"""隔离冒烟：拿装好的 ``axo-mcp-write.exe`` 走一遍提议 → 门控 → 台账 → 撤销。

**全程临时目录**：库、队列、审计都在 ``tempfile.mkdtemp()`` 里，跑完即删。不读也不写
用户的真实记忆库，也不碰真实设置——写档位的开关靠 ``AXO_PROPOSE_SETTING=1`` 覆盖，
而这个覆盖**只在显式给了数据目录时**才被服务端采信（见 ``propose_server.setting_enabled``）。

    python tools/_smoke_mcp_write.py                    # 默认测 dist/_mcp 里的产物
    python tools/_smoke_mcp_write.py --exe <路径> --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EXE = _ROOT / "dist" / "_mcp" / "axo-mcp-write.exe"

RESULTS: list[tuple[str, bool, str]] = []


def check(rule: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((rule, bool(ok), detail))


class Session:
    def __init__(self, exe: Path, state: Path, db: Path) -> None:
        env = dict(os.environ, AXO_PROPOSE_SETTING="1")
        self.proc = subprocess.Popen(
            [
                str(exe),
                "--data-dir",
                str(state),
                "--db",
                str(db),
                "--queue",
                str(state / "q.sqlite3"),
                "--audit",
                str(state / "a.jsonl"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._id = 0

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        message: dict = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            message["params"] = params
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"没有响应：{stderr[:300]}")
        return json.loads(line)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        reply = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        return json.loads(reply["result"]["content"][0]["text"])

    def tools(self) -> list[str]:
        return [tool["name"] for tool in self._send("tools/list")["result"]["tools"]]

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=20)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=_DEFAULT_EXE)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    if not args.exe.is_file():
        print(f"找不到 {args.exe}；先跑 tools/build_exe.py", file=sys.stderr)
        return 2

    sys.path.insert(0, str(_ROOT / "src"))
    from memcore.store import MemoryStore  # noqa: PLC0415 - 只有跑这个脚本才需要

    state = Path(tempfile.mkdtemp(prefix="axo-mcp-smoke-"))
    db = state / "mem.db"
    MemoryStore(db).close()  # 只读连接要求库已存在

    session = Session(args.exe, state, db)
    try:
        tools = session.tools()
        check("工具表 8 个", len(tools) == 8 and "memory.revert" in tools, str(tools))

        proposed = session.call(
            "propose.memory.write",
            {
                "text": "the gate accepted this line in an isolated run",
                "proposal_id": "iso-1",
                "source": "user",
            },
        )
        check(
            "提议被判 accepted",
            proposed.get("status") == "accepted",
            json.dumps(proposed, ensure_ascii=False),
        )

        hits = session.call("memory.search", {"query": "accepted this line"})
        check(
            "落库后可检索（多词查询）",
            hits.get("count", 0) >= 1,
            json.dumps(hits, ensure_ascii=False)[:120],
        )

        duplicate = session.call(
            "propose.memory.write",
            {"text": "the gate accepted this line in an isolated run", "proposal_id": "iso-2"},
        )
        check(
            "重复判 rejected",
            duplicate.get("status") == "rejected",
            json.dumps(duplicate, ensure_ascii=False),
        )

        replayed = session.call(
            "propose.memory.write",
            {"text": "the gate accepted this line in an isolated run", "proposal_id": "iso-1"},
        )
        check(
            "同一 proposal_id 幂等",
            replayed.get("seq") == proposed.get("seq"),
            json.dumps(replayed, ensure_ascii=False),
        )

        reverted = session.call(
            "memory.revert", {"seq": proposed["seq"], "reason": "isolated rollback"}
        )
        check(
            "撤销成功且库回到 0 条",
            reverted.get("reverted") is True and reverted.get("entries") == 0,
            json.dumps(reverted, ensure_ascii=False),
        )

        events = [
            json.loads(line)["event"]
            for line in (state / "a.jsonl").read_text("utf-8").splitlines()
        ]
        check(
            "审计事件齐全",
            {"proposal.received", "gate.accepted", "memory.applied", "memory.revert"}
            <= set(events),
            str(sorted(set(events))),
        )
    finally:
        session.close()

    width = max(len(rule) for rule, _, _ in RESULTS)
    for rule, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {rule.ljust(width)}  {detail[:110]}")
    failed = [rule for rule, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过；状态目录：{state}")
    if not args.keep and not failed:
        shutil.rmtree(state, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
