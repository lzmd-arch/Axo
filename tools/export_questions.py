"""从真实会话里抽出提问，生成一份**待标注**语料（只读会话，输出写仓外）。

    python tools/export_questions.py --out ~/axo-annotation
    python tools/export_questions.py --dry-run          # 只统计，不写文件

输出：`questions.jsonl`，每行 `{id, question, source_session, asked_at, label, note}`，
`label` 一律为 `null`，由人按 `docs/annotation-rubric.md` 填。

**它不判断“有没有记忆支撑”**——那是人的活。这里只做筛选与取样：去掉太短、纯附和、
明显的重复，并按时间分层抽样，让样本不至于全挤在某一天。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

#: 默认从 Axo 自己的数据根取会话目录（四档优先级由 :mod:`axo.paths` 决定）。
DEFAULT_SESSIONS: Path | None = None
DEFAULT_OUT = Path.home() / "axo-annotation"

#: 太短或纯附和的不值得标：它们既没有“事实”，也谈不上“有没有支撑”。
_TRIVIAL = {
    "好",
    "嗯",
    "哦",
    "行",
    "可以",
    "继续",
    "谢谢",
    "好的",
    "嗯嗯",
    "收到",
    "哈哈",
    "？",
    "?",
}
_MIN_CHARS = 5

#: 像“提问”的标志。真实会话里大量发言是碎片（“这个才是”“正在给你加”），
#: 它们既不是提问，也很难判“有没有记忆支撑”。默认只取带这些标志的句子，
#: 想连碎片一起标就加 ``--all``。
_QUESTION_MARKERS = ("吗", "什么", "怎么", "哪", "谁", "记得", "是不是", "多少", "几", "？", "?")


def _looks_like_question(text: str) -> bool:
    return any(marker in text for marker in _QUESTION_MARKERS)


def _load_turns(sessions: Path, *, questions_only: bool = True) -> list[dict]:
    turns: list[dict] = []
    for path in sorted(sessions.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("role") != "user":
                continue
            text = (record.get("content") or "").strip()
            if len(text) < _MIN_CHARS or text in _TRIVIAL:
                continue
            if questions_only and not _looks_like_question(text):
                continue
            turns.append(
                {
                    "question": text,
                    "source_session": path.stem,
                    "asked_at": record.get("at"),
                }
            )
    return turns


def _dedupe(turns: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for turn in turns:
        key = turn["question"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(turn)
    return unique


def _stratified(turns: list[dict], limit: int, seed: int) -> list[dict]:
    """按会话分层：每个会话先各取一份，再轮转补齐。

    单会话刷屏的样本会让“真实分布”变成“某一天的分布”，所以分层是按会话而不是按全局随机。
    """
    by_session: dict[str, list[dict]] = {}
    for turn in turns:
        by_session.setdefault(turn["source_session"], []).append(turn)

    rng = random.Random(seed)
    for items in by_session.values():
        rng.shuffle(items)

    picked: list[dict] = []
    round_index = 0
    while len(picked) < limit:
        added = False
        for session in sorted(by_session):
            items = by_session[session]
            if round_index < len(items):
                picked.append(items[round_index])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return picked


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true", help="连不像提问的碎片一起取")
    args = parser.parse_args(argv)

    if args.sessions is None:
        from axo import paths  # noqa: PLC0415 - 只有真要读盘时才导入

        args.sessions = paths.sessions_dir()
    if not args.sessions.is_dir():
        print(f"没有这个目录：{args.sessions}")
        return 2

    turns = _dedupe(_load_turns(args.sessions, questions_only=not args.all))
    picked = _stratified(turns, args.limit, args.seed)
    print(f"会话目录：{args.sessions}")
    scope = "全部发言" if args.all else "像提问的"
    print(f"可用（去重、去附和、{scope}）：{len(turns)}；本次取样：{len(picked)}")

    if args.dry_run:
        for item in picked[:5]:
            print("   ·", item["question"][:60])
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"q{index:03d}",
            "question": item["question"],
            "source_session": item["source_session"],
            "asked_at": item["asked_at"],
            "label": None,
            "note": "",
        }
        for index, item in enumerate(picked, start=1)
    ]
    target = args.out / "questions.jsonl"
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    target.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (args.out / "questions.sha256").write_text(
        f"{digest}  questions.jsonl\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {target}（{len(rows)} 条，sha256 {digest[:16]}…）")
    print(
        "下一步：按 docs/annotation-rubric.md 填 label（support / none / partial），再跑 eval_recall_threshold.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
