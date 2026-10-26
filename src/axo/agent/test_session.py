"""测试会话记忆：让同一轮里的多次 ``run_tests`` 能互相对照，而不是各跑各的。

模型最常见的失败模式是"失败 → 原样重跑 → 再失败 → 再原样重跑"。工具层能看见三件
它自己未必记得的事：这是第几次、上次的失败集合是什么、工作区文件自上次以来有没有变。
把这三件拼成一句回灌，比在系统提示里写十遍"不要原地重跑"更有用。

**纯层**：不 import Qt、不执行命令。快照只看 ``(相对路径, 大小, mtime)``，不读内容；
噪声目录（``__pycache__``、``.pytest_cache`` 等）不参与，否则跑一次测试就会把自己
写出的缓存当成"工作区变了"。文件太多时明确标截断，**不据此断言"没有变化"**。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .snapshot import Snapshot, snapshot_tree

__all__ = ["Snapshot", "TestRun", "TestSession", "snapshot_tree"]

_MAX_FAILURES = 40
_MAX_RUNS = 50
"""跨轮保留的最近测试次数。更老的只留计数，不占内存。"""

_BRIEF_CHARS = 500
"""注入 L1 的“上次测试”一句话上限。L1 是每轮都发的事实区，不能让它挤掉人格锚。"""


@dataclass(frozen=True)
class TestRun:
    __test__ = False  # pytest 收集标记，见 TestSession。

    index: int
    command: str
    cwd: str
    ok: bool
    exit_code: int | None
    failures: tuple[str, ...]
    snapshot: Snapshot


def _preview(lines: Sequence[str], limit: int = 3) -> str:
    shown = "；".join(lines[:limit])
    more = f" 等 {len(lines)} 项" if len(lines) > limit else ""
    return f"{shown}{more}"


class TestSession:
    """同一会话里的测试历史。**只活在内存里**，新会话/退出即丢。

    跨用户回合复用：上一轮的失败集合与文件快照留在这里，下一轮第一次 ``run_tests``
    就能立刻说出“与上次相比新增/消失/仍失败”，而不是从“第 1 次”重新开始。
    """

    # pytest 会把任何 ``Test*`` 类当测试收集；这个类有 ``__init__``，会报一条
    # 无用警告。``__test__ = False`` 是 pytest 认的显式退出标记。
    __test__ = False

    def __init__(self) -> None:
        self._runs: list[TestRun] = []
        self._total = 0
        self._lock = threading.Lock()

    @property
    def runs(self) -> tuple[TestRun, ...]:
        """最近 ``_MAX_RUNS`` 次；更老的已丢，但 ``index`` 不会回退。"""
        with self._lock:
            return tuple(self._runs)

    def brief(self) -> str:
        """注入 L1 的一句话“上次测试”。**没有历史时返回空串。**"""
        with self._lock:
            if not self._runs:
                return ""
            run = self._runs[-1]
        command = " ".join(run.command.split())
        state = "通过" if run.ok else f"失败（退出码 {run.exit_code}）"
        detail = ""
        if run.failures:
            shown = "；".join(run.failures[:2])
            more = f" 等 {len(run.failures)} 项" if len(run.failures) > 2 else ""
            detail = f"；失败：{shown}{more}"
        text = f"上次测试（本会话第 {run.index} 次）：`{command}` @ {run.cwd} → {state}{detail}。"
        if len(text) <= _BRIEF_CHARS:
            return text
        return text[: _BRIEF_CHARS - 1] + "…"

    def finish(
        self,
        *,
        command: str,
        cwd: Path,
        ok: bool,
        exit_code: int | None,
        failures: Sequence[str],
        before: Snapshot,
    ) -> str:
        """记录一次测试，返回给模型看的那句对照。"""
        kept = tuple(failures[:_MAX_FAILURES])
        with self._lock:
            previous = self._runs[-1] if self._runs else None
            self._total += 1
            run = TestRun(
                index=self._total,
                command=command,
                cwd=str(cwd),
                ok=ok,
                exit_code=exit_code,
                failures=kept,
                snapshot=before,
            )
            self._runs.append(run)
            if len(self._runs) > _MAX_RUNS:
                del self._runs[0]
        if previous is None:
            return f"本会话第 {run.index} 次测试（没有上次可对比）。"

        lines = [f"本会话第 {run.index} 次测试。"]
        if previous.command != command or previous.cwd != str(cwd):
            lines.append("命令或目录与上次不同，下面的失败对比仅供参考。")

        fixed = sorted(set(previous.failures) - set(kept))
        new = sorted(set(kept) - set(previous.failures))
        still = sorted(set(kept) & set(previous.failures))

        if ok:
            if previous.ok:
                lines.append("上次通过，这次也通过。")
            elif previous.failures:
                lines.append(f"上次的 {len(previous.failures)} 项失败这次都消失了。")
            else:
                lines.append("上次失败、这次通过。")
        elif previous.ok:
            if new:
                lines.append(f"上次通过，这次出现 {len(new)} 项失败：{_preview(new)}。")
            else:
                lines.append("上次通过，这次失败（没有摘出具体失败行）。")
        else:
            if still:
                lines.append(f"仍失败 {len(still)} 项：{_preview(still)}。")
            if new:
                lines.append(f"新增失败 {len(new)} 项：{_preview(new)}。")
            if fixed:
                lines.append(f"已消失 {len(fixed)} 项：{_preview(fixed)}。")
            if not (still or new or fixed):
                lines.append(
                    "失败与上次完全相同。" if kept else "这次没有摘出失败行，无法与上次逐项对比。"
                )

        unchanged = (
            before.digest is not None
            and previous.snapshot.digest is not None
            and before.digest == previous.snapshot.digest
        )
        if before.truncated or previous.snapshot.truncated:
            lines.append("（工作区文件太多，快照只看了一部分，不能据此断言“没有变化”。）")
        elif unchanged:
            if not ok and kept and not (new or fixed) and set(still) == set(previous.failures):
                lines.append(
                    "⚠ 工作区快照与上次测试前相同，失败也完全相同——原地重跑不会改变结果。"
                    "先读失败摘要指向的文件并修改，再跑。"
                )
            else:
                lines.append("工作区快照与上次测试前相同。")
        elif before.digest is not None and previous.snapshot.digest is not None:
            lines.append("工作区快照与上次测试前不同（上次测试自己写出的报告也算在内）。")
        return "\n".join(lines)
