"""daemon → UI 的单向事件通道：一个只追加的 JSONL 文件。

**为什么是文件而不是 socket**：UI 会关、会崩、会重启，而 daemon 是常驻的。
socket 把"谁先起来"变成一个必须处理的状态；文件把这件事降级成"读到第几行"——
UI 重启后从上次的行号继续，漏掉的事件还在那里。代价是**没有回执**：daemon 不知道
UI 有没有读过。这正是"单向"的意思，别在上行方向加请求/应答。

本模块**只用标准库**，所以 UI 可以 import 它而不带来 numpy / biosnn_bus。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["BusEvent", "append_event", "read_new_events"]

EVENT_KINDS = (
    "greeting_suggestion",
    "retrieval_intent",
    "dream_done",
    "daemon_start",
    "daemon_stop",
    "error",
)


@dataclass(frozen=True)
class BusEvent:
    at: float
    kind: str
    payload: dict[str, Any]


def append_event(path: Path, kind: str, payload: dict[str, Any], *, at: float) -> BusEvent:
    """追加一行。**写失败抛 ``OSError``**，由调用方决定是否让这一轮失败。

    只有写失败会让事件消失，所以这里不吞异常；反过来，读的那一头必须对坏行宽容
    （见 :func:`read_new_events`）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": float(at), "kind": str(kind), "payload": payload}, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return BusEvent(at=float(at), kind=str(kind), payload=dict(payload))


def read_new_events(path: Path, after_line: int = 0) -> tuple[list[BusEvent], int]:
    """读 ``after_line`` 之后的新事件，返回 ``(事件, 新行号)``。

    坏行**跳过但仍然计数**：否则一个写坏的半行会让读端每次从同一位置重来，之后
    所有事件都读不到。这是"宁可漏一条，不要卡住"。
    """
    path = Path(path)
    events: list[BusEvent] = []
    line_no = int(after_line)
    try:
        with path.open(encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                if i < after_line:
                    continue
                line_no = i + 1
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                events.append(
                    BusEvent(
                        at=float(payload.get("at", 0.0)),
                        kind=str(payload.get("kind", "")),
                        payload=payload.get("payload") or {},
                    )
                )
    except FileNotFoundError:
        return [], int(after_line)
    return events, line_no
