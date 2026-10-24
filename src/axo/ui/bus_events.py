"""UI 侧的总线事件接收：**只读文件，不认识 daemon**。

daemon 是独立进程，它把事件写进 ``<data>/bus/events.jsonl``（见
:mod:`axo.busd.events`）。这里只做三件事：记住读到第几行、把新事件发成信号、
把行号落盘——UI 重启后不会把旧建议再说一遍。

**顺序：先推进游标，再发信号。** 与提醒那条路同一条理由（先标 fired 再弹）：
反过来的话，消费方抛异常会让同一行每次轮询都重放一遍。代价是"记下了但没处理成"，
而那比"每次启动都重放旧建议"轻得多。
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..busd.events import read_new_events
from ..paths import data_dir

__all__ = ["BusEventPoller"]

POLL_MS = 4000


class BusEventPoller(QObject):
    """轮询事件文件。时间由 ``QTimer`` 驱动，测试可以直接调 :meth:`poll`。"""

    event = Signal(object)
    """一条 :class:`~axo.busd.events.BusEvent`。"""

    def __init__(
        self,
        *,
        events_path: Path | None = None,
        cursor_path: Path | None = None,
        poll_ms: int = POLL_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        root = data_dir() / "bus"
        self._events_path = Path(events_path) if events_path else root / "events.jsonl"
        self._cursor_path = Path(cursor_path) if cursor_path else root / "ui_cursor.json"
        self._line = self._load_cursor()
        self._timer = QTimer(self)
        self._timer.setInterval(max(200, int(poll_ms)))
        self._timer.timeout.connect(self.poll)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def line(self) -> int:
        """已经读到的行号（下一轮从这里继续）。"""
        return self._line

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._timer.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._timer.stop()

    @Slot()
    def poll(self) -> int:
        """读一次新事件，返回发出去的条数。**绝不抛**：坏文件在读取端已处理。"""
        events, line = read_new_events(self._events_path, self._line)
        if line != self._line:
            self._line = line
            self._save_cursor()
        for ev in events:
            self.event.emit(ev)
        return len(events)

    # ------------------------------------------------------------ 游标

    def _load_cursor(self) -> int:
        try:
            payload = json.loads(self._cursor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(payload, dict):
            return 0
        try:
            return max(0, int(payload.get("line", 0)))
        except (TypeError, ValueError):
            return 0

    def _save_cursor(self) -> None:
        try:
            self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
            self._cursor_path.write_text(
                json.dumps({"line": int(self._line)}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
