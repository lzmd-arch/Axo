"""主动搭话的另一种话头：**网上的热点新闻**。

与记忆那条分开一个 worker，因为它是**网络请求**：跑在界面线程里会把节拍器卡住，
而这只该在选中「热点新闻」时发生。取数字段与失败收场都由 :mod:`axo.agent.proactive_chat`
的纯函数决定，这里只负责"去问、回来交给界面"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..agent.proactive_chat import news_material

__all__ = ["NewsWorker"]


class NewsWorker(QObject):
    """搜一次热点，交回一条 ``Material``（没有可说的就是 ``None``）。"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, web: Any, query: str, *, now: datetime) -> None:
        super().__init__()
        self._web = web
        self._query = query
        self._now = now

    @Slot()
    def run(self) -> None:
        try:
            hits = self._web.search(self._query, count=5)
        except Exception as exc:  # noqa: BLE001 — 后台线程里抛出去没人接
            # 不把异常类型说给用户看（那是日志的事）；界面只需要知道"这次没搜到"。
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(news_material(hits, now=self._now))
