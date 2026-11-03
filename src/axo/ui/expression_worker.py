"""把表达层的一次渲染挪到工作线程。

**为什么不是又一个 ``ChatWorker``**：那句回答要**先核事实、再显示**，而流式显示与
"渲染完才发现丢了一个数"是冲突的——文字已经推出去了就没法收回。所以这里跑完整个
渲染、拿到 :class:`~axo.expression.layer.ExpressionOutcome` 之后才发一次信号，由界面
决定显示哪一份（表达层的、或退回的草稿）。

代价是这一轮的首字更晚出现（主模型写完 + 渲染写完）。表达层默认关闭，打开它的人
换的是"换模型不换说法"，这个延迟是明码标价的。
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from ..expression.layer import ExpressionLayer, ExpressionOutcome
from ..expression.mix import MixedRenderer
from ..expression.sir import SIR

__all__ = ["ExpressionWorker"]


class ExpressionWorker(QObject):
    """在工作线程里跑一次 :meth:`ExpressionLayer.render`。

    ``finished`` **永远发出**（成功、退回草稿、内部异常都走这一条）：界面据此收尾，
    少发一次就等于把这一轮卡在"正在统一语气…"。
    """

    finished = Signal(object)
    """:class:`~axo.expression.layer.ExpressionOutcome`。"""

    def __init__(self, layer: ExpressionLayer | MixedRenderer, sir: SIR) -> None:
        super().__init__()
        self._layer = layer
        self._sir = sir
        self._stop = threading.Event()

    def stop(self) -> None:
        """请求停止。渲染循环在下一片之前读它；不是即时中断（同 ``ChatWorker.stop``）。"""
        self._stop.set()

    @Slot()
    def run(self) -> None:
        try:
            outcome = self._layer.render(self._sir, keep_going=lambda: not self._stop.is_set())
        except Exception as exc:  # noqa: BLE001 —— 收尾必须发出，否则界面卡住
            outcome = ExpressionOutcome(
                self._sir.draft,
                "draft",
                f"表达层异常（{type(exc).__name__}: {exc}）",
            )
        self.finished.emit(outcome)
