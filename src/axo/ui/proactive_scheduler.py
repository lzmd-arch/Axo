"""她主动开口的**节拍器**：隔一会儿问一次"现在够格了吗"。

**它不做任何判定。** 判定在 :func:`axo.agent.proactive_chat.should_speak` 里（纯层，
逐格可断言），事实由 ``MainWindow`` 现取——他最后说话是什么时候、今天来过几次、设置说
静多久算静。这个类全部的状态就是"表在不在走"，所以它自己没有分支可测，gui 腿只需要
钉住起停。

**为什么是固定 60 秒的周期表，而不是像 :class:`~axo.ui.reminder_scheduler.ReminderScheduler`
那样"睡到下一个时刻"。** 那个类算得出下一次睡多久，是因为它等的是一个**已经存在的**
``due_at``；这里等的条件（静够 N 分钟、今天还没说够、不在免打扰时段）每一刻都在变，
没有那样的数可算。代价也一样：``ReminderScheduler`` 把单次睡眠压在 60 秒以内正是为了
防时钟跳变与休眠，所以"每分钟看一眼"并不比它贵，而遇到任何时钟异常也只是晚一分钟。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Slot

__all__ = ["TICK_MS", "ProactiveScheduler"]

#: 多久问一次。**不是"多久开口一次"**——那是设置里的最小间隔（默认 20 分钟）。
TICK_MS = 60_000


class ProactiveScheduler(QObject):
    """到点就问一次。**判定与取材都不归它管。**

    Args:
        tick_ms: 问一次的间隔。测试调小它，好在一次事件循环里看见两次。
        parent: Qt 父对象。
    """

    tick = Signal()
    """该问一次了。**载荷为空**——判定与取材都不在这里，所以这里只有"时间到了"这一
    件事可说。带一个"该不该开口"的结论出来会让这个类重新长出分支。"""

    def __init__(self, *, tick_ms: int = TICK_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tick_ms = tick_ms
        self._running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timeout)

    @property
    def running(self) -> bool:
        """表在不在走。"""
        return self._running

    @property
    def tick_ms(self) -> int:
        return self._tick_ms

    def start(self) -> None:
        """开始问。**幂等**——重复调用不会把周期重启一遍。"""
        if self._running:
            return
        self._running = True
        self._timer.start(self._tick_ms)

    def stop(self) -> None:
        """停表（退出时用）。**幂等。**"""
        self._running = False
        self._timer.stop()

    @Slot()
    def _on_timeout(self) -> None:
        self.tick.emit()
