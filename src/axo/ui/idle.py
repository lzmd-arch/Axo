"""无操作计时器：只回答“多久没有输入”，不碰 UI。

单次 ``QTimer`` + 60 秒上限，和 ``ReminderScheduler`` 同一种形状：系统睡眠/时钟跳变
后醒来，定时器可能一次晚很多；每次醒来重新比较 elapsed，而不是相信“定时器说到了”。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

__all__ = ["IdleMonitor"]


class IdleMonitor(QObject):
    """``idle`` 在连续 ``timeout_s`` 秒没有 ``touch()`` 时发一次。"""

    idle = Signal()

    def __init__(
        self,
        *,
        timeout_s: int = 900,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._timeout_s = max(60, min(86_400, int(timeout_s)))
        self._enabled = bool(enabled)
        self._last = self._clock()
        self._running = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    @property
    def running(self) -> bool:
        return self._running

    def configure(self, *, timeout_s: int, enabled: bool) -> None:
        self._timeout_s = max(60, min(86_400, int(timeout_s)))
        self._enabled = bool(enabled)
        if not self._enabled:
            self.stop()
        elif not self._running:
            # 从“关”切到“开”时必须真的 arm；只改布尔会让自动锁定静默不生效。
            self.start()
        else:
            self._arm()

    def start(self) -> None:
        self._last = self._clock()
        self._running = self._enabled
        if self._running:
            self._arm()
        else:
            self._timer.stop()

    def stop(self) -> None:
        self._running = False
        self._timer.stop()

    def touch(self) -> None:
        self._last = self._clock()

    def poll(self) -> None:
        """测试入口：按当前时钟检查一次；运行时由定时器调用。"""
        self._on_timeout()

    def _arm(self) -> None:
        remaining = max(0.0, self._timeout_s - (self._clock() - self._last))
        interval = int(min(60_000, max(5_000, remaining * 1000)))
        self._timer.start(interval)

    def _on_timeout(self) -> None:
        if not self._running:
            return
        if self._clock() - self._last >= self._timeout_s:
            self._running = False
            self._timer.stop()
            self.idle.emit()
            return
        self._arm()
