"""天气检查：在工作线程里查一次、跟基线比一次，交回变化。

与 ``news_worker`` 同一形状；区别是它**只读基线**，写基线由界面线程在"她真的说了"
之后做（见 ``WeatherBaseline`` 的模块文档）。
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..weather_watch import (
    WeatherBaseline,
    WeatherWatchResult,
    changed_lines,
    snapshot_from_report,
)

__all__ = ["WeatherWatchWorker"]


class WeatherWatchWorker(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, weather: Any, city: str, baseline: WeatherBaseline) -> None:
        super().__init__()
        self._weather = weather
        self._city = city
        self._baseline = baseline

    @Slot()
    def run(self) -> None:
        try:
            report = self._weather.lookup(self._city, days=1)
        except Exception as exc:  # noqa: BLE001 — 后台线程里抛出去没人接
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        snapshot = snapshot_from_report(report, city=self._city, now=time.time())
        previous = self._baseline.get(self._city)
        self.done.emit(
            WeatherWatchResult(snapshot=snapshot, changes=changed_lines(previous, snapshot))
        )
