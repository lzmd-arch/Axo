"""无操作计时器：纯时钟计算，不依赖真实等待。"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui.idle import IdleMonitor

pytestmark = pytest.mark.gui


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


def test_idle_fires_once_after_timeout(qapp: Any) -> None:
    clock = Clock()
    monitor = IdleMonitor(timeout_s=60, clock=clock)
    fired: list[bool] = []
    monitor.idle.connect(lambda: fired.append(True))
    monitor.start()
    clock.now += 59
    monitor.poll()
    assert fired == []
    clock.now += 2
    monitor.poll()
    assert fired == [True]
    monitor.poll()
    assert fired == [True]


def test_touch_resets_the_window(qapp: Any) -> None:
    clock = Clock()
    monitor = IdleMonitor(timeout_s=60, clock=clock)
    fired: list[bool] = []
    monitor.idle.connect(lambda: fired.append(True))
    monitor.start()
    clock.now += 50
    monitor.touch()
    clock.now += 50
    monitor.poll()
    assert fired == []
    clock.now += 11
    monitor.poll()
    assert fired == [True]


def test_stop_prevents_firing_and_configure_rearms(qapp: Any) -> None:
    clock = Clock()
    monitor = IdleMonitor(timeout_s=60, clock=clock)
    fired: list[bool] = []
    monitor.idle.connect(lambda: fired.append(True))
    monitor.start()
    monitor.stop()
    clock.now += 999
    monitor.poll()
    assert fired == []

    monitor.configure(timeout_s=120, enabled=True)
    monitor.start()
    clock.now += 121
    monitor.poll()
    assert fired == [True]
