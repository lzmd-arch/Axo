"""桌面工具回界面线程执行的那次交接。"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from PySide6.QtCore import QObject, QThread, Slot

from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Ui(QObject):
    _ui_call = MainWindow._ui_call

    def __init__(self) -> None:
        super().__init__()
        self._ui_calls: list[Any] = []
        self._ui_calls_lock = threading.Lock()

    @Slot()
    def _run_ui_calls(self) -> None:
        MainWindow._run_ui_calls(self)


class Worker(QThread):
    def __init__(self, ui: Ui) -> None:
        super().__init__()
        self.ui = ui
        self.result: Any = None

    def run(self) -> None:
        self.result = self.ui._ui_call(lambda: "界面线程的答案")


def test_a_call_from_the_main_thread_runs_inline(qapp: Any) -> None:
    ui = Ui()

    assert ui._ui_call(lambda: 42) == 42


def test_a_call_from_a_worker_thread_is_marshalled(qapp: Any) -> None:
    ui = Ui()
    worker = Worker(ui)
    worker.start()

    deadline = time.monotonic() + 5
    while worker.isRunning() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert worker.result == "界面线程的答案"
    worker.wait(2000)
