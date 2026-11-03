"""启动自检 worker：结果经信号回主线程，异常也不把启动拖垮。"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui import boot_checks as boot_checks_module
from axo.ui.boot_check_worker import BootCheckWorker
from axo.ui.boot_checks import BootCheck

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


def test_the_worker_emits_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (BootCheck("data", "数据目录", True, "ok"),)
    monkeypatch.setattr(boot_checks_module, "collect_checks", lambda *_args, **_kwargs: expected)
    received: list[object] = []
    worker = BootCheckWorker(None, None)
    worker.done.connect(received.append)

    worker.run()

    assert received == [expected]


def test_a_broken_check_run_emits_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> None:
        raise RuntimeError("坏了")

    monkeypatch.setattr(boot_checks_module, "collect_checks", boom)
    received: list[object] = []
    worker = BootCheckWorker(None, None)
    worker.done.connect(received.append)

    worker.run()

    assert received == [()]


def test_redact_flag_rewrites_details_before_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (BootCheck("data", "数据目录", True, "D:/secret/path"),)
    monkeypatch.setattr(boot_checks_module, "collect_checks", lambda *_args, **_kwargs: expected)
    received: list[object] = []
    worker = BootCheckWorker(None, None, redact=True)
    worker.done.connect(received.append)

    worker.run()

    assert received == [(BootCheck("data", "数据目录", True, "可写"),)]
