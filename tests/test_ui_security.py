"""启动闸门与锁屏遮罩：状态机与信号，不构造真主窗口/WebEngine。"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QDialog

from axo import passwords
from axo.ui.lock_overlay import LockOverlay
from axo.ui.settings_store import SecuritySettings
from axo.ui.startup_gate import StartupGate
from axo.ui.theme import Theme, resolve

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


def _gate(security: SecuritySettings) -> tuple[StartupGate, list[SecuritySettings]]:
    saved: list[SecuritySettings] = []
    gate = StartupGate(
        theme=Theme(),
        resolved=resolve(Theme(), "dark"),
        security=security,
        save=saved.append,
    )
    return gate, saved


def test_setup_does_not_save_before_the_recovery_code_is_acknowledged(qapp: Any) -> None:
    gate, saved = _gate(SecuritySettings())
    gate._password.setText("correct horse battery")
    gate._confirm.setText("correct horse battery")
    gate._submit()
    assert saved == []
    assert gate._pending is not None
    assert gate._code_view.text()
    gate._submit()  # 未勾选，不能放行
    assert saved == []
    gate._saved.setChecked(True)
    gate._submit()
    assert len(saved) == 1
    assert saved[0].has_password and saved[0].has_recovery
    assert gate.result() == QDialog.DialogCode.Accepted


def test_unlock_accepts_the_right_password(qapp: Any) -> None:
    verifier = passwords.hash_password("correct horse", iterations=100_000)
    gate, saved = _gate(SecuritySettings(password_verifier=verifier))
    gate._password.setText("wrong")
    gate._submit()
    assert saved == [] and gate.result() != QDialog.DialogCode.Accepted
    gate._password.setText("correct horse")
    gate._submit()
    assert gate.result() == QDialog.DialogCode.Accepted


def test_lock_overlay_emits_and_ignores_escape(qapp: Any) -> None:
    overlay = LockOverlay()
    submitted: list[str] = []
    overlay.unlock_requested.connect(submitted.append)
    overlay._password.setText("pw")
    overlay._submit()
    assert submitted == ["pw"]
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    overlay.keyPressEvent(event)
    assert event.isAccepted() is False
