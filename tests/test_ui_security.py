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


def test_corrupt_security_state_fails_closed_instead_of_setup(qapp: Any) -> None:
    """文件坏了不能变成“首次设置”：那等于删坏文件就能重设密码。"""
    gate, saved = _gate(SecuritySettings(storage="corrupt"))
    assert gate._blocked is True
    assert gate._setup_mode is False
    assert gate._password.isHidden()
    assert gate._confirm.isHidden()
    assert gate._recovery_button.isHidden()
    gate._password.setText("whatever")
    gate._confirm.setText("whatever")
    gate._submit()
    assert saved == []
    assert gate.result() != QDialog.DialogCode.Accepted


def test_gate_does_not_show_setup_when_verifier_is_empty_without_state(qapp: Any) -> None:
    """没有状态文件也没有 verifier 才是真正的首次启动。"""
    gate, _saved = _gate(SecuritySettings())
    assert gate._blocked is False
    assert gate._setup_mode is True


def test_unlock_throttles_repeated_failures(qapp: Any) -> None:
    """锁屏不能比启动闸门更容易试密码；连续失败后进入冷却，冷却期内正确也不放行。"""
    from types import SimpleNamespace

    from axo.ui.main_window import MainWindow

    class Overlay:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.busy: list[bool] = []

        def show_error(self, text: str) -> None:
            self.errors.append(text)

        def set_busy(self, value: bool) -> None:
            self.busy.append(value)

    overlay = Overlay()
    fake = SimpleNamespace(
        _locked=True,
        _unlock_not_before=0.0,
        _unlock_attempts=0,
        _security_settings=SecuritySettings(
            password_verifier=passwords.hash_password("correct horse", iterations=100_000)
        ),
        _ensure_lock_overlay=lambda: overlay,
        _finish_unlock=lambda: overlay.errors.append("unlocked"),
    )
    for _ in range(3):
        MainWindow._unlock(fake, "wrong")
    assert fake._unlock_attempts == 3
    assert fake._unlock_not_before > 0

    MainWindow._unlock(fake, "correct horse")
    assert "unlocked" not in overlay.errors

    fake._unlock_not_before = 0.0
    MainWindow._unlock(fake, "correct horse")
    assert "unlocked" in overlay.errors
