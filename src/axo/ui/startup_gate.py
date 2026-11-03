"""启动闸门：六幕之后、主窗口之前的应用密码关口。

这一层只做 UI 与状态切换；哈希/KDF 在 :mod:`axo.passwords`，持久化由注入的
``save`` 完成。**主窗口在闸门被接受之前不会构造**，所以内容不会先露出来再盖住。

模式：
- ``setup``：没有可用 verifier（首次启动）——新密码 + 确认 → 恢复码只显示一次 →
  勾选“已保存”后才落盘并放行；
- ``unlock``：每次启动输入密码；失败递增等待，不持久化；
- ``recovery``：恢复码 → 新密码 → 新恢复码，旧码作废。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import passwords
from ..ui.settings_store import SecuritySettings
from ..ui.theme import Resolved, Theme

__all__ = ["StartupGate"]


class StartupGate(QDialog):
    """全屏、无边框、不可用 Esc 绕过的启动闸门。"""

    def __init__(
        self,
        *,
        theme: Theme,
        resolved: Resolved,
        security: SecuritySettings,
        save: Callable[[SecuritySettings], None],
        mode: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(True)
        self.setWindowTitle("Axo · 解锁")
        self._resolved = resolved
        self._security = security
        self._save = save
        self._attempts = 0
        self._not_before = 0.0
        self._pending: SecuritySettings | None = None
        record_ok = passwords.password_record_ok(security.password_verifier)
        # 数据根里的状态文件存在但读不出来：**失败关闭**。能走到 setup 就等于
        # "把文件删坏"是一种重置密码的方法，那这把锁就没有意义了。
        self._blocked = security.storage == "corrupt"
        self._recovery_mode = not self._blocked and (
            mode == "recovery" or (not record_ok and bool(security.recovery_verifier))
        )
        self._setup_mode = not self._blocked and not record_ok and not self._recovery_mode
        self._build()
        self.setStyleSheet(
            f"QDialog {{ background: #080a0d; color: #eef3f1; }}"
            f"QLabel {{ color: #eef3f1; }}"
            f"QLineEdit {{ background: #11151a; color: #eef3f1; border: 1px solid #2a333b;"
            f" padding: 8px; border-radius: 6px; }}"
            f"QPushButton {{ background: {resolved.accent}; color: #080a0d; border: 0;"
            f" padding: 8px 18px; border-radius: 6px; font-weight: 600; }}"
            f"QPushButton:disabled {{ background: #2a333b; color: #7d8785; }}"
            f"QCheckBox {{ color: #aeb8b5; }}"
        )

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        card = QWidget(self)
        card.setFixedWidth(560)
        inner = QVBoxLayout(card)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(14)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        if self._blocked:
            title_text = "安全状态不可用"
        else:
            title_text = "设置启动密码" if self._setup_mode else "身份核验"
        title = QLabel(title_text, card)
        title.setStyleSheet("font-size: 46px; font-weight: 700;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        inner.addWidget(title)
        self._hint = QLabel("", card)
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._hint.setStyleSheet("color: #7d8785; font-size: 13px;")
        inner.addWidget(self._hint)

        self._password = QLineEdit(card)
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.setPlaceholderText("应用密码")
        self._password.returnPressed.connect(self._submit)
        inner.addWidget(self._password)
        self._confirm = QLineEdit(card)
        self._confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self._confirm.setPlaceholderText("再次输入密码")
        self._confirm.returnPressed.connect(self._submit)
        inner.addWidget(self._confirm)

        self._recovery_input = QLineEdit(card)
        self._recovery_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._recovery_input.setPlaceholderText("恢复码（32 位，可带空格）")
        self._recovery_input.returnPressed.connect(self._submit)
        inner.addWidget(self._recovery_input)

        self._error = QLabel("", card)
        self._error.setStyleSheet("color: #ff6b5e;")
        self._error.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        inner.addWidget(self._error)

        self._code_view = QLabel("", card)
        self._code_view.setWordWrap(True)
        self._code_view.setStyleSheet("font-family: Consolas; font-size: 20px;")
        self._code_view.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._code_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(self._code_view)

        self._saved = QCheckBox("我已保存恢复码（只显示这一次）", card)
        self._saved.toggled.connect(self._sync_buttons)
        inner.addWidget(self._saved)

        self._submit_button = QPushButton("继续", card)
        self._submit_button.clicked.connect(self._submit)
        inner.addWidget(self._submit_button)
        self._recovery_button = QPushButton("使用恢复码", card)
        self._recovery_button.clicked.connect(self._enter_recovery)
        inner.addWidget(self._recovery_button)
        self._quit_button = QPushButton("退出 Axo", card)
        self._quit_button.clicked.connect(self.reject)
        inner.addWidget(self._quit_button)
        self._render_mode()

    def _render_mode(self) -> None:
        if self._blocked:
            for widget in (
                self._password,
                self._confirm,
                self._recovery_input,
                self._code_view,
                self._saved,
                self._recovery_button,
            ):
                widget.setVisible(False)
            self._error.setText("数据根里的 security/state.json 存在但无法读取。")
            self._hint.setText(
                "为避免有人通过删改这个文件重设密码，Axo 不会进入设置流程。"
                "请从备份恢复它；如果确实要重新开始，请有意删除它后再启动。"
            )
            self._submit_button.setText("退出 Axo")
            self._submit_button.setEnabled(True)
            self._quit_button.setVisible(False)  # 两个按钮做同一件事只会让人犹豫
            return
        setup_fields = self._setup_mode and self._pending is None
        self._password.setVisible(
            setup_fields or (not self._recovery_mode and self._pending is None)
        )
        self._confirm.setVisible(setup_fields)
        self._recovery_input.setVisible(self._recovery_mode and self._pending is None)
        self._code_view.setVisible(self._pending is not None)
        self._saved.setVisible(self._pending is not None)
        self._recovery_button.setVisible(
            not setup_fields and self._pending is None and bool(self._security.recovery_verifier)
        )
        if self._recovery_mode and self._pending is None:
            self._hint.setText("输入恢复码后设置新密码。恢复码只用于重置，不保存明文。")
            self._submit_button.setText("验证恢复码")
        elif self._pending is not None:
            self._hint.setText("请把恢复码抄下来。关闭后无法再次查看。")
            self._submit_button.setText("我已保存，继续")
        elif self._setup_mode:
            self._hint.setText("至少 8 个字符。密码只用于本机 Axo 解锁。")
            self._submit_button.setText("设置密码")
        else:
            self._hint.setText("输入应用密码。输错不会泄露任何记忆内容。")
            self._submit_button.setText("解锁")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        if self._pending is not None:
            self._submit_button.setEnabled(self._saved.isChecked())
        else:
            self._submit_button.setEnabled(True)

    def _enter_recovery(self) -> None:
        if not self._security.recovery_verifier:
            self._error.setText("没有可用的恢复码。")
            return
        self._recovery_mode = True
        self._setup_mode = False
        self._render_mode()
        self._recovery_input.setFocus()

    def _submit(self) -> None:
        if self._blocked:
            self.reject()
            return
        if time.monotonic() < self._not_before:
            return
        if self._pending is not None:
            self._finish_pending()
            return
        if self._recovery_mode:
            self._submit_recovery()
            return
        if self._setup_mode:
            self._submit_setup()
            return
        self._submit_unlock()

    def _submit_unlock(self) -> None:
        candidate = self._password.text()
        if passwords.verify_password(candidate, self._security.password_verifier):
            self._attempts = 0
            self.accept()
            return
        self._fail("密码不正确。")

    def _submit_setup(self) -> None:
        first = self._password.text()
        if len(first) < 8 or len(first) > 1024:
            self._fail("密码需要 8–1024 个字符。")
            return
        if first != self._confirm.text():
            self._fail("两次输入的密码不一致。")
            return
        code = passwords.new_recovery_code()
        verifier = passwords.hash_recovery_code(code)
        password_verifier = passwords.hash_password(first)
        self._pending = replace(
            self._security,
            password_verifier=password_verifier,
            recovery_verifier=verifier,
        )
        self._show_code(code)

    def _submit_recovery(self) -> None:
        code = self._recovery_input.text()
        if not passwords.verify_recovery_code(code, self._security.recovery_verifier):
            self._fail("恢复码不正确。")
            return
        self._recovery_mode = False
        self._setup_mode = True
        self._recovery_input.clear()
        self._hint.setText("恢复码正确。请设置新密码，并保存新的恢复码。")
        self._render_mode()

    def _show_code(self, code: str) -> None:
        self._code_view.setText("恢复码（只显示这一次）\n" + passwords.display_recovery_code(code))
        self._saved.setChecked(False)
        self._render_mode()
        self._code_view.setVisible(True)
        self._saved.setVisible(True)

    def _finish_pending(self) -> None:
        if self._pending is None or not self._saved.isChecked():
            return
        try:
            self._save(self._pending)
        except Exception:  # noqa: BLE001 —— 写入失败必须留在闸门里，不能放行
            self._error.setText("安全设置没能写入，请检查数据目录权限后重试。")
            return
        self.accept()

    def _fail(self, text: str) -> None:
        self._attempts += 1
        self._error.setText(text)
        self._password.selectAll()
        delay = 0 if self._attempts < 3 else min(30, 2 ** (self._attempts - 3) * 2)
        if delay:
            self._not_before = time.monotonic() + delay
            self._submit_button.setEnabled(False)
            QTimer.singleShot(delay * 1000, self._sync_buttons)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self.isFullScreen():
            self.showFullScreen()
        self._password.setFocus()
