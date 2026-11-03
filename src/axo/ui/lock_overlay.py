"""锁屏遮罩：不透明、覆盖主窗口内容、没有 Esc 关闭。

它只收密码并发出信号；校验与恢复流程由 :class:`~axo.ui.main_window.MainWindow`
处理。这样这个控件可以单独在离屏下测试，不会构造 WebEngine。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LockOverlay"]


class LockOverlay(QWidget):
    unlock_requested = Signal(str)
    recovery_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, accent: str = "#6b8afd") -> None:
        super().__init__(parent)
        self.setObjectName("lockOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget#lockOverlay { background: #080a0d; }"
            "QLabel { color: #eef3f1; }"
            "QLineEdit { background: #11151a; color: #eef3f1; border: 1px solid #2a333b;"
            " padding: 9px; border-radius: 6px; }"
            f"QPushButton {{ background: {accent}; color: #080a0d; border: 0;"
            " padding: 9px 20px; border-radius: 6px; font-weight: 600; }"
            "QPushButton:disabled { background: #2a333b; color: #7d8785; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(80, 60, 80, 60)
        outer.addStretch(1)
        self._title = QLabel("已锁定", self)
        self._title.setStyleSheet("font-size: 40px; font-weight: 700;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(self._title)
        self._reason = QLabel("", self)
        self._reason.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._reason.setStyleSheet("color: #7d8785;")
        outer.addWidget(self._reason)
        self._password = QLineEdit(self)
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.setPlaceholderText("应用密码")
        self._password.returnPressed.connect(self._submit)
        outer.addWidget(self._password)
        self._error = QLabel("", self)
        self._error.setStyleSheet("color: #ff6b5e;")
        self._error.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(self._error)
        self._unlock = QPushButton("解锁", self)
        self._unlock.clicked.connect(self._submit)
        outer.addWidget(self._unlock)
        self._recovery = QPushButton("忘记密码，使用恢复码", self)
        self._recovery.clicked.connect(lambda: self.recovery_requested.emit())
        outer.addWidget(self._recovery)
        self._quit = QPushButton("退出 Axo", self)
        self._quit.clicked.connect(lambda: self.quit_requested.emit())
        outer.addWidget(self._quit)
        outer.addStretch(1)

    def set_reason(self, text: str) -> None:
        self._reason.setText(text)

    def show_error(self, text: str) -> None:
        self._error.setText(text)
        self._password.selectAll()

    def set_busy(self, busy: bool) -> None:
        self._password.setEnabled(not busy)
        self._unlock.setEnabled(not busy)
        self._recovery.setEnabled(not busy)

    def reset(self) -> None:
        self._password.clear()
        self._error.clear()
        self.set_busy(False)

    def focus_password(self) -> None:
        self._password.setFocus()

    def _submit(self) -> None:
        if not self._password.isEnabled():
            return
        self.unlock_requested.emit(self._password.text())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)
