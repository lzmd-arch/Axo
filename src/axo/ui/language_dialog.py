"""语言服务器安装/体检对话框。**把 CLI 的能力带进冻结产物。**

冻结的 exe 没有控制台，`axo-lsp-install` 这种 console entry point 也带不进去；所以
设置页里必须有一个入口。这个对话框只做三件事：体检、预览安装计划、确认后自动安装。
真正的安装逻辑仍在 :mod:`axo.agent.lsp_install`，这里只负责把它的 stdout 收进窗口。
"""

from __future__ import annotations

import contextlib
import io

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LanguageServersDialog"]


class _Worker(QThread):
    """在后台跑体检/安装；stdout/stderr 收成一段文本。"""

    finished_text = Signal(str)

    def __init__(self, mode: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode

    def run(self) -> None:
        buffer = io.StringIO()
        code = 1
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                if self.mode == "doctor":
                    from ..agent.language_doctor import main as doctor_main

                    code = doctor_main([])
                else:
                    from ..agent.lsp_install import main as install_main

                    code = install_main([] if self.mode == "plan" else ["--yes"])
        except Exception as exc:  # noqa: BLE001 — 对话框要显示，不让线程静默死
            buffer.write(f"\n内部错误：{type(exc).__name__}: {exc}\n")
        text = buffer.getvalue().strip() or "（没有输出）"
        self.finished_text.emit(f"{text}\n\n[退出码 {code}]")


class LanguageServersDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("语言服务器")
        self.resize(760, 540)
        self._worker: _Worker | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)
        outer.addWidget(
            QLabel(
                "LSP 的语义查询需要对应语言的服务器。先「检查现状」看缺什么；"
                "「预览计划」只打印命令；「自动安装」会联网调用 npm/rustup/go/dotnet/winget。"
                "安装器不提权、不自动同意许可证。",
                self,
            )
        )
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log.setPlainText(
            "点「检查现状」查看 26 个常见后缀的符号索引/LSP 服务器状态。\n"
            "没有服务器的语言仍可用 repo_map / find_symbol 回退。"
        )
        outer.addWidget(self.log, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.check_button = QPushButton("检查现状", self)
        self.check_button.clicked.connect(lambda: self._start("doctor"))
        row.addWidget(self.check_button)
        self.plan_button = QPushButton("预览安装计划", self)
        self.plan_button.clicked.connect(lambda: self._start("plan"))
        row.addWidget(self.plan_button)
        self.install_button = QPushButton("自动安装…", self)
        self.install_button.clicked.connect(self._install)
        row.addWidget(self.install_button)
        row.addStretch(1)
        close = QPushButton("关闭", self)
        close.clicked.connect(self.accept)
        row.addWidget(close)
        outer.addLayout(row)

        self.status = QLabel("", self)
        outer.addWidget(self.status)

    def _set_busy(self, busy: bool) -> None:
        for button in (self.check_button, self.plan_button, self.install_button):
            button.setEnabled(not busy)
        self.status.setText("正在运行…" if busy else "")

    def _start(self, mode: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if mode == "install" and not self._confirm():
            return
        self._set_busy(True)
        self.log.setPlainText("")
        worker = _Worker(mode, self)
        worker.finished_text.connect(self._on_finished)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        worker.start()

    def _install(self) -> None:
        self._start("install")

    def _confirm(self) -> bool:
        answer = QMessageBox.question(
            self,
            "自动安装语言服务器",
            "会联网调用本机已有的 npm/rustup/go/dotnet/winget，并安装缺失的服务器。\n"
            "不会提权，不会自动同意许可证；可能需要几分钟。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_finished(self, text: str) -> None:
        self.log.setPlainText(text)
        self._set_busy(False)

    def _on_thread_finished(self) -> None:
        self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if self._worker is not None and self._worker.isRunning():
            self.status.setText("安装器还在运行，等它结束再关。")
            event.ignore()
            return
        super().closeEvent(event)
