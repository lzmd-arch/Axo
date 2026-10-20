"""配对窗口：二维码 + 手动码 + 倒计时；以及服务器线程用的配对确认闸门。

配对窗口本身**不代表已经信任**：手机提交 secret 后，主窗口仍会弹一次
"这台设备要配对"的确认框。二维码被拍走的人最多能让确认框出现，不能替用户点头。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..sync.protocol import b64url_decode, encode_manual_code
from ..sync.qr import make_qr_svg
from .glass import GlassHost

__all__ = ["PairApprovalBroker", "SyncPairingDialog"]


class PairApprovalBroker:
    """服务器线程问、界面线程答的**一次性闸门**。

    服务器线程调用 :meth:`ask` 会阻塞到用户点了允许/拒绝或超时；界面线程用
    :meth:`pop_pending` 取走请求并调用 :meth:`resolve`。不把 Qt 信号直接跨线程
    发到窗口：加载/关闭窗口的时序会让"信号到了、窗口没了"变成一次静默的挂起。
    """

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []

    def ask(self, request: dict[str, Any]) -> bool:
        box: dict[str, Any] = {
            "request": dict(request),
            "event": threading.Event(),
            "granted": False,
        }
        with self._lock:
            self._pending.append(box)
        box["event"].wait(self.timeout_s)
        return bool(box["granted"])

    def pop_pending(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._pending:
                return None
            return self._pending.pop(0)

    @staticmethod
    def resolve(box: dict[str, Any], granted: bool) -> None:
        box["granted"] = bool(granted)
        box["event"].set()


class SyncPairingDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        *,
        controller: Any,
        theme: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加手机")
        self.setModal(False)
        self.resize(460, 620)
        self._controller = controller
        self._payload = controller.pairing()
        self._remaining = max(0, int(self._payload.expires_at - time.time()))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        outer.addWidget(QLabel("用手机扫这张二维码，或手动输入下面的信息。"))
        self.qr_label = QLabel(self)
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        svg = make_qr_svg(self._payload.to_qr_json())
        renderer = QSvgRenderer(QByteArray(svg))
        pixmap = QPixmap(300, 300)
        pixmap.fill(Qt.GlobalColor.white)
        from PySide6.QtGui import QPainter

        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        self.qr_label.setPixmap(pixmap)
        outer.addWidget(self.qr_label)

        hosts = " / ".join(self._payload.hosts)
        outer.addWidget(QLabel(f"地址：{hosts}"))
        outer.addWidget(QLabel(f"端口：{self._payload.port}"))
        pin_bytes = b64url_decode(self._payload.pin)
        code = encode_manual_code(self._payload.secret[:16], pin_bytes[:8])
        code_label = QLabel(code)
        code_label.setObjectName("syncManualCode")
        code_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        code_label.setStyleSheet("font-family: monospace; font-size: 15px; letter-spacing: 1px;")
        outer.addWidget(code_label)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        outer.addWidget(
            QLabel("手机提交后，这台电脑还会弹一次确认。二维码有效期很短；不要把屏幕截图发给别人。")
        )
        row = QHBoxLayout()
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        row.addWidget(self.close_button)
        row.addStretch(1)
        outer.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()
        if theme is not None:
            GlassHost(
                self,
                accent=getattr(theme, "accent", "#6b8afd"),
                animated=getattr(theme, "motion", True),
            )

    def _tick(self) -> None:
        if self._remaining <= 0:
            self.note.setText("配对窗口已过期。关闭这个窗口，重新点「添加手机…」。")
            self._timer.stop()
            return
        self.note.setText(f"还剩 {self._remaining} 秒。")
        self._remaining -= 1
