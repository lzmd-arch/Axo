"""Qt 外壳的液态玻璃底：**一整块缓慢流动的极光**。

网页那侧的玻璃靠 ``backdrop-filter``；Qt 这边没有，于是"背后有东西可糊"这件事要
自己造：一块铺满窗口、画着三团低透明度径向渐变的背景，位置随时间缓慢漂移。它上面
的顶栏、列表、输入框都是半透明的，于是整块界面读起来是"玻璃浮在有颜色的东西上"，
而不是"扁平色块上贴了几个圆角按钮"。

**只在可见且允许动效时计时。** 一个隐藏窗口还在 60ms 重绘一次，笔记本上就是白白
烧电；``set_animated(False)`` 时停表但**保留当前画面**——静止的玻璃仍然是玻璃。
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

__all__ = ["AuroraBackdrop"]


class AuroraBackdrop(QWidget):
    """铺满父控件的极光背景。**不吃鼠标事件。**"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._accent = QColor("#6b8afd")
        self._phase = 0.0
        self._animated = True
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._advance)
        if self.isVisible():
            self._timer.start()

    def set_accent(self, color: str, *, animated: bool) -> None:
        """换主题或动效开关。**画面立即更新**，不等下一帧定时器。"""
        candidate = QColor(color)
        if candidate.isValid():
            self._accent = candidate
        self._animated = bool(animated)
        self._sync_timer()
        self.update()

    def set_animated(self, animated: bool) -> None:
        self._animated = bool(animated)
        self._sync_timer()

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _sync_timer(self) -> None:
        if self._animated and self.isVisible():
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        self._phase += 0.022
        self.update()

    def showEvent(self, event: object) -> None:  # noqa: N802 — Qt 命名
        super().showEvent(event)  # type: ignore[arg-type]
        self._sync_timer()

    def hideEvent(self, event: object) -> None:  # noqa: N802
        super().hideEvent(event)  # type: ignore[arg-type]
        self._timer.stop()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        width = max(1, self.width())
        height = max(1, self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        blobs = (
            # (中心 x 比例, 中心 y 比例, 半径比例, alpha, 相位偏移)
            (0.12, 0.08, 0.56, 150, 0.0),
            (0.50, 0.02, 0.46, 110, 1.0),
            (0.88, 0.18, 0.56, 132, 2.1),
            (0.55, 0.92, 0.66, 96, 4.2),
        )
        for fx, fy, radius, alpha, offset in blobs:
            # 漂移幅度很小：它是背景，不是动画主角。
            cx = (fx + 0.035 * math.sin(self._phase + offset)) * width
            cy = (fy + 0.025 * math.cos(self._phase * 0.8 + offset)) * height
            reach = radius * max(width, height)
            color = QColor(self._accent)
            color.setAlpha(alpha)
            gradient = QRadialGradient(cx, cy, reach)
            gradient.setColorAt(0.0, color)
            color_end = QColor(self._accent)
            color_end.setAlpha(0)
            gradient.setColorAt(1.0, color_end)
            painter.fillRect(self.rect(), gradient)
        painter.end()
