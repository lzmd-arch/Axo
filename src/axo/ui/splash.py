"""启动时的开屏动画。

**它只做一件事：在 ``MainWindow`` 被构造的这几百毫秒里，让屏幕上有一个
"正在醒来"的东西，而不是一片空桌面。** 动画全部是 ``QPainter`` 画出来的，
不碰 :mod:`QtWebEngine`——离屏环境下第一次 ``runJavaScript`` 会让进程无声
消失（记在 ``offscreen-qtwebengine-hard-crash`` 那条里），而启动路径上没有
任何理由去承担那个风险。

**形象资源是可换的，代码不认识具体某一版立绘。** 读下面三个位置，优先级从
高到低：

1. ``assets/splash/character.gif`` / ``.webp``（动图，按原速播放）
2. ``assets/splash/character.png``（静态立绘，做呼吸与浮动）
3. ``axo-icon.png``（内置图标，兜底——资源缺失时开屏仍然成立）

也就是说换一版形象只需要替换文件，不用改这个模块。

**动效开关与主界面共用一份。** ``motion=False`` 时不跑任何定时器：直接画
一帧静态画面、主窗口就绪后立刻关闭。开屏是"启动装饰"，它不该在用户明确
关掉动效之后仍然自己动。
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEventLoop,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QMovie,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from .theme import Resolved, Theme

__all__ = ["SplashScreen", "splash_enabled"]

_ASSET_DIR = Path(__file__).with_name("assets") / "splash"
_MOVIE_NAMES = ("character.gif", "character.webp")
_STILL_NAME = "character.png"
_FALLBACK = Path(__file__).with_name("axo-icon.png")

_WIDTH = 740
_HEIGHT = 430
_RADIUS = 26
_SHADOW = 18
"""窗口四周留给投影的边（画在透明层里，不属于可见底板）。"""

_CHARACTER_BOX = QRectF(30, 12, 360, 414)
"""立绘的包围盒。形象换了也不用动——按 ``KeepAspectRatio`` 装进去。"""

_PREROLL_MS = 650
"""先让开屏自己动这么久，再去构造主窗口。

**为什么要有这一段。** ``MainWindow()`` 是同步的，构造期间事件循环不转，
动画会僵在原地；先跑一个嵌套事件循环把入场与呼吸播出来，主窗口的构造才
不至于把"动画"变成"一张图停了半秒"。这一段的成本是启动慢 0.65 秒，换来
的是开屏真的在动。``motion=False`` 或隔离模式下它是 0。
"""

_MIN_VISIBLE_MS = 900
"""从显示到允许关闭的最短时间。构造很快时也不至于一闪而过。"""


def splash_enabled() -> bool:
    """这次启动要不要开屏。

    **隔离模式（``AXO_ISOLATED_STATE``）一律跳过。** 那是打包自检与冒烟脚本
    的标记（见 ``ui/app.py`` 的同名变量）：那些进程等的是主窗口，多一个顶层
    窗口会让"可见窗口"的探测多一个候选，同时平白多等半秒。``AXO_NO_SPLASH``
    是给排查与录屏留的手动开关。
    """
    return not os.environ.get("AXO_ISOLATED_STATE") and not os.environ.get("AXO_NO_SPLASH")


def _load_character() -> tuple[QPixmap | None, QMovie | None]:
    """按优先级取形象。返回 ``(静态图, 动图)``，两者至多一个非空。"""
    for name in _MOVIE_NAMES:
        path = _ASSET_DIR / name
        if path.is_file():
            movie = QMovie(str(path))
            movie.setCacheMode(QMovie.CacheMode.CacheAll)
            if movie.isValid():
                return None, movie
    for path in (_ASSET_DIR / _STILL_NAME, _FALLBACK):
        if path.is_file():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                return pixmap, None
    return None, None


class SplashScreen(QWidget):
    """无边框、半透明、居中的开屏。"""

    def __init__(
        self,
        *,
        theme: Theme | None = None,
        resolved: Resolved,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(_WIDTH + 2 * _SHADOW, _HEIGHT + 2 * _SHADOW)

        self._theme = theme or Theme()
        self._resolved = resolved
        self._motion = bool(self._theme.motion)
        self._phase = 0.0
        self._intro = 0.0 if self._motion else 1.0
        self._stage = "正在醒来…"
        self._shown_at = 0.0
        self._closing = False

        self._still, self._movie = _load_character()
        if self._movie is not None:
            self._movie.frameChanged.connect(lambda _frame: self.update())
            self._movie.start()

        self._phase_anim = QPropertyAnimation(self, b"phase", self)
        self._phase_anim.setStartValue(0.0)
        self._phase_anim.setEndValue(1.0)
        self._phase_anim.setDuration(5200)
        self._phase_anim.setLoopCount(-1)

        self._intro_anim = QPropertyAnimation(self, b"intro", self)
        self._intro_anim.setStartValue(0.0)
        self._intro_anim.setEndValue(1.0)
        self._intro_anim.setDuration(520)
        self._intro_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(220)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._centre_on_screen()

    # ---------------------------------------------------------------- 动画属性

    def _get_phase(self) -> float:
        return self._phase

    def _set_phase(self, value: float) -> None:
        self._phase = float(value)
        self.update()

    phase = Property(float, _get_phase, _set_phase)

    def _get_intro(self) -> float:
        return self._intro

    def _set_intro(self, value: float) -> None:
        self._intro = float(value)
        self.update()

    intro = Property(float, _get_intro, _set_intro)

    # ---------------------------------------------------------------- 生命周期

    def _centre_on_screen(self) -> None:
        """居中到**光标所在的那块屏**，不是主屏。

        主窗口用的是同一套定位（见 ``ui/window_place.py``）；开屏与主窗口
        落在不同显示器上会像两个程序。
        """
        screen = QGuiApplication.screenAt(self.cursor().pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.center().x() - self.width() // 2,
            area.center().y() - self.height() // 2,
        )

    def showEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        super().showEvent(event)
        self._shown_at = time.monotonic()
        if not self._motion:
            self.setWindowOpacity(1.0)
            self.update()
            return
        self.setWindowOpacity(0.0)
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._intro_anim.start()
        self._phase_anim.start()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if self._movie is not None:
            self._movie.stop()
        self._phase_anim.stop()
        self._intro_anim.stop()
        self._fade.stop()
        super().closeEvent(event)

    def pre_roll(self, milliseconds: int = _PREROLL_MS) -> None:
        """用嵌套事件循环让开屏先动一会儿。

        **只在动画开启时真的等。** 关掉动效的人要的是"别耽误我"：这时它只
        重画一帧，然后立刻返回。
        """
        if not self._motion or milliseconds <= 0:
            self.repaint()
            return
        loop = QEventLoop(self)
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def set_stage(self, text: str) -> None:
        """更新那一行状态句（"正在打开记忆库…"）。"""
        self._stage = text
        self.update()

    def finish(self) -> None:
        """主窗口已经就绪：到最短展示时间后淡出并关闭。"""
        if self._closing:
            return
        self._closing = True
        minimum = _MIN_VISIBLE_MS if self._motion else 0
        remaining = minimum - (time.monotonic() - self._shown_at) * 1000.0
        QTimer.singleShot(max(0, int(remaining)), self._fade_out)

    def _fade_out(self) -> None:
        if not self._motion:
            self.close()
            return
        self._fade.stop()
        # ``_fade_out`` 由 ``_closing`` 保证只走一次，所以这里不需要先断开：
        # 上一次（淡入）没有连接 finished，硬断会打一条 RuntimeWarning。
        self._fade.finished.connect(self.close)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    # ------------------------------------------------------------------- 绘制

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        del event
        tokens = self._resolved.tokens
        accent = QColor(self._resolved.accent)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        plate = QRectF(_SHADOW, _SHADOW, _WIDTH, _HEIGHT)
        self._paint_shadow(painter, plate)
        painter.setPen(QPen(QColor(tokens.border), 1))
        painter.setBrush(QColor(tokens.bg))
        painter.drawRoundedRect(plate, _RADIUS, _RADIUS)

        clip = QPainterPath()
        clip.addRoundedRect(plate, _RADIUS, _RADIUS)
        painter.save()
        painter.setClipPath(clip)
        self._paint_aurora(painter, plate, accent)
        painter.restore()

        self._paint_character(painter, accent)
        self._paint_text(painter, tokens, accent)
        painter.end()

    def _paint_shadow(self, painter: QPainter, plate: QRectF) -> None:
        """假投影：几层逐渐扩散、逐渐变淡的圆角矩形。"""
        painter.setPen(Qt.PenStyle.NoPen)
        for spread, alpha in ((4, 26), (9, 18), (14, 10), (20, 5)):
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(
                plate.adjusted(-spread, -spread + 6, spread, spread + 6),
                _RADIUS + spread,
                _RADIUS + spread,
            )

    def _paint_aurora(self, painter: QPainter, plate: QRectF, accent: QColor) -> None:
        """两团跟随呼吸缓慢变亮的主色光，与主窗口的极光是同一套语言。"""
        glow = 0.5 + 0.5 * math.sin(self._phase * 2.0 * math.pi)
        left = QColor(accent)
        left.setAlpha(int(34 + 22 * glow))
        gradient = QRadialGradient(
            plate.left() + plate.width() * 0.18,
            plate.top() + plate.height() * 0.22,
            plate.width() * 0.62,
        )
        gradient.setColorAt(0.0, left)
        transparent = QColor(accent)
        transparent.setAlpha(0)
        gradient.setColorAt(1.0, transparent)
        painter.fillRect(plate, gradient)

        right = QColor(accent)
        right.setAlpha(int(22 + 14 * (1.0 - glow)))
        gradient = QRadialGradient(
            plate.right() - plate.width() * 0.10,
            plate.bottom() - plate.height() * 0.12,
            plate.width() * 0.52,
        )
        gradient.setColorAt(0.0, right)
        gradient.setColorAt(1.0, transparent)
        painter.fillRect(plate, gradient)

    def _paint_character(self, painter: QPainter, accent: QColor) -> None:
        breathe = math.sin(self._phase * 2.0 * math.pi)
        # 呼吸：上下 3px、缩放 1.5%。幅度再大就不是"呼吸"了，是"漂浮"。
        centre = _CHARACTER_BOX.center()
        scale = 1.0 + 0.015 * breathe * self._intro
        offset_y = (1.0 - self._intro) * 14.0 + breathe * 3.0

        glow = QRadialGradient(centre.x(), centre.y() + offset_y, _CHARACTER_BOX.width() * 0.62)
        halo = QColor(accent)
        halo.setAlpha(int((30 + 22 * (0.5 + 0.5 * breathe)) * self._intro))
        glow.setColorAt(0.0, halo)
        edge = QColor(accent)
        edge.setAlpha(0)
        glow.setColorAt(1.0, edge)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(centre, _CHARACTER_BOX.width() * 0.62, _CHARACTER_BOX.height() * 0.62)

        source: QPixmap | None = (
            self._movie.currentPixmap() if self._movie is not None else self._still
        )
        if source is None or source.isNull():
            return
        box = _CHARACTER_BOX.adjusted(
            _CHARACTER_BOX.width() * (1.0 - scale) / 2.0,
            _CHARACTER_BOX.height() * (1.0 - scale) / 2.0 + offset_y,
            -_CHARACTER_BOX.width() * (1.0 - scale) / 2.0,
            -_CHARACTER_BOX.height() * (1.0 - scale) / 2.0 + offset_y,
        )
        size = source.size().scaled(box.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF(
            box.center().x() - size.width() / 2.0,
            box.center().y() - size.height() / 2.0,
            size.width(),
            size.height(),
        )
        painter.setOpacity(self._intro)
        painter.drawPixmap(target, source, QRectF(source.rect()))
        painter.setOpacity(1.0)

    def _paint_text(self, painter: QPainter, tokens, accent: QColor) -> None:
        painter.setOpacity(self._intro)
        x = _SHADOW + 366
        painter.setPen(QColor(tokens.fg_strong))
        title = QFont(self.font())
        title.setPointSize(30)
        title.setBold(True)
        painter.setFont(title)
        painter.drawText(QRectF(x, _SHADOW + 136, 300, 46), Qt.AlignmentFlag.AlignLeft, "Axo")

        painter.setPen(QColor(tokens.fg_muted))
        subtitle = QFont(self.font())
        subtitle.setPointSize(11)
        painter.setFont(subtitle)
        painter.drawText(
            QRectF(x, _SHADOW + 183, 300, 24),
            Qt.AlignmentFlag.AlignLeft,
            "阿螈 · 记忆终端",
        )

        stage = QFont(self.font())
        stage.setPointSize(10)
        painter.setFont(stage)
        painter.setPen(QColor(tokens.fg))
        painter.drawText(QRectF(x, _SHADOW + 244, 300, 22), Qt.AlignmentFlag.AlignLeft, self._stage)

        self._paint_dots(painter, x, _SHADOW + 292, accent)

        version = QFont(self.font())
        version.setPointSize(8)
        painter.setFont(version)
        painter.setPen(QColor(tokens.fg_muted))
        painter.drawText(
            QRectF(_SHADOW, _SHADOW + _HEIGHT - 34, _WIDTH - 28, 20),
            Qt.AlignmentFlag.AlignRight,
            self._version_text(),
        )
        painter.setOpacity(1.0)

    def _paint_dots(self, painter: QPainter, x: float, y: float, accent: QColor) -> None:
        """三点涟漪。静止时也留三个淡点——那读作"在等"，不是"画错了"。"""
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(3):
            pulse = max(
                0.0,
                math.sin((self._phase - index * 0.08) * 2.0 * math.pi * 1.5),
            )
            colour = QColor(accent)
            colour.setAlpha(int(70 + 150 * pulse) if self._motion else 110)
            painter.setBrush(colour)
            painter.drawEllipse(QRectF(x + index * 15, y, 6, 6))

    @staticmethod
    def _version_text() -> str:
        from .. import __version__

        return f"v{__version__}"
