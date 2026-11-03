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
    QUrl,
    Signal,
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

from .boot import BootSequence
from .boot_checks import BootCheck
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

_CONFIRM_TIMEOUT_MS = 5000
"""确认幕最多等这么久。用户走开时启动不能永远停在开屏。"""

_SKIP_RECT = QRectF(_SHADOW + _WIDTH - 118, _SHADOW + 22, 96, 30)
"""画在右上角的「跳过」热区（窗口坐标）。"""

_BOOT_TICK_MS = 16
""""进度条"的重画间隔。只为动画平滑，不参与任何判定。"""


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

    closed = Signal()
    """窗口真正关掉（含淡出结束）。首启向导/搬迁对话框要等它之后再弹。"""

    def __init__(
        self,
        *,
        theme: Theme | None = None,
        resolved: Resolved,
        parent: QWidget | None = None,
        boot: BootSequence | None = None,
        boot_sound: bool = False,
        allow_skip: bool = True,
        require_confirm: bool = True,
        speed: float = 1.0,
        boot_theme: str = "warm",
        checks: tuple[BootCheck, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(_WIDTH + 2 * _SHADOW, _HEIGHT + 2 * _SHADOW)

        self._theme = theme or Theme()
        self._resolved = resolved
        self._motion = bool(self._theme.motion)
        self._phase = 0.0
        self._intro = 0.0 if self._motion else 1.0
        self._stage = "正在醒来…"
        self._shown_at = 0.0
        self._closing = False
        # ── 六幕启动（可选） ────────────────────────────────────────────────
        self._boot = boot if (boot is not None and self._motion) else None
        self._boot_index = 0
        self._boot_progress = 0.0
        self._boot_running = False
        self._boot_done = self._boot is None
        self._boot_sound = bool(boot_sound)
        self._boot_allow_skip = bool(allow_skip)
        self._boot_require_confirm = bool(require_confirm)
        self._boot_speed = max(0.5, min(2.0, float(speed)))
        self._boot_theme = boot_theme if boot_theme in ("warm", "dark") else "warm"
        # 真实自检结果。**空集不是"全绿"**——面板写"未检查"，绝不假装通过。
        self._boot_checks: dict[str, BootCheck] = {check.key: check for check in checks}
        # 六幕开着但结果还没送到：面板显示"检测中…"，而不是"未检查"或假 ✓。
        self._boot_checks_pending = self._boot is not None and not checks
        self._boot_confirm = False
        self._boot_skipped = False
        self._boot_waiting = False
        self._boot_phase = 0.0
        self._boot_loop: QEventLoop | None = None
        self._sound = None
        # 六幕要收键盘（Enter/Esc），所以**不**用"显示但不激活"。单段开屏仍然不抢焦点。
        if self._boot is None or not self._motion:
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

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

        if self._boot is not None and self._motion:
            screen = (
                QGuiApplication.screenAt(self.cursor().pos()) or QGuiApplication.primaryScreen()
            )
            if screen is not None:
                self.setFixedSize(screen.geometry().size())
                self.move(screen.geometry().topLeft())
        else:
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
        if self._sound is not None:
            self._sound.stop()
        self._phase_anim.stop()
        self._intro_anim.stop()
        self._fade.stop()
        self.closed.emit()
        super().closeEvent(event)

    def pre_roll(self, milliseconds: int = _PREROLL_MS) -> None:
        """把六幕**全部播完**再返回。

        用户要的是"每一幕结束之后才进主界面"：所以这里阻塞到序列结束（或跳过），
        主窗口之后才构造。确认幕最多等 5 秒；跳过随时可用（设置里关掉跳过时，
        仍然靠确认超时兜底）。
        """
        if not self._motion or milliseconds <= 0:
            self.repaint()
            return
        if self._boot is not None:
            self._run_boot_sequence()
            return
        loop = QEventLoop(self)
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    # ── 六幕启动 ────────────────────────────────────────────────────────────

    def _run_boot_sequence(self) -> None:
        assert self._boot is not None
        self._boot_running = True
        self._boot_skipped = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.activateWindow()
        self.setFocus()
        try:
            for index, act in enumerate(self._boot.acts):
                if self._boot_skipped:
                    break
                self._boot_index = index
                self._boot_progress = 0.0
                self._boot_confirm = False
                self._boot_waiting = False
                self._play_chime()
                self._wait_boot_act(act, confirm=act.confirm and self._boot_require_confirm)
            self._boot_done = True
        finally:
            self._boot_running = False
            self._boot_waiting = False
            self._boot_loop = None
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _wait_boot_act(self, act, *, confirm: bool = False) -> None:  # noqa: ANN001 — BootAct
        """嵌套事件循环里播完一幕；确认幕再等按键，最多 5 秒。"""
        duration = max(120, int(act.duration_ms / self._boot_speed))
        total = duration + (_CONFIRM_TIMEOUT_MS if confirm else 0)
        loop = QEventLoop(self)
        self._boot_loop = loop
        state = {"elapsed": 0}

        def tick() -> None:
            state["elapsed"] += _BOOT_TICK_MS
            # **动效有自己的时钟**：等待确认时进度条停住，但准星/掠扫/呼吸不能停。
            self._boot_phase += _BOOT_TICK_MS / 1000.0
            self._boot_progress = min(1.0, state["elapsed"] / duration)
            if confirm and state["elapsed"] >= duration:
                self._boot_waiting = True
            self.update()
            if self._boot_skipped or self._boot_confirm:
                loop.quit()
                return
            if confirm:
                if state["elapsed"] >= total:
                    loop.quit()
            elif state["elapsed"] >= duration:
                loop.quit()

        timer = QTimer(self)
        timer.setInterval(_BOOT_TICK_MS)
        timer.timeout.connect(tick)
        timer.start()
        loop.exec()
        timer.stop()
        self._boot_loop = None
        self._boot_confirm = False
        self._boot_waiting = False

    def _confirm_boot(self) -> None:
        if not self._boot_running or self._boot is None:
            return
        if self._boot_index >= len(self._boot.acts):
            return
        if self._boot.acts[self._boot_index].confirm:
            self._boot_confirm = True
            if self._boot_loop is not None:
                self._boot_loop.quit()

    def _skip_boot(self) -> None:
        if not self._boot_running or not self._boot_allow_skip:
            return
        self._boot_skipped = True
        if self._boot_loop is not None:
            self._boot_loop.quit()
        else:
            self._boot_index = len(self._boot.acts) if self._boot is not None else 0
            self._boot_done = True

    def _play_chime(self) -> None:
        if not self._boot_sound:
            return
        try:
            if self._sound is None:
                from PySide6.QtMultimedia import QMediaDevices, QSoundEffect

                from .boot_sound import ensure_chime

                if QMediaDevices.defaultAudioOutput().isNull():
                    self._boot_sound = False
                    return
                path = ensure_chime()
                if path is None:
                    self._boot_sound = False
                    return
                self._sound = QSoundEffect(self)
                self._sound.setSource(QUrl.fromLocalFile(str(path)))
                self._sound.setVolume(0.35)
            self._sound.play()
        except Exception:  # noqa: BLE001 —— 没有音频不该拦住启动
            self._boot_sound = False
            self._sound = None

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if self._boot_running:
            if event.key() == Qt.Key.Key_Escape:
                self._skip_boot()
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                self._confirm_boot()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if self._boot_running:
            if _SKIP_RECT.contains(event.position()) and self._boot_allow_skip:
                self._skip_boot()
                return
            self._confirm_boot()
            return
        super().mousePressEvent(event)

    def set_stage(self, text: str) -> None:
        """更新那一行状态句（"正在打开记忆库…"）。"""
        self._stage = text
        self.update()

    def set_checks(self, checks: tuple[BootCheck, ...]) -> None:
        """装入真实自检结果。必须在 :meth:`pre_roll` 之前调用。

        app 要先定下记忆库位置（``boot_data_root``）才能检测库，而开屏必须更早
        出现，所以是"先建窗、后填结果"两步。
        """
        self._boot_checks = {check.key: check for check in checks}
        self._boot_checks_pending = False
        self.update()

    def _self_check_warning(self) -> str:
        """有失败项时给一句醒目的汇总；全过就什么都不说。"""
        failed = [check for check in self._boot_checks.values() if not check.ok]
        return f"{len(failed)} 项自检未通过" if failed else ""

    def finish(self) -> None:
        """主窗口已经就绪：继续播剩下的幕，或到最短展示时间后淡出关闭。"""
        if self._closing or self._boot_running:
            return
        self._boot_done = True
        self._schedule_close()

    def _schedule_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        minimum = _MIN_VISIBLE_MS if self._motion else 0
        remaining = minimum - (time.monotonic() - self._shown_at) * 1000.0
        QTimer.singleShot(max(0, int(remaining)), self._fade_out)

    def has_pending_boot(self) -> bool:
        """``finish()`` 之后还会在屏幕上继续播六幕吗（对话框据此延后）。"""
        return self._boot is not None and not self._boot_done

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

        if self._boot is not None:
            if self._boot_done:
                self._paint_boot_hold(painter, accent)
            else:
                self._paint_boot_screen(painter, accent)
            painter.end()
            return

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

    # ── 六幕的绘制 ──────────────────────────────────────────────────────────

    # ── 六幕的全屏绘制 ──────────────────────────────────────────────────────

    def _paint_boot_hold(self, painter: QPainter, accent: QColor) -> None:
        """六幕结束、主窗口还在构造：保持同一块暗色底，**不再回小卡片**。"""
        width = float(self.width())
        height = float(self.height())
        bg, fg, muted = QColor("#080a0d"), QColor("#eef3f1"), QColor("#7d8785")
        painter.fillRect(QRectF(0, 0, width, height), bg)
        glow = QRadialGradient(width * 0.5, height * 0.40, max(width, height) * 0.6)
        halo = QColor(accent)
        halo.setAlpha(30)
        glow.setColorAt(0.0, halo)
        clear = QColor(accent)
        clear.setAlpha(0)
        glow.setColorAt(1.0, clear)
        painter.fillRect(QRectF(0, 0, width, height), glow)
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
        for y in range(0, int(height), 4):
            painter.drawLine(0, y, int(width), y)

        cx, cy = width * 0.5, height * 0.40
        radius = min(width, height) * 0.13
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(accent, 2))
        painter.drawArc(
            QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
            int(self._boot_phase * 70 % 360) * 16,
            250 * 16,
        )
        inner = QColor(accent)
        inner.setAlpha(70)
        painter.setPen(QPen(inner, 1))
        painter.drawEllipse(
            QRectF(cx - radius * 0.78, cy - radius * 0.78, radius * 1.56, radius * 1.56)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(QRectF(cx - 3, cy - 3, 6, 6))

        if self._still is not None and not self._still.isNull():
            target_h = height * 0.34
            scale = target_h / self._still.height()
            target_w = self._still.width() * scale
            painter.setOpacity(0.5)
            painter.drawPixmap(
                QRectF(cx - radius * 2.3 - target_w / 2, cy - target_h / 2, target_w, target_h),
                self._still,
                QRectF(self._still.rect()),
            )
            painter.setOpacity(1.0)

        mono = QFont("Consolas")
        mono.setPointSize(9)
        mono.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.0)
        painter.setFont(mono)
        painter.setPen(muted)
        painter.drawText(QRectF(28, 22, 420, 20), Qt.AlignmentFlag.AlignLeft, "AXO // READY")
        warning = self._self_check_warning()
        if warning:
            painter.setPen(QColor("#ff6b5e"))
            painter.drawText(
                QRectF(width - 448, 22, 420, 20),
                Qt.AlignmentFlag.AlignRight,
                f"! {warning}",
            )

        stage_font = QFont(self.font())
        stage_font.setPointSize(int(max(18, min(34, min(width, height) * 0.028))))
        painter.setFont(stage_font)
        painter.setPen(fg)
        painter.drawText(
            QRectF(width * 0.12, cy + radius + 48, width * 0.76, 48),
            Qt.AlignmentFlag.AlignHCenter,
            self._stage,
        )

        bar_w = min(width * 0.42, 620)
        bar = QRectF((width - bar_w) / 2, height - 66, bar_w, 3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(fg.red(), fg.green(), fg.blue(), 40))
        painter.drawRoundedRect(bar, 1.5, 1.5)
        painter.setBrush(accent)
        painter.drawRoundedRect(bar, 1.5, 1.5)

    def _paint_boot_screen(self, painter: QPainter, accent: QColor) -> None:
        """全屏六幕：暗色底、环形准星、扫描线、打字机与故障揭幕。"""
        assert self._boot is not None
        width = float(self.width())
        height = float(self.height())
        act = self._boot.acts[min(self._boot_index, len(self._boot.acts) - 1)]
        progress = self._boot_progress
        # **整体暗色**：六幕共用一套炭黑底，不再切暖白。
        bg, fg, muted = QColor("#080a0d"), QColor("#eef3f1"), QColor("#7d8785")

        painter.fillRect(QRectF(0, 0, width, height), bg)
        glow = QRadialGradient(width * 0.5, height * 0.40, max(width, height) * 0.6)
        halo = QColor(accent)
        pulse = 0.75 + 0.25 * math.sin(self._boot_phase * 2.0 * math.pi * 0.5)
        halo.setAlpha(int(34 * pulse))
        glow.setColorAt(0.0, halo)
        clear = QColor(accent)
        clear.setAlpha(0)
        glow.setColorAt(1.0, clear)
        painter.fillRect(QRectF(0, 0, width, height), glow)

        # 扫描线：每 3 px 一条，极淡；它是"仪器感"最便宜的一层。
        scan = QColor(255, 255, 255, 10)
        painter.setPen(QPen(scan, 1))
        for y in range(0, int(height), 4):
            painter.drawLine(0, y, int(width), y)

        # 暗角
        vignette = QRadialGradient(width * 0.5, height * 0.45, max(width, height) * 0.78)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 110))
        painter.fillRect(QRectF(0, 0, width, height), vignette)

        cx, cy = width * 0.5, height * 0.40
        radius = min(width, height) * 0.13
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(accent, 2))
        start = int((self._boot_index * 47 + self._boot_phase * 70) % 360)
        painter.drawArc(
            QRectF(cx - radius, cy - radius, radius * 2, radius * 2), start * 16, 250 * 16
        )
        inner = QColor(accent)
        inner.setAlpha(80)
        painter.setPen(QPen(inner, 1))
        painter.drawEllipse(
            QRectF(cx - radius * 0.78, cy - radius * 0.78, radius * 1.56, radius * 1.56)
        )
        for index in range(12):
            angle = math.radians(index * 30 + start * 0.2)
            inner_r = radius * 1.08
            outer_r = radius * (1.16 if index % 3 == 0 else 1.12)
            painter.drawLine(
                cx + inner_r * math.cos(angle),
                cy + inner_r * math.sin(angle),
                cx + outer_r * math.cos(angle),
                cy + outer_r * math.sin(angle),
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(QRectF(cx - 3, cy - 3, 6, 6))

        # 一道扫过整屏的细光——它是"系统在跑"的唯一持续运动。
        scan_y = ((self._boot_phase * 0.16) % 1.0) * height
        band = QColor(accent)
        band.setAlpha(14)
        painter.fillRect(QRectF(0, scan_y - 16, width, 32), band)
        line = QColor(accent)
        line.setAlpha(60)
        painter.setPen(QPen(line, 1))
        painter.drawLine(0, int(scan_y), int(width), int(scan_y))

        # ── 脉冲：环上的发放、扩散的脉冲环、底部一条发放轨迹 ────────────────
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index in range(36):
            angle = math.radians(index * 10)
            fire = 0.5 + 0.5 * math.sin(self._boot_phase * 6.0 + index * 2.399)
            if fire < 0.72:
                continue
            glow_alpha = int(40 + 160 * (fire - 0.72) / 0.28)
            colour = QColor(accent)
            colour.setAlpha(glow_alpha)
            painter.setPen(QPen(colour, 2))
            inner_r = radius * 1.06
            outer_r = radius * (1.12 + 0.18 * (fire - 0.72) / 0.28)
            painter.drawLine(
                cx + inner_r * math.cos(angle),
                cy + inner_r * math.sin(angle),
                cx + outer_r * math.cos(angle),
                cy + outer_r * math.sin(angle),
            )

        beat = (self._boot_phase % 1.4) / 1.4
        ring_colour = QColor(accent)
        ring_colour.setAlpha(int(56 * (1.0 - beat)))
        painter.setPen(QPen(ring_colour, 1))
        ring_r = radius * (0.7 + 1.7 * beat)
        painter.drawEllipse(QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2))

        trace_w = min(width * 0.46, 680)
        trace_x = (width - trace_w) / 2
        trace_y = height - 128
        painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 70), 1))
        steps = 72
        for index in range(steps):
            x = trace_x + trace_w * index / (steps - 1)
            value = (
                math.sin(index * 1.7 + self._boot_phase * 9.0)
                + math.sin(index * 3.1 - self._boot_phase * 5.0)
            ) / 2.0
            spike_height = max(0.0, value) ** 3 * 22.0
            painter.drawLine(int(x), trace_y, int(x), int(trace_y - spike_height))

        # 角标与跳过
        mono = QFont("Consolas")
        mono.setPointSize(9)
        mono.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.0)
        painter.setFont(mono)
        painter.setPen(muted)
        painter.drawText(
            QRectF(28, 22, 420, 20),
            Qt.AlignmentFlag.AlignLeft,
            f"AXO // BOOT {self._boot_index + 1:02d}/{len(self._boot.acts):02d}",
        )
        if self._boot_allow_skip:
            skip = QRectF(width - 152, 18, 124, 30)
            painter.setPen(QPen(muted, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(skip, 14, 14)
            painter.drawText(skip, Qt.AlignmentFlag.AlignCenter, "跳过  ESC")

        # 标题与副标题
        painter.setOpacity(min(1.0, 0.15 + progress * 2.6))
        title_size = int(max(30, min(64, min(width, height) * 0.052)))
        title_font = QFont(self.font())
        title_font.setPointSize(title_size)
        title_font.setBold(True)
        painter.setFont(title_font)
        text_top = cy + radius + 52
        title_rect = QRectF(width * 0.12, text_top, width * 0.76, title_size * 1.6)
        if act.kind == "glitch" and progress < 0.6:
            for dx, colour in ((-6, QColor(255, 59, 48, 110)), (6, QColor(41, 113, 184, 110))):
                painter.setPen(colour)
                painter.drawText(
                    title_rect.adjusted(dx, 0, dx, 0),
                    Qt.AlignmentFlag.AlignHCenter,
                    act.title,
                )
        painter.setPen(fg)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter, act.title)

        subtitle_font = QFont("Consolas")
        subtitle_font.setPointSize(10)
        subtitle_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        painter.setFont(subtitle_font)
        painter.setPen(muted)
        subtitle = act.subtitle
        if act.kind == "type":
            subtitle = subtitle[: max(0, int(len(subtitle) * min(1.0, progress * 1.7)))]
        painter.drawText(
            QRectF(width * 0.12, text_top + title_size * 2.0, width * 0.76, 26),
            Qt.AlignmentFlag.AlignHCenter,
            subtitle,
        )

        if self._boot_waiting:
            painter.setPen(QColor(accent))
            painter.drawText(
                QRectF(width * 0.12, text_top + title_size * 2.55, width * 0.76, 24),
                Qt.AlignmentFlag.AlignHCenter,
                "按 Enter 或点击继续",
            )

        # 中段：档案面板显示**真实自检结果**（app 在启动前跑好传进来）
        if act.kind == "glass" or act.id in ("permission", "identity"):
            if act.id == "identity":
                keys = (("identity", "设备识别"), ("password", "启动口令"))
            elif act.kind == "glass":
                keys = (("data", "数据目录"), ("archive", "记忆库"), ("persona", "人格锚"))
            else:
                keys = (("backend", "后端"), ("scope", "数据边界"), ("tools", "工具沙箱"))
            panel = QRectF(cx - 300, text_top + title_size * 2.65, 600, 150)
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 90), 1))
            panel_bg = QColor(255, 255, 255, 10)
            painter.setBrush(panel_bg)
            painter.drawRoundedRect(panel, 12, 12)
            panel_font = QFont(self.font())
            panel_font.setPointSize(10)
            painter.setFont(panel_font)
            metrics = painter.fontMetrics()
            for index, (key, fallback) in enumerate(keys):
                check = self._boot_checks.get(key)
                done = progress >= (index + 0.5) / len(keys)
                y = panel.top() + 18 + index * 30
                if check is None:
                    pending = self._boot_checks_pending
                    mark, label = "·", fallback
                    detail = "检测中…" if pending else "未检查"
                    colour = muted
                else:
                    mark = "✓" if check.ok else "✗"
                    label, detail = check.label, check.detail
                    colour = QColor(accent) if check.ok else QColor("#ff6b5e")
                painter.setPen(colour if done else muted)
                painter.drawText(
                    QRectF(panel.left() + 18, y, 24, 22),
                    Qt.AlignmentFlag.AlignLeft,
                    mark if done else "·",
                )
                painter.setPen(fg if done else muted)
                painter.drawText(
                    QRectF(panel.left() + 46, y, 120, 22),
                    Qt.AlignmentFlag.AlignLeft,
                    label,
                )
                if done:
                    elided = metrics.elidedText(detail, Qt.TextElideMode.ElideMiddle, 380)
                    painter.setPen(colour if (check is not None and not check.ok) else muted)
                    painter.drawText(
                        QRectF(panel.left() + 185, y, 390, 22),
                        Qt.AlignmentFlag.AlignRight,
                        elided,
                    )

        painter.setOpacity(1.0)
        # **立绘只在终幕出现**，在准星左侧淡入。
        if act.kind == "hero" and self._still is not None and not self._still.isNull():
            target_h = height * 0.42
            scale = target_h / self._still.height()
            target_w = self._still.width() * scale
            char_cx = cx - radius * 2.6
            painter.setOpacity(min(1.0, progress * 2.6) * 0.95)
            painter.drawPixmap(
                QRectF(char_cx - target_w / 2, cy - target_h / 2, target_w, target_h),
                self._still,
                QRectF(self._still.rect()),
            )
            painter.setOpacity(1.0)

        # 底部进度与幕点
        bar_w = min(width * 0.42, 620)
        bar = QRectF((width - bar_w) / 2, height - 66, bar_w, 3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(fg.red(), fg.green(), fg.blue(), 40))
        painter.drawRoundedRect(bar, 1.5, 1.5)
        painter.setBrush(accent)
        painter.drawRoundedRect(
            QRectF(bar.left(), bar.top(), bar.width() * progress, bar.height()), 1.5, 1.5
        )
        dots = len(self._boot.acts)
        for index in range(dots):
            x = width / 2 - (dots - 1) * 9 + index * 18
            painter.setBrush(
                accent if index <= self._boot_index else QColor(fg.red(), fg.green(), fg.blue(), 60)
            )
            painter.drawEllipse(QRectF(x - 3, height - 92, 6, 6))

        # 搬迁/自检期间的状态句。六幕画面原本不画 ``_stage``，于是大库搬迁时
        # 第一幕看上去像卡死；这一行至少让“还在动”有字可读。
        if self._stage:
            painter.setFont(mono)
            painter.setPen(muted)
            painter.drawText(
                QRectF(width * 0.12, height - 146, width * 0.76, 18),
                Qt.AlignmentFlag.AlignHCenter,
                self._stage,
            )

        # 左下角启动账：**每一行都是刚查出来的事实**，失败项带 !，不写"正在加载"。
        if self._boot_checks:
            logs = tuple(
                f"> {check.key}: {check.detail}" if check.ok else f"> ! {check.key}: {check.detail}"
                for check in self._boot_checks.values()
            )
        elif self._boot_checks_pending:
            logs = ("> checks: running…",)
        else:
            # 还没填结果时**什么都不写**：不写 "checks: not run" 这种会被当成失败的句子。
            logs = ()
        painter.setFont(mono)
        shown = logs[: min(len(logs), self._boot_index + 1)]
        for index, line in enumerate(shown):
            painter.setPen(muted if not line.startswith("> !") else QColor("#ff6b5e"))
            elided = painter.fontMetrics().elidedText(line, Qt.TextElideMode.ElideMiddle, 420)
            painter.drawText(
                QRectF(28, height - 118 + index * 18, 420, 18),
                Qt.AlignmentFlag.AlignLeft,
                elided,
            )
        warning = self._self_check_warning()
        if warning:
            painter.setPen(QColor("#ff6b5e"))
            painter.drawText(
                QRectF(width - 448, height - 118, 420, 18),
                Qt.AlignmentFlag.AlignRight,
                f"! {warning}",
            )

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
            "千岁（阿螈）· 记忆终端",
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
