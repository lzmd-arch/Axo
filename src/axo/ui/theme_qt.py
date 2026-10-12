"""主题的 Qt 绑定层：**这个包里唯一把配色接到 Qt 上的地方。**

:mod:`.color` / :mod:`.theme` / :mod:`.layout` / :mod:`.settings_store` 都是纯
函数，可以在无 Qt 的测试里逐条断言。而"这些颜色到底有没有真的画到窗口上"只能靠
Qt 回答，所以那部分全部收在这一个文件里，接口只有三个：
:func:`os_color_scheme` / :func:`build_palette` / :func:`apply_theme`。

**样式表与调色板要一起给，缺一不可。** 样式表管的是"我们写明了的地方"（顶栏、
列表、按钮……），调色板管的是"我们没写明的地方"——滚动条、下拉箭头、勾选框指示器、
文件对话框、消息框。只给样式表的话，那些控件会继续用默认的浅色调色板，在暗色界面
上就是几块刺眼的白；只给调色板的话，自绘的那部分完全没有主题。两样都给之后，
界面才是一个整体。

**样式表设在 ``QApplication`` 上，不逐窗口设。** 记忆窗口是懒建、而且是独立顶层
窗口；应用级样式表自动覆盖它，不必逐窗接线，换主题也只重设一次。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from .theme import (
    MODE_DARK,
    MODE_LIGHT,
    Resolved,
    Theme,
    build_qss,
    resolve,
)

__all__ = [
    "QSettingsBackend",
    "apply_theme",
    "build_palette",
    "os_color_scheme",
]

#: 问不到系统偏好时的答案。:func:`~axo.ui.theme.effective_mode` 只认
#: ``"light"``，所以这个值事实上等于"按暗色走"——本项目的默认观感就是暗色。
UNKNOWN = "unknown"


class QSettingsBackend:
    """把 ``QSettings`` 适配成 :class:`~.settings_store.SettingsBackend`。

    ``QSettings`` 自己就符合那个 Protocol（``value`` / ``setValue`` 两个方法都在），
    所以这个类其实只做一件事：**把那个事实写下来**。有了它，``settings_store``
    那一侧的测试就不用碰真实注册表，而这边也不需要为了可测性把 ``QSettings``
    包一层假接口。
    """

    def __init__(self, settings: object) -> None:
        self._settings = settings

    def value(self, key: str, default: object = None) -> object:
        return self._settings.value(key, default)  # type: ignore[attr-defined]

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 (Qt 命名)
        self._settings.setValue(key, value)  # type: ignore[attr-defined]


def os_color_scheme() -> str:
    """系统当前是明还是暗：``"dark"`` / ``"light"`` / ``"unknown"``。

    ``Qt 6.5+`` 才有 ``QStyleHints.colorScheme()``；本仓 ``PySide6 6.12`` 可用。
    它可能在三种情况下给不出答案——平台不支持、旧版本、以及某些 Linux 桌面环境，
    所以调用方**必须**处理 ``"unknown"``，而不是假定它只返回明暗之一。

    **两侧的明暗判定都以这个函数为唯一来源。** 网页侧不再用浏览器自己的
    ``prefers-color-scheme``：靠"Chromium 的检测恰好与 Qt 的判定一致"是脆的，
    而显式注入让两侧**可证明地**一致。
    """
    scheme = QGuiApplication.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Light:
        return MODE_LIGHT
    if scheme == Qt.ColorScheme.Dark:
        return MODE_DARK
    return UNKNOWN


def build_palette(resolved: Resolved) -> QPalette:
    """按解析后的主题造一个 ``QPalette``，管住样式表没写明的那部分。

    这里逐组都要给**禁用态**，否则 Qt 会自己把正常态算淡一档——在暗色底上它算
    出来的是一个更亮的灰，禁用按钮看起来反而比可用按钮显眼。
    """
    tokens = resolved.tokens
    roles = resolved.roles

    bg = QColor(tokens.bg)
    elevated = QColor(tokens.bg_elevated)
    sunken = QColor(tokens.bg_sunken)
    fg = QColor(tokens.fg)
    muted = QColor(tokens.fg_muted)
    accent = QColor(roles["accent"])
    accent_fg = QColor(roles["accentFg"])

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, fg)
    palette.setColor(QPalette.ColorRole.Base, sunken)
    palette.setColor(QPalette.ColorRole.AlternateBase, elevated)
    palette.setColor(QPalette.ColorRole.Text, fg)
    palette.setColor(QPalette.ColorRole.Button, elevated)
    palette.setColor(QPalette.ColorRole.ButtonText, fg)
    palette.setColor(QPalette.ColorRole.ToolTipBase, elevated)
    palette.setColor(QPalette.ColorRole.ToolTipText, fg)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, accent_fg)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(tokens.danger))

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, muted)
    return palette


def apply_theme(app: QApplication, theme: Theme, os_mode: str = UNKNOWN) -> Resolved:
    """把主题装到整个应用上，返回**解析后**的那一份。

    返回值不是可有可无的：网页那边要的是"实际画出来的主色"（浅色下可能已经换过
    一支），而不是用户选的原色。让它跟着一起返回，调用方就不必自己再 :func:`resolve`
    一遍——两处各算一次是这类代码最容易漂的地方。
    """
    resolved = resolve(theme, os_mode)
    app.setPalette(build_palette(resolved))
    app.setStyleSheet(build_qss(resolved))
    return resolved
