"""主题装到 Qt 上之后的样子。

gui 腿：它要一个真的 ``QApplication``，但**不碰 WebEngine、也不建 ``MainWindow``**
——后者会去连 Ollama、起四个后台线程、加载网页，重得不像单元测试，而且在没有
显示器时会静默地卡住。

这一份测的都是"纯函数那一侧证明不了"的事：QSS 字符串对不对由
``test_theme.py`` 断言，而"这个字符串真的被 Qt 收下了""调色板真的换了"
只能在这里看。用例故意少——它们真正的作用是**在管线断了的时候第一时间红**，
而不是把配色再验一遍。
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from axo.ui.theme import (
    CUSTOM_PRESET,
    DARK_TOKENS,
    LIGHT_TOKENS,
    PRESETS,
    Theme,
    resolve,
)
from axo.ui.theme_qt import (
    QSettingsBackend,
    apply_theme,
    build_palette,
    os_color_scheme,
)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # 用例之间会互相看见样式表与调色板，收尾时清掉，免得污染别的 gui 测试。
    app.setStyleSheet("")
    app.setPalette(QPalette())


# ── 系统明暗 ────────────────────────────────────────────────────────────────


def test_os_color_scheme_returns_a_known_word(qapp) -> None:
    """**只许返回这三个词之一。** 离屏平台下它给的是 ``"unknown"``——那条分支在
    CI 里是**必然**走到的，所以它必须是被想过的，而不是"线上不会发生"。"""
    assert os_color_scheme() in ("dark", "light", "unknown")


# ── 调色板 ──────────────────────────────────────────────────────────────────


def test_palette_carries_the_tokens(qapp) -> None:
    palette = build_palette(resolve(Theme(), "dark"))
    assert palette.color(QPalette.ColorRole.Window).name() == DARK_TOKENS.bg
    assert palette.color(QPalette.ColorRole.Base).name() == DARK_TOKENS.bg_sunken
    assert palette.color(QPalette.ColorRole.WindowText).name() == DARK_TOKENS.fg


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_palette_highlight_is_the_accent(qapp, preset) -> None:
    """``Highlight`` 管的是那些我们没写样式表的选中态（下拉框当前项、文本框里
    拖蓝的字）。它不跟主色走的话，改完主色会留下几处"没跟着变"的地方。"""
    resolved = resolve(Theme(preset=preset.key), "dark")
    palette = build_palette(resolved)
    assert palette.color(QPalette.ColorRole.Highlight).name() == preset.accent
    assert palette.color(QPalette.ColorRole.HighlightedText).name() == resolved.roles["accentFg"]


def test_palette_differs_between_light_and_dark(qapp) -> None:
    dark = build_palette(resolve(Theme(), "dark"))
    light = build_palette(resolve(Theme(mode="light"), "dark"))
    assert dark.color(QPalette.ColorRole.Window).name() == DARK_TOKENS.bg
    assert light.color(QPalette.ColorRole.Window).name() == LIGHT_TOKENS.bg
    assert dark.color(QPalette.ColorRole.Window) != light.color(QPalette.ColorRole.Window)


def test_disabled_text_is_muted_not_recomputed_by_qt(qapp) -> None:
    """**禁用态要显式给。** 不给的话 Qt 自己把正常态算淡一档，在暗色底上算出来的
    是一个更亮的灰——禁用按钮看起来反而比可用的显眼。"""
    palette = build_palette(resolve(Theme(), "dark"))
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
    assert disabled.name() == DARK_TOKENS.fg_muted


def test_danger_is_in_the_palette(qapp) -> None:
    """``BrightText`` 是 Qt 留给"比正常更醒目"的那一格，错误提示用它。"""
    palette = build_palette(resolve(Theme(), "dark"))
    assert palette.color(QPalette.ColorRole.BrightText).name() == DARK_TOKENS.danger


# ── 装上去 ──────────────────────────────────────────────────────────────────


def test_apply_theme_sets_stylesheet_and_palette(qapp) -> None:
    resolved = apply_theme(qapp, Theme(), "dark")
    assert resolved.mode == "dark"
    assert resolved.accent in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == DARK_TOKENS.bg


def test_apply_theme_returns_the_resolved_accent(qapp) -> None:
    """返回的是**实际画出来的**主色，不是用户选的原色。

    浅色模式下这两者不同（默认蓝紫会被换成 ``#3559c7``）。让 ``apply_theme`` 把
    解析结果一并返回，调用方——网页那一侧要的正是它——就不必自己再 :func:`resolve`
    一遍；两处各算一次是这类代码最容易漂的地方。
    """
    resolved = apply_theme(qapp, Theme(mode="light"), "dark")
    assert resolved.accent == PRESETS[0].light_accent
    assert resolved.accent != PRESETS[0].accent
    assert resolved.accent in qapp.styleSheet()


def test_reapplying_with_a_new_accent_updates_everything(qapp) -> None:
    apply_theme(qapp, Theme(preset="indigo"), "dark")
    assert PRESETS[1].accent not in qapp.styleSheet()

    apply_theme(qapp, Theme(preset="jade"), "dark")
    assert PRESETS[1].accent in qapp.styleSheet()
    assert PRESETS[0].accent not in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Highlight).name() == PRESETS[1].accent


def test_reapplying_between_modes_swaps_the_background(qapp) -> None:
    apply_theme(qapp, Theme(), "dark")
    assert DARK_TOKENS.bg in qapp.styleSheet()

    apply_theme(qapp, Theme(mode="light"), "dark")
    assert LIGHT_TOKENS.bg in qapp.styleSheet()
    assert DARK_TOKENS.bg not in qapp.styleSheet()


def test_custom_accent_reaches_the_stylesheet(qapp) -> None:
    theme = Theme(preset=CUSTOM_PRESET, accent="#ff8800")
    apply_theme(qapp, theme, "dark")
    assert "#ff8800" in qapp.styleSheet()


def test_apply_theme_never_leaves_a_placeholder(qapp) -> None:
    """装上去的那份里不该有残留占位符——有的话 Qt 会静默忽略整条规则，
    而外观上只表现为"有一处没生效"，极难定位。"""
    apply_theme(qapp, Theme(), "dark")
    assert "$" not in qapp.styleSheet()


# ── QSettings 适配 ──────────────────────────────────────────────────────────


class _FakeSettings:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def value(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 (Qt 命名)
        self.data[key] = value


def test_qsettings_backend_round_trips() -> None:
    from axo.ui.settings_store import SettingsStore

    backend = QSettingsBackend(_FakeSettings())
    store = SettingsStore(backend)
    theme = Theme(preset="amber", mode="light")
    store.save(theme)
    assert store.load() == theme


def test_qsettings_backend_is_a_real_qsettings_shape(qapp) -> None:
    """真的 ``QSettings`` 也符合那个 Protocol——这是适配器唯一要证明的事。

    用一个内存态的真 ``QSettings``，不落盘、不碰真实注册表。
    """
    from PySide6.QtCore import QSettings

    from axo.ui.settings_store import SettingsStore

    settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Axo-test", "x")
    store = SettingsStore(QSettingsBackend(settings))
    store.save(Theme(preset="jade", wallpaper_visible=33))
    theme = store.load()
    assert theme.preset == "jade"
    assert theme.wallpaper_visible == 33
