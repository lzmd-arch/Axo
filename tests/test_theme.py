"""主题数据与 QSS 生成。

QSS 最典型的失效方式是**静默不生效**：选择器少一个引号、属性名打错一个字母、
占位符没被替换——Qt 都不报错，只是那一条规则不出现。所以这里不测"看起来对不对"
（那要靠 ``tools/_smoke_window.py`` 目视），测的是**结构**：占位符全被替换掉了、
该有的选择器在场、换了主色输出真的跟着变。

另一件事是**两侧配色不许漂**。Qt 外壳与网页对话区是两套渲染引擎，没有中间格式
可以共用，值只能写在两个文件里。既然只能靠纪律，那就给纪律配一个执法者：这个
文件去读 ``web/css/app.css``，断言两边的主色默认值一致。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from axo.ui.color import contrast_ratio, parse_hex
from axo.ui.theme import (
    CUSTOM_PRESET,
    DARK_TOKENS,
    DEFAULT_THEME,
    LIGHT_TOKENS,
    MIN_ACCENT_CONTRAST,
    MODE_SYSTEM,
    PRESETS,
    WALLPAPER_VISIBLE_DEFAULT,
    WALLPAPER_VISIBLE_MAX,
    Theme,
    build_qss,
    clamp_mode,
    clamp_wallpaper_visible,
    describe_accent,
    effective_mode,
    preset_by_key,
    resolve,
)

_CSS = Path(__file__).resolve().parent.parent / "src/axo/ui/web/css/app.css"


def _css() -> str:
    """读网页样式表，**先剥掉注释**。

    下面三条断言都在回答"文件里有没有这条**规则**"，而 CSS 注释里出现同一个
    字符串是常事——那一块上面就有一段说明"明暗由 Python 显式驱动，不用
    prefers-color-scheme"，还引用了 ``:root[data-theme="light"]`` 这个名字。
    不剥注释的话，要么这几条断言永远假红，要么为了迁就它们把说明文字改成
    一个不能提关键词的样子——后者是让注释服务于测试，本末倒置。
    """
    return re.sub(r"/\*.*?\*/", "", _CSS.read_text(encoding="utf-8"), flags=re.DOTALL)


def _dark_qss() -> str:
    return build_qss(resolve(DEFAULT_THEME, "dark"))


# ── 预设 ────────────────────────────────────────────────────────────────────


def test_presets_have_unique_keys() -> None:
    assert len(PRESETS) == 4
    keys = [item.key for item in PRESETS]
    labels = [item.label for item in PRESETS]
    assert len(set(keys)) == len(keys)
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_preset_colours_are_parseable(preset) -> None:
    assert parse_hex(preset.accent)
    assert parse_hex(preset.light_accent)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_dark_preset_colours_are_readable_on_dark_bg(preset) -> None:
    ratio = contrast_ratio(parse_hex(preset.accent), parse_hex(DARK_TOKENS.bg))
    assert ratio >= MIN_ACCENT_CONTRAST, f"{preset.label} 在暗色底上只有 {ratio:.2f}"


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_light_preset_colours_are_readable_on_light_bg(preset) -> None:
    """**预设的浅色值必须手挑得过线，不能靠守卫算。** 见 ``Preset.light_accent``
    的说明：守卫只会推到"刚好过线"，而主色在网页侧当链接文字用。"""
    ratio = contrast_ratio(parse_hex(preset.light_accent), parse_hex(LIGHT_TOKENS.bg))
    assert ratio >= MIN_ACCENT_CONTRAST, f"{preset.label} 在浅色底上只有 {ratio:.2f}"


def test_light_preset_colours_are_in_the_same_band() -> None:
    """四支浅色值的对比度要彼此相当。

    不齐的话，**换预设会顺带改变"浅色下文字有多清楚"**——一个用户没在调、也不该
    被顺带改掉的量。这里不钉死具体数值（那会让调色变得很难），只钉住它们落在同一个
    窄带里。
    """
    ratios = [
        contrast_ratio(parse_hex(item.light_accent), parse_hex(LIGHT_TOKENS.bg)) for item in PRESETS
    ]
    assert max(ratios) - min(ratios) < 1.5, f"四支浅色值差了 {max(ratios) - min(ratios):.2f}"


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_preset_dark_and_light_differ(preset) -> None:
    assert preset.accent != preset.light_accent


def test_preset_by_key() -> None:
    assert preset_by_key("jade") is PRESETS[1]
    assert preset_by_key(CUSTOM_PRESET) is None
    assert preset_by_key("没这个") is None


# ── 两侧不漂 ────────────────────────────────────────────────────────────────


def test_web_css_uses_the_same_accents() -> None:
    """网页侧的 ``--accent`` 必须与 :data:`PRESETS` 的第一项一致——暗色那一支与
    浅色那一支都要。

    断言的是"文件里出现过这两个色号"而不是"某个块里写了什么"，是**故意**的：
    接下来那一块要从 ``@media (prefers-color-scheme: light)`` 改成
    ``:root[data-theme="light"]``（明暗由 Python 显式驱动，不再靠 Chromium 自己
    检测）。钉住选择器写法会让那条改动顺带改红这个测试，而它要守的其实只是
    "两边用的是同一个色号"。
    """
    css = _css()
    found = set(re.findall(r"--accent:\s*(#[0-9a-fA-F]{6})", css))
    default = PRESETS[0]
    assert default.accent in found, f"网页侧没有 {default.accent}；找到的是 {sorted(found)}"
    assert default.light_accent in found, f"网页侧没有 {default.light_accent}"


def test_web_css_light_block_overrides_danger() -> None:
    """网页侧的浅色块**必须**覆盖 ``--danger``。

    原本那一块只覆盖了 ``--accent``，于是浅色模式下错误提示用的还是暗色那支
    鲑鱼粉——在白底上对比度 2.5 左右，而它恰恰是最需要看清的一句。两侧的
    ``--danger`` 现在同源，这里断言网页侧那一支确实等于 :data:`LIGHT_TOKENS`。
    """
    css = _css()
    light_block = css.split(':root[data-theme="light"]', 1)
    assert len(light_block) == 2, '网页侧没有 [data-theme="light"] 块'
    # 只看那个块本身：整个文件里 --danger 出现两次（暗色一次、浅色一次），
    # 不切开的话"暗色那一支"会让这条断言永远通过。
    body = light_block[1].split("}", 1)[0]
    assert LIGHT_TOKENS.danger in body, f"浅色块没有覆盖 --danger：{body}"
    assert LIGHT_TOKENS.danger != DARK_TOKENS.danger


def test_web_css_drops_the_browser_side_scheme_detection() -> None:
    """**网页侧不许再自己检测明暗。**

    明暗如果两侧各判一次，迟早会在某台机器上分成两半（网页白了、外壳还黑着）。
    这条断言挡住的是"以后有人图省事又加回一个 media query"。
    """
    assert "prefers-color-scheme" not in _css()


# ── 浮在壁纸上的文字 ────────────────────────────────────────────────────────
#
# 壁纸是用户选的任意一张图，所以"浮在它上面的字读不读得清"不能靠眼睛看一遍——它
# 取决于用户选了哪张。能测的是**这件事有没有被交给壁纸决定**：带不透明底的文字与
# 壁纸无关（可以算），半透明或不打底的才取决于壁纸，而后者在某个壁纸上一定读不清。

_TEXT_CONTRAST_FLOOR = 4.5
"""正文与它所在色块之间的下限（WCAG AA）。``ui/color.py:44-45`` 记着这个口径：
正文 4.5、非文本图形 3.0。"""


def _palettes() -> tuple[dict[str, str], dict[str, str]]:
    """两套**有效**调色板 ``(暗, 浅)``。

    浅色是在暗色之上覆盖一层得到的——与 CSS 的层叠语义一致（浅色块只写它改的那几个
    变量）。只读浅色块本身的话，没被覆盖的变量会取不到。
    """
    css = _css()
    dark_body = css.split("{", 1)[1].split("}", 1)[0]
    light_body = css.split(':root[data-theme="light"]', 1)[1].split("}", 1)[0]
    dark = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", dark_body))
    light = {**dark, **dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", light_body))}
    return dark, light


def _caption_rules() -> list[str]:
    """所有提到 ``.msg-caption`` 的规则的**声明部分**。

    收全部而不是第一条：以后若有人加一条更具体的选择器去覆盖背景，只看第一条正好
    漏掉那次覆盖——而"覆盖成半透明"正是这个测试要挡的事。
    """
    return [
        declarations
        for selectors, declarations in re.findall(r"([^{}]*)\{([^{}]*)\}", _css())
        if ".msg-caption" in selectors
    ]


def test_the_caption_carries_an_opaque_backing() -> None:
    """对话流里的标题浮在壁纸上，所以它的底**必须不透明**。

    这条守的是一件真发生过的事：确认卡片的标题（「需要你确认 · write_memory」）与
    工具结果的标题都是 12px 的 ``--fg-muted``，直接画在壁纸上。哪怕蒙层按默认的 0.82
    压着，纯白壁纸下对合成底也只有 **3.59**、浅色配纯黑壁纸 **2.96**——都在正文下限
    之下，而这两张壁纸用户随手就能选到。

    **半透明的底不算数，而且比不打底更差。** 顶栏那套
    ``color-mix(..., 70%, transparent)`` 照抄过来的话壁纸还透上来 30%：暗色配纯白
    壁纸时合成底成了中灰 ``#5d5f64``，对 ``--fg-muted`` 只有 **2.08**。所以这里禁的
    是"透"，不是"有没有写 background"。
    """
    rules = _caption_rules()
    assert rules, "找不到 .msg-caption 的规则——它是不是被改名或删了？"
    for declarations in rules:
        assert "transparent" not in declarations, f".msg-caption 的底是半透明的：{declarations}"
        assert "color-mix" not in declarations, f".msg-caption 的底是混出来的：{declarations}"
    assert any("background" in declarations for declarations in rules), (
        ".msg-caption 没有背景——它直接画在壁纸上"
    )


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_the_caption_text_clears_the_text_floor(mode: str) -> None:
    """上一条的**前提**：``--fg-muted`` 对 ``--bg-elevated`` 真的过得了 4.5。

    上一条只证明"标题用的是不透明的底"，这一条证明"那块底上的字够清楚"。两条合起来
    才是完整的话——只留前一条的话，把 ``--bg-elevated`` 换成一支浅灰色也照样通过，
    而那时标题依然看不清。
    """
    dark, light = _palettes()
    palette = dark if mode == "dark" else light
    ratio = contrast_ratio(parse_hex(palette["fg-muted"]), parse_hex(palette["bg-elevated"]))
    assert ratio >= _TEXT_CONTRAST_FLOOR, (
        f"{mode} 下 --fg-muted 对 --bg-elevated 只有 {ratio:.2f}，低于正文下限 "
        f"{_TEXT_CONTRAST_FLOOR}；.msg-caption 用的正是这一对"
    )


# ── Theme 的不变式 ──────────────────────────────────────────────────────────


def test_default_theme_is_the_first_preset() -> None:
    assert DEFAULT_THEME.preset == PRESETS[0].key
    assert DEFAULT_THEME.accent == PRESETS[0].accent
    assert DEFAULT_THEME.mode == MODE_SYSTEM
    assert DEFAULT_THEME.wallpaper == ""
    assert DEFAULT_THEME.wallpaper_visible == WALLPAPER_VISIBLE_DEFAULT


def test_default_visible_matches_the_old_hardcoded_scrim() -> None:
    """默认可见度 18 ⇔ 遮罩 82% 不透明，正是网页侧原本写死的那个数。
    取它做默认，是为了让"接上壁纸功能"这件事本身不改变任何已有用户的观感。"""
    assert WALLPAPER_VISIBLE_DEFAULT == 18
    assert 100 - WALLPAPER_VISIBLE_DEFAULT == 82


def test_preset_wins_over_the_accent_field() -> None:
    """选了预设还带着一个对不上的 ``accent``，是两套真源，而两套必然会在某条路径
    上分叉——表现是"设置窗口里勾的是蓝紫，界面画的是上次自定义的红"。"""
    theme = Theme(preset="indigo", accent="#ff0000")
    assert theme.accent == PRESETS[0].accent


def test_custom_preset_keeps_its_accent() -> None:
    assert Theme(preset=CUSTOM_PRESET, accent="#ff0000").accent == "#ff0000"


def test_accent_is_normalised_to_lowercase_long_form() -> None:
    """设置里存的形式必须唯一，否则"同一个颜色"会有多个字符串表示。"""
    assert Theme(preset=CUSTOM_PRESET, accent="#ABC").accent == "#aabbcc"
    assert Theme(preset=CUSTOM_PRESET, accent="F80").accent == "#ff8800"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "purple"},
        {"preset": "没有这个预设"},
        {"preset": CUSTOM_PRESET, "accent": "不是颜色"},
        {"preset": CUSTOM_PRESET, "accent": ""},
        {"wallpaper_visible": -1},
        {"wallpaper_visible": WALLPAPER_VISIBLE_MAX + 1},
    ],
)
def test_theme_rejects_bad_values(kwargs: dict) -> None:
    """**非法值不许进入 Theme。** 等到渲染时才发现，那时已经写进设置、
    下次启动照样崩，而报错的地方离原因很远。"""
    with pytest.raises(ValueError):
        Theme(**kwargs)


def test_wallpaper_visible_upper_bound_is_deliberate() -> None:
    assert WALLPAPER_VISIBLE_MAX == 70
    assert Theme(wallpaper_visible=WALLPAPER_VISIBLE_MAX).wallpaper_visible == 70


# ── 兜底 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (50, 50),
        ("50", 50),
        (0, 0),
        (WALLPAPER_VISIBLE_MAX + 99, WALLPAPER_VISIBLE_MAX),
        (-5, 0),
        (None, WALLPAPER_VISIBLE_DEFAULT),
        ("不是数字", WALLPAPER_VISIBLE_DEFAULT),
        (3.9, 3),
    ],
)
def test_clamp_wallpaper_visible(value: object, expected: int) -> None:
    assert clamp_wallpaper_visible(value) == expected


@pytest.mark.parametrize("value", ["dark", "light", "system"])
def test_clamp_mode_passes_known_values(value: str) -> None:
    assert clamp_mode(value) == value


@pytest.mark.parametrize("value", ["purple", "", None, 3, b"dark"])
def test_clamp_mode_falls_back_to_system(value: object) -> None:
    assert clamp_mode(value) == MODE_SYSTEM


# ── 明暗解析 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,os_mode,expected",
    [
        (MODE_SYSTEM, "light", "light"),
        (MODE_SYSTEM, "dark", "dark"),
        (MODE_SYSTEM, "unknown", "dark"),
        # 指了明暗就以它为准，系统说什么都不算
        ("light", "dark", "light"),
        ("dark", "light", "dark"),
    ],
)
def test_effective_mode(mode: str, os_mode: str, expected: str) -> None:
    """**只有 ``"light"`` 会得到浅色。** 问不到系统偏好时回落暗色——本项目的默认
    观感是暗色，回落到浅色会让一个暗色主题的应用莫名其妙变白。"""
    assert effective_mode(Theme(mode=mode), os_mode) == expected


def test_resolve_dark_keeps_the_accent_untouched() -> None:
    resolved = resolve(Theme(), "dark")
    assert resolved.accent == PRESETS[0].accent
    assert resolved.accent_adjusted is False


def test_resolve_light_uses_the_preset_light_accent() -> None:
    """**默认主题在浅色下必须是网页侧原本那一支 ``#3559c7``**，而不是守卫算出来
    的近似值。差了这一步，"改进了配色系统"就成了"默认观感退步"。"""
    resolved = resolve(Theme(mode="light"), "dark")
    assert resolved.accent == PRESETS[0].light_accent
    assert resolved.accent_adjusted is True


def test_resolve_light_guards_a_custom_accent() -> None:
    """自定义主色没有手挑的浅色值可依，走守卫——而且**下限是 4.5 不是 3.0**：
    主色在网页侧当链接文字用，那是正文的阈值。"""
    theme = Theme(preset=CUSTOM_PRESET, accent="#6b8afd", mode="light")
    resolved = resolve(theme, "dark")
    assert resolved.accent_adjusted is True
    ratio = contrast_ratio(parse_hex(resolved.accent), parse_hex(LIGHT_TOKENS.bg))
    assert ratio >= MIN_ACCENT_CONTRAST, f"守卫只给到 {ratio:.2f}"


def test_resolve_light_leaves_a_dark_custom_accent_alone() -> None:
    """已经够暗的自定义色不该被动。"""
    theme = Theme(preset=CUSTOM_PRESET, accent="#1a1a1a", mode="light")
    resolved = resolve(theme, "dark")
    assert resolved.accent == "#1a1a1a"
    assert resolved.accent_adjusted is False


def test_describe_accent_says_what_happened() -> None:
    """**换过就要说。** 用户下次看到自己选的色号与屏幕上的不一样，
    会以为是程序坏了。"""
    unchanged = describe_accent(Theme(), resolve(Theme(), "dark"))
    assert unchanged == PRESETS[0].accent

    theme = Theme(mode="light")
    explained = describe_accent(theme, resolve(theme, "dark"))
    assert PRESETS[0].light_accent in explained
    assert theme.accent in explained  # 把用户选的那支也写出来


# ── QSS ─────────────────────────────────────────────────────────────────────


def test_qss_has_no_leftover_placeholders() -> None:
    """**最要紧的一条。** 漏替换的占位符会让 Qt 静默忽略那一条规则，
    外观上只表现为"有一处没生效"——极难定位。"""
    qss = _dark_qss()
    assert "$" not in qss


def test_qss_grows_with_every_placeholder_replaced() -> None:
    """模板里没有任何占位符的话上面那条断言会空转通过。"""
    template = (Path(__file__).resolve().parent.parent / "src/axo/ui/theme.qss").read_text(
        encoding="utf-8"
    )
    assert "$" in template
    assert len(_dark_qss()) > 1000


@pytest.mark.parametrize(
    "selector",
    [
        "QMainWindow",
        "QFrame#header",
        'QLabel#status[state="error"]',
        "#contextBar",
        "QListWidget#sessionList::item",
        "QTextBrowser",
        "QSlider::handle",
        "QSplitter::handle",
        'QPushButton[kind="primary"]',
        'QPushButton[kind="danger"]',
        # 设置窗口的挂载点。它们全是**新加的 objectName**，而一个新 objectName
        # 没有对应规则的症状是"这里看起来没被样式管过"——不报错、不崩，
        # 只是那几处悄悄用着 Fusion 的默认外观。
        "QListWidget#settingsCategories",
        "QFrame#settingsFooter",
        "QLabel#sectionTitle",
        'QLabel#remoteWarning[armed="yes"]',
    ],
)
def test_qss_has_the_selectors_that_matter(selector: str) -> None:
    """逐条钉住"这个挂载点有人管"。

    ``#contextBar`` 只在这里出现**槽**的规则，填充色由 ``_show_context`` 的控件级
    样式表给（那是语义交通灯，不随主题变）。两边共存已实测过——控件级样式表不会
    顶掉应用级的槽底规则。
    """
    assert selector in _dark_qss(), f"QSS 里没有 {selector}"


def test_qss_carries_the_accent_and_the_tokens() -> None:
    resolved = resolve(Theme(preset=CUSTOM_PRESET, accent="#ff8800"), "dark")
    text = build_qss(resolved)
    assert "#ff8800" in text
    assert resolved.roles["accentHover"] in text
    assert resolved.roles["accentSoft"] in text
    assert DARK_TOKENS.bg in text


def test_qss_changes_when_the_accent_changes() -> None:
    first = build_qss(resolve(Theme(preset="indigo"), "dark"))
    second = build_qss(resolve(Theme(preset="jade"), "dark"))
    assert first != second
    assert PRESETS[0].accent in first
    assert PRESETS[1].accent in second


def test_qss_changes_between_light_and_dark() -> None:
    dark = build_qss(resolve(Theme(), "dark"))
    light = build_qss(resolve(Theme(mode="light"), "dark"))
    assert dark != light
    assert LIGHT_TOKENS.bg in light
    assert DARK_TOKENS.bg not in light


def test_qss_danger_differs_between_modes() -> None:
    """浅色的 danger 是单独一支更深的红。用暗色那支鲑鱼粉在白底上对比度只有
    2.5 左右，而错误提示恰恰是最需要看得清的那一句。"""
    assert DARK_TOKENS.danger != LIGHT_TOKENS.danger
    for tokens in (DARK_TOKENS, LIGHT_TOKENS):
        ratio = contrast_ratio(parse_hex(tokens.danger), parse_hex(tokens.bg))
        assert ratio >= 4.5, f"{tokens.danger} 在 {tokens.bg} 上只有 {ratio:.2f}"


@pytest.mark.parametrize("tokens", [DARK_TOKENS, LIGHT_TOKENS], ids=["dark", "light"])
def test_body_text_is_readable(tokens) -> None:
    for name in ("fg", "fg_strong"):
        ratio = contrast_ratio(parse_hex(getattr(tokens, name)), parse_hex(tokens.bg))
        assert ratio >= 4.5, f"{name} 只有 {ratio:.2f}"


@pytest.mark.parametrize("tokens", [DARK_TOKENS, LIGHT_TOKENS], ids=["dark", "light"])
def test_muted_text_is_still_readable(tokens) -> None:
    """次要文字（时间戳、'上下文 —'）是 3.0 一档——它不需要抢注意力，
    但也不能糊成一片。"""
    ratio = contrast_ratio(parse_hex(tokens.fg_muted), parse_hex(tokens.bg))
    assert ratio >= 3.0, f"fg_muted 只有 {ratio:.2f}"
