"""配色数学。

这一份测的东西，在界面里**只能靠眼睛看**，而眼睛看不出来的恰恰是最要紧的那类
错误：悬停态与常态差 2%（等于没有反馈，用户只觉得"点了没反应"）、或者某个色在
浅色底上根本没有对比度（看起来只是"有点淡"）。搬到 :mod:`biosnn_memterm.ui.color`
之后这些都能逐条断言。

**最值得测的是"明暗两态的派生方向相反"这条。** 暗色底上"更突出"是把颜色变亮，
浅色底上是变暗——这是一条规则换了方向，不是两套设计。所以它必须是一个可断言的
``if mode``，而不是两处各写一遍（两处手写必然漂，而漂了之后只有一种模式会难看，
另一种看起来一切正常）。
"""

from __future__ import annotations

import pytest

from biosnn_memterm.ui.color import (
    MODE_DARK,
    MODE_LIGHT,
    contrast_ratio,
    darken,
    derive_roles,
    ensure_contrast,
    lighten,
    mix,
    parse_hex,
    readable_foreground,
    relative_luminance,
    to_hex,
)

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)

_ROLE_KEYS = {
    "accent",
    "accentHover",
    "accentPressed",
    "accentDisabled",
    "accentFg",
    "accentSoft",
}


# ── 解析与输出 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#ff8800", (255, 136, 0)),
        ("ff8800", (255, 136, 0)),
        ("#f80", (255, 136, 0)),
        ("#FFF", _WHITE),
        ("  #0a0b0c  ", (10, 11, 12)),
    ],
)
def test_parse_hex_accepts_common_spellings(text: str, expected: tuple[int, int, int]) -> None:
    assert parse_hex(text) == expected


@pytest.mark.parametrize(
    "bad",
    ["", "#", "#ff", "#ffff", "#fffff", "#fffffff", "#gggggg", "rgb(1,2,3)", "#ff8800ff"],
)
def test_parse_hex_rejects_anything_else(bad: str) -> None:
    """**八位（带 alpha）也要拒。** 本模块每个函数都假设颜色不透明，
    "支持"半透明只会让 :func:`contrast_ratio` 悄悄算错——而算错的表现是守卫
    放行了一个看不清的颜色，不是抛异常。"""
    with pytest.raises(ValueError):
        parse_hex(bad)


@pytest.mark.parametrize("value", ["#ff8800", "#f80", "ff8800", "#FFF"])
def test_to_hex_roundtrip_is_canonical(value: str) -> None:
    """进什么形式都行，**出只有一种**。设置里存的值必须唯一，否则同一个颜色会有
    多个字符串表示，比对与去重都会出错。"""
    assert to_hex(parse_hex(value)) == to_hex(parse_hex(value))
    assert to_hex(parse_hex("#F80")) == "#ff8800"


def test_to_hex_clamps_out_of_range() -> None:
    assert to_hex((-5, 999, 128)) == "#00ff80"


# ── 混合与亮度 ──────────────────────────────────────────────────────────────


def test_mix_endpoints_and_midpoint() -> None:
    assert mix(_BLACK, _WHITE, 0.0) == _BLACK
    assert mix(_BLACK, _WHITE, 1.0) == _WHITE
    assert mix(_BLACK, _WHITE, 0.5) == (128, 128, 128)


@pytest.mark.parametrize("t", [-0.1, 1.1])
def test_mix_rejects_out_of_range(t: float) -> None:
    with pytest.raises(ValueError):
        mix(_BLACK, _WHITE, t)


def test_lighten_and_darken_are_monotone_and_clamped() -> None:
    """单调很重要：不单调的话"更悬停一点"在某些色上反而更暗，
    表现为鼠标划过时按钮闪一下。"""
    base = (100, 140, 200)
    lighter = [relative_luminance(lighten(base, step / 10)) for step in range(11)]
    darker = [relative_luminance(darken(base, step / 10)) for step in range(11)]
    assert lighter == sorted(lighter)
    assert darker == sorted(darker, reverse=True)
    assert lighten(base, 1.0) == _WHITE
    assert darken(base, 1.0) == _BLACK
    assert lighten(base, 0.0) == base
    assert darken(base, 0.0) == base


def test_relative_luminance_bounds() -> None:
    assert relative_luminance(_BLACK) == pytest.approx(0.0)
    assert relative_luminance(_WHITE) == pytest.approx(1.0)


@pytest.mark.parametrize("rgb", [_BLACK, _WHITE, (18, 52, 86), (200, 30, 30)])
def test_contrast_ratio_is_symmetric(rgb: tuple[int, int, int]) -> None:
    assert contrast_ratio(rgb, _WHITE) == pytest.approx(contrast_ratio(_WHITE, rgb))


def test_contrast_ratio_white_on_black_is_21() -> None:
    """21 是 WCAG 公式的上界，也是唯一一个好核对的锚点——对不上说明公式写错了，
    而那会让所有守卫的阈值都失去意义。"""
    assert contrast_ratio(_WHITE, _BLACK) == pytest.approx(21.0)
    assert contrast_ratio((128, 128, 128), (128, 128, 128)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "background", ["#ffffff", "#000000", "#6b8afd", "#808080", "#d9a441", "#1f7a34"]
)
def test_readable_foreground_picks_the_better_side(background: str) -> None:
    picked = parse_hex(readable_foreground(parse_hex(background)))
    other = _BLACK if picked == _WHITE else _WHITE
    back = parse_hex(background)
    assert contrast_ratio(picked, back) >= contrast_ratio(other, back)


# ── 对比度守卫 ──────────────────────────────────────────────────────────────


def test_ensure_contrast_passes_through_when_already_fine() -> None:
    colour, adjusted = ensure_contrast("#000000", "#ffffff")
    assert (colour, adjusted) == ("#000000", False)


def test_ensure_contrast_pushes_away_from_background() -> None:
    """亮底往暗推、暗底往亮推。判据用**背景自己的亮度**而不是"哪个方向更远"
    ——后者在中间灰上会来回抖。"""
    on_light, light_adjusted = ensure_contrast("#eeeeee", "#ffffff")
    on_dark, dark_adjusted = ensure_contrast("#222222", "#000000")
    assert light_adjusted and dark_adjusted
    assert relative_luminance(parse_hex(on_light)) < relative_luminance(parse_hex("#eeeeee"))
    assert relative_luminance(parse_hex(on_dark)) > relative_luminance(parse_hex("#222222"))
    assert contrast_ratio(parse_hex(on_light), _WHITE) >= 3.0
    assert contrast_ratio(parse_hex(on_dark), _BLACK) >= 3.0


def test_ensure_contrast_can_raise_its_floor() -> None:
    """阈值是可调的——浅色模式的主色要按正文的 4.5 走，不是图形的 3.0。"""
    strict, _ = ensure_contrast("#6b8afd", "#f6f7f9", minimum=4.5)
    loose, _ = ensure_contrast("#6b8afd", "#f6f7f9", minimum=3.0)
    assert contrast_ratio(parse_hex(strict), parse_hex("#f6f7f9")) >= 4.5
    assert contrast_ratio(parse_hex(loose), parse_hex("#f6f7f9")) >= 3.0
    assert relative_luminance(parse_hex(strict)) <= relative_luminance(parse_hex(loose))


def test_ensure_contrast_reaches_the_floor_from_mid_grey() -> None:
    """中灰底上也能过线——白在暗底上、黑在亮底上，两个区间是拼起来的，
    覆盖了全部背景色。"""
    colour, adjusted = ensure_contrast("#808080", "#808080")
    assert adjusted
    assert contrast_ratio(parse_hex(colour), parse_hex("#808080")) >= 3.0


def test_ensure_contrast_falls_back_to_the_extreme_only_past_reach() -> None:
    """推到黑/白仍不达标时返回**那个极端值**，不是"迭代到第几步"的结果：够不够
    已经不是这一步能解决的（背景本身有问题），但颜色至少是确定的、可复现的。

    **要走到这条分支得把阈值抬到 8.0。** 常用的 3.0/4.5 在任何背景色下都至少有
    一头过线——所以下面这个兜底是防御性的，拿真实阈值去测只会测到别的分支，
    看起来绿着，其实那几行从没跑过。
    """
    colour, adjusted = ensure_contrast("#808080", "#808080", minimum=8.0)
    assert (colour, adjusted) == ("#ffffff", True)


# ── 角色派生 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode,bg", [(MODE_DARK, "#0f1116"), (MODE_LIGHT, "#f6f7f9")])
@pytest.mark.parametrize("accent", ["#6b8afd", "#3fb950", "#d9a441", "#ffffff", "#000000"])
def test_derive_roles_shape(mode: str, bg: str, accent: str) -> None:
    roles = derive_roles(accent, mode, bg=bg)
    assert set(roles) == _ROLE_KEYS
    for name, value in roles.items():
        assert parse_hex(value), name  # 每个都是合法色号，非法会在这里抛
        assert value == value.lower(), name  # to_hex 只产小写


@pytest.mark.parametrize("mode,bg", [(MODE_DARK, "#0f1116"), (MODE_LIGHT, "#f6f7f9")])
def test_derive_roles_hover_direction_flips_with_mode(mode: str, bg: str) -> None:
    """**这条是整份测试里最要紧的。** 同一个主色，暗色下悬停要比常态**亮**、
    浅色下要比常态**暗**。写反了不会有任何报错，只是其中一种模式下的悬停态
    看起来"有点怪"——而人眼很难指认"怪"在哪里。"""
    base = parse_hex("#6b8afd")
    roles = derive_roles("#6b8afd", mode, bg=bg)
    hover = relative_luminance(parse_hex(roles["accentHover"]))
    pressed = relative_luminance(parse_hex(roles["accentPressed"]))
    original = relative_luminance(base)

    if mode == MODE_DARK:
        assert hover > original
        assert pressed > hover
    else:
        assert hover < original
        assert pressed < hover


def test_derive_roles_disabled_moves_toward_background() -> None:
    for mode, bg in ((MODE_DARK, "#0f1116"), (MODE_LIGHT, "#f6f7f9")):
        roles = derive_roles("#6b8afd", mode, bg=bg)
        back = relative_luminance(parse_hex(bg))
        disabled = relative_luminance(parse_hex(roles["accentDisabled"]))
        original = relative_luminance(parse_hex("#6b8afd"))
        assert abs(disabled - back) < abs(original - back)


def test_accent_foreground_comes_from_the_base_colour_not_the_state() -> None:
    """``accentFg`` 必须只由 ``accent`` 决定。若按当前状态重算，鼠标划过一个刚好
    跨过黑白分界的色块时，块上的字会黑↔白闪一下——那种闪比对比度差 0.2 难看得多，
    而且看起来像 bug。"""
    roles = derive_roles("#808080", MODE_DARK, bg="#0f1116")
    assert roles["accentFg"] == readable_foreground(parse_hex("#808080"))


@pytest.mark.parametrize("accent", ["#6b8afd", "#3fb950", "#d9a441"])
@pytest.mark.parametrize("mode,bg", [(MODE_DARK, "#0f1116"), (MODE_LIGHT, "#f6f7f9")])
def test_accent_foreground_actually_readable(accent: str, mode: str, bg: str) -> None:
    """主色底上的字要看得清。三个预设 × 两态都不能例外。"""
    roles = derive_roles(accent, mode, bg=bg)
    ratio = contrast_ratio(parse_hex(roles["accentFg"]), parse_hex(accent))
    assert ratio >= 3.0, f"{accent} 上的 {roles['accentFg']} 只有 {ratio:.2f}"


def test_derive_roles_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        derive_roles("#6b8afd", "system", bg="#0f1116")
