"""磨砂按钮那一组颜色：**算得出来的部分，就不靠眼睛看**。

Qt 外壳这边做不到真模糊（QSS 没有 blur，窗口底色又是不透明的），所以"磨砂"在这里
只能是一层半透明填充 + 一条亮边 + 一道纵向渐变。**这层填充的成品色是可以算的**——
父层永远是某个不透明的 token 色，于是"叠上去之后字还看不看得清"是个有确定答案的
问题，而不是"反正很淡"。

这个文件守两件事：

1. **不能把字压糊。** alpha 一大，按钮底下透出来的就是背景色，而背景色是变的
   （窗口底、浮起层、以后可能还有别的落点）。逐格把成品色算出来、逐格量对比度，
   比在注释里写一句"很淡，没关系"结实得多。
2. **亮色那一支必须真的被用上。** 两张 alpha 表分列的理由是"亮"在两个模式下是
   两个方向；漏给某一侧的症状**不是报错**，是那一侧静默用上另一侧的数，看起来
   只是"颜色有点怪"。所以这里断言亮色输出里的值确实等于亮色表。

Qt 真的会不会照这个写法合成，归 ``test_ui_theme_glass.py``（那条要 QApplication，
只能在 gui 腿里跑）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biosnn_memterm.ui.color import (
    RGB,
    contrast_ratio,
    parse_hex,
    relative_luminance,
    rgba,
    to_hex,
)
from biosnn_memterm.ui.theme import (
    _GLASS_ALPHAS,
    _GLASS_EDGE_TOKENS,
    DARK_TOKENS,
    LIGHT_TOKENS,
    MODE_DARK,
    MODE_LIGHT,
    Theme,
    _substitutions,
    build_qss,
    resolve,
)

#: ``rgba(r, g, b, a)``。**只收 ``rgba`` 这一种写法**——:func:`rgba` 只产出它，
#: 收宽了就等于允许"两种写法都对"，而那时下面这些断言量的未必是发出去的东西。
_RGBA = re.compile(r"rgba\((\d+), (\d+), (\d+), ([0-9.]+)\)")

#: 填充 alpha 的上限。**这个数是 :func:`~biosnn_memterm.ui.color.rgba` 那段论据的
#: 边界**，改了它就要回头去看那段话还成不成立。这里断言一遍，是为了让"顺手调大
#: 一点更好看"这件事必须经过这条测试，而不是只经过一双手。
_FILL_ALPHA_MAX = 0.16

_FILL_NAMES = ("glassTop", "glassBottom", "glassTopHover", "glassBottomHover")
#: 三道边。**分开列**，因为"顶边比底边亮"这件事只有各自量一遍才知道——一条边
#: 写死给四个方向用时，它照样能过下面所有对比度断言，而屏幕上就是平的。
_EDGE_NAMES = ("glassLineTop", "glassLine", "glassLineBottom")
_ALL_GLASS_NAMES = (*_FILL_NAMES, *_EDGE_NAMES, "glassPressed")

_CSS = Path(__file__).resolve().parent.parent / "src/biosnn_memterm/ui/web/css/app.css"


def _over(top: str, bottom: str) -> RGB:
    """把 ``top``（``rgba``，或干脆是个不透明色号）叠在 ``bottom`` 上。"""
    match = _RGBA.fullmatch(top)
    if match is None:
        return parse_hex(top)
    base = parse_hex(bottom)
    channels = (int(match[1]), int(match[2]), int(match[3]))
    alpha = float(match[4])
    return (
        round(base[0] * (1 - alpha) + channels[0] * alpha),
        round(base[1] * (1 - alpha) + channels[1] * alpha),
        round(base[2] * (1 - alpha) + channels[2] * alpha),
    )


def _edge_over_fill(edge: str, fill: str, parent: str) -> RGB:
    """一道边叠在**填充之上**，填充再叠在父层上——三层。

    **这一条是照 Qt 的真实画法来的，不是想当然。** 样式表里那张背景铺满**整个
    边框盒**，边是后画上去压在它上面的，所以半透明的边最终与"它下面那一行填充"
    混色，而不是与窗口底混色。两种画法的差别**恰好就是"顶边到底算不算高光"**：
    按叠在父层上算，亮态那道黑边是个高光；照实际画法算，它是条暗线，亮态的按钮
    顶上根本没有光。

    实测在 ``test_ui_theme_glass.py``（那边量的是真渲染出来的像素），这个函数是
    把那个实测结论固定下来。
    """
    return _over(edge, to_hex(_over(fill, parent)))


def _subs(mode: str) -> dict[str, str]:
    """某一模式下真正会写进 QSS 的那份映射。

    第二参数是"系统当前是什么模式"，给 :data:`MODE_DARK` 是为了让它别插话——
    下面要的是 :class:`Theme` 自己选的那个模式，不是系统那个。
    """
    return _substitutions(resolve(Theme(mode=mode), MODE_DARK))


def _css() -> str:
    """读网页样式表，**先剥掉注释**（理由见 ``test_theme.py`` 里同名的那一个：
    注释里出现同一个词是常事，不剥就会为了迁就断言把说明文字改得不能提关键词）。"""
    return re.sub(r"/\*.*?\*/", "", _CSS.read_text(encoding="utf-8"), flags=re.DOTALL)


def _block(text: str, selector: str) -> str:
    """取某个选择器的规则块。**按第一层大括号配对**，不看缩进——CSS 里嵌套
    （``@supports``）是常态，用正则数括号才会在嵌套时悄悄取错一段。"""
    start = text.index(selector)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError(f"{selector} 那一段没有闭合")


def _decl(block: str, name: str) -> str:
    """取某个规则块里一条声明的值（不含分号）。"""
    match = re.search(rf"{re.escape(name)}:\s*([^;]+);", block)
    assert match is not None, f"{block[:40]}… 里没有 {name}"
    return match.group(1).strip()


# ── rgba 本身 ───────────────────────────────────────────────────────────────


def test_rgba_writes_the_one_form_both_engines_read() -> None:
    assert rgba("#ffffff", 0.06) == "rgba(255, 255, 255, 0.06)"
    assert rgba("#000000", 1.0) == "rgba(0, 0, 0, 1)"
    # 三位写法与不带 # 的写法都收（与 ``parse_hex`` 同一套容错），产出唯一一种。
    assert rgba("#fff", 0.5) == "rgba(255, 255, 255, 0.5)"
    assert rgba("0f1116", 0.0) == "rgba(15, 17, 22, 0)"


@pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0, -1.0])
def test_rgba_refuses_an_impossible_alpha(alpha: float) -> None:
    """越界的 alpha **当场炸**，不夹紧。

    夹紧的话 ``rgba(..., 1.4)`` 会变成全不透明，而调用处以为自己拿到的是 40% 的
    玻璃——那种错在屏幕上只是"这块有点实"，没人会往回查到这一行。
    """
    with pytest.raises(ValueError, match="不透明度"):
        rgba("#ffffff", alpha)


# ── 那层玻璃永远叠在不透明色上 ──────────────────────────────────────────────


@pytest.mark.parametrize("tokens", [DARK_TOKENS, LIGHT_TOKENS], ids=["dark", "light"])
def test_the_layer_under_the_glass_is_opaque(tokens: object) -> None:
    """**整段"可以绕开对比度守卫"的论据，前提就是这一条。**

    父层必须是不透明的 ``#rrggbb``：它一旦带上 alpha，"叠上去之后是什么颜色"
    就取决于再下面那一层，而那个是算不出来的（壁纸、别处的半透明）。所以这里
    把前提本身钉住——哪天有人给 ``bg`` 换了带 alpha 的值，红的是这一条，而不是
    若干无关的布局断言。
    """
    for name in ("bg", "bg_elevated"):
        value = getattr(tokens, name)
        assert re.fullmatch(r"#[0-9a-f]{6}", value), f"{name} 不是不透明色号：{value!r}"


# ── 七个占位符 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_every_glass_placeholder_is_filled(mode: str) -> None:
    """少一个就会让 ``Template.substitute`` 抛 ``KeyError``（这是刻意的，见
    ``build_qss`` 的说明）。这条测试把"八个都在"写成断言，于是**删掉一个占位符
    时红的是这个文件**，而不是等某天真的渲染时才炸。"""
    subs = _subs(mode)
    for name in _ALL_GLASS_NAMES:
        assert name in subs, f"{name} 没有对应的值"
        assert subs[name], f"{name} 是空的"
    assert "$" not in build_qss(resolve(Theme(mode=mode), MODE_DARK))


def test_the_two_modes_really_get_two_tables() -> None:
    """亮色那一支必须**真的被用上**。

    漏给一侧的症状不是报错：那一侧会静默用上另一侧的数。在暗态表（白叠 0.06）
    落在亮色底上，按钮几乎看不出来；反过来黑叠 0.09 落在暗色底上，是一块脏灰。
    两种都只是"颜色有点怪"。
    """
    assert set(_GLASS_ALPHAS) == {MODE_DARK, MODE_LIGHT}, "多了一个模式，或者少了一个"
    dark, light = _subs(MODE_DARK), _subs(MODE_LIGHT)
    assert light["glassTop"] == rgba(LIGHT_TOKENS.fg_strong, _GLASS_ALPHAS[MODE_LIGHT].top)
    assert dark["glassTop"] == rgba(DARK_TOKENS.fg_strong, _GLASS_ALPHAS[MODE_DARK].top)
    for name in _ALL_GLASS_NAMES:
        assert dark[name] != light[name], f"{name} 两侧同值，等于只写了一张表"


# ── 渐变的方向 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
@pytest.mark.parametrize("parent", ["bg", "bg_elevated"])
def test_the_gradient_is_lighter_at_the_top_in_both_modes(mode: str, parent: str) -> None:
    """**顶亮底暗**，两种模式都要成立。

    这一条是两张表"大小关系相反"的由来：暗态是往白里叠，叠得多才亮；亮态是往
    黑里叠，叠得少才亮。哪一张写反了，屏幕上就是"光从下面打上来"——说不出哪里
    不对，但看着就是不对。
    """
    subs = _subs(mode)
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    under = getattr(tokens, parent)
    top = relative_luminance(_over(subs["glassTop"], under))
    bottom = relative_luminance(_over(subs["glassBottom"], under))
    assert top > bottom, f"{mode}/{parent}：顶端没有比底端亮"


# ── 那道斜边 ────────────────────────────────────────────────────────────────
#
# 这一侧没有模糊、也没有投影，"这是一块玻璃"全靠上下两条边读出来：顶边是光，
# 底边是它贴着的影子。所以边不能只有一条——顶底要各占一个数，而且要顶亮底暗。
# 写成一条（或写反）在屏幕上只是"看着有点平"或"光从下面打上来"。
#
# **还有一层：这两条边不能跟填充共用色源。** 填充走 ``fg_strong``，因为它要的是
# "离开背景"的方向，而那个方向随明暗反转；边要的是绝对意义上的亮与暗，不反转。
# 共用的话，亮态的顶边会是一条**黑**线——而描边画在填充之上，叠黑只会更黑，
# 屏幕上就是"光从下面打上来"。所以下面既量关系，也量色源本身。


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_the_edges_take_their_own_two_colour_sources(mode: str) -> None:
    """顶边与左右取"该模式下最亮的 token"、底边取最暗的。

    这条与下面两条关系断言测的是同一个决定的**两半**：那两条量"叠出来亮不亮"，
    这条量"拿的是哪支颜色"。只有这条能在**关系还成立**的时候抓住"色源被换回
    ``fg_strong``"——暗态换回去照样顶亮底暗（白就是最亮的），亮态才会塌。
    留着它，是为了让那次改动的红点落在"你换了色源"这句话上，而不是落在
    "顶边不够亮"上。
    """
    subs = _subs(mode)
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    hi_name, lo_name = _GLASS_EDGE_TOKENS[mode]
    # 高光与影子**必须是两支不同的颜色**，否则"斜边"退化成一条同色描边。
    assert hi_name != lo_name, f"{mode}：高光与影子指向了同一个 token"
    assert subs["glassLineTop"] == rgba(getattr(tokens, hi_name), _GLASS_ALPHAS[mode].edge_top)
    assert subs["glassLine"] == rgba(getattr(tokens, hi_name), _GLASS_ALPHAS[mode].edge_side)
    assert subs["glassLineBottom"] == rgba(
        getattr(tokens, lo_name), _GLASS_ALPHAS[mode].edge_bottom
    )


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
@pytest.mark.parametrize("parent", ["bg", "bg_elevated"])
def test_the_bevel_is_light_on_top_and_dark_at_the_bottom(mode: str, parent: str) -> None:
    """顶边比底边亮，两种模式、两个父层都要成立。

    与上面那条"渐变顶亮底暗"是同一类断言，只是量的是边。哪一张表写反了，
    按钮读起来就是**下凹**的——而这恰好是"按下去"该有的样子，于是它会与
    ``:pressed`` 那条规则打架：一个常态看着像按下的按钮，用户不会去点它。

    **量的是"边叠在填充上"之后的样子**，不是边单独叠在父层上（见
    :func:`_edge_over_fill`）。亮态那一支只有这么量才抓得住：它的顶边按老算法
    是"黑叠 0.03"，看着比底边亮；照实际画法它是**暗**的，比底边还暗——而这条
    断言那时就红了。这正是它该红的地方。
    """
    subs = _subs(mode)
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    under = getattr(tokens, parent)
    top = relative_luminance(_edge_over_fill(subs["glassLineTop"], subs["glassTop"], under))
    bottom = relative_luminance(
        _edge_over_fill(subs["glassLineBottom"], subs["glassBottom"], under)
    )
    assert top > bottom, f"{mode}/{parent}：顶边没有比底边亮"


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_the_bevel_edges_stick_out_past_the_fill(mode: str) -> None:
    """顶边要比填充的**顶端**更亮、底边要比填充的**底端**更暗。

    只断言"顶边比底边亮"是不够的：两条边都缩在填充里面时（例如顶边 0.05、
    底边 0.04，而填充是 0.11 → 0.02），它们照样顶亮底暗，却看不出边界——那正是
    "磨砂不明显"的成因。这一条量的就是这个：**边要跨出填充之外**。

    比的基准是"同一层填充自己"——边与填充叠的是同一条底色，差别只在边那道
    描边，所以这一条量的纯粹是"描边有没有把它顶出去"。
    """
    subs = _subs(mode)
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    under = tokens.bg
    fill_top = relative_luminance(_over(subs["glassTop"], under))
    fill_bottom = relative_luminance(_over(subs["glassBottom"], under))
    edge_top = relative_luminance(_edge_over_fill(subs["glassLineTop"], subs["glassTop"], under))
    edge_bottom = relative_luminance(
        _edge_over_fill(subs["glassLineBottom"], subs["glassBottom"], under)
    )
    assert edge_top > fill_top, f"{mode}：顶边没有比填充的顶端更亮，看不出上边界"
    assert edge_bottom < fill_bottom, f"{mode}：底边没有比填充的底端更暗，看不出下边界"


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_the_bevel_also_covers_the_primary_fill(mode: str) -> None:
    """主按钮那道边也压在**它自己那道渐变**上，顶亮底暗。

    主按钮**不写** ``border-color``，那对边是从通用规则继承来的——"不写即时吃"
    这种接线最容易在某天被人顺手补一句就断掉，而屏幕上只是"主按钮看着有点平"。
    这条守卫就是给那一刻准备的，测的正是"继承来的那对边落在主色上还成不成立"。

    **这里能量精确值，gui 腿那边不能。** 主按钮的填充是**不透明**的主色，父层不
    参与合成，所以"边叠上去是什么色"是个不带光栅化的确定值。gui 腿量的是真像素，
    而两道边所在的那一行本身就**只有一半覆盖率**——底色是带圆角的矩形，光栅化到
    最外侧那一行时一半落在窗口底上，那一像素于是是"边叠在 0.5·填充 + 0.5·窗口底
    上"。亮态混进来的那一半近白把 R/G 抬得很高，而相对亮度里蓝只占 0.0722，于是
    "黑 0.22 压在深蓝上"明明更暗（0.0709 对 0.1203，掉四成），量出来反而更亮。
    所以 gui 腿那一路改成与"同一格把这道边擦掉"的渲染比；这一条留下量那半边
    不带光栅化的。
    """
    subs = _subs(mode)
    for fill_name, edge_name, darker in (
        ("accentTop", "glassLineTop", False),
        ("accentBottom", "glassLineBottom", True),
    ):
        fill = parse_hex(subs[fill_name])
        composited = relative_luminance(_over(subs[edge_name], subs[fill_name]))
        assert (composited < relative_luminance(fill)) is darker, (
            f"{mode}/{fill_name}：主色 {subs[fill_name]} 叠上 {subs[edge_name]} 得到 "
            f"{composited:.4f}，{'没有更暗' if darker else '没有更亮'}"
        )


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_the_primary_gradient_also_lights_from_above(mode: str) -> None:
    """主按钮那道实心渐变**方向也要随明暗翻**。

    ``accentHover`` 是"离背景更远"的那一支：暗态更远 = 更亮，亮态更远 = 更暗。
    直接写 ``stop:0 $accentHover, stop:1 $accent`` 的话，亮色模式下主按钮就成了
    上暗下亮——与旁边那些玻璃按钮的受光方向正好相反。暗态看不出来，所以这条
    错误的显形条件恰好是"用户把明暗切到浅色"。
    """
    subs = _subs(mode)
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    top = relative_luminance(_over(subs["accentTop"], tokens.bg))
    bottom = relative_luminance(_over(subs["accentBottom"], tokens.bg))
    assert top > bottom, f"{mode}：主按钮的渐变没有上亮下暗"


# ── 字还看得清吗 ────────────────────────────────────────────────────────────


def test_the_fill_never_washes_the_text_out() -> None:
    """逐格把成品色算出来、逐格量对比度。

    网格是"四种填充 × 两种模式 × 两个可能的父层"：按钮有时候落在窗口底
    （``$bg``）上，有时候落在浮起层（``$bgElevated``）上，两处都要过。
    """
    for mode, tokens in ((MODE_DARK, DARK_TOKENS), (MODE_LIGHT, LIGHT_TOKENS)):
        subs = _subs(mode)
        for name in _FILL_NAMES:
            for parent in (tokens.bg, tokens.bg_elevated):
                composited = to_hex(_over(subs[name], parent))
                worst = min(
                    contrast_ratio(parse_hex(tokens.fg), parse_hex(composited)),
                    contrast_ratio(parse_hex(tokens.fg_strong), parse_hex(composited)),
                )
                assert worst >= 4.5, (
                    f"{mode}/{name}/{parent}：叠出来是 {composited}，字压在上面只有 "
                    f"{worst:.2f} 的对比度"
                )


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_the_fill_alphas_stay_under_the_bound(mode: str) -> None:
    """上限本身也要有守卫。

    ``color.rgba`` 那段"可以绕开对比度守卫"的论据，边界就写在那个数上。上面那条
    对比度断言其实会先红——但那时报出来的是"某格不够清楚"，而这条报的是"你把
    论据的边界挪了"。两句话指向的是同一个决定，所以两条都留着。
    """
    subs = _subs(mode)
    alphas = []
    for name in _FILL_NAMES:
        match = _RGBA.fullmatch(subs[name])
        assert match is not None, f"{name} 不是 rgba 写法：{subs[name]!r}"
        alphas.append(float(match[4]))
    assert max(alphas) <= _FILL_ALPHA_MAX, f"{mode} 的填充最大叠到 {max(alphas)}"


def test_the_pressed_colour_is_one_of_our_own_and_readable() -> None:
    """按下态**故意不在磨砂那套里**（高光是变亮，按下去要变暗）。

    它是那个下沉 token 的原值，所以这里量的是"字压在下沉底上"——与磨砂无关，
    但同样是按钮上会出现的一格，一并有守卫。
    """
    for mode, tokens in ((MODE_DARK, DARK_TOKENS), (MODE_LIGHT, LIGHT_TOKENS)):
        assert _subs(mode)["glassPressed"] == tokens.bg_sunken
        assert contrast_ratio(parse_hex(tokens.fg), parse_hex(tokens.bg_sunken)) >= 4.5


# ── 网页侧 ──────────────────────────────────────────────────────────────────


def test_the_web_side_gets_both_glass_palettes() -> None:
    """两侧都要有一组，**而且必须是两组不同的值**。

    这条与 ``test_theme.py`` 里那条 ``--danger`` 的教训同源：浅色块漏覆盖的症状，
    是浅色下静默用上暗色那支（白填充 + 白亮边落在白底上，两块都看不见）。
    """
    css = _css()
    dark = _block(css, ":root {")
    light = _block(css, ':root[data-theme="light"] {')
    names = ("--glass-bg", "--glass-line", "--glass-hi", "--glass-lo", "--glass-shadow")
    for name in names:
        assert f"{name}:" in dark, f"暗色块没有 {name}"
        assert f"{name}:" in light, f"浅色块没有 {name}"
    for name in names:
        assert _decl(dark, name) != _decl(light, name), f"{name} 两侧同值"


@pytest.mark.parametrize("selector", [".tool-actions button {", "#send {"])
def test_the_web_glass_carries_all_three_shadows(selector: str) -> None:
    """三记投影一记都不能少——**这就是网页侧的斜边**。

    外壳那边"顶亮底暗"靠上下两条描边（QSS 没有 box-shadow），这一侧靠
    ``inset 0 1px 0`` / ``inset 0 -1px 0`` / ``0 1px 2px`` 三记。少一记的症状与
    外壳那边一模一样：按钮读起来是"一块浅一点的圆角矩形"。**这三记是这一侧唯一
    能读出来的玻璃证据**——默认没有壁纸，blur 糊的是纯色，等于没糊（见
    ``test_the_web_side_has_a_fallback_for_engines_without_blur`` 那段注释）。
    """
    block = _block(_css(), selector)
    shadow = _decl(block, "box-shadow")
    assert "inset 0 1px 0 var(--glass-hi)" in shadow, f"{selector} 少了顶上那记高光"
    assert "inset 0 -1px 0 var(--glass-lo)" in shadow, f"{selector} 少了底下那道暗边"
    assert "0 1px 2px var(--glass-shadow)" in shadow, f"{selector} 少了外投影"


@pytest.mark.parametrize("selector", [".tool-actions button {", "#send {"])
def test_the_web_blur_is_on_the_two_button_groups(selector: str) -> None:
    """真模糊只落在两处按钮上，且**带 ``-webkit-`` 前缀**。

    这一侧才是有东西可糊的那一侧（背后是 ``#scrim`` 压过的壁纸）；Qt 外壳那边
    连 blur 都没有，只能靠渐变拟态。两处都写上前缀是因为 QtWebEngine 的版本
    跟着 PySide6 走，不能假定它一定认无前缀的那种写法。
    """
    block = _block(_css(), selector)
    assert "backdrop-filter: blur(" in block
    assert "-webkit-backdrop-filter: blur(" in block


def test_the_web_side_has_a_fallback_for_engines_without_blur() -> None:
    """不支持 blur 时把那点半透明一并去掉。

    留着半透明却没有模糊，读出来是"这块有点脏"，比干脆不透明更差。这条断言很轻，
    但它让"哪天有人把 ``@supports`` 那一段删了"变成一次有意的改动。
    """
    css = _css()
    assert "@supports not" in css
    fallback = _block(css, "@supports not")
    assert "backdrop-filter" in fallback
    assert ".tool-actions button" in fallback and "#send" in fallback
