"""配色数学：纯函数，**不 import Qt**。

放在这里的理由与 :mod:`axo.session.indicator` 是同一条——界面里最难
测的就是"这个颜色对不对"。配色派生一旦住在 ``QWidget`` 里，它就只能靠人眼看，
而人眼看不出来的恰恰是最要紧的那类错误：悬停态与常态差 2%（等于没有反馈），
或者某个预设色放到浅色底上根本没有对比度。搬到这里就能逐条断言。

**一个纯色，派生方向随明暗反转。** 暗色底上"更突出"是把颜色变亮，浅色底上是
变暗——这不是两套设计，是同一条规则换了方向。所以它是
:func:`derive_roles` 里的一个 ``if``，而不是两处各写一遍（两处手写就会漂）。

RGB 一律是 ``(r, g, b)`` 三元组、每通道 0–255 的整数；对外的字符串形式一律是
``"#rrggbb"`` 小写。:func:`parse_hex` 容错地收 ``#rgb`` 与不带 ``#`` 的写法，
但:func:`to_hex` 只产出一种写法——**写出去的形式必须唯一**，否则设置里存的值
会随输入形式变来变去，比对与去重都会出错。
"""

from __future__ import annotations

__all__ = [
    "MODE_DARK",
    "MODE_LIGHT",
    "contrast_ratio",
    "darken",
    "derive_roles",
    "ensure_contrast",
    "lighten",
    "mix",
    "parse_hex",
    "readable_foreground",
    "relative_luminance",
    "rgba",
    "to_hex",
]

MODE_DARK = "dark"
MODE_LIGHT = "light"

RGB = tuple[int, int, int]

_HEX_DIGITS = set("0123456789abcdefABCDEF")
_CHANNEL_MAX = 255
_BYTE_MAX = 255.0

#: 正文/图标与它所在色块之间至少要有的对比度（WCAG 对非文本图形的要求是 3.0，
#: 正文是 4.5）。3.0 用在"主色块上的字"与"浅色模式的主色守卫"两处。
MIN_GRAPHIC_CONTRAST = 3.0

#: 悬停/按下的步长。**0.28 是上限**——再大就会把主色洗到与 :func:`readable_foreground`
#: 选出的前景色对不上（见 ``test_color_derive.py`` 里那条"三种状态都过对比度"的
#: 断言）。要调这两个数，先看那条断言还绿不绿。
_HOVER_STEP = 0.14
_PRESSED_STEP = 0.28

#: 禁用态往背景方向混的比例。0.60 之后基本只剩一个灰影，正好。
_DISABLED_MIX = 0.60

#: ``accentSoft``（选中行底色一类的低饱和底）往背景方向混的比例。
_SOFT_MIX = 0.78

#: WCAG 相对亮度公式里的 sRGB 反伽马常数。
_SRGB_LINEAR_THRESHOLD = 0.03928
_SRGB_DIVISOR = 12.92
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_EXPONENT = 2.4


def parse_hex(value: str) -> RGB:
    """``"#rrggbb"`` / ``"#rgb"`` / 不带 ``#`` → ``(r, g, b)``。

    只收 3 位与 6 位。**不收 8 位（带 alpha）**：本模块的每一个函数都假设颜色是
    不透明的，"支持"半透明只会让 :func:`contrast_ratio` 悄悄算错——而对比度算错
    的表现是守卫放行了一个看不清的颜色，不是抛异常。
    """
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or any(ch not in _HEX_DIGITS for ch in text):
        raise ValueError(f"不是合法的颜色字面量：{value!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def to_hex(rgb: RGB) -> str:
    """``(r, g, b)`` → ``"#rrggbb"``（小写）。通道越界一律夹紧。"""
    return "#" + "".join(f"{_clamp(channel):02x}" for channel in rgb)


def rgba(color: str, alpha: float) -> str:
    """``"#rrggbb"`` + 不透明度 → ``"rgba(r, g, b, a)"``。QSS 与 CSS 都收这一种写法。

    **这是本模块唯一产出半透明颜色的函数，也是唯一绕开对比度守卫的一处**——但
    它不是把 :func:`parse_hex` 那条"不收 8 位带 alpha"的规矩放宽了，而是那条规矩
    有一个明确的**适用边界**，这里落在边界外：

    磨砂按钮的半透明填充永远叠在一个**不透明的 token 色**上（``$bg`` /
    ``$bgElevated``），且 alpha 一律 ≤ 0.16（见 ``theme._GLASS_ALPHAS``）。父层
    既然是已知的，叠加后的有效底色就是**算得出来**的，于是"字还看不看得清"仍然
    是个可断言的问题——``test_theme_glass.py`` 把最坏那一格（亮态悬停时的底色，
    叠得最狠的一处）真的算了一遍，不是靠"反正很淡"糊过去。

    **这条论据的边界就在那个上限上。** 拿更大的 alpha 去做底，按钮底下就会透出
    不确定的东西，字压在一个算不出对比度的背景上；那时该做的是回头去量对比度
    （或干脆换回不透明的 token 色），而不是把这里的判断放宽。

    实测（``test_theme_glass.py``）：Qt 的 QSS 解析器认这个写法，且半透明填充与
    ``qlineargradient`` 的 rgba 色标都是**与父层颜色混合**后才落笔的，不是画成
    透明再被父层盖住。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"不透明度必须在 [0, 1] 内：{alpha!r}")
    red, green, blue = parse_hex(color)
    return f"rgba({red}, {green}, {blue}, {alpha:g})"


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """``t=0`` 得 ``a``，``t=1`` 得 ``b``。线性插值，逐通道四舍五入。"""
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"混合比例必须在 [0, 1] 内：{t!r}")
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def lighten(rgb: RGB, amount: float) -> RGB:
    """往白色方向混 ``amount``。``amount=0`` 是原色。"""
    return mix(rgb, (_CHANNEL_MAX, _CHANNEL_MAX, _CHANNEL_MAX), amount)


def darken(rgb: RGB, amount: float) -> RGB:
    """往黑色方向混 ``amount``。"""
    return mix(rgb, (0, 0, 0), amount)


def relative_luminance(rgb: RGB) -> float:
    """WCAG 2.x 的相对亮度，值域 ``[0, 1]``。

    **不是"人眼感知的明度"**：它是给:func:`contrast_ratio` 用的标准量。
    自己另写一个"看起来差不多"的亮度公式，对比度就会与所有现成的无障碍工具
    对不上，而那正是这条守卫要能被核对的前提。
    """
    channels = []
    for channel in rgb:
        value = channel / _BYTE_MAX
        if value <= _SRGB_LINEAR_THRESHOLD:
            channels.append(value / _SRGB_DIVISOR)
        else:
            channels.append(((value + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_EXPONENT)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(a: RGB, b: RGB) -> float:
    """两色的对比度，值域 ``[1.0, 21.0]``（黑白是 21）。"""
    first, second = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def readable_foreground(background: RGB) -> str:
    """在这个底色上写字，该用黑还是白。取对比度更大的那一边。"""
    if contrast_ratio(background, (0, 0, 0)) >= contrast_ratio(background, (_CHANNEL_MAX,) * 3):
        return "#000000"
    return "#ffffff"


def ensure_contrast(foreground: str, background: str, *, minimum: float = MIN_GRAPHIC_CONTRAST):
    """把 ``foreground`` 朝**远离背景**的方向推到对比度过线。

    返回 ``(颜色, 是否调过)``。第二项是给界面用的：设置窗口要在浅色模式下如实
    说一句"已为浅色模式调暗"，而不是悄悄换掉用户选的颜色——用户下次看到自己选的
    色号与屏幕上的不一样，会以为是程序坏了。

    **每步 5%，最多走到黑或白。** 走到头仍不达标就返回那个极端值并标为已调过：
    那时"够不够"已经不是这一步能解决的了（背景色本身有问题），但至少颜色是确定的、
    可以复现的，不是一个随迭代次数漂移的值。
    """
    target = parse_hex(foreground)
    back = parse_hex(background)
    if contrast_ratio(target, back) >= minimum:
        return to_hex(target), False

    # 背景亮就往暗里推，背景暗就往亮里推。判据用背景自己的亮度而不是"哪个方向
    # 更远"——后者在中间灰上会来回抖。
    toward_dark = relative_luminance(back) > 0.5
    extreme = (0, 0, 0) if toward_dark else (_CHANNEL_MAX,) * 3

    for step in range(1, 21):
        candidate = mix(target, extreme, step * 0.05)
        if contrast_ratio(candidate, back) >= minimum:
            return to_hex(candidate), True
    return to_hex(extreme), True


def derive_roles(accent: str, mode: str, *, bg: str) -> dict[str, str]:
    """从**一个**主色推出界面要用的全部颜色。

    Args:
        accent: 主色，``"#rrggbb"``。**应当是已经过 :func:`ensure_contrast` 的值**
            ——这里不再做守卫，否则 :func:`ensure_contrast` 的"是否调过"就无法
            如实告诉用户（两个地方各调一次，用户看到的是哪个就说不清了）。
        mode: :data:`MODE_DARK` 或 :data:`MODE_LIGHT`。
        bg: 当前模式的背景色。禁用态与柔和底是往**它**混出来的，所以这个参数
            不是可选的——少了它就只能往黑或白混，那在明暗两态里必有一态是错的。

    Returns:
        六个键，全部是 ``"#rrggbb"``：

        ``accent`` / ``accentHover`` / ``accentPressed`` / ``accentDisabled``
        / ``accentFg`` / ``accentSoft``。

    **``accentFg`` 从 ``accent`` 算，不从悬停/按下算。** 若按当前状态重算，
    鼠标划过一个刚好跨过黑白分界的色块时，块上的字会黑↔白闪一下——那种闪比
    对比度差 0.2 难看得多，而且看起来像 bug。
    """
    if mode not in (MODE_DARK, MODE_LIGHT):
        raise ValueError(f"未知的明暗模式：{mode!r}")
    base = parse_hex(accent)
    background = parse_hex(bg)

    away = darken if mode == MODE_LIGHT else lighten
    return {
        "accent": to_hex(base),
        "accentHover": to_hex(away(base, _HOVER_STEP)),
        "accentPressed": to_hex(away(base, _PRESSED_STEP)),
        "accentDisabled": to_hex(mix(base, background, _DISABLED_MIX)),
        "accentFg": readable_foreground(base),
        "accentSoft": to_hex(mix(base, background, _SOFT_MIX)),
    }


def _clamp(channel: int) -> int:
    return max(0, min(_CHANNEL_MAX, int(channel)))
