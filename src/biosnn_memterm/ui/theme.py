"""主题数据、明暗两套色，以及 Qt 样式表的生成。**纯函数，不 import Qt。**

与 :mod:`.color` 一样，这里刻意不碰 Qt：``build_qss`` 的产物是一个字符串，
于是"占位符有没有全部被替换掉""换了主色输出有没有跟着变""选择器写对没有"
全部可以逐条断言。这些断言抓的正是样式表最典型的失效方式——**静默不生效**：
QSS 里多一个空格、选择器少一个引号，Qt 不报错，只是那一条规则不出现。

**两侧共用一份配色，但只有一份真源。** 网页那边的 `:root` 变量表与这里的
:data:`DARK_TOKENS` / :data:`LIGHT_TOKENS` 是同一套值，写在两个文件里是因为
它们服务于两个渲染引擎（Chromium 与 Qt），没有中间格式可以共用。改色时要
**两边一起改**，``test_theme.py`` 会断言两边的主色默认值一致。

明暗两态**不是两套设计**：同一个 :class:`Tokens` 结构填两遍值，主色的派生
规则也只有一个（方向随明暗反转，见 :func:`biosnn_memterm.ui.color.derive_roles`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template

from .color import (
    MODE_DARK,
    MODE_LIGHT,
    derive_roles,
    ensure_contrast,
    parse_hex,
    rgba,
    to_hex,
)

__all__ = [
    "CUSTOM_PRESET",
    "DARK_TOKENS",
    "DEFAULT_THEME",
    "LIGHT_TOKENS",
    "MIN_ACCENT_CONTRAST",
    "MODE_SYSTEM",
    "PRESETS",
    "WALLPAPER_VISIBLE_DEFAULT",
    "WALLPAPER_VISIBLE_MAX",
    "Preset",
    "Resolved",
    "Theme",
    "Tokens",
    "build_qss",
    "clamp_mode",
    "clamp_wallpaper_visible",
    "describe_accent",
    "effective_mode",
    "preset_by_key",
    "resolve",
]

#: 「跟随系统」。与 :data:`MODE_DARK` / :data:`MODE_LIGHT` 并列的第三个取值，
#: 但它**不是**一种可以拿去渲染的模式——:func:`effective_mode` 负责把它解析成
#: 两者之一。让 "system" 一路传到渲染层是这类代码最常见的 bug：某个 ``if``
#: 没考虑到第三种取值，于是回落到"不是 dark 就是 light"里错误的那一边。
MODE_SYSTEM = "system"

_MODE_CHOICES = (MODE_SYSTEM, MODE_DARK, MODE_LIGHT)


@dataclass(frozen=True)
class Preset:
    """一个备选主色。

    **两支色，不是一个色加一个算法。** ``accent`` 给暗色底，``light_accent``
    给浅色底——后者是手挑的，不是把前者算暗得到的。
    """

    key: str
    label: str
    accent: str
    light_accent: str
    """浅色底上用的那一支。**必须手挑。**

    为什么不交给 :func:`~biosnn_memterm.ui.color.ensure_contrast` 自动算：那个
    函数的语义是"推到刚好过线"，而"刚好过线"对网页侧的主色是不够的——它在那边
    主要当**链接与强调文字**的颜色用。举例，``#6b8afd`` 按 3.0 的下限会被推到
    ``#6683f0``（对比 3.23），而网页侧原本手挑的是 ``#3559c7``（对比 5.75）。
    自动算的结果是**默认主题在浅色下比改动前更差**——一个"改进了配色系统"
    却让默认观感退步的改动，是最不该发生的那种。

    自定义主色没有手挑值可依，只能走守卫（下限见 :data:`MIN_ACCENT_CONTRAST`）。
    """


PRESETS: tuple[Preset, ...] = (
    Preset("indigo", "蓝紫", "#6b8afd", "#3559c7"),
    Preset("jade", "青绿", "#3fb950", "#1f7a34"),
    Preset("amber", "琥珀", "#d9a441", "#8a6410"),
)
"""三个预设主色。**唯一真源**——预设色块、默认值、测试断言全从这一处读。

第一个是默认值，也是网页侧 ``--accent`` 的既有取值，所以"没设置过"与"设成默认"
看起来完全一样，不需要额外的空值分支。

``light_accent`` 三支都落在对比度 5.0 上下（``#3559c7`` 是网页侧原本就有的那一支，
5.75），彼此相当——**不齐的话，换预设会顺带改变"浅色下文字有多清楚"**，
而那是个用户没在调、也不该被顺带改掉的量。``test_theme.py`` 把这区间钉住。
"""

#: 用户自定义主色时 :attr:`Theme.preset` 的取值。它与三个预设 key 并列，
#: 于是"当前选的是自定义还是某个预设"是一个字段就能回答的问题。
CUSTOM_PRESET = "custom"

#: **自定义**主色在浅色模式下要过的对比度下限。
#:
#: 比 :data:`~biosnn_memterm.ui.color.MIN_GRAPHIC_CONTRAST`（3.0）高，因为它俩
#: 管的不是一件事：3.0 是 WCAG 对**图形**（按钮底、图标）的要求，而主色在网页侧
#: 主要当**链接与强调文字**的颜色用，那是正文，AA 的线是 4.5。
#:
#: 三个预设不走近路——它们各带一支手挑的 ``light_accent``，落在 5.0 上下。
#: 这个常量只管自定义色，是**兜底**，不是设计目标。
MIN_ACCENT_CONTRAST = 4.5

#: 壁纸可见度的默认值。**18 不是一个随手定的数**：它等价于网页侧原本写死的
#: ``color-mix(in srgb, var(--bg) 82%, transparent)``——即遮罩 82% 不透明、
#: 壁纸透出 18%。默认值取它，是为了让"接上壁纸功能"这件事本身不改变任何
#: 已有用户的观感。
WALLPAPER_VISIBLE_DEFAULT = 18

#: 可见度上限。**70 之后遮罩只剩 0.30 的不透明度**，而消息是铺满整个对话区
#: 滚动的——没有哪一块区域是"纯装饰、可以不看"的。再高就是拿一张照片当背景、
#: 正文压在上面，可读性没有兜底。这是个刻意的硬上限，不是没做完。
WALLPAPER_VISIBLE_MAX = 70


@dataclass(frozen=True)
class Tokens:
    """一套（明或暗）界面色。字段名是 Python 风格，QSS 里的占位符名另有一张
    映射表（:func:`_substitutions`）——**不靠字段名自动对齐**，那样改个字段名
    就会静默换掉一个 QSS 占位符。"""

    bg: str
    """窗口底色。"""
    bg_elevated: str
    """浮起来的一层：面板、输入框、代码块。"""
    bg_sunken: str
    """沉下去的一层：列表空白处、文本框内部。"""
    fg: str
    fg_muted: str
    fg_strong: str
    border: str
    danger: str


DARK_TOKENS = Tokens(
    bg="#0f1116",
    bg_elevated="#171a22",
    bg_sunken="#0b0d12",
    fg="#e6e9f0",
    fg_muted="#8b93a7",
    fg_strong="#ffffff",
    border="#252a36",
    danger="#e2705f",
)

LIGHT_TOKENS = Tokens(
    bg="#f6f7f9",
    bg_elevated="#ffffff",
    bg_sunken="#eceef3",
    fg="#1f2430",
    fg_muted="#6a7286",
    fg_strong="#000000",
    border="#e2e5ec",
    # **浅色下必须单独给一个更深的红。** 网页侧原本的浅色块没有覆盖
    # ``--danger``，于是浅色模式下用的还是暗色那支 ``#e2705f``（一支鲑鱼粉），
    # 在白底上对比度只有 2.5 左右——错误提示恰恰是最需要看得清的那一句。
    # 这里取 ``#c0392b``，与上下文指示灯 OVER 态用的是同一个值。
    danger="#c0392b",
)

_TOKENS_BY_MODE = {MODE_DARK: DARK_TOKENS, MODE_LIGHT: LIGHT_TOKENS}


@dataclass(frozen=True)
class _GlassAlphas:
    """磨砂按钮那一组不透明度。**按明暗分列，不共用一个数。**

    颜色只有一处来源——``tokens.fg_strong``：暗态是白、亮态是黑。这是有意的，
    于是填充色随主题自动反转，**这一整套不引入任何新的色值，只引入 alpha**。

    但"亮"在两种模式下是**两个方向**：暗态往白推是变亮，亮态往黑推是变暗。
    所以两张表里 ``top`` 与 ``bottom`` 的大小关系是**反的**——暗态上亮下暗
    （``top > bottom``），亮态是黑叠得少才更亮（``top < bottom``）。两处各写一遍
    同一个数就会把这个关系写丢，症状是亮色模式下按钮看着"上暗下亮"，像光从
    下面打上来。

    **三道边（``edge_*``）才是"这是玻璃"读得出来的那一半。** 一层半透明填充加
    一道均匀的描边，看起来只是"一块浅一点的圆角矩形"——因为这一侧没有模糊、
    也没有投影（QSS 两样都没有），"厚度"只能靠**上下两条亮度不同的边**读出来：
    顶边那道亮线是光，底边那道暗线是它贴着的东西。三道边各给一个数、而不是给
    一个数让四个方向共用，是因为 ``border`` 简写会把四个方向一起写掉，那样顶底
    就分不出来了——而这个按钮的"玻璃感"恰好全在顶底这条差上。

    填充那几个值全部 ≤ 0.16（``color.rgba`` 那段论据的边界），
    ``test_theme_glass.py`` 逐格把成品色的对比度算出来。三道边**不在这条约束
    里**——描边不承载文字，它亮一点只影响"这个按钮有没有边界"。实测最狠的一格
    （暗态悬停、叠在 ``bg_elevated`` 上）字还有 8.69 的对比度，离 4.5 那条线很远。

    **亮态的填充 alpha 看起来偏大，不是笔误。** 亮态的底几乎是白的（``bg``
    ``#f6f7f9``），往黑里叠的可动范围只有 246→0 里的一小段：叠 0.02 与叠 0.05
    在屏幕上分不出来（差 9 级里的一级）。所以亮态要拿到"这块是灰玻璃"的观感，
    就得叠到 0.05 以上；暗态反过来，白色叠 0.10 已经很亮，再多就糊成一片。
    """

    top: float
    """渐变顶端（更亮的那一头）。"""
    bottom: float
    top_hover: float
    bottom_hover: float
    edge_top: float
    """顶边那道亮线。用 :data:`_GLASS_EDGE_TOKENS` 里的 ``hi``。"""
    edge_side: float
    """左右两条边。上下有亮暗之分，左右没有——光是从上面来的，不是从旁边。"""
    edge_bottom: float
    """底边那道暗线。用 ``lo``——"比填充更暗"这个方向，两种模式要的不是同一个人。"""


#: 磨砂那两道边各取哪个 token：``(hi, lo)`` = （最亮的、最暗的）。
#:
#: **不能像填充那样统一用 ``fg_strong``。** 填充要的是"离开背景"的方向，那个方向
#: 随明暗反转，``fg_strong`` 正好承担它；而顶边要的是**绝对意义上的亮**（光是白
#: 的），底边要的是**绝对意义上的暗**（影子是黑的）。这两个都**不随主题反转**：
#: 亮态的 ``fg_strong`` 是黑，拿它当高光会在按钮顶上画出一道暗线。
#:
#: 而且**描边是画在填充之上的**（样式表的背景铺满整个边框盒，边再压上去）——
#: 叠任何黑都只会更黑。所以亮态的高光**不可能是 "少叠一点黑"**，它必须换一个色源。
#: 这条是实测出来的，不是推的，见 ``test_ui_theme_glass.py``：那边把真渲染出来的
#: 顶边像素与填充像素比过，亮态下黑边确实比填充更暗。
#:
#: 两格都取"该模式下最极端的那个 token"：暗态最亮是 ``fg_strong``（白）、最暗是
#: ``bg_sunken``；亮态最亮是 ``bg_elevated``（白）、最暗是 ``fg_strong``（黑）。
#: 暗态的底边为什么是 ``bg_sunken`` 而不是纯黑：填充已经是"比背景亮"的一层，
#: 要压出影子就得比背景还暗，而那个方向能拿到的最远处就是下沉层。
_GLASS_EDGE_TOKENS: dict[str, tuple[str, str]] = {
    MODE_DARK: ("fg_strong", "bg_sunken"),
    MODE_LIGHT: ("bg_elevated", "fg_strong"),
}


_GLASS_ALPHAS = {
    MODE_DARK: _GlassAlphas(
        top=0.11,
        bottom=0.04,
        top_hover=0.16,
        bottom_hover=0.07,
        edge_top=0.22,
        edge_side=0.10,
        edge_bottom=0.85,
    ),
    MODE_LIGHT: _GlassAlphas(
        top=0.06,
        bottom=0.14,
        top_hover=0.10,
        bottom_hover=0.16,
        edge_top=0.85,
        edge_side=0.30,
        edge_bottom=0.22,
    ),
}
"""磨砂的 alpha 表。**浅色那一支必须有**——理由与 :data:`LIGHT_TOKENS` 里那条
``danger`` 的备注同源：漏给一侧，那一侧就会静默用上另一侧的数值。"""


@dataclass(frozen=True)
class Theme:
    """用户选定的外观。**这是被持久化的那个对象。**"""

    preset: str = PRESETS[0].key
    accent: str = PRESETS[0].accent
    """主色。**选了预设时它由预设决定，不由这个字段决定。**

    见 :meth:`__post_init__`：``preset`` 不是 ``"custom"`` 时这里会被改写成该
    预设的 ``accent``。留这个字段是为了让"自定义"有地方放色值，而不是为了与预设
    并存两套真源。
    """
    mode: str = MODE_SYSTEM
    wallpaper: str = ""
    """``wallpapers_dir()`` 里的**文件名**，不是完整路径。

    存文件名而不是路径，是为了让"用户把数据目录整体搬走"或"换了盘符"之后
    壁纸仍然找得到；存路径的话那两种情况下设置会指向一个不存在的文件，而
    症状是"壁纸莫名其妙没了"，看不出是路径问题。
    """
    wallpaper_visible: int = WALLPAPER_VISIBLE_DEFAULT

    def __post_init__(self) -> None:
        if self.mode not in _MODE_CHOICES:
            raise ValueError(f"未知的明暗模式：{self.mode!r}")
        if self.preset not in (*(item.key for item in PRESETS), CUSTOM_PRESET):
            raise ValueError(f"未知的预设：{self.preset!r}")
        # **预设说了算。** 选了预设还带着一个对不上的 ``accent``，是"两套真源"，
        # 而两套真源必然会在某条路径上分叉——表现是"设置窗口里勾的是蓝紫，
        # 界面画的是上次自定义的红"。这里当场收敛掉：不是 custom 就把 accent
        # 改写成预设的值。frozen 所以要绕过 setattr。
        #
        # 这条也顺手兜住了损坏的设置：注册表里 preset 与 accent 不一致时（手改、
        # 或某次写入只成功了一半），下一次读出来就是自洽的。
        #
        # 直接遍历 PRESETS 而不是调 :func:`preset_by_key`：那个函数定义在本类
        # **之后**，而模块级就有一句 ``DEFAULT_THEME = Theme()``。
        if self.preset != CUSTOM_PRESET:
            for _item in PRESETS:
                if _item.key == self.preset:
                    object.__setattr__(self, "accent", _item.accent)
                    break
        # 主色在这里就解析一次，**并归一化成 "#rrggbb" 小写**。非法主色不许进入
        # Theme——等到渲染时才发现，那时已经写进设置、下次启动照样崩，而报错的
        # 地方离原因很远。归一化同样重要：用户可能从取色器拿到 "#ABC"，而设置里
        # 存的形式**必须唯一**，否则"同一个颜色"会有多个字符串表示，比对与去重
        # 都会出错（"为什么我明明选了默认色，设置窗口却说这是自定义"）。
        object.__setattr__(self, "accent", to_hex(parse_hex(self.accent)))
        if not 0 <= self.wallpaper_visible <= WALLPAPER_VISIBLE_MAX:
            raise ValueError(
                f"壁纸可见度必须在 0–{WALLPAPER_VISIBLE_MAX} 内：{self.wallpaper_visible!r}"
            )


DEFAULT_THEME = Theme()


@dataclass(frozen=True)
class Resolved:
    """一个 :class:`Theme` 在当前明暗下**解析完**的样子。

    与 :class:`Theme` 分开是必要的：``Theme.mode`` 可以是 ``"system"``，而
    ``Resolved.mode`` 一定是 ``"dark"`` 或 ``"light"``；``Theme.accent`` 是用户
    选的原色，而 ``Resolved.accent`` 可能已经为浅色模式调暗过。把这两者混成一个
    对象，"用户选的是什么"与"实际画出来的是什么"就再也分不开了——而设置窗口
    恰恰要同时显示这两件事。
    """

    mode: str
    accent: str
    """实际使用的主色（可能已被 :func:`~biosnn_memterm.ui.color.ensure_contrast` 调整）。"""
    accent_adjusted: bool
    """主色是否被调过。界面要如实说出来，不能悄悄换掉用户选的色。"""
    tokens: Tokens
    roles: dict[str, str] = field(default_factory=dict)


def preset_by_key(key: str) -> Preset | None:
    """按 key 找预设；找不到返回 ``None``（自定义主色就是这种情况）。"""
    for item in PRESETS:
        if item.key == key:
            return item
    return None


def clamp_mode(value: object) -> str:
    """把任意输入收成合法的明暗模式，认不出就回落 ``"system"``。

    与 :func:`clamp_wallpaper_visible` 是同一类东西，存在的理由也一样：
    :class:`Theme` 对未知 mode 是**抛**的，而 :class:`~.settings_store.SettingsStore`
    承诺读设置绝不抛（见那个模块的注释）。所以"收一个不可信的值"这件事要有一个
    既不在构造函数里、也不在存储层里的落点——放这里，两边都调它。
    """
    return value if value in _MODE_CHOICES else MODE_SYSTEM  # type: ignore[return-value]


def clamp_wallpaper_visible(value: object) -> int:
    """把任意输入收成合法的可见度。``SettingsStore`` 用它兜住损坏的值。"""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return WALLPAPER_VISIBLE_DEFAULT
    return max(0, min(WALLPAPER_VISIBLE_MAX, number))


def effective_mode(theme: Theme, os_mode: str) -> str:
    """把 ``"system"`` 解析成明暗之一。

    ``os_mode`` 是操作系统给的说法，可能是 ``"dark"`` / ``"light"`` / ``"unknown"``
    （Qt 的 ``ColorScheme.Unknown``）。**只有 ``"light"`` 会得到浅色**，其余一律
    暗色：本项目的默认观感是暗色，而"问不到系统偏好"时回落到浅色会让一个
    暗色主题的应用莫名其妙变白。
    """
    if theme.mode in (MODE_DARK, MODE_LIGHT):
        return theme.mode
    return MODE_LIGHT if os_mode == MODE_LIGHT else MODE_DARK


def resolve(theme: Theme, os_mode: str = "unknown") -> Resolved:
    """算出实际要画的那套颜色。

    浅色模式下主色要换一支：网页侧原本的浅色 ``--accent`` 是 ``#3559c7`` 而不是
    暗色那支 ``#6b8afd``，因为后者在白底上对比度只有 2.93。两条路——

    * **预设**：走它自己手挑的 :attr:`Preset.light_accent`。见那个字段的说明：
      守卫只会把颜色推到"刚好过线"，而主色在这里当链接文字用，值得一个更好的值。
    * **自定义**：没有手挑值可依，走 :func:`~biosnn_memterm.ui.color.ensure_contrast`，
      下限 :data:`MIN_ACCENT_CONTRAST`。

    两条路都把"换过了"这个事实带出去（:attr:`Resolved.accent_adjusted`），
    设置窗口据此如实说明，而不是悄悄换掉用户选的色。
    """
    mode = effective_mode(theme, os_mode)
    tokens = _TOKENS_BY_MODE[mode]
    accent, adjusted = theme.accent, False
    if mode == MODE_LIGHT:
        chosen = preset_by_key(theme.preset)
        if chosen is not None:
            accent = chosen.light_accent
        else:
            accent, adjusted = ensure_contrast(theme.accent, tokens.bg, minimum=MIN_ACCENT_CONTRAST)
        adjusted = adjusted or accent != theme.accent
    return Resolved(
        mode=mode,
        accent=accent,
        accent_adjusted=adjusted,
        tokens=tokens,
        roles=derive_roles(accent, mode, bg=tokens.bg),
    )


def build_qss(resolved: Resolved) -> str:
    """把模板里的占位符换成实际颜色，得到可以 ``setStyleSheet`` 的字符串。

    用 :meth:`string.Template.substitute` 而不是 :meth:`str.format`：QSS 本身
    满地都是 ``{}``，``format`` 会把它们全当占位符。也刻意**不用**
    ``safe_substitute``——模板里写了一个没有对应值的占位符，那是模板与代码脱节，
    要当场炸掉；``safe_substitute`` 会把它原样留在样式表里，Qt 静默忽略那一条。
    """
    mapping = _substitutions(resolved)
    return Template(_template_source()).substitute(mapping)


@lru_cache(maxsize=1)
def _template_source() -> str:
    """读 ``theme.qss``。缓存一次：它在进程生命周期里不会变，而每次换主题
    重新读盘是白费——换主题本来就已经要重设整张样式表了。"""
    return Path(__file__).with_name("theme.qss").read_text(encoding="utf-8")


def _substitutions(resolved: Resolved) -> dict[str, str]:
    """QSS 占位符名 → 颜色。**显式列举，不靠 ``vars()``**：靠字段名自动对齐的话，
    重命名一个字段就会静默换掉一个占位符（或让 ``substitute`` 报一个与真正
    原因无关的 KeyError）。"""
    tokens = resolved.tokens
    glass = _GLASS_ALPHAS[resolved.mode]
    highlight = tokens.fg_strong
    # 三道边走另一对色源（最亮的 / 最暗的），理由见 :data:`_GLASS_EDGE_TOKENS`：
    # 填充那支要的是"离开背景"的方向，那个方向随明暗反转；边要的是绝对意义上的
    # 亮与暗，不随主题反转。混用会让亮态的顶边变成一条暗线。
    edge_hi_name, edge_lo_name = _GLASS_EDGE_TOKENS[resolved.mode]
    edge_hi = getattr(tokens, edge_hi_name)
    edge_lo = getattr(tokens, edge_lo_name)
    return {
        "bg": tokens.bg,
        "bgElevated": tokens.bg_elevated,
        "bgSunken": tokens.bg_sunken,
        "fg": tokens.fg,
        "fgMuted": tokens.fg_muted,
        "fgStrong": tokens.fg_strong,
        "border": tokens.border,
        "danger": tokens.danger,
        "glassTop": rgba(highlight, glass.top),
        "glassBottom": rgba(highlight, glass.bottom),
        "glassTopHover": rgba(highlight, glass.top_hover),
        "glassBottomHover": rgba(highlight, glass.bottom_hover),
        # 三道边分开给，而且**用的不是填充那个色源**。顶边与底边必须各占一个
        # 占位符：``border`` 简写会四个方向一起写，用一条边代替的话上下就一样
        # 亮，"光从上打下来"当场没了。左右两条走 ``hi``——它们没有亮暗之分，
        # 只是给圆角留一道薄边，用最亮的那支最省事（也最不容易在亮态出错）。
        "glassLineTop": rgba(edge_hi, glass.edge_top),
        "glassLine": rgba(edge_hi, glass.edge_side),
        "glassLineBottom": rgba(edge_lo, glass.edge_bottom),
        # 按下态**刻意不在磨砂那一套里**：磨砂的高光是"变亮"，而"按下去了"只能靠
        # 变暗来表达。暗态的高光色是白，拿它再叠只会让按钮更亮，读起来像悬停。
        # 所以这一格直接给下沉层的 token 色（不透明），与 ``QPushButton:checked``
        # 那类"态"用的是同一族颜色。它留着 ``glass`` 这个名字只是为了"按钮那一段
        # 只出现 $glass*"，别在别处也这么用。
        "glassPressed": tokens.bg_sunken,
        # 主按钮那道纵向渐变的两头。**顺序随明暗翻**：``accentHover`` 是
        # "离背景更远"的那一支，而"更远"在暗态是更亮、在亮态是更暗，所以
        # "上亮下暗"要的那一对在两种模式下正好互换。写死一个顺序的话，亮色模式下
        # 主按钮就是"光从下面打上来"——说不出哪里不对，但看着就是不对。
        "accentTop": resolved.roles["accentHover" if resolved.mode == MODE_DARK else "accent"],
        "accentBottom": resolved.roles["accent" if resolved.mode == MODE_DARK else "accentHover"],
        **resolved.roles,
    }


def describe_accent(theme: Theme, resolved: Resolved) -> str:
    """设置窗口里那一行说明。**换过就要说**，见 :attr:`Resolved.accent_adjusted`。

    措辞是"换用"而不是"调暗"：预设走的是手挑的浅色值（见
    :attr:`Preset.light_accent`），那不是把原色压暗算出来的，说"调暗"会让人以为
    程序在自作主张地改他的色。``theme.accent`` 已在 ``__post_init__`` 里归一化过，
    可以直接拼。
    """
    if not resolved.accent_adjusted:
        return resolved.accent
    return f"{resolved.accent}（浅色模式下换用这一支；你选的是 {theme.accent}）"
