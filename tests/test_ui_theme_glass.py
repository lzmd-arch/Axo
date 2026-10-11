"""Qt 到底会不会照那个写法合成——**这一条只能跑真渲染**。

"外壳这一侧做不了真模糊，所以磨砂只能靠半透明叠出拟态"这个说法，建立在两件
Qt 实现细节上：

1. QSS 的解析器认 ``rgba(r, g, b, a)`` 这种写法（而不是当成一个解析不了的值，
   把整条声明静默丢掉）；
2. 半透明填充、以及 ``qlineargradient`` 的 rgba 色标，都是**先与父层混合**再落笔
   ——不是先画成一个半透明的层、再被父层盖住。

两件都没有文档保证，而且**两件都错了也不报错**：整块按钮要么变成父层的颜色
（等于没画），要么变成别的什么。所以这里不测"好不好看"（那要靠眼睛，入口是
``tools/_smoke_window.py``），测的是"**它到底画没画上去**"：把窗口抓成位图，
在按钮的顶、底各取一个像素，与按 :func:`~biosnn_memterm.ui.color.rgba` 手算的
成品色逐通道比。

它单独一个文件，是因为它要 ``QApplication``；``test_ui_`` 这个前缀让它同时被核心
腿的 ``--ignore-glob`` 排除、被 gui 腿的 ``-m gui`` 收进来。
"""

from __future__ import annotations

import os
import re
from math import ceil, floor
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from biosnn_memterm.ui.color import RGB, parse_hex, relative_luminance
from biosnn_memterm.ui.theme import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    MODE_DARK,
    MODE_LIGHT,
    Theme,
    _substitutions,
    build_qss,
    resolve,
)

pytestmark = pytest.mark.gui

#: **把这条腿的平台钉到离屏**，与 ``tools/ci_local.py`` 给 gui 腿设的那一个一致
#: （CI 上也是它）。理由是这个文件的特殊性：它量的是**真渲染出来的像素**，而像素
#: 随平台变——原生 Windows 平台在 125% 缩放下 DPR 就是 1.25，一行逻辑像素落在设备网格
#: 上要跨两行，同一个断言在两种跑法下量的就是两张不同的图，于是"单跑这个文件绿、
#: 跑整条腿红"。
#:
#: 用 ``setdefault`` 而不是硬赋值：整条腿本来就是离屏，这里只是补上单独跑时缺的
#: 那一句；有人显式指定了别的平台时，不该被这个文件改掉（下面那几行按 DPR 换算
#: 设备坐标的代码就是给那种情形留的后路）。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: 探针窗口。**按钮比窗口小**，留出一圈能采到父层底色的边——"叠没叠上去"这个
#: 问题必须拿父层当参照物，铺满的话就没有参照物了。
_PROBE = "glassProbe"
_BOX = (160, 60)
_BUTTON_AT = (20, 10, 120, 40)

#: 手算与实渲之间允许的差。**4 是有余量的**：剩下的差来自渐变的插值与逐通道
#: 四舍五入，量级是 1–2。而"Qt 根本没合成"那条路（像素等于父层底色）差着十几到
#: 二十几，落在容差外面很远——这个门槛分得开这两种情况。
_TOLERANCE = 4

_RGBA = re.compile(r"rgba\((\d+), (\d+), (\d+), ([0-9.]+)\)")


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


def _over(top: str, bottom: str) -> RGB:
    """``top``（rgba 写法）叠在 ``bottom``（不透明色号）上。

    与核腿那份是同一套算法，这里重写一遍而不 import：那个文件在 gui 腿里也被
    收集，让这条腿反过来依赖它，红的时候要先想清楚是哪一边的问题。
    """
    match = _RGBA.fullmatch(top)
    assert match is not None, f"不是 rgba 写法：{top!r}"
    base = parse_hex(bottom)
    channels = (int(match[1]), int(match[2]), int(match[3]))
    alpha = float(match[4])
    return (
        round(base[0] * (1 - alpha) + channels[0] * alpha),
        round(base[1] * (1 - alpha) + channels[1] * alpha),
        round(base[2] * (1 - alpha) + channels[2] * alpha),
    )


def _render(
    mode: str, kind: str | None = None, *, bare_edges: bool = False
) -> tuple[dict[str, RGB], RGB]:
    """照真实的样子画一个按钮，返回 ``(采到的点, 父层底色)``。

    探针窗口的底色走的是**同一张样式表**里补的一条规则，不是另一次
    ``setStyleSheet``：控件级样式表会与它一起参与级联，分开设的话那条
    ``QPushButton`` 规则会被顶掉（``#contextBar`` 那里记过这条的另一个方向）。

    ``kind`` 传 ``"primary"`` 就画主按钮那条。**两种按钮都要画**：主按钮不写
    ``border-color``，那对边是从通用规则继承来的。这一格值不值得守，看下面那条
    测试的说明。

    ``bare_edges`` 把两道横边擦成透明，当作"**同一格没有这道边**"的对照。这条
    对照是斜边那对断言的依据，理由写在下面那条测试里。擦的是颜色不是宽度：那
    1 像素仍然占着，填充的位置不动，两次渲染的覆盖率完全一样，差的只剩那道边
    自己。
    """
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    holder = QWidget()
    holder.setObjectName(_PROBE)
    # 裸 ``QWidget`` 的样式表底色默认不画（要 ``WA_StyledBackground``）；不设这个
    # 属性的话抓出来的是系统调色板的灰，整套比对的参照物就错了。
    holder.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    button = QPushButton("好", holder)
    if kind is not None:
        # 动态属性必须在**设样式表之前**设：Qt 在 polish 时才按属性选选择器，
        # 而抓图之前不会再有第二次 polish。设晚了拿到的还是通用那条规则，
        # 探针画的就不是主按钮——而它看起来只是"颜色不对"，不会报错。
        button.setProperty("kind", kind)
    sheet = (
        build_qss(resolve(Theme(mode=mode), MODE_DARK))
        + f"\nQWidget#{_PROBE} {{ background: {tokens.bg}; }}"
    )
    if bare_edges:
        # 擦边的那条规则**必须连属性选择器一起列**：写在后面只赢同特异性的，
        # 而 ``QPushButton[kind="primary"]`` 比裸 ``QPushButton`` 高一档——只写
        # 后者的话，哪天有人往主按钮那条规则里补了边框，擦边就悄悄擦不掉，
        # 对照变回"自己跟自己比"，那对断言会以"永远成立"的方式绿掉。
        sheet += (
            '\nQPushButton, QPushButton[kind="primary"] {'
            " border-top-color: rgba(0, 0, 0, 0.0);"
            " border-bottom-color: rgba(0, 0, 0, 0.0); }"
        )
    holder.setStyleSheet(sheet)
    button.setGeometry(*_BUTTON_AT)
    holder.resize(*_BOX)

    image = holder.grab().toImage()
    left, top, width, height = _BUTTON_AT
    middle = left + width // 2

    # **抓出来的是设备像素，不是逻辑像素。** ``grab()`` 的位图按
    # ``devicePixelRatio`` 放大：原生 Windows 平台在 125% 缩放下 DPR 就是 1.25，
    # 160×60 的窗口抓出来是 200×75。拿逻辑坐标直接去索引，量到的位置就随 DPR 漂
    # ——而这里要量的恰好是"逻辑 1 像素宽"的那两道边，偏一格就落到填充或父层上。
    # 模块开头那句 ``setdefault`` 让默认跑法落在这条路径之外（离屏 DPR 是 1），
    # 留着它是给"有人显式指定了原生平台"那一种。
    dpr = image.devicePixelRatio()
    sample_x = round((middle + 0.5) * dpr)

    def at(device_y: int) -> RGB:
        colour = image.pixelColor(sample_x, device_y)
        return (colour.red(), colour.green(), colour.blue())

    def row(logical_y: float) -> RGB:
        """逻辑行**正中**那一个设备像素。填充分两处采，渐变在行间是连续的，
        取正中就代表这一行。"""
        return at(round((logical_y + 0.5) * dpr))

    def border_row(logical_y: float) -> RGB:
        """那道边**盖得最满**的那一个设备像素。

        DPR 不是整数时，一行边要跨两行设备像素，而跨在哪两行上取决于按钮被摆在
        哪——1.25 下逻辑 y=49 落在设备 61.25–62.5，其中 61 有七成五是边、62 只有
        五成。所以按**覆盖率**挑，不靠取整碰运气。

        **为什么不取那条带子里最极端的像素**（先前的写法）：带子最外侧那一行混着
        **窗口底**，而窗口底在暗态比填充还暗、在亮态比填充还亮。取极值等于请窗口底
        来替那道边作证——暗态下底边就算被 ``border-color`` 简写抹成一条亮线，混了
        窗底的边界行照样"比填充暗"，这条断言就永远绿。按覆盖率挑，邻行就进不来。
        """
        low, high = logical_y * dpr, (logical_y + 1) * dpr
        rows = range(floor(low), ceil(high))
        best = max(rows, key=lambda y: min(high, y + 1) - max(low, y))
        return at(best)

    # 上下各往里让 3 像素：外圈 1 像素是那道描边，再往边上是圆角，两者都不是
    # 填充色。横向取正中，避开圆角，也避开中间那个字。
    #
    # 另采**边本身**：顶上那一条与底下那一条，各取盖得最满的那一个设备像素。
    # 横向取正中，离两头的圆角很远，那两处是两条直线——这一对是"斜边画没画上"
    # 唯一的直接证据。
    #
    # **但"盖得最满"也只是那一格的一半。** 底色是个带圆角的矩形，光栅化到最外
    # 侧那一行时（第 10 与第 49 行）一半落在窗口底上，实测那一像素正是
    # ``0.5·填充 + 0.5·窗口底``（亮态下就是 (146, 162, 210)，与手算的
    # ``(0.5·(46,77,172) + 0.5·(246,247,249))`` 分毫不差）。边就压在这一格上。
    # 这一点是下面那条测试改用差分判据的全部原因——**别拿它去和填充内部的行比**。
    return (
        {
            "top": row(top + 3),
            "bottom": row(top + height - 4),
            "edge_top": border_row(top),
            "edge_bottom": border_row(top + height - 1),
        },
        parse_hex(tokens.bg),
    )


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
def test_qt_composites_the_glass_onto_the_parent(qapp: Any, mode: str) -> None:
    """顶、底两个采样点都等于"手算的成品色"。

    这是这个文件存在的全部理由：**Qt 真的把 rgba 当成半透明、并且与父层混合了**。
    换成不透明的写法（或 Qt 哪天不认这个写法），采到的就会是父层自己的颜色，
    于是这条当场红。
    """
    subs = _substitutions(resolve(Theme(mode=mode), MODE_DARK))
    tokens = DARK_TOKENS if mode == MODE_DARK else LIGHT_TOKENS
    sampled, parent = _render(mode)

    # 采样点离渐变的两头还差一点点（顶上的点在 10% 处、底下那个在 90% 处），
    # 这个差落在下面的容差里。
    expected = {
        "top": _over(subs["glassTop"], tokens.bg),
        "bottom": _over(subs["glassBottom"], tokens.bg),
    }
    for where in ("top", "bottom"):
        got, want = sampled[where], expected[where]
        delta = max(abs(a - b) for a, b in zip(got, want, strict=True))
        assert delta <= _TOLERANCE, (
            f"{mode}/{where}：画出来是 {got}，按 rgba 手算是 {want}（父层 {parent}）"
            f"——差 {delta}，说明那层没有叠在父层上"
        )


@pytest.mark.parametrize("mode", [MODE_DARK, MODE_LIGHT])
@pytest.mark.parametrize("kind", [None, "primary"], ids=["glass", "primary"])
def test_qt_draws_the_bevel_as_two_different_edges(qapp: Any, mode: str, kind: str | None) -> None:
    """两道边**各把它盖住的那一格按设计的方向推了一把**：上面提亮、下面压暗。

    这条守的失效方式很隐蔽：``border-top-color`` 与 ``border-color`` 简写长得
    几乎一样，而简写会**把四个方向一起写掉**（样式表里后写的那条赢），于是斜边
    静默消失。屏幕上只是"看着有点平"——不难看，只是不对，没人会往样式表上查。

    **主按钮这一路必须测。** 它不写 ``border-color``，那对边是从通用规则继承来的
    （**不写即时吃**）——所以哪天有人顺手往 ``QPushButton[kind="primary"]`` 里补一句
    ``border-color``，这里就红，而不是等用户看着主按钮说"这两个按钮不像一套"。

    这条也是**亮态那个错**的原始现场：那时顶边是一道黑线，实测报出
    ``顶边 (228, 229, 231) 没有比填充顶端 (230, 231, 232) 亮``。
    **这个错在核心腿上看不见**——那边的算法把边叠在**父层**上，亮态下"黑叠 0.03"
    确实是三格里最亮的。真正的画法是边压在**填充**上，叠黑只会更黑，所以
    ``test_theme_glass.py`` 里那两条也跟着改成了三层叠加。

    **判据是差分，不是"边与填充比大小"。** 两道边所在的那两行本身就只有一半
    覆盖率（见 :func:`_render` 里 ``border_row`` 的说明），那一像素是
    ``边 叠在 (0.5·填充 + 0.5·窗口底) 上``。玻璃按钮的填充是薄纱，把窗口底混进来
    差不了多少，所以它这一路"顶边比填充亮、底边比填充暗"两条都还量得出来；**主
    按钮那片是不透明的饱和主色**，混进来的那一半窗口底要么把它整个提亮（亮态，
    窗口底近白）、要么压暗（暗态，窗口底近黑），幅度远超 1 像素宽、alpha 0.22 的
    那道边。于是"底边比填充暗"在主按钮上**不是一条成立的断言**：

        亮态实测：底边那一格 (114, 126, 164)，填充底端 (46, 77, 172)
        相对亮度    0.212                对    0.091

    那不是画错了（那道边确实把它盖住的那一格从 (146, 162, 210) 压到了
    (114, 126, 164)），是**比错了对象**。所以主按钮这一路改与"同一格没有这道边"
    的渲染比（``bare_edges``）：两次渲染连覆盖率都一样，差的只剩那道边自己。
    这条关系在四个组合里都成立，守的也正是原来那条——简写把四边抹平、或者边
    压根没画上去，差分都当场塌。
    """
    sampled, _ = _render(mode, kind)
    bare, _ = _render(mode, kind, bare_edges=True)
    top, bottom = sampled["edge_top"], sampled["edge_bottom"]
    fill_top, fill_bottom = sampled["top"], sampled["bottom"]
    who = f"{mode}/{kind or 'glass'}"

    assert top != bottom, f"{who}：顶边与底边画成了同一个颜色（{top}），斜边没生效"
    assert relative_luminance(top) > relative_luminance(bare["edge_top"]), (
        f"{who}：顶边 {top} 没有把那一格提亮（把这道边擦掉之后是 {bare['edge_top']}）"
    )
    assert relative_luminance(bottom) < relative_luminance(bare["edge_bottom"]), (
        f"{who}：底边 {bottom} 没有把那一格压暗（把这道边擦掉之后是 {bare['edge_bottom']}）"
    )
    if kind is None:
        # 玻璃按钮上这两条还能直接与填充比——它那层填充是半透明的薄纱，窗口底
        # 混进边界行也改不了多少，那道边落下去就是几十级。主按钮比不得，见上。
        assert relative_luminance(top) > relative_luminance(fill_top), (
            f"{who}：顶边 {top} 没有比填充顶端 {fill_top} 亮，看不出上边界"
        )
        assert relative_luminance(bottom) < relative_luminance(fill_bottom), (
            f"{who}：底边 {bottom} 没有比填充底端 {fill_bottom} 暗，看不出下边界"
        )
