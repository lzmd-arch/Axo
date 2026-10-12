"""截一张屏幕，做成能交给模型看的样子。**只在界面线程调。**

三件事，每件都对着一个实测或评审逼出来的失效：

1. **把自己那一块涂掉。** 用户点「允许」那一下就把这个窗口顶到了最前，所以不处理的
   话，截到的图以对话区为主角——模型看到的会是**它自己刚说的话**（还有人格锚、记忆
   卡片）。做法是涂成一块纯色，**不是把像素裁掉**：窗口很少贴在某一条边上，裁掉会在图
   中间留一道缝，而涂掉得到的仍是一张正常的整屏截图。"模型看不到自己"是那条决定，
   涂还是裁只是实现。

2. **缩到长边 ≤ :data:`MAX_WIDTH`。** 屏幕上都是文字，PNG 压得好；缩放同时省 token 与
   带宽，而 ``session.budget.IMAGE_TOKENS`` 那个先验常数正是按这个尺寸估的——两处对不
   上，额度算的就不是实际发出去的那张图。

3. **全黑是一种收场，不是成功。** 锁屏、UAC 安全桌面、独占全屏都会给出全黑**且
   ``isNull()`` 为假**的图（离屏平台的假屏幕也一样，见 ``docs/agent.md`` 那条实测）。
   不判的话，模型会对着黑图编出一段描述——而用户看到的是"它说屏幕上有个登录框"。

**为什么在这里而不是在工具里**：截图必须发生在界面线程，而工具跑在 worker 线程
（``ui/agent_worker.py`` 那条交接把它送过来）。这一层只认识 Qt，返回的是
:class:`~axo.agent.tools.Capture`，所以它能在 gui 腿里被单独测——除了像素那两条
（离屏的假屏幕恒为一张黑图，那两条只能在真屏幕上手验）。
"""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QIODevice, QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap, QScreen

from ..agent.tools import Capture

__all__ = ["MAX_WIDTH", "grab", "looks_blank", "mask_rect", "scale_down"]

MAX_WIDTH = 1600
"""交出去的图最大多宽（长边）。见模块文档第 2 条。"""

MAX_BYTES = 8 * 1024**2
"""PNG 的字节上限。超了就再缩一档——一张纯色满屏的截图也可能压不到这个数以下。"""

_MASK_COLOR = "#3f3f46"
"""涂掉自己那一块用的颜色。选中性深灰：它既不吸引注意，也不会被当成屏幕内容的一部分
（纯黑会与"屏幕是黑的"那条判据混起来）。"""

_BLANK_SAMPLE_STEP = 24
"""判全黑时的采样步长（像素）。逐像素看一眼 4K 截图要几十毫秒，而这一判据只需要
"大概是不是全黑"。"""

_BLANK_LEVEL = 12
"""每个通道都 ≤ 这个值，这个采样点算"黑"。留一点余量：某些驱动给出的不是纯 0。"""

_BLANK_SHARE = 0.98
"""采样点里这么大比例是黑的 → 判"整张几乎全黑"。不取 1.0：锁屏上常有一条时钟或提示。"""


def grab(
    screen: QScreen | None = None,
    *,
    exclude: QRect | None = None,
) -> Capture:
    """截一块屏幕，返回可以直接交给模型的 :class:`Capture`。

    Args:
        screen: 截哪一块屏；``None`` = 主屏。**多屏时由调用方给**（应用所在那块），
            因为"用户的注意力在哪块屏上"只有它知道。
        exclude: **设备像素**坐标下要涂掉的一块（我们自己窗口的矩形）。调用方负责
            把逻辑坐标乘上 ``devicePixelRatio``——那一处换算只有它知道窗口在哪。

    失败一律是 :class:`Capture` 里的 ``reason``（一句人话），**不抛**：工具那一侧对
    这几种收场要做的是同一件事——如实告诉模型没看成。
    """
    target = screen or QGuiApplication.primaryScreen()
    if target is None:
        return Capture(reason="没找到屏幕（显示器都断开了？）")

    whole = target.grabWindow(0)
    if whole.isNull() or whole.width() == 0 or whole.height() == 0:
        return Capture(reason="这一次没有截到屏幕")

    if exclude is not None and not exclude.isEmpty():
        whole = mask_rect(whole, exclude)

    sized = scale_down(whole)
    if looks_blank(sized):
        return Capture(reason="屏幕看起来是黑的（可能是锁屏、UAC 或独占全屏）")

    png = _encode_png(sized)
    if png is None:
        return Capture(reason="截到了，但编码成 PNG 时失败了")
    if len(png) > MAX_BYTES:
        # 再缩一档。**不静默发一张超大的图**：那一头是上下文额度，而超了的表现是
        # 服务端静默截断最早的消息（见 budget.IMAGE_TOKENS 那条"上限是硬的"）。
        sized = scale_down(sized, max_width=MAX_WIDTH // 2)
        png = _encode_png(sized)
        if png is None or len(png) > MAX_BYTES:
            return Capture(reason=f"这张屏幕图太大（超过 {MAX_BYTES // 1024**2} MB），没有带进来")

    return Capture(png=png, width=sized.width(), height=sized.height())


def mask_rect(pixmap: QPixmap, rect: QRect) -> QPixmap:
    """把 ``rect``（设备像素）那一块涂成一块纯色。

    涂而**不裁**的理由见模块文档第 1 条。涂之前先与原图求交：调用方给的矩形可能部分
    落在屏幕外（窗口一半在另一块屏上），而"画到画布外面去"在 Qt 里是静默的——那样就会
    留下一块**没涂干净**的区域，而它看起来完全正常。
    """
    bounded = rect.intersected(QRect(0, 0, pixmap.width(), pixmap.height()))
    if bounded.isEmpty():
        return pixmap
    masked = QPixmap(pixmap)
    painter = QPainter(masked)
    try:
        painter.fillRect(bounded, QColor(_MASK_COLOR))
    finally:
        painter.end()
    return masked


def scale_down(pixmap: QPixmap, *, max_width: int = MAX_WIDTH) -> QPixmap:
    """长边缩到 ``max_width`` 以内。已经够小就原样返回（**不放大**）。

    按 ``devicePixelRatio`` 无关的设备像素缩放：交出去的是一张普通位图，模型那边没有
    "逻辑像素"这个概念。
    """
    longest = max(pixmap.width(), pixmap.height())
    if longest <= max_width:
        return pixmap
    return pixmap.scaled(
        max_width,
        max_width,
        # 等比：截图里是文字，拉扁了就没法看。
        Qt.AspectRatioMode.KeepAspectRatio,
        # 平滑：最近邻会在文字上留下锯齿，而模型要读的就是那些字。
        Qt.TransformationMode.SmoothTransformation,
    )


def looks_blank(pixmap: QPixmap) -> bool:
    """整张是不是**几乎全黑**。见模块文档第 3 条。

    采样判而不是逐像素：这一判据要的是"大概是黑的"，而 4K 逐像素看一遍要几十毫秒，
    那笔钱花在截图这一个动作上不值当（用户点了允许，等在那一头的是 worker 线程）。
    """
    image = pixmap.toImage()
    if image.isNull():
        return False
    dark = 0
    total = 0
    for y in range(0, image.height(), _BLANK_SAMPLE_STEP):
        for x in range(0, image.width(), _BLANK_SAMPLE_STEP):
            total += 1
            colour = image.pixelColor(x, y)
            if (
                colour.red() <= _BLANK_LEVEL
                and colour.green() <= _BLANK_LEVEL
                and colour.blue() <= _BLANK_LEVEL
            ):
                dark += 1
    return total > 0 and dark / total >= _BLANK_SHARE


def _encode_png(pixmap: QPixmap) -> bytes | None:
    """编码成 PNG 字节。失败返回 ``None``（调用方给一句人话）。"""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    try:
        if not pixmap.save(buffer, "PNG"):
            return None
        return bytes(buffer.data())
    finally:
        buffer.close()
