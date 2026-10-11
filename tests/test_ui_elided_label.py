"""``ElidedLabel``：顶栏那条状态文字不再被窗口右边裁掉。

标 ``gui``，但**不碰 WebEngine、不建 ``MainWindow``**。这里要验的东西必须有一个真
``QApplication`` 才能问：字体度量、布局给多宽、样式表有没有落到调色板上。

四条断言各守一件事：

* ``test_text_and_tooltip_keep_the_whole_thing`` —— **内容与显示分开**。省略只发生在
  画面上，``text()`` 仍给全文。``tools/_smoke_window.py`` 读的就是 ``text()``，
  它要是变成省略后的字符串，那个冒烟脚本会开始"看不见"状态变化而不报错。
* ``test_a_stylesheet_colour_reaches_the_palette`` —— :meth:`ElidedLabel.paintEvent`
  自己取色，取的是控件调色板。Qt 的样式表引擎会不会把 ``color:`` 落到调色板上，
  **是本文件唯一一个"猜错就会静默画成黑字"的假设**，所以它单独一条，把机理写下来。
* ``test_it_can_actually_be_squeezed`` —— 不覆盖 ``minimumSizeHint`` 的 ``QLabel``
  会坚持放下全文的宽度，于是把顶栏撑宽、把别人挤出去。这条钉住它肯被压。
* ``test_the_status_label_actually_gets_room_in_the_real_header`` —— **上面三条全是拿
  孤零零一个标签量的**，而那个标签的 ``sizeHint``、``minimumSizeHint`` 一直都没错，
  所以它们绿着、状态文字却从没显示过。宽度是**布局**给出来的，只有放进真顶栏才问得到。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from biosnn_memterm.ui.widgets import ElidedLabel

pytestmark = pytest.mark.gui

_LONG = "正在生成回答，已收到 128 个 token，上一个请求用了 3.4 秒 " * 4


@pytest.fixture(scope="module")
def qapp() -> Any:
    app = QApplication.instance() or QApplication([])
    yield app
    app.setStyleSheet("")


def _elided_width(label: ElidedLabel, width: int = 60) -> str:
    """按给定宽度算出会被画出来的那行字。**不真去画**——离屏下截图比对脆且没信息，
    要断言的是"省略发生了"，不是"像素长什么样"。"""
    return label.fontMetrics().elidedText(label.text(), label._mode, width)


def test_text_and_tooltip_keep_the_whole_thing(qapp) -> None:
    label = ElidedLabel(_LONG)
    label.resize(60, 24)
    assert label.text() == _LONG
    assert label.fullText() == _LONG
    assert label.toolTip() == _LONG


def test_it_actually_elides_when_narrow(qapp) -> None:
    """宽 60 时画出来的那行**短于全文且带省略号**。只断言"变短了"不够——那也可能是
    被硬裁了。省略号是"这里还有字"的唯一提示，所以它必须在。"""
    label = ElidedLabel(_LONG)
    label.resize(60, 24)
    shown = _elided_width(label)
    assert len(shown) < len(_LONG)
    assert shown.endswith("…")


def test_a_roomy_label_shows_everything(qapp) -> None:
    """够宽时**不许**加省略号。反过来的毛病（明明放得下却省了）同样难查，
    而且更常见——它看起来像"内容被截断了"。"""
    label = ElidedLabel("短状态")
    label.resize(600, 24)
    assert _elided_width(label, 600) == "短状态"


def test_it_can_actually_be_squeezed(qapp) -> None:
    """``minimumSizeHint`` 必须远小于 ``sizeHint``。

    **这一条是它能待在顶栏里的原因。** 两个数一样大的话，布局会为了放下全文而把顶栏
    撑宽——也就是说不修状态文字，加「设置」按钮之后窗口会先变宽再开始裁。

    策略断言写的是 ``Preferred`` 而**不是** ``Ignored``。这里原本写的是 ``Ignored``，
    那是把手段当成了目的：``Ignored`` 的 ``IgnoreFlag`` 连 ``minimumSizeHint`` 一起
    无视，于是布局给这个标签 **0 像素**，"肯被压"变成了"根本没宽过"。真正要的是
    ``Preferred``——``sizeHint`` 是它想要的宽度，布局可以一路压到 ``minimumSizeHint``，
    而窗口的最小宽度由后者决定，所以顶栏照样不会被撑宽。
    """
    label = ElidedLabel(_LONG)
    assert label.minimumSizeHint().width() < label.sizeHint().width()
    assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred


def test_the_status_label_actually_gets_room_in_the_real_header(qapp) -> None:
    """**这条拿真顶栏量，不拿孤零零一个标签量。**

    上面那些测的都是标签自己的性质——它的 ``sizeHint``、``minimumSizeHint`` 一直
    都是对的，所以**它们全都绿着，而顶栏那段状态文字从来没有显示过**。2026-10-10
    实测：``Ignored`` 策略下标签宽度是 **0**，``paintEvent`` 拿 0 去算省略号得到空串
    （``elidedText(全文, ElideRight, 0) == ''``），于是 `text()` 里内容齐全、tooltip
    里内容齐全、画面上一个像素都没有。要抓的是**布局真的给它多少空间**，所以必须
    把它放进那个真实的 ``QHBoxLayout`` 里。

    不建 ``MainWindow``（离屏下 QtWebEngine 一渲染就硬杀进程）。``_build_header``
    只建控件、不接信号，所以拿一个普通 ``QWidget`` 当 ``self`` 就能调它——哪天它长出
    新依赖，这里会以 ``AttributeError`` 说出来，而不是悄悄测了个别的东西。
    """
    from biosnn_memterm.ui.main_window import MainWindow

    text = "● Ollama 就绪 · 4 个对话模型"
    host = QWidget()
    header = MainWindow._build_header(host)  # type: ignore[arg-type]
    host.status_label.setText(text)
    header.resize(1600, 52)
    header.show()
    header.layout().activate()
    qapp.processEvents()

    label = host.status_label
    assert label.width() > 0, "布局给了状态标签 0 像素——它一个字节都画不出来"
    shown = label.fontMetrics().elidedText(label.fullText(), label._mode, label.width())
    assert shown.startswith("●"), f"状态文字画不出来（宽度 {label.width()}）：{shown!r}"
    assert shown == text, "1600 像素的顶栏放得下这句，不该省略"


def test_a_stylesheet_colour_reaches_the_palette(qapp) -> None:
    """**本文件里唯一在验机理的一条。**

    :meth:`ElidedLabel.paintEvent` 不调 ``super().paintEvent()``，所以它拿不到 Qt
    替它上的颜色，得自己从 ``QStyleOption.initFrom(self)`` 取。取不到的话症状是
    **状态文字画成默认黑字**——在暗色底上就是几乎看不见，而且不报任何错。

    这里证明的是：QSS 里的 ``color:`` 会被 Qt 的样式表引擎落到**控件自己的调色板**
    上（不是只存在字符串里），于是那条取色路径有效。
    """
    qapp.setStyleSheet("QLabel#status { color: #8b93a7; }")
    label = ElidedLabel("x")
    label.setObjectName("status")
    qapp.processEvents()  # polish 发生在事件处理里
    assert label.palette().color(QPalette.ColorRole.WindowText).name() == "#8b93a7"
    qapp.setStyleSheet("")


def test_the_default_pen_is_the_palette_window_text(qapp) -> None:
    """没被样式表点名时退回调色板的 ``WindowText``——也就是主题给的正文色。
    写死一个常量在这里，暗色主题下就会画成不可见的字。"""
    label = ElidedLabel("x")
    label.setObjectName("nobody-styles-me")
    qapp.processEvents()
    assert label.palette().color(QPalette.ColorRole.WindowText) == qapp.palette().color(
        QPalette.ColorRole.WindowText
    )


@pytest.mark.parametrize(
    "mode",
    [Qt.TextElideMode.ElideRight, Qt.TextElideMode.ElideMiddle, Qt.TextElideMode.ElideLeft],
)
def test_every_elide_mode_shortens(qapp, mode: Qt.TextElideMode) -> None:
    """三种省略模式都要真的起作用。样式留了口子（``mode`` 是构造参数），
    那就把三种都验一遍——只验默认那种的话，换个模式坏了没人知道。"""
    label = ElidedLabel(_LONG, mode=mode)
    label.resize(60, 24)
    shown = label.fontMetrics().elidedText(_LONG, mode, 60)
    assert len(shown) < len(_LONG)
    assert "…" in shown
