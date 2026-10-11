"""顶栏那条上下文指示灯：接线对不对、以及"没有这一轮"画成什么颜色。

阈值判定住在 :mod:`biosnn_memterm.session.indicator`，由核心腿的
``test_context_indicator.py`` 断言（``over_budget`` 优先于比值、0.85 的边界、
比值可以超过 100%）。这里只钉两件界面侧的事：**数值取自账本**，以及**没有这一轮
时不能画成绿色**。

这个文件标 ``gui``：核心腿用 ``--ignore-glob='tests/test_ui_*'`` 排除（**``--ignore``
不认通配符**，写成它一条都排不掉且不报错），与气体仓的分腿纪律一致。

**不构造真的 ``MainWindow``**（它一连 Ollama、起多个线程、加载 WebEngine）。
``_show_context`` 只碰三个属性，所以这里给它一个只带这三个属性的替身——
``test_ui_worker_track.py`` 对 ``MainWindow._track`` 用的是同一套办法。
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from biosnn_memterm.session.assemble import Assembly, AssemblyLedger
from biosnn_memterm.session.budget import TokenBudget
from biosnn_memterm.session.indicator import ContextState
from biosnn_memterm.ui.main_window import _CONTEXT_COLORS, MainWindow

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Stub:
    """``_show_context`` 真正用到的东西只有三个。"""

    _show_context = MainWindow._show_context

    def __init__(self, *, usable: int = 8900) -> None:
        from PySide6.QtWidgets import QLabel, QProgressBar

        self.context_bar = QProgressBar()
        self.context_label = QLabel()
        self._budget = SimpleNamespace(usable=usable)


def assembly(*, total: int, over: int = 0) -> Assembly:
    ledger = AssemblyLedger(budget=TokenBudget(window=16384), total_tokens=total, over_budget=over)
    return Assembly(messages=(), ledger=ledger)


def test_no_round_yet_is_empty_not_green(qapp: Any) -> None:
    """**没有这一轮 ≠ 额度用了 0%。**

    Qt 的 ``QProgressBar`` 默认就是绿色填充。不显式画成空态的话，"还没发过请求"
    看起来与"一切正常"一模一样——而这个灯存在的全部意义就是"不用细看也知道
    快满了没有"。
    """
    stub = Stub()
    stub._show_context(None)

    assert stub.context_label.text() == "上下文 —"
    assert stub.context_bar.value() == 0
    stylesheet = stub.context_bar.styleSheet()
    assert _CONTEXT_COLORS[ContextState.EMPTY] in stylesheet
    assert _CONTEXT_COLORS[ContextState.OK] not in stylesheet


def test_over_budget_shows_the_real_ratio_and_clamps_only_the_bar(qapp: Any) -> None:
    """比值**如实显示**，夹紧的只有进度条。

    把文字也夹到 100% 的话，"超了多少"就没地方看了——而超出去的量恰好决定了
    服务端会截掉多少条最早的消息（丢的是人格锚所在的位置）。
    """
    stub = Stub(usable=8900)
    stub._show_context(assembly(total=12000, over=3100))

    assert stub.context_label.text() == "上下文 12.0k / 8.9k"
    assert stub.context_bar.value() == 100
    assert _CONTEXT_COLORS[ContextState.OVER] in stub.context_bar.styleSheet()


def test_the_light_changes_colour_at_the_warning_line(qapp: Any) -> None:
    """0.85 是分界：0.84 绿、0.85 黄。数值本身在核心腿里钉，这里钉它真的接到了颜色。"""
    stub = Stub(usable=1000)
    stub._show_context(assembly(total=840))
    assert _CONTEXT_COLORS[ContextState.OK] in stub.context_bar.styleSheet()

    stub._show_context(assembly(total=850))
    assert _CONTEXT_COLORS[ContextState.WARN] in stub.context_bar.styleSheet()
    assert stub.context_bar.value() == 85


def test_the_numbers_come_from_the_ledger(qapp: Any) -> None:
    """灯读的是**这一轮账本的** ``total_tokens``，不是各层手工相加的数。

    账本是唯一会被人拿去核对的东西；灯另算一遍的话，两个数会在某次改动之后悄悄
    分家——而分家的症状只是"灯的数字看起来有点小"，没人会去查。
    """
    stub = Stub(usable=8000)
    stub._show_context(assembly(total=5000))
    assert "5.0k / 8.0k" in stub.context_label.text()
    assert stub.context_bar.value() == 62  # round(5000*100/8000) = 62.5 → 62

    stub._show_context(assembly(total=1000))
    assert "1.0k / 8.0k" in stub.context_label.text()
    assert stub.context_bar.value() == 12
