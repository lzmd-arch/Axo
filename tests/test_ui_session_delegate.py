"""会话列表的两行式 delegate。

标 ``gui``，不碰 WebEngine。这里断言的是**它有没有按说好的方式画**，而不是像素长
什么样（那要人眼，见 ``tools/_smoke_window.py``）。

三条要紧的：

* ``test_the_two_roles_do_not_collide_with_qt_or_each_other`` —— 自定义角色撞车会
  表现为"时间那行显示的是标题"，而 Qt 不会抱怨一句。
* ``test_it_falls_back_to_the_default_text`` —— 没设自定义角色的列表项**不能画成
  空白**。这个 delegate 装在会话列表上，但列表项是谁塞进去的它管不着；有人塞一条
  只设了 ``DisplayRole`` 的进去，屏幕上就少一行字。
* ``test_a_two_line_row_is_taller_than_a_one_line_row`` —— :meth:`sizeHint` 与
  :meth:`paint` 的边距必须一致。不一致的话行会互相压住或空出一截，看起来像字体
  度量算错了——那种毛病最容易被误诊成"系统 DPI 的问题"。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyleOptionViewItem,
)

from biosnn_memterm.ui.widgets import (
    SESSION_TIME_ROLE,
    SESSION_TITLE_ROLE,
    SessionItemDelegate,
)

pytestmark = pytest.mark.gui

_TITLE = "关于记忆注入的讨论"
_STAMP = "10-10 14:32"
_FILL = QColor("#123456")


@pytest.fixture(scope="module")
def qapp() -> Any:
    app = QApplication.instance() or QApplication([])
    yield app
    app.setStyleSheet("")


def _row(*, title: str | None, stamp: str | None, text: str = "") -> tuple[Any, Any, Any]:
    """造一个"只有一行、没有视图"的最小环境：列表、它的模型索引、和一个 ``option``。

    不建真窗口也不用 ``show()``——离屏下能画，而我们要的只是"画得出来"。
    """
    view = QListWidget()
    item = QListWidgetItem(text)
    if title is not None:
        item.setData(SESSION_TITLE_ROLE, title)
    if stamp is not None:
        item.setData(SESSION_TIME_ROLE, stamp)
    view.addItem(item)

    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 200, 40)
    option.widget = view
    return view, option, view.model().index(0, 0)


def _ink(option: QStyleOptionViewItem, index: Any, state: QStyle.StateFlag | None = None) -> int:
    """画一遍，数出**被改过的像素个数**。

    用一个不可能是主题里的颜色打底，于是"画上去了"就有了一个可断言的证据，
    而不必去比对像素内容（那要人眼看，且换个 Qt 版本就变）。
    """
    if state is not None:
        option.state |= state
    pixmap = QPixmap(200, 40)
    pixmap.fill(_FILL)
    painter = QPainter(pixmap)
    SessionItemDelegate().paint(painter, option, index)
    painter.end()
    image = pixmap.toImage()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y) != _FILL
    )


# ── 角色 ────────────────────────────────────────────────────────────────────


def test_the_two_roles_do_not_collide_with_qt_or_each_other() -> None:
    """角色号必须错开，且都在 Qt 留给应用的那一段里。

    撞车不会报错，只会让时间那行画出标题——**一类只在屏幕上才看得见的错**。
    """
    # 钉住**确切偏移**而不是"比 UserRole 大"：只验大小的话，两个角色都写成 +1
    # 也能通过一半，而那种撞车正是这条要拦的。
    base = int(Qt.ItemDataRole.UserRole)
    assert int(SESSION_TITLE_ROLE) == base + 1
    assert int(SESSION_TIME_ROLE) == base + 2


def test_the_display_role_is_left_alone() -> None:
    """``DisplayRole`` 是**别人的观测口**（``tools/_smoke_window.py`` 与若干测试靠
    ``item.text()`` 读 ``Session.describe()``）。delegate 只能加角色，不能改它。"""
    view, _, index = _row(title=_TITLE, stamp=_STAMP, text="描述行")
    assert index.data(Qt.ItemDataRole.DisplayRole) == "描述行"
    assert index.data(SESSION_TITLE_ROLE) == _TITLE
    assert index.data(SESSION_TIME_ROLE) == _STAMP
    del view


# ── 画 ──────────────────────────────────────────────────────────────────────


def test_it_draws_something(qapp) -> None:
    """最弱但最必要的一条：**别一声不响地什么都不画**。自绘控件写错了很容易变成
    一片空白，而空白与"列表里没有会话"看起来一模一样。"""
    view, option, index = _row(title=_TITLE, stamp=_STAMP)
    assert _ink(option, index) > 0
    del view


def test_it_draws_when_selected_too(qapp) -> None:
    """选中态走的是另一支取色（``HighlightedText``）。**那支没走过就不知道通不通**，
    而选中恰恰是用户最常看见的状态。"""
    view, option, index = _row(title=_TITLE, stamp=_STAMP)
    assert _ink(option, index, QStyle.StateFlag.State_Selected) > 0
    del view


def test_it_falls_back_to_the_default_text(qapp) -> None:
    """没设自定义角色的项退回 ``DisplayRole`` 单行，**不画成空白**。"""
    view, option, index = _row(title=None, stamp=None, text="只有默认文字的一行")
    assert _ink(option, index) > 0
    del view


def test_a_row_with_no_text_at_all_does_not_crash(qapp) -> None:
    """全空的列表项也不许炸。这条防的是"某个上游忘了填字段"演变成一次崩溃。"""
    view, option, index = _row(title="", stamp="")
    pixmap = QPixmap(200, 40)
    pixmap.fill(_FILL)
    painter = QPainter(pixmap)  # 持有到 paint 之后，别让临时对象提前析构
    SessionItemDelegate().paint(painter, option, index)
    painter.end()
    del view


# ── 高度 ────────────────────────────────────────────────────────────────────


def test_a_two_line_row_is_taller_than_a_one_line_row(qapp) -> None:
    """有时间的行明显更高。

    只断言"不同"是不够的——反了也是"不同"。这里钉住方向：两行 > 一行，
    因为 :meth:`SessionItemDelegate.paint` 把时间画在标题**下面**。
    """
    view, option, two_line = _row(title=_TITLE, stamp=_STAMP)
    _, _, one_line = _row(title=_TITLE, stamp=None, text="兜底")
    delegate = SessionItemDelegate()
    assert (
        delegate.sizeHint(option, two_line).height() > delegate.sizeHint(option, one_line).height()
    )
    del view


def test_the_height_leaves_room_for_both_lines(qapp) -> None:
    """:meth:`sizeHint` 给出的高度**装得下**它要画的两行。

    装不下的话第二行会被裁掉半截，看起来像行距没调好——这正是"paint 与 sizeHint
    各写一套边距"的典型症状，所以这条把它们绑在一起。
    """
    view, option, index = _row(title=_TITLE, stamp=_STAMP)
    height = SessionItemDelegate().sizeHint(option, index).height()
    line_height = option.fontMetrics.height()
    assert height >= 2 * line_height
    del view
