"""两个自绘控件。**放一起是因为它们解决的是同一类问题：Qt 默认控件做不到的排版。**

* :class:`ElidedLabel` —— 顶栏状态文字被窗口右边裁掉。Qt 的 ``QLabel`` 没有省略号，
  而顶栏那一条是"挤不下就顶出去"的自由布局：加了「设置」按钮之后它有五个按钮加一个
  模型下拉，状态文字是唯一一个可以被压的。
* :class:`SessionItemDelegate` —— 会话列表要两行（标题 + 时间），而
  ``QListWidgetItem`` 默认只有一行。

**两个都只在"怎么画"上动手，不改"是什么"。** :class:`ElidedLabel` 的 ``text()``
仍返回全文（``tools/_smoke_window.py`` 靠这个读状态），:class:`SessionItemDelegate`
也不碰 ``Session.describe()`` 那行文字——它只是把已有的两个自定义角色画成两行。
"""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOption,
    QStyleOptionViewItem,
    QWidget,
)

__all__ = [
    "SESSION_TIME_ROLE",
    "SESSION_TITLE_ROLE",
    "ElidedLabel",
    "SessionItemDelegate",
]

#: 会话列表项上的两个自定义角色。**不改 ``DisplayRole``**：``tools/_smoke_window.py``
#: 与别的测试靠 ``item.text()`` 读 ``Session.describe()``，换了它等于顺手改了别人的
#: 观测口。两个新角色是**加法**——delegate 只改怎么画。
SESSION_TITLE_ROLE = Qt.ItemDataRole.UserRole + 1
SESSION_TIME_ROLE = Qt.ItemDataRole.UserRole + 2

_ITEM_PAD_H = 8
_ITEM_PAD_V = 5
_LINE_GAP = 2


class ElidedLabel(QLabel):
    """会自己加省略号的 ``QLabel``，``text()`` 仍返回全文。

    为什么不是"把文字截断了再 ``setText``"：那样 ``text()`` 返回的是截断后的东西，
    而 tooltip 与自动化检查要的是全文。**内容与显示分开**——``_full`` 是内容，
    :meth:`paintEvent` 每次按当前宽度重新算显示。

    宽度策略是 ``Preferred`` 加一个很小的 ``minimumSizeHint``：**要多少给多少，不够
    了才压窄**。压窄之后是省略号而不是裁切——差别在于后者会让用户完全看不到后半句，
    而前者至少明说了"这里还有字"。

    **这里原本写的是 ``Ignored``，而它让这个标签从来没显示过。** ``Ignored`` 里那个
    ``IgnoreFlag`` 让布局把 ``sizeHint`` 与 ``minimumSizeHint`` **一起**无视，于是标签
    的宽度按 0 参与分配，``paintEvent`` 拿 0 去算省略号得到空串——顶栏那段状态文字
    （"● Ollama 就绪 · 4 个对话模型"）宽度实测为 0、画出来是 ``''``，从未可见。
    ``Preferred`` 才是那份注释真正描述的行为：``sizeHint`` 是它想要的宽度，布局可以
    一路压到 ``minimumSizeHint``；而窗口的最小宽度由后者决定，所以**顶栏不会被撑宽**
    （实测：1900px 窗宽给 252px 全显，1100px 时省略，窗口最小宽度只多 24 像素）。
    """

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
    ) -> None:
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt 命名)
        """存全文、设 tooltip，并让基类也拿到全文。

        基类那一份**不会被画出来**（:meth:`paintEvent` 不调 ``super()``），留着它是
        为了让 ``text()``、``sizeHint()`` 与无障碍接口都拿到完整内容。
        """
        self._full = text or ""
        self.setToolTip(self._full)
        super().setText(self._full)

    def fullText(self) -> str:  # noqa: N802 (与 text() 并列的读法)
        """全文。与 :meth:`text` 同值，但名字说清了它是"没被省略的那一份"。"""
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt 命名)
        """压到很小。

        **这一条是它能在顶栏里活下来的原因。** 不覆盖的话 ``QLabel`` 会坚持要放下
        全文的宽度，于是窗口被撑宽、或者状态文字被挤出可视区——正是要修的那个毛病。
        """
        hint = super().minimumSizeHint()
        return QSize(min(24, hint.width()), hint.height())

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt 命名)
        """自己画省略后的文本。

        **不调 ``super().paintEvent()``**：调了会把全文再画一遍，压在省略版上面。
        代价是要自己取颜色——用 ``QStyleOption.initFrom`` 拿调色板而不是写死，是为了
        让样式表里那些 ``color:`` 规则仍然生效（Qt 的样式表引擎会把 ``color`` 落到
        控件的调色板上）。
        """
        del event  # 整块重画，用不上脏矩形
        option = QStyleOption()
        option.initFrom(self)

        painter = QPainter(self)
        painter.setPen(option.palette.color(QPalette.ColorRole.WindowText))
        painter.setFont(self.font())
        text = self.fontMetrics().elidedText(self._full, self._mode, self.width())
        painter.drawText(self.rect(), int(self.alignment()), text)
        painter.end()


class SessionItemDelegate(QStyledItemDelegate):
    """会话列表的两行式：上行标题（加粗、省略），下行时间（次要色）。

    **不用 ``setItemWidget``。** 那个做法看着更直接，但它会把每一行变成真实子控件，
    于是选择点击被吃掉、选中高亮要自己画、``currentItemChanged`` 的语义也变得可疑。
    自绘 delegate 只改"怎么画"，列表的行为一个字都不用动。

    **底由基类画，字由自己画。** :meth:`paint` 先把 ``opt.text`` 清空再交给
    ``QStyle`` 画 ``CE_ItemViewItem``——于是样式表里 ``QListWidget::item`` 的
    ``:hover`` / ``:selected`` 背景仍然生效，而文字由下面自己排两行。完全自己画底的
    话，选中高亮就得在这里再实现一遍，两处颜色必然漂。

    时间那行用 ``QPalette.ColorRole.PlaceholderText``，它在
    :func:`~biosnn_memterm.ui.theme_qt.build_palette` 里被映射到 ``fg_muted``。
    绕这一圈是为了让 delegate **不必知道主题的存在**：颜色从调色板来，换主题时自己
    就跟着变了。
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""  # 底交给基类画，字留给自己
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        # 取色看 ``opt.state`` 而不是传进来的 ``option.state``：底是用 ``opt`` 画的，
        # 字也必须按同一份状态取色，否则某个 Qt 版本若在 ``initStyleOption`` 里动了
        # 状态，就会出现"底按选中画、字按未选中画"——选中行变成看不清的深色字。
        title = index.data(SESSION_TITLE_ROLE)
        stamp = index.data(SESSION_TIME_ROLE)
        if title is None:
            # 没写自定义角色的列表项：退回单行，用默认文字。**有这一条，这个 delegate
            # 才能安全地用在别处**，也才不至于在某天有人往列表里塞了一条没设角色的
            # 项时画出一片空白。
            title = index.data(Qt.ItemDataRole.DisplayRole) or ""
            stamp = ""

        rect = option.rect.adjusted(_ITEM_PAD_H, _ITEM_PAD_V, -_ITEM_PAD_H, -_ITEM_PAD_V)
        metrics = opt.fontMetrics
        line_height = metrics.height()
        align = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        primary = opt.palette.color(
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text
        )

        painter.save()
        title_font = painter.font()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(primary)
        painter.drawText(
            QRect(rect.left(), rect.top(), rect.width(), line_height),
            align,
            metrics.elidedText(str(title), Qt.TextElideMode.ElideRight, rect.width()),
        )

        if stamp:
            stamp_font = painter.font()
            stamp_font.setBold(False)
            painter.setFont(stamp_font)
            painter.setPen(opt.palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                QRect(rect.left(), rect.top() + line_height + _LINE_GAP, rect.width(), line_height),
                align,
                metrics.elidedText(str(stamp), Qt.TextElideMode.ElideRight, rect.width()),
            )
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        """两行的高度。**必须与 :meth:`paint` 用的是同一套边距**，不一致的话行会互相
        压住或者空出一截，而那种毛病看起来像字体度量算错了。"""
        base = super().sizeHint(option, index)
        lines = 2 if index.data(SESSION_TIME_ROLE) else 1
        height = lines * option.fontMetrics.height() + 2 * _ITEM_PAD_V
        if lines == 2:
            height += _LINE_GAP
        return QSize(base.width() + 2 * _ITEM_PAD_H, height)
