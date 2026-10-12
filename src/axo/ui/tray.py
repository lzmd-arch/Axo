"""系统托盘：**关窗不等于退出**，以及"什么时候该弹一下"的入口。

托盘不是一个图标，它是一条**行为改变**：窗口可以被收起来而进程继续活着。这条改变
是定时提醒能成立的前提——"到点了"要有东西在听，而关窗即退的进程在那一刻已经不在了
（见 :mod:`axo.agent.reminders` 的模块文档）。

## 这里的职责边界

这个类**不认识主窗口**，也不知道提醒是什么。它只做三件事：

1. 画一个图标放上去；
2. 菜单里那三项 → 三个信号（:attr:`show_requested` / :attr:`consolidate_requested` /
   :attr:`quit_requested`），由谁接、接到哪儿去是 ``MainWindow`` 的事；
3. 把一条消息（气泡）显示出来。

反过来（直接调 ``MainWindow.show()``）会让这个文件 import 主窗口，而主窗口已经
1800 行——单独的落点是为了让它保持可单测。

## 图标为什么是画出来的

这个仓库**零图片资产**（没有 ``.ico``/``.png``/``.svg``/``.qrc``），而 Windows 上
托盘图标**绝不能是 null**：那不是"没图标"，是一个空白方块，看起来像程序坏了。
所以用 ``QPainter`` 现画一个，底色取主题强调色——顺带跟着用户改的主题走。

## 托盘不可用时怎么办

:attr:`active` 为假时**这个对象什么都不做**，而且 ``MainWindow`` 也**不拦关窗**。
这不是防御性编程：一台托盘不可用的机器（某些精简版 Windows、没有面板的 Linux 会话）
上拦下关窗意味着窗口关掉了、图标也没有，应用**变得够不着**——比"关窗即退"坏得多。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

__all__ = ["ICON_SIZE", "TrayController"]

ICON_SIZE = 64
"""图标位图画多大。

64 而不是 16：Windows 会按 DPI 缩放托盘图标（150% 下要 24、200% 下要 32），交给
Qt 从大图缩下去比从小图放大清楚得多。这是个只算一次的 4 KB 位图，不值得省。
"""

_DOT_RADIUS = 0.18
"""中心那个点占图标边长的比例。太小在 16px 下会被缩没，太大就不像一个"点"。"""


def build_icon(accent: str, size: int = ICON_SIZE) -> QIcon:
    """按强调色画一个图标。**纯绘制，不读文件。**

    形状：圆角方块 + 中心一个点。选它是因为**在 16px 下还认得出来**——托盘图标实际
    就那么点大，笔画细的、带文字的在那个尺寸上都糊成一团。

    颜色解析走 :func:`~axo.ui.color.parse_hex` 那一套的宽容度：这个函数
    拿到的 ``accent`` 来自主题，而主题可能是手改过的注册表（见
    ``ui/settings_store.py`` 的三条纪律）。**画不出来就退回一个中性灰，绝不抛**——
    一个托盘图标不值得让程序打不开。

    ``size`` 是给打包时**手工拼多尺寸 ``.ico``** 用的（见 ``tools/build_exe.py``）：
    Windows 从任务栏到资源管理器会按 DPI 挑不同档位，每一档都得**现画**一张——把
    64px 那张缩到 16px 会把那个点缩没。托盘自己只用默认那一档。
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    colour = QColor(accent)
    if not colour.isValid():
        # 主题坏了（手改的注册表、降级回旧版本）。**这里不抛**：设置窗口会把坏值
        # 修好，而托盘是启动路径上的东西。
        colour = QColor("#6b7280")

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        # 圆角半径取边长的四分之一：再方一点像一个应用图标，再圆一点在小尺寸下
        # 会看不出圆角。
        painter.drawRoundedRect(0, 0, size, size, size / 4, size / 4)
        painter.setBrush(QColor("#ffffff"))
        radius = size * _DOT_RADIUS
        centre = size / 2 - radius
        painter.drawEllipse(int(centre), int(centre), int(radius * 2), int(radius * 2))
    finally:
        # ``end()`` 必须走到，**连抛的时候也要**：没结束的 QPainter 在析构时会往
        # stderr 打一句警告，而且那之后这个 pixmap 上再画什么都不可预期。
        painter.end()
    return QIcon(pixmap)


class TrayController(QObject):
    """托盘图标 + 菜单。**它的全部输出是两个信号与一条气泡。**

    Args:
        available: 托盘可不可用。``None``（默认）= 问 Qt
            （``QSystemTrayIcon.isSystemTrayAvailable()``）。可注入是为了让"不可用
            那台机器"这条路径能在测试里走到——那正是最容易写错、又最不容易被真人
            碰到的一支。
        accent: 图标底色（主题强调色）。
        parent: Qt 父对象。**传主窗口**，于是它跟着窗口一起销毁。
    """

    show_requested = Signal()
    """菜单里的「显示主窗口」/ 双击图标。"""

    consolidate_requested = Signal()
    """菜单里的「立即巩固」。**与 :attr:`show_requested` 同形**：托盘不认识主窗口，
    更不认识巩固是什么，它只把"用户按了这一项"说出去。

    这一项存在的理由是**定时那条路看不见**：巩固默认每天在后台跑一次，而"它到底跑
    没跑、跑出来什么"没有一处用户能主动触发的地方。一个只在后台悄悄发生、又改着用户
    的库的功能，必须有一个"我现在就要看一遍"的入口。
    """

    quit_requested = Signal()
    """菜单里的「退出」。**这是唯一会真的结束进程的那一项。**"""

    def __init__(
        self,
        *,
        available: bool | None = None,
        accent: str = "#6b7280",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._available = (
            QSystemTrayIcon.isSystemTrayAvailable() if available is None else available
        )
        self._tray: QSystemTrayIcon | None = None
        self._menu: QMenu | None = None
        #: 提示只给一次。见 :meth:`show_message_once`。
        self._hinted = False

        if not self._available:
            return

        self._menu = QMenu(parent)
        self._show_action = QAction("显示主窗口", self._menu)
        self._show_action.triggered.connect(self.show_requested)
        self._menu.addAction(self._show_action)
        # 排在「显示」之后、「退出」之前：它是一次真的计算（几秒到几分钟），而分隔线
        # 那一边是"结束进程"。夹在中间，误点的代价最小——最坏是白跑一轮。
        self._consolidate_action = QAction("立即巩固", self._menu)
        self._consolidate_action.triggered.connect(self.consolidate_requested)
        self._menu.addAction(self._consolidate_action)
        self._menu.addSeparator()
        self._quit_action = QAction("退出", self._menu)
        self._quit_action.triggered.connect(self.quit_requested)
        self._menu.addAction(self._quit_action)

        self._tray = QSystemTrayIcon(build_icon(accent), parent)
        self._tray.setToolTip("Axo")
        self._tray.setContextMenu(self._menu)
        # 双击与单击都叫回窗口：Windows 上双击是习惯，而 Linux 的某些面板只发单击。
        # 接两个不会有副作用——Qt 不会为一次双击同时发这两个信号。
        self._tray.activated.connect(self._on_activated)

    # ── 状态 ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """托盘**真的在**（可用且已经建好）。

        关窗那一支的判据就是它：为假时行为与托盘出现之前完全一样（关窗即退）。
        """
        return self._tray is not None

    def set_accent(self, accent: str) -> None:
        """主题换了，重画图标。**不可用时是空操作**，调用方不必先问。"""
        if self._tray is not None:
            self._tray.setIcon(build_icon(accent))

    def show(self) -> None:
        """放上去。**幂等**——``QSystemTrayIcon.show()`` 本来就是。"""
        if self._tray is not None:
            self._tray.show()

    def hide(self) -> None:
        """摘下来（退出时用）。

        **退出必须摘**：Windows 上留在通知区域的死图标要等鼠标划过才消失，而用户
        看到的是"我已经退出了，它还在那儿"。
        """
        if self._tray is not None:
            self._tray.hide()

    def show_message(self, title: str, body: str) -> None:
        """弹一条气泡。**托盘不在时什么都不做。**

        不做"退化成 QMessageBox"：一个模态框会挡住用户正在做的事，而这条消息的内容
        是"提醒你一下"，不是"需要你现在处理"。托盘不在的机器上本来也走不到这条路径
        （没有常驻，就没有定时触发）。
        """
        if self._tray is not None:
            self._tray.showMessage(title, body, QSystemTrayIcon.MessageIcon.Information)

    def show_message_once(self, title: str, body: str) -> bool:
        """同上，但**一个进程里只说得出口一次**。返回这次说没说。

        用例是"关窗收进托盘了"那句提示：第一次关窗不说的话，用户会以为程序退出了
        （图标在通知区域里，而那个区域默认是折叠的）；每次都说的话，它会在每天几十
        次关窗里变成噪音。**说一次就够，因为该知道的人一次就知道了。**

        返回值是**真的说了没有**，所以托盘不在时是假——把它做成"记下了要说"就
        成了谎话，而谎话的代价是调用方以为用户已经被告知过。
        """
        if self._tray is None or self._hinted:
            return False
        self._hinted = True
        self.show_message(title, body)
        return True

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """双击 / 单击图标 → 把窗口叫回来。右键菜单走的是另一条路（``contextMenu``），
        不会到这里。"""
        if reason in (
            QSystemTrayIcon.ActivationReason.DoubleClick,
            QSystemTrayIcon.ActivationReason.Trigger,
        ):
            self.show_requested.emit()
