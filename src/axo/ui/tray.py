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

from pathlib import Path

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


_ASSET = Path(__file__).with_name("axo-icon.png")
"""**品牌立绘**（阿螈）：窗口、托盘、exe 三处的图标都读这一份（exe 那份是打包时
从它生成的 ``.ico``）。放在源码树里与 ``theme.qss`` 同一姿态：``build_exe.py``
 把它镜像进产物（``DATAS``），运行时按 ``Path(__file__).with_name(...)`` 读——
 打包与否路径都对。可替换：测试把它指到别处来走兜底那条路。"""


def build_icon(size: int = ICON_SIZE) -> QIcon:
    """产品的图标：**品牌立绘**（阿螈）。**读不到资产就画兜底，绝不抛**。

    2026-10-13 换的：以前是按主题强调色画的记号（圆角方块 + 一点），理由是
    「16px 下还认得出来」；用户给了立绘，**图标不再跟主色走**——它是产品的脸，
    不是一块皮肤。16px 那档从**原图**平滑缩（立绘笔画粗，缩了还认得出）。

    ``size`` 的用途不变：打包时手工拼多尺寸 ``.ico`` 每档各取一张（见
    ``tools/build_exe.py``），托盘自己只用默认那一档。

    兜底（资产缺失/坏图）仍画那个中性灰记号——**图标不该让程序打不开**。
    """
    pixmap = QPixmap(str(_ASSET))
    if not pixmap.isNull():
        scaled = pixmap.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(scaled)
    return QIcon(_fallback_pixmap(size))


def _fallback_pixmap(size: int) -> QPixmap:
    """资产读不到时画的记号：圆角方块 + 中心一点（中性灰）。**画不出绝不抛。**"""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    colour = QColor("#6b7280")
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawRoundedRect(0, 0, size, size, size / 4, size / 4)
        painter.setBrush(QColor("#ffffff"))
        radius = size * _DOT_RADIUS
        centre = size / 2 - radius
        painter.drawEllipse(int(centre), int(centre), int(radius * 2), int(radius * 2))
    finally:
        painter.end()
    return pixmap


class TrayController(QObject):
    """托盘图标 + 菜单。**它的全部输出是两个信号与一条气泡。**

    Args:
        available: 托盘可不可用。``None``（默认）= 问 Qt
            （``QSystemTrayIcon.isSystemTrayAvailable()``）。可注入是为了让"不可用
            那台机器"这条路径能在测试里走到——那正是最容易写错、又最不容易被真人
            碰到的一支。
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

        self._tray = QSystemTrayIcon(build_icon(), parent)
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
