"""无头入口那一条通知：**一个托盘气泡，弹完就退**。

只有 :func:`show_balloon` 一个函数，而它只被 :mod:`axo.headless` 那一次调用（登录时，
通过 ``--remind``）。主窗口自己有托盘（``ui/tray.py``），不需要这里。

## 为什么是托盘气泡，不是 PowerShell toast

Windows 上还有一条路：``powershell -Command New-BurntToastNotification ...``。不用它的
理由是**中文要过控制台代码页**——``tools/build_exe.py`` 的 ``_make_shortcut`` 已经为
同一个理由绕过一次（那次改走 ``-EncodedCommand`` + UTF-16LE + base64）。托盘气泡没有
这个问题：它在进程内，字符串是 Python 的 ``str``。

## 两个前提，都是实测换来的

1. **必须有一个真 ``show()`` 过的托盘图标。** 没显示过的 ``QSystemTrayIcon`` 发不出
   消息——``showMessage`` 不报错，只是什么都不发生。
2. **事件循环要活到消息被 shell 收走。** 进程立刻退出的话，气泡在 Windows 上会随着
   图标一起消失，看起来像"什么都没发生"。所以这里自己转一个有限时长的事件循环
   （见 :data:`BALLOON_MS`），而不是发完就返回。

## 弹不出来要说"没弹出来"

返回值是**真的弹了没有**，与 :meth:`~axo.ui.tray.TrayController.show_message_once` 同一
条纪律：编一句"已经通知过你了"，代价是调用方以为用户被告知过，而这次提醒就永远消失了
（``headless.run`` 会照样把它标成响过，也正是"如实说"的意思）。
"""

from __future__ import annotations

__all__ = ["BALLOON_MS", "show_balloon"]

BALLOON_MS = 6000
"""气泡在屏幕上待多久（毫秒）。**也是这个进程从弹到退的总时长。**

Windows 自己那个通知的默认时长在 5 秒上下；这里取 6 秒是留一点余量，让"被 shell 收走"
这件事在慢机器上也能发生。它同时是登录体验的代价：那一次会占住进程 6 秒——一个登录
时跑的后台任务，这个代价可以接受，而**短于它就有气泡被吞掉的风险**（见模块文档第 2 条）。
"""


def show_balloon(
    title: str, body: str, *, wait_ms: int = BALLOON_MS, available: bool | None = None
) -> bool:
    """弹一条托盘气泡并等它被 shell 收走。返回**弹出来了没有**。

    Args:
        title: 气泡标题（``Alert.heading()``）。
        body: 气泡正文（``Alert.body()``）。
        wait_ms: 等多久再退。**可注入是为了可测**：真等 6 秒的测试没人愿意跑，而
            "它会不会永远挂在这儿"恰恰是这条路径唯一值得测的东西。
        available: 托盘在不在。``None`` = **问 Qt**，与
            :class:`~axo.ui.tray.TrayController` 那个同名参数同一套用法与同一个理由：
            "没有托盘"那一支真人最不容易碰到，而它必须有一条测试走到。
    """
    # 推迟到函数里 import：主窗口那条路不经过这里，而只要有人 import 这个模块
    # 就带上 Qt 是没必要的——真正需要它的是这一次调用。
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    from .tray import build_icon

    app = QApplication.instance() or QApplication([])
    if available is None:
        available = QSystemTrayIcon.isSystemTrayAvailable()
    if not available:
        return False

    tray = QSystemTrayIcon(build_icon(), app)
    tray.show()
    tray.showMessage(title, body, QSystemTrayIcon.MessageIcon.Information, wait_ms)
    # **有限时长的事件循环**，不是 ``app.exec()`` 到天荒地老：这条路径上没有窗口、
    # 没有「退出」可点，而一个不肯退的登录任务会在任务管理器里留一个看不见的进程。
    QTimer.singleShot(wait_ms, app.quit)
    try:
        app.exec()
    finally:
        # 摘掉图标：留着的话它会一直待在通知区域里，直到鼠标划过——用户看到的是
        # "提醒弹过了，可那是什么东西？"而程序其实早就退了（同 ``_shutdown`` 里
        # 那句 ``self._tray.hide()``）。
        tray.hide()
    return True
