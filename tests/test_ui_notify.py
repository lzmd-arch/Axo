"""登录时那一条气泡：**弹得出来、弹完自己退**。

``--remind`` 那条路上唯一与 Qt 打交道的地方。它跑在登录时（没有控制台），所以它要是
挂住了，症状是任务管理器里多一个看不见的进程；它要是吞掉了通知，症状是"那天早上的
提醒没响"——两种都没有任何界面会报出来。

这里量两件事：**没有托盘时不弹（且立刻返回）**、**有托盘时弹完自己退**。
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from axo.ui.notify import BALLOON_MS, show_balloon

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_no_tray_means_no_notification(qapp: Any) -> None:
    """没有托盘 → 立刻返回假，**不转事件循环**。

    返回值是"真的弹了没有"（同 ``TrayController.show_message_once`` 那条纪律）：编一句
    "已经通知过你了"，代价是 ``headless.run`` 把那条提醒标成响过、而用户从没看见过——
    一条真的提醒就这样消失了。
    """
    started = time.monotonic()

    assert show_balloon("标题", "正文", available=False) is False
    assert time.monotonic() - started < 1.0, "没有托盘还转了一圈事件循环"


def test_a_real_balloon_comes_back_on_its_own(qapp: Any) -> None:
    """**它必须自己退回来。** 等的是 ``wait_ms``，不是"等到有人关掉它"。

    这条路径上没有窗口、没有「退出」可点——一个不肯退的登录任务会在任务管理器里留一个
    看不见的进程（而用户完全不知道它在等什么）。

    ``available=True`` 是**注入**的，不是问 Qt：离屏环境下有没有托盘是实现细节，而这条
    测试要钉的是"走完那一趟会不会回来"。
    """
    started = time.monotonic()

    assert show_balloon("标题", "正文", wait_ms=1, available=True) is True
    assert time.monotonic() - started < 5.0


def test_the_default_wait_is_the_documented_one() -> None:
    """默认等待就是 :data:`BALLOON_MS`——它是"气泡在屏幕上待多久"，也是这个进程从弹到
    退的总时长。写死一个大数会让登录卡住，写太小会让气泡被 shell 吞掉（见模块文档）。
    """
    assert 3000 <= BALLOON_MS <= 15_000
