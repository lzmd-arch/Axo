"""登录时那一次：**看一眼有没有到点的提醒，弹一条就退**。

用户拍板的那条"程序没开着的那一晚"落到这里：开机自启 + 一个无头入口。它跑在登录时，
**没有控制台、没有人看着**，跑完就退。

## 为什么不复用 ``ReminderScheduler``

那个类的全部职责是"**什么时候再看一眼**"（单发 ``QTimer`` + 每次响过重排），而这里
**只看一眼**。把它搬进来会连带搬进 Qt、一个事件循环，以及"下一次什么时候醒"这个在
一次性进程里没有意义的问题——而真正要复用的三件东西（``due_now`` / ``Alert`` /
``mark_occurrences``）全在纯层，本来就与 Qt 无关。

## 顺序：先标 fired，再弹

与 GUI 那一侧（``ui/reminder_scheduler.py`` 的 ``_fire``）**逐字同序**，理由也一样：
反过来的话，弹出之后、落盘之前进程退出，下次登录会把同一条再弹一遍。两个方向的代价
不对称——重弹每次都会发生，而"标了却没弹成"很少发生。

**两个入口的顺序必须一致**，否则同一条提醒在两个入口下行为不一样，而那种不一致只会
以"有时候会弹两次"的形式出现（见下面那条已知取舍）。

## 已知取舍：无头与主程序同时到点会各弹一次

仓里已明说**没有单实例守卫、本版接受多开**，这条不改。它不是"每次都双弹"：两个进程
写的是**同一本** fired 账，只有各自在 ``due_now`` 与 ``mark_occurrences`` 之间那个很窄
的窗口里才会撞上，错开几十毫秒就只剩一次。窗口本身也很窄（无头只在登录时跑一次），
而且**可诊断**：``mark_occurrences`` 的 ``by`` 字段会写下 ``"gui"`` / ``"remind"``，
打开那个文件一看两行、两个 ``by``，就知道是这条已知的取舍又发生了，而不是去怀疑
"重复写入坏了"。

## 退出码

* ``0`` 正常（**包括"没有到点的"**——那是绝大多数登录的样子）；
* ``1`` 到点了但通知弹不出来（**已标 fired**，如实说，这次不重试）；
* ``2`` 提醒那几本账读不了（**没标、也没弹**，下次登录还会来）。

1 与 2 的差别不是程度，是**状态**：1 的那条提醒已经算响过了，2 的还没有。登录时跑的
东西没人看着，所以这个数就是它唯一的说法——``tools/build_exe.py`` 的 ``_check_remind``
判的是它。
"""

from __future__ import annotations

import sys
from typing import Protocol

from .agent.reminders import (
    Alert,
    CancelledStore,
    FiredStore,
    ReminderStore,
    due_now,
)

__all__ = ["EXIT_NOTIFY_FAILED", "EXIT_OK", "EXIT_UNREADABLE", "Notifier", "run"]

EXIT_OK = 0
EXIT_NOTIFY_FAILED = 1
EXIT_UNREADABLE = 2


class Notifier(Protocol):
    """弹一条通知。**返回"真的弹出来了没有"**。

    是个可注入的 Protocol（而不是直接调 ``ui/notify.py``）有两个理由：这一层因此不
    认识 Qt（``test_layering`` 钉着），而这整条逻辑在核心腿里用一个假件就能逐条断言
    ——包括"弹不出来"那一支，而那正是真人最不容易碰到、也最该被写死的一支。
    """

    def __call__(self, title: str, body: str) -> bool: ...


def run(
    *,
    reminders: ReminderStore,
    fired: FiredStore,
    cancelled: CancelledStore,
    now: float,
    notify: Notifier,
) -> int:
    """收掉一批到点、弹一条通知，返回退出码（见模块文档）。

    Args:
        reminders: 提醒簿（读）。
        fired: 响过的那些（读 + 写）。写它发生在弹之前。
        cancelled: 取消过的那些（读）。**与 GUI 那一侧一样要递**：少递一次，用户撤掉
            的那条会在下次登录时响。
        now: 现在几点。**由调用方给**，与纯层其余部分同一个约定：这一层要能在测试里
            把"现在"摆到任何一刻。
        notify: 弹通知的那一下（见 :class:`Notifier`）。**只在真有到点的时候才调**
            ——登录时绝大多数情况下没有，而那种时候不该去建一个 Qt 应用。
    """
    try:
        items = due_now(
            reminders.all(),
            fired.fired_keys(),
            now,
            cancelled=cancelled.cancelled_ids(),
        )
    except Exception as exc:  # noqa: BLE001 — 见下面那段：这是入口边界
        # **读不了就什么都不做。** 没标 fired、也没弹——于是下次登录还会来，而那正是
        # 对的：这一条提醒还没有被任何人口头告知过。
        #
        # **兜的是 `Exception` 而不是 `OSError`**：读盘这一步能出的错不止一种，而最
        # 容易漏的那个是**编码坏掉**——它是 `UnicodeDecodeError`，一个 `ValueError`，
        # 不是 `OSError`。漏出去的症状不是崩溃那么简单：一个没有控制台的进程打一份
        # 回溯，退出码是解释器给的 **1**，与"通知弹不出来"撞在一起，于是
        # ``_check_remind``（以及任何看退出码的人）会把一件真事故读成一次正常的失败。
        print(f"提醒那几本账读不了，这次什么都没做：{exc}", file=sys.stderr)
        return EXIT_UNREADABLE

    if not items:
        return EXIT_OK

    # ``late=True``：与 GUI 那一侧的 ``start()`` 同口径——它也是"启动时收掉一批"，
    # 标题因此是"错过了一条提醒"。登录时那些到点的绝大多数本来就是**当时没人在**
    # 才错过的，说"提醒"反倒不诚实（同 ``reminder_scheduler._fire`` 那段注释）。
    alert = Alert(items=items, late=True)
    try:
        fired.mark_occurrences(items, now, by="remind")
    except Exception as exc:  # noqa: BLE001 — 同上，入口边界
        print(f"提醒响过的账写不进去，这次没有弹：{exc}", file=sys.stderr)
        return EXIT_UNREADABLE

    if notify(alert.heading(), alert.body()):
        return EXIT_OK
    # **已经标过了，所以这次不重试**——重试的代价是下次登录再弹一遍同一条。这一行
    # 是留给源码检出跑的人的（打包产物没有控制台）。
    print("到点了，但通知没能弹出来（提醒已记成响过，这次不再重试）。", file=sys.stderr)
    return EXIT_NOTIFY_FAILED
