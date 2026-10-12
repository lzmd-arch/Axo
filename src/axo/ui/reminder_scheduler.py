"""定时触发：**到点了就响**。

:mod:`axo.agent.reminders` 负责"谁到点了"（纯判断），这里负责"什么时候
看一眼"（Qt）。一个**单发** ``QTimer``，每次响过之后**按最新的事实重排**下一次。

## 为什么是单发 + 重排，不是周期表

周期表（比如"每 10 秒扫一遍"）看着简单，但它有两个说不清的地方：**什么时候第一次
扫**（周期表在 10 秒之后才第一次响，而用户可能刚打开程序就有一条已经过期的提醒），
以及**扫的粒度就是提醒的精度**（10 秒的表意味着一条提醒最多晚 10 秒）。单发重排没有
这两个问题：下一次醒来的时刻**就是**下一条该响的时刻。

## 两个边界值

* **上限 60 秒**：`QTimer` 收的是毫秒的 ``int``，而 ``(next_due - now)`` 可以是几天。
  更要紧的是**时钟跳变与休眠**——机器睡 8 小时醒来时，`QTimer` 那一刻不一定按预期
  醒来，而每 60 秒重新对一次表是最省事的兜底。周期最长 60 秒，对一次 cpu 唤醒的
  代价可以忽略。
* **下限 1 秒**：`next_due` **可能是过去**（错过的那些还没标 fired 时）。允许 0 的话，
  一次没收干净（比如某个环节抛了）就变成一个 0 毫秒的紧循环，把 cpu 打满而界面上
  什么都看不出来。1 秒的下限让最坏情况退化成"每秒重试一次"。

## 先写 sidecar，再弹（write-ahead）

:meth:`_fire` 里 ``mark_all_fired`` 排在 ``due.emit`` **之前**。反过来的话，弹出之后、
落盘之前进程退出（关窗收进托盘的用户随时会这么做），下次启动会**重弹**同一条。
两个方向的代价不对称：重弹每次都会发生，而"标了却没弹成"很少发生。

## 一堆到点时聚合成一句

关了一周再打开，可能有二十条过期。逐条弹会变成一场通知风暴，而风暴的结局是用户把
通知关掉——那时**所有**提醒都失效了。所以多于一条时走
:func:`~axo.agent.reminders.missed_summary` 收成一句。**也不静默丢**：
错过的事情仍然要说出来，只是不必一次说五遍。

## 谁负责显示

这个类**不认识窗口、也不认识托盘**：算好标题与正文，发 :attr:`due` / :attr:`missed`
两个信号就完了。显示是 ``MainWindow`` 的事（聊天里一条 + 托盘一条气泡）。这样它可
单测，而且"提醒长什么样"这个决定只在一个地方。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..agent.reminders import (
    FiredStore,
    Reminder,
    ReminderStore,
    due_now,
    missed_summary,
    next_due_at,
)

__all__ = ["MAX_DELAY_MS", "MIN_DELAY_MS", "ReminderScheduler"]

MIN_DELAY_MS = 1000
"""两次扫之间至少隔这么久。见模块文档的"下限 1 秒"。"""

MAX_DELAY_MS = 60_000
"""最久睡这么久就重新对一次表。见模块文档的"上限 60 秒"。"""


class ReminderScheduler(QObject):
    """到点了就发信号。**它自己不看时钟以外的任何东西。**

    Args:
        reminders: 提醒簿（读）。位置由调用方给——这里不该知道应用的数据目录在哪儿。
        fired: 响过的那些（读 + 写）。写它是这个类**唯一**的副作用。
        now: 取当前时刻。默认 :func:`time.time`，可注入是为了让测试把"现在"摆到
            任何一刻：等真时钟走完一条提醒的等待是没法写进测试的。
        parent: Qt 父对象。
    """

    due = Signal(str, str)
    """**一条**到点了：(标题, 正文)，**已经算好、可以直接显示**。"""

    missed = Signal(str, str)
    """**一堆**到点了（程序没开着的那段时间里错过的）：同上，正文已经聚合成一句。"""

    def __init__(
        self,
        *,
        reminders: ReminderStore,
        fired: FiredStore,
        now: Callable[[], float] = time.time,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._reminders = reminders
        self._fired = fired
        self._now = now
        self._running = False
        self._timer = QTimer(self)
        # **单发**：响过之后由 :meth:`reschedule` 按最新的事实决定下一次什么时候。
        # 周期表在这里是错的（见模块文档）。
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """接管：**先把错过的收掉**，再排下一次。

        顺序不能反。先排表的话，那个"下一次"会被错过的那些压在后面（它们的
        ``due_at`` 在过去，于是排出来的延迟是下限 1 秒），用户要等一秒才看到本来
        早就该看到的东西。
        """
        self._running = True
        self._fire(late=True)
        self.reschedule()

    def stop(self) -> None:
        """停表（退出时用）。**幂等。**"""
        self._running = False
        self._timer.stop()

    def reschedule(self) -> None:
        """按**此刻的事实**重排下一次。

        新登记的提醒靠它立刻生效：没有这个调用，``set_reminder`` 写下的那条要等到
        当前这一觉睡满（最长 60 秒）才会被看见。``MainWindow`` 在
        ``_on_tool_finished`` 里见到 ``set_reminder`` 就调它一次。

        **每次重读文件**，不缓存：提醒簿是**另一个进程之外**的东西（用户手改、上一
        次运行的残留），而重读的代价是一次几十行的文本解析。
        """
        if not self._running:
            return
        pending = next_due_at(self._reminders.all(), self._fired.fired_ids())
        if pending is None:
            # 没有将来的提醒 → **把表停掉**，不是排一个很远的时刻：一个空转的定时器
            # 会在进程存活期间一直醒（每次都要重读一遍文件），而"没有东西可等"是
            # 一个能说清楚的状态。
            self._timer.stop()
            return
        self._timer.start(self._delay_ms(pending))

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _delay_ms(self, pending: float) -> int:
        """睡多久。夹在 :data:`MIN_DELAY_MS` 与 :data:`MAX_DELAY_MS` 之间。"""
        return int(min(MAX_DELAY_MS, max(MIN_DELAY_MS, (pending - self._now()) * 1000)))

    @Slot()
    def _on_timeout(self) -> None:
        """表响了：把到点的收掉，然后重新排。

        **最后一定要重排**（:meth:`_fire` 里可能什么都没收，那种情况下也要接着等
        下一条）——漏掉它的话，响过一次之后这个对象就永远沉默了，而症状是"第一条
        提醒准时，之后的一条都不响"。
        """
        self._fire(late=False)
        self.reschedule()

    def _fire(self, *, late: bool) -> None:
        """把现在到点、且没响过的**全部**收掉。

        Args:
            late: 这一批是不是"程序没开着的时候错过的"（:meth:`start` 那一路）。
                **只影响一条的措辞**：一条三小时前就该响的提醒，标题写成"提醒"是
                不诚实的——它暗示"就是现在"。多条那一支的措辞本来就带时间含义
                （"到点了"），不必再分。
        """
        now = self._now()
        due = due_now(self._reminders.all(), self._fired.fired_ids(), now)
        if not due:
            return
        # **先写后弹**（write-ahead）。见模块文档：反过来的代价是每次退出都重弹。
        self._fired.mark_all_fired([one.reminder_id for one in due], now)
        self._report(due, late=late)

    def _report(self, due: Sequence[Reminder], *, late: bool) -> None:
        if len(due) == 1:
            title = "错过了一条提醒" if late else "提醒"
            self.due.emit(title, due[0].text)
            return
        self.missed.emit(f"{len(due)} 条提醒到点了", missed_summary(due))
