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

:meth:`_fire` 里 ``mark_occurrences`` 排在 ``fired.emit`` **之前**。反过来的话，弹出之后、
落盘之前进程退出（关窗收进托盘的用户随时会这么做），下次启动会**重弹**同一条。
两个方向的代价不对称：重弹每次都会发生，而"标了却没弹成"很少发生。

## 一堆到点时聚合成一句

关了一周再打开，可能有二十条过期。逐条弹会变成一场通知风暴，而风暴的结局是用户把
通知关掉——那时**所有**提醒都失效了。所以多于一条时由
:meth:`~axo.agent.reminders.Alert.body` 收成一句。**也不静默丢**：
错过的事情仍然要说出来，只是不必一次说五遍。

## 取消过的不许再响

`cancelled` 是**第三本账**（见 :class:`~axo.agent.reminders.CancelledStore`）：
:func:`~axo.agent.reminders.due_now` 与 :func:`~axo.agent.reminders.next_due_at` 都收
它。这一层只做一件事——**每一次读表时都递进去**。两处都要递，而漏掉哪一处都是故障：

* 只漏 ``due_now``：取消过的提醒照旧弹出来——用户撤了它，而它到点还是响了；
* 只漏 ``next_due_at``：表会一直排在那条已经取消的提醒上。它到点之后
  ``next_occurrence`` 永远返回一个**过去的**时刻，于是定时器按 :data:`MIN_DELAY_MS`
  每秒醒一次、每次什么都不弹——cpu 白转，而界面上一点异常都看不出来。

## 谁负责显示

这个类**不认识窗口、也不认识托盘**：把这一批到点包成一个
:class:`~axo.agent.reminders.Alert` 发 :attr:`fired` 就完了。显示是 ``MainWindow``
的事（聊天里一条 + 托盘一条气泡）。这样它可单测，而且"提醒长什么样"这个决定只在一个
地方——那个地方现在是 :meth:`~axo.agent.reminders.Alert.heading` / ``body()``，落在
纯层，于是它逐条可断言（原先它是这个 Qt 对象上的一个方法，核心腿里测不到）。

**一个信号，不是两个**（这里的 ``due`` / ``missed`` 曾经是一对）。那两个信号的形状是
"一条"与"一堆"的差别，而那个差别本来就只是**措辞**——现在由 :class:`Alert` 自己按
``items`` 的长度决定。两个信号还逼着接收方写两份几乎一样的接线，而其中一份迟早会漏
掉一句：症状是托盘气泡少弹一次，或者对话区里少一条。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..agent.reminders import (
    Alert,
    CancelledStore,
    FiredStore,
    ReminderStore,
    due_now,
    next_due_at,
)

__all__ = ["MAX_DELAY_MS", "MIN_DELAY_MS", "ReminderScheduler"]

MIN_DELAY_MS = 1000
"""两次扫之间至少隔这么久。见模块文档的"下限 1 秒"。"""

MAX_DELAY_MS = 60_000
"""最久睡这么久就重新对一次表。见模块文档的"上限 60 秒"。"""


class ReminderScheduler(QObject):
    """到点了就发信号。**措辞与显示都不归它管。**

    Args:
        reminders: 提醒簿（读）。位置由调用方给——这里不该知道应用的数据目录在哪儿。
        fired: 响过的那些（读 + 写）。写它是这个类**唯一**的副作用。
        cancelled: 取消过的那些（读）。**没有默认值**：默认成"没有取消过"的话，一个
            忘了传它的调用点会表现为"用户撤掉的提醒照旧弹"，而那正是取消功能存在的
            全部意义。少传就是当场 :exc:`TypeError`。
        now: 取当前时刻。默认 :func:`time.time`，可注入是为了让测试把"现在"摆到
            任何一刻：等真时钟走完一条提醒的等待是没法写进测试的。
        parent: Qt 父对象。
    """

    fired = Signal(object)
    """到点了：**一个** :class:`~axo.agent.reminders.Alert`。

    载荷是 ``object`` 而不是把 ``(标题, 正文)`` 拆成两个 ``str``：这一个到点的身份是
    "哪条 + 哪一次"，而措辞（一条还是多条、是不是错过）是派生出来的。拆开传的话，
    "一条"与"一堆"就会在信号这一层重新分成两条路，而那正是这一版要收掉的东西。
    """

    def __init__(
        self,
        *,
        reminders: ReminderStore,
        fired: FiredStore,
        cancelled: CancelledStore,
        now: Callable[[], float] = time.time,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._reminders = reminders
        self._fired = fired
        self._cancelled = cancelled
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
        pending = next_due_at(
            self._reminders.all(),
            self._fired.fired_keys(),
            cancelled=self._cancelled.cancelled_ids(),
        )
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
                **只影响一条的措辞**（判据在
                :meth:`~axo.agent.reminders.Alert.heading` 里）：一条三小时前就该响
                的提醒，标题写成"提醒"是不诚实的——它暗示"就是现在"。多条那一支的
                措辞本来就带时间含义（"到点了"），不必再分。
        """
        now = self._now()
        due = due_now(
            self._reminders.all(),
            self._fired.fired_keys(),
            now,
            cancelled=self._cancelled.cancelled_ids(),
        )
        if not due:
            return
        # **先写后弹**（write-ahead）。见模块文档：反过来的代价是每次退出都重弹。
        self._fired.mark_occurrences(due, now, by="gui")
        self.fired.emit(Alert(items=due, late=late))
