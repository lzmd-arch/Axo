"""周期巩固的排程：**"距上次跑过多久"决定下一次什么时候醒**。

:mod:`axo.memory.consolidation` 管的是"这一轮按什么次序改库"（脏的那一
层），``ui.memory_worker.ConsolidationWorker`` 管的是"在后台把它跑完"（也脏），这里
只管**什么时候叫醒谁**：一个**单发** ``QTimer``，每次响过之后按最新的事实重排下一次。
形状抄自 :class:`~axo.ui.reminder_scheduler.ReminderScheduler`，单发 +
重排的理由也抄它——比周期表少两个说不清的地方（第一次什么时候扫、扫的粒度就是精度）。

## 为什么不复用 ``ReminderScheduler``

形状相近，**语义不同**，硬塞会让两个语义打架。提醒的语义是"**到点了**"：下一次醒来的
时刻**就是**下一条该响的时刻，错过一条就是错过一件事（所以它有 ``missed``，而且启动时
一定要把错过的补出来）。巩固的语义是"**周期性看一眼该不该跑**"：睡晚了只是跑得晚，
**没有"错过"这回事**——巩固的对象是库里的现状，早跑晚跑跑的是同一批条目。两个类各自
一个 ``QTimer``（各属各自的 ``QObject``）、两个不同的槽、**没有任何共享可变状态**，
所以它们并存不打架。

## 与提醒的三处刻意的不同

* **不抄那个 60 秒上限。** 那里的上限是为"机器休眠、每 60 秒重新对一次表"设计的；
  巩固是个 24 小时级的任务，按 60 秒对表等于每天白醒 1440 次，而每次醒来都要读一遍
  ``last.json``。这里**只留下限**。
* **不抄"没东西可等就停表"。** 提醒可以停表（"没有将来的提醒"是个说得清的状态，空转
  的定时器只是白读文件），巩固不能：**没有待办不等于不用再跑**——全库都被认过一遍
  之后，明天写进来的新记忆还要被认，而"明天"这件事只有时钟知道。
* **上限不是必需的兜底。** 提醒那条 60 秒上限还兼着"时钟跳变"的兜底，而这里的排程
  本来就是"读一下上次什么时候跑的、算出还差多久"，每次唤醒都重新读一遍文件——跳变
  最多让这一觉睡长一点或短一点，下一觉自动对齐。

## 下限为什么是 60 秒

``last.json`` 里的 ``at`` 可以是过去很久的（程序关了三天），那时"还差多久"是负数。
允许 0 的话就是一次 0 毫秒的紧循环，把 cpu 打满而界面上什么都看不出来。60 秒同时也是
"这个问题最早能得到答案"的时间——一轮本身可能跑几分钟，排得比这更密只会反复问同一件
事。:data:`MIN_DELAY_MS` 因此身兼三职：**从没跑过时的首次延迟**、**超期时的兜底**、
以及调用方"推迟一会儿再看"的步长（见 ``MainWindow`` 里那个代理 idle 检测）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..memory.consolidation import read_last_run

__all__ = ["INTERVAL_CEIL_S", "INTERVAL_FLOOR_S", "MIN_DELAY_MS", "ConsolidationScheduler"]

MIN_DELAY_MS = 60_000
"""两次唤醒之间至少隔这么久。见模块文档的"下限为什么是 60 秒"。"""

INTERVAL_FLOOR_S = MIN_DELAY_MS // 1000
"""周期能被调到的最短值（秒）。

与 :data:`MIN_DELAY_MS` **同值不是巧合**：比这更短的周期在这个功能上没有意义（一轮
本身可能跑几分钟），而夹到同一个数上，"手改注册表把间隔写成 0" 的后果就退化成"和最密
的合法周期一样"，不会变成一次紧循环。**注意这不该是设置界面的下限**——那一侧的下限是
小时（见 ``settings_dialog.MemoryPage``），这里挡的是绕过界面写进来的值。
"""

INTERVAL_CEIL_S = 7 * 24 * 3600
"""周期能被调到的最长值（秒）。

与设置界面上限（``settings_dialog._MAX_INTERVAL_HOURS`` = 168 小时）**同值是有意的**：
这个数只有一个理由——``QTimer.start`` 收的是 C 的 ``int``，而毫秒数一旦超过
``2**31 - 1``（约 24.8 天），PySide6 会在把 Python 整数转成 ``int`` 时抛
``OverflowError``。那会从 :meth:`ConsolidationScheduler.set_interval_s` 一路穿出
Qt 信号，也就是**一个设置项让主窗口在改它的那一刻崩掉**——正是这个夹子存在的理由。

取 7 天而不是贴着 24.8 天取：它同时是界面上的上限，而"两处上限是同一个数"比"两处上限
各自合法"更容易在下次改动里守住。
"""


class ConsolidationScheduler(QObject):
    """周期巩固的表。**除了一个 ``QTimer`` 与两行状态，它什么都没有。**

    Args:
        interval_s: 周期（秒）。**必填，没有默认值**——这个数的唯一真源是用户的设置
            （``memory/consolidation_interval_s``），在这里再写一个默认值就是第二个
            说法，而两个说法迟早会在某次改动里分岔。非整数、比
            :data:`INTERVAL_FLOOR_S` 小的值都会被夹住，见 :meth:`set_interval_s`。
        read_last: 读 ``last.json`` 的函数。默认 :func:`~axo.memory.
            consolidation.read_last_run`；可注入是为了让测试把"上次是什么时候跑的"
            摆到任何一刻——**等真时钟走完 24 小时是没法写进测试的**。
        now: 取当前时刻。默认 :func:`time.time`，可注入的理由同上。
        parent: Qt 父对象。
    """

    due = Signal()
    """**该跑一轮了。** 不带参数：跑不跑得成、跑成什么样是接收方的事（它要检查单飞、
    要等图、还可能把这一轮让掉）。这个类只报"时间到了"。"""

    def __init__(
        self,
        *,
        interval_s: int,
        read_last: Callable[[], dict | None] = read_last_run,
        now: Callable[[], float] = time.time,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._interval_s = _clamped_interval(interval_s)
        self._read_last = read_last
        self._now = now
        self._running = False
        self._timer = QTimer(self)
        # **单发**：响过之后由 :meth:`reschedule` 按最新的事实决定下一次是什么时候。
        # 周期表在这里是错的：它无从回答"第一次什么时候扫"，而那个问题正是启动路径。
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    # ── 状态 ────────────────────────────────────────────────────────────────

    @property
    def interval_s(self) -> int:
        """当前周期（夹过之后的那一个）。设置页要把用户填的值显示回去。"""
        return self._interval_s

    def set_interval_s(self, seconds: int) -> None:
        """周期改了。**当场重排**，不等这一觉睡完。

        不重排的话，把 24 小时改成 1 小时之后要等原来那一觉睡满（最长 24 小时）才
        生效——用户改的是一个"每多久跑一次"的设置，而它看起来没反应。
        """
        self._interval_s = _clamped_interval(seconds)
        self.reschedule()

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """接管：排上第一次。**不在这里同步发一枪。**

        与 :meth:`ReminderScheduler.start` 的区别就在这儿：那一侧启动时要把"程序没开
        着的时候错过的"补出来（**同步**发信号，所以调用方必须把槽接好之后再调它），
        而这一侧没有"错过"这个概念——要跑的那一轮等第一次唤醒就是了，早 60 秒晚 60
        秒跑的是同一批条目。
        """
        self._running = True
        self.reschedule()

    def stop(self) -> None:
        """停表（退出时用）。**幂等。**"""
        self._running = False
        self._timer.stop()

    def reschedule(self, delay_ms: int | None = None) -> None:
        """按**此刻的事实**重排下一次。

        Args:
            delay_ms: 指定延迟（毫秒）。``None``（默认）= **读 ``last.json`` 算出来**
                （:meth:`_delay_ms`），也就是"距该跑下一轮还有多久"。给一个数是为了让
                调用方推迟一段固定的时间（"现在正忙着生成，过一会儿再看"）——那种情况
                下重算一遍"距上次多久"是没意义的，答案明摆着还是"早就该跑了"。

        **``None`` 那一条走的是 :meth:`_delay_ms`，不是下限。** 直接排下限的话，这个
        表就退化成一个"每分钟问一次该不该跑"的轮询器——它照样能跑对（每次醒来重新
        读文件、算出来还是"早就该跑"），但**每天都白醒 1439 次**，而模块文档里那句
        "不抄那个 60 秒上限"就成了空话。
        """
        if not self._running:
            return
        self._timer.start(
            self._delay_ms() if delay_ms is None else max(MIN_DELAY_MS, int(delay_ms))
        )

    def reschedule_full_interval(self) -> None:
        """**隔一整个周期再看**。空轮与被让掉的那一轮走这条，不走 :meth:`reschedule`。

        为什么必须分开：``last.json`` 只在**真的跑完一轮**之后才写（见
        ``memory.consolidation.run_consolidation`` 的第 3 条）。于是"没有新记忆、
        这一轮让掉"和"这一轮失败了"这两种结局里，文件还是上一轮那个旧的 ``at``——
        下一次 :meth:`_delay_ms` 算出来必然是负数、被夹成 :data:`MIN_DELAY_MS`，也就
        是**每分钟醒一次、每分钟再让掉一次**，直到有东西变。那正是模块文档里说的
        "每天白醒 1440 次"，只不过这里会一直醒下去。

        给一个明确的延迟就绕开了那个算法：不问"距上次多久"，只问"从现在起隔多久"。
        失败走这条也是刻意的——**一轮失败不重试**，等下一个周期，理由与"没有待办不等
        于不用再跑"同源：失败的原因（库被别处占着、图过期）多半不会在一分钟后消失，
        而每分钟重试一次只会把一次失败变成一晚上的失败。
        """
        self.reschedule(self._interval_s * 1000)

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _delay_ms(self) -> int:
        """距"该跑下一轮"还有多久（毫秒），下限 :data:`MIN_DELAY_MS`。

        **从没跑过就是"现在就该跑"**（返回下限），这条决定了新装好的程序在启动一分钟
        之后跑第一轮。换成"再等一整个周期"的话，一个每天开关一次程序的人**永远等不
        到**那一轮——每一觉都从这次启动重新数 24 小时。

        ``read_last_run`` 自己对"文件没有 / 坏了 / 字段不对"一律返回 ``None``（见它的
        文档），所以这里只需要再挡一种情况：``at`` 不是数（手改过的 json）。**不编一个
        时间出来**——读不出来就是"不知道上次什么时候跑的"，与"从没跑过"在这里的后果
        相同，都是"现在就该跑"。

        **``inf`` / ``nan`` 也要挡，而且不能靠比较大小挡。** ``json.loads`` 收得下这两
        个字面量（它们不是合法 JSON，但 Python 的解析器默认接受）；``inf`` 会让下面的
        算式得到 ``inf``，而 ``int(inf)`` 直接抛 ``OverflowError``——那会从 ``start()``
        里穿出来，也就是**开机就崩**。``nan`` 更阴：``max(MIN_DELAY_MS, nan)`` 恰好返回
        下限，看起来是对的，全靠在比较里恒假。两种都当"读不出来"处理。
        """
        last = self._read_last()
        at = last.get("at") if isinstance(last, dict) else None
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
            return MIN_DELAY_MS
        remaining_ms = (float(at) + self._interval_s - self._now()) * 1000.0
        if not math.isfinite(remaining_ms):
            # ``now()`` 自己坏掉（注入的时钟返回 inf）时也走这条：与"读不出来"同一个
            # 出路。挡住它是因为 `int()` 会抛，而抛在 `start()` 里就是开机崩。
            return MIN_DELAY_MS
        return int(max(MIN_DELAY_MS, remaining_ms))

    @Slot()
    def _on_timeout(self) -> None:
        """表响了：**先重排，再发信号。**

        顺序是有讲究的，而且与提醒那一侧相反（那一侧是"先收掉错过的、再重排"，因为
        ``_fire`` 要写 sidecar）。这里先重排的理由是**别让一条抛异常的槽把表弄死**：
        ``due`` 是同步 emit 的，接收方一抛，异常会从 ``emit`` 里穿出来、穿过 Qt 的
        定时器回调——那之后这个对象就永远沉默了，而症状是"巩固跑过一次，之后再也
        不跑"。先排好表，最坏也只是那一轮没跑成。

        **重排算出来的多半是下限**（``last.json`` 此刻还是上一轮那个旧的）。那不是
        问题：真正的"下一次在 24 小时之后"由接收方在**这一轮结束**时重排（那时文件
        已经写了），这里排上的只是**兜底**——万一接收方忘了，"过一分钟再看一眼"仍
        然是个活着的行为，而不是一块停掉的表。
        """
        self.reschedule()
        self.due.emit()


def _clamped_interval(seconds: int) -> int:
    """把周期收成一个在 ``[INTERVAL_FLOOR_S, INTERVAL_CEIL_S]`` 里的整数秒。

    这里的宽容是**给手改过的注册表**留的，不是给设置页留的：那一侧的下限是小时、
    上限是 7 天，而绕过界面写进来的值可以是 0、可以是负数、可以是字符串、也可以
    是 10 的 30 次方。让它抛的话，症状是**主窗口在用户改设置的那一刻崩掉**——一个
    设置项把程序变成打不开，这是最不该有的一种失效。夹住它，最坏是"跑得比用户以为
    的密或疏"。上限那一条的理由见 :data:`INTERVAL_CEIL_S`（是溢出，不是策略）。
    """
    try:
        value = int(seconds)
    except (TypeError, ValueError, OverflowError):
        # ``OverflowError``：``int(float("inf"))`` 抛的就是它。它排在这里而不是
        # 让调用方去挡，是因为这个函数的全部意义就是"收不下就用默认的"。
        return INTERVAL_FLOOR_S
    return min(INTERVAL_CEIL_S, max(INTERVAL_FLOOR_S, value))
