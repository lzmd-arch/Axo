"""定时触发：**到点了真的会响，而且只响一次**。

这个文件量的是五件事，每一件对着一种"只有真跑起来才会发现"的失效：

1. **到点了要发信号**，而且**只发一次**——一个响过之后又响的提醒会把用户训练成
   无视通知，那时所有提醒都不再有效。
2. **先写 sidecar 再弹**：反过来的话，弹出之后进程退出，下次启动重弹同一条。
3. **重启不重弹**：这是 :class:`FiredStore` 存在的全部理由，而它坏掉的表现是
   "每次开程序都弹一遍上周的提醒"。
4. **一堆错过收成一句**：关了一周再开，二十条逐条弹 = 一场通知风暴，而风暴的结局
   是用户把通知关掉。
5. **表真的接上了**（最后一条用真时钟等一秒）：定时器连错槽、忘了 ``start``、
   单发之后没有重排——这三种都是"代码看着对、跑起来不响"，而它们**只能**靠一次
   真的等待来证。

时间一律注入（``now=``）：等真时钟走完一条提醒的等待是没法写进测试的。
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from biosnn_memterm.agent.reminders import FiredStore, ReminderStore
from biosnn_memterm.ui.reminder_scheduler import MAX_DELAY_MS, MIN_DELAY_MS, ReminderScheduler

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Clock:
    """可以摆到任何一刻的假时钟。"""

    def __init__(self, at: float = 1000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def _stores(tmp_path: Any) -> tuple[ReminderStore, FiredStore]:
    return ReminderStore(tmp_path / "reminders.jsonl"), FiredStore(tmp_path / "fired.jsonl")


def _scheduler(
    tmp_path: Any, *, clock: Clock | None = None
) -> tuple[ReminderScheduler, ReminderStore, FiredStore, list[tuple[str, str]]]:
    """建一个表，并把两个信号的输出都收进一个列表（顺序即顺序）。"""
    reminders, fired = _stores(tmp_path)
    scheduler = ReminderScheduler(reminders=reminders, fired=fired, now=clock or Clock())
    seen: list[tuple[str, str]] = []
    scheduler.due.connect(lambda title, body: seen.append(("due", title + "|" + body)))
    scheduler.missed.connect(lambda title, body: seen.append(("missed", title + "|" + body)))
    return scheduler, reminders, fired, seen


def _wait(qapp: Any, ms: int) -> None:
    """让**真的** wall clock 走完 ``ms`` 毫秒，并且期间转事件循环。

    仓里没有 ``QTest.qWait``，惯例是 ``processEvents()``；这里需要的是"真的等"，
    所以是 ``processEvents`` + 睡一小觉的循环。
    """
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()


# ── 到点了 ─────────────────────────────────────────────────────────────────


def test_a_missed_reminder_fires_when_the_program_starts(tmp_path: Any, qapp: Any) -> None:
    scheduler, reminders, fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    one = reminders.add("买牛奶", due_at=900.0, created_at=800.0)

    scheduler.start()

    assert len(seen) == 1, "启动时收掉错过的那些，是这条路径存在的理由"
    assert seen[0][0] == "due"
    assert "买牛奶" in seen[0][1]
    assert one.reminder_id in fired.fired_ids()


def test_it_does_not_fire_twice(tmp_path: Any, qapp: Any) -> None:
    """同一条响过之后，**再扫多少次都不该再响**。

    这是整个文件里最要紧的一条：一个会重复响的提醒会训练用户无视通知，而"无视
    通知"这件事坏了之后，**所有**提醒一起失效。
    """
    scheduler, reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    scheduler.start()

    scheduler._on_timeout()
    scheduler._on_timeout()
    scheduler.reschedule()

    assert len(seen) == 1


def test_the_sidecar_is_written_before_the_alert(tmp_path: Any, qapp: Any) -> None:
    """**先写后弹**（write-ahead）。

    证法是让"弹"这一侧在接到信号的那一刻**去读盘**：那时 id 必须已经在文件里了。
    反过来的话，弹出之后、落盘之前进程退出（关窗收进托盘的用户随时会这么做），
    下次启动会把同一条重弹一遍。
    """
    reminders, fired = _stores(tmp_path)
    scheduler = ReminderScheduler(reminders=reminders, fired=fired, now=Clock(1000.0))
    one = reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    at_alert_time: list[frozenset[str]] = []
    scheduler.due.connect(lambda *_args: at_alert_time.append(fired.fired_ids()))

    scheduler.start()

    assert at_alert_time, "信号根本没发出来，这条测试什么都没证"
    assert one.reminder_id in at_alert_time[0], "弹的时候还没落盘——进程此时退出就会重弹"


def test_a_restart_does_not_ring_again(tmp_path: Any, qapp: Any) -> None:
    """**重开一次程序不该重弹。** 这是 :class:`FiredStore` 存在的全部理由。

    这里真的建第二个 :class:`ReminderScheduler`（同一个数据目录、新的对象），而不是
    复用第一个——复用验的是内存里的状态，而真实情形是"进程没了，文件还在"。
    """
    first, reminders, _fired, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    first.start()
    first.stop()

    # 新的一次运行：同一批文件，全新的对象
    again, _reminders, _fired2, seen = _scheduler(tmp_path, clock=Clock(1100.0))
    again.start()

    assert seen == [], "重启之后又弹了一遍"


def test_a_future_reminder_waits_for_its_time(tmp_path: Any, qapp: Any) -> None:
    scheduler, reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("开会", due_at=2000.0, created_at=900.0)

    scheduler.start()
    assert seen == [], "还没到点就响了"
    scheduler._on_timeout()
    assert seen == [], "还没到点就响了"

    scheduler._now = Clock(2000.0)
    scheduler._on_timeout()

    assert len(seen) == 1
    assert seen[0][0] == "due"
    assert "开会" in seen[0][1]


def test_the_late_one_is_worded_as_late(tmp_path: Any, qapp: Any) -> None:
    """一条早就该响的提醒，标题不能写成"提醒"——那是在暗示"就是现在"。

    它可能已经过去三天了，而用户据这句话判断的是"这件事还要不要现在做"。
    """
    scheduler, reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("买牛奶", due_at=900.0, created_at=800.0)

    scheduler.start()

    assert "错过" in seen[0][1], seen[0][1]


def test_an_on_time_one_is_not_worded_as_late(tmp_path: Any, qapp: Any) -> None:
    """反过来也要验：在跑着的时候准点响的那条**不该**说"错过"。

    只验上一条的话，把标题写死成"错过了一条提醒"也能全绿——而那会让准点的提醒看
    起来像迟到的。
    """
    scheduler, reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("开会", due_at=1000.0, created_at=900.0)

    # **走运行中那一条路**，不是 ``start()``：启动那一批本来就是"程序没开着的时候
    # 错过的"，措辞不同是刻意的。
    scheduler._on_timeout()

    assert seen and "错过" not in seen[0][1], seen


# ── 一堆错过收成一句 ───────────────────────────────────────────────────────


def test_a_backlog_arrives_as_one_message(tmp_path: Any, qapp: Any) -> None:
    """**三条过期 → 一句话，不是三条通知。**

    关了一周再打开会变成一场风暴，而风暴的结局是用户把通知关掉——那时**所有**提醒
    都失效，包括没过期的。也不静默丢：错过的事情仍然要说出来。
    """
    scheduler, reminders, fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    ids = [
        reminders.add(f"第{n}件事", due_at=900.0 + n, created_at=800.0).reminder_id
        for n in range(3)
    ]

    scheduler.start()

    assert len(seen) == 1, f"三条过期发了 {len(seen)} 条通知"
    assert seen[0][0] == "missed"
    assert "3" in seen[0][1]
    for each in ids:
        assert each in fired.fired_ids(), "聚合了但没标 fired，下次启动会再来一遍"


def test_an_on_time_reminder_is_not_in_the_backlog(tmp_path: Any, qapp: Any) -> None:
    """聚合只收**已经到点**的：一条将来的提醒不该被顺手标掉。

    标掉它的后果是**它永远不响**——而用户完全没有机会发现，因为界面上什么都没发生。
    """
    scheduler, reminders, fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("过去的", due_at=900.0, created_at=800.0)
    future = reminders.add("将来的", due_at=5000.0, created_at=800.0)

    scheduler.start()

    assert len(seen) == 1
    assert future.reminder_id not in fired.fired_ids()


# ── 表排在什么时候 ─────────────────────────────────────────────────────────


def test_the_delay_is_clamped(tmp_path: Any, qapp: Any) -> None:
    """延迟夹在 ``[MIN_DELAY_MS, MAX_DELAY_MS]`` 之间。

    下限防的是**紧循环**：``next_due`` 可能是过去，允许 0 的话一次收不干净就变成
    0 毫秒的循环，把 cpu 打满而界面上什么都看不出来。
    上限防的是**时钟跳变与休眠**：睡 8 小时醒来时 ``QTimer`` 不一定按预期醒，每 60
    秒对一次表是最省事的兜底。
    """
    scheduler, _reminders, _fired, _seen = _scheduler(tmp_path, clock=Clock(1000.0))

    assert scheduler._delay_ms(1000.0) == MIN_DELAY_MS, "过去的事该立刻醒，但不该是 0"
    assert scheduler._delay_ms(500.0) == MIN_DELAY_MS
    assert scheduler._delay_ms(1030.0) == 30_000
    assert scheduler._delay_ms(1000.0 + 86_400) == MAX_DELAY_MS, "一天之后的事不该排一整天"


def test_the_timer_is_armed_for_the_next_one(tmp_path: Any, qapp: Any) -> None:
    """排表要**真的**排上，而且排的是最近的那一条。

    只验信号的话，一个从不 ``start()`` 定时器的实现也能全绿（测试都是手动调
    ``_on_timeout`` 的）——而它在真机上是一条都不响。
    """
    scheduler, reminders, _fired, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("远的", due_at=1000.0 + 300, created_at=900.0)
    reminders.add("近的", due_at=1000.0 + 30, created_at=900.0)

    scheduler.start()

    assert scheduler._timer.isActive()
    assert 0 < scheduler._timer.remainingTime() <= 30_000, scheduler._timer.remainingTime()


def test_nothing_pending_stops_the_timer(tmp_path: Any, qapp: Any) -> None:
    """没有将来的提醒 → **表停掉**，不是排一个很远的时刻。

    空转的定时器会在进程存活期间一直醒（每次都要重读一遍文件），而"没有东西可等"
    是一个说得清的状态。这里也顺带钉住"启动时不许炸"——第一次运行的程序正是这个
    状态。
    """
    scheduler, _reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))

    scheduler.start()

    assert seen == []
    assert scheduler._timer.isActive() is False


def test_a_newly_registered_one_is_picked_up_by_reschedule(tmp_path: Any, qapp: Any) -> None:
    """``set_reminder`` 写下一条之后，表要**立刻**改排。

    没有这个调用，新提醒要等当前这一觉睡满（最长 60 秒）才被看见。测法是先让表停
    下来（没有待办的），再登记一条 + ``reschedule()``，然后验表又转起来了。
    """
    scheduler, reminders, _fired, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    scheduler.start()
    assert scheduler._timer.isActive() is False

    reminders.add("三十秒后", due_at=1030.0, created_at=1000.0)
    scheduler.reschedule()

    assert scheduler._timer.isActive(), "新登记的提醒没把表叫醒"
    scheduler._on_timeout()
    assert seen == [], "还没到点"


def test_stop_stops_the_timer(tmp_path: Any, qapp: Any) -> None:
    """退出时要停表。**没停的表会在收尾之后还来拍一下门。**"""
    scheduler, reminders, _fired, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("三十秒后", due_at=1030.0, created_at=1000.0)
    scheduler.start()
    assert scheduler._timer.isActive()

    scheduler.stop()
    scheduler.stop()  # 幂等

    assert scheduler._timer.isActive() is False


# ── 真的等一次（这三条路径只能这样证） ─────────────────────────────────────


def test_the_real_timer_really_rings(tmp_path: Any, qapp: Any) -> None:
    """**用真时钟等一秒。**

    上一条证的是"表排上了"，这一条证的是"排上的那件事真的会发生"——定时器连错了
    槽、``setSingleShot`` 之后没有重排，这两种都是"代码看着对、跑起来不响"，而它们
    只能靠一次真的等待来证。

    等 1 秒而不是更久：下限就是 1 秒（见 ``_delay_ms``），所以这条提醒最早也要
    1 秒后才响。
    """
    reminders, fired = _stores(tmp_path)
    scheduler = ReminderScheduler(reminders=reminders, fired=fired)  # **真时钟**
    seen: list[tuple[str, str]] = []
    scheduler.due.connect(lambda title, body: seen.append((title, body)))
    reminders.add("一秒后", due_at=time.time() + 0.1, created_at=time.time())

    scheduler.start()
    assert seen == [], "还没到点"
    _wait(qapp, 1500)

    assert len(seen) == 1, f"等了 1.5 秒，收到 {len(seen)} 条"
    assert seen[0][1] == "一秒后"
