"""定时触发：**到点了真的会响，而且只响一次**。

这个文件量的是七件事，每一件对着一种"只有真跑起来才会发现"的失效：

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
6. **取消过的不许再响**，而且**两处都要认**：``due_now`` 漏了是"撤掉的照旧弹"，
   ``next_due_at`` 漏了是"表永远留在那条已撤的提醒上、每秒醒一次什么都不弹"。
   两处的症状差得很远，所以分两条测。
7. **重复日程按"哪一次"算**：每天的那条第二天要再响，而 fired 账里那两行是**两个
   不同的键**——这是 ``fired_key`` 的复合形状在调度器这一侧真的接上了的证据。

时间一律注入（``now=``）：等真时钟走完一条提醒的等待是没法写进测试的。
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pytest

from axo.agent.reminders import (
    REPEAT_DAILY,
    Alert,
    CancelledStore,
    FiredStore,
    ReminderStore,
)
from axo.ui.reminder_scheduler import MAX_DELAY_MS, MIN_DELAY_MS, ReminderScheduler

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


def _stores(tmp_path: Any) -> tuple[ReminderStore, FiredStore, CancelledStore]:
    """三本账（登记 / 响过 / 取消），与 ``MainWindow`` 里那三行一一对应。"""
    return (
        ReminderStore(tmp_path / "reminders.jsonl"),
        FiredStore(tmp_path / "fired.jsonl"),
        CancelledStore(tmp_path / "cancelled.jsonl"),
    )


def _scheduler(
    tmp_path: Any, *, clock: Clock | None = None
) -> tuple[ReminderScheduler, ReminderStore, FiredStore, CancelledStore, list[Alert]]:
    """建一个表，并把 :attr:`fired` 那一路上收到的通知按顺序收进一个列表。

    收**整个** :class:`~axo.agent.reminders.Alert` 而不是先拆成标题与正文：这一版
    的信号只有一个，"一条还是多条"是从 ``items`` 派生的——在测试里把它重新拆成两路
    的话，测的就是一个已经不存在的形状了。
    """
    reminders, fired, cancelled = _stores(tmp_path)
    scheduler = ReminderScheduler(
        reminders=reminders, fired=fired, cancelled=cancelled, now=clock or Clock()
    )
    seen: list[Alert] = []
    scheduler.fired.connect(lambda alert: seen.append(alert))
    return scheduler, reminders, fired, cancelled, seen


def _at(year: int, month: int, day: int, hour: int = 22, minute: int = 30) -> float:
    """某个本地时刻。与 ``tests/test_agent_reminders.py`` 的同名帮手是同一件事。

    **不从 epoch 常量硬编**（这个文件里原来那几个 ``1000.0`` 只够一次性用）：重复
    日程要走日历（``_at_same_clock``），而在 Windows 上 ``datetime.timestamp()``
    对 1970 年前后的时刻会当场 ``OSError``。真实的日期顺带让"周几"这一格能靠日历
    算出来。
    """
    return datetime(year, month, day, hour, minute).timestamp()


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
    scheduler, reminders, fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    one = reminders.add("买牛奶", due_at=900.0, created_at=800.0)

    scheduler.start()

    assert len(seen) == 1, "启动时收掉错过的那些，是这条路径存在的理由"
    assert seen[0].heading() == "错过了一条提醒"
    assert "买牛奶" in seen[0].body()
    assert one.reminder_id in fired.fired_keys()


def test_it_does_not_fire_twice(tmp_path: Any, qapp: Any) -> None:
    """同一条响过之后，**再扫多少次都不该再响**。

    这是整个文件里最要紧的一条：一个会重复响的提醒会训练用户无视通知，而"无视
    通知"这件事坏了之后，**所有**提醒一起失效。
    """
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
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
    reminders, fired, cancelled = _stores(tmp_path)
    scheduler = ReminderScheduler(
        reminders=reminders, fired=fired, cancelled=cancelled, now=Clock(1000.0)
    )
    one = reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    at_alert_time: list[frozenset[str]] = []
    scheduler.fired.connect(lambda _alert: at_alert_time.append(fired.fired_keys()))

    scheduler.start()

    assert at_alert_time, "信号根本没发出来，这条测试什么都没证"
    assert one.reminder_id in at_alert_time[0], "弹的时候还没落盘——进程此时退出就会重弹"


def test_a_restart_does_not_ring_again(tmp_path: Any, qapp: Any) -> None:
    """**重开一次程序不该重弹。** 这是 :class:`FiredStore` 存在的全部理由。

    这里真的建第二个 :class:`ReminderScheduler`（同一个数据目录、新的对象），而不是
    复用第一个——复用验的是内存里的状态，而真实情形是"进程没了，文件还在"。
    """
    first, reminders, _fired, _cancelled, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    first.start()
    first.stop()

    # 新的一次运行：同一批文件，全新的对象
    again, _reminders, _fired2, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1100.0))
    again.start()

    assert seen == [], "重启之后又弹了一遍"


def test_a_future_reminder_waits_for_its_time(tmp_path: Any, qapp: Any) -> None:
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("开会", due_at=2000.0, created_at=900.0)

    scheduler.start()
    assert seen == [], "还没到点就响了"
    scheduler._on_timeout()
    assert seen == [], "还没到点就响了"

    scheduler._now = Clock(2000.0)
    scheduler._on_timeout()

    assert len(seen) == 1
    assert seen[0].heading() == "提醒"
    assert "开会" in seen[0].body()


def test_the_late_one_is_worded_as_late(tmp_path: Any, qapp: Any) -> None:
    """一条早就该响的提醒，标题不能写成"提醒"——那是在暗示"就是现在"。

    它可能已经过去三天了，而用户据这句话判断的是"这件事还要不要现在做"。
    """
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("买牛奶", due_at=900.0, created_at=800.0)

    scheduler.start()

    assert seen[0].heading() == "错过了一条提醒", seen[0].heading()


def test_an_on_time_one_is_not_worded_as_late(tmp_path: Any, qapp: Any) -> None:
    """反过来也要验：在跑着的时候准点响的那条**不该**说"错过"。

    只验上一条的话，把标题写死成"错过了一条提醒"也能全绿——而那会让准点的提醒看
    起来像迟到的。
    """
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("开会", due_at=1000.0, created_at=900.0)

    # **走运行中那一条路**，不是 ``start()``：启动那一批本来就是"程序没开着的时候
    # 错过的"，措辞不同是刻意的。
    scheduler._on_timeout()

    assert seen and "错过" not in seen[0].heading(), seen


# ── 一堆错过收成一句 ───────────────────────────────────────────────────────


def test_a_backlog_arrives_as_one_message(tmp_path: Any, qapp: Any) -> None:
    """**三条过期 → 一句话，不是三条通知。**

    关了一周再打开会变成一场风暴，而风暴的结局是用户把通知关掉——那时**所有**提醒
    都失效，包括没过期的。也不静默丢：错过的事情仍然要说出来。
    """
    scheduler, reminders, fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    ids = [
        reminders.add(f"第{n}件事", due_at=900.0 + n, created_at=800.0).reminder_id
        for n in range(3)
    ]

    scheduler.start()

    assert len(seen) == 1, f"三条过期发了 {len(seen)} 条通知"
    assert seen[0].heading() == "3 条提醒到点了"
    assert "第0件事" in seen[0].body() and "第2件事" in seen[0].body()
    for each in ids:
        assert each in fired.fired_keys(), "聚合了但没标 fired，下次启动会再来一遍"


def test_an_on_time_reminder_is_not_in_the_backlog(tmp_path: Any, qapp: Any) -> None:
    """聚合只收**已经到点**的：一条将来的提醒不该被顺手标掉。

    标掉它的后果是**它永远不响**——而用户完全没有机会发现，因为界面上什么都没发生。
    """
    scheduler, reminders, fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    reminders.add("过去的", due_at=900.0, created_at=800.0)
    future = reminders.add("将来的", due_at=5000.0, created_at=800.0)

    scheduler.start()

    assert len(seen) == 1
    assert future.reminder_id not in fired.fired_keys()


# ── 取消过的不许再响 ───────────────────────────────────────────────────────


def test_a_cancelled_reminder_does_not_ring(tmp_path: Any, qapp: Any) -> None:
    """撤掉的那条到点了也不响，而且**不许**被标 fired。

    只验"没响"是不够的：把取消递给 ``due_now`` 而同时照旧标 fired 的写法也能过那一条。
    多出来那一行不是无害的——fired 的意思是"这一条响过了"，而它没响过。写错的后果在
    "撤销取消"这类功能上会露出来（那条已经躺在响过的账里了），而且文件的读法跟着变。
    """
    scheduler, reminders, fired, cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    one = reminders.add("买牛奶", due_at=900.0, created_at=800.0)
    cancelled.mark_cancelled([one.reminder_id], at=950.0)

    scheduler.start()

    assert seen == [], "撤掉的提醒还是响了"
    assert one.reminder_id not in fired.fired_keys(), "撤掉的提醒被标成了响过"


def test_a_cancelled_reminder_does_not_hold_the_timer(tmp_path: Any, qapp: Any) -> None:
    """取消之后**表也要放手**——这是"两处都要递"的另一半。

    只把取消递给 ``due_now``（漏了 ``next_due_at``）的写法在这里露馅：那条提醒的
    ``due_at`` 永远欠着，``next_occurrence`` 于是永远返回一个**过去的**时刻，定时器
    按下限每秒醒一次、每次什么都不弹——cpu 白转，而界面上一点异常都看不出来。
    """
    scheduler, reminders, _fired, cancelled, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
    one = reminders.add("买牛奶", due_at=5000.0, created_at=900.0)
    scheduler.start()
    assert scheduler._timer.isActive()

    cancelled.mark_cancelled([one.reminder_id], at=1000.0)
    scheduler.reschedule()

    assert scheduler._timer.isActive() is False, "表还留在一条已经撤掉的提醒上"


def test_cancelling_one_leaves_the_others_alone(tmp_path: Any, qapp: Any) -> None:
    """撤掉一条不该动到别的——``cancelled_ids`` 是个集合，而它按 id 比。"""
    scheduler, reminders, _fired, cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    doomed = reminders.add("撤回的", due_at=900.0, created_at=800.0)
    reminders.add("留下的", due_at=950.0, created_at=800.0)
    cancelled.mark_cancelled([doomed.reminder_id], at=999.0)

    scheduler.start()

    assert len(seen) == 1
    assert "留下的" in seen[0].body()


# ── 重复日程按"哪一次"算 ───────────────────────────────────────────────────


def test_a_repeating_reminder_rings_again_the_next_day(tmp_path: Any, qapp: Any) -> None:
    """每天的那条，**第二天还要再响**——而 fired 账里两行是**两个不同的键**。

    这条钉的是 ``fired_key`` 的复合形状在调度器这一侧真的接上了。键只有裸 id 的话，
    第二天那条会被当成"已经响过"而永远沉默；而读写两侧各判一次"这条是不是重复"的话，
    症状是**每次开程序都重弹**。两种都只有真跑第二遍才看得见。
    """
    due = _at(2026, 3, 4)
    scheduler, reminders, fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(due + 60))
    one = reminders.add("该睡了", due_at=due, created_at=due - 3600, repeat=REPEAT_DAILY)

    scheduler.start()
    assert len(seen) == 1

    scheduler._now = Clock(due + 86_400 + 60)
    scheduler._on_timeout()

    assert len(seen) == 2, "每天的提醒第二天没响"
    keys = {key for key in fired.fired_keys() if key.startswith(one.reminder_id)}
    assert len(keys) == 2, f"两次到点在账上成了同一个键：{keys}"


def test_a_repeating_reminder_says_how_many_were_missed(tmp_path: Any, qapp: Any) -> None:
    """关了三天再打开：**补一次，但把次数说出来**。

    合并是因为同一条提醒的三次通知内容完全相同（没有信息增量），而 ``missed`` 就是
    "不静默丢"的落点：用户看到的应当是"该睡了（错过了 3 次）"，而不是一条看起来像
    刚刚才到点的提醒。正文里那行重复的措辞也一并钉住——它是用户认出"这是每天那条"
    的唯一线索。
    """
    due = _at(2026, 3, 4)
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(
        tmp_path, clock=Clock(due + 2 * 86_400 + 60)
    )
    reminders.add("该睡了", due_at=due, created_at=due - 3600, repeat=REPEAT_DAILY)

    scheduler.start()

    assert len(seen) == 1, "三次错过只该弹一条"
    assert "错过了 3 次" in seen[0].body(), seen[0].body()
    assert "每天" in seen[0].body(), "重复的措辞没进正文"


# ── 表排在什么时候 ─────────────────────────────────────────────────────────


def test_the_delay_is_clamped(tmp_path: Any, qapp: Any) -> None:
    """延迟夹在 ``[MIN_DELAY_MS, MAX_DELAY_MS]`` 之间。

    下限防的是**紧循环**：``next_due`` 可能是过去，允许 0 的话一次收不干净就变成
    0 毫秒的循环，把 cpu 打满而界面上什么都看不出来。
    上限防的是**时钟跳变与休眠**：睡 8 小时醒来时 ``QTimer`` 不一定按预期醒，每 60
    秒对一次表是最省事的兜底。
    """
    scheduler, _reminders, _fired, _cancelled, _seen = _scheduler(tmp_path, clock=Clock(1000.0))

    assert scheduler._delay_ms(1000.0) == MIN_DELAY_MS, "过去的事该立刻醒，但不该是 0"
    assert scheduler._delay_ms(500.0) == MIN_DELAY_MS
    assert scheduler._delay_ms(1030.0) == 30_000
    assert scheduler._delay_ms(1000.0 + 86_400) == MAX_DELAY_MS, "一天之后的事不该排一整天"


def test_the_timer_is_armed_for_the_next_one(tmp_path: Any, qapp: Any) -> None:
    """排表要**真的**排上，而且排的是最近的那一条。

    只验信号的话，一个从不 ``start()`` 定时器的实现也能全绿（测试都是手动调
    ``_on_timeout`` 的）——而它在真机上是一条都不响。
    """
    scheduler, reminders, _fired, _cancelled, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
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
    scheduler, _reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))

    scheduler.start()

    assert seen == []
    assert scheduler._timer.isActive() is False


def test_a_newly_registered_one_is_picked_up_by_reschedule(tmp_path: Any, qapp: Any) -> None:
    """``set_reminder`` 写下一条之后，表要**立刻**改排。

    没有这个调用，新提醒要等当前这一觉睡满（最长 60 秒）才被看见。测法是先让表停
    下来（没有待办的），再登记一条 + ``reschedule()``，然后验表又转起来了。
    """
    scheduler, reminders, _fired, _cancelled, seen = _scheduler(tmp_path, clock=Clock(1000.0))
    scheduler.start()
    assert scheduler._timer.isActive() is False

    reminders.add("三十秒后", due_at=1030.0, created_at=1000.0)
    scheduler.reschedule()

    assert scheduler._timer.isActive(), "新登记的提醒没把表叫醒"
    scheduler._on_timeout()
    assert seen == [], "还没到点"


def test_stop_stops_the_timer(tmp_path: Any, qapp: Any) -> None:
    """退出时要停表。**没停的表会在收尾之后还来拍一下门。**"""
    scheduler, reminders, _fired, _cancelled, _seen = _scheduler(tmp_path, clock=Clock(1000.0))
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
    reminders, fired, cancelled = _stores(tmp_path)
    scheduler = ReminderScheduler(  # **真时钟**
        reminders=reminders, fired=fired, cancelled=cancelled
    )
    seen: list[Alert] = []
    scheduler.fired.connect(lambda alert: seen.append(alert))
    reminders.add("一秒后", due_at=time.time() + 0.1, created_at=time.time())

    scheduler.start()
    assert seen == [], "还没到点"
    _wait(qapp, 1500)

    assert len(seen) == 1, f"等了 1.5 秒，收到 {len(seen)} 条"
    assert seen[0].body() == "一秒后"
