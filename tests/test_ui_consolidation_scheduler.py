"""周期巩固的表：**什么时候醒，以及醒了之后谁接着排下一枪**。

这个文件量的是四件事，每一件对着一种"只有真跑起来才会发现"的失效：

1. **从没跑过 = 现在就该跑。** 换成"再等一整个周期"的话，一个每天开关一次程序的
   人**永远等不到**那一轮——每一觉都从这次启动重新数 24 小时。
2. **重排要再排上。** 单发 ``QTimer`` 响过之后自己不重排就永远沉默了，而症状是
   "巩固跑过一次，之后再也不跑"。这一条只有靠一次**真的等待**才证得了。
3. **让掉的那一轮要隔一整个周期。** ``last.json`` 只在真跑完之后写，所以让掉/失败
   之后走 :meth:`reschedule` 会算出负数、被夹成一分钟——每分钟醒一次、每分钟再让掉
   一次。:meth:`reschedule_full_interval` 就是为它存在的。
4. **坏输入不许抛。** 周期来自手改得动的注册表；抛在这里的症状是主窗口在用户改设置
   的那一刻崩掉，而那正是这个类存在的姿势要避免的。

时间一律注入（``now=``、``read_last=``）：等真时钟走完 24 小时是没法写进测试的。
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from axo.ui import consolidation_scheduler as module
from axo.ui.consolidation_scheduler import (
    INTERVAL_CEIL_S,
    INTERVAL_FLOOR_S,
    MIN_DELAY_MS,
    ConsolidationScheduler,
)

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


def _scheduler(
    *,
    interval_s: int = 86_400,
    last: dict[str, Any] | None = None,
    clock: Clock | None = None,
) -> tuple[ConsolidationScheduler, list[int]]:
    """建一个表，并把 ``due`` 收到一个列表里（列表长度即响过几次）。"""
    scheduler = ConsolidationScheduler(
        interval_s=interval_s,
        read_last=lambda: last,
        now=clock or Clock(),
    )
    seen: list[int] = []
    scheduler.due.connect(lambda: seen.append(1))
    return scheduler, seen


def _wait(qapp: Any, ms: int) -> None:
    """让**真的** wall clock 走完 ``ms`` 毫秒，并且期间转事件循环。

    与 ``test_ui_reminder_scheduler.py`` 里那个同形：仓里没有 ``QTest.qWait``，
    惯例是 ``processEvents()``，而这里需要的是"真的等"。
    """
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()


# ── 延迟算得对不对 ─────────────────────────────────────────────────────────


def test_never_run_means_the_first_round_is_now(qapp: Any) -> None:
    """从没跑过（``last.json`` 不在）→ 下限，也就是"现在就该跑"。

    这条决定了"新装好的程序在启动一分钟之后跑第一轮"。如果哪天有人把它改成"再等
    一个周期"，这一条会红——而那个改动在界面上完全看不出来，只有"用了一周还没有
    巩固过一次"这一个症状。
    """
    scheduler, _seen = _scheduler(last=None)

    assert scheduler._delay_ms() == MIN_DELAY_MS


@pytest.mark.parametrize("bad", ["昨天", True, None, [], float("nan"), float("inf"), float("-inf")])
def test_a_broken_timestamp_means_now_too(qapp: Any, bad: Any) -> None:
    """``at`` 不是可用的数 → 与"从没跑过"同一条出路，**不编一个时间出来**。

    ``read_last_run`` 保证的是"整个对象读得出来"，而手改过的 json 可以让 ``at`` 是
    字符串、``true``、``null``。``nan`` / ``inf`` 最容易漏：``json.loads`` **收得下**
    这两个字面量，而把它们交给 ``time`` 运算会得到 ``nan`` 比较恒假——排出来是一个
    永远不响的表（比抛更难发现）。
    """
    scheduler, _seen = _scheduler(last={"at": bad})

    assert scheduler._delay_ms() == MIN_DELAY_MS


def test_an_overdue_run_is_clamped_to_the_floor(qapp: Any) -> None:
    """早就该跑了（关了三天）→ 下限，**不是 0**。

    允许 0 就是一次 0 毫秒的紧循环，把 cpu 打满而界面上什么都看不出来。
    """
    scheduler, _seen = _scheduler(last={"at": 1000.0}, clock=Clock(1000.0 + 3 * 86_400))

    assert scheduler._delay_ms() == MIN_DELAY_MS


def test_a_mid_interval_last_run_returns_the_remainder(qapp: Any) -> None:
    """上次跑在 24 小时周期的第 6 小时 → 还差 18 小时。

    **这一条是"周期真的被读进去了"的唯一证据**：上面三条不管周期是多少都返回下限。
    """
    scheduler, _seen = _scheduler(last={"at": 1000.0}, clock=Clock(1000.0 + 6 * 3600))

    assert scheduler._delay_ms() == 18 * 3600 * 1000


def test_the_interval_decides_the_wait(qapp: Any) -> None:
    """同一份 ``last.json``、同一个"现在"，改周期就改等待——**乘除没错位**。"""
    now = Clock(1000.0 + 600)
    short, _ = _scheduler(interval_s=3600, last={"at": 1000.0}, clock=now)
    long, _ = _scheduler(interval_s=86_400, last={"at": 1000.0}, clock=now)

    assert short._delay_ms() == (3600 - 600) * 1000
    assert long._delay_ms() == (86_400 - 600) * 1000


# ── 周期本身：夹住坏值、改了就重排 ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (0, INTERVAL_FLOOR_S),
        (-1, INTERVAL_FLOOR_S),
        (59, INTERVAL_FLOOR_S),
        (60, 60),
        ("86400", 86_400),
        ("一天", INTERVAL_FLOOR_S),
        (None, INTERVAL_FLOOR_S),
        (True, INTERVAL_FLOOR_S),
        (float("nan"), INTERVAL_FLOOR_S),
        (float("inf"), INTERVAL_FLOOR_S),
        # 上限那一条是**溢出**，不是策略：毫秒数超过 2**31-1 时 ``QTimer.start``
        # 会在把 Python 整数转成 C 的 ``int`` 时抛，而那是从 Qt 信号里穿出来的。
        (10**30, INTERVAL_CEIL_S),
    ],
)
def test_the_interval_is_clamped_not_raised(qapp: Any, given: Any, expected: int) -> None:
    scheduler, _seen = _scheduler()

    scheduler.set_interval_s(given)

    assert scheduler.interval_s == expected


def test_the_ceiling_matches_the_settings_page(qapp: Any) -> None:
    """两处上限**是同一个数**（7 天），因为界面那一侧就是照它写的。

    守的是"下次改上限只改了一处"——那时界面允许 30 天而这里夹到 7 天，用户填 30 天
    保存后看到 168 小时，而没有任何一处说得清为什么。
    """
    from axo.ui.settings_dialog import _MAX_INTERVAL_HOURS

    assert INTERVAL_CEIL_S == _MAX_INTERVAL_HOURS * 3600


def test_an_interval_change_rearms_immediately(qapp: Any) -> None:
    """把 24 小时改成 1 小时之后**当场重排**，不等这一觉睡满。

    不重排的话最坏要等 24 小时才生效——用户改的是一个"每多久跑一次"的设置，而它
    看起来没反应。
    """
    scheduler, _seen = _scheduler(last={"at": 1000.0}, clock=Clock(1000.0))
    scheduler.start()
    assert scheduler._timer.remainingTime() > 23 * 3600 * 1000

    scheduler.set_interval_s(3600)

    assert 0 < scheduler._timer.remainingTime() <= 3600 * 1000


# ── 生命周期 ───────────────────────────────────────────────────────────────


def test_start_arms_the_timer_without_firing(qapp: Any) -> None:
    """``start()`` 排上第一次，**不同步发一枪**。

    这是与 ``ReminderScheduler.start`` 刻意的不同：那一侧启动时要把"程序没开着的时候
    错过的"补出来（同步发信号），而巩固没有"错过"这个概念——早 60 秒晚 60 秒跑的是
    同一批条目。所以 ``MainWindow`` 里它不必排在 ``chat_view`` 建好之后。
    """
    scheduler, seen = _scheduler(last=None)

    scheduler.start()

    assert seen == []
    assert scheduler._timer.isActive()


def test_reschedule_before_start_does_nothing(qapp: Any) -> None:
    """没接管之前，排表是空操作——不然退出之后还会有人来拍门。"""
    scheduler, _seen = _scheduler(last=None)

    scheduler.reschedule()

    assert scheduler._timer.isActive() is False


def test_stop_stops_the_timer(qapp: Any) -> None:
    """退出时要停表。**幂等**——``_shutdown`` 可能从两条路各走一次。"""
    scheduler, _seen = _scheduler(last=None)
    scheduler.start()
    assert scheduler._timer.isActive()

    scheduler.stop()
    scheduler.stop()

    assert scheduler._timer.isActive() is False
    # 停过之后再排也不该活过来：`reschedule` 认的是 `_running`，不是"表停着"。
    scheduler.reschedule()
    assert scheduler._timer.isActive() is False


def test_the_full_interval_reschedule_ignores_last_json(qapp: Any) -> None:
    """**让掉/失败之后走的那条路**：从现在起隔一整个周期，不问"距上次多久"。

    这一条是整个文件里最要紧的一条。让掉的一轮没有写 ``last.json``，所以此刻
    ``at`` 是上一轮那个旧值、``_delay_ms()`` 必然是下限——每分钟醒一次、每分钟再让掉
    一次，直到有东西变。而这条路径的症状（`last.json` 一直在被读、日志里什么都看不
    出来）恰恰是最难从现象追回代码的一种。
    """
    scheduler, _seen = _scheduler(last={"at": 1000.0}, clock=Clock(1000.0 + 3 * 86_400))
    scheduler.start()
    assert scheduler._delay_ms() == MIN_DELAY_MS, "前提：这一点上算出来确实是下限"

    scheduler.reschedule_full_interval()

    assert scheduler._timer.remainingTime() > 23 * 3600 * 1000


def test_a_timeout_reschedules_before_it_emits(qapp: Any) -> None:
    """表响了：**先重排，再发信号**。

    顺序与提醒那一侧相反，理由都在 :meth:`ConsolidationScheduler._on_timeout` 的文档
    里：``due`` 是同步 emit 的，接收方一抛就会从 ``emit`` 里穿出来、穿过 Qt 的定时器
    回调——那之后这个对象永远沉默，而症状是"巩固跑过一次，之后再也不跑"。

    测法是**在槽里看表**：槽被调到时表必须已经是活的。反过来写（先 emit 再排）时，
    槽里看到的是一张已经停掉的单发表——那正是"跑过一次就再也不跑"的实现。
    """
    scheduler, _seen = _scheduler(last=None)
    scheduler.start()
    at_emit: list[bool] = []
    scheduler.due.connect(lambda: at_emit.append(scheduler._timer.isActive()))

    scheduler._on_timeout()

    assert at_emit == [True], "槽被调到时表还没排上——这一枪之后就再也没有下一枪了"
    assert scheduler._timer.isActive()


# ── 真的等一次（"重排"只能这样证） ─────────────────────────────────────────


def test_the_real_timer_really_rings_and_rearms(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """**用真时钟等一次。**

    上一条证的是"表排上了"，这一条证的是"排上的那件事真的会发生，而且响过之后
    还有下一枪"——定时器连错槽、``setSingleShot(True)`` 之后忘了重排，这两种都是
    "代码看着对、跑起来不响"，而它们**只能**靠一次真的等待来证。

    把下限改小到 50 毫秒来等：真的等 60 秒是没法写进测试的。改的是模块常量，而
    :meth:`reschedule` 与 :meth:`_delay_ms` 都是在调用时读它。
    """
    monkeypatch.setattr(module, "MIN_DELAY_MS", 50)
    scheduler, seen = _scheduler(last=None)

    scheduler.start()
    assert seen == [], "start() 不该同步发一枪"

    _wait(qapp, 400)

    assert len(seen) >= 2, f"等了 0.4 秒，只响了 {len(seen)} 次——响了之后没有重排"
    assert scheduler._timer.isActive()
