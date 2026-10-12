"""登录时那一次：看一眼、标一下、弹一条、退。

这个文件量的是 **`--remind` 那条路走完之后世界上变了什么**——它跑在登录时、没有控制台、
没有人看着，所以"它到底做成了没有"只能靠这里的断言（以及构建自检里的
``_check_remind``，那一条起的是真产物）。

三件事各对着一种"只有真跑起来才会发现"的失效：

1. **先标 fired，再弹。** 反过来的话，弹完、落盘之前进程退出，下次登录会再弹一遍。
   证法是让"弹"这一侧在接到通知的那一刻**去读盘**——那时 id 必须已经在文件里了。
2. **弹不出来也要标。** 退出码 1 的意思是"到点了、我如实说我没弹成"，而不是"什么都没
   发生"——后者会让用户下次登录再收一遍同一条。
3. **读不了就什么都不做。** 退出码 2：没标、也没弹。这一支最要紧的是**不要**留下半截
   状态，因为那会让一条真正的提醒永远消失。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axo.agent.reminders import CancelledStore, FiredStore, ReminderStore
from axo.headless import EXIT_NOTIFY_FAILED, EXIT_OK, EXIT_UNREADABLE, run

NOW = 1_800_000_000.0


class Recorder:
    """假的通知通道：记下被弹了什么，并回一个预设的成败。"""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, body: str) -> bool:
        self.calls.append((title, body))
        return self.ok


def _stores(tmp_path: Path) -> tuple[ReminderStore, FiredStore, CancelledStore]:
    return (
        ReminderStore(tmp_path / "reminders.jsonl"),
        FiredStore(tmp_path / "reminders.fired.jsonl"),
        CancelledStore(tmp_path / "reminders.cancelled.jsonl"),
    )


def _run(tmp_path: Path, notify: Recorder, *, now: float = NOW) -> int:
    reminders, fired, cancelled = _stores(tmp_path)
    return run(reminders=reminders, fired=fired, cancelled=cancelled, now=now, notify=notify)


# ── 没东西可做 ──────────────────────────────────────────────────────────────


def test_a_quiet_login_is_a_quiet_zero(tmp_path: Path) -> None:
    """**登录时绝大多数情况就是这样**：没有到点的，于是连通知函数都不该被调到。

    这一条不是礼节：真实现里那个通知函数会**建一个 Qt 应用并转 6 秒事件循环**，而
    一个星期里九成九的登录都没有到点的提醒。
    """
    notify = Recorder()

    assert _run(tmp_path, notify) == EXIT_OK
    assert notify.calls == []


def test_a_future_one_waits(tmp_path: Path) -> None:
    """还没到点的不算（`due_now` 的判据），所以这一次也不该弹。"""
    reminders, _fired, _cancelled = _stores(tmp_path)
    reminders.add("开会", due_at=NOW + 600, created_at=NOW - 60)
    notify = Recorder()

    assert _run(tmp_path, notify) == EXIT_OK
    assert notify.calls == []


# ── 到点了 ──────────────────────────────────────────────────────────────────


def test_it_marks_before_it_tells(tmp_path: Path) -> None:
    """**先标后弹**（write-ahead）。证法是让"弹"的那一刻去读盘。

    与 GUI 那一侧（``ui/reminder_scheduler.py`` 的 ``_fire``）逐字同序：反过来的话，
    弹出之后、落盘之前进程退出，下次登录会把同一条再弹一遍——而登录时跑的进程被关掉
    是常事（用户注销、关机）。**两个入口的顺序必须一致**，否则同一条提醒在两个入口下
    行为不一样。
    """
    reminders, fired, _cancelled = _stores(tmp_path)
    one = reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)
    seen: list[frozenset[str]] = []

    def notify(title: str, body: str) -> bool:
        seen.append(fired.fired_keys())
        return True

    _run(tmp_path, notify)  # type: ignore[arg-type]

    assert seen, "通知函数根本没被调到，这条测试什么都没证"
    assert one.reminder_id in seen[0], "弹的时候还没落盘——进程此时退出就会重弹"


def test_the_ledger_line_says_it_was_the_headless_entry(tmp_path: Path) -> None:
    """``by`` 写的是 ``"remind"``。

    它是两个入口（GUI / 无头）唯一的区分标志，也是那条已知取舍的诊断依据：无头与主程序
    同时到点会各弹一次，而打开那个文件一看两行、两个 ``by``，就知道是它——不是"重复写入
    坏了"。
    """
    reminders, fired, _cancelled = _stores(tmp_path)
    reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)

    _run(tmp_path, Recorder())

    rows = [
        json.loads(line)
        for line in fired.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["by"] for row in rows] == ["remind"]


def test_the_heading_says_it_was_missed(tmp_path: Path) -> None:
    """标题是"错过了一条提醒"，与 GUI 那一侧的 ``start()`` 同口径。

    登录时那些到点的，绝大多数本来就是**当时没人在**才错过的；把它说成"提醒"是在暗示
    "就是现在"。
    """
    reminders, _fired, _cancelled = _stores(tmp_path)
    reminders.add("该睡了", due_at=NOW - 3600, created_at=NOW - 7200)
    notify = Recorder()

    _run(tmp_path, notify)

    assert notify.calls and "错过" in notify.calls[0][0], notify.calls


def test_a_backlog_arrives_as_one_notification(tmp_path: Path) -> None:
    """关了一周攒下的几条**只弹一次**（``Alert`` 自己把它们收成一句）。

    逐条弹会变成一场通知风暴，而风暴的结局是用户把通知关掉——那时**所有**提醒一起
    失效。这一条同时钉住"折叠"与"不静默丢"：正文里那几条的名字都在。
    """
    reminders, _fired, _cancelled = _stores(tmp_path)
    for index in range(3):
        reminders.add(f"第{index}件事", due_at=NOW - 600 + index, created_at=NOW - 900)
    notify = Recorder()

    _run(tmp_path, notify)

    assert len(notify.calls) == 1
    _title, body = notify.calls[0]
    for index in range(3):
        assert f"第{index}件事" in body


def test_cancelled_ones_do_not_come_back(tmp_path: Path) -> None:
    """用户撤掉的那条，登录时也不许响。

    **取消要在这里也递一次**（与 GUI 那一侧同一条）：它是第三本账，而"哪个入口读它"
    是一件容易被漏掉的事——漏了之后，撤掉的提醒会在下次登录时突然响起来，而用户
    取消它的时候正是为了不要这件事。
    """
    reminders, _fired, cancelled = _stores(tmp_path)
    doomed = reminders.add("撤回的", due_at=NOW - 60, created_at=NOW - 120)
    reminders.add("留下的", due_at=NOW - 30, created_at=NOW - 120)
    cancelled.mark_cancelled([doomed.reminder_id], at=NOW - 10)
    notify = Recorder()

    _run(tmp_path, notify)

    assert len(notify.calls) == 1
    assert "留下的" in notify.calls[0][1]


def test_a_second_login_does_not_repeat_it(tmp_path: Path) -> None:
    """跑第二遍什么都不做——这是 fired 账存在的全部理由。

    真的跑第二次 ``run``（同一个数据目录、新的对象），而不是复用第一个：真实情形是
    "进程没了，文件还在"。
    """
    reminders, _fired, _cancelled = _stores(tmp_path)
    reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)

    assert _run(tmp_path, Recorder()) == EXIT_OK
    second = Recorder()
    assert _run(tmp_path, second) == EXIT_OK
    assert second.calls == [], "第二次登录又弹了一遍"


# ── 弹不出来 ────────────────────────────────────────────────────────────────


def test_a_dead_balloon_is_reported_as_one(tmp_path: Path) -> None:
    """弹不出来 → 退出码 1，**而 fired 账照写**。

    这个数是登录时那条路唯一的说法（没有控制台、没有人看着）。"弹不出来"与"没到点"
    必须分得开——它们要做的事完全不同：前者什么都不用做（这次已经算响过了），后者
    要等下一次。
    """
    reminders, fired, _cancelled = _stores(tmp_path)
    one = reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)

    assert _run(tmp_path, Recorder(ok=False)) == EXIT_NOTIFY_FAILED
    assert one.reminder_id in fired.fired_keys(), "没弹成还标了——下次登录会再来一遍"


# ── 读不了 / 写不了 ─────────────────────────────────────────────────────────


def test_an_unreadable_ledger_does_nothing(tmp_path: Path) -> None:
    """读不了 → 退出码 2：**没标、也没弹**，下次登录还会来。

    这一支最要紧的是不要留下半截状态：标了却没弹，等于那条提醒**永远消失**（它已经
    "响过"了，而没有任何人被告知过）。

    造法是往 ``reminders.jsonl`` 里写一段**坏掉的字节**（不是合法 UTF-8）：读文件那一步
    会抛 ``UnicodeDecodeError``，而它是个 ``ValueError`` 不是 ``OSError``——**这一条
    是写测试时才发现的**：原先只兜 ``OSError``，于是这种坏法会带着一份回溯冲出去，而
    没有控制台的进程拿到的是解释器给的退出码 **1**，与"通知弹不出来"撞在一起。

    （先试过的那两种不行：文件不在、或者那个位置是个目录——``ReminderStore.all`` 对
    前者返回空表、对后者当作"没有这个文件"，两者都是**故意的宽容**（见那个模块的
    "为什么是追加写、格式宽容"）。所以这里要的是一种**读得到、但读不懂**的坏法。）
    """
    (tmp_path / "reminders.jsonl").write_bytes(
        b'{"reminder_id": "' + bytes([0xFF, 0xFE]) + b'"}' + bytes([10])
    )
    reminders, _fired, _cancelled = _stores(tmp_path)
    reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)
    notify = Recorder()

    assert _run(tmp_path, notify) == EXIT_UNREADABLE
    assert notify.calls == []
    assert (tmp_path / "reminders.fired.jsonl").is_file() is False, "读都读不了却写了账"


def test_an_unwritable_ledger_does_not_tell_anyone(tmp_path: Path) -> None:
    """写不进去 → 退出码 2：**也没弹**。

    顺序是"先标后弹"，所以写失败时通知还没发出去——那正好是我们要的：这条提醒没被
    任何人告知过，下次登录还会来。反过来说，如果这里弹了，那条提醒就永远消失了
    （没标 fired + 用户已经看过一眼）。
    """
    reminders, _fired, _cancelled = _stores(tmp_path)
    reminders.add("该睡了", due_at=NOW - 60, created_at=NOW - 120)
    # 让"追加写"失败：那个文件的位置上放一个目录，父目录建得出来、文件开不了。
    (tmp_path / "reminders.fired.jsonl").mkdir()
    notify = Recorder()

    assert _run(tmp_path, notify) == EXIT_UNREADABLE
    assert notify.calls == [], "写不进去却还是弹了——那条提醒下次不会再来了"


@pytest.mark.parametrize("code", [EXIT_OK, EXIT_NOTIFY_FAILED, EXIT_UNREADABLE])
def test_the_exit_codes_are_the_documented_three(code: int) -> None:
    """三个数各自有意义（0 正常 / 1 弹不出来 / 2 读不了），而且是**互不相同的三个**。

    它们是登录时那条路唯一的说法，`tools/build_exe.py` 的 `_check_remind` 判的就是
    这几个数——两个常量撞在一起的话，自检会把"读不了账"当成"正常"收下。
    """
    assert code in (0, 1, 2)
