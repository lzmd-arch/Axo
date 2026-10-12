"""提醒：**登记得对、读得回来**，以及 P3b 加上去的"谁到点了、谁已经响过"。

这个文件量的是六件事，每一件对着一种"看不出来"的失效：

1. **宽容的读**：坏行跳过、好行照读。反过来的实现（一行坏就整份打不开）会让用户
   还没被提醒的事情一起消失，而"打开不了"这个症状指不到那一行上。
2. **不猜时间**：``due_at`` 缺失或不是数的行被丢掉。给它编一个"大概明天"出来，
   就变成了一条用户从没约过的提醒——那是本模块唯一不能有的东西。
3. **真落盘、同步落盘**：``add`` 返回时字节已经在文件里。用户说完"提醒我明天买
   牛奶"，下一句话之前它就该在文件里；异步写在"说完立刻关窗口"时丢掉它，而那次
   恰好是最容易被试到的一次。
4. **追加不重写**：老行逐字节留着。提醒簿没有 WAL、没有备份，重写一次就是一次
   全损的机会。
5. **响过就不再响**（P3b）。没有这一条，每次启动都会把过去所有提醒重弹一遍——
   而"设了提醒之后每次开程序都弹"比没有提醒更让人想把通知关掉。
6. **错过的一堆收成一句**（P3b）。逐条弹会变成一场风暴，而风暴的结局同上。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from axo import paths
from axo.agent.reminders import (
    REPEAT_DAILY,
    REPEAT_ONCE,
    REPEAT_WEEKLY,
    FiredStore,
    Occurrence,
    Reminder,
    ReminderStore,
    due_now,
    fired_key,
    missed_summary,
    new_reminder_id,
    next_due_at,
    next_occurrence,
)

AT = 1_760_000_000.0


@pytest.fixture
def book(tmp_path: Path) -> ReminderStore:
    return ReminderStore(tmp_path / "提醒" / "reminders.jsonl")


def _record(book: ReminderStore, text: str = "明天买牛奶。", **overrides: object) -> Reminder:
    fields: dict[str, object] = {"due_at": AT + 86400, "created_at": AT}
    fields.update(overrides)
    return book.add(text, **fields)  # type: ignore[arg-type]


# ── 落盘与读回 ─────────────────────────────────────────────────────────────


def test_add_then_all_round_trips(book: ReminderStore) -> None:
    written = _record(book, "周三下午三点交材料。", session_id="S1", turn_seq=3, call_id="c1")

    (read,) = book.all()
    assert read == written, "读回来的和写下去的不是同一条"
    assert read.session_id == "S1" and read.turn_seq == 3 and read.call_id == "c1"


def test_add_creates_the_parent_directory(tmp_path: Path) -> None:
    """父目录不存在时自己建。

    落盘位置由调用方给（``paths.reminders_path()``），而它在用户第一次约提醒之前
    并不存在——一个"文件写不进去"的报错会让用户失去他刚说的话。
    """
    deep = tmp_path / "a" / "b" / "reminders.jsonl"
    ReminderStore(deep).add("随便。", due_at=AT, created_at=AT)
    assert deep.is_file()


def test_all_is_empty_before_anything_is_registered(book: ReminderStore) -> None:
    """**还没约过提醒是常态，不是错误。**文件不存在返回空表，不抛。"""
    assert book.path.is_file() is False
    assert book.all() == []
    assert len(book) == 0


def test_registration_order_is_kept(book: ReminderStore) -> None:
    for index in range(3):
        _record(book, f"第 {index} 条。")
    assert [r.text for r in book.all()] == ["第 0 条。", "第 1 条。", "第 2 条。"]


def test_a_second_add_does_not_rewrite_the_first_line(book: ReminderStore) -> None:
    """**追加，不重写。**

    提醒簿没有 WAL、没有快照、没有备份——重写一次就是一次全损的机会（写到一半断电
    就全没了）。所以这条直接在字节层面验：第一次写下的那行必须逐字节还在。
    """
    _record(book, "第一条。")
    first_line = book.path.read_text(encoding="utf-8").splitlines()[0]
    _record(book, "第二条。")

    lines = book.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == first_line, "第二次写入把第一行重写了"
    assert len(lines) == 2


def test_the_file_is_utf8_with_lf_endings(book: ReminderStore) -> None:
    """编码与行尾都要**逐字节可预期**。

    行尾这条不是洁癖：这个文件将来要被别的东西读（P3b 的托盘、用户自己的脚本、
    ``grep``），而 Windows 上的默认行为会把 ``\\n`` 写成 CRLF——症状是"别人的脚本
    读到的每一行末尾都多一个字符"，而在这个仓库里看不到任何异常。
    """
    _record(book, "中文内容要能原样读回来。")
    raw = book.path.read_bytes()
    assert b"\r\n" not in raw
    assert "中文内容要能原样读回来。".encode() in raw
    assert json.loads(raw.decode("utf-8").strip())["text"] == "中文内容要能原样读回来。"


def test_ids_are_unique(book: ReminderStore) -> None:
    """同一条提醒的 id 不能撞——将来"取消这条提醒"要靠它。"""
    ids = {_record(book, f"第 {index} 条。").reminder_id for index in range(20)}
    assert len(ids) == 20


def test_ids_sort_by_registration_time() -> None:
    """id 带时间前缀，于是它天然按登记顺序排（也是能当文件名的形状）。"""
    first, second = new_reminder_id(), new_reminder_id()
    assert first[:14] <= second[:14]


# ── 宽容的读 ───────────────────────────────────────────────────────────────


def test_a_broken_line_does_not_take_the_others_with_it(book: ReminderStore) -> None:
    """坏行跳过、好行照读。

    半行 JSON 是断电时最常见的一种残骸，而"因为第 2 行坏了所以第 3 行也不给你看"
    是最没有必要的一种失败——那几行里装着用户还没被提醒的事情。
    """
    _record(book, "第一条。")
    with book.path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"reminder_id": "x", "text": "断在这里\n')  # 半行 JSON
        handle.write("\n")  # 空行
        handle.write("[1, 2, 3]\n")  # 不是对象
        handle.write('"就一个字符串"\n')  # 也不是对象
    _record(book, "最后一条。")

    assert [r.text for r in book.all()] == ["第一条。", "最后一条。"]


def test_a_reminder_without_a_usable_time_is_dropped(book: ReminderStore) -> None:
    """**不猜缺的时间。**

    给它编一个"大概明天"出来，比丢一行坏数据严重得多——它会变成一条用户从没约过的
    提醒。所以 ``due_at`` 缺失、不是数、或者是个布尔值（``True`` 是 ``int`` 的子类）
    的行一律丢掉。
    """
    rows = [
        {"reminder_id": "r1", "text": "没有 due_at。"},
        {"reminder_id": "r2", "text": "due_at 是字符串。", "due_at": "明天"},
        {"reminder_id": "r3", "text": "due_at 是布尔。", "due_at": True},
        {"text": "没有 id。", "due_at": AT},
        {"reminder_id": "r5", "text": "", "due_at": AT},
        {"reminder_id": "", "text": "id 是空的。", "due_at": AT},
        {"reminder_id": "r7", "text": "   ", "due_at": AT},
        {"reminder_id": "r8", "text": "这一条是好的。", "due_at": AT},
    ]
    book.path.parent.mkdir(parents=True, exist_ok=True)
    book.path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    assert [r.text for r in book.all()] == ["这一条是好的。"], "坏行被收下了，或者好行被连坐了"


def test_a_missing_created_at_does_not_break_the_row() -> None:
    """``created_at`` 缺失只落成 0.0。

    与 ``due_at`` 的区别在于**它不影响这条提醒要不要发生**：什么时候登记的少一个数
    只是详情页少一行字，而丢掉整条会让用户还没被提醒的事情消失。两种字段两种纪律，
    写在这里免得后人把它们统一成一种。
    """
    reminder = Reminder.from_dict({"reminder_id": "r1", "text": "有正文。", "due_at": AT})
    assert reminder is not None
    assert reminder.created_at == 0.0
    assert reminder.session_id == "" and reminder.call_id == "" and reminder.turn_seq == 0


def test_optional_fields_are_omitted_when_empty() -> None:
    """空的可选键**不写进文件**。

    照 :meth:`~axo.session.store.Message.to_dict` 的范式：以后加字段时，
    老行不会变成一坨空键；而一行里塞满 ``"call_id": ""`` 会让"这个字段到底有没有
    值"变得要靠人读才知道。
    """
    payload = Reminder(reminder_id="r1", text="有正文。", due_at=AT, created_at=AT).to_dict()
    assert set(payload) == {"reminder_id", "text", "due_at", "created_at"}


def test_from_dict_refuses_a_non_dict() -> None:
    """签名上写着 ``dict``，但这个值来自**文件**——文件的形状不由类型注解保证。"""
    assert Reminder.from_dict([1, 2, 3]) is None  # type: ignore[arg-type]


# ── 落盘位置 ───────────────────────────────────────────────────────────────


def test_reminders_live_directly_in_the_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """提醒簿躺在数据根目录下，**不在 ``memory/`` 里**。

    进 ``memory/`` 是错的那个方向："清空记忆"「重建索引」这类操作以目录为单位下手，
    而用户还没被提醒的事情不该跟着走。反过来它也不需要自己的子目录——只有一行行的
    文本，凑不出记忆库那种"必须整体搬家"的一堆伴生文件。
    """
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    path = paths.reminders_path()
    assert path == tmp_path / "reminders.jsonl"
    assert path.parent == paths.data_dir()
    assert "memory" not in path.parts


def test_reminders_path_does_not_create_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**文件本身不在这里创建**（同 ``memory_db_path()``）。

    还没约过提醒是常态，先建一个空文件只会让"文件在但内容空"与"文件还没有"变成两种
    要分开处理的状态。真正建它的是 ``ReminderStore.add``。
    """
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    assert paths.reminders_path().is_file() is False


# ── 到点了没有 ─────────────────────────────────────────────────────────────


def _reminder(reminder_id: str, due_at: float, text: str = "该做的事。") -> Reminder:
    """直接造一条，**不经过落盘**：这一节测的是判断，不是文件。"""
    return Reminder(reminder_id=reminder_id, text=text, due_at=due_at, created_at=AT)


def test_due_now_takes_only_what_has_come_due() -> None:
    """到点了的进来，没到点的留在外面；已经响过的也留在外面。

    三样判据挤在一个断言里，因为**它们缺任何一条，症状都是同一个**：提醒在该响的
    时候没响，或者在不该响的时候又响了一遍。
    """
    past, future = _reminder("a", AT - 10), _reminder("b", AT + 10)
    found = due_now([past, future], {"c"}, AT)
    assert [one.reminder.reminder_id for one in found] == ["a"]


def test_a_boundary_reminder_counts_as_due() -> None:
    """``due_at == now`` 算到点。差一个等号的实现会让"约在整点"的提醒晚一轮才响，
    而那一轮的间隔由定时器决定——可能是 60 秒。"""
    assert len(due_now([_reminder("a", AT)], set(), AT)) == 1


def test_due_now_is_sorted_by_when_they_were_due() -> None:
    """**按到期时间排，不按登记顺序。**

    程序关了一周再打开时，用户想知道的是"最早该响的那件事"，而那与"哪天登记的"
    没有关系。反过来的排序看起来也对——直到两条提醒的登记顺序与到期顺序不一致，
    而那在"上周约的明天"与"昨天约的今天"之间是常态。
    """
    late = _reminder("晚", AT - 1)
    early = _reminder("早", AT - 100)
    assert [one.reminder.reminder_id for one in due_now([late, early], set(), AT)] == ["早", "晚"]


def test_a_fired_reminder_never_comes_back() -> None:
    """**这一条就是"重启不重弹"。**"""
    one = _reminder("a", AT - 10)
    assert due_now([one], {"a"}, AT) == ()
    assert due_now([one], set(), AT) == (Occurrence(reminder=one, at=one.due_at),)


def test_next_due_at_is_the_earliest_pending() -> None:
    reminders = [_reminder("a", AT + 300), _reminder("b", AT + 100)]
    assert next_due_at(reminders, set()) == AT + 100


def test_next_due_at_may_be_in_the_past() -> None:
    """**可能是过去的某一刻**，而那不是"没有"。调用方据此立刻醒一次把错过的收掉，
    而不是傻等一个永远不来的时刻。"""
    assert next_due_at([_reminder("a", AT - 500)], set()) == AT - 500


def test_next_due_at_is_none_when_nothing_is_pending() -> None:
    """``None`` 与"过去的某一刻"是两件事：前者意味着**没有东西可等**，而那是可以
    真的把定时器停掉的状态。"""
    assert next_due_at([], set()) is None
    assert next_due_at([_reminder("a", AT + 5)], {"a"}) is None


# ── 已经响过的那本账 ───────────────────────────────────────────────────────


@pytest.fixture
def fired(tmp_path: Path) -> FiredStore:
    return FiredStore(tmp_path / "提醒" / "reminders.fired.jsonl")


def test_marking_then_reading_back(fired: FiredStore) -> None:
    fired.mark_occurrences([_occ("a", AT)], AT)
    assert fired.fired_keys() == frozenset({"a"})


def test_the_sidecar_survives_a_restart(fired: FiredStore) -> None:
    """**新开一个对象读同一个文件**，等价于下次启动。

    没有这一条，``fired_ids()`` 可以返回一个进程内的内存集合、测试全绿，而重新
    启动之后一切重来——那正是这个 sidecar 存在的理由。
    """
    fired.mark_occurrences([_occ("a", AT)], AT)
    fired.mark_occurrences([_occ("b", AT)], AT)
    restarted = FiredStore(fired.path)
    assert restarted.fired_keys() == frozenset({"a", "b"})


def test_an_empty_sidecar_is_an_empty_set(fired: FiredStore) -> None:
    """还没响过任何一条是常态，不是错误——同 :meth:`ReminderStore.all`。"""
    assert fired.fired_keys() == frozenset()


def test_marking_the_same_id_twice_is_harmless(fired: FiredStore) -> None:
    """重复标记是无害的，所以崩溃后重放**不必先查一遍**——"先查再写"在一个只追加
    的文件上本来也防不住竞态。"""
    fired.mark_occurrences([_occ("a", AT)], AT)
    fired.mark_occurrences([_occ("a", AT + 1)], AT + 1)
    assert fired.fired_keys() == frozenset({"a"})


@pytest.mark.parametrize("line", ['"就一个字符串"', "[1, 2, 3]", "{}", '{"reminder_id": 7}', "{"])
def test_broken_sidecar_lines_are_skipped(fired: FiredStore, line: str) -> None:
    """读不懂的行跳过，**其余照读**。

    反过来的实现（一行坏就整份打不开）会让所有提醒在下次启动时**一起重弹**，
    而症状指不到那一行上。
    """
    fired.mark_occurrences([_occ("好", AT)], AT)
    with fired.path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    assert fired.fired_keys() == frozenset({"好"})


def test_mark_all_writes_every_id(fired: FiredStore) -> None:
    """错过的一批一次标完。**一次打开文件**：二十次打开/关闭里任何一次失败都会留下
    一半已标一半没标的状态，而那正是"有些弹了两次、有些一次没弹"。"""
    fired.mark_occurrences([_occ(name, AT) for name in ("a", "b", "c")], AT)
    assert fired.fired_keys() == frozenset({"a", "b", "c"})


def test_mark_all_with_nothing_does_not_create_the_file(tmp_path: Path) -> None:
    """空批次不建文件：还没错过任何东西时不该在数据目录里留下一份空的账。"""
    store = FiredStore(tmp_path / "fired.jsonl")
    store.mark_occurrences([], AT)
    assert store.path.is_file() is False


def test_the_sidecar_is_not_the_register(tmp_path: Path) -> None:
    """**两本账，两个文件。**

    掺在一起的话，``ReminderStore.all()`` 会开始返回两种东西，而它的每一个调用方
    都是按"读出全部提醒"写的——症状是界面上多出一些来历不明的条目。
    """
    monkeypatch_path = tmp_path / "sessions.jsonl"
    del monkeypatch_path
    book = ReminderStore(tmp_path / "reminders.jsonl")
    fired = FiredStore(tmp_path / "reminders.fired.jsonl")
    _record(book)
    fired.mark_occurrences([_occ("a", AT)], AT)
    assert len(book.all()) == 1, "登记簿里只有提醒，没有 fired 行"
    assert fired.fired_keys() == frozenset({"a"})


# ── 错过的一堆收成一句 ─────────────────────────────────────────────────────


def test_missed_summary_lists_them_in_order() -> None:
    missed = [_occ_of(_reminder("a", AT, "买牛奶")), _occ_of(_reminder("b", AT, "交材料"))]
    assert missed_summary(missed) == "买牛奶、交材料"


def test_missed_summary_counts_the_overflow() -> None:
    """超过上限的折成"还有 N 条"——**不静默丢**，但也不必一次说五遍。"""
    missed = [_occ_of(_reminder(str(index), AT, f"第{index}件")) for index in range(8)]
    summary = missed_summary(missed, limit=3)
    assert summary == "第0件、第1件、第2件；还有 5 条"


def test_missed_summary_of_nothing_is_empty() -> None:
    """没有错过的不该冒出一句"还有 0 条"。"""
    assert missed_summary([]) == ""


def test_the_fired_path_is_its_own_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    fired_path = paths.reminders_fired_path()
    assert fired_path == tmp_path / "reminders.fired.jsonl"
    assert fired_path != paths.reminders_path(), "两本账必须是两个文件"
    assert fired_path.is_file() is False, "同 reminders_path：文件不在这里创建"


def _occ_of(reminder: Reminder) -> Occurrence:
    return Occurrence(reminder=reminder, at=reminder.due_at)


def _occ(reminder_id: str, at: float) -> Occurrence:
    """一条**一次性**的到点。改 ``repeat`` 的时候记得：键的形状跟着变（见
    :func:`fired_key` 与 ``FiredStore.fired_keys`` 那条不变量）。"""
    return Occurrence(
        reminder=Reminder(reminder_id=reminder_id, text=reminder_id, due_at=at, created_at=AT),
        at=at,
    )


def _at(year: int, month: int, day: int, hour: int = 22, minute: int = 30) -> float:
    """某个本地时刻。**不从 epoch 常量硬编**：星期几要靠日历算，而那是这里最容易
    写错的一格（``tm_wday`` 的 0 是周一，ISO 的 1 是周一）。"""
    return datetime(year, month, day, hour, minute).timestamp()


def _repeating(reminder_id: str, due_at: float, repeat: str, *, text: str = "该睡了") -> Reminder:
    return Reminder(reminder_id=reminder_id, text=text, due_at=due_at, created_at=AT, repeat=repeat)


# ── 重复 ────────────────────────────────────────────────────────────────────


def test_a_daily_reminder_fires_again_the_next_day() -> None:
    """**"每天"的核心判据。** 一次性那条响过就永久出局，而这一条明天还得回来。"""
    due = _at(2026, 3, 4)
    one = _repeating("d", due, REPEAT_DAILY)

    first = due_now([one], set(), due + 60)
    assert len(first) == 1 and first[0].missed == 1

    fired = {fired_key(one, first[0].at)}
    again = due_now([one], fired, due + 86400 + 60)

    assert len(again) == 1, "第二天没回来"
    assert again[0].at > first[0].at


def test_a_fired_one_shot_never_comes_back() -> None:
    """对照组：同样过了两天，一次性的那条不回来。"""
    due = _at(2026, 3, 4)
    one = Reminder(reminder_id="o", text="交房租", due_at=due, created_at=AT)
    fired = {fired_key(one, due)}

    assert due_now([one], fired, due + 3 * 86400) == ()


def test_a_daily_reminder_says_how_many_it_missed() -> None:
    """**补一次，但把次数说出来。** 三条理由都指向同一个方向：同一条提醒的三次通知
    内容完全相同（没有信息增量）、不静默丢（那个数就是承认它）、顺手避开 fired 账
    三倍增长。"""
    due = _at(2026, 3, 4)
    one = _repeating("d", due, REPEAT_DAILY)

    found = due_now([one], set(), due + 3 * 86400 + 60)

    assert len(found) == 1, "折叠成一次"
    assert found[0].missed == 4, "第 0/1/2/3 天，一共该响四次"
    assert "错过了 4 次" in missed_summary(found)


def test_a_weekly_reminder_keeps_the_same_weekday_and_clock() -> None:
    """每周那条：落在同一个星期几、同一个时刻，一周只一次。"""
    due = _at(2026, 3, 4, 20, 0)
    one = _repeating("w", due, REPEAT_WEEKLY, text="游泳课")
    base = time.localtime(due)

    found = due_now([one], set(), due + 6 * 86400)

    assert len(found) == 1
    got = time.localtime(found[0].at)
    assert got.tm_wday == base.tm_wday
    assert (got.tm_hour, got.tm_min) == (20, 0)


def test_a_weekly_reminder_without_weekdays_uses_the_due_day() -> None:
    """``weekly`` 但没给 ``weekdays`` → 取 ``due_at`` 那一天的星期几。

    这不是猜：它是这一行能算出来的唯一一件事实，也是用户说「每周三 22:30」时最可能
    的意思（工具那边正是这么落的盘）。"""
    due = _at(2026, 3, 4)
    one = Reminder(reminder_id="w", text="游泳课", due_at=due, created_at=AT, repeat=REPEAT_WEEKLY)

    assert one.weekdays == ()
    assert time.localtime(next_occurrence(one, set())).tm_wday == time.localtime(due).tm_wday


# ── 取消 ────────────────────────────────────────────────────────────────────


def test_a_cancelled_reminder_never_fires() -> None:
    """两级都要挡：到点那一步、以及"下一次什么时候醒"那一步。**少了后一条，定时器
    会一直为一条永远不会响的提醒设表。**"""
    one = _reminder("a", AT - 10)

    assert due_now([one], set(), AT, cancelled={"a"}) == ()
    assert next_due_at([one], set(), cancelled={"a"}) is None


def test_the_cancelled_path_is_its_own_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**第三本账、第三个文件。** 混进登记簿会让它的契约变成假的（那个文件装的是
    "还没被提醒的事情"），而重写老行更不行——它是只追加的。"""
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    cancelled = paths.reminders_cancelled_path()

    assert cancelled == tmp_path / "reminders.cancelled.jsonl"
    assert cancelled not in (paths.reminders_path(), paths.reminders_fired_path())
    assert cancelled.is_file() is False, "同另外两本：文件不在这里创建"


# ── 两个键的形状 ────────────────────────────────────────────────────────────


def test_the_two_key_shapes_do_not_collide() -> None:
    """一次性的是裸 id（与老行逐字节相容），重复的是复合键。**复合键里那一次的计划
    时刻就是这一次的身份**——用"实际响的时刻"会在延迟之后对不上，于是同一次响两遍。"""
    once = _reminder("a", AT)
    many = _repeating("a", AT, REPEAT_DAILY)

    assert fired_key(once, AT) == "a"
    assert fired_key(many, AT) == f"a@{int(AT)}"
    assert fired_key(many, AT) != fired_key(many, AT + 86400)


# ── 读坏数据的方向 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("stored", ["everyday", "DAILY", "hourly", 7, None, [], True])
def test_an_unknown_repeat_reads_as_once(stored: object) -> None:
    """**方向与 ``due_at`` 相反，这是刻意的。** 给一条没有时间的提醒编时间是造出一条
    用户从没约过的提醒（所以缺就丢行）；而给一个不认识的 ``repeat`` 猜"每天"只会让它
    响得比用户约的多——通知风暴的结局是用户把通知关掉，那时**所有**提醒一起失效。"""
    payload = {
        "reminder_id": "a",
        "text": "记住",
        "due_at": AT,
        "created_at": AT,
        "repeat": stored,
    }

    one = Reminder.from_dict(payload)

    assert one is not None
    assert one.repeat == REPEAT_ONCE


@pytest.mark.parametrize(
    "stored,expected",
    [
        ([0, 7], (0,)),  # 越界的丢掉，不 % 7 圆过去
        ([-1, 2], (2,)),
        ([1, "2", 1], (1,)),  # 不是整数的丢掉；重复的去掉
        ([True], ()),
        ("周三", ()),  # 根本不是列表
        (None, ()),
    ],
)
def test_out_of_range_weekdays_are_dropped(stored: object, expected: tuple[int, ...]) -> None:
    payload = {
        "reminder_id": "a",
        "text": "游泳课",
        "due_at": AT,
        "created_at": AT,
        "repeat": REPEAT_WEEKLY,
        "weekdays": stored,
    }

    one = Reminder.from_dict(payload)

    assert one is not None
    assert one.weekdays == expected


def test_a_plain_line_has_no_repeat_keys_on_disk() -> None:
    """**老行的键集合一个都不变。** 两个新字段非空才写——这是"以后加字段不会让老行
    变成一坨空键"那条范式的执行点。"""
    once = _reminder("a", AT)
    assert set(once.to_dict()) == {"reminder_id", "text", "due_at", "created_at"}

    many = _repeating("b", AT, REPEAT_DAILY)
    assert set(many.to_dict()) == {"reminder_id", "text", "due_at", "created_at", "repeat"}

    weekly = Reminder(
        reminder_id="c",
        text="游泳课",
        due_at=AT,
        created_at=AT,
        repeat=REPEAT_WEEKLY,
        weekdays=(0, 2),
    )
    assert weekly.to_dict()["weekdays"] == [0, 2]
