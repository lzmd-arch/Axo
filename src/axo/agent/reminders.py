"""提醒的登记，以及**谁到点了、谁已经响过**。

``set_reminder`` 工具把一条提醒写进这里的文件。P3a 时这个模块的职责只有一条
——**登记得对、读得回来**——因为"到点了"要有常驻进程在听，而那要等 P3b 的托盘
（否则那是一个没有任何东西在听的时刻）。

P3b 到了，于是这里多了三件**不碰 Qt 的事**（判断全在这一层，Qt 那一侧只负责
"什么时候看一眼"与"弹出来"）：

* :func:`due_now` —— 现在该响的是哪几条；
* :func:`next_due_at` —— 下一次该在什么时候醒（``None`` = 没有将来的提醒）；
* :func:`missed_summary` —— 程序没开着的那些时刻里错过的、聚合成一句话。

以及 :class:`FiredStore`：**已经响过的那些**记在哪儿。它必须存在，否则每次启动
都会把过去所有提醒重弹一遍。

## 为什么它和记忆库分开

提醒是**还没发生的事**，记忆是**发生过的事**。混进 ``memcore`` 的第一个后果是
召回会开始返回一些未来的句子——它们在语义上就是"用户提到过的一句话"，权重、
衰减、SDR 全都算得出来，于是它会安安静静地出现在注入账本里。第二个后果更实在：
"清空聊天记录"与"删掉一条记忆"这类动作不该顺手动到用户还没被提醒的事情。

于是落成一个自己的 JSONL 文件（位置由调用方给，见下）。

## 为什么是追加写、格式宽容

同 :mod:`axo.session.store`：一行一条，只追加。读的时候**跳过读不懂的
行**，而不是让整份文件打不开——这个文件里装着用户还没被提醒的事情，而"因为第 3
行坏了所以第 4 行也不给你看"是最没有必要的一种失败。

## 为什么"已经响过"是**另一个**文件

``reminders.jsonl`` 的契约是"这个文件装的是**还没被提醒的事情**"。往里追加第二种
记录（"这条响过了"）会让那句话变成假的，也会让 :meth:`ReminderStore.all` 从
"读出全部提醒"变成"读出两种东西"——而它现有的每一个调用方都是按前者写的。

所以 :class:`FiredStore` 写一个 sidecar（位置见
:func:`~axo.paths.reminders_fired_path`）。它**只追加**：一次已触发是
一件发生过的事，而"发生过"这件事没有"改"这个动作。重复标记同一个 id 是无害的
（读出来是集合），于是崩溃后重放的代价只是多一行。

## 谁决定文件在哪儿

**调用方**（``ui/main_window.py`` 从 :func:`~axo.paths.reminders_path`
取），不是本模块。理由与 :class:`~memcore.store.MemoryStore` 收一个 ``path`` 一样：
纯层里的东西不该知道"应用的数据目录"长什么样——那是 ``paths.py`` 的事，而测试要能
把它指到 ``tmp_path``。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "CATCH_UP_LIMIT",
    "REPEAT_DAILY",
    "REPEAT_ONCE",
    "REPEAT_WEEKLY",
    "Alert",
    "CancelledStore",
    "FiredStore",
    "Occurrence",
    "Reminder",
    "ReminderStore",
    "due_now",
    "fired_key",
    "missed_summary",
    "new_reminder_id",
    "next_due_at",
    "next_occurrence",
    "occurrences_after",
    "repeat_label",
]

MISSED_LIMIT = 5
"""错过的那一堆里，一句话最多列几条正文。其余折成"还有 N 条"。"""

REPEAT_ONCE = ""
REPEAT_DAILY = "daily"
REPEAT_WEEKLY = "weekly"
"""重复的三种取值。``""`` 是**一次性的**，也是**读不出来时的回落值**（见
:meth:`Reminder.from_dict`）。

**只有三种。** "每隔 N 小时"与"每月某日"都刻意没做：前者没有"几点"这个概念（起点算
哪一刻得单独定），后者多一个"31 号在二月怎么办"的边界要定——两件都不值当在这一版里
给一个提醒系统加第二种时间模型。"""

_REPEATS = (REPEAT_ONCE, REPEAT_DAILY, REPEAT_WEEKLY)

CATCH_UP_LIMIT = 366
"""一条重复提醒一次最多补出多少次。**先验值。**

它挡的不是性能（走三百多步是微秒级），是一种说不清的状态：``due_at`` 设在 1970 年的
每日提醒会算出"错过 20000 次"，而那个数对用户没有任何意义、对界面也不友好。留一年是
因为"整整一年没开过这个程序"已经越过了所有合理的用法。

**截断不丢提醒**：调用方按最近的那一次标 fired，它下一条照旧。"""

_WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
"""``time.localtime().tm_wday`` 的 0 是周一。**别用 ISO 的周一=1** —— 两套编号混用
是这类代码最经典的一个错，而症状是"每周三"显示成"每周二"。"""


@dataclass(frozen=True)
class Occurrence:
    """**某一条提醒的某一次**该响。

    ``at`` 是**计划时刻**，不是实际响的时刻——它是这一次的身份，也是 fired 账里
    那一行的键。两者分开是有后果的：程序关了三天再打开，这一次的 ``at`` 仍然是
    三天前那个 22:30，而不是现在。

    ``missed`` 是从上次响过到这一次一共**该响几次**。关了一周之后，"每天 22:30"
    那一条只弹一次，但要说出来它错过了七次——不静默丢，也不必一次说七遍。
    """

    reminder: Reminder
    at: float
    missed: int = 1


@dataclass(frozen=True)
class Alert:
    """这一次到点了：给**显示**与给**开口**的同一份事实。

    存在的理由是 ``ui/reminder_scheduler.py`` 模块文档里那句「提醒长什么样这个决定
    只在一个地方」——而那个地方原先是一个 Qt 对象的方法，于是它在核心腿里测不到。
    搬到这里之后它是纯函数，可以逐条断言。
    """

    items: tuple[Occurrence, ...]
    late: bool = False

    def heading(self) -> str:
        if len(self.items) > 1:
            return f"{len(self.items)} 条提醒到点了"
        if self.late or any(one.missed > 1 for one in self.items):
            return "错过了一条提醒"
        return "提醒"

    def body(self) -> str:
        if len(self.items) == 1:
            return _occurrence_line(self.items[0])
        return missed_summary(self.items)


def fired_key(reminder: Reminder, occurrence_at: float) -> str:
    """这一条**这一次**在 fired 账里的键。

    一次性的 → 就是 ``reminder_id``（与这个 sidecar 出现以来的每一行逐字节相容）；
    重复的 → ``f"{reminder_id}@{int(occurrence_at)}"``。

    **两个形状由这一处决定，因为读写必须对称。** 写的时候多写了、读的时候按复合键
    去比，两边各判一次"这条是不是重复"——错一边的症状很具体：**重复提醒每次开程序
    都重弹**（或者反过来，第二次之后再也不会响）。
    """
    if reminder.repeat == REPEAT_ONCE:
        return reminder.reminder_id
    return _compose_key(reminder.reminder_id, occurrence_at)


def _compose_key(reminder_id: str, occurrence_at: float) -> str:
    """复合键的**唯一**拼法。读的一侧（:func:`_fired_key`）也走这里——两处各写一遍
    格式串的话，"改了其中一处"这种最常见的疏漏不会被任何测试抓住。"""
    return f"{reminder_id}@{int(occurrence_at)}"


def _weekday_of(ts: float) -> int:
    """epoch 秒 → 这一天的星期几（0=周一 … 6=周日，本地时间）。"""
    return time.localtime(ts).tm_wday


def _repeat_days(reminder: Reminder) -> tuple[int, ...]:
    """这条提醒**一周里的哪几天**响。一次性的返回空表。"""
    if reminder.repeat == REPEAT_DAILY:
        return tuple(range(7))
    if reminder.repeat == REPEAT_WEEKLY:
        # 没有 ``weekdays`` 时退回 ``due_at`` 那一天——这个不变量在
        # :meth:`Reminder.from_dict` 里也兜了一次，两处都要有：那里兜的是"从文件
        # 读进来的"，这里兜的是"代码里直接构造的"。
        return reminder.weekdays or (_weekday_of(reminder.due_at),)
    return ()


def _at_same_clock(base: float, days: int) -> float:
    """把 ``base`` 的**本地墙上时间**搬到 ``days`` 天之后。

    **不是 ``base + days * 86400``。** 那个写法在有夏令时的时区里会把"每天 22:30"
    在换季那天变成 21:30 或 23:30——而"每天几点"这句话的语义是**墙上时间**，不是
    "隔了 86400 秒"。
    """
    parts = time.localtime(base)
    moved = datetime(
        parts.tm_year, parts.tm_mon, parts.tm_mday, parts.tm_hour, parts.tm_min, parts.tm_sec
    ) + timedelta(days=days)
    return moved.timestamp()


def _day_offset(base: float, ts: float) -> int:
    """``ts`` 比 ``base`` 晚几个**日历天**（按本地日期算，不是按秒差）。"""
    left = time.localtime(base)
    right = time.localtime(ts)
    return (
        datetime(right.tm_year, right.tm_mon, right.tm_mday)
        - datetime(left.tm_year, left.tm_mon, left.tm_mday)
    ).days


def _last_fired_at(reminder: Reminder, fired: Set[str]) -> float | None:
    """这条重复提醒**上一次响过的那一次**的计划时刻；一次都没响过返回 ``None``。"""
    prefix = reminder.reminder_id + "@"
    stamps = [
        int(key[len(prefix) :])
        for key in fired
        if key.startswith(prefix) and key[len(prefix) :].isdigit()
    ]
    return float(max(stamps)) if stamps else None


def occurrences_after(
    reminder: Reminder, *, after: float, until: float, limit: int = CATCH_UP_LIMIT
) -> tuple[float, ...]:
    """``(after, until]`` 里这条提醒该响的**计划时刻**，按时间先后。

    一次性的：``due_at`` 落在区间里就是它，否则空表（**没有第二次**）。
    重复的：从那一天起逐天看，星期几落在 :func:`_repeat_days` 里的那些天各出一个。

    **时刻永远照抄 ``due_at`` 的时分秒**，不做任何"往前挪一点"的智能调整——提醒是
    这个项目里唯一"到点有后果"的东西，不给它猜（同 ``catalog._resolve_due`` 里
    "只给日期不当成零点"那条）。
    """
    if until <= after:
        return ()
    if reminder.repeat == REPEAT_ONCE:
        return (reminder.due_at,) if after < reminder.due_at <= until else ()

    weekdays = _repeat_days(reminder)
    if not weekdays:
        return ()

    found: list[float] = []
    # **从 ``after`` 那一天起步，不是从 ``due_at`` 那一天。** 一条 1970 年的每日
    # 提醒从起点逐天走会白走两万天，而从结果那一头找完全等价——真正要防的是
    # "每次重排定时器都走两万步"这件事，它每 60 秒发生一次。
    offset = max(_day_offset(reminder.due_at, after), 0)
    horizon = offset + limit * 7 + 8
    while offset <= horizon and len(found) < limit:
        at = _at_same_clock(reminder.due_at, offset)
        if at > until:
            break
        if at > after and _weekday_of(at) in weekdays:
            found.append(at)
        offset += 1
    return tuple(found)


def next_occurrence(reminder: Reminder, fired: Set[str]) -> float | None:
    """这条提醒**欠着的那一次**；没有下一次时返回 ``None``。

    一次性的：没响过就是 ``due_at``，响过就 ``None``（它没有第二次）。
    重复的：在最后一次响过之后的下一次；**一次都没响过就还是 ``due_at``，哪怕它已经
    是过去**——那正是 `reschedule()` 要靠它立刻醒一次把欠的收掉的原因。
    """
    if reminder.repeat == REPEAT_ONCE:
        return None if fired_key(reminder, reminder.due_at) in fired else reminder.due_at
    last = _last_fired_at(reminder, fired)
    since = reminder.due_at - 1.0 if last is None else last
    hits = occurrences_after(reminder, after=since, until=float("inf"), limit=1)
    return hits[0] if hits else None


def due_now(
    reminders: Sequence[Reminder],
    fired: Set[str],
    now: float,
    *,
    cancelled: Set[str] = frozenset(),
    catch_up: int = CATCH_UP_LIMIT,
) -> tuple[Occurrence, ...]:
    """到点了、而且**这一次**还没响过的那些，**按计划时刻先后**。

    ``now`` 由调用方给而不是在这里读时钟：这一层要能在测试里把"现在"摆到任何
    一刻，而"到点了"的全部判断就是这一次比较。同理 ``fired`` 是**传进来的集合**，
    不是在这里读文件——纯函数与 IO 分开，后者在 :class:`FiredStore` 里。

    返回的是 :class:`Occurrence` 而不是 :class:`Reminder`：**一次到点的身份就是
    "哪条 + 哪一次"**。只带 ``Reminder`` 的话调用方拿不到那个身份，于是标 fired 时
    得自己再算一遍——而它算出来的可能与这里用的不是同一个。

    排序而不是按登记顺序：程序关着了一周之后，用户想知道的是"最早该响的那件事"，
    而那与"哪天登记的"没有关系。
    """
    found: list[Occurrence] = []
    for one in reminders:
        if one.reminder_id in cancelled:
            continue
        if one.repeat == REPEAT_ONCE:
            if one.due_at <= now and fired_key(one, one.due_at) not in fired:
                found.append(Occurrence(reminder=one, at=one.due_at))
            continue
        last = _last_fired_at(one, fired)
        since = one.due_at - 1.0 if last is None else last
        hits = occurrences_after(one, after=since, until=now, limit=catch_up)
        if hits:
            # **折叠成一次**，把漏了几次放进 ``missed``。三条理由都指向同一个方向：
            # 同一条提醒的三次通知内容完全相同（合并不是为了省事，是因为没有信息
            # 增量）；不静默丢（那三次必须被承认，承认的方式是这个数）；顺手避开
            # fired 文件三倍增长。
            found.append(Occurrence(reminder=one, at=hits[-1], missed=len(hits)))
    return tuple(sorted(found, key=lambda one: (one.at, one.reminder.reminder_id)))


def next_due_at(
    reminders: Sequence[Reminder], fired: Set[str], *, cancelled: Set[str] = frozenset()
) -> float | None:
    """下一次该在什么时候醒。**没有任何将来的提醒时返回 ``None``**。

    返回值可能**是过去**（错过的那些还没被标 fired 时）：调用方据此立刻醒一次把
    它们收掉，而不是傻等一个永远不来的时刻。``None`` 与"过去的某一刻"是两件不同
    的事——前者意味着"没有东西可等"，而这是可以真的把定时器停掉的状态。

    **刻意不收 ``now``**：给了它，这一层就会开始"往前看"，而"欠着的那一次已经过去
    了"正是这里唯一要表达的意思。判据放在调用方（谁的定时器谁决定怎么醒）。
    """
    pending: list[float] = []
    for one in reminders:
        if one.reminder_id in cancelled:
            continue
        at = next_occurrence(one, fired)
        if at is not None:
            pending.append(at)
    return min(pending) if pending else None


def repeat_label(reminder: Reminder) -> str:
    """``""`` / ``"每天 22:30"`` / ``"每周三 22:30"``。

    **一处措辞、三处用**（回灌给模型、托盘气泡、对话区）。分开写三遍的话，它们迟早
    会在"周三还是周二"这种地方分叉——而 `_WEEKDAY_NAMES` 那段已经说过，这类编号
    混用是这里最经典的一个错。
    """
    if reminder.repeat == REPEAT_ONCE:
        return ""
    parts = time.localtime(reminder.due_at)
    clock = f"{parts.tm_hour:02d}:{parts.tm_min:02d}"
    if reminder.repeat == REPEAT_DAILY:
        return f"每天 {clock}"
    names = "、".join(_WEEKDAY_NAMES[day] for day in sorted(set(_repeat_days(reminder))))
    return f"每{names} {clock}"


def _occurrence_line(one: Occurrence) -> str:
    """一次到点在正文里长什么样。标题归 :meth:`Alert.heading`。"""
    label = repeat_label(one.reminder)
    head = f"{label} · {one.reminder.text}" if label else one.reminder.text
    return f"{head}（错过了 {one.missed} 次）" if one.missed > 1 else head


def missed_summary(items: Sequence[Occurrence], *, limit: int = MISSED_LIMIT) -> str:
    """把错过的那一堆收成**一句话**。

    聚合成一条而不是逐条弹：关了一周再打开会变成一场通知风暴，而风暴的结局是用户
    把通知关掉——那时**所有**提醒都不再有效，包括没过期的。

    也不静默丢掉：错过的事情仍然要说出来，只是不必一次说五遍。
    """
    if not items:
        return ""
    listed = "、".join(_occurrence_line(one) for one in items[:limit])
    if len(items) > limit:
        return f"{listed}；还有 {len(items) - limit} 条"
    return listed


def new_reminder_id() -> str:
    """提醒 id。形状与 :func:`~memcore.ids.new_ulid` 同类：可排序、可当文件名。

    不 import ``memcore.ids`` 是因为它不是记忆——用同一个 id 空间会让人以为
    "提醒也能被召回"。时间前缀在前，于是文件里天然按登记顺序排。
    """
    return f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class Reminder:
    """一条登记下来的提醒。

    Attributes:
        reminder_id: 稳定 id。将来"取消这条提醒"要用它。
        text: 到点要告诉用户的那句话。
        due_at: 什么时候（epoch 秒，本地时钟）。
        created_at: 什么时候登记的。与 :attr:`due_at` 分开，因为"三天前约的今天的
            事"和"今天约的今天的事"在界面上是两回事。
        session_id: 在哪次会话里登记的。用于溯源，不用于触发。
        turn_seq: 那次会话的第几个回合，同上的定位成分。
        call_id: 是哪一次工具调用登记的。**它是这个数据结构里唯一能把提醒接回
            逐字稿的东西**——没有它，一条提醒就是一句来历不明的话。
        repeat: ``""``（一次性）/ ``"daily"`` / ``"weekly"``。见 :data:`REPEAT_ONCE`。
        weekdays: 一周里的哪几天响（0=周一 … 6=周日），**只有 weekly 且一次要定好几个
            星期几时才非空**。「每周三 22:30」这种情况由 :attr:`due_at` 那一天推出来，
            不需要模型去算星期几——那消掉了整类"周三算成 3 还是 2"的错。
    """

    reminder_id: str
    text: str
    due_at: float
    created_at: float
    session_id: str = ""
    turn_seq: int = 0
    call_id: str = ""
    repeat: str = REPEAT_ONCE
    weekdays: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """落盘的那个字典。**非空才写**——照 :meth:`~axo.session.store.Message.to_dict`
        的范式，于是以后加字段不会让老行变成一坨空键。"""
        payload: dict[str, Any] = {
            "reminder_id": self.reminder_id,
            "text": self.text,
            "due_at": self.due_at,
            "created_at": self.created_at,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.turn_seq:
            payload["turn_seq"] = self.turn_seq
        if self.call_id:
            payload["call_id"] = self.call_id
        if self.repeat:
            payload["repeat"] = self.repeat
            if self.weekdays:
                payload["weekdays"] = list(self.weekdays)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> Reminder | None:
        """读回来。**形状不对一律返回 ``None``**（调用方跳过那一行）。

        参数标成 ``Any`` 而不是 ``dict`` 是如实：这个值来自 ``json.loads``，一行
        ``"就一个字符串"`` 或者 ``[1, 2, 3]`` 都是合法 JSON。类型注解在这里管不住
        文件，所以这道检查必须在**这里**——``all()`` 只管按行读，形状的事一处说了算。

        **不做单位换算、不猜缺的时间**：``due_at`` 缺失或不是数的行直接丢掉。
        给一条没有时间的提醒编一个"大概明天"出来，比丢一行坏数据严重得多——
        它会变成一条用户从没约过的提醒。

        **``repeat`` 那一格的方向与 ``due_at`` 相反，这是刻意的。** 读不出来或不认识
        就按**一次性**读：给一条没有时间的提醒编时间是造出一条用户从没约过的提醒，
        而给一个不认识的 ``repeat`` 猜"每天"只会让它**响得比用户约的多**——通知风暴
        的结局是用户把通知关掉，那时**所有**提醒一起失效。两边的方向都由代价决定：
        这一格只能往"少响"那一侧落。

        **``weekly`` 但 ``weekdays`` 缺失时，取 ``due_at`` 那一天的星期几。** 这不是
        猜：它是这一行能算出来的唯一一件事实，也是用户说「每周三 22:30」时最可能的
        意思（工具那边正是这么落的盘）。

        ``weekdays`` 里的非整数与越界值（``-1``、``7``）**逐项丢掉**，丢完为空就退回
        上一条。越界值不许 ``% 7`` 圆过去——那是把一次错误吸收掉，而它下次还会犯。
        """
        if not isinstance(payload, dict):
            return None
        reminder_id = payload.get("reminder_id")
        text = payload.get("text")
        due_at = payload.get("due_at")
        if not isinstance(reminder_id, str) or not reminder_id:
            return None
        if not isinstance(text, str) or not text.strip():
            # 全空白的正文等于一条**没有内容**的提醒：它会在将来某次"到点了"里
            # 显示成一行空白，而用户没法从那一行看出该做什么。判据用 ``strip()``
            # 是因为 ``"   "`` 与非空字符串一样能过 ``isinstance``。
            # （存下去的仍是原文，不在这里替它修剪——读的一侧只做"收不收"的判断。）
            return None
        if isinstance(due_at, bool) or not isinstance(due_at, (int, float)):
            return None
        created_at = payload.get("created_at")
        repeat = _text(payload.get("repeat"))
        if repeat not in _REPEATS:
            repeat = REPEAT_ONCE
        return cls(
            reminder_id=reminder_id,
            text=text,
            due_at=float(due_at),
            created_at=float(created_at) if isinstance(created_at, (int, float)) else 0.0,
            session_id=_text(payload.get("session_id")),
            turn_seq=_count(payload.get("turn_seq")),
            call_id=_text(payload.get("call_id")),
            repeat=repeat,
            weekdays=_weekdays(payload.get("weekdays")),
        )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _weekdays(value: Any) -> tuple[int, ...]:
    """读 ``weekdays`` 那一格：**非整数与越界值逐项丢掉，不圆、不猜。**

    顺序保留、重复去掉。读不懂整体（不是列表）就返回空表——空表在
    :func:`_repeat_days` 那里会退回"``due_at`` 那一天的星期几"，那是这一行能算出来的
    唯一一件事实。
    """
    if not isinstance(value, (list, tuple)):
        return ()
    seen: list[int] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            continue
        if not 0 <= entry <= 6:
            continue
        if entry not in seen:
            seen.append(entry)
    return tuple(seen)


class ReminderStore:
    """提醒的落盘。**只追加，不改写。**"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def add(
        self,
        text: str,
        *,
        due_at: float,
        created_at: float,
        session_id: str = "",
        turn_seq: int = 0,
        call_id: str = "",
        repeat: str = REPEAT_ONCE,
        weekdays: Sequence[int] = (),
    ) -> Reminder:
        """登记一条。返回真正落下去的那条（带它自己的 id）。

        **同步写**（同 :class:`~axo.memory.writer.MemoryWriter` 的
        "每轮实时写入"）：用户说完"提醒我明天买牛奶"，这句话在下一句话之前就该
        在文件里。异步写会让"说完立刻关窗口"丢掉它——而这恰好是最容易被试到的
        那一次。

        **失败要炸**，不吞：这条路径上没有"让谁重试"的当事人（模型在闸门那边，
        这里只是一次落盘），所以静默失败等于用户以为约好了而其实没有。
        """
        reminder = Reminder(
            reminder_id=new_reminder_id(),
            text=text,
            due_at=due_at,
            created_at=created_at,
            session_id=session_id,
            turn_seq=turn_seq,
            call_id=call_id,
            repeat=repeat if repeat in _REPEATS else REPEAT_ONCE,
            weekdays=tuple(weekdays),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``newline="\n"``：Windows 上默认会把 ``\n`` 写成 CRLF，而这个文件将来要
        # 被别的工具读（P3b 的托盘、用户自己的脚本）。逐字节可预期的行尾比省一次
        # 参数重要。
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(reminder.to_dict(), ensure_ascii=False) + "\n")
        return reminder

    def all(self) -> list[Reminder]:
        """读出全部提醒，**按登记顺序**。

        文件不存在返回空表而不是抛：还没登记过任何提醒是**常态**，不是错误。
        """
        if not self._path.is_file():
            return []
        found: list[Reminder] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 半行 JSON：断电时最常见的一种残骸
                reminder = Reminder.from_dict(payload)
                if reminder is not None:
                    found.append(reminder)
        return found

    def __len__(self) -> int:
        return len(self.all())


class FiredStore:
    """**已经响过**的提醒 id。只追加，读出来是个集合。

    它存在的原因是 :func:`due_now` 需要知道"这一条已经响过了"——没有它，每次启动
    都会把过去所有提醒重弹一遍，而"设了提醒每次开程序都弹"比没有提醒更烦人。

    **写在前、弹在后**（write-ahead，调用方的纪律）：反过来的话，弹出之后、落盘
    之前进程退出，下次启动会重弹同一条。而反过来的代价（标了却因为弹失败而没弹）
    是用户错过一次提醒——两害相权，重弹更糟：它每次都发生，而弹出失败很少发生。

    与 :class:`ReminderStore` 同一条宽容度：**读的时候跳过读不懂的行**，绝不抛。
    这个文件里装的只是"响过没有"，一行坏了不该让其余全部重弹。
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def fired_keys(self) -> frozenset[str]:
        """响过的那些**次**。文件不存在返回空集合（还没响过任何一条是常态）。

        **两个形状**：一行里有 ``due_at``（重复提醒才写）→ 键是
        ``"<reminder_id>@<那次计划时刻>"``；没有 → 键就是 ``reminder_id``。
        这个 sidecar 出现以来写的每一行都是后者，所以**老行天然成立、不需要任何
        迁移**。

        > **不变量：``due_at`` 被写下来 ⟺ 这条提醒是重复的。** 它由
        > :meth:`mark_occurrences` 一处保证。读的一侧据此判断"这一行是不是
        > 复合键"——两边依据的信号不同（那边看 ``Reminder.repeat``，这边看字段在不在），
        > 所以这条不变量断了就会**静默错位**：重复提醒每次开程序都重弹，或者响过一次
        > 之后再也不会响。改了写的那一侧，这条注释与读的那一侧都要跟着看。
        """
        if not self._path.is_file():
            return frozenset()
        found: set[str] = set()
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = _fired_key(payload)
                if key:
                    found.add(key)
        return frozenset(found)

    def mark_occurrences(
        self, occurrences: Sequence[Occurrence], at: float, *, by: str = ""
    ) -> None:
        """把一批到点记进账里。**同步写、失败要炸**——同 :meth:`ReminderStore.add`：
        静默失败的表现是下次启动重弹一条用户已经看过的提醒。

        **一次打开文件**，不是每条开一次——错过二十条时那二十次打开/关闭里任何一次
        失败都会留下一半已标、一半没标的状态。

        ``by`` 是 ``"gui"`` / ``"remind"``（**谁弹的**，见 :class:`~axo.headless` 与
        ``ui/reminder_scheduler.py``）。它**不参与去重**，只让"同一次到点弹了两次"
        这件已知的失效可诊断：打开这个文件一看两行、两个 ``by``，就知道那是无头入口
        与主程序撞上了，而不是去怀疑"重复写入坏了"。**一个被接受的失效，如果不可
        诊断，下一个人会花一整轮去查一个已知答案。**
        """
        if not occurrences:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            for one in occurrences:
                payload: dict[str, Any] = {"reminder_id": one.reminder.reminder_id, "at": at}
                if one.reminder.repeat != REPEAT_ONCE:
                    # 见 :meth:`fired_keys` 那条不变量：这一句是它的唯一执行点。
                    payload["due_at"] = one.at
                if by:
                    payload["by"] = by
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class CancelledStore:
    """**取消过**的提醒 id。只追加，读出来是个集合。

    **为什么是第三个文件而不是往登记簿里写一行"这条撤了"。** :class:`ReminderStore`
    只会追加这条事实不退让（``test_a_second_add_does_not_rewrite_the_first_line`` 是
    逐字节守卫，而提醒簿没有 WAL、没有备份，重写一次就是一次全损），所以"取消"只能在
    **别处**记一件发生过的事。

    而"别处"有三个选择，选了这个：把取消行混进 ``reminders.jsonl`` 会让
    :meth:`ReminderStore.all` 从"读出全部提醒"变成"读出折叠后的当前状态"——它现有的
    每一个调用方都是按前者写的；更要紧的是它会**违反 ``paths`` 与模块文档里写明的
    契约**（"这个文件装的是还没被提醒的事情"）。多一个文件的代价是
    :class:`FiredStore` 已经付过的那笔账：三本账、三个集合、同一套宽容读。收益是
    :class:`ReminderStore` 与它那四条测试**一个字都不用动**。

    > **再加第四种记录之前，先把这三个 sidecar 合成一本事件账。** 三个是"三件事各自
    > 发生过"，四个就是"我们其实需要一本次序账"了。

    与 :class:`FiredStore` 逐字段同构，理由也一样：读的时候跳过读不懂的行，绝不抛。
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def cancelled_ids(self) -> frozenset[str]:
        """取消过的那些 id。文件不存在返回空集合（一条都没取消过是常态）。"""
        if not self._path.is_file():
            return frozenset()
        found: set[str] = set()
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reminder_id = _fired_id(payload)
                if reminder_id:
                    found.add(reminder_id)
        return frozenset(found)

    def mark_cancelled(self, reminder_ids: Sequence[str], at: float) -> None:
        """一次标一批，**一次打开文件**（同 :meth:`FiredStore.mark_occurrences`）。

        **同步写、失败要炸**：静默失败的表现是用户以为撤掉了、而它照旧到点响。
        """
        if not reminder_ids:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            for reminder_id in reminder_ids:
                payload = {"reminder_id": reminder_id, "at": at}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _fired_key(payload: Any) -> str:
    """一行 fired 记录读出来的**键**。形状不对返回空串（调用方跳过那一行）。

    比 :meth:`Reminder.from_dict` 宽容得多，因为它要的信息只有"这是哪一条、哪一次"。
    ``due_at`` 读不出来就退回裸 id（见 :meth:`FiredStore.fired_keys` 那条不变量）。
    """
    reminder_id = _fired_id(payload)
    if not reminder_id:
        return ""
    at = payload.get("due_at") if isinstance(payload, dict) else None
    if isinstance(at, bool) or not isinstance(at, (int, float)):
        return reminder_id
    return _compose_key(reminder_id, float(at))


def _fired_id(payload: Any) -> str:
    """一行记录里的 id。形状不对返回空串（调用方跳过那一行）。"""
    if not isinstance(payload, dict):
        return ""
    reminder_id = payload.get("reminder_id")
    return reminder_id if isinstance(reminder_id, str) else ""
