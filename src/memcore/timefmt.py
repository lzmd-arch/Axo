"""记忆行上那个时间怎么写。

**为什么单独一个模块。** 时间戳会出现在两个地方：装配时注入上下文的记忆行
（``axo/session/assemble.py``）、归并产物的正文（``memcore/merge.py``）。两处各写
一套 ``strftime`` 就会长出两种时间格式，而模型拿"两种长得不一样的时间"去比大小，
正是这一批要治的病。格式只许有这一处定义。

**为什么用本地时区。** L1 环境行写的是本地时间（「当前时间：2026-10-13 星期二
22:41」），记忆行要是 UTC，模型得自己换算才能回答「昨天我说了什么」——而它换算
不稳。同一个时钟、同一种格式，那两行才可比。

**没时间就不编时间。** ``created_at <= 0`` 表示这条记忆的时间不可知（老数据、或
调用方手里确实没有）。渲染方该省掉方括号，而不是打一个 1970-01-01 上去——编出来
的时间比没有时间更坏，它会被模型当事实读。
"""

from __future__ import annotations

from datetime import datetime

__all__ = ["format_when", "format_clock", "format_span", "has_time"]


def has_time(ts: float) -> bool:
    """这个时间戳可不可信。``<= 0`` 一律当作「不知道」。

    ``MemoryItem.created_at`` 的默认值是 ``now_ts()``，所以真库里几乎没有假时间；
    会出现 ``0`` 的是手搓的注入对象与老测试夹具。
    """
    return ts > 0


def format_when(ts: float) -> str:
    """单点时刻：``2026-10-13 22:41``。

    与 L1 环境行的 ``%Y-%m-%d %H:%M`` 同族。那里多一个星期几，是因为「今天星期几」
    是模型自己推不出来的常数；记忆行不需要——行里本来就有完整日期。
    """
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_clock(ts: float) -> str:
    """同一天内的短刻度：``13:02``。

    给「原话附在后面」那种行用：上下文里已经有完整日期了，每句原话再写一遍年份
    是白烧 token。
    """
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def format_span(start: float, end: float) -> str:
    """一段时间。同一天收成 ``2026-10-13 13:02–13:05``；跨天把两边都写全：
    ``2026-10-13 22:41 – 2026-10-14 01:02``。

    跨天的分隔符前后加空格、同一天的不加：一眼就分得出「同一天里的几分钟」和
    「跨了夜」。``end <= start`` 退化成单点——脏数据不该让渲染抛。
    """
    if end <= start:
        return format_when(start)
    left = datetime.fromtimestamp(start)
    right = datetime.fromtimestamp(end)
    if left.date() == right.date():
        return f"{format_when(start)}–{format_clock(end)}"
    return f"{format_when(start)} – {format_when(end)}"
