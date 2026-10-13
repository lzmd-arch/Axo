"""从用户这一句里认出「哪段时间」。

「昨天我说了什么」——这句的检索键不是词面，是**时间窗**。词法与语义通道都看不出
「昨天」意味着什么，它们只比字面，于是这句注定召不回昨天写的那几条，而用户看到
的是「她记不住事」。

**绝对时刻由系统算，不让任何模型碰。** N.E.K.O 的 ``memory/temporal.py`` 划的是同
一条边界：那边模型只许给相对偏移（``{"offset": -1, "unit": "day"}``），换算成 ISO
是系统的事。这边更简单——用户的话本身就是相对的，换算规则完全确定。

**认不出就返回 None，宁可不认不能认错。** 一个错的时间窗会把整轮召回导向错误的
那一天，症状是「她把前天的事说成昨天的」，比召不回更坏。所以：

* 不认裸数字日期（``10-13`` 可能是比分、区间、编号）；认 ``X月X日/号``。
* 不认模糊量（「最近」「前几天」「前段时间」）——它们没有边界。
* 时区一律本地：「昨天」是**这台机器**的昨天，与 L1 环境行同一个时钟。

半开区间 ``[since, until)``：``until`` 是下一段时间的零点，包含关系不会在边界上
把某一条算两遍或漏掉。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

__all__ = ["parse_time_window", "day_bounds", "week_bounds", "month_bounds"]

#: 「X月X日」。**不认 ``10-13``**：裸数字的连字符形式可能是比分、区间、编号，
#: 而认错一天比召不回更坏（见模块文档）。
_DATE_RE = re.compile(r"(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]?")

#: 「最近 N 天」「近 3 天」「过去两天」的天数。
_RECENT_RE = re.compile(r"(?:最近|近|过去的?)\s*(?P<n>\d+)\s*天")

#: 日界词。**长的在前**：「大前天」要先于「前天」命中，「昨晚」先于「昨天」。
_DAY_WORDS: tuple[tuple[str, int], ...] = (
    ("大前天", 3),
    ("前天", 2),
    ("前日", 2),
    ("昨天", 1),
    ("昨日", 1),
    ("今天", 0),
    ("今日", 0),
)

#: 同时说了「哪天」和「哪一段」的复合词。
_COMPOUNDS: tuple[tuple[str, int, int, int], ...] = (
    ("今早", 0, 5, 12),
    ("今晚", 0, 18, 24),
    ("昨晚", 1, 18, 24),
    ("昨夜", 1, 20, 24),
)

#: 一天里的时段（起、止）——**小时数**。写成小时而不是 delta，是因为它们要嵌进
#: 某一天的零点上去。
_PARTS: tuple[tuple[str, int, int], ...] = (
    ("凌晨", 0, 5),
    ("清晨", 5, 8),
    ("早上", 5, 12),
    ("早晨", 5, 12),
    ("上午", 5, 12),
    ("中午", 12, 14),
    ("下午", 12, 18),
    ("傍晚", 17, 20),
    ("晚上", 18, 24),
    ("夜里", 20, 24),
)

_WEEK_WORDS: tuple[tuple[str, int], ...] = (
    ("这个星期", 0),
    ("本星期", 0),
    ("这星期", 0),
    ("本周", 0),
    ("这周", 0),
    ("上个星期", 1),
    ("上星期", 1),
    ("上礼拜", 1),
    ("上周", 1),
)

_MONTH_WORDS: tuple[tuple[str, int], ...] = (
    ("这个月", 0),
    ("本月", 0),
    ("上个月", 1),
    ("上月", 1),
)


def day_bounds(moment: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    """某一天的 ``[零点, 次日零点)``。``offset`` 以「今天」为 0，正数往过去。"""
    day = (moment - timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
    return day, day + timedelta(days=1)


def week_bounds(moment: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    """某一周的 ``[周一零点, 下周一零点)``。``offset`` 以「本周」为 0。

    **周一为一周之始**——与 ``datetime.weekday()`` 一致，也与中文口语里「这周」
    的默认理解一致。不按周日为始：那会让「这周」在星期天说出口时含着明天。
    """
    start = (moment - timedelta(days=moment.weekday() + 7 * offset)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start, start + timedelta(days=7)


def month_bounds(moment: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    """某个月的 ``[1 日零点, 次月 1 日零点)``。``offset`` 以「本月」为 0。"""
    first = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    year, month = first.year, first.month - offset
    while month <= 0:
        month += 12
        year -= 1
    start = first.replace(year=year, month=month)
    end = start.replace(year=year + 1, month=1) if month == 12 else start.replace(month=month + 1)
    return start, end


def parse_time_window(text: str, now: datetime | None = None) -> tuple[float, float] | None:
    """把这一句里的「哪段时间」翻成 ``[since, until)`` 的 epoch 秒。

    认不出返回 ``None``。**只认得出来的时候才认**：错的一天比没有更坏。

    认法按从具体到模糊的顺序试：复合词（今晚/昨晚）→ 日 + 时段（昨天晚上）→
    X月X日 → 周 → 月 → 最近 N 天。同一个词在更具体的规则里被吃掉，就不会再被
    模糊规则截胡。
    """
    moment = now or datetime.now()
    if not text or not text.strip():
        return None

    for word, offset, start_hour, end_hour in _COMPOUNDS:
        if word in text:
            day_start, _ = day_bounds(moment, offset)
            since = day_start + timedelta(hours=start_hour)
            until = day_start + timedelta(hours=end_hour)
            return since.timestamp(), until.timestamp()

    day_offset = _first_match(text, _DAY_WORDS)
    part = _first_part(text)
    if day_offset is not None:
        day_start, day_end = day_bounds(moment, day_offset)
        if part is not None:
            start_hour, end_hour = part
            since = day_start + timedelta(hours=start_hour)
            until = day_start + timedelta(hours=end_hour)
        else:
            since, until = day_start, day_end
        return since.timestamp(), until.timestamp()

    dated = _DATE_RE.search(text)
    if dated is not None:
        window = _dated_window(moment, int(dated.group("month")), int(dated.group("day")))
        if window is not None:
            return window

    week_offset = _first_match(text, _WEEK_WORDS)
    if week_offset is not None:
        since, until = week_bounds(moment, week_offset)
        return since.timestamp(), until.timestamp()

    month_offset = _first_match(text, _MONTH_WORDS)
    if month_offset is not None:
        since, until = month_bounds(moment, month_offset)
        return since.timestamp(), until.timestamp()

    recent = _RECENT_RE.search(text)
    if recent is not None:
        days = int(recent.group("n"))
        if days <= 0:
            return None
        since, _ = day_bounds(moment, days - 1)
        return since.timestamp(), moment.timestamp()

    return None


def _first_match(text: str, words: tuple[tuple[str, int], ...]) -> int | None:
    """按**声明顺序**取第一个命中的词。长词在前是声明顺序的责任。"""
    for word, offset in words:
        if word in text:
            return offset
    return None


def _first_part(text: str) -> tuple[int, int] | None:
    for word, start_hour, end_hour in _PARTS:
        if word in text:
            return start_hour, end_hour
    return None


def _dated_window(moment: datetime, month: int, day: int) -> tuple[float, float] | None:
    """「X月X日」→ 那一天。**年份靠推断**：未来太久的日期算作去年。"""
    if not (1 <= month <= 12):
        return None
    for year in (moment.year, moment.year - 1):
        try:
            start = datetime(year, month, day)
        except ValueError:
            continue  # 2月30日这种
        if start <= moment + timedelta(days=1):
            end = start + timedelta(days=1)
            return start.timestamp(), end.timestamp()
    return None
