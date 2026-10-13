"""``memcore.temporal_window``：把「昨天」翻成一段 epoch 秒。

这一小块决定「昨天我说了什么」这句能不能命中。词法与语义都看不出「昨天」是
什么意思，于是这句注定召不回昨天写的那几条——用户看到的是「她记不住事」。

这里的断言全部指向一条纪律：**认不出就返回 None，宁可不认不能认错。** 错的一天
比没有更坏——她会把前天的事说成昨天的，而那比「我不知道」更难发现。
"""

from __future__ import annotations

from datetime import datetime

from memcore.temporal_window import (
    day_bounds,
    month_bounds,
    parse_time_window,
    week_bounds,
)

#: 2026-10-14 是星期三。
NOW = datetime(2026, 10, 14, 15, 30)


def window(text: str) -> tuple[datetime, datetime] | None:
    parsed = parse_time_window(text, NOW)
    if parsed is None:
        return None
    return datetime.fromtimestamp(parsed[0]), datetime.fromtimestamp(parsed[1])


def test_yesterday_is_one_whole_local_day() -> None:
    """「昨天我说了什么」——这一句是整个批次的起点。"""
    assert window("昨天我说了什么") == (
        datetime(2026, 10, 13, 0, 0),
        datetime(2026, 10, 14, 0, 0),
    )


def test_a_day_with_a_part_of_day_narrows_the_window() -> None:
    """「昨天晚上」不是昨天一整天——时段要收窄，否则会把中午的事也算进来。"""
    assert window("昨天晚上我说了什么") == (
        datetime(2026, 10, 13, 18, 0),
        datetime(2026, 10, 14, 0, 0),
    )


def test_compounds_carry_their_own_part_of_day() -> None:
    assert window("今晚吃什么") == (
        datetime(2026, 10, 14, 18, 0),
        datetime(2026, 10, 15, 0, 0),
    )
    assert window("今早那件事") == (
        datetime(2026, 10, 14, 5, 0),
        datetime(2026, 10, 14, 12, 0),
    )
    assert window("昨晚的提醒") == (
        datetime(2026, 10, 13, 18, 0),
        datetime(2026, 10, 14, 0, 0),
    )


def test_the_day_words_run_three_days_back() -> None:
    assert window("前天呢")[0] == datetime(2026, 10, 12, 0, 0)
    assert window("大前天呢")[0] == datetime(2026, 10, 11, 0, 0)
    assert window("今天的事")[0] == datetime(2026, 10, 14, 0, 0)


def test_a_written_out_date_is_resolved_to_a_day() -> None:
    assert window("10月13日那天") == (
        datetime(2026, 10, 13, 0, 0),
        datetime(2026, 10, 14, 0, 0),
    )
    assert window("10 月 3 号的事")[0] == datetime(2026, 10, 3, 0, 0)


def test_a_date_that_looks_like_the_future_belongs_to_last_year() -> None:
    """跨年的时候「12月25日」说的多半是去年——这是年份推断的唯一规则。"""
    new_year = datetime(2027, 1, 5, 10, 0)

    parsed = parse_time_window("12月25日的那次", new_year)

    assert parsed is not None
    start = datetime.fromtimestamp(parsed[0])
    assert (start.year, start.month, start.day) == (2026, 12, 25)


def test_weeks_and_months_run_on_calendar_bounds() -> None:
    assert window("这周发生了什么") == (
        datetime(2026, 10, 12, 0, 0),
        datetime(2026, 10, 19, 0, 0),
    )
    assert window("上周的对话") == (
        datetime(2026, 10, 5, 0, 0),
        datetime(2026, 10, 12, 0, 0),
    )
    assert window("这个月的记忆") == (
        datetime(2026, 10, 1, 0, 0),
        datetime(2026, 11, 1, 0, 0),
    )
    assert window("上个月的事")[0] == datetime(2026, 9, 1, 0, 0)


def test_weeks_start_on_monday() -> None:
    """周一起算——周日起算会让「这周」在星期天说出口时含着明天。"""
    assert week_bounds(NOW)[0].weekday() == 0
    assert week_bounds(NOW, 1)[0] == datetime(2026, 10, 5, 0, 0)


def test_the_last_n_days_counts_calendar_days_including_today() -> None:
    assert window("最近3天") == (datetime(2026, 10, 12, 0, 0), NOW)


def test_vague_quantities_are_not_windows() -> None:
    """「最近」「前几天」「前段时间」没有边界——不认，不是认成今天。"""
    assert parse_time_window("最近还好吗", NOW) is None
    assert parse_time_window("前几天那件事", NOW) is None
    assert parse_time_window("前段时间", NOW) is None


def test_a_plain_question_is_not_a_window() -> None:
    assert parse_time_window("路由器放哪了", NOW) is None
    assert parse_time_window("", NOW) is None
    assert parse_time_window("   ", NOW) is None


def test_bare_numeric_dates_are_refused() -> None:
    """10-13 可能是比分、区间、编号——认错一天比召不回更坏。"""
    assert parse_time_window("10-13 那天", NOW) is None
    assert parse_time_window("比分 3-1", NOW) is None


def test_the_window_is_half_open() -> None:
    """[since, until)：边界上那一条只算一遍，不会漏也不会重。"""
    since, until = day_bounds(NOW, 1)

    assert since.timestamp() <= datetime(2026, 10, 13, 23, 59).timestamp() < until.timestamp()
    assert not (since.timestamp() <= until.timestamp() < until.timestamp())


def test_month_bounds_cross_the_year() -> None:
    start, end = month_bounds(datetime(2026, 1, 15), 1)

    assert start == datetime(2025, 12, 1)
    assert end == datetime(2026, 1, 1)
