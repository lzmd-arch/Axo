"""记忆行上那个时间的排版（``memcore.timefmt``）。

只有几个格式函数，但它们决定的是**模型能不能形成时间观念**：两行记忆的时间长得
不一样，模型比先后就会出错；而一个编出来的时间比没有时间更坏——它会被当成事实
读。所以这里的断言全是关于"形状"的：同一天怎么写、跨天怎么写、没时间怎么写。
"""

from __future__ import annotations

from datetime import datetime

from memcore.timefmt import format_clock, format_span, format_when, has_time

WHEN = datetime(2026, 10, 13, 22, 41).timestamp()
SAME_DAY_END = datetime(2026, 10, 13, 13, 5).timestamp()
SAME_DAY_START = datetime(2026, 10, 13, 13, 2).timestamp()
NEXT_DAY = datetime(2026, 10, 14, 1, 2).timestamp()


def test_a_single_moment_is_date_then_clock() -> None:
    """``2026-10-13 22:41``——与 L1 环境行的 ``%Y-%m-%d %H:%M`` 同族。

    两处格式必须可比：环境行说"现在是 2026-10-13 星期二 22:41"，记忆行说这件事是
    "2026-10-13 22:41"，模型直接比字符串的前半段就够判断是不是同一天。
    """
    assert format_when(WHEN) == "2026-10-13 22:41"


def test_the_clock_alone_is_for_quotes_inside_a_dated_block() -> None:
    """``13:02``：原话附在后面那几行用。完整日期已经有了，再写一遍是白烧 token。"""
    assert format_clock(SAME_DAY_START) == "13:02"


def test_a_span_inside_one_day_keeps_the_date_once() -> None:
    """同一天收成 ``2026-10-13 13:02-13:05`` 那样的紧凑形状。"""
    assert format_span(SAME_DAY_START, SAME_DAY_END) == f"{format_when(SAME_DAY_START)}–13:05"


def test_a_span_across_days_writes_both_dates_in_full() -> None:
    """跨天两边都写全。只写一次日期会让"跨了夜"这件事看不出来。"""
    text = format_span(WHEN, NEXT_DAY)
    assert text == f"{format_when(WHEN)} – {format_when(NEXT_DAY)}"
    assert "2026-10-13" in text and "2026-10-14" in text


def test_a_backwards_span_degenerates_to_one_point() -> None:
    """``end <= start`` 是脏数据，不该让渲染抛——退化成单点。"""
    assert format_span(SAME_DAY_END, SAME_DAY_START) == format_when(SAME_DAY_END)


def test_zero_is_never_a_time() -> None:
    """``<= 0`` 一律是「不知道」。

    ``created_at = 0`` 渲染出来的 1970-01-01 会被模型当成"很久很久以前"，而真实
    情况是"这条没有时间"。两者对模型是完全不同的事实。
    """
    assert has_time(WHEN) is True
    assert has_time(0.0) is False
    assert has_time(-1.0) is False
