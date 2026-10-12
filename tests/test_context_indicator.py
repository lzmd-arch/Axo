"""上下文指示灯的判定与排版。

全是纯算术，所以这个文件短——但它守的是**主窗口顶栏与侧栏账本同源**这件事的
下半截：指示灯读的是 ``AssemblyLedger.total_tokens / budget.usable / over_budget``，
与侧栏逐行账本读的是同一份。灯上写 90%、账本写 40% 的界面无从排查，而且用户会
先信灯。

三条边界值得单独说：``over_budget`` 优先于比值、``0.85`` 是**含**的、比值可以
大于 1 而文字如实显示（夹到 100% 的是进度条，不是这行字）。
"""

from __future__ import annotations

import pytest

from axo.session.indicator import (
    WARN_RATIO,
    ContextState,
    context_state,
    format_context,
)


def test_no_assembly_yet_is_empty_not_ok() -> None:
    """**「额度用了 0%」与「没有额度」是两件事。**

    前者会让人以为一次请求正在顺利进行——而实际上这一轮还没发生（没有会话，
    或者首次装配还没回来）。
    """
    assert context_state(total_tokens=0, usable=8900, over_budget=0) is ContextState.EMPTY


def test_ordinary_use_is_ok() -> None:
    assert context_state(total_tokens=3200, usable=8900, over_budget=0) is ContextState.OK


def test_over_budget_wins_over_the_ratio() -> None:
    """**判定顺序不能换：先看 ``over_budget``，再看比值。**

    超出一点点时比值可能仍然落在警示区里（缺了这个判断就会显示黄），而超了就该是红
    ——它很可能已经被服务端截掉了最早的消息，丢的正是人格锚所在的位置。
    """
    assert context_state(total_tokens=8900, usable=8900, over_budget=1) is ContextState.OVER
    assert context_state(total_tokens=101, usable=100, over_budget=1) is ContextState.OVER
    # 没有超出时，同样的比值只是黄。
    assert context_state(total_tokens=8900, usable=8900, over_budget=0) is ContextState.WARN


def test_the_warn_threshold_is_inclusive() -> None:
    """``>= WARN_RATIO``。0.849 绿、0.85 黄——边界上偏"提前告警"的一侧。"""
    usable = 10000
    below = int(usable * WARN_RATIO) - 1
    at = int(usable * WARN_RATIO)

    assert context_state(total_tokens=below, usable=usable, over_budget=0) is ContextState.OK
    assert context_state(total_tokens=at, usable=usable, over_budget=0) is ContextState.WARN


def test_uncomputable_budget_is_over_not_ok() -> None:
    """额度算不出来（窗口与预留输出太接近）时判 OVER。

    这个配置下装配器基本发不出东西，说"一切正常"是错的。
    """
    assert context_state(total_tokens=500, usable=0, over_budget=0) is ContextState.OVER


@pytest.mark.parametrize(
    ("total", "usable", "expected"),
    [
        (0, 0, "上下文 —"),
        (-1, 8900, "上下文 —"),
        (999, 8900, "上下文 999 / 8.9k"),
        (3200, 8900, "上下文 3.2k / 8.9k"),
        (12000, 8900, "上下文 12.0k / 8.9k"),
    ],
)
def test_format_context(total: int, usable: int, expected: str) -> None:
    assert format_context(total_tokens=total, usable=usable) == expected


def test_format_does_not_clamp_an_over_budget_ratio() -> None:
    """**如实显示，不夹紧到 100%。**

    夹到 100% 的是进度条（画法的限制），不是这行字。两处都夹的话，「超了多少」
    就没有地方看了。
    """
    text = format_context(total_tokens=12000, usable=8900)

    assert "12.0k" in text
    assert "8.9k" in text


def test_the_four_states_are_distinct_strings() -> None:
    """状态值是给样式表用的键（``over`` → 红），不能互相撞上。"""
    assert {state.value for state in ContextState} == {"empty", "ok", "warn", "over"}
    assert str(ContextState.WARN) == "warn"
