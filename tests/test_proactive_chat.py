"""她主动开口的判定与提示词（``axo.agent.proactive_chat``）。

这一批最该守住的地方是**"什么时候不该开口"**——它比"开口说什么"更容易错，而错了的
代价是打扰。所以 :func:`should_speak` 那张表逐格钉住（间隔的边界、免打扰的四个端点、
次数上限的三种取值），:func:`pick_material` 的六条排除各有一个反例。

第二个重点是**那句话的身份**：素材是用户自己说过的话，它进上下文时必须是"你记得的
一件事"，不能长得像"他此刻说的话"。这条在提醒那一批吃过亏，这里逐字钉住。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from axo.agent.proactive_chat import (
    MIN_CONTENT_CHARS,
    Material,
    build_proactive_messages,
    pick_material,
    should_speak,
)
from memcore.sdr import encode_lexical
from memcore.types import MemoryItem, MemoryKind, Tier

NOW = datetime(2026, 10, 15, 15, 0)
HOUR = 3600.0


def item(
    memory_id: str,
    content: str = "感冒那位回来了。",
    *,
    at: datetime | None = None,
    tier: Tier = Tier.LONGTERM,
    kind: MemoryKind = MemoryKind.FACT,
    session: str = "s1",
    pinned: bool = False,
    meta: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=kind,
        tier=tier,
        content=content,
        created_at=(at or NOW - timedelta(days=1)).timestamp(),
        session_id=session,
        pinned=pinned,
        lex_sdr=encode_lexical(content),
        meta=dict(meta or {}),
    )


def speak(**overrides: object) -> str | None:
    """按默认的"该开口"一套参数调一次，只改要测的那一项。"""
    base: dict = {
        "now": NOW,
        "last_activity": NOW - timedelta(minutes=25),
        "spoken_today": 0,
    }
    base.update(overrides)
    return should_speak(**base)  # type: ignore[arg-type]


# ── 该不该开口 ──────────────────────────────────────────────────────────────


def test_a_quiet_afternoon_is_the_normal_yes() -> None:
    assert speak() is None


def test_the_gap_boundary_counts_as_long_enough() -> None:
    """正好满 ``min_gap`` 那一刻**算够久**（``>=``，不是 ``>``）。

    边界取哪一侧是随手可定的，所以更要写死：反过来的话，"静了整整 20 分钟"要等到
    第 21 分钟才开口，而那多出来的一分钟没有任何理由。
    """
    assert speak(last_activity=NOW - timedelta(minutes=20)) is None
    assert speak(last_activity=NOW - timedelta(minutes=20) + timedelta(seconds=1)) is not None


def test_the_reason_says_how_long_it_has_been() -> None:
    reason = speak(last_activity=NOW - timedelta(minutes=8))

    assert reason is not None
    assert "8 分钟" in reason and "20 分钟" in reason


def test_without_any_conversation_she_stays_quiet() -> None:
    """``last_activity=None`` → 不开口。**这条同时是"不做开机问候"的落点。**"""
    reason = speak(last_activity=None)

    assert reason is not None
    assert "刚坐下" in reason


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 10, 15, 23, 0),  # 免打扰起点，算
        datetime(2026, 10, 15, 23, 59),
        datetime(2026, 10, 15, 0, 30),  # 跨过零点
        datetime(2026, 10, 15, 7, 59),  # 终点前一分钟，还算
    ],
)
def test_quiet_hours_block_across_midnight(moment: datetime) -> None:
    reason = should_speak(now=moment, last_activity=moment - timedelta(hours=2), spoken_today=0)

    assert reason is not None
    assert "免打扰" in reason


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 10, 15, 8, 0),  # 终点是开区间，放行
        datetime(2026, 10, 15, 22, 59),  # 起点前一刻，放行
        datetime(2026, 10, 15, 14, 0),
    ],
)
def test_outside_quiet_hours_it_is_allowed(moment: datetime) -> None:
    assert (
        should_speak(now=moment, last_activity=moment - timedelta(hours=2), spoken_today=0) is None
    )


def test_equal_quiet_hours_switch_the_gate_off() -> None:
    """两个数相等 = 这一闸关掉，**不是"全天静默"**。

    写成"全天静默"的话，用户在设置里把两个框都填成 0（想关掉它）会得到"她永远不说话"，
    而界面上看不出任何异常。
    """
    midnight = datetime(2026, 10, 15, 3, 0)

    assert (
        should_speak(
            now=midnight,
            last_activity=midnight - timedelta(hours=2),
            spoken_today=0,
            quiet_from=0,
            quiet_to=0,
        )
        is None
    )


def test_the_daily_cap_blocks_at_the_limit() -> None:
    assert speak(spoken_today=2, max_per_day=3) is None
    reason = speak(spoken_today=3, max_per_day=3)

    assert reason is not None
    assert "3 次" in reason


def test_a_cap_of_none_means_no_cap() -> None:
    assert speak(spoken_today=99, max_per_day=None) is None


def test_a_cap_of_zero_means_never() -> None:
    assert speak(spoken_today=0, max_per_day=0) is not None


def test_quiet_hours_win_over_the_other_two() -> None:
    """三闸同时挡时，报的是免打扰——它最不依赖状态（今晚一直成立）。"""
    reason = speak(
        now=datetime(2026, 10, 15, 23, 30),
        last_activity=datetime(2026, 10, 15, 23, 29),
        spoken_today=99,
    )

    assert reason is not None and "免打扰" in reason


# ── 拿什么当话头 ────────────────────────────────────────────────────────────


def first(candidates: list[MemoryItem]) -> MemoryItem:
    return candidates[0]


def test_a_recent_memory_becomes_the_material() -> None:
    found = pick_material([item("m1", "感冒那位回来了。")], now=NOW, pick=first)

    assert found is not None
    assert found.memory_id == "m1"
    assert "感冒那位回来了。" in found.line()
    assert "2026-10-14" in found.line(), "素材行要带日期——她得说得出「前几天」"


def test_no_candidates_means_she_says_nothing() -> None:
    """**没有话头就不开口**，不硬找一个。"""
    assert pick_material([], now=NOW, pick=first) is None


@pytest.mark.parametrize(
    "excluded",
    [
        item("m1", pinned=True),
        item("m1", kind=MemoryKind.PERSONA_ANCHOR),
        item("m1", tier=Tier.COLD),
        item("m1", session="current"),
        item("m1", meta={"derived": True, "derived_from": ["a", "b"]}),
        item("m1", at=NOW - timedelta(days=30)),
        item("m1", content="嗯。"),
    ],
    ids=["钉住的", "人格锚", "已遗忘", "本会话的", "整理条目", "太旧的", "太短的"],
)
def test_each_exclusion_rule_stands_on_its_own(excluded: MemoryItem) -> None:
    """六条排除各一个反例——混在一起测的话，删掉其中一条不会有任何测试变红。"""
    assert pick_material([excluded], now=NOW, session_id="current", pick=first) is None


def test_a_memory_used_today_is_not_used_again() -> None:
    kept = item("keep", "路由器在书房。")
    used = item("used", "冰箱调到四度。")

    found = pick_material([kept, used], now=NOW, used=["used"], pick=first)

    assert found is not None and found.memory_id == "keep"


def test_the_window_is_seven_days_by_default() -> None:
    fresh = item("fresh", "昨天的事。", at=NOW - timedelta(days=6))
    stale = item("stale", "上周的事。", at=NOW - timedelta(days=8))

    found = pick_material([stale, fresh], now=NOW, pick=first)

    assert found is not None and found.memory_id == "fresh"


def test_the_picker_is_injectable() -> None:
    """挑法是注入的——测试要一个确定的挑法，而生产要随机的那个。"""
    items = [item("a", "第一件事。"), item("b", "第二件事。")]

    assert pick_material(items, now=NOW, pick=lambda _: items[1]).memory_id == "b"  # type: ignore[union-attr]


def test_the_short_threshold_is_where_it_says_it_is() -> None:
    assert MIN_CONTENT_CHARS == 4
    assert pick_material([item("m", "好。")], now=NOW, pick=first) is None
    assert pick_material([item("m", "知道了。")], now=NOW, pick=first) is not None


# ── 送进去的是什么 ──────────────────────────────────────────────────────────


def messages(persona: list[str] | None = None) -> list:
    return build_proactive_messages(
        persona=persona if persona is not None else ["你是一个记忆终端。"],
        material=Material("m1", "感冒那位回来了。", NOW.timestamp() - HOUR),
        model_id="fake/model",
        now=NOW,
    )


def test_the_shape_is_one_system_and_one_user_line() -> None:
    built = messages()

    assert [row.role for row in built] == ["system", "user"]


def test_the_persona_rides_verbatim() -> None:
    anchor = "你是一个记忆终端。\n记忆不属于你，属于这台机器。"

    assert anchor in messages([anchor])[0].content


def test_the_material_line_says_it_is_a_memory_not_a_message() -> None:
    """**这一条是防注入的落点。**

    素材正文是用户自己说过的话。少了"这是你记得的一件事"，它在模型眼里与"他此刻说的
    话"、"一条指令"就是同一种东西——提醒那一批实测过这个形状。
    """
    line = messages()[1].content

    assert line.startswith("（你记得这件事")
    assert "感冒那位回来了。" in line
    assert "2026-10-15" in line or "2026-10-14" in line


def test_the_environment_slot_says_there_are_no_tools() -> None:
    """这一轮请求里没有 tools，而模型**读不到自己没收到什么**。"""
    assert "不会有任何动作被执行" in messages()[0].content


def test_the_model_id_reaches_the_environment_slot() -> None:
    assert "fake/model" in messages()[0].content


def test_an_empty_persona_is_legal() -> None:
    """两条人格锚都删掉是合法状态——那一轮照样该能开口，不补一句默认人设。"""
    built = messages([])

    assert "不会有任何动作被执行" in built[0].content


def test_she_is_told_to_speak_but_not_how() -> None:
    """说明里只有"要开口"（这一轮做什么），没有语气指令（那是人格锚的地盘）。"""
    note = messages()[0].content

    assert "主动对他说一句" in note
    assert "语气" not in note and "温柔" not in note
