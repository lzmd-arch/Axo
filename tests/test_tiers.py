"""层级升降（``memcore.tiers``）。

真库的那组数（141 条里 139 条停在 working、42 条权重已衰减到 0）就是这一层存在的理由。
断言分两族：**什么时候该升/降**，以及**什么时候绝对不许动**——后者更重要：让它升错的
代价是一次性的，让它降错（或动了不该动的）的代价是那条记忆开始快速淡出。
"""

from __future__ import annotations

from memcore.sdr import encode_lexical
from memcore.tiers import DEMOTE_AFTER_DAYS, PROMOTE_AT, plan_tier_moves
from memcore.types import MemoryItem, MemoryKind, Tier

NOW = 1_800_000_000.0
DAY = 86400.0


def item(
    memory_id: str,
    *,
    tier: Tier = Tier.WORKING,
    reinforces: int = 0,
    accesses: int = 0,
    last_access: float | None = None,
    pinned: bool = False,
    kind: MemoryKind = MemoryKind.FACT,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=kind,
        tier=tier,
        content="路由器在书房。",
        created_at=NOW - 40 * DAY,
        reinforce_count=reinforces,
        access_count=accesses,
        last_access_ts=NOW - 40 * DAY if last_access is None else last_access,
        pinned=pinned,
        lex_sdr=encode_lexical("路由器在书房。"),
    )


def moves(items: list[MemoryItem], *, reinforced: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    return [
        (move.memory_id, move.decision)
        for move in plan_tier_moves(items, now=NOW, reinforced=reinforced)
    ]


# ── 升 ──────────────────────────────────────────────────────────────────────


def test_enough_reinforcement_promotes_one_step() -> None:
    """回放两次都认得出自己 → 从 working 升到 recent。"""
    found = plan_tier_moves([item("m", reinforces=PROMOTE_AT)], now=NOW)

    assert [(m.memory_id, m.from_tier, m.to_tier) for m in found] == [
        ("m", Tier.WORKING, Tier.RECENT)
    ]


def test_this_rounds_reinforcement_counts_toward_the_threshold() -> None:
    """``reinforced`` 里的那条按 +1 算——SQL 层的自增还没发生。

    少了这一条，一条「上一轮被认出一回、这一轮又被认出」的记忆要等到第三轮才升，
    而它其实已经够了。
    """
    found = plan_tier_moves([item("m", reinforces=PROMOTE_AT - 1)], now=NOW, reinforced=("m",))

    assert [m.decision for m in found] == ["promote"]


def test_promotion_is_capped_at_long_term() -> None:
    """**封顶 LONGTERM**：``ARCHIVE`` 是"长期不用但确实重要"的位置，该由人说。"""
    well_used = item("m", tier=Tier.LONGTERM, reinforces=PROMOTE_AT, accesses=5)

    found = plan_tier_moves([well_used], now=NOW)

    assert found == [], "已经到顶了就不该再动"


def test_archive_is_never_touched() -> None:
    found = plan_tier_moves([item("m", tier=Tier.ARCHIVE, reinforces=99, accesses=0)], now=NOW)

    assert found == []


# ── 降 ──────────────────────────────────────────────────────────────────────


def test_a_long_term_memory_nobody_uses_falls_back_to_recent() -> None:
    found = plan_tier_moves([item("m", tier=Tier.LONGTERM, accesses=1)], now=NOW)

    assert [(m.from_tier, m.to_tier) for m in found] == [(Tier.LONGTERM, Tier.RECENT)]


def test_demotion_never_goes_down_to_working() -> None:
    """**不降到 working**：那一层的两小时半衰期是给"本次会话临时记的"设计的，
    把一个 30 天没人用的记忆扔进去等于判它死刑——它该慢慢淡，不是立刻消失。"""
    found = plan_tier_moves([item("m", tier=Tier.RECENT, accesses=0)], now=NOW)

    assert found == []


def test_a_recently_used_long_term_memory_stays() -> None:
    fresh = item("m", tier=Tier.LONGTERM, accesses=1, last_access=NOW - 3 * DAY)

    assert plan_tier_moves([fresh], now=NOW) == []


def test_a_well_used_long_term_memory_stays_even_if_stale() -> None:
    used = item("m", tier=Tier.LONGTERM, accesses=5, last_access=NOW - 90 * DAY)

    assert plan_tier_moves([used], now=NOW) == []


def test_just_inside_the_horizon_is_not_stale() -> None:
    """边界：正好 30 天前访问过——**不算**"没人用"（拿 `<` 判，不是 `<=`）。"""
    edge = item(
        "m",
        tier=Tier.LONGTERM,
        accesses=1,
        last_access=NOW - DEMOTE_AFTER_DAYS * DAY,
    )

    assert plan_tier_moves([edge], now=NOW) == []


def test_a_memory_reinforced_this_round_is_never_demoted() -> None:
    """它这一轮显然还活着，不该同时被判"没人用"。"""
    stale = item("m", tier=Tier.LONGTERM, accesses=0, last_access=NOW - 400 * DAY)

    assert plan_tier_moves([stale], now=NOW, reinforced=("m",)) == []


# ── 不许动的 ────────────────────────────────────────────────────────────────


def test_pinned_anchors_and_forgotten_ones_are_left_alone() -> None:
    pinned = item("a", reinforces=99, pinned=True)
    anchor = item("b", kind=MemoryKind.PERSONA_ANCHOR, reinforces=99)
    forgotten = item("c", tier=Tier.COLD, reinforces=99)

    assert plan_tier_moves([pinned, anchor, forgotten], now=NOW) == []


def test_one_move_per_memory_per_round() -> None:
    """一条记忆一轮最多一个移动——升和降不叠加。"""
    found = plan_tier_moves([item("m", reinforces=PROMOTE_AT, accesses=0)], now=NOW)

    assert len(found) == 1
