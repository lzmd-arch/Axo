"""把「同一件事的碎条目」并成一条带时间的记录（``memcore.merge``）。

用户要的是两件事同时成立：碎条目不再是散着的一句句，**以及**已有记忆一个字都不许
改。这两条一起把设计压成了一个形状——归并产出**新的一行**，原话逐字附在后面。

所以这里的断言分两类：判据（什么算同一件事）与形状（并出来长什么样）。判据宁可
少并：并错了原话还在、原句也还在库里，但模型读到的那条整理条目会把两件事说成一件。
"""

from __future__ import annotations

from memcore.merge import (
    MAX_GROUP,
    FragmentGroup,
    group_fragments,
    render_group,
    span_of,
)
from memcore.sdr import encode_lexical
from memcore.types import MemoryItem, MemoryKind, Tier

BASE = 1_760_000_000.0


def item(
    memory_id: str,
    content: str,
    *,
    at: float = BASE,
    session: str = "s1",
    pinned: bool = False,
    kind: MemoryKind = MemoryKind.FACT,
    meta: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=kind,
        tier=Tier.RECENT,
        content=content,
        created_at=at,
        session_id=session,
        pinned=pinned,
        lex_sdr=encode_lexical(content),
        meta=dict(meta or {}),
    )


def only(items: list[MemoryItem]) -> FragmentGroup:
    groups = group_fragments(items)
    assert len(groups) == 1, f"期望恰好一组，得到 {len(groups)} 组"
    return groups[0]


# ── 判据 ────────────────────────────────────────────────────────────────────


def test_two_short_fragments_of_one_event_become_one_group() -> None:
    """用户给的那对原话：「感冒那位回来了」「还带了杯奶茶」。

    两句没有共同的字组，靠的是「紧接着说的 + 一边是碎句」——那正是碎条目的形状。
    """
    group = only(
        [
            item("a", "感冒那位回来了。", at=BASE),
            item("b", "还带了杯奶茶。", at=BASE + 120),
        ]
    )

    assert group.memory_ids == ("a", "b")
    assert group.span == (BASE, BASE + 120)


def test_shared_vocabulary_merges_across_a_longer_gap() -> None:
    """有共同字组时间隔可以放宽到 30 分钟——它们在说同一样东西。"""
    group = only(
        [
            item("a", "路由器在书房。", at=BASE),
            item("b", "路由器在书房书架顶层。", at=BASE + 20 * 60),
        ]
    )

    assert len(group) == 2


def test_different_sessions_never_merge() -> None:
    groups = group_fragments(
        [
            item("a", "感冒那位回来了。", at=BASE, session="s1"),
            item("b", "还带了杯奶茶。", at=BASE + 60, session="s2"),
        ]
    )

    assert groups == []


def test_a_long_gap_stops_the_chain() -> None:
    groups = group_fragments(
        [
            item("a", "路由器在书房。", at=BASE),
            item("b", "路由器在书房。", at=BASE + 40 * 60),
        ]
    )

    assert groups == []


def test_unrelated_long_sentences_stay_apart_even_in_one_burst() -> None:
    """长句各自是完整事实，挨着说不代表是一件事——不并。"""
    groups = group_fragments(
        [
            item("a", "医保报销材料交到街道服务中心二楼窗口。", at=BASE),
            item("b", "阳台晾衣杆的螺丝松了得借个内扳手。", at=BASE + 60),
        ]
    )

    assert groups == []


def test_pinned_and_anchors_are_left_alone() -> None:
    """钉住条目与人格锚不参与归并——它们不是「这场里说的」。"""
    groups = group_fragments(
        [
            item("a", "路由器在书房。", at=BASE, pinned=True),
            item("b", "路由器在书房书架。", at=BASE + 60, pinned=True),
            item("c", "人格锚", at=BASE, kind=MemoryKind.PERSONA_ANCHOR),
            item("d", "人格锚又一条", at=BASE + 60, kind=MemoryKind.PERSONA_ANCHOR),
        ]
    )

    assert groups == []


def test_derived_items_are_not_merged_again() -> None:
    """归并产物不再归并——那会得到「整理条目的整理条目」。"""
    groups = group_fragments(
        [
            item("a", "感冒那位回来了。", at=BASE, meta={"derived": True, "derived_from": ["x"]}),
            item(
                "b", "还带了杯奶茶。", at=BASE + 60, meta={"derived": True, "derived_from": ["y"]}
            ),
        ]
    )

    assert groups == []


def test_a_chain_is_capped_so_it_never_becomes_a_whole_session() -> None:
    items = [
        item(f"m{index}", "还带了杯奶茶。", at=BASE + index * 60) for index in range(MAX_GROUP + 3)
    ]

    groups = group_fragments(items)

    assert all(len(group) <= MAX_GROUP for group in groups)
    assert sum(len(group) for group in groups) <= MAX_GROUP + 3


def test_a_single_item_is_not_a_group() -> None:
    assert group_fragments([item("a", "路由器在书房。", at=BASE)]) == []


# ── 形状 ────────────────────────────────────────────────────────────────────


def test_render_group_lists_the_originals_verbatim() -> None:
    """头部是并列，原话逐字照抄并各自带时刻。

    头部**不许合成因果断言**（「她回来时带了杯奶茶」是新的话）：并列句只说「这几句
    是这段时间里说的」，那永远是真的。
    """
    group = only(
        [
            item("a", "感冒那位回来了。", at=BASE),
            item("b", "还带了杯奶茶。", at=BASE + 120),
        ]
    )

    lines = render_group(group).splitlines()

    assert lines[0] == "感冒那位回来了；还带了杯奶茶。"
    assert lines[1].strip().startswith("↳ 原话：")
    assert "「感冒那位回来了。」" in lines[1]
    assert "「还带了杯奶茶。」" in lines[1]
    assert "13:02" in lines[1] or ":" in lines[1], "原话该带自己的时刻"


def test_the_head_keeps_every_original_word() -> None:
    """「不改写用词」：头部是原句去掉句读后的并列，一个字都不许添。"""
    group = only(
        [
            item("a", "妈妈对花生过敏。", at=BASE),
            item("b", "做饭不能放花生油。", at=BASE + 30),
        ]
    )

    head = render_group(group).splitlines()[0]

    assert head == "妈妈对花生过敏；做饭不能放花生油。"


def test_span_of_reads_meta_and_ignores_garbage() -> None:
    good = item("a", "路由器在书房。", meta={"span": [BASE, BASE + 60]})
    assert span_of(good) == (BASE, BASE + 60)

    for bad in (
        item("b", "没有 span。"),
        item("c", "类型不对。", meta={"span": "显然是坏的"}),
        item("d", "长度不对。", meta={"span": [1.0]}),
        item("e", "不是数字。", meta={"span": ["x", "y"]}),
        item("f", "起点是 0。", meta={"span": [0.0, 1.0]}),
    ):
        assert span_of(bad) is None
