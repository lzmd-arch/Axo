"""混合召回与注入账本。

这个文件守三件事：

1. **可解释性不是幻觉**（:func:`test_explanation_matches_the_scores`）。账本里那
   句话里的每个数字都必须来自参与排序的同一个值。允许解释与数值脱钩，账本就
   退化成一段读起来很合理的散文。
2. **融合算子的性质**（:func:`test_fusion_never_lowers_a_candidate`）。噪
   声而不是加权平均，这条选择是实测逼出来的：用平均的话，打开语义通道会让
   纯词法命中的记忆掉到阈值以下。
3. **预算与去冗的行为**（``test_budget_*`` / ``test_redundant_*``）。它们决定
   了每一轮真正塞进上下文的是什么。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from memcore.audit import RejectReason
from memcore.ids import new_ulid
from memcore.recall import RecallConfig, _fuse, recall
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

SPACE = make_space_id("test", "fake-embed", 4)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        opened.register_space(Space(space_id=SPACE, dim=4))
        yield opened


def add(
    store: MemoryStore,
    content: str,
    *,
    tier: Tier = Tier.LONGTERM,
    weight: float = 0.8,
    pinned: bool = False,
    kind: MemoryKind = MemoryKind.FACT,
    vector: np.ndarray | None = None,
    created_at: float = 0.0,
    meta: dict | None = None,
) -> MemoryItem:
    """往库里放一条。``created_at`` 给 0 表示「用现在」（默认工厂）。

    只有要钉住时刻的测试才传它——而**时刻不许被顺手改成别的**：``MemoryItem`` 的
    ``__post_init__`` 会拿 ``created_at`` 去对齐衰减起点，所以必须在构造时就给定，
    不能构造完再改。
    """
    payload: dict = {
        "memory_id": new_ulid(),
        "kind": kind,
        "tier": tier,
        "content": content,
        "taint": Taint.USER,
        "weight": weight,
        "pinned": pinned,
        "lex_sdr": encode_lexical(content),
        "meta": dict(meta or {}),
    }
    if created_at > 0:
        payload["created_at"] = created_at
    item = MemoryItem(**payload)
    store.add(item)
    if vector is not None:
        store.set_embedding(item.memory_id, SPACE, vector)
    return item


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# ── 基本路径 ────────────────────────────────────────────────────────────────


def test_lexical_only_works_without_any_embedding(store: MemoryStore) -> None:
    """没有嵌入模型时系统必须照常可用——这是正常状态，不是降级路径。"""
    add(store, "我家的路由器放在书房书架顶层。")
    add(store, "今天午饭吃的是番茄炒蛋和米饭。")

    ledger = recall(store, query_text="路由器放在哪？", budget_tokens=500)
    assert ledger.channels == ("lexical",)
    assert len(ledger.injected) == 1
    assert "路由器" in ledger.injected[0].content


def test_semantic_channel_joins_when_a_vector_is_given(store: MemoryStore) -> None:
    add(store, "完全不相干的一句话。", vector=unit(1, 0, 0, 0))
    ledger = recall(
        store,
        query_text="随便问问",
        query_vector=unit(1, 0, 0, 0),
        space_id=SPACE,
        budget_tokens=500,
    )
    assert ledger.channels == ("lexical", "semantic")
    assert ledger.space_id == SPACE


def test_space_is_blank_in_the_ledger_without_a_vector(store: MemoryStore) -> None:
    """没走语义通道时账本里不该出现一个空间标识——那会让人以为走了。"""
    add(store, "一条记忆。")
    assert recall(store, query_text="记忆", budget_tokens=500).space_id == ""


def test_dimension_mismatch_raises_rather_than_truncating(store: MemoryStore) -> None:
    """查询向量与空间维度不符时报错，不靠截断或补齐蒙混。"""
    from memcore.errors import StoreError

    with pytest.raises(StoreError, match="维"):
        recall(
            store,
            query_text="任意",
            query_vector=np.ones(9, dtype=np.float32),
            space_id=SPACE,
            budget_tokens=100,
        )


def test_an_unregistered_space_falls_back_to_lexical(store: MemoryStore) -> None:
    """问一个**还没登记**的空间：答案是"这一轮没有语义通道"，不是"召回失败"。

    写路径与读路径在这里必须分开：``set_embedding`` 往未登记的空间写向量要当场
    拒绝（否则"记忆独立于模型"这条承诺就失去了校验点），而这里只是问了一句
    "有没有这个空间的向量"——答案没有，仅此而已。

    这条测试的由来：原来的写法是 ``space = store.registry.get(...)`` 再判 ``None``，
    而 ``SpaceRegistry.get`` 未登记时**抛异常**，于是那个分支永远不执行、真实行为
    是把 ``SpaceNotFoundError`` 抛给调用方。
    """
    add(store, "我家的路由器放在书房书架顶层。")
    ledger = recall(
        store,
        query_text="路由器放在哪",
        query_vector=unit(1, 0, 0, 0),
        space_id="test/从未见过@4",
        budget_tokens=500,
    )
    assert ledger.channels == ("lexical",)
    assert "未登记" in "".join(ledger.notes)
    assert len(ledger.injected) == 1, "词法通道照常工作"


def test_pinned_items_never_come_through_recall(store: MemoryStore) -> None:
    """人格锚走 L0 槽，不参与常规召回——否则它会跟逐字稿抢预算。"""
    anchor = add(store, "你是记忆终端，回答简洁。", pinned=True, kind=MemoryKind.PERSONA_ANCHOR)
    add(store, "记忆终端的回答风格是简洁的。")

    ledger = recall(store, query_text="记忆终端 回答 简洁", budget_tokens=500)
    assert anchor.memory_id not in ledger.injected_ids


def test_excluded_ids_are_skipped_silently(store: MemoryStore) -> None:
    """已在逐字稿里出现过的记忆不再重复注入。"""
    item = add(store, "我家的路由器放在书房书架顶层。")
    ledger = recall(store, query_text="路由器在哪", budget_tokens=500, exclude_ids=[item.memory_id])
    assert ledger.injected == ()
    assert item.memory_id not in {r.memory_id for r in ledger.rejected}


def test_cold_tier_is_rejected_with_a_reason(store: MemoryStore) -> None:
    add(store, "我家的路由器放在书房书架顶层。", tier=Tier.COLD)
    ledger = recall(store, query_text="路由器在哪", budget_tokens=500)
    assert ledger.injected == ()
    assert RejectReason.COLD in {r.reason for r in ledger.rejected}


def test_item_without_any_encoding_is_reported_as_a_data_problem(store: MemoryStore) -> None:
    """没有编码的条目要报 NO_ENCODING，而不是混进"分数不够"里。

    它指向的是向量回填断过，性质与"排不上去"完全不同——账本里必须显眼。
    """
    store.add(
        MemoryItem(
            memory_id=new_ulid(),
            kind=MemoryKind.FACT,
            tier=Tier.LONGTERM,
            content="一条还没有任何编码的记忆。",
            lex_sdr=None,
        )
    )
    ledger = recall(store, query_text="记忆", budget_tokens=500)
    assert RejectReason.NO_ENCODING in {r.reason for r in ledger.rejected}


# ── 融合算子的性质 ──────────────────────────────────────────────────────────


def test_fusion_never_lowers_a_candidate() -> None:
    """**多一条通道只会让候选更容易入选，永远不会更难。**

    这是选噪声或而不是加权平均的**全部理由**。用平均的话，一条词法已经明确命中
    的记忆（0.25）只要语义通道没有同感（0.00），就会被拉到 0.1375 掉到阈值以下
    ——"打开语义通道反而让召回变差"，而且这不是调参能修好的。
    """
    weights = {"lexical": 0.9, "semantic": 0.45}
    for lexical in (0.0, 0.05, 0.12, 0.25, 0.5, 0.9):
        for semantic in (0.0, 0.1, 0.3, 0.7):
            alone = _fuse({"lexical": lexical}, weights)
            both = _fuse({"lexical": lexical, "semantic": semantic}, weights)
            assert both >= alone - 1e-12, f"加了语义通道反而降低了得分：{alone} → {both}"


def test_fusion_is_monotone_in_each_channel() -> None:
    weights = {"lexical": 0.9, "semantic": 0.45}
    previous = -1.0
    for lexical in np.linspace(0.0, 1.0, 21):
        value = _fuse({"lexical": float(lexical), "semantic": 0.3}, weights)
        assert value >= previous
        previous = value


def test_fusion_stays_in_unit_interval() -> None:
    weights = {"lexical": 0.9, "semantic": 0.45}
    assert _fuse({"lexical": 1.0, "semantic": 1.0}, weights) <= 1.0
    assert _fuse({}, weights) == 0.0


def test_semantic_only_needs_more_evidence_than_lexical_only() -> None:
    """语义通道单独要把记忆顶过阈值，需要比词法强一倍的证据。

    依据是 ``tools/bench_space.py`` 的实测 AUC：词法 0.884、nomic-embed-text
    0.760。这条断言把那个结论固定成行为，防止有人把两个可靠度调成相等。
    """
    settings = RecallConfig()
    weights = {"lexical": settings.lexical_weight, "semantic": settings.semantic_weight}
    modulation = 0.6 + 0.4 * 0.8  # 权重 0.8 的记忆

    lexical_fused = _fuse({"lexical": 0.20}, weights)
    semantic_fused = _fuse({"semantic": 0.20}, weights)

    assert lexical_fused * modulation >= settings.min_score
    assert semantic_fused * modulation < settings.min_score


# ── 预算与去冗 ──────────────────────────────────────────────────────────────


def test_budget_rejects_what_does_not_fit(store: MemoryStore) -> None:
    long_text = "记忆终端的存储用 SQLite 加 WAL 模式。" * 12
    add(store, long_text)
    ledger = recall(store, query_text="记忆终端的存储用什么", budget_tokens=10)
    assert ledger.injected == ()
    assert RejectReason.OVER_BUDGET in {r.reason for r in ledger.rejected}


def test_budget_keeps_looking_after_one_does_not_fit(store: MemoryStore) -> None:
    """装不下一条**不结束**循环——后面可能有更短的。

    "按得分从高到低，装不下就停"会让一条 447 token 的记忆把后面所有 4 token 的
    候选一起挡在门外。这里刻意让长的**得分最高**（把整句重复二十遍——重复不增加
    token 集合，所以它的词法 SDR 与查询逐位相同，Jaccard 是 1.0），这样"继续往
    下看"才是唯一能让短条目入选的原因。
    """
    long_item = add(store, "路由器放在书房书架顶层。" * 20)
    short = add(store, "顶层的书架")
    ledger = recall(store, query_text="路由器放在书房书架顶层", budget_tokens=60)

    scores = {m.memory_id: m.scores.fused for m in ledger.injected}
    over = [r for r in ledger.rejected if r.reason is RejectReason.OVER_BUDGET]
    assert over and over[0].memory_id == long_item.memory_id, "长的没被记成装不下"
    assert over[0].score > scores[short.memory_id], (
        "前提没成立：装不下的那条并不是得分最高的那条，这个测试就测不到'继续往下看'"
    )
    assert short.memory_id in ledger.injected_ids


def test_budget_is_respected(store: MemoryStore) -> None:
    for index in range(10):
        add(store, f"记忆终端第 {index} 条关于存储与召回的记忆。")
    ledger = recall(store, query_text="记忆终端 存储 召回", budget_tokens=60)
    assert ledger.tokens_used <= 60
    assert ledger.tokens_used == sum(m.tokens for m in ledger.injected)


def test_max_items_caps_the_injection(store: MemoryStore) -> None:
    for index in range(20):
        add(store, f"记忆终端第 {index} 条互不相同的记忆，各自提到不同的细节 {index}。")
    settings = RecallConfig(max_items=3, min_score=0.0)
    ledger = recall(store, query_text="记忆终端 记忆", budget_tokens=10_000, config=settings)
    assert len(ledger.injected) <= 3


def test_near_duplicates_are_deduplicated(store: MemoryStore) -> None:
    """同一件事被说两遍时只注入一条——去冗用的是**词法 SDR 的 Jaccard**。

    用 embedding 余弦去冗的话，换一个嵌入模型就会得到不同的去冗结果，于是
    "同一段对话在不同模型下装进上下文的东西不一样"——那正是本项目要消灭的现象。
    """
    first = add(store, "我家的路由器放在书房书架顶层。")
    second = add(store, "我家的路由器放在书房书架的顶层。")  # 只差一个"的"

    settings = RecallConfig(min_score=0.0)
    ledger = recall(store, query_text="路由器放在哪里", budget_tokens=10_000, config=settings)

    injected = set(ledger.injected_ids)
    assert len(injected & {first.memory_id, second.memory_id}) == 1, (
        "两条几乎一样的记忆都被注入了——去冗没生效"
    )
    redundant = [r for r in ledger.rejected if r.reason is RejectReason.REDUNDANT]
    assert redundant, "被去掉的那条应当留下 REDUNDANT 记录"
    assert "相似度" in redundant[0].detail


def test_deduplication_does_not_use_embeddings(store: MemoryStore) -> None:
    """去掉嵌入模型之后，去冗的结果**完全一样**。"""
    add(store, "我家的路由器放在书房书架顶层。")
    add(store, "我家的路由器放在书房书架的顶层。")
    settings = RecallConfig(min_score=0.0)

    without = recall(store, query_text="路由器放在哪里", budget_tokens=10_000, config=settings)
    with_embeddings = recall(
        store,
        query_text="路由器放在哪里",
        query_vector=unit(1, 0, 0, 0),
        space_id=SPACE,
        budget_tokens=10_000,
        config=settings,
    )
    assert len(without.injected) == len(with_embeddings.injected) == 1


# ── 账本 ────────────────────────────────────────────────────────────────────


def test_explanation_matches_the_scores(store: MemoryStore) -> None:
    """**解释里的每个数字都必须与参与排序的值一致。**

    这是"可解释性可能是幻觉"那条风险的验收点：`explain` 是从 ChannelScores
    现算的模板文本，不是独立字段。一旦有人把它改成可传入的字符串，这条会失败。
    """
    add(store, "我在做一个叫 Axo的项目，记忆必须独立于所调用的模型。")
    ledger = recall(store, query_text="记忆终端为什么要独立于模型？", budget_tokens=500)

    assert ledger.injected, "这条样本应当命中"
    for memory in ledger.injected:
        assert memory.explain
        # 每个被列出的通道，它的数值都原样出现在解释里
        for _, value in memory.scores.active():
            assert f"{value:.2f}" in memory.explain, (memory.explain, value)
        assert str(memory.tokens) in memory.explain
        assert f"{memory.weight:.2f}" in memory.explain


def test_explanation_format_matches_the_acceptance_criterion(store: MemoryStore) -> None:
    """计划书 §3.4 的验收原话是「因语义 0.87 被注入，184 token」。"""
    add(store, "我在做一个叫 Axo的项目，记忆必须独立于所调用的模型。")
    ledger = recall(store, query_text="记忆终端为什么要独立于模型？", budget_tokens=500)
    explain = ledger.injected[0].explain
    assert explain.startswith("因词法 ")
    assert " 被注入，" in explain
    assert explain.endswith(" token；权重 0.80")


def test_explanation_is_unavailable_when_attribution_is_all_zero() -> None:
    """归因全 0 时**如实显示「归因不可用」**，不编一个理由。"""
    from memcore.audit import ChannelScores, RecalledMemory

    memory = RecalledMemory(
        memory_id="01J",
        content="一条被注入了但没有归因的记忆。",
        kind="fact",
        tier="longterm",
        weight=0.8,
        tokens=20,
        scores=ChannelScores(),
    )
    assert "归因不可用" in memory.explain

    pinned = RecalledMemory(
        memory_id="01J",
        content="人格锚。",
        kind="persona_anchor",
        tier="longterm",
        weight=1.0,
        tokens=20,
        scores=ChannelScores(),
        pinned=True,
    )
    assert "钉住条目" in pinned.explain


def test_ledger_lists_rejections_with_reasons(store: MemoryStore) -> None:
    add(store, "我家的路由器放在书房书架顶层。")
    add(store, "今天午饭吃的是番茄炒蛋和米饭。")
    ledger = recall(store, query_text="路由器在哪", budget_tokens=500)

    assert ledger.injected
    rejected = {r.memory_id: r.reason for r in ledger.rejected}
    assert RejectReason.BELOW_THRESHOLD in rejected.values()
    for record in ledger.rejected:
        assert record.explain
        assert record.content_preview


def test_ledger_summary_is_one_line(store: MemoryStore) -> None:
    add(store, "一条记忆。")
    ledger = recall(store, query_text="记忆", budget_tokens=500)
    assert "\n" not in ledger.summary()
    assert "扫过 1 条" in ledger.summary()


def test_query_preview_is_collapsed_and_truncated(store: MemoryStore) -> None:
    add(store, "一条记忆。")
    ledger = recall(store, query_text="第一行\n第二行\t" + "很长" * 60, budget_tokens=500)
    assert "\n" not in ledger.query_preview
    assert ledger.query_preview.endswith("…")
    assert len(ledger.query_preview) <= 41


# ── 写回 ────────────────────────────────────────────────────────────────────


def test_injected_items_get_their_access_counted(store: MemoryStore) -> None:
    item = add(store, "我家的路由器放在书房书架顶层。", weight=0.8)
    recall(store, query_text="路由器在哪", budget_tokens=500)

    reloaded = store.get(item.memory_id)
    assert reloaded is not None
    assert reloaded.access_count == 1


def test_rejected_items_are_not_touched(store: MemoryStore) -> None:
    """淘汰的候选连权重都不必结算——它们的值不影响任何结果。"""
    item = add(store, "今天午饭吃的是番茄炒蛋和米饭。")
    recall(store, query_text="路由器在哪", budget_tokens=500)

    reloaded = store.get(item.memory_id)
    assert reloaded is not None
    assert reloaded.access_count == 0


def test_recall_is_read_only_apart_from_access_counting(store: MemoryStore) -> None:
    item = add(store, "我家的路由器放在书房书架顶层。")
    before = store.get(item.memory_id)
    recall(store, query_text="路由器在哪", budget_tokens=500)
    after = store.get(item.memory_id)

    assert before is not None and after is not None
    assert after.content == before.content
    assert after.weight == pytest.approx(before.weight)
    assert store.count() == 1


def test_empty_library_returns_an_empty_ledger(store: MemoryStore) -> None:
    ledger = recall(store, query_text="什么都没有", budget_tokens=500)
    assert ledger.injected == ()
    assert ledger.scanned == 0
    assert ledger.summary()


def test_query_without_encodable_characters_still_works(store: MemoryStore) -> None:
    """全标点的输入没有词法通道，但**不能报错**——用户可能只是发了个表情。"""
    add(store, "一条记忆。")
    ledger = recall(store, query_text="？？？", budget_tokens=500)
    assert ledger.channels == ()
    assert "不含可编码字符" in " ".join(ledger.notes)


WHEN = datetime(2026, 10, 13, 22, 41).timestamp()


def test_injected_items_carry_their_time(store: MemoryStore) -> None:
    """召回要把 ``created_at`` 带出来——装配层只能渲染它拿到的东西。

    时间戳不是装饰：没有它，「昨天我说了什么」在召回这一层就已经答不了了。
    """
    add(store, "感冒那位回来了。", created_at=WHEN)

    ledger = recall(
        store,
        query_text="感冒那位",
        budget_tokens=500,
        config=RecallConfig(min_score=0.0),
    )

    assert ledger.injected, "这一条该被召回"
    assert ledger.injected[0].created_at == WHEN
    assert ledger.injected[0].span is None


def test_a_merged_entry_carries_its_span(store: MemoryStore) -> None:
    """归并产物的区间从 ``meta["span"]`` 来；原句没有区间，记 ``None``。"""
    end = WHEN + 180
    add(
        store,
        "感冒那位回来了，还带了杯奶茶。",
        created_at=WHEN,
        meta={"derived": True, "derived_from": ["a", "b"], "span": [WHEN, end]},
    )

    ledger = recall(
        store,
        query_text="感冒那位",
        budget_tokens=500,
        config=RecallConfig(min_score=0.0),
    )

    assert ledger.injected[0].span == (WHEN, end)


def test_a_dirty_span_is_ignored_rather_than_failing_the_round(store: MemoryStore) -> None:
    """``meta`` 是自由格式 JSON，脏数据不该让整轮召回失败。"""
    add(store, "路由器在书房。", meta={"span": "显然是坏的"})

    ledger = recall(
        store, query_text="路由器", budget_tokens=500, config=RecallConfig(min_score=0.0)
    )

    assert ledger.injected, "召回本身不受影响"
    assert ledger.injected[0].span is None


NOW = datetime(2026, 10, 14, 15, 30).timestamp()
YESTERDAY = datetime(2026, 10, 13, 22, 41).timestamp()
LAST_WEEK = datetime(2026, 10, 6, 9, 0).timestamp()


def test_time_window_hits_outrank_the_rest(store: MemoryStore) -> None:
    """「昨天我说了什么」——窗内那条必须排在窗外那条前面。

    这是整批的验收点。以前两条打成平手（词面都撞不上「昨天」），于是召回按权重
    随缘排，用户看到的是「她记不住事」。
    """
    outside = add(store, "路由器在书房书架顶层。", created_at=LAST_WEEK)
    inside = add(store, "感冒那位回来了。", created_at=YESTERDAY)

    ledger = recall(
        store,
        query_text="昨天我说了什么",
        budget_tokens=10_000,
        config=RecallConfig(min_score=0.0, max_items=2),
        now=NOW,
    )

    assert list(ledger.injected_ids) == [inside.memory_id, outside.memory_id]


def test_the_ledger_says_which_window_it_used(store: MemoryStore) -> None:
    """账本要能回答「你说的昨天是哪一天」——两行的时间戳与它必须可比。"""
    add(store, "感冒那位回来了。", created_at=YESTERDAY)

    ledger = recall(
        store,
        query_text="昨天我说了什么",
        budget_tokens=10_000,
        config=RecallConfig(min_score=0.0),
        now=NOW,
    )

    assert "temporal" in ledger.channels
    window_notes = [note for note in ledger.notes if note.startswith("时间窗")]
    assert window_notes, f"账本没说用了哪个时间窗：{ledger.notes}"
    assert "2026-10-13 00:00" in window_notes[0]
    assert "窗内 1 条" in window_notes[0]


def test_a_window_hit_survives_even_without_an_encoding(store: MemoryStore) -> None:
    """窗内命中不因缺编码被挡——用户问的是那段时间里有什么。

    缺编码是数据问题，但它在任何不带时间词的轮次里都会以 NO_ENCODING 冒出来，
    不会因为这一条放行就永远瞒住。
    """
    bare = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content="那天说了路由器的事。",
        taint=Taint.USER,
        created_at=YESTERDAY,
    )
    store.add(bare)

    ledger = recall(
        store,
        query_text="昨天我说了什么",
        budget_tokens=10_000,
        config=RecallConfig(min_score=0.0),
        now=NOW,
    )

    assert list(ledger.injected_ids) == [bare.memory_id]
    assert ledger.injected[0].scores.temporal == 1.0


def test_without_a_time_phrase_there_is_no_temporal_channel(store: MemoryStore) -> None:
    """没认出时间就没有这个通道——不是「都算命中」。"""
    add(store, "路由器在书房。", created_at=YESTERDAY)

    ledger = recall(
        store,
        query_text="路由器在哪",
        budget_tokens=10_000,
        config=RecallConfig(min_score=0.0),
        now=NOW,
    )

    assert "temporal" not in ledger.channels
    assert all(item.scores.temporal == 0.0 for item in ledger.injected)
