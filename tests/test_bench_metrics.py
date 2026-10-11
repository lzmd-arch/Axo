"""四项指标的边界情形，以及一条必须永远成立的不变量。

指标函数是纯函数，所以这里全部**手搓账本**——测"``K`` 大于注入条数怎么办"不该
需要建一个库、跑一次召回。这也是 :class:`QueryOutcome` 被设计成一个独立 dataclass
而不是"召回账本的包装"的理由。

最后一条 ``test_fused_does_not_depend_on_selection_knobs`` 是另一种测试：它不测
指标，它测**指标的前提**。AUC 的口径依赖"``fused`` 与阈值无关"，而那个性质一旦
被破坏，AUC 不会报错，只会静默地变成另一个量。
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import pytest

from biosnn_memterm.bench import (
    QueryOutcome,
    auc,
    census_config,
    census_scores,
    false_injection_rate,
    hit_at_k,
    mrr,
    mrr_answered,
    no_answer_injections,
    recall_at_k,
    run,
    summarize,
)
from biosnn_memterm.bench.goldset import GoldMemory, GoldQuery, GoldSet
from biosnn_memterm.bench.metrics import K_VALUES
from memcore.graph import GraphIndex
from memcore.recall import RecallConfig, recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

NOW = 1700000000.0


def make(
    query_id: str,
    injected: tuple[str, ...],
    *,
    expect: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    never: tuple[str, ...] = (),
    scores: dict[str, float] | None = None,
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        category="test",
        injected=injected,
        expect=expect,
        optional=optional,
        never=never,
        scores=scores,
    )


def is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


# ── hit@K ───────────────────────────────────────────────────────────────────


def test_hit_at_k_leaves_no_answer_out_of_the_denominator() -> None:
    """``expect`` 为空查询不进分母。

    三个查询：一个命中、一个没命中、一个库里根本没有答案（却被注入了 8 条）。
    hit@1 只该看前两个 → 0.5。把第三个算成"没命中"会得到 0.333——那是把
    "我们没有编造"记成一笔失败。
    """
    outcomes = [
        make("q-1", ("m-1",), expect=("m-1",)),
        make("q-2", ("m-2",), expect=("m-9",)),
        make("q-3", ("m-3", "m-4"), expect=()),
    ]
    assert hit_at_k(outcomes, 1) == pytest.approx(0.5)
    assert mrr(outcomes) == pytest.approx((1.0 + 0.0) / 2)


def test_hit_at_k_past_the_end_is_not_an_error() -> None:
    """``K`` 大于注入条数时按实际长度算。"""
    outcomes = [make("q-1", ("m-1",), expect=("m-1",))]
    assert hit_at_k(outcomes, 8) == 1.0
    assert hit_at_k(outcomes, 99) == 1.0


def test_hit_at_k_uses_the_injection_order() -> None:
    """命中的判定看**顺序**：第 3 位的期望条目不算 hit@1。"""
    outcomes = [make("q-1", ("m-x", "m-y", "m-1"), expect=("m-1",))]
    assert hit_at_k(outcomes, 1) == 0.0
    assert hit_at_k(outcomes, 3) == 1.0
    assert recall_at_k(outcomes, 1) == 0.0


def test_hit_and_recall_differ_when_a_query_has_two_truths() -> None:
    """macro 与 micro 的分母不同：一条查询有两个期望条时两个数会分开。"""
    outcomes = [
        make("q-1", ("m-1", "m-z"), expect=("m-1", "m-2")),
        make("q-2", ("m-3",), expect=("m-3",)),
    ]
    assert hit_at_k(outcomes, 2) == 1.0  # 两条查询都至少命中一个
    assert recall_at_k(outcomes, 2) == pytest.approx(2 / 3)  # 三个期望条里找回两个


# ── MRR ─────────────────────────────────────────────────────────────────────


def test_mrr_counts_a_miss_as_zero() -> None:
    outcomes = [make("q-1", ("m-1",), expect=("m-1",)), make("q-2", ("m-2",), expect=("m-9",))]
    assert mrr(outcomes) == pytest.approx(0.5)
    assert mrr_answered(outcomes) == pytest.approx(1.0)


def test_mrr_answered_is_nan_when_nothing_hit() -> None:
    """一条都没命中时是 ``nan``，不是 0。

    0 的意思是"排得完全靠后"，而这里根本没有可排的东西。把两者混起来，一次语料
    事故就会看起来像一次性能崩溃。
    """
    outcomes = [make("q-1", ("m-2",), expect=("m-9",))]
    assert is_nan(mrr_answered(outcomes))
    assert mrr(outcomes) == 0.0


# ── 误注入率 ────────────────────────────────────────────────────────────────


def test_false_injection_rate_is_micro_not_macro() -> None:
    """按**注入条数**汇总，不是按查询平均。

    q-1 注入 1 条且全错、q-2 注入 4 条且全对 → micro = 1/5 = 0.2，而按查询平均
    会得到 0.5。两者在 token 上的代价差了八倍，而 macro 把这两条查询看成一样重。
    """
    outcomes = [
        make("q-1", ("m-x",), expect=("m-1",)),
        make("q-2", ("m-2", "m-3", "m-4", "m-5"), expect=("m-2", "m-3", "m-4", "m-5")),
    ]
    assert false_injection_rate(outcomes) == pytest.approx(1 / 5)


def test_false_injection_rate_treats_optional_as_relevant() -> None:
    """``optional`` 不算错。判它错会把"去冗做得好"变成一个负分项。"""
    outcomes = [make("q-1", ("m-1", "m-2"), expect=("m-1",), optional=("m-2",))]
    assert false_injection_rate(outcomes) == 0.0


def test_no_answer_injections_counts_every_injection_as_wrong() -> None:
    """``expect`` 为空时任何注入都是错的，而且**不设上限**。"""
    outcomes = [
        make("q-1", ("m-1", "m-2", "m-3", "m-4", "m-5", "m-6", "m-7", "m-8"), expect=()),
        make("q-2", (), expect=()),
        make("q-3", ("m-1",), expect=("m-1",)),  # 有答案的查询不进这个分母
    ]
    assert no_answer_injections(outcomes) == pytest.approx(4.0)


# ── AUC ─────────────────────────────────────────────────────────────────────


def test_auc_counts_a_tie_as_one_half() -> None:
    """并列按 ½ 计。

    正例 0.5 对负例 {0.5, 0.4}：赢 1、平 1 → (1 + 0.5) / 2 = 0.75。把并列算成
    输会得到 0.5，那正好等于"瞎猜"——一个恒定的打分函数会被误报成没有区分度。
    """
    outcomes = [make("q-1", (), expect=("m-p",), scores={"m-p": 0.5, "m-n1": 0.5, "m-n2": 0.4})]
    assert auc(outcomes) == pytest.approx(0.75)


def test_auc_is_nan_when_either_side_is_empty() -> None:
    only_positive = [make("q-1", (), expect=("m-p",), scores={"m-p": 0.5})]
    assert is_nan(auc(only_positive))
    no_scores = [make("q-1", (), expect=("m-p",))]
    assert is_nan(auc(no_scores))


def test_auc_excludes_optional_from_the_negatives() -> None:
    """负集是"库里不属于 ``expect ∪ optional`` 的条"。"""
    outcomes = [
        make(
            "q-1",
            (),
            expect=("m-p",),
            optional=("m-o",),
            scores={"m-p": 0.9, "m-o": 0.1, "m-n": 0.8},
        )
    ]
    # 负集只有 m-n（0.8）：正例 0.9 赢 → 1.0。若 m-o 被当成负例，会得到 0.5。
    assert auc(outcomes) == pytest.approx(1.0)


# ── nan 的边界 ──────────────────────────────────────────────────────────────


def test_empty_outcomes_give_nan_not_zero() -> None:
    """空集会返回 ``nan``。**不是 0**——0 是"排得完全相反"。"""
    for value in (
        hit_at_k([], 1),
        mrr([]),
        false_injection_rate([]),
        no_answer_injections([]),
        auc([]),
    ):
        assert is_nan(value)


def test_summarize_covers_every_headline_metric() -> None:
    """汇总字典里四项 headline 与每一个 K 都在。"""
    summary = summarize([make("q-1", ("m-1",), expect=("m-1",))])
    for k in K_VALUES:
        assert f"hit@{k}" in summary
        assert f"recall@{k}" in summary
    for name in ("mrr", "false_injection_rate", "auc"):
        assert name in summary


# ── 指标的前提：fused 与选择旋钮无关 ────────────────────────────────────────


def _tiny_store() -> MemoryStore:
    store = MemoryStore(":memory:")
    for index, text in enumerate(
        (
            "盐度探头每月标定一次，标定用的是标准海水。",
            "蓄电池电量低于百分之二十就告警。",
            "运行日志保留九十天，到期之后压缩成周报。",
            "采集箱每十五分钟往岸上推一次数据包。",
        )
    ):
        store.add(
            MemoryItem(
                memory_id=f"m-{index}",
                kind=MemoryKind.FACT,
                tier=Tier.LONGTERM,
                content=text,
                created_at=NOW,
                weight=0.7,
                lex_sdr=encode_lexical(text),
            )
        )
    return store


def test_fused_does_not_depend_on_selection_knobs() -> None:
    """同一条候选在两种阈值配置下的 ``fused`` **逐位相同**。

    这是 :func:`~biosnn_memterm.bench.metrics.auc` 合法性的前提：AUC 用的是普查
    调用（``min_score=0``、``max_items=1``）里的分，而报告出来的是部署配置下的数。
    两者能并列，只因为 ``fused`` 只由通道得分与通道权重决定（``recall._fuse``），
    与 ``min_score`` / ``max_items`` / ``budget`` / ``mmr_lambda`` 全都无关。

    将来有人把阈值塞进 ``_fuse``（比如除以 ``min_score`` 做归一化），AUC 会静默
    变成另一个量而**没有任何一处会报错**——这条断言就是那件事的哨兵。
    """
    store = _tiny_store()
    query = "盐度探头多久标定一次"

    def injected_fused(config: RecallConfig) -> dict[str, float]:
        ledger = recall(
            store, query_text=query, budget_tokens=10_000, config=config, now=NOW, settle=False
        )
        return {m.memory_id: m.scores.fused for m in ledger.injected}

    strict = injected_fused(RecallConfig(min_score=0.05, max_items=2))
    loose = injected_fused(RecallConfig(min_score=0.0, max_items=8))
    shared = set(strict) & set(loose)
    assert len(shared) == 2, (
        f"两种配置共同注入的应该正好是 strict 的那 2 条，实际 {sorted(shared)}"
        f"（strict {sorted(strict)}，loose {sorted(loose)}）——"
        f"这个测试没有在测东西。"
    )
    for memory_id in shared:
        assert strict[memory_id] == loose[memory_id], (
            f"{memory_id} 的 fused 随 min_score/max_items 变了："
            f"{strict[memory_id]} vs {loose[memory_id]}。"
        )


def test_census_config_reaches_the_whole_corpus() -> None:
    """普查调用要能看到**每一条**非 pinned 的条目。

    少一条就意味着 AUC 的负集少一条，而它不会报错。这条同时钉住
    :func:`census_config` 里 ``max_items=1`` 那个反直觉的选择：写 1 而不是
    ``corpus_size`` 是为了避开 O(n³)，而它之所以等价，是因为落选的那些会以
    ``MAX_ITEMS`` 的形态带着完整的 ``fused`` 进 rejected——这个测试验证的正是
    那个等价性。
    """
    store = _tiny_store()
    scores = census_scores(store, "盐度探头多久标定一次", RecallConfig())
    assert len(scores) == store.count()
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_run_builds_a_graph_exactly_when_the_channel_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run`` 的兜底建图**只在权重为正时发生，而且只发生一次**。

    两个方向都是真的失败：

    * **不为正也建**（比如写成"总是建"）——结果不会错（``recall`` 会短路），但
      每一档权重都白建一张图。扫描四档就是四张一模一样的图，而它没有任何症状。
    * **为正却不建**——这是更坏的那个：``recall`` 收到 ``graph=None``，只会往账本
      里写一句"没有可用的关联图"，然后**给出与权重 0 完全相同的数**。扫描表上看
      起来就是"联想没用"。

    后者用 ``GraphIndex.build`` 的调用次数钉住——账本里那句 note 不流进
    :class:`QueryOutcome`，从外面看不见，只能从这里看。
    """

    built: list[int] = []
    real = GraphIndex.build.__func__

    def spy(cls, items, **kwargs):
        materialized = list(items)
        built.append(len(materialized))
        return real(cls, materialized, **kwargs)

    monkeypatch.setattr(GraphIndex, "build", classmethod(spy))

    goldset = GoldSet(
        memories=(
            GoldMemory("m-0", "盐度探头每月标定一次，标定用的是标准海水。"),
            GoldMemory("m-1", "蓄电池电量低于百分之二十就告警。"),
        ),
        queries=(GoldQuery("q-1", "盐度探头多久标定一次", "fact", expect=("m-0",)),),
        gold_dir=Path(),
    )

    run(goldset, config=RecallConfig(graph_weight=0.0))
    assert built == [], "权重是 0，不该建图。"

    run(goldset, config=RecallConfig(graph_weight=0.5))
    assert built == [2], "权重为正时应当建一张图（2 个节点），而且只建一次。"


def test_run_forwards_the_graph_to_the_census_path() -> None:
    """``run`` 要把图**分别传给两条路**：端到端召回那条，与普查（AUC）那条。

    这条是补一次真实的翻车：``run`` 一开始只传给了 ``recall``，**忘了传给
    ``census_scores``**。症状是扫描表上 AUC 四列**一模一样**——而"一模一样"看起来
    太正常了，差点被当成"AUC 与阈值无关、所以不响应配置"（那句注释确实就写在上面）。
    真正的问题是：**一个在扫描里纹丝不动的指标，要么真的是不变量，要么根本没接线**，
    而这两者在报告上长得一样。

    **必须从 ``run`` 走，不能直接调 ``census_scores``。** 第一版就是直接调的——那
    只能证明 ``census_scores`` 认自己那个 ``graph`` 形参，而漏掉的那一跳恰好是
    ``run`` 到它之间。测试停在坏掉的那一层的下面，就永远抓不到它。
    """
    goldset = GoldSet(
        memories=(
            GoldMemory("m-0", "路由器放在客厅电视柜上，信号到卧室就只剩一格了。"),
            GoldMemory("m-1", "客厅的路由器挪到走廊之后，卧室的信号好了一格。"),
            GoldMemory("m-2", "路由器重启之后，客厅到卧室的信号稳定多了。"),
            GoldMemory("m-3", "水管接口渗水，缠生料带之前要把螺纹擦干净。"),
        ),
        queries=(GoldQuery("q-1", "客厅的路由器信号到卧室。", "fact", expect=("m-0",)),),
        gold_dir=Path(),
    )

    off = run(goldset, config=RecallConfig(graph_weight=0.0))[0].scores
    on = run(goldset, config=RecallConfig(graph_weight=1.0))[0].scores

    assert off is not None and on is not None
    assert set(on) == set(off), "两次普查的候选集不该不同（只差一个通道权重）。"
    assert on != off, "换了 graph_weight 而普查得分一字未变——图没传到普查那一路。"


def test_census_config_copies_channel_weights_verbatim() -> None:
    """普查只放开**选择与阈值**那几个旋钮，别的字段一个都不动。

    动了通道权重或衰减，量到的就不是"受测配置的打分函数"了——而那正是 AUC 声称在
    量的东西。

    **比的是全部字段，不是逐个数出来的几项。** 第一版把这几行写成"抄了 lexical、
    semantic、graph 三个权重"的断言，于是 P4b 加 ``graph_decay`` 时它照样绿：新增的
    字段既没被抄也没被断言，而漏抄的后果是"扫描出来的 AUC 用的是另一个衰减"——
    一个不会报错、只会让数对不上的偏差。这里改成"放开的那几个之外必须逐字段相等"，
    以后再加字段时它会自己红。
    """
    base = RecallConfig(
        lexical_weight=0.31, semantic_weight=0.77, graph_weight=0.5, graph_decay=0.125
    )
    census = census_config(corpus_size=200, base=base)

    #: 这些是**有意放开**的：把它们写成基线的值，普查就取不到全库得分了。
    relaxed = {
        "min_score": 0.0,
        "redundancy_threshold": 1.0,
        "mmr_lambda": 1.0,
        "max_items": 1,
        "candidate_limit": 200,
        "max_rejected": 200,
    }
    for field in fields(RecallConfig):
        name = field.name
        if name in relaxed:
            assert getattr(census, name) == relaxed[name], f"{name} 没放开。"
        else:
            assert getattr(census, name) == getattr(base, name), (
                f"{name} 被普查改掉了——它不是选择旋钮，是打分函数的一部分。"
                "新加的字段要么照抄，要么写进 relaxed 并说明为什么。"
            )
