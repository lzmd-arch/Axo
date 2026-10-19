"""联想通道接进召回：接线、短路、以及"开了开关没反应"要说话。

P4a 把关联图 A 建了出来并量了出来，结论是**负的**（图上的扩散 ``hit@1`` 0.8250
低于词法基线 0.8500）。所以 P4b 接的是线，不是开关：``RecallConfig.graph_weight``
默认 **0**，图要调用方自己注入。这个文件守的就是"接了线"与"没打开"这两件事同时
成立——它们互为对方的证明，缺一条另一条就没有意义：

1. **``graph_weight=0`` 时行为逐位不变**（:func:`test_a_zero_weight_is_bit_identical_
   with_and_without_a_graph`）。没有这条，"默认关"就只是一句注释：某天有人把图无条件
   塞进融合，默认配置的用户会在毫不知情的情况下换了排序。
2. **权重为正时联想确实进了融合**（:func:`test_the_channel_joins_the_fusion`）。
   没有这条，"接了线"同样只是一句注释——而**接了线却零调用**是这个仓踩过的坑，
   症状看起来像"模型不听话"。
3. **扩散在权重为 0 时根本不算**（:func:`test_a_zero_weight_never_runs_the_diffusion`）。
   第 1 条比的是结果，这条比的是**代价**：算了不用，账本一样对，但关着开关的人
   白白付了一趟全库解码。
4. **开了开关却用不上时，账本要说出来**（``test_*_says_so``）。"我明明打开了"
   是最难自查的一类问题：界面上开关是勾着的，召回里什么都没发生，而没有任何
   地方能看出这一点。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from memcore.graph import GraphConfig, GraphIndex
from memcore.ids import new_ulid
from memcore.recall import RecallConfig, _graph_channel, recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import SDR, MemoryItem, MemoryKind, Taint, Tier

#: 三句同主题 + 两句无关。同主题那三句共享"路由器／客厅／卧室／信号"这些二元组，
#: 所以图上一定连得起来；两句无关的用来当"不该被联想拉进来"的对照。
TEXTS = (
    "路由器放在客厅电视柜上，信号到卧室就只剩一格了。",
    "客厅的路由器挪到走廊之后，卧室的信号好了一格。",
    "路由器重启之后，客厅到卧室的信号稳定多了。",
    "水管接口渗水，缠生料带之前要把螺纹擦干净。",
    "厨房下水道有点慢，倒了一壶热水下去。",
)

QUERY = "客厅的路由器信号到卧室很弱。"

#: 两次调用之间的时钟**必须钉住**。权重是结算过惰性衰减的，而衰减读 ``now``——
#: 不钉的话两次运行会差在小数点后十位上，逐位比较就成了在比时钟。
NOW = 1_700_000_000.0

#: ``top_m`` 与 ``min_weight`` 都比默认档紧，为的是把图压稀——默认档在这五句上会
#: 连出 18 条边（几乎全连通），那样"邻居传过来的分"与"自己命中的分"就分不开了。
_SPARSE = GraphConfig(top_m=3, min_weight=0.15)


def _item(content: str, *, pinned: bool = False, weight: float = 0.8) -> MemoryItem:
    return MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content=content,
        taint=Taint.USER,
        weight=weight,
        pinned=pinned,
        lex_sdr=encode_lexical(content),
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        for text in TEXTS:
            opened.add(_item(text))
        yield opened


@pytest.fixture
def graph(store: MemoryStore) -> GraphIndex:
    return GraphIndex.build(store.iter_items(), config=_SPARSE)


def _run(store: MemoryStore, *, graph: GraphIndex | None, weight: float):
    """固定输入跑一轮召回。``settle=False`` 与固定的 ``now`` 都是为了让两次调用
    只差在那一个参数上——否则比的是时钟，不是接线。"""
    return recall(
        store,
        query_text=QUERY,
        budget_tokens=2000,
        config=RecallConfig(graph_weight=weight, max_items=8),
        graph=graph,
        now=NOW,
        settle=False,
    )


# ── 短路：默认关 ────────────────────────────────────────────────────────────


def test_a_zero_weight_is_bit_identical_with_and_without_a_graph(
    store: MemoryStore, graph: GraphIndex
) -> None:
    """**本版唯一一条"不许变坏"的硬约束。**

    ``graph_weight=0`` 时，手里有一张真图与根本没有图必须产出**逐位相同**的账本。
    比的是整个账本对象（含每条候选的 :class:`~memcore.audit.ChannelScores`），
    只把 ``elapsed_ms`` 抹掉——它本来就会不同，而且它与"召回结果变没变"无关。

    做法是**短路**而不是分支：权重为 0 时 :func:`_graph_channel` 根本不被调用
    （见 :func:`test_a_zero_weight_never_runs_the_diffusion`），``fired`` 与权重
    表里都不会出现 ``graph`` 这个键，于是融合的连乘与今天一字不差。
    """
    with_graph = _run(store, graph=graph, weight=0.0)
    without = _run(store, graph=None, weight=0.0)

    assert dataclasses.replace(with_graph, elapsed_ms=0.0) == dataclasses.replace(
        without, elapsed_ms=0.0
    )
    assert with_graph.channels == without.channels == ("lexical",)
    for injected in with_graph.injected:
        assert injected.scores.graph == 0.0, "权重是 0，联想分不该有任何来源。"


def test_a_zero_weight_never_runs_the_diffusion(
    store: MemoryStore, graph: GraphIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """权重为 0 时扩散**根本不算**——不是算了不用。

    这条比的是代价，不是结果。上面那条逐位相同对"算了不用"一样成立（乘 0 不影响
    连乘），而"算了"的代价是一趟额外的 ``store.iter_items()``：它会把全库每条记忆的
    SDR 都解码一遍，与库的大小成正比。关着开关的多数人不该付它。
    """
    calls: list[float] = []
    real = GraphIndex.diffuse

    def spy(self: GraphIndex, base, *, decay: float = 0.5):
        calls.append(decay)
        return real(self, base, decay=decay)

    monkeypatch.setattr(GraphIndex, "diffuse", spy)

    _run(store, graph=graph, weight=0.0)
    assert calls == [], "权重为 0 时不该算扩散。"

    _run(store, graph=graph, weight=0.5)
    assert len(calls) == 1, "权重为正时每条查询恰好扩散一次。"


# ── 接线：权重为正 ──────────────────────────────────────────────────────────


def test_the_channel_joins_the_fusion(store: MemoryStore, graph: GraphIndex) -> None:
    """权重为正时，联想确实进了融合——账本、通道列表、解释三处都看得见。

    这三处一起断言是刻意的：`channels` 是给机器读的，`ChannelScores.graph` 是给
    界面读的，``explain`` 是给人读的。只对一处会让"接了线却有一半没通"看起来正常。
    """
    ledger = _run(store, graph=graph, weight=0.5)

    assert "graph" in ledger.channels
    credited = [m for m in ledger.injected if m.scores.graph > 0.0]
    assert credited, "没有一条拿到联想分——通道接上了但没通。"
    assert any("联想" in m.explain for m in credited), (
        "账本里有联想分，解释里却没有——归因与数值脱钩了。"
    )


def test_the_channel_can_only_raise_a_candidate(store: MemoryStore, graph: GraphIndex) -> None:
    """打开联想**只会让候选的分变高**，不会压低任何一条。

    这是噪声或算子"只增不减"性质在真实路径上的表现：多一条通道就是多一个
    ``(1 − r·s)`` 因子，而那个因子永远 ≤ 1。若哪天有人给联想加一个惩罚项
    （"邻居太像就扣分"），这条会红。
    """
    off = {m.memory_id: m.scores.fused for m in _run(store, graph=graph, weight=0.0).injected}
    on = {m.memory_id: m.scores.fused for m in _run(store, graph=graph, weight=1.0).injected}

    shared = set(off) & set(on)
    assert shared, "两次召回的入选集合没有交集，比不出来。"
    for memory_id in shared:
        assert on[memory_id] > off[memory_id], (
            f"{memory_id} 打开联想之后分没变高——通道进了融合却没起作用。"
        )


def test_a_heavier_weight_lifts_more(store: MemoryStore, graph: GraphIndex) -> None:
    """可靠度是**绝对**乘在得分上的：权重越大，联想分对融合的贡献越大。"""
    light = {m.memory_id: m.scores.fused for m in _run(store, graph=graph, weight=0.25).injected}
    heavy = {m.memory_id: m.scores.fused for m in _run(store, graph=graph, weight=1.0).injected}

    assert heavy, "权重 1.0 下一无所得，比不出来。"
    for memory_id, fused in heavy.items():
        if memory_id in light:
            assert fused > light[memory_id]


def test_the_seed_ignores_pinning_and_exclusion(tmp_path: Path) -> None:
    """种子**不过滤** pinned / excluded。这是个显式决定，不是漏掉了。

    ``base`` 是**图的属性**——"哪些节点与查询词面沾边"；pinned 与 excluded 是
    **候选池**的事，发生在 :func:`_scan` 里。把过滤提前到种子上的话，一条钉住的
    记忆会连带失去"当桥梁"的资格：它的邻居明明该被拉进来，却因为没有种子而收不到。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        store.add(_item(TEXTS[0], pinned=True))
        for text in TEXTS[1:]:
            store.add(_item(text))
        built = GraphIndex.build(store.iter_items(), config=_SPARSE)

        scores = _graph_channel(store, query_lex=encode_lexical(QUERY), graph=built, decay=0.5)

        assert scores, "钉住的那条没当种子，于是扩散什么也没传出来。"
        # 而它自己不会被注入：过滤发生在候选池那一层。
        ledger = _run(store, graph=built, weight=1.0)
        assert all(not m.pinned for m in ledger.injected)


def test_an_all_zero_spread_still_counts_as_the_channel_running(
    store: MemoryStore, graph: GraphIndex
) -> None:
    """图在，但一个种子都填不出来——这时通道**仍然算跑过**（返回 ``{}``，不是 ``None``）。

    空字典与 ``None`` 在这里是两件事：``None`` 是"这条通道这一轮不存在"，``{}``
    是"它跑了，什么也没传出来"。混起来会让"开了开关但没反应"与"开关根本没生效"
    看起来一模一样。

    这里用一个**宽度对不上**的查询码把种子全部挡掉。这同时是那条宽度守卫的第一条
    测试：两套不可比的表示算出来的 Jaccard 是垃圾，而它会以"看起来正常"的分数混进
    扩散——所以宽度对不上的节点**不给种子**（照 :func:`~memcore.recall._lexical_score`
    的同一口径）。

    **"查询与全库毫无词面重合"这条造不出来**：词法码是定宽哈希，任意两段文本的
    Jaccard 有一个由碰撞撑起来的地板（实测拉丁字母乱码对中文约 0.08）。所以想要
    全零扩散，只能从宽度上切开。
    """
    width = next(item.lex_sdr.width for item in store.iter_items() if item.lex_sdr is not None)
    foreign = SDR(idx=np.array([0, 1, 2]), weights=np.ones(3), width=width + 1)

    scores = _graph_channel(store, query_lex=foreign, graph=graph, decay=0.5)

    assert scores == {}


# ── 说了做不到的事，就要说出来 ──────────────────────────────────────────────


def test_enabling_the_channel_without_a_graph_says_so(store: MemoryStore) -> None:
    """开了开关却没有图——账本里必须有一句，而不是安静地什么都不做。"""
    ledger = _run(store, graph=None, weight=0.5)

    assert ledger.channels == ("lexical",)
    assert any("联想通道已启用" in note and "关联图" in note for note in ledger.notes), (
        f"开了开关却没有图，账本里没提这件事：{ledger.notes}"
    )


def test_enabling_the_channel_without_a_seed_says_so(store: MemoryStore, graph: GraphIndex) -> None:
    """图在，但这一轮的输入编不出词法码（全标点）——扩散没有种子。

    这一句的理由与上一条不同（是"这轮用不上"而不是"还没有图"），所以文案也不同。
    两句都指向同一件事：**别让用户对着一个勾着的开关猜。**
    """
    ledger = recall(
        store,
        query_text="。。。",
        budget_tokens=2000,
        config=RecallConfig(graph_weight=0.5),
        graph=graph,
        settle=False,
    )

    assert any("没有词法编码" in note for note in ledger.notes), ledger.notes


def test_a_zero_weight_says_nothing_about_the_channel(
    store: MemoryStore, graph: GraphIndex
) -> None:
    """反空转：那句提示只属于"开了开关"。

    权重是 0 时它是**没打开**，不是**打开了没用上**——写出来会让默认配置的每一轮
    账本上都挂着一条不存在的故障。
    """
    ledger = _run(store, graph=graph, weight=0.0)

    assert not any("联想" in note for note in ledger.notes), ledger.notes


def test_seeds_come_from_the_lexical_index_not_a_full_decode(
    store: MemoryStore, graph: GraphIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图通道的种子分不该再遍历全库、解码每条 SDR。"""

    def fail():
        raise AssertionError("图通道不该调用 iter_items()")

    monkeypatch.setattr(store, "iter_items", fail)

    scores = _graph_channel(store, query_lex=encode_lexical(QUERY), graph=graph, decay=0.5)

    assert scores
