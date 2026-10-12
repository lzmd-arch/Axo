"""基质语料上的端到端断言：结构不变量、可复现性、以及两项门槛。

:mod:`tests.test_recall_gate` 与 :mod:`tests.test_router_gate` 的同位体。它
**不是**在测"``encode_assoc`` 选位对不对""W 的算术对不对"——那些由
:mod:`tests.test_sdr_assoc`、:mod:`tests.test_graph`、:mod:`tests.test_substrate`
用可控的小例子逐条断言。它测的是三件只有"整套量具一起跑"才暴露得出来的事：

1. **结构不变量**：语料里的每一条都在图里、图里没有语料外的节点、每条查询的
   ``expect`` 都与它同簇（见 :func:`~axo.bench.substrate.substrate_structural_violations`）。
2. **可复现**：连跑两遍逐位相同。这套量具的全部输入是 blake2b 与 numpy，**没有
   随机数、没有模型、没有 GPU、没有网络**——所以这一条比召回那边容易成立，
   **正因为它容易成立，它才是一道便宜的哨兵**：任何一天有人往子采样或图里塞一个
   随机种子、一个时间戳，它立刻红。它同时是容差取值的证据（见
   :data:`~axo.bench.substrate.DEFAULT_TOLERANCE`）。
3. **门槛**：``neighbor_auc`` 与 ``completion_top1`` 不跌破 ``baseline.json``
   的容差，且基线自己像一份真实测量（防空转）。

**后两条断言刻意固化了两次验红**（见各自的 docstring）——它们不是复述实现，
而是把"这道守卫真的会红"这件事变成一条每次提交都会跑的断言。

默认档在 200 条语料上是亚秒级的，但仍然共用一个 module 级的结果：语料只该读一次盘，
编码（200 次词法码 + 200 × 64 次 blake2b）也只该算一次。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from axo.bench import (
    CodedMemory,
    NeighborPairs,
    SubstrateCorpus,
    build_graph,
    coded_memories,
    completion_outcomes,
    completion_summary,
    default_substrate_dir,
    graph_outcomes,
    graph_summary,
    load_substrate_corpus,
    neighbor_auc,
    neighbor_pairs,
    neighbor_summary,
    probe_sample,
    substrate_digests,
)
from axo.bench.gates import gate_rows, sanity_problems
from axo.bench.substrate import (
    DEFAULT_TOLERANCE,
    GATED,
    LOWER_IS_BETTER,
    SANITY_CEILING,
    SANITY_FLOOR,
    substrate_structural_violations,
)
from memcore.sdr import ASSOC_K
from memcore.types import DEFAULT_SDR_WIDTH, SDR

SUBSTRATE = default_substrate_dir()
BASELINE = SUBSTRATE / "baseline.json"

#: 默认档里**不含语料规模**的那部分参数。规模是语料自己的条数，所以它单独算。
#:
#: 这些值必须与 ``tools/bench_substrate.py`` 的 argparse 默认值一致，由
#: :func:`test_baseline_config_is_the_default_tier` 机械地钉住——测试与工具跑的是
#: 同一档，比较才有意义（否则"门槛过了"可能只是两边量了不同的东西）。
DEFAULT_TIER = {
    "assoc": "hash",
    "drop": 0.5,
    "probes": 200,
    "top_m": 8,
    "min_weight": 0.10,
    "decay": 0.5,
}


@pytest.fixture(scope="module")
def corpus() -> SubstrateCorpus:
    return load_substrate_corpus()


@pytest.fixture(scope="module")
def baseline() -> dict:
    """基线文件。**必须存在**——缺失是失败，不是跳过。

    把"没基线"写成 ``skip`` 的话，守卫就变成了一个空转的东西，而空转的守卫比没有
    守卫更糟：它让人以为有人在看。
    """
    assert BASELINE.is_file(), (
        f"找不到 {BASELINE}。基质基线是要入库的资产（计划书 §四）。"
        f"生成它：`python tools/bench_substrate.py --freeze`。"
    )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _expected_config(corpus: SubstrateCorpus) -> dict:
    return {**DEFAULT_TIER, "k": ASSOC_K, "scale": len(corpus.memories)}


def _measure(corpus: SubstrateCorpus, config: dict) -> dict:
    """在默认档上把四个数全量出来。**与 ``tools/bench_substrate.py::_measure`` 同序**。

    刻意不复用工具里的那个函数：``tools/`` 不是一个包，而**测试不该 import 工具**——
    工具的入口是命令行，这样"测试跑的是测试自己搭的那条链"，两边同时写错才会两边
    都绿。真正让两边不会分叉的是 :func:`test_baseline_config_is_the_default_tier`
    与 :func:`test_gate_passes`：它们把这一档与基线文件绑在一起。
    """
    coded = coded_memories(corpus.memories, k=config["k"], mode=config["assoc"])
    graph = build_graph(
        coded,
        top_m=config["top_m"],
        min_weight=config["min_weight"],
        k=config["k"],
        mode=config["assoc"],
    )
    cluster_of = {memory.memory_id: memory.cluster for memory in corpus.memories}
    pairs = neighbor_pairs(graph, cluster_of)
    sampled = probe_sample(corpus.memories, config["probes"])
    completions = completion_outcomes(
        coded, [memory.memory_id for memory in sampled], drop=config["drop"], top=5
    )
    hits = graph_outcomes(coded, corpus.queries, graph, decay=config["decay"])
    neighbors = neighbor_summary(pairs)
    completion = completion_summary(completions)
    return {
        "coded": coded,
        "graph": graph,
        "pairs": pairs,
        "completion_outcomes": completions,
        "graph_outcomes": hits,
        "neighbors": neighbors,
        "completion": completion,
        "graph_hit": graph_summary(hits),
        "metrics": {
            "neighbor_auc": neighbors["neighbor_auc"],
            "completion_top1": completion["completion_top1"],
        },
    }


@pytest.fixture(scope="module")
def measured(corpus: SubstrateCorpus, baseline: dict) -> dict:
    return _measure(corpus, _expected_config(corpus))


# ── 结构不变量 ──────────────────────────────────────────────────────────────


def test_structural_invariants(corpus: SubstrateCorpus, measured: dict) -> None:
    """语料与图对得上，且每条查询的 ``expect`` 都与它同簇。

    **这些不是"变差了"，是 bug。** 与 ``route_structural_violations`` 同一个定位，
    工具给它一个与 1 不同的退出码（3）。混进指标里的话，一条期望写错了簇只让
    ``graph_hit`` 掉 0.025——看起来像噪声，而症状是"那条查询永远够不到答案"。
    """
    violations = substrate_structural_violations(measured["graph"], corpus.memories, corpus.queries)
    assert violations == [], "结构不变量被违反——这是 bug，不是标定问题：\n  " + "\n  ".join(
        violations
    )


def test_every_memory_is_a_node(corpus: SubstrateCorpus, measured: dict) -> None:
    """自检：图里恰好是语料那些条，一条不多一条不少。

    条数单独断言，是因为"图里少一条"在结构不变量里只表现为一行文字，而它在指标上
    的表现是**所有分母一起小一点**——那不会让任何一项变红。
    """
    graph = measured["graph"]
    assert len(graph) == len(corpus.memories)
    assert sorted(graph.node_ids) == sorted(m.memory_id for m in corpus.memories)


def test_no_orphan_cluster_in_the_queries(corpus: SubstrateCorpus, measured: dict) -> None:
    """查询侧的簇必须在图里有节点。

    ``graph_hit`` 的候选集是图的节点；一个只出现在 queries.jsonl 里的簇名（拼写
    错误）会让那些查询的 ``expect`` 全是异簇的——那条由结构不变量报出来，但报的
    是"期望必须是同簇的"。这一条给出更早、更准的那句话。
    """
    nodes = set(measured["graph"].node_ids)
    for query in corpus.queries:
        missing = sorted(set(query.expect) - nodes)
        assert not missing, f"{query.query_id} 期望的 {missing} 不在图里。"


# ── 可复现 ──────────────────────────────────────────────────────────────────


def test_substrate_benchmark_is_reproducible(corpus: SubstrateCorpus) -> None:
    """连跑两遍，图、逐条结果、聚合指标都逐位相同。

    这条是 :data:`~axo.bench.substrate.DEFAULT_TOLERANCE` 取 0.02 / 0.05
    的**前提**：容差之所以不是"采样噪声"，正是因为这里没有任何随机来源。所以它不
    只是可复现性检查，它是那个容差取值的证据。

    逐条结果也一起比（不只是聚合指标）：两条不同的错误在总数上可以互相抵消，而
    "总数一样"会让人以为什么都没变。图比的是 ``GraphIndex.__eq__``，也就是
    ``node_ids`` / ``nbr_idx`` / ``nbr_w`` 三个数组的逐元素相等。
    """
    config = _expected_config(corpus)
    first = _measure(corpus, config)
    second = _measure(corpus, config)
    assert first["graph"] == second["graph"], (
        "两遍建出来的图不一样——图里混进了隐藏状态（随机种子、时间、字典序不稳的集合）。"
    )
    assert first["completion_outcomes"] == second["completion_outcomes"]
    assert first["graph_outcomes"] == second["graph_outcomes"]
    assert first["metrics"] == second["metrics"]
    assert first["graph_hit"] == second["graph_hit"]


def test_graph_outcomes_routes_through_the_shared_diffusion_operator(
    corpus: SubstrateCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """量具的扩散**必须**走 :meth:`memcore.graph.GraphIndex.diffuse`，不许在量具里另写一份。

    这条守的不是数值，是**"同一个算子只有一处定义"**。P4b 把这段计算从
    :func:`~axo.bench.substrate.graph_outcomes` 搬到了图上，理由写在
    ``diffuse`` 的 docstring 里：量具与生产要共用它，而两份实现漂移正是这个仓最忌讳
    的形状。搬迁本身不会有人反对，**搬回去**也不会——只要没人看着，复制一份出来
    改一个参数（``decay`` 的默认值、max 换成 sum）是最自然的"顺手优化"，而两份
    实现的分叉在指标上表现为容差内的偏移：``--check`` 照样绿。

    所以这里把 ``diffuse`` 换成一层记录用的外壳，跑完确认：**每条查询各调一次**、
    种子分的形状是整张图、衰减传的是配置里那个值，且结果与不装外壳时逐位相同
    （后者说明量具确实只走了这一条路，没有"某条分支绕开外壳"的暗门）。
    """
    from memcore.graph import GraphIndex

    config = _expected_config(corpus)
    coded = coded_memories(corpus.memories, k=config["k"], mode=config["assoc"])
    graph = build_graph(
        coded,
        top_m=config["top_m"],
        min_weight=config["min_weight"],
        k=config["k"],
        mode=config["assoc"],
    )
    expected = graph_outcomes(coded, corpus.queries, graph, decay=config["decay"])

    seen: list[tuple[tuple[int, ...], float]] = []
    real = GraphIndex.diffuse

    def spy(self: GraphIndex, base: np.ndarray, *, decay: float = 0.5) -> np.ndarray:
        seen.append((base.shape, decay))
        return real(self, base, decay=decay)

    monkeypatch.setattr(GraphIndex, "diffuse", spy)
    actual = graph_outcomes(coded, corpus.queries, graph, decay=config["decay"])

    assert len(seen) == len(corpus.queries), "每条查询应当各扩散一次。"
    assert {shape for shape, _ in seen} == {(len(graph),)}, "种子分应当覆盖整张图的行序。"
    assert {decay for _, decay in seen} == {config["decay"]}, (
        "扩散用错了衰减——量具的参数没有流到算子上。"
    )
    assert actual == expected, (
        "装了记录外壳之后结果变了：说明量具有一条绕开 `GraphIndex.diffuse` 的路径，"
        "扩散因此有了两份实现。"
    )


# ── 门槛 ────────────────────────────────────────────────────────────────────


def test_the_gate_tier_matches_the_reported_tier(measured: dict, baseline: dict) -> None:
    """进门槛的那一档就是报告里那一档。

    这条挡的是"量具量了 A、门槛判了 B"——比如有人让 ``--drop`` 流进扫描行却没流进
    门槛行。症状是门槛永远绿，而报告上的数是另一个参数下的。
    """
    assert measured["metrics"] == baseline["metrics"], (
        "测试在默认档上量出来的数与基线里记的不一致。若这是有意改的（换了参数、换了"
        "语料），请重跑 `python tools/bench_substrate.py --freeze`；若是无意的，"
        "那是这一档的两条链已经分叉了。"
    )


# ── 门槛 ────────────────────────────────────────────────────────────────────


def test_gate_passes(baseline: dict, measured: dict) -> None:
    """两项指标不跌破基线的容差。

    走 :func:`~axo.bench.gates.gate_rows` 并**显式传**
    ``gated=GATED`` 与 ``lower_is_better=LOWER_IS_BETTER``：前者的默认值是召回那
    一套名单，后者的默认值含误注入率。测试与 ``tools/bench_substrate.py`` 必须
    经过同一对参数，否则工具里哪天传错了名单这边仍然绿——那正是 ``bench/gates.py``
    存在的理由。
    """
    rows = gate_rows(
        baseline["metrics"],
        measured["metrics"],
        baseline.get("tolerance"),
        gated=GATED,
        lower_is_better=LOWER_IS_BETTER,
    )
    assert len(rows) == len(GATED), (
        f"只产生了 {len(rows)} 条判定，而 GATED 有 {len(GATED)} 项——基线里少了一项。"
        f"这种情况下的 `all([])` 是 True，测试会静默变绿。"
    )
    failed = [row for row in rows if not row.passed]
    assert not failed, (
        "以下指标跌破了容差：\n  "
        + "\n  ".join(
            f"{row.metric}: 基线 {row.baseline:.4f} → 本轮 {row.live:.4f}"
            f"（差 {row.delta:+.4f}，下限 {row.limit:.4f}）"
            for row in failed
        )
        + (
            "\n\n这不是程序出错。先分流：语料被改过？还是编码/选位/建图的规则真的变了？"
            "（`python tools/bench_substrate.py --check` 会把两条线索都打出来。）"
        )
    )


def test_baseline_is_not_vacuous(baseline: dict) -> None:
    """基线自己必须像一份真实测量。**这一条就是防空转。**

    没有它，"把 baseline 的 ``neighbor_auc`` 写成 0.5、``completion_top1`` 写成 0
    就永远绿"是两步就能走到的状态，而且它看起来完全正常。
    """
    problems = sanity_problems(baseline["metrics"], floor=SANITY_FLOOR, ceiling=SANITY_CEILING)
    assert not problems, "基线文件不像一份真实测量：\n  " + "\n  ".join(problems)


def test_sanity_ceiling_actually_fires_on_orthogonal_codes() -> None:
    """**验红固化**：``completion_top1`` 的天花板不是一条空转的断言。

    默认档上它**触不到**——hash 档的关联码挤在 552 维子空间里，``W = AᵀA`` 被公共
    分量淹没，**连不扰动的探针（``drop=0``）都找不回自己**（实测 0.0250，与
    ``drop=0.5`` 一样）。所以不加这一组的话，"这道守卫会红吗"只能靠推理回答。

    这里用一组**互不相交**的 64 位码把整条链走通：50 条码两两无交集，W 干净，
    ``drop=0`` 的探针必然找回自己 → ``completion_top1 = 1.0`` → 防空转报出来。
    结论有两条：上界可达（不是空转），以及**它在默认档上触不到**（所以它拦的不是
    当前这份基线，而是"哪天有人把编码器改成近似正交之后，把 drop=0 下恒为 1 的
    数冻成基线"）。
    """
    items = _orthogonal_items(count=50, k=64)
    ids = [item.memory_id for item in items]
    summary = completion_summary(completion_outcomes(items, ids, drop=0.0, top=5))
    assert summary["completion_top1"] == 1.0, (
        f"正交码上不扰动的探针没找回自己（top1 = {summary['completion_top1']}）——"
        f"下面的防空转断言会因此变成空转。"
    )
    problems = sanity_problems(
        {"neighbor_auc": 0.9, "completion_top1": summary["completion_top1"]},
        floor=SANITY_FLOOR,
        ceiling=SANITY_CEILING,
    )
    assert any("completion_top1" in problem for problem in problems), (
        f"整条链走通了却一条问题都没报出来——SANITY_CEILING 是个摆设。收到 {problems}。"
    )


def test_sanity_floor_catches_a_blind_graph() -> None:
    """**验红的另一半**：``neighbor_auc`` 的地板也不是摆设。

    AUC 掉到 0.5 以下不是"差一点的图"，是"边权与簇**逆向**相关"——同簇的边比异簇的
    还弱。那是边的口径写反了（正负例接反、Jaccard 写成了"不相似度"），不是标定问题。
    冻结这种基线等于把「测反了」冻成「合格」。

    **边界是严格小于**（:func:`~axo.bench.gates.sanity_problems` 用的是
    ``value < low``）：恰好 0.5 放行。这条边界与召回那边 ``hit@8`` 的地板同形，
    而它是**有意**的——0.5 是"什么都没分开"，而那本来就该由门槛自己（基线 ± 容差）
    去管；这道守卫拦的是符号错，不是"效果差"。下面第二段把这条边界也钉住，
    免得哪天有人把它改成 ``<=`` 而没人知道。
    """
    blind = sanity_problems(
        {"neighbor_auc": 0.49, "completion_top1": 0.5},
        floor=SANITY_FLOOR,
        ceiling=SANITY_CEILING,
    )
    assert any("neighbor_auc" in problem for problem in blind), (
        f"AUC 掉到瞎猜线以下却没被拦下来——SANITY_FLOOR 是个摆设。收到 {blind}。"
    )

    on_the_line = sanity_problems(
        {"neighbor_auc": 0.5, "completion_top1": 0.5},
        floor=SANITY_FLOOR,
        ceiling=SANITY_CEILING,
    )
    assert not on_the_line, (
        f"恰好 0.5 被拦下来了——地板的边界从「严格小于」变成了「小于等于」。"
        f"改边界当然可以，但那是一次有意的政策改动，不该悄悄发生。收到 {on_the_line}。"
    )


def test_reversing_the_auc_fails_the_gate(baseline: dict, measured: dict) -> None:
    """**验红固化**：把 ``neighbor_auc`` 的正负例写反，门槛必须红。

    验红时真的做过一次：把 ``neighbor_auc`` 改成 ``_auc(pairs.diff, pairs.same)``
    之后 ``--check`` 给出 0.2917 = 1 − 0.7083 并退出码 1。这里把同一件事固化成
    一条**不依赖改动源码**的断言——方向写反在数值上等价于"本轮的值取反"。

    为什么值得单列：AUC 的分子分母写反是**静默**的（依然落在 [0,1]，依然是个漂亮的
    数），而它的含义完全变了；而门槛比较的正是它的绝对值。
    """
    reversed_metrics = {
        "neighbor_auc": 1.0 - measured["metrics"]["neighbor_auc"],
        "completion_top1": measured["metrics"]["completion_top1"],
    }
    rows = gate_rows(
        baseline["metrics"],
        reversed_metrics,
        baseline.get("tolerance"),
        gated=GATED,
        lower_is_better=LOWER_IS_BETTER,
    )
    failed = {row.metric for row in rows if not row.passed}
    assert "neighbor_auc" in failed, (
        "把 AUC 反过来之后门槛仍然是绿的——那说明方向或者容差有一处没接上。"
    )
    assert "completion_top1" not in failed, (
        "只有 neighbor_auc 被改动了，completion_top1 不该跟着红——"
        "那说明两条指标串了（比如共用一个变量）。"
    )


def test_direction_comes_from_the_table_that_was_passed_in() -> None:
    """**方向必须来自调用方传进来的那张表**，不是 :mod:`.gates` 的模块级默认表。

    这份量具是仓库里**第一个传自定义方向表**的调用方（它传空集，见
    :data:`LOWER_IS_BETTER`）。在那之前，"``limit`` 按传进来的表算、``passed``
    回头查模块级那张"这个不一致碰不到——两份名单的指标都不在模块级那张表里。
    它一旦碰上就是静默的：``limit`` 说"涨了 0.05 就跌破"，``passed`` 却按"越大
    越好"判，于是**越小越好的指标涨了也报达标**。

    这里用一个**当前不存在**的指标名（越小越好的混淆率）把那条路走通：前两条断言
    钉 ``limit`` 与这一行记下的方向，第三条钉 ``passed`` 用的是同一张表。
    """
    baseline = {"confusion_rate": 0.10}
    live = {"confusion_rate": 0.30}  # 涨了三倍：按"越小越好"必须跌破
    rows = gate_rows(
        baseline,
        live,
        {"confusion_rate": 0.05},
        gated=("confusion_rate",),
        lower_is_better=LOWER_IS_BETTER | {"confusion_rate"},
    )
    (row,) = rows
    assert row.limit == pytest.approx(0.15), (
        "方向没进 limit——上限应该是 基线 + 容差 = 0.15，而不是基线 − 容差。"
    )
    assert row.lower_is_better is True, "方向没被记进这一行，判定的依据就无从核对。"
    assert not row.passed, (
        "0.30 越过了上限 0.15，却判成达标——``passed`` 读的是模块级那张方向表"
        "（里面没有这个指标），而不是调用方传进来的那张。这正是「方向必须与指标"
        "同源」那条纪律被半途丢掉的样子。"
    )


def test_neighbor_auc_direction() -> None:
    """方向在**源头**也钉一次：同簇边权更高才是 > 0.5。

    上面的门槛断言用的是"在基线的数上取反"，它证明判定链是对的；这一条证明
    :func:`neighbor_auc` 自己没把两堆接反——两处都有，才是"方向写死"这件事的
    完整守卫。
    """
    assert neighbor_auc(NeighborPairs(same=(0.9, 0.8), diff=(0.1, 0.2), orphans=0)) == 1.0
    assert neighbor_auc(NeighborPairs(same=(0.1, 0.2), diff=(0.9, 0.8), orphans=0)) == 0.0
    assert neighbor_auc(NeighborPairs(same=(0.5,), diff=(0.5,), orphans=0)) == 0.5  # 并列算半票


# ── 基线的其它不变量 ────────────────────────────────────────────────────────


def test_baseline_was_frozen_on_this_corpus(baseline: dict) -> None:
    """门槛比的必须是**同一把尺子**。

    语料改了而基线没重跑时，比较照样出结果——只是那个差值没有意义，而它会以"达标"
    的形态出现（两边的数碰巧接近）或者"跌破"（不接近），两种都不会提到真正的原因。
    """
    assert baseline["corpus"] == substrate_digests(SUBSTRATE), (
        "baseline.json 记的语料指纹与现在的不一致。改语料要在同一次提交里更新"
        "freeze.json 并重跑 `--freeze`。"
    )


def test_baseline_config_is_the_default_tier(corpus: SubstrateCorpus, baseline: dict) -> None:
    """基线记的参数必须就是默认档。

    ``neighbor_auc`` 与 ``completion_top1`` 都随 ``--k`` / ``--scale`` / ``--drop``
    变化，所以"基线记的是哪一档"是这两个数唯一的上下文。有人拿 ``--drop 0.2``
    冻了一份基线的话，它比现在松——而文件里如果没有这一节，没人看得出来。

    这一条同时把测试自己那份 :data:`DEFAULT_TIER` 与工具绑在一起：参数改了而这里
    没跟上，两边就会各量各的。
    """
    assert baseline["config"] == _expected_config(corpus), (
        f"基线记的参数是 {baseline['config']}，而默认档是 {_expected_config(corpus)}。"
        f"**门槛只绑默认档**：扫描出来的那些行是用来裁决参数该取多少的。"
    )


def test_tolerance_is_written_into_the_baseline(baseline: dict) -> None:
    """容差必须**抄进文件**，而不是只在代码里。

    调容差因此变成一次看得见的 diff（改文件），而不是一次没人看见的代码改动——
    后者是"悄悄放宽门槛"最常见的形态。
    """
    assert baseline["tolerance"] == DEFAULT_TOLERANCE


def test_graph_hit_is_reported_but_not_gated(baseline: dict, measured: dict) -> None:
    """第三通道与词法基线的数**并排报**，但**不进门槛**。

    并排是刻意的：第三通道的全部意义是"带来词法没有的东西"，把两个数分开报会让
    "图没带来什么"这件事可以被含糊过去。

    不进门槛也是刻意的：它衡量的是"第三通道该不该上"这个**尚未裁决**的问题
    （P4b 的 go/no-go）。冻住一个还没裁决的数，会让 P4b 从"够不够好"变成"不许
    变坏"——那是把一个待决问题伪装成一个已决问题。
    """
    assert set(baseline["metrics"]) == set(GATED), (
        f"基线里进了门槛的指标是 {sorted(baseline['metrics'])}，而 GATED 是 {GATED}。"
        f"只报告的数不许出现在这里。"
    )
    for channel in ("lexical", "graph"):
        for name in ("hit@1", "hit@8", "mrr"):
            assert f"{channel}_{name}" in measured["graph_hit"], (
                f"报告里少了 {channel}_{name}——两条通道必须并排。"
            )


def test_the_lexical_baseline_is_in_the_baseline_too(baseline: dict) -> None:
    """词法基线的数也要入库。

    它是第三通道的**参照物**：只留图的那几个数，下一版就没有"图带来了什么"的
    基线可读——而那个差值是 P4b 唯一的 go/no-go 依据。
    """
    hits = baseline["graph_hit"]
    for channel in ("lexical", "graph"):
        for name in ("hit@1", "hit@8", "mrr"):
            assert f"{channel}_{name}" in hits, f"基线里少了 {channel}_{name}。"


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _orthogonal_items(*, count: int, k: int) -> list[CodedMemory]:
    """``count`` 条两两不相交的 ``k`` 位码。

    码长取 :data:`~memcore.types.DEFAULT_SDR_WIDTH`，并要求 ``count × k`` 塞得下：
    塞不下就不是正交而是循环重叠了，那样这条断言量的是别的东西。
    """
    assert count * k <= DEFAULT_SDR_WIDTH, "正交码放不下——调小 count 或 k。"
    items: list[CodedMemory] = []
    for index in range(count):
        idx = np.arange(index * k, (index + 1) * k)
        pattern = SDR(idx=idx, weights=np.ones(k, dtype=np.float64), width=DEFAULT_SDR_WIDTH)
        items.append(
            CodedMemory(
                memory_id=f"orth-{index:03d}",
                cluster="orthogonal",
                lex_sdr=pattern,
                assoc_sdr=pattern,
            )
        )
    return items
