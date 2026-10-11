"""基质量具：图 A 的边权分不分得开簇、W 还剩多少吸引域、第三通道值不值。

**为什么另起一份语料。** ``benchmarks/gold/`` 每行只有 ``{id, content}``，
**没有簇标签**，而这份量具的核心判据是"同簇 vs 异簇"——没有标签就没有判据。
而且金标准集是为**召回**造的：里面全是"每五分钟 / 每十分钟 / 每十五分钟"这种
近义改写（那些是故意的硬负例）。拿它给图当尺子会犯 README 记着的那种错（拿错
尺子），更糟的是近义改写恰好会让关联图的 top-m 邻居全是自己的近重复项，量出来的
precision 好看但它量的是"我认得出自己的改写"，不是"我把相关的连起来了"。

新语料 ``benchmarks/substrate/`` 的每一行带 ``cluster``，并有自己的 freeze.json。
它有 10 个簇、每簇 20 条。**簇不是主题分类学，是判据**——它只需要满足"同簇的该被
连起来、异簇的不该"，别的都不要求。

**全部由 blake2b 与 numpy 决定：无随机数、无模型、无 GPU、无网络。** 所以它能在
CI 与 pre-commit 里跑得动——这正是能把 :data:`GATED` 里那两项绑成门槛的前提。
语义通道那条"要 Ollama 所以不进门槛"的理由在这里不成立。

报什么、判什么::

    neighbor_auc          进门槛   同簇边的权重高于异簇边的概率（0.5 = 瞎猜）
    completion_top1       进门槛   ρ 丢位之后还认得自己的比例
    ------------------------------------------------------------------
    neighbor_precision@m  只报告   现有边里落在同簇的比例
    completion_top5       只报告
    confusion_rate        只报告   top-1 是**别的**记忆（P4c 的 sleep_confusion）
    graph_hit@* / mrr     只报告   第三通道 vs 词法基线，并排

**为什么 graph_hit 不进门槛**：它衡量的是"第三通道该不该上"这个**尚未裁决**的
问题（P4b 的 go/no-go）。冻住一个还没裁决的数，会让 P4b 从"够不够好"变成
"不许变坏"——那是把一个待决问题伪装成一个已决问题。

这一层不碰 Qt、不碰 HTTP，也不碰库（``mem.db``）。它吃的是内存里的语料与
``memcore`` 的纯对象，与 :mod:`~biosnn_memterm.bench.metrics` 同一条纪律。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from memcore.graph import GraphConfig, GraphIndex
from memcore.sdr import ASSOC_K, drop_bits, encode_assoc, encode_lexical
from memcore.substrate import Substrate, SubstrateConfig
from memcore.types import SDR

from .corpus import file_sha256, read_jsonl

__all__ = [
    "DEFAULT_TOLERANCE",
    "GATED",
    "LOWER_IS_BETTER",
    "SANITY_CEILING",
    "SANITY_FLOOR",
    "CodedMemory",
    "CompletionOutcome",
    "GraphOutcome",
    "NeighborPairs",
    "SubstrateCorpus",
    "SubstrateMemory",
    "SubstrateQuery",
    "build_graph",
    "coded_memories",
    "completion_outcomes",
    "completion_summary",
    "default_substrate_dir",
    "drop_bits",
    "graph_outcomes",
    "graph_summary",
    "load_substrate_corpus",
    "load_substrate_memories",
    "load_substrate_queries",
    "neighbor_auc",
    "neighbor_pairs",
    "neighbor_precision_at_m",
    "neighbor_summary",
    "probe_sample",
    "scaled_memories",
    "substrate_digests",
    "substrate_structural_violations",
]

_SCHEMA_VERSION = 1

#: "找不到文件"那句话。逐份语料各写各的，理由同 :data:`~.goldset._GOLD_HINT`。
_SUBSTRATE_HINT = (
    "基质语料在仓库根的 benchmarks/substrate/ 下；从源码检出跑的话它就在。"
    "找不到通常是工作目录不对，用 --corpus-dir 指定。"
)


# ── 门槛政策（名单与方向必须同源，理由见 :mod:`.gates` 的模块文档）────────────


#: 进门槛的两项。**它们是基质的性质，离线可跑**——与语义通道那条"要 Ollama
#: 所以不进门槛"的理由（README 记着的那次实测否定结果）不是一回事。
GATED = ("neighbor_auc", "completion_top1")

#: 这份量具里没有"越小越好"的指标。**必须显式传空集**给 :func:`~.gates.gate_rows`：
#: 它的默认值是召回那一套（含误注入率），拿召回的默认来判基质会把
#: ``false_injection_rate`` 当成越大越好——虽然它有"基线里没有就跳过"兜底，但
#: 那份兜底不该是唯一防线。
LOWER_IS_BETTER: frozenset[str] = frozenset()

#: 容差。**会被抄进基线文件**，判定时以文件里的为准。
#:
#: ``neighbor_auc`` 走 0.02：它的配对数是"同簇边 × 异簇边"，200 条语料上约
#: 6×10⁵ 对，抽样噪声远小于指标本身的可分辨度（理由同金标准集的 AUC）。
#: ``completion_top1`` 走 0.05：n=200 个探针时 p≈0.9 的标准误约 0.021，2σ 是
#: ±0.043——容差比这更紧，红的就是噪声而不是回归。
DEFAULT_TOLERANCE = {
    "neighbor_auc": 0.02,
    "completion_top1": 0.05,
}

#: 防空转的反向断言：**基线自己必须先像一份基线**。
#:
#: ``neighbor_auc`` 的地板是 0.5（瞎猜）。一张连簇都分不开的图不是"差一点的
#: 图"，是"边权与簇无关"，而那说明边的口径写错了（比如 Jaccard 算成了交集
#: 计数），不是标定问题。
SANITY_FLOOR = {"neighbor_auc": 0.5}

#: ``completion_top1`` 的天花板。**它拦的正是"探针没被扰动"这件空转**
#: （``--drop 0`` 时补全恒为 1，这个指标就什么都没量）。定在 0.995 而不是 0.99：
#: ρ>0 的探针真的丢掉了三分之一的位，"几乎全对"是可能的，**全对不是**。
SANITY_CEILING = {"completion_top1": 0.995}


# ── 语料 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubstrateMemory:
    """语料里的一条记忆。**带簇标签**——这是它区别于金标准集的地方。"""

    memory_id: str
    content: str
    cluster: str


@dataclass(frozen=True)
class SubstrateQuery:
    """一条查询与它的期望集。

    ``expect`` 是**同簇的另外几条**：其中通常有一条与查询共享词汇（词法通道
    够得到），另外几条要靠图的边才走得到。词法基线够不着的正是后者——这份量具
    要回答的就是"第三通道能不能把它们捞回来"。

    ``expect`` 里的每一条都必须与 ``cluster`` 同簇，由
    :func:`substrate_structural_violations` 机械检查：不是同簇的"期望"会让
    ``graph_hit`` 变成一个既不是图也不是词法的量。
    """

    query_id: str
    query: str
    cluster: str
    expect: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubstrateCorpus:
    """一整份基质语料。"""

    memories: tuple[SubstrateMemory, ...]
    queries: tuple[SubstrateQuery, ...]
    corpus_dir: Path

    @property
    def by_id(self) -> dict[str, SubstrateMemory]:
        return {memory.memory_id: memory for memory in self.memories}

    @property
    def clusters(self) -> dict[str, int]:
        """``{簇名: 条数}``，按簇名排序。"""
        counts: dict[str, int] = {}
        for memory in self.memories:
            counts[memory.cluster] = counts.get(memory.cluster, 0) + 1
        return dict(sorted(counts.items()))


def _read_jsonl(path: Path) -> list[dict]:
    return read_jsonl(path, hint=_SUBSTRATE_HINT, what="语料")


def _cluster_of(row: dict, path: Path) -> str:
    cluster = row.get("cluster")
    if not isinstance(cluster, str) or not cluster:
        raise ValueError(
            f"{path} 里的 cluster 必须是个非空字符串，收到 {cluster!r}。"
            f"**没有簇标签这份语料就没有判据**——同簇/异簇是 neighbor_auc 的全部分母。"
        )
    return cluster


def load_substrate_memories(path: Path) -> list[SubstrateMemory]:
    """读 ``memories.jsonl``。名字带 ``substrate_`` 前缀：``load_memories`` 已经被
    金标准集占了，两个同名导入不会报错——它只会**静默顶替**（见 :mod:`.routes`）。"""
    return [
        SubstrateMemory(
            memory_id=row["id"],
            content=row["content"],
            cluster=_cluster_of(row, path),
        )
        for row in _read_jsonl(path)
    ]


def load_substrate_queries(path: Path) -> list[SubstrateQuery]:
    """读 ``queries.jsonl``。"""
    queries: list[SubstrateQuery] = []
    for row in _read_jsonl(path):
        expect = row.get("expect", [])
        if isinstance(expect, str):
            raise ValueError(f"{path} 里的 expect 必须是数组，收到字符串 {expect!r}。")
        queries.append(
            SubstrateQuery(
                query_id=row["id"],
                query=row["query"],
                cluster=_cluster_of(row, path),
                expect=tuple(expect),
            )
        )
    return queries


def default_substrate_dir() -> Path:
    """定位仓库根的 ``benchmarks/substrate/``。

    按"那里有没有 memories.jsonl"判断，而不是按固定的层数——层数会随包结构调整，
    而"语料在那儿"不会。理由与 :func:`~.goldset.default_gold_dir` 一字不差。
    """
    for candidate in Path(__file__).resolve().parents:
        directory = candidate / "benchmarks" / "substrate"
        if (directory / "memories.jsonl").is_file():
            return directory
    raise FileNotFoundError(
        "在源码树的任何一层都没找到 benchmarks/substrate/memories.jsonl。"
        "基质语料只在源码检出里（sdist 会带上，wheel 不带）——"
        "如果你是从安装好的包跑，请用 --corpus-dir 指向语料目录。"
    )


def load_substrate_corpus(corpus_dir: Path | None = None) -> SubstrateCorpus:
    """把一份基质语料整个读进来。"""
    directory = Path(corpus_dir) if corpus_dir is not None else default_substrate_dir()
    return SubstrateCorpus(
        memories=tuple(load_substrate_memories(directory / "memories.jsonl")),
        queries=tuple(load_substrate_queries(directory / "queries.jsonl")),
        corpus_dir=directory,
    )


def substrate_digests(corpus_dir: Path) -> dict[str, object]:
    """两份 JSONL 的条数与 sha256，外加簇的分布。**冻结判据就是它。**"""
    directory = Path(corpus_dir)
    memories_path = directory / "memories.jsonl"
    queries_path = directory / "queries.jsonl"
    memories = load_substrate_memories(memories_path)
    queries = load_substrate_queries(queries_path)
    counts: dict[str, int] = {}
    for memory in memories:
        counts[memory.cluster] = counts.get(memory.cluster, 0) + 1
    return {
        "schema_version": _SCHEMA_VERSION,
        "memories": {"count": len(memories), "sha256": file_sha256(memories_path)},
        "queries": {"count": len(queries), "sha256": file_sha256(queries_path)},
        "clusters": dict(sorted(counts.items())),
    }


# ── 规模与探针的确定性抽样 ──────────────────────────────────────────────────


def probe_sample(memories: Sequence[SubstrateMemory], count: int) -> tuple[SubstrateMemory, ...]:
    """从语料里等距抽 ``count`` 条当探针。

    **探针是抽的不是全量的**，这是为了让不同 ``--scale`` 之间可比：n=200 时
    200 个探针，n=20000 时还是 200 个。全量探针会让"n 变大"同时改变两件事
    （容量与测量精度），而容量悬崖是本版唯一想看的东西。

    用等距而不是随机：**没有随机数**，所以同一个 n 每次跑出同一批探针，门槛才
    绑得住。``np.unique`` 顺带保证升序且不重复，因此返回条数**可能略少于**
    ``count``（``linspace`` 取整时会撞在一起）——调用方要按实际长度算分母。
    """
    if count < 1:
        raise ValueError(f"探针条数必须 >= 1，收到 {count!r}。")
    total = len(memories)
    if total == 0:
        return ()
    if count >= total:
        return tuple(memories)
    picks = np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))
    return tuple(memories[int(position)] for position in picks)


def scaled_memories(memories: Sequence[SubstrateMemory], n: int) -> tuple[SubstrateMemory, ...]:
    """把语料确定性扩到 ``n`` 条（``n`` 更小时就是前 ``n`` 条）。

    **合成的那部分是同簇两条句子的拼接**，不是某一条的副本。这个选择是有理由的：
    副本的 Jaccard 与本体是 1.0，于是"最近邻"永远是它自己的影子、``neighbor_auc``
    被抬到虚高，而``completion_top1`` 会塌在"分不清自己和自己的复制品"上——那量
    的是语料的重复度，不是基质的容量。拼接出来的条目与任何单条的重叠都在 0.5 上下，
    簇标签仍然正确（两条都来自同一个簇）。

    **它仍然是合成的**：``--scale 20000`` 上那些数不是"两万条真实记忆"上的数，
    是"两万条同簇拼接语料"上的数。报的时候要连着这句一起报（工具的
    ``--scale`` 分列标题里就写着）。
    """
    if n < 1:
        raise ValueError(f"规模必须 >= 1，收到 {n!r}。")
    if n <= len(memories):
        return tuple(memories[:n])

    groups: dict[str, list[SubstrateMemory]] = {}
    for memory in memories:
        groups.setdefault(memory.cluster, []).append(memory)

    out = list(memories)
    index = len(out)
    while len(out) < n:
        cluster = memories[index % len(memories)].cluster
        pool = groups[cluster]
        first = pool[(index * 7) % len(pool)]
        second = pool[(index * 13 + 5) % len(pool)]
        if second.memory_id == first.memory_id:
            second = pool[(index * 13 + 6) % len(pool)]
        out.append(
            SubstrateMemory(
                memory_id=f"s-{index + 1:04d}",
                content=first.content + second.content,
                cluster=cluster,
            )
        )
        index += 1
    return tuple(out)


# ── 编码 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CodedMemory:
    """一条编好码的记忆。**字段名必须与 :class:`memcore.types.MemoryItem` 一致**
    ——:meth:`~memcore.substrate.Substrate.build` 与
    :meth:`~memcore.graph.GraphIndex.build` 是按名字取的。

    刻意不复用 ``MemoryItem``：那是个可变的、带十几个字段的写入侧结构，量具只用
    其中三个，而借用它会让"改一条语料"看起来像"改一条记忆"。
    """

    memory_id: str
    cluster: str
    lex_sdr: SDR
    assoc_sdr: SDR


def coded_memories(
    memories: Sequence[SubstrateMemory], *, k: int = ASSOC_K, mode: str = "hash"
) -> tuple[CodedMemory, ...]:
    """把语料编成词法码 + 关联码。**算一次，喂给图和基质两边。**

    不这么做的话 ``--scale 20000`` 会把同一批 64 次哈希算两遍（图一遍、基质
    一遍），而那是两万条 × 64 次 blake2b——不值得花两遍。
    """
    return tuple(_code(memory, k=k, mode=mode) for memory in memories)


def _code(memory: SubstrateMemory, *, k: int, mode: str) -> CodedMemory:
    lex = encode_lexical(memory.content)
    return CodedMemory(
        memory_id=memory.memory_id,
        cluster=memory.cluster,
        lex_sdr=lex,
        assoc_sdr=encode_assoc(lex, k=k, mode=mode),
    )


def build_graph(
    coded: Sequence[CodedMemory],
    *,
    top_m: int = 8,
    min_weight: float = 0.10,
    k: int = ASSOC_K,
    mode: str = "hash",
) -> GraphIndex:
    """按编好的条目建图。参数与 :class:`~memcore.graph.GraphConfig` 一一对应。"""
    return GraphIndex.build(
        coded, config=GraphConfig(top_m=top_m, min_weight=min_weight, k=k, mode=mode)
    )


# ── 图 A：边权分不分得开簇 ──────────────────────────────────────────────────


@dataclass(frozen=True)
class NeighborPairs:
    """图里全部无向边（一条边只算一次）的权重，按两端是否同簇分开。

    ``orphans`` 是一条边都没有的节点数。它不是判据，但一个"图建出来了"而
    ``orphans`` 接近节点数的结果，说明 ``min_weight`` 或 ``top_m`` 把边滤光了
    ——只看 AUC 是看不出来的（分母为零时 AUC 是 ``nan``，而 ``nan`` 很容易被
    当成"没测"而不是"没边"）。
    """

    same: tuple[float, ...]
    diff: tuple[float, ...]
    orphans: int


def neighbor_pairs(graph: GraphIndex, cluster_of: Mapping[str, str]) -> NeighborPairs:
    """把图摊成同簇边与异簇边两堆。

    **只看上三角**（``column > row``）：图存的是每个节点自己的有向 top-m，一条
    无向边通常被记两次；两堆都收的话同一条边会在 AUC 里投两票，而两票的权重
    未必相等（两侧各自选出来的 top-m 不要求对称）——那会让 AUC 偏向"被双向选中的
    那些对"，而这不是要量的东西。

    Raises:
        ValueError: 图里有语料里没有的节点——那时"同簇还是异簇"无从判断。
            工具在算指标**之前**用 :func:`substrate_structural_violations` 挡一次，
            这里是第二道（这个函数也可以被单独调用）。
    """
    same: list[float] = []
    diff: list[float] = []
    orphans = 0
    for row, source_id in enumerate(graph.node_ids):
        if source_id not in cluster_of:
            raise ValueError(
                f"图里的节点 {source_id} 不在语料里——它属于哪个簇无从判断，"
                f"而同簇/异簇是这份量具的全部分母。"
            )
        degree = 0
        for column, weight in zip(graph.nbr_idx[row], graph.nbr_w[row], strict=True):
            target = int(column)
            if target < 0 or target <= row:
                continue
            target_id = graph.node_ids[target]
            if target_id not in cluster_of:
                raise ValueError(f"图里的节点 {target_id} 不在语料里。")
            degree += 1
            bucket = same if cluster_of[source_id] == cluster_of[target_id] else diff
            bucket.append(float(weight))
        if degree == 0:
            orphans += 1
    return NeighborPairs(same=tuple(same), diff=tuple(diff), orphans=orphans)


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """任取一个正例与一个负例，正例更大的概率。**并列按 ½。** 0.5 = 瞎猜。

    走秩和（Mann–Whitney U）而不是两层循环：``--scale 20000`` 上两堆各有几万条，
    逐对比较是十亿次。并列取平均秩，这与逐对比较里"并列算半票"是同一个数。
    """
    size = len(positive) + len(negative)
    if not positive or not negative:
        return float("nan")
    combined = np.concatenate(
        [np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)]
    )
    order = np.argsort(combined, kind="stable")
    values = combined[order]
    # 并列组：值发生变化的位置就是组边界，组内各得同一个（平均）秩。
    starts = np.flatnonzero(np.concatenate([[True], values[1:] != values[:-1]]))
    counts = np.diff(np.append(starts, size))
    average_rank = starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(size, dtype=np.float64)
    ranks[order] = np.repeat(average_rank, counts)
    n_pos = len(positive)
    rank_sum = float(ranks[:n_pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * len(negative))


def neighbor_auc(pairs: NeighborPairs) -> float:
    """``neighbor_auc`` —— 同簇边的权重高于异簇边的概率。

    这是"图 A 的边权能不能把相关的与无关的分开"的直接答案，也是它进门槛的理由：
    一张 AUC 停在 0.5 附近的图，喂给扩散与可视化只会输出噪声。
    """
    return _auc(pairs.same, pairs.diff)


def neighbor_precision_at_m(pairs: NeighborPairs) -> float:
    """``neighbor_precision@m`` —— 现有边里落在同簇的比例。**直观版。**

    分母是边数而不是节点数：这份量具关心的是"连出来的边对不对"。无向边已经被
    :func:`neighbor_pairs` 去重过了。
    """
    total = len(pairs.same) + len(pairs.diff)
    if not total:
        return float("nan")
    return len(pairs.same) / total


def neighbor_summary(pairs: NeighborPairs) -> dict[str, float]:
    """图 A 那几个数，扁平成一个字典。键名就是基线文件与 README 里用的名字。"""
    return {
        "edge_same": float(len(pairs.same)),
        "edge_diff": float(len(pairs.diff)),
        "orphan_nodes": float(pairs.orphans),
        "neighbor_auc": neighbor_auc(pairs),
        "neighbor_precision@m": neighbor_precision_at_m(pairs),
    }


# ── W：模式补全 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompletionOutcome:
    """一条探针跑完一次补全之后的证据。指标函数只吃它。"""

    memory_id: str
    cluster: str
    ranked: tuple[str, ...]


#: ``drop_bits`` 由 :mod:`memcore.sdr` 再导出（见上面的 import）。P4c 的睡眠巩固
#: 把这个算子搬进了 ``memcore``（搬了，不是复制）——理由与 ``GraphIndex.diffuse``
#: 那次搬家一字不差：
#: **让量具与生产用同一个算子**。巩固的判据「回放辨认率」必须与这里的
#: ``completion_top1`` 同口径，探针的造法一旦有两份实现，两个数字就不再可比，
#: 而它们看起来仍然都像个合理的百分比。这个再导出让既有的
#: ``from ...bench.substrate import drop_bits`` 继续可用，实现只有一份。


def completion_outcomes(
    coded: Sequence[CodedMemory],
    probes: Sequence[str],
    *,
    drop: float = 0.5,
    top: int = 5,
) -> tuple[CompletionOutcome, ...]:
    """逐条探针跑一次 :meth:`~memcore.substrate.Substrate.complete`。

    ``probes`` 是 memory_id 的序列（用 :func:`probe_sample` 抽出来）。不在
    ``coded`` 里的 id 会被跳过——语料与探针来自同一批，正常不会发生；真发生了
    说明调用方传错，而"少测几条"比"当场炸"更隐蔽，所以这里数出来由调用方对账。

    返回条数可能少于 ``len(probes)``。
    """
    if top < 1:
        raise ValueError(f"top 必须 >= 1，收到 {top!r}。")
    substrate = Substrate.build(coded, config=SubstrateConfig())
    by_id = {item.memory_id: item for item in coded}
    outcomes: list[CompletionOutcome] = []
    for memory_id in probes:
        item = by_id.get(memory_id)
        if item is None:
            continue
        probe = drop_bits(item.assoc_sdr, drop)
        ranked = tuple(name for name, _score in substrate.complete(probe, top=top))
        outcomes.append(CompletionOutcome(memory_id=memory_id, cluster=item.cluster, ranked=ranked))
    return tuple(outcomes)


def _top1_is_self(outcome: CompletionOutcome) -> bool:
    return bool(outcome.ranked) and outcome.ranked[0] == outcome.memory_id


def completion_summary(outcomes: Sequence[CompletionOutcome]) -> dict[str, float]:
    """``completion_top1`` / ``top5`` / ``confusion_rate``。

    ``confusion_rate`` 是"top-1 是**别的**记忆"的比例。**它与 ``top1`` 互补**
    （只要每个探针都有排名，两者之和恒为 1），所以它不是第四个独立的数。单列
    它的理由是它是 :attr:`memcore.types.ConsolidationEvent.sleep_confusion`
    的同一个原语——P4c 那个字段按的就是这个口径（见
    :func:`memcore.consolidation.replay_cluster`），而不是就地发明一个。

    分母是**探针数**，不是语料条数：``--scale`` 放大语料时探针数不变，这几个数
    才跨规模可比。
    """
    total = len(outcomes)
    if not total:
        return {
            "n_probes": 0.0,
            "completion_top1": float("nan"),
            "completion_top5": float("nan"),
            "confusion_rate": float("nan"),
        }
    top1 = sum(1 for outcome in outcomes if _top1_is_self(outcome))
    top5 = sum(1 for outcome in outcomes if outcome.memory_id in outcome.ranked[:5])
    return {
        "n_probes": float(total),
        "completion_top1": top1 / total,
        "completion_top5": top5 / total,
        "confusion_rate": (total - top1) / total,
    }


# ── 第三通道：图上的扩散 vs 词法基线 ────────────────────────────────────────


@dataclass(frozen=True)
class GraphOutcome:
    """一条查询在两条通道上的排名证据。

    ``*_rank`` 是**第一条期望记忆的位次**（1 起），``0`` 表示"前 ``depth`` 条里
    没有"。存位次而不是整条排名表：n=20000 时四十条查询的完整排名表是八十万个
    id，而 ``--json`` 要把它写进 artifact。
    """

    query_id: str
    cluster: str
    expect: tuple[str, ...]
    lexical_rank: int
    graph_rank: int


def _first_rank(ranked: np.ndarray, node_ids: Sequence[str], expected: set[str], depth: int) -> int:
    for position in range(min(depth, len(ranked))):
        if node_ids[int(ranked[position])] in expected:
            return position + 1
    return 0


def graph_outcomes(
    coded: Sequence[CodedMemory],
    queries: Sequence[SubstrateQuery],
    graph: GraphIndex,
    *,
    decay: float = 0.5,
    depth: int = 10,
) -> tuple[GraphOutcome, ...]:
    """query → 词法种子 → 沿 A 扩散一步 → 排名。与词法基线并排。

    两条通道吃的是**同一个**查询编码与同一批候选，唯一的差别是有没有扩散这一步：
    词法排名按 Jaccard 降序，图排名在它上面加 ``decay × max(种子分 × 边权)``。
    取 ``max`` 而不是求和——求和会让连接数多的节点被抬起来（hub 爆炸），而
    "从哪个种子走过来的"才是这一步的语义。

    扩散的方向是**双向的**：图存的是每个节点自己的有向 top-m，这里遍历全部行，
    所以 ``i→j`` 与 ``j→i`` 两条记录各自贡献一次，等价于把两个方向并起来。

    ``depth`` 之内的位次是精确的，之外一律记 0（= 没找到）。所以 ``mrr`` 是
    **下界**；两条通道用的是同一个 ``depth``，比较仍然成立。
    """
    if decay < 0.0:
        # ``diffuse`` 自己也会拦，但它那一句要等循环里第一条查询才执行——查询列表
        # 空着时它永远不会执行。参数校验不该取决于输入是不是空的。
        raise ValueError(f"扩散系数不能是负数，收到 {decay!r}。")
    node_ids = graph.node_ids
    lex_of = {item.memory_id: item.lex_sdr for item in coded}
    base = np.zeros(len(node_ids), dtype=np.float64)

    outcomes: list[GraphOutcome] = []
    for query in queries:
        query_sdr = encode_lexical(query.query)
        for row, node_id in enumerate(node_ids):
            source = lex_of.get(node_id)
            base[row] = 0.0 if source is None else query_sdr.jaccard(source)

        combined = base + graph.diffuse(base, decay=decay)

        lexical_rank = np.argsort(-base, kind="stable")
        graph_rank = np.argsort(-combined, kind="stable")
        expected = set(query.expect)
        outcomes.append(
            GraphOutcome(
                query_id=query.query_id,
                cluster=query.cluster,
                expect=query.expect,
                lexical_rank=_first_rank(lexical_rank, node_ids, expected, depth),
                graph_rank=_first_rank(graph_rank, node_ids, expected, depth),
            )
        )
    return tuple(outcomes)


def graph_summary(outcomes: Sequence[GraphOutcome]) -> dict[str, float]:
    """两条通道的 hit@1 / hit@8 / mrr，并排放在同一个字典里。

    并排是**刻意的**：第三通道的全部意义是"带来词法没有的东西"，把两个数分开
    报会让"图没带来什么"这件事可以被含糊过去。
    """
    total = len(outcomes)
    if not total:
        return {
            "n_queries": 0.0,
            **{
                f"{channel}_{name}": float("nan")
                for channel in ("lexical", "graph")
                for name in ("hit@1", "hit@8", "mrr")
            },
        }
    summary: dict[str, float] = {"n_queries": float(total)}
    for channel in ("lexical", "graph"):
        ranks = [getattr(outcome, f"{channel}_rank") for outcome in outcomes]
        summary[f"{channel}_hit@1"] = sum(1 for rank in ranks if rank == 1) / total
        summary[f"{channel}_hit@8"] = sum(1 for rank in ranks if 1 <= rank <= 8) / total
        summary[f"{channel}_mrr"] = sum(1.0 / rank for rank in ranks if rank) / total
    return summary


# ── 结构不变量 ──────────────────────────────────────────────────────────────


def substrate_structural_violations(
    graph: GraphIndex,
    memories: Sequence[SubstrateMemory],
    queries: Sequence[SubstrateQuery],
) -> list[str]:
    """语料与图对不上、或者 ``expect`` 不长在同一个簇上时返回理由（空 = 通过）。

    **这些不是"变差了"，是 bug。** 与 :func:`~.routeeval.route_structural_violations`
    同一个定位，工具给它一个与 1 不同的退出码。

    为什么 ``expect`` 必须同簇：``graph_hit`` 量的是"沿边能不能走到正确答案"。
    一条期望里混进异簇的条目，那个数就变成了"图能不能跨簇走"——那是另一个问题，
    而且答案（"不能，也不该"）会让指标恒定偏低而看不出原因。
    """
    problems: list[str] = []
    by_id = {memory.memory_id: memory for memory in memories}

    for node_id in graph.node_ids:
        if node_id not in by_id:
            problems.append(f"图里有语料里没有的节点 {node_id}。")
    for memory in memories:
        if graph.index_of(memory.memory_id) < 0:
            problems.append(f"语料里的 {memory.memory_id} 不在图里。")

    for query in queries:
        if not query.expect:
            problems.append(f"{query.query_id} 没有期望集——它进不了 hit/MRR 的分母。")
        for expected in query.expect:
            memory = by_id.get(expected)
            if memory is None:
                problems.append(f"{query.query_id} 期望的 {expected} 不在语料里。")
            elif memory.cluster != query.cluster:
                problems.append(
                    f"{query.query_id} 是 {query.cluster} 簇，却期望 {expected}"
                    f"（{memory.cluster} 簇）——期望必须是同簇的。"
                )
    return problems
