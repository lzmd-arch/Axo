"""睡眠巩固 —— 把记忆按关联图分成簇，再在**每一簇自己的**基质上回放它。

## 为什么是分簇 + 局部 W，而不是全库一个 W

这一版的形状不是设计出来的，是 **P4a 量出来的那个负数逼出来的**。
:mod:`memcore.substrate` 的全局 W 在真实规模上一步补全不可靠，容量悬崖实测为
（``weight`` 档、50% 丢位探针）::

    n=200     0.8700
    n=500     0.2250
    n=1000    0.0900
    n=2000    0.0300

``benchmarks/substrate/CORPUS.md`` 因此写下结论：P4c 的吸引子回放要建在**分簇、
局部 W、或迭代去噪**上。本模块取第一条——把全库切成一簇一簇，每簇自己建一份 W，
于是"W 的容量"这个约束的作用域从全库缩到簇内，而簇的大小由 :data:`MAX_CLUSTER`
管住。**容量随簇规模扩展，不随全库扩展**，这就是分簇的全部意义。

## 两档关联码：图用 ``hash``，W 用 ``idf``（``weight`` 保留为对照）

同一份词法码现算两份关联码，互不干扰：

* **图 A 用 ``hash`` 档**（``GraphConfig`` 的默认）——它的 ``neighbor_auc`` 是
  0.7083，而 ``weight`` 档只有 0.5287。
* **局部 W 默认用 ``idf`` 档**——按**这一簇成员**的文档频率压掉跨条目公共位
  （:func:`~memcore.sdr.encode_assoc_idf`），是 ALIF 稳态适应的选位等价形式。
  2026-10-18 在 166 条真实记忆上量到回放辨认率 0.6988 → 0.8976。``hash`` 档的 W
  在**每一档规模上都是死的**（连 ``--drop 0`` 也只有 0.0200）；``weight`` 档在
  n=200 上是 0.8700，保留为 A/B 与历史事件回放档。

一个静默陷阱要堵住：:meth:`~memcore.substrate.Substrate.build` 会**优先吃**
``item.assoc_sdr``，只有它是 ``None`` 时才按 ``config`` 现算。所以本模块**显式**
现算关联码（``idf`` 或 ``weight``），不把
:class:`~memcore.types.MemoryItem` 直接递给 ``build``——否则哪天有人预先算了一份
``hash`` 档的 ``assoc_sdr`` 塞进条目，``code_mode`` 会被**无声忽略**，
而结果看起来仍然像个合理的辨认率。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MemCoreError
from .graph import GraphIndex
from .sdr import ASSOC_K, drop_bits, encode_assoc, encode_assoc_idf
from .substrate import Substrate, SubstrateConfig
from .types import SDR, ConsolidationEvent, MemoryItem, Tier

__all__ = [
    "BASE_SEED",
    "DEFAULT_DROP",
    "EPOCH",
    "EVENT_FIELDS",
    "MAX_CLUSTER",
    "MIN_CLUSTER",
    "REINFORCE",
    "SKIP",
    "SLEEP_CONFUSION",
    "SLEEP_REPLAY_MISS",
    "W_MODE",
    "W_MODES",
    "ReplayOutcome",
    "RunReport",
    "append_events",
    "cluster_event_rows",
    "compare_events",
    "consolidate",
    "event_rows",
    "events_of",
    "partition",
    "read_events",
    "replay_cluster",
    "run_id_for",
    "run_seed_for",
]

#: 五个决策名。取值与 :class:`~memcore.types.ConsolidationEvent` 的 docstring 同一套，
#: 写成一列常量是因为它们要被**比对**（``--replay`` 逐条比、报告里分类计数），
#: 而散在代码里的字符串字面量拼错一个字母不会报错，只会让某一类决策永远数不到。
REINFORCE = "reinforce"
SLEEP_CONFUSION = "sleep_confusion"
SLEEP_REPLAY_MISS = "sleep_replay_miss"
SKIP = "skip"
MERGE = "merge"
PROMOTE = "promote"
DEMOTE = "demote"
"""碎条目归并成一条派生条目。**只增不改**：写出的是新的一行，原句一字不动。

回放那三个决策改的都是 ``reinforce_count``，这一条**什么旧账都不改**——它只是
把散着的几句合成一条带时间的记录。所以它也不该出现在回放的比对里：回放比的是
同一个 ``run_seed`` 重跑会不会得到同一批 decision，而归并的判据里没有随机成分。"""

BASE_SEED = 0
"""整轮的基准种子。**常量而不是随机数**。

一轮巩固必须**完全可复现**——:class:`~memcore.types.ConsolidationEvent` 的整个
卖点就是"同一个 ``run_seed`` 重跑一遍、逐条比对"。所以种子不掷骰子：它由
(``base_seed``, ``run_ts``) 派生（见 :func:`run_seed_for`），整轮的差异只来自
``run_ts``。``--seed`` 可以覆盖它，覆盖的用途是**故意换一个抽样**做对照，
不是"让每轮不一样"。

代价记在这里：同一个安装在两台机器上、同一秒跑，``run_id`` 会一样。真需要
"每个安装不同"的话，得把一个随机 ``base_seed`` 持久化下来——本版不做。
"""

W_MODE = "idf"
"""局部 W 用的关联码档位。**默认是簇内文档频率的 ``idf`` 档。**

三档的关系：

* ``hash``：图 A 用的档。建 W **在每一档规模上都是死的**（n=200 的
  ``completion_top1`` 是 0.0250，连 ``--drop 0``、探针不残缺时也只有 0.0200）。
* ``weight``：P4c 原来用的档，n=200 上是 0.8700。它比 ``hash`` 好，但取的是
  各自词频最高的位，跨条目公共位没有被压掉。
* ``idf``：按**簇内**文档频率压公共位（:func:`~memcore.sdr.encode_assoc_idf`），
  是 ALIF 稳态适应的选位等价形式。2026-10-18 在 166 条真实记忆上量到回放辨认率
  **0.6988 → 0.8976**；在合成语料的分簇回放里 n=200 0.9000 → 0.9950、n=500
  0.4400 → 0.5840、n=1000 0.1370 → 0.3130、n=2000 0.1020 → 0.1960。

**它不是单条记忆的纯函数**：``df`` 是一簇成员的全量统计，所以只在
:func:`replay_cluster` 这种"整簇现算"的路径上使用；逐条写入路径不能算它。
``weight`` 保留为 A/B 与历史事件的回放档（见 :func:`replay_cluster` 的
``code_mode``）。
"""

W_MODES = ("idf", "weight")
"""``replay_cluster`` 认得的码档。``hash`` 不在里面——它是图 A 的档，不是 W 的。"""

DEFAULT_DROP = 0.5
"""回放探针默认丢掉多少比例的活跃位。

与 ``tools/bench_substrate.py`` 的 ``--drop`` 默认档一致——**这不是巧合，是同
口径的要求**：巩固的判据「回放辨认率」要与基线的 ``completion_top1`` 可比，
探针的残缺程度必须一样。见 :func:`~memcore.sdr.drop_bits`。
"""


@dataclass(frozen=True)
class _Coded:
    """喂给 :meth:`~memcore.substrate.Substrate.build` 的一行。

    只要 ``memory_id`` 与**已经算好的**关联码。``assoc_sdr`` 显式给值而
    ``lex_sdr`` 留 ``None``，是为了堵住 ``build`` 的"优先吃 ``assoc_sdr``"那条
    路径（见模块文档开头那段）：调用方若已经算过一份 ``hash`` 档的码塞进条目，
    ``code_mode`` 会被**无声忽略**，而结果看起来仍然像个合理的辨认率。这里把码
    算在外面、显式传进来，就没有那个空间。
    """

    memory_id: str
    assoc_sdr: SDR
    lex_sdr: None = None


@dataclass(frozen=True)
class ReplayOutcome:
    """一条记忆回放完了之后的证据。**判据与字段都从这里读，不另算一遍。**

    Attributes:
        memory_id: 被回放的那条。
        decision: ``"reinforce"`` / ``"sleep_confusion"`` / ``"sleep_replay_miss"``
            / ``"skip"``。取值与 :class:`~memcore.types.ConsolidationEvent` 的
            docstring 同一套。
        jaccard: 补全给出的第一名与**被回放的那条**之间的关联码 Jaccard。
            认出自己时恒为 1.0；认成别人时是一个小于 1 的正数，**直接量化串扰
            有多远**。
        confused_with: ``decision == "sleep_confusion"`` 时是顶替它的那条，否则
            ``None``。
        detail: 附带的证据。``skip`` 时是**具体原因**（不是"不满足条件"这种
            说了等于没说的东西）；其余情况带 ``probe_jaccard``（探针残度，
            ``drop=0.5`` 时约 0.5，可以自检丢位真的生效了）与加固前后的计数。
    """

    memory_id: str
    decision: str
    jaccard: float | None = None
    confused_with: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


MAX_CLUSTER = 200
"""一个簇最多装多少条记忆。

**取值依据是量出来的悬崖，不是 Hopfield 的 ``0.138·N``。** 上面那张表里
n=200 是唯一一个补全还可靠的实测点（0.8700），n=500 已经塌到 0.2250。经典公式
对 N=4096 给出约 570 条，而这个仓的**相关码把悬崖提前了**——CORPUS.md 记着这件事。
所以这里取那个唯一活着的实测点，而不是理论值。

**它是占位，不是标定过的。** 语料是合成的（``--scale`` 出来的一批长得很像的
句子），真实库上最优的簇规模没有实测。P4c 落地后应当拿真实库上
``reinforce_count`` 的分布回头标定这个数。
"""

MIN_CLUSTER = 2
"""小于这个规模的簇不回放。

单成员簇是**必须**挡掉的，不是为了整齐：它自己的 W 只有一个模式，
:meth:`~memcore.substrate.Substrate.complete` 对任何探针都恒回它自己——
于是"回放辨认率"被一批必然命中的样本抬高，而那个数字看起来完全正常。
"""

_MAX_LPA_ITERS = 20
"""标签传播的轮数上限。

真实图上几乎必然在前几轮就稳定；这个上限是给"不收敛"准备的（标签在两组之间
来回摆是 LPA 的已知行为）。到了上限就停下用当前标签——**第 4 步的切分对任何
输入都终止**，所以就算这里不收敛，整条路径也不会挂住。
"""


def _symmetrise(graph: GraphIndex) -> list[dict[int, float]]:
    """把有向 top-m 邻接表折成**无向**的加权邻接表，同一对边取 ``max``。

    **取 max 不取 sum**，与 :meth:`~memcore.graph.GraphIndex.diffuse` 的口径一致。
    图存的是"每个节点自己最像的 m 个"，所以 i→j 与 j→i 的权重本来就不同；求和会
    把**互相都认为对方很像**的那一对抬成两倍，而"像的程度"不该由记账次数决定。

    **两个方向都要写。** :meth:`~memcore.graph.GraphIndex.edges` 给的是有向边，
    i→j 存在而 j→i 不存在是常态（j 的 top-m 里没有 i）。只按行写一格的话
    得到的是**有向的**邻接表，而被它喂大的标签传播与区域生长都会变成方向相关的
    ——症状是分簇结果取决于哪一端"更主动"，而结果看起来仍然像个合理的分簇。

    返回 ``list[dict]``，第 ``i`` 项是 ``{邻居行号: 权重}``。**键的插入顺序由
    :meth:`~memcore.graph.GraphIndex.edges` 决定，而它是确定的**（``nonzero``
    按行序、列序），所以下游一切遍历都是可复现的。

    代价：5 万条 × 双向约 16 度时，这份 ``list[dict]`` 是上百 MB 的临时对象。
    分簇一天跑一次、跑在后台线程里，所以接受；真要压的话得换成压缩稀疏行。
    """
    rows, targets, weights = graph.edges()
    size = len(graph)
    sym: list[dict[int, float]] = [{} for _ in range(size)]
    for row, target, weight in zip(rows.tolist(), targets.tolist(), weights.tolist(), strict=True):
        if target == row:
            continue
        for a, b in ((row, target), (target, row)):
            current = sym[a].get(b)
            if current is None or weight > current:
                sym[a][b] = weight
    return sym


def _label_propagate(sym: Sequence[dict[int, float]]) -> list[int]:
    """确定性加权标签传播。返回每个行号的标签。

    **决策规则必须逐字确定**，否则同一张图两次跑出两个分簇，而 ``--replay``
    的逐条比对就永远失败：

    * 只有**严格大于**当前标签的票数才切换（票数并列时保留当前标签）；
    * 并列的最高票里取**标签值最小**的那个。

    实现上就是把"最高票、其次最小标签"写成一个 key，再要求它严格超过当前票数
    ——那一条同时给出了上面两条：当前标签若是最高票之一，无论它是不是最小的那个
    标签，都切不过去。
    """
    labels = list(range(len(sym)))
    for _ in range(_MAX_LPA_ITERS):
        changed = False
        for node in range(len(sym)):
            tally: dict[int, float] = {}
            for neighbor, weight in sym[node].items():
                label = labels[neighbor]
                tally[label] = tally.get(label, 0.0) + weight
            if not tally:
                continue
            # 最高票优先；票数相同时标签值小的优先。max 在 key 上取最大，
            # 所以第二项写成 -label。
            best_label, best_tally = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))
            if best_tally > tally.get(labels[node], 0.0):
                labels[node] = best_label
                changed = True
        if not changed:
            break
    return labels


def _grow_clusters(
    members: list[int], sym: Sequence[dict[int, float]], max_cluster: int
) -> list[list[int]]:
    """把一个超限的标签组切成若干个不超过 ``max_cluster`` 的簇。

    区域生长：种子取组内**加权度最大**的节点（并列取行号最小，而行号序即
    ``memory_id`` 序），然后每次并入"**到当前簇的边权之和最大**"的未分配邻居。

    用"到簇的权重之和"而不是单链接（"和簇里任一节点最像的那个"）：单链接会把
    一条细长链一节一节拉进来，最后得到一个又大又散的簇——而簇的意义正是
    "这些条记忆互相关联"，链式串联恰好破坏了它。

    候选必须与簇**已有边**才进得来（连接度为 0 的不选）。所以一个在组内没有
    任何邻居的节点会自己成为一个单成员簇，由 :func:`partition` 按
    :data:`MIN_CLUSTER` 丢掉。

    并列一律**取行号最小者**。``members`` 由 :func:`partition` 按行号升序给出，
    而行号序就是 ``memory_id`` 序（``GraphIndex.node_ids`` 已排序），所以
    ``-node`` 这个次键就是"取 memory_id 最小的那个"。
    """
    remaining: set[int] = set(members)
    # 加权度只在"到组内其他成员的边"上算——组外的不参与，否则播种会被全图的
    # hub 带偏，而切分的目的恰恰是让每簇内部自洽。
    clusters: list[list[int]] = []

    while remaining:
        seed = max(
            remaining,
            key=lambda node: (
                sum(w for nbr, w in sym[node].items() if nbr in remaining and nbr != node),
                -node,
            ),
        )
        remaining.discard(seed)
        cluster = [seed]
        # 到当前簇的累计边权。只在候选真的与簇相邻时才非零，所以它同时是
        # "够不够格进来"的判据。
        conn: dict[int, float] = {}
        while len(cluster) < max_cluster:
            for nbr, weight in sym[cluster[-1]].items():
                if nbr in remaining:
                    conn[nbr] = conn.get(nbr, 0.0) + weight
            candidates = [(score, -node, node) for node, score in conn.items() if score > 0.0]
            if not candidates:
                break
            _, _, chosen = max(candidates)
            remaining.discard(chosen)
            cluster.append(chosen)
            del conn[chosen]
        clusters.append(cluster)
    return clusters


def partition(
    graph: GraphIndex,
    *,
    max_cluster: int = MAX_CLUSTER,
    min_cluster: int = MIN_CLUSTER,
) -> tuple[tuple[str, ...], ...]:
    """把关联图分成若干个不超过 ``max_cluster`` 条的簇，返回 ``memory_id`` 的元组。

    两步：

    1. **标签传播**（:func:`_label_propagate`）——能细分就细分；
    2. **超限的标签组按区域生长切分**（:func:`_grow_clusters`）——**无条件执行**。

    第 2 步不是"保险"，是主路径：真实库上最短路的 top-m 图几乎必然是一整个巨大
    连通分量，标签传播给不出细簇。它同时也兜住了第 1 步不收敛的情形——切分对
    任何输入都终止。

    **返回的顺序是规范的**，因为 ``tools/sleep_once.py --replay`` 要逐条比对事件
    序列，而序列的顺序由簇的顺序与簇内成员顺序共同决定：

    * 簇内：切分出来的簇按**生长顺序**（种子在前）；没被切分的标签组按行序
      （也就是 ``memory_id`` 序）。
    * 簇间：按**第一个成员的 ``memory_id``** 升序。

    **返回的簇不覆盖图里的每个节点。** 规模小于 ``min_cluster`` 的簇被丢掉，所以
    孤立节点（组内没有邻居、自己成一个单成员簇）不在结果里——它们本版不回放。
    这是刻意的：单成员簇的补全恒为命中，收进来只会虚增辨认率。

    Args:
        graph: 关联图。只读它的 :meth:`~memcore.graph.GraphIndex.edges`。
        max_cluster: 簇规模上限。见 :data:`MAX_CLUSTER`。
        min_cluster: 簇规模下限，小于它的簇不进结果。见 :data:`MIN_CLUSTER`。

    Returns:
        ``((memory_id, ...), ...)``。没有满足条件的簇时返回 ``()``。
    """
    if min_cluster < 1:
        raise ValueError(f"min_cluster 必须 >= 1，收到 {min_cluster!r}。")
    if max_cluster < min_cluster:
        raise ValueError(
            f"max_cluster({max_cluster}) 不能小于 min_cluster({min_cluster})——"
            f"那样没有任何簇能同时满足上界与下界。"
        )

    node_ids = graph.node_ids
    if not node_ids:
        return ()

    sym = _symmetrise(graph)
    labels = _label_propagate(sym)

    groups: dict[int, list[int]] = {}
    for node, label in enumerate(labels):
        groups.setdefault(label, []).append(node)

    clusters: list[list[int]] = []
    for label in sorted(groups):
        members = groups[label]
        if len(members) > max_cluster:
            clusters.extend(_grow_clusters(members, sym, max_cluster))
        else:
            clusters.append(members)

    kept = [cluster for cluster in clusters if len(cluster) >= min_cluster]
    kept.sort(key=lambda cluster: node_ids[cluster[0]])
    return tuple(tuple(node_ids[node] for node in cluster) for cluster in kept)


# ── 回放：在一簇自己的基质上问"还认不认得自己" ─────────────────────────────


def _skip_reason(item: MemoryItem) -> str | None:
    """这条记忆该不该被回放。返回具体原因，``None`` 表示可以回放。

    **四种原因各自有理由，不是"不满足条件"**：

    * ``"pinned"``——钉住的条目连衰减都不参与，睡眠期去"加固"它没有意义；
    * ``"cold"``——已经冷掉的记忆不再参与召回，回放它等于给一个用户看不到的东西
      记功；
    * ``"no_lex_sdr"``——还没有编码，没有码就没有探针。这在实时写入路径上是常态
      （编码是异步补的），不是错误。

    第四种"不在图里"不在这里判——:func:`partition` 压根不会把这类节点放进簇。
    两者是同一件事的两处写法，别在两个地方各判一次。
    """
    if item.pinned:
        return "pinned"
    if item.tier is Tier.COLD:
        return "cold"
    if item.lex_sdr is None:
        return "no_lex_sdr"
    return None


def _classify(
    item: MemoryItem,
    ranked: Sequence[tuple[str, float]],
    codes: dict[str, SDR],
    probe: SDR,
) -> ReplayOutcome:
    """把一次补全的结果读成一条决策。**这一步是纯的，所以它单独可测。**

    三个分支与 ``benchmarks/substrate/CORPUS.md`` 里那个 ``confusion_rate`` 是
    **同一个原语**：那里叫"top-1 是别的记忆"，这里叫 ``sleep_confusion``。
    分开发明一个口径的话，巩固报的串扰率与基线的串扰率就没法互相印证了。

    ``sleep_replay_miss`` 那支**目前从整条流水线走不到**：只要基质非空，
    ``complete`` 至少回一条，而空基质意味着这一簇全是 ``skip``。留着它是因为
    "补全什么都没给出来"是一个真实的、需要被看见的结局——哪天真出现了，
    它该是一条事件，而不是一个被 ``ranked[0]`` 撞出来的 ``IndexError``。
    """
    code = codes[item.memory_id]
    probe_jaccard = probe.jaccard(code)
    if not ranked:
        return ReplayOutcome(
            item.memory_id, SLEEP_REPLAY_MISS, detail={"probe_jaccard": probe_jaccard}
        )

    top1 = ranked[0][0]
    if top1 == item.memory_id:
        return ReplayOutcome(
            item.memory_id,
            REINFORCE,
            # 自己与自己的 Jaccard 恒为 1.0——写出来是为了让这个字段在
            # "认出自己"与"认成别人"两种情形下的读法一致，而不是一个特例。
            jaccard=codes[top1].jaccard(code),
            detail={
                "probe_jaccard": probe_jaccard,
                "reinforce_before": item.reinforce_count,
                "reinforce_after": item.reinforce_count + 1,
            },
        )
    return ReplayOutcome(
        item.memory_id,
        SLEEP_CONFUSION,
        jaccard=codes[top1].jaccard(code),
        confused_with=top1,
        detail={"probe_jaccard": probe_jaccard},
    )


def replay_cluster(
    members: Sequence[MemoryItem],
    *,
    drop: float = DEFAULT_DROP,
    code_mode: str = W_MODE,
) -> tuple[ReplayOutcome, ...]:
    """在**这一簇自己的**基质上回放它的每一条，返回与 ``members`` 同序的结果。

    这就是 P4a 那个负数指向的补救：全库一个 W 在目标规模上补不出来，而一簇
    200 条以内是可以的。所以这里的 W 只装这一簇的码——簇的边界就是 W 的容量边界。

    **码在这里现算。** 不读 ``item.assoc_sdr``：那是图 A 的 ``hash`` 档，拿它建 W
    是一个恒死的算子，而它的输出看起来完全正常。

    Args:
        members: 簇里的条目，**按 :func:`partition` 给出的顺序**。返回值的顺序
            与它一致，而那个顺序决定了事件序列的顺序——``--replay`` 靠它逐条比对。
        drop: 探针丢掉多少比例的活跃位。见 :data:`DEFAULT_DROP`。
        code_mode: ``"idf"``（默认）或 ``"weight"``。``idf`` 用**这一簇成员**的
            文档频率现算（:func:`~memcore.sdr.encode_assoc_idf`），``weight`` 是
            P4c 原来的档，保留给 A/B 与历史事件的回放。

    Returns:
        与 ``members`` 等长、同序的 :class:`ReplayOutcome` 元组。

    Raises:
        ValueError: ``drop <= 0``，或 ``code_mode`` 不在 :data:`W_MODES` 里。
            **空转的探针会制造一个假的满分**：探针就是目标本身时补全恒为命中，
            辨认率 100%。量具里留着 ``--drop 0`` 是为了看"完美探针的天花板"
            （那是诊断），生产路径上没有这个需要。
    """
    if drop <= 0.0:
        raise ValueError(
            f"drop 必须 > 0，收到 {drop!r}——drop=0 时探针就是目标本身，"
            f"补全恒为命中，会量出一个假的 100%。"
        )
    if code_mode not in W_MODES:
        raise ValueError(f"code_mode 只认得 {W_MODES}，收到 {code_mode!r}。")

    codes: dict[str, SDR] = {}
    reasons: dict[str, str] = {}
    included: list[MemoryItem] = []
    lex_codes: list[SDR] = []
    for item in members:
        reason = _skip_reason(item)
        if reason is not None:
            reasons[item.memory_id] = reason
            continue
        # _skip_reason 已经保证了 lex_sdr 不是 None。
        assert item.lex_sdr is not None
        included.append(item)
        lex_codes.append(item.lex_sdr)

    if code_mode == "idf" and lex_codes:
        width = max(code.width for code in lex_codes)
        density = np.zeros(width, dtype=np.int64)
        for code in lex_codes:
            density[code.idx] += 1
        for item, lex in zip(included, lex_codes, strict=True):
            codes[item.memory_id] = encode_assoc_idf(lex, df=density, n=len(included), k=ASSOC_K)
    else:
        for item, lex in zip(included, lex_codes, strict=True):
            codes[item.memory_id] = encode_assoc(lex, k=ASSOC_K, mode="weight")

    substrate = Substrate.build(
        [
            _Coded(memory_id=item.memory_id, assoc_sdr=codes[item.memory_id])
            for item in members
            if item.memory_id in codes
        ],
        # 码已经显式算好，`SubstrateConfig` 的 mode 只在缺码时才会被用到；
        # 用 weight 是为了不给 `encode_assoc` 传一个它不认识的 `"idf"`。
        config=SubstrateConfig(k=ASSOC_K, mode="weight"),
    )

    outcomes: list[ReplayOutcome] = []
    for item in members:
        reason = reasons.get(item.memory_id)
        if reason is not None:
            outcomes.append(
                ReplayOutcome(
                    item.memory_id,
                    SKIP,
                    detail={"reason": reason, "code_mode": code_mode},
                )
            )
            continue
        code = codes[item.memory_id]
        probe = drop_bits(code, drop)
        outcome = _classify(item, substrate.complete(probe, top=1), codes, probe)
        outcomes.append(replace(outcome, detail={**outcome.detail, "code_mode": code_mode}))
    return tuple(outcomes)


# ── 一轮：谁参与、跑成什么样 ────────────────────────────────────────────────


def run_id_for(base_seed: int, at: float) -> str:
    """该轮巩固的标识。派生式与 :class:`~memcore.types.ConsolidationEvent` 写的一致。

    ``run_ts`` 用 ``at`` **完整的浮点值**（``!r`` 取的是能往返的最短表示，同一台
    机器上同一个 ``double`` 永远拼出同一个串）。早先那版取 ``int(at)``——理由是
    "同一秒内跑两次本来也该是同一轮"，而这句话被手跑实测推翻了：一秒内跑两轮会
    拿到**同一个** ``run_id``，于是第二轮的事件被追加进第一轮那个文件里，
    ``--replay`` 读出来是一份 60 行、哪一轮都对不上的记录，然后报"30 处不一致，
    这是 bug"——**而它是这个取整造出来的假警报**。两轮巩固是两轮：命令行连点两次
    就是两轮，各拍各的快照、各落各的事件。

    ``at`` 从事件文件读回来时是同一个浮点（JSON 的浮点在 Python 侧按 ``repr``
    往返），所以 ``--replay`` 仍能派生出一模一样的 ``run_id``。
    """
    material = f"{base_seed}|sleep|{at!r}"
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()[:16]


def run_seed_for(base_seed: int, at: float) -> int:
    """该轮的抽样种子。**重跑时用它复现同一批候选**。

    与 :func:`run_id_for` 同一份材料、不同的后缀，所以 ``run_seed`` 与 ``run_id``
    一一对应——``--replay`` 读出一轮的 ``run_id`` 就能取回它的 ``run_seed``。
    """
    material = f"{base_seed}|sleep|{at!r}|seed"
    return int.from_bytes(hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "big")


def _round_order(
    clusters: Sequence[tuple[str, ...]], *, run_seed: int, limit: int | None
) -> tuple[tuple[str, ...], ...]:
    """这一轮要看哪些簇。

    ``limit is None`` 时按 :func:`partition` 的**规范顺序**全看——那是 ``--replay``
    逐条比对的前提，也是默认档。给了 ``limit`` 时换个次序：

    * **次序由 ``run_seed`` 决定**（``blake2b(run_seed | 簇首成员的 id)``，取摘要
      升序）。真正的随机数在这里不能用——:mod:`memcore.sdr` 记着
      ``np.random.default_rng`` 不保证跨版本给出同一串数，一次 ``pip install -U
      numpy`` 就能让"重跑一遍"漂移。摘要由规范固定，跨版本、跨平台一样。
    * **整簇进、整簇出，按条数预算截断**。切半个簇是不行的：W 是按簇建的，少一条
      成员就换了一份 W，"辨认率"随之变成一个随 ``--limit`` 变化的数——那它与基线
      的 ``completion_top1`` 就不再可比，而两个数看起来都仍然正常。

    **这条与计划书里那句话有出入，记在这里**：计划书把 ``--limit`` 写作「回放多少
    条」，落成的是「本轮看多少条记忆」。差别在**谁决定看哪些簇**：计划书的读法会
    让抽样把 W 一起缩小，读法本条的抽样只决定"这一轮看哪儿"。取本条的读法，是因为
    前一种读法会让同一个库上的辨认率随 ``--limit`` 漂移。

    预算装不下任何一个簇时返回空元组——那意味着"这一轮什么都没看"，由调用方与
    :attr:`RunReport.recognition_rate` 的 ``None`` 如实报告，不是静默的空转。
    """
    if limit is None:
        return tuple(clusters)

    def _key(members: tuple[str, ...]) -> bytes:
        material = f"{run_seed}|{members[0]}"
        return hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()

    keyed = sorted(clusters, key=_key)
    chosen: list[tuple[str, ...]] = []
    spent = 0
    for members in keyed:
        if spent + len(members) > limit:
            continue
        chosen.append(members)
        spent += len(members)
    return tuple(chosen)


@dataclass(frozen=True)
class RunReport:
    """一轮巩固跑完之后的样子。**给调用方看的东西全在这里，别在别处再算一遍。**

    这是**纯计算的结果**，不含任何写入。真写库的是
    :meth:`~memcore.store.MemoryStore.bump_reinforce`，喂给它的正是
    :attr:`reinforced`——"改库改什么"这件事因此只有一处出口：只有 reinforce 的
    那批 id 会被列出来，而 store 侧只有 ``bump_reinforce`` 一个方法碰这个计数。

    Attributes:
        run_id: 该轮标识。事件文件名与 ``--replay`` 都认它。
        run_seed: 该轮抽样种子。
        at: 这一轮的时间（Unix 秒）。
        clusters: 逐簇的决策，**簇序即规范序**（``--limit`` 档下是抽样序）。
        missing_from_store: 图里有、库里没有的成员数。它们是**被彻底删除**的记忆
            （图是上次建图时的快照），不是错误——但也不能假装没看见：一个越来越大的
            这个数说明图该重建了。
    """

    run_id: str
    run_seed: int
    at: float
    clusters: tuple[tuple[ReplayOutcome, ...], ...]
    missing_from_store: int = 0

    @property
    def outcomes(self) -> tuple[ReplayOutcome, ...]:
        """所有决策，按簇序展平。事件落盘要的就是这个次序。"""
        return tuple(outcome for cluster in self.clusters for outcome in cluster)

    @property
    def replayed(self) -> int:
        """真回放了的条数——**辨认率的分母**。

        ``skip`` 不进分母，与基线的 :func:`~axo.bench.substrate.completion_summary`
        拿探针数当分母同一姿态：被跳过的条目根本没有探针，把它们算进分母等于
        把"没测"记成"没测中"。
        """
        return sum(1 for outcome in self.outcomes if outcome.decision != SKIP)

    @property
    def reinforced(self) -> tuple[str, ...]:
        """认出自己的那些 ``memory_id``，按簇序。**写库要的就是它。**"""
        return tuple(o.memory_id for o in self.outcomes if o.decision == REINFORCE)

    @property
    def recognition_rate(self) -> float | None:
        """回放辨认率 = 认出自己的条数 / 真回放了的条数。

        **一条都没真回放时返回 ``None``，不是 ``0.0``。** "这一轮没有数据"与
        "这一轮一条都没认出来"是两件完全不同的事，而 ``0.0`` 会让它们在报告里
        长得一模一样——然后有人会去调参数修一个根本不存在的问题。
        """
        total = self.replayed
        return None if total == 0 else len(self.reinforced) / total

    @property
    def skip_reasons(self) -> dict[str, int]:
        """``skip`` 的原因分布。**按原因分开数**，不然"跳了 12 条"没法判断该改谁。"""
        tally: dict[str, int] = {}
        for outcome in self.outcomes:
            if outcome.decision != SKIP:
                continue
            reason = str(outcome.detail.get("reason", "unknown"))
            tally[reason] = tally.get(reason, 0) + 1
        return tally


def consolidate(
    graph: GraphIndex,
    items: Iterable[MemoryItem],
    *,
    at: float,
    base_seed: int = BASE_SEED,
    drop: float = DEFAULT_DROP,
    max_cluster: int = MAX_CLUSTER,
    limit: int | None = None,
    code_mode: str = W_MODE,
) -> RunReport:
    """跑一轮：分簇 → 每簇建局部 W → 回放 → 汇总。

    **纯函数**：它不改库、不写文件、不拍快照。写库是调用方拿
    :attr:`RunReport.reinforced` 去调
    :meth:`~memcore.store.MemoryStore.bump_reinforce` 的事。这样切开有两个好处——
    这一层可以在没有库的情况下测（本仓的 ``--dry-run`` 也就成了"算完不写"这一步
    之差），而"改库改什么"这个决定只有一个出口。

    Args:
        graph: 关联图。它的 ``node_ids`` 是分簇的输入；图**过期**（建图之后才写的
            记忆不在里面）不是错误，那些记忆这一轮不参与，注释见下。
        items: 候选记忆。通常是 :meth:`~memcore.store.MemoryStore.iter_items` 的
            全部。**库里没有的图节点会被如实计数**，见 ``missing_from_store``。
        at: 这一轮的时间。``run_id`` / ``run_seed`` 由它与 ``base_seed`` 派生。
        base_seed: 见 :data:`BASE_SEED`。
        drop: 探针丢位比例，见 :data:`DEFAULT_DROP`。
        max_cluster: 簇上限，见 :data:`MAX_CLUSTER`。
        limit: 本轮最多看多少条记忆。``None`` 表示全看。见 :func:`_round_order`。
        code_mode: 局部 W 的码档，见 :data:`W_MODE`。默认 ``"idf"``；``"weight"``
            留给 A/B 与历史事件回放。

    Returns:
        :class:`RunReport`。

    Raises:
        ValueError: ``limit`` 小于 :data:`MIN_CLUSTER`——那种预算装不下任何一簇，
            与其安静地跑出一轮空转，不如当场拒掉。``drop`` 的校验在
            :func:`replay_cluster` 里。
    """
    if limit is not None and limit < MIN_CLUSTER:
        raise ValueError(
            f"limit 必须 >= MIN_CLUSTER（{MIN_CLUSTER}），收到 {limit!r}——"
            f"再小的预算装不下任何一个簇，这一轮会安静地什么都不做。"
        )

    by_id = {item.memory_id: item for item in items}
    ordered = _round_order(
        partition(graph, max_cluster=max_cluster),
        run_seed=run_seed_for(base_seed, at),
        limit=limit,
    )

    clusters: list[tuple[ReplayOutcome, ...]] = []
    missing = 0
    for members in ordered:
        present = [by_id[memory_id] for memory_id in members if memory_id in by_id]
        # 图是上次建图时的快照。建图**之后**写的记忆不在这里（这一轮不参与，
        # 下次建图后自然会有）；反过来"图里有、库里没有"只有一个来源——被彻底
        # 删除。两者都不该让巩固失败，但后者的数量要报出来：它涨上去就说明图该重建了。
        missing += len(members) - len(present)
        if not present:
            continue
        clusters.append(replay_cluster(present, drop=drop, code_mode=code_mode))

    return RunReport(
        run_id=run_id_for(base_seed, at),
        run_seed=run_seed_for(base_seed, at),
        at=at,
        clusters=tuple(clusters),
        missing_from_store=missing,
    )


# ── 事件：落盘与比对 ────────────────────────────────────────────────────────

#: 事件行的字段与**顺序**。顺序是规格的一部分：JSONL 一行一眼扫过去，
#: 前三个是"哪一轮、第几条"，中间是决策，``detail`` 在最后——它是唯一会长大的
#: 字段，放前面会把真正的决策挤到屏幕外。
EVENT_FIELDS: tuple[str, ...] = (
    "run_id",
    "run_seed",
    "seq",
    "memory_id",
    "decision",
    "at",
    "epoch",
    "jaccard",
    "confused_with",
    "detail",
)

#: 一轮只有一个 epoch（见计划里"不做多 epoch"）。字段留着是为了将来真的做多轮
#: 衰减时，事件格式不用改。
EPOCH = 0

#: ``jaccard`` 的比对容差。浮点写出去再读回来本该逐位相同（``repr`` 往返），
#: 留一点余量是给"将来某一版把 jaccard 换成 float32 存"这种改动。
_JACCARD_TOL = 1e-6

#: 逐条比对时**要比**的字段。``memory_id`` 与 ``decision`` 是主体，``seq`` 是
#: 顺序，``confused_with`` 是串扰指向谁，``epoch`` 目前恒 0 但比了才知道它变了。
#:
#: **刻意不比**：``at``（重跑的时间天然不同）、``run_id`` / ``run_seed``（由 ``at``
#: 派生，重跑必然不同——但抽样顺序若因此变了，会在 ``memory_id`` 那一列上显形）、
#: ``detail``（里面的 ``reinforce_before`` 是**库的当前状态**，重跑时库已经加过了；
#: 拿它比对等于要求"巩固过一次的库"与"没巩固过的库"给出同一份证据）。
_COMPARED = ("seq", "memory_id", "decision", "confused_with", "epoch")


def events_of(report: RunReport) -> tuple[ConsolidationEvent, ...]:
    """把一轮报告摊成 :class:`~memcore.types.ConsolidationEvent`，**按簇序**。

    这个类型在 P4c 之前是全仓零生产者的空壳。它就是事件的**契约**：字段名、
    取值表、``epoch`` 的语义都写在它的 docstring 里，所以这里不另起一套，只是
    给它填上值——落盘那一层再往契约外面补 ``run_seed`` 与 ``seq``（它们是**行的
    属性**，不是某一条决策的属性，见 :data:`EVENT_FIELDS`）。
    """
    return tuple(
        ConsolidationEvent(
            run_id=report.run_id,
            memory_id=outcome.memory_id,
            decision=outcome.decision,
            at=report.at,
            epoch=EPOCH,
            jaccard=outcome.jaccard,
            confused_with=outcome.confused_with,
            detail=dict(outcome.detail),
        )
        for outcome in report.outcomes
    )


def event_rows(report: RunReport) -> tuple[dict[str, Any], ...]:
    """整轮的事件行，``seq`` 从 0 起**连续**跨簇编号。

    ``seq`` 从 0 起是有意的：它是**下标**（第 0 条、第 1 条），不是**计数**
    （第 1 条、第 2 条）。下面 :func:`compare_events` 会拿它逐条对位，两种起点
    都能自洽，所以必须在某一处定死——这里定死，并由 ``test_..._seq_starts_at_zero``
    钉住，免得有一天改了起点、而已落盘的历史事件还按老起点排。
    """
    return _rows_of(events_of(report), run_seed=report.run_seed)


def cluster_event_rows(report: RunReport, index: int) -> tuple[dict[str, Any], ...]:
    """只取第 ``index`` 簇的行，``seq`` 与整轮的一致。

    存在的理由是**崩溃时的形状**：真跑时每簇"改库 + 追加事件"在同一次循环里
    完成，中途挂掉留下的就是一个 ``seq`` 连续的前缀——库改了前 k 簇，事件也就
    只记了前 k 簇，两者对得上。整轮写完才落盘的话，崩在中途会得到"库改了、
    事件没有"这种**对不上账**的状态。
    """
    events = events_of(report)
    start = sum(len(cluster) for cluster in report.clusters[:index])
    return _rows_of(
        events[start : start + len(report.clusters[index])],
        run_seed=report.run_seed,
        seq_from=start,
    )


def _rows_of(
    events: Sequence[ConsolidationEvent], *, run_seed: int, seq_from: int = 0
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for offset, event in enumerate(events):
        rows.append(
            {
                "run_id": event.run_id,
                "run_seed": run_seed,
                "seq": seq_from + offset,
                "memory_id": event.memory_id,
                "decision": event.decision,
                "at": event.at,
                "epoch": event.epoch,
                "jaccard": event.jaccard,
                "confused_with": event.confused_with,
                "detail": event.detail,
            }
        )
    return tuple(rows)


def append_events(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """把若干事件行**追加**到 ``path``，返回这一批写了几行。

    追加而不是重写：一轮的事件是分簇写出去的（见 :func:`cluster_event_rows`），
    而"重写"会让第二次追加把第一次的覆盖掉。

    **不写文件头**——每行自足（``run_id`` / ``run_seed`` / ``seq`` 都在行里）。
    文件被截断、被人在中间删掉一行时，剩下的行仍然可解析；写文件头的话，
    头一丢整个文件就废了，而它是审计证据。

    父目录由这里建（``mkdir(parents=True, exist_ok=True)``）：调用方拿到的是
    ``paths.consolidation_events_path()``，那个函数只解析位置、不做 I/O（paths 模块
    的纪律），所以"目录还不存在"是常态而不是错误。
    """
    materialised = tuple(rows)
    if not materialised:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in materialised:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(materialised)


def read_events(path: Path) -> tuple[dict[str, Any], ...]:
    """读一份事件文件。**坏行当场报错**，不当成空行跳过。

    读不出来就意味着审计链断了——那正是这个文件存在的全部意义。安静跳过坏行
    会让 ``--replay`` 拿一份短了的记录去比对，然后报"序列不一致"，把责任推给
    巩固的确定性，而真正的问题是这个文件。

    Raises:
        MemCoreError: 文件不存在，或是某一行不是合法的 JSON 对象。
    """
    if not path.exists():
        raise MemCoreError(f"没有这一轮的事件文件：{path}。run_id 写错了吗？")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MemCoreError(
                f"{path} 第 {number} 行不是合法 JSON：{exc}。这一行是坏数据，"
                f"继续读下去只会把损坏扩散到比对结果里。"
            ) from exc
        if not isinstance(row, dict):
            raise MemCoreError(
                f"{path} 第 {number} 行反序列化后是 {type(row).__name__}，不是对象。"
                f"事件行永远是一个对象——即使它只有一个字段。"
            )
        rows.append(row)
    return tuple(rows)


def compare_events(
    recorded: Sequence[dict[str, Any]], replayed: Sequence[dict[str, Any]]
) -> tuple[str, ...]:
    """逐条比对**已落盘的一轮**与**刚重跑出来的一轮**，返回不一致的描述。

    返回空元组 = 复现成功。返回值是给人看的字符串列表，不是 ``bool``：一轮巩固
    不可复现是要去查的，而"第 37 条：记录的 decision 是 reinforce、重跑是
    sleep_confusion"比"不一致"有用得多。

    比什么、不比什么见 :data:`_COMPARED`——那里也写了每一处"不比"的理由。
    """
    problems: list[str] = []
    for index in range(max(len(recorded), len(replayed))):
        if index >= len(recorded):
            problems.append(f"第 {index} 条：记录里没有，重跑多出来了")
            continue
        if index >= len(replayed):
            problems.append(f"第 {index} 条：记录里有，重跑没跑出来")
            continue
        old, new = recorded[index], replayed[index]
        for field_name in _COMPARED:
            if old.get(field_name) != new.get(field_name):
                problems.append(
                    f"第 {index} 条：{field_name} 记录是 {old.get(field_name)!r}、"
                    f"重跑是 {new.get(field_name)!r}"
                )
        old_j, new_j = old.get("jaccard"), new.get("jaccard")
        if old_j is None and new_j is None:
            continue
        if old_j is None or new_j is None or abs(old_j - new_j) > _JACCARD_TOL:
            problems.append(f"第 {index} 条：jaccard 记录是 {old_j!r}、重跑是 {new_j!r}")
    return tuple(problems)
