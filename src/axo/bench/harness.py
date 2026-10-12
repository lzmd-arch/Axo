"""把语料跑成证据：建库 → 逐条 recall → 汇总成 :class:`QueryOutcome`。

**库在内存里，而且是一次性的**（``MemoryStore(":memory:")``，``store.py:110`` 明确
它是合法路径）。这条不是图快，是图**不碰用户的东西**：基准脚本会被 pre-commit 与
CI 每次提交都跑一遍，它要是读了 ``AXO_DATA`` 指向的真库，就会有"跑一次
基准把用户的记忆权重动了一遍"这种事故——而且它不会有任何症状，因为召回只写访问
计数与衰减结算，账面看起来一切正常。所以这里既不碰磁盘，也不碰环境变量。

**每一轮都是 ``settle=False``。** 理由与探针式召回相同（``recall.py:174-180``）：
基准要量的是"库原样不动时召回会给出什么"。若开着结算，第一遍跑完权重就变了，
第二遍跑出来的数就不一样——:func:`axo.bench.harness.store_fingerprint`
就是用来把这条钉成断言的。

**"现在"是固定的**（:data:`~axo.bench.goldset.CREATED_AT`）。惰性衰减
依赖"现在几点"，不固定就没有可复现性可言。建库时同时把 ``last_decay_ts`` 也设成
它，于是 ``at <= last_decay_ts`` 成立、衰减因子恒为 1——语料里写的权重是多少，
召回看到的权重就是多少。
"""

from __future__ import annotations

import hashlib

import numpy as np

from axo.memory.backfill import Embedder, sync_embeddings
from memcore.audit import RejectReason
from memcore.errors import SDRCapacityError
from memcore.graph import GraphIndex
from memcore.recall import RecallConfig, recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

from .goldset import CREATED_AT, GoldMemory, GoldSet
from .metrics import QueryOutcome

__all__ = [
    "build_store",
    "census_config",
    "census_scores",
    "run",
    "store_fingerprint",
    "structural_violations",
]

#: 普查调用的 token 预算。写成"足够大"而不是"语料总长"：后者要求先给全库正文算
#: 一遍 token，而那正是我们要让 ``recall`` 自己去做的事——在这里再实现一次估算，
#: 就会出现两处会漂移的 token 账。
_CENSUS_BUDGET = 10**9


# ── 建库 ────────────────────────────────────────────────────────────────────


def build_store(
    memories: tuple[GoldMemory, ...] | list[GoldMemory],
    *,
    embedder: Embedder | None = None,
) -> MemoryStore:
    """把语料灌进一个全新的 ``:memory:`` 库。

    词法 SDR 在建库时就算好（``encode_lexical``，纯 blake2b 哈希，不调模型），
    所以**不给 ``embedder`` 时这套基准完全离线**——它量到的是词法通道的天花板，
    那是一个有意义且可复现的数。

    Args:
        memories: 语料里的记忆。
        embedder: 可选的嵌入后端。给了就给全库补向量、启用语义通道。它的形状
            是 :class:`~axo.memory.backfill.Embedder`——**只有三个成员
            的窄协议**，所以这个模块不必认识 httpx，也不必认识 Ollama。

    Raises:
        SDRCapacityError: 某条正文里没有一个可编码字符。**不吞**，并且把 id 报出来
            ——那是一条语料 bug，静默跳过会让"这条永远是 0 分"看起来像召回问题。
    """
    store = MemoryStore(":memory:")
    for gold in memories:
        try:
            lex = encode_lexical(gold.content)
        except SDRCapacityError as exc:
            raise SDRCapacityError(
                f"语料 {gold.memory_id} 的正文编不出词法 SDR：{exc}。"
                f"金标准语料里不该有条目连一个可编码字符都没有——请修语料，"
                f"而不是让它在基准里静默地永远得 0 分。"
            ) from exc
        store.add(
            MemoryItem(
                memory_id=gold.memory_id,
                kind=MemoryKind(gold.kind),
                tier=Tier(gold.tier),
                content=gold.content,
                taint=gold.taint,
                created_at=gold.created_at,
                # 两个时间戳在这里显式对齐到 created_at。``MemoryItem.__post_init__``
                # （``types.py:377-380``）本来就会把 ≤0 的它们拉平到 created_at，所以
                # 这一行**不是**在补一个会在别处发生的 bug；写出来是为了让"基准里衰减
                # 因子恒为 1、权重就是语料里那个数"这条约定在建库这一处读得出来，
                # 而不必让读者追进 types.py 才知道。
                last_access_ts=gold.created_at,
                last_decay_ts=gold.created_at,
                weight=gold.weight,
                pinned=gold.pinned,
                session_id=gold.session_id,
                lex_sdr=lex,
            )
        )
    if embedder is not None:
        sync_embeddings(store, embedder)
    return store


def store_fingerprint(store: MemoryStore) -> str:
    """库状态的摘要。**只取召回可能改写的那几列。**

    用来断言"跑一遍基准没有改变库"：召回若忘了关 ``settle``，改的就是权重、
    访问计数与两个时间戳。把这些列（而不是全表）纳入摘要，是为了让这个指纹
    描述"跑基准会不会留下痕迹"，而不是"库文件的字节"。全表摘要会被 schema
    字段顺序之类的无关变动扰动，于是它红的时候没人知道是不是真出事了。

    行按 ``memory_id`` 排序（``iter_items`` 本来就是），所以两遍的结果可比。
    """
    digest = hashlib.sha256()
    for item in store.iter_items():
        digest.update(
            "\x1f".join(
                (
                    item.memory_id,
                    f"{item.weight:.12f}",
                    str(item.access_count),
                    f"{item.last_access_ts:.6f}",
                    f"{item.last_decay_ts:.6f}",
                )
            ).encode("utf-8")
        )
        digest.update(b"\x1e")
    return digest.hexdigest()


# ── 普查：拿全库的融合得分 ──────────────────────────────────────────────────


def census_config(corpus_size: int, base: RecallConfig) -> RecallConfig:
    """把受测配置"放开"成一次能看见全库得分的调用。**通道权重逐字段照抄。**

    真实配置下账本只留注入项与前 ``max_rejected=40`` 条淘汰项，被 ``min_score``
    早早砍掉的条取不到分——而 AUC 需要**每一条**的 ``fused``。所以放开的只有
    选择与阈值那几个旋钮（``min_score``、``redundancy_threshold``、``mmr_lambda``、
    ``max_items``、``candidate_limit``、``max_rejected``），**通道权重一个都不动**：
    动了它们，量到的就不是"受测配置的打分函数"了。

    **``max_items`` 必须是 1，不是 ``corpus_size``。** 第一版写成
    ``corpus_size``，于是 MMR 每一步都要扫整个候选池——200 条库上单次查询就是
    约 4×10⁶ 次 Jaccard，整个扫描跑不完（探针就是这么挂住的）。写成 1 之后
    ``_select`` 只走一步，**其余全部以 ``MAX_ITEMS`` 的形态落进 rejected 并带着
    完整的 ``fused``**——这正好是我们想要的全库得分表，而复杂度从 O(n³) 掉到
    O(n log n)。

    代价是那条被选中的 entries 走的是 ``injected`` 而不是 ``rejected``，所以
    :func:`census_scores` 要把两边合起来看。
    """
    size = max(1, corpus_size)
    return RecallConfig(
        lexical_weight=base.lexical_weight,
        semantic_weight=base.semantic_weight,
        graph_weight=base.graph_weight,
        # **衰减也照抄。** 它不是"选择旋钮"而是算子的一部分：普查若用了另一个
        # 衰减，量到的 ``fused`` 就不是受测配置的打分函数——而 AUC 声称量的正是那个。
        graph_decay=base.graph_decay,
        min_score=0.0,
        redundancy_threshold=1.0,
        mmr_lambda=1.0,
        max_items=1,
        candidate_limit=size,
        max_rejected=size,
    )


def census_scores(
    store: MemoryStore,
    query_text: str,
    base: RecallConfig,
    *,
    query_vector: np.ndarray | None = None,
    space_id: str = "",
    now: float = CREATED_AT,
    graph: GraphIndex | None = None,
) -> dict[str, float]:
    """全库的 ``{memory_id: fused}``。

    只有 :func:`~axo.bench.metrics.auc` 用它。**缺席即 0.0**（在
    :func:`~axo.bench.metrics.auc` 里按 0.0 取）：``pinned`` 条目在
    扫描期就被跳过，连通道都没进——"它不在这一轮的候选里"与"它得了 0 分"在
    排序意义上是同一件事。

    纯函数式的：``settle=False``，跑完库一个字节都不变。

    ``graph`` 只在 :attr:`RecallConfig.graph_weight` 为正时才有意义；为 0 时传了也
    不会被算（``recall`` 会短路）。**这里不自己建图**——图与权重无关，建一次就够，
    而那属于调用方（见 :func:`run`）。
    """
    size = store.count()
    ledger = recall(
        store,
        query_text=query_text,
        budget_tokens=_CENSUS_BUDGET,
        query_vector=query_vector,
        space_id=space_id,
        config=census_config(size, base),
        now=now,
        settle=False,
        graph=graph,
    )
    scores = {memory.memory_id: memory.scores.fused for memory in ledger.injected}
    for rejected in ledger.rejected:
        if rejected.reason is RejectReason.MAX_ITEMS:
            scores[rejected.memory_id] = rejected.score
    return scores


# ── 跑 ──────────────────────────────────────────────────────────────────────


def run(
    goldset: GoldSet,
    *,
    config: RecallConfig | None = None,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
    now: float = CREATED_AT,
    graph: GraphIndex | None = None,
) -> list[QueryOutcome]:
    """逐条查询跑一遍，返回证据。**顺序与 ``goldset.queries`` 一致。**

    Args:
        goldset: 语料。
        config: 受测配置。``None`` 用 ``RecallConfig()`` 的默认值——基准默认量
            的就是出厂配置。
        embedder: 给库补向量、给查询算向量的后端。``None`` = 纯词法。
        store: 复用一个已经建好的库。``None`` 就现建一个（每次调用都是一份全新的
            内存库，所以两次调用互不影响）。
        now: "现在几点"。默认 :data:`CREATED_AT`，与建库时的时间戳一致。
        graph: 关联图。**权重为 0 时连看都不看它**——多传一张图不会改变任何结果
            （``recall`` 会短路），但也不会白建一张。只有权重为正、而调用方没给图时
            才在这里现建一张，见下。

    Returns:
        每条查询一个 :class:`QueryOutcome`，指标函数只吃它。

    **图的建立条件写得很紧，是刻意的**：权重为 0 时一张都不建。这个函数的调用方
    通常在一个循环里跑好几个权重（见 ``tools/bench_recall.py --graph-weight``），
    若每次都给建，扫四档就是建四张一模一样的图。所以工具那边先建一张传进来；
    这里的兜底只服务于"单跑一次、没准备图"的调用方。
    """
    settings = config or RecallConfig()
    if store is None:
        store = build_store(goldset.memories, embedder=embedder)
    if graph is None and settings.graph_weight > 0.0:
        graph = GraphIndex.build(store.iter_items())

    vectors = _query_vectors(embedder, goldset)
    space_id = embedder.space_id if embedder is not None else ""

    outcomes: list[QueryOutcome] = []
    for index, query in enumerate(goldset.queries):
        vector = None if vectors is None else vectors[index]
        ledger = recall(
            store,
            query_text=query.query,
            budget_tokens=query.budget,
            query_vector=vector,
            space_id=space_id,
            config=settings,
            exclude_ids=query.exclude,
            now=now,
            settle=False,
            graph=graph,
        )
        outcomes.append(
            QueryOutcome(
                query_id=query.query_id,
                category=query.category,
                injected=tuple(ledger.injected_ids),
                expect=query.expect,
                optional=query.optional,
                never=query.never,
                scores=census_scores(
                    store,
                    query.query,
                    settings,
                    query_vector=vector,
                    space_id=space_id,
                    now=now,
                    graph=graph,
                ),
                tokens_used=ledger.tokens_used,
                tokens_budget=ledger.tokens_budget,
            )
        )
    return outcomes


def _query_vectors(embedder: Embedder | None, goldset: GoldSet) -> np.ndarray | None:
    """一次把全部查询编完。**一次调用而不是逐条**——后端自己分片，逐条调会把
    它的批处理整个绕过去。"""
    if embedder is None or not goldset.queries:
        return None
    matrix = np.asarray(
        embedder.embed([query.query for query in goldset.queries]), dtype=np.float32
    )
    if matrix.shape[0] != len(goldset.queries):
        raise ValueError(
            f"嵌入后端返回了 {matrix.shape[0]} 条查询向量，而查询有 "
            f"{len(goldset.queries)} 条。数量对不上说明向量与查询的对应关系已经断了"
            f"——按下标硬配会把 A 的向量挂到 B 上，而相似度照样算得出一个数。"
        )
    return matrix.reshape(len(goldset.queries), -1)


# ── 结构不变量 ──────────────────────────────────────────────────────────────


def structural_violations(goldset: GoldSet, outcomes: list[QueryOutcome]) -> list[str]:
    """返回违例说明；**必须为空**。空列表是通过，不是"没检查"。

    这里查的四条与四项指标是**两种语言的断言**，不能混在一起聚合：

    * ``pinned`` 的条目进了注入（人格锚是 pinned，让它参与常规召回是设计事故）；
    * ``tier="cold"`` 的条目进了注入（已遗忘的东西被想起来）；
    * 查询 ``exclude`` 掉的 id 进了注入（逐字稿里已经有的又注入一遍）；
    * ``tokens_used > tokens_budget``（硬预算被突破）。

    前三条在 :func:`memcore.recall._scan` 里是"跳过"，第四条在 :func:`_select`
    里是"装不下换下一条"——都是**永真**的性质，不是统计倾向。

    **为什么不能只靠平均值兜住**：这些一旦发生就是 bug，而 bug 会在均值里被
    平均掉。200 条语料里混进一条人格锚，hit@1 可能还涨了 0.005——数字更好看，
    系统更坏。所以它们是 ``== []``，不进任何聚合。

    第四条还有一个只有基准会在意的理由：预算被突破时，多出来的那几条仍然算
    "注入了"，于是误注入率的分母里混进了本来不该存在的东西——量具自己先坏了。
    """
    pinned = {gold.memory_id for gold in goldset.memories if gold.pinned}
    cold = {gold.memory_id for gold in goldset.memories if gold.tier == "cold"}

    violations: list[str] = []
    for query, outcome in zip(goldset.queries, outcomes, strict=True):
        injected = set(outcome.injected)
        for label, forbidden in (
            ("pinned", pinned),
            ("cold", cold),
            ("exclude", set(query.exclude)),
        ):
            overlap = sorted(injected & forbidden)
            if overlap:
                violations.append(f"{query.query_id} 注入了 {label} 的条目：{overlap}")
        if outcome.tokens_used > outcome.tokens_budget:
            violations.append(
                f"{query.query_id} 的 token {outcome.tokens_used} 超出预算 {outcome.tokens_budget}"
            )
    return violations
