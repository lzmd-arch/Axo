"""召回基准：金标准语料 + 四项指标 + 可复现的运行器。

**这一层是量具，不是功能。** 它回答的问题只有一个："改了一个参数之后，召回是
变好了还是变坏了"。在它存在之前，``memcore.recall.RecallConfig`` 的九个参数
全是初值，改任何一个都只能靠读代码判断。

为什么单独一个包而不是塞进 :mod:`axo.memory`：读路径与写路径都不该
知道"我们正在被评测"。量具要有能力**观察**被测对象而不被它观察——所以这里的
每一行都只调 :func:`memcore.recall.recall` 的公开签名，不从库里偷内部状态。

**它不碰 Qt、不碰 HTTP**，与 ``memory/``、``session/`` 同一条纪律
（``tests/test_layering.py`` 的 ``_PURE_LAYERS`` 钉住）。嵌入后端是以一个窄协议
（:class:`axo.memory.backfill.Embedder`）传进来的，所以语义腿有没有
都跑得动；没有嵌入时这套基准量到的是**纯词法通道**的天花板，那是一个有意义且
完全离线的数。

模块地图::

    corpus.py    三份语料共用的读盘：指纹 + 逐行 JSON + 禁词表（不碰库）
    goldset.py   召回语料：GoldMemory / GoldQuery / 指纹 / 目录定位
    metrics.py   四项指标的纯函数（不碰库、不碰盘）
    harness.py   建 :memory: 库 → 逐条 recall → 汇总成 QueryOutcome
    routes.py    路由语料：Utterance / 指纹 / 目录定位
    routeeval.py 逐条路由 → Outcome → 准确率/混淆/结构不变量（不碰库、不碰盘）
    substrate.py 关联基质语料与量具（图 A 的边权、W 的补全、k / scale 扫描）
"""

from __future__ import annotations

from .corpus import PERSONAL_DENYLIST, file_sha256, personal_identifier_hits, read_jsonl
from .goldset import (
    CREATED_AT,
    GOLD_WEIGHT,
    GoldMemory,
    GoldQuery,
    GoldSet,
    corpus_digests,
    default_gold_dir,
    load_goldset,
    load_memories,
    load_queries,
)
from .harness import (
    build_store,
    census_config,
    census_scores,
    run,
    store_fingerprint,
    structural_violations,
)
from .metrics import (
    QueryOutcome,
    auc,
    false_injection_rate,
    hard_negative_auc,
    hit_at_k,
    mrr,
    mrr_answered,
    no_answer_injections,
    recall_at_k,
    summarize,
)
from .routeeval import (
    Outcome,
    evaluate,
    lexicon_violations,
    per_category,
    route_structural_violations,
    route_summary,
)
from .routes import (
    REGISTERS,
    RouteCorpus,
    Utterance,
    default_routes_dir,
    load_corpus,
    load_utterances,
    routes_digests,
)
from .substrate import (
    CodedMemory,
    CompletionOutcome,
    GraphOutcome,
    NeighborPairs,
    SubstrateCorpus,
    SubstrateMemory,
    SubstrateQuery,
    build_graph,
    coded_memories,
    completion_outcomes,
    completion_summary,
    default_substrate_dir,
    drop_bits,
    graph_outcomes,
    graph_summary,
    load_substrate_corpus,
    load_substrate_memories,
    load_substrate_queries,
    neighbor_auc,
    neighbor_pairs,
    neighbor_precision_at_m,
    neighbor_summary,
    probe_sample,
    scaled_memories,
    substrate_digests,
    substrate_structural_violations,
)

# 路由那边的两个函数带 ``route_`` 前缀，**不是风格偏好，是这一层逼出来的**：
# ``structural_violations`` 与 ``summarize`` 已经被召回侧占了（来自 .harness 与
# .metrics）。同名再导入一次不会报错——它只会**静默顶替**前面那个，而调用方拿到的是
# 另一个模块的函数（症状是一条指向无关属性的 ``AttributeError``）。曾经真的发生过，
# 见 :mod:`axo.bench.routeeval` 的模块文档。

__all__ = [
    "CREATED_AT",
    "GOLD_WEIGHT",
    "PERSONAL_DENYLIST",
    "REGISTERS",
    "CodedMemory",
    "CompletionOutcome",
    "GoldMemory",
    "GoldQuery",
    "GoldSet",
    "GraphOutcome",
    "NeighborPairs",
    "Outcome",
    "QueryOutcome",
    "RouteCorpus",
    "SubstrateCorpus",
    "SubstrateMemory",
    "SubstrateQuery",
    "Utterance",
    "auc",
    "build_graph",
    "build_store",
    "census_config",
    "census_scores",
    "coded_memories",
    "completion_outcomes",
    "completion_summary",
    "corpus_digests",
    "default_gold_dir",
    "default_routes_dir",
    "default_substrate_dir",
    "drop_bits",
    "evaluate",
    "false_injection_rate",
    "file_sha256",
    "graph_outcomes",
    "graph_summary",
    "hard_negative_auc",
    "hit_at_k",
    "lexicon_violations",
    "load_corpus",
    "load_goldset",
    "load_memories",
    "load_queries",
    "load_substrate_corpus",
    "load_substrate_memories",
    "load_substrate_queries",
    "load_utterances",
    "mrr",
    "mrr_answered",
    "neighbor_auc",
    "neighbor_pairs",
    "neighbor_precision_at_m",
    "neighbor_summary",
    "no_answer_injections",
    "per_category",
    "personal_identifier_hits",
    "probe_sample",
    "read_jsonl",
    "recall_at_k",
    "route_structural_violations",
    "route_summary",
    "routes_digests",
    "run",
    "scaled_memories",
    "store_fingerprint",
    "structural_violations",
    "substrate_digests",
    "substrate_structural_violations",
    "summarize",
]
