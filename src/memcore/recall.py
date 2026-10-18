"""混合召回：从全库挑出这几条注入上下文。

三条通道，后两条都可以缺席::

    词法通道   query 的词法 SDR × 条目的词法 SDR 的 Jaccard
               —— 不依赖任何模型，换 embedding 后行为逐位不变
    语义通道   query 向量与库里向量的余弦（暴力全扫）
               —— 依赖当前 space_id，换模型后重算
    联想通道   把与查询词面沾边的条目当种子，沿关联图 A 扩散一步
               —— 需要调用方注入一张图（``graph=``），**默认权重是 0**

通道算完，加权融合，再用**结算过惰性衰减的权重**调制，然后过阈值、MMR
去冗、token 预算贪心。每一步的结果都记进 :class:`memcore.audit.RecallLedger`。

**联想通道默认关**（``RecallConfig.graph_weight`` 是 0），而且它需要调用方先建好图
并用 ``graph=`` 传进来——:mod:`memcore` 不做 I/O，图从哪来是上层的事（照
``query_vector`` 的先例）。给一个非零权重会让 ``min_score`` 不再对应原来的严格程度，
所以"开不开"是一个标定问题，实测数在 ``benchmarks/gold/CORPUS.md`` 的联想腿那页。

**没有图、或权重为 0 时，扩散根本不算**（不是算了不用）：这样"关着"与"这个功能还不
存在"在账本上逐位相同。

**MMR 去冗用的是词法 SDR 的 Jaccard，不是 embedding 余弦。** 这一条是刻意的：
用余弦去冗，换一个 embedding 模型就会得到不同的去冗结果，"同一段对话在不同
模型下装进上下文的东西不一样"——那正是本项目要消灭的现象。

**这个函数只读，唯一的写是"把用过的记忆记一笔"**（``touch_access`` 与惰性衰减
的结算，可用 ``settle=False`` 关掉）。召回不应该产生用户看不见的副作用。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .audit import (
    ChannelScores,
    RecalledMemory,
    RecallLedger,
    RejectedCandidate,
    RejectReason,
)
from .errors import SDRCapacityError, SpaceNotFoundError, StoreError
from .graph import GraphIndex
from .sdr import encode_lexical
from .store import MemoryStore
from .temporal_window import parse_time_window
from .timefmt import format_when
from .tokens import estimate_tokens
from .types import SDR, MemoryItem, Tier, now_ts

__all__ = ["RecallConfig", "recall"]

#: 权重调制的地板。最终得分 = 融合相似度 × (地板 + (1-地板)·权重)。
#:
#: 为什么不直接用权重乘：一条刚写的记忆权重是 0.5，直接被砍半，那么"相似的
#: 新记忆"会永远输给"不太相似但被访问过很多次的旧记忆"。加了地板之后，权重
#: 最多造成 1/0.6 ≈ 1.67 倍的倾斜——足以在相似度接近时决定胜负，不足以让一条
#: 低相似度的老旧记忆挤掉高相似度的新记忆。
_WEIGHT_FLOOR = 0.6

#: 少于这么多条向量时不做语义对比度归一化（见 :func:`_contrast`）。
_CONTRAST_MIN_SAMPLES = 8

#: 归一化除法的分母下限。基线逼近 1.0 时 ``1 - baseline`` 会趋近 0，把整段
#: 缩放拉到极端。留 5% 的余量，最坏情况下放大 20 倍——而不是放大到无穷。
_CONTRAST_MIN_HEADROOM = 0.05


@dataclass(frozen=True)
class RecallConfig:
    """召回的调参。**这些数字是初值，不是标定值。**

    标定它们的是 ``tools/bench_recall.py``——**工具已落地**（P1.5）：它在一份
    200 条记忆 / 76 条查询的金标准语料上跑完整召回，报 hit@K、MRR、误注入率、
    AUC 四项指标，并与冻结基线比对::

        uv run python tools/bench_recall.py --no-semantic          # 只跑词法通道
        uv run python tools/bench_recall.py --no-semantic --check  # 与基线比对
        uv run python tools/bench_recall.py --help                 # 其余开关

    **但本版只报告、不改默认值**：下面每一个数字都还是初值，**没有一个是靠它
    标定出来的**。金标准语料是合成的、n≈76，它够格当"改了参数有没有变坏"的
    相对闸门，不够格当"这个值是最好的"的依据——那要在真实库上扫。之所以把它们
    集中在一个 frozen dataclass 里，就是让标定变成"换一组参数重跑"，而不是
    "去代码里找那几个魔数"。
    """

    lexical_weight: float = 0.9
    """词法通道的**可靠度**（见 :func:`_fuse`：得分先乘可靠度，再噪声或合并）。

    取自 ``tools/bench_space.py`` 在本机 40 对中文短句上的实测 AUC：词法 **0.884**，
    nomic-embed-text **0.760**。0.9 是"按实测排序质量的保守折算"——不取 1.0，因为
    AUC 0.88 离完美排序还有距离。**这是 n=20 的一组样本给出的数，不是定论**；
    P1.5 的 ``bench_recall.py`` 已在金标准语料上跑通（``--no-semantic`` 即纯词法
    基线，实测数记在 ``benchmarks/gold/CORPUS.md``），但**本版没有据此改这个数**：
    合成语料给不出"该取 0.9 还是 0.85"的答案，它只能告诉你改动有没有让它变坏。
    """

    semantic_weight: float = 0.45

    time_weight: float = 1.0
    """时间窗通道的权重。**默认给满**，这是刻意的。

    「昨天我说了什么」里的「昨天」不是形容词，是**范围限定**：窗内的那一批就是
    用户要的东西，窗内的条目按定义该排在窗外的前面。给 0.6 这种半截权重的话，
    一条窗外但词面撞车的记忆照样压得过窗内那条，症状是「她把前天的事说成昨天的」。

    窗外的条目照常打分：窗里一条都没有时（那天确实什么都没记）不至于召回空手。
    """
    """语义通道的可靠度。**只有词法的一半**，理由见上。

    这一条直接决定"语义通道单独能把一条记忆顶过阈值吗"：一条只有语义命中、
    对比度归一化后得 0.22 的记忆，融合值是 ``0.22 × 0.45 = 0.099``，仍然低于
    阈值——它必须比词法命中强一倍才有同等话语权。考虑到实测 AUC 只有 0.76
    （瞎猜是 0.5），这个保守程度是数据支持的，不是拍脑袋。

    **为什么可靠度是绝对施加而不是归一化到最强通道**：归一化会让"只启用一条
    通道"与"两条都启用"下的孤立命中拿到同一个分数，于是语义通道单独命中 0.16
    就能进来，和词法一样容易——而那正是实测否定的东西。
    """

    graph_weight: float = 0.0
    """联想通道的**可靠度**。**默认 0 = 关。**

    **这不是「没做」，是量出来的「不划算」。** 2026-10-17 在金标准语料（200 条记忆 /
    76 条查询）上扫了四档（``tools/bench_recall.py --no-semantic --graph-weight
    0,0.25,0.5,1``）：

    ========  =======  =======  =======  =======
    指标      w=0.00   w=0.25   w=0.50   w=1.00
    ========  =======  =======  =======  =======
    hit@8     0.9412   0.9265   0.9265   0.9265
    MRR       0.8975   0.8939   0.8939   0.8914
    误注入率  0.8672   0.8793   0.8812   0.8843
    AUC       0.9093   0.8921   0.8757   0.8450
    ========  =======  =======  =======  =======

    **每一档都比关着差，而且 w 越大越差**；AUC 在 w=0.5 上就跌破了门槛的 0.8893
    下限。P4b 那次在基质语料上的结论（0.8250 vs 0.8500）是同一个方向。

    原因是**这张图与语料都还不是真的**：边是词法相似度的 top-m，语料是人造句子。
    所以它是一个标定问题、不是一个实现问题：等边换成真的语义关联、语料换成真实
    记忆后再扫一次。在那之前，默认关是有数支撑的，不是惯性。
    """

    graph_decay: float = 0.5
    """扩散一步的衰减。**与 :attr:`graph_weight` 是两件事。**

    这里管"传得动多远"（沿边每走一步乘多少），上面那个管"这条通道说的话有多算数"。
    与 :func:`axo.bench.substrate.graph_outcomes` 的 ``decay`` 同值——
    量具与生产要量的是同一个算子，各调各的就等于在量两个东西。
    """

    min_score: float = 0.10
    """融合得分低于此值即淘汰。

    标定链：``tools/bench_space.py --sweep`` 在 40 对标注样本上给出词法 Jaccard
    的最优阈值 **0.12**（召回 15/20，误报 2/20，Youden's J）。换算到融合量纲要
    除以词法可靠度 0.9（0.12 × 0.9 ≈ 0.108），再考虑到权重调制对一条权重 0.5
    的新记忆额外乘 0.8——最终取 **0.10**，对应原始 Jaccard 约 0.139，**略高于
    实测最优点**，也就是略微偏向精确（少注入一条无关注记忆，而不是多注入）。

    账本里保留每个通道的分数，就是为了让"这个阈值定得对不对"可以被核对——它是
    一个**标定过的初值**，不是一个定律。
    """

    redundancy_threshold: float = 0.70
    """与已入选条目的词法 Jaccard 超过此值即判为重复。

    0.70 的由来：两条记忆的长度相近时，词法 SDR 的全部活跃位里重合七成，意味
    着它们几乎共享同一批字与二元组——那是同一件事被说了两遍，不是两件相关的
    事。低于这个值就开始误杀"同一主题的不同侧面"。
    """

    mmr_lambda: float = 0.75
    """MMR 的相关性/多样性权衡。1.0 = 完全按得分排，0.0 = 只挑最不一样的。"""

    max_items: int | None = None
    """本轮最多注入几条。**默认 ``None`` = 不限条数。**

    以前默认是 8，那是自设的额度：用户要的是「无论哪一个对话都要能够使用完整的
    记忆」，而"一次最多 8 条"正是"能回想起的记忆少"的直接原因之一。现在条数只受
    剩余额度约束（窗口装不下才丢），丢谁由得分决定、账本如实说。

    显式给一个数仍然有效——记忆浏览器的探针给 20，那是**浏览页的大小**，不是上下
    文额度（``memory/browse.py`` 的 ``PROBE_CONFIG``）。"""

    candidate_limit: int = 200
    """进入 MMR 的候选上限（按融合得分取前 N）。

    MMR 是 O(选中数 × 候选数) 的两两比较，不限流的话全库两万条就是 16 万次
    Jaccard。截断到 200 之后，一次召回的开销与库的大小无关。
    """

    max_rejected: int = 40
    """账本里最多记几条被淘汰的候选（按得分从高到低取）。账本要给人看，
    五百条"分数不够"不是解释，是噪声。"""


#: 纯词法召回时倒排索引最多 gather 多少条候选。
#:
#: 只影响**扫描成本**，不改变融合/MMR 的语义；语义/时间/图通道存在时走全量扫描，
#: 因为它们的命中可能没有词法重叠。5,000 是在 50k 真实文本放大语料上、top-1 与
#: 全量一致的前提下取的值（见 sweep_results/axo_recall_rare_bit）。
_PREFILTER_LIMIT = 5000

#: 各通道候选并集的上限；超过它就退回全量扫描（时间窗很大时）。
_CANDIDATE_UNION_LIMIT = 20000


@dataclass
class _Candidate:
    """扫描阶段的内部候选。**不进账本**，账本里只有最终形态。"""

    item: MemoryItem
    weight: float
    """结算过惰性衰减的权重。"""

    tokens: int
    scores: ChannelScores
    sdr: SDR | None


def recall(
    store: MemoryStore,
    *,
    query_text: str,
    budget_tokens: int,
    query_vector: np.ndarray | None = None,
    space_id: str = "",
    config: RecallConfig | None = None,
    exclude_ids: Iterable[str] = (),
    now: float | None = None,
    settle: bool = True,
    graph: GraphIndex | None = None,
) -> RecallLedger:
    """召回要注入上下文的记忆。

    Args:
        store: 记忆库。
        query_text: 用户这一轮的输入。词法通道的输入，也是唯一必需的。
        budget_tokens: 这些记忆总共能占多少 token。**不含**装配器加在它们外面的
            说明文字与消息模板开销——那是 :mod:`axo.session.budget`
            的账。
        query_vector: 用户输入在 ``space_id`` 下的 embedding。给 ``None`` 就
            只用词法通道——**这是一种正常状态，不是降级**：没装嵌入模型时
            系统必须照常可用。
        space_id: 语义通道的记忆空间。与 ``query_vector`` 同时给出才生效。
        config: 调参。``None`` 用默认值。
        exclude_ids: 不参与召回的 id（通常是在逐字稿里已经出现过的）。
        now: 时间旅行的入口（测试用）。
        settle: 是否把"这些被用了"记进库（结算衰减 + 访问计数）。默认要。**探针式
            调用（记忆界面里的「以这条为线索召回一次」）必须传 ``False``**：那是
            "看一眼"，不是"想起来"。不关掉的话，用户每点一次探索按钮，被看到的
            那几条就会显得更常用、衰减得更慢——**一次观察改变了被观察的东西**，
            而真实召回的排序从此带着探索的痕迹，没有任何地方能看出这一点。
        graph: 关联图。``None`` 时联想通道整条不存在——**这是默认，也是正常状态**。
            图必须由调用方建好传进来，:mod:`memcore` 不做 I/O（照 ``query_vector``
            的先例）。配 :attr:`RecallConfig.graph_weight` 使用：权重是 0 时即便传了
            图，扩散也**根本不算**，这样"关着开关"与"没有这个功能"在账本上逐位相同。

    Returns:
        一份完整的账本，含入选、淘汰与原因。

    Raises:
        StoreError: ``query_vector`` 的维度与登记的空间不符。
    """
    started = time.perf_counter()
    settings = config or RecallConfig()
    at = now_ts() if now is None else now
    excluded = set(exclude_ids)
    notes: list[str] = []

    query_lex = _query_sdr(query_text, notes)
    # 时间窗跟着 now 走：now 是这条路径上唯一的时间旅行入口，而「昨天」不跟着
    # 它走的话，同一句查询在今天和明天召回的东西不一样，测试写不出来。
    window = parse_time_window(query_text, datetime.fromtimestamp(at))
    window_ids: set[str] | None = None
    if window is not None:
        since, until = window
        window_ids = {item.memory_id for item in store.items_created_between(since, until)}
        notes.append(
            f"时间窗 {format_when(since)} – {format_when(until)}（窗内 {len(window_ids)} 条）"
        )
    scores_by_id = _semantic_channel(
        store, query_vector=query_vector, space_id=space_id, notes=notes
    )
    graph_scores = _graph_channel(
        store,
        query_lex=query_lex,
        graph=graph if settings.graph_weight > 0.0 else None,
        decay=settings.graph_decay,
    )
    if settings.graph_weight > 0.0 and graph_scores is None:
        # **不写这句，开了开关没反应就是静默的。** 而"我明明打开了"是最难自查的
        # 一类问题：界面上开关是勾着的，召回里确实什么都没发生。
        reason = (
            "还没有可用的关联图（还没建，或库太大没有自动建）"
            if graph is None or len(graph) == 0
            else "本轮输入没有词法编码，扩散没有种子"
        )
        notes.append(f"联想通道已启用，但{reason}，本轮它不参与")
    channel_names = (
        (("temporal",) if window_ids is not None else ())
        + (("lexical",) if query_lex is not None else ())
        + (("semantic",) if scores_by_id is not None else ())
        + (("graph",) if graph_scores is not None else ())
    )

    candidate_ids = _prefilter_ids(
        store,
        query_lex=query_lex,
        scores_by_id=scores_by_id,
        graph_scores=graph_scores,
        window_ids=window_ids,
    )
    candidates, rejected, scanned, live = _scan(
        store,
        query_lex=query_lex,
        scores_by_id=scores_by_id,
        graph_scores=graph_scores,
        window_ids=window_ids,
        excluded=excluded,
        at=at,
        settings=settings,
        notes=notes,
        candidate_ids=candidate_ids,
    )

    injected, more_rejected, used = _select(
        candidates, budget_tokens=budget_tokens, settings=settings
    )
    rejected.extend(more_rejected)

    # 只有真的被注入的条目才写库：一次召回给八条记忆各写一行的代价，会在每轮
    # 对话里重复一次。淘汰的候选连权重都不必结算——它们的值不会影响任何结果。
    if settle:
        _settle(store, injected, at)

    rejected.sort(key=lambda candidate: -candidate.score)

    return RecallLedger(
        query_preview=_preview(query_text),
        scanned=scanned,
        channels=channel_names,
        space_id=space_id if scores_by_id is not None else "",
        injected=tuple(injected),
        rejected=tuple(rejected[: settings.max_rejected]),
        tokens_used=used,
        tokens_budget=budget_tokens,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        notes=tuple(notes) + (() if live else ("全部候选都已遗忘",)),
    )


# ── 通道 ────────────────────────────────────────────────────────────────────


def _query_sdr(query_text: str, notes: list[str]) -> SDR | None:
    """把输入编成词法 SDR。全标点/全空白的输入没有词法通道。"""
    try:
        return encode_lexical(query_text)
    except SDRCapacityError:
        notes.append("输入不含可编码字符，本轮无词法通道")
        return None


def _semantic_channel(
    store: MemoryStore,
    *,
    query_vector: np.ndarray | None,
    space_id: str,
    notes: list[str],
) -> tuple[dict[str, float] | None, tuple[str, ...]]:
    """算全库的语义得分。返回 ``{memory_id: 余弦}``，不可用时返回 ``None``。"""
    if query_vector is None or not space_id:
        if query_vector is not None and not space_id:
            notes.append("给了 query_vector 却没有 space_id，语义通道未启用")
        return None

    # ``SpaceRegistry.get`` 未登记时**抛异常**而不是返回 None（那是刻意的：往未登记
    # 的空间写向量必须当场被拒）。但读路径不一样——问一个还没登记的空间，答案是
    # "这一轮没有语义通道"，不是"召回失败"。这里的 try 就是那两种意图的分界。
    #
    # 曾经写成 ``if space is None``，于是这个分支**永远不会执行**：它看起来像在
    # 处理"空间没登记"，实际行为是把 SpaceNotFoundError 抛到调用方，而调用方要么
    # 没接、要么接成了"整轮召回失败"——两种情况都比"少一条通道"糟得多。
    try:
        space = store.registry.get(space_id)
    except SpaceNotFoundError:
        notes.append(f"记忆空间 {space_id!r} 未登记，本轮只有词法通道")
        return None

    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size != space.dim:
        raise StoreError(
            f"查询向量是 {query.size} 维，而空间 {space_id!r} 登记的是 {space.dim} 维。"
            f"这两者对不上说明当前用的嵌入模型与库里的向量不是同一个——"
            f"**不能靠截断或补齐蒙混过去**，那样算出来的相似度没有意义。"
        )

    ids, matrix = store.normalized_embedding_matrix(space_id)
    if not ids:
        notes.append(f"空间 {space_id!r} 里还没有任何向量，本轮只有词法通道")
        return None

    # 矩阵已按 embedding revision 缓存并归一化；查询侧仍需归一化。
    sims = matrix @ _normalized(query[None, :])[0]
    # 负余弦是"意思相反"，不是"负相关"。融合得分要当权重用，放任负值进来会
    # 让一条词法强命中的记忆被语义项拖到阈值以下。
    sims = np.clip(sims, 0.0, 1.0)
    adjusted = _contrast(sims, notes)
    # strict=True：三个序列都源自同一次查询，长度不等说明列序错位了。静默截断
    # 会把 A 的余弦挂到 B 的 id 上——那是**看起来完全正常**的错误召回。
    return {
        memory_id: (float(raw), float(norm))
        for memory_id, raw, norm in zip(ids, sims, adjusted, strict=True)
    }


def _contrast(sims: np.ndarray, notes: list[str]) -> np.ndarray:
    """把余弦从"绝对相似度"换算成"相对本库的突出程度"。

    **为什么必须做这一步**：自然语言上余弦有很高的基线——两段毫无关系的中文也
    常有 0.3~0.5。不做处理的话，语义通道会变成一个"什么都通过"的通道，阈值拦
    不住它，而它还会通过融合把词法通道的判断一起抬过去。一个通道如果对所有输入
    都给高分，它就不携带任何信息。

    基线取自**本次查询对本库全体条目的中位数**，不是一个常数：中位数是"这个查询
    一般能问到多像的东西"的估计，减掉它之后留下的才是峰值。用中位数而不是均值，
    是因为均值会被少数极高的相似度拉高——于是"有一条真的很像"反而把那一条自己
    压下去，方向正好反了。

    样本太少时（库还小）**不做归一化**：三条记忆里挑不出"一般水平"。此时原样
    返回余弦，并记一条备注——让"这一轮没有对比度归一化"在账本里可见。
    """
    if sims.size < _CONTRAST_MIN_SAMPLES:
        if sims.size:
            notes.append(
                f"本空间只有 {sims.size} 条向量，少于 {_CONTRAST_MIN_SAMPLES} 条，"
                f"本轮不做语义对比度归一化"
            )
        return sims
    baseline = float(np.median(sims))
    headroom = max(1.0 - baseline, _CONTRAST_MIN_HEADROOM)
    return np.clip((sims - baseline) / headroom, 0.0, 1.0)


def _normalized(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。零范数的行原样留下（乘出来是 0，不会产生 NaN）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def _preview(text: str, limit: int = 40) -> str:
    """账本里留的输入摘要。压掉换行，免得侧栏那一行被撑开。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


# ── 扫描 ────────────────────────────────────────────────────────────────────


def _graph_channel(
    store: MemoryStore,
    *,
    query_lex: SDR | None,
    graph: GraphIndex | None,
    decay: float,
) -> dict[str, float] | None:
    """沿关联图扩散一步，返回 ``{memory_id: 联想得分}``。不该算时返回 ``None``。

    **返回 ``spread``，不是 ``base + spread``。** ``base`` 就是各条记忆与查询的词法
    Jaccard，而词法通道（:func:`_lexical_score`）已经带着它进过一次融合了；再算一遍
    等于把同一份证据记两遍。``spread`` 的语义恰好是"**不是本条命中，而是从它的邻居
    传过来的证据**"。

    **种子不过滤 pinned / excluded / COLD。** ``base`` 是图的属性（哪些节点与查询词面
    沾边），过滤是**候选池**的事、发生在 :func:`_scan`。副作用是"本会话刚说的话"也能
    当种子把邻居拉进来——这在联想语义下正是想要的效果。这是个**显式决定**。

    代价：这一趟是**第二次** ``store.iter_items()``（它解码 SDR）。权重为 0 时
    :func:`recall` 连调都不调它，所以关着开关的人付不到这个代价。
    """
    if graph is None or query_lex is None or len(graph) == 0:
        return None

    lex_of: dict[str, SDR] = {}
    for item in store.iter_items():
        if item.lex_sdr is not None:
            lex_of[item.memory_id] = item.lex_sdr

    base = np.zeros(len(graph), dtype=np.float64)
    for row, node_id in enumerate(graph.node_ids):
        source = lex_of.get(node_id)
        # 宽度对不上的节点**不给种子**：两套不可比的表示算出来的 Jaccard 是垃圾，
        # 而它会以"看起来正常"的分数混进扩散（照 _lexical_score 的同一口径）。
        if source is None or source.width != query_lex.width:
            continue
        base[row] = source.jaccard(query_lex)

    spread = graph.diffuse(base, decay=decay)
    return {
        node_id: float(spread[row])
        for row, node_id in enumerate(graph.node_ids)
        if spread[row] > 0.0
    }


def _prefilter_ids(
    store: MemoryStore,
    *,
    query_lex: SDR | None,
    scores_by_id: dict[str, float] | None,
    graph_scores: dict[str, float] | None,
    window_ids: set[str] | None,
) -> set[str] | None:
    """纯词法/多通道的候选并集。返回 ``None`` = 退回全量扫描。"""
    if query_lex is None and scores_by_id is None and graph_scores is None and window_ids is None:
        return None
    # 有「没有任何词法码」的条目时，候选路径看不到它们，而它们必须以
    # NO_ENCODING 出现在账本里（数据问题要可见）。
    if store.unencoded_count() > 0:
        return None

    ids: set[str] = set()
    if query_lex is not None:
        ids.update(store.lexical_index().candidates(query_lex, limit=_PREFILTER_LIMIT))
    if scores_by_id is not None:
        semantic = sorted(scores_by_id.items(), key=lambda pair: pair[1][1], reverse=True)[
            :_PREFILTER_LIMIT
        ]
        ids.update(memory_id for memory_id, _scores in semantic)
    if window_ids is not None:
        ids.update(window_ids)
    if graph_scores is not None:
        ids.update(memory_id for memory_id, score in graph_scores.items() if score > 0.0)
    if len(ids) > _CANDIDATE_UNION_LIMIT:
        return None
    return ids


def _scan(
    store: MemoryStore,
    *,
    query_lex: SDR | None,
    scores_by_id: dict[str, float] | None,
    graph_scores: dict[str, float] | None,
    window_ids: set[str] | None,
    excluded: set[str],
    at: float,
    settings: RecallConfig,
    notes: list[str],
    candidate_ids: set[str] | None = None,
) -> tuple[list[_Candidate], list[RejectedCandidate], int, bool]:
    """走一遍全库，算出每条候选的融合得分。

    返回 ``(候选, 淘汰记录, 扫描条数, 是否有活着的条目)``。

    **扫描期间不写库。** :meth:`MemoryStore.iter_items` 在迭代期间持有库锁，
    而惰性衰减的结算是一次 UPDATE——在打开的读游标上写同一个库，行为依赖
    SQLite 的内部状态。所以扫描阶段只用纯函数
    :meth:`MemoryItem.decayed_weight`，写回留到选完之后。

    **为什么保持单趟、不改成"先 gather 再 fuse"**：那会在融合前把全库的
    ``MemoryItem``（含解码后的 SDR）留在内存里。5 万条的峰值是量级估算、**未实测**，
    而这是给"关着开关的多数人"平白加的回归。多出来的那趟留在
    :func:`_graph_channel` 里，只在开关打开时才付。
    """
    lexical_weight = settings.lexical_weight
    semantic_weight = settings.semantic_weight
    # 权重表**按有没有图通道来建**，不是每轮判断：``_fuse`` 对 ``fired`` 里每个名字
    # 都要在权重表里找到键。只在确实算过扩散时才加这一项，是为了让"没开"与"没有
    # 这个功能"走到完全一样的连乘上去。
    weights = {"lexical": lexical_weight, "semantic": semantic_weight}
    if window_ids is not None:
        weights["temporal"] = settings.time_weight
    if graph_scores is not None:
        weights["graph"] = settings.graph_weight
    pool: list[_Candidate] = []
    rejected: list[RejectedCandidate] = []
    scanned = 0
    live = False

    if candidate_ids is None:
        iterator = store.iter_items()
    else:
        iterator = store.get_many(sorted(candidate_ids))
    for item in iterator:
        if item.pinned or item.memory_id in excluded:
            continue
        scanned += 1

        if item.tier is Tier.COLD:
            rejected.append(_reject(item, RejectReason.COLD, 0.0))
            continue
        live = True

        lexical = _lexical_score(query_lex, item)
        pair = None if scores_by_id is None else scores_by_id.get(item.memory_id)
        semantic = None if pair is None else pair[1]
        temporal = 1.0 if window_ids is not None and item.memory_id in window_ids else None

        # 时间窗命中**不因为缺编码被挡在外面**：用户问的是那段时间里有什么，而缺
        # 编码这件事在任何不带时间词的轮次里都会以 NO_ENCODING 冒出来，瞒不住。
        if lexical is None and semantic is None and temporal is None:
            rejected.append(_reject(item, RejectReason.NO_ENCODING, 0.0))
            continue

        graph = None if graph_scores is None else graph_scores.get(item.memory_id)
        fired: dict[str, float] = {}
        if temporal is not None:
            fired["temporal"] = temporal
        if lexical is not None:
            fired["lexical"] = lexical
        if semantic is not None:
            fired["semantic"] = semantic
        if graph is not None:
            fired["graph"] = graph
        fused = _fuse(fired, weights)

        weight = item.decayed_weight(at, half_life_days=item.tier.half_life_days)
        # ``ranking`` 是**排序用的那个分**：融合分再按可信度/新旧打折。
        ranking = fused * (_WEIGHT_FLOOR + (1.0 - _WEIGHT_FLOOR) * weight)

        # **去留看融合分，不看打了折的那个。** 权重是「这条有多可信、多新」，它该
        # 影响**排序**，不该影响**要不要进上下文**——一条两小时前的 working 记忆
        # 权重会衰减到 0（WORKING 的半衰期是 0.08 天），而它可能正是唯一答得上来的
        # 一条。用 ``ranking`` 判阈值时它要再打六折才够得着 ``min_score``，于是
        # 「完整记忆」在旧记忆上名不副实。
        #
        # 阈值本身的量纲因此变了（严格程度略降）：``min_score`` 是按打折后的分标定
        # 的，现在直接比融合分。这一点由金标准门槛盯着——它跌了会红。
        if fused < settings.min_score:
            rejected.append(
                _reject(
                    item,
                    RejectReason.BELOW_THRESHOLD,
                    fused,
                    detail=_score_detail(lexical, semantic, graph, temporal=temporal),
                )
            )
            continue

        pool.append(
            _Candidate(
                item=item,
                weight=weight,
                tokens=estimate_tokens(item.content),
                scores=ChannelScores(
                    temporal=temporal or 0.0,
                    lexical=lexical or 0.0,
                    semantic=semantic or 0.0,
                    semantic_raw=0.0 if pair is None else pair[0],
                    graph=graph or 0.0,
                    fused=ranking,
                ),
                sdr=item.lex_sdr,
            )
        )

    if not live and scanned:
        notes.append("库里的条目全部已遗忘或缺少编码")
    elif candidate_ids is not None and not scanned:
        notes.append("词法候选为空：没有与查询共享词法位的记忆")
    return pool, rejected, scanned, live


def _fuse(fired: dict[str, float], weights: dict[str, float]) -> float:
    """把各通道的得分合成一个数：**噪声或** ``1 − ∏(1 − r_c·s_c)``。

    不用加权平均。这件事值得写清楚，因为平均看起来更自然：

    平均假设两条通道是**对同一个量的两次测量**，一次高一次低就该往中间靠。但
    词法通道与语义通道不是这样——它们是**两条独立的线索**，各自都可能单独想起
    同一条记忆。用平均的话，一条词法已经明确命中的记忆（0.25）只要语义通道没
    有同感（0.00），就会被拉到 0.1375 掉到阈值以下。**打开语义通道反而让召回
    变差**，这不是调参能修好的，是算子选错了。

    噪声或在概率论里正是"多个独立原因各自可能导致同一结果"的合成方式，它有三条
    我们需要的性质：

    * **单调**：任何一条通道的得分上升，融合值上升；
    * **只增不减**：给一条候选增加一条通道，融合值只会变大——所以"换一个嵌入
      模型"永远不会让它原本认得的东西变得不认得；
    * **一致增益**：两条通道都命中时高于任何单独一条。

    权重 ``w_c`` 是**绝对可靠度**，直接乘在得分上。这一条是实测逼出来的：若把
    可靠度归一化到最强通道（``w_c / max(w)``），只启用一条通道时它永远拿满分，
    于是"语义通道单独命中"与"词法通道单独命中"一样容易把记忆顶过阈值——而
    ``tools/bench_space.py`` 量出两者的排序质量分别是 0.760 与 0.884。**可靠度
    必须是绝对的**，才能让弱通道需要更强的证据。

    代价是没有"两条通道都开时最强通道仍拿满分"这个性质。那是可以接受的：词法
    通道的贡献在两种配置下都等于 ``0.9 × s``，仍然不随配置漂移；变的只是阈值
    本身要以 ``0.9`` 为量纲标定（见 :attr:`RecallConfig.min_score`）。
    """
    if not fired:
        return 0.0
    survival = 1.0
    for name, score in fired.items():
        survival *= 1.0 - score * weights[name]
    return 1.0 - survival


def _lexical_score(query_lex: SDR | None, item: MemoryItem) -> float | None:
    """词法通道得分。任一侧没有 SDR 就没有这个得分。"""
    if query_lex is None or item.lex_sdr is None:
        return None
    if item.lex_sdr.width != query_lex.width:
        # 宽度不同无法比较。这不是"相似度为零"，是两套不可比的表示——静默返回
        # 0 会让一次宽度变更表现为"召回质量突然下降"。
        return None
    return item.lex_sdr.jaccard(query_lex)


def _score_detail(
    lexical: float | None,
    semantic: float | None,
    graph: float | None = None,
    temporal: float | None = None,
) -> str:
    parts = []
    if temporal is not None:
        parts.append(f"时间 {temporal:.2f}")
    if lexical is not None:
        parts.append(f"词法 {lexical:.2f}")
    if semantic is not None:
        parts.append(f"语义 {semantic:.2f}")
    if graph is not None:
        parts.append(f"联想 {graph:.2f}")
    return " · ".join(parts)


def _reject(
    item: MemoryItem, reason: RejectReason, score: float, *, detail: str = ""
) -> RejectedCandidate:
    return RejectedCandidate(
        memory_id=item.memory_id,
        content_preview=_preview(item.content, limit=60),
        reason=reason,
        score=score,
        detail=detail,
    )


# ── 选择 ────────────────────────────────────────────────────────────────────


def _jaccard_matrix(pool: list[_Candidate]) -> np.ndarray:
    """候选两两的 Jaccard 矩阵，一次性算完。

    MMR 每一步都要问"这个候选与已选里最像的那条有多像"；逐对调
    :meth:`SDR.jaccard` 会让 ``candidate_limit=200`` 退化成约 4 万次
    ``np.intersect1d``（50k 库上实测 4–8 秒，见 2026-10-18 的 profiling）。
    这里把候选的活跃位铺成 ``(n, width)`` 的 0/1 矩阵，一次矩阵乘得到交集计数；
    ``SDR.jaccard`` **刻意忽略权重**，所以 0/1 矩阵与它同义。

    宽度不一致的候选（正常不会发生）按 0 相似度处理，与
    :func:`_max_jaccard` 原来的跳过行为一致。
    """
    width = max((candidate.sdr.width for candidate in pool if candidate.sdr is not None), default=0)
    matrix = np.zeros((len(pool), width), dtype=np.float32)
    for row, candidate in enumerate(pool):
        if candidate.sdr is not None and candidate.sdr.width == width:
            matrix[row, candidate.sdr.idx] = 1.0
    cardinality = matrix.sum(axis=1)
    intersection = matrix @ matrix.T
    union = cardinality[:, None] + cardinality[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def _select(
    pool: list[_Candidate],
    *,
    budget_tokens: int,
    settings: RecallConfig,
) -> tuple[list[RecalledMemory], list[RejectedCandidate], int]:
    """按 MMR + token 预算贪心地选。

    MMR（最大边际相关）每一步在"得分高"与"跟已选的不像"之间取最大：

        ``mmr(c) = λ·score(c) − (1−λ)·max_jaccard(c, 已选)``

    预算被装不下时**不退出循环**而是继续看下一条——剩下的候选里有更短的，
    它可能装得下。"按得分从高到低，装不下就停"会让一条 800 token 的记忆把
    后面所有 60 token 的候选一起挡在门外。

    **相似度用预计算的矩阵，不逐对算。** ``max_sim`` 维护"每个候选与已选的最大
    Jaccard"；选中一条后用矩阵的一行做一次 ``maximum`` 更新。语义与原逐对版本
    相同，只是把 O(k²) 次 Python/numpy 调用换成一次矩阵乘。
    """
    pool.sort(key=lambda candidate: -candidate.scores.fused)
    pool = pool[: settings.candidate_limit]
    if not pool:
        return [], [], 0

    jaccard = _jaccard_matrix(pool)
    scores = np.asarray([candidate.scores.fused for candidate in pool], dtype=np.float64)
    max_sim = np.zeros(len(pool), dtype=np.float64)
    alive = np.ones(len(pool), dtype=bool)

    selected: list[_Candidate] = []
    rejected: list[RejectedCandidate] = []
    used = 0

    # ``max_items`` 为 None 时只受额度约束——见它自己的注释。
    while alive.any() and (settings.max_items is None or len(selected) < settings.max_items):
        values = settings.mmr_lambda * scores - (1.0 - settings.mmr_lambda) * max_sim
        values = np.where(alive, values, -np.inf)
        index = int(np.argmax(values))
        candidate = pool[index]
        redundancy = float(max_sim[index])
        alive[index] = False

        if redundancy > settings.redundancy_threshold:
            rejected.append(
                _reject(
                    candidate.item,
                    RejectReason.REDUNDANT,
                    candidate.scores.fused,
                    detail=f"与已入选的相似度 {redundancy:.2f}",
                )
            )
            continue

        if used + candidate.tokens > budget_tokens:
            rejected.append(
                _reject(
                    candidate.item,
                    RejectReason.OVER_BUDGET,
                    candidate.scores.fused,
                    detail=f"需要 {candidate.tokens}，剩余 {max(0, budget_tokens - used)}",
                )
            )
            continue

        selected.append(candidate)
        used += candidate.tokens
        max_sim = np.maximum(max_sim, jaccard[index])

    for index, candidate in enumerate(pool):
        if alive[index]:
            rejected.append(_reject(candidate.item, RejectReason.MAX_ITEMS, candidate.scores.fused))

    injected = [
        RecalledMemory(
            memory_id=candidate.item.memory_id,
            content=candidate.item.content,
            kind=str(candidate.item.kind),
            tier=str(candidate.item.tier),
            weight=candidate.weight,
            tokens=candidate.tokens,
            scores=candidate.scores,
            pinned=candidate.item.pinned,
            session_id=candidate.item.session_id,
            created_at=candidate.item.created_at,
            span=_span_of(candidate.item),
            derived_from=_derived_of(candidate.item),
        )
        for candidate in selected
    ]
    return injected, rejected, used


def _span_of(item: MemoryItem) -> tuple[float, float] | None:
    """归并产物的（起、止）时刻，从 ``meta["span"]`` 取；不是归并产物就是 ``None``。

    **取不到就当没有。** ``meta`` 是自由格式的 JSON 对象，脏数据不该让整轮召回
    失败——那会变成「这一轮没有记忆」，离原因很远（同 ``_row_to_item`` 对坏
    ``meta_json`` 的态度）。
    """
    raw = item.meta.get("span")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        start, end = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if start > 0 else None


def _derived_of(item: MemoryItem) -> tuple[str, ...]:
    """归并产物代表的原句 id；不是归并产物就是空元组。

    与 :func:`_span_of` 同一条纪律：``meta`` 是自由格式的，读不出来就当没有。
    """
    raw = item.meta.get("derived_from")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(each) for each in raw)


# ── 写回 ────────────────────────────────────────────────────────────────────


def _settle(store: MemoryStore, injected: list[RecalledMemory], at: float) -> None:
    """把"这些被用了"记进库：结算衰减 + 访问计数。

    读回来的条目再写一次而不是直接 UPDATE 权重，是因为 :meth:`MemoryStore.
    apply_decay` 自己要做"变化太小就不写"的判断——那个判断依赖库里的原始
    时间戳，不能由调用方替它做。
    """
    ids: list[str] = []
    for memory in injected:
        item = store.get(memory.memory_id)
        if item is None:
            # 扫描与选完之间被删了。并发下可能发生（用户正在记忆浏览器里删条目），
            # 但绝不可能是常态——这是"召回结果指向一条不存在的记忆"的唯一入口。
            continue
        store.apply_decay(item, at)
        ids.append(memory.memory_id)
    store.touch_access(ids, at)
