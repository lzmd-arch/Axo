"""词法 SDR —— 一个**不依赖任何模型**的文本编码器。

这个模块是「记忆独立于模型」这条承诺在代码里的第一个承重点。它的输出只由输入
文本的字节决定，与装了哪个 embedding 模型、跑在什么硬件上、numpy 是什么版本
全都无关。换模型时语义通道（:mod:`memcore.recall` 里的余弦项）会重算，而词法
SDR 逐位不变——因此建在它上面的去冗、MMR、以及将来的关联基质都活得下来。

**为什么必须自己写，而不是用 numpy 的随机数生成器**：``np.random.default_rng``
**不保证跨版本给出同一串随机数**（NumPy 的明确政策：生成器的流可以随版本变，
``RandomState`` 才是冻结的）。用它来派生"哪个词激活哪些位"，意味着一次
``pip install -U numpy`` 就能让全库的词法 SDR 静默漂移——召回质量下降，而
没有任何一处会报错。所以位的派生只用 :mod:`hashlib` 里的 blake2b：它的输出由
规范固定，跨版本、跨平台、跨解释器都稳定。

编码方法（哈希 SDR 的常规做法）::

    文本 → 规范化 → 切 token（字/词 + 相邻二元组）→ 每个 token 哈希出若干位
         → 按位累计权重 → 取权重最高的 k 位 → 归一化成 SDR

权重的值域是 ``(0, 1]``：最高频的那一位是 1.0，其余按比例。**不要对它调
``SDR.binary()`` 的默认阈值**（见 :class:`memcore.types.SDR`）。

**"两个尺度"在本模块有两条不同的轴，别混为一谈。** 仓里两处 docstring 各说了
一半，读起来像同一件事：

* **按模型依赖性分**：词法 SDR（:func:`encode_lexical`，模型无关）与语义 SDR
  （依赖当前 embedding 模型，属于某个 ``space_id``）。后者到今天还没有编码器。
* **按稀疏度分**：词法码 ``k = 573``（``s = 0.14``，计划书 §4.3）与关联码
  ``k = 64``（:func:`encode_assoc`，喂神经元级基质 W）。

``README.md`` 把「记忆的编码分两个尺度」写在模型依赖性那一轴上，而阶段表里 P4 的
「SDR 双尺度」落的是稀疏度那一轴。两条都成立，但把它们说成一条，会让"双尺度已经
做完"与"语义尺度还不存在"同时看起来为真。
"""

from __future__ import annotations

import hashlib

import numpy as np

from .errors import SDRCapacityError
from .text import tokenize
from .types import DEFAULT_SDR_WIDTH, SDR

__all__ = ["ASSOC_K", "LEXICAL_K", "drop_bits", "encode_assoc", "encode_lexical"]

LEXICAL_K = 573
"""词法 SDR 的默认活跃位数。

4096 × 0.14 ≈ 573，取自计划书 §4.3 的稀疏度 ``s = 0.14``。这个数在本项目里
被用在它**本来**的位置——相似度、去冗、以及关联基质的地基。神经元级基质 W 用的是
另一档更稀疏的编码（:data:`ASSOC_K`）；那一档的出处不在计划书里，见 :data:`ASSOC_K`。
"""

_BITS_PER_TOKEN = 8
"""每个 token 激活多少位。

标定依据：一条典型记忆条目（约 40 个汉字 → 约 13 个 token 编码后的 26 个）在
本参数下第一轮就产出约 200 个活跃位，两三轮即可填满 k=573。设得太大则长文本
的位冲突变多、区分度下降；太小则短文本要跑很多轮。
"""

_MAX_ROUNDS = 128
"""最多做几轮扩充。

存在的理由：文本很短时（一个词、一个字），一轮的位不够 k 个。此时按轮次再哈希
一批新位补上，直到攒够 k 个或触顶。上限纯粹是防呆——单 token 文本约 72 轮即可
填满，128 是留给非拉丁文字与极短输入余量。
"""

_PERSON_PREFIX = b"memcore-lex"
"""blake2b 的 ``person`` 参数前缀，限长 16 字节，这里占 11 字节。"""


# 切分（tokenize）与字符分类在 :mod:`memcore.text` 里——那是编码器与预算器
# 共用的判断，放在这里会让 :mod:`memcore.tokens` 不得不从一个"编码器模块"里
# import 一个与编码无关的工具函数。


# ── 位派生 ──────────────────────────────────────────────────────────────────


def _bits_for(token: str, round_index: int, *, width: int, count: int) -> np.ndarray:
    """一个 token 在第 ``round_index`` 轮激活的位索引。

    **取 16 位无符号整数再对 width 取模**，而不是取 64 位再取模：当 ``width``
    是 2 的幂且不超过 65536 时，``65536 % width == 0``，取模是**均匀**的，没有
    模偏置。width 不是 2 的幂时会有轻微偏置（最大偏差 ``width / 65536``），
    对相似度计算没有实际影响。
    """
    person = _PERSON_PREFIX + round_index.to_bytes(2, "little")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=count * 2, person=person).digest()
    return np.frombuffer(digest, dtype=">u2").astype(np.int64) % width


def encode_lexical(
    text: str,
    *,
    width: int = DEFAULT_SDR_WIDTH,
    k: int = LEXICAL_K,
) -> SDR:
    """把文本编成词法 SDR。

    **同一个字符串永远得到逐位相同的 SDR**，与 numpy 版本、平台、进程无关。
    这条性质由 ``tests/test_model_agnostic.py`` 钉住。

    Raises:
        SDRCapacityError: 文本不含任何可编码字符（全是标点或空白），或者
            ``width`` / ``k`` 不合法。
    """
    if width < 1:
        raise SDRCapacityError(f"SDR 宽度必须 >= 1，收到 {width!r}。")
    if k < 1 or k > width:
        raise SDRCapacityError(f"活跃位数 k 必须落在 [1, {width}]，收到 {k!r}。")

    tokens = tokenize(text)
    if not tokens:
        raise SDRCapacityError(
            f"文本 {text[:40]!r} 里没有任何可编码的字符（字、词、数字）。"
            f"全标点或全空白的输入无法产生 SDR——调用方应当在这一步之前就跳过它。"
        )

    # 先做词频统计再去哈希：同一 token 在文本里出现 n 次，它的位就带 n 倍权重。
    # 去重还顺带把哈希调用数从"token 数"降到"不同 token 数"——一段反复提到
    # 「咖啡」的文本不必哈希「咖啡」十遍。
    frequency: dict[str, int] = {}
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1

    uniques = list(frequency)
    reps = np.array([frequency[token] for token in uniques], dtype=np.float64)
    # 每个唯一 token 贡献 _BITS_PER_TOKEN 个位，把这 _BITS_PER_TOKEN 个位的权重
    # 都设成该 token 的词频，一次 bincount 就把整轮的累计做完。
    per_token_weights = np.repeat(reps, _BITS_PER_TOKEN)

    counts = np.zeros(width, dtype=np.float64)
    for round_index in range(_MAX_ROUNDS):
        drawn = np.concatenate(
            [_bits_for(token, round_index, width=width, count=_BITS_PER_TOKEN) for token in uniques]
        )
        counts += np.bincount(drawn, weights=per_token_weights, minlength=width)
        if int(np.count_nonzero(counts)) >= k:
            break

    # 取权重最高的 k 位。并列时按索引升序——**必须有确定的打破平局规则**，
    # 否则同一段文本在不同机器上会选出不同的位，"逐位相同"就不成立了。
    # lexsort 的最后一个键是主键，所以主键是 -counts，次键是索引。
    order = np.lexsort((np.arange(width), -counts))
    # 触顶时可能攒不够 k 位（宽度很大而文本极短）。此时**只取实际非零的位**，
    # 而不是用零权重的位凑数：一个 weight=0 的活跃位在 Jaccard 里算"活跃"，
    # 却对相似度毫无贡献，还会污染将来的基质写入。宁可 k 小于标称值。
    take = min(k, int(np.count_nonzero(counts)))
    selected = np.sort(order[:take])

    weights = counts[selected]
    weights = (weights / weights.max()).astype(np.float32)
    return SDR(idx=selected.astype(np.int32), weights=weights, width=width)


# ── 关联码：同一批位上的第二档稀疏度 ────────────────────────────────────────

ASSOC_K = 64
"""关联码的默认活跃位数——神经元级基质 W 的输入。

**这个 64 没有可考的出处。** 计划书里与它对得上的只有 §4.3 的 ``s = 0.14``，
那是词法码那一档，不是关联码。

所以它**由量具扫出来**，不靠引用背书：``tools/bench_substrate.py --k 32,64,128,256``
报出邻居质量与模式补全准确率随 k 的变化。在扫描结果落进 ``benchmarks/substrate/``
之前，任何"k=64 是标定过的"说法都是假的。
"""

_PERSON_ASSOC = b"memcore-assoc"
"""关联码位选择的 ``person`` 前缀（13 字节，限长 16）。

**必须与词法码的 ``b"memcore-lex"`` 不同。** 两者的输入空间本来就不同（那边哈希
token，这边哈希位下标），所以共用前缀并不会直接撞车；但共用会让"选哪 64 位"与
"哪些 token 激活哪些位"落进同一个哈希族，两条路径在同一个下标上产生相关性——而
关联码的全部意义就是做词法码**独立**的一次抽样。
"""

_ASSOC_MODES = frozenset({"hash", "weight"})
"""``encode_assoc`` 认得的选择规则。取值与对照数据见 :func:`encode_assoc`。"""


def _assoc_key(index: int) -> int:
    """一个位下标在关联码里的选择键。

    **只由下标决定，与权重无关**——这正是"哈希子采样"与"取权重最高"的全部区别。
    取 64 位大端，所以返回值的取值空间是 ``[0, 2**64)``，碰撞概率可忽略；即便
    碰撞，:func:`encode_assoc` 的 ``lexsort`` 次键是下标本身，结果仍然确定。

    下标按 4 字节小端编码：宽度上限 :data:`memcore.types.DEFAULT_SDR_WIDTH` 是
    4096，远在 ``2**32`` 之内。宽度真涨到十亿级时这里会 ``OverflowError`` 而不是
    静默截断——那是想要的失败方式。
    """
    digest = hashlib.blake2b(
        index.to_bytes(4, "little"), digest_size=8, person=_PERSON_ASSOC
    ).digest()
    return int.from_bytes(digest, "big")


def encode_assoc(lex: SDR, *, k: int = ASSOC_K, mode: str = "hash") -> SDR:
    """把词法 SDR 再稀疏化成 ``k`` 位的关联码——**神经元级基质 W 的输入**。

    纯 ``SDR -> SDR``：不重新分词、不碰文本。输出宽度与 ``lex`` 相同，``idx`` 是
    ``lex.idx`` 的子集，权重逐位照搬。也就是说 **``assoc.to_dense()`` 恰好等于
    ``lex.to_dense()`` 在入选位上的限制**——关联码是词法码的一个视图，不是另一套
    编码。

    Args:
        lex: 词法 SDR。
        k: 目标活跃位数。**可能取不满**：``lex`` 自己不足 k 位时只取实际有的那些
            （同 :func:`encode_lexical` 的 ``take`` 处置）。宁可 k 小于标称值，也
            不用权重的位凑数——零权重的活跃位在 Jaccard 里算"活跃"却对相似度毫无
            贡献。
        mode: 选择规则，``"hash"`` 或 ``"weight"``。默认 ``"hash"``。两个规则的
            实测差异见下。

    Returns:
        宽度与 ``lex`` 相同的 SDR。

    Raises:
        SDRCapacityError: ``k`` 不在 ``[1, lex.width]``、``mode`` 不认识，或者
            ``lex`` 是空的。

    **一条推论，量图 A 的边权时要知道**：关联码是词法码的子集，所以两者的 Jaccard
    是同一量纲的估计。在 ``benchmarks/gold/`` 的 200 条上实测（19900 对）：词法 J
    均值 0.1970、最大 0.9490；关联 J 均值 0.2056、最大 1.0000；逐对相关 0.9862。
    **低中位几乎重合，高位被系统性抬高**（词法 0.949 的那一对在关联码上是 1.000）
    ——高度重叠的两个位集，各自的 top-64 会趋向同一批位。所以图 A 的边权阈值可以
    沿用词法量纲，但要预期高位非线性，别拿它去分辨"很像"和"极像"。
    （金标准语料是硬负例语料，全是近义改写，**不能**用它的均值当"无关对"的基线。）

    **两个规则的实测对照**（同一批语料，另见 ``benchmarks/substrate/`` 的量具）：

    ==========  ============  ============  ==============
    规则        top-64 并集    两两 J 均值    最高频位被选中
    ==========  ============  ============  ==============
    ``hash``    **504** 位     0.2056        182 / 200
    ``weight``  **1902** 位    0.0917        138 / 200
    ==========  ============  ============  ==============

    这两列合起来说清了一件事，而它与直觉相反：**铺不开的是 ``hash``，不是
    ``weight``**。``hash`` 按摘要排名取前 k，于是摘要小（但固定）的那批下标在任何
    条目里都容易入选——200 条记忆的关联码全挤在 504 个位上。``weight`` 取各自词频
    最高的位，不同条目的高频词不同，反而铺到 1902 位。

    代价记在两个不同的地方：``hash`` 保住了 Jaccard 量纲（0.2056 ≈ 词法的
    0.1970），却把 W 的**有效维数**压到 504——按 Hopfield 的 ``0.138·N``，容量
    只剩约 70 条模式，比 N=4096 的 570 条低一个量级，而且是我们自己压下去的。
    ``weight`` 把有效维数抬到 1902（容量约 260 条），代价是 Jaccard 被压到词法的
    一半。**哪一侧更值，由量具的 ``--assoc`` 扫描裁决，不在这一版里替它决定。**
    """
    if k < 1 or k > lex.width:
        raise SDRCapacityError(f"关联码的 k 必须落在 [1, {lex.width}]，收到 {k!r}。")
    if mode not in _ASSOC_MODES:
        raise SDRCapacityError(f"不认识的选择规则 {mode!r}，只认得 {sorted(_ASSOC_MODES)}。")
    if lex.k < 1:
        raise SDRCapacityError("空的词法 SDR 里没有位可以抽样，调用方应当在此之前跳过它。")

    if mode == "hash":
        keys = np.array([_assoc_key(int(i)) for i in lex.idx], dtype=np.uint64)
        # 主键是摘要，次键是下标：摘要有碰撞时结果仍然确定（见 :func:`_assoc_key`）。
        positions = np.lexsort((lex.idx, keys))[: min(k, lex.k)]
    else:
        # 主键是 -权重（即权重降序），次键同样是下标。
        positions = np.lexsort((lex.idx, -lex.weights))[: min(k, lex.k)]

    # 选出来之后按位下标重排：SDR 要求 idx 严格升序。两个 key 都唯一，所以这一步
    # 与排序算法无关——但显式写 stable，免得将来有人换 key 时踩到并列。
    positions = positions[np.argsort(lex.idx[positions], kind="stable")]
    return SDR(
        idx=lex.idx[positions].astype(np.int32),
        weights=lex.weights[positions],
        width=lex.width,
    )


# ── 残缺探针：从一份码里确定性地挖掉一部分位 ────────────────────────────────

_DROP_STRIDE = 7919
"""丢位的步长。7919 与 1000 互素，所以 ``(序号 × 7919) % 1000`` 在 0..999 上铺得开。"""


def drop_bits(sdr: SDR, ratio: float) -> SDR:
    """确定性地丢掉 ``ratio`` 比例的活跃位，做出一个残缺探针。

    规则是"位在码里的序号 × 7919 再模 1000，小于 ``ratio × 1000`` 的丢掉"。
    用序号而不是位索引、用固定步长而不是随机数：**量具必须可复现**，否则门槛
    绑的就是一次抽签，而抽签的门槛会在某天毫无理由地红一次。

    ``ratio = 0`` 时原样返回。那是一次**空转的测量**（探针就是目标本身，补全
    恒为 1），由 :data:`~biosnn_memterm.bench.substrate.SANITY_CEILING` 在冻结
    之前拦掉——它在这里不做判断，因为"看看完美探针的天花板"也是一个合法的诊断。

    **住在 ``memcore`` 而不是 ``bench``**：理由与
    :func:`memcore.graph.GraphIndex.diffuse` 一致——**让量具与生产用同一个算子**。
    巩固的判据是"回放辨认率"，它必须与基线里的 ``completion_top1`` 同口径；探针的
    造法一旦有两份实现，两个数字就不再可比，而它们看起来仍然都像个合理的百分比。
    """
    if ratio <= 0.0:
        return sdr
    keep = ((np.arange(sdr.k) * _DROP_STRIDE) % 1000) >= int(ratio * 1000)
    if not keep.any():
        # 全丢光就不是"残缺的探针"，是一个空探针——而 SDR 不允许全零。
        keep[0] = True
    return SDR(idx=sdr.idx[keep], weights=sdr.weights[keep], width=sdr.width)
