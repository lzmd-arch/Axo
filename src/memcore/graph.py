"""关联图 A —— 记忆之间的相似度图。

边是**关联码的 Jaccard**，每个节点保留自己最像的 ``top_m`` 个邻居（另有权重下限）。
它和 :mod:`memcore.substrate` 的 W 是**同一套关联码的两个视图**：A 是图的形态
（可解释、喂扩散与可视化），W 是矩阵的形态（模式补全）。两者都从
:func:`~memcore.sdr.encode_assoc` 的产物来。

## 为什么边取"相似"而不是"共现"

同会话共现（``MemoryItem.session_id``）在生物学上更像"关联"，也是唯一真正独立
于词法通道的信号。但它需要**真实使用历史**才有边，而 ``benchmarks/`` 下没有任何
可用的共现结构——**离线量不出质量**。这一版的纪律是"只报告能离线跑的数"，所以
共现边登记给后面，等真实库上有数据。

## 建图是显式全量重建，不做增量

增量有**三条会静默发散的路径**：编辑改了正文而旧边留下、删除产生悬空节点、并发写
与召回撞锁。全量重建是幂等、可复现的。图是 ``mem.db`` 的**派生缓存**——删掉一条
的正确答案是"下次重建时它自然消失"，而不是维护一条删边路径。所以
:meth:`GraphIndex.is_stale` 的用途是让读者知道该不该重建，**不是**给
``MemoryStore.delete`` 挂回调。

## 代价：先算账再选算法

朴素的双层循环对 Jaccard 是 ``n²/2`` 次集合运算，n=5e4 时不可接受。这里走稠密
分块矩阵乘：把每块 ``(chunk, N)`` 的二值指示矩阵与另一块相乘得到**交点计数**。

* **绝不整块算 ``(M, M)``**：M=5e4 时那张 float32 矩阵是 10 GB。分块后每块的
  ``(chunk, M)`` 只有几十 MB，算完立刻做 top-m 选择，全矩阵从不落地。
* FLOPs 是 ``n²·N/2``（N=4096），**不是** ``n²·k``：numpy 里没有稀疏矩阵，
  二值指示矩阵的乘法按稠密算。

**实测**（开发机、Python 3.12、numpy 单线程 BLAS、``ASSOC_K=64``、``top_m=8``）::

    n=200     建图 0.01 s    含编码 0.25 s
    n=2000    建图 0.55 s    含编码 3.0 s
    n=20000   建图 40 s      含编码 66 s
    n=50000   建图 259 s     含编码 322 s

两次十倍规模各涨 55× 与 73×，与 ``n²`` 同阶（最后一次 2.5 倍规模涨 6.5×）。
**5 万条要四分半**——所以只能手动触发，别把它挂到写入路径上。

"含编码"那一列不是凑数：``encode_assoc`` 在 5 万条上要 63 s，与建图同一量级
——**重建一次的成本是两者之和**。分开报也是诊断的需要：编码慢是 ``k`` 的问题，
建图慢是分块与 ``top_m`` 的问题。
"""

from __future__ import annotations

import hashlib
import time
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import SDRCapacityError, StoreCorruptError
from .sdr import ASSOC_K, encode_assoc
from .types import SDR, MemoryItem

__all__ = ["FORMAT_VERSION", "GraphConfig", "GraphHeat", "GraphIndex", "graph_heat"]

FORMAT_VERSION = 1
"""``graph.npz`` 的格式版本。**与 :data:`memcore.store.SCHEMA_VERSION` 无关**：
图是派生缓存，重建它不涉及任何迁移——版本对不上就丢掉重算，而库文件对不上要迁移。
"""

_CHUNK = 256
"""分块的边长（行数）。每块的稠密指示矩阵是 ``(_CHUNK, N)`` float32 ≈ 4 MB。"""


@dataclass(frozen=True)
class GraphConfig:
    """建图参数。

    **这两个默认值是占位的，不是标定过的。** ``tools/bench_substrate.py`` 的
    ``--top-m`` / ``--min-weight`` 扫描会给出曲线；在曲线落进
    ``benchmarks/substrate/`` 之前，不要引用这两个数当作"调好的"。

    Attributes:
        top_m: 每个节点保留多少个邻居。
        min_weight: 边权下限。低于它的对不连边——没有它就只是"最像的 m 个"，
            哪怕最像的那个也只像 0.01。
        k: 关联码的活跃位数。
        mode: 关联码的选择规则，透传给 :func:`~memcore.sdr.encode_assoc`。
    """

    top_m: int = 8
    min_weight: float = 0.10
    k: int = ASSOC_K
    mode: str = "hash"


_DEFAULT_CONFIG = GraphConfig()
"""``build`` 的默认参数。理由同 :data:`memcore.substrate._DEFAULT_CONFIG`。"""


def _extract(
    items: Iterable[MemoryItem], *, k: int, mode: str
) -> tuple[tuple[str, ...], list[SDR], int]:
    """把条目抽成 ``(已排序的 id, 关联码, 宽度)``。

    :meth:`GraphIndex.build` 与 :meth:`GraphIndex.is_stale` 共用它——两者必须
    用**同一套**规则决定"图里有哪些节点、每个节点的码是什么"，否则指纹永远对
    不上（这个 bug 真的发生过：``is_stale`` 一开始让调用方自己传码，而调用方
    手里只有词法码）。

    **没有 ``lex_sdr`` 的条目被跳过**，不是报错：还没回填编码的条目在实时写入
    路径上是常态。
    """
    pairs: list[tuple[str, SDR]] = []
    width = 0
    for item in items:
        sdr = item.assoc_sdr
        if sdr is None:
            if item.lex_sdr is None:
                continue
            sdr = encode_assoc(item.lex_sdr, k=k, mode=mode)
        if width == 0:
            width = sdr.width
        elif sdr.width != width:
            raise SDRCapacityError(
                f"条目 {item.memory_id} 的关联码宽度是 {sdr.width}，"
                f"而前面的是 {width}——不同宽度的关联码算不出 Jaccard。"
            )
        pairs.append((item.memory_id, sdr))

    # 按 memory_id 排序：ULID 按时间单调递增，所以字典序就是创建序。
    pairs.sort(key=lambda pair: pair[0])
    return tuple(node_id for node_id, _ in pairs), [sdr for _, sdr in pairs], width


def _source_digest(node_ids: Sequence[str], patterns: Sequence[SDR]) -> str:
    """节点集合与它们关联码的指纹。

    **必须含 ``idx`` 的字节，不能只比 id 集合**——"改了正文但 id 不变"是最常见的
    陈旧来源，只比 id 会让这种情况下 :meth:`GraphIndex.is_stale` 返回 False。

    按 id 排序后哈希，所以指纹与输入顺序无关（行序另由 :meth:`GraphIndex.build`
    固定，见那里的说明）。
    """
    digest = hashlib.sha256()
    for node_id, sdr in sorted(zip(node_ids, patterns, strict=True), key=lambda pair: pair[0]):
        digest.update(node_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(np.asarray(sdr.idx, dtype=np.int32).tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class GraphIndex:
    """关联图。**行序是 ``node_ids`` 的字典序**，不是输入顺序。

    Attributes:
        node_ids: ``(M,)`` 的 memory_id，**已排序**。行序就是它，所以
            :meth:`neighbors` 可以用二分查找。
        nbr_idx: ``(M, top_m)`` int32，每行的邻居行号，``-1`` 是空位。
        nbr_w: ``(M, top_m)`` float32，对应的 Jaccard 边权，空位为 0。
        source_digest: 建图时节点集合与关联码的指纹，见 :meth:`is_stale`。
        built_at: 建图时刻（Unix 秒）。仅供诊断。
        top_m / min_weight / assoc_k / assoc_mode: 建图参数。存下来有两个用处：
            **差异诊断**（改了参数却没重建，这几项就是证据），以及让
            :meth:`is_stale` 能用当初那套参数重算关联码——它必须和
            :meth:`build` 用同一套，否则指纹永远对不上。
    """

    node_ids: tuple[str, ...]
    nbr_idx: np.ndarray
    nbr_w: np.ndarray
    source_digest: str
    built_at: float
    top_m: int
    min_weight: float
    assoc_k: int = ASSOC_K
    assoc_mode: str = "hash"

    def __post_init__(self) -> None:
        nbr_idx = np.ascontiguousarray(self.nbr_idx, dtype=np.int32)
        nbr_w = np.ascontiguousarray(self.nbr_w, dtype=np.float32)
        node_ids = tuple(self.node_ids)
        size = len(node_ids)

        if nbr_idx.shape != nbr_w.shape:
            raise StoreCorruptError(
                f"nbr_idx 与 nbr_w 形状必须一致，收到 {nbr_idx.shape} 与 {nbr_w.shape}。"
            )
        if nbr_idx.ndim != 2 or nbr_idx.shape[0] != size:
            raise StoreCorruptError(
                f"邻居表应当是 ({size}, top_m) 的二维数组，收到 {nbr_idx.shape}。"
            )
        if nbr_idx.size and (nbr_idx.min() < -1 or nbr_idx.max() >= size):
            # 越界就是损坏：一个指向不存在节点的行号会让 neighbors 返回一个
            # 别的记忆的 id，而那是**看起来完全正常**的错误答案。
            raise StoreCorruptError(
                f"邻居表里有越界的行号：值域应为 [-1, {size})，"
                f"收到 [{int(nbr_idx.min())}, {int(nbr_idx.max())}]。"
            )
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "nbr_idx", nbr_idx)
        object.__setattr__(self, "nbr_w", nbr_w)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphIndex):
            return NotImplemented
        return (
            self.node_ids == other.node_ids
            and self.top_m == other.top_m
            and self.min_weight == other.min_weight
            and self.assoc_k == other.assoc_k
            and self.assoc_mode == other.assoc_mode
            and self.source_digest == other.source_digest
            and np.array_equal(self.nbr_idx, other.nbr_idx)
            and np.array_equal(self.nbr_w, other.nbr_w)
        )

    def __len__(self) -> int:
        return len(self.node_ids)

    # ── 构建 ────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        items: Iterable[MemoryItem],
        *,
        config: GraphConfig = _DEFAULT_CONFIG,
        built_at: float | None = None,
    ) -> GraphIndex:
        """全量重建关联图。

        **行序强制按 memory_id 字典序**，不用输入顺序：这样 ``source_digest``
        与 ``nbr_idx`` 都只由节点集合决定，同一批记忆无论以什么顺序传进来都得到
        逐位相同的图。ULID 按时间单调递增，所以字典序就是创建序——一个顺带的好处。

        **没有 ``lex_sdr`` 的条目被跳过**（同 :meth:`~memcore.substrate.Substrate.build`）。

        Raises:
            SDRCapacityError: 条目的关联码宽度不一致，或者 ``config`` 里的参数越界。
        """
        if config.top_m < 1:
            raise SDRCapacityError(f"top_m 必须 >= 1，收到 {config.top_m!r}。")
        if config.min_weight < 0.0 or config.min_weight > 1.0:
            raise SDRCapacityError(
                f"边权下限是 Jaccard 的量纲，必须落在 [0, 1]，收到 {config.min_weight!r}。"
            )

        node_ids, patterns, width = _extract(items, k=config.k, mode=config.mode)
        digest = _source_digest(node_ids, patterns)
        stamp = time.time() if built_at is None else float(built_at)

        size = len(patterns)
        if size == 0:
            return cls(
                node_ids=(),
                nbr_idx=np.zeros((0, config.top_m), dtype=np.int32),
                nbr_w=np.zeros((0, config.top_m), dtype=np.float32),
                source_digest=digest,
                built_at=stamp,
                top_m=config.top_m,
                min_weight=config.min_weight,
                assoc_k=config.k,
                assoc_mode=config.mode,
            )

        k_max = max(sdr.k for sdr in patterns)
        idx = np.zeros((size, k_max), dtype=np.int32)
        active = np.zeros((size, k_max), dtype=bool)
        for row, sdr in enumerate(patterns):
            idx[row, : sdr.k] = sdr.idx
            # padding 位的 idx 是 0，但它不是真的活跃位。用 active 掩码把它挡在
            # 指示矩阵之外——否则每条短记忆都会凭空多出一个"0 号位"。
            active[row, : sdr.k] = True
        # 关联码不带权重的语义（Jaccard 只看集合），所以指示矩阵是二值的。
        cardinality = active.sum(axis=1).astype(np.int32)

        nbr_idx = np.full((size, config.top_m), -1, dtype=np.int32)
        nbr_w = np.zeros((size, config.top_m), dtype=np.float32)
        for start in range(0, size, _CHUNK):
            stop = min(start + _CHUNK, size)
            block = _indicator(idx[start:stop], active[start:stop], width)
            # 交点计数：block @ B^T，B 逐块现造，全量的 (M, N) 从不落地。
            inter = np.empty((stop - start, size), dtype=np.int32)
            for other in range(0, size, _CHUNK):
                other_stop = min(other + _CHUNK, size)
                other_block = _indicator(idx[other:other_stop], active[other:other_stop], width)
                inter[:, other:other_stop] = np.rint(block @ other_block.T).astype(np.int32)
            _select_top_m(
                inter,
                cardinality[start:stop],
                cardinality,
                start,
                config,
                nbr_idx[start:stop],
                nbr_w[start:stop],
            )

        return cls(
            node_ids=node_ids,
            nbr_idx=nbr_idx,
            nbr_w=nbr_w,
            source_digest=digest,
            built_at=stamp,
            top_m=config.top_m,
            min_weight=config.min_weight,
            assoc_k=config.k,
            assoc_mode=config.mode,
        )

    # ── 查询 ────────────────────────────────────────────────────────────────

    def index_of(self, node_id: str) -> int:
        """节点所在的行号。找不到返回 ``-1``。

        二分查找而不是建字典：``node_ids`` 已排序，而一张 M 条的 id→行号映射
        在 5 万条时是几 MB 的常驻内存——图是缓存，不该反客为主。
        """
        position = bisect_left(self.node_ids, node_id)
        if position < len(self.node_ids) and self.node_ids[position] == node_id:
            return position
        return -1

    def neighbors(
        self, node_id: str, *, live: Iterable[str] | None = None
    ) -> list[tuple[str, float]]:
        """一个节点的邻居，按边权降序。

        Args:
            node_id: 要查的节点。不在图里就返回空列表——**不抛错**：图是缓存，
                一条刚写进去还没重建的记忆在图里查不到是常态。
            live: 当前库里真实存在的 memory_id。给定时，指向已删除节点的边被过滤
                掉。**不给就不过滤**，此时调用方拿到的是"建图那一刻"的邻居。

        Returns:
            ``[(memory_id, weight), ...]``。
        """
        row = self.index_of(node_id)
        if row < 0:
            return []
        allowed = None if live is None else set(live)
        result: list[tuple[str, float]] = []
        for column, weight in zip(self.nbr_idx[row], self.nbr_w[row], strict=True):
            if column < 0:
                continue
            neighbor_id = self.node_ids[int(column)]
            if allowed is not None and neighbor_id not in allowed:
                continue
            result.append((neighbor_id, float(weight)))
        return result

    def is_stale(self, items: Iterable[MemoryItem]) -> bool:
        """按现在这批条目，这份图是不是过期的。

        比较的是 :func:`_source_digest`，所以"改了正文但 id 不变"也算过期
        （见那里的说明）。**顺序不敏感**：调用方不必费心排序。关联码用图自己
        记着的 :attr:`assoc_k` / :attr:`assoc_mode` 重算——必须与
        :meth:`build` 当初那套参数一致，否则指纹永远对不上。

        代价是一次全量重编码（每条 64 次哈希）。这是个偶尔调用的判断，不是热路径。
        """
        node_ids, patterns, _ = _extract(items, k=self.assoc_k, mode=self.assoc_mode)
        return self.source_digest != _source_digest(node_ids, patterns)

    def edge_count(self) -> int:
        """有多少条有向边（每条无向边通常被记两次，因为两侧各自保留 top-m）。"""
        return int(np.count_nonzero(self.nbr_idx >= 0))

    # ── 扩散 ────────────────────────────────────────────────────────────────

    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """有向边的三件套 ``(rows, targets, weights)``，只取非空槽。

        图存的是**每个节点自己**的有向 top-m，所以"这两个像"会被记两次（i→j 与
        j→i）。遍历全部行因此等价于把两个方向并起来——:meth:`diffuse` 要的正是这个，
        不需要额外对称化。

        **不缓存。** ``GraphIndex`` 是 frozen，加缓存字段就要在 :meth:`__post_init__`
        里 ``setattr``，而那会让"图"这份纯数据多出一份随调用漂移的状态。代价可以
        接受：一次 ``nonzero`` 在 5 万条 × top-m=8 上扫 40 万个元素。
        """
        rows, slots = np.nonzero(self.nbr_idx >= 0)
        targets = self.nbr_idx[rows, slots].astype(np.int64)
        weights = self.nbr_w[rows, slots].astype(np.float64)
        return rows.astype(np.int64), targets, weights

    def diffuse(self, base: np.ndarray, *, decay: float = 0.5) -> np.ndarray:
        """把 ``base`` 沿边传播一步，返回每个节点**收到的**分数。

        **取 max 不取 sum。** 求和会让连接数多的节点被抬起来（hub 爆炸），而"从哪个
        邻居走过来的"才是这一步的语义。它住在 ``memcore`` 而不是 ``bench``，是为了让
        **量具与生产用同一个算子**——两份实现漂移正是这个仓最忌讳的形状。

        Args:
            base: ``(M,)`` 的种子分，按 :attr:`node_ids` 的行序。通常是各条记忆与
                查询的词法 Jaccard。
            decay: 每步的衰减。**与通道可靠度是两件事**（那是
                ``RecallConfig.graph_weight``）：这里管"传得动多远"，那里管"这条
                通道说的话有多算数"。

        Returns:
            ``(M,)`` 的 float64。**不含 ``base`` 自己**——要 ``base + spread`` 的
            调用方自己加。分开是刻意的：``base`` 是"这个词法分是我自己算出来的"，
            ``spread`` 是"这个分是邻居传给我的"，而词法通道已经带着前者进过一次
            融合了，混在一起等于把同一份证据记两遍。
        """
        if decay < 0.0:
            raise ValueError(f"扩散系数不能是负数，收到 {decay!r}。")
        scores = np.asarray(base, dtype=np.float64)
        if scores.shape != (len(self.node_ids),):
            # 行序配错了不会抛异常，只会算出**另一个记忆的邻居**——那种错误看起来
            # 完全正常。所以这里必须自己拦。
            raise ValueError(
                f"种子分应当是 ({len(self.node_ids)},) 的数组，收到 {scores.shape}。"
                "行序必须与 node_ids 一致。"
            )
        rows, targets, weights = self.edges()
        spread = np.zeros(len(self.node_ids), dtype=np.float64)
        np.maximum.at(spread, targets, scores[rows] * weights * decay)
        return spread

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """写 ``graph.npz``。**不开 ``allow_pickle``**，所以只有定长数组能进来。"""
        np.savez(
            path,
            format_version=np.array(FORMAT_VERSION, dtype=np.int64),
            node_ids=np.array(self.node_ids, dtype="<U26"),
            nbr_idx=self.nbr_idx,
            nbr_w=self.nbr_w,
            source_digest=np.array(self.source_digest, dtype="<U64"),
            built_at=np.array(self.built_at, dtype=np.float64),
            top_m=np.array(self.top_m, dtype=np.int64),
            min_weight=np.array(self.min_weight, dtype=np.float64),
            assoc_k=np.array(self.assoc_k, dtype=np.int64),
            assoc_mode=np.array(self.assoc_mode, dtype="<U8"),
        )

    @classmethod
    def load(cls, path: Path) -> GraphIndex:
        """读 ``graph.npz``。

        Raises:
            StoreCorruptError: 文件缺键、版本对不上，或者 id 超出了 ``'<U26'``
                能装下的长度（ULID 恰好 26 字符；更长的 id 在保存时**会被静默
                截断**，所以这里宁可报错）。
        """
        with np.load(path, allow_pickle=False) as data:
            missing = {
                "format_version",
                "node_ids",
                "nbr_idx",
                "nbr_w",
                "source_digest",
                "built_at",
                "top_m",
                "min_weight",
                "assoc_k",
                "assoc_mode",
            } - set(data.files)
            if missing:
                raise StoreCorruptError(f"{path} 缺少这些键：{sorted(missing)}。")
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise StoreCorruptError(
                    f"{path} 的格式版本是 {version}，本代码只认 {FORMAT_VERSION}。"
                    f"图是派生缓存——**丢掉重建**，不要写迁移。"
                )
            node_ids = tuple(str(node_id) for node_id in data["node_ids"])
            if any(len(node_id) > 26 for node_id in node_ids):
                raise StoreCorruptError(f"{path} 里有超过 26 个字符的 id，定长字段装不下。")
            return cls(
                node_ids=node_ids,
                nbr_idx=data["nbr_idx"],
                nbr_w=data["nbr_w"],
                source_digest=str(data["source_digest"]),
                built_at=float(data["built_at"]),
                top_m=int(data["top_m"]),
                min_weight=float(data["min_weight"]),
                assoc_k=int(data["assoc_k"]),
                assoc_mode=str(data["assoc_mode"]),
            )


def _indicator(idx: np.ndarray, active: np.ndarray, width: int) -> np.ndarray:
    """一块 ``(chunk, N)`` 的二值指示矩阵。"""
    rows, columns = idx.shape
    block = np.zeros((rows, width), dtype=np.float32)
    block[np.repeat(np.arange(rows), columns)[active.ravel()], idx.ravel()[active.ravel()]] = 1.0
    return block


def _select_top_m(
    inter: np.ndarray,
    cardinality: np.ndarray,
    all_cardinality: np.ndarray,
    offset: int,
    config: GraphConfig,
    nbr_idx_out: np.ndarray,
    nbr_w_out: np.ndarray,
) -> None:
    """在交点数上就地选出每行的 top-m 邻居，写进输出块。**全矩阵从不落地。**"""
    # Jaccard = |A∩B| / (|A| + |B| − |A∩B|)。对角线是"自己和自己"，分子分母都是
    # 自己的势，得到 1.0——先把它们打成 -1，免得每个节点把自己当第一名。
    union = cardinality[:, None] + all_cardinality[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)
    rows = np.arange(inter.shape[0])
    similarity[rows, rows + offset] = -1.0
    # 先过权重下限，再做 top-m：低于下限的对连候选都不该是。
    similarity[similarity < config.min_weight] = -1.0

    top_m = nbr_idx_out.shape[1]
    available = similarity.shape[1]
    if available <= top_m:
        # 候选比 top_m 还少：全排序，然后**只填前 available 列**。剩下的列保持
        # 初始的 -1 / 0.0——按 (c, top_m) 整块赋值会以形状不符当场炸掉，而
        # "节点数少于 top_m"在小语料上是很常见的配置。
        order = np.argsort(-similarity, axis=1, kind="stable")
        picked = np.take_along_axis(similarity, order, axis=1)
        keep = (picked >= config.min_weight) & (picked > 0.0)
        nbr_idx_out[:, :available] = np.where(keep, order, -1).astype(np.int32)
        nbr_w_out[:, :available] = np.where(keep, picked, 0.0).astype(np.float32)
        return

    # argpartition 是 O(M) 的，全排序是 O(M log M)——M=5e4 时这不是小事。
    order = np.argpartition(-similarity, top_m, axis=1)[:, :top_m]
    # partition 只保证"前 top_m 个就是最大的那些"，不保证顺序，所以再排一次。
    order = np.take_along_axis(
        order,
        np.argsort(-np.take_along_axis(similarity, order, 1), axis=1, kind="stable"),
        axis=1,
    )
    picked = np.take_along_axis(similarity, order, axis=1)
    # ``> 0`` 与下限是两件事：``min_weight=0`` 时 Jaccard 恰好为 0 的那些对仍然
    # 会被 ``>= 0`` 放进来，而"一点都不像"不该连成边。
    keep = (picked >= config.min_weight) & (picked > 0.0)
    nbr_idx_out[:] = np.where(keep, order, -1).astype(np.int32)
    nbr_w_out[:] = np.where(keep, picked, 0.0).astype(np.float32)


@dataclass(frozen=True)
class GraphHeat:
    """关联图上的**逐条记忆**热度。**速率级，不是脉冲。**

    这是「SNN 热度」面板的数据源：对每条记忆 i，把它的 ``top-m`` 出边权重的平方
    求和（``support_i = sum_j w_ij**2``），再按全库最大值归一化到 ``[0, 1]``。
    它衡量的是"这条记忆被它的邻居支撑得有多强"——强边比弱边贡献大，孤立节点是 0。
    它**不是**一步补全的 argmax：每条记忆都有自己的标量，这正是面板要的形状。

    **它来自关联图 A（``hash`` 档），不是 W（``idf`` 档）。** 图是窗口里已经建好/
    载入的那一份，算它只多一次 ``edges()`` + ``bincount``，5 万条 x top-m=8 上是
    400k 个元素，毫秒级；不需要为了面板再建一份 W。
    """

    values: dict[str, float]
    """``memory_id -> 归一化热度``，0..1。"""
    support: dict[str, float]
    """``memory_id -> 原始支撑度``（边权平方和）。"""
    node_count: int
    edge_count: int
    max_support: float
    median: float
    """归一化热度的中位数。"""


def graph_heat(graph: GraphIndex) -> GraphHeat:
    """算 :class:`GraphHeat`。**纯函数，无随机数。**"""
    if len(graph) == 0:
        return GraphHeat(
            values={},
            support={},
            node_count=0,
            edge_count=0,
            max_support=0.0,
            median=0.0,
        )

    rows, _targets, weights = graph.edges()
    support = np.bincount(rows, weights=weights**2, minlength=len(graph)).astype(np.float64)
    max_support = float(support.max()) if support.size else 0.0
    values = support / max_support if max_support > 0.0 else np.zeros_like(support)
    return GraphHeat(
        values={
            node_id: float(value) for node_id, value in zip(graph.node_ids, values, strict=True)
        },
        support={
            node_id: float(value) for node_id, value in zip(graph.node_ids, support, strict=True)
        },
        node_count=len(graph),
        edge_count=int(rows.size),
        max_support=max_support,
        median=float(np.median(values)) if values.size else 0.0,
    )
