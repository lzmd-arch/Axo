"""词法倒排索引：给纯词法召回做候选预筛。

只在**没有语义/时间/图通道**时启用（见 :func:`memcore.recall.recall`）：这些通道
的命中可能没有词法重叠，只按词法位过滤会漏掉它们。

**候选必须按精确 Jaccard 排序，不能按倒排表的出现顺序截断。** 5 万条放大语料上
的实测（研究副本 ``axo_recall_candidate_quality.py``）：只按"稀有位先 gather、凑满
上限就停"选 5,000 条，精确 top-8 的覆盖率只有 0.48，top-1 命中 0；改成按
``|A∩B| / |A∪B|`` 给所有共享位的条目排序、取前 1,000 条，hit@1 0.85、hit@8 0.975。
差别不在上限大小，而在**谁被留下**：一条与查询共享 3 个公共位的记忆，Jaccard 可能
高于只共享 1 个稀有位的记忆，而出现顺序会先撞上后者。

排序用的是与 :meth:`memcore.types.SDR.jaccard` 相同的集合口径（忽略权重），所以
候选阶段不会改变最终融合/MMR 的语义。上限只影响扫描成本；仍由
:data:`memcore.recall._PREFILTER_LIMIT` 控制。

**新增条目走 delta，不重建整库。** 50k 条上整库构建实测 5 s；如果每写一条记忆都
重建，50k 库的每一轮对话都要付这 5 s。:meth:`LexicalIndex.with_added` 把新条目
追加进一个小的 delta CSR，:meth:`candidates` 把 base 与 delta 的 posting 一起
bincount；delta 超过 `MemoryStore` 的阈值时，下一次 `lexical_index()` 才做一次
全量合并。删除仍然走整库重建——删除是低频的用户动作，不值得为它维护 tombstone。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import numpy as np

from .types import SDR

__all__ = ["LexicalIndex"]


_Csr = tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _empty_csr() -> _Csr:
    return (
        (),
        np.zeros(0, dtype=np.int32),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
    )


def _csr(entries: Iterable[tuple[str, np.ndarray, int]], width: int) -> _Csr:
    """把 ``(memory_id, 活跃位, 宽度)`` 编成 CSR。

    返回 ``(ids, rows, offsets, counts, lengths, widths)``；``widths`` 是每条记忆
    自己的表示宽度。召回必须保留它：宽度不同的两套码算出来的 Jaccard 是垃圾，
    而它看起来会像一个正常分数（见 :func:`memcore.recall._lexical_score`）。
    """
    ids: list[str] = []
    flat_bits: list[np.ndarray] = []
    lengths: list[int] = []
    widths: list[int] = []
    for memory_id, idx, row_width in entries:
        idx = np.asarray(idx, dtype=np.int32)
        if idx.size == 0:
            continue
        ids.append(str(memory_id))
        lengths.append(int(idx.size))
        widths.append(int(row_width))
        flat_bits.append(idx)
    if not ids:
        return _empty_csr()

    bits = np.concatenate(flat_bits)
    # 行号按长度展开，不逐条造 ``np.full``：5 万条 × 573 位时那是 2.9 万个短数组，
    # 合起来只为了再 concatenate 一次。``np.repeat`` 一步到位，峰值少一份与
    # ``bits`` 等大的 int32 数组。
    del flat_bits
    # ``bincount`` 与顺序无关，所以不必先排好再数——省掉一份 ``sorted_bits``。
    counts = np.bincount(bits, minlength=width).astype(np.int64)
    order = np.argsort(bits, kind="stable")
    del bits
    rows = np.repeat(np.arange(len(ids), dtype=np.int32), np.asarray(lengths, dtype=np.int64))
    sorted_rows = rows[order]
    del rows, order
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return (
        tuple(ids),
        sorted_rows.astype(np.int32, copy=False),
        offsets,
        counts,
        np.asarray(lengths, dtype=np.int32),
        np.asarray(widths, dtype=np.int32),
    )


@dataclass(frozen=True)
class LexicalIndex:
    """按词法位建立的 CSR 倒排索引，外加一小段未合并的新增条目。"""

    ids: tuple[str, ...]
    rows: np.ndarray
    """按位排序的 memory 行号（int32）。"""
    offsets: np.ndarray
    counts: np.ndarray
    """每个位的文档频率；保留给调用方做稀有度判断。"""
    lengths: np.ndarray
    """每条记忆的词法活跃位数（与 ``ids`` 同序，int32）。"""
    widths: np.ndarray
    """每条记忆自己的表示宽度（与 ``ids`` 同序，int32）。"""
    width: int
    delta_ids: tuple[str, ...] = ()
    """还没并进 base 的新增条目，按写入顺序。"""
    delta_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    delta_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    delta_counts: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    delta_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    delta_widths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    delta_bits: tuple[np.ndarray, ...] = ()
    """与 ``delta_ids`` 同序的活跃位；只用于下一次 ``with_added`` 时重建 delta CSR。"""

    @classmethod
    def build(cls, rows: Iterable[tuple[str, object, object]]) -> LexicalIndex:
        """从 ``(memory_id, lex_idx_blob, lex_width)`` 构建。没有词法码的条目跳过。"""
        entries: list[tuple[str, np.ndarray]] = []
        width = 0
        for memory_id, idx_blob, row_width in rows:
            if idx_blob is None or row_width is None:
                continue
            idx = np.frombuffer(bytes(idx_blob), dtype=np.int32)
            if idx.size == 0:
                continue
            width = max(width, int(row_width))
            entries.append((str(memory_id), idx, int(row_width)))
        ids, flat_rows, offsets, counts, lengths, widths = _csr(entries, width)
        return cls(
            ids=ids,
            rows=flat_rows,
            offsets=offsets,
            counts=counts,
            lengths=lengths,
            widths=widths,
            width=max(1, width),
        )

    def contains(self, memory_id: str) -> bool:
        """索引里有没有这个 id（base 或 delta）。"""
        return memory_id in self.ids or memory_id in self.delta_ids

    def with_added(self, memory_id: str, idx: np.ndarray, width: int) -> LexicalIndex:
        """返回追加了一条记忆的新索引；不重建 base。

        同一个 id 在 delta 里已经存在时替换它（base 里的旧副本由调用方负责——
        ``MemoryStore`` 在 id 已存在时直接失效整库索引）。
        """
        idx = np.asarray(idx, dtype=np.int32)
        if idx.size == 0:
            return self
        entries = [
            (existing_id, bits, entry_width)
            for existing_id, bits, entry_width in zip(
                self.delta_ids, self.delta_bits, self.delta_widths, strict=True
            )
            if existing_id != memory_id
        ]
        entries.append((str(memory_id), idx, int(width)))
        new_width = max(self.width, int(width))
        (
            delta_ids,
            delta_rows,
            delta_offsets,
            delta_counts,
            delta_lengths,
            delta_widths,
        ) = _csr(entries, new_width)
        return replace(
            self,
            width=new_width,
            delta_ids=delta_ids,
            delta_rows=delta_rows,
            delta_offsets=delta_offsets,
            delta_counts=delta_counts,
            delta_lengths=delta_lengths,
            delta_widths=delta_widths,
            delta_bits=tuple(bits for _memory_id, bits, _width in entries),
        )

    def _touched(self, query: SDR) -> tuple[np.ndarray, np.ndarray]:
        """返回与查询共享位、且宽度一致的条目：``(位置, Jaccard 分)``。"""
        base_size = len(self.ids)
        total = base_size + len(self.delta_ids)
        if total == 0 or query.k == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

        # 所有共享位的 postings 一次 concatenate 再 bincount：比逐位 ``np.add.at``
        # 快一个量级，而这里每个查询本来就要扫 5 万×数百个 posting。delta 的行号
        # 加上 base 大小，合并成一个位置空间。
        postings: list[np.ndarray] = []
        for bit in query.idx:
            bit = int(bit)
            if bit >= 0 and bit + 1 < self.offsets.size:
                start, stop = int(self.offsets[bit]), int(self.offsets[bit + 1])
                if stop > start:
                    postings.append(self.rows[start:stop])
            if bit >= 0 and bit + 1 < self.delta_offsets.size:
                start, stop = int(self.delta_offsets[bit]), int(self.delta_offsets[bit + 1])
                if stop > start:
                    postings.append(self.delta_rows[start:stop] + base_size)
        if not postings:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

        intersection = np.bincount(np.concatenate(postings), minlength=total).astype(np.int64)
        touched = np.flatnonzero(intersection)
        if touched.size == 0:
            return touched, np.zeros(0, dtype=np.float64)

        inter = intersection[touched]
        lengths = np.empty(touched.size, dtype=np.int32)
        widths = np.empty(touched.size, dtype=np.int32)
        in_base = touched < base_size
        if in_base.any():
            lengths[in_base] = self.lengths[touched[in_base]]
            widths[in_base] = self.widths[touched[in_base]]
        if (~in_base).any():
            delta_positions = touched[~in_base] - base_size
            lengths[~in_base] = self.delta_lengths[delta_positions]
            widths[~in_base] = self.delta_widths[delta_positions]

        # 宽度不同的两套码不能算相似度。候选路径原先靠 ``_scan`` 再拦一次，
        # 但图通道的种子分只看这里，必须在源头就挡掉。
        compatible = widths == query.width
        if not compatible.all():
            touched = touched[compatible]
            inter = inter[compatible]
            lengths = lengths[compatible]
        if touched.size == 0:
            return touched, np.zeros(0, dtype=np.float64)

        union = query.k + lengths - inter
        return touched, inter / np.maximum(union, 1)

    def candidates(self, query: SDR, *, limit: int) -> tuple[str, ...]:
        """按精确 Jaccard 取前 ``limit`` 条词法候选。

        只返回与查询至少共享一个位的条目——Jaccard 为 0 的条目本来就不可能被
        词法通道选中，放进候选只会增加扫描成本。
        """
        if limit <= 0:
            return ()
        touched, score = self._touched(query)
        if touched.size == 0:
            return ()

        # 同分按位置稳定排序：候选集合必须可复现，否则相同输入在两次运行里会
        # 给出不同的 top-k，预算边缘的条目忽进忽出。
        order = np.lexsort((touched, -score))[:limit]
        base_size = len(self.ids)
        return tuple(
            self.ids[int(position)]
            if int(position) < base_size
            else self.delta_ids[int(position) - base_size]
            for position in touched[order]
        )

    def scores(self, query: SDR) -> dict[str, float]:
        """每条与查询共享位、且宽度一致的记忆的 Jaccard 分。

        图通道的种子分只需要这个字典；不要为了它再 ``iter_items()`` 解码一遍
        全库 SDR——50k 上那是数秒，而这里是一次 posting 扫描。
        """
        touched, score = self._touched(query)
        if touched.size == 0:
            return {}
        base_size = len(self.ids)
        return {
            (
                self.ids[int(position)]
                if int(position) < base_size
                else self.delta_ids[int(position) - base_size]
            ): float(value)
            for position, value in zip(touched, score, strict=True)
        }
