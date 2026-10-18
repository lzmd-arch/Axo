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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .types import SDR

__all__ = ["LexicalIndex"]


@dataclass(frozen=True)
class LexicalIndex:
    """按词法位建立的 CSR 倒排索引。"""

    ids: tuple[str, ...]
    rows: np.ndarray
    """按位排序的 memory 行号（int32）。"""
    offsets: np.ndarray
    counts: np.ndarray
    """每个位的文档频率；保留给调用方做稀有度判断。"""
    lengths: np.ndarray
    """每条记忆的词法活跃位数（与 ``ids`` 同序，int32）。"""
    width: int

    @classmethod
    def build(cls, rows: Iterable[tuple[str, object, object]]) -> LexicalIndex:
        """从 ``(memory_id, lex_idx_blob, lex_width)`` 构建。没有词法码的条目跳过。"""
        ids: list[str] = []
        flat_bits: list[np.ndarray] = []
        flat_rows: list[np.ndarray] = []
        lengths: list[int] = []
        width = 0
        for memory_id, idx_blob, row_width in rows:
            if idx_blob is None or row_width is None:
                continue
            idx = np.frombuffer(bytes(idx_blob), dtype=np.int32)
            if idx.size == 0:
                continue
            position = len(ids)
            ids.append(str(memory_id))
            lengths.append(int(idx.size))
            width = max(width, int(row_width))
            flat_bits.append(idx.astype(np.int32, copy=False))
            flat_rows.append(np.full(idx.size, position, dtype=np.int32))
        if not ids:
            return cls(
                ids=(),
                rows=np.zeros(0, dtype=np.int32),
                offsets=np.zeros(1, dtype=np.int64),
                counts=np.zeros(1, dtype=np.int64),
                lengths=np.zeros(0, dtype=np.int32),
                width=max(1, width),
            )

        bits = np.concatenate(flat_bits)
        order = np.argsort(bits, kind="stable")
        sorted_bits = bits[order]
        sorted_rows = np.concatenate(flat_rows)[order]
        counts = np.bincount(sorted_bits, minlength=width).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(
            ids=tuple(ids),
            rows=sorted_rows.astype(np.int32, copy=False),
            offsets=offsets,
            counts=counts,
            lengths=np.asarray(lengths, dtype=np.int32),
            width=width,
        )

    def candidates(self, query: SDR, *, limit: int) -> tuple[str, ...]:
        """按精确 Jaccard 取前 ``limit`` 条词法候选。

        只返回与查询至少共享一个位的条目——Jaccard 为 0 的条目本来就不可能被
        词法通道选中，放进候选只会增加扫描成本。
        """
        if limit <= 0 or not self.ids or query.k == 0:
            return ()

        # 所有共享位的 postings 一次 concatenate 再 bincount：比逐位 ``np.add.at``
        # 快一个量级，而这里每个查询本来就要扫 5 万×数百个 posting。
        postings: list[np.ndarray] = []
        for bit in query.idx:
            bit = int(bit)
            if bit < 0 or bit + 1 >= self.offsets.size:
                continue
            start, stop = int(self.offsets[bit]), int(self.offsets[bit + 1])
            if stop > start:
                postings.append(self.rows[start:stop])
        if not postings:
            return ()

        intersection = np.bincount(np.concatenate(postings), minlength=len(self.ids)).astype(
            np.int64
        )
        touched = np.flatnonzero(intersection)
        if touched.size == 0:
            return ()
        inter = intersection[touched]
        union = query.k + self.lengths[touched] - inter
        score = inter / np.maximum(union, 1)

        # 同分按行号（即 memory_id 升序）稳定排序：候选集合必须可复现，否则
        # 相同输入在两次运行里会给出不同的 top-k，预算边缘的条目忽进忽出。
        order = np.lexsort((touched, -score))[:limit]
        return tuple(self.ids[int(touched[int(position)])] for position in order)
