"""词法倒排索引：给纯词法召回做候选预筛。

只在**没有语义/时间/图通道**时启用（见 :func:`memcore.recall.recall`）：这些通道
的命中可能没有词法重叠，只按词法位过滤会漏掉它们。

候选选择按**位的文档频率**从稀有到常见 gather：公共位在 5 万条放大语料上会把
候选拉回全库，稀有位的区分度更高。候选上限由 :data:`memcore.recall._PREFILTER_LIMIT`
控制；它只影响扫描成本，不改变最终融合/MMR 的语义。
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
    width: int

    @classmethod
    def build(cls, rows: Iterable[tuple[str, object, object]]) -> LexicalIndex:
        """从 ``(memory_id, lex_idx_blob, lex_width)`` 构建。没有词法码的条目跳过。"""
        ids: list[str] = []
        flat_bits: list[np.ndarray] = []
        flat_rows: list[np.ndarray] = []
        width = 0
        for memory_id, idx_blob, row_width in rows:
            if idx_blob is None or row_width is None:
                continue
            idx = np.frombuffer(bytes(idx_blob), dtype=np.int32)
            if idx.size == 0:
                continue
            position = len(ids)
            ids.append(str(memory_id))
            width = max(width, int(row_width))
            flat_bits.append(idx.astype(np.int32, copy=False))
            flat_rows.append(np.full(idx.size, position, dtype=np.int32))
        if not ids:
            return cls(
                ids=(),
                rows=np.zeros(0, dtype=np.int32),
                offsets=np.zeros(1, dtype=np.int64),
                counts=np.zeros(1, dtype=np.int64),
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
            width=width,
        )

    def candidates(self, query: SDR, *, limit: int) -> tuple[str, ...]:
        """按稀有位的倒排表 gather，最多 ``limit`` 条。"""
        if limit <= 0 or not self.ids or query.k == 0:
            return ()
        order = np.argsort(self.counts[query.idx], kind="stable")
        selected: set[int] = set()
        for bit in query.idx[order]:
            bit = int(bit)
            if bit < 0 or bit + 1 >= self.offsets.size:
                continue
            start, stop = int(self.offsets[bit]), int(self.offsets[bit + 1])
            for row in self.rows[start:stop]:
                selected.add(int(row))
                if len(selected) >= limit:
                    return tuple(self.ids[row] for row in selected)
        return tuple(self.ids[row] for row in selected)
