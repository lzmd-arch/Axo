"""元认知门控（MG）：这一轮召回**够不够支撑一次有记忆的回答**。

## 它是什么，不是什么

这是 BioSNN-Plug 里 ``run_metacognitive_gate_v11.py`` 那套机制的**Axo 版**：

* 原版：在 4 维情景记忆任务上，取读出残差的 ``[max_errors, errors, state_norm]``
  三个标量，用 kNN(5) 密度门给出置信度，阈值只在校准集上取（覆盖 0.80）。
* 这一版：把这三个标量换成**召回账本上真实可算的量**——
  ``top_score``（最强的一条）、``mean_topk``（整体能量）、``density``（支撑密度）。
  机制不变：校准集上标准化 → kNN 密度 → 经验分布给分位 → 阈值。

**它不是 SNN。** 原版也没有 512 个神经元——那是另一件还没做的事（见 BioSNN-Plug
计划书 amend 里的排期条目）。把这一层写成"SNN 门控"会让后来的人以为脉冲那部分
已经存在。

## 为什么先做成纯层、且先不接线

判据是**离线指标**：陌生话题的拦截率、有记忆支撑时的保留率，以及**它比"召回有没有
命中"这条平凡规则强多少**。平凡规则（``injected`` 非空就注入）在同一条链上白拿，
而一个不比它强的门控只是把阈值换个说法。指标出来之前不改变任何回答路径。

## 三个决策档

置信度先换成**校准集上的分位**（0–1，越大越像"见过的"），再按两档切：

* ``>= high_percentile``（默认 0.7）→ ``inject``
* ``>= medium_percentile``（默认 0.3）→ ``inject_with_caution``
* 否则 → ``uncertain``

分位而不是原始距离：原始距离的尺度随库变，而"和校准集比，这条排在第几成"换了库
仍然读得懂。两档的数值直接沿用任务书的 0.7 / 0.3。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AXO_COLUMNS",
    "GateResult",
    "KnnDensityGate",
    "ledger_features",
]

#: 进入门控的三个特征（顺序固定：校准矩阵按它排列）。
#:
#: 与 v11 的三个标量一一对应：``top_score`` ↔ ``max_errors``（最强证据）、
#: ``mean_topk`` ↔ ``errors``（整体能量）、``density`` ↔ ``state_norm``（支撑的
#: 尺度）。另外两个（``channel_spread`` / ``semantic_raw_top``）留作消融，默认不进。
AXO_COLUMNS: tuple[str, ...] = ("top_score", "mean_topk", "density")

#: 账本里最多看前几条来算 ``mean_topk``。与 v11 的 ``neighbors=5`` 同数——它不是
#: 从 Axo 的召回调出来的，**是一个随机制带过来的常数**，所以写在这里让人能看见它
#: 是个选择而不是发现。
TOPK = 5


def _fused(item: Any) -> float:
    """一条召回的融合分。形状不对就当 0——账本永远是这条链上最可信的来源。"""
    scores = getattr(item, "scores", None)
    value = getattr(scores, "fused", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _channel_spread(item: Any) -> float:
    """一条召回在各通道上的分数离散度（v11 ``disagreement`` 的类比）。

    四个通道的分数本就是同一量纲（融合前各自归一化过），所以直接取标准差；
    只用了一条通道时它是 0——**0 在这里的含义是"没有分歧可谈"，不是"完全一致"**。
    """
    scores = getattr(item, "scores", None)
    if scores is None:
        return 0.0
    values: list[float] = []
    for name in ("temporal", "lexical", "semantic", "graph"):
        try:
            values.append(float(getattr(scores, name)))
        except (TypeError, ValueError):
            values.append(0.0)
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def _semantic_raw(item: Any) -> float:
    scores = getattr(item, "scores", None)
    if scores is None:
        return 0.0
    try:
        return float(scores.semantic_raw)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def ledger_features(ledger: Any) -> dict[str, float]:
    """一轮召回 → 门控特征。

    Args:
        ledger: :class:`memcore.audit.RecallLedger`（或同形状的替身）。只读
            ``injected`` / ``scanned``——**不读 ``rejected``**：被淘汰的候选在
            "分数不够"与"重复"之间混在一起，拿它做密度会把两件事算成一件。

    Returns:
        五个键（:data:`AXO_COLUMNS` 三个，加两个消融用的）。

    **空召回是合法输入**：特征全 0。它意味着"这一轮记忆里什么都没有"，而门控要把
    这句话说成一个数，而不是抛异常——抛出去会让聊天在一轮没有记忆时直接失败。
    """
    injected = list(getattr(ledger, "injected", ()) or ())
    scanned = getattr(ledger, "scanned", 0)
    try:
        scanned = int(scanned)
    except (TypeError, ValueError):
        scanned = 0

    if not injected:
        return {
            "top_score": 0.0,
            "mean_topk": 0.0,
            "density": 0.0,
            "channel_spread": 0.0,
            "semantic_raw_top": 0.0,
        }

    ordered = sorted(injected, key=lambda item: -_fused(item))
    top = ordered[0]
    scores = [max(0.0, _fused(item)) for item in ordered[:TOPK]]
    return {
        "top_score": max(0.0, _fused(top)),
        "mean_topk": float(np.mean(scores)) if scores else 0.0,
        # 密度用**扫过的条数**做分母：同一句话在 20 条的库里与 2000 条的库里，
        # "命中 3 条"是完全不同的两件事，而它们的分母正是这个差别。
        "density": float(len(injected)) / float(max(1, scanned)),
        "channel_spread": _channel_spread(top),
        "semantic_raw_top": _semantic_raw(top),
    }


@dataclass(frozen=True)
class GateResult:
    """一次门控的结论。

    ``percentile`` 是**校准集上的分位**（0–1）：0.5 表示"和校准集里一半的点一样
    常见"。它比原始距离好读，也比一个裸阈值稳——库换了，分位的含义不变。
    """

    confidence: float
    """kNN 密度分（越大越像校准集）。**有量纲、随库变**，只在同一份校准下可比。"""

    percentile: float
    """``confidence`` 在校准集里的经验分位（0–1）。对外只说这个。"""

    decision: str
    """``inject`` / ``inject_with_caution`` / ``uncertain``。"""


@dataclass
class KnnDensityGate:
    """校准在**已知有记忆支撑**的查询上，然后按密度判一条新查询。

    用法（离线）：``fit(rows)`` → 需要阈值时 ``decide`` 自己按分位切；批次评测用
    :meth:`confidence_batch` + :meth:`percentiles`。没有全局状态，同输入同输出。
    """

    columns: tuple[str, ...] = AXO_COLUMNS
    neighbors: int = 5
    high_percentile: float = 0.7
    medium_percentile: float = 0.3
    mean: np.ndarray = field(init=False, repr=False)
    scale: np.ndarray = field(init=False, repr=False)
    reference: np.ndarray = field(init=False, repr=False)
    _sorted_confidence: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(len(self.columns))
        self.scale = np.ones(len(self.columns))
        self.reference = np.zeros((0, len(self.columns)))
        self._sorted_confidence = np.zeros(0)

    # ── 校准 ────────────────────────────────────────────────────────────────

    def fit(self, rows: Sequence[Mapping[str, float]]) -> KnnDensityGate:
        """用校准集拟合。**校准集必须是"有记忆支撑"的查询**（见模块文档）。

        Raises:
            ValueError: 校准集为空，或 ``neighbors`` 大于校准点数——后者会让
                "近邻距离"恒为无穷，任何阈值都失去意义。
        """
        if not rows:
            raise ValueError("校准集是空的：没有校准就定不出阈值。")
        matrix = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.columns] for row in rows],
            dtype=np.float64,
        )
        if matrix.shape[0] < self.neighbors:
            raise ValueError(
                f"校准集只有 {matrix.shape[0]} 条，少于 neighbors={self.neighbors}："
                "近邻距离会退化。"
            )
        self.mean = matrix.mean(axis=0)
        self.scale = np.maximum(matrix.std(axis=0), 1e-8)
        self.reference = (matrix - self.mean) / self.scale
        # 校准点自己的分位要**排除自身**，否则每个点的距离都是 0、分位系统性偏高。
        self._sorted_confidence = np.sort(self.confidence_batch(rows, exclude_self=True))
        return self

    # ── 打分 ────────────────────────────────────────────────────────────────

    def _normalized(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        values = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.columns] for row in rows],
            dtype=np.float64,
        )
        return (values - self.mean) / self.scale

    def confidence_batch(
        self, rows: Sequence[Mapping[str, float]], *, exclude_self: bool = False
    ) -> np.ndarray:
        """每行到校准集**最近 k 个点的平均距离**取负。越大越像校准集。"""
        if not rows:
            return np.zeros(0)
        values = self._normalized(rows)
        # 展开的平方距离：(|x|² + |r|² − 2x·r)。照抄 v11 的算法，
        # 两次矩阵乘就能算完一整批。
        distances = (
            (values**2).sum(axis=1)[:, None]
            + (self.reference**2).sum(axis=1)[None, :]
            - 2.0 * values @ self.reference.T
        )
        if exclude_self:
            np.fill_diagonal(distances, np.inf)
        nearest = np.partition(distances, self.neighbors - 1, axis=1)[:, : self.neighbors]
        return -nearest.mean(axis=1)

    def percentiles(self, confidence: np.ndarray) -> np.ndarray:
        """相对校准集的经验分位（0–1）。用右插入，最大值给 1.0。"""
        if self._sorted_confidence.size == 0:
            raise ValueError("还没有 fit：分位无从算起。")
        values = np.atleast_1d(np.asarray(confidence, dtype=np.float64))
        return np.searchsorted(self._sorted_confidence, values, side="right") / float(
            self._sorted_confidence.size
        )

    # ── 决策 ────────────────────────────────────────────────────────────────

    def decide(self, features: Mapping[str, float]) -> GateResult:
        """一条特征 → 三档结论。"""
        confidence = float(self.confidence_batch([features])[0])
        percentile = float(self.percentiles(np.array([confidence]))[0])
        if percentile >= self.high_percentile:
            decision = "inject"
        elif percentile >= self.medium_percentile:
            decision = "inject_with_caution"
        else:
            decision = "uncertain"
        return GateResult(confidence=confidence, percentile=percentile, decision=decision)
