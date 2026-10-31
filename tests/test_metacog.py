"""元认知门控：特征怎么从账本里来、密度门怎么判。

**这里测的不是"门控好不好用"**——那是离线评测的事（``tools/eval_metacog.py`` 在
真实库上跑）。这里钉的是三条会被悄悄改坏的性质：

1. 特征**只从账本能算的量**里来，空召回是合法输入（全 0），不是异常；
2. 密度门的排序性质：离校准集越远，分位越低——反过来的实现也能让均值好看，
   但它在拦截陌生话题时会恰好反过来；
3. 校准点自己的分位要**排除自身**，否则每个点的距离都是 0、阈值整体偏高。

**不建任何 Qt 对象**：这一层是纯的，连 QCoreApplication 都不需要。
"""

from __future__ import annotations

import numpy as np
import pytest

from axo.metacog import AXO_COLUMNS, KnnDensityGate, ledger_features
from memcore.audit import ChannelScores, RecalledMemory, RecallLedger


def _item(score: float, *, semantic_raw: float = 0.0, lexical: float = 0.0) -> RecalledMemory:
    """一条被注入的记忆。通道分数按需给——特征里只有 top 那条的它们会被读。"""
    return RecalledMemory(
        memory_id=f"m{int(score * 1000)}",
        content="正文",
        kind="fact",
        tier="working",
        weight=1.0,
        tokens=10,
        scores=ChannelScores(
            lexical=lexical,
            semantic=score,
            fused=score,
            semantic_raw=semantic_raw,
        ),
    )


def _ledger(items: list[RecalledMemory], *, scanned: int = 100) -> RecallLedger:
    return RecallLedger(query_preview="测试", scanned=scanned, injected=tuple(items))


# ── 特征 ────────────────────────────────────────────────────────────────────


def test_empty_recall_is_all_zeros_not_an_exception() -> None:
    """这一轮记忆里什么都没有是**正常状态**：门控要把它说成一个数。"""
    features = ledger_features(_ledger([]))
    assert features["top_score"] == 0.0
    assert features["mean_topk"] == 0.0
    assert features["density"] == 0.0
    assert set(AXO_COLUMNS) <= set(features)


def test_top_and_mean_use_the_scores_in_descending_order() -> None:
    ledger = _ledger([_item(0.2), _item(0.8), _item(0.5)], scanned=200)
    features = ledger_features(ledger)
    assert features["top_score"] == pytest.approx(0.8)
    assert features["mean_topk"] == pytest.approx((0.8 + 0.5 + 0.2) / 3)
    # 密度用**扫过的条数**做分母：3/200，不是 3/3。
    assert features["density"] == pytest.approx(3 / 200)


def test_mean_only_looks_at_the_first_five() -> None:
    """``mean_topk`` 是前 5 条的平均，不是全体的平均——与 v11 的 k=5 同数。"""
    items = [_item(1.0 - index * 0.1) for index in range(10)]
    features = ledger_features(_ledger(items))
    first_five = [1.0, 0.9, 0.8, 0.7, 0.6]
    assert features["mean_topk"] == pytest.approx(sum(first_five) / 5)


def test_channel_spread_is_about_the_top_item() -> None:
    """分歧只看**最强那条**：弱的那些本来就可能只有一条通道撑着。"""
    ledger = _ledger([_item(0.9, semantic_raw=0.33), _item(0.1)])
    features = ledger_features(ledger)
    assert features["semantic_raw_top"] == pytest.approx(0.33)
    assert features["channel_spread"] >= 0.0


def test_a_malformed_item_does_not_take_the_turn_down() -> None:
    """账本里混进一条形状不对的条目时，特征照样算得出来（当 0 分）。"""

    class _Broken:
        scores = None

    ledger = RecallLedger(query_preview="x", scanned=10, injected=(_item(0.6), _Broken()))
    features = ledger_features(ledger)
    assert features["top_score"] == pytest.approx(0.6)


# ── 密度门 ──────────────────────────────────────────────────────────────────


def _cloud(count: int = 40, seed: int = 7) -> list[dict[str, float]]:
    """一圈"有记忆支撑"的校准点：三个特征各自 N(0, 1)，稍微相关一点更像真实账本。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(count):
        top = float(rng.normal(0.6, 0.1))
        rows.append(
            {
                "top_score": top,
                "mean_topk": top * 0.8 + float(rng.normal(0, 0.03)),
                "density": float(abs(rng.normal(0.02, 0.005))),
            }
        )
    return rows


def test_points_inside_the_calibration_cloud_rank_higher_than_far_ones() -> None:
    gate = KnnDensityGate().fit(_cloud())
    inside = gate.percentiles(gate.confidence_batch([_cloud()[0]]))[0]
    far = gate.percentiles(
        gate.confidence_batch([{"top_score": 0.0, "mean_topk": 0.0, "density": 0.0}])
    )[0]
    assert inside > far


def test_a_far_point_decides_uncertain_and_a_typical_one_does_not() -> None:
    gate = KnnDensityGate().fit(_cloud())
    typical = gate.decide(_cloud()[3])
    far = gate.decide({"top_score": 0.0, "mean_topk": 0.0, "density": 0.0})
    assert far.decision == "uncertain"
    assert typical.decision != "uncertain"
    assert 0.0 <= far.percentile <= 1.0


def test_percentiles_are_monotone_in_confidence() -> None:
    gate = KnnDensityGate().fit(_cloud())
    low = gate.percentiles(
        np.array([gate.confidence_batch([{"top_score": -5, "mean_topk": -5, "density": -1}])[0]])
    )[0]
    high = gate.percentiles(np.array([gate.confidence_batch([_cloud()[0]])[0]]))[0]
    assert high > low


def test_calibration_points_do_not_count_themselves_as_neighbours() -> None:
    """排除自身之后，校准点的分位不再全是 1.0——否则阈值会系统性偏高。"""
    rows = _cloud()
    gate = KnnDensityGate().fit(rows)
    own = gate.confidence_batch(rows, exclude_self=True)
    assert np.isfinite(own).all()
    assert np.all(own < 0.0)


def test_the_gate_is_deterministic() -> None:
    rows = _cloud()
    first = (
        KnnDensityGate().fit(rows).decide({"top_score": 0.55, "mean_topk": 0.44, "density": 0.01})
    )
    second = (
        KnnDensityGate().fit(rows).decide({"top_score": 0.55, "mean_topk": 0.44, "density": 0.01})
    )
    assert first == second


def test_a_tiny_calibration_set_is_rejected() -> None:
    """三五个点配 k=5 会让近邻距离退化——当场报错比给一个假阈值好。"""
    with pytest.raises(ValueError):
        KnnDensityGate().fit(_cloud()[:3])
    with pytest.raises(ValueError):
        KnnDensityGate().fit([])


def test_percentiles_before_fit_raise() -> None:
    with pytest.raises(ValueError):
        KnnDensityGate().percentiles(np.zeros(1))
