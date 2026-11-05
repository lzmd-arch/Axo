"""特征重排器：特征数量钉死、输出与候选等长、不产生 NaN。"""

from __future__ import annotations

import numpy as np

from axo.memory.rerank import (
    FEATURE_NAMES,
    HeuristicReranker,
    LinearReranker,
    extract_features,
)
from memcore.audit import ChannelScores, RecalledMemory


def item(content: str, fused: float = 0.5, session_id: str = "s1") -> RecalledMemory:
    return RecalledMemory(
        memory_id=content,
        content=content,
        kind="fact",
        tier="longterm",
        weight=0.8,
        tokens=10,
        scores=ChannelScores(lexical=fused, fused=fused),
        session_id=session_id,
    )


def test_feature_matrix_has_the_declared_width() -> None:
    matrix = extract_features(
        "When did Alice visit Paris in 2024?",
        [item("Alice: I visited Paris in June 2024."), item("Bob: I like tea.")],
    )
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_reranker_returns_one_score_per_candidate() -> None:
    reranker = LinearReranker(weights=tuple(1.0 for _ in FEATURE_NAMES))
    scores = reranker("alpha", [item("alpha one"), item("beta two")])
    assert len(scores) == 2
    assert all(isinstance(value, float) for value in scores)


def test_heuristic_returns_one_score_per_candidate() -> None:
    scores = HeuristicReranker()("alpha 2024", [item("alpha 2024"), item("beta")])
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_wrong_weight_count_is_rejected() -> None:
    try:
        LinearReranker(weights=(1.0,))
    except ValueError as exc:
        assert "特征" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("权重数量不对却没有报错")
