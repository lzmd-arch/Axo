"""召回的第二段重排：**只看候选与查询，不碰库、不调模型。**

第一段召回已经把候选压到 200 条并算好了通道分；这一层只回答"同样这 200 条，
谁该排在前面"。所以它不需要 store、不需要 embedding、不需要网络——特征全是
查询与正文的字面/结构量，加第一段已经给出的分数。

这条边界是刻意的：重排器若能从库里再捞候选，它就不是"读出"而是"第二套召回"，
两个阶段出错时也无法归因。:func:`memcore.recall.recall` 的 ``rerank`` 钩子传进来的
就是这 200 条候选。

模型本身是一个线性打分（特征见 :data:`FEATURE_NAMES`）。线性不是省事：预注册
里把特征先定死，训练只学 11 个权重，过拟合的空间小，出问题时也能逐项看。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from memcore.audit import RecalledMemory

__all__ = [
    "FEATURE_NAMES",
    "HeuristicReranker",
    "LOCOMO_V1",
    "LinearReranker",
    "extract_features",
]

FEATURE_NAMES: tuple[str, ...] = (
    "fused",
    "lexical",
    "semantic",
    "semantic_raw",
    "bm25",
    "word_overlap",
    "char_ngram",
    "number_overlap",
    "entity_overlap",
    "length_ratio",
    "rank_reciprocal",
)

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")
_ENTITY_STOP = {
    "I",
    "The",
    "A",
    "An",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "Did",
    "Do",
    "Does",
    "Is",
    "Are",
    "Was",
    "Were",
    "Can",
    "Could",
    "Would",
    "Should",
    "Which",
    "If",
    "In",
    "On",
    "At",
    "For",
    "To",
}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    flat = " ".join(text.lower().split())
    return {flat[index : index + n] for index in range(max(0, len(flat) - n + 1))}


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text.lower()))


def _entities(text: str) -> set[str]:
    found = {item for item in _ENTITY.findall(text) if item not in _ENTITY_STOP}
    # 中文没有大小写；引号里的词当作实体。
    for left, right in re.findall(r'"([^"]{1,30})"|「([^」]{1,30})」', text):
        found.add(left or right)
    return {item for item in found if item}


def _jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class _Profile:
    words: tuple[str, ...]
    ngrams: frozenset[str]
    numbers: frozenset[str]
    lowered: str


@lru_cache(maxsize=8192)
def _profile(text: str) -> _Profile:
    """一条正文的字面画像。**缓存的是纯函数结果**，不改变任何特征值。

    同一个库里的同一条记忆会在很多轮召回里重复出现；每次重新分词/切 n-gram 是
    O(正文长度)，200 条候选乘上去就是几十毫秒。缓存把这条成本摊到"第一次见到
    这条记忆"上，重排的稳态成本才进得了 10 ms 预算。
    """
    return _Profile(
        words=tuple(_words(text)),
        ngrams=frozenset(_char_ngrams(text)),
        numbers=frozenset(_numbers(text)),
        lowered=text.lower(),
    )


def _bm25(query_words: list[str], documents: list[list[str]]) -> list[float]:
    """候选池内 BM25。IDF 用这 N 条候选估计，不碰全库统计。"""
    count = len(documents)
    if count == 0 or not query_words:
        return [0.0] * count
    document_frequency: Counter[str] = Counter()
    for words in documents:
        document_frequency.update(set(words))
    average_length = sum(len(words) for words in documents) / count or 1.0
    idf = {
        term: math.log(1.0 + (count - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }
    k1, b = 1.2, 0.75
    query_terms = set(query_words)
    scores: list[float] = []
    for words in documents:
        frequencies = Counter(words)
        length = len(words) or 1
        total = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            total += (
                idf.get(term, 0.0)
                * tf
                * (k1 + 1)
                / (tf + k1 * (1 - b + b * length / average_length))
            )
        scores.append(total)
    return scores


@dataclass(frozen=True)
class HeuristicReranker:
    """预注册里的无监督权重：**不训练、可解释、可逐项核对**。

    权重在看到任何结果之前写死（``docs/prereg-locomo-rerank-v1.md``）：
    ``0.60*fused + 0.15*bm25_norm + 0.10*number + 0.10*entity + 0.05*rank``。
    ``bm25_norm`` 只在**本查询的候选内**做 min-max，不跨查询比较。
    """

    def __call__(self, query: str, items: Sequence[RecalledMemory]) -> list[float]:
        matrix = extract_features(query, items)
        if not len(matrix):
            return []
        columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
        bm25 = matrix[:, columns["bm25"]]
        span = float(bm25.max() - bm25.min())
        bm25 = (bm25 - bm25.min()) / span if span > 0 else np.zeros_like(bm25)
        scores = (
            0.60 * matrix[:, columns["fused"]]
            + 0.15 * bm25
            + 0.10 * matrix[:, columns["number_overlap"]]
            + 0.10 * matrix[:, columns["entity_overlap"]]
            + 0.05 * matrix[:, columns["rank_reciprocal"]]
        )
        return [float(value) for value in scores]


@dataclass(frozen=True)
class LinearReranker:
    """``score = bias + w · (x - mean) / std``。权重由训练/标定给出。"""

    weights: tuple[float, ...]
    bias: float = 0.0
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(f"权重有 {len(self.weights)} 个，特征有 {len(FEATURE_NAMES)} 个。")

    def __call__(self, query: str, items: Sequence[RecalledMemory]) -> list[float]:
        matrix = extract_features(query, items)
        if self.mean is not None and self.std is not None:
            matrix = (matrix - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
                self.std, dtype=np.float32
            )
        scores = matrix @ np.asarray(self.weights, dtype=np.float32) + self.bias
        return [float(value) for value in scores]


def extract_features(query: str, items: Sequence[RecalledMemory]) -> np.ndarray:
    """``(len(items), len(FEATURE_NAMES))`` 的 float32 矩阵。

    **特征顺序就是 :data:`FEATURE_NAMES` 的顺序**；训练与产品加载同一份。
    """
    if not items:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    contents = [item.content for item in items]
    profiles = [_profile(content) for content in contents]
    query_profile = _profile(query)
    query_words = list(query_profile.words)
    query_word_set = set(query_words)
    query_ngrams = query_profile.ngrams
    query_numbers = query_profile.numbers
    query_entities = _entities(query)
    query_length = max(1, len(query))
    documents = [list(profile.words) for profile in profiles]
    bm25 = _bm25(query_words, documents)
    ranks = np.argsort(np.argsort([-item.scores.fused for item in items]))

    rows: list[list[float]] = []
    for index, item in enumerate(items):
        profile = profiles[index]
        words = set(profile.words)
        content = item.content
        numbers = profile.numbers
        entities = {entity.lower() for entity in query_entities}
        lowered = profile.lowered
        rows.append(
            [
                float(item.scores.fused),
                float(item.scores.lexical),
                float(item.scores.semantic),
                float(item.scores.semantic_raw),
                float(bm25[index]),
                _jaccard(query_word_set, words),
                _jaccard(query_ngrams, profile.ngrams),
                (len(query_numbers & numbers) / len(query_numbers)) if query_numbers else 0.0,
                (
                    sum(1 for entity in entities if entity in lowered) / len(entities)
                    if entities
                    else 0.0
                ),
                min(query_length, len(content)) / max(query_length, len(content), 1),
                1.0 / (1.0 + float(ranks[index])),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


#: LoCoMo 英文上按对话留一训练后的线性重排权重（10 折平均；最终全量拟合在
#: ``sweep_results/locomo_rerank/rerank_v1_cached.json`` 的 ``final`` 里）。
#: **只在英文 LoCoMo 上验过**；产品默认关，打开前看
#: ``benchmarks/locomo/RERANK.md`` 的边界一节。
LOCOMO_V1 = LinearReranker(
    weights=(
        0.2766,
        -0.2612,
        0.3352,
        0.3288,
        0.2927,
        -0.2044,
        0.1445,
        0.2271,
        0.2293,
        -0.3027,
        0.0142,
    ),
    bias=3.263,
    mean=(0.1585, 0.1044, 0.1389, 0.5758, 2.0290, 0.0540, 0.0545, 0.0, 0.2846, 0.5177, 0.0056),
    std=(0.0349, 0.0173, 0.1021, 0.0610, 1.7529, 0.0367, 0.0268, 0.0056, 0.4224, 0.2293, 0.0335),
)
