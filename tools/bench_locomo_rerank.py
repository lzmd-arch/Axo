"""LoCoMo 第二段重排实验（预注册见 docs/prereg-locomo-rerank-v1.md）。

第一段仍是 `memcore.recall.recall()`；本工具用它的 `rerank=` 钩子把 top-200
候选原样截下来，然后在**同一批候选**上比较三种读出：

* baseline：按第一段 `fused` 排；
* heuristic：预注册里写死的无监督权重；
* learned：线性 RankNet，按 conversation 留一训练。

它不写库、不碰用户数据；库是 `:memory:`，召回全部 `settle=False`。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_locomo import build_goldset, load_payload, make_embedder  # noqa: E402

from axo.bench import CREATED_AT, build_store  # noqa: E402
from axo.memory.rerank import FEATURE_NAMES, extract_features  # noqa: E402
from memcore.recall import RecallConfig, recall  # noqa: E402

DEFAULT_SHA = "553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365"
K_VALUES = (1, 3, 5, 8)


def metrics(rankings: list[list[str]], rows: list[dict]) -> dict:
    answered = [row for row in rows if row["expect"]]
    result = {}
    for k in K_VALUES:
        hits = 0
        found = 0
        total = 0
        for ranking, row in zip(rankings, rows, strict=True):
            if not row["expect"]:
                continue
            top = set(ranking[:k])
            hits += 1 if top & set(row["expect"]) else 0
            found += len(top & set(row["expect"]))
            total += len(row["expect"])
        result[f"hit@{k}"] = hits / len(answered) if answered else 0.0
        result[f"recall@{k}"] = found / total if total else 0.0
    reciprocal = 0.0
    for ranking, row in zip(rankings, rows, strict=True):
        expect = set(row["expect"])
        for index, memory_id in enumerate(ranking, start=1):
            if memory_id in expect:
                reciprocal += 1.0 / index
                break
    result["mrr"] = reciprocal / len(answered) if answered else 0.0
    result["answered"] = len(answered)
    return result


def cat14_hit5(rankings: list[list[str]], rows: list[dict]) -> float:
    subset = [
        (ranking, row)
        for ranking, row in zip(rankings, rows, strict=True)
        if row["category"] in {"1", "2", "3", "4"} and row["expect"]
    ]
    if not subset:
        return 0.0
    return sum(1 for ranking, row in subset if set(ranking[:5]) & set(row["expect"])) / len(subset)


def rank_by(records: list[dict], key: str) -> list[list[str]]:
    return [
        [record["items"][index]["memory_id"] for index in np.argsort(-np.asarray(record[key]))]
        for record in records
    ]


def heuristic_scores(record: dict) -> np.ndarray:
    matrix = record["features"]
    if not len(matrix):
        return np.zeros(0, dtype=np.float32)
    names = {name: index for index, name in enumerate(FEATURE_NAMES)}
    bm25 = matrix[:, names["bm25"]]
    span = float(bm25.max() - bm25.min())
    bm25 = (bm25 - bm25.min()) / span if span > 0 else np.zeros_like(bm25)
    return (
        0.60 * matrix[:, names["fused"]]
        + 0.15 * bm25
        + 0.10 * matrix[:, names["number_overlap"]]
        + 0.10 * matrix[:, names["entity_overlap"]]
        + 0.05 * matrix[:, names["rank_reciprocal"]]
    )


def pairs_for(records: list[dict], *, seed: int, negatives: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    positives: list[np.ndarray] = []
    negatives_out: list[np.ndarray] = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=bool)
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        if not len(pos) or not len(neg):
            continue
        matrix = record["features"]
        for index in pos:
            chosen = rng.sample(list(neg), min(negatives, len(neg)))
            for other in chosen:
                positives.append(matrix[index])
                negatives_out.append(matrix[other])
    if not positives:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0, len(FEATURE_NAMES)))
    return np.asarray(positives, dtype=np.float32), np.asarray(negatives_out, dtype=np.float32)


def fit_ranknet(
    records: list[dict],
    *,
    seed: int = 0,
    epochs: int = 220,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    negatives: int = 12,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if not records:
        zeros = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        return zeros, 0.0, zeros, np.ones(len(FEATURE_NAMES), dtype=np.float32)
    matrix = np.concatenate([record["features"] for record in records])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    for record in records:
        record["features_std"] = (record["features"] - mean) / std

    pos, neg = pairs_for(records, seed=seed, negatives=negatives)
    if not len(pos):
        return np.zeros(len(FEATURE_NAMES)), 0.0, mean, std
    pos = (pos - mean) / std
    neg = (neg - mean) / std

    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    bias = np.float32(0.0)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        difference = pos - neg
        logits = difference @ weights + bias
        sigma = 1.0 / (1.0 + np.exp(np.clip(logits, -60, 60)))
        grad_w = -(sigma[:, None] * difference).mean(axis=0) + l2 * weights
        grad_b = -sigma.mean()
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
        weights -= (
            learning_rate * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + eps)
        )
        bias -= learning_rate * (m_b / (1 - beta1**step)) / (np.sqrt(v_b / (1 - beta2**step)) + eps)
    return weights, float(bias), mean, std


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoCoMo 召回重排实验")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expect-sha256", default=DEFAULT_SHA)
    parser.add_argument("--arm", default="nomic", choices=("lexical", "nomic", "local"))
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--local-model-dir", default="")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep_results" / "locomo_rerank")
    parser.add_argument("--tag", default="rerank_v1")
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    payload, digest = load_payload(args.data, args.expect_sha256)
    if args.limit_samples:
        payload = payload[: args.limit_samples]
    goldset, info = build_goldset(payload, args.data.parent)
    embedder = make_embedder(args.arm, args)
    store = build_store(goldset.memories, embedder=embedder)
    vectors = None
    space_id = ""
    if embedder is not None:
        vectors = np.asarray(
            embedder.embed([query.query for query in goldset.queries]), dtype=np.float32
        )
        space_id = embedder.space_id

    records: list[dict] = []
    first_stage_ms: list[float] = []
    feature_ms: list[float] = []
    for index, query in enumerate(goldset.queries):
        vector = None if vectors is None else vectors[index]
        captured: list = []

        def capture(_text: str, items, pool=captured) -> list[float]:
            pool.extend(items)
            return [item.scores.fused for item in items]

        before = time.perf_counter()
        recall(
            store,
            query_text=query.query,
            budget_tokens=query.budget,
            query_vector=vector,
            space_id=space_id,
            config=RecallConfig(max_items=8, candidate_limit=200),
            now=CREATED_AT,
            settle=False,
            rerank=capture,
        )
        first_stage_ms.append((time.perf_counter() - before) * 1000.0)
        if not captured:
            continue
        feature_start = time.perf_counter()
        features = extract_features(query.query, captured)
        feature_ms.append((time.perf_counter() - feature_start) * 1000.0)
        items = [
            {
                "memory_id": item.memory_id,
                "content": item.content,
                "session_id": item.session_id,
                "fused": item.scores.fused,
            }
            for item in captured
        ]
        labels = [1 if item.memory_id in set(query.expect) else 0 for item in captured]
        records.append(
            {
                "query_id": query.query_id,
                "sample": query.query_id.split(":", 1)[0],
                "category": query.category,
                "expect": list(query.expect),
                "items": items,
                "labels": labels,
                "features": features,
                "fused": [item.scores.fused for item in captured],
            }
        )

    covered = 0
    total_expect = 0
    for record in records:
        top = {item["memory_id"] for item in record["items"]}
        covered += len(top & set(record["expect"]))
        total_expect += len(record["expect"])
    candidate_coverage = covered / total_expect if total_expect else 0.0

    baseline_rankings = rank_by(records, "fused")
    baseline = metrics(baseline_rankings, records)

    for record in records:
        record["heuristic"] = heuristic_scores(record).tolist()
    heuristic_rankings = rank_by(records, "heuristic")
    heuristic = metrics(heuristic_rankings, records)

    samples = sorted({record["sample"] for record in records})
    learned_rankings: list[list[str]] = [[] for _ in records]
    fold_weights: list[dict] = []
    for held_out in samples:
        train = [record for record in records if record["sample"] != held_out]
        test_indices = [
            index for index, record in enumerate(records) if record["sample"] == held_out
        ]
        weights, bias, mean, std = fit_ranknet(train, epochs=args.epochs)
        fold_weights.append(
            {
                "held_out": held_out,
                "weights": weights.tolist(),
                "bias": bias,
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )
        for index in test_indices:
            record = records[index]
            matrix = (record["features"] - mean) / std
            scores = matrix @ weights + bias
            learned_rankings[index] = [
                record["items"][item_index]["memory_id"] for item_index in np.argsort(-scores)
            ]
    learned = metrics(learned_rankings, records)

    final_weights, final_bias, final_mean, final_std = fit_ranknet(records, epochs=args.epochs)

    result = {
        "kind": "locomo rerank v1",
        "arm": args.arm,
        "data_sha256": digest,
        "samples": len(payload),
        "memories": len(goldset.memories),
        "queries": len(goldset.queries),
        "records": len(records),
        "skipped_no_candidates": len(goldset.queries) - len(records),
        "candidate_limit": 200,
        "candidate_coverage": candidate_coverage,
        "features": list(FEATURE_NAMES),
        "baseline": baseline,
        "heuristic": heuristic,
        "learned": learned,
        "cat14": {
            "baseline": cat14_hit5(baseline_rankings, records),
            "heuristic": cat14_hit5(heuristic_rankings, records),
            "learned": cat14_hit5(learned_rankings, records),
        },
        "first_stage_ms_median": float(np.median(first_stage_ms)) if first_stage_ms else 0.0,
        "feature_ms_median": float(np.median(feature_ms)) if feature_ms else 0.0,
        "final": {
            "weights": final_weights.tolist(),
            "bias": final_bias,
            "mean": final_mean.tolist(),
            "std": final_std.tolist(),
        },
        "fold_weights": fold_weights,
        "feature_ms_p90": float(np.percentile(feature_ms, 90)) if feature_ms else 0.0,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "excluded": info["excluded"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"候选覆盖@200 {candidate_coverage:.3f} | 查询 {len(records)} | "
        f"特征 {result['feature_ms_median']:.1f} ms/查询"
    )
    for name in ("baseline", "heuristic", "learned"):
        row = result[name]
        print(
            f"{name:<9} hit@1 {row['hit@1']:.3f} / hit@5 {row['hit@5']:.3f} / "
            f"hit@8 {row['hit@8']:.3f} | 覆盖@8 {row['recall@8']:.3f} | MRR {row['mrr']:.3f}"
        )
    print(
        f"cat1-4 hit@5 基线 {result['cat14']['baseline']:.3f} / "
        f"特征 {result['cat14']['heuristic']:.3f} / 学习 {result['cat14']['learned']:.3f}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
