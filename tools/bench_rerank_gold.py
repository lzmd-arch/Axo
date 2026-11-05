"""中文金标准上的第二段重排实验（预注册见 docs/prereg-gold-rerank-zh-v1.md）。

第一段仍是 `memcore.recall.recall()`；本工具截下它阈值后的候选，在同一批候选上比较：

* baseline：不重排；
* heuristic：预注册里写死的无监督权重；
* learned：线性 RankNet，5 折按查询 id 的 CRC32 分折（探索）。

指标用**产品选择后**的注入序列（MMR + 预算 + max_items=8），不是候选 top-k。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_locomo_rerank import fit_ranknet, heuristic_scores  # noqa: E402

from axo.bench import build_store, load_goldset  # noqa: E402
from axo.bench.metrics import (  # noqa: E402
    QueryOutcome,
    false_injection_rate,
    hit_at_k,
    mrr,
    no_answer_injections,
    recall_at_k,
)
from axo.memory.rerank import extract_features  # noqa: E402
from memcore.recall import RecallConfig, recall  # noqa: E402

DEFAULT_MODEL_DIR = ROOT / "build" / "bundle" / "embedding" / "bge-small-zh-v1.5"
FOLDS = 5


def outcome_for(record: dict, injected: tuple[str, ...]) -> QueryOutcome:
    return QueryOutcome(
        query_id=record["query_id"],
        category=record["category"],
        injected=injected,
        expect=tuple(record["expect"]),
        optional=tuple(record["optional"]),
        never=tuple(record["never"]),
    )


def summarize(outcomes: list[QueryOutcome]) -> dict[str, float]:
    return {
        **{f"hit@{k}": hit_at_k(outcomes, k) for k in (1, 3, 5, 8)},
        "recall@8": recall_at_k(outcomes, 8),
        "mrr": mrr(outcomes),
        "false_injection_rate": false_injection_rate(outcomes),
        "no_answer_injections": no_answer_injections(outcomes),
        "answered": sum(1 for outcome in outcomes if outcome.answered),
    }


def per_category(outcomes: list[QueryOutcome]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category, []).append(outcome)
    return {
        category: {
            "answered": sum(1 for outcome in rows if outcome.answered),
            "hit@1": hit_at_k(rows, 1),
            "hit@5": hit_at_k(rows, 5),
        }
        for category, rows in sorted(buckets.items())
    }


def lookup_hook(scores: dict[str, float]):
    def hook(_query: str, items) -> list[float]:
        return [float(scores.get(item.memory_id, 0.0)) for item in items]

    return hook


def rerun(store, query, vector, space_id, scores: dict[str, float], config) -> tuple[str, ...]:
    ledger = recall(
        store,
        query_text=query.query,
        budget_tokens=query.budget,
        query_vector=vector,
        space_id=space_id,
        config=config,
        settle=False,
        rerank=lookup_hook(scores),
    )
    return ledger.injected_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中文金标准重排实验")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "benchmarks" / "gold")
    parser.add_argument("--arm", default="bge", choices=("bge", "lexical"))
    parser.add_argument("--local-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep_results" / "gold_rerank")
    parser.add_argument("--tag", default="zh_v1")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--candidate-limit", type=int, default=200)
    args = parser.parse_args(argv)
    config = RecallConfig(max_items=8, candidate_limit=args.candidate_limit)
    if args.min_score is not None:
        config = replace(config, min_score=args.min_score)

    started = time.perf_counter()
    goldset = load_goldset(args.gold_dir)
    embedder = None
    if args.arm == "bge":
        from axo.llm.local_embed import LocalEmbeddingBackend

        embedder = LocalEmbeddingBackend(model_dir=args.local_model_dir)
    store = build_store(goldset.memories, embedder=embedder)
    vectors = None
    space_id = ""
    if embedder is not None:
        vectors = np.asarray(
            embedder.embed([query.query for query in goldset.queries]), dtype=np.float32
        )
        space_id = embedder.space_id

    records: list[dict] = []
    baseline_outcomes: list[QueryOutcome] = []
    feature_ms: list[float] = []
    first_ms: list[float] = []
    for index, query in enumerate(goldset.queries):
        vector = None if vectors is None else vectors[index]
        captured: list = []

        def capture(_text: str, items, pool=captured) -> list[float]:
            pool.extend(items)
            return [item.scores.fused for item in items]

        before = time.perf_counter()
        ledger = recall(
            store,
            query_text=query.query,
            budget_tokens=query.budget,
            query_vector=vector,
            space_id=space_id,
            config=config,
            settle=False,
            rerank=capture,
        )
        first_ms.append((time.perf_counter() - before) * 1000.0)
        feature_start = time.perf_counter()
        features = extract_features(query.query, captured)
        feature_ms.append((time.perf_counter() - feature_start) * 1000.0)
        record = {
            "query_id": query.query_id,
            "category": query.category,
            "expect": list(query.expect),
            "optional": list(query.optional),
            "never": list(query.never),
            "items": [{"memory_id": item.memory_id, "content": item.content} for item in captured],
            "labels": [1 if item.memory_id in set(query.expect) else 0 for item in captured],
            "features": features,
            "fused": [item.scores.fused for item in captured],
            "vector": vector,
        }
        records.append(record)
        baseline_outcomes.append(outcome_for(record, ledger.injected_ids))

    covered = sum(
        len({item["memory_id"] for item in record["items"]} & set(record["expect"]))
        for record in records
    )
    total_expect = sum(len(record["expect"]) for record in records)
    coverage = covered / total_expect if total_expect else 0.0

    for record in records:
        scores = heuristic_scores(record)
        record["heuristic"] = {
            item["memory_id"]: float(score)
            for item, score in zip(record["items"], scores, strict=True)
        }
    heuristic_outcomes = [
        outcome_for(
            record,
            rerun(
                store,
                goldset.queries[index],
                record["vector"],
                space_id,
                record["heuristic"],
                config,
            ),
        )
        for index, record in enumerate(records)
    ]

    folds = [zlib.crc32(record["query_id"].encode("utf-8")) % FOLDS for record in records]
    learned_outcomes: list[QueryOutcome | None] = [None] * len(records)
    for fold in range(FOLDS):
        train = [record for record, value in zip(records, folds, strict=True) if value != fold]
        test_indices = [index for index, value in enumerate(folds) if value == fold]
        weights, bias, mean, std = fit_ranknet(train, epochs=args.epochs)
        for index in test_indices:
            record = records[index]
            matrix = (record["features"] - mean) / std
            scores = matrix @ weights + bias
            score_map = {
                item["memory_id"]: float(score)
                for item, score in zip(record["items"], scores, strict=True)
            }
            learned_outcomes[index] = outcome_for(
                record,
                rerun(store, goldset.queries[index], record["vector"], space_id, score_map, config),
            )
    learned = [outcome for outcome in learned_outcomes if outcome is not None]

    result = {
        "kind": "gold rerank zh v1",
        "arm": args.arm,
        "memories": len(goldset.memories),
        "queries": len(goldset.queries),
        "candidate_coverage": coverage,
        "config": {"min_score": config.min_score, "candidate_limit": config.candidate_limit},
        "baseline": summarize(baseline_outcomes),
        "heuristic": summarize(heuristic_outcomes),
        "learned": summarize(learned),
        "per_category": {
            "baseline": per_category(baseline_outcomes),
            "heuristic": per_category(heuristic_outcomes),
            "learned": per_category(learned),
        },
        "first_stage_ms_median": float(np.median(first_ms)),
        "feature_ms_median": float(np.median(feature_ms)),
        "feature_ms_p90": float(np.percentile(feature_ms, 90)),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"候选覆盖 {coverage:.3f} | 特征 {result['feature_ms_median']:.1f} ms/查询")
    for name in ("baseline", "heuristic", "learned"):
        row = result[name]
        print(
            f"{name:<9} hit@1 {row['hit@1']:.3f} / hit@5 {row['hit@5']:.3f} / "
            f"hit@8 {row['hit@8']:.3f} | 覆盖@8 {row['recall@8']:.3f} | "
            f"MRR {row['mrr']:.3f} | 误注入 {row['false_injection_rate']:.3f}"
        )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
