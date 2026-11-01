"""基质补全：线性一步 vs 脉冲群体（预注册 docs/prereg-substrate-spiking-v1.md）。

    python tools/bench_substrate_spiking.py --seeds 5
    python tools/bench_substrate_spiking.py --grid        # 验证集选参、留出集报数

判据 S1：脉冲 top-1 ≥ 线性 +5pp；S3：每次补全的额外延迟 < 20 ms。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from memcore.sdr import encode_assoc, encode_lexical
from memcore.substrate import Substrate

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "substrate"


def load(corpus: Path):
    memories = [
        json.loads(line) for line in (corpus / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [
        json.loads(line) for line in (corpus / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    return memories, queries


def build(memories):
    from types import SimpleNamespace

    items = [
        SimpleNamespace(memory_id=m["id"], lex_sdr=encode_lexical(m["content"]), assoc_sdr=None)
        for m in memories
    ]
    return Substrate.build(items)


def evaluate(sub: Substrate, queries, *, seed: int, neurons: int, steps: int):
    hits1 = hits5 = 0
    spiking_ms, linear_ms = [], []
    fired = 0
    for q in queries:
        probe = encode_assoc(encode_lexical(q["query"]))
        expect = set(q.get("expect") or [])

        start = time.perf_counter()
        sub.complete(probe, top=5)
        linear_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        spiking = [
            mid
            for mid, _ in sub.complete_spiking(
                probe, top=5, neurons=neurons, steps=steps, seed=seed
            )
        ]
        spiking_ms.append((time.perf_counter() - start) * 1000)

        hits1 += bool(spiking and spiking[0] in expect)
        hits5 += bool(expect and set(spiking[:5]) & expect)
        fired += bool(spiking)
    n = len(queries)
    return {
        "seed": seed,
        "spiking_hit1": hits1 / n,
        "spiking_hit5": hits5 / n,
        "spiking_ms": statistics.median(spiking_ms),
        "linear_ms": statistics.median(linear_ms),
    }


def linear_only(sub: Substrate, queries) -> dict:
    hits1 = hits5 = 0
    for q in queries:
        probe = encode_assoc(encode_lexical(q["query"]))
        expect = set(q.get("expect") or [])
        top = [mid for mid, _ in sub.complete(probe, top=5)]
        hits1 += bool(top and top[0] in expect)
        hits5 += bool(expect and set(top) & expect)
    n = len(queries)
    return {"hit1": hits1 / n, "hit5": hits5 / n}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--neurons", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--grid", action="store_true", help="验证集选参、留出集报数")
    args = parser.parse_args(argv)

    memories, queries = load(args.corpus)
    sub = build(memories)
    base = linear_only(sub, queries)
    print(f"语料：{len(memories)} 条记忆 / {len(queries)} 条查询")
    print(f"线性一步：hit@1 {base['hit1']:.3f}  hit@5 {base['hit5']:.3f}")

    if args.grid:
        half = len(queries) // 2
        valid, test = queries[:half], queries[half:]
        grid = [(n, s) for n in (256, 512, 1024) for s in (5, 10, 20)]
        scored = []
        for neurons, steps in grid:
            rows = [evaluate(sub, valid, seed=1, neurons=neurons, steps=steps) for _ in range(2)]
            scored.append((statistics.median(r["spiking_hit1"] for r in rows), neurons, steps))
        scored.sort(reverse=True)
        best_hit, best_neurons, best_steps = scored[0]
        print(f"验证集选参：neurons={best_neurons} steps={best_steps}（验证 hit@1 {best_hit:.3f}）")
        test_rows = [
            evaluate(sub, test, seed=seed, neurons=best_neurons, steps=best_steps)
            for seed in range(1, args.seeds + 1)
        ]
        spiking_hit1 = statistics.median(r["spiking_hit1"] for r in test_rows)
        print(f"留出集：脉冲 hit@1 {spiking_hit1:.3f}（线性 {linear_only(sub, test)['hit1']:.3f}）")
        return 0

    rows = [
        evaluate(sub, queries, seed=seed, neurons=args.neurons, steps=args.steps)
        for seed in range(1, args.seeds + 1)
    ]
    hit1 = statistics.median(r["spiking_hit1"] for r in rows)
    hit5 = statistics.median(r["spiking_hit5"] for r in rows)
    spiking_ms = statistics.median(r["spiking_ms"] for r in rows)
    linear_ms = statistics.median(r["linear_ms"] for r in rows)
    print(
        f"脉冲（{args.neurons} 神经元 × {args.steps} 步，{len(rows)} 种子）：hit@1 {hit1:.3f}  hit@5 {hit5:.3f}"
    )
    print(
        f"延迟：脉冲 {spiking_ms:.1f} ms / 线性 {linear_ms:.1f} ms（多 {spiking_ms - linear_ms:+.1f} ms）"
    )
    print(f"S1（≥ +0.05）：{'过' if hit1 - base['hit1'] >= 0.05 else '不过'}")
    print(f"S3（< +20 ms）：{'过' if spiking_ms - linear_ms < 20 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
