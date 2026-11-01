"""查询改写能不能抬高硬改写的命中（预注册 docs/prereg-query-rewrite-v1.md）。

    python tools/bench_query_rewrite.py --corpus benchmarks/paraphrase_v1_hard --limit 20

判据：Q1 硬改写 hit@1 ≥0.60（基线 0.40）；Q2 单次改写延迟 <300 ms；Q3 gold 不倒退。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from memcore.recall import RecallConfig, recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore

OLLAMA = "http://127.0.0.1:11434/api/generate"
INSTRUCTION = (
    "用另一种说法描述下面这个场景：不要沿用原句里的名词与数字写法（把“两成”说成“百分之二十”、"
    "不要回答问题，不要解释，只输出改写后的问句。\n\n提问："
)


def rewrite(model: str, query: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": INSTRUCTION + query,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 96},
        "think": False,
    }
    # 本机 Ollama：绕开系统代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    start = time.perf_counter()
    request = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with opener(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body.get("response", "")
    text = __import__("re").sub(r"<think>.*?</think>", "", text, flags=__import__("re").S)
    text = __import__("re").sub(r"^\s*<think>.*", "", text, flags=__import__("re").S)
    return text.strip(), (time.perf_counter() - start) * 1000


def ranked(store, text: str, k: int = 5) -> list[str]:
    ledger = recall(
        store,
        query_text=text,
        budget_tokens=8000,
        config=RecallConfig(max_items=k),
        settle=False,
    )
    return [m.memory_id for m in ledger.injected]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/paraphrase_v1_hard"))
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=2, help="打印前几条改写结果")
    args = parser.parse_args(argv)

    memories = [
        json.loads(line)
        for line in (args.corpus / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [
        json.loads(line) for line in (args.corpus / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    if args.limit:
        queries = queries[: args.limit]

    from memcore.types import MemoryItem, MemoryKind, Tier

    real = []
    for m in memories:
        real.append(
            MemoryItem(
                memory_id=m["id"],
                kind=MemoryKind.FACT,
                tier=Tier.WORKING,
                content=m["content"],
                lex_sdr=encode_lexical(m["content"]),
            )
        )
    store2 = MemoryStore(":memory:")
    for item in real:
        store2.add(item)

    base1 = rewrite1 = 0
    latencies: list[float] = []
    samples: list[tuple[str, str]] = []
    for index, q in enumerate(queries, start=1):
        expect = set(q.get("expect") or [])
        original = q["query"]
        base_top = ranked(store2, original)
        base1 += bool(base_top and base_top[0] in expect)

        rewritten, ms = rewrite(args.model, original)
        latencies.append(ms)
        if len(samples) < args.show:
            samples.append((original, rewritten))

        merged = []
        for mid in ranked(store2, rewritten) + base_top:
            if mid not in merged:
                merged.append(mid)
        rewrite1 += bool(merged and merged[0] in expect)
        if index % 5 == 0:
            print(f"  {index}/{len(queries)}", flush=True)

    n = len(queries)
    print(f"\n语料：{args.corpus.name}  n={n}")
    print(f"基线 hit@1：{base1 / n:.3f}")
    print(f"改写+并集 hit@1：{rewrite1 / n:.3f}")
    print(f"改写延迟：中位 {statistics.median(latencies):.0f} ms（最大 {max(latencies):.0f} ms）")
    print(f"Q1（≥0.60）：{'过' if rewrite1 / n >= 0.60 else '不过'}")
    print(f"Q2（<300 ms）：{'过' if statistics.median(latencies) < 300 else '不过'}")
    for original, rewritten in samples:
        print(f"\n  原：{original}\n  改：{rewritten[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
