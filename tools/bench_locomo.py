"""LoCoMo 召回基线（口径见 docs/prereg-locomo-retrieval-v1.md）。

只量召回，不量答题：把每轮对话当作一条记忆、把每条 QA 的 evidence 当作 ground truth，
看 Axo 的 recall() 能不能把证据轮排进前 k。

数据不入库，运行时从本地副本读；只登记 URL 与 sha256。

用法::

    python tools/bench_locomo.py --data <path>/locomo10.json --arm lexical
    python tools/bench_locomo.py --data <path>/locomo10.json --arm nomic --out <仓外>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from axo.bench import (  # noqa: E402
    CREATED_AT,
    GOLD_WEIGHT,
    GoldMemory,
    GoldQuery,
    GoldSet,
    run,
)
from axo.bench.metrics import K_VALUES, hit_at_k, mrr, recall_at_k  # noqa: E402

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text:latest"


def load_payload(path: Path, expected: str) -> tuple[dict, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise SystemExit(f"locomo10.json 的 sha256 不符：{digest}（预期 {expected}）")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _turns(conversation: dict):
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(value, list):
            continue
        for turn in value:
            yield key, turn


def build_goldset(payload: list[dict], gold_dir: Path) -> tuple[GoldSet, dict]:
    memories: list[GoldMemory] = []
    queries: list[GoldQuery] = []
    excluded = {"empty_evidence": 0, "missing_id": 0}
    per_sample: list[dict] = []
    for sample in payload:
        sample_id = str(sample.get("sample_id") or "")
        known: set[str] = set()
        for session_key, turn in _turns(sample.get("conversation") or {}):
            dia = turn.get("dia_id")
            if not dia:
                continue
            memory_id = f"{sample_id}:{dia}"
            memories.append(
                GoldMemory(
                    memory_id=memory_id,
                    content=f"{turn.get('speaker', '')}: {turn.get('text', '')}",
                    kind="fact",
                    tier="longterm",
                    taint="user",
                    weight=GOLD_WEIGHT,
                    pinned=False,
                    session_id=f"{sample_id}/{session_key}",
                    created_at=CREATED_AT,
                )
            )
            known.add(memory_id)
        kept = 0
        for index, qa in enumerate(sample.get("qa") or []):
            evidence = qa.get("evidence") or []
            if not evidence:
                excluded["empty_evidence"] += 1
                continue
            expect = [f"{sample_id}:{dia}" for dia in evidence]
            if any(item not in known for item in expect):
                excluded["missing_id"] += 1
                continue
            queries.append(
                GoldQuery(
                    query_id=f"{sample_id}:qa{index}",
                    query=str(qa.get("question", "")),
                    category=str(qa.get("category", "")),
                    expect=tuple(dict.fromkeys(expect)),
                )
            )
            kept += 1
        per_sample.append({"sample_id": sample_id, "memories": len(known), "queries": kept})
    goldset = GoldSet(memories=tuple(memories), queries=tuple(queries), gold_dir=gold_dir)
    return goldset, {"excluded": excluded, "per_sample": per_sample}


def make_embedder(arm: str, args):
    if arm == "lexical":
        return None
    if arm == "nomic":
        from axo.llm.ollama import OllamaEmbeddingBackend

        return OllamaEmbeddingBackend(model=args.model, dim=args.dim, host=args.host)
    if arm == "local":
        from axo.llm.local_embed import LocalEmbeddingBackend
        from axo.paths import bundled_embedding_dir

        model_dir = Path(args.local_model_dir) if args.local_model_dir else bundled_embedding_dir()
        if model_dir is None:
            raise SystemExit("找不到随包 bge 模型；用 --local-model-dir 指过去。")
        return LocalEmbeddingBackend(model_dir=model_dir)
    raise SystemExit(f"未知的臂：{arm}")


def summarize_arm(outcomes) -> dict:
    row = {f"hit@{k}": hit_at_k(outcomes, k) for k in K_VALUES}
    row.update({f"recall@{k}": recall_at_k(outcomes, k) for k in K_VALUES})
    row["mrr"] = mrr(outcomes)
    row["answered"] = sum(1 for outcome in outcomes if outcome.expect)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoCoMo 召回基线")
    parser.add_argument("--data", type=Path, required=True, help="locomo10.json 的本地路径")
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument("--arm", default="lexical", choices=("lexical", "nomic", "local"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--local-model-dir", default="")
    parser.add_argument("--limit-samples", type=int, default=0, help="只跑前 N 段（冒烟用）")
    parser.add_argument("--out", type=Path, default=_ROOT / "benchmarks" / "locomo")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    payload, digest = load_payload(args.data, args.expect_sha256)
    if args.limit_samples:
        payload = payload[: args.limit_samples]
    goldset, info = build_goldset(payload, args.data.parent)
    embedder = make_embedder(args.arm, args)
    outcomes = run(goldset, embedder=embedder)
    result = {
        "kind": "axo locomo retrieval baseline",
        "arm": args.arm,
        "data_sha256": digest,
        "samples": len(payload),
        "memories": len(goldset.memories),
        "queries": len(goldset.queries),
        "model": args.model if args.arm == "nomic" else args.arm,
        "space_id": getattr(embedder, "space_id", ""),
        **info,
        "metrics": summarize_arm(outcomes),
        "per_category": _per_category(outcomes),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.arm
    path = args.out / f"{tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    metrics = result["metrics"]
    print(
        f"[{tag}] 记忆 {result['memories']} / 查询 {result['queries']}"
        f"（排除 空证据 {info['excluded']['empty_evidence']}、"
        f"缺 id {info['excluded']['missing_id']}）"
    )
    print(
        f"[{tag}] hit@1 {metrics['hit@1']:.3f} / hit@3 {metrics['hit@3']:.3f} / "
        f"hit@5 {metrics['hit@5']:.3f} / hit@8 {metrics['hit@8']:.3f} | "
        f"证据覆盖@8 {metrics['recall@8']:.3f} | MRR {metrics['mrr']:.3f}"
    )
    print(f"写入 {path}")
    return 0


def _per_category(outcomes) -> dict:
    buckets: dict[str, list] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category, []).append(outcome)
    return {
        category: {
            "answered": sum(1 for outcome in rows if outcome.expect),
            "hit@5": hit_at_k(rows, 5),
            "recall@8": recall_at_k(rows, 8),
        }
        for category, rows in sorted(buckets.items())
    }


if __name__ == "__main__":
    raise SystemExit(main())
