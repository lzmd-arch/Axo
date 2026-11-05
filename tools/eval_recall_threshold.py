"""在**标注语料**上复评召回门槛的两条判据（库用真实库的一致快照，只读源库）。

    python tools/eval_recall_threshold.py --questions ~/axo-annotation/questions.jsonl
    python tools/eval_recall_threshold.py --questions … --sweep 0.20,0.25,0.30
    python tools/eval_recall_threshold.py --questions … --corpus gold   # 先复现旧语料上的数

判据（`docs/prereg-recall-threshold-v1.md`）：

1. 陌生话题（`label: none`）的注入条数应 **< 5**；
2. 高相关（`label: support`）里「至少注入一条」的比例应 **≥ 95%**。

`partial` 不算正样本（见 `docs/annotation-rubric.md`）。

**库是 SQLite WAL 模式。** 只 `copy2` 主文件会漏掉 `-wal` 里还没 checkpoint 的写入，
两次跑出来的库状态不同——所以这里用 SQLite 的 backup API 做一个**一致快照**；
`--sweep` 的几档门槛共用同一份快照，差值是同一状态下的差。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
import tempfile
from pathlib import Path


def source_digest(db: Path) -> str:
    """主文件 + WAL 的摘要。SHM 是读路径的运行时索引，不计。"""
    digest = hashlib.sha256()
    for path in (db, db.with_name(db.name + "-wal")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot(db: Path, dest: Path) -> None:
    """SQLite backup API：拿到包含 WAL 的一致快照，不碰源库。"""
    source = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(dest)
    try:
        with target:
            source.backup(target)
    finally:
        target.close()
        source.close()


def measure(rows: list[dict], store, recall, config) -> list[dict]:
    results: list[dict] = []
    for row in rows:
        ledger = recall(
            store,
            query_text=row["question"],
            budget_tokens=8000,
            config=config,
            settle=False,
        )
        results.append({**row, "injected": len(ledger.injected)})
    return results


def summarize(results: list[dict]) -> dict:
    support = [r for r in results if r["label"] == "support"]
    none = [r for r in results if r["label"] == "none"]
    partial = [r for r in results if r["label"] == "partial"]
    hit_rate = (sum(1 for r in support if r["injected"] >= 1) / len(support)) if support else None
    none_counts = [r["injected"] for r in none]
    quiet_share = (
        sum(1 for count in none_counts if count < 5) / len(none_counts) if none_counts else None
    )
    return {
        "results": results,
        "support": support,
        "none": none,
        "partial": partial,
        "hit_rate": hit_rate,
        "quiet_share": quiet_share,
        "none_counts": none_counts,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path.home() / ".axo" / "memory" / "mem.db")
    parser.add_argument("--limit", type=int, default=5, help="每次召回最多注入多少条")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="产品召回门槛（默认 0.25；库默认 0.133，不是产品口径）",
    )
    parser.add_argument(
        "--sweep",
        default="",
        help="同一快照上跑多档，逗号分隔，例如 0.20,0.25,0.30",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    thresholds = (
        [float(part) for part in args.sweep.split(",") if part.strip()]
        if args.sweep.strip()
        else [args.min_score]
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from memcore.recall import RecallConfig, recall  # noqa: PLC0415
    from memcore.store import MemoryStore  # noqa: PLC0415

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unlabelled = [row for row in rows if row.get("label") not in {"support", "none", "partial"}]
    if unlabelled:
        print(f"还有 {len(unlabelled)} 条没有标注（label 为空或非法），先按锚定标完再跑。")
        return 2

    before = source_digest(args.db)
    work = Path(tempfile.mkdtemp(prefix="axo-threshold-eval-"))
    copy = work / "mem.db"
    snapshot(args.db, copy)
    if source_digest(args.db) != before:
        print("做快照期间源库变了（有别的进程在写）；本次结果对应快照，不代表源库的静止状态。")
    print(f"快照：{copy}")

    store = MemoryStore(copy)
    try:
        output: list[dict] = []
        for threshold in thresholds:
            config = RecallConfig(max_items=args.limit, min_score=threshold)
            summary = summarize(measure(rows, store, recall, config))
            output.append({"min_score": threshold, **summary})
            support, none, partial = summary["support"], summary["none"], summary["partial"]
            print(
                f"语料：{len(rows)} 条（support {len(support)} / none {len(none)} / "
                f"partial {len(partial)}）"
            )
            print(f"门槛：min_score={threshold:g}，max_items={args.limit}")
            print(
                f"判据 2（support 至少注入一条）：{summary['hit_rate']:.1%}（门槛 ≥95%）"
                if summary["hit_rate"] is not None
                else "判据 2：无 support 样本"
            )
            if summary["none_counts"]:
                print(
                    f"判据 1（none 注入 <5 的比例）：{summary['quiet_share']:.1%}；"
                    f"中位 {statistics.median(summary['none_counts'])} 条，"
                    f"最大 {max(summary['none_counts'])} 条"
                )
            if partial:
                print(
                    f"partial（不算正样本）：{len(partial)} 条，"
                    f"中位注入 {statistics.median([r['injected'] for r in partial])} 条"
                )
            print()
    finally:
        store.close()

    after = source_digest(args.db)
    print(f"源库未变（db+wal）：{after == before}")
    print(f"工作目录：{work}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "thresholds": output,
                    "source_unchanged": after == before,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if after == before else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
