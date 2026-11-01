"""在**标注语料**上复评召回门槛的两条判据（库用真实库的副本，只读源库）。

    python tools/eval_recall_threshold.py --questions D:/Projects/annotation_2026-11/questions.jsonl
    python tools/eval_recall_threshold.py --questions … --corpus gold   # 先复现旧语料上的数

判据（`docs/prereg-recall-threshold-v1.md`）：

1. 陌生话题（`label: none`）的注入条数应 **< 5**；
2. 高相关（`label: support`）里「至少注入一条」的比例应 **≥ 95%**。

`partial` 不算正样本（见 `docs/annotation-rubric.md`）。

**只动副本**：源库复制一次，运行前后各取 sha256 并比对；不等就报错退出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("D:/AxoData/memory/mem.db"))
    parser.add_argument("--limit", type=int, default=5, help="每次召回最多注入多少条")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

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

    before = sha256(args.db)
    work = Path(tempfile.mkdtemp(prefix="axo-threshold-eval-"))
    copy = work / "mem.db"
    shutil.copy2(args.db, copy)
    if sha256(args.db) != before:
        print("拷贝之后源库哈希变了——立刻停下。")
        return 3

    store = MemoryStore(copy)
    results: list[dict] = []
    for row in rows:
        ledger = recall(
            store,
            query_text=row["question"],
            budget_tokens=8000,
            config=RecallConfig(max_items=args.limit),
            settle=False,
        )
        results.append({**row, "injected": len(ledger.injected)})
    store.close()

    support = [r for r in results if r["label"] == "support"]
    none = [r for r in results if r["label"] == "none"]
    partial = [r for r in results if r["label"] == "partial"]

    hit_rate = (sum(1 for r in support if r["injected"] >= 1) / len(support)) if support else None
    none_counts = [r["injected"] for r in none]
    quiet_share = (
        sum(1 for count in none_counts if count < 5) / len(none_counts) if none_counts else None
    )

    print(
        f"语料：{len(rows)} 条（support {len(support)} / none {len(none)} / partial {len(partial)}）"
    )
    print(
        f"判据 2（support 至少注入一条）：{hit_rate:.1%}（门槛 ≥95%）"
        if hit_rate is not None
        else "判据 2：无 support 样本"
    )
    if none_counts:
        print(
            f"判据 1（none 注入 <5 的比例）：{quiet_share:.1%}；"
            f"中位 {statistics.median(none_counts)} 条，最大 {max(none_counts)} 条"
        )
    if partial:
        print(
            f"partial（不算正样本）：{len(partial)} 条，中位注入 {statistics.median([r['injected'] for r in partial])} 条"
        )

    after = sha256(args.db)
    print(f"源库未变：{after == before}")
    print(f"工作目录：{work}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "rows": results,
                    "hit_rate": hit_rate,
                    "quiet_share": quiet_share,
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
