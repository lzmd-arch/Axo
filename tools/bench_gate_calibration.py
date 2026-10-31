"""真实库**副本**上的门控标定与验收。

两条硬规矩，脚本自己保证：

1. **源库只读**：只在开头复制一次，之后所有读写都发生在临时目录；结束前再取一次源库
   sha256 并与开头比对，不等就报错退出。
2. **嵌入用应用自己的那个**（随包 `bge-small-zh-v1.5`），不是词法替身——标定要回答的
   正是"0.90 在真嵌入下是不是等于重复"，换个替身就不算数。

    python tools/bench_gate_calibration.py                 # 默认读 D:/AxoData/memory/mem.db
    python tools/bench_gate_calibration.py --source <db> --limit 200
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

DEFAULT_SOURCE = Path(r"D:/AxoData/memory/mem.db")
MODEL_DIR = _ROOT / "build" / "bundle" / "embedding" / "bge-small-zh-v1.5"

NOVEL = [
    "今天下午三点在会议室讨论下季度的排期安排",
    "冰箱里的牛奶还剩半盒，明天记得买新的",
    "把论文第三节的图表重新导出一次，字体统一成 Times",
    "周末去爬山，路线从北门上山，全程大约四小时",
    "服务器上的日志轮转配置需要改成按天切分",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b) -> float:
    import numpy as np  # noqa: PLC0415

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--limit", type=int, default=80, help="参与标定的条目抽样上限")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"找不到源库：{args.source}", file=sys.stderr)
        return 2

    before = sha256(args.source)
    print(f"源库 sha256（拷贝前）：{before[:16]}…")

    work = Path(tempfile.mkdtemp(prefix="axo-gate-calib-"))
    copy = work / "mem.db"
    shutil.copy2(args.source, copy)
    after_copy = sha256(args.source)
    if after_copy != before:
        print("拷贝之后源库哈希变了——立刻停下。", file=sys.stderr)
        return 3
    print(f"副本：{copy}（源库未变）")

    from axo.llm.local_embed import LocalEmbeddingBackend  # noqa: PLC0415
    from axo.propose_consumer import GateConfig, ProposeConsumer  # noqa: PLC0415
    from memcore.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(copy)
    entries = list(store.iter_items())
    total = len(entries)
    sample = entries[: args.limit]
    print(f"库内条目：{total}；抽样：{len(sample)}")

    backend = LocalEmbeddingBackend(model_dir=MODEL_DIR)
    entry_vectors = backend.embed([item.content for item in sample])

    verbatim = [item.content for item in sample]
    edited = [item.content + " 补充一句。" for item in sample]
    probes = verbatim + edited + NOVEL
    probe_vectors = backend.embed(probes)

    def max_cos(vector, *, skip_self: int | None = None) -> float:
        best = 0.0
        for index, other in enumerate(entry_vectors):
            if skip_self is not None and index == skip_self:
                continue
            best = max(best, cosine(vector, other))
        return best

    # 四组各答一个问题，别混：
    #   1) 把库里已有的句子原样再提一次 —— 重复判定能不能抓住（期望 ≈ 1.0）
    #   2) 同一句轻微改写 —— 阈值对"几乎重复"有多敏感
    #   3) 库内两条**不同**条目的最大相似度 —— 阈值上方还剩多少余量（假阳性的天花板）
    #   4) 完全无关的新句子 —— 基线有多高
    groups = {
        "原句再提（含自身）": [max_cos(probe_vectors[i]) for i in range(len(verbatim))],
        "轻微改写（不含自身）": [
            max_cos(probe_vectors[len(verbatim) + i], skip_self=i) for i in range(len(edited))
        ],
        "库内异条目互比": [max_cos(probe_vectors[i], skip_self=i) for i in range(len(verbatim))],
        "无关新文本": [
            max_cos(probe_vectors[len(verbatim) + len(edited) + i]) for i in range(len(NOVEL))
        ],
    }

    threshold = GateConfig().duplicate_similarity
    print("\n## 相似度分布（与应用同一套 BGE 嵌入）\n")
    print("| 组 | 条数 | min | 中位数 | max | ≥ 0.90 的比例 |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    report: dict[str, dict] = {}
    for name, values in groups.items():
        over = sum(1 for v in values if v >= threshold) / len(values)
        report[name] = {**stats(values), "over_threshold": round(over, 3), "count": len(values)}
        print(
            f"| {name} | {len(values)} | {min(values):.4f} | {statistics.median(values):.4f} "
            f"| {max(values):.4f} | {over:.0%} |"
        )

    # 副本上的端到端：提议 → 门控 → 落库 → 撤销
    consumer = ProposeConsumer(store, work / "queue.sqlite3", audit_path=work / "audit.jsonl")
    entry_count = store.count()
    proposal = consumer.ingest(
        "服务器上的日志轮转配置需要改成按天切分", proposal_id="calib-1", source="user"
    )
    consumer.decide(proposal["seq"])
    status = consumer.queue.get(proposal["seq"])["status"]
    after_propose = store.count()
    consumer.revert(proposal["seq"], "calibration rollback") if status == "accepted" else None
    final = store.count()
    store.close()

    print("\n## 副本上的端到端\n")
    print(f"- 提议判定：`{status}`")
    print(f"- 库条目：{entry_count} → {after_propose} → {final}（撤销后）")

    after_run = sha256(args.source)
    print(f"\n源库 sha256（结束）：{after_run[:16]}…  一致：{after_run == before}")
    print(f"工作目录：{work}")

    out = work / "calibration.json"
    out.write_text(
        json.dumps(
            {"source_unchanged": after_run == before, "groups": report, "proposal_status": status},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if after_run == before else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
