"""元认知门控的离线标定与评测（真实库**副本**上跑）。

它回答三个问题，按重要性排：

1. **门控能不能把"陌生话题"挡住**（OOD 拦截率）；
2. **它会不会把"有记忆支撑"的那批也挡掉**（保留率）；
3. **它比平凡规则强多少**——同一批数据上，"召回有东西就注入"与"只看最强分"各自
   的 AUROC。**不比它们强，这一层就是装饰**，不该接线。

规矩照 ``bench_gate_calibration.py`` 抄：源库只在开头复制一次、结束前再取一次
sha256 比对；嵌入用应用自己那个 bge（不是词法替身）；召回一律 ``settle=False``
（看一眼，不是想起来）。

用法::

    python tools/eval_metacog.py                      # 默认读 D:/AxoData/memory/mem.db
    python tools/eval_metacog.py --positives 120 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

DEFAULT_SOURCE = Path(r"D:/AxoData/memory/mem.db")

#: 嵌入模型（随包 bge）。**三个候选按序找**：`build/bundle` 是构建资产目录，它可能
#: 被清理；`dist/axo/_internal` 是已构建产物里的那一份；``AXO_EMBEDDING_DIR`` 优先。
MODEL_CANDIDATES = (
    _ROOT / "build" / "bundle" / "embedding" / "bge-small-zh-v1.5",
    _ROOT / "dist" / "axo" / "_internal" / "embedding" / "bge-small-zh-v1.5",
)

#: 陌生话题：任务书里那 20 条 OOD 的主题 + 几条日常的。**全部与记忆库无关**。
NOVEL = (
    "量子纠缠的最新实验进展是什么",
    "如何养热带鱼",
    "推荐几个成都火锅店",
    "2026 年诺贝尔物理学奖给了谁",
    "怎么用 Rust 写一个异步 web 服务",
    "海王星的大气层主要成分是什么",
    "法式可颂的开酥手法",
    "如何评价最近的国际油价走势",
    "git rebase 和 merge 有什么区别",
    "教我做一道麻婆豆腐",
    "股票的分红除权是什么意思",
    "如何练习自由泳的打腿",
    "推荐几本科幻小说",
    "空调制冷效果差怎么排查",
    "解释一下傅里叶变换",
    "马丘比丘在哪个国家",
    "怎样挑选一台入门相机",
    "为什么天空是蓝色的",
    "怎么给绿萝换盆",
    "三体里黑暗森林法则讲的是什么",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_model_dir() -> Path:
    for candidate in MODEL_CANDIDATES:
        if (candidate / "model_optimized.onnx").is_file():
            return candidate
    raise SystemExit(
        "找不到随包嵌入模型。先跑 tools/fetch_bundle_assets.py，或用 AXO_EMBEDDING_DIR 指过去。"
    )


def auroc(positive: list[float], negative: list[float]) -> float:
    """正样本得分高于负样本的概率（Mann–Whitney）。"""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    wins = 0.0
    for value in positive:
        for other in negative:
            if value > other:
                wins += 1.0
            elif value == other:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--positives", type=int, default=100, help="参与评测的记忆条数上限")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"找不到源库：{args.source}", file=sys.stderr)
        return 2

    before = sha256(args.source)
    work = Path(tempfile.mkdtemp(prefix="axo-metacog-"))
    copy = work / "mem.db"
    shutil.copy2(args.source, copy)
    if sha256(args.source) != before:
        print("拷贝之后源库哈希变了——立刻停下。", file=sys.stderr)
        return 3

    import numpy as np  # noqa: PLC0415

    from axo.llm.local_embed import LocalEmbeddingBackend  # noqa: PLC0415
    from axo.metacog import KnnDensityGate, ledger_features  # noqa: PLC0415
    from memcore.recall import RecallConfig, recall  # noqa: PLC0415
    from memcore.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(copy)
    items = list(store.iter_items())
    items.sort(key=lambda item: item.memory_id)
    if not items:
        print("库是空的。", file=sys.stderr)
        return 4

    # 按 id 的哈希分校准/评测两半：**按条目分而不是随机分**，同一条记忆不会同时
    # 出现在两边——那是这类评测最经典的一种泄漏。
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    selected = [items[index] for index in order[: min(args.positives, len(items))]]
    split = [
        int(hashlib.sha1(item.memory_id.encode()).hexdigest(), 16) % 2 == 0 for item in selected
    ]

    backend = LocalEmbeddingBackend(model_dir=resolve_model_dir())
    recall_config = RecallConfig()
    budget = 12000

    def features_for(query: str) -> dict[str, float]:
        vector = backend.embed([query])[0]
        ledger = recall(
            store,
            query_text=query,
            budget_tokens=budget,
            query_vector=vector,
            space_id=backend.space_id,
            config=recall_config,
            settle=False,
        )
        return ledger_features(ledger)

    print(f"库内条目：{len(items)}；参与评测：{len(selected)}（按 id 哈希分两半）")
    positive_rows = [features_for(item.content) for item in selected]
    novel_rows = [features_for(query) for query in NOVEL]

    # **难一点的正面样本**：只给记忆的**前半截**（前 12 字）。逐字整句提问在真实使用
    # 里几乎不会出现，而"半句 + 记忆里有完整那条"才是日常——也正是单特征阈值最可能
    # 露馅的地方。前 12 字短于 6 字的条目跳过（那已经不是提问，是一个词）。
    hard_queries = [item.content[:12] for item in selected if len(item.content) >= 6]
    hard_rows = [features_for(query) for query in hard_queries]

    calibration = [row for row, flag in zip(positive_rows, split, strict=True) if flag]
    held_out = [row for row, flag in zip(positive_rows, split, strict=True) if not flag]
    if len(calibration) < 10 or len(held_out) < 10:
        print("校准/评测任一半的点太少（<10），调大 --positives。", file=sys.stderr)
        return 5

    gate = KnnDensityGate().fit(calibration)
    cal_conf = gate.confidence_batch(calibration, exclude_self=True)
    held_conf = gate.confidence_batch(held_out)
    novel_conf = gate.confidence_batch(novel_rows)
    gate_auroc = auroc(list(held_conf), list(novel_conf))

    # 两个平凡基线：单特征（top_score）与二值（召回有没有东西）。
    cal_top = np.asarray([row["top_score"] for row in calibration])
    held_top = np.asarray([row["top_score"] for row in held_out])
    novel_top = np.asarray([row["top_score"] for row in novel_rows])
    top_auroc = auroc(held_top, novel_top)
    binary_hits = [1.0 if row["density"] > 0 else 0.0 for row in held_out]
    binary_miss = [1.0 if row["density"] > 0 else 0.0 for row in novel_rows]
    binary_auroc = auroc(binary_hits, binary_miss)

    # **保留率是旋钮，不是指标。** 密度门的阈值由"愿意保留多少"定（v11 的
    # ``COVERAGE=0.80`` 就是这个意思），所以真正要报的是：**在同一个保留目标下，
    # 门控比"只看最强分"多挡住多少陌生话题**。两边的阈值都只在 calibration 上取。
    targets = (0.80, 0.90, 0.95)
    rows_table: list[dict[str, float]] = []
    for target in targets:
        gate_cut = float(np.quantile(cal_conf, 1.0 - target))
        top_cut = float(np.quantile(cal_top, 1.0 - target))
        rows_table.append(
            {
                "target": target,
                "gate_retained": float(np.mean(held_conf >= gate_cut)),
                "gate_ood_abstain": float(np.mean(novel_conf < gate_cut)),
                "top_retained": float(np.mean(held_top >= top_cut)),
                "top_ood_abstain": float(np.mean(novel_top < top_cut)),
            }
        )

    empty_ood = sum(1 for row in novel_rows if row["density"] == 0)
    empty_pos = sum(1 for row in held_out if row["density"] == 0)

    print()
    print("## 结果（同一批数据、同一套嵌入）")
    print()
    print("| 指标 | 值 |")
    print("| :--- | ---: |")
    print(f"| 门控 AUROC | {gate_auroc:.3f} |")
    print(f"| 基线①：只看 top_score 的 AUROC | {top_auroc:.3f} |")
    print(f"| 基线②：召回非空就注入的 AUROC | {binary_auroc:.3f} |")
    print(f"| OOD 里召回一条都没命中的比例 | {empty_ood / len(novel_rows):.1%} |")
    print(f"| 正样本里召回一条都没命中的比例 | {empty_pos / len(held_out):.1%} |")

    print()
    print("## 同一保留目标下：门控 vs 只看最强分")
    print()
    print("| 保留目标 | 门控实际保留 | 门控 OOD 拦截 | top_score 实际保留 | top_score OOD 拦截 |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows_table:
        print(
            f"| {row['target']:.0%} | {row['gate_retained']:.1%} | {row['gate_ood_abstain']:.1%} "
            f"| {row['top_retained']:.1%} | {row['top_ood_abstain']:.1%} |"
        )

    print()
    print("## 难一点的正面样本（记忆前半截）")
    print()
    print(
        f"条数：{len(hard_rows)}；召回一条都没命中的比例："
        f"{sum(1 for row in hard_rows if row['density'] == 0) / max(1, len(hard_rows)):.1%}"
    )
    print()
    print("| 保留目标 | 门控实际保留 | 门控 OOD 拦截 | top_score 实际保留 | top_score OOD 拦截 |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    hard_conf = gate.confidence_batch(hard_rows)
    hard_top = np.asarray([row["top_score"] for row in hard_rows])
    for target in targets:
        gate_cut = float(np.quantile(cal_conf, 1.0 - target))
        top_cut = float(np.quantile(cal_top, 1.0 - target))
        print(
            f"| {target:.0%} | {np.mean(hard_conf >= gate_cut):.1%} "
            f"| {np.mean(novel_conf < gate_cut):.1%} "
            f"| {np.mean(hard_top >= top_cut):.1%} "
            f"| {np.mean(novel_top < top_cut):.1%} |"
        )
    print()
    print(f"门控 AUROC（难正样本 vs OOD）：{auroc(list(hard_conf), list(novel_conf)):.3f}")
    print(f"top_score AUROC（难正样本 vs OOD）：{auroc(list(hard_top), list(novel_top)):.3f}")

    print()
    print("## 特征分布（中位数）")
    print()
    print("| 组 | top_score | mean_topk | density |")
    print("| :--- | ---: | ---: | ---: |")
    for name, rows in (("有记忆支撑", held_out), ("陌生话题", novel_rows)):
        med = {
            key: float(np.median([row[key] for row in rows]))
            for key in ("top_score", "mean_topk", "density")
        }
        print(
            f"| {name} | {med['top_score']:.3f} | {med['mean_topk']:.3f} | {med['density']:.4f} |"
        )

    after = sha256(args.source)
    print(f"\n源库 sha256 一致：{after == before}")
    payload = {
        "source": str(args.source),
        "source_unchanged": after == before,
        "seed": args.seed,
        "items": len(items),
        "evaluated": len(selected),
        "calibration": len(calibration),
        "held_out": len(held_out),
        "ood": len(novel_rows),
        "by_target": rows_table,
        "gate_auroc": round(gate_auroc, 4),
        "top_score_auroc": round(top_auroc, 4),
        "binary_auroc": round(binary_auroc, 4),
        "model_dir": str(resolve_model_dir()),
    }
    out = args.out or (work / "metacog_eval.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果：{out}")
    store.close()
    return 0 if after == before else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
