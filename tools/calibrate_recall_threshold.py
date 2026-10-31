"""标定召回层的 ``min_score``（"弱记忆不再注入"那条阈值）。

要回答的只有一个问题：**把融合分的下限从 0.10 抬到多少，既不丢掉该记住的，又能把
陌生话题那一坨弱相关挡在门外。** 所以这个脚本只量分布，不下结论——阈值由
``docs/prereg-recall-threshold-v1.md`` 的判据定。

分三组量：

* **正样本（整句）**：库内每条记忆的正文当查询。它代表"这句话我确实有"。
* **正样本（半句）**：正文前 12 字。真实的"只记得一半"。
* **陌生话题**：与库无关的 20 条。它们的注入条目按定义**全是不该注入的**。
* **真实提问**：会话里用户说过的话（如果给了 ``--sessions``）。**这一组没有标签**，
  只报分布，不看对错——它是用来看"阈值会拦掉多少日常提问"的。

规矩同 ``bench_gate_calibration.py``：源库只复制不改、前后比对 sha256、``settle=False``。
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
DEFAULT_SESSIONS = Path(r"D:/AxoData/sessions")

MODEL_CANDIDATES = (
    _ROOT / "build" / "bundle" / "embedding" / "bge-small-zh-v1.5",
    _ROOT / "dist" / "axo" / "_internal" / "embedding" / "bge-small-zh-v1.5",
)

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
    raise SystemExit("找不到随包嵌入模型：先跑 tools/fetch_bundle_assets.py。")


def quantiles(
    values: list[float], points: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    result: dict[str, float] = {"min": round(ordered[0], 4), "max": round(ordered[-1], 4)}
    for point in points:
        index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
        result[f"p{int(point * 100)}"] = round(ordered[index], 4)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--positives", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"找不到源库：{args.source}", file=sys.stderr)
        return 2

    before = sha256(args.source)
    work = Path(tempfile.mkdtemp(prefix="axo-recall-calib-"))
    copy = work / "mem.db"
    shutil.copy2(args.source, copy)
    if sha256(args.source) != before:
        print("拷贝之后源库哈希变了——立刻停下。", file=sys.stderr)
        return 3

    import numpy as np  # noqa: PLC0415

    from axo.llm.local_embed import LocalEmbeddingBackend  # noqa: PLC0415
    from memcore.recall import RecallConfig, recall  # noqa: PLC0415
    from memcore.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(copy)
    items = sorted(store.iter_items(), key=lambda item: item.memory_id)
    rng = np.random.default_rng(args.seed)
    selected = [items[index] for index in rng.permutation(len(items))[: args.positives]]

    backend = LocalEmbeddingBackend(model_dir=resolve_model_dir())
    config = RecallConfig()  # **默认值**：这里量的就是用户现在拿到的那个召回

    def scores_for(query: str) -> list[float]:
        vector = backend.embed([query])[0]
        ledger = recall(
            store,
            query_text=query,
            budget_tokens=12000,
            query_vector=vector,
            space_id=backend.space_id,
            config=config,
            settle=False,
        )
        return sorted((float(item.scores.fused) for item in ledger.injected), reverse=True)

    groups: dict[str, list[list[float]]] = {
        "正样本（整句）": [scores_for(item.content) for item in selected],
        "正样本（半句）": [
            scores_for(item.content[:12]) for item in selected if len(item.content) >= 6
        ],
        "陌生话题": [scores_for(query) for query in NOVEL],
    }

    session_queries: list[str] = []
    if args.sessions.is_dir():
        from axo.session.store import SessionStore  # noqa: PLC0415

        for session in SessionStore(args.sessions).list_sessions(limit=40):
            loaded = SessionStore(args.sessions).load(session.session_id)
            for message in loaded.messages:
                if message.role == "user":
                    text = (message.content or "").strip()
                    if 4 <= len(text) <= 120:
                        session_queries.append(text)
    # 去重、限量：真实提问没有标签，只用来读分布。
    session_queries = list(dict.fromkeys(session_queries))[:200]
    if session_queries:
        groups["真实提问（无标签）"] = [scores_for(query) for query in session_queries]

    print(f"库内条目：{len(items)}；正样本：{len(selected)}；真实提问：{len(session_queries)}")
    report: dict[str, dict] = {}
    for name, per_query in groups.items():
        tops = [row[0] if row else 0.0 for row in per_query]
        counts = [len(row) for row in per_query]
        print(f"\n## {name}\n")
        print(f"- 条数：{len(per_query)}")
        print(
            f"- 注入条数：中位 {np.median(counts):.0f}、p95 {np.quantile(counts, 0.95):.0f}、最大 {max(counts)}"
        )
        print(f"- top 融合分分位：{quantiles(tops)}")
        all_items = [score for row in per_query for score in row]
        if all_items:
            print(f"- 逐条融合分分位：{quantiles(all_items)}")
        report[name] = {
            "queries": len(per_query),
            "injected_median": float(np.median(counts)),
            "injected_p95": float(np.quantile(counts, 0.95)),
            "top": quantiles(tops),
            "items": quantiles(all_items) if all_items else {},
        }

    # 直接回答"抬到 T 之后还剩多少"。
    print("\n## 阈值扫描：每组的注入条数中位数 / 保留比例\n")
    print("| T | " + " | ".join(groups) + " |")
    print("| ---: | " + " | ".join("---:" for _ in groups) + " |")
    for threshold in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        cells = []
        for per_query in groups.values():
            kept = [sum(1 for score in row if score >= threshold) for row in per_query]
            share = sum(1 for value in kept if value > 0) / max(1, len(kept))
            cells.append(f"{np.median(kept):.0f} / {share:.0%}")
        print(f"| {threshold} | " + " | ".join(cells) + " |")

    after = sha256(args.source)
    print(f"\n源库 sha256 一致：{after == before}")
    out = args.out or (work / "recall_threshold_calibration.json")
    out.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "unchanged": after == before,
                "seed": args.seed,
                "groups": report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"结果：{out}")
    store.close()
    return 0 if after == before else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
