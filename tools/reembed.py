"""把库里所有记忆按指定嵌入空间补算一遍向量（**只写 embeddings 表，不碰 memories**）。

    python tools/reembed.py --db <副本路径> --space local/bge-small-zh-v1.5@512 \
        --model-dir build/bundle/embedding/bge-small-zh-v1.5

安全边界（结构性的，不靠自觉）：

* 只调用 ``MemoryStore.set_embedding``，它按 ``(memory_id, space_id)`` 插入；
  **没有任何一条路径会 update/delete memories 行**；
* 结束后逐字节比较 memories 表的内容哈希与条数，不等就报错；
* 默认拒绝在 ``D:/AxoData`` 上运行，除非显式加 ``--i-mean-the-real-library``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def memories_digest(store) -> tuple[int, str]:
    rows = []
    for item in store.iter_items():
        rows.append(
            (item.memory_id, item.content, item.kind.value, item.tier.value, float(item.weight))
        )
    rows.sort()
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()
    return len(rows), digest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--missing-only", action="store_true", help="只补这个空间里还没有向量的条目"
    )
    parser.add_argument("--i-mean-the-real-library", action="store_true")
    args = parser.parse_args(argv)

    if "AxoData" in str(args.db) and not args.i_mean_the_real_library:
        print(
            "拒绝在真实库上运行。先复制一份，或用 --i-mean-the-real-library 显式确认。",
            file=sys.stderr,
        )
        return 2

    from axo.llm.local_embed import LocalEmbeddingBackend
    from memcore.store import MemoryStore

    store = MemoryStore(args.db)
    before_count, before_digest = memories_digest(store)
    print(f"库：{args.db}（{before_count} 条记忆，digest {before_digest[:16]}…）")

    backend = LocalEmbeddingBackend(model_dir=args.model_dir)
    items = list(store.iter_items())
    if args.missing_only:
        missing = set(store.missing_embedding_ids(args.space))
        items = [item for item in items if item.memory_id in missing]
    done = 0
    for start in range(0, len(items), args.batch):
        chunk = items[start : start + args.batch]
        vectors = backend.embed([item.content for item in chunk])
        for item, vector in zip(chunk, vectors, strict=True):
            store.set_embedding(item.memory_id, args.space, vector)
            done += 1
        print(f"  {done}/{len(items)}", flush=True)

    after_count, after_digest = memories_digest(store)
    embedded = store.count_embeddings(args.space)
    store.close()

    ok = after_count == before_count and after_digest == before_digest
    print(f"记忆条数 {before_count} -> {after_count}；内容哈希{'一致' if ok else '**不一致**'}")
    print(f"{args.space} 空间里的向量：{embedded} 条")
    return 0 if ok and embedded == before_count else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
