# 全库重嵌入：把记忆补算到 bge 空间（2026-11-01）

背景：`benchmarks/recall_embedding_swap.md` 四档语料一致显示 bge-small-zh 优于 nomic，
且权重两向扫描打平（不用调）。剩下的唯一硬活是**把已有记忆补算到新空间**。

## 安全边界（结构性保证，不靠自觉）

| 保证 | 怎么做到 |
| :--- | :--- |
| **不改记忆内容** | 工具只调 `MemoryStore.set_embedding`，它按 `(memory_id, space_id)` 插入；没有任何 update/delete memories 的路径 |
| **不删记忆** | 同上；且结束前比对 memories 的条数与内容哈希（含 id/content/kind/tier/weight） |
| **旧向量保留** | 新向量写在**另一个 space_id** 下，`spaces` 表分行存；旧空间原样保留，随时可切回 |
| **可重入** | 按 `missing_embedding_ids` 可以只补缺的；重复跑不会重复插 |
| **真实库要显式确认** | 默认拒绝在含 `AxoData` 的路径上跑，必须加 `--i-mean-the-real-library` |

## 副本上的实跑（2026-11-01）

```
python tools/reembed.py --db ~/axo-reembed/copy.db \
    --space local/bge-small-zh-v1.5@512 \
    --model-dir build/bundle/embedding/bge-small-zh-v1.5
→ 217 条记忆，内容哈希一致，217 条向量写入新空间
```

备份：`~/axo-reembed/backup-mem.db`（原库 3,063,808 字节）。

## 在真库上跑之前

1. **先关掉 Axo**——库是 WAL 模式，应用在跑时既有锁竞争，备份也可能漏掉 WAL 里的最新写入；
2. 关掉之后重新备份一次（应用退出时会把 WAL 合并进主库）；
3. 再跑同一条命令，加 `--i-mean-the-real-library`。

## 跑完之后不要做的事

**不要立刻把默认嵌入切成 bge。** 先确认：
- `bench_recall --local-embed` 在真实库上仍然不差于现在（现在还只在合成语料上验过）；
- 应用的嵌入探针能认出新空间（`spaces` 里有 `local/bge-small-zh-v1.5@512`）。
