# 换嵌入模型：nomic-embed-text vs 随包 bge-small-zh（2026-11-01）

动机：分类拆分显示失分集中在 `paraphrase`（0.2）与 `redundant_pair`（0.4），而权重扫描
证明**瓶颈不在融合权重**、在语义通道本身。所以换模型、不动权重，做一次干净对比。

跑法：

```bash
python tools/bench_recall.py --model nomic-embed-text --json                      # Ollama
python tools/bench_recall.py --local-embed build/bundle/embedding/bge-small-zh-v1.5 --json
```

## 结果（gold 85 条查询，权重同为 1.2 / 0.45 / 1.5）

| 指标 | nomic-embed-text (768) | **bge-small-zh (512)** | 差 |
| :--- | ---: | ---: | ---: |
| hit@1 | 0.831 | **0.909** | **+7.8pp** |
| hit@5 | 0.909 | **0.974** | **+6.5pp** |
| paraphrase | 0.200 | **0.600** | **+40pp** |
| redundant_pair | 0.400 | **0.600** | +20pp |
| distractor_topic | 0.750 | **0.875** | +12.5pp |
| exact | 0.875 | 0.875 | 0 |
| budget | 0.600 | **0.800** | +20pp |

提升集中在**改写与近重复**两类——正是我们事前指出的窟窿。**方向是对的，幅度超过预期。**

## 但这条不能直接采纳，原因有三

1. **换嵌入 = 换向量空间**。库里已有条目的向量是按 `ollama/nomic-embed-text@768` 存的，
   bge 是 `local/bge-small-zh-v1.5@512`。`spaces` 表支持按空间分行存，所以旧向量不会坏，
   但**要在新空间上重算一遍**（全库重嵌入），这是一次数据迁移，不是改个默认值。
2. **样本量依旧很小**：paraphrase 与 redundant_pair 各 n=5，一条查询就是 20 个百分点。
   0.909 这个数需要更多查询支撑才敢当结论。
3. **权重要重扫**：1.2 / 1.5 是在 nomic 上调出来的，换 bge 之后最优点可能不同；
   现在拿旧权重配新模型，等于让新模型背着别人的参数跑。

## 下一步（按顺序）

1. 把 paraphrase / redundant_pair 各扩到 ≥20 条；
2. 在新语料上**用 bge 重扫一次权重**（同样是验证集选参、留出集报数）；
3. 两项都稳了，再谈"默认嵌入换成 bge"——那一步要连带做**全库重嵌入**与迁移说明。
