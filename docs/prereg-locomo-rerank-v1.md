# 预注册：LoCoMo 召回重排器 v1（2026-11-05）

**状态：判据先落盘，代码后写。**

## 问题

LoCoMo 上 nomic 臂 hit@5 = **0.495**、证据覆盖@8 = **0.439**、MRR = **0.350**
（`benchmarks/locomo/README.md`）。配置层已经到顶（`benchmarks/recall_tuning.md`：
`redundancy_threshold` / `min_score` / 融合权重 / 图通道都试过），剩下的是**排序**。

这一轮只问一件事：**在同一批候选上换一个读出（重排），能不能把证据往前挪？**
不换嵌入、不换候选生成、不改召回配置。

## 口径

- 数据：`locomo10.json`，sha256
  `553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365`；
  5882 条记忆 / 1973 条查询；排除规则与 v1 基线逐条相同。
- 第一段：现有 `recall()`，nomic 嵌入（`nomic-embed-text:latest`），
  `RecallConfig()` 出厂默认，`candidate_limit=200`，**取 top-200 候选**；
  时间窗按基准固定为中立（`CREATED_AT`）。
- 重排只改这 200 条的**顺序**；阈值、MMR、token 预算、`max_items` 均不变。
- 指标：hit@1/3/5/8、证据覆盖@8、MRR；另报 cat 1–4 池化 hit@5。
- 训练/评测划分：**按 conversation（sample）留一**，10 折；每个查询只由没见过
  它所在 conversation 的模型打分。池化各折的留出结果。
- 训练标签只来自训练折的 `evidence`；留出折的标签只用于算指标。

## 两条臂

### A. 特征重排（线性 RankNet，主臂）

候选特征（**先定死，跑之前不再加**）：

| 特征 | 定义 |
| :--- | :--- |
| `fused` | 第一段融合分 |
| `lexical` | 词法通道分 |
| `semantic` | 语义通道分（归一化后） |
| `semantic_raw` | 原始余弦 |
| `bm25` | 查询词与正文的 BM25（IDF 用本查询的 200 条候选估计） |
| `word_overlap` | 小写词集合的 Jaccard |
| `char_ngram` | 字符 3-gram 的 Jaccard（中英文都可用） |
| `number_overlap` | 查询里的数字在正文中出现的比例 |
| `entity_overlap` | 查询里的首字母大写词 / 引号词在正文中出现的比例 |
| `length_ratio` | `min(len(q), len(c)) / max(...)` |
| `rank_reciprocal` | `1 / (1 + 第一段名次)` |

模型：线性打分 `w·x`，pairwise logistic loss（正例 vs 采样负例），L2=1e-3，
Adam，固定种子 0；每折最多 20 个负例/正例对。**不使用任何非线性模型或外部标注。**

### B. 无监督特征重排（副臂，可解释）

固定权重，不训练：

```
score = 0.60*fused + 0.15*bm25_norm + 0.10*number_overlap
      + 0.10*entity_overlap + 0.05*rank_reciprocal
```

`bm25_norm` 是本查询候选内 min-max 归一化。权重先写死，看到结果后不改。

## 判据

| # | 判据 | 门槛 |
| :--- | :--- | :--- |
| **R0** | 前置：候选里有证据 | top-200 的证据覆盖率 ≥ 0.80 |
| **R1** | 主臂把证据往前挪 | 留出池化 hit@5 ≥ **0.520**（+2.5pp）且 coverage@8 ≥ **0.464**（+2.5pp） |
| **R2** | 不是用 MRR 换 hit@5 | 留出池化 MRR ≥ **0.345** |
| **R3** | 不伤简单档 | cat 1–4 池化 hit@5 ≥ **0.540**（不低于基线） |
| **R4** | 副臂也有正向 | 副臂 hit@5 ≥ 基线 + **0.010** |
| **R5** | 成本 | 单查询重排 < **10 ms**（纯 Python/numpy，不含第一段） |

R0 不过是**无效实验**（候选里根本没有证据，重排无从谈起），不是重排失败。
R1/R2/R3 任一不过是负结果：**不接线、不改默认**，把数写进 README。

## 不做的事

- 不换嵌入、不改 `candidate_limit`、不改第一段权重；
- 不在留出折上调权重、不按类别分模型；
- 不把 LoCoMo 的标签、`dia_id` 或 conversation 结构直接做成特征；
- 不因为一次不过就换模型/换特征再试——那要另写预注册。

## 通过之后

1. 把主臂权重冻结，接到 `memcore.recall(rerank=...)` 的钩子上；
2. 产品默认**仍关**，先在真实库副本上量延迟与注入条数；
3. 中文/真实库是否成立另写预注册——LoCoMo 是英文，特征虽语言无关，权重未必。
