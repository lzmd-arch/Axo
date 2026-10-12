"""四项指标：hit@K / MRR / 误注入率 / AUC。**纯函数，不碰库也不碰盘。**

口径写死在这里，是因为"指标"这种东西最容易被悄悄重新定义：分母换一下、把
``no_answer`` 算进 hit 的分母、或者把"没命中"记成 0 而不是排除——每一种都会让
数字变好看，而没有任何一处会报错。所以每条指标上面那段 docstring 不是在解释，
是在**定义**；改它等于改量具。

记号：查询 ``q`` 的期望集 ``E_q``、可选集 ``O_q``、注入序列 ``inj_q``（顺序 =
``recall`` 返回顺序 = MMR 选择顺序 = 模型在提示里看到的先后），``topK_q = inj_q[:K]``。

**为什么 AUC 与阈值无关、因而是上界。** ``ChannelScores.fused`` 只由通道得分与
通道权重决定（:func:`memcore.recall._fuse`），**与 ``min_score``、``max_items``、
``mmr_lambda``、``budget`` 全都无关**。所以 AUC 描述的是"这个打分函数能不能把
对的排在错的前面"，在任何部署配置下都成立——它因此是**上界**：当前配置实际达到
的质量不可能超过它。这条性质由 ``tests/test_bench_metrics.py`` 里那条"同一条候选
在 ``min_score`` 0.10 与 0.0 下 fused 相同"的断言钉住。将来有人把阈值塞进
:func:`memcore.recall._fuse`，AUC 会静默变味，那条断言会红。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "K_VALUES",
    "QueryOutcome",
    "auc",
    "false_injection_rate",
    "hard_negative_auc",
    "hit_at_k",
    "mrr",
    "mrr_answered",
    "no_answer_injections",
    "recall_at_k",
    "summarize",
]

#: 报告哪几个 K。1 是"排第一了吗"，8 是 ``RecallConfig.max_items`` 的默认值
#: ——超过它的 K 永远等于 ``hit@8``，报出来只会让人以为多测了一个点。
K_VALUES: tuple[int, ...] = (1, 3, 5, 8)


@dataclass(frozen=True)
class QueryOutcome:
    """一条查询跑完之后的全部证据。指标函数只吃它。

    把"跑"与"算"分开，是为了让指标的边界情形能被单元测试**手搓**——测
    "``K > len(inj)`` 怎么办"不该需要建一个库。

    Attributes:
        injected: 注入序列，**有序**。
        scores: ``{memory_id: fused}``，来自一次放开了全部阈值的"普查调用"
            （见 :func:`axo.bench.harness.census_scores`）。只有 AUC
            用它。缺失的 id 按 0.0 处理——那表示"它在这一轮连通道都没进"。
    """

    query_id: str
    category: str
    injected: tuple[str, ...]
    expect: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    scores: Mapping[str, float] | None = None
    tokens_used: int = 0
    tokens_budget: int = 0

    @property
    def answered(self) -> bool:
        """这条查询**有没有正确答案**。决定它进不进 hit/MRR 的分母。"""
        return bool(self.expect)

    @property
    def relevant(self) -> set[str]:
        """相关的 id：期望的 ∪ 可选的。"""
        return set(self.expect) | set(self.optional)


def _answered(outcomes: Sequence[QueryOutcome]) -> list[QueryOutcome]:
    return [outcome for outcome in outcomes if outcome.answered]


def hit_at_k(outcomes: Sequence[QueryOutcome], k: int) -> float:
    """``hit@K`` —— 前 K 条里**至少有一条**期望记忆的查询占比。macro。

    分母是 ``Q_ans = {q : E_q ≠ ∅}``。``no_answer`` 查询**不进分母**：它的
    ``E_q`` 是空的，"命中"这个词对它没有定义，硬算成 0 会让"库里确实没有的
    东西我们没有编出来"变成一个负分项。

    ``K`` 大于注入条数时按实际长度算，不是错误。
    """
    answered = _answered(outcomes)
    if not answered:
        return float("nan")
    hits = sum(1 for outcome in answered if set(outcome.expect) & set(outcome.injected[:k]))
    return hits / len(answered)


def recall_at_k(outcomes: Sequence[QueryOutcome], k: int) -> float:
    """``recall@K`` —— **micro**：全部期望条目里，有多少条进了前 K。

    与 :func:`hit_at_k` 的差别只在分母：hit 是"这条查询答对了吗"（macro），
    recall 是"该找的条目找回来几成"（micro）。一条查询有两个期望条时，两个数
    会分开——这正是它值得单独报的理由。
    """
    answered = _answered(outcomes)
    total = sum(len(outcome.expect) for outcome in answered)
    if not total:
        return float("nan")
    found = sum(len(set(outcome.expect) & set(outcome.injected[:k])) for outcome in answered)
    return found / total


def mrr(outcomes: Sequence[QueryOutcome]) -> float:
    """``MRR`` —— 第一条期望记忆的排名倒数的均值。**未命中计 0**，不是排除。

    排除未命中会让这个数变成"答对的时候答得多靠前"，那是另一个量
    （:func:`mrr_answered`）。两个都报，但门槛绑的是这一个——它把"答不出"算进去。
    """
    answered = _answered(outcomes)
    if not answered:
        return float("nan")
    total = 0.0
    for outcome in answered:
        rank = _first_rank(outcome)
        total += 1.0 / rank if rank else 0.0
    return total / len(answered)


def mrr_answered(outcomes: Sequence[QueryOutcome]) -> float:
    """只对**命中了的**查询算的 MRR。**只报告，不作门槛。**"""
    ranks = [rank for outcome in _answered(outcomes) if (rank := _first_rank(outcome))]
    if not ranks:
        return float("nan")
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def _first_rank(outcome: QueryOutcome) -> int:
    """第一条期望记忆在第几位（1 起）。没有就是 0。"""
    for index, memory_id in enumerate(outcome.injected, start=1):
        if memory_id in set(outcome.expect):
            return index
    return 0


def false_injection_rate(outcomes: Sequence[QueryOutcome]) -> float:
    """``误注入率`` —— 注入的东西里，有多少条既不是期望的也不是可选的。**micro**。

    ``Σ|inj ∖ (E∪O)| / Σ|inj|``，在全部查询上汇总（含 ``no_answer``）。

    为什么不用"含错注入的查询比例"：``max_items=8`` 下那几乎恒为 1。为什么用
    micro 而不是 macro：macro 会让一条只注入 1 条的查询与一条注入 8 条的查询同权，
    而模型看到的是 token——多注入 7 条垃圾的代价是实打实的。

    **这个口径在槽位填满时会饱和**（每条查询 1 条正确、8 个槽位 → 上限 0.875）。
    所以它主要响应"注入了多少条"，而不是"排得准不准"；排得准不准由
    :func:`hit_at_k` 与 :func:`mrr` 管。两者互补，不重复。
    """
    injected = sum(len(outcome.injected) for outcome in outcomes)
    if not injected:
        return float("nan")
    wrong = sum(len(set(outcome.injected) - outcome.relevant) for outcome in outcomes)
    return wrong / injected


def no_answer_injections(outcomes: Sequence[QueryOutcome]) -> float:
    """库**确实没有**答案的查询，平均被注入了几条。**越低越好。**

    这是整套指标里最尖锐的一个。``no_answer`` 的期望集是空的，所以**任何注入都是
    错的**——而它不设上限：分数阈值只要形同虚设，这里就会稳定地给出 8。

    用"平均条数"而不是"比例"（比例恒为 1，因为 8 条里 8 条都错），也不是"有没有
    注入"（布尔量，丢掉了 8 与 1 的区别——而 8 与 1 的差别正是浪费的 token）。
    """
    queries = [outcome for outcome in outcomes if not outcome.answered]
    if not queries:
        return float("nan")
    return sum(len(outcome.injected) for outcome in queries) / len(queries)


def auc(outcomes: Sequence[QueryOutcome], *, negatives: str = "all") -> float:
    """``AUC`` —— 任取一个正例与一个负例，正例分数更高的概率。0.5 = 瞎猜。

    正例 = ``E_q`` 里每一条的 ``fused``；负例按 ``negatives`` 取：

    * ``"all"``：库里不属于 ``E_q ∪ O_q`` 的每一条。这是**主口径**——它问的是
      "这个打分函数能把相关的与全库其它东西分开吗"。
    * ``"never"``：只取同主题干扰项。这是最难的那一半，见
      :func:`hard_negative_auc`。

    逐查询配对之后 **pooled**（所有查询的正负例倒进同一个池子），不是"每条查询
    算一个 AUC 再平均"：后者在正例只有 1 条时，每条查询的 AUC 就是"这一条排在
    多少负例前面"，平均之后与 pooled 数值接近但含义更绕。

    并列按 ½ 计。任一侧为空时返回 ``nan``，**不是 0**——0 是"排得完全相反"，
    而空集是"没有可比的"，把后者写成前者会让一次语料事故看起来像一次性能崩溃。
    """
    positives: list[float] = []
    negative_scores: list[float] = []
    for outcome in _answered(outcomes):
        if outcome.scores is None:
            continue
        pool = outcome.relevant if negatives == "all" else set(outcome.never)
        positives.extend(outcome.scores.get(memory_id, 0.0) for memory_id in outcome.expect)
        negative_scores.extend(
            score for memory_id, score in outcome.scores.items() if memory_id not in pool
        )
    pairs = len(positives) * len(negative_scores)
    if not pairs:
        return float("nan")
    wins = ties = 0
    for positive in positives:
        for negative in negative_scores:
            wins += positive > negative
            ties += positive == negative
    return (wins + 0.5 * ties) / pairs


def hard_negative_auc(outcomes: Sequence[QueryOutcome]) -> float:
    """把负例限定成 ``never``（同主题干扰项）的 AUC。**只报告。**"""
    return auc(outcomes, negatives="never")


def summarize(outcomes: Sequence[QueryOutcome]) -> dict[str, float]:
    """把四项指标（外加几个二级量）汇总成一个扁平字典。

    键名就是 README 与基线文件里用的名字。加一个键要同时改三处：这里、基线文件
    的结构、以及 README 的公式表——它们是同一件事的三个视图。
    """
    summary: dict[str, float] = {}
    for k in K_VALUES:
        summary[f"hit@{k}"] = hit_at_k(outcomes, k)
        summary[f"recall@{k}"] = recall_at_k(outcomes, k)
    summary["mrr"] = mrr(outcomes)
    summary["mrr_answered"] = mrr_answered(outcomes)
    summary["false_injection_rate"] = false_injection_rate(outcomes)
    summary["no_answer_injections"] = no_answer_injections(outcomes)
    summary["auc"] = auc(outcomes)
    summary["hard_negative_auc"] = hard_negative_auc(outcomes)
    return summary
