"""路由语料上的读数：准确率、混淆、结构不变量。

**它是 :mod:`.harness` 的同位体**——那一个把金标准集灌进 ``:memory:`` 库再逐条召回，
这一个把路由语料逐条喂给 :func:`~axo.agent.router.decide_turn`。两者都是
"建一个环境 → 逐条跑 → 汇总"，区别只在被测对象。

**为什么量的是无状态的那一票。** 迟滞是跨轮状态，而语料是一句句独立的输入：把状态机
接进来，读数就会取决于语料的排列顺序，而顺序是任意的。所以这里走
:func:`~axo.agent.router.decide_turn` 而不是 ``route_turn``，测的是
"这一层判了什么"。迟滞本身由 ``tests/test_router.py`` 单独测——它不是准确率问题，
是**跨轮行为**问题。

**两把尺子。** 规则层确定性、离线、进 CI 门槛；模型层要 Ollama，**只报告**。把门槛
绑在模型上等于让"这台机器装没装模型"决定提交能不能过（召回那边同一条纪律，见
``tools/bench_recall.py``）。报告把两者分列，是因为它们回答的是两个不同的问题：
"表够不够用"与"模型能不能补上表够不着的那些"。

**一个不参与判定但必须印出来的数**：:func:`route_summary` 里的 ``majority_baseline``。
这份语料里闲聊占多数，于是"一律判闲聊"本身就有不错的准确率——不印出来，读者会把
"准确率 0.85"读成"规则层挺行"。

**两个函数带 ``route_`` 前缀，是因为平铺命名空间里已经有同名的了。** 召回那边的
:func:`~axo.bench.metrics.summarize` 与
:func:`~axo.bench.harness.structural_violations` 都经
``bench/__init__.py`` 导出，本模块的同位体如果也叫这两个名字，就会在那一层**互相顶替**
——而且顶替掉的那一个还会**静默地**换掉调用方拿到的东西（症状是一条莫名其妙的
``AttributeError``，指向一个看起来完全无关的属性）。这不是假想的：本节写下的第一版就
是那么写的，被 ``tests/test_recall_gate.py::test_structural_invariants`` 当场抓住。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from axo.agent.router import (
    Classifier,
    ModelRole,
    Register,
    RouteSource,
    decide_turn,
)

from .corpus import personal_identifier_hits
from .gates import GateRow
from .gates import gate_rows as _gate_rows
from .gates import sanity_problems as _sanity_problems
from .routes import RouteCorpus

__all__ = [
    "AMBIGUOUS_FLAG_FLOOR",
    "DEFAULT_TOLERANCE",
    "GATED",
    "SANITY_FLOOR",
    "Outcome",
    "evaluate",
    "gate_rows",
    "lexicon_violations",
    "per_category",
    "route_structural_violations",
    "route_summary",
    "sanity_problems",
]

#: 进门槛的指标。四项都是**越大越好**。
#:
#: ``accuracy`` 与 ``coverage`` 必须成对，理由是这一层有一档"我不知道"：
#:
#: * 只看准确率，**一句"不知道"会在闲聊占多数的语料上白拿一个对**（歧义退回 CHAT，
#:   而多数标注恰好是 chat）。一个把 40% 的输入都推给用户确认的规则层，准确率可以
#:   接近 1.0，而那正是最该被看见的退化。
#: * 只看覆盖率，一张"什么都判 task"的表能拿满分。
#:
#: 所以 ``accuracy`` 的分母是**敢判的那些**，``coverage`` 是敢判的比例。两者一起
#: 才说得清这张表是"准"还是"怂"。
#:
#: ``task_recall``/``chat_recall`` 与 ``task_precision``/``chat_precision`` 在"每条
#: 语料非此即彼"的前提下两两同源（一条判错的 task 既拉低前者也拉低后者），所以各取
#: 一个进门槛，另一个只报告。
GATED = ("accuracy", "coverage", "task_recall", "chat_recall")

#: 容差。**会被抄进基线文件**，判定时以文件里的为准。
#:
#: **它不是采样噪声，是"允许几条翻转而不惊动人"。** 这份测量是确定性的——同一份
#: 语料跑两遍逐位相同，有一条测试钉住它（``test_router_benchmark_is_idempotent``）——
#: 所以按统计学算标准误在这里是错的：没有任何随机来源。
#:
#: 0.02 折成条数大约是一到两条（分母各不相同：准确率 95、覆盖率 120、任务查全 42、
#: 闲聊查全 53）。取这么紧是因为这张表是手写的、每次改都是一个词：**改一个词而
#: 弄坏了另一条，正是这个门槛要抓的那件事**，而它抓不住的话这个量具就白做了。
#: 反过来，改善（把 ``hard`` 里那几条收进表里）只会把数往上推，任何时候都是绿的。
DEFAULT_TOLERANCE = {
    "accuracy": 0.02,
    "coverage": 0.02,
    "task_recall": 0.02,
    "chat_recall": 0.02,
}

#: 防空转的反向断言：**基线自己必须先像一份基线**。
#:
#: 与召回那边同一条理由——没有它，"把基线写成全 0 就永远绿"是两步就能走到的状态，
#: 而它看起来完全正常。0.5 这个数对准确率还有一层意思：这份语料的
#: ``majority_baseline`` 在 0.6 上下，一条低于 0.5 的规则层还不如"一律判闲聊"。
SANITY_FLOOR = {"accuracy": 0.5, "coverage": 0.5}

#: ``ambiguous`` 那一类里，歧义标志**至少**该亮多少比例。
#:
#: 它是一条反空转断言，不是一个准确率指标：那一类的定义就是"规则层判不出来"，
#: 如果标志不亮，要么标志是死的（**一个不出标志的歧义层看起来和没有歧义层一样**），
#: 要么这些条目其实不属于这一类。
#:
#: 取 0.8 而不是 1.0，是因为**规则层变好会让这个桶变小**：某天有人给表里补了一个
#: 词，命中了这里的某一条，那条就不再是歧义了——那是进步，不该把 CI 变红。所以
#: 失败消息必须同时说出"要么挪走条目、要么规则退化了"。
AMBIGUOUS_FLAG_FLOOR = 0.8

#: 结构不变量专门看的类别名。**它们是语料的约定**，判据在
#: :func:`route_structural_violations` 里。
_SLASH_CATEGORY = "slash"
_AMBIGUOUS_CATEGORY = "ambiguous"


@dataclass(frozen=True)
class Outcome:
    """一条语料的判定结果。

    Attributes:
        source: 谁定下来的（``slash``/``rule``/``model``/``default``）。它是
            :class:`~axo.agent.router.RouteSource` 的**值**——存字符串
            而不是枚举，理由与 :class:`.routes.Utterance` 存 ``expect: str`` 一样：
            这份结果要进 JSON 报告。
    """

    utterance_id: str
    category: str
    expect: str
    got: str
    source: str
    confidence: float
    ambiguous: bool

    @property
    def correct(self) -> bool:
        return self.expect == self.got


def evaluate(
    corpus: RouteCorpus,
    *,
    classify: Classifier | None = None,
    model_role: ModelRole = ModelRole.OFF,
) -> list[Outcome]:
    """逐条跑一遍，返回与语料**同序**的结果。

    同序不是洁癖：报告要能按行对照语料，而"结果按类别排过序"在排查某一条时是纯负担。

    ``model_role`` 直接透给 :func:`~axo.agent.router.decide_turn`，
    于是报告里的三行数字（``off`` / ``fallback`` / ``only``）走的是**同一段判定
    代码**，差别只在参数。
    """
    outcomes: list[Outcome] = []
    for utterance in corpus.utterances:
        _, decision = decide_turn(utterance.text, classify=classify, model_role=model_role)
        outcomes.append(
            Outcome(
                utterance_id=utterance.utterance_id,
                category=utterance.category,
                expect=utterance.expect,
                got=decision.register.value,
                source=decision.source.value,
                confidence=decision.confidence,
                ambiguous=decision.ambiguous,
            )
        )
    return outcomes


def _ratio(numerator: int, denominator: int) -> float:
    """分母为零时返回 ``nan`` 而不是 0。

    0 是"一条都没判对"，``nan`` 是"这一项没有可比的"——两者混起来，一份缺了某一类
    的语料会显示成一个刺眼的低分，而不是"这里取不到数"。与 :mod:`.metrics` 同一条
    口径。**一个整桶都标了歧义的类别**（``ambiguous`` 那一类就是）会在这里拿到
    ``nan``——那是它的正确读数，不是异常。
    """
    return numerator / denominator if denominator else float("nan")


def decided(outcomes: Sequence[Outcome]) -> list[Outcome]:
    """规则层**敢判**的那些（没有亮歧义标志）。见 :data:`GATED` 里 accuracy 的说明。"""
    return [outcome for outcome in outcomes if not outcome.ambiguous]


def route_summary(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """准确率与两类的查全/查准。

    与召回那边的 :func:`~axo.bench.metrics.summarize` 同名不同物，所以带
    ``route_`` 前缀（见模块文档）。

    ``accuracy`` 的分母是**敢判的那些**，``accuracy_all`` 是把它放回全长度的那个
    数。两个都印出来是因为它们回答不同的问题：前者是"这张表判对了吗"，后者是
    "用户这一轮的处境有多大概率是对的"——后者才包含"它说不知道，而默认的闲聊
    恰好蒙对"那一部分。
    """
    total = len(outcomes)
    live = decided(outcomes)
    wants_task = sum(1 for o in live if o.expect == Register.TASK.value)
    wants_chat = sum(1 for o in live if o.expect == Register.CHAT.value)
    got_task = sum(1 for o in live if o.got == Register.TASK.value)
    hit_task = sum(1 for o in live if o.expect == o.got == Register.TASK.value)
    hit_chat = sum(1 for o in live if o.expect == o.got == Register.CHAT.value)
    all_task = sum(1 for o in outcomes if o.expect == Register.TASK.value)
    all_chat = sum(1 for o in outcomes if o.expect == Register.CHAT.value)
    return {
        "n": float(total),
        "coverage": _ratio(len(live), total),
        "accuracy": _ratio(sum(1 for o in live if o.correct), len(live)),
        "accuracy_all": _ratio(sum(1 for o in outcomes if o.correct), total),
        "task_recall": _ratio(hit_task, wants_task),
        "task_precision": _ratio(hit_task, got_task),
        "chat_recall": _ratio(hit_chat, wants_chat),
        "chat_precision": _ratio(hit_chat, len(live) - got_task),
        "ambiguous_rate": _ratio(total - len(live), total),
        # "一律判闲聊"能拿到多少。**不参与判定**，进报告是为了让上面的准确率有一个
        # 可比的底——这份语料闲聊占多数，不看这个数会把 0.85 读成"挺行"。
        "majority_baseline": _ratio(max(all_task, all_chat), total),
    }


def per_category(outcomes: Sequence[Outcome]) -> dict[str, dict[str, float]]:
    """按类别分列。**只报告，不进门槛**（有的类只有几条，标准误比差值大）。

    它进报告是因为"哪一类在退步"比"总分掉了半个点"信息量大得多：``negation`` 掉
    0.1 与 ``chitchat`` 掉 0.1 是完全不同的两件事。

    ``accuracy`` 与整体同口径（分母是这一桶里敢判的那些），所以 ``ambiguous`` 那一
    桶会是 ``nan``——见 :func:`_ratio`。
    """
    buckets: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category, []).append(outcome)
    return {
        category: {
            "n": float(len(group)),
            "coverage": _ratio(len(decided(group)), len(group)),
            "accuracy": _ratio(sum(1 for o in decided(group) if o.correct), len(decided(group))),
            "ambiguous_rate": _ratio(sum(1 for o in group if o.ambiguous), len(group)),
        }
        for category, group in sorted(buckets.items())
    }


def lexicon_violations(corpus: RouteCorpus) -> list[str]:
    """语料里出现了禁词就报出来。见 :data:`.corpus.PERSONAL_DENYLIST`。

    与 ``tests/test_router_corpus_integrity.py`` 的那条断言同源——**在这一层也留
    一份**，是为了让 ``--freeze`` 自己拒绝冻结一份会泄漏的语料：如果只在测试里，
    泄漏会先进 ``baseline.json``，然后才在 CI 上红。
    """
    problems: list[str] = []
    for utterance in corpus.utterances:
        hits = personal_identifier_hits(utterance.text)
        if hits:
            problems.append(f"{utterance.utterance_id} 的正文里出现了 {hits}")
    return problems


def route_structural_violations(corpus: RouteCorpus, outcomes: Sequence[Outcome]) -> list[str]:
    """结构不变量。**有违例就是 bug，不是"变差了"**——调用方给一个与 1 不同的退出码。

    与召回那边的 :func:`~axo.bench.harness.structural_violations` 同名不同物，
    所以带 ``route_`` 前缀（见模块文档）。

    两条，都不看总体准确率：

    1. ``slash`` 类的每一条，来源必须是 :attr:`~RouteSource.SLASH`，且判定必须与
       标注一致。这一档是**解析层**，与词表、与模型都无关——它错了说明斜杠解析被
       改坏了，而那种坏法的症状（"用户显式指定的语域没生效"）在总准确率上几乎看不
       出来，因为它只影响几条。
    2. ``ambiguous`` 类的歧义标志率不低于 :data:`AMBIGUOUS_FLAG_FLOOR`。见那个常量
       的说明。
    """
    problems: list[str] = []
    by_id = {outcome.utterance_id: outcome for outcome in outcomes}

    for utterance in corpus.utterances:
        if utterance.category != _SLASH_CATEGORY:
            continue
        outcome = by_id[utterance.utterance_id]
        if outcome.source != RouteSource.SLASH.value:
            problems.append(
                f"{outcome.utterance_id} 是 slash 类，但来源是 {outcome.source!r} 而不是 "
                f"{RouteSource.SLASH.value!r}——斜杠解析没认出这条命令。"
            )
        if not outcome.correct:
            problems.append(
                f"{outcome.utterance_id} 显式指定了 {utterance.expect}，却判成了 {outcome.got}。"
            )

    bucket = [o for o in outcomes if o.category == _AMBIGUOUS_CATEGORY]
    rate = _ratio(sum(1 for o in bucket if o.ambiguous), len(bucket))
    if rate != rate or rate < AMBIGUOUS_FLAG_FLOOR:
        problems.append(
            f"{_AMBIGUOUS_CATEGORY} 类的歧义标志率是 {rate:.3f}，低于 "
            f"{AMBIGUOUS_FLAG_FLOOR}（{len(bucket)} 条）。要么这些条目其实不属于这一类"
            f"（规则层已经认得它们了——那就把它们挪到合适的类别里），要么规则层退化了。"
        )
    return problems


def gate_rows(
    baseline_metrics: Mapping[str, float],
    live_metrics: Mapping[str, float],
    tolerance: Mapping[str, float] | None = None,
) -> list[GateRow]:
    """:func:`axo.bench.gates.gate_rows` 带上**路由这一套名单**。

    **判定逻辑不在这里**——它有两个调用者（``tools/bench_router.py`` 与
    ``tests/test_router_gate.py``），两处各写一遍的话，一边把方向写反了另一边仍然
    会绿。见 :mod:`axo.bench.gates` 的模块文档。
    """
    return _gate_rows(baseline_metrics, live_metrics, tolerance, gated=GATED)


def sanity_problems(metrics: Mapping[str, float]) -> list[str]:
    """同上，防空转那一条。见 :data:`SANITY_FLOOR`。"""
    return _sanity_problems(metrics, floor=SANITY_FLOOR, ceiling={})
