"""路由语料上的端到端断言：结构不变量、可复现性、规则层门槛。

:mod:`tests.test_recall_gate` 的同位体——那一个跑金标准集判召回，这一个跑路由语料判
语域。它**不是**在测"斜杠解析对不对""某个词在不在表里"——那些由
:mod:`tests.test_router` 用可控的单句逐条断言。它测的是三件只有"整套量具一起跑"才暴露
得出来的事：

1. **结构不变量**：斜杠类的每条都必须被显式认出来，`ambiguous` 类的歧义标志必须照亮
   （见 :func:`~biosnn_memterm.bench.routeeval.route_structural_violations`）。
2. **可复现**：连跑两遍逐位相同。路由是纯函数，所以这一条比召回那边更容易成立——
   **正因为它容易成立，它才是一道便宜的哨兵**：任何一天有人往 `decide` 里塞一个时间、
   一个随机、一段跨轮状态，它立刻红。
3. **门槛**：四项指标不跌破 ``baseline.json`` 的容差。**门槛只绑规则层。**

120 条全跑完是毫秒级的，但仍然共用一个 module 级的结果：语料只该读一次盘。
"""

from __future__ import annotations

import json

import pytest

from biosnn_memterm.bench import (
    RouteCorpus,
    default_routes_dir,
    evaluate,
    load_corpus,
    routes_digests,
)
from biosnn_memterm.bench.routeeval import (
    gate_rows,
    route_structural_violations,
    route_summary,
    sanity_problems,
)

BASELINE = default_routes_dir() / "baseline.json"


@pytest.fixture(scope="module")
def corpus() -> RouteCorpus:
    return load_corpus()


@pytest.fixture(scope="module")
def outcomes(corpus: RouteCorpus):
    return evaluate(corpus)


# ── 结构不变量 ──────────────────────────────────────────────────────────────


def test_structural_invariants(corpus: RouteCorpus, outcomes) -> None:
    """斜杠层照做、歧义标志照亮、`ambiguous` 那一档没被浪费。

    **这两条不参与任何聚合。** 它们是"永真"的性质（显式指定的语域必须生效；判不出来
    必须说出来），一旦为假就是 bug。混进准确率里的话，一次"斜杠没生效"只影响 7 条里
    的 1 条——准确率掉 0.008，看起来像噪声，而症状是"用户显式说的话被无视了"。
    """
    violations = route_structural_violations(corpus, outcomes)
    assert violations == [], "结构不变量被违反——这是 bug，不是标定问题：\n  " + "\n  ".join(
        violations
    )


def test_every_utterance_produced_an_outcome(corpus: RouteCorpus, outcomes) -> None:
    """自检：条数与顺序都要对得上。

    顺序也断言，因为 ``evaluate`` 承诺"与语料同序"——报告是按行对照语料的，而**顺序
    错了不会让任何一个指标变坏**，只会让排查时看的那一行是别人的。
    """
    assert len(outcomes) == len(corpus.utterances)
    assert [o.utterance_id for o in outcomes] == [u.utterance_id for u in corpus.utterances]


# ── 可复现 ──────────────────────────────────────────────────────────────────


def test_router_benchmark_is_idempotent(corpus: RouteCorpus) -> None:
    """连跑两遍，指标与逐条判定都逐位相同。

    这条是 :data:`~biosnn_memterm.bench.routeeval.DEFAULT_TOLERANCE` 取 0.02 的**前提**：
    容差之所以不是"采样噪声"，正是因为这里没有任何随机来源。所以它不只是可复现性
    检查，它是那个容差取值的证据。

    逐条判定也一起比（不只是聚合指标）：两条不同的错误在总数上可以互相抵消，
    而"总数一样"会让人以为什么都没变。
    """
    first = evaluate(corpus)
    second = evaluate(corpus)
    assert first == second, (
        "两遍跑出来的判定不一样——规则层里混进了隐藏状态（时间、随机、跨轮状态）。"
        "路由必须是一个纯函数，否则门槛的容差就得按噪声重新定。"
    )
    assert route_summary(first) == route_summary(second)


# ── 门槛 ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def baseline() -> dict:
    """基线文件。**必须存在**——缺失是失败，不是跳过。

    把"没基线"写成 ``skip`` 的话，守卫就变成了一个空转的东西，而空转的守卫比没有
    守卫更糟：它让人以为有人在看。
    """
    assert BASELINE.is_file(), (
        f"找不到 {BASELINE}。路由基线是要入库的资产（计划书 §五）。"
        f"生成它：`python tools/bench_router.py --no-model --freeze`。"
    )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_rule_layer_gate(baseline: dict, outcomes) -> None:
    """四项指标不跌破基线的容差。**门槛只绑规则层。**

    走 :func:`~biosnn_memterm.bench.routeeval.gate_rows` 而不是直接调
    :func:`~biosnn_memterm.bench.gates.gate_rows`：前者带着路由这一套名单与方向。
    测试与 ``tools/bench_router.py`` 必须**经过同一个转发函数**，否则两边哪天各传一份
    名单，工具里的那份写反了这边仍然绿——那正是 ``bench/gates.py`` 存在的理由。
    """
    metrics = route_summary(outcomes)
    rows = gate_rows(baseline["metrics"], metrics, baseline.get("tolerance"))
    assert rows, (
        "一条判定都没产生——基线里没有一项与 GATED 对得上。"
        "这种情况下的 `all([])` 是 True，测试会静默变绿。"
    )
    failed = [row for row in rows if not row.passed]
    assert not failed, (
        "以下指标跌破了容差：\n  "
        + "\n  ".join(
            f"{row.metric}: 基线 {row.baseline:.4f} → 本轮 {row.live:.4f}"
            f"（差 {row.delta:+.4f}，下限 {row.limit:.4f}）"
            for row in failed
        )
        + (
            "\n\n这不是程序出错。先分流：语料被改过？还是规则表真的变了？"
            "（`python tools/bench_router.py --no-model --check` 会把两条线索都打出来。）"
        )
    )


def test_baseline_is_not_vacuous(baseline: dict) -> None:
    """基线自己必须像一份真实测量。

    没有这一条，"把 baseline 写成全 0 就永远绿"是两步就能走到的状态，而且它看起来
    完全正常。
    """
    problems = sanity_problems(baseline["metrics"])
    assert not problems, "基线文件不像一份真实测量：\n  " + "\n  ".join(problems)


def test_baseline_row_was_frozen_on_this_corpus(baseline: dict) -> None:
    """门槛比的必须是**同一把尺子**。

    语料改了而基线没重跑时，比较照样出结果——只是那个差值没有意义，而它会以"达标"
    的形态出现（两边的数碰巧接近）或者"跌破"（不接近），两种都不会提到真正的原因。
    """
    assert baseline["corpus"] == routes_digests(default_routes_dir()), (
        "baseline.json 记的语料指纹与现在的不一致。改语料要在同一次提交里更新"
        "freeze.json 并重跑 `--freeze`。"
    )


def test_model_layer_is_not_gated(baseline: dict) -> None:
    """**基线里不许有模型层的数。** 这条是门槛纪律的守卫。

    模型层要 Ollama。哪天有人图省事把某个模型跑出来的数一起 ``--freeze`` 进去，门槛
    就会在**装了这个模型的机器上**绿、在别的机器上红——而红的那台看起来像"路由退化
    了"。所以基线里的 ``model_layer`` 必须是空的，这一条让它保持空的。
    """
    assert not baseline.get("model_layer"), (
        "baseline.json 里出现了模型层的数。门槛只绑规则层（见 tools/bench_router.py 的"
        "模块文档）：把模型数冻进基线，等于让装没装模型决定提交能不能过。"
    )
