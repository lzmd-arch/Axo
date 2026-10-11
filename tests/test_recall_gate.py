"""金标准集上的端到端断言：结构不变量、可复现性、词法门槛。

这个文件把 :mod:`biosnn_memterm.bench` 真正跑起来。它**不是**在测"召回对不对"
——召回的质量由 ``tests/test_recall.py`` 用可控的小库逐条断言。它测的是三件
只有"整套量具一起跑"才暴露得出来的事：

1. **结构不变量**：pinned / cold / excluded 一条都不许进注入。这是 bug，不是
   统计倾向，所以它是 ``== []`` 而不是某个门槛（见
   :func:`~biosnn_memterm.bench.harness.structural_violations`）。
2. **可复现**：连跑两遍逐位相同。``settle=False`` 加 ``:memory:`` 库加固定的
   "现在"，三者合起来才让"这个数字可复现"成立——少一个就有一遍会不一样。
3. **门槛**：四项 headline 指标不跌破 ``baseline_lexical.json`` 的容差。

一条 76 个查询的完整跑约 3.5 秒，所以整个模块共用一个 session 级的库与结果。
"""

from __future__ import annotations

import json

import pytest

from biosnn_memterm.bench import (
    GoldSet,
    QueryOutcome,
    build_store,
    default_gold_dir,
    load_goldset,
    run,
    store_fingerprint,
    structural_violations,
    summarize,
)
from biosnn_memterm.bench.gates import gate_rows, sanity_problems

BASELINE = default_gold_dir() / "baseline_lexical.json"


@pytest.fixture(scope="module")
def goldset() -> GoldSet:
    return load_goldset()


@pytest.fixture(scope="module")
def store(goldset: GoldSet):
    """一份共享的库。**只读地共享**——基准不写库，这正是可复现性的前提。"""
    return build_store(goldset.memories)


@pytest.fixture(scope="module")
def outcomes(goldset: GoldSet, store) -> list[QueryOutcome]:
    return run(goldset, store=store)


# ── 结构不变量 ──────────────────────────────────────────────────────────────


def test_structural_invariants(goldset: GoldSet, outcomes: list[QueryOutcome]) -> None:
    """pinned / cold / excluded 一条都不许进注入，token 不许超预算。

    **这四条不参与任何聚合。** 它们是"永真"的性质（``_scan`` 跳过 pinned、
    直接淘汰 cold；``_select`` 装不下就换下一条），一旦为假就是 bug。混进均值里
    的话，一条人格锚被注入可能让 hit@1 还涨了 0.005——数字更好看，系统更坏。
    """
    violations = structural_violations(goldset, outcomes)
    assert violations == [], "结构不变量被违反——这是 bug，不是标定问题：\n  " + "\n  ".join(
        violations
    )


def test_every_query_produced_an_outcome(goldset: GoldSet, outcomes: list[QueryOutcome]) -> None:
    """自检：条数对不上说明 ``run`` 漏跑了——后面的指标会在一个残缺的集合上算，
    而它们全都不会报错。"""
    assert len(outcomes) == len(goldset.queries)
    assert [o.query_id for o in outcomes] == [q.query_id for q in goldset.queries]


# ── 可复现 ──────────────────────────────────────────────────────────────────


def test_benchmark_is_idempotent(goldset: GoldSet, store) -> None:
    """连跑两遍：指标逐位相同，库的指纹一个字节都不变。

    两件事一起断言，因为它们坏掉的原因不同：

    * 指标不同 → 有隐藏状态（时间、随机、字典序）；
    * 指纹变了 → 有人把 ``settle`` 打开了，基准开始**改写被它观察的东西**。
      那比指标漂移更糟：它意味着跑一次基准会让下一次的结果变，而 CI 上每天
      跑一次，数字会缓慢地爬。
    """
    before = store_fingerprint(store)
    first = summarize(run(goldset, store=store))
    second = summarize(run(goldset, store=store))

    assert first == second, "两遍跑出来的指标不一样——基准有隐藏状态。"
    assert store_fingerprint(store) == before, (
        "跑完基准之后库变了——有人把 settle 打开了，或者写回路径漏了。"
        "基准必须能观察被测对象而不改变它。"
    )


# ── 门槛 ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def baseline() -> dict:
    """基线文件。**必须存在**——缺失是失败，不是跳过。

    把"没基线"写成 ``skip`` 的话，守卫就变成了一个空转的东西，而空转的守卫比
    没有守卫更糟：它让人以为有人在看。缺失时下面的断言会给出一条能照做的指示。
    """
    assert BASELINE.is_file(), (
        f"找不到 {BASELINE}。金标准基线是要入库的资产（计划书 §四）。"
        f"生成它：`python tools/bench_recall.py --no-semantic --freeze`。"
    )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_lexical_gate(baseline: dict, outcomes: list[QueryOutcome]) -> None:
    """四项 headline 指标不跌破基线的容差。**门槛只绑词法通道。**"""
    metrics = summarize(outcomes)
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
            "\n\n这不是程序出错。先分流：语料被改过？还是召回行为真的变了？"
            "（`python tools/bench_recall.py --check` 会把两条线索都打出来。）"
        )
    )


def test_baseline_is_not_vacuous(baseline: dict) -> None:
    """基线自己必须像一份真实测量。

    没有这一条，"把 baseline 写成全 0 就永远绿"是两步就能走到的状态，而且它
    看起来完全正常。
    """
    problems = sanity_problems(baseline["metrics"])
    assert not problems, "基线文件不像一份真实测量：\n  " + "\n  ".join(problems)


def test_baseline_row_was_frozen_on_this_corpus(baseline: dict) -> None:
    """门槛比的必须是**同一把尺子**。

    语料改了而基线没重跑时，比较照样出结果——只是那个差值没有意义，而它会以
    "达标"的形态出现（两边的数碰巧接近）或者"跌破"（不接近），两种都不会提到
    真正的原因。
    """
    from biosnn_memterm.bench import corpus_digests

    assert baseline["corpus"] == corpus_digests(default_gold_dir()), (
        "baseline_lexical.json 记的语料指纹与现在的不一致。改语料要在同一次提交里"
        "更新 freeze.json 并重跑 `--freeze`。"
    )
