"""门槛的政策：哪几项、什么方向、多大容差、以及"这份基线像不像基线"。

**为什么这一段在纯层而不是在 ``tools/bench_recall.py`` 里。** 判"过了没有"的
逻辑有两个调用者：命令行工具（人看的报告）和 ``tests/test_recall_gate.py``
（CI 的断言）。两处各写一遍的话，工具里的方向写反了——把"越小越好"的误注入率
当成"越大越好"——测试仍然会用自己那份正确的逻辑跑绿，于是**没有任何一处会红**。
口径只能有一个出处，这条与 :mod:`~axo.bench.metrics` 是同一个理由。

**三把尺子共用这段机制，各带各的名单。** 召回（:data:`GATED`）、路由
（:mod:`~axo.bench.routeeval` 的 ``GATED``）与关联基质
（:mod:`~axo.bench.substrate` 的 ``GATED``）判的东西毫不相干，但
"基线 ± 容差、方向写死、基线要先像一份基线"是同一套政策。所以名单、
方向表与防空转的上下界都是**参数**，默认值是召回那一套（第一个用到它的地方）。
后面那两份名单如果自己再写一遍判定，这个模块存在的理由就没了。

**方向表是参数这件事要一路走到 ``GateRow.passed``。** 只把方向表用在算 ``limit``
上、却让 ``passed`` 回头去读模块级的那张，等于把"方向必须与指标同源"这条纪律
半途丢掉：调用方传了自定义方向表时两者会不一致，症状是"越小越好的指标涨了也判
达标"，而没有任何一处报错。基质那份量具是第一个传自定义方向表的调用方（它传空集），
所以这条不再是纸上谈兵。

这里只有政策，没有 I/O：不读文件、不打印、不退出。工具负责把 :func:`gate_rows`
的结果排版，测试负责把它的 ``passed`` 变成断言。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "DEFAULT_TOLERANCE",
    "GATED",
    "LOWER_IS_BETTER",
    "SANITY_CEILING",
    "SANITY_FLOOR",
    "GateRow",
    "gate_rows",
    "sanity_problems",
]

#: 进门槛的指标。
#:
#: 四项 headline（hit@K / MRR / 误注入率 / AUC）取其中最稳的代表：hit@8 是
#: "该找的到底找回来了没有"，hit@1 是最容易被看见的那个数——只报不判会让人
#: 以为它不受保护。**per-category 不进门槛**：n≈5 时标准误约 0.18，拿它当判据
#: 等于每天红一次。
GATED = ("hit@1", "hit@8", "mrr", "false_injection_rate", "auc")

#: **越小越好**的指标。方向必须写死在这里，不能让容差的符号去推断——推断错的
#: 症状是"误注入率涨了 0.05 也算通过"，而没有任何一处会报错。
LOWER_IS_BETTER = frozenset({"false_injection_rate", "no_answer_injections"})

#: 容差。**会被抄进基线文件**，判定时以文件里的为准，这样调容差本身是一次可见的
#: diff，而不是一次没人看见的代码改动。
#:
#: 0.05 是 n≈80 逼出来的下限：p≈0.8 时标准误 ≈ √(0.8×0.2/80) ≈ 0.045，2σ 就是
#: ±0.09。容差比这更紧，红的就是噪声而不是回归。AUC 走 0.02：它的配对数是 90 万，
#: 抽样噪声远小于指标本身的可分辨度。
DEFAULT_TOLERANCE = {
    "hit@1": 0.05,
    "hit@8": 0.05,
    "mrr": 0.05,
    "false_injection_rate": 0.05,
    "auc": 0.02,
}

#: 防空转的反向断言：**基线自己必须先像一份基线**。
#:
#: 没有这两条，"把 baseline 写成全 0 就永远绿"是两步就能走到的状态，而它看起来
#: 完全正常。粗设即可——它们拦的是"这份基线是假的"，不是"召回好不好"。
SANITY_FLOOR = {"hit@8": 0.5}
SANITY_CEILING = {"false_injection_rate": 0.9}


@dataclass(frozen=True)
class GateRow:
    """一项指标的判定结果。工具把它排版，测试看 ``passed``。"""

    metric: str
    baseline: float
    live: float
    tolerance: float
    limit: float
    """刚好及格的边界值。低于（或高于，看方向）它就是跌破。"""
    delta: float
    lower_is_better: bool = False
    """这一项的方向。**由 :func:`gate_rows` 从它收到的那张方向表里填进来。**

    刻意存成字段而不是让 :attr:`passed` 去查模块级的 :data:`LOWER_IS_BETTER`：
    ``limit`` 是按调用方传进来的方向表算的，``passed`` 若按另一张表判，两者会
    不一致——而那个不一致的症状是静默的（见模块文档）。默认 ``False`` 只服务于
    手写构造的场景，生产路径一律由 :func:`gate_rows` 显式填。
    """

    @property
    def passed(self) -> bool:
        return self.live <= self.limit if self.lower_is_better else self.live >= self.limit


def gate_rows(
    baseline_metrics: Mapping[str, float],
    live_metrics: Mapping[str, float],
    tolerance: Mapping[str, float] | None = None,
    *,
    gated: tuple[str, ...] = GATED,
    lower_is_better: frozenset[str] = LOWER_IS_BETTER,
) -> list[GateRow]:
    """逐项判定。**基线里没有的指标跳过**——它没有参照物，判不了。

    容差缺省时用 :data:`DEFAULT_TOLERANCE`，而不是 0：一份漏写了 tolerance 的
    基线会因此变成"必须逐位相同"，而那在浮点下几乎是必红。

    ``gated``/``lower_is_better`` 是给另外两把尺子（路由、基质）留的口子。方向表
    跟着名单一起传，是因为**方向必须与指标同源**：调用方只传名单、让方向默认成
    召回的，路由或基质里哪天加一个"越小越好"的指标就会静默按"越大越好"判。
    传进来的方向表还会逐项写进 :attr:`GateRow.lower_is_better`——判定与显示都读
    那一个字段，不回头查模块级的默认表。
    """
    settings = DEFAULT_TOLERANCE if tolerance is None else tolerance
    rows: list[GateRow] = []
    for name in gated:
        if name not in baseline_metrics or name not in live_metrics:
            continue
        want = float(baseline_metrics[name])
        live = float(live_metrics[name])
        tol = float(settings.get(name, DEFAULT_TOLERANCE.get(name, 0.05)))
        descending = name in lower_is_better
        limit = want + tol if descending else want - tol
        rows.append(
            GateRow(
                metric=name,
                baseline=want,
                live=live,
                tolerance=tol,
                limit=limit,
                delta=live - want,
                lower_is_better=descending,
            )
        )
    return rows


def sanity_problems(
    metrics: Mapping[str, float],
    *,
    floor: Mapping[str, float] = SANITY_FLOOR,
    ceiling: Mapping[str, float] = SANITY_CEILING,
) -> list[str]:
    """这份指标不像真实测量时返回理由（空列表 = 通过）。

    ``nan`` 一律算不合格：分母为零的指标本来就取不到，而一份**基线文件**里出现
    ``nan`` 说明冻结那一步的语料是坏的——拿它当门槛的结果是每项比较都是 False，
    于是"永远红"，看起来像召回坏了。
    """
    problems: list[str] = []
    for name, low in floor.items():
        if name not in metrics:
            continue
        value = metrics[name]
        if value != value or value < low:
            problems.append(f"{name} = {value} < {low}")
    for name, high in ceiling.items():
        if name not in metrics:
            continue
        value = metrics[name]
        if value != value or value > high:
            problems.append(f"{name} = {value} > {high}")
    return problems
