"""基质基准：图 A 的边权分不分得开簇、W 还剩多少吸引域、第三通道值不值。

P4a 交付的**是一个数**，不是一次行为变更：这一版没有任何东西把基质接进
``memcore.recall``（``graph_weight`` 保持 0、``ChannelScores.graph`` 保持恒 0）。
理由记在 README 的那次实测否定结果里——加语义通道让 AUC 涨了而 hit@8 跌了，因为
``min_score`` 是按词法量纲标定的。**换通道组合要重标阈值，而那要真实库上的数据。**

所以这个脚本要回答的只有三个问题：

1. 图 A 的边权能不能把同簇的和异簇的分开（``neighbor_auc``）；
2. 神经元级 W 在多大规模上还剩吸引域（``completion_top1`` 随 ``--scale`` 的曲线）；
3. 关联码取多少位合适（``--k`` 曲线）——``memcore/sdr.py`` 里 ``ASSOC_K = 64``
   的出处是一处翻不到的计划书引用，**这条曲线是它的替代品**。

指标的口径写在 :mod:`axo.bench.substrate` 里，**那里是定义，这里只是
报告**。门槛的政策（哪几项、什么方向、防空转的上下界）也在那里。

用法::

    python tools/bench_substrate.py                              # 默认档，报四个数
    python tools/bench_substrate.py --k 32,64,128,256            # 关联码取多少位
    python tools/bench_substrate.py --scale 200,2000,20000       # 容量悬崖在哪
    python tools/bench_substrate.py --assoc weight               # 换一条子采样规则
    python tools/bench_substrate.py --drop 0.5 --probes 200      # 探针有多残、抽几条
    python tools/bench_substrate.py --check                      # 与基线比，跌破容差退 1
    python tools/bench_substrate.py --freeze                     # 重算并覆盖基线
    python tools/bench_substrate.py --json                       # 额外输出机器可读报告

退出码：0 = 达标（或只是报告），1 = 跌破基线容差，2 = 用法或环境不对，3 = 结构
不变量被违反。**1 不等于程序出错**——它表达的是"基质比登记在案的基线差了"；
3 是 bug（见 :func:`~axo.bench.substrate.substrate_structural_violations`）。

**门槛只绑默认档**（``--assoc hash``、``k=ASSOC_K``、``scale`` = 语料条数）。
扫描出来的那些行只报告：它们是**用来裁决参数该取多少**的，冻住其中一个就等于
把"还没裁决"伪装成"已经裁决"。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from axo.bench.gates import gate_rows, sanity_problems  # noqa: E402
from axo.bench.substrate import (  # noqa: E402
    DEFAULT_TOLERANCE,
    GATED,
    LOWER_IS_BETTER,
    SANITY_CEILING,
    SANITY_FLOOR,
    SubstrateCorpus,
    build_graph,
    coded_memories,
    completion_outcomes,
    completion_summary,
    default_substrate_dir,
    graph_outcomes,
    graph_summary,
    load_substrate_corpus,
    neighbor_pairs,
    neighbor_summary,
    probe_sample,
    scaled_memories,
    substrate_digests,
    substrate_structural_violations,
)
from memcore.sdr import ASSOC_K  # noqa: E402

BASELINE_NAME = "baseline.json"
BASELINE_SCHEMA_VERSION = 1
ASSOC_MODES = ("hash", "weight")


# ── 用法与环境 ──────────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """用法或环境不对。**退出码 2 而不是 1**——1 留给"指标跌了"。"""
    print(f"\n{message}", file=sys.stderr)
    raise SystemExit(2)


def _int_list(raw: str, flag: str) -> list[int]:
    """``"32,64"`` → ``[32, 64]``。空串给空列表（调用方决定默认值）。"""
    if not raw.strip():
        return []
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        try:
            values.append(int(piece))
        except ValueError:
            _die(f"{flag} 要一串用逗号隔开的整数，收到 {piece!r}。")
    return values


# ── 一次测量 ────────────────────────────────────────────────────────────────


class _Config:
    """这一轮的全部参数。它是基线文件里 ``config`` 那一节，也是差异诊断的依据。"""

    def __init__(self, args: argparse.Namespace, mode: str) -> None:
        self.drop = float(args.drop)
        self.probes = int(args.probes)
        self.top_m = int(args.top_m)
        self.min_weight = float(args.min_weight)
        self.decay = float(args.decay)
        self.assoc = mode

    def as_dict(self, *, k: int, scale: int) -> dict[str, Any]:
        return {
            "assoc": self.assoc,
            "k": k,
            "scale": scale,
            "drop": self.drop,
            "probes": self.probes,
            "top_m": self.top_m,
            "min_weight": self.min_weight,
            "decay": self.decay,
        }


def _measure(corpus: SubstrateCorpus, config: _Config, *, k: int, scale: int) -> dict:
    """在 ``(k, scale)`` 这一档上把四个数全量出来。

    每一步都从头做（重新编码、重新建图），**不做跨档复用**：不同 k 的关联码本来
    就不是同一批位，而复用会让"这一档的数"与"上一档的数"共享一部分状态——那样
    曲线上的拐点可能是复用的产物。

    返回的字典是**纯数据**（写得进 JSON），所以图和条目都不在里面——它们由
    :func:`_run` 就地拿去算结构不变量，不跨出这一层。
    """
    memories = scaled_memories(corpus.memories, scale)
    coded = coded_memories(memories, k=k, mode=config.assoc)
    graph = build_graph(
        coded, top_m=config.top_m, min_weight=config.min_weight, k=k, mode=config.assoc
    )
    cluster_of = {memory.memory_id: memory.cluster for memory in memories}
    pairs = neighbor_pairs(graph, cluster_of)

    sampled = probe_sample(memories, config.probes)
    completion = completion_outcomes(
        coded, [memory.memory_id for memory in sampled], drop=config.drop, top=5
    )

    hits = graph_outcomes(coded, corpus.queries, graph, decay=config.decay)

    neighbors = neighbor_summary(pairs)
    completions = completion_summary(completion)
    return {
        "config": config.as_dict(k=k, scale=scale),
        "nodes": float(len(graph)),
        "edges": float(graph.edge_count()),
        "neighbors": neighbors,
        "completion": completions,
        "graph_hit": graph_summary(hits),
        "metrics": {
            "neighbor_auc": neighbors["neighbor_auc"],
            "completion_top1": completions["completion_top1"],
        },
        "graph": graph,
        "memories": memories,
    }


# ── 报告 ────────────────────────────────────────────────────────────────────


def _fmt(value: float) -> str:
    return "  n/a " if value != value else f"{value:.4f}"


def _report_row(label: str, row: dict) -> None:
    neighbors = row["neighbors"]
    completions = row["completion"]
    hits = row["graph_hit"]
    print(
        f"  {label:<22} AUC {_fmt(neighbors['neighbor_auc'])}"
        f"  边准 {_fmt(neighbors['neighbor_precision@m'])}"
        f"  top1 {_fmt(completions['completion_top1'])}"
        f"  top5 {_fmt(completions['completion_top5'])}"
        f"  词法mrr {_fmt(hits['lexical_mrr'])}  图mrr {_fmt(hits['graph_mrr'])}"
    )


def _report(
    corpus: SubstrateCorpus,
    gate_row: dict,
    sweeps: list[tuple[str, dict]],
    violations: list[str],
) -> None:
    neighbors = gate_row["neighbors"]
    completions = gate_row["completion"]
    hits = gate_row["graph_hit"]
    config = gate_row["config"]

    print(
        f"\n── 这一档（默认档，进门槛）k={config['k']} assoc={config['assoc']}"
        f" scale={config['scale']} drop={config['drop']:.2f} probes={config['probes']}"
        f" top_m={config['top_m']} min_weight={config['min_weight']} ──"
    )
    print(
        f"  节点 {int(gate_row['nodes'])}   有向边 {int(gate_row['edges'])}"
        f"   无向边 {int(neighbors['edge_same']) + int(neighbors['edge_diff'])}"
        f"（同簇 {int(neighbors['edge_same'])} / 异簇 {int(neighbors['edge_diff'])}）"
        f"   孤点 {int(neighbors['orphan_nodes'])}"
    )
    print(f"  neighbor_auc          {_fmt(neighbors['neighbor_auc'])}   ← 进门槛（0.5 = 瞎猜）")
    print(f"  neighbor_precision@m  {_fmt(neighbors['neighbor_precision@m'])}")
    print(f"  completion_top1       {_fmt(completions['completion_top1'])}   ← 进门槛")
    print(f"  completion_top5       {_fmt(completions['completion_top5'])}")
    print(
        f"  confusion_rate        {_fmt(completions['confusion_rate'])}"
        f"   （top-1 是别的记忆；P4c 的 sleep_confusion 就是这个口径）"
    )

    print("\n── 第三通道 vs 词法基线（只报告：第三通道该不该上还没有裁决）──")
    print(f"  {'通道':<8}{'hit@1':>10}{'hit@8':>10}{'mrr':>10}")
    for channel, label in (("lexical", "词法"), ("graph", "图")):
        print(
            f"  {label:<8}{_fmt(hits[f'{channel}_hit@1']):>10}"
            f"{_fmt(hits[f'{channel}_hit@8']):>10}{_fmt(hits[f'{channel}_mrr']):>10}"
        )
    print(f"  （{int(hits['n_queries'])} 条查询）")

    if sweeps:
        print("\n── 扫描（只报告：这些行是用来裁决参数该取多少的）──")
        for label, row in sweeps:
            _report_row(label, row)

    if violations:
        print(f"\n── 结构不变量违例 {len(violations)} 条（这是 bug，不是标定问题）──")
        for line in violations:
            print(f"  {line}")
    else:
        print("\n── 结构不变量：0 违例 ──")


# ── 基线 ────────────────────────────────────────────────────────────────────


def _check_sanity(metrics: dict[str, float], *, where: str) -> None:
    """基线（或候选基线）自己必须像一份基线。见 :data:`SANITY_FLOOR` / ``_CEILING``。"""
    problems = sanity_problems(metrics, floor=SANITY_FLOOR, ceiling=SANITY_CEILING)
    if problems:
        _die(
            f"{where}的指标不像一份真实测量：\n  "
            + "\n  ".join(problems)
            + "\n\n这是防空转的反向断言。``--drop 0`` 时补全恒为 1，那个数什么都没量；"
            "一张连簇都分不开的图（AUC≈0.5）说明边的口径写错了。两者都不该被冻结。"
        )


def _freeze(payload: dict, path: Path, metrics: dict[str, float]) -> int:
    """重算并覆盖基线。**强制重跑，不接受手工输入。**"""
    _check_sanity(metrics, where="这一轮实测")
    payload["schema_version"] = BASELINE_SCHEMA_VERSION
    payload["tolerance"] = dict(DEFAULT_TOLERANCE)
    payload["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["note"] = (
        "这是**实测值，不是目标值**（先跑后填）。门槛 = 本文件的值 ± tolerance，"
        "方向见 axo.bench.substrate.GATED（两项都是越大越好）。"
        "**门槛只绑默认档**（assoc=hash、k=ASSOC_K、scale=语料条数）：扫描出来的"
        "那几行是用来裁决参数该取多少的，冻住其中一个等于把待决问题伪装成已决问题。"
        "改语料要同时改 freeze.json 并重跑 --freeze。"
        "graph_hit@* 刻意不进门槛——它回答的是「第三通道该不该上」，而那还没有裁决。"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n"
    # newline="\n"：Python 的文本模式在 Windows 上会写 CRLF，而冻结判据是原始字节
    # ——一份自己把行尾写歪的基线文件会在下一台机器上对不上自己的 sha256。
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\n基线已写入 {path}（{len(GATED)} 项指标，容差 {DEFAULT_TOLERANCE}）")
    return 0


def _check(payload: dict, path: Path, metrics: dict[str, float]) -> tuple[int, list]:
    """与基线比。基线一定存在——缺基线在 :func:`_run` 里就停住了（跑之前）。"""
    baseline = json.loads(path.read_text(encoding="utf-8"))
    _check_sanity(baseline.get("metrics", {}), where="基线文件")

    if baseline.get("corpus") != payload["corpus"]:
        print(
            "⚠️ 语料的 sha256 与基线记录的不一致——**门槛比较的已经不是同一把尺子**。"
            "若这是有意改题，请在同一次提交里更新 benchmarks/substrate/freeze.json "
            "并重跑 --freeze。",
            file=sys.stderr,
        )
    if baseline.get("config") != payload["config"]:
        print(
            "⚠️ 这一轮的参数与基线里记的不一致——门槛比较的可能已经不是同一档。"
            "改参数要重跑 --freeze，否则比较的是两个不同的东西。",
            file=sys.stderr,
        )

    rows = gate_rows(
        baseline["metrics"],
        metrics,
        baseline.get("tolerance"),
        gated=GATED,
        lower_is_better=LOWER_IS_BETTER,
    )
    tolerance = baseline.get("tolerance", {})
    print(
        f"\n── 门槛（基线 {path.name}；"
        + "、".join(f"{name} 容差 {tolerance.get(name, 0.05)}" for name in GATED)
        + "） ──"
    )
    failed = [row for row in rows if not row.passed]
    for row in rows:
        print(
            f"  {row.metric:<18} 基线 {row.baseline:.4f}  本轮 {row.live:.4f}"
            f"  差 {row.delta:+.4f}  下限 {row.limit:.4f}  "
            f"{'达标' if row.passed else '**跌破**'}"
        )
    if failed:
        print(
            f"\n{len(failed)} 项跌破容差。这不是程序出错——基质的性质确实变了。"
            f"先看是不是语料被改过（上面的 sha256 警告），再看是不是有意改的。"
        )
        return 1, rows
    print("\n全部在容差内。")
    return 0, rows


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", default="", help="基质语料目录（默认自动定位）")
    parser.add_argument(
        "--baseline", default="", help=f"基线文件（默认 <语料目录>/{BASELINE_NAME}）"
    )
    parser.add_argument(
        "--k", default="", help=f"关联码活跃位数，逗号分隔（默认只有默认档 {ASSOC_K}）"
    )
    parser.add_argument("--scale", default="", help="语料规模，逗号分隔（默认 = 语料条数）")
    parser.add_argument("--assoc", default="hash", choices=ASSOC_MODES, help="关联码的选择规则")
    parser.add_argument("--drop", type=float, default=0.5, help="探针丢掉多少比例的活跃位")
    parser.add_argument("--probes", type=int, default=200, help="抽多少条探针（跨规模可比）")
    parser.add_argument("--top-m", type=int, default=8, help="图里每个节点保留几个邻居")
    parser.add_argument("--min-weight", type=float, default=0.10, help="边权下限")
    parser.add_argument("--decay", type=float, default=0.5, help="扩散一步的衰减系数")
    parser.add_argument("--check", action="store_true", help="与基线比，跌破容差即退出码 1")
    parser.add_argument("--freeze", action="store_true", help="重算并覆盖基线")
    parser.add_argument("--json", action="store_true", help="额外输出一份机器可读的报告")
    parser.add_argument("--quiet", action="store_true", help="只留结论（pre-commit 用）")
    args = parser.parse_args(argv)

    # ``--json`` 时把"人看的"那一路整体挪去 stderr，stdout 只留一份 JSON。理由与
    # bench_router.py 一字不差：不这样，`> report.json` 出来的东西谁都解析不了。
    human = sys.stderr if args.json else sys.stdout
    try:
        with contextlib.redirect_stdout(human):
            code, payload = _run(args)
    except FileNotFoundError as exc:
        # 缺语料、缺基线、路径打错，都是**环境**问题，不是"指标跌了"。理由与
        # bench_router.py 一字不差：退出码 1 的含义是"行为确实变了"，而"找不到
        # 工作目录"退 1 会让 CI 上一条环境错误伪装成一条门槛失败。
        _die(str(exc))

    if args.json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return code


def _run(args: argparse.Namespace) -> tuple[int, dict]:
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else default_substrate_dir()
    corpus = load_substrate_corpus(corpus_dir)
    config = _Config(args, args.assoc)

    if not 0.0 <= config.drop <= 1.0:
        _die(f"--drop 是「丢掉多少比例的活跃位」，必须落在 [0, 1]，收到 {config.drop!r}。")
    if config.top_m < 1:
        _die(f"--top-m 必须 >= 1，收到 {config.top_m!r}。")
    if not 0.0 <= config.min_weight <= 1.0:
        _die(f"--min-weight 是 Jaccard 的量纲，必须落在 [0, 1]，收到 {config.min_weight!r}。")

    total = len(corpus.memories)
    if (args.check or args.freeze) and args.assoc != "hash":
        _die(
            "--check/--freeze 按定义只在默认档上做（门槛不绑扫描）。"
            f"基线冻结的是 assoc=hash 那一档，而这一轮是 {args.assoc}。"
        )

    baseline_file = Path(args.baseline) if args.baseline else corpus_dir / BASELINE_NAME
    if args.check and not baseline_file.is_file():
        _die(
            f"找不到基线文件 {baseline_file}。**没有基线就不检查**——把「缺基线」写成"
            f"跳过，守卫就变成了一个空转的东西，而空转的守卫比没有守卫更糟：它让人"
            f"以为有人在看。\n先跑一次 `python tools/bench_substrate.py --freeze`。"
        )

    if not args.quiet:
        print(f"语料 {corpus_dir}")
        print(
            f"记忆 {total} 条，簇 {len(corpus.clusters)} 个 {corpus.clusters}；"
            f"查询 {len(corpus.queries)} 条"
        )

    # 默认档永远跑，且永远是进门槛的那一行。
    gate_row = _measure(corpus, config, k=ASSOC_K, scale=total)

    requested = [
        (k, scale)
        for k in (_int_list(args.k, "--k") or [ASSOC_K])
        for scale in (_int_list(args.scale, "--scale") or [total])
    ]
    sweeps: list[tuple[str, dict]] = []
    for k, scale in requested:
        if (k, scale) == (ASSOC_K, total):
            continue
        row = _measure(corpus, config, k=k, scale=scale)
        sweeps.append((f"k={k} scale={scale}", row))

    # 结构不变量在**默认档**的图上查，用**语料里的**记忆与查询：扫描出来的那些
    # 图是参数实验，拿它们报"语料与图对不上"会把一次配置尝试说成一次 bug。
    violations = substrate_structural_violations(gate_row["graph"], corpus.memories, corpus.queries)

    if not args.quiet:
        _report(corpus, gate_row, sweeps, violations)

    payload: dict = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "corpus": substrate_digests(corpus_dir),
        "config": gate_row["config"],
        "nodes": gate_row["nodes"],
        "edges": gate_row["edges"],
        "neighbors": gate_row["neighbors"],
        "completion": gate_row["completion"],
        "graph_hit": gate_row["graph_hit"],
        "metrics": gate_row["metrics"],
        "sweeps": {label: _row_payload(row) for label, row in sweeps},
        "structural_violations": violations,
    }

    code = 0
    if args.freeze:
        code = _freeze(payload, baseline_file, gate_row["metrics"])
        payload["tolerance"] = dict(DEFAULT_TOLERANCE)
    elif args.check:
        code, gates = _check(payload, baseline_file, gate_row["metrics"])
        payload["gates"] = gates

    if violations:
        # 结构不变量是 bug，不是"变差了"。给一个与 1 不同的码，免得 CI 上
        # "基质退步"与"语料与图对不上"看起来是同一种红。
        if not args.quiet:
            print(f"\n⚠️ 结构不变量 {len(violations)} 条违例——这是 bug，不是标定问题。")
        code = 3 if code == 0 else code

    if args.quiet:
        neighbors = gate_row["neighbors"]
        completions = gate_row["completion"]
        print(
            f"AUC {_fmt(neighbors['neighbor_auc'])}  边准 {_fmt(neighbors['neighbor_precision@m'])}  "
            f"top1 {_fmt(completions['completion_top1'])}  "
            f"图mrr {_fmt(gate_row['graph_hit']['graph_mrr'])}  "
            f"→ {'达标' if code == 0 else '不达标'}"
        )
    return code, payload


def _row_payload(row: dict) -> dict:
    """一行测量里**写得进 JSON** 的那部分。

    ``_measure`` 的返回里多带了两样东西——建好的图和编好的条目——它们是给结构
    不变量就地用的，不是报告的一部分。图在 n=20000 时是几万个元素的数组，把它
    序列化进 artifact 只会让那份文件没法读。
    """
    return {key: value for key, value in row.items() if key not in ("graph", "memories")}


def _jsonable(value: object) -> object:
    """把一份报告变成 ``json.dumps`` 认得的形状。

    两件事，都不是洁癖：``nan`` → ``null``（分母为零的指标是 ``nan``，而它进的是
    版本控制的 artifact，别的工具读不回来）、frozen dataclass → 字典（
    :class:`~axo.bench.gates.GateRow` 要进 artifact）。
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
