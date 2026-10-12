"""召回基准：金标准语料上的四项指标，以及它们的门槛。

:mod:`memcore.recall` 有九个可调参数，而在这之前**没有任何东西能告诉你改了一个
参数之后变好了还是变坏了**。既有的 ``tools/bench_space.py`` 量的是配对级的分隔度
（一句话对一条记忆的 Jaccard 能不能分开），它**没有候选池**——所以真正决定召回
行为的 K、阈值、MMR 去冗、token 预算驱逐，一条都没被量到。

这个脚本量的是那个缺口：200 条语料、76 条查询，每条都带着 ground truth，跑的是
真正的 ``recall()``。四项指标（hit@K / MRR / 误注入率 / AUC）的精确口径写在
:mod:`axo.bench.metrics` 里，**那里是定义，这里只是报告**。

**门槛只绑词法通道。** 词法 SDR 是 blake2b 哈希编码，不需要模型、不需要网络、
不需要 GPU，所以它在 CI 与 pre-commit 里每次都跑得动，而语义通道需要 Ollama——
把门槛绑在它上面等于让"有没有装模型"决定提交能不能过。语义数字仍然会报，它的
价值是回答"换一个嵌入模型会不会更好"，那不是一个 CI 判据。见计划书 §三。

用法::

    python tools/bench_recall.py                    # 报告；探测到 Ollama 就用语义通道
    python tools/bench_recall.py --no-semantic      # 纯词法，不探测（CI / pre-commit）
    python tools/bench_recall.py --semantic         # 强制要求语义通道，连不上即失败
    python tools/bench_recall.py --check            # 与基线比，跌破容差即退出码 1
    python tools/bench_recall.py --freeze           # 重算并覆盖基线（门槛只容词法）
    python tools/bench_recall.py --json             # 额外输出一份机器可读的报告

    # 联想通道的扫描（图由它自己建；只报告，不进门槛，不与 --check 同用）
    python tools/bench_recall.py --no-semantic --graph-weight 0,0.25,0.5,1.0

退出码：0 = 达标（或只是报告），1 = 跌破基线容差，2 = 用法或环境不对。
**1 不等于程序出错**——它表达的是"召回行为比登记在案的基线差了"。3 = 结构不变量
被违反，那既不是"变差"也不是"没配好"，是 bug。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import unicodedata
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from axo.bench import (  # noqa: E402
    QueryOutcome,
    build_store,
    corpus_digests,
    default_gold_dir,
    hit_at_k,
    load_goldset,
    run,
    structural_violations,
    summarize,
)
from axo.bench.gates import (  # noqa: E402
    DEFAULT_TOLERANCE,
    GATED,
    gate_rows,
    sanity_problems,
)
from axo.bench.metrics import K_VALUES, mrr  # noqa: E402
from memcore.graph import GraphIndex  # noqa: E402
from memcore.recall import RecallConfig  # noqa: E402

BASELINE_NAME = "baseline_lexical.json"
BASELINE_SCHEMA_VERSION = 1

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text:latest"


# ── 语义通道的有无 ──────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """用法或环境不对。**退出码 2 而不是 1**——1 留给"指标跌了"。"""
    print(f"\n{message}", file=sys.stderr)
    raise SystemExit(2)


def _resolve_embedder(args: argparse.Namespace):
    """决定这一轮有没有语义通道。返回后端或 ``None``。

    三种模式的差别只在"连不上时怎么办"：

    * ``--no-semantic``：不探测。CI 与 pre-commit 走这条——**不探测**与"探测了
      但没连上"是不同的：前者不会在 CI 的日志里留下一次徒劳的连接尝试与超时。
    * ``--semantic``：连不上就是失败（退出码 2）。想量语义通道的人要的是这个数，
      静默退回词法会给他一份"看起来是语义、其实是词法"的报告。
    * 默认：探测，连不上就退回词法并**打印一行**。这不是降级——``recall.py:168``
      写得很清楚："没装嵌入模型时系统必须照常可用"。退出码不受影响。
    """
    if args.no_semantic:
        return None

    from axo.llm.base import EmbeddingBackendError
    from axo.llm.ollama import (
        OllamaEmbeddingBackend,
        probe,
        probe_embedding_dim,
    )

    if probe(args.host) is None:
        if args.semantic:
            _die(f"--semantic 要求语义通道，但 {args.host} 上没有响应。先启动 Ollama。")
        print(f"未发现嵌入后端（{args.host} 无响应），本轮只有词法通道——这是正常状态，")
        print("不是降级（recall.py:168-171）。退出码不受影响。")
        return None

    try:
        dim = probe_embedding_dim(args.model, args.host)
    except EmbeddingBackendError as exc:
        if args.semantic:
            _die(f"--semantic 要求语义通道，但探测 {args.model} 失败：{exc}")
        print(f"探测嵌入模型 {args.model} 失败（{exc}），本轮只有词法通道。")
        return None

    print(f"语义通道：{args.model} @ {args.host}（{dim} 维）")
    return OllamaEmbeddingBackend(model=args.model, dim=dim, host=args.host)


# ── 报告 ────────────────────────────────────────────────────────────────────


def _per_category(outcomes: list[QueryOutcome]) -> dict[str, dict[str, float]]:
    """按类别分的 hit@1 与 MRR。**只报告，不进门槛**（见 :data:`GATED` 的注释）。

    参数里没有 ``goldset``：类别是跑的时候从查询上抄进 :class:`QueryOutcome` 的，
    所以到这一步语料本身不必在场。
    """
    buckets: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category, []).append(outcome)
    return {
        category: {
            "n": len(group),
            "hit@1": hit_at_k(group, 1),
            "mrr": mrr(group),
        }
        for category, group in sorted(buckets.items())
    }


def _fmt(value: float) -> str:
    return "  n/a " if value != value else f"{value:.4f}"


def _pad(text: str, width: int) -> str:
    """按**显示宽度**左对齐。中文一个字占两格，而 ``str.ljust`` 数的是字符数。

    用 ``ljust`` 的话表头与行会错开——"误注入率"比 "MRR" 少两个字符却宽两格，
    于是它那一列的数会往左挤。这一页是要被人直接抄进 ``CORPUS.md`` 的，错位会
    让人以为抄错了列。
    """
    shown = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
    return text + " " * max(0, width - shown)


def _report(
    metrics: dict[str, float],
    categories: dict[str, dict[str, float]],
    violations: list[str],
    *,
    channels: list[str],
) -> None:
    print(f"\n── 通道 {'+'.join(channels)} ──")
    print("  " + "  ".join(f"hit@{k} {_fmt(metrics[f'hit@{k}'])}" for k in K_VALUES))
    print(f"  MRR {_fmt(metrics['mrr'])}（只算命中者 {_fmt(metrics['mrr_answered'])}）")
    print(
        f"  误注入率 {_fmt(metrics['false_injection_rate'])}"
        f"   空答注入数 {_fmt(metrics['no_answer_injections'])} 条/查询"
    )
    print(
        f"  AUC {_fmt(metrics['auc'])}（同主题干扰项 {_fmt(metrics['hard_negative_auc'])}）"
        "  —— 只启用词法通道时这是**打分函数自身的天花板**，不是系统上界"
    )

    print("\n── 各类别（只报告，不进门槛：n≈5 时标准误约 0.18） ──")
    for category, stats in categories.items():
        if stats["hit@1"] != stats["hit@1"]:
            # hit@1 是 nan 只可能是这个桶里一条带 expect 的查询都没有。
            # 分母为零时 nan 而不是 0——0 是"排得完全不对"，两回事。
            print(f"  {category:<20} (n={int(stats['n'])})  不进 hit 分母")
        else:
            print(
                f"  {category:<20} hit@1 {_fmt(stats['hit@1'])}  MRR {_fmt(stats['mrr'])}"
                f"  (n={int(stats['n'])})"
            )

    if violations:
        print(f"\n── 结构不变量违例 {len(violations)} 条（这是 bug，不是标定问题） ──")
        for line in violations:
            print(f"  {line}")
    else:
        print("\n── 结构不变量：0 违例（pinned / cold / excluded 一条都没被注入） ──")


def _gate_rows(baseline: dict, metrics: dict[str, float]) -> list:
    """把基线文件与实测指标交给 :func:`axo.bench.gates.gate_rows`。

    判定逻辑**不在这个文件里**——它有两个调用者（这里与
    ``tests/test_recall_gate.py``），两处各写一遍的话，这里把方向写反了测试仍然
    会绿。见该模块的文档。
    """
    return gate_rows(baseline.get("metrics", {}), metrics, baseline.get("tolerance"))


# ── 联想通道扫描 ────────────────────────────────────────────────────────────


def _parse_weights(raw: str) -> list[float]:
    """把 ``--graph-weight 0,0.25,0.5`` 解析成一串非负的可靠度。

    解析失败要**当场死掉**（退出码 2），不能悄悄退回默认值：这个开关的全部意义
    就是"把这几档跑出来对比"，少跑一档而报告上看不出来，是最坏的一种错。
    """
    parts = [piece.strip() for piece in raw.replace("，", ",").split(",")]
    try:
        weights = [float(piece) for piece in parts if piece]
    except ValueError:
        _die(f"--graph-weight 只认逗号分隔的数字，收到 {raw!r}。")
    if not weights:
        _die(f"--graph-weight 至少要给一个数，收到 {raw!r}。")
    negative = [value for value in weights if value < 0.0]
    if negative:
        _die(f"--graph-weight 不能是负数，收到 {negative}。")
    return weights


_SWEEP_ROWS = (
    ("hit@1", "hit@1"),
    ("hit@8", "hit@8"),
    ("MRR", "mrr"),
    ("误注入率", "false_injection_rate"),
    ("空答注入数", "no_answer_injections"),
    ("AUC", "auc"),
    ("同主题干扰项 AUC", "hard_negative_auc"),
)


def _report_sweep(rows: list[tuple[float, dict[str, float]]], *, graph: GraphIndex) -> None:
    """把几档权重并排打出来。**宽度按列对齐**，因为这一页要直接抄进 CORPUS.md。

    表头先打图的摘要。少了它，"图上没生效"会被读成"联想没用"——而这两件事的
    修法完全不同（前者是接线问题，后者是结论）。
    """
    print(
        f"\n── 关联图 ── {len(graph)} 个节点、{graph.edge_count()} 条有向边"
        f"（top_m={graph.top_m}、min_weight={graph.min_weight}）"
    )
    header = f"  {_pad('指标', 18)}" + "".join(f"{f'w={w:.2f}':>12}" for w, _ in rows)
    print("\n── 联想通道扫描（**只报告，不进门槛**） ──")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for label, key in _SWEEP_ROWS:
        line = f"  {_pad(label, 18)}"
        for _, metrics in rows:
            line += f"{_fmt(metrics.get(key, float('nan'))):>12}"
        print(line)
    print(
        "\n  怎么读：w 是 ``RecallConfig.graph_weight``（联想通道的可靠度）。"
        "w=0.00 那一列就是出厂默认。"
    )


def _check_sanity(metrics: dict[str, float], *, where: str) -> None:
    """基线（或候选基线）自己必须像一份基线。见 :mod:`axo.bench.gates`。"""
    problems = sanity_problems(metrics)
    if problems:
        _die(
            f"{where}的指标不像一份真实测量：\n  "
            + "\n  ".join(problems)
            + "\n\n这是防空转的反向断言。一份把数字写成 0 的基线会让门槛永远绿，"
            "而它看起来完全正常——所以宁可在这里停住。"
        )


def _baseline_path(args: argparse.Namespace, gold_dir: Path) -> Path:
    return Path(args.baseline) if args.baseline else gold_dir / BASELINE_NAME


def _freeze(payload: dict, path: Path, metrics: dict[str, float]) -> int:
    """重算并覆盖基线。**强制重跑，不接受手工输入。**"""
    _check_sanity(metrics, where="这一轮实测")
    payload["schema_version"] = BASELINE_SCHEMA_VERSION
    payload["tolerance"] = dict(DEFAULT_TOLERANCE)
    payload["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["note"] = (
        "这是**实测值，不是目标值**。门槛 = 本文件的值 ± tolerance，"
        "方向见 tools/bench_recall.py 的 LOWER_IS_BETTER。改语料要同时改 "
        "freeze.json 并重跑 --freeze。"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # _jsonable：``no_answer`` 那一桶的 hit@1 是 nan（分母为零），而 nan 不是合法
    # JSON。不转换的话这个文件会带着一个裸 ``NaN`` 字面量落盘——Python 读得回来，
    # 别的任何工具读不回来，而它进的是版本控制。
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n"
    # newline="\n"：Python 的文本模式在 Windows 上会写 CRLF，而冻结判据是原始字节
    # ——一份自己把行尾写歪的基线文件会在下一台机器上对不上自己的 sha256。
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\n基线已写入 {path}（{len(GATED)} 项指标，容差 {DEFAULT_TOLERANCE}）")
    return 0


def _check(payload: dict, path: Path, metrics: dict[str, float]) -> int:
    """与基线比。基线一定存在——缺基线在 :func:`_run` 里就停住了（跑之前）。"""
    baseline = json.loads(path.read_text(encoding="utf-8"))
    _check_sanity(baseline.get("metrics", {}), where="基线文件")

    if baseline.get("config") != payload["config"]:
        print(
            "⚠️ 本轮配置与基线记录的不一致——两个不同的配置比出来的差值不是"
            "「变好还是变坏」，只是「换了个东西」。",
            file=sys.stderr,
        )

    frozen_corpus = baseline.get("corpus", {})
    for key in ("memories", "queries"):
        if frozen_corpus.get(key, {}).get("sha256") != payload["corpus"][key]["sha256"]:
            print(
                f"⚠️ 语料 {key}.jsonl 的 sha256 与基线记录的不一致——"
                f"**门槛比较的已经不是同一把尺子**。若这是有意改题，"
                f"请在同一次提交里更新 benchmarks/gold/freeze.json 并重跑 --freeze。",
                file=sys.stderr,
            )

    rows = _gate_rows(baseline, metrics)
    tolerance = baseline.get("tolerance", {})
    print(
        f"\n── 门槛（基线 {path.name}；"
        + "、".join(f"{name} 容差 {tolerance.get(name, 0.05)}" for name in GATED)
        + "） ──"
    )
    failed = [row for row in rows if not row.passed]
    for row in rows:
        # 方向读这一行自己记着的那个（由 gate_rows 按传进去的方向表填），不再原地
        # 查模块级那张表：判定与显示必须是同一个来源，否则会出现"箭头说越小越好、
        # 判定却按越大越好"这种自相矛盾的输出。
        arrow = "↓越小越好" if row.lower_is_better else ""
        print(
            f"  {row.metric:<22} 基线 {row.baseline:.4f}  本轮 {row.live:.4f}"
            f"  差 {row.delta:+.4f}  下限 {row.limit:.4f}  "
            f"{'达标' if row.passed else '**跌破**'} {arrow}"
        )
    if failed:
        print(
            f"\n{len(failed)} 项跌破容差。这不是程序出错——召回行为确实变了。"
            f"先看是不是语料被改过（上面的 sha256 警告），再看是不是有意改的。"
        )
        return 1
    print("\n全部在容差内。")
    return 0


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-semantic",
        action="store_true",
        help="强制纯词法，不探测 Ollama（CI 与 pre-commit 用这条）",
    )
    group.add_argument("--semantic", action="store_true", help="强制要求语义通道，连不上即失败")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="嵌入模型名")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama 地址")
    parser.add_argument("--corpus-dir", default="", help="金标准语料目录（默认自动定位）")
    parser.add_argument(
        "--baseline", default="", help=f"基线文件（默认 <语料目录>/{BASELINE_NAME}）"
    )
    parser.add_argument("--config", default="", help="受测配置的 JSON 文件（默认用出厂默认值）")
    parser.add_argument(
        "--graph-weight",
        default="",
        help="联想通道可靠度的扫描（逗号分隔，如 0,0.25,0.5,1）。只报告，不与 --check 同用",
    )
    parser.add_argument("--check", action="store_true", help="与基线比，跌破容差即退出码 1")
    parser.add_argument("--freeze", action="store_true", help="重算并覆盖基线")
    parser.add_argument("--json", action="store_true", help="额外输出一份机器可读的报告")
    parser.add_argument("--quiet", action="store_true", help="只留结论（pre-commit 用）")
    args = parser.parse_args(argv)

    # ``--json`` 时把"人看的"那一路整体挪去 stderr，stdout 只留一份 JSON。不这样
    # 的话报告正文会和 JSON 挤在同一个流里，`> report.json` 出来的东西谁都解析
    # 不了——而 CI 正是要把它当 artifact 传上去。用 redirect_stdout 而不是给每个
    # 打印点加一个 out 参数：流的归属是**这一整段**的性质，不是每一行的性质。
    human = sys.stderr if args.json else sys.stdout
    try:
        with contextlib.redirect_stdout(human):
            code, payload = _run(args)
    except FileNotFoundError as exc:
        # 缺语料、缺基线、``--config`` 指错，都是**环境**问题（工作目录不对、
        # 路径打错），不是"指标跌了"。让它冒到顶层的话，CI 上会伪装成门槛没过——
        # 退出码 1 的含义是"行为确实变了，去看看是不是有意改的"，而这里根本还没
        # 跑到量指标那一步。共用的退出码约定见模块头注与 README 的排障表。
        _die(str(exc))

    if args.json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return code


def _run(args: argparse.Namespace) -> tuple[int, dict]:
    """跑一轮并把报告打出来。返回 ``(退出码, 那份 JSON payload)``。

    人看的输出全部走 ``sys.stdout``（``main`` 可能把它临时接到了 stderr 上），
    JSON 由 ``main`` 单独打印——这样"报告"与"产物"不会混在同一个流里。
    """
    gold_dir = Path(args.corpus_dir) if args.corpus_dir else default_gold_dir()
    goldset = load_goldset(gold_dir)

    config = RecallConfig()
    if args.config:
        config = RecallConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))

    # 门槛只绑词法通道（计划书 §三）。--check / --freeze 与 --semantic 是矛盾的：
    # 前者拿一份纯词法的基线去比语义跑出来的数，必然红，而红的原因不是召回变差。
    if (args.check or args.freeze) and args.semantic:
        _die("--check/--freeze 按定义只在词法通道上做（门槛不绑模型）。去掉 --semantic。")
    if (args.check or args.freeze) and not args.no_semantic:
        args.no_semantic = True
        print("门槛只绑词法通道，本轮按 --no-semantic 跑。")

    # 门槛绑的是**默认 config**（``graph_weight=0``），而扫描量的是"换一档会怎样"。
    # 两者混在一起的话，门槛会拿一个非默认的权重去比一份默认权重下冻的基线——
    # 那必然红，而红的原因不是召回变差。
    if args.graph_weight and (args.check or args.freeze):
        _die(
            "--graph-weight 是扫描，--check/--freeze 是门槛，两者不能同用：\n"
            "  门槛绑的是默认配置（graph_weight=0）。要看联想，单独跑一次扫描；"
            "要把结果冻成基线，得先改 RecallConfig 的默认值。"
        )

    # 缺基线要在**跑之前**就停住。放到 _check 里也行，但那时 3.5 秒的基准已经跑完
    # 了，而且"门槛只绑词法"那行会出现在报错后面——读起来像顺序出了错。
    baseline_file = _baseline_path(args, gold_dir)
    if args.check and not baseline_file.is_file():
        _die(
            f"找不到基线文件 {baseline_file}。**没有基线就不检查**——把「缺基线」写成"
            f"跳过，守卫就变成了一个空转的东西，而空转的守卫比没有守卫更糟：它让人"
            f"以为有人在看。\n先跑一次 `python tools/bench_recall.py --no-semantic --freeze`。"
        )

    if not args.quiet:
        print(f"语料 {gold_dir}")
        print(f"记忆 {len(goldset.memories)} 条，查询 {len(goldset.queries)} 条")

    embedder = _resolve_embedder(args)

    if args.graph_weight:
        return _run_sweep(args, goldset, config, embedder)

    outcomes = run(goldset, config=config, embedder=embedder)
    metrics = summarize(outcomes)
    categories = _per_category(outcomes)
    violations = structural_violations(goldset, outcomes)

    channels = ["lexical"] + (["semantic"] if embedder is not None else [])
    if not args.quiet:
        _report(metrics, categories, violations, channels=channels)

    payload: dict = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "corpus": corpus_digests(gold_dir),
        "channels": channels,
        "space_id": embedder.space_id if embedder is not None else "",
        "config": asdict(config),
        "metrics": metrics,
        "per_category": categories,
        "structural_violations": violations,
    }

    code = 0
    if args.freeze:
        code = _freeze(payload, baseline_file, metrics)
    elif args.check:
        code = _check(payload, baseline_file, metrics)
        payload["gates"] = _gate_rows(
            json.loads(baseline_file.read_text(encoding="utf-8")), metrics
        )

    if violations:
        # 结构不变量是 bug，不是"变差了"。给一个与 1 不同的码，免得 CI 上
        # "召回退步"与"人格锚被注入了"看起来是同一种红。
        print(f"\n⚠️ 结构不变量 {len(violations)} 条违例——这是 bug，不是标定问题。")
        code = 3 if code == 0 else code

    if args.quiet:
        print(
            f"hit@1 {_fmt(metrics['hit@1'])}  hit@8 {_fmt(metrics['hit@8'])}  "
            f"MRR {_fmt(metrics['mrr'])}  误注入率 {_fmt(metrics['false_injection_rate'])}  "
            f"AUC {_fmt(metrics['auc'])}  → {'达标' if code == 0 else '不达标'}"
        )
    return code, payload


def _run_sweep(
    args: argparse.Namespace, goldset, config: RecallConfig, embedder
) -> tuple[int, dict]:
    """按几档 ``graph_weight`` 各跑一遍，打一张并排表。

    **图只建一次。** 图与权重无关——它只取决于库里的关联码——所以四档建四张一模一样
    的图是纯浪费，而 :func:`~axo.bench.harness.run` 的兜底建图是按调用
    次数触发的。这里先把库建好、把图建好，逐档复用同一份。

    库也复用：``run`` 的每一轮都是 ``settle=False``，跑完一个字节都不变，所以两次
    调用本来就等价于同一个库；复用只是省掉一次灌库。
    """
    weights = _parse_weights(args.graph_weight)
    store = build_store(goldset.memories, embedder=embedder)
    graph = GraphIndex.build(store.iter_items())

    rows: list[tuple[float, dict[str, float]]] = []
    violations: list[str] = []
    for weight in weights:
        tuned = replace(config, graph_weight=weight)
        outcomes = run(goldset, config=tuned, embedder=embedder, store=store, graph=graph)
        rows.append((weight, summarize(outcomes)))
        violations.extend(structural_violations(goldset, outcomes))

    channels = ["lexical"] + (["semantic"] if embedder is not None else []) + ["graph"]
    if not args.quiet:
        _report_sweep(rows, graph=graph)
    else:
        print("  ".join(f"w={w:.2f}:{_fmt(m['hit@1'])}" for w, m in rows))

    payload: dict = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "corpus": corpus_digests(Path(args.corpus_dir) if args.corpus_dir else default_gold_dir()),
        "channels": channels,
        "space_id": embedder.space_id if embedder is not None else "",
        "config": asdict(config),
        "graph": {
            "nodes": len(graph),
            "edges": graph.edge_count(),
            "top_m": graph.top_m,
            "min_weight": graph.min_weight,
            "assoc_k": graph.assoc_k,
            "assoc_mode": graph.assoc_mode,
        },
        "sweep": {f"{weight:g}": metrics for weight, metrics in rows},
        "structural_violations": violations,
    }

    code = 3 if violations else 0
    if violations:
        print(f"\n⚠️ 结构不变量 {len(violations)} 条违例——这是 bug，不是标定问题。")
    return code, payload


def _jsonable(value: object) -> object:
    """把一份报告变成 ``json.dumps`` 认得的形状。

    两件事，都不是洁癖：

    * **``nan`` → ``null``。** 分母为零的指标是 ``nan``（见
      :func:`~axo.bench.metrics.auc`），而 ``nan`` 不是合法 JSON。
      ``allow_nan=False`` 会直接抛，不写的话会吐出一个 ``NaN`` 字面量——那个
      字面量只有 Python 的解析器认，CI 上产出的 artifact 会在任何别的工具里坏掉。
      写成 ``null`` 也正好是它的意思："这一项没有可比的"，与"得了 0 分"分得开。
    * **numpy 标量 → Python 标量。** 它们不进 ``json`` 的类型表。
    * **frozen dataclass（:class:`~axo.bench.gates.GateRow`）→ 字典。**
      判定结果要进 artifact，而 artifact 是 JSON。
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
