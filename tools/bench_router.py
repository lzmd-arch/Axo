"""路由基准：语域判定的准确率，以及它的门槛。

``agent/router.py`` 的规则层是一张**手写的标记词表**。在它之前，"改了表里的一个词
之后路由是变准了还是变不准了"这个问题没有答案——只能靠读代码判断，而读代码判断不出
"``别``"这个否定词该放在 task 之前还是之后。

这个脚本量的是那个缺口：125 条带人工标注的输入，逐条走
:func:`~axo.agent.router.decide_turn`，报准确率与两类的查全/查准。
指标的口径写在 :mod:`axo.bench.routeeval` 里，**那里是定义，这里只是报告**。

**门槛只绑规则层。** 规则层是纯函数、离线、确定性，所以它在 CI 与 pre-commit 里每次
都跑得动；模型自分类要 Ollama——把门槛绑在它上面等于让"这台机器装没装模型、装了哪个"
决定提交能不能过（与 ``tools/bench_recall.py`` 把门槛只绑词法通道是同一条纪律）。
模型层的数字仍然会报，因为**"该不该把 ``MODEL_CLASSIFY_DEFAULT`` 打开"正是这份报告
要回答的问题**，而它不是一个 CI 判据。

用法::

    python tools/bench_router.py               # 报规则层；探测到 Ollama 就再报模型层
    python tools/bench_router.py --no-model    # 只有规则层，不探测（CI / pre-commit）
    python tools/bench_router.py --model qwen3-4b:latest   # 指名模型，连不上即失败
    python tools/bench_router.py --check       # 与基线比，跌破容差即退出码 1
    python tools/bench_router.py --freeze      # 重算并覆盖基线（门槛只容规则层）
    python tools/bench_router.py --json        # 额外输出一份机器可读的报告

退出码：0 = 达标（或只是报告），1 = 跌破基线容差，2 = 用法或环境不对，3 = 结构不变量
被违反。**1 不等于程序出错**——它表达的是"路由比登记在案的基线差了"；3 既不是"变差"
也不是"没配好"，是 bug（见 :func:`~axo.bench.routeeval.route_structural_violations`）。
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

from axo.agent.router import (  # noqa: E402
    Classifier,
    ModelRole,
    Register,
    classify_with_model,
)
from axo.bench.routeeval import (  # noqa: E402
    DEFAULT_TOLERANCE,
    GATED,
    Outcome,
    evaluate,
    gate_rows,
    lexicon_violations,
    per_category,
    route_structural_violations,
    route_summary,
    sanity_problems,
)
from axo.bench.routes import (  # noqa: E402
    RouteCorpus,
    default_routes_dir,
    load_corpus,
    routes_digests,
)

BASELINE_NAME = "baseline.json"
BASELINE_SCHEMA_VERSION = 1

DEFAULT_HOST = "http://127.0.0.1:11434"
#: 本机装着的那个（见 ``tools/smoke_agent.py`` 的实测记录）。**它只是默认值**：
#: 报告里那份数字必须连同模型名一起写进基线，否则"换了个模型"会看起来像"变好了"。
DEFAULT_MODEL = "qwen3-4b:latest"


# ── 用法与环境 ──────────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """用法或环境不对。**退出码 2 而不是 1**——1 留给"指标跌了"。"""
    print(f"\n{message}", file=sys.stderr)
    raise SystemExit(2)


def _resolve_classifier(args: argparse.Namespace) -> Classifier | None:
    """决定这一轮有没有模型层。返回那个分类器或 ``None``。

    三种模式的差别只在"连不上时怎么办"（与 ``bench_recall.py`` 的
    ``_resolve_embedder`` 一字不差的分工）：

    * ``--no-model``：不探测。CI 与 pre-commit 走这条——**不探测**与"探测了但没连上"
      是不同的：前者不会在 CI 日志里留下一次徒劳的连接尝试与超时。
    * ``--model NAME``：连不上就是失败（退出码 2）。指名要量它的人要的是那个数，
      静默退回规则层会给他一份"看起来含模型、其实只有规则"的报告。
    * 默认：探测，连不上就只跑规则层并**打印一行**。这不是降级——"没装模型时系统
      照常可用"是本项目的常态（:data:`~axo.agent.router.MODEL_CLASSIFY_DEFAULT`
      默认就是关的）。退出码不受影响。
    """
    if args.no_model:
        return None

    from axo.llm.ollama import OllamaChatBackend, probe

    if probe(args.host) is None:
        if args.model:
            _die(f"--model 要求模型层，但 {args.host} 上没有响应。先启动 Ollama。")
        print(f"未发现对话后端（{args.host} 无响应），本轮只量规则层——这是正常状态，")
        print("不是降级：模型自分类的默认值本来就是关的。退出码不受影响。")
        return None

    name = args.model or DEFAULT_MODEL
    backend = OllamaChatBackend(model=name, host=args.host)
    print(f"模型层：{name} @ {args.host}（上下文 {backend.context_window} tokens）")

    # 逐条问一次，所以要把"问到第几条了"打出来——本地 4B 模型 125 条要跑几分钟，
    # 一段没有任何输出的几分钟与"卡死了"在终端上看起来一模一样。
    seen = 0

    def classifier(text: str) -> Register | None:
        nonlocal seen
        seen += 1
        if seen % 20 == 0:
            print(f"  模型层 已判 {seen} 条…")
        return classify_with_model(text, backend, max_tokens=args.max_tokens).register

    return classifier


# ── 报告 ────────────────────────────────────────────────────────────────────


def _fmt(value: float) -> str:
    return "  n/a " if value != value else f"{value:.4f}"


def _misses(corpus: RouteCorpus, outcomes: list[Outcome]) -> list[dict[str, Any]]:
    """判错的那些条，**连正文一起**。

    它是一个报告项而不是调试残留：语料的 ``note`` 解释了"为什么这条该是 X"，
    而判错的清单是把那张表读一遍的最短路径——没有它，"准确率 0.95"要排查就得自己
    去跑一遍并逐条对。
    """
    by_id = {utterance.utterance_id: utterance for utterance in corpus.utterances}
    return [
        {
            "id": outcome.utterance_id,
            "category": outcome.category,
            "expect": outcome.expect,
            "got": outcome.got,
            "source": outcome.source,
            "ambiguous": outcome.ambiguous,
            "text": by_id[outcome.utterance_id].text,
            "note": by_id[outcome.utterance_id].note,
        }
        for outcome in outcomes
        if not outcome.correct
    ]


def _report_metrics(title: str, metrics: dict[str, float], *, gated: bool) -> None:
    mark = "（进门槛）" if gated else "（只报告）"
    print(f"\n── {title} {mark} ──")
    print(
        f"  条数 {int(metrics['n'])}   覆盖 {_fmt(metrics['coverage'])}"
        f"   准确率 {_fmt(metrics['accuracy'])}（在敢判的那些里）"
    )
    print(
        f"  含「不知道」一起算 {_fmt(metrics['accuracy_all'])}"
        f"   歧义标志 {_fmt(metrics['ambiguous_rate'])}"
    )
    print(f"  任务  查全 {_fmt(metrics['task_recall'])}   查准 {_fmt(metrics['task_precision'])}")
    print(f"  闲聊  查全 {_fmt(metrics['chat_recall'])}   查准 {_fmt(metrics['chat_precision'])}")
    print(f"  「一律判闲聊」的准确率 {_fmt(metrics['majority_baseline'])}   ← 低于它就是不如不判")


def _report(
    corpus: RouteCorpus,
    rule: dict[str, float],
    categories: dict[str, dict[str, float]],
    misses: list[dict[str, Any]],
    extra: list[tuple[str, dict[str, float]]],
    violations: list[str],
    lexicon: list[str],
) -> None:
    _report_metrics("规则层（off）", rule, gated=True)

    if extra:
        print("\n── 分列（同一份语料、同一段判定代码，只换 model_role） ──")
        print(f"  {'层':<10}{'覆盖':>10}{'准确率':>10}{'任务查全':>12}{'闲聊查全':>12}{'歧义':>10}")
        for name, metrics in [("rule", rule), *extra]:
            print(
                f"  {name:<10}{_fmt(metrics['coverage']):>10}"
                f"{_fmt(metrics['accuracy']):>10}"
                f"{_fmt(metrics['task_recall']):>12}{_fmt(metrics['chat_recall']):>12}"
                f"{_fmt(metrics['ambiguous_rate']):>10}"
            )

    print("\n── 各类别（只报告，不进门槛：有的类只有几条，标准误比差值大） ──")
    for category, stats in categories.items():
        print(
            f"  {category:<14} 覆盖 {_fmt(stats['coverage'])}"
            f"  准确率 {_fmt(stats['accuracy'])}"
            f"  歧义 {_fmt(stats['ambiguous_rate'])}  (n={int(stats['n'])})"
        )

    if misses:
        print(f"\n── 规则层判错的 {len(misses)} 条 ──")
        for miss in misses:
            tail = f"  # {miss['note']}" if miss["note"] else ""
            print(
                f"  {miss['id']}  [{miss['category']}]  期望 {miss['expect']}"
                f" → 判成 {miss['got']}（{miss['source']}）{tail}"
            )
            print(f"      {miss['text']}")
    else:
        print("\n── 规则层一条都没判错 ──")

    if lexicon:
        print(f"\n── 语料里出现了禁词 {len(lexicon)} 处（这是不可逆的问题，必须在提交前修） ──")
        for line in lexicon:
            print(f"  {line}")

    if violations:
        print(f"\n── 结构不变量违例 {len(violations)} 条（这是 bug，不是标定问题） ──")
        for line in violations:
            print(f"  {line}")
    else:
        print("\n── 结构不变量：0 违例（斜杠层照做、歧义标志照亮） ──")


# ── 基线 ────────────────────────────────────────────────────────────────────


def _baseline_path(args: argparse.Namespace, routes_dir: Path) -> Path:
    return Path(args.baseline) if args.baseline else routes_dir / BASELINE_NAME


def _check_sanity(metrics: dict[str, float], *, where: str) -> None:
    """基线（或候选基线）自己必须像一份基线。见 :data:`SANITY_FLOOR`。"""
    problems = sanity_problems(metrics)
    if problems:
        _die(
            f"{where}的指标不像一份真实测量：\n  "
            + "\n  ".join(problems)
            + "\n\n这是防空转的反向断言。一份把准确率写成 0 的基线会让门槛永远绿，"
            "而它看起来完全正常——所以宁可在这里停住。"
        )


def _freeze(payload: dict, path: Path, metrics: dict[str, float]) -> int:
    """重算并覆盖基线。**强制重跑，不接受手工输入。**"""
    _check_sanity(metrics, where="这一轮实测")
    payload["schema_version"] = BASELINE_SCHEMA_VERSION
    payload["tolerance"] = dict(DEFAULT_TOLERANCE)
    payload["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["note"] = (
        "这是**实测值，不是目标值**（先跑后填）。门槛 = 本文件的值 ± tolerance，"
        "方向见 axo.bench.routeeval.GATED（四项都是越大越好），"
        "容差的含义见那里的 DEFAULT_TOLERANCE（这份测量是确定性的，容差不是噪声，"
        "是「允许几条翻转而不惊动人」）。改语料要同时改 freeze.json 并重跑 --freeze。"
        "**门槛只绑规则层**：模型层要 Ollama，把门槛绑上去等于让装没装模型决定提交"
        "能不能过。"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n"
    # newline="\n"：Python 的文本模式在 Windows 上会写 CRLF，而冻结判据是原始字节
    # ——一份自己把行尾写歪的基线文件会在下一台机器上对不上自己的 sha256。
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\n基线已写入 {path}（{len(GATED)} 项指标，容差 {DEFAULT_TOLERANCE}）")
    return 0


def _check(payload: dict, path: Path, metrics: dict[str, float]) -> tuple[int, list]:
    """与基线比。基线一定存在——缺基线在 :func:`_run` 里就停住了（跑之前）。

    返回 ``(退出码, 判定行)``：判定行要进 ``--json`` 的 artifact，而在返回之前
    再读一遍基线文件只为拿那几个数，是白读一次盘。
    """
    baseline = json.loads(path.read_text(encoding="utf-8"))
    _check_sanity(baseline.get("metrics", {}), where="基线文件")

    if baseline.get("corpus") != payload["corpus"]:
        print(
            "⚠️ 语料 utterances.jsonl 的 sha256 与基线记录的不一致——"
            "**门槛比较的已经不是同一把尺子**。若这是有意改题，"
            "请在同一次提交里更新 benchmarks/routes/freeze.json 并重跑 --freeze。",
            file=sys.stderr,
        )

    rows = gate_rows(baseline["metrics"], metrics, baseline.get("tolerance"))
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
            f"\n{len(failed)} 项跌破容差。这不是程序出错——路由行为确实变了。"
            f"先看是不是语料被改过（上面的 sha256 警告），再看是不是有意改的。"
        )
        return 1, rows
    print("\n全部在容差内。")
    return 0, rows


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    channels = parser.add_mutually_exclusive_group()
    channels.add_argument(
        "--no-model",
        action="store_true",
        help="只量规则层，不探测 Ollama（CI 与 pre-commit 用这条）",
    )
    channels.add_argument("--model", default="", help=f"指名对话模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama 地址")
    parser.add_argument(
        "--max-tokens", type=int, default=8, help="模型自分类的最大输出（要的是一个词）"
    )
    parser.add_argument("--corpus-dir", default="", help="路由语料目录（默认自动定位）")
    parser.add_argument(
        "--baseline", default="", help=f"基线文件（默认 <语料目录>/{BASELINE_NAME}）"
    )
    parser.add_argument("--check", action="store_true", help="与基线比，跌破容差即退出码 1")
    parser.add_argument("--freeze", action="store_true", help="重算并覆盖基线")
    parser.add_argument("--json", action="store_true", help="额外输出一份机器可读的报告")
    parser.add_argument("--quiet", action="store_true", help="只留结论（pre-commit 用）")
    args = parser.parse_args(argv)

    # ``--json`` 时把"人看的"那一路整体挪去 stderr，stdout 只留一份 JSON。理由与
    # bench_recall.py 一字不差：不这样，`> report.json` 出来的东西谁都解析不了。
    human = sys.stderr if args.json else sys.stdout
    try:
        with contextlib.redirect_stdout(human):
            code, payload = _run(args)
    except FileNotFoundError as exc:
        # 缺语料、缺基线、``--config`` 指错，都是**环境**问题（工作目录不对、
        # 路径打错），不是"指标跌了"。理由与 bench_recall.py 一字不差：退出码 1
        # 的含义是"行为确实变了"，而"找不到工作目录"退 1 会让 CI 上一条环境错误
        # 伪装成一条门槛失败。共用的退出码约定见模块头注与 README 的排障表。
        _die(str(exc))

    if args.json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return code


def _run(args: argparse.Namespace) -> tuple[int, dict]:
    routes_dir = Path(args.corpus_dir) if args.corpus_dir else default_routes_dir()
    corpus = load_corpus(routes_dir)

    # 门槛只绑规则层（见模块文档）。--check/--freeze 与模型层是矛盾的：拿一份纯规则
    # 层的基线去比带模型跑出来的数，差值不是"变好还是变坏"，只是"换了个东西"。
    if (args.check or args.freeze) and args.model:
        _die("--check/--freeze 按定义只在规则层上做（门槛不绑模型）。去掉 --model。")
    if (args.check or args.freeze) and not args.no_model:
        args.no_model = True
        print("门槛只绑规则层，本轮按 --no-model 跑。")

    # 缺基线要在**跑之前**就停住。放到 _check 里也行，但那时报告已经打完了，
    # 而"门槛只绑规则层"那行会出现在报错后面——读起来像顺序出了错。
    baseline_file = _baseline_path(args, routes_dir)
    if args.check and not baseline_file.is_file():
        _die(
            f"找不到基线文件 {baseline_file}。**没有基线就不检查**——把「缺基线」写成"
            f"跳过，守卫就变成了一个空转的东西，而空转的守卫比没有守卫更糟：它让人"
            f"以为有人在看。\n先跑一次 `python tools/bench_router.py --no-model --freeze`。"
        )

    # 规则层永远跑，且永远是进门槛的那一行。
    outcomes = evaluate(corpus)
    metrics = route_summary(outcomes)

    if not args.quiet:
        print(f"语料 {routes_dir}")
        print(f"输入 {len(corpus.utterances)} 条，类别 {corpus.categories}")

    extra: list[tuple[str, dict[str, float]]] = []
    model_name = ""
    classifier = _resolve_classifier(args)
    if classifier is not None:
        model_name = args.model or DEFAULT_MODEL
        # `only`：规则层不参与，每句都问。它回答"光靠模型够不够"。
        extra.append(
            (
                "model",
                route_summary(evaluate(corpus, classify=classifier, model_role=ModelRole.ONLY)),
            )
        )
        # `fallback`：线上打开开关时的真实形态。
        extra.append(
            (
                "hybrid",
                route_summary(evaluate(corpus, classify=classifier, model_role=ModelRole.FALLBACK)),
            )
        )

    categories = per_category(outcomes)
    misses = _misses(corpus, outcomes)
    violations = route_structural_violations(corpus, outcomes)
    lexicon = lexicon_violations(corpus)

    if not args.quiet:
        _report(corpus, metrics, categories, misses, extra, violations, lexicon)

    payload: dict = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "corpus": routes_digests(routes_dir),
        "model": model_name,
        "model_layer": dict(extra),
        "metrics": metrics,
        "per_category": categories,
        "misses": misses,
        "structural_violations": violations,
        "lexicon_violations": lexicon,
    }

    code = 0
    if args.freeze:
        # 会泄漏的语料**不许冻结**：只在测试里挡的话，泄漏会先进 baseline.json，
        # 然后才在 CI 上红——那时它已经在版本控制里了。
        if lexicon:
            _die("语料里有禁词，不冻结基线：\n  " + "\n  ".join(lexicon))
        code = _freeze(payload, baseline_file, metrics)
        payload["tolerance"] = dict(DEFAULT_TOLERANCE)
    elif args.check:
        code, gates = _check(payload, baseline_file, metrics)
        payload["gates"] = gates

    if violations:
        # 结构不变量是 bug，不是"变差了"。给一个与 1 不同的码，免得 CI 上
        # "路由退步"与"歧义标志失效"看起来是同一种红。
        if not args.quiet:
            print(f"\n⚠️ 结构不变量 {len(violations)} 条违例——这是 bug，不是标定问题。")
        code = 3 if code == 0 else code

    if args.quiet:
        print(
            f"覆盖 {_fmt(metrics['coverage'])}  准确率 {_fmt(metrics['accuracy'])}  "
            f"任务查全 {_fmt(metrics['task_recall'])}  闲聊查全 {_fmt(metrics['chat_recall'])}  "
            f"（一律判闲聊 {_fmt(metrics['majority_baseline'])}）  "
            f"→ {'达标' if code == 0 else '不达标'}"
        )
    return code, payload


def _jsonable(value: object) -> object:
    """把一份报告变成 ``json.dumps`` 认得的形状。

    两件事，都不是洁癖：``nan`` → ``null``（分母为零的指标是 ``nan``，而它进的是
    版本控制的 artifact，别的工具读不回来）、frozen dataclass → 字典（
    :class:`~axo.bench.gates.GateRow` 要进 artifact）。与
    ``bench_recall.py`` 的同名函数是同一套理由。
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
