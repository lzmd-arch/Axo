"""巩固的基准：全局一个 W 与分簇局部 W 的并排表。

P4a 交付的是一个**负数**（``benchmarks/substrate/CORPUS.md``）：全局 Hebbian W 的
一步补全在目标规模上不可靠，容量悬崖实测为（``weight`` 档、50% 丢位探针）::

    n=200     0.8700
    n=500     0.2250
    n=1000    0.0900
    n=2000    0.0300

P4c 的形状就是对它的回应：把全库切成一簇一簇，每簇建**自己的** W，于是"容量"这个
约束的作用域从全库缩到簇内。所以这个脚本要回答的只有一件事——**分簇把事情解决了
多少**。

它读**同一份**语料（``benchmarks/substrate/memories.jsonl``，不新造语料、不新冻
freeze），用**同一个**丢位算子（:func:`memcore.sdr.drop_bits`），把三列摆在一起：

=================  ==============  ================  ====================
规模               全局 W(hash)    全局 W(weight)    分簇局部 W(weight)
=================  ==============  ================  ====================

并排报而不是分开报，理由与 ``graph_summary`` 并排报两条通道一字不差：分开报的话
"分簇到底有没有用"会被含糊过去——两边的数看起来都像个合理的百分比。

**默认扫四档而不是只跑默认档。** 语料全量那一档（n=200）恰好是**唯一一个全局 W
还活着的点**，只报它等于只报了"两边都很好"，而分簇的全部意义在它右边那几档。四档
就是 P4a 悬崖表的那四个结点。

**进不进门槛：不进。** 本仓的政策是**只有默认路径进门槛**（CORPUS.md 记着这条），
而巩固**没有默认路径**——它不参与召回，没有"用户实际走到的那条路"可言。所以这里
只报告：``baseline.json`` 一个字节都不动，退出码里也**没有 1**（1 的含义是"比基线
差了"，而这里没有基线）。

用法::

    python tools/bench_consolidation.py                       # 四档并排表
    python tools/bench_consolidation.py --scale 200,500,2000  # 自己挑档
    python tools/bench_consolidation.py --drop 0.3            # 探针不那么残
    python tools/bench_consolidation.py --json                # 机器可读

退出码：0 = 报告完了，2 = 用法或环境不对。**没有 1、也没有 3**——这个脚本不裁决
任何东西，它只把几个数摆在一起；裁决写在 ``benchmarks/substrate/CORPUS.md`` 里。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, NoReturn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from biosnn_memterm.bench.substrate import (  # noqa: E402
    SubstrateCorpus,
    SubstrateMemory,
    build_graph,
    coded_memories,
    completion_outcomes,
    completion_summary,
    default_substrate_dir,
    load_substrate_corpus,
    probe_sample,
    scaled_memories,
    substrate_digests,
)
from memcore.consolidation import (  # noqa: E402
    DEFAULT_DROP,
    MAX_CLUSTER,
    MIN_CLUSTER,
    SLEEP_CONFUSION,
    consolidate,
)
from memcore.sdr import ASSOC_K, encode_lexical  # noqa: E402
from memcore.types import MemoryItem, MemoryKind, Tier  # noqa: E402

#: 默认规模阶梯里除语料全量之外的那几个结点。**它们就是 P4a 悬崖表上的点**——
#: 拿同一批 n 值再量一遍，两个数才摆得到同一张表上。
_DEFAULT_LADDER = (500, 1000, 2000)

#: 语料全量那一档的**探针条数**。与 ``tools/bench_substrate.py --probes`` 的默认档
#: 一致，前两列因此与基线记的是同一个数（见 :func:`_cross_check_baseline`）。
_PROBES = 200

#: ``consolidate`` 要一个 ``at``。**固定值。** ``limit=None`` 时
#: :func:`~memcore.consolidation._round_order` 直接返回规范序，而 ``at`` 只参与
#: ``run_id`` / ``run_seed`` 的派生——那两个在这个脚本里一个数都不影响。给个能重复
#: 跑出同一份报告的值就够了。
_BENCH_AT = 1_700_000_000.0

#: 判"这两条本来就几乎一样"的那条线。**只是报告的一个分档**，不参与任何判定：
#: 串扰的 Jaccard 落在 0.9 以上，说明 top-1 认错的那条与被回放的那条在关联码上
#: 几乎重合。
_NEAR_DUPLICATE_J = 0.9

_BASELINE_NAME = "baseline.json"

#: 与基线对表时的容差。**不是**基线文件里那一个——理由见
#: :func:`_cross_check_baseline`：这里比的是同一个测量，差一点点就是口径变了。
#: 留这 1e-6 是给"换了一版 numpy 让某个并列的打破顺序变了"这类情况一点余量，而那种
#: 情况下数字会真变（一档 200 条探针里差一条就是 0.005），照样该报。
_CROSS_CHECK_TOL = 1e-6


# ── 用法与环境 ──────────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """用法或环境不对。**退出码 2**——这个脚本没有"指标跌了"那一档（1）。"""
    print(f"\n{message}", file=sys.stderr)
    raise SystemExit(2)


def _int_list(raw: str, flag: str) -> list[int]:
    """``"200,500"`` → ``[200, 500]``。空串给空列表（调用方决定默认值）。"""
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


def _items_of(memories: tuple[SubstrateMemory, ...]) -> tuple[MemoryItem, ...]:
    """把语料行变成 :class:`~memcore.types.MemoryItem`——``consolidate`` 吃的形状。

    **只填 ``lex_sdr``，``assoc_sdr`` 留 ``None``。** 这不是省事：
    :func:`~memcore.consolidation.replay_cluster` 会**显式**现算 ``weight`` 档的
    关联码，而 :meth:`~memcore.substrate.Substrate.build` 优先吃
    ``item.assoc_sdr``——预先塞一份 ``hash`` 档的码进去，``mode="weight"`` 会被
    **无声忽略**，而结果看起来仍然像个合理的辨认率。留 ``None`` 是让那条路径根本
    没有入口。

    ``tier`` 取 ``RECENT`` 而**不能取 ``COLD``**：冷掉的记忆会被
    :func:`~memcore.consolidation._skip_reason` 整批跳过，辨认率的分母随即空掉——
    而 ``recognition_rate`` 那时返回 ``None``（不是 ``0.0``），表上是一格 ``n/a``，
    看起来像"还没测"，不像"量错了"。

    ``created_at`` 显式给一个常量，让条目本身也是确定的；这个字段不进任何指标，
    但"同一份语料跑两次是同一批对象"是这个脚本的立身之本。
    """
    return tuple(
        MemoryItem(
            memory_id=memory.memory_id,
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content=memory.content,
            created_at=_BENCH_AT,
            lex_sdr=encode_lexical(memory.content),
        )
        for memory in memories
    )


def _measure(
    corpus: SubstrateCorpus,
    scale: int,
    *,
    probes: int,
    drop: float,
    max_cluster: int,
) -> dict[str, Any]:
    """在 ``scale`` 这一档上把这一行要的数全量出来。**返回纯数据**（写得进 JSON）。

    每一步都从头做（重新编码、重新建图），**不做跨档复用**：不同档的条目本来就不是
    同一批（:func:`~biosnn_memterm.bench.substrate.scaled_memories` 拼出来的那些与
    原始条目无关），而复用会让"这一档的数"与"上一档的数"共享一部分状态。

    图**与 ``bench_substrate.py`` 的默认档同一档**（``assoc=hash``、``k=ASSOC_K``、
    ``top_m=8``、``min_weight=0.10``）——前两列因此就是那份基线记录的数，而不是
    "另建了一张图上的数"。见 :func:`_cross_check_baseline`。
    """
    memories = scaled_memories(corpus.memories, scale)
    items = _items_of(memories)
    # 同一份 hash 档的码喂给图与第一列（`coded_memories` 的 docstring 就是为这件事
    # 写的）。weight 档的那一份另算——它是 W 的档，与图无关。
    coded_hash = coded_memories(memories, k=ASSOC_K, mode="hash")
    graph = build_graph(coded_hash, k=ASSOC_K, mode="hash")

    sample = probe_sample(memories, probes)
    probe_ids = [memory.memory_id for memory in sample]
    global_hash = completion_summary(completion_outcomes(coded_hash, probe_ids, drop=drop, top=5))[
        "completion_top1"
    ]
    global_weight = completion_summary(
        completion_outcomes(
            coded_memories(memories, k=ASSOC_K, mode="weight"), probe_ids, drop=drop, top=5
        )
    )["completion_top1"]

    report = consolidate(graph, items, at=_BENCH_AT, drop=drop, max_cluster=max_cluster)
    sizes = sorted((len(cluster) for cluster in report.clusters), reverse=True)
    confusions = [
        outcome.jaccard
        for outcome in report.outcomes
        if outcome.decision == SLEEP_CONFUSION and outcome.jaccard is not None
    ]
    return {
        "scale": scale,
        "memories": len(memories),
        "nodes": len(graph),
        "edges": graph.edge_count(),
        "probes": len(probe_ids),
        "global_hash": global_hash,
        "global_weight": global_weight,
        "recognition_rate": report.recognition_rate,
        "replayed": report.replayed,
        "reinforced": len(report.reinforced),
        "clusters": len(report.clusters),
        "largest_cluster": sizes[0] if sizes else 0,
        "median_cluster": statistics.median(sizes) if sizes else 0,
        "cluster_sizes": sizes,
        "confusions": len(confusions),
        "confusion_jaccard_median": statistics.median(confusions) if confusions else None,
        "confusion_jaccard_over_line": (
            sum(1 for value in confusions if value >= _NEAR_DUPLICATE_J) / len(confusions)
            if confusions
            else None
        ),
        "skip_reasons": report.skip_reasons,
        "missing_from_store": report.missing_from_store,
    }


# ── 报告 ────────────────────────────────────────────────────────────────────


def _fmt(value: float | None) -> str:
    """``None`` 与 ``nan`` 都显示成 ``n/a``。

    **它们不是 0。** ``recognition_rate`` 在一轮什么都没真回放时返回 ``None``——
    "没有数据"与"一条都没认出来"是两件完全不同的事；``completion_top1`` 在探针为
    空时是 ``nan``。把两者印成 ``0.0000`` 会让人去调参数修一个不存在的问题。
    """
    if value is None or value != value:
        return "n/a"
    return f"{value:.4f}"


def _pct(value: float | None) -> str:
    """占比。与 :func:`_fmt` 同一姿态：``None`` 是 ``n/a``，不是 ``0.0%``。"""
    return "n/a" if value is None else f"{value:.1%}"


def _cell(text: str, width: int) -> str:
    """按**显示宽度**右对齐补齐。中文占两列，而 ``str.rjust`` 只数一个。

    这个脚本的全部产出就是一张表，表头全是中文。不这样，表头与数字各按各的规则
    对齐，读者看到的是"这一列怎么歪了"——然后开始怀疑数字本身。
    """
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return " " * max(0, width - shown) + text


def _report(corpus: SubstrateCorpus, rows: list[dict[str, Any]], *, drop: float) -> None:
    print(
        f"记忆 {len(corpus.memories)} 条，语料自带的簇标签 {len(corpus.clusters)} 个"
        f"（每簇 {sorted(set(corpus.clusters.values()))} 条）"
        f"；探针每档等距抽 {rows[0]['probes']} 条"
    )
    print(
        f"图：assoc=hash k={ASSOC_K} top_m=8 min_weight=0.10（与 bench_substrate.py "
        f"的默认档同一档）；探针丢位 {drop:.2f}；簇上限 {MAX_CLUSTER}"
    )

    print("\n── 并排：全局一个 W vs 分簇局部 W ──")
    widths = (6, 13, 15, 15, 6, 8, 8, 11, 11)
    head = (
        "规模",
        "全局W(hash)",
        "全局W(weight)",
        "分簇W(weight)",
        "簇数",
        "最大簇",
        "回放",
        "认错J中位",
        "认错J≥线",
    )
    print("  " + "".join(_cell(name, width) for name, width in zip(head, widths, strict=True)))
    for row in rows:
        cells = (
            str(row["scale"]),
            _fmt(row["global_hash"]),
            _fmt(row["global_weight"]),
            _fmt(row["recognition_rate"]),
            str(row["clusters"]),
            str(row["largest_cluster"]),
            str(row["replayed"]),
            _fmt(row["confusion_jaccard_median"]),
            _pct(row["confusion_jaccard_over_line"]),
        )
        print("  " + "".join(_cell(cell, width) for cell, width in zip(cells, widths, strict=True)))

    print("\n── 逐档明细 ──")
    for row in rows:
        skips = "、".join(f"{name} {count}" for name, count in sorted(row["skip_reasons"].items()))
        # 百分号先算成字符串再拼：嵌在 f-string 里要三层引号，而嵌套同类引号是
        # Python 3.12 才放开的，本仓的 requires-python 是 >=3.10。
        near_text = _pct(row["confusion_jaccard_over_line"])
        print(
            f"  {row['scale']}：图 {row['nodes']} 节点 / {row['edges']} 边；"
            f"簇 {row['clusters']} 个（最大 {row['largest_cluster']}、"
            f"中位 {row['median_cluster']:g}）"
        )
        print(
            f"        辨认率 {_fmt(row['recognition_rate'])} = "
            f"{row['reinforced']}/{row['replayed']}；认错 {row['confusions']} 条"
            f"（J 中位 {_fmt(row['confusion_jaccard_median'])}，"
            f"≥{_NEAR_DUPLICATE_J:g} 的占 {near_text}）"
            f"；skip {skips or '无'}；图里有、库里没有 {row['missing_from_store']} 条"
        )

    print(
        "\n── 怎么读这张表 ──\n"
        "  三列问的是同一句话：丢位探针经**一步补全**之后，top-1 是不是它自己。\n"
        "  前两列是同一个**全局** W 换两条关联码（hash 是图 A 的那一档，weight 是 W\n"
        "  的那一档）；第三列把 W 换成「只装它自己那一簇的成员」，其余不变。\n"
        "  前两列的分母是等距抽的探针条数（跨规模可比）；第三列覆盖该档的**全部**\n"
        "  记忆，所以它的分母随规模涨。两种分母都在表上印着。\n"
        "  「认错 J」是认错时 top-1 与被回放那条之间的关联码 Jaccard——**认错得有多\n"
        "  远**。接近 1 表示那两条本来就几乎一样：这时辨认率量的是语料的重复度，不是\n"
        "  W 的容量。这一列是给辨认率配的，别单独看它。\n"
        "  **这一节不进门槛**：巩固没有默认路径（它不参与召回），所以只报告。"
    )
    print(
        "\n  （只报告；退出码 1 在本脚本里永不出现，因为这里没有基线可比。"
        "真实的库上那一轮由 tools/sleep_once.py 输出。）"
    )


def _cross_check_baseline(corpus_dir: Path, rows: list[dict[str, Any]]) -> None:
    """拿冻结在案的 ``completion_top1`` 校第一列。**只警告，不改退出码。**

    两列理应是**同一个数**：基线那一行是 ``assoc=hash`` 的默认档，而这张表的第一列
    也是同一档、同一份语料、同一批探针、同一张图。对不上只有一个可能——**口径变了**
    （丢位算子、探针抽法、图参数任一），而"口径变了"这件事在两张表里看起来都完全
    正常，甚至在两张表里都更好看。这是这个脚本唯一一处能自己发现自己量错了的地方。

    基线文件缺、或表里没有它记的那一档，就安静地跳过——那不是错误，只是这一次没
    什么可对照的。**不读它的 ``metrics`` 之外的任何东西**：这份文件的 schema 归
    ``bench_substrate.py`` 管，这里只是借一个数来对表。

    **容差用自己的，不借基线文件里的那一个。** 基线记的 ``tolerance``（0.05）是给
    "基质确实变了"用的——那是一个**行为的**判据，比它小的一律放过。而这里比的是
    **同一个测量**：同一份语料、同一批探针、同一个算子，唯一的差别是有没有重新算
    一遍，所以**差一点点就说明口径变了**。借那个容差来比，这个守卫会形同虚设：
    实测过，``--drop 0.3`` 让第一列从 0.0250 掉到 0.0000，而 0.0250 < 0.05 照样
    不响——一个只会说"没事"的守卫比没有守卫更糟。
    """
    path = corpus_dir / _BASELINE_NAME
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = (payload.get("metrics") or {}).get("completion_top1")
    scale = (payload.get("config") or {}).get("scale")
    row = next((one for one in rows if one["scale"] == scale), None)
    if frozen is None or row is None:
        return
    live = row["global_hash"]
    if live != live:  # nan：这一档没量出东西，没什么可对的
        return
    delta = abs(live - frozen)
    if delta > _CROSS_CHECK_TOL:
        print(
            f"⚠️ 第一列与基线里的 completion_top1 对不上：基线 {frozen:.4f}、"
            f"本轮 {live:.4f}（差 {delta:.4f}，容差 {_CROSS_CHECK_TOL:g}）。两者按定义"
            f"是**同一个测量**，对不上说明口径变了——先看丢位算子与探针抽法，再看图"
            f"参数。（这一档是 {scale}；基线记的容差 "
            f"{(payload.get('tolerance') or {}).get('completion_top1', 0.05)} 是给「基质"
            f"真的变了」用的，比它小的一律放过，不能拿来比这一个数。）",
            file=sys.stderr,
        )


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", default="", help="基质语料目录（默认自动定位）")
    parser.add_argument(
        "--scale",
        default="",
        help=f"语料规模，逗号分隔（默认 = 语料条数 + {','.join(map(str, _DEFAULT_LADDER))}）",
    )
    parser.add_argument("--probes", type=int, default=_PROBES, help="抽多少条探针（跨规模可比）")
    parser.add_argument("--drop", type=float, default=DEFAULT_DROP, help="探针丢掉多少比例的活跃位")
    parser.add_argument("--max-cluster", type=int, default=MAX_CLUSTER, help="簇规模上限")
    parser.add_argument("--json", action="store_true", help="额外输出一份机器可读的报告")
    parser.add_argument("--quiet", action="store_true", help="只留每一档那一行")
    args = parser.parse_args(argv)

    # ``--json`` 时把"人看的"那一路整体挪去 stderr，stdout 只留一份 JSON。理由与
    # bench_substrate.py 一字不差：不这样，`> report.json` 出来的东西谁都解析不了。
    human = sys.stderr if args.json else sys.stdout
    try:
        with contextlib.redirect_stdout(human):
            payload = _run(args)
    except FileNotFoundError as exc:
        # 缺语料、路径打错，都是**环境**问题。这个脚本没有"指标跌了"那一档，所以
        # 用一个与 0 不同的码把"环境不对"与"报告完了"分开。
        _die(str(exc))

    if args.json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else default_substrate_dir()
    corpus = load_substrate_corpus(corpus_dir)

    if not 0.0 < args.drop <= 1.0:
        _die(
            f"--drop 是「丢掉多少比例的活跃位」，必须落在 (0, 1]，收到 {args.drop!r}。"
            f"**0 不许**：探针就是目标本身时补全恒为命中，那会量出一个假的 100%。"
        )
    if args.probes < 1:
        _die(f"--probes 必须 >= 1，收到 {args.probes!r}。")
    if args.max_cluster < MIN_CLUSTER:
        _die(
            f"--max-cluster 必须 >= {MIN_CLUSTER}（簇规模下限），收到 {args.max_cluster!r}"
            f"——比下限还小的组会被 :func:`~memcore.consolidation.partition` 整个丢掉，"
            f"那一档会安静地什么都不回放。"
        )

    scales = _int_list(args.scale, "--scale") or sorted({len(corpus.memories), *_DEFAULT_LADDER})
    if any(scale < 1 for scale in scales):
        _die(f"--scale 是语料条数，必须 >= 1，收到 {scales}。")
    if not args.quiet:
        # 只印这一行：下面几档里最大的一档要跑几秒，而"什么都没印"与"卡住了"
        # 在终端上长得一样。其余的表头归 :func:`_report`。
        print(f"语料 {corpus_dir}")

    rows: list[dict[str, Any]] = []
    for scale in scales:
        row = _measure(
            corpus, scale, probes=args.probes, drop=args.drop, max_cluster=args.max_cluster
        )
        rows.append(row)
        if args.quiet:
            print(
                f"  {scale}  全局W(hash) {_fmt(row['global_hash'])}  "
                f"全局W(weight) {_fmt(row['global_weight'])}  "
                f"分簇W(weight) {_fmt(row['recognition_rate'])}"
            )

    if not args.quiet:
        _report(corpus, rows, drop=args.drop)

    _cross_check_baseline(corpus_dir, rows)

    return {
        "corpus": substrate_digests(corpus_dir),
        "config": {
            "assoc": "hash",
            "k": ASSOC_K,
            "top_m": 8,
            "min_weight": 0.10,
            "drop": args.drop,
            "probes": args.probes,
            "max_cluster": args.max_cluster,
            "gated": False,
        },
        "rows": rows,
    }


def _jsonable(value: object) -> object:
    """把一份报告变成 ``json.dumps`` 认得的形状。

    只做一件事：``nan`` → ``null``。分母为零的指标是 ``nan``（``None`` 已经是
    ``None``），而它进的是可能被存下来的 artifact——``NaN`` 不是合法 JSON 字面量，
    别的工具读不回来。
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
