"""睡眠巩固跑一轮：分簇 → 每簇局部 W 回放 → 改库 → 落事件。

**这个 ``--replay`` 与 P3a 的逐字稿回放不是一回事。** 那边说的是"把一段会话的消息
按顺序再喂一遍"（``agent/loop.py``、``session/assemble.py``）；这里的 ``--replay``
是 :class:`~memcore.types.ConsolidationEvent` 预先指定的那个意思：拿同一份
``run_seed`` 把**一轮巩固**重跑一遍，与落盘的事件逐条比对。旗标名沿用类型里写死的
那个，不改；两处同名靠上下文区分。

用法::

    python tools/sleep_once.py --dry-run             # 真库上跑一遍，什么都不写
    python tools/sleep_once.py                       # 真跑：拍快照 → 改库 → 落事件
    python tools/sleep_once.py --limit 500           # 本轮只看 500 条
    python tools/sleep_once.py --replay <run_id>     # 重跑并逐条比对
    python tools/sleep_once.py --replay <run_id> --db <快照副本>
    python tools/sleep_once.py --json

退出码：``0`` = 跑完（``--dry-run`` 也是 0；``--replay`` 则是"比对一致"），
``1`` = **v1 不产生**（巩固没有默认路径、没有基线门槛，这一档留给将来的
``--check``），``2`` = 用法或环境不对，``3`` = 结构不变量被违反——**是 bug**：
一轮巩固重跑出来不一样，那就推翻了 ``ConsolidationEvent`` 存在的理由。

三件"这个脚本不做什么"：

* **不建库、不建图。** 缺哪个都退 2 并说清怎么办。巩固是**读**已有记忆的夜间
  任务，不是初始化工具——在一台还没有记忆的机器上"跑一轮巩固"是在初始化，
  那件事该由应用做（它同时还要建空间、写人格锚）。
* **不轮转快照、不写 ``last.json``**（两件事都在巩固入口里，和"拍快照"挨着——
  写入的次序只有那一份实现，见 ``memory/consolidation.py`` 的模块文档）。
* **``--replay`` 不改任何东西**，也不拍快照。

``--replay`` 要带上**原轮的抽样参数**（``--limit`` 尤其）：事件行里只有结果，
没有"当时用了多少预算"。指错了会得到一串"第 k 条对不上"，而那不是 bug，是重跑的
输入与原轮不同——序列不一致退 3 的含义是"同一份输入给出了不同结果"。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NoReturn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from axo.memory.consolidation import (  # noqa: E402
    run_consolidation,
    summary_line,
)
from axo.paths import (  # noqa: E402
    consolidation_events_path,
    graph_path,
    memory_db_path,
    snapshot_dir,
)
from memcore.consolidation import (  # noqa: E402
    DEFAULT_DROP,
    MAX_CLUSTER,
    MIN_CLUSTER,
    REINFORCE,
    SKIP,
    SLEEP_CONFUSION,
    SLEEP_REPLAY_MISS,
    W_MODE,
    W_MODES,
    RunReport,
    compare_events,
    consolidate,
    event_rows,
    read_events,
)
from memcore.errors import MemCoreError  # noqa: E402
from memcore.graph import GraphIndex  # noqa: E402
from memcore.store import MemoryStore  # noqa: E402

_RUN_ID_CHARS = frozenset("0123456789abcdef")
_RUN_ID_LEN = 16

Payload = dict[str, Any]


# ── 用法与环境 ──────────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """用法或环境不对。**退出码 2 而不是 1**——1 留给"指标跌了"，而 v1 还没有
    任何一条会跌的指标。"""
    print(f"\n{message}", file=sys.stderr)
    raise SystemExit(2)


def _checked_run_id(raw: str) -> str:
    """``--replay`` 的 ``run_id`` 是**外部输入**，而它要拼进文件名。

    验形状而不是"清理"它：``../`` 这类值会让 ``consolidation_events_path`` 安静地
    跑出巩固目录，去读一个与巩固毫无关系的文件；而 ``run_id`` 的形状是**定死的**
    （:func:`~memcore.consolidation.run_id_for` 给出 16 位十六进制），所以拒掉一个
    不合形状的值不会误伤任何真实用法。
    """
    if len(raw) != _RUN_ID_LEN or any(char not in _RUN_ID_CHARS for char in raw):
        _die(
            f"--replay 要的是 run_id（{_RUN_ID_LEN} 位十六进制），收到 {raw!r}。\n"
            f"它是 run_id_for() 派生的，一轮一个；事件文件名就是 <run_id>.jsonl，"
            f"躺在 {consolidation_events_path('0' * _RUN_ID_LEN).parent} 下。"
        )
    return raw


def _load_graph(path: Path) -> GraphIndex:
    """图是分簇的**前提**，缺了或坏了这一轮就没有意义——退 2，不是退 1。

    与界面不同（那里盘上的图坏了要当"没有图"、直接重建，见 ``GraphWorker``）：
    这个脚本**不建图**。建图在 5 万条上要几分钟，把它藏在一个夜间任务的副作用里，
    用户下一次打开记忆窗口会发现自己盯着一个"正在建立关联图"。
    """
    if not path.exists():
        _die(
            f"找不到关联图 {path}。\n"
            f"巩固靠图分簇，没有图就没有这一轮。让应用建一张"
            f"（打开记忆窗口会自动建，设置页的「重建联想」可以强制重建）再回来。"
        )
    try:
        return GraphIndex.load(path)
    except (OSError, MemCoreError, ValueError) as exc:
        _die(f"{path} 读不出来：{type(exc).__name__}：{exc}\n图是派生缓存，重建一张就好。")
    raise AssertionError("unreachable")  # pragma: no cover — _die 必定抛出


def _open_store(db_path: Path, *, human: Any, stack: contextlib.ExitStack) -> MemoryStore:
    """打开库。**库必须已经存在**，缺了退 2。

    ``MemoryStore`` 打开不存在的路径会**建一个空库**，于是"路径打错"与"还没有
    记忆"会得到同一个结果（0 条、跑完、退 0），而这两种情况的正确处置完全不同。

    **快照不会被就地打开。** ``--replay --db <快照>`` 里那份快照是"改之前那一份
    完整的库"，而 SQLite 一打开就会往文件里写 WAL 头部、留下 ``-wal``/``-shm``
    边车——证据就不再是原来那一份了。所以来自
    :func:`~axo.paths.snapshot_dir` 的路径一律**先复制再打开**，
    那一行输出就是这件事的告知（不是悄悄帮忙：整库复制是几百 MB 的写，
    用户该知道它在发生）。
    """
    if not db_path.exists():
        _die(
            f"找不到记忆库 {db_path}。\n"
            f"**这个脚本不建库**——在一台还没有记忆的机器上跑一轮巩固没有意义。"
            f"先让应用跑起来（它会建库、建空间、写人格锚）。"
        )

    try:
        inside_snapshots = db_path.resolve().is_relative_to(snapshot_dir().resolve())
    except OSError:  # pragma: no cover — resolve 只在极端路径下失败
        inside_snapshots = False
    if inside_snapshots:
        holder = stack.enter_context(tempfile.TemporaryDirectory(prefix="biosnn-replay-"))
        copy = Path(holder) / db_path.name
        shutil.copy2(db_path, copy)
        print(f"快照按副本打开（原文件不动）：{copy}", file=human)
        db_path = copy

    return stack.enter_context(MemoryStore(db_path))


# ── 报告 ────────────────────────────────────────────────────────────────────


def _tally(report: RunReport) -> dict[str, int]:
    """按决策分类计数。三个名字**从常量取**，不在这拼字面量——拼错一个字母的后果
    是那一类永远数到 0，而报告看起来完全正常。"""
    counts = {REINFORCE: 0, SLEEP_CONFUSION: 0, SLEEP_REPLAY_MISS: 0}
    for outcome in report.outcomes:
        if outcome.decision in counts:
            counts[outcome.decision] += 1
    return counts


def _payload(report: RunReport, **extra: Any) -> Payload:
    counts = _tally(report)
    return {
        "run_id": report.run_id,
        "run_seed": report.run_seed,
        "at": report.at,
        "clusters": len(report.clusters),
        "replayed": report.replayed,
        "reinforced": len(report.reinforced),
        "confusion": counts[SLEEP_CONFUSION],
        "replay_miss": counts[SLEEP_REPLAY_MISS],
        "recognition_rate": report.recognition_rate,
        "skip_reasons": report.skip_reasons,
        "missing_from_store": report.missing_from_store,
        **extra,
    }


def _print_report(report: RunReport, *, human: Any, quiet: bool) -> None:
    """人看的那几行。**没数据就说没数据**：辨认率拿不到时不打一个 0。

    ``--quiet`` 时只留 :func:`~axo.memory.consolidation.summary_line`
    那一行——那句话在终端、对话区、托盘气泡三处被读，所以它只有一份实现。
    """
    if quiet:
        print(summary_line(report), file=human)
        return
    counts = _tally(report)
    print(f"巩固 {report.run_id}（种子 {report.run_seed}）", file=human)
    print(
        f"本轮 {len(report.clusters)} 簇 / 回放 {report.replayed} 条："
        f"认出 {counts[REINFORCE]}、串扰 {counts[SLEEP_CONFUSION]}、"
        f"没回放 {counts[SLEEP_REPLAY_MISS]}",
        file=human,
    )
    rate = report.recognition_rate
    if rate is None:
        print("辨认率 没有数据（一条都没回放到）", file=human)
    else:
        print(f"辨认率 {rate:.4f}", file=human)
    if report.skip_reasons:
        joined = "、".join(
            f"{reason} {count}" for reason, count in sorted(report.skip_reasons.items())
        )
        print(f"跳过 {joined}", file=human)
    if report.missing_from_store:
        print(f"图上多出 {report.missing_from_store} 条（库里没有）——图该重建了", file=human)


# ── 三条路径 ────────────────────────────────────────────────────────────────


def _dry_run(
    store: MemoryStore, graph: GraphIndex, args: argparse.Namespace, *, human: Any
) -> tuple[int, Payload]:
    """算完就走。**连库都不写**，所以它与真跑的差别就是"调不调那个入口"。"""
    code_mode = args.code_mode or W_MODE
    report = consolidate(
        graph,
        list(store.iter_items()),
        at=args.at,
        base_seed=args.seed,
        drop=args.drop,
        max_cluster=args.max_cluster,
        limit=args.limit,
        code_mode=code_mode,
    )
    _print_report(report, human=human, quiet=args.quiet)
    if not args.quiet:
        print("--dry-run：没有改库、没有拍快照、没有落事件。", file=human)
    return 0, _payload(
        report, dry_run=True, snapshot=None, events=None, events_written=0, code_mode=code_mode
    )


def _real_run(
    store: MemoryStore, graph: GraphIndex, args: argparse.Namespace, *, human: Any
) -> tuple[int, Payload]:
    code_mode = args.code_mode or W_MODE
    run = run_consolidation(
        store,
        graph,
        at=args.at,
        base_seed=args.seed,
        drop=args.drop,
        max_cluster=args.max_cluster,
        limit=args.limit,
        snapshot=not args.no_snapshot,
        code_mode=code_mode,
    )
    _print_report(run.report, human=human, quiet=args.quiet)
    if not args.quiet:
        if run.snapshot is not None:
            print(f"快照 {run.snapshot}", file=human)
        else:
            print("快照 无（这一轮没有要改的东西）", file=human)
        print(f"事件 {run.events_path}（{run.events_written} 行）", file=human)
        print(f"加固 {run.reinforced} 条", file=human)
    return 0, _payload(
        run.report,
        dry_run=False,
        snapshot=None if run.snapshot is None else str(run.snapshot),
        events=str(run.events_path),
        events_written=run.events_written,
        code_mode=code_mode,
    )


def _replay(
    store: MemoryStore, graph: GraphIndex, run_id: str, args: argparse.Namespace, *, human: Any
) -> tuple[int, Payload]:
    """重跑一轮并与落盘的事件逐条比对。**只读**：不拍快照、不改库、不落事件。"""
    try:
        recorded = read_events(consolidation_events_path(run_id))
    except MemCoreError as exc:
        _die(str(exc))
    if not recorded:
        _die(
            f"{consolidation_events_path(run_id)} 是空的。这个脚本写不出空文件"
            f"（空轮不落盘），所以它要么被截断过，要么不是这个脚本写的。"
        )

    # 回放只比对**回放决策**。同一份事件文件里还有 ``merge`` / ``promote`` /
    # ``demote`` 行——它们由 :func:`~axo.memory.consolidation.run_consolidation`
    # 在回放之后追加，``consolidate`` 不产生它们；拿它们去比对会让任何一轮有归并
    # 或升降层的巩固都报"记录里有、重跑没跑出来"（这是 2026-10-18 在 166 条真库上
    # 第一次跑真轮时撞到的假警报）。``MERGE`` 的 docstring 本来就写着它"不该出现在
    # 回放的比对里"；这里按决策名过滤，而不是假设它们一定排在末尾。
    replay_decisions = frozenset({REINFORCE, SLEEP_CONFUSION, SLEEP_REPLAY_MISS, SKIP})
    recorded_replay = tuple(row for row in recorded if row.get("decision") in replay_decisions)
    if not recorded_replay:
        _die(
            f"{consolidation_events_path(run_id)} 里没有回放决策行"
            f"（只有归并/升降层之类的附加事件），没有可比对的东西。"
        )

    # 用**记录里的**时刻重跑：run_seed 由 (base_seed, at) 派生，换一个时刻就换了
    # 抽样顺序，于是"复现"变成"用另一份输入跑一遍"——那样比出来的任何差异都说明
    # 不了问题。base_seed 从 --seed 来（默认就是原轮那个常量 0）。
    at = float(recorded[0]["at"])
    # 码档也取记录里的：W_MODE 从 weight 换成 idf 之后，旧事件仍要能用原来的档
    # 重跑。旧事件没有这个字段，默认 weight——那是它们当时用的档。
    recorded_mode = str(recorded[0].get("detail", {}).get("code_mode", "weight"))
    if recorded_mode not in W_MODES:
        _die(f"事件里记录的 code_mode 是 {recorded_mode!r}，不在 {W_MODES} 里。")
    if args.code_mode is not None and args.code_mode != recorded_mode:
        _die(
            f"--replay 必须用原轮记录的码档：这一轮记的是 {recorded_mode!r}，"
            f"命令行给的是 {args.code_mode!r}。换档会让重跑与记录不可比。"
        )
    report = consolidate(
        graph,
        list(store.iter_items()),
        at=at,
        base_seed=args.seed,
        drop=args.drop,
        max_cluster=args.max_cluster,
        limit=args.limit,
        code_mode=recorded_mode,
    )
    if report.run_id != run_id:
        _die(
            f"重跑出来的 run_id 是 {report.run_id}，与文件名 {run_id} 不同。\n"
            f"它由 blake2b(base_seed|sleep|<at>) 派生，而 <at> 取自事件文件"
            f"（{at}）——所以对不上的只可能是 --seed（这一轮给的是 {args.seed}）。"
        )

    problems = compare_events(recorded_replay, event_rows(report))
    if problems:
        print(f"{run_id} 重跑出来与落盘的记录不一致（{len(problems)} 处）：", file=human)
        for problem in problems:
            print(f"  {problem}", file=human)
        print(
            "\n**这是 bug，不是环境问题。** 同一轮巩固重跑出同一串事件是"
            "ConsolidationEvent 存在的全部理由；做不到就说明分组、抽样或回放里有一处"
            "不是确定性的。",
            file=human,
        )
        return 3, _payload(
            report,
            replay=True,
            rows=len(recorded_replay),
            rows_total=len(recorded),
            problems=list(problems),
            code_mode=recorded_mode,
        )

    print(summary_line(report), file=human)
    print(f"与 {run_id} 逐条一致（{len(recorded_replay)} 条）。", file=human)
    return 0, _payload(
        report,
        replay=True,
        rows=len(recorded_replay),
        rows_total=len(recorded),
        problems=[],
        code_mode=recorded_mode,
    )


def _run(args: argparse.Namespace, *, human: Any) -> tuple[int, Payload]:
    if not 0.0 <= args.drop <= 1.0:
        _die(f"--drop 是「丢掉多少比例的活跃位」，必须落在 [0, 1]，收到 {args.drop!r}。")
    if args.max_cluster < MIN_CLUSTER:
        _die(f"--max-cluster 必须 >= MIN_CLUSTER（{MIN_CLUSTER}），收到 {args.max_cluster!r}。")
    if args.limit is not None and args.limit < MIN_CLUSTER:
        _die(
            f"--limit 必须 >= MIN_CLUSTER（{MIN_CLUSTER}），收到 {args.limit!r}——"
            f"再小的预算装不下任何一个簇，这一轮会安静地什么都不做。"
        )
    if args.seed < 0:
        _die(f"--seed 是 base_seed，必须 >= 0，收到 {args.seed!r}。")
    if args.replay and args.dry_run:
        _die("--replay 本身就是只读的，--dry-run 没有意义。去掉一个。")
    # 形状也在这一批里验：**用法不对要在碰任何文件之前说完**。放到下面那个调用点上
    # 的话，一个写错的 run_id 会先去读图、开库，然后才被拒——而"图不在"那条错误会
    # 顶掉"run_id 写错了"这条更贴近原因的话。
    run_id = _checked_run_id(args.replay) if args.replay else ""

    graph = _load_graph(Path(args.graph) if args.graph else graph_path())
    db_path = Path(args.db) if args.db else memory_db_path()
    with contextlib.ExitStack() as stack:
        store = _open_store(db_path, human=human, stack=stack)
        if args.replay:
            return _replay(store, graph, run_id, args, human=human)
        if args.dry_run:
            return _dry_run(store, graph, args, human=human)
        return _real_run(store, graph, args, human=human)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="", help="记忆库（默认数据目录下的 mem.db）")
    parser.add_argument("--graph", default="", help="关联图缓存（默认数据目录下的 graph.npz）")
    parser.add_argument(
        "--limit", type=int, default=None, help=f"本轮最多看多少条记忆（下限 {MIN_CLUSTER}）"
    )
    parser.add_argument("--drop", type=float, default=DEFAULT_DROP, help="探针丢位比例")
    parser.add_argument("--max-cluster", type=int, default=MAX_CLUSTER, help="簇上限")
    parser.add_argument("--seed", type=int, default=0, help="base_seed（整轮可复现性的常量）")
    parser.add_argument(
        "--code-mode",
        default=None,
        choices=[*W_MODES],
        help=f"局部 W 的码档（默认 {W_MODE}；weight 是历史档，用于 A/B 与旧事件回放）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="跑完整流程但不改库、不拍快照、不落事件"
    )
    parser.add_argument("--replay", default="", help="重跑某一轮并与落盘的事件逐条比对")
    parser.add_argument("--no-snapshot", action="store_true", help="不拍快照（测试用）")
    parser.add_argument("--json", action="store_true", help="额外输出一份机器可读的报告")
    parser.add_argument("--quiet", action="store_true", help="只留一行结论")
    args = parser.parse_args(argv)
    # 整轮只看一次时间：快照名、run_id、run_seed 都由它派生，中途再取一次会出现
    # "报告里的时刻与快照文件名差一秒"这种没法解释的东西。
    args.at = time.time()

    # ``--json`` 时把"人看的"那一路整体挪去 stderr，stdout 只留一份 JSON。理由与
    # bench_substrate.py 一字不差：不这样，`> report.json` 出来的东西谁都解析不了。
    human = sys.stderr if args.json else sys.stdout
    try:
        code, payload = _run(args, human=human)
    except (MemCoreError, ValueError, OSError) as exc:
        # 兜底。**不是为了让错误好看，是为了别让它的退码说谎**：Python 未捕获异常退出
        # 码是 1，而这张表里 1 的意思是"指标跌了"——于是"库被人删了"会被读成"这一轮
        # 的指标没过线"。这一档收的是"用法或环境不对"，一律退 2，消息原样带出去。
        # 巩固入口拒绝重复轮次时抛的 ValueError 也走这里（见 memory/consolidation.py）。
        _die(f"{type(exc).__name__}：{exc}")
    if args.json:
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return code


def _jsonable(value: object) -> object:
    """把一份报告变成 ``json.dumps`` 认得的形状（``nan`` → ``null``，理由同
    ``tools/bench_substrate.py``：它可能进 artifact，别的工具读不回来）。"""
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
