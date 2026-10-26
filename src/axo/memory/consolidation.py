"""巩固的**入口**：真跑一轮时，改库这件事在这里发生。

上一层 :mod:`memcore.consolidation` 是纯的——分簇、回放、汇总，不碰库也不碰盘。
这一层是脏的，它按固定次序做四件事，**次序本身就是要紧的东西**：

1. **拍快照**（``snapshot_to``）——只在"本轮确实要改库"时拍，且在第一次写之前。
   快照失败就**不继续改库**：宁可不巩固，也不要一次没退路的批量修改。这是本版
   唯一一处无人值守的批量改库，它得有退路。拍完顺手轮转（:func:`rotate_snapshots`，
   留最近 :data:`SNAPSHOT_KEEP` 份）。
2. **逐簇「改库 + 追加事件」**——两件事在同一次循环迭代里完成，所以中途挂掉留下的
   是一个**对得上账的前缀**：库改了前 k 簇，事件也就只记了前 k 簇。整轮改完再落
   事件的话，崩在中途会得到"库改了、事件没有"的状态，而事件存在的全部意义就是
   能回答"这轮改了哪些"。
3. **写 ``last.json``**——调度器靠它回答"24 小时到了没有"，热度面板靠它说"最近
   一次巩固是什么时候"。它**不是证据**，是给下一次跑用的记录，所以它坏了不算事故
   （见 :func:`read_last_run`）。**空轮也要写**：它同时是调度器的钟，不写的话那一轮
   的 24 小时从来没有开始过，而调度器会每分钟问一次"到了没有"。
4. 返回一个 :class:`ConsolidationRun`，把"报告 + 落到哪 + 改了多少"一次交出去。

**动手之前先看事件文件在不在**：``run_id`` 由 ``(base_seed, at)`` 派生，同名文件已经在
盘上就说明这一轮跑过，此时**拒绝**（``ValueError``）而不是追加——追加会把两轮混成一份
``--replay`` 永远对不上的记录，然后报一串假的"这是 bug"。这道门在拍快照之前，理由见
:func:`run_consolidation`。

**这一层只有一份。** CLI（``tools/sleep_once.py``）与界面上的定时/手动两条路径都
走 :func:`run_consolidation`——改库的次序一旦有第二份实现，两份就会在某个 bug
修复里分岔，而分岔的那一份是"看起来完全正常"的那种。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memcore.consolidation import (
    BASE_SEED,
    DEFAULT_DROP,
    DEMOTE,
    MAX_CLUSTER,
    MERGE,
    PROMOTE,
    REINFORCE,
    W_MODE,
    RunReport,
    append_events,
    cluster_event_rows,
    consolidate,
)
from memcore.graph import GraphIndex
from memcore.ids import new_ulid
from memcore.merge import FragmentGroup, group_fragments, render_group
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.tiers import TierMove, plan_tier_moves
from memcore.types import MemoryItem, MemoryKind

from ..paths import consolidation_events_path, consolidation_last_path, snapshot_dir

__all__ = [
    "SNAPSHOT_KEEP",
    "ConsolidationRun",
    "last_run",
    "newest_memory_id",
    "read_last_run",
    "rotate_snapshots",
    "run_consolidation",
    "snapshot_name",
    "summary_line",
    "write_last_run",
]

SNAPSHOT_KEEP = 5
"""留几份快照。**不是 3**——``paths.memory_db_path`` 的老注释曾写「保留最近 3 份」，
那是 P4c 之前留的占位；定在 5 是因为巩固**每天一次**：5 份给的是"发现不对之后还
有几天可退"，而整库复制在几百 MB 量级上，5 份的磁盘代价可以接受。周期被调短的
话这条要跟着重算（写磁盘的开销与周期成正比）。"""

_SNAPSHOT_GLOB = "mem-*.db"
"""轮转只认这个形状。**不是 ``*.db``**——那个目录将来可能住进别的东西（手工挪进来
的一份备份、另一套工具的产物），而轮转的全部动作是 ``unlink``。删错一个文件是不可
撤销的，所以宁可不认。"""


def rotate_snapshots(keep: int = SNAPSHOT_KEEP) -> tuple[Path, ...]:
    """把快照留到最近 ``keep`` 份，返回删掉的那些（按删除序）。

    **按文件名排序，不按 mtime。** 名字里就写着这一轮的时刻（UTC、定宽、字典序即
    时间序），而 mtime 是"这个文件最后一次被写"——把快照目录从备份里拷回来之后，
    每一份的 mtime 都变成"刚才"，按它排序等于**随机**删，第一个被删的可能正是最新的
    那一份。名字是那一轮自己留下的，拷来拷去都不变。

    Raises:
        ValueError: ``keep < 1``。不设这道门的话 ``keep=0`` 会安静地**一份都不删**
            （空切片），于是"我配了不保留快照"与"快照一直在攒"看起来一模一样。
    """
    if keep < 1:
        raise ValueError(f"keep 至少是 1，收到 {keep!r}——0 份快照等于没有退路。")
    snapshots = sorted(snapshot_dir().glob(_SNAPSHOT_GLOB))
    doomed = snapshots[: max(0, len(snapshots) - keep)]
    for path in doomed:
        path.unlink()
    return tuple(doomed)


def newest_memory_id(store: MemoryStore) -> str:
    """库里最新那条的 id；**空库是空串**。

    "最新"按 ``memory_id`` 排——它是 ULID，单调递增，所以字典序即时间序（这条是
    ``write_last_run`` 里那个 ``newest_id`` 能成立的前提，见它的文档）。

    **借 ``summaries(limit=1)``，不自己 ``max(iter_items())``。** 后者要把全库每条
    记忆的 SDR 反序列化出来（5 万条约几十秒、上百 MB 的临时对象），而这里只问一个
    字符串；``summaries`` 的文档明说它按 ``memory_id`` 降序、走主键索引，所以
    "降序第一条"就是要的那一条。它那一句"调用方要检查是否命中上限"在这里不适用：
    要的就是一条，空表返回空串。

    **这个方法存在，是为了让"有没有新东西"这个问题有两个问法一致的答案**：巩固
    入口写 ``last.json`` 时用它，调度器那条路径让掉一轮时也用它。两边各写一遍
    ``max(...)`` 的话，某一天其中一边换了排法，症状是"巩固要么天天跑、要么再也不
    跑"，而两种都不报错。
    """
    rows = store.summaries(limit=1)
    return rows[0].memory_id if rows else ""


def snapshot_name(at: float) -> str:
    """快照文件名 ``mem-<UTC yyyymmddTHHMMSS>.db``。

    **UTC 而不是本地时间**：文件名是给人看"哪一份更早"的，而按 mtime 轮转、
    按名字读，三处必须用同一个时区；混用会在夏令时切换那天排序错乱。用 UTC 是因为
    它没有那个切换。
    """
    stamp = datetime.fromtimestamp(at, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"mem-{stamp}.db"


@dataclass(frozen=True)
class ConsolidationRun:
    """一轮真跑的产出。"""

    report: RunReport
    """纯计算的那一份。辨认率、跳过原因、``reinforced`` 都从它读。"""

    events_path: Path
    events_written: int
    """这一轮落盘的事件行数。它应当等于 ``len(report.outcomes)``。"""

    reinforced: int
    """真的 ``+1`` 了的条数，= ``len(report.reinforced)``。"""

    moved: tuple[TierMove, ...]
    """这一轮升降了哪些层。**算「记账」，不是「改记忆」**——只动 ``tier`` 一列。"""

    merged: tuple[MemoryItem, ...]
    """归并出来的派生条目。它们是**新增的行**，不改任何既有行——见
    :func:`write_merges`。"""

    snapshot: Path | None
    """拍下来的快照；``None`` 表示这一轮没拍（没东西要改，或调用方关掉了）。"""


def run_consolidation(
    store: MemoryStore,
    graph: GraphIndex,
    *,
    at: float,
    base_seed: int = BASE_SEED,
    drop: float = DEFAULT_DROP,
    max_cluster: int = MAX_CLUSTER,
    limit: int | None = None,
    snapshot: bool = True,
    code_mode: str = W_MODE,
) -> ConsolidationRun:
    """跑一轮并**改库**。``--dry-run`` 走的是 :func:`~memcore.consolidation.consolidate`
    本身，不经过这里。

    Args:
        store: 打开的库。这一层不负责开关它——调用方（CLI / worker）各自知道自己的
            生命周期。
        graph: 关联图。**分簇的前提**，没有图就没有这一轮（调用方决定怎么报这件事）。
        at: 这一轮的时间。``run_id`` / ``run_seed`` 与快照文件名都由它派生。
        snapshot: 拍不拍快照，默认拍。``False`` 是给测试用的——生产路径上没有理由
            关掉它。
        code_mode: 局部 W 的码档，默认 :data:`~memcore.consolidation.W_MODE`
            （``"idf"``）。``"weight"`` 是历史档，给 A/B 与旧事件回放用；见
            :func:`~memcore.consolidation.replay_cluster`。

    Returns:
        :class:`ConsolidationRun`。

    Raises:
        ValueError: 这一轮的 ``run_id`` 已经用过（同名事件文件在盘上）。**在拍快照
            之前就拒**——晚一步会留下一份保护着"没有发生的那次修改"的快照。
    """
    items = list(store.iter_items())
    report = consolidate(
        graph,
        items,
        at=at,
        base_seed=base_seed,
        drop=drop,
        max_cluster=max_cluster,
        limit=limit,
        code_mode=code_mode,
    )

    # 先看事件文件在不在，**在拍快照之前**：这一轮的 run_id 已经在盘上就说明它跑过，
    # 此时拒绝。拒绝要赶在任何副作用之前——晚一步就会留下一份"被拒了这一轮"的快照，
    # 而快照是几百 MB 的写。
    events_path = consolidation_events_path(report.run_id)
    if events_path.exists():
        raise ValueError(
            f"这一轮（{report.run_id}）已经跑过：{events_path} 在盘上。\n"
            f"事件文件按 run_id 命名，而 run_id 由 (base_seed, at) 派生——同一个 at 跑第二遍"
            f"会追加进第一遍那个文件，得到一份两轮混在一起、--replay 永远对不上的记录，"
            f"然后报一串「这是 bug」。所以这里拒绝，而不是覆盖或追加。"
        )

    # 归并的**规划**排在快照之前，**写**排在快照之后——见 plan_merges 的注释。
    planned_merges = plan_merges(items)
    # 升降层同理：先算，快照之后才写。``reinforced`` 传进去是为了让「这一轮刚被认出」
    # 一次就能凑够第二次（SQL 层的自增还没发生）。
    planned_moves = plan_tier_moves(items, now=at, reinforced=report.reinforced)

    snapshot_path: Path | None = None
    if snapshot and (report.reinforced or planned_merges or planned_moves):
        # 空转的一轮（一条都没加固）不拍：整库复制是几百 MB 的写，而它保护的那次
        # 修改根本不存在。**判断用的是 `reinforced` 而不是"跑完没有"**——只有
        # 认出自己才会写库，串扰与跳过都只进事件。
        snapshot_path = store.snapshot_to(snapshot_dir() / snapshot_name(at))
        # 轮转排在**写成功之后**：先删后写的话，写失败就白白少了一份退路。它又排在
        # 改库**之前**——万一 unlink 失败（文件被占着），这一轮就停在这个位置：快照
        # 在、库没动、last.json 没写，调度器下次会当它没跑过再排一轮。每一处失败都
        # 落在"什么都没改"这一侧。
        rotate_snapshots()

    written = 0
    for index, cluster in enumerate(report.clusters):
        # 逐簇写。谓词与 `RunReport.reinforced` 是同一条（都是 `decision == REINFORCE`），
        # 这里拆开写只是为了按簇切片——见模块文档第 2 条。
        store.bump_reinforce(
            [outcome.memory_id for outcome in cluster if outcome.decision == REINFORCE]
        )
        written += append_events(cluster_event_rows(report, index), events_path)

    merges = write_merges(
        store,
        planned_merges,
        events_path=events_path,
        run_id=report.run_id,
        at=at,
        run_seed=report.run_seed,
        seq_from=written,
    )
    written += merges.events_written

    moves = write_tier_moves(
        store,
        planned_moves,
        events_path=events_path,
        run_id=report.run_id,
        at=at,
        run_seed=report.run_seed,
        seq_from=written,
    )
    written += moves

    # ``last.json`` 排在**所有写之后**，而且**空轮也写**（见模块文档第 3 条）。
    # ``newest_id`` 在这里现问一次库，不用上面那份 ``items``：那一份是这一轮**开始时**
    # 读的，而这一轮可能跑了几分钟——期间写进来的新记忆才是"下次该不该跑"要看的。
    write_last_run(report, count=len(report.reinforced), newest_id=newest_memory_id(store))

    return ConsolidationRun(
        report=report,
        events_path=events_path,
        events_written=written,
        reinforced=len(report.reinforced),
        merged=merges.created,
        moved=tuple(planned_moves),
        snapshot=snapshot_path,
    )


def plan_merges(items: Sequence[MemoryItem]) -> list[FragmentGroup]:
    """这一轮该并哪几组。**纯读不写。**

    拆成规划与写入两步，是为了让快照落在**任何写之前**：归并也是改库（虽然只增
    不改），而这一层的纪律是"没有退路的批量写不许发生"。

    已经并过的组不再并：跑第二遍会读到派生条目，而它带的 ``derived_from`` 正是
    上一遍那几条原句的 id。
    """
    covered = _covered_fragments(items)
    return [group for group in group_fragments(items) if frozenset(group.memory_ids) not in covered]


@dataclass(frozen=True)
class MergeResult:
    """归并这一遍的产出。"""

    created: tuple[MemoryItem, ...]
    events_written: int


def write_merges(
    store: MemoryStore,
    groups: Sequence[FragmentGroup],
    *,
    events_path: Path,
    run_id: str,
    at: float,
    run_seed: int,
    seq_from: int = 0,
) -> MergeResult:
    """把规划好的组写成派生条目 + 事件行。**只增不改**：只调 ``store.add``。

    逐组「写一条 + 记一行」放在同一次迭代里，理由与逐簇那条一样：中途挂掉留下的是
    一个对得上账的前缀。**绝不进 ``transform``**——那是改既有行的唯一入口，而用户的
    约束是「已有的记忆不允许被你改动」。
    """
    created: list[MemoryItem] = []
    written = 0
    for offset, group in enumerate(groups):
        derived = _derived_item(group)
        store.add(derived)
        created.append(derived)
        written += append_events(
            (
                {
                    "run_id": run_id,
                    "run_seed": run_seed,
                    "seq": seq_from + offset,
                    "memory_id": derived.memory_id,
                    "decision": MERGE,
                    "at": at,
                    "epoch": 0,
                    "jaccard": None,
                    "confused_with": None,
                    "detail": {
                        "derived_from": list(group.memory_ids),
                        "count": len(group),
                    },
                },
            ),
            events_path,
        )
    return MergeResult(created=tuple(created), events_written=written)


def write_tier_moves(
    store: MemoryStore,
    moves: Sequence[TierMove],
    *,
    events_path: Path,
    run_id: str,
    at: float,
    run_seed: int,
    seq_from: int = 0,
) -> int:
    """把升降写进库、并落事件行。返回写了多少条事件。

    **一次 ``executemany`` 改完**，然后按条落事件：层级是单列更新，拆成逐条写只会
    多付几百次事务的钱；而事件是证据，逐条落才有「崩在中间留下的是对得上账的前缀」
    那个性质。
    """
    if not moves:
        return 0
    store.set_tiers([(move.memory_id, move.to_tier) for move in moves])
    append_events(
        tuple(
            {
                "run_id": run_id,
                "run_seed": run_seed,
                "seq": seq_from + index,
                "memory_id": move.memory_id,
                "decision": PROMOTE if move.decision == "promote" else DEMOTE,
                "at": at,
                "epoch": 0,
                "jaccard": None,
                "confused_with": None,
                "detail": {"from": move.from_tier.value, "to": move.to_tier.value},
            }
            for index, move in enumerate(moves)
        ),
        events_path,
    )
    return len(moves)


def _covered_fragments(items: Sequence[MemoryItem]) -> set[frozenset[str]]:
    """已经并过的那几组原句 id。用来保证跑第二遍不重复造条目。"""
    covered: set[frozenset[str]] = set()
    for item in items:
        if not item.meta.get("derived"):
            continue
        raw = item.meta.get("derived_from")
        if isinstance(raw, (list, tuple)):
            covered.add(frozenset(str(each) for each in raw))
    return covered


def _derived_item(group: FragmentGroup) -> MemoryItem:
    """造一条派生条目。**原句一个字都不动**，所以它只是新的一行。

    ``kind`` 取原句共同的那一个，不一致就退到 FACT——派生条目不发明新的语义类别。
    ``tier`` 取最早那条的：它代表的就是那一场，不该比它的第一条活得更短。
    """
    rendered = render_group(group)
    kinds = {item.kind for item in group.items}
    return MemoryItem(
        memory_id=new_ulid(),
        kind=group.items[0].kind if len(kinds) == 1 else MemoryKind.FACT,
        tier=group.items[0].tier,
        content=rendered,
        taint=group.items[0].taint,
        created_at=group.start,
        lex_sdr=encode_lexical(rendered),
        session_id=group.session_id,
        meta={
            "derived": True,
            "derived_from": list(group.memory_ids),
            "span": [group.start, group.end],
        },
    )


def summary_line(report: RunReport) -> str:
    """一轮的**一句话**。CLI 的 ``--quiet``、界面的系统消息、托盘气泡都用它。

    同一件事会被读三遍（终端、对话区、气泡），三处各写一遍的结果是它们会在某次
    改动里分岔，然后用户看到两个说法。所以这里只有一份。

    收的是 :class:`~memcore.consolidation.RunReport` 而不是
    :class:`ConsolidationRun`：``--dry-run`` 只有前者，而"这一轮认出了多少"这句话
    在算完的那一刻就已经定了——它不该因为"写没写库"而说得不一样。

    **没有数据时不编数字**（``recognition_rate`` 是 ``None``）：写成"辨认率 0%"
    会把"没测"说成"没测中"，而这个区别正是 :attr:`RunReport.recognition_rate`
    返回 ``None`` 的理由。跳过数照报——它是这一轮真实发生了的事。
    """
    head = f"巩固 {report.run_id[:8]}："
    skipped = sum(report.skip_reasons.values())
    tail = f"，跳过 {skipped} 条" if skipped else ""
    gap = (
        f"，另有 {report.missing_from_graph} 条新记忆不在图里" if report.missing_from_graph else ""
    )
    rate = report.recognition_rate
    if rate is None:
        return f"{head}没有可回放的条目（跳过 {skipped} 条）{gap}"
    return (
        f"{head}回放 {report.replayed} 条，认出 {len(report.reinforced)} 条"
        f"（辨认率 {rate:.1%}）{tail}{gap}"
    )


def write_last_run(
    report: RunReport,
    *,
    count: int,
    newest_id: str,
    path: Path | None = None,
) -> Path:
    """写 ``last.json``：这一轮跑在什么时候、加固了多少条、当时最新的记忆是哪条。

    ``newest_id`` 是给调度器的"有没有新东西"用的：库里的 id 是 ULID（按时间
    排序），所以"最新那条还是上次那条"等价于"这期间没有新的记忆写进来"。自动那条
    路径据此可以让掉一轮——巩固的全部作用对象是**已有的**记忆，库里没新东西时
    重跑一遍只是把同一批条目再认一遍。

    ``indent=2``：这个文件小，而它是要给人看的（"最近一次巩固是什么时候"）。
    """
    target = path or consolidation_last_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "at": report.at,
        "count": count,
        "newest_id": newest_id,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return target


def read_last_run(path: Path | None = None) -> dict[str, Any] | None:
    """读 ``last.json``；**没有、坏了、字段不对，一律返回 ``None``**。

    这个文件不是证据，是给下一次跑用的记录：它丢了只会让调度器把下一轮当成
    "上次跑过"（多跑一轮无害），让热度面板说"还没跑过"（少一句而不假一句）。
    **所以坏数据不该把界面炸掉，也不该编一个数字出来**——这正是"没数据就说没数据"
    那条纪律：拿不到就返回 ``None``，由调用方说"不知道"，而不是在这里体贴地
    补一个零。
    """
    target = path or consolidation_last_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "run_id" not in payload or "at" not in payload:
        return None
    return payload


def last_run() -> tuple[float, int] | None:
    """上一轮的 ``(什么时候跑的, 加固了几条)``；**读不出来时 ``None``**。

    :func:`read_last_run` 保证的只是"整个对象读得出来"，而手改过的 json 可以让
    ``at`` 是 ``"昨天"``、``true``、``null``，甚至 ``Infinity``——最后那两个不是合法
    JSON 字面量，但 Python 的解析器照收，而 ``datetime.fromtimestamp`` 收不下。

    **它存在，是为了让"上一轮是什么时候跑的"只有一个答案。** 界面这一侧至少两处要
    说这句话（设置页那行说明、记忆窗口的热度面板），两处各写一遍那套守卫的话，某天
    其中一处忘了挡 ``true``——``bool`` 是 ``int`` 的子类，``float(True)`` 是
    ``1.0``——于是"最近一次"会显示成 1970 年，一个既不像错误也不像占位的年份。

    ``count`` 读不出来时给 ``0`` 而不是另开一个三态：调用方拿它决定"要不要说加固了
    几条"，而"这轮加固了零条"与"不知道加固了几条"在界面上说同一句话（都不说）。
    """
    payload = read_last_run()
    if payload is None:
        return None
    at = payload.get("at")
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
        return None
    count = payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int):
        count = 0
    return float(at), count
