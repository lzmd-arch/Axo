"""一轮真跑**在盘上留下什么**：快照、``last.json``、事件文件的既有性。

事件行**长什么样**（字段、``seq``、逐条比对）在 ``test_consolidation_events.py``；
分簇与回放的**判断**在 ``test_consolidation_partition.py`` / ``test_consolidation_replay.py``。
这个文件只管一件事：那一次真跑之后，文件系统上多了什么、没多什么，以及**在什么
情况下它拒绝动手**。

重心是**次序**。这一层是本版唯一一处"无人值守的批量改库"，它全部的退路就是快照
在写之前拍下来。次序错了代码照样跑完、报告照样好看——所以这里量的是快照里那条
``reinforce_count`` 是**改之前**的还是改之后的。
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from biosnn_memterm.memory import consolidation as entry
from biosnn_memterm.paths import (
    consolidation_last_path,
    memory_db_path,
    snapshot_dir,
)
from memcore.graph import GraphIndex
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

AT = 1_760_000_000.0

_TEXTS = [
    "我家的路由器放在书房书架顶层。",
    "今天午饭吃的是番茄炒蛋和米饭。",
    "每周三晚上八点有一节游泳课。",
]


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把数据目录指到 ``tmp_path``。**这个文件里每条用例都要它**，所以铺一个夹具。

    ``paths`` 那几个函数只认 ``BIOSNN_MEMTERM_DATA``，而巩固写的每一处位置（库、
    快照、事件、``last.json``）都从它派生——不铺这一层，用例会往真实数据目录里写。
    """
    monkeypatch.setenv("BIOSNN_MEMTERM_DATA", str(tmp_path))
    return tmp_path


def _items(**changes: object) -> list[MemoryItem]:
    items = [
        MemoryItem(
            memory_id=f"m{index:03d}",
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content=text,
            lex_sdr=encode_lexical(text),
        )
        for index, text in enumerate(_TEXTS)
    ]
    return [item.copy(**changes) for item in items] if changes else items


def _store(**changes: object) -> MemoryStore:
    """开一个库并写进三条记忆。**这条路径会覆盖已有的行**——要接着上一轮往下跑，
    用 :func:`_open`，别再写一遍。"""
    store = MemoryStore(memory_db_path())
    for item in _items(**changes):
        store.add(item)
    return store


def _open() -> MemoryStore:
    """开**已有**的那个库（不写任何东西）。"""
    return MemoryStore(memory_db_path())


def _one_cluster(size: int = len(_TEXTS)) -> GraphIndex:
    """一整个全连的团（与 ``test_consolidation_run.py`` 里那份同一个形状）。

    全连的小团在回放里**认得出自己**——这一轮的 ``reinforced`` 非空，于是快照那条
    分支才会走到。一份"跑完但没加固"的输入会让下面每条用例都退化成"什么都没发生"。
    """
    top_m = size - 1
    nbr_idx = np.full((size, top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, top_m), dtype=np.float32)
    for row in range(size):
        for slot, target in enumerate(index for index in range(size) if index != row):
            nbr_idx[row, slot] = target
            nbr_w[row, slot] = 1.0
    return GraphIndex(
        node_ids=tuple(f"m{index:03d}" for index in range(size)),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="d",
        built_at=0.0,
        top_m=top_m,
        min_weight=0.1,
    )


def _reinforce_counts(db: Path) -> dict[str, int]:
    """从**另一个连接**读一份库的计数，且不碰 ``MemoryStore``。

    被测的那一份可能正开着；这里要看的是盘上那份字节，不能顺带把它初始化一遍。
    """
    connection = sqlite3.connect(str(db))
    try:
        rows = connection.execute(
            "SELECT memory_id, reinforce_count FROM memories ORDER BY memory_id"
        ).fetchall()
    finally:
        connection.close()
    return dict(rows)


# ── 快照本身 ────────────────────────────────────────────────────────────────


def test_the_snapshot_is_a_readable_database_with_the_same_rows(data_dir: Path) -> None:
    """拍出来的是一份**能开的库**，行数与内容都与原库一致。

    这条挡的是"快照其实是个空壳"：``snapshot_to`` 写出去的文件如果是个 truncate 过
    的空文件，巩固照样跑完、报告照样好看，而真出事那天才发现手里什么都没有。
    """
    with _store() as store:
        snapshot = store.snapshot_to(data_dir / "snap.db")

    assert snapshot == data_dir / "snap.db"
    assert _reinforce_counts(snapshot) == {"m000": 0, "m001": 0, "m002": 0}


def test_the_snapshot_carries_what_is_still_only_in_the_wal(data_dir: Path) -> None:
    """**尚未检查点的东西也要在里面。** 这条是"为什么不用 ``shutil.copy``"的量具。

    库跑在 WAL 模式（``MemoryStore._configure``）：刚提交的东西还躺在 ``mem.db-wal``
    里，主库文件上只有一页头——连 ``memories`` 这张表本身都还不在里面。而巩固恰恰是
    在**刚被写过**的库上拍快照，裸拷漏掉的正好是最新的那几条。所以这里同时走两遍：
    裸拷主库文件，打开来是"没有这张表"；``snapshot_to`` 写出来的那份，行都在。

    两边都在库还开着的时候做。SQLite 在**最后一个连接关闭**时会做一次检查点并删掉
    ``-wal``，那时裸拷就"恰好"也对了——那正是这条用例要避免的假绿。
    """
    wal = Path(f"{memory_db_path()}-wal")
    with _store() as store:
        # 两件事都在库里还开着的时候做完，尤其那一次裸拷：关库会做检查点并把
        # `-wal` 删掉，之后裸拷就"恰好"也对了。
        assert wal.stat().st_size > 0, "前提：确实还有东西没检查点"
        naive = data_dir / "naive.db"
        shutil.copy2(memory_db_path(), naive)
        thorough = store.snapshot_to(data_dir / "thorough.db")

    with pytest.raises(sqlite3.DatabaseError, match="no such table"):
        _reinforce_counts(naive)
    assert _reinforce_counts(thorough) == {"m000": 0, "m001": 0, "m002": 0}


def test_the_snapshot_name_is_utc_and_second_resolution() -> None:
    """文件名是 ``mem-<UTC yyyymmddTHHMMSS>.db``。

    拿 epoch 断言，这样"用的是 UTC 还是本地时间"有确定答案：本地是 UTC+8 的机器上
    按本地时间写会得到 ``19700101T080000``。文件名、mtime 排序、轮转三处必须同一个
    时区，混用会在夏令时切换那天排序错乱——UTC 没有那个切换。
    """
    assert entry.snapshot_name(0.0) == "mem-19700101T000000.db"
    assert entry.snapshot_name(AT) == "mem-20251009T085320.db"


# ── 次序：快照在写之前 ──────────────────────────────────────────────────────


def test_the_snapshot_is_taken_before_the_database_is_written(data_dir: Path) -> None:
    """快照里是**改之前**的计数，库里的计数已经 +1。

    这一层唯一的退路就是它：出事之后要能回到"这一轮没跑过"那个状态。快照要是拍在
    写之后，它保护的就是一次已经发生的修改——看起来一切正常，直到真要用它那天。
    """
    with _store() as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)
        after = {item.memory_id: item.reinforce_count for item in store.iter_items()}

    assert run.snapshot is not None
    assert _reinforce_counts(run.snapshot) == {"m000": 0, "m001": 0, "m002": 0}
    assert after == {"m000": 1, "m001": 1, "m002": 1}


def test_an_idle_round_snapshots_nothing(data_dir: Path) -> None:
    """一条都没加固的轮次**不拍快照**（连快照目录都没有）。

    整库复制是几百 MB 的写，而它保护的那次修改根本不存在。判断用的是 ``reinforced``
    而不是"跑完没有"——``pinned`` 的条目会被跳过、只进事件，那一轮什么都没改。
    """
    with _store(pinned=True) as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)

    assert run.reinforced == 0
    assert run.snapshot is None
    assert run.events_path.exists(), "事件照样要落：那一轮确实发生过"
    # 查目录本身，不查 ``snapshot_dir()``——那个函数**顺手 mkdir**，拿它来断言
    # "目录不存在"是自己把自己弄成真的。
    assert not (memory_db_path().parent / "snapshots").exists()


# ── 拒绝：同一轮跑第二遍 ────────────────────────────────────────────────────


def test_a_second_run_of_the_same_round_is_refused(data_dir: Path) -> None:
    """同一个 ``at`` 跑第二遍**当场拒掉**，而且不留下任何痕迹。

    事件文件按 ``run_id`` 命名，``run_id`` 由 ``(base_seed, at)`` 派生。往里追加的
    结果是"两轮混成一份、``--replay`` 永远对不上"的记录，而报出来的是"这是 bug"——
    一个重复的 ``at`` 就能造出一份永久有效的假警报。所以宁可拒绝。

    "不留下痕迹"是这条的另一半：拒绝要赶在拍快照**之前**，否则会攒下一份保护着
    "没有发生的那次修改"的快照，而快照是几百 MB 的写。
    """

    def lines(path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines()

    with _store() as store:
        first = entry.run_consolidation(store, _one_cluster(), at=AT)
        snapshots_after_first = sorted(snapshot_dir().glob("*.db"))

        with pytest.raises(ValueError, match="已经跑过"):
            entry.run_consolidation(store, _one_cluster(), at=AT)

        assert len(lines(first.events_path)) == 3, "被拒的那一轮不该往事件里追一行"
        assert sorted(snapshot_dir().glob("*.db")) == snapshots_after_first
        assert {item.memory_id: item.reinforce_count for item in store.iter_items()} == {
            "m000": 1,
            "m001": 1,
            "m002": 1,
        }, "被拒的那一轮不该动库"

    # 换一个 at 就是另一轮：这道门挡的是"同一轮跑两遍"，不是"跑两遍"。
    with _open() as store:
        second = entry.run_consolidation(store, _one_cluster(), at=AT + 0.001)
        assert {item.memory_id: item.reinforce_count for item in store.iter_items()} == {
            "m000": 2,
            "m001": 2,
            "m002": 2,
        }
    assert second.report.run_id != first.report.run_id


# ── 轮转：留最近 5 份 ───────────────────────────────────────────────────────


def _fake_snapshots(*names: str) -> list[str]:
    """在快照目录里摆几个**空文件**，返回按名字排好的那串。

    轮转只看文件名，所以不需要真库——一条真的 ``snapshot_to`` 换不来这里要问的东西
    （"删的是哪几个"），却要跑一遍完整的回放。
    """
    directory = snapshot_dir()
    for name in names:
        (directory / name).write_bytes(b"")
    return sorted(name for name in names)


def test_rotation_keeps_the_newest_five(data_dir: Path) -> None:
    """留最近 5 份，删的是**名字最旧**的那些，返回值把删掉的报出来。

    名字是 ``mem-<UTC yyyymmddTHHMMSS>.db``，定宽、字典序即时间序，所以"最旧"有
    确定答案。返回值不是装饰：轮转删的是**用户的退路**，调用方与将来的 ``--prune``
    要能说清它删了什么。
    """
    names = _fake_snapshots(*(f"mem-2026010{day}T030000.db" for day in range(1, 8)))

    doomed = entry.rotate_snapshots()

    assert [path.name for path in doomed] == names[:2]
    assert sorted(path.name for path in snapshot_dir().glob("*.db")) == names[2:]
    assert len(names[2:]) == entry.SNAPSHOT_KEEP


def test_rotation_looks_at_the_name_not_the_mtime(data_dir: Path) -> None:
    """排序键是**文件名**，不是 mtime。

    差别在"快照目录从备份里拷回来"这件事上：拷完每份的 mtime 都是"刚才"，按 mtime
    排序等于随机删，而第一个被删的可能正是最新的那一份。这里把 mtime 反过来设
    （最新那份的 mtime 最旧），按名字轮转仍然留下它。
    """
    names = _fake_snapshots("mem-20260101T030000.db", "mem-20260102T030000.db")
    directory = snapshot_dir()
    for name, age in zip(names, (10_000.0, 1.0), strict=True):
        os.utime(directory / name, (age, age))

    doomed = entry.rotate_snapshots(keep=1)

    assert [path.name for path in doomed] == ["mem-20260101T030000.db"]
    assert [path.name for path in directory.glob("*.db")] == ["mem-20260102T030000.db"], (
        "mtime 更新的那一份不是被留下的理由；名字里的时刻才是"
    )


def test_rotation_leaves_files_that_are_not_snapshots_alone(data_dir: Path) -> None:
    """只认 ``mem-*.db``。躺在同一个目录里的别的东西**一份都不删**。

    轮转的全部动作是 ``unlink``，删错一个不可撤销。那个目录将来住进别的东西是完全
    可能的（手工挪进来的一份备份、另一套工具的产物），而"顺手把目录清空到只剩 5 个
    文件"在测试里看起来和正确行为一模一样。
    """
    _fake_snapshots(
        "mem-20260101T030000.db",
        "mem-20260102T030000.db",
        "mem-20260103T030000.db",
    )
    directory = snapshot_dir()
    (directory / "notes.txt").write_text("别删我", encoding="utf-8", newline="\n")
    (directory / "backup.db").write_bytes(b"")

    entry.rotate_snapshots(keep=1)

    assert sorted(path.name for path in directory.iterdir()) == [
        "backup.db",
        "mem-20260103T030000.db",
        "notes.txt",
    ]


def test_rotation_refuses_a_keep_below_one(data_dir: Path) -> None:
    """``keep=0`` 当场拒掉，**不安静地一份都不删**。

    ``snapshots[:-0]`` 是空切片，于是"我配了不保留快照"与"快照一直在攒"看起来一模
    一样——一种没人会去查的错。轮转本身是"留 N 份"的意思，N 是 0 的时候它没有意义。
    """
    _fake_snapshots("mem-20260101T030000.db")

    with pytest.raises(ValueError, match="至少是 1"):
        entry.rotate_snapshots(keep=0)

    assert (snapshot_dir() / "mem-20260101T030000.db").exists()


def test_a_real_run_rotates_the_directory_to_five(data_dir: Path) -> None:
    """真跑六轮 → 盘上留 5 份，且是最新的那 5 份。

    上面几条量的是轮转函数本身，这条量的是**它真的被调用了**——纯层全绿不等于接线
    在。六轮各差一小时，快照名因此各不相同（名字精确到秒）。
    """
    with _store() as store:
        for index in range(6):
            run = entry.run_consolidation(store, _one_cluster(), at=AT + 3600 * index)

    remaining = sorted(path.name for path in snapshot_dir().glob("*.db"))
    assert len(remaining) == entry.SNAPSHOT_KEEP
    assert run.snapshot is not None
    assert remaining[-1] == run.snapshot.name, "最新那一份必须还在"
    assert entry.snapshot_name(AT) not in remaining, "第一轮那份该被轮转掉"


# ── last.json ───────────────────────────────────────────────────────────────


def test_last_run_round_trips(data_dir: Path) -> None:
    """``last.json`` 写出去再读回来是同一份东西，落点在巩固目录里。

    调度器靠它回答"24 小时到了没有"，热度面板靠它说"最近一次巩固是什么时候"。
    ``newest_id`` 是给前者用的：库里的 id 按时间排序，"最新那条还是上次那条"等价于
    "这期间没有新记忆写进来"。
    """
    with _store() as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)
    path = entry.write_last_run(run.report, count=run.reinforced, newest_id="m002")

    assert path == consolidation_last_path()
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert entry.read_last_run() == {
        "run_id": run.report.run_id,
        "at": AT,
        "count": 3,
        "newest_id": "m002",
    }


def test_last_run_returns_none_instead_of_inventing_a_number(data_dir: Path) -> None:
    """没跑过、文件坏了、字段不对——一律 ``None``，**不返回一个像样的默认字典**。

    这个文件不是证据，是给下一次跑用的记录。坏数据该让调用方说"不知道"，而不是在
    这里体贴地补一个零：``at=0`` 会让热度面板写"最近一次巩固：1970-01-01"，而那是
    编出来的。
    """
    assert entry.read_last_run() is None, "还没跑过"

    consolidation_last_path().write_text("{ 这不是 JSON", encoding="utf-8", newline="\n")
    assert entry.read_last_run() is None

    consolidation_last_path().write_text(
        json.dumps({"at": AT}, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    assert entry.read_last_run() is None, "缺 run_id 就是不全"


def test_last_run_hands_back_a_time_and_a_count(data_dir: Path) -> None:
    """``last_run`` 是那一份 json 的**唯一**读法：``(时刻, 条数)``。

    两个界面成员要用它（设置页那行说明、记忆窗口的热度面板），而它们各自还要自己
    判一次"这个数拿得去算吗"的话，某天其中一处漏了——``bool`` 是 ``int`` 的子类，
    ``float(True)`` 是 ``1.0``——症状是"最近一次"显示成 1970 年。
    """
    with _store() as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)
    entry.write_last_run(run.report, count=run.reinforced, newest_id="m002")

    assert entry.last_run() == (AT, 3)


def test_last_run_refuses_a_time_it_cannot_use(data_dir: Path) -> None:
    """``at`` 不是个能算的数 → ``None``。**不返回一个补过的时刻。**

    ``read_last_run`` 只保证"整个对象读得出来"，而手改过的 json 可以让 ``at`` 是
    字符串、``true``，或 ``Infinity``——最后那个不是合法 JSON，但 Python 的解析器
    照收（默认的 ``parse_constant`` 不拦），而 ``datetime.fromtimestamp`` 收不下。
    """
    assert entry.last_run() is None, "还没跑过"

    for bad in ("昨天", True, None, [AT], float("inf"), float("nan")):
        consolidation_last_path().write_text(
            json.dumps({"run_id": "r", "at": bad, "count": 3}, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        assert entry.last_run() is None, f"{bad!r} 不该被当成一个时刻"


def test_a_missing_count_is_zero_not_a_reason_to_say_nothing_at_all(data_dir: Path) -> None:
    """时刻好、条数坏 → **时刻留下，条数给 0**。

    两个字段的失效后果不一样：时刻坏了整句话就说不出来（"最近一次"没有宾语），而
    条数坏了只是少说一句"加固了几条"。把整条都丢掉，等于让一个不重要字段的坏值
    把一句真话也一起弄没。
    """
    consolidation_last_path().write_text(
        json.dumps({"run_id": "r", "at": AT, "count": "三条"}, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    assert entry.last_run() == (AT, 0)


# ── 一句话结论 ──────────────────────────────────────────────────────────────


def test_the_summary_line_says_no_rate_when_there_is_none(data_dir: Path) -> None:
    """一条都没回放到时，那句话里**不提辨认率**。

    "辨认率 0%" 会把"没测"说成"没测中"，而这两件事在 ``recognition_rate`` 那里本来
    就靠 ``None`` 分开。跳过数照报——它是这一轮真实发生了的事，不是编出来的。
    """
    with _store(tier=Tier.COLD) as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)

    line = entry.summary_line(run.report)
    assert run.report.recognition_rate is None
    assert line == f"巩固 {run.report.run_id[:8]}：没有可回放的条目（跳过 3 条）"
    assert "辨认率" not in line


def test_the_summary_line_carries_the_rate_when_there_is_one(data_dir: Path) -> None:
    """有数据时那句写的是真数字——两态的另一半。

    与上一条各自管一段输入。两条挤在同一段文本上断言的话，"没有数据"那一半会被写成
    "辨认率是某个数"，而这两种输入本来就走两条分支。
    """
    with _store() as store:
        run = entry.run_consolidation(store, _one_cluster(), at=AT)

    assert entry.summary_line(run.report) == (
        f"巩固 {run.report.run_id[:8]}：回放 3 条，认出 3 条（辨认率 100.0%）"
    )
