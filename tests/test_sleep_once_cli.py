"""``tools/sleep_once.py`` 的**进程级**契约：退码，以及它肯不肯动手。

为什么是子进程而不是 import 进来调 ``main()``：这个脚本的产出**就是退出码**，而
退出码只有真进程才有。仓里对 ``tools/`` 的态度是"测试不该 import 工具"（见
``test_substrate_gate.py:106`` 那段理由）——那条在这里刚好也成立：库那一侧的逻辑
由 ``test_consolidation_*.py`` 直接调库来测，这里测的是**包在外面那一层**。

退码表：``0`` 跑完（``--dry-run`` 也是 0），``1`` **v1 不产生**，``2`` 用法或环境
不对，``3`` 结构不变量被违反。**``1`` 与 ``2`` 的区别是要紧的**：未捕获异常退出码
也是 1，所以"库被人删了"会伪装成"指标没过线"——这个文件里那几条退 2 的用例
盯的就是这件事。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from biosnn_memterm.paths import graph_path, memory_db_path
from memcore.graph import GraphIndex
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "sleep_once.py"
_TIMEOUT = 180

_TEXTS = [
    "我家的路由器放在书房书架顶层。",
    "今天午饭吃的是番茄炒蛋和米饭。",
    "每周三晚上八点有一节游泳课。",
]


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """一个装好库与图的合成数据目录。

    三条记忆 + 一张全连的团。**图是手搭的、不是 ``GraphIndex.build`` 出来的**：
    三句话在真实的建图门槛下很可能连不出一条边，那样"跑一轮"会安静地变成"跑了
    个空轮"，而每条用例的期望值都要跟着解释一遍"为什么什么都没发生"。
    """
    monkeypatch.setenv("BIOSNN_MEMTERM_DATA", str(tmp_path))
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
    with MemoryStore(memory_db_path()) as store:
        for item in items:
            store.add(item)
    size, top_m = len(items), len(items) - 1
    nbr_idx = np.full((size, top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, top_m), dtype=np.float32)
    for row in range(size):
        for slot, target in enumerate(index for index in range(size) if index != row):
            nbr_idx[row, slot] = target
            nbr_w[row, slot] = 1.0
    GraphIndex(
        node_ids=tuple(item.memory_id for item in items),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="d",
        built_at=0.0,
        top_m=top_m,
        min_weight=0.1,
    ).save(graph_path())
    return tmp_path


def _cli(data_dir: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """跑一次真进程。**显式把数据目录传进环境**，不靠外面碰巧设对了。"""
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=_TIMEOUT,
        check=False,
        env={**os.environ, "BIOSNN_MEMTERM_DATA": str(data_dir)},
    )


def _event_files(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "memory" / "consolidation").glob("*.jsonl"))


def _snapshots(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "memory" / "snapshots").glob("*.db"))


def _reinforce_counts(data_dir: Path) -> dict[str, int]:
    connection = sqlite3.connect(str(memory_db_path()))
    try:
        rows = connection.execute(
            "SELECT memory_id, reinforce_count FROM memories ORDER BY memory_id"
        ).fetchall()
    finally:
        connection.close()
    return dict(rows)


# ── 跑一场 ──────────────────────────────────────────────────────────────────


def test_dry_run_reports_and_writes_nothing(data_dir: Path) -> None:
    """``--dry-run`` 退 0，且**盘上什么都没多**。

    "什么都不写"是这条的全部内容，所以要查三处：事件（连带 ``last.json``，它俩同住
    巩固目录）、快照、计数。只查其中一处的话，另外两处各写一点也照样绿。

    查的是**文件**不是目录：``snapshots/`` 那个空目录会出现——``_open_store`` 要判断
    "给的库是不是在快照目录里"，而 ``paths.snapshot_dir()`` 按规定顺手 mkdir。那是
    位置解析的副作用（``paths`` 只做 mkdir 这一种 I/O），不是这一轮写下的东西。
    """
    result = _cli(data_dir, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "辨认率 1.0000" in result.stdout, result.stdout
    assert "没有改库、没有拍快照、没有落事件" in result.stdout
    assert not (data_dir / "memory" / "consolidation").exists()
    assert _snapshots(data_dir) == []
    assert _reinforce_counts(data_dir) == {"m000": 0, "m001": 0, "m002": 0}


def test_a_real_run_leaves_events_a_snapshot_and_a_last_run(data_dir: Path) -> None:
    """真跑一次：事件 3 行、快照 1 份、``last.json`` 指向这一轮、计数 +1。

    四样东西缺一不可，而它们各自坏掉的样子都很安静：事件少了就重放不出来，快照
    少了就没有退路，``last.json`` 少了调度器每次启动都当成到点。
    """
    result = _cli(data_dir)

    assert result.returncode == 0, result.stderr
    events = _event_files(data_dir)
    assert len(events) == 1
    rows = [json.loads(line) for line in events[0].read_text(encoding="utf-8").splitlines()]
    assert [row["memory_id"] for row in rows] == ["m000", "m001", "m002"]
    assert events[0].stem == rows[0]["run_id"], "文件名就是那一轮的 run_id"
    assert len(_snapshots(data_dir)) == 1
    assert _reinforce_counts(data_dir) == {"m000": 1, "m001": 1, "m002": 1}

    last = json.loads(
        (data_dir / "memory" / "consolidation" / "last.json").read_text(encoding="utf-8")
    )
    assert last["run_id"] == rows[0]["run_id"]
    assert last["count"] == 3
    assert last["newest_id"] == "m002", "库里的 id 排序即时间序，最新那条是 m002"


def test_json_goes_to_stdout_and_the_human_report_to_stderr(data_dir: Path) -> None:
    """``--json`` 时 stdout 上**只有**一份能解析的 JSON，人看的那几行去了 stderr。

    不这么分的话 ``> report.json`` 出来的东西谁都解析不了——报告与 JSON 掺在一起，
    而"混着"这种坏法在下游是静默的（解析失败被当成"这次没有结果"）。
    """
    result = _cli(data_dir, "--json", "--quiet")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["replayed"] == 3
    assert payload["reinforced"] == 3
    assert payload["recognition_rate"] == 1.0
    assert payload["dry_run"] is False
    assert payload["events_written"] == 3
    assert payload["events"].endswith(".jsonl")
    assert payload["snapshot"].endswith(".db")
    assert "辨认率" in result.stderr and "辨认率" not in result.stdout


# ── --replay ────────────────────────────────────────────────────────────────


def test_replaying_the_same_round_matches(data_dir: Path) -> None:
    """真跑一轮之后按 ``run_id`` 重跑，逐条一致、退 0。

    这是 ``ConsolidationEvent`` 存在的全部理由：同一轮巩固重跑出同一串事件。它成立
    靠的是"抽样与分组全用 blake2b、一处不用 numpy RNG"（见 ``memcore.sdr`` 开头那段）。
    """
    assert _cli(data_dir).returncode == 0
    run_id = _event_files(data_dir)[0].stem

    result = _cli(data_dir, "--replay", run_id)

    assert result.returncode == 0, result.stderr
    assert f"与 {run_id} 逐条一致（3 条）" in result.stdout


def test_a_replay_that_disagrees_exits_three(data_dir: Path) -> None:
    """记录被人动过之后重跑对不上，**退 3**（结构不变量被违反 = bug）。

    这里手动把第 1 行的决策改掉再重放：退 3 而不是退 0，且输出里点名是第几条、
    哪个字段。**退码是被测的东西**——它要是滑成 0，一份对不上的审计记录会被当成
    "复现成功"读过去。
    """
    assert _cli(data_dir).returncode == 0
    events = _event_files(data_dir)
    lines = events[0].read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["decision"] = "sleep_confusion"
    events[0].write_text(
        "\n".join([json.dumps(first, ensure_ascii=False), *lines[1:]]) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = _cli(data_dir, "--replay", events[0].stem)

    assert result.returncode == 3, result.stdout + result.stderr
    assert "重跑出来与落盘的记录不一致（1 处）" in result.stdout
    assert "第 0 条：decision 记录是 'sleep_confusion'、重跑是 'reinforce'" in result.stdout
    assert "这是 bug，不是环境问题" in result.stdout


def test_a_snapshot_is_opened_as_a_copy_and_the_original_is_untouched(data_dir: Path) -> None:
    """``--replay --db <快照>`` 打开的是**副本**，原文件一个字节都不变。

    SQLite 一打开就会往文件里写 WAL 头、留下 ``-wal``/``-shm`` 边车——证据就不再是
    原来那一份了。快照的意义正是"事后回到那一刻重放"，所以它必须只读地进。命令行上
    那一行"快照按副本打开"是这件事的告知：整库复制是几百 MB 的写，不该悄悄发生。
    """
    assert _cli(data_dir).returncode == 0
    snapshot = _snapshots(data_dir)[0]
    before = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    result = _cli(data_dir, "--replay", _event_files(data_dir)[0].stem, "--db", str(snapshot))

    assert result.returncode == 0, result.stderr
    assert "快照按副本打开（原文件不动）" in result.stdout
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == before
    assert not Path(f"{snapshot}-wal").exists(), "边车不该留在快照旁边"


# ── 环境与用法 ──────────────────────────────────────────────────────────────


def test_a_missing_database_exits_two(data_dir: Path) -> None:
    """库不在就退 2，**而不是建一个空库跑完退 0**。

    ``MemoryStore`` 打开不存在的路径会建一个新库，于是"路径打错"与"还没有记忆"会
    得到同一个结果（0 条、跑完、退 0）——而这两件事的正确处置完全不同：前者要人去看
    路径，后者要人去用应用。
    """
    memory_db_path().unlink()

    result = _cli(data_dir)

    assert result.returncode == 2, result.stdout
    assert "找不到记忆库" in result.stderr
    assert "这个脚本不建库" in result.stderr
    assert not memory_db_path().exists(), "退 2 之前不该已经把它建出来了"


def test_a_missing_graph_exits_two(data_dir: Path) -> None:
    """图不在就退 2——巩固靠图分簇，没有图就没有这一轮。

    这里也**不顺手建一张**：五万条上建图要几分钟，藏在夜间任务的副作用里，用户
    下次打开记忆窗口会发现自己盯着"正在建立关联图"。
    """
    graph_path().unlink()

    result = _cli(data_dir)

    assert result.returncode == 2, result.stdout
    assert "找不到关联图" in result.stderr
    assert not graph_path().exists()


def test_a_corrupt_graph_exits_two(data_dir: Path) -> None:
    """图坏了也退 2，**不崩**。图是派生缓存，重建一张就好——而崩掉的话退出码是 1，
    那个数字在这张表里说的是"指标跌了"。"""
    graph_path().write_bytes(b"not an npz at all")

    result = _cli(data_dir)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "读不出来" in result.stderr
    assert "重建" in result.stderr


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (("--replay", "not-a-run-id"), "16 位十六进制"),
        (("--replay", "../escape"), "16 位十六进制"),
        (("--drop", "1.5"), "--drop"),
        (("--max-cluster", "1"), "--max-cluster"),
        (("--limit", "1"), "--limit"),
        (("--seed", "-1"), "--seed"),
        (("--replay", "0" * 16, "--dry-run"), "--dry-run"),
    ],
)
def test_usage_and_environment_errors_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...], needle: str
) -> None:
    """用法不对一律退 2，并在 stderr 上说清是哪一处。

    ``--replay ../escape`` 这一格是**路径穿越**：那个值要拼进文件名，而清理它不如
    验形状——``run_id`` 的形状是定死的（16 位十六进制），拒绝一个不合形状的值不会
    误伤任何真实用法。
    """
    monkeypatch.setenv("BIOSNN_MEMTERM_DATA", str(tmp_path))

    result = _cli(tmp_path, *argv)

    assert result.returncode == 2, result.stdout + result.stderr
    assert needle in result.stderr
