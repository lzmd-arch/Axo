"""搬迁：动盘的那一段，以及它**在什么时候拒绝动盘**。

这是本次改动里唯一会碰用户数据的地方，所以这里量三件事，每一件都对着一条可能
静默出错的路径：

1. **搬过去的东西与原来逐字节相同**（名字、字节数、sha256 三条都比），而且
   **源一个字节没动**——失败路径也一样。
2. **失败要能被抓住并回滚**：复制坏了、目标里本来就有人家的东西、目标与源互相
   包含……每一条都要么拒绝、要么原样收回去，不能留下半个状态。
3. **不枚举子目录**：根里用户自己随手丢的文件也得跟着走。这条是拿一个
   `notes.txt` 证的——枚举表会随 `paths.py` 长出第 11 个函数而变旧，而
   "复制整个目录"不会，所以这条测试防的是将来有人把它改回枚举。

`execute_move` **不写位置指针、也不删源**：那两件分别属于 `data_root.py` 与
用户点头之后的那一问。这里顺带钉住它俩没被顺手做掉。
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from axo.ui.data_move import MoveKind, execute_move, manifest, plan_move

_ANCHOR = "你是 Axo 的助手。"


def _library(root: Path) -> Path:
    """在 `root` 造一份**形状齐全**的数据根。

    刻意每一样都放一点：库（带 WAL 伴生文件）、图缓存、快照、巩固事件、逐字稿、
    壁纸、两个提醒文件，外加一个**用户自己丢进来的**文件。搬迁的判据是"整个目录",
    所以夹具也得是整个目录——只放一个 `mem.db` 的话，"漏搬了 sessions/"这类错
    在测试里根本看不出来。
    """
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "snapshots").mkdir()
    (root / "memory" / "consolidation").mkdir()
    (root / "sessions").mkdir()
    (root / "wallpapers").mkdir()

    connection = sqlite3.connect(root / "memory" / "mem.db")
    connection.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY, body TEXT)")
    connection.execute("INSERT INTO memories VALUES ('m1', ?)", (_ANCHOR,))
    connection.commit()
    # **不 close**：模拟 main_window._shutdown 那条路径——它刻意不关库，所以退出时
    # -wal 常常还在。搬迁必须把它带走，只拷 mem.db 会少一批已提交的事务。
    # （这里留着连接，-wal 就是活的。）
    (root / "memory" / "mem.db-wal").write_bytes(b"pretend-wal")
    (root / "memory" / "mem.db-shm").write_bytes(b"pretend-shm")
    (root / "memory" / "graph.npz").write_bytes(b"\x00\x01npz")
    (root / "memory" / "snapshots" / "mem-20260101T000000.db").write_bytes(b"snapshot")
    (root / "memory" / "consolidation" / "last.json").write_text("{}", encoding="utf-8")
    (root / "memory" / "consolidation" / "run1.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "sessions" / "s1.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
    (root / "wallpapers" / "wallpaper-abc.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "reminders.jsonl").write_text('{"id": "r1"}\n', encoding="utf-8")
    (root / "reminders.fired.jsonl").write_text('{"id": "r0"}\n', encoding="utf-8")
    # 用户自己丢进来的：不枚举的证据（见模块文档第 3 条）。
    (root / "notes.txt").write_text("我自己放这儿的", encoding="utf-8")
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _library(tmp_path / "src")


# ── 搬对了没有 ──────────────────────────────────────────────────────────────


def test_every_file_arrives_byte_for_byte(source: Path, tmp_path: Path) -> None:
    """**这一条是这个功能的全部内容。** 名字、字节数、sha256 三条都比。"""
    destination = tmp_path / "dst"
    before = manifest(source)

    plan = plan_move(source, destination)
    assert plan.kind is MoveKind.COPY

    result = execute_move(plan)

    assert result.ok is True, result.reason
    assert manifest(destination) == before
    assert manifest(source) == before, "源被改动了——这个功能不该碰源"


def test_the_copy_does_not_touch_the_source_or_the_pointer(source: Path, tmp_path: Path) -> None:
    """`execute_move` 只复制。**删源与改指针都不在它这儿**。

    删源是"校验通过之后、用户点头之后"那一问的事（见 `MainWindow.report_data_move`），
    指针是 `boot_data_root` 唯一的提交点。这里把两条都钉住：谁将来顺手加一句
    `shutil.rmtree(plan.source)` 或者 `save_data_root(...)`，先红的是这条。
    """
    destination = tmp_path / "dst"
    before = manifest(source)

    execute_move(plan_move(source, destination))

    assert source.is_dir()
    assert manifest(source) == before


def test_any_file_in_the_root_travels(source: Path, tmp_path: Path) -> None:
    """**不枚举子目录**的证据：根里用户自己放的文件也跟着走。

    这一条防的是"改成按 `paths.py` 的清单逐个搬"。那张表看起来更精确，实际是把它
    与 `paths.py` 绑在一起：那边长出第 11 个函数而这边没跟上时，**它是静默漏搬**——
    用户的文件还在旧位置，而界面上一切正常。
    """
    destination = tmp_path / "dst"

    execute_move(plan_move(source, destination))

    assert (destination / "notes.txt").read_text(encoding="utf-8") == "我自己放这儿的"


def test_the_wal_travels_and_the_shm_does_not(source: Path, tmp_path: Path) -> None:
    """`-wal` 必须跟着走，`-shm` 不能。

    只拷 `mem.db` 会得到一个**少了一批已提交事务**的库，而那个库打开时不报错——
    用户的记忆看起来"少了几条"，指不到搬迁这里。`-shm` 反过来：它装的是进程内
    索引状态，带过去反而让 SQLite 读到不该读的页。
    """
    destination = tmp_path / "dst"

    execute_move(plan_move(source, destination))

    assert (destination / "memory" / "mem.db-wal").is_file()
    assert not (destination / "memory" / "mem.db-shm").exists()
    assert "memory/mem.db-shm" not in manifest(destination)


def test_the_copied_library_is_actually_readable(source: Path, tmp_path: Path) -> None:
    """复制出来的库**能用**——哈希相等已经说明了，但这条说的是它还是个 sqlite 库。

    哈希能证明字节一样，证明不了"打开它对"。两条都要，因为它们将来会在不同的改动
    下红：把 `copy2` 换成某种"只拷数据不拷权限/时间"的写法，哈希这条抓不到。
    """
    destination = tmp_path / "dst"

    execute_move(plan_move(source, destination))

    connection = sqlite3.connect(
        f"file:{(destination / 'memory' / 'mem.db').as_posix()}?mode=ro", uri=True
    )
    rows = connection.execute("SELECT memory_id, body FROM memories").fetchall()
    connection.close()
    assert rows == [("m1", _ANCHOR)]


def test_progress_is_reported_for_every_file(source: Path, tmp_path: Path) -> None:
    """进度回调按"第几个 / 共几个"报——确认框要靠它说"可能要等一会儿"。"""
    seen: list[tuple[int, int]] = []

    execute_move(
        plan_move(source, tmp_path / "dst"),
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen, "一次都没报"
    assert seen == sorted(seen)
    assert len(seen) == seen[-1][1] == len(manifest(source))


# ── 坏掉了要被抓住 ──────────────────────────────────────────────────────────


def test_a_truncated_copy_is_caught_and_rolled_back(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """复制坏了 → 失败 → **目标侧收回、源完好**。

    这条测的是"校验真的在校验"。把 `copy2` 换成"少写一半"的那种写法，然后断言
    三件事：报了失败、目标被清掉（不留半成品给下一次误认为已完成）、源没动。

    **字节数变了所以哈希与大小两条都会不匹配**——正是想要的，因为真实世界里磁盘
    写满就是长这样。
    """
    destination = tmp_path / "dst"
    real_copy = shutil.copy2

    def _half(source_file: Path | str, target_file: Path | str, **kwargs: object) -> object:
        result = real_copy(source_file, target_file, **kwargs)  # type: ignore[arg-type]
        target = Path(target_file)
        if target.name == "graph.npz":
            target.write_bytes(b"\x00")
        return result

    monkeypatch.setattr("axo.ui.data_move.shutil.copy2", _half)

    result = execute_move(plan_move(source, destination))

    assert result.ok is False
    assert "对不上" in result.reason
    assert not destination.exists(), "半成品留在了目标位置"
    assert (source / "memory" / "graph.npz").read_bytes() == b"\x00\x01npz"


# ── 什么时候拒绝 ────────────────────────────────────────────────────────────


def test_a_non_empty_destination_is_refused(source: Path, tmp_path: Path) -> None:
    """目标里有别的东西 → 拒绝，**绝不覆盖**。

    这是最要紧的一条拒绝：`shutil.rmtree` 是这里唯一不可逆的动作，而"用户选错了
    文件夹"是最常见的一种误操作（选了个装满照片的目录）。
    """
    destination = tmp_path / "dst"
    destination.mkdir()
    (destination / "photos").mkdir()
    (destination / "photos" / "a.jpg").write_bytes(b"x")

    plan = plan_move(source, destination)

    assert plan.kind is MoveKind.REFUSE
    assert plan.discards_destination is False
    assert (destination / "photos" / "a.jpg").is_file()


def test_a_destination_inside_the_source_is_refused(source: Path) -> None:
    """目标在源里面 → 拒绝。

    不挡的话复制会把"正在写的东西"再复制一遍。它不会报错，只会得到一棵没法解释的
    目录树——而且会随着复制过程长大。
    """
    plan = plan_move(source, source / "inner")

    assert plan.kind is MoveKind.REFUSE
    assert not (source / "inner").exists()


def test_a_source_inside_the_destination_is_refused(source: Path, tmp_path: Path) -> None:
    """源在目标里面 → 拒绝。校验会把上一层那些旧内容也算进来，于是永远对不上。"""
    wrapper = tmp_path
    plan = plan_move(source, wrapper)

    assert plan.kind is MoveKind.REFUSE


def test_the_same_folder_is_refused(source: Path) -> None:
    assert plan_move(source, source).kind is MoveKind.REFUSE


def test_a_destination_that_is_a_file_is_refused(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "not-a-folder"
    target.write_text("x", encoding="utf-8")

    plan = plan_move(source, target)

    assert plan.kind is MoveKind.REFUSE
    assert target.read_text(encoding="utf-8") == "x"


def test_a_source_without_a_library_is_refused(tmp_path: Path) -> None:
    """源里没有 `memory/mem.db` = 那不是一份数据根，没有可搬的东西。

    方向要紧：这里若放行，用户会得到一个"搬过去但什么都没有"的新位置，而旧位置
    还留着真东西——两份看起来都像数据根。
    """
    empty = tmp_path / "not-a-library"
    empty.mkdir()
    (empty / "random.txt").write_text("x", encoding="utf-8")

    assert plan_move(empty, tmp_path / "dst").kind is MoveKind.REFUSE


# ── 接手与重来 ──────────────────────────────────────────────────────────────


def test_an_existing_library_at_the_destination_is_adopted(source: Path, tmp_path: Path) -> None:
    """目标里已经有一份记忆 → **用它**，一个字节都不复制。

    这条覆盖"我把库拷到 D 盘了"与"U 盘插到另一台机器"：用户要的是用那一份，而不是
    把现在这份盖上去——盖上去的话，他插上盘那一刻就把盘里那份记忆覆盖掉了。
    """
    destination = _library(tmp_path / "dst")

    plan = plan_move(source, destination)

    assert plan.kind is MoveKind.ADOPT
    assert plan.total_bytes == 0
    assert plan.files == ()


def test_adopting_does_not_execute_anything(source: Path, tmp_path: Path) -> None:
    """`ADOPT` 不动盘——`execute_move` 直接回一个"不需要执行"。"""
    destination = _library(tmp_path / "dst")
    before = manifest(destination)

    result = execute_move(plan_move(source, destination))

    assert result.ok is False
    assert manifest(destination) == before


def test_a_retry_after_a_half_finished_copy_clears_our_own_leftovers(
    source: Path, tmp_path: Path
) -> None:
    """上次搬到一半断了 → 认出那堆东西是**我们自己的**，清掉重来。

    `discards_destination` 只在"上一次是我们自己复制的"这一条分支上为真——它也是
    `execute_move` 里唯一允许删目录的依据。所以这条同时钉住：按 `previous_attempt`
    判、清得干净、以及清完之后搬出来的东西是完整的那一份。
    """
    destination = tmp_path / "dst"
    destination.mkdir()
    # 上次复制到一半留下的：半个库 + 一个已经复制好的文件。
    (destination / "memory").mkdir()
    (destination / "memory" / "mem.db").write_bytes(b"half")
    (destination / "notes.txt").write_text("stale", encoding="utf-8")

    plan = plan_move(source, destination, previous_attempt=True)
    assert plan.kind is MoveKind.COPY
    assert plan.discards_destination is True

    result = execute_move(plan)

    assert result.ok is True, result.reason
    assert manifest(destination) == manifest(source)


def test_a_half_finished_copy_is_not_mistaken_for_a_library(source: Path, tmp_path: Path) -> None:
    """**上一次搬到一半的目标里有 `mem.db`，也绝不能当成"这里已经有一份记忆"。**

    复制断在半路时，`memory/mem.db` 往往**已经在那儿了**（它排得靠前，是最先被复制
    的那一批）。先判 `ADOPT` 的话，接手过来的是一个缺了一截的库——而用户看到的是
    "搬完了，但记忆少了一截"，指不到这里。

    判据是硬的：标记里那面 ``started`` 旗只在**成功之后**才清，所以它还在就等于
    "这次复制没有成功过，里面无论有什么都是我们自己的半成品"。

    对照组就在下面两行：**没有那面旗时，同一个目录看起来就是一份现成的库**，那时
    接手才是对的。
    """
    destination = tmp_path / "dst"
    destination.mkdir()
    (destination / "memory").mkdir()
    (destination / "memory" / "mem.db").write_bytes(b"half")

    assert plan_move(source, destination, previous_attempt=True).kind is MoveKind.COPY
    assert plan_move(source, destination).kind is MoveKind.ADOPT


def test_a_strange_non_empty_destination_is_refused(source: Path, tmp_path: Path) -> None:
    """目标非空、里面没有库、也没有那面旗 → **拒绝，不许猜。**

    这堆东西可能是用户的、可能是别的程序的。分不清的时候正确的做法是拒绝；而
    `previous_attempt` 是"上一次确实是我们搬的"的**唯一**证据，它由注册表里那个
    搬迁标记提供。
    """
    destination = tmp_path / "dst"
    destination.mkdir()
    (destination / "notes.txt").write_text("stale", encoding="utf-8")

    plan = plan_move(source, destination)

    assert plan.kind is MoveKind.REFUSE
    assert plan.discards_destination is False
    # 有旗时同一个目录才被认作我们自己的半成品。
    assert plan_move(source, destination, previous_attempt=True).kind is MoveKind.COPY


def test_a_failed_copy_leaves_a_pre_existing_empty_folder_in_place(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标本来就是用户建的空文件夹 → 失败时只收回我们复制的那部分，**不删那个文件夹**。

    删掉它是"顺手"的一步，而那一步在目标其实是个挂载点或符号链接时是灾难。
    所以回滚必须精确到"这一次复制出来的条目"。
    """
    destination = tmp_path / "dst"
    destination.mkdir()
    real_copy = shutil.copy2

    def _half(source_file: Path | str, target_file: Path | str, **kwargs: object) -> object:
        result = real_copy(source_file, target_file, **kwargs)  # type: ignore[arg-type]
        target = Path(target_file)
        if target.name == "graph.npz":
            target.write_bytes(b"\x00")
        return result

    monkeypatch.setattr("axo.ui.data_move.shutil.copy2", _half)

    assert execute_move(plan_move(source, destination)).ok is False

    assert destination.is_dir(), "用户建的那个空文件夹被删掉了"
    assert list(destination.iterdir()) == [], "没收回干净"
