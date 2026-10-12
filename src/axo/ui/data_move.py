"""把一份数据根**完整地**复制到另一个地方，并证明它完整。

这是"用户自选记忆存哪"里唯一会**动盘**的一段，所以它被写成一个不需要 Qt、不需要
设置的纯函数对（:func:`plan_move` / :func:`execute_move`），好让测试能把每一种
拒绝与每一种失败都摆出来。

**三条不许商量的规矩：**

1. **源一个字节都不改。** 不搬、不删、不重命名。这个功能叫「搬迁」，但"删掉旧的
   那份"是另一件事、发生在校验通过之后、而且要用户点头——见 ``ui/data_root.py``
   的 ``MoveReport`` 与主窗口那一问。所以这里的每条失败路径都以"源完好"收场。
2. **不用 ``shutil.move`` / ``os.replace`` / ``rename``。** ``os.replace`` 在
   Windows 上要求目标目录**不存在**，而"目标是用户刚建的空文件夹"是最常见的
   情形；``shutil.move`` 跨卷时退化成 ``copytree`` + ``rmtree``——**它删源**；
   而 rename 再原子也不校验。要的从来不是原子性，是"目标完整之后才提交"，而
   拷贝 + 逐文件哈希给的正是这个。少一条分支就少一条没人走过的分支。
3. **不枚举子目录。** 整个根逐字复制。`memory/`（库 + `graph.npz` + `consolidation/`
   + `snapshots/`）、`sessions/`、`wallpapers/`、两个 `reminders*.jsonl`，以及用户
   自己往里放的任何文件——一张枚举表会随 ``paths.py`` 长出第 11 个函数而变旧，
   而"复制整个目录"不会。漏掉 `sessions/` 的症状是侧栏空掉（用户会认定"记忆丢了"），
   漏掉 `reminders.fired.jsonl` 的症状是旧提醒**重新响一遍**。
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "MoveKind",
    "MovePlan",
    "MoveResult",
    "execute_move",
    "manifest",
    "plan_move",
]

#: 记忆库的真身。它在不在，是"这里是不是一份数据根"的判据。
_LIBRARY = ("memory", "mem.db")

#: SQLite 的共享内存文件（``mem.db-shm``）。
#:
#: **不复制。** 它装的是 WAL 索引的进程内状态，里面没有任何需要保留的内容——下次
#: 打开库时按 ``-wal`` 自动重建。带着它走反而更坏：一个与 ``-wal`` 对不上的 ``-shm``
#: 会让 SQLite 认为索引可用而读到没提交的页。
#:
#: ``-wal`` **必须跟着走**，理由见 ``ui/main_window.py`` 的 ``_shutdown``：它刻意
#: 不关记忆库（理由写在那里），所以退出时 ``mem.db-wal`` 常常还在——只拷 ``mem.db``
#: 会得到一个少了一批**已提交**事务的库，而那个库看起来完全正常。
_SKIP_SUFFIX = "-shm"

_CHUNK = 1 << 20


class MoveKind(str, Enum):
    """用户点了「更改位置」之后，这件事是什么性质。"""

    COPY = "copy"
    """把源整棵复制过去，然后改用目标。"""

    ADOPT = "adopt"
    """目标里已经有一份记忆——直接用它，**一个字节都不复制**。

    这条覆盖的是"我把库拷到 D 盘了"与"U 盘插到另一台机器"：用户要的是**用那一份**，
    而不是把现在这份覆盖上去。
    """

    REFUSE = "refuse"
    """这个目标不能用，理由在 :attr:`MovePlan.summary`。"""


@dataclass(frozen=True)
class MovePlan:
    """一次搬迁的**判断结果**。纯数据，不动盘。"""

    kind: MoveKind
    source: Path
    destination: Path
    summary: str
    """一句给用户看的人话。界面直接贴它，不自己拼句子。"""

    discards_destination: bool = False
    """``True`` 表示 :func:`execute_move` 会先清空目标。

    只在"目标里是我们自己上次复制的一半"时为真。**单列一个字段而不是让
    ``execute_move`` 自己判断**：删掉一整个目录是这里唯一不可逆的动作，它该由一个
    显式的、可断言的开关控制，而不是由一段"看起来是半成品"的推断。
    """

    files: tuple[str, ...] = ()
    """源的相对路径清单（已排除 :data:`_SKIP_SUFFIX`）。"""

    total_bytes: int = 0

    @property
    def is_actionable(self) -> bool:
        """要不要动盘（``REFUSE`` 不动）。"""
        return self.kind is not MoveKind.REFUSE


@dataclass(frozen=True)
class MoveResult:
    """一次搬迁的**执行结果**。"""

    ok: bool
    source: Path
    destination: Path
    copied_bytes: int = 0
    copied_files: int = 0
    reason: str = ""
    """失败原因（成功时为空）。人话，界面直接贴。"""

    notes: tuple[str, ...] = field(default=())
    """附注。**不许单独判红**——见 :func:`execute_move` 里那条关于 ``quick_check``
    的说明。"""


# ── 清单与哈希 ──────────────────────────────────────────────────────────────


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_skipped(path: Path) -> bool:
    """这个文件在搬迁里被跳过吗（见 :data:`_SKIP_SUFFIX`）。"""
    return path.name.endswith(_SKIP_SUFFIX)


def _walk_files(root: Path) -> Iterator[Path]:
    for current, _dirs, names in os.walk(root):
        for name in names:
            path = Path(current) / name
            if not is_skipped(path):
                yield path


def manifest(root: Path) -> dict[str, tuple[int, str]]:
    """``{相对路径: (字节数, sha256)}``，**不含被跳过的那些**。

    比"复制完打开库看一眼"强的地方在于它不依赖任何格式细节：``PRAGMA quick_check``
    要先用 ``mode=ro`` 打开库，而 WAL 库在只读打开时可能因为建不出 ``-shm`` 而直接
    失败——那会变成一条**假的**"搬迁失败"。哈希是纯 I/O 的，不会因为格式细节翻脸。

    路径一律用 posix 形状的相对路径：``\\`` 与 ``/`` 混着比会在跨平台时对不上，
    而这里比的是"同一份东西在两个目录里"，用相对路径才与根在哪无关。
    """
    root = Path(root)
    found: dict[str, tuple[int, str]] = {}
    for path in _walk_files(root):
        relative = path.relative_to(root).as_posix()
        found[relative] = (path.stat().st_size, _digest(path))
    return found


# ── 判断 ────────────────────────────────────────────────────────────────────


def _within(child: Path, parent: Path) -> bool:
    """``child`` 是不是在 ``parent`` 里面（或就是它）。"""
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def _is_empty(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    except OSError:
        return False
    return False


def _looks_like_a_library(root: Path) -> bool:
    return (root / _LIBRARY[0] / _LIBRARY[1]).is_file()


def plan_move(
    source: Path | str, destination: Path | str, *, previous_attempt: bool = False
) -> MovePlan:
    """判断"把 ``source`` 换到 ``destination``"这件事是什么性质。

    ``previous_attempt`` 为真表示"上一次搬到一半断了，这是同一个目标"。它只影响
    一条分支：目标非空、里面又没有库时，那堆东西可能是**我们自己**上次留下的半成品
    ——见 :attr:`MovePlan.discards_destination` 关于为什么这一条值得单开。
    """
    source = Path(source)
    destination = Path(destination)

    if not _looks_like_a_library(source):
        return MovePlan(
            kind=MoveKind.REFUSE,
            source=source,
            destination=destination,
            summary=f"{source} 看起来不是一份记忆库（里面没有 memory/mem.db），没有可搬的东西。",
        )

    if source == destination:
        return MovePlan(
            kind=MoveKind.REFUSE,
            source=source,
            destination=destination,
            summary="新位置与现在的位置是同一个文件夹。",
        )

    # **互相包含必须挡住。** 目标在源里面时，复制会把自己正在写的东西再复制一遍
    # （无限长大或至少复制出一棵畸形树）；源在目标里面时，校验会把上一层的旧内容
    # 也算进来。两种都不会报错，只会得到一棵没法解释的目录树。
    if _within(destination, source) or _within(source, destination):
        return MovePlan(
            kind=MoveKind.REFUSE,
            source=source,
            destination=destination,
            summary="新旧位置不能互相包含——新的不能在旧的里面（反过来也不行）。",
        )

    if destination.exists() and not destination.is_dir():
        return MovePlan(
            kind=MoveKind.REFUSE,
            source=source,
            destination=destination,
            summary=f"{destination} 是一个文件，不是文件夹。",
        )

    if destination.exists() and not _is_empty(destination):
        # **这一条必须排在 ADOPT 前面。** 上一次复制断在半路时，目标里**很可能
        # 已经有 `memory/mem.db` 了**（按字母序它排得靠前，而且正是最先被复制的那
        # 批）——先判 ADOPT 的话，会把一个**残缺的库**当成"这里已经有一份记忆"
        # 接手过来，而用户看到的是"搬完了，但记忆少了一截"。
        #
        # 判据是硬的：标记里的 ``started`` 表示"我们开始往这儿复制了"，而标记只在
        # **成功之后**才清。所以只要它还带着 ``started``，这次复制就没有成功过，
        # 里面无论有什么都是我们自己的半成品。
        if previous_attempt:
            files, total = _survey(source)
            return MovePlan(
                kind=MoveKind.COPY,
                source=source,
                destination=destination,
                summary=(
                    f"上一次搬到一半断了。先把 {destination} 里那份半成品清掉，"
                    f"再把 {source} 整个搬过去。"
                ),
                discards_destination=True,
                files=files,
                total_bytes=total,
            )
        if _looks_like_a_library(destination):
            return MovePlan(
                kind=MoveKind.ADOPT,
                source=source,
                destination=destination,
                summary=(
                    f"{destination} 里已经有一份记忆，会直接用它——"
                    f"现在这份留在 {source} 不动，一个字节都不会复制。"
                ),
            )
        return MovePlan(
            kind=MoveKind.REFUSE,
            source=source,
            destination=destination,
            summary=(
                f"{destination} 里已经有别的东西了。挑一个空的文件夹，或者一个已经装过记忆的。"
            ),
        )

    files, total = _survey(source)
    return MovePlan(
        kind=MoveKind.COPY,
        source=source,
        destination=destination,
        summary=f"把 {source} 里的全部内容复制到 {destination}，然后改用新位置。",
        files=files,
        total_bytes=total,
    )


def _survey(source: Path) -> tuple[tuple[str, ...], int]:
    """源里要复制的文件清单与总字节数（给确认框说"要复制多大"）。

    **不在这里算哈希**：清单只用来报数，而哈希是 :func:`manifest` 的事、在真正
    复制的那一次才算。分开是因为 :func:`plan_move` 会被调两次（点击那一次、
    新进程启动那一次），而只有后者需要哈希。
    """
    names: list[str] = []
    total = 0
    for path in _walk_files(source):
        names.append(path.relative_to(source).as_posix())
        total += path.stat().st_size
    names.sort()
    return tuple(names), total


# ── 执行 ────────────────────────────────────────────────────────────────────


def _clean_up(destination: Path, copied: Iterable[str], *, whole: bool) -> None:
    """失败之后把**我们自己弄出来的东西**收回去，一个字节都不多删。

    ``whole`` 为真（目标本来就是我们上次的半成品）时整棵删掉；否则只删这一次复制
    出来的那些条目，然后从底往上摘掉空目录——用户建的那个空文件夹要留下。
    """
    if whole:
        shutil.rmtree(destination, ignore_errors=True)
        return
    for relative in copied:
        target = destination / relative
        with contextlib.suppress(OSError):
            target.unlink()
    # **自底向上逐个 rmdir，不先看"这层还有没有东西"。** ``os.walk(topdown=False)``
    # 的产出顺序是自底向上，但它给每层列的 ``dirs`` 是**进这层时**扫到的——那时下层
    # 还没被删，于是"这一层非空"会一直成立，空壳目录一个都收不掉。``rmdir`` 自己
    # 只在空目录上成功，让它去判反而准。
    for current, _dirs, _files in os.walk(destination, topdown=False):
        path = Path(current)
        if path == destination:
            # **用户自己建的那个空文件夹要留下**——它是他被要求挑的那一个。
            continue
        with contextlib.suppress(OSError):
            path.rmdir()


def execute_move(
    plan: MovePlan, *, on_progress: Callable[[int, int], None] | None = None
) -> MoveResult:
    """执行 :func:`plan_move` 给的方案。**只有 ``COPY`` 会动盘。**

    顺序是刻意的：先落清单 → 再复制 → 再落清单 → 两份逐条比 → **通过了才算数**。
    任何一步不过就把目标侧收回去、源一个字节没动——所以任何时刻断电，盘上只有两种
    状态：源完整（指针还指着它），或者源完整 + 目标完整。
    """
    if plan.kind is not MoveKind.COPY:
        return MoveResult(
            ok=False,
            source=plan.source,
            destination=plan.destination,
            reason=f"这个方案不需要执行（{plan.kind.value}）。",
        )

    destination = plan.destination
    # **这个目录是不是我们刚建出来的**，决定失败时收得多彻底：我们建的整棵删掉，
    # 用户建的那个空文件夹只收回我们复制进去的东西（见 :func:`_clean_up`）。
    created = not destination.exists()

    # ① 目标可能已经有东西——**只在方案明确说了"清掉它"时才清**。见
    # ``MovePlan.discards_destination``：这个字段是唯一允许删目录的依据。
    if plan.discards_destination:
        shutil.rmtree(destination, ignore_errors=True)

    copied: list[str] = []
    copied_bytes = 0
    try:
        destination.mkdir(parents=True, exist_ok=True)

        before = manifest(plan.source)
        total = len(before)

        for index, (relative, _entry) in enumerate(sorted(before.items()), start=1):
            source_file = plan.source / relative
            target_file = destination / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied.append(relative)
            copied_bytes += source_file.stat().st_size
            if on_progress is not None:
                on_progress(index, total)

        after = manifest(destination)
    except OSError as error:
        _clean_up(destination, copied, whole=plan.discards_destination or created)
        return MoveResult(
            ok=False,
            source=plan.source,
            destination=destination,
            reason=f"复制到 {destination} 的时候失败了：{error}",
            notes=(f"{plan.source} 一个字都没动。",),
        )

    # ② 逐条比。**多出来的文件也算不通过**——目标里凭空多出一个，说明它本来就不是
    # 空的（或者有别的东西在同时往里写），而那两种情况下"用这个目标"都是错的。
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    different = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    if missing or extra or different:
        _clean_up(destination, copied, whole=plan.discards_destination or created)
        detail = []
        if missing:
            detail.append(f"少了 {len(missing)} 个（{missing[0]}…）")
        if extra:
            detail.append(f"多了 {len(extra)} 个（{extra[0]}…）")
        if different:
            detail.append(f"{len(different)} 个内容对不上（{different[0]}）")
        return MoveResult(
            ok=False,
            source=plan.source,
            destination=destination,
            reason="复制出来的东西与原来的对不上：" + "、".join(detail),
            notes=(
                f"已经把 {destination} 里刚复制的那部分收回来了。",
                f"{plan.source} 一个字都没动。",
            ),
        )

    return MoveResult(
        ok=True,
        source=plan.source,
        destination=destination,
        copied_bytes=copied_bytes,
        copied_files=len(copied),
    )
