"""启动时定下"记忆放在哪"，以及把一次**跨重启**的搬迁办完。

这里做三件互相关联的事，都不碰 Qt（``QSettings`` 由调用方以
:class:`~axo.ui.settings_store.SettingsBackend` 的形状传进来）：

1. **解析位置**——四档优先级的次序在 :mod:`axo.paths` 里，这里负责把"注册表里
   那个"读出来再告知它（``paths`` 不许 import Qt，理由见那个模块的文档）。
2. **搬迁的日志**——用户点「更改位置」时只写一条标记，真正动盘发生在**下一次启动**。
3. **收尾那一问**——搬完了要不要删掉旧的那份。

## 为什么搬迁要跨一次重启

原地搬需要在**同一个进程里**停掉四条线程、关掉记忆库、搬、校验、改指针、再重启。
而 ``ui/main_window.py`` 的 ``_shutdown`` 是"唯一的退出路径"、终结性的，且它
**刻意不关记忆库**（那段注释专门挡住将来那句好心的 ``self._memory.close()``）。
原地搬要求的正是那条被挡住的路：一个"停线程但别退、并且这次要关库"的半停机状态，
之后还要把四个 store 与整张图重新接线。

而既然最后都要重启，新进程里**零个打开的句柄**——在旧进程里做的每一步都是白做的
风险。所以：点击 → 写标记 → 立刻重启 → 新进程在打开任何东西之前搬完。用户的体验
仍是一次点击，代价是一次重启。

好处不止是省事：**任何时刻断电，盘上只有两种状态**——源完整（指针还指着它），
或者源完整 + 目标完整（指针指着后者）。指针是唯一一次写（:func:`boot_data_root`
里那一行 ``save_data_root``），它写在复制并校验通过**之后**。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from .data_move import MoveKind, MoveResult, execute_move, plan_move
from .settings_store import SettingsBackend, SettingsStore

__all__ = [
    "MoveMarker",
    "MoveReport",
    "boot_data_root",
    "data_root_description",
    "discard_source",
    "read_marker",
    "stage_move",
]

#: 待办搬迁的标记。
#:
#: **一个键装一整个 JSON，不是三个键。** 三个字段必须一致，而多一个键就多一个
#: "写了一半"的中间态——这个仓库对"文件在但内容是空的"已经很敏感了，同一个病没
#: 理由在注册表里再犯一次。
#:
#: 它放在这里而不是 ``settings_store`` 的键区：它**不是一条设置**，是一篇一次性的
#: 流水账——写下来、办完、划掉。设置是"用户希望怎样"，而这一格是"还有一件事没做"。
_MARKER_KEY = "app/pending_move"


@dataclass(frozen=True)
class MoveMarker:
    """一条待办搬迁。"""

    source: Path
    destination: Path
    started: bool
    """**已经开始往目标复制了。**

    它是"目标里那堆东西是我们自己的半成品"的唯一证据，也决定了重来时要不要先清
    目标（见 :func:`~axo.ui.data_move.plan_move` 的 ``previous_attempt``）。落旗的
    时刻在动盘**之前**，清旗只在成功**之后**——所以它还在，就等于"那次复制没有
    成功过"。
    """


def read_marker(backend: SettingsBackend) -> MoveMarker | None:
    """读出标记。**读不出来就当没有**，并且不抛。

    这个值在注册表里，可以被手改、被别的版本写坏。偏执地逐字段验一遍（是对象、
    三个键都在、类型对、路径非空）：半个标记**绝不能动盘**——一个把
    ``destination`` 读成 ``null`` 的标记，会让"搬迁"变成"往某个地方复制"。
    """
    raw = backend.value(_MARKER_KEY, "")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    source = payload.get("from")
    destination = payload.get("to")
    # **`started` 必须真的在、且真的是 bool。** 缺字段时默认成 ``False`` 看着无害，
    # 实际是**不安全的那一边**：那面旗就是"目标里那堆东西是我们自己的半成品"的唯一
    # 证据，把它默认成"没开始过"，一个残缺的库就会被当成"这里已经有一份记忆"接手。
    # 读不准就当整条标记没有——这条的容错方向只能是拒绝。
    started = payload.get("started")
    if not isinstance(source, str) or not source.strip():
        return None
    if not isinstance(destination, str) or not destination.strip():
        return None
    if not isinstance(started, bool):
        return None
    return MoveMarker(source=Path(source), destination=Path(destination), started=started)


def write_marker(backend: SettingsBackend, marker: MoveMarker) -> None:
    backend.setValue(
        _MARKER_KEY,
        json.dumps(
            {"from": str(marker.source), "to": str(marker.destination), "started": marker.started},
            ensure_ascii=False,
        ),
    )


def clear_marker(backend: SettingsBackend) -> None:
    backend.setValue(_MARKER_KEY, "")


def _has_raw_marker(backend: SettingsBackend) -> bool:
    """那一格里有非空的东西吗（不管读不读得懂）。"""
    raw = backend.value(_MARKER_KEY, "")
    return isinstance(raw, str) and bool(raw.strip())


def stage_move(backend: SettingsBackend, source: Path, destination: Path) -> None:
    """记下"下次启动时把 ``source`` 搬到 ``destination``"。**只写标记，不碰指针。**

    指针要等到复制并校验通过之后才写——那是 :func:`boot_data_root` 里唯一一次提交。
    所以这一步之后无论怎么断电，位置都还指着那份完整的源。
    """
    write_marker(
        backend, MoveMarker(source=Path(source), destination=Path(destination), started=False)
    )


@dataclass(frozen=True)
class MoveReport:
    """这一次启动在搬迁上做了什么。**给用户看的那一句在 :attr:`summary`。**

    ``None``（没有报告）表示"这次启动没有待办搬迁"——与"办了但失败了"是两件事，
    所以 :func:`boot_data_root` 的返回值是可空的。
    """

    ok: bool
    summary: str
    source: Path | None = None
    destination: Path | None = None
    offers_to_delete_source: bool = False
    """要不要问那句"旧的那份删不删"。

    只有真正复制过（:attr:`~axo.ui.data_move.MoveKind.COPY` 且成功）才为真。
    ``ADOPT`` 为假——那一支根本没复制，源不是"旧的那份"，是用户另一处的东西。
    """

    notes: tuple[str, ...] = ()


def _refused(summary: str, source: Path, destination: Path, *notes: str) -> MoveReport:
    return MoveReport(
        ok=False, summary=summary, source=source, destination=destination, notes=notes
    )


def resolve_data_root(backend: SettingsBackend) -> None:
    """**只解析**：把这次进程用哪个数据根定下来。不动盘、不读标记、不办搬迁。

    抽出来是因为**无头入口也要走同一条解析**（``--remind``，登录时跑一次，见
    :mod:`axo.headless`）：那一次要读提醒簿，而"提醒簿在哪"与主窗口是同一个问题。
    它要是自己拼一个默认位置，症状是——用户在设置里把数据目录搬到了 ``D:\\AxoData``，
    而无头入口读到的是一本**空的**提醒簿，于是**每天开机什么都不做，界面上一个字都
    不显示**。四档优先级（环境变量 > 便携指针 > 注册表 > 默认）只该有一处实现，
    这一处就是它。

    **不办搬迁，也不读标记。** 迁移是跨重启的大事（先落旗、再动盘、最后提交位置），
    不该由一个登录任务顺手办掉——那会让"搬了一半"发生在一个没人看着的进程里。
    要办搬迁的只有 :func:`boot_data_root` 那一条路。
    """
    store = SettingsStore(backend)
    stored = store.stored_data_root()
    paths.set_data_root(Path(stored) if stored else None)


def boot_data_root(
    backend: SettingsBackend,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> MoveReport | None:
    """定下这次进程用哪个数据根，并把待办的搬迁办完。返回发生了什么（没事就是 ``None``）。

    **必须在建任何 store 之前调**，而且要在建主窗口之前：主窗口一建好，四个 store
    与整张图就把 ``paths`` 现算出来的路径焊死了（见 ``ui/main_window.py:379`` 起）。
    """

    def _apply(root: Path) -> None:
        paths.set_data_root(root)

    store = SettingsStore(backend)
    resolve_data_root(backend)

    marker = read_marker(backend)
    if marker is None:
        # 有一格读不出来的东西就划掉它。它是手改过或旧版本留下的，留着不影响功能，
        # 但"标记在、却永远不生效"会让排查的人以为搬迁坏了——而真正的原因是那格内容
        # 根本解析不出来。**空值不算**（那是常态，每次都划一下没必要）。
        if _has_raw_marker(backend):
            clear_marker(backend)
        return None

    # 环境变量在生效时**不动盘，也不清标记**。这一档最高，用户（或某个脚本）明确
    # 说了"这次用那个目录"；此时把标记划掉就等于把他点的那一下悄悄扔掉，而下次
    # 打开时环境变量很可能已经不在了。留着它，下次接着办。
    if paths.data_root_source() == "env":
        return _refused(
            "环境变量 AXO_DATA 正在生效，这次不搬。它比界面里选的位置优先。",
            marker.source,
            marker.destination,
            "搬迁还记着，下次不带那个环境变量启动时会接着办。",
        )

    plan = plan_move(marker.source, marker.destination, previous_attempt=marker.started)

    if plan.kind is MoveKind.REFUSE:
        clear_marker(backend)
        return _refused(plan.summary, marker.source, marker.destination, "这次没有动任何文件。")

    if plan.kind is MoveKind.ADOPT:
        store.save_data_root(plan.destination)
        _apply(plan.destination)
        clear_marker(backend)
        return MoveReport(
            ok=True,
            summary=plan.summary,
            source=plan.source,
            destination=plan.destination,
            notes=(f"现在用的是 {plan.destination}，设置里也记下了。",),
        )

    # **先落旗，再动盘。** 反过来（复制完才落旗）的话，复制中途断电会留下一个
    # 看起来"从没开始过"的目标，而它里面全是半成品——下次启动时它会撞上
    # "目标非空且没有库"那条拒绝，于是用户被卡在一个自己解不开的状态里。
    if not marker.started:
        write_marker(
            backend,
            MoveMarker(source=marker.source, destination=marker.destination, started=True),
        )

    result: MoveResult = execute_move(plan, on_progress=on_progress)

    if not result.ok:
        clear_marker(backend)
        return MoveReport(
            ok=False,
            summary=result.reason,
            source=plan.source,
            destination=plan.destination,
            notes=result.notes,
        )

    # ★ **唯一的提交点。** 上面所有步骤都不改位置，就这一行改。
    store.save_data_root(plan.destination)
    _apply(plan.destination)
    clear_marker(backend)
    return MoveReport(
        ok=True,
        summary=f"记忆与数据已经搬到 {plan.destination} 了。",
        source=plan.source,
        destination=plan.destination,
        offers_to_delete_source=True,
        notes=(
            f"一共 {result.copied_files} 个文件、{_megabytes(result.copied_bytes)}。",
            f"原来那份还在 {plan.source}，一个字节都没动。",
        ),
    )


def _megabytes(count: int) -> str:
    return f"{count / 1024**2:.1f} MB" if count >= 1024**2 else f"{count / 1024:.0f} KB"


def discard_source(root: Path | str, *, current_root: Path | str) -> bool:
    """删掉搬迁之后留在原处的那一份。成功返回 ``True``。

    只在用户明确回答"删掉"之后调用。三道闸，每一道挡一种"删错东西"：

    1. **它不是现在正在用的那个根。** 传错参数就是"程序把自己的数据删了"。
    2. **它看起来像一份数据根**（有 ``memory/mem.db``）。否则那可能是个普通的文件夹，
       而它只是恰好被写在标记里。
    3. **当前根不在它里面。** 现在的根若是它的子目录，删它就是删当前根。

    第 2 条并不是多余的谨慎：这个函数唯一的调用点是"要不要删掉旧的"那一问，而那
    一问出现在一次**成功复制之后**——它删错东西的代价是把用户还没确认过的东西抹掉。
    """
    root = Path(root)
    current = Path(current_root)

    if root == current:
        return False
    if not (root / "memory" / "mem.db").is_file():
        return False
    try:
        current.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        pass
    else:
        return False

    shutil.rmtree(root, ignore_errors=True)
    return not root.exists()


def data_root_description(source: str) -> str:
    """:func:`~axo.paths.data_root_source` 那个取值的一句话解释（界面直接贴）。

    分开写是因为界面上必须能回答"**为什么我点了没反应**"——位置对了不够，还得说清
    它为什么是那个。
    """
    if source == "env":
        return "现在由环境变量 AXO_DATA 决定，界面里改它不生效。"
    if source == "portable":
        pointer = paths.portable_pointer_path()
        name = pointer.name if pointer is not None else "axo-data.txt"
        return f"由程序文件夹旁边的 {name} 决定，界面里改它不生效。"
    if source == "stored":
        return "这是你在设置里选的位置。"
    return "这是默认位置。"
