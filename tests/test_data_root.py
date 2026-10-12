"""启动时定位置、办搬迁、收尾：整条**跨重启**的流程。

搬迁被拆成"点击时写标记"与"下次启动时执行"两半（理由见 `axo/ui/data_root.py`），
所以这个文件量的是那两半之间的契约——而它恰好全在"什么时刻写了哪个键"上：

* 点了「更改位置」之后、重启之前，**位置指针必须还是旧的**（那一刻盘上只有源是
  完整的，指针指过去才对）。所以这里有一条专门让复制失败，然后断言指针没动。
* 复制**之前**要先落 `started` 旗，复制**成功之后**才写指针、才划掉标记。三条的
  顺序错一条，症状都是"断电之后卡在一个自己解不开的状态"。

搬迁算法本身（复制得对不对、什么时候拒绝）在 `test_data_move.py`；这里只管把它
接进"一次启动"里。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axo import paths
from axo.ui import data_root as entry
from axo.ui.data_move import MoveResult
from axo.ui.data_root import (
    MoveMarker,
    boot_data_root,
    data_root_description,
    discard_source,
    read_marker,
    stage_move,
)
from axo.ui.settings_store import SETTINGS_APP, SETTINGS_ORG

_ANCHOR = "你是 Axo 的助手。"


class FakeBackend:
    """`SettingsBackend` 的替身（同 `test_settings_store.py` 那个）。"""

    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self.data: dict[str, object] = dict(initial or {})

    def value(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 (Qt 命名)
        self.data[key] = value


@pytest.fixture(autouse=True)
def _clean_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """隔离第 4 档并复位进程内覆盖。

    两条都不能省：`data_dir()` 会 mkdir，漏了 `LOCALAPPDATA` 就真去建使用者的数据
    目录；而 `paths._override` 是模块级状态，会漏给别的测试文件
    （`test_repo_hygiene` 量的是**默认值**，覆盖不随 `delenv` 消失）。
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.delenv("AXO_DATA", raising=False)
    paths.set_data_root(None)
    yield
    paths.set_data_root(None)


def _library(root: Path) -> Path:
    """造一份最小的数据根（形状够 `plan_move` 认出来即可）。"""
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "mem.db").write_bytes(b"sqlite-bytes")
    (root / "notes.txt").write_text("用户的文件", encoding="utf-8")
    return root


# ── 没有待办时什么都不做 ────────────────────────────────────────────────────


def test_nothing_staged_means_no_report(tmp_path: Path) -> None:
    """没标记 = 没报告（与"办了但失败了"是两件事，所以返回值可空）。"""
    assert boot_data_root(FakeBackend()) is None
    assert paths.data_dir() == tmp_path / "localappdata" / "axo"


def test_a_stored_pointer_decides_where_we_start(tmp_path: Path) -> None:
    """注册表里那个会被读出来并**真的生效**——这是"界面里选完重启就换地方"。

    断言的是 `memory_db_path()`（一个派生路径），不是 `data_dir()`：被消费的是那
    10 个派生函数。
    """
    chosen = tmp_path / "chosen"

    boot_data_root(FakeBackend({"app/data_root": str(chosen)}))

    assert paths.memory_db_path() == chosen / "memory" / "mem.db"
    assert paths.data_root_source() == "stored"


# ── 搬迁：正常一路 ──────────────────────────────────────────────────────────


def test_a_staged_move_ends_up_in_the_pointer_and_on_disk(tmp_path: Path) -> None:
    """**这一条是整件事的主干。** 写标记 → 下一次启动 → 复制过去 → 指针指过去。"""
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    backend = FakeBackend()

    stage_move(backend, source, destination)
    assert "app/data_root" not in backend.data, "写标记那一步不该顺手把位置也改了"

    report = boot_data_root(backend)

    assert report is not None and report.ok is True
    assert (destination / "memory" / "mem.db").read_bytes() == b"sqlite-bytes"
    assert (destination / "notes.txt").read_text(encoding="utf-8") == "用户的文件"
    assert backend.data["app/data_root"] == str(destination)
    assert read_marker(backend) is None, "办完了标记还留着，下次会白搬一遍"
    assert paths.data_dir() == destination


def test_the_marker_is_written_before_the_copy_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**先落旗，再动盘。**

    反过来（复制完才落旗）的话，复制中途断电会留下一个看起来"从没开始过"的目标，
    而它里面全是半成品——下次启动时它撞上"目标非空且没有库"那条拒绝，于是用户被
    卡在一个自己解不开的状态里：他不知道该删哪个文件夹，界面上也不会说。

    验法是趁 `execute_move` 被调的那一刻回头看注册表。
    """
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    backend = FakeBackend()
    stage_move(backend, source, destination)

    seen: list[MoveMarker | None] = []
    real_execute = entry.execute_move

    def _peek(plan: object, **kwargs: object) -> MoveResult:
        seen.append(read_marker(backend))
        return real_execute(plan, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(entry, "execute_move", _peek)

    assert boot_data_root(backend) is not None

    assert seen and seen[0] is not None, "动盘的时候标记已经不在了"
    assert seen[0].started is True


def test_the_first_kind_of_a_staged_move_is_copy_then_adopt_on_the_second_boot(
    tmp_path: Path,
) -> None:
    """标记里那面 `started` 旗只影响分支，不影响目的地最终落在哪。

    搬完之后再启动一次（同一个 backend）不该再有任何动作——标记已经划掉了。
    """
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    backend = FakeBackend()
    stage_move(backend, source, destination)
    boot_data_root(backend)

    assert boot_data_root(backend) is None
    assert paths.data_dir() == destination


# ── 搬迁：失败与拒绝 ────────────────────────────────────────────────────────


def test_the_pointer_is_not_written_when_the_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**这一条是"任何时刻断电都不丢数据"的机器证据。**

    复制失败了 → 指针必须还是**原来那个**（此刻只有源是完整的），标记划掉（不许
    无限重试一个走不通的目标），源一个字节没动。
    """
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    backend = FakeBackend({"app/data_root": str(source)})
    stage_move(backend, source, destination)

    monkeypatch.setattr(
        entry,
        "execute_move",
        lambda plan, **_: MoveResult(
            ok=False, source=plan.source, destination=plan.destination, reason="磁盘满了"
        ),
    )

    report = boot_data_root(backend)

    assert report is not None and report.ok is False
    assert "磁盘满了" in report.summary
    assert backend.data["app/data_root"] == str(source), "复制没成功就把指针改过去了"
    assert read_marker(backend) is None
    assert (source / "memory" / "mem.db").read_bytes() == b"sqlite-bytes"


def test_a_target_with_strangers_in_it_is_refused_and_reported(tmp_path: Path) -> None:
    """目标里有别人的东西 → 拒绝、说清理由、划掉标记。"""
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    destination.mkdir()
    (destination / "photos.jpg").write_bytes(b"x")
    backend = FakeBackend()
    stage_move(backend, source, destination)

    report = boot_data_root(backend)

    assert report is not None and report.ok is False
    assert (destination / "photos.jpg").is_file(), "拒绝了却动了盘"
    assert read_marker(backend) is None


def test_a_target_that_already_has_a_library_is_adopted(tmp_path: Path) -> None:
    """目标里已经有一份记忆 → 接手，不复制。

    这条覆盖"我把库拷到 D 盘了"与"U 盘插到另一台机器"：用户要的是**用那一份**。
    """
    source = _library(tmp_path / "src")
    destination = _library(tmp_path / "dst")
    (destination / "memory" / "mem.db").write_bytes(b"other-library")
    backend = FakeBackend()
    stage_move(backend, source, destination)

    report = boot_data_root(backend)

    assert report is not None and report.ok is True
    assert backend.data["app/data_root"] == str(destination)
    assert (destination / "memory" / "mem.db").read_bytes() == b"other-library", "被覆盖了"
    assert report.offers_to_delete_source is False, "这一支没复制，不存在'旧的那份'"


def test_the_environment_variable_holds_the_move_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """环境变量在生效时：**不动盘，也不划掉标记。**

    这一档最高，说明用户（或某个脚本）明确要求"这次用那个目录"。此时把标记划掉就
    等于把他点的那一下悄悄扔掉——而下次打开时那个环境变量很可能已经不在了。留着，
    下次接着办。
    """
    source = _library(tmp_path / "src")
    destination = tmp_path / "dst"
    backend = FakeBackend()
    stage_move(backend, source, destination)
    monkeypatch.setenv("AXO_DATA", str(tmp_path / "env-root"))

    report = boot_data_root(backend)

    assert report is not None and report.ok is False
    assert not destination.exists()
    assert read_marker(backend) is not None, "标记被划掉了，用户点的那一下就没了"
    assert paths.data_root_source() == "env"


# ── 标记本身 ────────────────────────────────────────────────────────────────


def test_the_marker_round_trips(tmp_path: Path) -> None:
    backend = FakeBackend()
    marker = MoveMarker(source=tmp_path / "a", destination=tmp_path / "b", started=True)
    entry.write_marker(backend, marker)

    assert read_marker(backend) == marker


@pytest.mark.parametrize(
    "stored",
    [
        "{",  # 半个 JSON
        "[]",  # 不是对象
        '{"from": "C:/a"}',  # 缺 to
        '{"from": "", "to": "C:/b", "started": false}',  # 空路径
        '{"from": "C:/a", "to": "C:/b"}',  # 缺 started
        '{"from": "C:/a", "to": "C:/b", "started": "yes"}',  # started 不是 bool
        '{"from": 7, "to": "C:/b", "started": false}',  # 路径不是字符串
        7,  # 压根不是字符串
        ["from"],
    ],
)
def test_a_broken_marker_is_ignored(stored: object) -> None:
    """**半个标记绝不动盘。**

    一个把 `destination` 读成 `null` 的标记，会让"搬迁"变成"往某个地方复制"。
    所以逐字段验类型，验不过就当没有。
    """
    assert read_marker(FakeBackend({"app/pending_move": stored})) is None


def test_a_broken_marker_is_cleared_so_it_stops_lying(tmp_path: Path) -> None:
    """读不出来的标记**要划掉**。

    留着不影响功能，但"标记在、却永远不生效"会让排查的人以为搬迁坏了——而真正的
    原因只是那一格解析不出来。
    """
    source = _library(tmp_path / "src")
    backend = FakeBackend({"app/pending_move": '{"from": "C:/a"'})

    assert boot_data_root(backend) is None
    assert backend.data["app/pending_move"] == ""
    assert (source / "memory" / "mem.db").is_file()


def test_clearing_leaves_an_empty_string_not_a_missing_key() -> None:
    """划掉是写空串。`None` 在 `QSettings` 里会变成"删掉这个键"，而两种状态在
    读侧本来就是一回事——写空串是为了让注册表里那一格**看得见地被清过**。"""
    backend = FakeBackend()
    entry.clear_marker(backend)

    assert backend.data["app/pending_move"] == ""


def test_a_staged_marker_is_a_json_object_with_three_fields(tmp_path: Path) -> None:
    """**一个键装一个 JSON，不是三个键。** 三个字段必须一致，多一个键就多一个
    "写了一半"的中间态。"""
    backend = FakeBackend()
    stage_move(backend, tmp_path / "a", tmp_path / "b")

    payload = json.loads(str(backend.data["app/pending_move"]))
    assert set(payload) == {"from", "to", "started"}
    assert payload["started"] is False


# ── 删掉旧的那一份 ──────────────────────────────────────────────────────────


def test_discarding_the_source_removes_it(tmp_path: Path) -> None:
    source = _library(tmp_path / "src")
    current = _library(tmp_path / "dst")

    assert discard_source(source, current_root=current) is True
    assert not source.exists()


def test_the_current_root_is_never_discarded(tmp_path: Path) -> None:
    """传错参数就是"程序把自己的数据删了"——第一道闸。"""
    current = _library(tmp_path / "dst")

    assert discard_source(current, current_root=current) is False
    assert current.is_dir()


def test_something_that_is_not_a_library_is_never_discarded(tmp_path: Path) -> None:
    """没有 `memory/mem.db` 就不删。第二道闸：那一格是标记里的一段字符串，而它可能
    指向一个普通的文件夹——`shutil.rmtree` 不问里面是什么。"""
    stranger = tmp_path / "photos"
    stranger.mkdir()
    (stranger / "a.jpg").write_bytes(b"x")

    assert discard_source(stranger, current_root=tmp_path / "dst") is False
    assert (stranger / "a.jpg").is_file()


def test_a_folder_containing_the_current_root_is_never_discarded(tmp_path: Path) -> None:
    """现在的根若在它里面，删它就是删当前根——第三道闸。"""
    outer = _library(tmp_path / "outer")
    current = _library(outer / "inner")

    assert discard_source(outer, current_root=current) is False
    assert (current / "memory" / "mem.db").is_file()


def test_discarding_a_missing_folder_is_not_an_error(tmp_path: Path) -> None:
    """已经被用户删掉了 → 返回假，不抛（那一问可能在几次启动之后才被回答）。"""
    assert discard_source(tmp_path / "gone", current_root=tmp_path / "dst") is False


# ── 界面要用的那句话 ────────────────────────────────────────────────────────


def test_every_source_has_a_sentence() -> None:
    """四档各有各的解释，而且**都不为空、互不相同**。

    界面上必须能回答"为什么我点了没反应"——位置显示对了不够，还得说清它为什么是
    那个。互不相同是因为"默认位置"与"你在设置里选的"会给出同一个路径字符串，
    分不开的话用户就没法判断那个按钮该不该灰。
    """
    sentences = {
        source: data_root_description(source) for source in ("env", "portable", "stored", "default")
    }

    assert all(sentences.values())
    assert len(set(sentences.values())) == 4


def test_the_settings_names_are_the_ones_we_think() -> None:
    """`main()` 与 `MainWindow` 必须建**同一个** `QSettings`。

    两处各写一份字符串的话，改了其中一处就会变成"启动时读的是 A、运行时读写的
    是 B"，症状是"设置改了不生效"。这条把它钉成一条可断言的事实。
    """
    assert (SETTINGS_ORG, SETTINGS_APP) == ("Axo", "axo")
