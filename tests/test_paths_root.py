"""数据根**听谁的**：四档优先级的次序，以及每一档的坏输入怎么办。

`paths.data_dir()` 原来只有两档（环境变量 → 默认），改位置只能靠设环境变量。
现在多出两档（程序目录旁的便携指针文件、界面里选过并存进注册表的那个），于是
"谁赢"从一句话变成一张表——而**优先级写反了不会报错**，只会让用户在一个地方
改了、程序在另一个地方读写。所以这里逐档钉一遍。

位置本身的**派生**（`memory/`、`sessions/`、`wallpapers/` 这些）不归这里管，
`test_repo_hygiene.py` 管的是"默认值不在仓库里"，而"搬迁"是另一件事
（`test_data_move.py`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo import paths


@pytest.fixture(autouse=True)
def _reset_root() -> None:
    """每个用例前后都把进程内覆盖清掉。

    **这不是可选的。** `paths._override` 是模块级状态，而
    `test_repo_hygiene.py::test_default_data_dir_is_outside_the_repo` 量的是
    **默认值**——它 `delenv("AXO_DATA")` 之后再问 `data_dir()`，覆盖却不会随之
    消失。漏了这条 fixture，一次泄漏就能让那条守卫对着一个临时目录点头。
    """
    paths.set_data_root(None)
    yield
    paths.set_data_root(None)


@pytest.fixture(autouse=True)
def _isolate_local_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """兜住第 4 档，免得哪个用例漏设覆盖时**真的**去动使用者的数据目录。

    `data_dir()` 会 `mkdir`，所以"没设任何覆盖"在这台机器上不是无害的：它会建出
    真的 `%LOCALAPPDATA%\\axo`。
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AXO_DATA", raising=False)


def _default_root(tmp_path: Path) -> Path:
    """当前平台的默认根。Windows 看 ``LOCALAPPDATA``，macOS 看 HOME。"""
    if sys.platform == "darwin":
        return tmp_path / "home" / "Library" / "Application Support" / "axo"
    return tmp_path / "localappdata" / "axo"


def _write_pointer(monkeypatch: pytest.MonkeyPatch, path: Path, text: str) -> None:
    """把便携指针文件放到 `path`，并让 `paths` 去那里找。

    不能真往"程序目录"里写：源码态那个目录是**仓库根**，而那里有一整族禁止写入的
    守卫（`tests/test_repo_hygiene.py`）。
    """
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(paths, "portable_pointer_path", lambda: path)


# ── 四档各一条 ──────────────────────────────────────────────────────────────


def test_the_environment_variable_wins_over_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**环境变量最高**——这条同时是守卫。

    全部测试、打包自检（`build_exe._probe`）与冒烟脚本都靠 `AXO_DATA` 做隔离。
    把它降到注册表之下，隔离会**静默失效**：那些脚本照跑，只是开始动使用者的真
    数据目录。而这件事在日志里一个字都不显示。
    """
    from_env = tmp_path / "from-env"
    _write_pointer(monkeypatch, tmp_path / "axo-data.txt", str(tmp_path / "from-file"))
    monkeypatch.setenv("AXO_DATA", str(from_env))
    paths.set_data_root(tmp_path / "from-registry")

    assert paths.data_dir() == from_env
    assert paths.data_root_source() == "env"


def test_the_portable_file_wins_over_the_stored_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """便携文件压过注册表。

    它表达的是"**这一份程序**属于那一份数据"——把整个文件夹拷到另一台机器、或者
    U 盘插到别处，跟着走的应该是它。反过来（注册表赢）就等于说这个文件没用。
    """
    from_file = tmp_path / "from-file"
    _write_pointer(monkeypatch, tmp_path / "axo-data.txt", str(from_file))
    paths.set_data_root(tmp_path / "from-registry")

    assert paths.data_dir() == from_file
    assert paths.data_root_source() == "portable"


def test_the_stored_pointer_wins_over_the_default(tmp_path: Path) -> None:
    """界面里选过的那个压过默认值——这就是"改了要生效"本身。"""
    chosen = tmp_path / "chosen"
    paths.set_data_root(chosen)

    assert paths.data_dir() == chosen
    assert paths.data_root_source() == "stored"


def test_the_default_is_used_when_nothing_is_set(tmp_path: Path) -> None:
    """什么都没设时回到平台默认：Windows ``%LOCALAPPDATA%\\axo``、
    macOS ``~/Library/Application Support/axo``。"""
    assert paths.data_dir() == _default_root(tmp_path)
    assert paths.data_root_source() == "default"


# ── 生效性：钉进去之后，派生出来的路径真的跟着走 ────────────────────────────


def test_setting_the_root_moves_every_derived_path(tmp_path: Path) -> None:
    """**这一条才是"生效了"的证据。**

    只断言 `data_dir()` 返回了新的那个是不够的：真正被消费的是那 10 个派生函数，
    而 `MainWindow.__init__` 里是它们（不是 `data_dir()`）被求值、被焊进四个
    store。所以这里必须问一个**派生**路径。
    """
    chosen = tmp_path / "chosen"
    paths.set_data_root(chosen)

    assert paths.memory_db_path() == chosen / "memory" / "mem.db"
    assert paths.sessions_dir() == chosen / "sessions"
    assert paths.wallpapers_dir() == chosen / "wallpapers"
    assert paths.reminders_path() == chosen / "reminders.jsonl"
    assert paths.snapshot_dir() == chosen / "memory" / "snapshots"


def test_data_dir_creates_the_root_but_not_the_files(tmp_path: Path) -> None:
    """建目录、**不建文件**——`memory_db_path` 那条契约还在。

    先建一个空 `mem.db` 会让"文件在但表不存在"变成一个要单独处理的中间态，
    `memcore.store.MemoryStore` 才是建库的那一个。
    """
    chosen = tmp_path / "chosen"
    paths.set_data_root(chosen)
    db = paths.memory_db_path()

    assert chosen.is_dir()
    assert db.parent.is_dir()
    assert not db.exists()


def test_resetting_puts_it_back(tmp_path: Path) -> None:
    """`set_data_root(None)` 是唯一的复位口（上面那条 fixture 也靠它）。"""
    paths.set_data_root(tmp_path / "chosen")
    assert paths.data_dir() == tmp_path / "chosen"

    paths.set_data_root(None)
    assert paths.data_dir() == _default_root(tmp_path)
    assert paths.data_root_source() == "default"


# ── 坏输入：一律当没有，绝不抛 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "data\n",  # 相对路径：含义取决于工作目录 = 一次静默的位置变更
        "\n",  # 空文件
        "# 只有注释\n",
        "",
    ],
)
def test_a_portable_file_with_no_usable_line_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
) -> None:
    """便携文件是**用户手写**的，格式不对不该让程序起不来——当没有就是了。"""
    _write_pointer(monkeypatch, tmp_path / "axo-data.txt", text)

    assert paths.portable_data_root() is None
    assert paths.data_dir() == _default_root(tmp_path)
    assert paths.data_root_source() == "default"


def test_a_missing_portable_file_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "portable_pointer_path", lambda: tmp_path / "nope.txt")

    assert paths.portable_data_root() is None


def test_comments_and_blank_lines_around_the_path_are_fine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """第一行有效内容才算数，前后的注释与空行不影响。"""
    target = tmp_path / "real"
    _write_pointer(
        monkeypatch,
        tmp_path / "axo-data.txt",
        f"# 我的记忆放这儿\n\n  {target}  \n# 下面这行不算\n{tmp_path / 'other'}\n",
    )

    assert paths.portable_data_root() == target


def test_the_pointer_for_the_program_directory_follows_the_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """冻结态挂在 exe 旁，源码态挂在仓库根——**都不是当前工作目录**。

    挂工作目录的话，从桌面快捷方式起与 `cd` 到别处再起会指向不同文件。
    """
    fake_exe = tmp_path / "axo" / "axo.exe"
    fake_exe.parent.mkdir(parents=True)
    fake_exe.touch()
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "executable", str(fake_exe))

    assert paths.portable_pointer_path() == fake_exe.resolve().parent / "axo-data.txt"
