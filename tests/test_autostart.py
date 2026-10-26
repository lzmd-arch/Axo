"""开机自启那条注册表值：写什么、什么时候写、绝不写的时候。

这一层里有三件事值得钉，而它们各对着一种"用户看不见"的失效：

1. **命令行怎么拼。** 路径不带引号的话，Windows 会拿第一个空格去断命令行——症状是
   "设了自启，登录时什么都没发生"，而注册表里那一行看起来完全正常。
2. **值一样就不写。** 每次启动都动那个键，会让杀软与用户都觉得这程序在做可疑的事。
   而反过来（永远不写）会让"换了安装目录之后自启指着旧路径"这件事永远修不好——所以
   这一条测的是**两个方向**。
3. **源码检出/非 Windows 上什么都不做。** 那台机器上根本没有可自启的东西，而"写了
   一个 python.exe 进去"的症状要到下次登录才出现一次。

**这里的注册表是一个假件**（``sys.modules`` 里塞一个 ``winreg``）：真跑一遍会把自启
写进**跑测试的这台机器**，而那正是本仓最不想发生的一类副作用——它不留痕迹、也不报错。
假件同时让"值一样就不写"这件事**可断言**：真注册表上"写了但值没变"是看不出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo import autostart

EXE = Path(r"C:\Program Files\Axo\axo.exe")


class FakeWinreg:
    """``winreg`` 的替身：**一个键、一个值**，够这一层用。

    真 ``winreg`` 的形状只用到这几样：``OpenKey`` / ``CreateKeyEx`` 是可用的上下文
    管理器、``QueryValueEx`` 在值不在时抛 ``FileNotFoundError``、``DeleteValue``
    同理。``writes`` 记下每一次 ``SetValueEx``——"没写"才是这一层最要紧的断言。
    """

    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 0x0002
    REG_SZ = 1

    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.writes: list[tuple[str, str]] = []
        self.deletes: list[str] = []

    # ── 键对象 ──

    class _Key:
        def __init__(self, registry: FakeWinreg) -> None:
            self._registry = registry

        def __enter__(self) -> FakeWinreg._Key:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def OpenKey(self, _root: object, _path: str, _reserved: int = 0, _access: int = 0) -> _Key:  # noqa: N802
        if self.value is None:
            raise FileNotFoundError("那个键或那个值不在")
        return FakeWinreg._Key(self)

    def CreateKeyEx(self, _root: object, _path: str, _reserved: int = 0, _access: int = 0) -> _Key:  # noqa: N802
        return FakeWinreg._Key(self)

    # ── 值 ──

    def QueryValueEx(self, _key: object, name: str) -> tuple[str, int]:  # noqa: N802
        if self.value is None or name != autostart.VALUE_NAME:
            raise FileNotFoundError(name)
        return self.value, FakeWinreg.REG_SZ

    def SetValueEx(self, _key: object, name: str, _reserved: int, _kind: int, value: str) -> None:  # noqa: N802
        self.value = value
        self.writes.append((name, value))

    def DeleteValue(self, _key: object, name: str) -> None:  # noqa: N802
        if self.value is None or name != autostart.VALUE_NAME:
            raise FileNotFoundError(name)
        self.value = None
        self.deletes.append(name)


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> FakeWinreg:
    """**一台装了假注册表的 Windows**：``sys.platform`` 与 ``sys.modules["winreg"]``
    都换掉，于是这一层真的走到读写那几行，而盘上什么都没发生。"""
    monkeypatch.setattr(sys, "platform", "win32")
    registry = FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", registry)
    return registry


# ── 那一行命令行 ────────────────────────────────────────────────────────────


def test_the_command_is_quoted() -> None:
    """**带引号，而且是全路径。**（见模块文档第 1 条）"""
    assert autostart.command_for(EXE) == f'"{EXE}" {autostart.REMIND_FLAG}'


def test_the_command_starts_with_the_quoted_path_not_something_else() -> None:
    """引号包住的必须是**路径本身**：写成 ``"exe --remind"`` 的话，整串会被当成一个
    文件名，而 Windows 会去找一个叫这个名字的程序。"""
    command = autostart.command_for(EXE)

    assert command.startswith('"') and command.split('"')[1] == str(EXE)


# ── 有没有可自启的东西 ──────────────────────────────────────────────────────


def test_a_source_checkout_has_nothing_to_autostart() -> None:
    """源码检出下 ``current_exe()`` 是 ``None``。

    ``sys.executable`` 那时是 python.exe，写进去等于让登录时开一个**没有脚本可跑的
    Python**——而那个症状只在下次登录时出现一次，且看起来与这个设置毫无关系。
    """
    assert autostart.current_exe() is None


def test_a_frozen_build_points_at_its_own_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    """冻结产物下就是 ``sys.executable``（PyInstaller 里那就是 exe 自己）。"""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(EXE))

    assert autostart.current_exe() == EXE


def test_unavailable_reason_says_which_case_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """两种"设不了"各有各的说法，而且都能设时是空串。

    设置页把那句话原样显示出来、把勾选框灰掉——**"勾了却没生效"比"勾不上"坏得多**。
    """
    monkeypatch.setattr(sys, "platform", "linux")
    assert "Windows" in autostart.unavailable_reason()

    monkeypatch.setattr(sys, "platform", "win32")
    assert "打包" in autostart.unavailable_reason()  # 源码检出

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(EXE))
    assert autostart.unavailable_reason() == ""


# ── 写与不写 ────────────────────────────────────────────────────────────────


def test_sync_writes_when_the_value_is_missing(windows: FakeWinreg) -> None:
    assert autostart.sync(autostart.command_for(EXE)) is True
    assert windows.value == autostart.command_for(EXE)


def test_sync_does_not_rewrite_the_same_value(windows: FakeWinreg) -> None:
    """**值真的一样就什么都不做。**

    这是三条对策里最要紧的一条：每次启动都动那个键，会让杀软与用户都觉得这个程序在做
    可疑的事（它天天改注册表，而界面上什么都没发生）。所以这里断言的是 ``writes``
    空——真注册表上"写了但值没变"是看不出来的，这也是这个假件存在的理由。
    """
    windows.value = autostart.command_for(EXE)

    assert autostart.sync(autostart.command_for(EXE)) is False
    assert windows.writes == []


def test_sync_repairs_a_stale_path(windows: FakeWinreg) -> None:
    """值不一样才写——**这就是"搬家之后的自愈"**（用户把 exe 挪了、或者换了安装目录）。

    不写的话，注册表里那一行会一直指着旧路径，而症状是"登录时什么都没发生"，离这次
    改动已经过去几周。
    """
    windows.value = autostart.command_for(r"C:\Old\axo.exe")

    assert autostart.sync(autostart.command_for(EXE)) is True
    assert windows.value == autostart.command_for(EXE)


def test_remove_takes_our_value_and_leaves_the_key(windows: FakeWinreg) -> None:
    """删的是**自己那一个值**，不是那个键：``Run`` 是 Windows 的，里面还住着别的程序。"""
    windows.value = autostart.command_for(EXE)

    assert autostart.remove() is True
    assert windows.value is None
    assert windows.deletes == [autostart.VALUE_NAME]


def test_removing_what_is_not_there_is_not_an_error(windows: FakeWinreg) -> None:
    """本来就没有 → 假（没动过），但**不抛**。

    调用方要的结果是"它现在不在那儿"，而那个结果已经成立了。
    """
    assert autostart.remove() is False


def test_apply_parks_everything_on_the_setting(
    windows: FakeWinreg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``apply`` 是这一层唯一的入口：勾上 → 写，取消 → 删。

    **源码检出下勾上也什么都不写**（没有可自启的东西）——这一条正是"只在用户显式勾选
    时写"那条对策的另一半：不是"用户勾了就一定写"，而是"用户勾了、而且这台机器上真的
    有那个东西，才写"。
    """
    assert autostart.apply(True) is False, "源码检出下不该写任何东西"
    assert windows.writes == []

    windows.value = autostart.command_for(EXE)
    assert autostart.apply(False) is True
    assert windows.value is None


# ── 非 Windows ──────────────────────────────────────────────────────────────


def test_a_non_windows_machine_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Windows 上每一个写操作都是"没做成"，而且**绝不 import ``winreg``**
    （那个模块在那儿根本不存在）。

    这几行要在 Linux 的 CI 上被 import 到，所以这条不是洁癖：真去 import 一次，
    整个核心腿会以一个 ImportError 的形式红在收集阶段。
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "winreg", None)  # 真 import 会当场炸

    assert autostart.is_supported() is False
    assert autostart.read_command() is None
    assert autostart.sync(autostart.command_for(EXE)) is False
    assert autostart.remove() is False
    assert autostart.apply(True) is False


# ── macOS LaunchAgent ───────────────────────────────────────────────────────


def test_macos_writes_and_removes_a_launch_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS 用 LaunchAgent 完成同一件事：一个 plist、RunAtLoad、一条 --remind。

    **不真写 ``~/Library/LaunchAgents``。** 测试给出的路径换成临时目录；真正要断言的
    是 plist 里的 ``ProgramArguments`` 与"值一样就不重写"这两个行为。
    """
    import plistlib

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(autostart, "_launch_agent_path", lambda: tmp_path / "reminders.plist")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Axo.app/Contents/MacOS/Axo"))
    command = autostart.command_for(sys.executable)

    assert autostart.is_supported() is True
    assert autostart.unavailable_reason() == ""
    assert autostart.sync(command) is True
    record = plistlib.loads((tmp_path / "reminders.plist").read_bytes())
    assert record["ProgramArguments"] == [sys.executable, autostart.REMIND_FLAG]
    assert record["RunAtLoad"] is True
    assert autostart.read_command() == command
    assert autostart.sync(command) is False, "值一样时不该重写 plist"
    assert autostart.remove() is True
    assert autostart.read_command() is None
    assert autostart.remove() is False


def test_macos_source_checkout_says_it_cannot_autostart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")

    assert autostart.current_exe() is None
    assert "打包" in autostart.unavailable_reason()
    assert autostart.apply(True) is False
