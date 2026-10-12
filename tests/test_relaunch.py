"""重启命令怎么拼：冻结态与源码态走的是两个入口。

这个功能只有一条路会真的用到它（改完记忆位置要重启），而**拼错了的症状是"点了重启
之后程序没起来"**——那时搬迁的标记已经在注册表里，所以用户下次自己打开照样能办完，
一切看起来只是"重启那个按钮不太灵"。所以两条分支在这里各钉一遍。

`restart_application()` 本身不测：它只有一句 `QProcess.startDetached`，而真起一个进程
是另一回事（那属于交付物手验）。这里量的是**喂给它的那三个值**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo.ui.relaunch import restart_command


def test_the_frozen_build_restarts_its_own_exe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """冻结态：`sys.executable` 就是 `axo.exe`，参数原样带过去，工作目录是它自己那儿。

    工作目录不能是"当前目录"——从桌面快捷方式起、或者从别处 `cd` 过来，当前目录都
    不一样，而 onedir 产物是**相对于 exe 所在目录**找它的那些 DLL 与资源的。
    """
    exe = tmp_path / "axo" / "axo.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "argv", [str(exe), "--some-flag"])

    program, arguments, working_directory = restart_command()

    assert program == str(exe)
    assert arguments == ["--some-flag"]
    assert Path(working_directory) == exe.parent.resolve()


def test_the_source_checkout_restarts_through_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """源码态：用同一个解释器 + `-m axo.ui.app`。

    **不去猜 venv 的路径、也不要求 uv 在。** `sys.executable` 在 `uv run` 下就是那个
    venv 的解释器；而 `run.bat` 与 `python -m axo` 走的本来就是同一个 `main()`，
    所以这个入口与平时启动是同一个。
    """
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\some\venv\python.exe")

    program, arguments, _working_directory = restart_command()

    assert program == r"C:\some\venv\python.exe"
    assert arguments == ["-m", "axo.ui.app"]
