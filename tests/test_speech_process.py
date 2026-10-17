"""随程序拉起语音组件（``axo.ui.speech_process``）。

三条判据各对应一种真实的坏结果：**先探再拉**（否则起第二个、抢不到端口）、**只收自己
拉的**（用户手动开的那个不许动）、**拉不起来要说**（沉默会让"自动启动开了"与"其实
没起来"在界面上长得一模一样）。
"""

from __future__ import annotations

from typing import Any

from axo.ui.speech_process import CREATE_FLAGS, SpeechLauncher


class FakeProcess:
    """够用的 ``Popen`` 替身：只被问 ``pid`` 与 ``poll()``。"""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -1


def launcher(
    command: str = r"C:\组件\启动.bat",
    *,
    alive: bool = False,
    process: FakeProcess | None = None,
    error: OSError | None = None,
) -> tuple[SpeechLauncher, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    spawns: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spawn(*args: Any, **kwargs: Any) -> Any:
        spawns.append((args, kwargs))
        if error is not None:
            raise error
        return process or FakeProcess()

    return SpeechLauncher(command, probe=lambda: alive, spawn=spawn), spawns


def test_an_empty_command_does_nothing() -> None:
    """没配命令 = 不自动启动。**不猜路径**：猜错一个的后果是每次开机多一行错。"""
    built, spawns = launcher("   ")

    assert built.ensure() == ""
    assert spawns == []


def test_if_something_is_already_answering_it_does_not_start_a_second() -> None:
    """**先探再拉。** 起第二个抢不到端口，然后在组件日志里留下一串与用户无关的错。"""
    built, spawns = launcher(alive=True)

    assert built.ensure() == ""
    assert spawns == []
    assert built.started is False


def test_a_dead_port_gets_the_component_started() -> None:
    built, spawns = launcher()

    assert built.ensure() == ""
    assert len(spawns) == 1
    assert built.started is True


def test_a_batch_file_goes_through_cmd() -> None:
    """``CreateProcess`` 直接执行 ``.bat`` 时参数与工作目录的处置与用户双击不一样。"""
    built, spawns = launcher()

    built.ensure()

    assert spawns[0][0][0] == ["cmd", "/c", r"C:\组件\启动.bat"]


def test_an_exe_is_started_directly() -> None:
    built, spawns = launcher(r"C:\组件\speech.exe")

    built.ensure()

    assert spawns[0][0][0] == [r"C:\组件\speech.exe"]


def test_it_starts_without_a_console_window_and_in_its_own_group() -> None:
    """``CREATE_NO_WINDOW`` 不弹控制台；``CREATE_NEW_PROCESS_GROUP`` 是整棵树能收掉的前提。"""
    built, spawns = launcher()

    built.ensure()

    assert spawns[0][1]["creationflags"] == CREATE_FLAGS
    assert spawns[0][1]["stdin"] is not None  # DEVNULL


def test_a_failure_to_start_is_told_not_swallowed() -> None:
    built, _spawns = launcher(error=OSError("找不到文件"))

    note = built.ensure()

    assert "没能自动启动" in note and "找不到文件" in note
    assert built.started is False


def test_stop_does_nothing_when_we_never_started_it() -> None:
    """**用户自己开着的那个不许动。**"""
    built, _spawns = launcher()

    built.stop()  # 不抛、不做任何事

    assert built.started is False


def test_started_goes_false_once_the_process_exits() -> None:
    process = FakeProcess()
    built, _spawns = launcher(process=process)
    built.ensure()

    process.returncode = 1

    assert built.started is False
