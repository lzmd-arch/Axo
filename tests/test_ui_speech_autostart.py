"""程序启动时把语音组件拉起来（``MainWindow._autostart_speech`` 那一段接线）。

**为什么不建 ``MainWindow``**：离屏下 QtWebEngine 一渲染就把整个进程硬杀掉（同
``test_ui_graph_chain.py``）。这里拿未绑定方法配一个人造 ``self``，并把 ``SpeechLauncher``
换成记名的假件——真件会往端口上发一次请求、失败时还会真起一个进程。

量的是四条：**开关没开就什么都不问**、**没配命令就不猜路径**、**拉起来了要说一句**、
**没拉起来也要说一句，而且不把自己记成"管着它"**。最后一条最容易漏：记上了的话退出时
``stop()`` 会去收一个根本不是我们拉起来的进程。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axo.ui import main_window as main_window_module
from axo.ui.main_window import MainWindow
from axo.ui.settings_store import SpeechSettings

pytestmark = pytest.mark.gui

COMMAND = "C:/组件/启动.bat"


class FakeLauncher:
    """记下构造参数；``ensure`` 按脚本回一句话，``started`` 按脚本给。"""

    made: list[dict[str, Any]] = []
    note = ""
    responds = True

    def __init__(self, command: str, *, probe: Any) -> None:
        self.command = command
        self.probe = probe
        self.stopped = 0
        FakeLauncher.made.append({"command": command, "probe": probe})

    def ensure(self) -> str:
        return FakeLauncher.note

    def stop(self) -> None:
        self.stopped += 1

    @property
    def started(self) -> bool:
        return FakeLauncher.responds


class FakeRemote:
    """持有地址的那个对象——``probe`` 是它的绑定方法，所以出生地还查得到。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def probe(self) -> bool:
        return False


def window(*, autostart: bool, command: str, base_url: str = "http://pc:9000") -> Any:
    said: list[tuple[str, str]] = []
    self = SimpleNamespace(
        _voice_settings=SpeechSettings(
            autostart=autostart, launch_command=command, base_url=base_url
        ),
        _speech_launcher=None,
        chat_view=SimpleNamespace(append_message=lambda role, text, *_: said.append((role, text))),
    )
    return self, said


@pytest.fixture(autouse=True)
def _fake_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeLauncher.made = []
    FakeLauncher.note = ""
    FakeLauncher.responds = True
    monkeypatch.setattr(main_window_module, "SpeechLauncher", FakeLauncher)
    monkeypatch.setattr(main_window_module, "RemoteSpeech", FakeRemote)


def test_nothing_happens_when_the_switch_is_off() -> None:
    """默认关。**连探都不探**——探一次也是一次请求。"""
    self, said = window(autostart=False, command=COMMAND)

    MainWindow._autostart_speech(self)

    assert FakeLauncher.made == []
    assert self._speech_launcher is None
    assert said == []


def test_an_empty_command_is_not_guessed() -> None:
    """开了开关但没配命令 = 什么都不做。**不猜路径**：猜错就是每次开机多一行错。"""
    self, said = window(autostart=True, command="   ")

    MainWindow._autostart_speech(self)

    assert FakeLauncher.made == []
    assert said == []


def test_a_started_component_is_kept_and_explained() -> None:
    """拉起来了就留着（退出时才收），并说一句——第一次合成要加载模型。"""
    self, said = window(autostart=True, command=COMMAND)

    MainWindow._autostart_speech(self)

    assert [one["command"] for one in FakeLauncher.made] == [COMMAND]
    assert self._speech_launcher is not None
    assert len(said) == 1
    assert said[0][0] == "system"
    assert "正在启动语音组件" in said[0][1]


def test_the_probe_points_at_the_configured_endpoint() -> None:
    """探活要问**用户配的那个地址**，不是默认的 localhost——组件可能跑在另一台机器上。"""
    self, _said = window(autostart=True, command=COMMAND, base_url="http://pc:9000")

    MainWindow._autostart_speech(self)

    probe = FakeLauncher.made[0]["probe"]
    assert probe.__self__.base_url == "http://pc:9000"


def test_a_failed_start_is_told_and_leaves_nothing_to_stop() -> None:
    """**没拉起来要说话，且不记成"我管着它"。** 记上了的话退出时 stop() 会去收一个
    不是我们拉起来的进程。"""
    FakeLauncher.responds = False
    FakeLauncher.note = "语音组件没能自动启动（C:/组件/启动.bat）：找不到文件"
    self, said = window(autostart=True, command=COMMAND)

    MainWindow._autostart_speech(self)

    assert self._speech_launcher is None
    assert said == [("system", FakeLauncher.note)]
