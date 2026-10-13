"""「听不听得见原声」的能力门与贴音频：本机恒假、云端问用户、发完即清。

与 `test_ui_screen_note.py` 同一个手法（未绑定方法 + 人造 self，离屏不建真窗口）。
两条纪律各有一组判据：

* **不静默降级**：勾了「听得见」却被这一路收不了（Ollama）时要说一句原因；
* **音频只活在这一轮**：`_pending_audio` 发完即清，下一轮不许捎带上一轮的声音。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from axo.llm.base import ChatMessage
from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append((role, text))


def _window(*, cloud: bool, hears: bool, audio: bytes = b"", ride: str | None = "user") -> Any:
    window = SimpleNamespace(
        _backend_settings=SimpleNamespace(is_cloud=cloud),
        _tool_settings=SimpleNamespace(audio_in=hears),
        _backend=SimpleNamespace(audio_ride_on=ride),
        _pending_audio=audio,
        chat_view=_Recorder(),
    )
    # `_attach_pending_audio` 内部会调 `self._model_can_hear()`——替身不是 MainWindow，
    # 照 test_ui_tools_wire 的手法把未绑定方法接上去（用替身重写就不是真件那一行了）。
    window._model_can_hear = lambda: MainWindow._model_can_hear(window)  # type: ignore[attr-defined]
    return window


@dataclass(frozen=True)
class _Asm:
    """`Assembly` 的替身——**必须是 frozen dataclass**：`_attach_pending_audio` 用
    `dataclasses.replace` 重建，SimpleNamespace 会当场 TypeError。"""

    messages: tuple[ChatMessage, ...] = ()


def _assembly(*rows: ChatMessage) -> Any:
    return _Asm(tuple(rows))


# ── 能力门 ─────────────────────────────────────────────────────────────────


def test_local_ollama_never_hears(qapp: Any) -> None:
    """本机恒假——**协议收不了**（官方 ChatMessage 无 audio 字段），勾了也没用。"""
    assert MainWindow._model_can_hear(_window(cloud=False, hears=True)) is False


def test_cloud_asks_the_user(qapp: Any) -> None:
    """云端问用户那一格（`/models` 不报能力，与 cloud_vision 同一困境）。"""
    assert MainWindow._model_can_hear(_window(cloud=True, hears=True)) is True
    assert MainWindow._model_can_hear(_window(cloud=True, hears=False)) is False


# ── 贴音频 ─────────────────────────────────────────────────────────────────


def test_the_recording_reaches_the_user_row(qapp: Any) -> None:
    """听得见 + 收得了：贴到**最后一条 user 行**上（原声就是用户说的话，不需要 carrier），
    **发完即清**。"""
    window = _window(cloud=True, hears=True, audio=b"WAV")
    assembly = _assembly(
        ChatMessage(role="system", content="环境"),
        ChatMessage(role="user", content="他说的"),
        ChatMessage(role="assistant", content="她说的"),
        ChatMessage(role="user", content="他说的第二句"),
    )

    rebuilt, rides = MainWindow._attach_pending_audio(window, assembly)

    assert rides is True
    assert rebuilt.messages[3].audio != (), "贴错行了——该贴最后一条 user"
    assert rebuilt.messages[1].audio == (), "第一轮那条不该被带上"
    assert window._pending_audio == b"", "发完没清——下一轮会捎带上一轮的声音"


def test_without_the_declaration_only_text_goes_out(qapp: Any) -> None:
    """没勾「听得见」→ 静默只发转写文字（设计好的默认路，没什么好解释）。"""
    window = _window(cloud=True, hears=False, audio=b"WAV")
    assembly = _assembly(ChatMessage(role="user", content="他说的"))

    rebuilt, rides = MainWindow._attach_pending_audio(window, assembly)

    assert rides is False
    assert rebuilt.messages[0].audio == ()
    assert window.chat_view.calls == []


def test_declared_but_unrefusable_says_why(qapp: Any) -> None:
    """**勾了却被这一路收不了**（Ollama）→ 说一句原因。刚做完一个声明就被静默降级，
    界面上必须留下为什么（同「工具没出现要解释」那条纪律）。"""
    window = _window(cloud=False, hears=True, audio=b"WAV")
    assembly = _assembly(ChatMessage(role="user", content="他说的"))

    rebuilt, rides = MainWindow._attach_pending_audio(window, assembly)

    assert rides is False
    assert rebuilt.messages[0].audio == ()
    said = [text for role, text in window.chat_view.calls if role == "system"]
    assert said and "原声没有发出去" in said[0]
    assert "Ollama" in said[0], "没说清是哪一路收不了"
