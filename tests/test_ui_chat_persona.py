"""首启人设卡片的那条写路径：写什么进库、失败时卡片留不留、话说给谁听。

**这里不验卡片长什么样**（那是 ``tools/_smoke_webview.py`` 的事，它需要显示器，
见 README「别在离屏下跑渲染路径」）。这里验的是 Python 这一侧的三件事：

1. 卡片收下的答案**真的落进记忆库**，而且 ``asking`` 随之翻掉——网页侧"能不能打字"
   就是靠那次推送解开的，所以"写进去了但没推"是一个必须被钉住的半成品状态；
2. 空串（「这一次先不写」）**什么都不写**，不是写一条空的；
3. 写失败时**卡片留着**（不推 ``asking=False``），并且说一句为什么。

**不构造真的 ``MainWindow``。** ``_apply_persona`` 与它调到的四个方法都是普通的
实例方法，所以这里给一个只带那几个属性的替身类——``test_ui_context_strip.py`` 与
``test_ui_worker_track.py`` 用的是同一套办法：把 ``MainWindow`` 的方法**赋成类属性**，
于是它们之间的相互调用（``_apply_persona`` → ``_push_persona_state``）照常工作。
替身缺了哪个属性会当场 ``AttributeError``，而不是悄悄测了别的东西。

**对话区的替身写在这里，不用 ``tools/tools_support.py`` 那个。** 那个是给冒烟脚本
（真进程）用的，而 ``tools/`` 不是包、也不在测试的 ``sys.path`` 上——为它铺一条 import
路径是让测试依赖开发脚本。这个替身只需要记住三样东西，抄一遍比接线便宜。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from biosnn_memterm.memory.persona import (
    BUILTIN_REF,
    DEFAULT_PERSONA,
    builtin_anchor,
    ensure_builtin_anchor,
    set_user_persona,
    user_anchor,
)
from biosnn_memterm.ui.main_window import MainWindow
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Recorder:
    """对话区的替身：只记下被推了什么。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.persona_state: dict[str, Any] | None = None

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append(f"appendMessage({role})")

    def set_persona_state(self, *, asking: bool, builtin: str) -> None:
        self.persona_state = {"asking": bool(asking), "builtin": builtin}
        self.calls.append(f"setPersonaState({asking})")


class FakeDialog:
    """设置窗口的替身：只记下被回填了什么。"""

    def __init__(self) -> None:
        self.persona: tuple[str, str] | None = None
        self.note: str = ""

    def set_persona(self, builtin: str, user: str) -> None:
        self.persona = (builtin, user)

    def set_persona_note(self, note: str) -> None:
        self.note = note


class Stub:
    """``_apply_persona`` 与它调到的那些方法。"""

    _persona_texts = MainWindow._persona_texts
    _push_persona_state = MainWindow._push_persona_state
    _refresh_persona_page = MainWindow._refresh_persona_page
    _apply_persona = MainWindow._apply_persona
    _on_persona_answered = MainWindow._on_persona_answered
    _on_persona_save_requested = MainWindow._on_persona_save_requested
    _on_persona_clear_requested = MainWindow._on_persona_clear_requested

    def __init__(self, store: MemoryStore, *, asking: bool, dialog: FakeDialog | None = None):
        self._memory = store
        self.chat_view = Recorder()
        self._persona_asking = asking
        self._settings_dialog = dialog


@pytest.fixture
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        ensure_builtin_anchor(opened)
        yield opened


def _system_messages(stub: Stub) -> list[str]:
    return [call for call in stub.chat_view.calls if call == "appendMessage(system)"]


# ── 卡片那条路 ──────────────────────────────────────────────────────────────


def test_an_answer_from_the_card_lands_in_the_library(store: MemoryStore, qapp: Any) -> None:
    stub = Stub(store, asking=True)

    stub._on_persona_answered("说话简短一点，叫我老离。")

    written = user_anchor(store)
    assert written is not None
    assert written.content == "说话简短一点，叫我老离。"
    assert stub._persona_asking is False
    assert stub.chat_view.persona_state == {
        "asking": False,
        "builtin": DEFAULT_PERSONA,
    }, "没推 asking=False 的话，网页侧的输入框永远解不开"


def test_skipping_the_card_writes_nothing(store: MemoryStore, qapp: Any) -> None:
    """**空串不是一个空的人设，是"没写"。**

    写成一条空正文的锚会让库里多出一条说不出话的记忆，而它还会参与下次启动的
    "要不要问"判定——于是用户选了「先不写」，卡片反而永远不再出现。
    """
    stub = Stub(store, asking=True)

    stub._on_persona_answered("")

    assert user_anchor(store) is None
    assert stub._persona_asking is False, "「先不写」也要放行，否则输入框永远解不开"
    assert _system_messages(stub) == []


def test_the_answer_is_stripped_before_it_is_stored(store: MemoryStore, qapp: Any) -> None:
    """首尾空白不留。网页侧已经把框里的字 trim 过一道，但那条路不是唯一入口
    （设置页、将来的导入都走 :meth:`_apply_persona`），所以这里再定一次。"""
    stub = Stub(store, asking=True)

    stub._on_persona_answered("  说话简短。  ")

    written = user_anchor(store)
    assert written is not None
    assert written.content == "说话简短。"


def test_a_write_that_fails_keeps_the_card_and_says_why(store: MemoryStore, qapp: Any) -> None:
    """**失败不推 ``asking=False``。**

    推了的话卡片会收起来、用户打的字也没了（网页侧清空输入框是那次推送触发的），
    而他看到的是一个正常开始说话的界面——下次启动卡片回来时他会以为是另一个 bug。
    """
    stub = Stub(store, asking=True)
    stub._push_persona_state()  # 模拟启动时那一次推送
    before = list(stub.chat_view.calls)

    stub._on_persona_answered("。。。！？")  # 全标点：can_encode 会拒绝

    assert user_anchor(store) is None
    assert stub._persona_asking is True, "卡片要留着，让他改一下再试"
    assert stub.chat_view.calls[len(before) :] == ["appendMessage(system)"], (
        "失败时只该多一条系统消息，不该有 setPersonaState"
    )
    assert stub.chat_view.persona_state == {"asking": True, "builtin": DEFAULT_PERSONA}


def test_the_builtin_text_pushed_is_the_one_in_the_library(store: MemoryStore, qapp: Any) -> None:
    """**卡片上那段只读文字读库，不填常量。**

    库里那条可能是旧版本写下的字面量（``DEFAULT_PERSONA`` 改过），而卡片要展示的
    是"现在这个模型读到的身份"。填常量的话，展示一处、库里一处，模型读到的与用户
    看到的不是同一份——而这正是这个功能存在的理由。

    构造方式：把 ensure 写下的那条删掉、换成一条正文不同的——那就是"旧版本留下的
    那条"的样子。不删的话库里会有两条内置锚，而 ``builtin_anchor`` 只返回第一条。
    """
    created = builtin_anchor(store)
    assert created is not None
    store.delete(created.memory_id)
    store.add(
        MemoryItem(
            memory_id=new_ulid(),
            kind=MemoryKind.PERSONA_ANCHOR,
            tier=Tier.LONGTERM,
            content="（旧版本写下的那三句）",
            pinned=True,
            lex_sdr=encode_lexical("旧版本写下的那三句"),
            source_ref=BUILTIN_REF,
        )
    )
    stub = Stub(store, asking=True)

    stub._push_persona_state()

    assert stub.chat_view.persona_state is not None
    assert stub.chat_view.persona_state["builtin"] == "（旧版本写下的那三句）"
    assert stub.chat_view.persona_state["builtin"] != DEFAULT_PERSONA


def test_a_second_launch_asks_nothing(store: MemoryStore, qapp: Any) -> None:
    """库里已经有用户那条时，``asking`` 是假——卡片不出现、输入框直接可用。"""
    set_user_persona(store, "说话简短。")
    stub = Stub(store, asking=False)

    stub._push_persona_state()

    assert stub.chat_view.persona_state is not None
    assert stub.chat_view.persona_state["asking"] is False


# ── 设置页那条路 ────────────────────────────────────────────────────────────


def test_saving_from_the_settings_page_reports_there_not_in_the_chat(
    store: MemoryStore, qapp: Any
) -> None:
    """设置页保存的反馈**写在它自己那一页上**。

    它其实也能往对话流里塞一条系统消息（通道是同一个 chat_view），但那时用户正
    看着设置窗口——把话说到他看不见的另一个窗口里，等于没说。
    """
    dialog = FakeDialog()
    stub = Stub(store, asking=False, dialog=dialog)

    stub._on_persona_save_requested("说话简短，不要客套话。")

    assert user_anchor(store) is not None
    assert dialog.note.startswith("已写入")
    assert _system_messages(stub) == [], "成功不需要往对话流里说话"
    assert dialog.persona is not None, "写完要回填，否则「保存」键不会变灰"


def test_clearing_from_the_settings_page_promises_to_ask_again(
    store: MemoryStore, qapp: Any
) -> None:
    """裁决 4 的原话：删掉之后**下次启动**重新问。

    所以删完这一句必须说出来——不说的话，用户下次启动看到卡片回来会以为删失败了。
    而本进程内**不重弹**：卡片住在对话流里，那时用户多半正在跟它说话。
    """
    set_user_persona(store, "说话简短。")
    dialog = FakeDialog()
    stub = Stub(store, asking=False, dialog=dialog)

    stub._on_persona_clear_requested()

    assert user_anchor(store) is None
    assert builtin_anchor(store) is not None, "删人设不该把内置那条也删了"
    assert "下次启动" in dialog.note
    assert stub._persona_asking is False, "本进程不再弹，下次启动才问"
    assert stub.chat_view.persona_state is not None
    assert stub.chat_view.persona_state["asking"] is False


def test_a_failed_save_says_so_on_the_settings_page(store: MemoryStore, qapp: Any) -> None:
    dialog = FakeDialog()
    stub = Stub(store, asking=False, dialog=dialog)

    stub._on_persona_save_requested("。。。")  # 全标点

    assert user_anchor(store) is None
    assert dialog.note.startswith("没写进去：")
    assert "可编码" in dialog.note, "界面说的理由要与 can_encode 给的同源"
