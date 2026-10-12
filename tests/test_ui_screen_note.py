"""「这一句要看屏幕，而这一轮没有那个工具」时，界面要**说清为什么**。

这个文件是 2026-10-12 一次真实困惑的产物：用户做完了屏幕识别，问「先帮我确认一下，你现在
能看到我的屏幕吗」，模型答

    看不到。我没有屏幕、窗口这类接口——你没接给我。

**模型说的是实话**——``look_at_screen`` 只在当前模型自报看得见图时才登记，而他当时用的
是云端后端（那条协议的工具结果装不了图）。但界面上**一个字都没说为什么**，所以那句话
读起来像"这个功能根本不存在"，用户接着就得来问人。

所以 `MainWindow._explain_the_missing_screen_tool` 补一句。这个文件钉四件事：说、按后端的
两种说法、以及两处**反空转**（看得见图时不啰嗦、不是在说屏幕时不插嘴）。

**为什么不建 ``MainWindow``**（同 ``test_ui_capture_chain.py``）：离屏下 QtWebEngine 一渲染
就把整个进程硬杀掉。这里拿那个方法的**未绑定版本**配一个人造的 ``self``——它碰到的只有
五个成员，缺了哪个会当场 ``AttributeError``。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


class _Recorder:
    """对话区的替身：只记被推了什么。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append((role, text))


class _Combo:
    def __init__(self, name: str | None) -> None:
        self._name = name

    def currentData(self) -> Any:  # noqa: N802 (Qt 命名)
        return self._name


def _window(*, can_see: bool, cloud: bool, text: str, model: str | None = "qwen3-4b") -> Any:
    """那个人造 ``self``：**只有这个方法碰得到的那五样**。"""
    window = SimpleNamespace(
        _pending_text=text,
        _backend_settings=SimpleNamespace(is_cloud=cloud),
        model_combo=_Combo(model),
        chat_view=_Recorder(),
    )
    window._model_can_see = lambda: can_see  # type: ignore[attr-defined]
    return window


def _explain(window: Any) -> list[str]:
    MainWindow._explain_the_missing_screen_tool(window)  # type: ignore[arg-type]
    return [text for _role, text in window.chat_view.calls]


# ── 该说的时候说，而且说得对 ────────────────────────────────────────────────


def test_the_cloud_backend_says_that_it_is_the_protocol_not_the_feature() -> None:
    """云端：**要指到"这条协议装不了图"**，而不是含糊地说"看不见"。

    只说"看不了"的话，用户合理的下一步是去翻设置、换模型——而云端的模型列表里
    **换哪个都一样**。所以这句话必须把原因指到后端上，并给出真正能走的那一步。
    """
    said = _explain(_window(can_see=False, cloud=True, text="你现在能看到我的屏幕吗"))

    assert len(said) == 1
    assert "云端后端装不了图" in said[0]
    assert "minicpm-v" in said[0], "没给出能走的那一步"


def test_a_local_model_without_vision_is_named() -> None:
    """本机：**把当前模型的名字写进去**——"没有 vision"这句话得有个主语。"""
    said = _explain(
        _window(can_see=False, cloud=False, text="我屏幕现在是什么", model="qwen3-4b:latest")
    )

    assert len(said) == 1
    assert "qwen3-4b:latest" in said[0]
    assert "vision" in said[0]


def test_every_screen_marker_gets_the_explanation() -> None:
    """路由与界面**共用那一张词表**，所以每个说法都要走到这句解释。

    各写一张表的话，症状是"有时候解释、有时候不解释"——比不解释更难查。
    """
    from axo.agent.router import SCREEN_MARKERS

    for marker in SCREEN_MARKERS:
        said = _explain(_window(can_see=False, cloud=True, text=f"{marker}现在是什么"))
        assert said, f"「{marker}」那一句没拿到解释"


# ── 反空转：不该说的时候一个字都不说 ────────────────────────────────────────


def test_a_model_that_can_see_gets_no_excuse() -> None:
    """看得见图就没什么好解释的——这一条挡的是"永远弹一句"那种假绿。"""
    assert _explain(_window(can_see=True, cloud=True, text="我屏幕现在是什么")) == []


def test_an_ordinary_turn_is_left_alone() -> None:
    """不是在说屏幕的句子不插嘴：那句话是给"你去看一眼"这个意图准备的。

    少了这一条，一个把 ``SCREEN_MARKERS`` 判错的实现（比如把整个 ``_TASK_MARKERS``
    当成屏幕词表）也能全绿，而它会在每一轮任务里糊一句"看不了屏幕"。
    """
    assert _explain(_window(can_see=False, cloud=True, text="帮我查一下我之前记过什么")) == []
    assert _explain(_window(can_see=False, cloud=True, text="今天天气不错")) == []
