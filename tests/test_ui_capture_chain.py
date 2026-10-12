"""截图那根线：worker 要一张图 → 界面线程截 → 交回 worker。

这一条交接横跨三处，而每一处单独看都正常：``AgentWorker.grab_screen``（worker 线程里
登记并等）、``capture_requested`` 信号（把请求送到界面线程）、``MainWindow`` 的槽
（截图并直连送回去）。任何一处断了，症状都是同一句话——"它说看不到屏幕"——而那句话
指不到断在哪一层。

**为什么不建 ``MainWindow``**（同 ``test_ui_open_settings`` 与 ``test_ui_graph_chain``）：
离屏下 QtWebEngine 一渲染就把整个进程硬杀掉。所以这里拿 ``MainWindow`` 的**未绑定方法**
配一个人造的 ``self``；塞进去的那几个属性就是那个方法碰到的全部。

**为什么不建真 ``AgentWorker``**：它要一个后端、一个注册表、一份预算——而这条链路要验的
只是"三步接没接上"。``grab_screen`` / ``provide_capture`` 这两个方法碰到的 ``self`` 上
只有三样东西（见 :class:`_FakeWorker`），所以拿未绑定方法配替身就够，而且缺了哪样会当场
``AttributeError``。
"""

from __future__ import annotations

from typing import Any

import pytest

from axo.agent.handoff import CaptureSlot
from axo.agent.tools import Capture
from axo.ui import agent_worker as agent_worker_module
from axo.ui.agent_worker import AgentWorker
from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui

SHOT = Capture(png=b"PNG", width=1600, height=900)


class _FakeSignal:
    """``capture_requested`` 那一半：``emit`` 时**就地**把界面线程会做的事做掉。

    真实现里这两步跨线程（信号排队到界面线程，结果再由界面线程直连送回来）。测试里
    顺序一样、距离为零：这条链路要验的是"接没接上"，而线程本身的事由
    ``tests/test_agent_handoff.py`` 那条真线程测试守着。
    """

    def __init__(self, on_emit: Any) -> None:
        self._on_emit = on_emit
        self.tokens: list[str] = []

    def emit(self, token: str) -> None:
        self.tokens.append(token)
        self._on_emit(token)


class _FakeWorker:
    """``AgentWorker`` 那一半：只带 ``grab_screen`` 碰得到的三样。"""

    #: 两个方法都从真类上取下来（**未绑定**），替身只补状态。
    grab_screen = AgentWorker.grab_screen
    provide_capture = AgentWorker.provide_capture

    def __init__(self, shot: Capture | None = SHOT, *, answer: bool = True) -> None:
        self._stop_requested = False
        self._capture_slot = CaptureSlot()
        self._shot = shot
        self._answer = answer
        self.given: list[tuple[str, Capture]] = []
        self.capture_requested = _FakeSignal(self._on_capture_requested)

    def _on_capture_requested(self, token: str) -> None:
        """界面线程那一侧做的事：截一张（这里是预设的），再直连送回去。"""
        if not self._answer:
            return
        assert self._shot is not None
        self.given.append((token, self._shot))
        assert self.provide_capture(token, self._shot) is True


def test_a_request_comes_back_as_a_capture() -> None:
    """整条路走通：登记 → 发信号 → 界面回话 → worker 拿到那张图。

    ``token`` 一路带过去又带回来（``handoff`` 那条身份机制），所以这里顺带钉住"两边
    用的是同一个字符串"——不是同一个的话，回话会被丢掉，而症状正是"它说没等到截图"。
    """
    worker = _FakeWorker()

    shot = worker.grab_screen()

    assert shot == SHOT
    assert len(worker.capture_requested.tokens) == 1
    assert worker.given and worker.given[0][0] == worker.capture_requested.tokens[0]


def test_a_stopped_worker_does_not_ask_for_a_picture() -> None:
    """停止之后**连问都不问**：这一轮正在被拆掉，而截一张屏是有代价的（用户得点一次头
    才走到这里，停止之后再弹一张卡片是没有道理的）。"""
    worker = _FakeWorker()
    worker._stop_requested = True

    shot = worker.grab_screen()

    assert shot.ok is False
    assert "停止" in shot.reason
    assert worker.capture_requested.tokens == [], "停了还发了截图请求"


def test_a_silent_gui_comes_back_as_a_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """界面那一侧不回话时，worker **有界地**回来并给一句人话。

    有界这一条是硬的：收尾只等 3 秒，任何一次新的等待都必须在这个预算里结束——一个
    无限等的截图会把退出变成 `0xC0000409`（同 ``CAPTURE_TIMEOUT_S`` 那段说明）。
    """
    monkeypatch.setattr(agent_worker_module, "CAPTURE_TIMEOUT_S", 0.05)
    worker = _FakeWorker(answer=False)

    shot = worker.grab_screen()

    assert shot.ok is False
    assert "没等到" in shot.reason


def test_the_window_slot_hands_the_picture_back() -> None:
    """``MainWindow._on_capture_requested``：截图（``_grab_screen``）→ 直连送回去。

    这条钉的是**那一处直连**：走信号的话，收尾路径上事件循环已经在退出、排队的东西未必
    还能被派发（同 ``_on_tool_confirm`` 那条理由），而结果是 worker 白等满 5 秒。
    """
    captured: list[tuple[str, Capture]] = []

    class _Worker:
        def provide_capture(self, token: str, shot: Capture) -> bool:
            captured.append((token, shot))
            return True

    window = type("_Window", (), {})()
    window._agent_worker = _Worker()
    window._grab_screen = lambda: SHOT  # type: ignore[attr-defined]

    MainWindow._on_capture_requested(window, "tok-1")  # type: ignore[arg-type]

    assert captured == [("tok-1", SHOT)]


def test_the_window_slot_survives_a_dead_worker() -> None:
    """这一轮已经结束（``_agent_worker`` 是 ``None``）时什么都不做，**不抛**。

    这一支真的会到：卡片上的「允许」与这一轮的收尾之间没有锁，而那个槽已经在
    ``cancel`` 之后被丢掉引用。抛出去的话，异常会从一个 Qt 槽里冲出来——PySide6 会把它
    吞掉，于是没有任何地方看得见它。
    """
    window = type("_Window", (), {})()
    window._agent_worker = None

    MainWindow._on_capture_requested(window, "tok-1")  # type: ignore[arg-type]
