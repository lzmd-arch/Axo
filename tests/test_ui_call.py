"""通话接线层：开麦/关麦、半双工、插话、看门狗与挂断。

**不碰真麦克风**：``CallController`` 的两个依赖（录音器、识别 worker）都可注入。
这里量的是"谁在什么时候被叫到"，而不是 Qt 的音频设备——后者在 CI 上不存在。
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal

from axo.asr import Transcription
from axo.call import CallState
from axo.ui.call import CallController

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeRecorder:
    """只记开始/停止。``problem`` 非空时起不来。"""

    def __init__(self, *, problem: str = "") -> None:
        self.listening = False
        self.starts = 0
        self.stops = 0
        self.problem = problem

    def start(self, _sink: Any) -> str:
        self.starts += 1
        if self.problem:
            return self.problem
        self.listening = True
        return ""

    def stop(self) -> None:
        self.stops += 1
        self.listening = False


class FakeWorker(QObject):
    """``SegmentWorker`` 的信号面，``run`` 什么都不做。"""

    utterances = Signal(object)
    problem = Signal(str)
    level = Signal(int)
    speech_started = Signal()
    speech_ended = Signal()
    finished = Signal()

    def __init__(self, *, problem: str = "") -> None:
        super().__init__()
        self.recognizer = SimpleNamespace(problem=problem)
        self.pushed = 0
        self.stop_calls: list[bool] = []

    def push(self, _pcm: bytes) -> None:
        self.pushed += 1

    def stop(self, *, discard: bool = False) -> None:
        self.stop_calls.append(discard)

    def run(self) -> None:
        return


_CREATED: list[CallController] = []


@pytest.fixture(autouse=True)
def _shut_down_calls() -> Any:
    """每个测试结束都收掉通话线程。**主线程退出时还在跑的 QThread 会让进程
    0xC0000409**（同 gui 腿那条老纪律）。"""
    yield
    for controller in _CREATED:
        controller.shutdown()
    # **不清空**：``track`` 让线程走 ``deleteLater``，那一次删除要到下一轮事件循环
    # 才落地。此时如果把控制器（线程的父对象）也放掉，删除会顺着父子关系再来一次
    # ——Windows 上的表现是整个进程在下一个测试的 setup 里直接 Abort。


def build(problem: str = "") -> tuple[CallController, FakeRecorder, FakeWorker, list[tuple]]:
    recorder = FakeRecorder(problem=problem)
    worker = FakeWorker()
    controller = CallController(recorder=recorder, worker_factory=lambda: worker)
    _CREATED.append(controller)
    states: list[tuple] = []
    controller.state_changed.connect(lambda state, meta: states.append((state, meta)))
    return controller, recorder, worker, states


def test_start_opens_the_microphone_and_reports_listening(qapp: Any) -> None:
    controller, recorder, _worker, states = build()

    assert controller.start() == ""
    assert recorder.starts == 1
    assert controller.live is True
    assert controller.state is CallState.LISTENING
    assert states[-1][0] == "listening"


def test_a_microphone_that_cannot_open_ends_the_call(qapp: Any) -> None:
    controller, recorder, _worker, _states = build(problem="麦克风被系统拦下了")

    assert controller.start() == "麦克风被系统拦下了"
    assert recorder.starts == 1
    assert controller.live is False


def test_an_utterance_closes_the_microphone_and_goes_to_thinking(qapp: Any) -> None:
    controller, recorder, worker, _states = build()
    controller.start()
    got: list[Transcription] = []
    controller.utterance.connect(got.append)

    worker.utterances.emit(Transcription(text="你好"))

    assert controller.state is CallState.THINKING
    assert recorder.listening is False
    assert [one.text for one in got] == ["你好"]


def test_a_spoken_reply_keeps_the_microphone_closed_until_playback_ends(qapp: Any) -> None:
    controller, recorder, worker, _states = build()
    controller.start()
    worker.utterances.emit(Transcription(text="你好"))
    starts_before = recorder.starts

    controller.on_reply_finished(will_speak=True)
    assert controller.state is CallState.SPEAKING
    assert recorder.starts == starts_before, "她说话时麦克风必须关着"

    controller.on_playback_finished()
    assert controller.state is CallState.LISTENING
    assert recorder.listening is True


def test_a_text_only_reply_returns_to_listening_immediately(qapp: Any) -> None:
    controller, recorder, worker, _states = build()
    controller.start()
    worker.utterances.emit(Transcription(text="你好"))

    controller.on_reply_finished(will_speak=False)

    assert controller.state is CallState.LISTENING
    assert recorder.listening is True


def test_interrupt_stops_playback_and_opens_the_microphone(qapp: Any) -> None:
    controller, recorder, worker, _states = build()
    controller.start()
    worker.utterances.emit(Transcription(text="你好"))
    controller.on_reply_finished(will_speak=True)
    asked: list[str] = []
    controller.interrupt_requested.connect(lambda: asked.append("stop"))

    controller.interrupt()

    assert asked == ["stop"]
    assert controller.state is CallState.LISTENING
    assert recorder.listening is True


def test_mute_closes_and_reopens_the_microphone(qapp: Any) -> None:
    controller, recorder, _worker, _states = build()
    controller.start()

    controller.set_muted(True)
    assert recorder.listening is False
    assert controller.muted is True

    controller.set_muted(False)
    assert recorder.listening is True


def test_a_typed_turn_also_closes_the_microphone(qapp: Any) -> None:
    """通话中打字提交：麦克风也必须关，否则她一边答、一边还在录。"""
    controller, recorder, _worker, _states = build()
    controller.start()

    controller.on_user_turn_started()

    assert controller.state is CallState.THINKING
    assert recorder.listening is False


def test_the_watchdog_recovers_a_stuck_capture(qapp: Any) -> None:
    controller, recorder, _worker, _states = build()
    controller.start()
    controller._session.move(CallState.CAPTURING, time.monotonic() - 30)  # noqa: SLF001

    controller._tick()  # noqa: SLF001

    assert controller.state is CallState.LISTENING


def test_idle_hangup_ends_the_call(qapp: Any) -> None:
    controller, _recorder, _worker, _states = build()
    controller.start()
    controller._session.last_activity = time.monotonic() - 10_000  # noqa: SLF001

    controller._tick()  # noqa: SLF001

    assert controller.live is False


def test_stop_is_idempotent_and_tells_the_worker(qapp: Any) -> None:
    controller, _recorder, worker, _states = build()
    controller.start()

    controller.stop()
    controller.stop()

    assert controller.live is False
    assert worker.stop_calls == [True]


def test_a_missing_speech_model_prevents_the_call(qapp: Any) -> None:
    """模型不在时**不开麦**：录三十秒再说"没有模型"，用户会以为它一直在听。"""
    recorder = FakeRecorder()
    worker = FakeWorker(problem="语音模型加载不了：目录不在")
    controller = CallController(recorder=recorder, worker_factory=lambda: worker)
    _CREATED.append(controller)

    assert controller.start() == "语音模型加载不了：目录不在"
    assert controller.live is False
    assert recorder.starts == 0
