"""分句流式语音：一句完整了就合成、就播，一条回复一条回执。"""

from __future__ import annotations

import time
from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal

from axo.tts import SpeechResult
from axo.ui.speech_pipeline import SpeechPipeline

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakePlayer(QObject):
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.played: list[bytes] = []
        self.stopped = 0
        self.problem = ""

    def speak(self, wav: bytes) -> str:
        if self.problem:
            return self.problem
        self.played.append(wav)
        return ""

    def stop(self) -> None:
        self.stopped += 1


class FakeBackend:
    def __init__(self, *, fail_on: str = "") -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
        self.calls.append(text)
        if self.fail_on and self.fail_on in text:
            return SpeechResult(reason="组件没在跑")
        return SpeechResult(wav=b"WAV:" + text.encode("utf-8"))


_CREATED: list[SpeechPipeline] = []


@pytest.fixture(autouse=True)
def _shut_down_pipelines() -> Any:
    yield
    for pipeline in _CREATED:
        pipeline.shutdown()
    # **不清空**：线程走 ``deleteLater``，下一次事件循环才删；此时把父对象放掉会
    # 让删除再来一次，Windows 上表现为下一个测试 setup 里直接 Abort。


def make() -> tuple[SpeechPipeline, FakePlayer, FakeBackend]:
    player = FakePlayer()
    backend = FakeBackend()
    pipeline = SpeechPipeline(backend, player)  # type: ignore[arg-type]
    _CREATED.append(pipeline)
    return pipeline, player, backend


def wait_until(qapp: Any, predicate: Any, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_first_complete_sentence_is_spoken_before_the_rest_arrives(qapp: Any) -> None:
    pipeline, player, backend = make()
    try:
        pipeline.feed("第一句。第二")

        assert wait_until(qapp, lambda: player.played)
        assert player.played[0].startswith(b"WAV:")
        assert backend.calls == ["第一句。"]
    finally:
        pipeline.shutdown()


def test_the_next_sentence_is_already_synthesising_while_one_plays(qapp: Any) -> None:
    """**预取**：播放第 N 句时，第 N+1 句已经在合成。"""
    pipeline, player, backend = make()
    try:
        pipeline.feed("第一句。第二句。")
        assert wait_until(qapp, lambda: player.played)
        assert wait_until(qapp, lambda: backend.calls == ["第一句。", "第二句。"])

        player.finished.emit()
        assert wait_until(qapp, lambda: len(player.played) == 2)
        assert player.played[1] == "WAV:第二句。".encode()
    finally:
        pipeline.shutdown()


def test_finish_flushes_a_tail_without_punctuation(qapp: Any) -> None:
    pipeline, player, _backend = make()
    try:
        pipeline.feed("没有标点")
        qapp.processEvents()
        assert player.played == [], "没有句末标点就不该先念半句"

        pipeline.finish()

        assert wait_until(qapp, lambda: player.played)
        assert player.played[0] == "WAV:没有标点".encode()
    finally:
        pipeline.shutdown()


def test_a_synthesis_failure_is_reported_once(qapp: Any) -> None:
    player = FakePlayer()
    backend = FakeBackend(fail_on="第一句")
    pipeline = SpeechPipeline(backend, player)  # type: ignore[arg-type]
    _CREATED.append(pipeline)
    failures: list[str] = []
    pipeline.failed.connect(failures.append)
    try:
        pipeline.speak("第一句。")

        assert wait_until(qapp, lambda: failures)
        assert failures == ["组件没在跑"]
    finally:
        pipeline.shutdown()


def test_finished_fires_once_after_every_sentence(qapp: Any) -> None:
    pipeline, player, _backend = make()
    done: list[str] = []
    pipeline.finished.connect(lambda: done.append("done"))
    try:
        pipeline.speak("第一句。第二句。")
        assert wait_until(qapp, lambda: player.played)
        player.finished.emit()
        assert wait_until(qapp, lambda: len(player.played) == 2)
        player.finished.emit()
        assert wait_until(qapp, lambda: done)
        assert done == ["done"]
    finally:
        pipeline.shutdown()


def test_cancel_clears_the_queue_and_stops_playback(qapp: Any) -> None:
    pipeline, player, _backend = make()
    finished: list[str] = []
    pipeline.finished.connect(lambda: finished.append("done"))
    try:
        pipeline.speak("第一句。第二句。")
        assert wait_until(qapp, lambda: player.played)

        pipeline.cancel()

        player.finished.emit()
        qapp.processEvents()
        assert player.stopped == 1
        assert finished == []
        assert pipeline.active is False
    finally:
        pipeline.shutdown()


def test_started_fires_only_once_for_many_sentences(qapp: Any) -> None:
    pipeline, player, _backend = make()
    started: list[str] = []
    pipeline.started.connect(lambda: started.append("x"))
    try:
        pipeline.speak("第一句。第二句。")
        assert wait_until(qapp, lambda: player.played)
        player.finished.emit()
        assert wait_until(qapp, lambda: len(player.played) == 2)

        assert started == ["x"]
    finally:
        pipeline.shutdown()
