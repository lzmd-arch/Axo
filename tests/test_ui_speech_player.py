"""``SpeechPlayer`` 与 ``TtsWorker``：播/停/说完/坏输入/丢弃迟到结果。

播放器的替身是**假输出端**（``SinkFactory``，同 ``voice.SourceFactory`` 的手法）——
真 ``QAudioSink`` 要音频设备，而 CI 上没有；注入之后这条链在没有设备的机器上照跑，
而需要设备的测试最后总是被跳过（仓里那条老纪律）。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal

from axo.audio import AudioClip
from axo.tts import SpeechResult
from axo.ui.speech_player import SpeechPlayer, TtsWorker

pytestmark = pytest.mark.gui


def _wav(seconds: float = 0.05) -> bytes:
    return AudioClip(pcm=b"\x00\x00" * int(16000 * seconds)).as_wav()


class FakeSink(QObject):
    """假输出端：记 start/stop，状态可被测试拨动。

    **必须是 QObject 子类**：``Signal()`` 只活在 QObject 上——普通类上声明的
    ``stateChanged`` 是个死属性（第一版就是这么红了四条）。
    """

    stateChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.started: list[Any] = []
        self.stops = 0
        self.volume: float | None = None
        self._state = 0  # 0=Stopped 1=Active 2=Idle

    def state(self) -> int:
        # 与真 ``QAudioSink.state()`` 同形：**是方法不是属性**（第一版就错在这）。
        return self._state

    def start(self, io: Any) -> None:
        self.started.append(io)
        self._state = 1

    def stop(self) -> None:
        self.stops += 1
        self._state = 0

    def setVolume(self, volume: float) -> None:  # noqa: N802
        self.volume = volume


@pytest.fixture
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeDevice:
    """一个"什么格式都支持"的输出设备。**测试不该依赖这台机器的音频设备。**"""

    def isNull(self) -> bool:  # noqa: N802 — 与 QAudioDevice 同形
        return False

    def isFormatSupported(self, _fmt: Any) -> bool:  # noqa: N802
        return True


def _player(sink: FakeSink, **kwargs: Any) -> SpeechPlayer:
    return SpeechPlayer(
        sink_factory=lambda _fmt, _dev: sink,
        device_factory=FakeDevice,
        **kwargs,
    )


def test_speaking_starts_the_sink_with_raw_pcm(qapp, qapp_fixture=None) -> None:
    """开播：喂进 ``QBuffer`` 的是**裸 PCM**（格式头拆掉了——``QAudioSink`` 直推 PCM）。"""
    sink = FakeSink()
    player = _player(sink)

    problem = player.speak(_wav())

    assert problem == ""
    assert sink.started, "没开播"
    buffer = sink.started[0]
    assert bytes(buffer.data().data()).startswith(b"\x00\x00"), "喂进去的不是裸 PCM"
    assert sink.volume == pytest.approx(0.8), "默认音量 80 没带上"


def test_garbage_audio_says_so_instead_of_playing(qapp) -> None:
    """坏 wav / 空体 → 一句人话，不播也不抛（同 ``asr`` 的失败纪律）。"""
    sink = FakeSink()
    player = _player(sink)

    assert player.speak(b"not a wav") != ""
    assert player.speak(b"") != ""
    assert sink.started == []


def test_stop_is_idempotent_and_never_says_finished(qapp) -> None:
    """``stop()`` 幂等；**打断不发 ``finished``**（那是"自然播完"专用）。"""
    sink = FakeSink()
    player = _player(sink)
    said: list[int] = []
    player.finished.connect(lambda: said.append(1))

    player.speak(_wav())
    player.stop()
    player.stop()

    assert sink.stops == 1, "stop 了两次就停了两次——幂等没做"
    assert said == [], "被打断也报了 finished"


def test_natural_end_emits_finished_once(qapp) -> None:
    """数据喂完（Idle）→ ``finished``；**只发一次**。"""
    sink = FakeSink()
    player = _player(sink)
    said: list[int] = []
    player.finished.connect(lambda: said.append(1))

    player.speak(_wav())
    sink._state = 2
    sink.stateChanged.emit()

    assert said == [1]
    assert player.playing is False


def test_speaking_again_stops_the_previous_one(qapp) -> None:
    """新段开播前先停旧段：两段叠播是她两句话混在一起，谁也听不清。"""
    sink = FakeSink()
    player = _player(sink)

    player.speak(_wav())
    player.speak(_wav())

    assert sink.stops == 1


def test_volume_is_clamped_and_applies_next_time(qapp) -> None:
    sink = FakeSink()
    player = _player(sink)
    player.set_volume(500)
    player.speak(_wav())
    assert sink.volume == pytest.approx(1.0)
    player.stop()
    player.set_volume(-5)
    player.speak(_wav())
    assert sink.volume == pytest.approx(0.0)


# ── TtsWorker ──────────────────────────────────────────────────────────────


class FakeBackend:
    def __init__(self, result: SpeechResult) -> None:
        self.result = result
        self.asked: list[str] = []

    def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
        self.asked.append(f"{voice}:{text}")
        return self.result


def _run(worker: TtsWorker) -> list[Any]:
    got: list[Any] = []
    worker.done.connect(got.append)
    worker.run()  # 直接跑：这个类的线程纪律由 thread_tracking 的测试守着
    return got


def test_the_worker_hands_the_receipt_over(qapp) -> None:
    worker = TtsWorker(FakeBackend(SpeechResult(wav=b"WAV")), "你好", voice="she")

    got = _run(worker)

    assert [r.wav for r in got] == [b"WAV"]


def test_a_crashing_backend_becomes_a_sentence_not_a_dead_thread(qapp) -> None:
    """合成层"约定不抛"，但真抛了（编程错误）也要变成一句人话——
    后台线程里抛出去没有人接得住，用户只会看到"没响"。"""

    class Boom:
        def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
            raise RuntimeError("炸了")

    got = _run(TtsWorker(Boom(), "你好"))

    assert len(got) == 1
    assert got[0].ok is False
    assert "炸了" in got[0].reason


def test_a_discarded_worker_keeps_its_result_to_itself(qapp) -> None:
    """``stop()`` 之后的结果**不发**（让路那一族：迟到的不许再响）。"""
    worker = TtsWorker(FakeBackend(SpeechResult(wav=b"WAV")), "你好")
    worker.stop()

    got = _run(worker)

    assert got == []
