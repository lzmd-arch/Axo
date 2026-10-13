"""录音与解码：**片怎么来、怎么进去、什么时候停**。

采集那一半在界面线程（``QAudioSource`` 的 ``readyRead`` 是它属主线程的信号），解码那一半
在工作线程。这个文件把两头都拿假件替换掉，量的是它们之间的几条约定——而那几条都是"在真机
上要按着麦克风才能看出来"的：

* **拿不到麦克风时要说一句话**，而不是起一个永远不响的录音（Windows 的隐私设置会整个
  拦下桌面应用，而 Qt 那边只给一个空设备、不报错）；
* **片一片不少地到识别器手里**（队列那一层，跨线程）；
* **两种收场都发 `finished`**（用户按停 / 录满上限自己停）；
* **丢弃那一遍不出结果**（用户开口说了话，这一遍不要了）。

真麦克风的声音长什么样、识别得准不准，只能在真机上按一次——这条链在路上只到"把 PCM
交出去"为止。
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from axo.asr import Transcription
from axo.ui import voice as voice_module
from axo.ui.voice import MAX_SECONDS, DecodeWorker, VoiceRecorder

pytestmark = pytest.mark.gui

#: 两片"音频"。具体字节无所谓（这一层不解释它们），用可读的字母是为了断言失败时看得懂。
ONE = b"ab"
TWO = b"cd"


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeSource:
    """一个假音频源：把回调记下来，让测试自己往里灌字节；``problem`` 非空时起不来。"""

    def __init__(self, on_chunk: Any, *, problem: str = "") -> None:
        self.on_chunk = on_chunk
        self.problem = problem
        self.stopped = 0

    def start(self) -> str:
        return self.problem

    def stop(self) -> None:
        self.stopped += 1


def _factory(problem: str = "") -> tuple[Any, list[FakeSource]]:
    """一个假的源工厂 + 它造出来的那些源（测试据此往录音里灌字节）。"""
    built: list[FakeSource] = []

    def build(on_chunk: Any) -> FakeSource:
        source = FakeSource(on_chunk, problem=problem)
        built.append(source)
        return source

    return build, built


class FakeRecognizer:
    """假识别器：只记下喂进来的字节，并按脚本回一句。"""

    def __init__(self, result: Transcription | None = None) -> None:
        self.audio = b""
        self.feeds = 0
        self.result = result or Transcription(text="帮我记一下")
        self.finished = 0

    def feed(self, pcm: bytes) -> None:
        self.feeds += 1
        self.audio += pcm

    def finish(self) -> Transcription:
        self.finished += 1
        return self.result


# ── 采集那一半 ──────────────────────────────────────────────────────────────


def test_the_chunks_reach_the_sink(qapp: Any) -> None:
    """起录之后，来的每一片都交给 ``sink``——**顺序与内容都不变**。"""
    factory, built = _factory()
    heard: list[bytes] = []
    recorder = VoiceRecorder(source_factory=factory)

    assert recorder.start(heard.append) == ""
    assert recorder.listening is True
    built[0].on_chunk(ONE)
    built[0].on_chunk(TWO)

    assert heard == [ONE, TWO]


def test_a_dead_microphone_is_a_sentence_and_no_listening(qapp: Any) -> None:
    """拿不到麦克风 → 起录返回一句人话，而且**不进入"正在听"**。

    这一支最常见的来源是 Windows 的隐私设置（它按"桌面应用"整个放行或整个拦下）。不说
    的话，用户按了按钮、界面切成"正在听"、然后永远没有下文。
    """
    factory, _built = _factory(problem="拿不到麦克风（隐私设置？）")
    recorder = VoiceRecorder(source_factory=factory)
    finished: list[str] = []
    recorder.finished.connect(lambda: finished.append("x"))

    problem = recorder.start(lambda _pcm: None)

    assert "麦克风" in problem
    assert recorder.listening is False
    assert finished == [], "没起来却发了 finished"


def test_starting_twice_does_not_open_a_second_stream(qapp: Any) -> None:
    """已经在听时再按一下**不重开一路**（那一支由调用方处理成"停"）。"""
    factory, built = _factory()
    recorder = VoiceRecorder(source_factory=factory)
    recorder.start(lambda _pcm: None)

    assert recorder.start(lambda _pcm: None) == "已经在听了"
    assert len(built) == 1


def test_stopping_finishes_once_and_releases_the_device(qapp: Any) -> None:
    """停：发一次 ``finished``、放掉设备，而且**幂等**。

    "放掉设备"要紧：`QAudioSource` 活着就占着麦克风，而"录完了麦还开着"在托盘图标上
    没有任何痕迹。
    """
    factory, built = _factory()
    recorder = VoiceRecorder(source_factory=factory)
    recorder.start(lambda _pcm: None)
    finished: list[str] = []
    recorder.finished.connect(lambda: finished.append("x"))

    recorder.stop()
    recorder.stop()

    assert finished == ["x"]
    assert built[0].stopped >= 1
    assert recorder.listening is False


def test_it_stops_itself_at_the_cap(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """录满上限**自己停**（``MAX_SECONDS``）——它是一条隐私边界，不是体验参数。

    忘了按第二次不该让麦克风一直开着。这里把上限压到几十毫秒，再转一小会儿事件循环。
    """
    monkeypatch.setattr(voice_module, "MAX_SECONDS", 0.05)
    factory, _built = _factory()
    recorder = VoiceRecorder(source_factory=factory)
    finished: list[str] = []
    recorder.finished.connect(lambda: finished.append("x"))
    recorder.start(lambda _pcm: None)

    deadline = time.monotonic() + 3.0
    while not finished and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert finished == ["x"], "到点了没自己停"
    assert recorder.listening is False
    assert MAX_SECONDS == 30, "默认上限被人改小了"


# ── 解码那一半 ──────────────────────────────────────────────────────────────


def test_the_worker_feeds_everything_then_hands_back_one_result(qapp: Any) -> None:
    """片按顺序喂完，最后交出**一个**结果。``run()`` 在这里同步跑（不起线程）。"""
    recognizer = FakeRecognizer(Transcription(text="把明天下午的会记一下"))
    worker = DecodeWorker(recognizer)  # type: ignore[arg-type]
    got: list[Any] = []
    worker.done.connect(got.append)

    worker.push(ONE)
    worker.push(TWO)
    worker.stop()
    worker.run()

    assert recognizer.audio == ONE + TWO
    assert recognizer.finished == 1
    assert len(got) == 1
    assert (got[0].text, got[0].reason) == ("把明天下午的会记一下", "")
    # **原声与转写是同一次采集的产物**（2026-10-13 起）：收成里带的 WAV 要真的
    # 解得回那两片 PCM——少带了音频的话，"模型听得见"那条路拿不到东西。
    from axo.audio import decode_wav

    clip = decode_wav(got[0].audio)
    assert clip is not None and clip.pcm == ONE + TWO


def test_a_discarded_pass_produces_no_result(qapp: Any) -> None:
    """丢弃那一遍**连结果都不算**（用户开口说了话，这一遍不要了）。

    只是"发了结果但没人理"的话，那段文字会从另一条路冒出来——而用户已经改口说别的了。
    """
    recognizer = FakeRecognizer()
    worker = DecodeWorker(recognizer)  # type: ignore[arg-type]
    got: list[Any] = []
    worker.done.connect(got.append)

    worker.push(ONE)
    worker.stop(discard=True)
    worker.run()

    assert got == []
    assert recognizer.finished == 0


def test_push_after_a_discard_is_dropped(qapp: Any) -> None:
    """丢弃之后再来的片也不进队列（界面那边可能还有一片在路上）。"""
    recognizer = FakeRecognizer()
    worker = DecodeWorker(recognizer)  # type: ignore[arg-type]
    worker.stop(discard=True)

    worker.push(ONE)
    worker.run()

    assert recognizer.audio == b""


def test_the_worker_builds_a_real_recognizer_by_default(qapp: Any) -> None:
    """不传识别器时它自己造一个（真机上的那条路）。**这里只验形状。**

    造出来的那一个在没装库／没放模型的机器上带着一句人话（``problem``），而那句话由
    ``asr`` 那一层负责；这里只确认"默认值存在、形状对"。
    """
    worker = DecodeWorker()

    assert hasattr(worker.recognizer, "feed")
    assert hasattr(worker.recognizer, "finish")
