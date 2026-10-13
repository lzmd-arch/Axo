"""录一段话，交给识别器。**采集在界面线程、解码在工作线程。**

## 为什么采集不搬进工作线程

``QAudioSource`` 的 ``readyRead`` 是它**属主线程**上的信号（Qt 的多媒体类不做跨线程
搬家），把它搬走要自己管一个事件循环与一次对象迁移——那个坑（连同它的收尾）在这里不
值得踩。采集本身极轻（每十几毫秒一次 ``memcpy``），真正的重活是解码，那一件必须离开
界面线程。

## 解码边录边喂

收尾那一次 ``finish()`` 只剩尾巴。反过来的话，一段 30 秒的录音要在最后一起解码，而那
一下足够超过收尾给线程的 3 秒预算（``thread_tracking.THREAD_WAIT_MS``）——超时的表现是
``0xC0000409``，没有回溯、没有 stderr。

## 片走队列，不走信号

``DecodeWorker.push`` 从任意线程被调，内部是一个 :class:`queue.Queue`。**不用 Qt 信号**：
这个仓库里的 worker 都不 ``moveToThread``（见 ``ui/thread_tracking.py`` 那段事故史），
所以发给它们的排队信号会在**界面线程**执行——而那正是这里要避开的那件事。与
``ApprovalSlot`` 用 ``Lock`` + ``Event`` 是同一个理由：跨线程的交接用标准库，不借 Qt 的
排队机制。

## 设备层可注入

CI 上没有音频设备，所以"从哪儿拿 PCM"是一个可替换的入口：测试塞一个喂固定字节的假源，
真机上是 :class:`_QtSource`。没有这一层的话，这条链在 CI 上只能整条跳过——而需要设备的
测试最后总是被跳过（同 ``test_layering`` 那段注释里的同一条纪律）。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..asr import SAMPLE_RATE, Recognizer, Transcription
from ..audio import AudioClip

__all__ = ["MAX_SECONDS", "DecodeWorker", "SourceFactory", "VoiceRecorder"]

MAX_SECONDS = 30
"""一次最长录多久（秒）。

**它是一条隐私边界，不是体验参数**：忘了按第二次（或者按完就走了）不该让麦克风一直开着。
到点自己停，并且照常把已经录到的部分交出去。
"""


class _AudioSource(Protocol):
    """一个"正在录"的东西。够这一层用的只有两个方法。"""

    def start(self) -> str:
        """开始录。**空串 = 成了**，否则是一句可以直接显示的人话。"""
        ...

    def stop(self) -> None:
        """停下并释放设备。**幂等。**"""
        ...


#: 造一个音频源：收一个"来了一片"的回调，返回那个源。测试用它塞假件。
SourceFactory = Callable[[Callable[[bytes], None]], _AudioSource]


class VoiceRecorder(QObject):
    """把麦克风的声音一片片交出去。**采集在界面线程。**

    它只做**采集**：拿到一片就交给 :meth:`start` 收下的那个 ``sink``（识别在别处——
    "重活离开界面线程"的当事人是调用方，它才知道后台线程的收尾清单长什么样）。

    ``finished`` 在**两种**收场都发：用户按了第二次，或者录到了 :data:`MAX_SECONDS`。
    调用方据此收尾（把解码线程叫停、把按钮复位），不必区分是哪一种。
    """

    finished = Signal()

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        source_factory: SourceFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._factory = source_factory or _qt_source_factory(sample_rate)
        self._source: _AudioSource | None = None
        self._sink: Callable[[bytes], None] | None = None
        self._cap = QTimer(self)
        self._cap.setSingleShot(True)
        self._cap.timeout.connect(self.stop)

    @property
    def listening(self) -> bool:
        return self._source is not None

    def start(self, sink: Callable[[bytes], None]) -> str:
        """起录：来的每一片交给 ``sink``。**空串 = 起好了**，否则是一句人话。

        人话那一种（拿不到麦克风）最常见的来源是 Windows 的隐私设置——它按"桌面应用"
        整个放行或整个拦下，而拦下时 Qt 这边只会给一个空设备，不报错。
        """
        if self._source is not None:
            return "已经在听了"
        source = self._factory(sink)
        problem = source.start()
        if problem:
            source.stop()
            return problem
        self._source = source
        self._sink = sink
        self._cap.start(MAX_SECONDS * 1000)
        return ""

    def stop(self) -> None:
        """结束这一遍。**幂等**；没在录时什么都不做。

        ``sink`` 那一侧（解码线程）由调用方收尾：这里只保证**不再有新的片**进来。
        """
        source, self._source = self._source, None
        self._sink = None
        self._cap.stop()
        if source is None:
            return
        source.stop()
        self.finished.emit()


class DecodeWorker(QObject):
    """把音频片喂给识别器，最后交出结果。**``run()`` 跑在工作线程里。**

    生命周期照这个仓库的惯例：``started`` 直连 :meth:`run`，调用方用 ``_track`` 接管
    那一对 ``(线程, worker)``，收尾时 ``stop()`` 让它把队列排干就退出。
    """

    done = Signal(object)
    """这一遍的收成：一个 :class:`~axo.asr.Transcription`。**只有不丢弃时才发。**"""

    def __init__(self, recognizer: Recognizer | None = None) -> None:
        super().__init__()
        self._recognizer = recognizer if recognizer is not None else Recognizer()
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stopping = threading.Event()
        self._discard = False
        #: 这一遍的**原声**（按片攒）。转写与音频是同一次采集的产物（见
        #: ``asr.Transcription.audio``），在同一个 ``run()`` 里攒完一起交出去。
        self._pcm: list[bytes] = []

    @property
    def recognizer(self) -> Recognizer:
        return self._recognizer

    def _recognizer_sample_rate(self) -> int:
        """识别器用的采样率。**与采集侧同一个数**（``SAMPLE_RATE``）：音频要与文字
        对得上，两边采样率不同的话，模型听到的是变调的你。"""
        return SAMPLE_RATE

    def push(self, pcm: bytes) -> None:
        """一片音频。**任意线程**（内部是一把线程安全的队列）。"""
        if not self._discard:
            self._queue.put(bytes(pcm))

    def stop(self, *, discard: bool = False) -> None:
        """收尾：把队列里剩下的喂完再取结果。**界面线程直接调**（同 ``ChatWorker.stop``）。

        ``discard`` 给"用户开口说了话，这一遍不要了"那条路：连结果都不必算。
        """
        self._discard = discard
        self._stopping.set()

    @Slot()
    def run(self) -> None:
        """在线程里执行。由 ``QThread.started`` 触发。"""
        while not self._stopping.is_set() or not self._queue.empty():
            try:
                piece = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._pcm.append(piece)
            self._recognizer.feed(piece)
        if self._discard:
            return
        result = self._recognizer.finish()
        clip = AudioClip(pcm=b"".join(self._pcm), sample_rate=self._recognizer_sample_rate())
        self.done.emit(Transcription(text=result.text, reason=result.reason, audio=clip.as_wav()))


def _qt_source_factory(sample_rate: int) -> SourceFactory:
    """真的那一个：``QAudioSource``。**只在有 Qt 的那条路上被调到。**"""

    def build(on_chunk: Callable[[bytes], None]) -> _AudioSource:
        return _QtSource(sample_rate=sample_rate, on_chunk=on_chunk)

    return build


class _QtSource:
    """``QAudioSource`` 的薄封装。**只在界面线程里用。**"""

    def __init__(self, *, sample_rate: int, on_chunk: Callable[[bytes], None]) -> None:
        from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

        self._on_chunk = on_chunk
        self._source: object | None = None
        self._io: object | None = None

        fmt = QAudioFormat()
        fmt.setSampleRate(sample_rate)
        fmt.setChannelCount(1)
        # 16 位有符号小端：`asr` 那边按"2 字节一个采样"切，两处必须同一个格式。
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        device = QMediaDevices.defaultAudioInput()
        if device.isNull() or not device.isFormatSupported(fmt):
            self._device = None
            return
        self._device = device
        self._format = fmt
        self._klass = QAudioSource

    def start(self) -> str:
        if self._device is None:
            return (
                "拿不到麦克风。可能是这台机器没有输入设备，或者 Windows 的"
                "「麦克风隐私设置」把桌面应用整个拦掉了。"
            )
        self._source = self._klass(self._device, self._format)
        self._io = self._source.start()
        if self._io is None:  # pragma: no cover - Qt 在设备打开失败时返回 None
            self._source = None
            return "麦克风打不开（设备可能被别的程序占着）。"
        self._io.readyRead.connect(self._pull)
        return ""

    def stop(self) -> None:
        if self._source is None:
            return
        try:
            self._source.stop()
        finally:
            # **一定要丢掉引用**：`QAudioSource` 活着就占着设备，而"录完了麦克风还开着"
            # 在托盘图标上没有痕迹（用户只能去系统设置里看）。
            self._source = None
            self._io = None

    @Slot()
    def _pull(self) -> None:
        """``readyRead``：把这一片读出来交上去。"""
        if self._io is None:
            return
        data = self._io.readAll()
        raw = bytes(data)
        if raw:
            self._on_chunk(raw)
