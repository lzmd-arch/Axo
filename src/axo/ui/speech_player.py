"""让她的话变成声音：播放 + 合成的接线。**采集在界面线程那一套纪律原样适用。**

## 播放：``QAudioSink`` 直推 PCM，**不建线程**

三个理由，前两个是实测逼出来的：

1. **打断要立刻**（``QMediaPlayer`` 走 ffmpeg 插件，``stop`` 常有几百毫秒尾巴，而
   打断是这条链的立论）——``QAudioSink.stop()`` 是即时的。
2. **"还在不在出声"要有一个不含糊的事实源**（``QAudioSink.state()`` 就是它）。
3. ``QAudioSink.start(QIODevice)`` 是 Qt 自己拉数据，``QBuffer`` 喂整段 PCM 即可——
   **播放没有线程，就没有收尾归宿问题**（``thread_tracking`` 那段事故史管的是 worker）。

WAV 拆包在 :mod:`axo.audio`（纯层，三家共用这一处编解码）。

## 合成：``TtsWorker`` 走既有 worker 纪律

合成是一次 HTTP 往返（克隆 TTS 第一次要加载参考音，几秒到十几秒），**必须离开界面
线程**。形状照 ``ui/voice.py`` 的 ``DecodeWorker``：``started`` 直连 ``run``、
**不 ``moveToThread``**、调用方用 ``_track`` 收 ``（线程, worker）`` 对。

**结果永远是收条**（``tts.SpeechResult``）：合成层不抛，这里兜一层 ``except Exception``
也只是为了把"编程错误"变成一句人话——后台线程里抛出去没有人接得住，而用户看到的会是
"按了没响"。

## 失败的收场只有一种：说出来

组件没在跑、播不出来、没有输出设备——一律由调用方写进对话区（``**没有出声** + 原因``），
**不静默**：静默的话，用户对着麦克风说了话、她答了文字，而"该出声的没出声"与"这个功能
本来就不出声"长得一模一样。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Protocol

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, Signal, Slot
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from ..audio import decode_wav
from ..tts import SpeechBackend, SpeechResult

__all__ = ["SinkFactory", "SpeechPlayer", "TtsWorker"]


class _SinkLike(Protocol):
    """``QAudioSink`` 上我们用到的那几格。测试替身实现这一个就够。"""

    stateChanged: Any

    def start(self, io: Any) -> None: ...

    def stop(self) -> None: ...

    def setVolume(self, volume: float) -> None: ...  # noqa: N802 (Qt 命名)


#: 造一个输出端：收 ``(格式, 设备)``，返回那个端。测试塞假件（同 ``voice.SourceFactory``）。
SinkFactory = Callable[[QAudioFormat, Any], _SinkLike]


class SpeechPlayer(QObject):
    """播一段 wav。**界面线程单例**；``stop()`` 幂等、打断即时。

    ``finished`` **只在自然播完时发**：``stop()`` 是"被打断"，那是一个不同的收场
    （调用方不需要为它做收尾——它自己的调用方就是收尾方）。
    """

    finished = Signal()

    def __init__(
        self,
        *,
        sink_factory: SinkFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sink_factory = sink_factory
        self._sink: _SinkLike | None = None
        self._buffer: Any = None
        self._volume = 80

    @property
    def playing(self) -> bool:
        return self._sink is not None

    def set_volume(self, value: int) -> None:
        """0–100。**下一次播放生效**（QAudioSink 的 setVolume 是 0..1 浮点）。"""
        self._volume = max(0, min(100, int(value)))
        if self._sink is not None:
            self._sink.setVolume(self._volume / 100)

    def speak(self, wav: bytes) -> str:
        """开播。**空串 = 起来了**，否则是一句人话（不抛，同 ``asr`` 的纪律）。

        先停掉上一段再说新的：两段叠播是最坏的形态（她的两句话混在一起，谁也听不清）。
        """
        self.stop()
        clip = decode_wav(wav)
        if clip is None or clip.empty:
            return "这段声音播不了（认不出它的格式，或者它是空的）"
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            return "没有音频输出设备。可能是这台机器没接扬声器/耳机，或者系统的输出被独占了。"
        fmt = QAudioFormat()
        fmt.setSampleRate(clip.sample_rate)
        fmt.setChannelCount(clip.channels)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            return f"这台机器不支持这个声音格式（{clip.sample_rate} Hz / {clip.channels} 声道）。"

        buffer = QBuffer()
        buffer.setData(QByteArray(clip.pcm))
        if not buffer.open(QIODevice.OpenModeFlag.ReadOnly):
            return "声音缓冲开不出来（内存不够？）"
        sink = self._make_sink(fmt, device)
        sink.setVolume(self._volume / 100)
        sink.stateChanged.connect(self._on_state_changed)
        sink.start(buffer)
        self._sink, self._buffer = sink, buffer
        return ""

    def stop(self) -> None:
        """打断：**立刻、幂等**（同 ``VoiceRecorder.stop``）。不发 ``finished``。"""
        sink, self._sink = self._sink, None
        self._buffer = None
        if sink is not None:
            # Qt 对象在极少数路径上已死（包装器被回收），那时 stop 无从谈起——跳过即可。
            with contextlib.suppress(RuntimeError):
                sink.stop()

    def _make_sink(self, fmt: QAudioFormat, device: Any) -> _SinkLike:
        if self._sink_factory is not None:
            return self._sink_factory(fmt, device)
        return QAudioSink(device, fmt, self)

    @Slot()
    def _on_state_changed(self, *_args: Any) -> None:
        """播到头 = 停下来。**只有这一种收场发 ``finished``**（被打断时 ``_sink``
        已经是 ``None``，这里进不来）。"""
        sink = self._sink
        if sink is None:
            return
        state = getattr(sink, "state", lambda: None)()
        # QAudioSink.State：Active=1 Idle=2 Stopped=0 Suspended=3。**Idle = 数据喂完了**，
        # 那就是"自然播完"；Stopped 在这里是被打断（上面已置空）或设备消失。
        if state == 2:
            self._sink, self._buffer = None, None
            self.finished.emit()


class TtsWorker(QObject):
    """一次合成（HTTP 到组件）。``run()`` 在工作线程里执行，**不 ``moveToThread``**。

    生命周期照 ``DecodeWorker``：``started`` 直连 ``run``，调用方 ``_track`` 接管对。
    ``stop()`` = 这一遍结果不要了（让路那一族：迟到的 ``done`` 不许落地）。
    """

    done = Signal(object)

    def __init__(self, backend: SpeechBackend, text: str, voice: str = "") -> None:
        super().__init__()
        self._backend = backend
        self._text = text
        self._voice = voice
        self._discarded = False

    def stop(self) -> None:
        """丢掉这一遍（HTTP 已在飞就随它去，结果不落地）。**幂等。**"""
        self._discarded = True

    @Slot()
    def run(self) -> None:
        """在线程里执行。由 ``QThread.started`` 触发。"""
        try:
            result = self._backend.synthesize(self._text, voice=self._voice)
        except Exception as exc:  # noqa: BLE001 - 后台线程里抛出去没有人接得住
            result = SpeechResult(reason=f"合成声音时出了错：{type(exc).__name__}: {exc}")
        if self._discarded:
            return
        self.done.emit(result)
