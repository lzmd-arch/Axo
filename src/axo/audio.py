"""音频的纯层：PCM 与 WAV 的来回、时长、base64。**stdlib only，不 import Qt/httpx。**

## 为什么单列一个模块

语音对话这一族有三个消费者要同一种音频形状（16 kHz 单声道 Int16 的 PCM，出门时包成 WAV）：

* **采集**（``ui/voice.py``）：麦克风给的是裸 PCM，交给识别、也要**整段留一份**给模型听；
* **预算**（``session/budget.py``）：算"这条音频占多少 token"要先知道它多少秒；
* **播放**（``ui/speech_player.py``）：``QAudioSink`` 吃的是 PCM，所以收回来的 WAV 要拆开。

三家各自写一遍 WAV 头解析的话，错一处的症状是**莫名其妙的噪音或预算算飞**——所以
编解码只此一处。stdlib 的 ``wave`` 不收非标准 WAV（有些引擎出的头里有未知 chunk），
所以这里手写解析（RIFF 手遍），**写用 ``wave``**（写出的头是标准的）。

## 成功与失败同形

``decode_wav`` 认不出来返回 ``None``（同 ``asr._text_of`` 对坏 JSON 的姿态）：这条路径上
"认不出"是常态（引擎回了个 mp3、回了半段），各调用方要的收场一致——**当没有这段音频**，
而不是抛。
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

__all__ = ["AudioClip", "decode_wav", "wav_seconds"]


@dataclass(frozen=True)
class AudioClip:
    """一段音频：**裸 PCM** + 采样率/声道/位宽。"""

    pcm: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    """一个采样几个字节（2 = 16 位）。与 ``asr._SAMPLE_WIDTH`` 同一个数——采集、识别、
    预算三家按同一个格式切，错开一个字节的症状是"识别出一串莫名其妙的字"。"""

    @property
    def duration_s(self) -> float:
        """多少秒。PCM 而不是 WAV 头：这个对象手里就是裸数据。"""
        frame = self.channels * self.sample_width
        return len(self.pcm) / (self.sample_rate * frame) if frame else 0.0

    @property
    def empty(self) -> bool:
        return not self.pcm

    def as_wav(self) -> bytes:
        """包成 WAV 字节（标准头，手写解析的那一侧认它）。"""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(self.channels)
            writer.setsampwidth(self.sample_width)
            writer.setframerate(self.sample_rate)
            writer.writeframes(self.pcm)
        return buffer.getvalue()

    def as_base64(self) -> str:
        """WAV 的 base64（**不带 ``data:`` 前缀**——同 ``ChatMessage.images`` 的约定）。"""
        return base64.b64encode(self.as_wav()).decode("ascii")


def decode_wav(data: bytes) -> AudioClip | None:
    """WAV 字节 → :class:`AudioClip`。**认不出来返回 ``None``，不抛**（见模块文档）。

    用 ``wave`` 读：我们自己写的、以及主流引擎出的 WAV 它都认。**异常要收得比想象宽**
    （``wave.Error`` / ``EOFError`` / ``ValueError`` / 裸 ``RuntimeError``——实测四类
    都从坏输入里冒过）：抛出去的话它会在后台线程里炸出来，那里没有人接得住。真遇到它也摇头的
    （ffmpeg 变体的非标准头），那种音频本来也喂不进 ``QAudioSink``——两个消费者给出
    同一个答案，比其中一个认、另一个不认好。
    """
    if not data:
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            pcm = reader.readframes(reader.getnframes())
            return AudioClip(
                pcm=pcm,
                sample_rate=reader.getframerate(),
                channels=reader.getnchannels(),
                sample_width=reader.getsampwidth(),
            )
    except (wave.Error, EOFError, ValueError, RuntimeError):
        return None


def wav_seconds(data: bytes) -> float | None:
    """WAV 字节 → 秒数。**认不出来返回 ``None``**（预算那边有自己的回落）。"""
    clip = decode_wav(data)
    return None if clip is None else clip.duration_s
