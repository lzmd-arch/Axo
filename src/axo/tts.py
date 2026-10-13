"""语音合成（TTS）的纯层：协议、请求形状、音色扫描。**不 import Qt，也不 import httpx。**

## 引擎在独立组件里

"她的声音"用**离线克隆 TTS**（零样本：一段 10–30 秒参考音定音色），而那一族引擎全在
torch 栈上——**torch 不进本仓**（``pyproject.toml`` 里那条纪律：一份召回基准测试不该
需要装 Qt 才能跑，同理一个记忆终端不该拖着深度学习框架）。所以这一族是两半：

* 这一层（纯）：说清"要发什么、回来什么、失败说哪句话"；
* :mod:`axo.llm.speech`（网络）：一个 httpx 调用端，贴 OpenAI 的 ``/v1/audio/speech``。

组件是用户自己装的独立程序（与 Ollama 同姿态）。**检测不到不出声，但要说一句人话**
（同 :mod:`axo.asr` 的三条失败纪律：这些路径跑在界面上或后台线程里，抛出去没有人
接得住，而用户看到的是"按了没反应"）。

## 音色 = 一个参考音频目录

``models/voices/<名字>/`` 下放一段参考音（wav/mp3/flac），**一个目录 = 一个音色**。
零样本克隆不需要训练：引擎拿参考音当音色模板。这与 vosk 模型同一个姿态——资产用户
自己放、程序不下载、跟着数据根走（搬迁会带上，见 ``paths.voices_dir``）。

"音色单独训练"的最轻实现就是这个形状：真要微调一个音色，产出仍然落在同一个目录结构里
（换掉参考音/加一个模型文件），调用端一个字都不用改。

## 成功与失败都是收条

:class:`SpeechResult` 与 ``Transcription`` / ``Capture`` 同一个形状：``ok`` 为真 =
有 wav；否则 ``reason`` 是**能直接显示的一句话**。失败一律不抛。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_SPEECH_ENDPOINT",
    "SUPPORTED_SAMPLE_SUFFIXES",
    "SpeechBackend",
    "SpeechResult",
    "Voice",
    "speech_payload",
    "voices_in",
]

DEFAULT_SPEECH_ENDPOINT = "http://127.0.0.1:8800"
"""组件默认监听的地址。**localhost、不设鉴权**：它与 Axo 在同一台机器上，把它暴露到
局域网上等于把"让她说什么"交给整个网络。要远程用是另一件事（那时要有 key，不在这里）。"""

SUPPORTED_SAMPLE_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg")
"""一个目录算不算"有参考音"的判据。**按后缀认**：引擎自己认哪些格式是引擎的事，
这里只要能区分"放了参考音的目录"与"空目录/放错东西的目录"。"""


@dataclass(frozen=True)
class SpeechResult:
    """一次合成的收条。**成功与失败都是它**（同 ``Transcription`` / ``Capture``）。"""

    wav: bytes = b""
    reason: str = ""
    seconds: float = 0.0
    """这段音频多长（秒）。由收下它的一方算好填进来——预算要用它（见
    ``session/budget.AUDIO_TOKENS``），而算它只需要 wav 头（``budget.wav_seconds``）。"""

    @property
    def ok(self) -> bool:
        return bool(self.wav) and not self.reason


@dataclass(frozen=True)
class Voice:
    """一个音色：``models/voices/`` 下的一个目录。"""

    name: str
    """目录名，也就是设置里选的那个名字。**不猜显示名**：用户放的是什么就叫什么。"""
    path: Path
    """参考音所在目录（引擎要整目录或其中的样本文件时用）。"""
    sample: str
    """目录里那个参考音的文件名（引擎只收单文件时用）。"""


@runtime_checkable
class SpeechBackend(Protocol):
    """一个能把"一段文字"变成"一段声音"的东西。"""

    def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
        """合成一段。**不抛**：失败进 :attr:`SpeechResult.reason`。"""
        ...


def speech_payload(text: str, *, voice: str = "", model: str = "") -> dict[str, str]:
    """``POST /v1/audio/speech`` 的请求体。**纯函数**——形状的测试不要网络。

    贴 OpenAI 的那个形状（``input`` / ``voice`` / ``response_format``），两点刻意：

    * ``response_format`` 固定 ``wav``：播放与预算都按 wav 走（``QAudioSink`` 直接吃
      PCM，而 ``budget.wav_seconds`` 认 wav 头），换 mp3 的省流量在这个本机组件的
      形态下换不来什么。
    * ``model`` **非空才写**：OpenAI 的 API 要它，自建组件多数不看；空着不发，免得
      那一侧把一个空模型名当真。
    """
    payload: dict[str, str] = {"input": text, "voice": voice, "response_format": "wav"}
    if model:
        payload["model"] = model
    return payload


def voices_in(root: Path) -> tuple[Voice, ...]:
    """扫出 ``root`` 下的音色。**不创建、不修改**；目录不存在就是空元组。

    一个目录算一个音色的判据是"里面至少有一个参考音文件"（按后缀认，见
    :data:`SUPPORTED_SAMPLE_SUFFIXES`）。空目录与放了别的东西的目录都不算——
    界面上的下拉框少一项比多一项好：多的那一项用户点了之后是引擎报错，而那个错
    看起来像我们坏了。
    """
    folder = Path(root)
    if not folder.is_dir():
        return ()
    found: list[Voice] = []
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir():
            continue
        samples = sorted(
            item.name
            for item in entry.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_SAMPLE_SUFFIXES
        )
        if not samples:
            continue
        found.append(Voice(name=entry.name, path=entry, sample=samples[0]))
    return tuple(found)
