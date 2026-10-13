"""语音识别：把一段 PCM 变成一句话。**不 import Qt。**

## 引擎藏在后面

现在用的是 **vosk**（Apache-2.0）：中文小模型约 42 MB、完全离线、接口就是"喂片、取结果"。
将来换 sherpa-onnx / whisper 只动这一个模块——调用方（``ui/voice.py``）看到的是
:class:`Recognizer` 与 :class:`Transcription` 两个形状。

## 模型不打包、不自动下载；库随产物发

* **模型**（:data:`~axo.paths.VOICE_MODEL_NAME`）放在**数据根**下的 ``models/`` 里，由用户
  自己放——与"Ollama 的模型要你自己 pull"同一个姿态：发布包不因此长 42 MB，仓库里也不放
  二进制。找不到就返回一句人话（去哪儿下、放到哪个目录）。
* **库**（vosk 那个 wheel，约 13 MB）**随产物发**：冻结产物里装不了 pip 包，不随产物发的
  话，打包版的语音输入根本不可能工作。

## 三条失败各给一句人话，都不抛

库没装、模型没放、没听清——三种都走 :class:`Transcription` 的 ``reason``。调用方（界面）
对它们的处理是同一种：把那句话说出来。抛异常的话，那条路径上没有任何人接得住（识别跑在
后台线程里），而用户看到的是"按了没反应"。

## 加载模型与说话并行

构造 :class:`Recognizer` **不加载模型**，第一片 PCM 进来时才加载（几百毫秒到两秒）。这不是
省事：那时用户正在说话，而这段时间正好盖住加载——"按一下就开始听"比"先等两秒再开始听"
少丢掉开头两个字。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .audio import AudioClip
from .paths import VOICE_MODEL_NAME, voice_model_dir

__all__ = ["SAMPLE_RATE", "Recognizer", "Transcription", "recognize"]

SAMPLE_RATE = 16000
"""喂给识别器的采样率。

vosk 的小模型按 16 kHz 训练，而 Qt 那边也按这个数打开麦克风——**两处必须同一个数**。
不一致的症状不是报错，而是识别出一串莫名其妙的字。
"""

_SAMPLE_WIDTH = 2
"""一个采样几个字节（16 位有符号）。

用来判"这一片是不是半个采样"：不是的话**把剩下的那半个字节留到下一片**，而不是就地丢掉
或硬喂。差一个字节之后每一片都会错位，而症状同样是"识别出一串莫名其妙的字"。
"""


@dataclass(frozen=True)
class Transcription:
    """一次识别的收成。**成功与失败都是它**（同 :class:`~axo.agent.tools.Capture` 那个形状）。

    ``reason`` 非空 = 这一次没成，而那句话是**可以直接显示给用户的**（缺库、缺模型、
    没听清各有一句）。
    """

    text: str = ""
    reason: str = ""
    audio: bytes = b""
    """这一次采集的**原声**（WAV 字节）。**转写与音频是同一次采集的产物**，一起交出来
    ——模型听得见原声时，原声与转写文字一起走（见 ``ChatMessage.audio``）；听不见时
    音频随那一轮用完即弃，**不落盘**（与截图同姿态）。空 = 这一收成没有配音频
    （``recognize`` 那种"已经有一段"的调用方也带上；纯失败路径上它是空的）。"""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.reason


def _library_problem() -> str:
    """库装没装。装了就返回空串。

    **函数内 import**：`import axo.asr` 因此不要求装了 vosk（CI 与任何只跑核心腿的机器
    都能 import 这个模块），而"库在不在"这件事在这里变成一句人话，不是一次 ImportError。
    """
    try:
        import vosk  # noqa: F401
    except ImportError:
        return (
            "没有装语音识别库。源码检出下装它：`uv sync --extra voice`；"
            "打包版不该出现这一句（库随产物发）。"
        )
    return ""


def _model_problem(model_dir: Path) -> str:
    """模型放没放。"""
    if not model_dir.is_dir():
        return (
            f"还没有语音模型：把 {VOICE_MODEL_NAME}.zip 解压到 {model_dir} 就能用"
            "（约 42 MB，完全离线，声音不出这台机器）。"
        )
    return ""


def _text_of(raw: object) -> str:
    """vosk 的 ``FinalResult()`` 是一段 JSON 文本，取其中的 ``text``。

    读不出来一律当"没听清"（空串）——识别结果这一段坏掉的原因很多（版本差异、空输入），
    而它们的收场是同一个。
    """
    if not isinstance(raw, str):
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    text = payload.get("text") if isinstance(payload, dict) else None
    return text.strip() if isinstance(text, str) else ""


class Recognizer:
    """一次识别：**喂片、取结果**（见模块文档"加载模型与说话并行"）。

    它不是线程安全的，也不该跨两次录音复用：一次录音一个实例，用完丢掉。
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE, model_dir: Path | None = None) -> None:
        self._sample_rate = sample_rate
        self._model_dir = Path(model_dir) if model_dir is not None else voice_model_dir()
        self._problem = _library_problem() or _model_problem(self._model_dir)
        self._model: object | None = None
        self._session: object | None = None
        self._pending = b""
        self._heard = False

    @property
    def problem(self) -> str:
        """这一路走不走得通；空串 = 能走（那时模型还没加载）。"""
        return self._problem

    def feed(self, pcm: bytes) -> None:
        """喂一片 PCM（16 位小端、单声道）。**第一片进来时才加载模型。**"""
        if self._problem or not pcm:
            return
        if self._session is None:
            self._load()
            if self._problem:
                return
        data = self._pending + pcm
        aligned = len(data) - (len(data) % _SAMPLE_WIDTH)
        self._pending = data[aligned:]
        if aligned:
            self._session.AcceptWaveform(data[:aligned])  # type: ignore[attr-defined]
            self._heard = True

    def finish(self) -> Transcription:
        """收尾：把半片的尾巴丢掉，取最终结果。"""
        if self._problem:
            return Transcription(reason=self._problem)
        if self._session is None or not self._heard:
            return Transcription(reason="没有听到声音（麦克风没进来数据，或者你还没开口）")
        raw = self._session.FinalResult()  # type: ignore[attr-defined]
        text = _text_of(raw)
        if not text:
            return Transcription(reason="没有听清（离麦克风近一点，或者再说一遍）")
        return Transcription(text=text)

    def _load(self) -> None:
        """加载模型与建立识别会话。**失败归成一句人话**，不抛。

        兜的是 ``Exception``：模型目录被删了一半、版本不匹配、磁盘坏块——vosk 那层会抛出
        好几种东西，而这一层对它们的收场是同一个（"模型加载不了"）。这条路径跑在后台线程
        里，抛出去没有人接得住。
        """
        import vosk

        try:
            self._model = vosk.Model(str(self._model_dir))
            self._session = vosk.KaldiRecognizer(self._model, self._sample_rate)
        except Exception as exc:  # noqa: BLE001 — 见上面那段
            self._problem = f"语音模型加载不了：{exc}"


def recognize(
    pcm: bytes, *, sample_rate: int = SAMPLE_RATE, model_dir: Path | None = None
) -> Transcription:
    """一次性的便利入口：喂完、取结果。给测试与"已经有一段完整音频"的调用方用。

    边录边识别那条路直接用 :class:`Recognizer`（它可以在录音期间就喂）。
    """
    recognizer = Recognizer(sample_rate=sample_rate, model_dir=model_dir)
    recognizer.feed(pcm)
    result = recognizer.finish()
    # **转写与原声一起交**（见 ``Transcription.audio``）：调用方手里已有这段 PCM，
    # 不用它的话，"模型听得见"那条路要再回采集层要一遍。
    clip = AudioClip(pcm=pcm, sample_rate=sample_rate)
    return Transcription(text=result.text, reason=result.reason, audio=clip.as_wav())
