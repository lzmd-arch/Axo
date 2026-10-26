"""随包的千岁 TTS：ZipVoice 零样本克隆，CPU 上跑，不依赖外部组件。

分发版不能假设用户装好了 torch / F5-TTS / 一个 8800 端口的 sidecar。这里用
`sherpa-onnx` 的 **ZipVoice**（distill-int8，中英双语）把"文字 → 千岁的声音"做在
进程内：

* 参考音与逐字稿随包放在 ``tts/voices/qiansui/``；
* ONNX 模型约 110 MB，vocoder 54 MB，CPU 上 4 步生成约 4–5 秒一句；
* 不需要 GPU，也不联网。

## 音色不是"模型里自带的"

ZipVoice 是零样本克隆：每次合成都要一份参考音 + 它的**逐字文本**。用户放在
``models/voices/<名字>/`` 下的目录照样能用（与外部组件同一套目录约定）；不选时用
随包的千岁参考音。参考文本对不上会让音色/咬字明显变差，所以缺 ``ref.txt`` 时
**不猜**，直接回一句"这个音色缺逐字稿"。

## 许可

ZipVoice 的预训练权重按其模型卡是 **CC-BY-NC-4.0**（训练数据 Emilia 的限制）。
随包分发只用于非商业用途；包里的 ``THIRD-PARTY-NOTICES.txt`` 与
``licenses/CC-BY-NC-4.0.txt`` 是这份义务的落点。代码与 sherpa-onnx 是 Apache-2.0。
"""

from __future__ import annotations

import io
import threading
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..tts import SpeechResult

__all__ = ["LocalSpeech", "is_available"]

_MODEL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokens.txt",
    "lexicon.txt",
    "lexicon-en-us.txt",
    "vocos_24khz.onnx",
)


def is_available(assets_dir: Path | str | None) -> bool:
    """随包 TTS 资产齐不齐。坏目录当没有，不抛。"""
    if assets_dir is None:
        return False
    root = Path(assets_dir)
    return root.is_dir() and all((root / "zipvoice" / name).is_file() for name in _MODEL_FILES)


@dataclass
class LocalSpeech:
    """进程内的 ZipVoice 合成器。**延迟加载模型**，第一次合成才付那一两秒。"""

    assets_dir: Path
    voices_dir: Path
    num_steps: int = 4
    threads: int = 4
    _tts: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _gen_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.assets_dir = Path(self.assets_dir)
        self.voices_dir = Path(self.voices_dir)
        if not is_available(self.assets_dir):
            raise ValueError(f"随包 TTS 资产不完整：{self.assets_dir}")
        if not self.voices_dir.is_dir():
            self.voices_dir = self.assets_dir / "voices"

    # ── 模型 ────────────────────────────────────────────────────────────────

    def _ensure(self) -> object:
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            try:
                import sherpa_onnx
            except ImportError as exc:  # pragma: no cover - 分发版一定带
                raise RuntimeError("随包 TTS 需要 sherpa-onnx；当前环境缺它。") from exc
            root = self.assets_dir / "zipvoice"
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                        tokens=str(root / "tokens.txt"),
                        encoder=str(root / "encoder.int8.onnx"),
                        decoder=str(root / "decoder.int8.onnx"),
                        lexicon=f"{root / 'lexicon.txt'},{root / 'lexicon-en-us.txt'}",
                        vocoder=str(root / "vocos_24khz.onnx"),
                    ),
                    num_threads=max(1, int(self.threads)),
                    provider="cpu",
                )
            )
            if not config.validate():
                raise RuntimeError("随包 ZipVoice 配置校验失败。")
            self._tts = sherpa_onnx.OfflineTts(config)
            return self._tts

    # ── 音色 ────────────────────────────────────────────────────────────────

    def _reference(self, voice: str) -> tuple[Path, Path]:
        candidates: list[Path] = []
        if voice.strip():
            candidates.append(self.voices_dir / voice.strip())
        candidates.append(self.assets_dir / "voices" / "qiansui")
        for folder in candidates:
            if not folder.is_dir():
                continue
            text = folder / "ref.txt"
            samples = sorted(
                item
                for item in folder.iterdir()
                if item.is_file() and item.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")
            )
            if text.is_file() and samples:
                return samples[0], text
        raise FileNotFoundError(
            "没有可用的参考音与逐字稿。放一个 models/voices/<名字>/ref.wav + ref.txt，"
            "或者确认随包资产里带着 tts/voices/qiansui。"
        )

    def _read_audio(self, path: Path) -> tuple[object, int]:
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32")
        except ImportError as exc:  # pragma: no cover - 分发版一定带
            raise RuntimeError("随包 TTS 需要 soundfile；当前环境缺它。") from exc
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        return audio, int(sample_rate)

    # ── 合成 ────────────────────────────────────────────────────────────────

    def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
        """合成一句。**不抛**；失败进 :attr:`SpeechResult.reason`。"""
        text = text.strip()
        if not text:
            return SpeechResult(reason="没有要念的文字（这一句是空的）")
        try:
            reference, transcript = self._reference(voice)
            ref_text = transcript.read_text(encoding="utf-8").strip()
            if not ref_text:
                return SpeechResult(reason=f"{transcript} 是空的——ZipVoice 需要参考音的逐字稿。")
            audio, sample_rate = self._read_audio(reference)
            tts = self._ensure()
            import sherpa_onnx

            generation = sherpa_onnx.GenerationConfig()
            generation.reference_audio = audio
            generation.reference_sample_rate = sample_rate
            generation.reference_text = ref_text
            generation.num_steps = max(1, int(self.num_steps))
            # 同一时刻只跑一次生成：两个 TTS worker（回复 + 提醒）可能撞上，而
            # sherpa-onnx 的 OfflineTts 没有承诺并发安全。
            with self._gen_lock:
                generated = tts.generate(text, generation)
            samples = getattr(generated, "samples", None)
            output_rate = int(getattr(generated, "sample_rate", 0) or 0)
            if samples is None or len(samples) == 0 or output_rate <= 0:
                return SpeechResult(reason="随包 TTS 没有生成出声音（模型返回空结果）。")
            wav = _wav_bytes(samples, output_rate)
            return SpeechResult(wav=wav, seconds=len(samples) / output_rate)
        except Exception as exc:  # noqa: BLE001 — 音频线程里不能把异常丢给 Qt
            return SpeechResult(reason=f"随包语音合成失败：{type(exc).__name__}：{exc}")


def _wav_bytes(samples: Sequence[float], sample_rate: int) -> bytes:
    """float32 单声道 → 16 位 PCM WAV。与 ``SpeechResult`` 的播放/预算约定一致。"""
    import numpy as np

    values = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (values * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return buffer.getvalue()
