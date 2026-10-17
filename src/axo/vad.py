"""从麦克风 PCM 里认出「一句说完了」。

电话式对话的第一个硬需求不是识别得多准，而是**知道什么时候该停下听**。这个模块
只做那一件事：逐帧看能量，认出语音从哪一刻开始、沉默够久之后结束。它**不认识
Qt，也不认识 vosk**——喂进来的是二十毫秒的 numpy 数组，吐出去的是一次状态迁移。

## 为什么是能量而不是模型

Silero VAD 更准，但它带 onnxruntime（几十 MB）与一个需要随包分发的模型文件；而
这里要分辨的只是「有没有人在说话」与「停了多久」，在安静的室内、单麦克风、16 kHz
的条件下，自适应噪声底足够。真正的语义判断交给 ASR 与模型，这一层只负责切段。

**自适应噪声底是承重件**：固定阈值在风扇声大一点的房间就永远触发、在安静房间又
把轻声说话漏掉。这里用「不在说话时缓慢跟随环境能量」的办法，并且在开头几帧只学
不判（``warmup_frames``）——否则开机第一声噪声就会被当成语音。

## 门槛与迟滞

* 起始要求 ``start_margin_db``，结束只看 ``end_margin_db``（更松）；
* 结束要连续 ``end_silence_ms`` 的安静，避免词与词之间的换气被切成两句；
* 短于 ``min_speech_ms`` 的一段发 ``drop`` 而不是 ``end``——咳嗽、碰桌子不该
  触发一次提交，但**调用方仍要清空缓冲**，所以这两种事件必须分开。

帧长由调用方切好（见 :func:`frame_size`）；本模块不攒残余字节，那是
``SegmentWorker`` 的事。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FRAME_MS",
    "MIN_SPEECH_MS",
    "Vad",
    "VadConfig",
    "frame_size",
    "level_percent",
    "rms_db",
]

#: 一帧多少毫秒。20 ms 是语音端点检测的常用粒度：足够快，也不会让每帧的开销
#: 变得显著（16 kHz 下 320 个采样点）。
FRAME_MS = 20

#: 短于这个时长的一段不算一句话。250 ms 大约是一个两字词的时长。
MIN_SPEECH_MS = 250

#: 能量下限（dBFS）。低于它的帧按 -100 dB 处理，避免 log(0)。
_SILENCE_DB = -100.0


def frame_size(sample_rate: int = 16000, frame_ms: int = FRAME_MS) -> int:
    """一帧有多少采样点。"""
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate 与 frame_ms 都必须是正数。")
    return max(1, round(sample_rate * frame_ms / 1000))


def rms_db(frame: np.ndarray) -> float:
    """一帧的均方根能量，单位 dBFS（0 = 满幅）。"""
    if frame.size == 0:
        return _SILENCE_DB
    values = np.asarray(frame, dtype=np.float64)
    if np.issubdtype(frame.dtype, np.integer):
        # int16 是采集层的原生格式；按它的满幅归一化，别按数据里出现过的最大值
        # 归一化——那样一个安静房间里的一声咳嗽会被放大成满幅。
        values = values / 32768.0
    rms = float(np.sqrt(np.mean(np.square(values))))
    if rms <= 1e-10:
        return _SILENCE_DB
    return 20.0 * float(np.log10(rms))


def level_percent(frame: np.ndarray, *, floor_db: float = -60.0) -> int:
    """给界面用的 0–100 电平。**只表示响度，不表示有没有说话。**"""
    span = -floor_db
    if span <= 0:
        return 0
    db = max(floor_db, rms_db(frame))
    return int(round(max(0.0, min(1.0, (db - floor_db) / span)) * 100))


@dataclass
class VadConfig:
    """端点检测的参数。

    Attributes:
        sample_rate: 采样率，只用来把毫秒换成帧数。
        frame_ms: 帧长。
        start_margin_db: 高于噪声底多少算开口。8 dB 在实测里能分开轻声与底噪。
        end_margin_db: 结束的迟滞门槛，比起始低。太接近起始会让换气把句子切断。
        end_silence_ms: 连续安静多久算这一句结束。
        min_speech_ms: 短于它的段发 ``drop``。
        warmup_frames: 开头多少帧只学噪声底、不做判断。
        floor_fall: 噪声底向下跟随的速度（安静环境突然变安静时用快的）。
        floor_rise: 噪声底向上跟随的速度（比向下慢，避免把一句话的开头学成噪声）。
    """

    sample_rate: int = 16000
    frame_ms: int = FRAME_MS
    start_margin_db: float = 8.0
    end_margin_db: float = 4.0
    end_silence_ms: int = 600
    min_speech_ms: int = MIN_SPEECH_MS
    warmup_frames: int = 5
    floor_fall: float = 0.25
    floor_rise: float = 0.02

    def __post_init__(self) -> None:
        if self.end_silence_ms <= 0 or self.min_speech_ms <= 0:
            raise ValueError("end_silence_ms 与 min_speech_ms 必须为正。")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames 不能为负。")


class Vad:
    """逐帧端点检测。**返回事件字符串，调用方据此切段。**

    Events:
        ``""``: 没有变化。
        ``"start"``: 刚刚开口。
        ``"end"``: 一段有效语音结束。
        ``"drop"``: 一段太短，丢掉。
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self._floor_db = -60.0
        self._warmup = 0
        self._in_speech = False
        self._voiced_frames = 0
        self._silence_frames = 0
        self._above_start = 0
        self._silence_needed = max(1, round(self.config.end_silence_ms / self.config.frame_ms))
        self._start_needed = 3

    @property
    def noise_floor_db(self) -> float:
        return self._floor_db

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, frame: np.ndarray) -> str:
        """喂一帧，返回事件。帧长不必精确——短帧只是让判断粗一点。"""
        db = rms_db(frame)

        if self._warmup < self.config.warmup_frames:
            self._warmup += 1
            self._floor_db += (db - self._floor_db) / self._warmup
            return ""

        if not self._in_speech:
            self._follow_quiet(db)
            if db > self._floor_db + self.config.start_margin_db:
                self._above_start += 1
            else:
                self._above_start = 0
            if self._above_start >= self._start_needed:
                self._in_speech = True
                self._voiced_frames = self._above_start
                self._silence_frames = 0
                self._above_start = 0
                return "start"
            return ""

        if db > self._floor_db + self.config.end_margin_db:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._silence_frames >= self._silence_needed:
            # **只数有声的帧**。把结尾那 600 ms 静音算进来，任何一次咳嗽都会
            # 被抬过 min_speech_ms，``drop`` 就永远不会发生。
            duration_ms = self._voiced_frames * self.config.frame_ms
            self._in_speech = False
            self._voiced_frames = 0
            self._silence_frames = 0
            self._above_start = 0
            return "end" if duration_ms >= self.config.min_speech_ms else "drop"
        return ""

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _follow_quiet(self, db: float) -> None:
        """不在说话时让噪声底跟随环境。**只在安静帧上跟**，否则一句话的开头
        会把底抬高、把后半句判成没在说话。"""
        if db > self._floor_db + self.config.start_margin_db:
            return
        rate = self.config.floor_fall if db < self._floor_db else self.config.floor_rise
        self._floor_db += (db - self._floor_db) * rate
