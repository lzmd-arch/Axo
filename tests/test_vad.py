"""端点检测：什么时候算开口、什么时候算说完。"""

from __future__ import annotations

import numpy as np
import pytest

from axo.vad import FRAME_MS, Vad, VadConfig, frame_size, level_percent, rms_db


def _tone(amplitude: float, frames: int, *, sample_rate: int = 16000) -> np.ndarray:
    size = frame_size(sample_rate) * frames
    wave = np.sin(2 * np.pi * 440 * np.arange(size) / sample_rate)
    return (wave * amplitude * 32767).astype(np.int16)


def _silence(frames: int, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(frame_size(sample_rate) * frames, dtype=np.int16)


def _feed(vad: Vad, samples: np.ndarray) -> list[str]:
    """按帧喂进去，收集所有非空事件。"""
    step = frame_size()
    events: list[str] = []
    for start in range(0, len(samples), step):
        event = vad.push(samples[start : start + step])
        if event:
            events.append(event)
    return events


def test_a_sentence_starting_and_ending_is_one_segment() -> None:
    vad = Vad()

    events = _feed(vad, np.concatenate([_silence(10), _tone(0.1, 25), _silence(40)]))

    assert events == ["start", "end"]
    assert vad.in_speech is False


def test_a_short_blip_is_dropped_not_ended() -> None:
    """咳嗽、碰桌子不该触发一次提交。**但调用方仍要清空缓冲**，所以事件要分开。"""
    vad = Vad()

    events = _feed(vad, np.concatenate([_silence(10), _tone(0.1, 5), _silence(40)]))

    assert events == ["start", "drop"]


def test_constant_background_noise_does_not_trigger_after_warmup() -> None:
    """**自适应噪声底是承重件**：固定阈值在风扇声里会永远触发。"""
    vad = Vad()

    events = _feed(vad, _tone(0.02, 120))

    assert events == []
    assert vad.noise_floor_db > -50


def test_a_quick_pause_inside_a_sentence_does_not_split_it() -> None:
    """换气不是句末：结束要连续安静够久。"""
    vad = Vad()
    samples = np.concatenate(
        [
            _silence(10),
            _tone(0.1, 15),
            _silence(10),  # 200 ms，短于 end_silence_ms
            _tone(0.1, 15),
            _silence(40),
        ]
    )

    events = _feed(vad, samples)

    assert events == ["start", "end"]


def test_a_frame_can_be_any_length() -> None:
    """帧长由调用方切；短帧只是让判断粗一点，不该报错。"""
    vad = Vad()
    frame = np.zeros(100, dtype=np.int16)

    assert vad.push(frame) == ""


def test_empty_frame_is_silence_not_a_crash() -> None:
    vad = Vad()

    assert vad.push(np.zeros(0, dtype=np.int16)) == ""


def test_the_noise_floor_cannot_exceed_the_frame_that_raises_it() -> None:
    """底噪跟随是渐进的：一帧不会把它抬到那一帧之上。"""
    config = VadConfig(warmup_frames=1, floor_rise=0.5)
    vad = Vad(config)
    frame = _tone(0.03, 1)

    vad.push(frame)

    assert vad.noise_floor_db <= rms_db(frame) + 0.001


def test_level_is_bounded_and_ordered() -> None:
    quiet = np.zeros(320, dtype=np.int16)
    loud = (np.sin(2 * np.pi * 440 * np.arange(320) / 16000) * 32767).astype(np.int16)

    assert level_percent(quiet) == 0
    assert 0 <= level_percent(loud) <= 100
    assert level_percent(loud) > level_percent(_tone(0.01, 1))


def test_bad_config_is_rejected_early() -> None:
    with pytest.raises(ValueError):
        VadConfig(end_silence_ms=0)
    with pytest.raises(ValueError):
        VadConfig(min_speech_ms=0)
    with pytest.raises(ValueError):
        VadConfig(warmup_frames=-1)
    with pytest.raises(ValueError):
        frame_size(0)
    with pytest.raises(ValueError):
        frame_size(16000, 0)


def test_frame_size_matches_the_documented_rate() -> None:
    assert frame_size() == 320
    assert FRAME_MS == 20
