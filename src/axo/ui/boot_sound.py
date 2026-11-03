"""启动提示音：**程序生成**，不随包带任何音频素材。

180 ms 的双音（880 → 1320 Hz），带 10 ms 淡入淡出与指数衰减；只在第一次启动时
写到 ``data_dir()/boot/chime.wav``，之后复用。素材自制，许可上没有第三方问题
（DSH 的那几个启动插件音频取自第三方视频，明确"未授权商业用途"，不能借）。

失败一律静默：没有数据目录、没有音频设备、写文件失败——启动都不该因此停下。
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

__all__ = ["chime_bytes", "ensure_chime"]

_RATE = 44_100
_SECONDS = 0.18


def chime_bytes() -> bytes:
    """生成一段 16-bit 单声道 WAV 的完整字节。**确定性**，便于测试与复用。"""
    frames = int(_RATE * _SECONDS)
    samples: list[int] = []
    for index in range(frames):
        t = index / _RATE
        # 两段音高：前 40% 880 Hz，之后 1320 Hz；指数衰减 + 两端 10 ms 淡入淡出。
        frequency = 880.0 if t < _SECONDS * 0.4 else 1320.0
        envelope = math.exp(-6.0 * t / _SECONDS)
        fade = min(1.0, t / 0.01, (_SECONDS - t) / 0.01)
        value = math.sin(2.0 * math.pi * frequency * t) * envelope * fade * 0.35
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    payload = struct.pack("<" + "h" * len(samples), *samples)
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, _RATE, _RATE * 2, 2, 16)
    data = b"data" + struct.pack("<I", len(payload)) + payload
    return header + fmt + data


def ensure_chime() -> Path | None:
    """把提示音写到数据目录并返回路径；任何失败返回 ``None``。"""
    try:
        from ..paths import data_dir

        path = data_dir() / "boot" / "chime.wav"
        if not path.is_file() or path.stat().st_size <= 44:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(chime_bytes())
        return path
    except OSError:
        return None
