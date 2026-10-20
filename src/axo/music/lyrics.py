"""歌词：LRC 解析、旁挂文件、内嵌歌词与 LRCLIB 响应的形状。

**解析器只管把时间戳变成秒**，不判断"这句该不该高亮"——那是窗口按播放位置做的
一次二分/线性扫描。LRC 的 ``[offset:±ms]`` 按常见约定处理：正值让歌词**提前**出现
（从时间戳里减掉），负值推后。
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import LyricLine, Lyrics, Track

__all__ = [
    "embedded_lyrics",
    "lrclib_query",
    "parse_lrc",
    "parse_lrclib",
    "plain_lyrics",
    "sidecar_lyrics",
]

_STAMP = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_OFFSET = re.compile(r"\[offset:\s*([+-]?\d+)\s*\]", re.IGNORECASE)


def _seconds(minutes: int, seconds: int, fraction: str | None) -> float:
    value = minutes * 60 + seconds
    if fraction:
        value += int(fraction) / (10 ** len(fraction))
    return value


def parse_lrc(text: str) -> tuple[LyricLine, ...]:
    """把 LRC 正文变成按时间排好的行。**没有时间戳的行直接丢掉**（那是元数据）。"""
    if not text:
        return ()
    offset_s = 0.0
    match = _OFFSET.search(text)
    if match:
        offset_s = int(match.group(1)) / 1000.0
    found: list[LyricLine] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stamps = list(_STAMP.finditer(raw))
        if not stamps:
            continue
        body = _STAMP.sub("", raw).strip()
        if not body:
            continue
        for stamp in stamps:
            at_s = _seconds(int(stamp.group(1)), int(stamp.group(2)), stamp.group(3)) - offset_s
            found.append(LyricLine(at_s=max(0.0, at_s), text=body))
    found.sort(key=lambda line: line.at_s)
    return tuple(found)


def plain_lyrics(text: str, *, source: str = "embedded") -> Lyrics:
    """纯文本歌词：每行时间戳为 0，窗口按普通文本展示。"""
    lines = tuple(
        LyricLine(at_s=0.0, text=line.strip())
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    )
    return Lyrics(lines=lines, synced=False, source=source if lines else "")


def sidecar_lyrics(track: Track) -> Lyrics:
    """同目录同名 ``.lrc``。没有就空结果；读失败也只是没有。"""
    path = Path(track.path).with_suffix(".lrc")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return Lyrics()
    lines = parse_lrc(text)
    if lines:
        return Lyrics(lines=lines, synced=True, source="lrc")
    return plain_lyrics(text, source="lrc")


def embedded_lyrics(text: str) -> Lyrics:
    text = (text or "").strip()
    if not text:
        return Lyrics()
    lines = parse_lrc(text)
    if lines:
        return Lyrics(lines=lines, synced=True, source="embedded")
    return plain_lyrics(text, source="embedded")


def lrclib_query(track: Track) -> dict[str, str | int]:
    """``GET /api/get`` 的查询参数。**只有元数据，没有路径、没有文件内容。**"""
    return {
        "artist_name": track.artist,
        "track_name": track.title,
        "album_name": track.album,
        "duration": int(round(track.duration_s)) if track.duration_s else 0,
    }


def parse_lrclib(payload: object) -> Lyrics:
    """LRCLIB 的一条响应。同步歌词优先，纯文本兜底，都没有就空。"""
    if not isinstance(payload, dict):
        return Lyrics()
    synced = payload.get("syncedLyrics")
    if isinstance(synced, str) and synced.strip():
        lines = parse_lrc(synced)
        if lines:
            return Lyrics(lines=lines, synced=True, source="lrclib")
    plain = payload.get("plainLyrics")
    if isinstance(plain, str) and plain.strip():
        return plain_lyrics(plain, source="lrclib")
    return Lyrics()
