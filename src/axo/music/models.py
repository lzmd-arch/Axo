"""音乐库的纯层：曲目、歌单、歌词与播放队列。**不 import Qt，也不 import httpx。**

这一层只做"读目录、读标签、算顺序"三件事，所以能在一台没有声卡、没有 Qt 的机器上
逐条断言。播放器在 :mod:`axo.ui.music_player`，在线歌词在 :mod:`axo.llm.lyrics`。

**曲目 id 是路径的哈希，不是路径本身。** 工具面要把它交给模型，而绝对路径里带着
用户名与目录结构；哈希只保证"同一条曲目在本次会话里认得出"，不承担去重之外的用途。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "LyricLine",
    "Lyrics",
    "LyricsResult",
    "MusicError",
    "MusicSource",
    "PlayerState",
    "Playlist",
    "RepeatMode",
    "Track",
]


class MusicError(RuntimeError):
    """一句能直接显示给使用者的失败原因。**不携带路径之外的内部状态。**"""


@dataclass(frozen=True)
class Track:
    """一条曲目。``title`` 永远非空：标签坏了就退回文件名。"""

    track_id: str
    path: Path
    title: str
    artist: str = ""
    album: str = ""
    duration_s: float = 0.0
    track_no: int = 0
    year: str = ""

    @property
    def display(self) -> str:
        if self.artist:
            return f"{self.artist} — {self.title}"
        return self.title


@dataclass(frozen=True)
class Playlist:
    """一个 ``.m3u`` / ``.m3u8`` 歌单，``track_ids`` 只保留库里存在的那些。"""

    name: str
    path: Path
    track_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LyricLine:
    at_s: float
    text: str


@dataclass(frozen=True)
class Lyrics:
    """一次歌词查询的结果。``synced`` 为假时 ``at_s`` 全部是 0。"""

    lines: tuple[LyricLine, ...] = ()
    synced: bool = False
    source: str = ""
    """``lrc`` / ``embedded`` / ``lrclib``；空串 = 没找到。"""

    @property
    def ok(self) -> bool:
        return bool(self.lines)


@dataclass(frozen=True)
class LyricsResult:
    """一次查询的结果 + 失败时那一句人话（成功时为空串）。"""

    lyrics: Lyrics = field(default_factory=Lyrics)
    reason: str = ""


class RepeatMode(str, Enum):
    OFF = "off"
    ONE = "one"
    ALL = "all"


@dataclass(frozen=True)
class PlayerState:
    """播放器对外的一次快照。**任何一项变了都重建整份**，不传可变引用。"""

    track: Track | None = None
    playing: bool = False
    paused: bool = False
    position_s: float = 0.0
    duration_s: float = 0.0
    volume: int = 80
    shuffle: bool = False
    repeat: RepeatMode = RepeatMode.OFF
    queue_length: int = 0
    queue_index: int = -1
    error: str = ""


@runtime_checkable
class MusicSource(Protocol):
    """以后接 Subsonic/Navidrome 时实现这一组；v1 只有本地文件夹。"""

    def search(self, query: str, limit: int = 20) -> tuple[Track, ...]: ...

    def playlists(self) -> tuple[Playlist, ...]: ...


@dataclass
class LibraryIndex:
    """一次扫描的结果。**索引只活在内存里**；扫描是后台线程，写盘不是 v1 的事。"""

    tracks: tuple[Track, ...] = ()
    playlists: tuple[Playlist, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    def by_id(self, track_id: str) -> Track | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
