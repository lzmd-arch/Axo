"""本地音乐库。纯层见 :mod:`axo.music.models` / :mod:`axo.music.library` /
:mod:`axo.music.lyrics` / :mod:`axo.music.queue`；Qt 播放器在 ``axo.ui.music_player``。"""

from __future__ import annotations

from .library import (
    SUPPORTED_SUFFIXES,
    format_duration,
    parse_m3u,
    playlists_in,
    read_cover,
    read_embedded_lyrics,
    scan_library,
    search_tracks,
    track_from_path,
    track_id_for,
)
from .lyrics import (
    embedded_lyrics,
    lrclib_query,
    parse_lrc,
    parse_lrclib,
    plain_lyrics,
    sidecar_lyrics,
)
from .models import (
    LibraryIndex,
    LyricLine,
    Lyrics,
    MusicError,
    MusicSource,
    PlayerState,
    Playlist,
    RepeatMode,
    Track,
)
from .queue import PlayQueue

__all__ = [
    "SUPPORTED_SUFFIXES",
    "LibraryIndex",
    "LyricLine",
    "Lyrics",
    "MusicError",
    "MusicSource",
    "PlayQueue",
    "PlayerState",
    "Playlist",
    "RepeatMode",
    "Track",
    "embedded_lyrics",
    "format_duration",
    "lrclib_query",
    "parse_lrc",
    "parse_lrclib",
    "parse_m3u",
    "plain_lyrics",
    "playlists_in",
    "read_cover",
    "read_embedded_lyrics",
    "scan_library",
    "search_tracks",
    "sidecar_lyrics",
    "track_from_path",
    "track_id_for",
]
