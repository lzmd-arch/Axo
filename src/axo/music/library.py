"""扫描本地音乐文件夹、读标签、解析 M3U。**只读用户盘，不改任何文件。**

**标签解析是"尽力而为"。** tinytag 读不了的文件（损坏、后缀对不上、扩展名是伪造的）
不应该让整次扫描失败：那一条退回文件名，其余照常入库。界面上能看出"这条没有标签"，
比"整个库空了"好得多——后者看起来像扫描坏了。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from .models import LibraryIndex, Playlist, Track

__all__ = [
    "SUPPORTED_SUFFIXES",
    "format_duration",
    "parse_m3u",
    "playlists_in",
    "read_cover",
    "read_embedded_lyrics",
    "scan_library",
    "search_tracks",
    "track_from_path",
    "track_id_for",
]

SUPPORTED_SUFFIXES = (
    ".mp3",
    ".mp2",
    ".flac",
    ".m4a",
    ".mp4",
    ".ogg",
    ".oga",
    ".opus",
    ".wav",
    ".wma",
    ".aac",
)

_IGNORED_DIRS = {"$RECYCLE.BIN", "System Volume Information", ".git", ".svn"}
"""扫描时直接剪掉的目录。**不是按隐藏名单**：普通音乐目录里也可能有以点开头的
文件夹（专辑名 ".5"），那一种是用户放的，照扫。"""

_PLAYLIST_SUFFIXES = (".m3u", ".m3u8")


def track_id_for(path: Path | str) -> str:
    """路径的稳定短 id。大小写与斜杠方向不影响结果（Windows 上同一条曲目只有一个 id）。"""
    normalized = os.path.normcase(str(Path(path).resolve())).replace("\\", "/")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _tag(path: Path) -> object | None:
    """读一次标签；没装 tinytag 或文件坏了都返回 ``None``。"""
    try:
        from tinytag import TinyTag  # 函数内 import：核心腿不要求装它
    except ImportError:
        return None
    try:
        return TinyTag.get(str(path))
    except Exception:  # noqa: BLE001 — 坏文件、未知格式、权限，一律退回文件名
        return None


def track_from_path(path: Path | str) -> Track:
    """一条曲目。标签缺失时标题取文件名、专辑取父目录名。"""
    folder = Path(path)
    tag = _tag(folder)
    title = str(getattr(tag, "title", "") or "").strip() or folder.stem
    artist = str(getattr(tag, "artist", "") or "").strip()
    album = str(getattr(tag, "album", "") or "").strip() or folder.parent.name
    try:
        duration = float(getattr(tag, "duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        track_no = int(getattr(tag, "track", 0) or 0)
    except (TypeError, ValueError):
        track_no = 0
    year = str(getattr(tag, "year", "") or "").strip()
    return Track(
        track_id=track_id_for(folder),
        path=folder,
        title=title,
        artist=artist,
        album=album,
        duration_s=max(0.0, duration),
        track_no=max(0, track_no),
        year=year,
    )


def _walk_audio(root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name for name in dir_names if not name.startswith(".") and name not in _IGNORED_DIRS
        )
        for name in sorted(file_names):
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(Path(current) / name)
    return found


def scan_library(
    root: Path | str,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> LibraryIndex:
    """递归扫描一个文件夹。**不抛**：读不了的条目进 ``errors``。

    ``progress(done, total)`` 每 50 条调一次；``should_stop`` 为真时提前收工，
    返回已经扫到的部分（关闭窗口/换根时用）。
    """
    folder = Path(root)
    if not folder.is_dir():
        return LibraryIndex(errors=(f"音乐文件夹不存在：{folder}",))
    files = _walk_audio(folder)
    total = len(files)
    tracks: list[Track] = []
    errors: list[str] = []
    for index, path in enumerate(files, start=1):
        if should_stop is not None and should_stop():
            break
        try:
            tracks.append(track_from_path(path))
        except Exception as exc:  # noqa: BLE001 — 单条失败不中断整次扫描
            errors.append(f"{path.name}：{type(exc).__name__}")
        if progress is not None and (index % 50 == 0 or index == total):
            progress(index, total)
    ordered = tuple(
        sorted(tracks, key=lambda one: (one.album.casefold(), one.track_no, one.title.casefold()))
    )
    return LibraryIndex(
        tracks=ordered,
        playlists=playlists_in(folder, ordered),
        errors=tuple(errors),
    )


def _playlist_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name for name in dir_names if not name.startswith(".") and name not in _IGNORED_DIRS
        )
        for name in sorted(file_names):
            if not name.startswith(".") and Path(name).suffix.lower() in _PLAYLIST_SUFFIXES:
                found.append(Path(current) / name)
    return found


def parse_m3u(path: Path | str) -> tuple[Path, ...]:
    """读一个 M3U/M3U8。**只返回本地、存在的文件**；网络地址与缺失项跳过。

    相对路径按**歌单所在目录**解析——M3U 的通用约定，也是用户把歌单和音乐放在
    同一个文件夹时唯一说得通的那种读法。
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ()
    base = source.parent
    found: list[Path] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "://" in line:
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = base / candidate
        try:
            if candidate.is_file():
                found.append(candidate.resolve())
        except OSError:
            continue
    return tuple(found)


def playlists_in(root: Path | str, tracks: Sequence[Track]) -> tuple[Playlist, ...]:
    """扫描根目录下的歌单，只保留库里认得出的曲目。"""
    known = {os.path.normcase(str(track.path.resolve())): track.track_id for track in tracks}
    result: list[Playlist] = []
    for path in _playlist_files(Path(root)):
        ids: list[str] = []
        for entry in parse_m3u(path):
            track_id = known.get(os.path.normcase(str(entry)))
            if track_id is not None and track_id not in ids:
                ids.append(track_id)
        if ids:
            result.append(Playlist(name=path.stem, path=path, track_ids=tuple(ids)))
    return tuple(sorted(result, key=lambda one: one.name.casefold()))


def read_cover(path: Path | str) -> bytes:
    """内嵌封面，没有就空串。**只在选中/播放时调**，不在全库扫描时读图。"""
    try:
        from tinytag import TinyTag
    except ImportError:
        return b""
    try:
        tag = TinyTag.get(str(path), image=True)
    except Exception:  # noqa: BLE001
        return b""
    images = getattr(tag, "images", None)
    cover = getattr(images, "front_cover", None)
    data = getattr(cover, "data", b"") if cover is not None else b""
    return bytes(data or b"")


def read_embedded_lyrics(path: Path | str) -> str:
    """标签里的歌词正文（没有就空串）。tinytag 2.x 放在 ``other["lyrics"]``。"""
    tag = _tag(path)
    other = getattr(tag, "other", None) or {}
    text = other.get("lyrics") if isinstance(other, dict) else None
    if isinstance(text, list):
        text = "\n".join(str(one) for one in text)
    return str(text or "").strip()


def search_tracks(tracks: Sequence[Track], query: str, *, limit: int = 20) -> tuple[Track, ...]:
    """按标题/艺术家/专辑/文件名做子串匹配，按"越像标题越靠前"排序。"""
    needle = query.strip().casefold()
    if not needle:
        return tuple(tracks[: max(0, limit)])
    scored: list[tuple[int, str, Track]] = []
    for track in tracks:
        title = track.title.casefold()
        artist = track.artist.casefold()
        album = track.album.casefold()
        name = track.path.stem.casefold()
        if needle == title:
            score = 0
        elif title.startswith(needle):
            score = 1
        elif needle in title:
            score = 2
        elif needle in artist:
            score = 3
        elif needle in album:
            score = 4
        elif needle in name:
            score = 5
        else:
            continue
        scored.append((score, title, track))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in scored[: max(0, limit)])


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"
