"""音乐服务：把扫描、队列、播放器与歌词串起来。**播放器的所有调用都在 UI 线程。**

窗口与工具面都只认这一层：窗口拿信号刷新界面，工具拿 ``search/play/control`` 这些
方法。扫描与在线歌词各走一个 ``QThread``，并且一律经 :func:`thread_tracking.track`
接管——收尾时少等一个线程，Windows 上就是 ``0xC0000409``。
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ..llm.lyrics import RemoteLyrics
from ..music.library import read_embedded_lyrics, scan_library, search_tracks
from ..music.lyrics import embedded_lyrics, sidecar_lyrics
from ..music.models import (
    LibraryIndex,
    Lyrics,
    LyricsResult,
    MusicError,
    PlayerState,
    RepeatMode,
    Track,
)
from ..music.queue import PlayQueue
from .music_player import MusicPlayer
from .settings_store import MusicSettings
from .thread_tracking import track as track_thread
from .thread_tracking import wait_for_all

__all__ = ["MusicService"]


class _ScanWorker(QObject):
    progress = Signal(int, int)
    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self, root: Path, cancel: threading.Event, generation: int, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._root = root
        self._cancel = cancel
        self._generation = generation

    def run(self) -> None:
        try:
            index = scan_library(
                self._root,
                progress=lambda done, total: self.progress.emit(done, total),
                should_stop=self._cancel.is_set,
            )
        except Exception as exc:  # noqa: BLE001 — 线程里抛出去没有人接
            self.failed.emit(self._generation, f"扫描失败：{type(exc).__name__}")
            return
        self.done.emit(self._generation, index)


class _LyricsWorker(QObject):
    done = Signal(str, object)

    def __init__(
        self,
        track: Track,
        online: RemoteLyrics,
        *,
        allow_online: bool,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._track = track
        self._online = online
        self._allow_online = allow_online

    def run(self) -> None:
        try:
            lyrics = sidecar_lyrics(self._track)
            if lyrics.ok:
                self.done.emit(self._track.track_id, LyricsResult(lyrics))
                return
            embedded = embedded_lyrics(read_embedded_lyrics(self._track.path))
            if embedded.ok:
                self.done.emit(self._track.track_id, LyricsResult(embedded))
                return
            if self._allow_online and self._online.available:
                self.done.emit(self._track.track_id, self._online.fetch(self._track))
                return
            self.done.emit(self._track.track_id, LyricsResult(reason="本地没有歌词"))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(
                self._track.track_id, LyricsResult(reason=f"读取歌词失败：{type(exc).__name__}")
            )


class MusicService(QObject):
    library_changed = Signal(object)
    scan_started = Signal()
    scan_progress = Signal(int, int)
    scan_finished = Signal(object)
    scan_failed = Signal(str)
    state_changed = Signal(object)
    current_changed = Signal(object)
    lyrics_changed = Signal(object)
    notice = Signal(str)

    def __init__(
        self,
        settings: MusicSettings,
        *,
        player: Any | None = None,
        online: RemoteLyrics | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._index = LibraryIndex()
        self._queue = PlayQueue()
        self._player = player if player is not None else MusicPlayer(self, volume=settings.volume)
        self._online = (
            online if online is not None else RemoteLyrics(enabled=settings.online_lyrics)
        )
        self._live_threads: list[QThread] = []
        self._cancel = threading.Event()
        self._scanning = False
        self._scan_root: Path | None = None
        self._scan_generation = 0
        self._lyrics: Lyrics = Lyrics()
        self._lyrics_reason = ""
        self._lyrics_for: str = ""
        self._tool_lyrics_cache: dict[str, LyricsResult] = {}
        if hasattr(self._player, "state_changed"):
            self._player.state_changed.connect(self._on_player_state)
        if hasattr(self._player, "failed"):
            self._player.failed.connect(self._on_player_failed)
        if hasattr(self._player, "finished"):
            self._player.finished.connect(self._on_finished)
        self._apply_volume(settings.volume)

    # ── 查询 ────────────────────────────────────────────────────────────────

    @property
    def settings(self) -> MusicSettings:
        return self._settings

    @property
    def index(self) -> LibraryIndex:
        return self._index

    @property
    def player(self) -> Any:
        return self._player

    @property
    def has_root(self) -> bool:
        root = self._settings.library_root
        return root is not None and Path(root).is_dir()

    @property
    def available(self) -> bool:
        """工具面只在这一刻登记音乐工具：有根目录且已经扫出曲目。"""
        return self.has_root and bool(self._index.tracks)

    def state(self) -> PlayerState:
        base = self._player.state()
        return replace(
            base,
            shuffle=self._queue.shuffle,
            repeat=self._queue.repeat,
            queue_length=self._queue.length,
            queue_index=self._queue.index,
        )

    def search(self, query: str, *, limit: int = 20) -> tuple[Track, ...]:
        return search_tracks(self._index.tracks, query, limit=max(1, min(500, limit)))

    def playlists(self) -> tuple[Any, ...]:
        return self._index.playlists

    def current_track(self) -> Track | None:
        return self._player.track if hasattr(self._player, "track") else self._queue.current()

    def current_lyrics(self) -> tuple[Lyrics, str]:
        return self._lyrics, self._lyrics_reason

    def lyrics_for(self, query: str = "") -> tuple[Track | None, LyricsResult]:
        """工具面用的同步查询：先本地，再（开启时）在线。可以在工作线程里调。"""
        track = self.current_track()
        if query.strip():
            found = self.search(query, limit=1)
            track = found[0] if found else None
        if track is None:
            return None, LyricsResult(reason="没有找到这首曲目")
        cached = self._tool_lyrics_cache.get(track.track_id)
        if cached is not None:
            return track, cached
        lyrics = sidecar_lyrics(track)
        if not lyrics.ok:
            lyrics = embedded_lyrics(read_embedded_lyrics(track.path))
        if lyrics.ok:
            result = LyricsResult(lyrics)
        elif self._settings.online_lyrics and self._online.available:
            result = self._online.fetch(track)
        else:
            result = LyricsResult(reason="本地没有歌词")
        self._tool_lyrics_cache[track.track_id] = result
        return track, result

    # ── 扫描 ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._settings.scan_on_start and self.has_root:
            self.scan()

    def scan(self) -> None:
        root = self._settings.library_root
        if root is None or not Path(root).is_dir():
            self.scan_failed.emit(f"音乐文件夹不存在：{root}" if root else "还没有选择音乐文件夹")
            return
        # 同一个根已经在扫就不重复起；换了根则取消旧的、起新的。旧线程的 done
        # 带着 generation 回来，被 `_on_scan_done` 丢掉——它不能拿旧根的结果覆盖新根。
        if self._scanning and self._scan_root == Path(root):
            return
        self._cancel.set()
        self._scanning = True
        self._scan_root = Path(root)
        self._scan_generation += 1
        generation = self._scan_generation
        self._cancel = threading.Event()
        self.scan_started.emit()
        worker = _ScanWorker(Path(root), self._cancel, generation, self)
        thread = QThread(self)
        worker.progress.connect(self.scan_progress)
        worker.done.connect(self._on_scan_done)
        worker.failed.connect(self._on_scan_failed)
        track_thread(self, thread, worker, what="音乐扫描", slot="scan")
        thread.start()

    def _on_scan_done(self, generation: int, index: LibraryIndex) -> None:
        if generation != self._scan_generation:
            return
        self._scanning = False
        self._index = index
        self._queue.drop_missing(track.track_id for track in index.tracks)
        self._tool_lyrics_cache.clear()
        self.library_changed.emit(index)
        self.scan_finished.emit(index)
        if index.errors:
            self.notice.emit(f"有 {len(index.errors)} 个文件读不了，已跳过（{index.errors[0]}）")

    def _on_scan_failed(self, generation: int, reason: str) -> None:
        if generation != self._scan_generation:
            return
        self._scanning = False
        self.scan_failed.emit(reason)

    # ── 播放 ────────────────────────────────────────────────────────────────

    def play(self, *, query: str = "", track_id: str = "", playlist: str = "") -> str:
        """从查询/id/歌单开始播。**找不到就抛 ``MusicError``**（工具面收成人话）。"""
        tracks: tuple[Track, ...]
        start_id = ""
        if playlist:
            chosen = next(
                (
                    one
                    for one in self._index.playlists
                    if one.name.casefold() == playlist.casefold()
                ),
                None,
            )
            if chosen is None:
                names = "、".join(one.name for one in self._index.playlists) or "（没有歌单）"
                raise MusicError(f"没有叫「{playlist}」的歌单；现有：{names}")
            tracks = tuple(
                track for track in (self._index.by_id(one) for one in chosen.track_ids) if track
            )
            start_id = (
                track_id
                if track_id and any(one.track_id == track_id for one in tracks)
                else (tracks[0].track_id if tracks else "")
            )
        elif track_id:
            found = self._index.by_id(track_id)
            if found is None:
                raise MusicError(f"没有 id 为 {track_id} 的曲目")
            tracks, start_id = self._index.tracks, found.track_id
        else:
            found = search_tracks(self._index.tracks, query, limit=20)
            if not found:
                raise MusicError(f"音乐库里没有找到「{query}」")
            tracks, start_id = self._index.tracks, found[0].track_id
        if not tracks:
            raise MusicError("这个歌单里没有可播放的曲目")
        self._queue.set_tracks(tracks, start_id=start_id)
        self._play_current()
        return self._describe_current()

    def control(self, action: str, value: Any = None) -> str:
        action = action.strip().lower()
        if action in {"pause", "resume", "playpause"}:
            if self._player.track is None:
                raise MusicError("现在没有在播放的曲目")
            if action == "pause":
                self._player.pause()
            elif action == "resume" or self._player.paused:
                self._player.resume()
            else:
                self._player.pause()
            return self._describe_current()
        if action == "stop":
            self._player.stop()
            return "已停止。"
        if action in {"next", "prev"}:
            found = self._queue.next() if action == "next" else self._queue.prev()
            if found is None:
                self._player.stop()
                return "已经是队列的尽头。"
            self._play_track(found, skip_missing=True)
            return self._describe_current()
        if action == "shuffle":
            self._queue.set_shuffle(not self._queue.shuffle)
            self._emit_state()
            return "随机播放已打开。" if self._queue.shuffle else "随机播放已关闭。"
        if action == "repeat":
            if value in {"off", "all", "one"}:
                self._queue.set_repeat(RepeatMode(str(value)))
            elif value is None:
                self._queue.cycle_repeat()
            else:
                raise MusicError("repeat 只能是 off / all / one")
            self._emit_state()
            return f"循环模式：{self._queue.repeat.value}。"
        if action == "volume":
            try:
                volume = max(0, min(100, int(value)))
            except (TypeError, ValueError) as exc:
                raise MusicError("volume 要 0–100 的整数") from exc
            self._apply_volume(volume)
            return f"音量 {volume}%。"
        if action == "seek":
            try:
                position = max(0.0, float(value))
            except (TypeError, ValueError) as exc:
                raise MusicError("seek 要秒数") from exc
            self._player.seek(position)
            return f"跳到 {position:.0f} 秒。"
        raise MusicError(f"不认识的动作：{action}")

    def _play_current(self) -> None:
        track = self._queue.current()
        if track is None:
            raise MusicError("队列是空的")
        self._play_track(track)

    def _play_track(self, track: Track, *, skip_missing: bool = False) -> None:
        if not Path(track.path).is_file():
            self._queue.remove(track.track_id)
            self.notice.emit(f"「{track.title}」的文件不在了，已从队列里摘掉。")
            if not skip_missing:
                self._player.stop()
                raise MusicError(f"「{track.title}」的文件不在了：{track.path}")
            next_track = self._queue.current()
            if next_track is None:
                self._player.stop()
                raise MusicError("队列里的文件都不在了")
            track = next_track
        self._player.play(track)
        self._lyrics = Lyrics()
        self._lyrics_reason = ""
        self._lyrics_for = ""
        self.current_changed.emit(track)
        self.lyrics_changed.emit(self._lyrics)
        self._request_lyrics(track)
        self._emit_state()

    def _on_finished(self) -> None:
        found = self._queue.next(auto=True)
        if found is None:
            self._player.stop()
            self._emit_state()
            return
        self._play_track(found, skip_missing=True)

    def _on_player_failed(self, reason: str) -> None:
        self.notice.emit(reason)
        self._emit_state()

    def _on_player_state(self, _state: object) -> None:
        self._emit_state()

    def _emit_state(self) -> None:
        self.state_changed.emit(self.state())

    def _apply_volume(self, volume: int) -> None:
        with contextlib.suppress(Exception):  # 假播放器可能没有这一格
            self._player.set_volume(max(0, min(100, int(volume))))

    def _describe_current(self) -> str:
        track = self.current_track()
        if track is None:
            return "当前没有曲目。"
        return f"正在播放：{track.display}"

    # ── 歌词 ────────────────────────────────────────────────────────────────

    def _request_lyrics(self, track: Track) -> None:
        if self._lyrics_for == track.track_id:
            return
        self._lyrics_for = track.track_id
        worker = _LyricsWorker(
            track,
            self._online,
            allow_online=self._settings.online_lyrics,
            parent=self,
        )
        thread = QThread(self)
        worker.done.connect(self._on_lyrics)
        track_thread(self, thread, worker, what="音乐歌词", slot="lyrics")
        thread.start()

    def _on_lyrics(self, track_id: str, result: LyricsResult) -> None:
        if track_id != self._lyrics_for:
            return
        self._lyrics = result.lyrics
        self._lyrics_reason = result.reason
        self.lyrics_changed.emit(result.lyrics)

    # ── 设置与收尾 ──────────────────────────────────────────────────────────

    def apply_settings(self, settings: MusicSettings) -> None:
        old = self._settings
        self._settings = settings
        if settings.volume != old.volume:
            self._apply_volume(settings.volume)
        if settings.online_lyrics != old.online_lyrics:
            self._online.close()
            self._online = RemoteLyrics(enabled=settings.online_lyrics)
            self._tool_lyrics_cache.clear()
        if settings.library_root != old.library_root:
            self._cancel.set()
            self._index = LibraryIndex()
            self._queue.set_tracks(())
            self.library_changed.emit(self._index)
            if settings.library_root is not None and Path(settings.library_root).is_dir():
                self.scan()
            else:
                self._player.stop()
                self._emit_state()
        self._emit_state()

    def shutdown(self) -> None:
        """停播、取消扫描、等所有音乐线程。**幂等。**"""
        self._cancel.set()
        wait_for_all(list(self._live_threads))
        with contextlib.suppress(Exception):
            self._player.shutdown()
        self._online.close()
