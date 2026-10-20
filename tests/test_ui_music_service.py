"""音乐服务：队列、播放器调用、歌词来源与收尾。**用假播放器，不碰声卡。**"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from axo.music.models import LibraryIndex, MusicError, PlayerState, Playlist, RepeatMode, Track
from axo.ui.music_service import MusicService
from axo.ui.settings_store import MusicSettings


class FakePlayer(QObject):
    state_changed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.track: Track | None = None
        self.paused = False
        self.volume = 80
        self.played: list[str] = []
        self.stopped = False

    def play(self, track: Track, *, position_s: float = 0.0) -> None:
        self.track = track
        self.paused = False
        self.played.append(track.track_id)
        self.state_changed.emit(self.state())

    def pause(self) -> None:
        self.paused = True
        self.state_changed.emit(self.state())

    def resume(self) -> None:
        self.paused = False
        self.state_changed.emit(self.state())

    def stop(self) -> None:
        self.stopped = True
        self.paused = False
        self.state_changed.emit(self.state())

    def seek(self, position_s: float) -> None:
        self.position_s = position_s

    def set_volume(self, volume: int) -> None:
        self.volume = volume

    def shutdown(self) -> None:
        self.stopped = True

    def state(self) -> PlayerState:
        return PlayerState(
            track=self.track,
            paused=self.paused,
            playing=not self.paused and self.track is not None,
            volume=self.volume,
        )


def _track(tmp_path: Path, name: str) -> Track:
    path = tmp_path / f"{name}.mp3"
    path.write_bytes(b"")
    return Track(name, path, f"歌{name}", "某人", "某张", 120.0)


def _service(
    tmp_path: Path, tracks: tuple[Track, ...], *, root: Path | None = None
) -> MusicService:
    player = FakePlayer()
    settings = MusicSettings(library_root=root or tmp_path, scan_on_start=False)
    service = MusicService(settings, player=player, online=None)  # type: ignore[arg-type]
    service._index = LibraryIndex(tracks=tracks)
    service._request_lyrics = lambda _track: None  # type: ignore[method-assign]
    return service


def test_play_query_fills_queue_and_starts_selected_track(tmp_path: Path) -> None:
    tracks = tuple(_track(tmp_path, name) for name in ("一", "二", "三"))
    service = _service(tmp_path, tracks)
    message = service.play(query="歌二")
    assert "歌二" in message
    assert service.player.track.track_id == "二"  # type: ignore[union-attr]
    assert service.state().queue_length == 3


def test_next_prev_and_auto_finish(tmp_path: Path) -> None:
    tracks = tuple(_track(tmp_path, name) for name in ("一", "二"))
    service = _service(tmp_path, tracks)
    service.play(track_id="一")
    service.control("next")
    assert service.player.track.track_id == "二"  # type: ignore[union-attr]
    service.player.finished.emit()
    assert service.player.stopped
    service.control("prev")
    assert service.player.track.track_id == "一"  # type: ignore[union-attr]


def test_repeat_one_replays_on_finish(tmp_path: Path) -> None:
    tracks = (_track(tmp_path, "一"),)
    service = _service(tmp_path, tracks)
    service.play(track_id="一")
    service.control("repeat", "one")
    service.player.finished.emit()
    assert service.player.played == ["一", "一"]


def test_missing_file_is_removed_and_reported(tmp_path: Path) -> None:
    good = _track(tmp_path, "好的")
    gone = Track("没了", tmp_path / "missing.mp3", "没了")
    service = _service(tmp_path, (good, gone))
    with pytest.raises(MusicError):
        service.play(track_id="没了")
    assert "没了" not in {track.track_id for track in service._queue.tracks}


def test_lyrics_for_uses_sidecar(tmp_path: Path) -> None:
    good = _track(tmp_path, "有词")
    (tmp_path / "有词.lrc").write_text("[00:01.00] 你好\n", encoding="utf-8")
    service = _service(tmp_path, (good,))
    service.player.track = good
    track, result = service.lyrics_for()
    assert track is not None and result.lyrics.ok and result.lyrics.synced


def test_playlist_play_uses_its_order(tmp_path: Path) -> None:
    tracks = tuple(_track(tmp_path, name) for name in ("一", "二"))
    service = _service(tmp_path, tracks)
    service._index = LibraryIndex(
        tracks=tracks,
        playlists=(Playlist("顺序", tmp_path / "x.m3u", ("二", "一")),),
    )
    service.play(playlist="顺序")
    assert service.player.track.track_id == "二"  # type: ignore[union-attr]
    service.control("next")
    assert service.player.track.track_id == "一"  # type: ignore[union-attr]


def test_root_change_clears_queue_and_scans(tmp_path: Path, monkeypatch) -> None:
    tracks = (_track(tmp_path, "一"),)
    service = _service(tmp_path, tracks)
    service.play(track_id="一")
    calls: list[bool] = []
    new_root = tmp_path / "new"
    new_root.mkdir()
    monkeypatch.setattr(service, "scan", lambda: calls.append(True))
    service.apply_settings(MusicSettings(library_root=new_root, scan_on_start=False))
    assert calls == [True]
    assert service.state().queue_length == 0


def test_control_rejects_unknown_action(tmp_path: Path) -> None:
    service = _service(tmp_path, (_track(tmp_path, "一"),))
    with pytest.raises(MusicError):
        service.control("dance")


def test_shutdown_stops_player_and_is_idempotent(tmp_path: Path) -> None:
    service = _service(tmp_path, ())
    service.shutdown()
    service.shutdown()
    assert service.player.stopped


def test_repeat_mode_cycle_and_shuffle(tmp_path: Path) -> None:
    service = _service(tmp_path, (_track(tmp_path, "一"),))
    service.play(track_id="一")
    service.control("shuffle")
    assert service.state().shuffle
    service.control("repeat")
    assert service.state().repeat is RepeatMode.ALL


def test_a_stale_scan_result_is_ignored(tmp_path: Path) -> None:
    """换了根之后，旧扫描带着旧 generation 回来——它不能覆盖新根的结果。"""
    service = _service(tmp_path, (_track(tmp_path, "一"),))
    old = LibraryIndex(tracks=(_track(tmp_path, "旧"),))
    service._scan_generation = 3
    service._on_scan_done(2, old)
    assert service.index.tracks[0].title == "歌一", "旧 generation 的结果不该覆盖当前库"


def test_playlist_play_can_start_at_the_clicked_track(tmp_path: Path) -> None:
    tracks = tuple(_track(tmp_path, name) for name in ("一", "二"))
    service = _service(tmp_path, tracks)
    service._index = LibraryIndex(
        tracks=tracks,
        playlists=(Playlist("顺序", tmp_path / "x.m3u", ("一", "二")),),
    )
    service.play(playlist="顺序", track_id="二")
    assert service.player.track.track_id == "二"  # type: ignore[union-attr]
    assert service.state().queue_length == 2
