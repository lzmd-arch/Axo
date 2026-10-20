"""音乐窗口：空库、列表过滤、双击播放与歌词高亮。**离屏，收掉顶层窗口。**"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal

from axo.music.models import (
    LibraryIndex,
    LyricLine,
    Lyrics,
    PlayerState,
    Playlist,
    RepeatMode,
    Track,
)
from axo.ui.music_window import MusicWindow
from axo.ui.settings_store import MusicSettings

pytestmark = pytest.mark.gui


class FakeService(QObject):
    library_changed = Signal(object)
    scan_started = Signal()
    scan_progress = Signal(int, int)
    scan_finished = Signal(object)
    scan_failed = Signal(str)
    state_changed = Signal(object)
    current_changed = Signal(object)
    lyrics_changed = Signal(object)
    notice = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.tracks = (
            Track("1", Path("/music/一.mp3"), "一", "某人", "某张", 120.0),
            Track("2", Path("/music/二.mp3"), "二", "别人", "另一张", 90.0),
        )
        self.index = LibraryIndex(
            tracks=self.tracks, playlists=(Playlist("通勤", Path("x"), ("2",)),)
        )
        self.settings = MusicSettings()
        self.played: list[str] = []
        self.controls: list[tuple[str, Any]] = []
        self.scanned = 0

    def state(self) -> PlayerState:
        return PlayerState(volume=50, repeat=RepeatMode.OFF)

    def scan(self) -> None:
        self.scanned += 1

    def search(self, query: str, *, limit: int = 20) -> tuple[Track, ...]:
        return tuple(track for track in self.tracks if query in track.title)

    def play(self, *, query: str = "", track_id: str = "", playlist: str = "") -> str:
        self.played.append(track_id or query or playlist)
        return "开始播放"

    def control(self, action: str, value: Any = None) -> str:
        self.controls.append((action, value))
        return f"已执行 {action}"


def test_empty_library_shows_zero_rows(qtbot) -> None:
    service = FakeService()
    service.index = LibraryIndex()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    assert window.track_table.rowCount() == 0


def test_library_fills_rows_and_search_filters(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    assert window.track_table.rowCount() == 2
    window.search_edit.setText("二")
    assert window.track_table.rowCount() == 1
    assert window.track_table.item(0, 0).text() == "二"


def test_double_click_plays_the_row(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    window._play_row(1, 0)
    assert service.played == ["2"]


def test_playlist_filter_uses_its_ids(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    window.playlist_combo.setCurrentIndex(1)
    assert window.track_table.rowCount() == 1
    assert window.track_table.item(0, 0).text() == "二"


def test_transport_buttons_call_the_service(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    window._control("next")
    window._control("volume", 30)
    assert service.controls == [("next", None), ("volume", 30)]


def test_lyrics_highlight_follows_position(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    window._on_lyrics(
        Lyrics(
            lines=(LyricLine(1.0, "第一句"), LyricLine(5.0, "第二句")),
            synced=True,
            source="lrc",
        )
    )
    window._highlight_lyrics(0.4)
    assert window.lyrics_view.currentRow() == -1
    window._highlight_lyrics(5.2)
    assert window.lyrics_view.currentRow() == 1


def test_close_waits_and_hides(qtbot) -> None:
    service = FakeService()
    window = MusicWindow(service)
    qtbot.addWidget(window)
    window.close()
    assert not window.isVisible()
