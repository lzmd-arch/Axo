"""音乐工具：条件登记、只读/副作用、失败句与不过闸门。**不碰真播放器。**"""

from __future__ import annotations

from pathlib import Path

import pytest

from axo.agent.catalog import default_tools
from axo.agent.music_tools import (
    MusicControlTool,
    MusicLyricsTool,
    MusicPlaylistsTool,
    MusicPlayTool,
    MusicSearchTool,
    MusicStatusTool,
)
from axo.agent.tools import ToolContext, ToolRegistry
from axo.music.models import (
    Lyrics,
    LyricsResult,
    PlayerState,
    Playlist,
    RepeatMode,
    Track,
)
from memcore.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def _track(name: str) -> Track:
    return Track(name, Path(f"/music/{name}.mp3"), f"歌{name}", "某人", "某张", 120.0)


class FakeMusic:
    available = True

    def __init__(self) -> None:
        self.tracks = (_track("一"), _track("二"))
        self.played: list[dict[str, str]] = []
        self.controls: list[tuple[str, object]] = []

    def search(self, query: str, *, limit: int = 20) -> tuple[Track, ...]:
        return tuple(track for track in self.tracks if query in track.title)[:limit]

    def playlists(self) -> tuple[Playlist, ...]:
        return (Playlist("通勤", Path("/music/x.m3u"), ("一", "二")),)

    def lyrics_for(self, query: str = "") -> tuple[Track | None, LyricsResult]:
        if query and query not in "歌一":
            return None, LyricsResult(reason="没有")
        return self.tracks[0], LyricsResult(Lyrics(lines=(), synced=False, source="lrc"), reason="")

    def state(self) -> PlayerState:
        return PlayerState(
            track=self.tracks[0],
            playing=True,
            duration_s=120.0,
            position_s=10.0,
            volume=60,
            repeat=RepeatMode.OFF,
            queue_length=2,
            queue_index=0,
        )

    def play(self, **kwargs: str) -> str:
        self.played.append(kwargs)
        return "正在播放：歌一"

    def control(self, action: str, value: object = None) -> str:
        self.controls.append((action, value))
        return f"已执行 {action}"


def _ctx(store: MemoryStore, music: object | None = None) -> ToolContext:
    return ToolContext(store=store, music=music)


def test_music_tools_are_registered_only_with_an_available_channel(store: MemoryStore) -> None:
    plain = [tool.spec.name for tool in default_tools()]
    assert not any(name.startswith("music_") for name in plain)
    with_music = [tool.spec.name for tool in default_tools(music=FakeMusic())]
    assert with_music[-6:] == [
        "music_search",
        "music_playlists",
        "music_lyrics",
        "music_status",
        "music_play",
        "music_control",
    ]


def test_search_formats_ids_and_bounds(store: MemoryStore) -> None:
    tool = MusicSearchTool()
    outcome = tool.run({"query": "歌"}, _ctx(store, FakeMusic()))
    assert outcome.ok
    assert "歌一" in outcome.content and "[一]" in outcome.content


def test_playlists_lists_counts(store: MemoryStore) -> None:
    outcome = MusicPlaylistsTool().run({}, _ctx(store, FakeMusic()))
    assert outcome.ok and "通勤（2 首）" in outcome.content


def test_lyrics_without_lyrics_says_so_instead_of_inventing(store: MemoryStore) -> None:
    outcome = MusicLyricsTool().run({}, _ctx(store, FakeMusic()))
    assert not outcome.ok
    assert "没有歌词" in outcome.content or "没有歌词" in (outcome.detail or "")


def test_status_is_not_replayable(store: MemoryStore) -> None:
    outcome = MusicStatusTool().run({}, _ctx(store, FakeMusic()))
    assert outcome.ok and "正在播放" in outcome.content
    assert outcome.replayable is False


def test_play_and_control_do_not_ask_for_approval(store: MemoryStore) -> None:
    registry = ToolRegistry([MusicPlayTool(), MusicControlTool()])
    assert not registry.needs_approval("music_play")
    assert not registry.needs_approval("music_control")
    music = FakeMusic()
    ctx = _ctx(store, music)
    played = registry.dispatch(_call("music_play", query="歌一"), ctx)
    assert played.ok and music.played == [{"query": "歌一", "track_id": "", "playlist": ""}]
    controlled = registry.dispatch(_call("music_control", action="volume", volume=30), ctx)
    assert controlled.ok and music.controls == [("volume", 30)]


def test_failures_are_readable_without_a_channel(store: MemoryStore) -> None:
    outcome = MusicSearchTool().run({"query": "歌"}, _ctx(store, None))
    assert not outcome.ok and "音乐库" in outcome.content


def test_seek_requires_a_number(store: MemoryStore) -> None:
    outcome = MusicControlTool().run({"action": "seek"}, _ctx(store, FakeMusic()))
    assert not outcome.ok and "秒数" in outcome.content


class _Call:
    def __init__(self, name: str, arguments: dict[str, object]) -> None:
        self.name = name
        self.arguments = arguments
        self.id = "c1"
        self.argument_error = ""


def _call(name: str, **arguments: object) -> _Call:
    return _Call(name, arguments)
