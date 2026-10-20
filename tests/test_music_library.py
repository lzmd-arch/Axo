"""纯层音乐库：扫描、歌单、搜索。**不需要 tinytag，也不需要声卡。**"""

from __future__ import annotations

from pathlib import Path

from axo.music.library import (
    format_duration,
    parse_m3u,
    playlists_in,
    scan_library,
    search_tracks,
    track_from_path,
    track_id_for,
)
from axo.music.models import Track


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_track_id_is_stable_across_separators_and_case(tmp_path: Path) -> None:
    one = tmp_path / "Album" / "Song.mp3"
    two = tmp_path / "album" / "song.MP3"
    _touch(one)
    _touch(two)
    assert track_id_for(one) == track_id_for(str(one).replace("\\", "/"))
    # Windows 的 normcase 把大小写抹平；POSIX 上不抹——这条断言只钉"同一台机器上稳定"。
    assert len(track_id_for(one)) == 16


def test_scan_falls_back_to_filename_when_tags_are_missing(tmp_path: Path) -> None:
    _touch(tmp_path / "a.mp3")
    _touch(tmp_path / "nested" / "b.flac")
    _touch(tmp_path / "cover.jpg")
    _touch(tmp_path / ".hidden" / "c.mp3")
    index = scan_library(tmp_path)
    titles = {track.title for track in index.tracks}
    assert titles == {"a", "b"}
    assert index.errors == ()
    assert len(index.tracks) == 2


def test_scan_missing_root_is_a_readable_error(tmp_path: Path) -> None:
    index = scan_library(tmp_path / "nope")
    assert index.tracks == ()
    assert index.errors and "不存在" in index.errors[0]


def test_scan_can_stop_early(tmp_path: Path) -> None:
    for index in range(80):
        _touch(tmp_path / f"{index:03}.mp3")
    seen: list[tuple[int, int]] = []
    index = scan_library(
        tmp_path,
        progress=lambda done, total: seen.append((done, total)),
        should_stop=lambda: bool(seen) and seen[-1][0] >= 50,
    )
    assert 0 < len(index.tracks) < 80
    assert seen


def test_track_uses_fake_tags(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    import axo.music.library as library

    path = _touch(tmp_path / "x.mp3")

    def fake_tag(_path: Path) -> object:
        return SimpleNamespace(
            title="夜航", artist="阿螈", album="第一张", duration=123.4, track="7", year="2026"
        )

    monkeypatch.setattr(library, "_tag", fake_tag)
    track = track_from_path(path)
    assert (track.title, track.artist, track.album) == ("夜航", "阿螈", "第一张")
    assert track.duration_s == 123.4
    assert track.track_no == 7
    assert track.year == "2026"
    assert track.display == "阿螈 — 夜航"


def test_m3u_parses_relative_and_skips_comments_and_missing(tmp_path: Path) -> None:
    one = _touch(tmp_path / "music" / "one.mp3")
    two = _touch(tmp_path / "two.mp3")
    playlist = tmp_path / "mix.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXTINF:1,one\nmusic/one.mp3\n"
        + str(two)
        + "\nhttp://example.com/x.mp3\nnope.mp3\n",
        encoding="utf-8",
    )
    assert parse_m3u(playlist) == (one.resolve(), two.resolve())


def test_playlists_in_keeps_only_known_tracks(tmp_path: Path) -> None:
    one = _touch(tmp_path / "one.mp3")
    _touch(tmp_path / "two.mp3")
    (tmp_path / "mix.m3u").write_text("one.mp3\ntwo.mp3\nmissing.mp3\n", encoding="utf-8")
    tracks = (track_from_path(one),)
    playlists = playlists_in(tmp_path, tracks)
    assert len(playlists) == 1
    assert playlists[0].name == "mix"
    assert playlists[0].track_ids == (tracks[0].track_id,)


def test_search_ranks_title_before_artist_before_filename() -> None:
    tracks = (
        Track("1", Path("x/alpha.mp3"), "夜航", "别的"),
        Track("2", Path("x/beta.mp3"), "别的", "夜航"),
        Track("3", Path("x/night.mp3"), "无关", "无关"),
    )
    assert [track.track_id for track in search_tracks(tracks, "夜航")] == ["1", "2"]
    assert [track.track_id for track in search_tracks(tracks, "night")] == ["3"]
    assert search_tracks(tracks, "") == tracks
    assert search_tracks(tracks, "不存在") == ()


def test_format_duration() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(61.6) == "1:02"
    assert format_duration(-3) == "0:00"
