"""LRC 解析与歌词来源的形状。**不联网。**"""

from __future__ import annotations

from pathlib import Path

from axo.music.lyrics import embedded_lyrics, lrclib_query, parse_lrc, parse_lrclib, sidecar_lyrics
from axo.music.models import Track


def test_parse_lrc_orders_lines_and_strips_stamps() -> None:
    lines = parse_lrc("[00:03.50] 第二句\n[00:01.00] 第一句\n[ti:标题]\n")
    assert [(line.at_s, line.text) for line in lines] == [(1.0, "第一句"), (3.5, "第二句")]


def test_parse_lrc_handles_multiple_stamps_on_one_line() -> None:
    lines = parse_lrc("[00:01.00][00:10.00] 副歌\n")
    assert [(line.at_s, line.text) for line in lines] == [(1.0, "副歌"), (10.0, "副歌")]


def test_parse_lrc_applies_positive_offset_as_earlier() -> None:
    lines = parse_lrc("[offset:+500]\n[00:02.00] 一句\n")
    assert lines[0].at_s == 1.5


def test_parse_lrc_ignores_metadata_and_blank_lyrics() -> None:
    assert parse_lrc("[ar:某人]\n[al:某张]\n[00:01.00]\n") == ()


def test_sidecar_lyrics_prefers_lrc(tmp_path: Path) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"")
    (tmp_path / "song.lrc").write_text("[00:01.00] 你好\n", encoding="utf-8")
    track = Track("id", audio, "song")
    lyrics = sidecar_lyrics(track)
    assert lyrics.ok and lyrics.synced and lyrics.source == "lrc"
    assert lyrics.lines[0].text == "你好"


def test_sidecar_lyrics_absent_is_empty(tmp_path: Path) -> None:
    track = Track("id", tmp_path / "song.mp3", "song")
    assert not sidecar_lyrics(track).ok


def test_embedded_lyrics_accepts_plain_text() -> None:
    lyrics = embedded_lyrics("第一行\n第二行\n")
    assert lyrics.ok and not lyrics.synced
    assert [line.text for line in lyrics.lines] == ["第一行", "第二行"]


def test_lrclib_query_sends_metadata_only() -> None:
    track = Track("id", Path("secret/path/song.mp3"), "晴天", "周杰伦", "叶惠美", 269.4)
    assert lrclib_query(track) == {
        "artist_name": "周杰伦",
        "track_name": "晴天",
        "album_name": "叶惠美",
        "duration": 269,
    }


def test_parse_lrclib_prefers_synced_then_plain() -> None:
    synced = parse_lrclib({"syncedLyrics": "[00:01.00] 同步\n", "plainLyrics": "纯文本"})
    assert synced.synced and synced.source == "lrclib" and synced.lines[0].text == "同步"
    plain = parse_lrclib({"syncedLyrics": None, "plainLyrics": "第一行\n第二行"})
    assert plain.ok and not plain.synced
    assert parse_lrclib({}) == parse_lrclib(None)
