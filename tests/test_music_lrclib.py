"""LRCLIB 客户端：精确查、搜索兜底、限流与超时。**用 MockTransport，不联网。**"""

from __future__ import annotations

from pathlib import Path

import httpx

from axo.llm.lyrics import RemoteLyrics
from axo.music.models import Track

TRACK = Track("id", Path("secret/song.mp3"), "晴天", "周杰伦", "叶惠美", 269.4)


def _client(handler) -> RemoteLyrics:
    return RemoteLyrics(transport=httpx.MockTransport(handler))


def test_exact_hit_returns_synced_lyrics() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "trackName": "晴天",
                "syncedLyrics": "[00:01.00] 故事的小黄花",
                "plainLyrics": "故事的小黄花",
            },
        )

    result = _client(handler).fetch(TRACK)
    assert result.lyrics.ok and result.lyrics.synced
    assert result.lyrics.source == "lrclib"
    assert seen[0].url.params["track_name"] == "晴天"
    assert "secret" not in str(seen[0].url)


def test_search_fallback_matches_the_title() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/get"):
            return httpx.Response(404, json={"message": "nope"})
        return httpx.Response(
            200,
            json=[
                {"trackName": "别的歌", "syncedLyrics": "[00:01.00] 不对"},
                {"trackName": "晴天", "syncedLyrics": "[00:02.00] 对了"},
            ],
        )

    result = _client(handler).fetch(TRACK)
    assert result.lyrics.ok and result.lyrics.lines[0].text == "对了"


def test_rate_limit_is_a_readable_reason() -> None:
    result = _client(lambda _request: httpx.Response(429)).fetch(TRACK)
    assert not result.lyrics.ok and "限流" in result.reason


def test_timeout_is_a_readable_reason() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    result = _client(handler).fetch(TRACK)
    assert not result.lyrics.ok and "超时" in result.reason


def test_disabled_makes_no_request() -> None:
    called: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={})

    client = RemoteLyrics(enabled=False, transport=httpx.MockTransport(handler))
    result = client.fetch(TRACK)
    assert not result.lyrics.ok and "没有开启" in result.reason
    assert called == []


def test_missing_artist_or_title_does_not_call_the_network() -> None:
    called: list[httpx.Request] = []
    client = _client(lambda request: called.append(request) or httpx.Response(200, json={}))
    result = client.fetch(Track("id", Path("x.mp3"), "只有标题"))
    assert not result.lyrics.ok and "缺少" in result.reason
    assert called == []
