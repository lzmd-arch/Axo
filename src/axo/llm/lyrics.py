"""LRCLIB 客户端：本地没有歌词时，拿曲目元数据换一份同步歌词。

**只发元数据**：曲名、艺术家、专辑、时长。不发文件、不发路径、不发整库清单。
这一层与 :mod:`axo.llm.speech` 同姿态——协议形状在纯层
(:mod:`axo.music.lyrics`)，这里只有发请求与把失败收成一句话。

**超时压在 2 秒**：窗口关闭与程序退出会等歌词线程（``THREAD_WAIT_MS`` 是 3 秒），
让一次网络请求把退出拖进超时，代价比"这次没查到"大得多。
"""

from __future__ import annotations

import httpx

from ..music.lyrics import lrclib_query, parse_lrclib
from ..music.models import Lyrics, LyricsResult, Track

__all__ = ["RemoteLyrics"]

_BASE = "https://lrclib.net"
_TIMEOUT_S = 2.0
_UA = "Axo/0.10 (local music library; https://github.com/lzmd-arch/Axo)"


class RemoteLyrics:
    def __init__(
        self,
        *,
        enabled: bool = True,
        transport: httpx.BaseTransport | None = None,
        timeout: float = _TIMEOUT_S,
    ) -> None:
        self._enabled = bool(enabled)
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=_BASE,
            timeout=timeout,
            headers={"User-Agent": _UA},
            follow_redirects=True,
            transport=transport,
        )

    @property
    def available(self) -> bool:
        return self._enabled

    def close(self) -> None:
        self._client.close()

    def fetch(self, track: Track) -> LyricsResult:
        """先精确查，再按 ``q`` 搜一次。**不抛**：失败进 ``reason``。"""
        if not self._enabled:
            return LyricsResult(reason="在线歌词没有开启")
        if not track.artist or not track.title:
            return LyricsResult(reason="这首歌缺少艺术家或曲名，查不了在线歌词")
        try:
            response = self._client.get("/api/get", params=lrclib_query(track))
            if response.status_code == 200:
                lyrics = parse_lrclib(response.json())
                if lyrics.ok:
                    return LyricsResult(lyrics)
            elif response.status_code == 429:
                return LyricsResult(reason="LRCLIB 限流了，过一会儿再试")
            elif response.status_code not in (200, 404):
                return LyricsResult(reason=f"LRCLIB 返回了 {response.status_code}")

            found = self._search(track)
            if found is not None:
                return LyricsResult(found)
            return LyricsResult(reason="LRCLIB 没有收录这首的歌词")
        except httpx.TimeoutException:
            return LyricsResult(reason="LRCLIB 没有响应（超时）")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return LyricsResult(reason=f"LRCLIB 请求失败：{type(exc).__name__}")

    def _search(self, track: Track) -> Lyrics | None:
        response = self._client.get(
            "/api/search",
            params={"q": f"{track.artist} {track.title}", "track_name": track.title},
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, list):
            return None
        wanted_title = track.title.casefold()
        for hit in payload:
            if not isinstance(hit, dict):
                continue
            if str(hit.get("trackName") or "").casefold() != wanted_title:
                continue
            lyrics = parse_lrclib(hit)
            if lyrics.ok:
                return lyrics
        return None
