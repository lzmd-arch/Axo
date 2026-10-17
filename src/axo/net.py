"""联网读：DeepSeek 原生的 Search / Fetch。

**为什么用官方这两个端点而不是自己抓网页**：用户手里已经有一把 DeepSeek 的 key，
同一把 key 能调 ``/v1/tools/search`` 与 ``/v1/tools/fetch``（5 元/千次，失败不计费）。
自己抓 Bing/Google 既踩 ToS 又脆弱；自己抓任意 URL 还要处理 SSRF——而官方 Fetch
替我们做了这一层。

**这不是 axo.agent 里的纯层**：它真的要发 HTTP，所以 httpx 在函数内 import，
工具侧通过 :class:`~axo.agent.tools.ToolContext` 注入一个实例。CI 里注入替身，
一次网络都不发。

**默认不登记**：联网是出网行为，URL 会离开本机。设置里打开 ``web_enabled`` 之后
``MainWindow._tools()`` 才把这两个工具交给模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULT_SEARCH_URL", "DeepSeekWeb", "WebError", "WebHit"]

DEFAULT_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
DEFAULT_FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"

#: 一次搜索最多要多少条。官方上限 50；这里默认 8，够回答"最近发生了什么"。
DEFAULT_COUNT = 8

#: 抓回来的正文最多回灌多少字符。**与读文件同一个理由**：模型读到的是片段时，
#: 它必须知道自己读到的是片段。
MAX_FETCH_CHARS = 24_000

_TIMEOUT = 30.0


class WebError(RuntimeError):
    """搜不到/抓不到。消息是写给模型看的。"""


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    date: str = ""


class DeepSeekWeb:
    """DeepSeek 原生 Search / Fetch 的薄封装。

    Args:
        api_key: DeepSeek 的 key。空串时 :func:`available` 为假，工具不登记。
        search_url / fetch_url: 端点。默认官方地址；自建网关可以改。
    """

    def __init__(
        self,
        api_key: str,
        *,
        search_url: str = DEFAULT_SEARCH_URL,
        fetch_url: str = DEFAULT_FETCH_URL,
    ) -> None:
        self._key = api_key.strip()
        self.search_url = search_url
        self.fetch_url = fetch_url

    @property
    def available(self) -> bool:
        return bool(self._key)

    def search(self, query: str, *, count: int = DEFAULT_COUNT) -> list[WebHit]:
        """搜网页。**空结果是正常结果**（没有命中），不是错误。"""
        query = query.strip()
        if not query:
            raise WebError("query 是空的，请给出要搜的内容。")
        if not self._key:
            raise WebError("没有可用的 DeepSeek API key，联网搜索不可用。")
        count = max(1, min(50, int(count)))
        payload = self._post(
            self.search_url,
            {"query": query[:128], "count": count, "need_content": False},
            what="联网搜索",
        )
        raw = payload.get("results")
        if not isinstance(raw, list):
            return []
        hits: list[WebHit] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            hits.append(
                WebHit(
                    title=str(entry.get("title") or "").strip(),
                    url=str(entry.get("url") or "").strip(),
                    snippet=str(entry.get("snippet") or "").strip(),
                    summary=str(entry.get("summary") or "").strip(),
                    date=str(entry.get("date") or "").strip(),
                )
            )
        return hits

    def fetch(self, url: str) -> tuple[str, str]:
        """抓一页，返回 ``(标题, 正文 Markdown)``。正文截到 :data:`MAX_FETCH_CHARS`。"""
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            raise WebError("只抓 http/https 地址。")
        if not self._key:
            raise WebError("没有可用的 DeepSeek API key，联网抓取不可用。")
        payload = self._post(self.fetch_url, {"url": url}, what="抓取网页")
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "")
        if len(content) > MAX_FETCH_CHARS:
            content = (
                content[:MAX_FETCH_CHARS]
                + f"\n\n[…]（正文只给了前 {MAX_FETCH_CHARS} 字，后面还有，"
                "需要更多时请直接告诉用户去看原文）"
            )
        if not content.strip():
            raise WebError("那一页没有可读正文。")
        return title, content

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _post(self, url: str, body: dict[str, Any], *, what: str) -> dict[str, Any]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                response = client.post(url, json=body, headers=headers)
        except httpx.RequestError as exc:
            raise WebError(f"{what}失败：连不上 {url}（{type(exc).__name__}）。") from exc
        if not response.is_success:
            detail = response.text.strip()[:300]
            raise WebError(f"{what}失败：HTTP {response.status_code} {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebError(f"{what}返回的不是 JSON。") from exc
        if not isinstance(payload, dict):
            raise WebError(f"{what}返回的形状不认识。")
        return payload
