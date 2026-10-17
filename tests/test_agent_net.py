"""联网读：DeepSeek 原生 Search / Fetch 的解析与失败。**一次网络都不发。**"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from axo.net import DeepSeekWeb, WebError


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.is_success = 200 <= status < 300
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    payload: dict[str, Any] = {}
    status = 200

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        return

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> FakeResponse:
        FakeClient.calls.append((url, json, headers))
        return FakeResponse(FakeClient.payload, FakeClient.status)


@pytest.fixture(autouse=True)
def fake_http(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.calls = []
    FakeClient.payload = {}
    FakeClient.status = 200
    monkeypatch.setattr(httpx, "Client", FakeClient)


def test_no_key_means_unavailable() -> None:
    assert DeepSeekWeb("").available is False
    assert DeepSeekWeb("  ").available is False
    assert DeepSeekWeb("sk-x").available is True


def test_search_parses_results_and_sends_the_key() -> None:
    FakeClient.payload = {
        "results": [
            {
                "title": "标题",
                "url": "https://example.com/a",
                "snippet": "摘要",
                "summary": "总结",
                "date": "2026-10-01",
            }
        ]
    }
    web = DeepSeekWeb("sk-test")

    hits = web.search("DeepSeek", count=5)

    assert [hit.title for hit in hits] == ["标题"]
    assert hits[0].url == "https://example.com/a"
    url, body, headers = FakeClient.calls[0]
    assert url.endswith("/v1/tools/search")
    assert body == {"query": "DeepSeek", "count": 5, "need_content": False}
    assert headers["Authorization"] == "Bearer sk-test"


def test_an_empty_result_list_is_not_an_error() -> None:
    FakeClient.payload = {"results": []}

    assert DeepSeekWeb("sk-test").search("没有这回事") == []


def test_fetch_returns_title_and_content_and_truncates() -> None:
    FakeClient.payload = {"title": "一页", "content": "正" * 30_000}

    title, content = DeepSeekWeb("sk-test").fetch("https://example.com")

    assert title == "一页"
    assert content.startswith("正")
    assert "只给了前" in content
    assert len(content) < 30_000


def test_fetch_refuses_non_http_urls() -> None:
    with pytest.raises(WebError, match="http"):
        DeepSeekWeb("sk-test").fetch("file:///etc/passwd")


def test_http_errors_become_a_sentence() -> None:
    FakeClient.status = 403
    FakeClient.payload = {"error": "blocked"}

    with pytest.raises(WebError, match="HTTP 403"):
        DeepSeekWeb("sk-test").search("x")


def test_a_missing_key_is_refused_in_plain_words() -> None:
    with pytest.raises(WebError, match="API key"):
        DeepSeekWeb("").search("x")
