"""HTTP 错误 → 能直接显示给用户的句子，**且里面没有 key**。

两条各自独立的东西：

1. **状态码要指对方向。** 抄 Ollama 那份的后果是 404 的提示变成"先跑 ollama pull
   <模型名>"——云端下这句荒谬，而且它会把用户支去查一个根本不存在的本地模型。
   401 与"连不上"也必须分开：前者是 key 不对，后者是网络/地址，混在一起会让用户
   去查地址，而地址没问题。

2. **key 绝不进任何要给用户看的句子。** 它一旦进了错误串，就会出现在状态栏、日志、
   以及用户发出去的截图里。而"服务端把 key 原样回显"是真实存在的一种行为。
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from axo.llm.base import ChatBackendError
from axo.llm.openai_compat import (
    _error_detail,
    _raise_for_status,
    _wrap_transport_error,
    probe,
)

_KEY = "sk-0123456789abcdef0123456789abcdef"
_REQUEST = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")


def _unread(status: int, body: bytes) -> httpx.Response:
    """**还没被读过正文**的响应——正是 ``client.stream(...)`` 里的那样。

    传 ``stream=`` 而不是 ``content=``：后者在构造时就 ``read()`` 掉了，
    ``.text`` 不会抛——那样测的是另一件事（见 ``test_ollama_errors``）。
    """
    return httpx.Response(status, stream=httpx.ByteStream(body), request=_REQUEST)


def _message(status: int, body: bytes, *, api_key: str = _KEY) -> str:
    response = _unread(status, body)
    with pytest.raises(ChatBackendError) as caught:
        _raise_for_status(response, what="对话（deepseek-chat）", api_key=api_key)
    return str(caught.value)


def _fake_get(body: str, *, status: int = 200) -> Any:
    """替掉 ``httpx.Client.get`` 的替身工厂。

    写成工厂而不是就地一个 ``lambda``：绑上去的是**未绑定的方法**，第一个参数是
    ``self``，而 ``self`` / ``path`` 在替身里都用不上——就地写就得给三个用不着的
    参数起名，而 ruff 的 ``ARG005`` 不认 ``lambda`` 里的下划线前缀。
    """

    def fake(_client: httpx.Client, _path: str, **_kw: Any) -> httpx.Response:
        return httpx.Response(status, text=body)

    return fake


def test_a_streaming_error_becomes_a_sentence_not_a_traceback() -> None:
    """流式请求撞上非 2xx 时，正文还在网络上——此刻取 ``.text`` 抛
    ``httpx.ResponseNotRead``，于是一条干净的错误变成一个 httpx 回溯。"""
    text = _message(400, b'{"error": {"message": "invalid model"}}')

    assert "invalid model" in text
    assert "HTTP 400" in text


def test_each_status_code_points_somewhere_useful() -> None:
    """各状态码那句话。**互相不能换**：指错方向比没有提示更费时间。"""
    assert "API key 无效" in _message(401, b'{"error":{"message":"auth failed"}}')
    assert "余额不足" in _message(402, b'{"error":{"message":"insufficient balance"}}')
    assert "权限" in _message(403, b'{"error":{"message":"forbidden"}}')
    assert "限流" in _message(429, b'{"error":{"message":"rate limited"}}')


def test_404_talks_about_the_endpoint_not_about_ollama() -> None:
    """**这一条是"不能复用 ollama 那份"的守卫。** ollama 的 404 提示会让人去
    ``ollama pull`` 一个云端的模型名。"""
    text = _message(404, b'{"error":{"message":"Not Found"}}')

    assert "/v1" in text
    assert "ollama pull" not in text


def test_5xx_says_it_is_not_a_local_problem() -> None:
    text = _message(500, b'{"error":{"message":"internal error"}}')

    assert "云端服务端报错" in text
    assert "ollama" not in text.lower()


def test_a_non_json_error_body_is_still_shown() -> None:
    """反代返回的 HTML 错误页。解析不了就退化成"无附加信息"，等于把唯一的线索丢掉。"""
    text = _message(502, b"<html><body>Bad Gateway</body></html>")

    assert "Bad Gateway" in text


def test_an_empty_error_body_says_so_instead_of_showing_nothing() -> None:
    assert "无附加信息" in _message(503, b"")


def test_the_error_message_is_unwrapped_from_its_extra_layer() -> None:
    """OpenAI 兼容端点把话藏在 ``error.message`` 里——**比 Ollama 多一层嵌套**。
    不剥的话用户看到的是整段 ``{'message': ..., 'type': ...}`` 的 repr。"""
    assert _error_detail('{"error": {"message": "boom", "type": "x"}}') == "boom"
    assert _error_detail('{"error": "plain"}') == "plain"


# ── key 不外泄 ──────────────────────────────────────────────────────────────


def test_a_server_that_echoes_the_key_back_does_not_leak_it() -> None:
    """**这一条是这个模块的安全底线。**

    服务端（或者中间的反代/网关）把 key 原样回显到错误体里——这是真实存在的一种
    行为，而且回显是最容易被我们原样转给用户的那一种。
    """
    text = _message(401, f'{{"error": {{"message": "bad key {_KEY}"}}}}'.encode())

    assert _KEY not in text
    assert "sk-" not in text, "按形状兜的那一道也要生效"


def test_a_key_shaped_string_from_elsewhere_is_also_masked() -> None:
    """按形状兜那一道，抓的是被引号、前缀或日志格式动过手脚的回显——第一道
    （精确替换）匹配不上，形状还认得出来。"""
    text = _message(401, b'{"error":{"message":"key sk-ffffffffffffffffffff rejected"}}')

    assert "sk-ffffff" not in text


def test_a_transport_error_carries_the_address_but_not_the_key() -> None:
    exc = httpx.ConnectError("connection refused")
    with pytest.raises(ChatBackendError) as caught:
        _wrap_transport_error(exc, "https://api.deepseek.com/v1", what="对话（x）", api_key=_KEY)

    text = str(caught.value)
    assert "连不上 https://api.deepseek.com/v1" in text
    assert "connection refused" in text
    assert _KEY not in text


def test_a_successful_response_is_never_read() -> None:
    """**成功路径一个字都不许读。**

    这个响应紧接着要交给 ``_iter_sse_chunks`` 去逐片解析——在这里顺手 ``read()``
    掉，上层就什么都收不到了。
    """
    response = _unread(200, b"data: [DONE]\n\n")

    _raise_for_status(response, what="对话（x）", api_key=_KEY)

    assert not response.is_stream_consumed, "成功路径提前读掉了正文，上层会收到一个空流"


# ── 探活 ────────────────────────────────────────────────────────────────────


def test_probe_reports_a_missing_key_as_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """地址对、能列模型、但没填 key——**这是通的，只是聊不了**。

    报成"失败"会让人以为地址写错了。而这句话必须在一次点击之后就说出来：等到用户
    打完一整句话再被 401 拦下，那一次输入就白费了。
    """
    monkeypatch.setattr(httpx.Client, "get", _fake_get('{"data":[]}'))

    ok, detail = probe("https://api.deepseek.com/v1", api_key="")

    assert ok is True
    assert "API key" in detail


def test_probe_returns_the_reason_instead_of_just_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """401 → 一句话说明是 key 的问题，而不是笼统的"连不上"。"""
    monkeypatch.setattr(
        httpx.Client, "get", _fake_get('{"error":{"message":"bad key"}}', status=401)
    )

    ok, detail = probe("https://api.deepseek.com/v1", api_key=_KEY)

    assert ok is False
    assert "API key 无效" in detail


def test_probe_never_raises_on_an_untrusted_address() -> None:
    """地址不可信时 ``_client`` 会抛——探活也要把它变成一句话，因为调用方是界面。"""
    ok, detail = probe("http://evil.example.com/v1", api_key=_KEY)

    assert ok is False
    assert "https" in detail
