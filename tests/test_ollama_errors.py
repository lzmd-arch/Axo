"""HTTP 错误 → 能直接显示给用户的句子。

这一层唯一真实的失败模式不是"翻译错了"，是**在流式路径上根本翻不
出来**：``client.stream(...)`` 返回时正文还在网络上，此刻取 ``response.text`` 抛
``httpx.ResponseNotRead``——于是一条干净的错误（"模型没下载，先 ollama pull"）变成
一个 httpx 回溯。触发条件是**流式请求撞上非 2xx**：模型名写错、Ollama 没起、上下文
超限。全是用户最常撞上的那几种，也正是最需要一句人话的时候。

所以这里第一条守卫复现的就是那个现场：**一个没被读过的响应**。传 ``content=`` 构造
出来的响应在构造时就已读好，``.text`` 不会抛，也就复现不出这个 bug——测试会绿着，
而线上照旧给回溯。
"""

from __future__ import annotations

import httpx
import pytest

from axo.llm.base import ChatBackendError
from axo.llm.ollama import _raise_for_status

_REQUEST = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")


def _unread(status: int, body: bytes) -> httpx.Response:
    """**还没被读过正文**的响应——正是 ``client.stream(...)`` 里的那样。

    传 ``stream=`` 而不是 ``content=``：后者在构造时就 ``read()`` 掉了，
    ``_content`` 已就位，``.text`` 不会抛——那样测的是另一件事。
    """
    return httpx.Response(status, stream=httpx.ByteStream(body), request=_REQUEST)


def test_a_streaming_404_becomes_a_sentence_not_a_traceback() -> None:
    """这一条就是那个 bug 的现场。修之前它抛的是 ``httpx.ResponseNotRead``。"""
    response = _unread(404, b'{"error":"model \'qwen9\' not found, try pulling it first"}')

    with pytest.raises(ChatBackendError) as caught:
        _raise_for_status(response, what="对话（qwen9）")

    message = str(caught.value)
    assert "model 'qwen9' not found, try pulling it first" in message
    assert "HTTP 404" in message
    # 服务端的话要原样带出来，加的那句提示是给"模型没下载"这条最常见原因的。
    assert "ollama pull" in message


def test_the_server_side_hint_is_attached_on_5xx() -> None:
    """5xx 给的是"服务端自己报错"那句，而不是"模型没下载"。两条提示不能互换——
    指错方向会让人去重装一个本来就装好的模型。"""
    response = _unread(500, b'{"error":"out of memory"}')

    with pytest.raises(ChatBackendError) as caught:
        _raise_for_status(response, what="对话（qwen3-4b）")

    assert "out of memory" in str(caught.value)
    assert "Ollama 服务端自身报错" in str(caught.value)
    assert "ollama pull" not in str(caught.value)


def test_a_non_json_error_body_is_still_shown() -> None:
    """错误体不是 JSON 时（反代返回的 HTML 错误页），正文要原样带出来，
    不能因为"解析不了"就退化成一句"无附加信息"。"""
    response = _unread(502, b"<html><body>Bad Gateway</body></html>")

    with pytest.raises(ChatBackendError) as caught:
        _raise_for_status(response, what="对话（x）")

    assert "Bad Gateway" in str(caught.value)


def test_an_empty_error_body_says_so_instead_of_showing_nothing() -> None:
    """正文为空时给一句"无附加信息"——否则错误末尾会拖着一段空白，
    看起来像消息被截断了。"""
    response = _unread(503, b"")

    with pytest.raises(ChatBackendError) as caught:
        _raise_for_status(response, what="对话（x）")

    assert "无附加信息" in str(caught.value)


def test_a_successful_response_is_never_read() -> None:
    """**成功路径一个字都不许读。**

    这个响应在真实调用里紧接着要交给 :func:`_iter_ndjson_chunks` 去逐行解析——
    在这里顺手 ``read()`` 掉，上层就什么都收不到了。这正是修 bug 时最容易顺手
    犯的错（"反正读一下没坏处"），所以单独钉一条。
    """
    response = _unread(200, b'{"message":{"role":"assistant","content":"hi"},"done":true}\n')

    _raise_for_status(response, what="对话（x）")

    assert not response.is_stream_consumed, "成功路径提前读掉了正文，上层会收到一个空流"
