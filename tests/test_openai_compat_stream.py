"""云端 SSE 流解析，以及**碎片累积**。

两个真实的失败模式，都在这里钉死：

1. **``delta.tool_calls`` 是分片的。** 第一片带 id 与工具名，参数是一段段拼出来的
   JSON 字符串。写错了的表现不是报错，是"模型明明调了工具，界面却什么都没发生"
   ——``agent/loop.py`` 是**覆盖**取值的，它拿到的会是半截参数，然后工具收到一个
   它看不懂的入参。
2. **``[DONE]`` 未必会来。** 有些反代把它吃掉。少发收尾那一片的后果同上：工具调用
   彻底消失，而账面上一切正常。

第 2 条尤其值得一条用例，因为它不会有任何症状指向这里。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from biosnn_memterm.llm.base import ChatBackendError
from biosnn_memterm.llm.openai_compat import _as_arguments, _iter_sse_chunks


class _FakeResponse:
    """只实现 ``iter_bytes`` 的假响应（与 ``test_ollama_stream`` 同一招）。

    :func:`_iter_sse_chunks` 只用这一个方法，所以不必去 mock httpx——测试越小越不
    容易因为库升级而失效。
    """

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = pieces

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._pieces


def _sse(payload: object) -> bytes:
    """一片 SSE 事件。``\\r\\n`` 行尾——真实服务端就是这么发的。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return b"data: " + body.encode("utf-8") + b"\r\n\r\n"


def _delta(**fields: object) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": dict(fields)}]}


def _chunks(*payloads: object) -> list:
    raw = b"".join(_sse(payload) for payload in payloads)
    return list(_iter_sse_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]


# ── 文本 ────────────────────────────────────────────────────────────────────


def test_parses_content_deltas_and_marks_the_end() -> None:
    chunks = _chunks(_delta(content="你好"), _delta(content="世界"), "[DONE]")

    assert [chunk.text for chunk in chunks] == ["你好", "世界", ""]
    assert chunks[-1].done is True
    assert all(not chunk.done for chunk in chunks[:-1]), "中途的片不该被当成结尾"


def test_reassembles_events_split_across_reads() -> None:
    """一次 read 切在事件中间——真实网络下的常态。"""
    whole = _sse(_delta(content="被切开的一句话")) + _sse("[DONE]")
    pieces = [whole[:20], whole[20:45], whole[45:]]
    chunks = list(_iter_sse_chunks(_FakeResponse(pieces)))  # type: ignore[arg-type]

    assert chunks[0].text == "被切开的一句话"
    assert chunks[-1].done is True


def test_skips_heartbeat_comments_and_blank_lines() -> None:
    """反代常拿 ``: ping`` 当心跳。不跳的话它每几秒被当成一个坏负载丢一次——
    而"偶尔丢一行"这种症状没人会怀疑到心跳头上。"""
    raw = (
        b": ping\r\n\r\n"
        + _sse(_delta(content="甲"))
        + b"\r\n"
        + _sse(_delta(content="乙"))
        + _sse("[DONE]")
    )
    chunks = list(_iter_sse_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["甲", "乙", ""]


def test_a_bad_json_event_is_skipped_not_fatal() -> None:
    raw = _sse("data: { this is not json") + _sse(_delta(content="还在")) + _sse("[DONE]")
    chunks = list(_iter_sse_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["还在", ""]


def test_an_error_inside_the_stream_is_raised() -> None:
    """服务端在流中间报错。装作没看见的话，用户看到的是"回复说着说着停了"。"""
    raw = _sse(_delta(content="开头")) + _sse({"error": {"message": "quota exceeded"}})

    with pytest.raises(ChatBackendError) as caught:
        list(_iter_sse_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]

    assert "quota exceeded" in str(caught.value)


def test_reasoning_content_never_becomes_text() -> None:
    """reasoner 的思考过程**不进正文**。进去的话它会显示在对话区、也会被写进逐字稿，
    而且因为它只在某些模型上出现，看起来就像"模型突然话多"。"""
    chunks = _chunks(
        _delta(reasoning_content="让我想想……"),
        _delta(reasoning_content="还在想……", content="答案是 42"),
        "[DONE]",
    )

    # 只有一片有字，思考那一片连空片都不发——界面上不该因为它多出一次刷新。
    assert [chunk.text for chunk in chunks] == ["答案是 42", ""]
    # 但它照样在 raw 里——厂商概念的唯一出口。
    assert chunks[0].raw["choices"][0]["delta"]["reasoning_content"] == "还在想……"


def test_a_content_array_is_flattened() -> None:
    """少数端点回分块数组形式。``[["text"]]`` 直接进 ``text`` 的话，界面上会出现
    一个 Python repr。"""
    chunks = _chunks(
        _delta(content=[{"type": "text", "text": "分块"}, {"type": "text", "text": "形式"}]),
        "[DONE]",
    )

    assert chunks[0].text == "分块形式"


# ── 工具调用：碎片累积 ──────────────────────────────────────────────────────


def _call_fragment(index: int, **fields: object) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, **fields}]}}]}


#: 一次真实形状的调用**被切成四片**。第一片带 id 与函数名，参数是四段拼起来的
#: JSON。拼完应当恰好是 ``{"text": "买牛奶", "at": "明天 9:00"}``。
_FRAGMENTS: list[dict[str, Any]] = [
    _call_fragment(0, id="call_abc", function={"name": "set_reminder", "arguments": '{"te'}),
    _call_fragment(0, function={"arguments": 'xt": "买牛奶", '}),
    _call_fragment(0, function={"arguments": '"at": "明天 9'}),
    _call_fragment(0, function={"arguments": ':00"}'}),
]


def test_arguments_are_stitched_across_four_fragments() -> None:
    """**这一条是整块设计的核心。**

    碎片累加写错（例如只取最后一片）不会报任何错——``arguments`` 那一串只是变成了
    ``':00"}'``，工具收到一个它看不懂的入参然后在内部炸掉，而错误信息把锅甩给工具。
    """
    chunks = _chunks(*_FRAGMENTS, "[DONE]")

    calls = chunks[-1].tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call_abc", "id 只在第一片里，丢了就配不上工具结果"
    assert calls[0].name == "set_reminder"
    assert calls[0].arguments == {"text": "买牛奶", "at": "明天 9:00"}


def test_tool_calls_only_ride_the_final_chunk() -> None:
    """碎片期间一片都不许带 ``tool_calls`` 出去。

    ``agent/loop.py`` 是**覆盖**取值（它照着 ``base.ChatChunk`` 那句"整条一次到齐"
    写的）。中途发出去的话，它拿到的会是半截参数——而那是本模块引入的、唯一能让
    那句注释变成假话的写法。
    """
    chunks = _chunks(*_FRAGMENTS, "[DONE]")

    assert all(not chunk.tool_calls for chunk in chunks[:-1])
    assert chunks[-1].tool_calls


def test_two_parallel_calls_do_not_merge() -> None:
    """两个 index 交错到达。按"当前片"累加的话，两次调用的参数会拼成一团，
    而那次调用看起来仍然"有一个名字"。"""
    chunks = _chunks(
        _call_fragment(0, id="call_a", function={"name": "read_file", "arguments": '{"pa'}),
        _call_fragment(1, id="call_b", function={"name": "list_files", "arguments": '{"di'}),
        _call_fragment(0, function={"arguments": 'th": "a.txt"}'}),
        _call_fragment(1, function={"arguments": 'r": "."}'}),
        "[DONE]",
    )

    calls = {call.id: call for call in chunks[-1].tool_calls}
    assert calls["call_a"].arguments == {"path": "a.txt"}
    assert calls["call_b"].arguments == {"dir": "."}
    assert calls["call_a"].name == "read_file"
    assert calls["call_b"].name == "list_files"


def test_a_fragment_without_index_continues_the_previous_call() -> None:
    """少数端点每片都不带 index。按位置数的话会把一次调用切成两次，而两次都没有
    完整参数——症状与"碎片累积写错"一模一样。"""
    chunks = _chunks(
        _call_fragment(0, id="call_a", function={"name": "f", "arguments": '{"x"'}),
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"function": {"arguments": ": 1}"}}]}}]},
        "[DONE]",
    )

    assert [call.arguments for call in chunks[-1].tool_calls] == [{"x": 1}]


def test_an_arguments_object_is_accepted() -> None:
    """有的端点不分片、直接给对象（与 Ollama 同形）。"""
    chunks = _chunks(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c",
                                "function": {"name": "f", "arguments": {"x": 1}},
                            }
                        ]
                    },
                }
            ]
        },
        "[DONE]",
    )

    assert chunks[-1].tool_calls[0].arguments == {"x": 1}


def test_a_nameless_fragment_is_dropped_not_fatal() -> None:
    """没有函数名的调用无从派发。丢掉它，而不是让整段回复作废。"""
    chunks = _chunks(
        _call_fragment(0, function={"arguments": "{}"}),
        _call_fragment(1, id="b", function={"name": "good", "arguments": '{"x": 1}'}),
        "[DONE]",
    )

    assert [call.name for call in chunks[-1].tool_calls] == ["good"]


def test_a_missing_id_is_synthesized_per_call() -> None:
    """id 缺失要合成，**而且两次调用不能拿到同一个**：同一段对话里两次 assistant
    消息都会有 ``index=0``，按 index 编号会让服务端把结果配到错误的请求上，
    而配错不报错。"""
    chunks = _chunks(
        _call_fragment(0, function={"name": "f", "arguments": "{}"}),
        _call_fragment(1, function={"name": "g", "arguments": "{}"}),
        "[DONE]",
    )

    ids = [call.id for call in chunks[-1].tool_calls]
    assert len(set(ids)) == 2
    assert all(ids)


def test_truncated_arguments_raise_instead_of_guessing() -> None:
    """拼出来不是合法 JSON 就抛——猜一个 ``{}`` 塞进去会让工具收到看不懂的参数、
    然后在内部炸掉，而错误会把锅甩给工具实现。"""
    with pytest.raises(ChatBackendError) as caught:
        _chunks(
            _call_fragment(0, id="c", function={"name": "set_reminder", "arguments": '{"te'}),
            "[DONE]",
        )

    assert "set_reminder" in str(caught.value)
    assert "合法 JSON" in str(caught.value)


def test_an_empty_arguments_string_means_no_arguments() -> None:
    """无参工具的 ``arguments`` 就是空串（或者干脆没有）。那不是"参数坏了"。"""
    chunks = _chunks(
        _call_fragment(0, id="c", function={"name": "now", "arguments": ""}),
        "[DONE]",
    )

    assert chunks[-1].tool_calls[0].arguments == {}
    assert _as_arguments("   ", "now") == {}


# ── 收尾 ────────────────────────────────────────────────────────────────────


def test_usage_only_chunk_lands_on_the_final_chunk() -> None:
    """开了 ``include_usage`` 之后，统计信息**单独占一片**，那一片的 ``choices``
    是空列表。把它当成畸形会丢掉 token 数——而它是上下文指示器唯一的数据源。"""
    chunks = _chunks(
        _delta(content="说完了"),
        {"choices": [], "usage": {"prompt_tokens": 31, "completion_tokens": 7}},
        "[DONE]",
    )

    assert chunks[-1].done is True
    assert chunks[-1].prompt_tokens == 31
    assert chunks[-1].completion_tokens == 7


def test_a_stream_without_done_still_ends_and_keeps_the_calls() -> None:
    """**没有 ``[DONE]`` 也要收尾。** 有些反代在连接收尾时把它吃掉。

    少发这一片的后果不是"少一段字"，是**工具调用彻底消失**：模型明明调了，闸门
    那边一个都没看见，这一轮静默地结束。这一条就是那个失败的守卫。
    """
    chunks = _chunks(*_FRAGMENTS)

    assert chunks[-1].done is True, "没收到 [DONE]，收尾片照样要发"
    assert chunks[-1].tool_calls[0].arguments == {"text": "买牛奶", "at": "明天 9:00"}


def test_an_empty_stream_still_produces_a_done_chunk() -> None:
    """一个字都没有的流（连接立刻断掉）。上层靠 ``done`` 复位状态，不发的话界面
    永远停在"正在生成…"。"""
    chunks = list(_iter_sse_chunks(_FakeResponse([])))  # type: ignore[arg-type]

    assert len(chunks) == 1
    assert chunks[0].done is True
    assert chunks[0].text == ""
