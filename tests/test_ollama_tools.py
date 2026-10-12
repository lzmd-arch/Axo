"""工具调用的**两端**：我们发出去的声明，与拿回来的调用。

:func:`_build_tool` / :func:`_parse_tool_calls` / :func:`_as_arguments` /
:func:`_build_message` 里那几个分支曾经**零生产调用、零测试覆盖**，整条 P3 路线的
地基只是一句注释。2026-10-10 用 ``tools/smoke_agent.py`` 对 ``qwen3-4b`` 真跑了一轮，
结论是**形状与注释所述完全一致**，并把它冻进了 ``fixtures/ollama_tool_calls.json``。
本文件把那份字节喂给真正的解析器——**探针一次，回归永久**：以后谁动了
``_parse_tool_calls``，这些断言会立刻红。

夹具里能直接看出来的三件事（也就是 Ollama 与 OpenAI 的三处差异）：

1. ``tool_calls`` 与 ``done: true`` 在**同一片**到达，不分片——所以消费端不需要
   任何碎片累积状态机；
2. 返回的 ``tool_calls`` **没有** ``{"type": "function"}`` 外壳（而我们**发出去**的
   声明**要**这个外壳，方向不对称）；
3. ``function.arguments`` 是**对象**，不是 OpenAI 那种 JSON 字符串。

第 3 条是 :class:`ToolCall.arguments` 存 ``dict`` 而不是 ``str`` 的理由，也是
:func:`_as_arguments` 必须同时接受两种形状的理由——云端 OpenAI 兼容端点是字符串。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axo.llm.base import (
    ChatBackendError,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolSpec,
)
from axo.llm.ollama import (
    OllamaChatBackend,
    _build_message,
    _build_options,
    _build_tool,
    _parse_line,
    _parse_tool_calls,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ollama_tool_calls.json"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _chunks() -> list[dict]:
    return [entry["chunk"] for entry in _fixture()["events"]]


#: ``_build_payload`` 是实例方法，但**不能构造真的实例**：``__post_init__`` 会去
#: 问服务端要上下文窗口（``query_context_window``），核心腿一旦联网就不再是核心腿
#: ——本仓的第一条承诺就是"一份召回基准测试不该需要装 Qt/Ollama 才能跑"。
#: 这里只喂它真正读的两个属性，非绑定调用。多读一个属性这条会立刻 AttributeError，
#: 那正是想要的：它提醒把这个替身补齐，而不是悄悄跳过。
_STAND_IN = SimpleNamespace(model="qwen3-4b:latest", keep_alive="5m")


def _payload(request: ChatRequest) -> dict:
    return OllamaChatBackend._build_payload(_STAND_IN, request)  # type: ignore[arg-type]


# ── 真实字节：入参形状 ─────────────────────────────────────────────────────


def test_fixture_is_real_ollama_output() -> None:
    """夹具自称的出身必须能核对。

    这不是形式主义：一份**编出来的**夹具会让下面每条断言都变成"我们的想象与我们的
    想象一致"。``provenance`` 块是这份文件的说明书，读它的人要能追到怎么重放。
    """
    doc = _fixture()
    assert doc["events"], "夹具是空的——那下面的断言全是空转。"
    assert doc["provenance"]["via"].startswith("tools/smoke_agent.py")
    assert doc["provenance"]["model"].startswith("qwen3-4b")


def test_every_fixture_chunk_carries_tool_calls() -> None:
    """反空转：夹具里每一片都真的带工具调用。

    否则把 ``events`` 清空、或塞进四条纯文本片，下面那些循环断言会**全部通过**
    ——一个守卫最坏的失败方式不是漏报，是报绿。
    """
    for chunk in _chunks():
        calls = chunk["message"]["tool_calls"]
        assert calls, f"夹具里混进了不带 tool_calls 的片：{chunk}"


def test_tool_calls_arrive_whole_alongside_done() -> None:
    """第 1 条差异：工具调用与流结束标记**在同一片**。

    这条是消费端"不需要碎片累积状态机"这个结论的全部依据。若将来某版 Ollama 把
    ``tool_calls`` 挪到单独一片，**这份夹具不会自己变红**（字节是冻的）——它红在
    :func:`test_stream_reassembles_tool_calls_from_fixture_bytes` 上，那一条把夹具
    当**逐字节的流**喂给 ``_iter_ndjson_chunks``，测的是"按片收集"这条路径本身。
    """
    for chunk in _chunks():
        assert chunk["done"] is True
        assert chunk["message"]["tool_calls"]


def test_response_has_no_openai_type_wrapper() -> None:
    """第 2 条差异：**返回**的 tool_call 没有 ``type: function`` 外壳。

    与 :func:`test_build_tool_adds_the_openai_wrapper` 是一对：入参要有、出参没有。
    写反了的话，``_parse_tool_calls`` 会读到 ``function is None`` 而静默丢掉每一次
    调用——模型"调了工具但什么也没发生"。
    """
    for chunk in _chunks():
        for call in chunk["message"]["tool_calls"]:
            assert "type" not in call, f"Ollama 的返回里出现了 OpenAI 的外壳：{call}"
            assert "function" in call


def test_arguments_are_objects_not_json_strings() -> None:
    """第 3 条差异：``arguments`` 是对象。

    这一条直接决定 :class:`ToolCall` 的字段类型。夹具里既有空对象（``get_time``
    无参）也有带键的对象（``search_memory``），两种都要在。
    """
    shapes = [
        call["function"]["arguments"]
        for chunk in _chunks()
        for call in chunk["message"]["tool_calls"]
    ]
    assert all(isinstance(value, dict) for value in shapes)
    assert any(value for value in shapes), "夹具里没有一条带参数的调用——参数形状没被覆盖。"
    assert any(not value for value in shapes), "夹具里没有一条无参调用——空参数路径没被覆盖。"


# ── 真实字节：解析器读出来什么 ─────────────────────────────────────────────


def test_parse_line_reads_fixture_bytes() -> None:
    """把夹具的**原始字节**喂给 ``_parse_line``，逐条核对。

    走 ``_parse_line`` 而不是直接调 ``_parse_tool_calls``：前者才是流里的真实入口，
    它还要负责 ``done`` / token 数 / ``raw`` 的填充。只测内层等于漏掉外层。
    """
    seen: list[ToolCall] = []
    for chunk in _chunks():
        line = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
        parsed = _parse_line(line)
        assert parsed is not None
        assert parsed.done is True
        assert parsed.text == "", "带工具调用的片不该同时吐正文（夹具实测如此）"
        assert parsed.tool_calls, f"这一片没解出工具调用：{chunk}"
        seen.extend(parsed.tool_calls)

    assert len(seen) == 4
    assert [call.name for call in seen] == [
        "get_time",
        "search_memory",
        "search_memory",
        "write_memory",
    ]
    # id 逐条非空：回灌时工具结果靠它配对，丢了就配不上，而**配错不报错**。
    assert all(call.id for call in seen)
    assert len({call.id for call in seen}) == len(seen), "夹具里出现了重复 id，配对会串。"
    # 参数按名字核对，两个 search_memory 的 query 不同——正是它们把那两次调用
    # 区分开。
    assert seen[0].arguments == {}
    assert seen[1].arguments == {"query": "备用钥匙"}
    assert seen[3].arguments and "content" in seen[3].arguments


def test_stream_reassembles_tool_calls_from_fixture_bytes() -> None:
    """按**流**走一遍：夹具字节被切成不规则的块，收集端仍然拿到完整调用。

    模拟的是真实网络的不对齐分片（与 ``test_ollama_stream.py`` 同一手法）。这条是
    "消费端不需要碎片累积状态机"的**执行版**依据：切得再碎，每片自己都是完整的
    JSON。
    """
    from axo.llm.ollama import _iter_ndjson_chunks

    class _Response:
        def __init__(self, pieces: list[bytes]) -> None:
            self._pieces = pieces

        def iter_bytes(self):  # noqa: ANN201 — 与 httpx 的鸭子形状对齐
            yield from self._pieces

    whole = b"".join(
        (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8") for chunk in _chunks()
    )
    # 切点刻意错开行边界：每 97 字节一刀，与任何一行都不对齐。
    pieces = [whole[i : i + 97] for i in range(0, len(whole), 97)]

    collected: list[ToolCall] = []
    for chunk in _iter_ndjson_chunks(_Response(pieces)):  # type: ignore[arg-type]
        collected.extend(chunk.tool_calls)

    assert [call.name for call in collected] == [
        "get_time",
        "search_memory",
        "search_memory",
        "write_memory",
    ]


# ── 出参：我们发出去的声明 ─────────────────────────────────────────────────


def test_build_tool_adds_the_openai_wrapper() -> None:
    """**入参**要 ``type: function`` 外壳——与返回方向相反，注意别写反。"""
    spec = ToolSpec(name="get_time", description="查时间", parameters={"type": "object"})
    built = _build_tool(spec)
    assert built["type"] == "function"
    assert built["function"]["name"] == "get_time"
    assert built["function"]["parameters"] == {"type": "object"}


def test_build_tool_supplies_empty_schema_when_parameters_missing() -> None:
    """``parameters`` 缺省时补一个合法的空对象 schema，而不是留 ``None``。

    JSON Schema 的位置上放 ``None``，严格的端点会拒掉整个请求——症状是"加了工具
    就发不出请求"，而锅看起来在工具身上。
    """
    built = _build_tool(ToolSpec(name="ping", description=""))
    assert built["function"]["parameters"] == {"type": "object", "properties": {}}


def test_payload_omits_tools_key_when_empty() -> None:
    """``tools`` 为空时**整个键都不出现**。

    不是"省略一个空数组"这么轻：``ChatRequest.tools`` 的 docstring 承诺"不带工具的
    请求与加这个字段之前逐字节一致"，这条就是那个承诺的执行版。多一个 ``"tools": []``
    足以让某些端点改变行为。
    """
    request = ChatRequest(messages=[ChatMessage(role="user", content="你好")])
    assert "tools" not in _payload(request)


def test_payload_includes_tools_when_present() -> None:
    """有工具时逐条套壳，且条数一致——少一条就是模型看不见那个工具。"""
    specs = (
        ToolSpec(name="get_time", description="查时间"),
        ToolSpec(name="search_memory", description="查记忆"),
    )
    request = ChatRequest(messages=[ChatMessage(role="user", content="你好")], tools=specs)
    payload = _payload(request)
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "get_time",
        "search_memory",
    ]


def test_think_goes_top_level_not_into_options() -> None:
    """``think`` 属于请求体顶层。落进 ``options`` 会被 Ollama **静默忽略**。

    静默忽略是这里唯一值得测的失败模式：症状是"明明关了思考，token 账还是开着思考
    的数"，不报错、不掉字。2026-10-10 的实测显示开着思考时输出 token 涨到 3–10 倍
    且首字延迟从 0.06s 涨到 1–2s，所以这个开关真的会被用到。
    """
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="你好")], extra={"think": False}
    )
    payload = _payload(request)
    assert payload["think"] is False
    assert "think" not in _build_options(request)


# ── 回灌：把调用与结果配回去 ───────────────────────────────────────────────


def test_build_message_serializes_tool_calls_with_id_but_without_index() -> None:
    """assistant 行回灌时带 ``id``，**不带** Ollama 自己的 ``index``。

    ``index`` 是 Ollama 输出时的内部编号，不是协议的一部分；带上等于把它当接口。
    实测不带 ``index`` 回灌，它照样把结果配到了正确的那次调用上。
    """
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="abc123", name="get_time", arguments={}),),
    )
    payload = _build_message(message)
    assert payload["tool_calls"] == [
        {"id": "abc123", "function": {"name": "get_time", "arguments": {}}}
    ]
    assert "index" not in payload["tool_calls"][0]["function"]


def test_build_message_writes_tool_call_id_on_tool_rows() -> None:
    """tool 行必须带 ``tool_call_id``——它靠这个找到自己对应的那次调用。"""
    payload = _build_message(
        ChatMessage(role="tool", content="2026-10-10 12:59:39", tool_call_id="abc123")
    )
    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "abc123"


def test_round_trip_of_a_real_call() -> None:
    """夹具里的真实调用 → 回灌序列化 → 仍是可配对的形状。

    这一条把两端接起来：解析出来的东西能不能原样发回去。中间少了 ``id`` 或形状
    变了，配对就断在服务端，而服务端**不报错**。
    """
    chunk = _chunks()[1]
    calls = _parse_tool_calls(chunk["message"])
    assert len(calls) == 1

    payload = _build_message(ChatMessage(role="assistant", content="", tool_calls=calls))
    sent = payload["tool_calls"][0]
    assert sent["id"] == calls[0].id
    assert sent["function"]["name"] == "search_memory"
    assert sent["function"]["arguments"] == {"query": "备用钥匙"}


# ── 畸形输入 ───────────────────────────────────────────────────────────────


_MISSING = object()


def _call(
    *,
    _id: object = "x1",
    name: str = "search_memory",
    arguments: object = _MISSING,
) -> dict:
    """一条 Ollama 形状的 tool_call。

    ``id`` 是 tool_call 的**顶层**键、``name``/``arguments`` 在 ``function`` 里
    ——这条分界正是夹具里那三处差异的形态，helper 要是把它抹平了，测出来的东西
    就不是协议。

    三个参数都要能表达"显式给了个怪值"：``_id=None``（服务端没返回 id）、
    ``arguments=None``（无参调用的另一种写法）、``name=""``（畸形）。所以
    ``arguments`` 用哨兵而不是默认 ``{}``。
    """
    function: dict[str, object] = {
        "name": name,
        "arguments": {} if arguments is _MISSING else arguments,
    }
    return {"id": _id, "function": function}


def test_missing_id_is_synthesized_and_unique_per_call() -> None:
    """没有 ``id`` 就合成，且**两次合成不相等**。

    为什么不是 ``call-{index}``：同一段对话里两条 assistant 消息都会有 ``index=0``
    的那次调用，那种 id 会在服务端把结果配到**错误**的请求上——而配错不报错。
    老版本 Ollama 与部分兼容端点确实不返回 id，所以这条不是假设。
    """
    parsed = _parse_tool_calls({"tool_calls": [_call(_id=None), _call(_id=None)]})
    assert len(parsed) == 2
    assert all(call.id for call in parsed)
    assert parsed[0].id != parsed[1].id


def test_openai_shaped_string_arguments_are_parsed() -> None:
    """OpenAI 兼容端点把 ``arguments`` 给成 JSON **字符串**，同样要能读。

    这是 ``ChatBackend`` 抽象存在的意义：换端点不该换上层代码。
    """
    parsed = _parse_tool_calls({"tool_calls": [_call(arguments='{"query": "钥匙"}')]})
    assert parsed[0].arguments == {"query": "钥匙"}


def test_openai_shaped_wrapper_is_tolerated() -> None:
    """带 ``type: function`` 外壳的调用也读得出来——兼容端点会给。"""
    parsed = _parse_tool_calls(
        {
            "tool_calls": [
                {"type": "function", "id": "x", "function": {"name": "ping", "arguments": {}}}
            ]
        }
    )
    assert [call.name for call in parsed] == ["ping"]


@pytest.mark.parametrize(
    "bad",
    ["不是 JSON 的字符串", "[1, 2, 3]", "123", "True", [1], 42],
    ids=["非JSON字符串", "JSON数组字符串", "JSON数字字符串", "JSON布尔字符串", "数组", "数字"],
)
def test_unparseable_arguments_raise(bad: object) -> None:
    """读不成 dict 就抛，**不猜**。

    猜一个 dict 塞进去的后果是：工具收到看不懂的参数、在内部炸掉，错误信息把锅
    甩给工具实现。抛在这里，错就明确挂在"端点形状与假设不符"上。
    """
    with pytest.raises(ChatBackendError):
        _parse_tool_calls({"tool_calls": [_call(arguments=bad)]})


def test_none_arguments_become_empty_dict() -> None:
    """``arguments`` 是 ``None`` 当无参调用——这是正常形状，不是错误。"""
    parsed = _parse_tool_calls({"tool_calls": [_call(arguments=None)]})
    assert parsed[0].arguments == {}


def test_nameless_call_is_skipped_not_fatal() -> None:
    """没名字的调用跳过，其余照常解析。

    一条畸形的 tool_call 不该让整段回复作废；而"模型想调一个不存在的工具"是模型的
    问题，不是协议的问题。
    """
    parsed = _parse_tool_calls(
        {"tool_calls": [_call(name=""), _call(name="ping"), "不是字典", {"id": "y"}]}
    )
    assert [call.name for call in parsed] == ["ping"]


@pytest.mark.parametrize(
    "message", [{}, {"tool_calls": None}, {"tool_calls": "x"}, {"tool_calls": 3}]
)
def test_absent_or_non_list_tool_calls_yield_empty(message: dict) -> None:
    """纯文本回复（没有 ``tool_calls`` 键，或它是个怪值）返回空元组，不抛。"""
    assert _parse_tool_calls(message) == ()
