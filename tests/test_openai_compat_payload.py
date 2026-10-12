"""请求体长什么样。

**这里一个真实例都不构造。** :meth:`OpenAICompatChatBackend.__post_init__` 会去问
端点上下文窗口——那是一次网络请求。用一个**没过 ``__post_init__``** 的实例（见
:func:`_bare`）非绑定地调 ``_build_payload``，测的就是纯粹的映射。

其中 ``_build_message`` 那一条最要紧：**回灌方向与 Ollama 差两处**，而且差的方式
是"发出去不报错、只是模型看不懂"。``arguments`` 是对象还是 JSON 字符串、外壳有没有
``type``，这两处错了都不会得到一条清晰的错误。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from axo.llm.base import (
    ChatBackendError,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolSpec,
)
from axo.llm.openai_compat import (
    DEFAULT_OPENAI_MODEL,
    OpenAICompatChatBackend,
    _build_message,
    _build_tool,
)


def _bare(**fields: Any) -> OpenAICompatChatBackend:
    """一个**没过 ``__post_init__``** 的后端实例。

    ``__post_init__`` 会联网问上下文窗口，核心里不能有网络。``__new__`` 绕开
    ``__init__``（因此也绕开 ``__post_init__``），字段自己填。
    """
    backend = OpenAICompatChatBackend.__new__(OpenAICompatChatBackend)
    backend.model = DEFAULT_OPENAI_MODEL
    backend.base_url = "https://api.deepseek.com/v1"
    backend.api_key = "sk-test"
    backend.context_window = 65536
    for key, value in fields.items():
        setattr(backend, key, value)
    return backend


def _payload(request: ChatRequest, **fields: Any) -> dict[str, Any]:
    return _bare(**fields)._build_payload(request)


# ── 顶层字段 ────────────────────────────────────────────────────────────────


def test_the_basic_shape() -> None:
    payload = _payload(ChatRequest(messages=[ChatMessage(role="user", content="在吗")]))

    assert payload["model"] == DEFAULT_OPENAI_MODEL
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "在吗"}]
    assert isinstance(payload["temperature"], float)


def test_usage_is_asked_for_explicitly() -> None:
    """**不带 ``stream_options`` 的话云端根本不给 ``usage``。**

    它不报的表现是"上下文指示器一直空着"，而不是任何一条错误——所以这条断言守的是
    一个不会自己喊疼的东西。
    """
    payload = _payload(ChatRequest(messages=[ChatMessage(role="user", content="x")]))

    assert payload["stream_options"] == {"include_usage": True}


def test_optional_fields_appear_only_when_set() -> None:
    """没设的字段不出现。带上 ``"stop": []`` 这类空值，有些端点会当成语法错误。"""
    bare = _payload(ChatRequest(messages=[ChatMessage(role="user", content="x")]))
    assert "max_tokens" not in bare
    assert "stop" not in bare
    assert "tools" not in bare

    full = _payload(
        ChatRequest(messages=[ChatMessage(role="user", content="x")], max_tokens=64, stop=("停",))
    )
    assert full["max_tokens"] == 64
    assert full["stop"] == ["停"]


def test_extra_goes_to_the_top_level() -> None:
    """与 Ollama 的差别：那边的 ``extra`` 进 ``options``，这里 OpenAI 的请求体是平的。"""
    payload = _payload(
        ChatRequest(messages=[ChatMessage(role="user", content="x")], extra={"top_p": 0.5})
    )

    assert payload["top_p"] == 0.5
    assert "options" not in payload


def test_extra_may_not_silently_override_an_explicit_field() -> None:
    """静默覆盖会让人以为设的是 ``ChatRequest`` 上的字段，而实际生效的是 ``extra``。"""
    with pytest.raises(ChatBackendError) as caught:
        _payload(
            ChatRequest(messages=[ChatMessage(role="user", content="x")], extra={"model": "别的"})
        )

    assert "model" in str(caught.value)


# ── 工具声明（入参方向） ────────────────────────────────────────────────────


def test_a_tool_spec_gets_the_openai_shell() -> None:
    spec = ToolSpec(name="now", description="现在几点")
    built = _build_tool(spec)

    assert built["type"] == "function"
    assert built["function"]["name"] == "now"
    assert built["function"]["parameters"] == {"type": "object", "properties": {}}


def test_tools_are_included_when_offered() -> None:
    payload = _payload(
        ChatRequest(
            messages=[ChatMessage(role="user", content="x")],
            tools=(ToolSpec(name="now", description="现在几点"),),
        )
    )

    assert payload["tools"][0]["function"]["name"] == "now"


# ── 回灌方向的两处不对称 ────────────────────────────────────────────────────


def test_arguments_go_back_as_a_json_string() -> None:
    """**这一条是回灌方向最容易写错的地方。**

    Ollama 收对象，OpenAI 收 JSON 字符串。发错方向不会有清晰的报错——多数端点会回
    一个 400，或者干脆**忽略整条调用**，而"忽略了整条调用"的表现是模型收到一段它
    无法解释的对话，然后开始胡说。
    """
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="call_a", name="now", arguments={"x": 1}),),
    )

    call = _build_message(message)["tool_calls"][0]

    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"x": 1}


def test_the_tool_call_shell_is_added() -> None:
    """Ollama 返回的 ``tool_calls`` 没有 ``type`` 外壳，回灌给它也不需要；**这里需要**。"""
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="call_a", name="now", arguments={}),),
    )

    assert _build_message(message)["tool_calls"][0]["type"] == "function"


def test_a_tool_result_carries_its_call_id() -> None:
    """``tool_call_id`` 丢了，模型无法把结果与它发出的那次调用配对。"""
    built = _build_message(ChatMessage(role="tool", content="9:00", tool_call_id="call_a"))

    assert built["tool_call_id"] == "call_a"
    assert built["role"] == "tool"


def test_non_ascii_arguments_are_not_escaped_into_puzzles() -> None:
    """``ensure_ascii=False``。转义本身是合法的，但它在日志与错误信息里读起来像乱码，
    而排查工具调用问题时人正是要读那一串。"""
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="c", name="set_reminder", arguments={"text": "买牛奶"}),),
    )

    assert "买牛奶" in _build_message(message)["tool_calls"][0]["function"]["arguments"]


# ── 身份 ────────────────────────────────────────────────────────────────────


def test_model_id_names_the_backend_type_not_the_brand() -> None:
    """进注入账本的就是它。

    前缀写 ``deepseek/`` 的话，同一段代码接 OpenRouter 或自建 vLLM 时那个前缀就在
    撒谎——而账本里的这一栏是"这段记忆是哪个模型读的"的唯一记录。
    """
    assert _bare(model="deepseek-chat").model_id == "openai-compat/deepseek-chat"
    assert _bare(model="qwen2.5-7b").model_id == "openai-compat/qwen2.5-7b"


def test_count_tokens_is_honestly_unknown() -> None:
    """返回 ``None`` 是契约允许的分支（装配预算必须回退到启发式）。

    编一个"字数除以二"之类的数才是错的：它会让预算看起来精确，而实际偏差随语言、
    随模型变化。
    """
    assert _bare().count_tokens("随便一段话") is None
