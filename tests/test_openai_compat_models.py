"""模型清单解析、挑模型、以及地址护栏。

**形状与 Ollama 那份逐键对齐**是本文件的主题。消费它的地方（下拉框、主窗口的装配、
挑模型）只认那一套键，而"少一个键"的失败方式不是报错——是某一处静默地取到 ``None``，
表现为"列表里那个模型选了没反应"。

``capabilities`` 那一条单独值得一个断言：云端条目要**显式**打上 ``completion``，
不能靠 ``is_chat_capable`` 里"空 capabilities 当可用"那个巧合。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from axo.llm.base import ChatBackendError
from axo.llm.openai_compat import (
    DEFAULT_OPENAI_MODEL,
    FALLBACK_CLOUD_MODELS,
    _known_context_window,
    _parse_models,
    endpoint_problem,
    fallback_models,
    pick_cloud_chat_model,
    supports_audio_input,
)

#: ``list_local_models`` 返回的条目里有的键。**云端条目必须全都有**——少了哪个，
#: 读它的那一处就会静默取到 ``None``。
_OLLAMA_KEYS = frozenset(
    {
        "name",
        "short_name",
        "size",
        "parameter_size",
        "quantization",
        "family",
        "modified_at",
        "capabilities",
    }
)


def _body(*entries: dict[str, Any]) -> str:
    return json.dumps({"object": "list", "data": list(entries)}, ensure_ascii=False)


# ── 解析 ────────────────────────────────────────────────────────────────────


def test_parses_the_openai_list_shape() -> None:
    models = _parse_models(
        _body({"id": "deepseek-reasoner"}, {"id": "deepseek-chat"}, {"id": "deepseek-coder"})
    )

    assert [entry["name"] for entry in models] == [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ], "按名字排序——下拉框的次序要稳定"
    assert all(entry["short_name"] == entry["name"] for entry in models)


def test_every_entry_carries_the_keys_the_ui_reads() -> None:
    """**这一条是"逐键对齐"的守卫。** 少一个键不会报错，只会让读它的地方取到
    ``None``，而症状是"那个模型选了没反应"。"""
    entry = _parse_models(_body({"id": "deepseek-chat"}))[0]

    assert set(entry) >= _OLLAMA_KEYS


def test_capabilities_are_stated_explicitly() -> None:
    """云端模型要**自己说**它能对话。

    不能靠 ``is_chat_capable`` 里"空元组当可用"那个巧合：那条语义一收紧，云端模型
    会**整批**从下拉框消失，而症状是"明明配好了，列表却是空的"。
    """
    entry = _parse_models(_body({"id": "deepseek-chat"}))[0]

    assert entry["capabilities"] == ("completion",)


def test_the_context_window_is_taken_when_the_endpoint_reports_it() -> None:
    """DeepSeek 的 ``/models`` 会回这个字段，所以它是首选，猜测只是兜底。"""
    entry = _parse_models(_body({"id": "deepseek-chat", "context_window": 131072}))[0]

    assert entry["context_window"] == 131072


def test_a_missing_context_window_stays_none() -> None:
    """``None`` 不是 0：读它的地方要能区分"端点没报"与"报了个 0"。"""
    assert _parse_models(_body({"id": "x"}))[0]["context_window"] is None


def test_entries_without_an_id_are_dropped() -> None:
    models = _parse_models(_body({"id": ""}, {"object": "model"}, {"id": "keep"}))

    assert [entry["name"] for entry in models] == ["keep"]


def test_a_top_level_array_is_accepted_too() -> None:
    """有的兼容端点直接回一个数组。为它写一条分支比在用户那里变成"列表是空的"便宜。"""
    models = _parse_models(json.dumps([{"id": "a"}]))

    assert [entry["name"] for entry in models] == ["a"]


def test_a_broken_body_raises_instead_of_returning_nothing() -> None:
    """**空列表与"解析不了"必须分开。**

    都返回空列表的话，界面上的表现一模一样（下拉框空了），而用户会去查地址、
    查 key，唯独不会想到是响应的形状不对。
    """
    with pytest.raises(ChatBackendError):
        _parse_models("<html>Bad Gateway</html>")
    with pytest.raises(ChatBackendError):
        _parse_models('{"object": "list"}')


# ── 挑模型 ──────────────────────────────────────────────────────────────────


def test_the_default_model_is_preferred() -> None:
    """默认值与"从清单里挑哪个"是同一个决定的两半：分成两个名字就会出现
    "默认按 A 建后端，却把 B 记进了账本"。"""
    models = _parse_models(_body({"id": "deepseek-coder"}, {"id": DEFAULT_OPENAI_MODEL}))

    assert pick_cloud_chat_model(models) == DEFAULT_OPENAI_MODEL


def test_the_first_is_used_when_the_default_is_absent() -> None:
    models = _parse_models(_body({"id": "qwen2.5-7b"}, {"id": "zzz"}))

    assert pick_cloud_chat_model(models) == "qwen2.5-7b"


def test_nothing_to_pick_gives_an_empty_string() -> None:
    assert pick_cloud_chat_model([]) == ""


# ── 兜底预设 ────────────────────────────────────────────────────────────────


def test_the_fallback_entries_have_the_same_shape_as_a_real_list() -> None:
    """兜底预设是拿去**直接显示**的，所以它不能长得跟真清单不一样。"""
    entries = fallback_models()

    assert [entry["name"] for entry in entries] == list(FALLBACK_CLOUD_MODELS)
    assert FALLBACK_CLOUD_MODELS, "兜底不能是空的——那等于没有兜底"
    for entry in entries:
        assert set(entry) >= _OLLAMA_KEYS
        assert entry["capabilities"] == ("completion",)


def test_the_fallback_is_a_fresh_list_each_time() -> None:
    """改了返回值不该改到那份常量——它是别的调用点也会拿到的东西。"""
    first = fallback_models()
    first[0]["name"] = "被改过了"

    assert fallback_models()[0]["name"] == FALLBACK_CLOUD_MODELS[0]


def test_the_fallback_carries_a_window_so_nobody_probes_again() -> None:
    """兜底预设**必须自带** ``context_window``。

    这一份只出现在"问端点那一趟已经失败"之后，而主窗口正是在那之后、在**界面
    线程上**构造后端的：窗口是 ``None`` 的话，构造函数会照着
    :func:`query_context_window` 再发一次 ``GET /models``——那次请求必然以同一个
    原因失败，代价是先把 ``connect`` 那 10 秒卡满。而它拿回来的还是这个数。
    """
    for entry in fallback_models():
        assert isinstance(entry["context_window"], int), entry["name"]
    # 与服务端**没报**时留下的 ``None`` 是两回事（见上面那条）。
    assert _parse_models(_body({"id": "x"}))[0]["context_window"] is None


# ── 地址护栏 ────────────────────────────────────────────────────────────────


def test_https_is_always_fine() -> None:
    assert endpoint_problem("https://api.deepseek.com/v1") == ""
    assert endpoint_problem("https://openrouter.ai/api/v1") == ""


def test_plain_http_is_fine_for_a_local_endpoint() -> None:
    """自建推理机（vLLM、LM Studio）走明文 http 是正常的——那台机器就是本机。"""
    assert endpoint_problem("http://127.0.0.1:8000/v1") == ""
    assert endpoint_problem("http://localhost:1234/v1") == ""


def test_plain_http_to_someone_elses_machine_is_refused() -> None:
    """**这一条是护栏的全部意义。** key 跟着每个请求发出去，明文 http 意味着它在
    路上可以被任何人读到。"""
    problem = endpoint_problem("http://api.deepseek.com/v1")

    assert problem, "远程的明文 http 必须被拦下"
    assert "https" in problem and "API key" in problem


def test_an_incomplete_address_is_refused_with_an_example() -> None:
    """用户最常见的写法是漏掉协议头。错误要说清"该长什么样"。"""
    problem = endpoint_problem("api.deepseek.com")

    assert problem
    assert "https://api.deepseek.com/v1" in problem


def test_a_file_url_is_refused() -> None:
    """别让 ``file://`` 之类的东西从"不是 http"这条缝里溜过去。"""
    assert endpoint_problem("file:///etc/passwd")
    assert endpoint_problem("ftp://example.com/v1")


# ── 2026-10 的模型事实 ──────────────────────────────────────────────────────


def test_the_default_model_is_the_current_one() -> None:
    """默认值跟着官方当前模型走；旧名字会被转发，但账本里记的应该是真名。"""
    assert DEFAULT_OPENAI_MODEL == "deepseek-flash"
    assert DEFAULT_OPENAI_MODEL in FALLBACK_CLOUD_MODELS


def test_known_models_get_their_documented_context_window() -> None:
    """1M 是官方定价页写着的数；**问端点仍优先**，这张表只管问不到的时候。"""
    assert _known_context_window("deepseek-flash") == 1_000_000
    assert _known_context_window("deepseek-v4-pro") == 1_000_000
    assert _known_context_window("未知模型") == 131072


def test_fallback_windows_follow_the_known_table() -> None:
    for entry in fallback_models():
        if entry["name"].startswith("deepseek-"):
            assert entry["context_window"] == 1_000_000, entry["name"]


def test_deepseek_models_do_not_accept_audio() -> None:
    """2026-10 已核实：两个当前模型都不收 ``input_audio``。

    这条提前门控的收益是**不发那段必定被拒的 1 MB 上传**；对未知端点仍然按
    "可能收"处理，让退化重发去回答。
    """
    assert supports_audio_input("deepseek-flash") is False
    assert supports_audio_input("deepseek-v4-pro") is False
    assert supports_audio_input("deepseek-chat") is False
    assert supports_audio_input("gpt-4o-audio-preview") is True
    assert supports_audio_input("qwen2-audio") is True
