"""NDJSON 流解析。

这是 P0 里唯一有真实失败模式的逻辑：网络给的字节流**不保证按行对齐**，一个
JSON 对象被切成两片是常态而不是异常。写错了的表现是"偶尔丢一段字"或"回复卡在
一半"——都是很难复现、很难归因的症状。所以它值得一个不依赖网络的单元测试。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from biosnn_memterm.llm.base import ChatBackendError
from biosnn_memterm.llm.ollama import (
    _iter_ndjson_chunks,
    _l2_normalize,
    _parse_line,
    _parse_num_ctx,
    check_host_is_local,
    short_model_name,
)


class _FakeResponse:
    """只实现 ``iter_bytes`` 的假响应。

    :func:`_iter_ndjson_chunks` 只用这一个方法，所以不必去 mock httpx——
    测试越小越不容易因为库升级而失效。
    """

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = pieces

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._pieces


def _event(text: str, done: bool = False, **extra: object) -> bytes:
    payload = {"message": {"role": "assistant", "content": text}, "done": done}
    payload.update(extra)
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


# ── 正常路径 ────────────────────────────────────────────────────────────────


def test_parses_aligned_lines() -> None:
    pieces = [_event("你好"), _event("世界", done=True, prompt_eval_count=12, eval_count=7)]
    chunks = list(_iter_ndjson_chunks(_FakeResponse(pieces)))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["你好", "世界"]
    assert chunks[-1].done is True
    assert chunks[-1].prompt_tokens == 12
    assert chunks[-1].completion_tokens == 7


def test_reassembles_split_json() -> None:
    """一个 JSON 对象被切在中间——这是真实网络下的常态。"""
    whole = _event("被切开的一句话")
    pieces = [whole[:12], whole[12:30], whole[30:]]
    chunks = list(_iter_ndjson_chunks(_FakeResponse(pieces)))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["被切开的一句话"]


def test_handles_multiple_objects_in_one_piece() -> None:
    """多个 JSON 对象挤在一次 read 里。"""
    pieces = [_event("一") + _event("二") + _event("三", done=True)]
    chunks = list(_iter_ndjson_chunks(_FakeResponse(pieces)))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["一", "二", "三"]


def test_handles_stream_without_trailing_newline() -> None:
    """流不以换行结尾（HTTP 分块边界不保证对齐行边界）。"""
    raw = _event("结尾没有换行", done=True).rstrip(b"\n")
    chunks = list(_iter_ndjson_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["结尾没有换行"]
    assert chunks[-1].done is True


def test_skips_blank_lines() -> None:
    pieces = [_event("甲"), b"\n\n", _event("乙", done=True)]
    chunks = list(_iter_ndjson_chunks(_FakeResponse(pieces)))  # type: ignore[arg-type]

    assert [chunk.text for chunk in chunks] == ["甲", "乙"]


def test_done_chunk_with_empty_content_is_kept() -> None:
    """结尾片常常没有文本，只有统计信息——它不能被当成空行丢掉。"""
    raw = (
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 3})
        + "\n"
    ).encode()
    chunks = list(_iter_ndjson_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]

    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].done is True
    assert chunks[0].completion_tokens == 3


# ── 错误路径 ────────────────────────────────────────────────────────────────


def test_bad_json_line_is_skipped_not_fatal() -> None:
    """半行坏 JSON 不该打断整段回复。

    本地模型的统计行偶尔会吐出坏 JSON。为此中止整段对话，对用户来说是
    "回复突然断了，不知道为什么"——比丢一行更糟。
    """
    assert _parse_line(b"{ this is not json") is None


def test_stream_error_is_raised() -> None:
    """流中间的 error 字段必须抛出去——后端明确报错了，不能装作没看见。"""
    raw = (json.dumps({"error": "model runner has crashed"}) + "\n").encode()
    with pytest.raises(ChatBackendError, match="model runner has crashed"):
        list(_iter_ndjson_chunks(_FakeResponse([raw])))  # type: ignore[arg-type]


def test_missing_message_field_yields_empty_text() -> None:
    """Ollama 的某些事件没有 message 字段，不该 KeyError。"""
    chunk = _parse_line(b'{"done": false}')
    assert chunk is not None
    assert chunk.text == ""
    assert chunk.done is False


# ── 后端地址护栏 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://localhost",
        "http://[::1]:11434",
        "http://0.0.0.0:11434",
    ],
)
def test_local_hosts_are_recognised(host: str) -> None:
    assert check_host_is_local(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "http://192.168.1.50:11434",
        "http://inference.internal:11434",
        "https://api.example.com",
        "not a url",
        "",
    ],
)
def test_remote_hosts_are_not_local(host: str) -> None:
    assert check_host_is_local(host) is False


# ── 归一化 ──────────────────────────────────────────────────────────────────


def test_l2_normalize_produces_unit_rows() -> None:
    import numpy as np

    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = _l2_normalize(matrix)

    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


def test_l2_normalize_keeps_zero_rows_at_zero() -> None:
    """零向量必须保持是零向量。

    直接除以范数会得到 NaN，而 NaN 一旦进了召回打分就会静默污染整次排序——
    症状是"有时候一条都召不回来"，且不报错。
    """
    import numpy as np

    matrix = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    normalized = _l2_normalize(matrix)

    assert not np.isnan(normalized).any()
    assert np.array_equal(normalized[0], np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("qwen3-4b:latest", "qwen3-4b"),
        ("qwen3-4b", "qwen3-4b"),
        # 非 latest 的标签是实义的，绝不能脱——`4b` 是尺寸，量化标记是精度，
        # 脱掉它们会把两个真正不同的模型变成同一个名字。
        ("qwen3:4b", "qwen3:4b"),
        ("llama3.2:3b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        # 只有末尾那一段才算标签。`:latest` 出现在名字中间不是标签边界。
        ("weird:latest-name", "weird:latest-name"),
        ("", ""),
    ],
)
def test_short_model_name_strips_only_the_default_tag(raw: str, expected: str) -> None:
    assert short_model_name(raw) == expected


def test_short_model_name_never_produces_an_empty_name() -> None:
    """``":latest"`` 这种退化输入不能归一化成空串。

    归一化成一个空名字的后果是下拉框里出现一个无法选中的条目，而错误信息里
    又是一个空字符串——归因成本远高于在这里写一条断言。
    """
    assert short_model_name(":latest") != ""


# ── /api/show 的 num_ctx 解析 ───────────────────────────────────────────────


def test_parse_num_ctx_reads_the_alignment_padded_form() -> None:
    """``/api/show`` 的 ``parameters`` 是**空格对齐**，不是 ``key=value``。

    写成 ``split("=")`` 会一条都取不到，然后静默退回"按名字猜"——预算就此算错，
    而这里没有任何报错。所以这条断言必须钉住真实格式。
    """
    parameters = (
        "top_k                          20\n"
        "top_p                          0.95\n"
        "num_ctx                        16384\n"
        "num_predict                    2048\n"
        "temperature                    0.6\n"
    )
    assert _parse_num_ctx(parameters) == 16384


def test_parse_num_ctx_returns_none_when_absent() -> None:
    """没写 ``num_ctx`` 时返回 ``None``，让调用方去猜。

    不能返回 0 或某个默认值：那会让调用方以为"问到了"，从而不再走猜测分支，
    把 Ollama 的默认窗口（通常远小于上表里那些数）当成真实窗口用。
    """
    assert _parse_num_ctx("temperature 0.8\n") is None
    assert _parse_num_ctx("") is None


def test_parse_num_ctx_does_not_confuse_a_prefix_key() -> None:
    """``num_ctx`` 的匹配必须是整键相等，不能是前缀。

    ``startswith("num_ctx")`` 会把 ``num_ctx_foo`` 也吞掉，症状是读到某个
    陌生参数的值当窗口——比猜错更难查。
    """
    assert _parse_num_ctx("num_ctx_extra 512\nnum_ctx 8192\n") == 8192


def test_parse_num_ctx_tolerates_a_bad_value() -> None:
    """值不是整数时返回 ``None`` 而不是抛异常。

    它跑在后端构造路径上：为一个畸形参数让"选个模型"整个失败，代价不成比例。
    """
    assert _parse_num_ctx("num_ctx                        not-a-number\n") is None
