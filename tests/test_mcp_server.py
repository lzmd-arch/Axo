"""Axo 只读 MCP Server 测试（内存库，不碰真数据）。"""

from __future__ import annotations

import io
import json

from axo.mcp_server import TOOLS, MemoryCore, handle_message, serve_stdio
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier


def _core() -> MemoryCore:
    store = MemoryStore(":memory:")
    for index, text in enumerate(("spiking working memory", "episodic memory recall")):
        store.add(
            MemoryItem(
                memory_id=f"m{index}",
                kind=MemoryKind.FACT,
                tier=Tier.WORKING,
                content=text,
                lex_sdr=encode_lexical(text),
            )
        )
    return MemoryCore(store)


def test_tools_are_read_only_names():
    assert [tool["name"] for tool in TOOLS] == ["state.query", "memory.search", "trigger.status"]


def test_state_query_counts_memories():
    reply = handle_message(
        _core(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "state.query", "arguments": {}},
        },
    )
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert payload["memory_entries"] == 2
    assert payload["read_only"] is True


def test_search_returns_hits_without_writing():
    core = _core()
    before = core.store.count()
    reply = handle_message(
        core,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory.search",
                "arguments": {"query": "episodic memory", "top_k": 2},
            },
        },
    )
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert payload["count"] == 2
    assert payload["hits"][0]["id"] == "m1"
    assert core.store.count() == before


def test_trigger_status_is_not_armed():
    reply = handle_message(
        _core(),
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "trigger.status", "arguments": {}},
        },
    )
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert payload["armed"] is None


def test_bad_arguments_are_tool_errors():
    reply = handle_message(
        _core(),
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "memory.search", "arguments": {"query": ""}},
        },
    )
    assert reply["result"]["isError"] is True


def test_stdio_skips_notifications():
    source = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
    )
    sink = io.StringIO()
    serve_stdio(_core(), stdin=source, stdout=sink)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(replies) == 1
    assert replies[0]["id"] == 1
