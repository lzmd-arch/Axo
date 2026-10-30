"""Axo 只读 MCP Server 测试（内存库，不碰真数据）。"""

from __future__ import annotations

import io
import json
from pathlib import Path

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


# ── 记忆库的写保护（2026-10-30 用户要求：Axo 侧只允许读）─────────────────────


_WRITE_VERBS = (
    "write",
    "delete",
    "update",
    "set",
    "add",
    "remove",
    "arm",
    "feed",
    "settle",
    "forget",
    "merge",
)


def test_no_tool_name_carries_a_write_verb():
    """工具名是契约：新增任何带写语义的动词都要先过这一关（并改这条测试）。"""
    for tool in TOOLS:
        lowered = tool["name"].lower()
        assert not any(verb in lowered for verb in _WRITE_VERBS), tool["name"]


def test_read_only_tools_still_cover_the_three_reads():
    assert {tool["name"] for tool in TOOLS} == {"state.query", "memory.search", "trigger.status"}


def test_open_store_is_always_read_only(tmp_path):
    from axo.mcp_server import open_store

    db = tmp_path / "mem.db"
    MemoryStore(db).close()  # 先建一个合法的库
    store = open_store(db)
    try:
        assert store.read_only is True
    finally:
        store.close()


def test_open_store_refuses_to_create_a_missing_database(tmp_path):
    import pytest

    from axo.mcp_server import open_store

    with pytest.raises(SystemExit):
        open_store(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_a_full_session_leaves_the_database_byte_identical(tmp_path):
    """跑一遍完整的只读会话，库目录逐字节不变——比"我们没有写"更强的证据。"""
    import hashlib

    from axo.mcp_server import open_store

    db = tmp_path / "mem.db"
    store = MemoryStore(db)
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
    store.close()

    def _digest() -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(tmp_path.iterdir())
        }

    before = _digest()
    core = MemoryCore(open_store(db))
    try:
        for index, (name, arguments) in enumerate(
            (
                ("state.query", {}),
                ("memory.search", {"query": "memory", "top_k": 5}),
                ("trigger.status", {}),
                ("memory.search", {"query": "spiking", "top_k": 1}),
            )
        ):
            reply = handle_message(
                core,
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            assert reply["result"]["isError"] is False
    finally:
        core.store.close()

    after = _digest()
    # 1) 库文件本身逐字节不变——这是"记忆库没被改"的硬证据；
    for name, digest in before.items():
        assert after[name] == digest, f"{name} 被改动了"
    # 2) 新增的只允许是 SQLite 的运行时索引：`-wal` 必须是空文件、`-shm` 是共享内存索引。
    #    库是 WAL 模式，只读连接需要一个共享内存索引才能正确看到未检查点的提交；
    #    不用 `immutable=1` 就是因为它会**忽略** WAL，在应用同时在写时读出旧数据。
    for name in set(after) - set(before):
        assert name.endswith(("-wal", "-shm")), f"意外的新文件：{name}"
        if name.endswith("-wal"):
            assert Path(tmp_path, name).read_bytes() == b"", f"{name} 里出现了数据"


def test_memory_core_refuses_a_writable_file_store(tmp_path):
    """第二道闸：装配写错（传了可写的落盘库）时立刻炸，而不是悄悄允许写。"""
    import pytest

    db = tmp_path / "mem.db"
    writable = MemoryStore(db)
    try:
        with pytest.raises(ValueError, match="只读"):
            MemoryCore(writable)
    finally:
        writable.close()
