"""写档位的 MCP 入口（``axo-mcp-write``）：提议进队列、门控落库、可撤销。

只读入口一行没改，由 ``tests/test_mcp_server.py`` 继续守着；这里只验新增的那半边。
"""

from __future__ import annotations

import json

from axo.mcp_server import handle_message
from axo.propose_consumer import ProposeConsumer
from axo.propose_server import TOOLS_WRITE, WriteCore, main_write
from memcore.store import MemoryStore


def _core(tmp_path) -> WriteCore:
    db = tmp_path / "mem.db"
    MemoryStore(db).close()
    consumer = ProposeConsumer(
        MemoryStore(db), tmp_path / "queue.sqlite3", audit_path=tmp_path / "audit.jsonl"
    )
    return WriteCore(db, consumer)


def _call(core, name, arguments=None):
    reply = handle_message(
        core,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
        tools=TOOLS_WRITE,
    )
    return json.loads(reply["result"]["content"][0]["text"]), reply["result"].get("isError", False)


def test_tool_table_adds_four_to_the_three_read_only(tmp_path):
    assert [tool["name"] for tool in TOOLS_WRITE] == [
        "state.query",
        "memory.search",
        "trigger.status",
        "propose.memory.write",
        "queue.status",
        "queue.list",
        "memory.revert",
    ]


def test_propose_lands_through_the_gate(tmp_path):
    core = _core(tmp_path)
    payload, error = _call(
        core,
        "propose.memory.write",
        {"text": "the gate accepted this line at 20 trials", "proposal_id": "p1", "source": "user"},
    )
    assert error is False, payload
    assert payload["status"] == "accepted", payload
    assert payload["entries"] == 1

    hits, _ = _call(core, "memory.search", {"query": "gate"})
    assert hits["count"] >= 1


def test_duplicate_is_rejected_and_nothing_is_written(tmp_path):
    core = _core(tmp_path)
    text = "the gate accepted this line at 20 trials"
    _call(core, "propose.memory.write", {"text": text, "proposal_id": "p1", "source": "user"})
    payload, _ = _call(core, "propose.memory.write", {"text": text, "proposal_id": "p2"})
    assert payload["status"] == "rejected"
    assert "duplicate" in payload["reason"]
    assert payload["entries"] == 1


def test_proposal_without_id_is_refused(tmp_path):
    core = _core(tmp_path)
    payload, error = _call(core, "propose.memory.write", {"text": "no id here"})
    assert error is True
    assert "proposal_id" in payload["error"]


def test_queue_tools_and_revert(tmp_path):
    core = _core(tmp_path)
    propose, _ = _call(
        core, "propose.memory.write", {"text": "a line that will be undone", "proposal_id": "p1"}
    )
    status, _ = _call(core, "queue.status")
    assert status["counts"]["accepted"] == 1

    listing, _ = _call(core, "queue.list", {"status": "accepted"})
    assert listing["count"] == 1

    reverted, error = _call(core, "memory.revert", {"seq": propose["seq"], "reason": "test"})
    assert error is False and reverted["reverted"] is True
    assert reverted["entries"] == 0


def test_entry_point_refuses_when_the_setting_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr("axo.propose_server.setting_enabled", lambda **_: False)
    assert main_write(["--db", str(tmp_path / "mem.db")]) == 2
