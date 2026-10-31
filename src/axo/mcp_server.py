"""Axo 只读 MCP Server：把本地记忆库以三个只读工具暴露给 MCP 客户端。

红线（2026-10-30 用户明确要求）：**记忆库只允许读，不允许任何更改**。

* 库以 SQLite 的 ``mode=ro`` 打开（:func:`open_store`）——写入在**引擎层**被拒绝，
  不依赖调用方自觉；
* `memory.search` 走 `memcore.recall(..., settle=False)`，不落权重、不写访问计数；
* 不调 LLM、不读 key；传输用 stdio 的 JSON-RPC 2.0，无第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from memcore.recall import RecallConfig, recall
from memcore.store import MemoryStore

from . import paths

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "axo-memory", "version": "0.1.0"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "state.query",
        "description": "Read Axo memory counts, spaces and data-root source (read-only).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory.search",
        "description": "Search Axo local memories; never settles weights or writes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "budget_tokens": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger.status",
        "description": "Read whether autonomous triggering is exposed (not in this server).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class MemoryCore:
    def __init__(self, store: MemoryStore) -> None:
        # 第二道闸：即使以后有人改了装配，只要传进来的是**可写的落盘库**就直接炸，
        # 而不是悄悄允许写。内存库（测试用）例外——它碰不到真实记忆库。
        if not store.read_only and store.path != ":memory:":
            raise ValueError(
                "MCP 适配器只接受只读打开的落盘库（用 open_store / "
                "MemoryStore(..., read_only=True)）；用户要求记忆库只读。"
            )
        self.store = store

    def state(self) -> dict[str, Any]:
        return {
            "memory_entries": self.store.count(),
            "spaces": [space.space_id for space in self.store.spaces()],
            "data_root_source": paths.data_root_source(),
            "read_only": True,
        }

    def search(self, query: str, top_k: int = 5, budget_tokens: int = 8000) -> list[dict[str, Any]]:
        ledger = recall(
            self.store,
            query_text=query,
            budget_tokens=budget_tokens,
            config=RecallConfig(max_items=top_k),
            settle=False,
        )
        return [
            {
                "id": memory.memory_id,
                "text": memory.content,
                "score": float(memory.scores.fused),
                "tier": memory.tier,
                "pinned": bool(memory.pinned),
                "created_at": float(memory.created_at),
            }
            for memory in ledger.injected
        ]

    def trigger_status(self) -> dict[str, Any]:
        return {
            "armed": None,
            "reason": "read-only memory MCP; proactive control is not exposed",
        }


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _text(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": is_error,
    }


def _call(core: MemoryCore, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "state.query":
        if arguments:
            raise ValueError("state.query 不接受参数。")
        return _text(core.state())
    if name == "trigger.status":
        if arguments:
            raise ValueError("trigger.status 不接受参数。")
        return _text(core.trigger_status())
    if name == "memory.search":
        unknown = set(arguments) - {"query", "top_k", "budget_tokens"}
        if unknown:
            raise ValueError(f"未知参数：{sorted(unknown)}")
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        budget = arguments.get("budget_tokens", 8000)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串。")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 50:
            raise ValueError("top_k 必须是 1..50 的整数。")
        if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 100000:
            raise ValueError("budget_tokens 必须是 1..100000 的整数。")
        hits = core.search(query, top_k, budget)
        return _text({"hits": hits, "count": len(hits)})
    if name in ("propose.memory.write", "queue.status", "queue.list", "memory.revert"):
        if not hasattr(core, "queue_status"):
            raise KeyError(name)
        if name == "propose.memory.write":
            unknown = set(arguments) - {"text", "proposal_id", "source"}
            if unknown:
                raise ValueError(f"未知参数：{sorted(unknown)}")
            return _text(
                core.propose(
                    arguments.get("text"),
                    proposal_id=arguments.get("proposal_id"),
                    source=str(arguments.get("source", "client")),
                )
            )
        if name == "queue.status":
            if arguments:
                raise ValueError("queue.status 不接受参数。")
            return _text(core.queue_status())
        if name == "queue.list":
            unknown = set(arguments) - {"status", "limit"}
            if unknown:
                raise ValueError(f"未知参数：{sorted(unknown)}")
            return _text(
                core.queue_list(
                    status=arguments.get("status"), limit=int(arguments.get("limit", 20))
                )
            )
        unknown = set(arguments) - {"seq", "reason"}
        if unknown:
            raise ValueError(f"未知参数：{sorted(unknown)}")
        return _text(core.revert(arguments.get("seq"), arguments.get("reason")))
    raise KeyError(name)


def handle_message(
    core: MemoryCore,
    message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if message.get("jsonrpc") != "2.0":
        return _error(message.get("id"), -32600, "jsonrpc 必须是 '2.0'。")
    method = message.get("method")
    message_id = message.get("id")
    params = message.get("params") or {}
    if not isinstance(method, str):
        return _error(message_id, -32600, "缺少 method。")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _result(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return _result(message_id, {})
    if method == "tools/list":
        return _result(message_id, {"tools": tools if tools is not None else TOOLS})
    if method == "tools/call":
        if not isinstance(params, dict):
            return _error(message_id, -32602, "params 必须是对象。")
        name, arguments = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return _error(message_id, -32602, "tools/call 需要 name 与 arguments。")
        try:
            return _result(message_id, _call(core, name, arguments))
        except KeyError:
            return _error(message_id, -32602, f"未知工具：{name}")
        except ValueError as error:
            return _result(message_id, _text({"error": str(error)}, is_error=True))
        except Exception as error:  # pragma: no cover - 防御未知崩溃
            return _error(message_id, -32603, f"工具内部错误：{error}")
    return _error(message_id, -32601, f"未知方法：{method}")


def serve_stdio(
    core: MemoryCore,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    source, sink = stdin or sys.stdin, stdout or sys.stdout
    for line in source:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("消息必须是 JSON 对象。")
        except (json.JSONDecodeError, ValueError) as error:
            sink.write(json.dumps(_error(None, -32700, f"解析失败：{error}")) + "\n")
            sink.flush()
            continue
        response = handle_message(core, message, tools=tools)
        if response is not None:
            sink.write(json.dumps(response, ensure_ascii=False) + "\n")
            sink.flush()


def open_store(db: Path) -> MemoryStore:
    """**只读**打开记忆库。写操作会被 SQLite 直接拒绝（``mode=ro``）。

    MCP 适配器是"只允许看"的调用方：用户 2026-10-30 明确要求不得对记忆库做任何更改，
    所以这里不开放任何可写路径，也不提供开关。
    """
    try:
        return MemoryStore(db, read_only=True)
    except FileNotFoundError as error:
        raise SystemExit(f"{error}（只读模式不会新建记忆库）") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Axo read-only memory MCP server")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.data_dir is not None:
        paths.set_data_root(args.data_dir)
    db = args.db or paths.memory_db_path()
    serve_stdio(MemoryCore(open_store(db)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
