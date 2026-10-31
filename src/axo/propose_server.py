"""写档位的 MCP 入口（``axo-mcp-write``）：外部客户端只能提议，门控决定要不要落库。

与 ``axo-mcp`` 的关系不是"把只读放开"，而是**多了一个消费者**：

* 三个只读工具照旧（读的是只读连接，行为与 ``axo-mcp`` 完全一致）；
* 新增 ``propose.memory.write``：只往队列里放一条，**不直接改库**；
* 新增 ``queue.status`` / ``queue.list`` / ``memory.revert``：看判定、看理由、撤回一条。

判定在 ``propose`` 同一次调用里跑完（``decide_pending``）：客户端提完就能拿到
accept / pending / reject 的结论，不会出现"提了但永远没人处置"的队列。

**默认不存在。** ``memory/propose_enabled`` 关着时这个入口直接退出，并告诉你去哪里打开——
"关着"的形状是"没有这个服务器"，不是"有这个服务器但什么也不写"。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from memcore.store import MemoryStore

from . import paths
from .mcp_server import TOOLS as READ_TOOLS
from .mcp_server import MemoryCore, open_store, serve_stdio
from .propose_consumer import ProposeConsumer

SETTING_KEY_HINT = "设置 → 记忆 → 允许外部客户端提议写入记忆（MCP）"

_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS_WRITE: list[dict[str, Any]] = list(READ_TOOLS) + [
    {
        "name": "propose.memory.write",
        "description": (
            "Propose a memory entry. It only enters the queue; the gate decides whether it is "
            "kept. Requires a client-generated proposal_id (idempotency key)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "proposal_id": {"type": "string", "minLength": 1},
                "source": {"type": "string", "enum": ["user", "client", "llm"]},
            },
            "required": ["text", "proposal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "queue.status",
        "description": "Queue depth, quota usage and the gate configuration (read-only).",
        "inputSchema": _EMPTY,
    },
    {
        "name": "queue.list",
        "description": "List proposals with their verdicts and reasons (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "accepted", "rejected", "reverted"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "memory.revert",
        "description": "Undo one accepted proposal (deletes the memory it created). Requires a reason.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seq": {"type": "integer", "minimum": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["seq", "reason"],
            "additionalProperties": False,
        },
    },
]


class WriteCore:
    """三个只读工具转发给只读核；其余四个转发给消费者。"""

    def __init__(self, db: Path, consumer: ProposeConsumer) -> None:
        # 收**库路径**而不是现成的连接：检索用的只读连接在启动时打开，而落库走另一个
        # 可写连接——**只读连接看不见之后提交的写入**（WAL 下的快照），症状是
        # "提议被接受、库里也有一条，但紧接着 memory.search 返回空"。所以每次落库之后
        # 换一条新的只读连接。
        self._db = db
        self._reader = MemoryCore(open_store(db))
        self._consumer = consumer

    def _refresh_reader(self) -> None:
        old = self._reader
        self._reader = MemoryCore(open_store(self._db))
        with contextlib.suppress(Exception):  # 关不掉不影响新连接
            old.store.close()

    # ---- 只读（与 axo-mcp 同一份实现）
    def state(self) -> dict[str, Any]:
        payload = self._reader.state()
        payload.update({"read_only": False, "write_tier": "propose", **self._consumer.status()})
        return payload

    def search(self, query: str, top_k: int = 5, budget_tokens: int = 8000) -> list[dict[str, Any]]:
        return self._reader.search(query, top_k, budget_tokens)

    def trigger_status(self) -> dict[str, Any]:
        return self._reader.trigger_status()

    # ---- 写档位
    def propose(self, text: Any, *, proposal_id: Any, source: str = "client") -> dict[str, Any]:
        row = self._consumer.ingest(text, proposal_id=proposal_id, source=source)
        if row["status"] == "pending":
            self._consumer.decide(row["seq"])
        decided = self._consumer.queue.get(row["seq"])
        if decided["status"] == "accepted":
            self._refresh_reader()
        return {
            "seq": decided["seq"],
            "proposal_id": decided["proposal_id"],
            "status": decided["status"],
            "reason": decided["reason"],
            "salience": decided["salience"],
            "replayed": row.get("replayed", False),
            "entries": self._consumer.store.count(),
        }

    def queue_status(self) -> dict[str, Any]:
        return self._consumer.status()

    def queue_list(self, *, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        if status is not None and status not in ("pending", "accepted", "rejected", "reverted"):
            raise ValueError("status 必须是 pending / accepted / rejected / reverted。")
        if not 1 <= int(limit) <= 100:
            raise ValueError("limit 必须是 1..100 的整数。")
        rows = self._consumer.queue.list(status=status, limit=int(limit))
        return {
            "proposals": [
                {
                    "seq": item["seq"],
                    "proposal_id": item["proposal_id"],
                    "status": item["status"],
                    "reason": item["reason"],
                    "salience": item["salience"],
                    "created_at": item["created_at"],
                    "text": item["payload"].get("text"),
                }
                for item in rows
            ],
            "count": len(rows),
        }

    def revert(self, seq: Any, reason: Any) -> dict[str, Any]:
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError("seq 必须是整数。")
        payload = self._consumer.revert(seq, reason)
        self._refresh_reader()
        return payload


#: 隔离测试用的覆盖开关。**只在同时显式给了数据目录时才被采信**——不然它就能
#: 悄悄把默认库的写档位打开，而那正是"默认只读"要防的事。
ENV_OVERRIDE = "AXO_PROPOSE_SETTING"


def setting_enabled(*, isolated: bool = False) -> bool:
    """读真实设置。Qt 只在调用时导入——测试与纯逻辑不需要它。

    ``isolated`` 由调用方在**显式指定了数据目录**时为真；只有那种情况下
    :data:`ENV_OVERRIDE` 才会被采纳。生产路径永远读设置。
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override is not None and isolated:
        return override.strip() == "1"

    from PySide6.QtCore import QSettings

    from .ui.settings_store import SETTINGS_APP, SETTINGS_ORG, SettingsStore
    from .ui.theme_qt import QSettingsBackend

    store = SettingsStore(QSettingsBackend(QSettings(SETTINGS_ORG, SETTINGS_APP)))
    return bool(store.load_memory().propose_enabled)


def main_write(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="axo-mcp-write",
        description="Axo MCP server, write tier (proposals -> gate -> memory store)",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--queue", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.data_dir is not None:
        paths.set_data_root(args.data_dir)
    if not setting_enabled(isolated=args.data_dir is not None or args.db is not None):
        print(
            f"写档位没有打开。在 Axo 里打开：{SETTING_KEY_HINT}，再重启这个服务器。",
            file=sys.stderr,
        )
        return 2

    db = args.db or paths.memory_db_path()
    queue = args.queue or paths.propose_queue_path()
    audit = args.audit or paths.propose_audit_path()
    consumer = ProposeConsumer(MemoryStore(db), queue, audit_path=audit)
    serve_stdio(WriteCore(db, consumer), tools=TOOLS_WRITE)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_write())
