"""提议消费者：外部客户端只能提议，门控决定要不要落库（实施计划 P3 / §4.2）。

三条规矩，和 MCP 写版本一一对应：

1. **提议不是写入**：``ingest`` 只往队列里放一条，库一个字节都不动；
2. **门控是唯一写入口**：这个模块里只有 :meth:`ProposeConsumer._apply` 会调
   ``MemoryWriter.record_if_absent``；
3. **全量审计**：提议、判定、落库各一行 JSONL，事件名与 biosnn-mcp 写版本一致
   （``proposal.received`` / ``gate.accepted`` / ``gate.rejected`` / ``gate.pending`` /
   ``memory.applied``），两个副作用可以放在同一个对账脚本下。

**队列格式是协议 v1**：与 ``biosnn-mcp`` 写版本同一张表结构（``proposals``），
字段见 :data:`SCHEMA`。两边各存各的文件，不互相 import——跨仓直接依赖会让"两个版本
必然漂移"变成"一次升级两边一起坏"。

**默认相似度是词法重合（Jaccard）**，不是余弦：Axo 的真实嵌入要连本机模型，而门控
必须能在没有模型的机器上跑完判据。``similarity`` 参数可以把真嵌入换进来；换进来之后
0.90 这个阈值才按"余弦"解释，这一点写进结果 README，不靠读者猜。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from axo.memory.policy import MemoryDraft
from axo.memory.writer import MemoryWriter
from memcore.store import MemoryStore

# 冻结于 biosnn 计划书 §9.1（2026-10-31）。改这里等于改判据，必须 amend。
DUPLICATE_SIMILARITY = 0.90
ACCEPT_SALIENCE = 0.60
FLOOR_SALIENCE = 0.30
QUOTA_HOURLY = 20
QUOTA_DAILY = 200

SOURCE_WEIGHTS = {"user": 0.45, "client": 0.25, "llm": 0.15}
LENGTH_WEIGHT = 0.20
NOVELTY_WEIGHT = 0.25
ENTITY_WEIGHT = 0.10
LENGTH_MIN, LENGTH_MAX = 20, 400

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id  TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    reason       TEXT    NOT NULL DEFAULT '',
    salience     REAL    NOT NULL DEFAULT 0,
    max_cosine   REAL    NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    decided_at   TEXT
);
CREATE TABLE IF NOT EXISTS applied (
    seq         INTEGER PRIMARY KEY,
    memory_id   TEXT    NOT NULL,
    reverted_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lexical_similarity(left: str, right: str) -> float:
    """词法重合：没有嵌入时的替身，只用来让门控可测、可离线跑。"""
    a, b = set(left.lower().split()), set(right.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class GateConfig:
    duplicate_similarity: float = DUPLICATE_SIMILARITY
    accept_salience: float = ACCEPT_SALIENCE
    floor_salience: float = FLOOR_SALIENCE
    quota_hourly: int = QUOTA_HOURLY
    quota_daily: int = QUOTA_DAILY

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def salience(text: str, *, source: str, max_similarity: float) -> float:
    weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["client"])
    length = len(text.strip())
    if LENGTH_MIN <= length <= LENGTH_MAX:
        length_part = LENGTH_WEIGHT
    elif length > LENGTH_MAX:
        length_part = LENGTH_WEIGHT / 2
    else:
        length_part = 0.0
    novelty = NOVELTY_WEIGHT * max(0.0, 1.0 - max_similarity)
    entity = ENTITY_WEIGHT if any(ch.isdigit() for ch in text) else 0.0
    return round(min(1.0, weight + length_part + novelty + entity), 4)


class ProposalQueue:
    """最小的队列：只放提议与判定，不碰记忆库。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def enqueue(self, proposal_id: str, payload: dict[str, Any]) -> tuple[int, bool]:
        row = self._db.execute(
            "SELECT seq FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if row is not None:
            return int(row["seq"]), False
        cursor = self._db.execute(
            "INSERT INTO proposals (proposal_id, kind, payload, status, created_at) "
            "VALUES (?, 'memory.write', ?, 'pending', ?)",
            (proposal_id, json.dumps(payload, ensure_ascii=False), _now()),
        )
        self._db.commit()
        return int(cursor.lastrowid), True

    def get(self, seq: int) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM proposals WHERE seq = ?", (seq,)).fetchone()
        if row is None:
            raise KeyError(seq)
        return {
            "seq": int(row["seq"]),
            "proposal_id": str(row["proposal_id"]),
            "payload": json.loads(row["payload"]),
            "status": str(row["status"]),
            "reason": str(row["reason"]),
            "salience": float(row["salience"]),
            "max_cosine": float(row["max_cosine"]),
            "created_at": str(row["created_at"]),
            "decided_at": row["decided_at"],
        }

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT seq FROM proposals WHERE status = 'pending' ORDER BY seq LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [self.get(int(row["seq"])) for row in rows]

    def decide(
        self,
        seq: int,
        status: str,
        reason: str,
        *,
        salience_score: float = 0.0,
        max_similarity: float = 0.0,
    ) -> None:
        if status not in ("pending", "accepted", "rejected", "reverted"):
            raise ValueError(f"未知状态：{status}")
        self._db.execute(
            "UPDATE proposals SET status = ?, reason = ?, salience = ?, max_cosine = ?, "
            "decided_at = ? WHERE seq = ?",
            (status, reason, salience_score, max_similarity, _now(), seq),
        )
        self._db.commit()

    def list(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT seq FROM proposals"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [self.get(int(row["seq"])) for row in self._db.execute(sql, params)]

    def record_apply(self, seq: int, *, memory_id: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO applied (seq, memory_id) VALUES (?, ?)", (seq, memory_id)
        )
        self._db.commit()

    def applied_memory_id(self, seq: int) -> str | None:
        row = self._db.execute(
            "SELECT memory_id, reverted_at FROM applied WHERE seq = ?", (seq,)
        ).fetchone()
        if row is None or row["reverted_at"] is not None:
            return None
        return str(row["memory_id"])

    def mark_reverted(self, seq: int) -> None:
        stamp = _now()
        self._db.execute("UPDATE applied SET reverted_at = ? WHERE seq = ?", (stamp, seq))
        self._db.execute(
            "UPDATE proposals SET status = 'reverted', decided_at = ? WHERE seq = ?",
            (stamp, seq),
        )
        self._db.commit()

    def counts(self) -> dict[str, int]:
        counts = {"pending": 0, "accepted": 0, "rejected": 0, "reverted": 0}
        for row in self._db.execute("SELECT status, COUNT(*) AS n FROM proposals GROUP BY status"):
            counts[str(row["status"])] = int(row["n"])
        return counts

    def accepted_since(self, since: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM proposals WHERE status = 'accepted' AND decided_at >= ?",
            (since,),
        ).fetchone()
        return int(row["n"])

    def quota_usage(self) -> tuple[int, int]:
        now = datetime.now(timezone.utc)
        hour = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        day = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.accepted_since(hour), self.accepted_since(day)

    def close(self) -> None:
        self._db.close()


class ProposeConsumer:
    """把队列里的提议一条条判过去；只有它能把 accepted 变成库里的条目。"""

    def __init__(
        self,
        store: MemoryStore,
        queue_path: Path | str,
        *,
        audit_path: Path | str | None = None,
        config: GateConfig | None = None,
        writer: MemoryWriter | None = None,
        similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        if store.read_only:
            raise ValueError("提议消费者需要一个可写库；只读库请用 axo-mcp。")
        self.store = store
        self.queue = ProposalQueue(queue_path)
        self.config = config or GateConfig()
        self._writer = writer or MemoryWriter(store)
        self._similarity = similarity or lexical_similarity
        self._audit_path = Path(audit_path) if audit_path is not None else None

    # ------------------------------------------------------------ 提议
    def ingest(
        self,
        text: str,
        *,
        proposal_id: str,
        source: str = "client",
        tags: Iterable[str] = (),
    ) -> dict[str, Any]:
        body = (text or "").strip()
        if not body:
            raise ValueError("text 必须是非空字符串。")
        if not (proposal_id or "").strip():
            raise ValueError("proposal_id 必须是非空字符串。")
        if source not in SOURCE_WEIGHTS:
            raise ValueError("source 必须是 user / client / llm 之一。")
        seq, created = self.queue.enqueue(
            proposal_id,
            {"text": body, "tags": [str(tag) for tag in tags], "source": source},
        )
        if created:
            self._audit("proposal.received", {"seq": seq, "proposal_id": proposal_id, "text": body})
        return {**self.queue.get(seq), "replayed": not created}

    # ------------------------------------------------------------ 判定
    def decide_pending(self, limit: int = 20) -> dict[str, Any]:
        decided = []
        for row in self.queue.pending(limit):
            decided.append(self.decide(row["seq"]))
        counts = self.queue.counts()
        return {"decided": decided, "counts": counts, "entries": self.store.count()}

    def decide(self, seq: int) -> dict[str, Any]:
        row = self.queue.get(seq)
        if row["status"] != "pending":
            return row
        text = row["payload"]["text"]
        source = str(row["payload"].get("source", "client"))

        similarity = self._max_similarity(text)
        hour, day = self.queue.quota_usage()
        score = salience(text, source=source, max_similarity=similarity)

        if similarity >= self.config.duplicate_similarity:
            return self._record(
                seq,
                "rejected",
                f"duplicate: similarity {similarity:.3f} ≥ {self.config.duplicate_similarity:.2f}",
                score,
                similarity,
            )
        if score < self.config.floor_salience:
            return self._record(
                seq,
                "rejected",
                f"not salient enough: {score:.3f} < {self.config.floor_salience:.2f}",
                score,
                similarity,
            )
        over_quota = hour >= self.config.quota_hourly or day >= self.config.quota_daily
        if over_quota:
            return self._record(
                seq,
                "pending",
                f"quota exhausted: {hour}/{self.config.quota_hourly} per hour, "
                f"{day}/{self.config.quota_daily} per day; held for review",
                score,
                similarity,
            )
        if score >= self.config.accept_salience:
            return self._apply(seq, row, score, similarity)
        return self._record(
            seq,
            "pending",
            f"salience {score:.3f} between {self.config.floor_salience:.2f} and "
            f"{self.config.accept_salience:.2f}; held for review",
            score,
            similarity,
        )

    # ------------------------------------------------------------ 落库
    def _apply(
        self, seq: int, row: dict[str, Any], score: float, similarity: float
    ) -> dict[str, Any]:
        """**唯一**会改记忆库的方法。"""
        text = row["payload"]["text"]
        source = str(row["payload"].get("source", "client"))
        draft = MemoryDraft(content=text)
        item, created = self._writer.record_if_absent(
            draft,
            session_id="",
            source_ref=f"propose://{row['proposal_id']}",
        )
        if item is None:
            return self._record(
                seq, "rejected", "not encodable as a lexical fingerprint", score, similarity
            )
        status = "accepted"
        reason = f"salience {score:.3f} ≥ {self.config.accept_salience:.2f}, quota free"
        self.queue.decide(seq, status, reason, salience_score=score, max_similarity=similarity)
        self.queue.record_apply(seq, memory_id=item.memory_id)
        self._audit("gate.accepted", {"seq": seq, "reason": reason, "source": source})
        self._audit(
            "memory.applied",
            {"seq": seq, "memory_id": item.memory_id, "created": created},
        )
        return {**self.queue.get(seq), "memory_id": item.memory_id}

    def _record(
        self, seq: int, status: str, reason: str, score: float, similarity: float
    ) -> dict[str, Any]:
        self.queue.decide(seq, status, reason, salience_score=score, max_similarity=similarity)
        self._audit(f"gate.{status}", {"seq": seq, "reason": reason, "salience": score})
        return self.queue.get(seq)

    def revert(self, seq: int, reason: str) -> dict[str, Any]:
        """撤销一条 accepted：把那次写进去的记忆删掉，库回到写前。"""
        why = (reason or "").strip()
        if not why:
            raise ValueError("reason 必须是非空字符串。")
        row = self.queue.get(seq)
        if row["status"] == "reverted":
            raise ValueError(f"第 {seq} 条已经撤销过。")
        if row["status"] != "accepted":
            raise ValueError(f"第 {seq} 条不是 accepted，撤销无从谈起。")
        memory_id = self.queue.applied_memory_id(seq)
        if memory_id is None:
            raise ValueError(f"第 {seq} 条找不到落库记录，拒绝撤销（宁可报错也不猜）。")
        deleted = self.store.delete(memory_id)
        self.queue.mark_reverted(seq)
        self._audit(
            "memory.revert",
            {"seq": seq, "reason": why, "memory_id": memory_id, "deleted": deleted},
        )
        return {
            "seq": seq,
            "reverted": True,
            "memory_id": memory_id,
            "entries": self.store.count(),
            "reason": why,
        }

    # ------------------------------------------------------------ 辅助
    def _max_similarity(self, text: str) -> float:
        best = 0.0
        for item in self.store.iter_items():
            best = max(best, self._similarity(text, item.content))
            if best >= 1.0:
                break
        return best

    def status(self) -> dict[str, Any]:
        hour, day = self.queue.quota_usage()
        counts = self.queue.counts()
        return {
            "counts": counts,
            "quota": {"hour": hour, "day": day},
            "gate": self.config.as_dict(),
            "entries": self.store.count(),
        }

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        if self._audit_path is None:
            return
        entry = {"ts": _now(), "event": event, **payload}
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_consumer(
    store: MemoryStore,
    *,
    enabled: bool,
    queue_path: Path | str,
    audit_path: Path | str | None = None,
) -> ProposeConsumer | None:
    """按设置决定要不要建消费者：**关着返回 None，不是返回一个不写库的消费者**。

    "关着"的正确形状是"这个东西不存在"，而不是"存在但什么也不做"——后者一旦有
    哪条路径忘了检查开关，写就发生了，而所有测试仍然是绿的。
    """
    if not enabled:
        return None
    return ProposeConsumer(store, queue_path, audit_path=audit_path)
