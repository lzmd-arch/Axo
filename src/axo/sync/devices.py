"""配对的权威账：设备、一次性配对窗口、幂等操作、锁定。

**纯层**：只依赖 sqlite3 与 ``protocol``；DPAPI/Keystore 通过注入的
``protect``/``unprotect`` 进来，测试传恒等函数。

这里只存 **token 的派生签名键**，从不存 token 本身。数据库被拿走时，攻击者
拿到的是每台设备可吊销的签名键，而不是可以再次配对的 token；配对窗口一旦关闭，
里面的 secret 也只剩 sha256。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcore.ids import new_ulid

from .protocol import PAIR_MAX_ATTEMPTS, PAIR_WINDOW_S, b64url_decode, b64url_encode, signing_key

__all__ = ["DeviceRecord", "DeviceStore", "PairResult"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    signing_key   TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_seen_ts  REAL NOT NULL,
    last_ip       TEXT NOT NULL DEFAULT '',
    revoked       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pair_windows (
    pair_id       TEXT PRIMARY KEY,
    secret_sha256 TEXT NOT NULL,
    expires_at    REAL NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    closed        INTEGER NOT NULL DEFAULT 0,
    source_ip     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sync_ops (
    device_id    TEXT NOT NULL,
    op_id        TEXT NOT NULL,
    memory_id    TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (device_id, op_id)
);

CREATE TABLE IF NOT EXISTS lockouts (
    ip        TEXT PRIMARY KEY,
    until_ts  REAL NOT NULL,
    reason    TEXT NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    name: str
    created_at: float
    last_seen_ts: float
    last_ip: str
    revoked: bool


@dataclass(frozen=True)
class PairResult:
    device_id: str
    token: bytes
    name: str
    created_at: float


def _protect_bytes(protect: Callable[[str], str], data: bytes) -> str:
    return protect(b64url_encode(data))


def _unprotect_bytes(unprotect: Callable[[str], str], stored: str) -> bytes | None:
    try:
        return b64url_decode(unprotect(stored))
    except (TypeError, ValueError):
        return None


class DeviceStore:
    def __init__(
        self,
        path: Path | str,
        *,
        protect: Callable[[str], str] = lambda value: value,
        unprotect: Callable[[str], str] = lambda value: value,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._protect = protect
        self._unprotect = unprotect
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> DeviceStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── 配对窗口 ────────────────────────────────────────────────────────────

    def create_pair_window(
        self,
        secret: bytes,
        *,
        ttl_s: int = PAIR_WINDOW_S,
        source_ip: str = "",
        now: float | None = None,
    ) -> tuple[str, float]:
        """开一个一次性窗口，返回 ``(pair_id, expires_at)``。旧的未用窗口一并关闭。"""
        moment = time.time() if now is None else now
        pair_id = new_ulid()
        expires_at = moment + max(30, min(600, int(ttl_s)))
        digest = hashlib.sha256(secret).hexdigest()
        with self._lock:
            self._connection.execute("UPDATE pair_windows SET closed = 1 WHERE closed = 0")
            self._connection.execute(
                "INSERT INTO pair_windows (pair_id, secret_sha256, expires_at, source_ip) "
                "VALUES (?, ?, ?, ?)",
                (pair_id, digest, expires_at, source_ip),
            )
            self._connection.commit()
        return pair_id, expires_at

    def consume_pair(
        self,
        pair_id: str,
        secret: bytes,
        *,
        device_name: str,
        source_ip: str = "",
        now: float | None = None,
    ) -> PairResult | None:
        """校验一次性 secret 并创建设备。失败返回 ``None``；连续失败会锁定来源 IP。"""
        moment = time.time() if now is None else now
        name = _clean_name(device_name)
        with self._lock:
            if self._locked(source_ip, moment):
                return None
            row = self._connection.execute(
                "SELECT * FROM pair_windows WHERE pair_id = ?", (pair_id,)
            ).fetchone()
            if row is None or row["closed"] or float(row["expires_at"]) < moment:
                return None
            if int(row["attempts"]) >= PAIR_MAX_ATTEMPTS:
                self._connection.execute(
                    "UPDATE pair_windows SET closed = 1 WHERE pair_id = ?", (pair_id,)
                )
                self._connection.commit()
                return None
            expected = str(row["secret_sha256"])
            actual = hashlib.sha256(secret).hexdigest()
            if not hmac.compare_digest(expected, actual):
                attempts = int(row["attempts"]) + 1
                self._connection.execute(
                    "UPDATE pair_windows SET attempts = ?, closed = ? WHERE pair_id = ?",
                    (attempts, 1 if attempts >= PAIR_MAX_ATTEMPTS else 0, pair_id),
                )
                self._connection.commit()
                if attempts >= PAIR_MAX_ATTEMPTS and source_ip:
                    self._lock_ip(source_ip, 900, "pair_failures", moment)
                return None
            token = secrets.token_bytes(32)
            device_id = "dev_" + b64url_encode(secrets.token_bytes(9))
            stored_key = _protect_bytes(self._protect, signing_key(token))
            if not stored_key:
                return None
            self._connection.execute(
                "INSERT INTO devices (device_id, name, signing_key, created_at, last_seen_ts, last_ip) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, name, stored_key, moment, moment, source_ip),
            )
            self._connection.execute(
                "UPDATE pair_windows SET closed = 1 WHERE pair_id = ?", (pair_id,)
            )
            self._connection.commit()
        return PairResult(device_id=device_id, token=token, name=name, created_at=moment)

    def consume_pair_by_secret(
        self,
        secret: bytes,
        *,
        device_name: str,
        source_ip: str = "",
        now: float | None = None,
    ) -> PairResult | None:
        """手动码没有 pair_id：按 secret 的 sha256 找当前打开的窗口。"""
        moment = time.time() if now is None else now
        digest = hashlib.sha256(secret).hexdigest()
        with self._lock:
            row = self._connection.execute(
                "SELECT pair_id FROM pair_windows "
                "WHERE secret_sha256 = ? AND closed = 0 AND expires_at >= ? "
                "ORDER BY expires_at DESC LIMIT 1",
                (digest, moment),
            ).fetchone()
        if row is None:
            return None
        return self.consume_pair(
            str(row["pair_id"]),
            secret,
            device_name=device_name,
            source_ip=source_ip,
            now=moment,
        )

    # ── 设备 ────────────────────────────────────────────────────────────────

    def authenticate(
        self, device_id: str, *, source_ip: str = "", now: float | None = None
    ) -> bytes | None:
        """返回设备签名键；未知或已吊销返回 ``None``。"""
        moment = time.time() if now is None else now
        with self._lock:
            row = self._connection.execute(
                "SELECT signing_key, revoked FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
            if row is None or int(row["revoked"]):
                return None
            key = _unprotect_bytes(self._unprotect, str(row["signing_key"]))
            if key is None or len(key) != 32:
                return None
            self._connection.execute(
                "UPDATE devices SET last_seen_ts = ?, last_ip = ? WHERE device_id = ?",
                (moment, source_ip, device_id),
            )
            self._connection.commit()
            return key

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE devices SET revoked = 1 WHERE device_id = ? AND revoked = 0",
                (device_id,),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def revoke_all(self) -> int:
        with self._lock:
            cursor = self._connection.execute("UPDATE devices SET revoked = 1 WHERE revoked = 0")
            self._connection.commit()
            return cursor.rowcount

    def list_devices(self) -> list[DeviceRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT device_id, name, created_at, last_seen_ts, last_ip, revoked "
                "FROM devices ORDER BY created_at ASC"
            ).fetchall()
        return [
            DeviceRecord(
                device_id=str(row["device_id"]),
                name=str(row["name"]),
                created_at=float(row["created_at"]),
                last_seen_ts=float(row["last_seen_ts"]),
                last_ip=str(row["last_ip"]),
                revoked=bool(row["revoked"]),
            )
            for row in rows
        ]

    def device_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM devices WHERE revoked = 0"
            ).fetchone()
        return int(row["n"]) if row else 0

    # ── 幂等操作账 ──────────────────────────────────────────────────────────

    def get_op(self, device_id: str, op_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sync_ops WHERE device_id = ? AND op_id = ?",
                (device_id, op_id),
            ).fetchone()
        return None if row is None else dict(row)

    def record_op(
        self,
        device_id: str,
        op_id: str,
        *,
        memory_id: str,
        content_sha256: str,
        result_json: str,
        now: float | None = None,
    ) -> None:
        moment = time.time() if now is None else now
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO sync_ops "
                "(device_id, op_id, memory_id, content_sha256, result_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, op_id, memory_id, content_sha256, result_json, moment),
            )
            self._connection.commit()

    # ── 锁定 ────────────────────────────────────────────────────────────────

    def _locked(self, ip: str, now: float) -> bool:
        if not ip:
            return False
        row = self._connection.execute(
            "SELECT until_ts FROM lockouts WHERE ip = ?", (ip,)
        ).fetchone()
        return row is not None and float(row["until_ts"]) > now

    def is_locked(self, ip: str, *, now: float | None = None) -> bool:
        moment = time.time() if now is None else now
        with self._lock:
            return self._locked(ip, moment)

    def _lock_ip(self, ip: str, seconds: int, reason: str, now: float) -> None:
        self._connection.execute(
            "INSERT INTO lockouts (ip, until_ts, reason) VALUES (?, ?, ?) "
            "ON CONFLICT(ip) DO UPDATE SET until_ts=excluded.until_ts, reason=excluded.reason",
            (ip, now + max(1, seconds), reason),
        )
        self._connection.commit()

    def lock_ip(self, ip: str, seconds: int = 900, reason: str = "manual") -> None:
        with self._lock:
            self._lock_ip(ip, seconds, reason, time.time())

    def cleanup(self, *, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        with self._lock:
            self._connection.execute(
                "DELETE FROM pair_windows WHERE expires_at < ? AND closed = 1",
                (moment - 3600,),
            )
            self._connection.execute("DELETE FROM lockouts WHERE until_ts < ?", (moment - 3600,))
            self._connection.commit()


def _clean_name(name: str) -> str:
    cleaned = " ".join(str(name or "").split())
    cleaned = "".join(char for char in cleaned if ord(char) >= 0x20)
    return (cleaned or "未命名设备")[:64]
