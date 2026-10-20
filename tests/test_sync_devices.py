"""配对的权威账与审计：token 不落盘、失败锁定、幂等操作、hash 链。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from axo.sync.audit import AuditLog
from axo.sync.devices import DeviceStore
from axo.sync.protocol import signing_key
from memcore.ids import new_ulid


def test_pair_creates_device_and_returns_token_once(tmp_path: Path) -> None:
    store = DeviceStore(tmp_path / "authority.db")
    secret = b"a" * 16
    pair_id, expires_at = store.create_pair_window(secret)
    result = store.consume_pair(pair_id, secret, device_name="Pixel")
    assert result is not None
    assert result.name == "Pixel"
    assert len(result.token) == 32
    key = store.authenticate(result.device_id)
    assert key == signing_key(result.token)

    # token 本身不能出现在数据库里；只应存签名键（这里恒等 protect，所以是 b64url）。
    with sqlite3.connect(tmp_path / "authority.db") as raw:
        rows = raw.execute("SELECT signing_key FROM devices").fetchall()
    assert rows and result.token.hex() not in rows[0][0]

    # 窗口一次性。
    assert store.consume_pair(pair_id, secret, device_name="Again") is None


def test_pair_wrong_secret_locks_source(tmp_path: Path) -> None:
    store = DeviceStore(tmp_path / "authority.db")
    secret = b"b" * 16
    pair_id, _ = store.create_pair_window(secret, source_ip="192.168.1.9")
    for _ in range(5):
        assert (
            store.consume_pair(pair_id, b"wrong", device_name="x", source_ip="192.168.1.9") is None
        )
    assert store.is_locked("192.168.1.9")
    # 即使给对 secret，窗口已关闭。
    assert store.consume_pair(pair_id, secret, device_name="x", source_ip="192.168.1.9") is None


def test_revoke_blocks_auth(tmp_path: Path) -> None:
    store = DeviceStore(tmp_path / "authority.db")
    secret = b"c" * 16
    pair_id, _ = store.create_pair_window(secret)
    result = store.consume_pair(pair_id, secret, device_name="Laptop")
    assert result is not None
    assert store.revoke(result.device_id)
    assert store.authenticate(result.device_id) is None


def test_protect_is_applied(tmp_path: Path) -> None:
    seen: list[str] = []

    def protect(value: str) -> str:
        seen.append(value)
        return "wrapped:" + value

    def unprotect(value: str) -> str:
        assert value.startswith("wrapped:")
        return value[len("wrapped:") :]

    store = DeviceStore(tmp_path / "authority.db", protect=protect, unprotect=unprotect)
    pair_id, _ = store.create_pair_window(b"d" * 16)
    result = store.consume_pair(pair_id, b"d" * 16, device_name="Phone")
    assert result is not None
    assert seen
    assert store.authenticate(result.device_id) == signing_key(result.token)


def test_op_ledger_is_idempotent(tmp_path: Path) -> None:
    store = DeviceStore(tmp_path / "authority.db")
    pair_id, _ = store.create_pair_window(b"e" * 16)
    result = store.consume_pair(pair_id, b"e" * 16, device_name="Phone")
    assert result is not None
    op_id = new_ulid()
    store.record_op(
        result.device_id,
        op_id,
        memory_id=new_ulid(),
        content_sha256="abc",
        result_json='{"ok":true}',
    )
    store.record_op(
        result.device_id,
        op_id,
        memory_id=new_ulid(),
        content_sha256="xyz",
        result_json='{"ok":false}',
    )
    row = store.get_op(result.device_id, op_id)
    assert row is not None
    assert row["content_sha256"] == "abc"


def test_audit_hash_chain_and_redaction(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    assert log.write({"event": "append", "memory_id": "m1", "len": 3})
    assert log.write({"event": "recall", "query_sha256": "abc"})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["prev_hash"] == "0" * 64
    assert second["prev_hash"] == first["hash"]
    assert "content" not in first and "content" not in second
