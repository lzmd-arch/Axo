"""MemoryBridge 业务层：配对、只增、全量快照、加密负载。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from axo.memory.policy import MemoryDraft
from axo.memory.writer import MemoryWriter
from axo.sync.protocol import (
    b64url_decode,
    b64url_encode,
    make_nonce,
    open_record,
    seal_record,
    sign_request,
    signing_key,
)
from axo.sync.service import SyncConfig, SyncRequest, SyncService, _AesGcmCipher
from memcore.ids import new_ulid
from memcore.store import MemoryStore


@pytest.fixture
def service(tmp_path: Path) -> SyncService:
    store = MemoryStore(tmp_path / "memory" / "mem.db")
    writer = MemoryWriter(store)
    writer.record_direct(
        MemoryDraft(content="桌面原有记忆"), session_id="s", source_ref="turn://s/1"
    )
    return SyncService(
        store,
        writer,
        tmp_path,
        config=SyncConfig(server_name="test", port=1, hosts=("127.0.0.1",)),
        protect=lambda value: value,
        unprotect=lambda value: value,
        pair_approver=lambda _request: True,
    )


def _call(
    service: SyncService,
    device_id: str,
    token: bytes,
    method: str,
    path: str,
    *,
    query: dict[str, str] | None = None,
    body: dict | None = None,
    authenticated: bool = True,
) -> tuple[int, dict]:
    raw = (
        b""
        if body is None
        else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    )
    query = query or {}
    query_text = "&".join(f"{key}={value}" for key, value in sorted(query.items()))
    headers: dict[str, str] = {}
    if authenticated:
        stamp = int(time.time())
        nonce = make_nonce()
        signature = sign_request(signing_key(token), method, path, query_text, stamp, nonce, raw)
        headers = {
            "x-axo-device": device_id,
            "x-axo-ts": str(stamp),
            "x-axo-nonce": nonce,
            "x-axo-sig": signature,
        }
    response = service.dispatch(SyncRequest(method, path, query, headers, raw, "127.0.0.1"))
    return response.status, response.body


def _pair(service: SyncService) -> tuple[str, bytes, bytes]:
    payload = service.open_pairing()
    response = service.dispatch(
        SyncRequest(
            "POST",
            "/v1/pair",
            {},
            {},
            json.dumps(
                {
                    "v": 1,
                    "pair_id": payload.pair_id,
                    "secret": b64url_encode(payload.secret),
                    "device_name": "test-phone",
                }
            ).encode(),
            "127.0.0.1",
        )
    )
    assert response.status == 200 and response.body["ok"]
    return (
        response.body["device_id"],
        b64url_decode(response.body["token"]),
        b64url_decode(response.body["memory_key"]),
    )


def test_status_requires_auth(service: SyncService) -> None:
    status, body = _call(service, "", b"", "GET", "/v1/status", authenticated=False)
    assert status == 401 and not body["ok"]


def test_pair_snapshot_and_append(service: SyncService) -> None:
    device, token, memory_key = _pair(service)
    status, body = _call(service, device, token, "GET", "/v1/status")
    assert status == 200 and body["ok"] and body["count"] == 1

    status, body = _call(service, device, token, "GET", "/v1/vault/snapshot")
    assert status == 200 and len(body["items"]) == 1
    record = open_record(_AesGcmCipher(), memory_key, body["items"][0])
    assert record is not None and record["payload"]["content"] == "桌面原有记忆"

    memory_id = new_ulid()
    op_id = new_ulid()
    payload = {
        "v": 1,
        "memory_id": memory_id,
        "op_id": op_id,
        "content": "手机新增",
        "kind": "fact",
        "created_at": time.time(),
    }
    envelope = seal_record(
        _AesGcmCipher(), memory_key, memory_id=memory_id, seq=0, op="append", payload=payload
    )
    status, body = _call(
        service,
        device,
        token,
        "POST",
        "/v1/vault/append",
        body={"v": 1, "record": envelope},
    )
    assert status == 201 and body["ok"] and body["duplicate"] is False
    assert service.store.count() == 2

    # 同一个 op 重试：幂等。
    status, body = _call(
        service,
        device,
        token,
        "POST",
        "/v1/vault/append",
        body={"v": 1, "record": envelope},
    )
    assert status == 200 and body["memory_id"] == memory_id and body["op_id"] == op_id
    assert service.store.count() == 2


def test_append_conflict_does_not_change_existing(service: SyncService) -> None:
    device, token, memory_key = _pair(service)
    existing_id = service.store.summaries(limit=1)[0].memory_id
    payload = {
        "v": 1,
        "memory_id": existing_id,
        "op_id": new_ulid(),
        "content": "想覆盖成别的",
        "kind": "fact",
        "created_at": time.time(),
    }
    envelope = seal_record(
        _AesGcmCipher(), memory_key, memory_id=existing_id, seq=0, op="append", payload=payload
    )
    status, body = _call(
        service,
        device,
        token,
        "POST",
        "/v1/vault/append",
        body={"v": 1, "record": envelope},
    )
    assert status == 409
    item = service.store.get(existing_id)
    assert item is not None and item.content == "桌面原有记忆"


def test_no_update_or_delete_routes(service: SyncService) -> None:
    device, token, _ = _pair(service)
    for method in ("PUT", "PATCH", "DELETE"):
        status, _ = _call(service, device, token, method, "/v1/vault/append")
        assert status in {401, 405}


def test_replay_is_rejected(service: SyncService) -> None:
    device, token, _ = _pair(service)
    raw = b""
    stamp = int(time.time())
    nonce = make_nonce()
    signature = sign_request(signing_key(token), "GET", "/v1/status", "", stamp, nonce, raw)
    headers = {
        "x-axo-device": device,
        "x-axo-ts": str(stamp),
        "x-axo-nonce": nonce,
        "x-axo-sig": signature,
    }
    first = service.dispatch(SyncRequest("GET", "/v1/status", {}, headers, raw, "127.0.0.1"))
    second = service.dispatch(SyncRequest("GET", "/v1/status", {}, headers, raw, "127.0.0.1"))
    assert first.status == 200
    assert second.status == 401
