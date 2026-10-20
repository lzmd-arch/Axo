"""MemoryBridge 的业务层：把加密信封、设备审计与记忆库接起来。

这里没有 Qt、没有 socket。服务器只把已经解析好的 :class:`SyncRequest` 交进来，
拿回 :class:`SyncResponse`。所有"能不能做"的判断都在这一层：

* 默认拒绝：方法/路径不在白名单、设备未配对、签名/时间/nonce/限速不通过，
  一律拒绝；
* 记忆只走 ``record_if_absent``；手机端请求里没有 update/delete 这种动词；
* 桌面端到手机端的所有记忆负载都用**记忆专用密钥**做 AEAD，TLS 只是外层。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axo.memory.policy import MemoryDraft
from axo.memory.writer import MemoryWriter
from memcore.recall import RecallConfig, recall
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint

from .audit import AuditLog
from .certs import CertInfo, ensure_server_cert
from .devices import DeviceStore
from .protocol import (
    CLOCK_SKEW_S,
    MAX_LIST_LIMIT,
    MEMORY_KEY_BYTES,
    NONCE_RETENTION_S,
    PROTOCOL_VERSION,
    PairingPayload,
    b64url_decode,
    b64url_encode,
    open_record,
    seal_record,
    validate_append,
    validate_recall,
    verify_request,
)
from .qr import make_qr_svg

__all__ = ["SyncConfig", "SyncRequest", "SyncResponse", "SyncService"]


@dataclass(frozen=True)
class SyncConfig:
    server_name: str = "Axo"
    port: int = 47831
    hosts: tuple[str, ...] = ()
    max_devices: int = 4
    audit_enabled: bool = True


@dataclass(frozen=True)
class SyncRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: Mapping[str, str]
    body: bytes
    source_ip: str
    source_port: int = 0


@dataclass(frozen=True)
class SyncResponse:
    status: int
    body: dict[str, Any]


class _Buckets:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > window_s:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


class _AesGcmCipher:
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(key).encrypt(nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(key).decrypt(nonce, ciphertext, aad)


def ensure_master_key(
    path: Path,
    *,
    protect: Callable[[str], str],
    unprotect: Callable[[str], str],
) -> bytes:
    if path.is_file():
        try:
            key = b64url_decode(unprotect(path.read_text(encoding="utf-8").strip()))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "记忆专用密钥读不出来（可能换了 Windows 账户）。请在设置里重置信任。"
            ) from exc
        if len(key) != MEMORY_KEY_BYTES:
            raise RuntimeError("记忆专用密钥长度不对；请重置信任。")
        return key
    key = secrets.token_bytes(MEMORY_KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(protect(b64url_encode(key)), encoding="utf-8", newline="\n")
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return key


class SyncService:
    def __init__(
        self,
        store: MemoryStore,
        writer: MemoryWriter,
        data_dir: Path | str,
        *,
        config: SyncConfig,
        protect: Callable[[str], str],
        unprotect: Callable[[str], str],
        pair_approver: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.store = store
        self.writer = writer
        self.config = config
        self.data_dir = Path(data_dir)
        self.sync_dir = self.data_dir / "sync"
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        self.store.install_change_log()
        self.devices = DeviceStore(
            self.sync_dir / "authority.sqlite3", protect=protect, unprotect=unprotect
        )
        self.audit = AuditLog(self.sync_dir / "audit.jsonl", enabled=config.audit_enabled)
        self.master_key = ensure_master_key(
            self.sync_dir / "memory.key.dpapi", protect=protect, unprotect=unprotect
        )
        self.cert: CertInfo = ensure_server_cert(
            self.sync_dir / "tls", protect=protect, unprotect=unprotect
        )
        self.pair_approver = pair_approver
        self._cipher = _AesGcmCipher()
        self._nonces: dict[str, float] = {}
        self._nonce_lock = threading.Lock()
        self._buckets = _Buckets()
        self._pairing: tuple[PairingPayload, str] | None = None
        self._pairing_lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self.devices.close()

    # ── 配对 ────────────────────────────────────────────────────────────────

    def open_pairing(self) -> PairingPayload:
        if self.devices.device_count() >= max(1, self.config.max_devices):
            raise RuntimeError("已配对设备数到上限；先吊销一台再配对。")
        secret = secrets.token_bytes(16)
        pair_id, expires_at = self.devices.create_pair_window(secret)
        payload = PairingPayload(
            pair_id=pair_id,
            expires_at=expires_at,
            hosts=tuple(self.config.hosts) or ("127.0.0.1",),
            port=self.config.port,
            pin=self.cert.pin,
            secret=secret,
        )
        with self._pairing_lock:
            self._pairing = (payload, pair_id)
        return payload

    def pairing_svg(self) -> bytes:
        payload = self.open_pairing()
        return make_qr_svg(payload.to_qr_json())

    def _pair(self, request: SyncRequest) -> SyncResponse:
        parsed = _json_object(request.body)
        if parsed is None:
            return _error(400, "请求不是 JSON 对象")
        allowed = {"v", "pair_id", "secret", "device_name"}
        if set(parsed) - allowed:
            return _error(400, "不认识的字段")
        if parsed.get("v") != PROTOCOL_VERSION:
            return _error(400, "协议版本不对")
        pair_id = parsed.get("pair_id")
        if pair_id is not None and (not isinstance(pair_id, str) or not pair_id):
            return _error(400, "pair_id 不合法")
        try:
            secret = b64url_decode(str(parsed.get("secret", "")))
        except (ValueError, TypeError):
            return _error(400, "secret 不合法")
        if len(secret) != 16:
            return _error(400, "secret 长度不对")
        name = str(parsed.get("device_name") or "新设备")[:64]
        with self._pairing_lock:
            current = self._pairing
        if pair_id and (current is None or current[0].pair_id != pair_id):
            return _error(404, "配对窗口不存在或已关闭")
        if self.pair_approver is not None and not self.pair_approver(
            {"pair_id": pair_id, "device_name": name, "source_ip": request.source_ip}
        ):
            self.audit.write({"event": "pair_denied", "pair_id": pair_id, "ip": request.source_ip})
            return _error(403, "桌面端没有确认这次配对")
        if pair_id:
            result = self.devices.consume_pair(
                pair_id, secret, device_name=name, source_ip=request.source_ip
            )
        else:
            result = self.devices.consume_pair_by_secret(
                secret, device_name=name, source_ip=request.source_ip
            )
        if result is None:
            self.audit.write({"event": "pair_failed", "pair_id": pair_id, "ip": request.source_ip})
            return _error(403, "配对失败")
        with self._pairing_lock:
            self._pairing = None
        self.audit.write(
            {
                "event": "pair_ok",
                "device": result.device_id,
                "ip": request.source_ip,
                "name": result.name,
            }
        )
        return SyncResponse(
            200,
            {
                "ok": True,
                "v": PROTOCOL_VERSION,
                "device_id": result.device_id,
                "token": b64url_encode(result.token),
                "memory_key": b64url_encode(self.master_key),
                "pin": self.cert.pin,
                "server_name": self.config.server_name,
            },
        )

    # ── 入口 ────────────────────────────────────────────────────────────────

    def dispatch(self, request: SyncRequest) -> SyncResponse:
        if self._closed:
            return _error(503, "同步服务已停止")
        started = time.monotonic()
        try:
            if request.path == "/v1/pair" and request.method == "POST":
                if not self._buckets.allow(f"pair:{request.source_ip}", 10):
                    return _error(429, "配对请求太频繁")
                return self._pair(request)
            if not self._buckets.allow("global", 600):
                return _error(429, "请求太频繁")
            auth = self._authenticate(request)
            if isinstance(auth, SyncResponse):
                return auth
            device_id, key = auth
            if not self._buckets.allow(f"dev:{device_id}", 120):
                return _error(429, "请求太频繁")
            if request.method == "GET" and request.path == "/v1/status":
                return self._status(request)
            if request.method == "GET" and request.path == "/v1/vault/snapshot":
                return self._snapshot(request)
            if request.method == "GET" and request.path == "/v1/vault/events":
                return self._events(request)
            if request.method == "POST" and request.path == "/v1/recall":
                if not self._buckets.allow(f"recall:{device_id}", 30):
                    return _error(429, "召回太频繁")
                return self._recall(request)
            if request.method == "POST" and request.path == "/v1/vault/append":
                if not self._buckets.allow(f"append:{device_id}", 30, window_s=3600):
                    return _error(429, "追加太频繁")
                return self._append(request, device_id)
            if request.method == "POST" and request.path == "/v1/unpair":
                self.devices.revoke(device_id)
                self.audit.write({"event": "unpair", "device": device_id, "ip": request.source_ip})
                return SyncResponse(200, {"ok": True})
            if request.method in {"PUT", "PATCH", "DELETE"}:
                return _error(405, "不支持的方法")
            return _error(404, "没有这个接口")
        finally:
            if request.path != "/v1/pair":
                self.audit.write(
                    {
                        "event": "request",
                        "method": request.method,
                        "path": request.path,
                        "ip": request.source_ip,
                        "ms": int((time.monotonic() - started) * 1000),
                    }
                )

    def _authenticate(self, request: SyncRequest) -> tuple[str, bytes] | SyncResponse:
        device_id = request.headers.get("x-axo-device", "").strip()
        timestamp = request.headers.get("x-axo-ts", "").strip()
        nonce = request.headers.get("x-axo-nonce", "").strip()
        signature = request.headers.get("x-axo-sig", "").strip()
        if not device_id or not timestamp or not nonce or not signature:
            return _error(401, "未认证")
        try:
            stamp = int(timestamp)
        except ValueError:
            return _error(401, "时间戳不合法")
        if abs(time.time() - stamp) > CLOCK_SKEW_S:
            return _error(401, "时间戳超出窗口")
        if len(nonce) > 64:
            return _error(401, "nonce 不合法")
        with self._nonce_lock:
            moment = time.monotonic()
            for key, seen in list(self._nonces.items()):
                if moment - seen > NONCE_RETENTION_S:
                    del self._nonces[key]
            if nonce in self._nonces:
                return _error(401, "重放")
            if len(self._nonces) > 100_000:
                return _error(503, "nonce 缓存已满")
        key = self.devices.authenticate(device_id, source_ip=request.source_ip)
        if key is None:
            return _error(401, "设备未配对或已吊销")
        query = _query_string(request.query)
        if not verify_request(
            key, signature, request.method, request.path, query, stamp, nonce, request.body
        ):
            return _error(401, "签名不匹配")
        with self._nonce_lock:
            self._nonces[nonce] = moment
        return device_id, key

    # ── 读 ──────────────────────────────────────────────────────────────────

    def _status(self, _request: SyncRequest) -> SyncResponse:
        return SyncResponse(
            200,
            {
                "ok": True,
                "v": PROTOCOL_VERSION,
                "server": self.config.server_name,
                "rev": self.store.change_head(),
                "count": self.store.count(),
                "devices": self.devices.device_count(),
                "server_time": time.time(),
            },
        )

    def _snapshot(self, request: SyncRequest) -> SyncResponse:
        cursor = request.query.get("cursor", "")
        try:
            limit = max(1, min(MAX_LIST_LIMIT, int(request.query.get("limit", "200"))))
        except ValueError:
            return _error(400, "limit 不合法")
        items, next_cursor = self.store.items_page(
            after_id=cursor, limit=limit, descending=True, exclude_persona=True
        )
        head = self.store.change_head()
        envelopes = [
            seal_record(
                self._cipher,
                self.master_key,
                memory_id=item.memory_id,
                seq=head,
                op="snapshot",
                payload=_memory_payload(item),
            )
            for item in items
        ]
        return SyncResponse(
            200,
            {
                "ok": True,
                "v": PROTOCOL_VERSION,
                "rev": head,
                "items": envelopes,
                "next_cursor": next_cursor,
                "has_more": bool(next_cursor),
                "total": self.store.count(),
            },
        )

    def _events(self, request: SyncRequest) -> SyncResponse:
        try:
            since = max(0, int(request.query.get("since", "0")))
            limit = max(1, min(500, int(request.query.get("limit", "200"))))
        except ValueError:
            return _error(400, "since/limit 不合法")
        changes = self.store.changes_since(since, limit=limit)
        envelopes: list[dict[str, Any]] = []
        next_seq = since
        for change in changes:
            next_seq = change.seq
            if change.op == "delete":
                envelopes.append(
                    seal_record(
                        self._cipher,
                        self.master_key,
                        memory_id=change.memory_id,
                        seq=change.seq,
                        op="delete",
                        payload={},
                    )
                )
                continue
            item = self.store.get(change.memory_id)
            if item is None or item.kind is MemoryKind.PERSONA_ANCHOR:
                continue
            envelopes.append(
                seal_record(
                    self._cipher,
                    self.master_key,
                    memory_id=item.memory_id,
                    seq=change.seq,
                    op="upsert",
                    payload=_memory_payload(item),
                )
            )
        return SyncResponse(
            200,
            {
                "ok": True,
                "v": PROTOCOL_VERSION,
                "rev": self.store.change_head(),
                "items": envelopes,
                "next_seq": next_seq,
                "has_more": len(changes) >= limit,
            },
        )

    def _recall(self, request: SyncRequest) -> SyncResponse:
        parsed = _json_object(request.body)
        if parsed is None:
            return _error(400, "请求不是 JSON 对象")
        fields, error = validate_recall(parsed)
        if error:
            return _error(400, error)
        assert fields is not None
        ledger = recall(
            self.store,
            query_text=fields["query"],
            budget_tokens=fields["budget_tokens"],
            config=RecallConfig(max_items=fields["limit"]),
            exclude_ids=fields["exclude_ids"],
            settle=False,
        )
        head = self.store.change_head()
        envelope_items = []
        for item in ledger.injected:
            payload = {
                "memory_id": item.memory_id,
                "content": item.content,
                "kind": item.kind,
                "created_at": item.created_at,
                "score": float(item.scores.fused),
                "explain": item.explain,
                "weight": float(item.weight),
            }
            envelope_items.append(
                seal_record(
                    self._cipher,
                    self.master_key,
                    memory_id=item.memory_id,
                    seq=head,
                    op="recall",
                    payload=payload,
                )
            )
        return SyncResponse(
            200,
            {
                "ok": True,
                "v": PROTOCOL_VERSION,
                "rev": head,
                "items": envelope_items,
                "truncated": False,
            },
        )

    # ── 写（只追加） ────────────────────────────────────────────────────────

    def _append(self, request: SyncRequest, device_id: str) -> SyncResponse:
        parsed = _json_object(request.body)
        if parsed is None or set(parsed) - {"v", "record"}:
            return _error(400, "请求不是合法的追加信封")
        record = parsed.get("record")
        if not isinstance(record, Mapping):
            return _error(400, "record 不是对象")
        opened = open_record(self._cipher, self.master_key, record)
        if opened is None:
            return _error(400, "记忆信封打不开")
        if opened.get("op") != "append":
            return _error(405, "只接受 append")
        payload = opened.get("payload")
        if not isinstance(payload, Mapping):
            return _error(400, "payload 不是对象")
        fields, error = validate_append(payload)
        if error:
            return _error(400, error)
        assert fields is not None
        if opened.get("memory_id") != fields["memory_id"]:
            return _error(400, "信封与 payload 的 memory_id 不一致")
        previous = self.devices.get_op(device_id, fields["op_id"])
        if previous is not None:
            return SyncResponse(200, json.loads(previous["result_json"]))
        draft = MemoryDraft(
            content=fields["content"],
            kind=MemoryKind(fields["kind"]),
            taint=Taint.USER,
            weight=0.5,
            meta={"via": "mobile_sync", "device": device_id, "op_id": fields["op_id"]},
        )
        item, created = self.writer.record_if_absent(
            draft,
            session_id=f"mobile/{device_id}",
            source_ref=f"mobile://{device_id}/{fields['op_id']}",
            memory_id=fields["memory_id"],
            created_at=fields["created_at"],
        )
        if item is None:
            return _error(422, "这条内容无法编码成记忆")
        if not created and item.content != fields["content"]:
            return _error(409, "这条 id 已存在且内容不同；同步不允许改已有记忆")
        result = {
            "ok": True,
            "v": PROTOCOL_VERSION,
            "memory_id": item.memory_id,
            "op_id": fields["op_id"],
            "duplicate": not created,
            "rev": self.store.change_head(),
        }
        self.devices.record_op(
            device_id,
            fields["op_id"],
            memory_id=item.memory_id,
            content_sha256=hashlib.sha256(fields["content"].encode("utf-8")).hexdigest(),
            result_json=json.dumps(result, ensure_ascii=False),
        )
        self.audit.write(
            {
                "event": "append",
                "device": device_id,
                "ip": request.source_ip,
                "memory_id": item.memory_id,
                "op_id": fields["op_id"],
                "len": len(fields["content"]),
                "content_sha256": hashlib.sha256(fields["content"].encode("utf-8")).hexdigest()[
                    :16
                ],
                "duplicate": not created,
            }
        )
        return SyncResponse(201 if created else 200, result)


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _query_string(query: dict[str, str]) -> str:
    if not query:
        return ""
    return "&".join(f"{key}={value}" for key, value in sorted(query.items()))


def _memory_payload(item: MemoryItem) -> dict[str, Any]:
    meta = dict(item.meta)
    encoded = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if len(encoded) > 16_384:
        meta = {"omitted": True}
    return {
        "memory_id": item.memory_id,
        "content": item.content,
        "kind": str(item.kind),
        "tier": str(item.tier),
        "taint": str(item.taint),
        "created_at": float(item.created_at),
        "source_ref": item.source_ref,
        "session_id": item.session_id,
        "pinned": bool(item.pinned),
        "meta": meta,
    }


def _error(status: int, message: str) -> SyncResponse:
    return SyncResponse(status, {"ok": False, "error": message})
