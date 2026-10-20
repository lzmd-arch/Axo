"""同步协议的**纯层**：常量、规范化、签名、配对码与输入校验。

这一层不碰 Qt、不碰 socket、不碰文件；桌面的服务器与 Android 的客户端导入同一份
文件，所以"两边算出来的签名是否一致"不靠两处各写一遍来保证。

安全边界只在这里定义一次：

* 请求签名覆盖 **方法 + 路径 + 查询串 + 时间戳 + nonce + body 的 sha256**；
* 配对码是 128/256 位随机串，不是可猜的短码；
* 所有进来给服务端的 JSON 都按固定 schema 校验，未知字段直接拒绝；
* ULID 由客户端生成但必须形状合法，服务端仍用 ``add_if_absent`` 兜底。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from memcore.ids import is_ulid

__all__ = [
    "CLOCK_SKEW_S",
    "MAX_BODY_BYTES",
    "MAX_CONTENT_CHARS",
    "MAX_EVENTS_LIMIT",
    "MAX_LIST_LIMIT",
    "MAX_QUERY_CHARS",
    "NONCE_RETENTION_S",
    "PAIR_MAX_ATTEMPTS",
    "PAIR_WINDOW_S",
    "PROTOCOL_VERSION",
    "MEMORY_KEY_BYTES",
    "MemoryCipher",
    "NONCE_BYTES",
    "PairingPayload",
    "b64url_decode",
    "b64url_encode",
    "canonical_json",
    "canonical_request",
    "cert_pin",
    "decode_manual_code",
    "decode_pairing_qr",
    "encode_pairing_qr",
    "encode_manual_code",
    "make_nonce",
    "memory_aad",
    "open_record",
    "seal_record",
    "derive_memory_key",
    "sign_request",
    "signing_key",
    "validate_append",
    "validate_recall",
    "verify_request",
]

PROTOCOL_VERSION = 1

MEMORY_KEY_BYTES = 32
NONCE_BYTES = 12
VAULT_CONTEXT = b"axo-memory-v1"

MAX_BODY_BYTES = 65_536
MAX_QUERY_CHARS = 1_000
MAX_CONTENT_CHARS = 4_000
MAX_LIST_LIMIT = 200
MAX_EVENTS_LIMIT = 500
CLOCK_SKEW_S = 300
NONCE_RETENTION_S = 600
PAIR_WINDOW_S = 120
PAIR_MAX_ATTEMPTS = 5

_SIGN_CONTEXT = b"axo-sync-sign-v1"
_PAIR_KIND = "axo-pair"
_ALLOWED_KINDS = frozenset({"fact", "preference", "decision", "conclusion", "open_question"})
_PAIR_HOST_RE = re.compile(r"^[0-9A-Fa-f:.]+$")
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_INDEX = {char: index for index, char in enumerate(_CROCKFORD)}


def canonical_json(value: Any) -> str:
    """固定键序、无多余空白的 JSON。签名与配对码都靠它保持一致。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def signing_key(token: bytes) -> bytes:
    """设备 token 的派生键。服务端只存这个，不存 token 本身。"""
    return hmac.new(token, _SIGN_CONTEXT, hashlib.sha256).digest()


def canonical_request(
    method: str,
    path: str,
    query: str,
    timestamp: int | str,
    nonce: str,
    body: bytes,
) -> bytes:
    return "\n".join(
        (
            method.upper().strip(),
            path.strip(),
            query.strip(),
            str(timestamp),
            nonce,
            hashlib.sha256(body or b"").hexdigest(),
        )
    ).encode("utf-8")


def sign_request(
    key: bytes,
    method: str,
    path: str,
    query: str,
    timestamp: int | str,
    nonce: str,
    body: bytes,
) -> str:
    message = canonical_request(method, path, query, timestamp, nonce, body)
    return b64url_encode(hmac.new(key, message, hashlib.sha256).digest())


def verify_request(
    key: bytes,
    signature: str,
    method: str,
    path: str,
    query: str,
    timestamp: int | str,
    nonce: str,
    body: bytes,
) -> bool:
    expected = sign_request(key, method, path, query, timestamp, nonce, body)
    return hmac.compare_digest(expected, signature or "")


def make_nonce() -> str:
    return b64url_encode(secrets.token_bytes(16))


def cert_pin(der: bytes) -> str:
    return b64url_encode(hashlib.sha256(der).digest())


# ── 记忆保管库：只给记忆用的 AEAD 层 ────────────────────────────────────────
#
# 与 TLS 是两层：TLS 保护"这段连接"，这一层保护"这条记忆"。即使同步日志被
# 拷贝、经过盲中继、或以后换成别的传输，没有 MK 也只有密文。MK 在配对时经
# 已固定的 TLS 通道交给手机，桌面用 DPAPI、手机用 Keystore 保存。


class MemoryCipher(Protocol):
    """AEAD 的最小接口。桌面用 ``cryptography``，Android 用 Java Cipher。"""

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes: ...

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes: ...


def _hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    output = bytearray()
    block = b""
    counter = 1
    while len(output) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:length])


def derive_memory_key(master_key: bytes, memory_id: str) -> bytes:
    """每条记忆一把子钥匙：同一条记忆的密文不会在别的 id 上解开。"""
    if len(master_key) != MEMORY_KEY_BYTES:
        raise ValueError("memory master key 必须是 32 字节")
    if not memory_id:
        raise ValueError("memory_id 不能为空")
    return _hkdf_sha256(master_key, memory_id.encode("utf-8"), VAULT_CONTEXT, 32)


def memory_aad(memory_id: str, seq: int, op: str) -> bytes:
    return canonical_json(
        {"v": PROTOCOL_VERSION, "memory_id": memory_id, "seq": int(seq), "op": op}
    ).encode("utf-8")


def seal_record(
    cipher: MemoryCipher,
    master_key: bytes,
    *,
    memory_id: str,
    seq: int,
    op: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """把一条记忆封成可传输/可落盘的密文信封。"""
    plaintext = canonical_json(
        {
            "v": PROTOCOL_VERSION,
            "memory_id": memory_id,
            "seq": int(seq),
            "op": op,
            "payload": dict(payload),
        }
    ).encode("utf-8")
    nonce = secrets.token_bytes(NONCE_BYTES)
    key = derive_memory_key(master_key, memory_id)
    data = cipher.encrypt(key, nonce, plaintext, memory_aad(memory_id, seq, op))
    return {
        "v": PROTOCOL_VERSION,
        "memory_id": memory_id,
        "seq": int(seq),
        "op": op,
        "nonce": b64url_encode(nonce),
        "data": b64url_encode(data),
    }


def open_record(
    cipher: MemoryCipher,
    master_key: bytes,
    envelope: Mapping[str, Any],
) -> dict[str, Any] | None:
    """解开信封。任何字段不匹配都返回 ``None``，不抛异常给上层。"""
    try:
        if envelope.get("v") != PROTOCOL_VERSION:
            return None
        memory_id = str(envelope["memory_id"])
        seq = int(envelope["seq"])
        op = str(envelope["op"])
        nonce = b64url_decode(str(envelope["nonce"]))
        data = b64url_decode(str(envelope["data"]))
        if len(nonce) != NONCE_BYTES:
            return None
        key = derive_memory_key(master_key, memory_id)
        try:
            plaintext = cipher.decrypt(key, nonce, data, memory_aad(memory_id, seq, op))
        except Exception:  # noqa: BLE001 —— 密文来自网络，AEAD 失败必须当作"打不开"
            return None
        record = json.loads(plaintext.decode("utf-8"))
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if (
        record.get("v") != PROTOCOL_VERSION
        or record.get("memory_id") != memory_id
        or int(record.get("seq", -1)) != seq
        or record.get("op") != op
        or not isinstance(record.get("payload"), dict)
    ):
        return None
    return record


# ── 配对码 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairingPayload:
    pair_id: str
    expires_at: float
    hosts: tuple[str, ...]
    port: int
    pin: str
    secret: bytes

    def to_qr_json(self) -> str:
        return canonical_json(
            {
                "v": PROTOCOL_VERSION,
                "t": _PAIR_KIND,
                "id": self.pair_id,
                "exp": int(self.expires_at),
                "h": list(self.hosts),
                "p": int(self.port),
                "pin": self.pin,
                "s": b64url_encode(self.secret),
            }
        )


def encode_pairing_qr(payload: PairingPayload) -> str:
    return payload.to_qr_json()


def decode_pairing_qr(text: str) -> PairingPayload | None:
    """解析桌面二维码/``axo-pair.txt``。**任何字段不合法都返回 ``None``。**"""
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    if set(raw) != {"v", "t", "id", "exp", "h", "p", "pin", "s"}:
        return None
    if raw.get("v") != PROTOCOL_VERSION or raw.get("t") != _PAIR_KIND:
        return None
    pair_id = raw.get("id")
    if not isinstance(pair_id, str) or not is_ulid(pair_id):
        return None
    expires_at = raw.get("exp")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    hosts = raw.get("h")
    if not isinstance(hosts, list) or not 1 <= len(hosts) <= 4:
        return None
    if not all(
        isinstance(host, str) and _PAIR_HOST_RE.match(host) and len(host) <= 45 for host in hosts
    ):
        return None
    port = raw.get("p")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        return None
    pin = raw.get("pin")
    if not isinstance(pin, str) or len(pin) < 40 or len(pin) > 60:
        return None
    try:
        secret = b64url_decode(raw.get("s", ""))
    except (ValueError, TypeError):
        return None
    if len(secret) != 16:
        return None
    return PairingPayload(
        pair_id=pair_id,
        expires_at=float(expires_at),
        hosts=tuple(hosts),
        port=port,
        pin=pin,
        secret=secret,
    )


def encode_manual_code(secret: bytes, pin_prefix: bytes) -> str:
    """16 字节 secret + 8 字节 pin 前缀 → 12 组 4 位 Crockford。

    pin 前缀取 8 字节而不是 4：手动流程没有二维码里的完整 pin，配对期间只能靠
    这个前缀把服务器证书钉住。64 位足以挡住一次定向 MITM；配对响应里的完整 pin
    还会再核对一次。
    """
    if len(secret) != 16 or len(pin_prefix) != 8:
        raise ValueError("手动码必须是 16 字节 secret + 8 字节 pin 前缀")
    raw = secret + pin_prefix
    chars: list[str] = []
    for byte in raw:
        chars.append(_CROCKFORD[byte >> 4])
        chars.append(_CROCKFORD[byte & 0x0F])
    text = "".join(chars)
    return "-".join(text[index : index + 4] for index in range(0, len(text), 4))


def decode_manual_code(code: str) -> tuple[bytes, bytes] | None:
    cleaned = "".join(char for char in (code or "").upper() if char not in "- \t\r\n")
    if len(cleaned) != 48:
        return None
    raw = bytearray()
    for index in range(0, 48, 2):
        high = _CROCKFORD_INDEX.get(cleaned[index])
        low = _CROCKFORD_INDEX.get(cleaned[index + 1])
        if high is None or low is None:
            return None
        raw.append((high << 4) | low)
    return bytes(raw[:16]), bytes(raw[16:24])


# ── 输入校验 ────────────────────────────────────────────────────────────────


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str]) -> str:
    unknown = set(payload) - allowed
    if unknown:
        return f"不认识的字段：{', '.join(sorted(unknown))}"
    return ""


def _valid_text(value: Any, *, max_chars: int, allow_newlines: bool = True) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        return False
    for char in value:
        if char in ("\n", "\t") and allow_newlines:
            continue
        if ord(char) < 0x20 or ord(char) == 0x7F:
            return False
    return True


def validate_append(payload: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """校验手机 append 请求。返回 ``(清洗后的字段, 错误)``；错误为空串表示通过。"""
    if not isinstance(payload, Mapping):
        return None, "请求必须是 JSON 对象"
    error = _reject_unknown(payload, {"v", "memory_id", "op_id", "content", "kind", "created_at"})
    if error:
        return None, error
    if payload.get("v") != PROTOCOL_VERSION:
        return None, "协议版本不对"
    memory_id = payload.get("memory_id")
    if not isinstance(memory_id, str) or not is_ulid(memory_id):
        return None, "memory_id 不是合法 ULID"
    op_id = payload.get("op_id")
    if not isinstance(op_id, str) or not is_ulid(op_id):
        return None, "op_id 不是合法 ULID"
    content = payload.get("content")
    if not _valid_text(content, max_chars=MAX_CONTENT_CHARS):
        return None, "content 为空、过长或包含控制字符"
    kind = payload.get("kind", "fact")
    if kind not in _ALLOWED_KINDS:
        return None, "kind 不在允许的集合里"
    created_at = payload.get("created_at")
    if created_at is not None:
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            return None, "created_at 必须是数字"
        if not 0 <= float(created_at) <= time.time() + CLOCK_SKEW_S:
            return None, "created_at 不在合理范围"
    return {
        "memory_id": memory_id,
        "op_id": op_id,
        "content": content,
        "kind": kind,
        "created_at": None if created_at is None else float(created_at),
    }, ""


def validate_recall(payload: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(payload, Mapping):
        return None, "请求必须是 JSON 对象"
    error = _reject_unknown(payload, {"v", "query", "limit", "budget_tokens", "exclude_ids"})
    if error:
        return None, error
    if payload.get("v") != PROTOCOL_VERSION:
        return None, "协议版本不对"
    query = payload.get("query")
    if not _valid_text(query, max_chars=MAX_QUERY_CHARS):
        return None, "query 为空、过长或包含控制字符"
    limit = payload.get("limit", 8)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        return None, "limit 必须是 1..50"
    budget = payload.get("budget_tokens", 1200)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 64 <= budget <= 100_000:
        return None, "budget_tokens 必须是 64..100000"
    exclude = payload.get("exclude_ids", [])
    if not isinstance(exclude, list) or len(exclude) > 50:
        return None, "exclude_ids 必须是最多 50 个 id 的数组"
    if not all(isinstance(one, str) and is_ulid(one) for one in exclude):
        return None, "exclude_ids 里有非法 ULID"
    return {
        "query": query,
        "limit": limit,
        "budget_tokens": budget,
        "exclude_ids": tuple(exclude),
    }, ""
