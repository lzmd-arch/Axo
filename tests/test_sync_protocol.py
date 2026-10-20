"""同步协议的纯层向量：签名、配对码、输入校验。"""

from __future__ import annotations

import json
import time

from axo.sync.protocol import (
    PairingPayload,
    b64url_decode,
    b64url_encode,
    cert_pin,
    decode_manual_code,
    decode_pairing_qr,
    encode_manual_code,
    encode_pairing_qr,
    sign_request,
    signing_key,
    validate_append,
    validate_recall,
    verify_request,
)
from memcore.ids import new_ulid


def test_canonical_json_is_stable() -> None:
    assert json.dumps({"b": 1, "a": 2}, sort_keys=True, separators=(",", ":")) == ('{"a":2,"b":1}')


def test_request_signature_round_trip_and_tamper() -> None:
    key = signing_key(b"x" * 32)
    body = b'{"v":1}'
    signature = sign_request(key, "POST", "/v1/recall", "", 100, "nonce", body)
    assert verify_request(key, signature, "POST", "/v1/recall", "", 100, "nonce", body)
    assert not verify_request(key, signature, "POST", "/v1/recall", "", 101, "nonce", body)
    assert not verify_request(key, signature, "POST", "/v1/recall", "", 100, "other", body)
    assert not verify_request(key, signature, "POST", "/v1/recall", "", 100, "nonce", b"x")


def test_pairing_qr_round_trip() -> None:
    payload = PairingPayload(
        pair_id=new_ulid(),
        expires_at=time.time() + 120,
        hosts=("192.168.1.2", "100.64.0.2"),
        port=47831,
        pin="x" * 43,
        secret=b"y" * 16,
    )
    decoded = decode_pairing_qr(encode_pairing_qr(payload))
    assert decoded is not None
    assert decoded.hosts == payload.hosts
    assert decoded.port == 47831
    assert decoded.secret == b"y" * 16


def test_pairing_qr_rejects_unknown_field() -> None:
    raw = json.loads(
        encode_pairing_qr(
            PairingPayload(
                pair_id=new_ulid(),
                expires_at=time.time() + 120,
                hosts=("127.0.0.1",),
                port=1,
                pin="x" * 43,
                secret=b"y" * 16,
            )
        )
    )
    raw["evil"] = True
    assert decode_pairing_qr(json.dumps(raw)) is None


def test_manual_code_round_trip() -> None:
    secret = bytes(range(16))
    pin_prefix = bytes(range(8))
    code = encode_manual_code(secret, pin_prefix)
    assert decode_manual_code(code) == (secret, pin_prefix)
    assert decode_manual_code(code.lower().replace("-", "")) == (secret, pin_prefix)
    assert decode_manual_code("not-a-code") is None


def test_b64url_is_unpadded() -> None:
    assert "=" not in b64url_encode(b"hello")
    assert b64url_decode(b64url_encode(b"\x00\x01\x02")) == b"\x00\x01\x02"


def test_cert_pin_is_43_chars() -> None:
    assert len(cert_pin(b"fake der")) == 43


def test_validate_append_accepts_valid_and_rejects_unknown() -> None:
    payload = {
        "v": 1,
        "memory_id": new_ulid(),
        "op_id": new_ulid(),
        "content": "手机新增的一条记忆",
        "kind": "fact",
        "created_at": time.time(),
    }
    parsed, error = validate_append(payload)
    assert error == "" and parsed is not None
    parsed, error = validate_append({**payload, "tier": "longterm"})
    assert parsed is None and "不认识" in error


def test_validate_append_rejects_bad_content_and_kind() -> None:
    base = {
        "v": 1,
        "memory_id": new_ulid(),
        "op_id": new_ulid(),
        "content": "ok",
        "kind": "fact",
    }
    for bad in ("", "x\x00y", "x" * 4001):
        parsed, error = validate_append({**base, "content": bad})
        assert parsed is None and error
    parsed, error = validate_append({**base, "kind": "persona_anchor"})
    assert parsed is None and error


def test_validate_recall() -> None:
    parsed, error = validate_recall(
        {"v": 1, "query": "路由器", "limit": 8, "budget_tokens": 1200, "exclude_ids": []}
    )
    assert error == "" and parsed is not None
    parsed, error = validate_recall({"v": 1, "query": "x", "limit": 0})
    assert parsed is None and error
