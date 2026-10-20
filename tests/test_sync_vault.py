"""记忆保管库：每条记忆单独 AEAD，信封字段绑定 id/seq/op。"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from axo.sync.protocol import (
    MEMORY_KEY_BYTES,
    derive_memory_key,
    open_record,
    seal_record,
)


class AesGcmCipher:
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return AESGCM(key).encrypt(nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)


def test_derive_memory_key_is_deterministic_and_id_bound() -> None:
    master = os.urandom(32)
    first = derive_memory_key(master, "01JDAAAAAAAAAAAAAAAAAAAAAA")
    second = derive_memory_key(master, "01JDAAAAAAAAAAAAAAAAAAAAAA")
    other = derive_memory_key(master, "01JDBBBBBBBBBBBBBBBBBBBBBB")
    assert len(first) == MEMORY_KEY_BYTES
    assert first == second
    assert first != other


def test_seal_open_round_trip() -> None:
    cipher = AesGcmCipher()
    master = os.urandom(32)
    memory_id = "01JDAAAAAAAAAAAAAAAAAAAAAA"
    envelope = seal_record(
        cipher,
        master,
        memory_id=memory_id,
        seq=7,
        op="upsert",
        payload={"content": "只有手机和桌面能看懂", "kind": "fact"},
    )
    assert "只有手机和桌面能看懂" not in str(envelope)
    opened = open_record(cipher, master, envelope)
    assert opened is not None
    assert opened["memory_id"] == memory_id
    assert opened["seq"] == 7
    assert opened["payload"]["content"] == "只有手机和桌面能看懂"


def test_open_rejects_tampered_fields() -> None:
    cipher = AesGcmCipher()
    master = os.urandom(32)
    memory_id = "01JDAAAAAAAAAAAAAAAAAAAAAA"
    envelope = seal_record(
        cipher, master, memory_id=memory_id, seq=1, op="upsert", payload={"content": "x"}
    )
    assert open_record(cipher, master, {**envelope, "seq": 2}) is None
    assert open_record(cipher, master, {**envelope, "op": "delete"}) is None
    assert open_record(cipher, os.urandom(32), envelope) is None
    # 直接改密文 -> AEAD 拒绝；open_record 返回 None，不把异常抛给上层。
    assert open_record(cipher, master, {**envelope, "data": envelope["data"][:-4] + "AAAA"}) is None
