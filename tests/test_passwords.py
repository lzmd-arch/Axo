"""应用密码/恢复码：只验算法与边界，不碰真实用户密码。"""

from __future__ import annotations

import base64

import pytest

from axo import passwords


def _fast(password: str = "correct horse battery") -> str:
    return passwords.hash_password(password, iterations=100_000)


def test_roundtrip_and_wrong_password() -> None:
    record = _fast()
    assert record.startswith("pbkdf2-sha256$1$100000$")
    assert passwords.verify_password("correct horse battery", record) is True
    assert passwords.verify_password("wrong password", record) is False
    assert passwords.password_record_ok(record) is True


def test_the_record_never_contains_the_plaintext() -> None:
    assert "correct horse" not in _fast()


@pytest.mark.parametrize(
    "record",
    [
        "",
        "not-a-record",
        "pbkdf2-sha256$1$100000$bad$bad",
        "argon2$1$100000$AAAA$AAAA",
        "pbkdf2-sha256$1$999$AAAA$AAAA",
        "pbkdf2-sha256$1$100000$AAAA$AAAA",
        "pbkdf2-sha256$1$3000000$AAAA$AAAA",
    ],
)
def test_malformed_records_are_rejected(record: str) -> None:
    assert passwords.parse_verifier(record) is None
    assert passwords.verify_password("correct horse battery", record) is False


def test_a_tampered_digest_does_not_verify() -> None:
    record = _fast()
    prefix, digest = record.rsplit("$", 1)
    tampered = prefix + "$" + base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")
    assert passwords.verify_password("correct horse battery", tampered) is False


def test_short_passwords_are_rejected() -> None:
    with pytest.raises(ValueError):
        passwords.hash_password("short")


def test_recovery_code_display_and_roundtrip() -> None:
    code = passwords.new_recovery_code()
    assert len(code) == 32
    shown = passwords.display_recovery_code(code)
    assert passwords.normalize_recovery_code(shown) == code
    assert passwords.normalize_recovery_code(code.lower()) == code
    assert passwords.normalize_recovery_code("not a code") is None

    verifier = passwords.hash_recovery_code(code)
    assert passwords.verify_recovery_code(shown, verifier) is True
    assert passwords.verify_recovery_code("A" * 32, verifier) is False


def test_self_test_vector() -> None:
    assert passwords.self_test() is True
