"""数据根里的安全状态文件：原子写、严格读。

这一层是"删掉注册表也不会退回首次设置"的承重件，所以读的规则必须逐条测到：
**文件在但坏了 = 失败关闭**，不能和"文件不存在"混为一谈。
"""

from __future__ import annotations

import base64
import json

import pytest

from axo.security_state import SecurityStateError, load_state, save_state


def _record(fill: bytes = b"a") -> str:
    """一条格式合法的 verifier；这里测的是文件结构，不跑 PBKDF2。"""
    salt = base64.urlsafe_b64encode(fill * 16).decode("ascii").rstrip("=")
    digest = base64.urlsafe_b64encode(fill * 32).decode("ascii").rstrip("=")
    return f"pbkdf2-sha256$1$100000${salt}${digest}"


def test_missing_file_is_first_run(tmp_path) -> None:
    assert load_state(tmp_path / "security" / "state.json") is None


def test_roundtrip_creates_parent_and_records_version(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    save_state(path, password_verifier=_record(), recovery_verifier="")
    state = load_state(path)
    assert state is not None
    assert state.password_verifier == _record()
    assert state.recovery_verifier == ""
    assert state.updated_at
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 1


def test_save_replaces_existing_file_atomically(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    save_state(path, password_verifier=_record(b"o"), recovery_verifier="")
    save_state(path, password_verifier=_record(b"n"), recovery_verifier=_record(b"r"))
    state = load_state(path)
    assert state is not None and state.password_verifier == _record(b"n")
    assert [p.name for p in path.parent.iterdir()] == ["state.json"]


def test_invalid_json_is_an_error_not_first_run(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    path.parent.mkdir()
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SecurityStateError):
        load_state(path)


def test_unknown_version_is_an_error(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"version": 99, "password_verifier": "x"}), encoding="utf-8")
    with pytest.raises(SecurityStateError):
        load_state(path)


def test_missing_or_empty_password_record_is_an_error(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    path.parent.mkdir()
    for payload in (
        {"version": 1, "recovery_verifier": "rec"},
        {"version": 1, "password_verifier": "   "},
        {"version": 1, "password_verifier": "x"},
        {"version": 1, "password_verifier": 42},
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SecurityStateError):
            load_state(path)


def test_directory_instead_of_file_is_an_error(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    path.mkdir(parents=True)
    with pytest.raises(SecurityStateError):
        load_state(path)


def test_non_utf8_bytes_are_an_error_not_a_crash(tmp_path) -> None:
    path = tmp_path / "security" / "state.json"
    path.parent.mkdir()
    path.write_bytes(b"\xff\xfe{")
    with pytest.raises(SecurityStateError):
        load_state(path)
