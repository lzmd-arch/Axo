"""设备身份文件：随机 ID、哈希指纹、四态比较，不泄漏原值。"""

from __future__ import annotations

from pathlib import Path

from axo import device_identity


def test_fingerprint_is_stable_and_domain_separated(monkeypatch) -> None:
    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "machine-a")
    monkeypatch.setattr(device_identity, "_macos_platform_uuid", lambda: "")
    monkeypatch.setattr(device_identity, "_linux_machine_id", lambda: "")
    first = device_identity.current_machine_fingerprint()
    second = device_identity.current_machine_fingerprint()
    assert first == second
    assert first[0] != "machine-a"
    assert "machine-a" not in first[0]


def test_identity_is_created_once_and_fingerprint_is_not_refreshed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "machine-a")
    monkeypatch.setattr(device_identity, "_macos_platform_uuid", lambda: "")
    monkeypatch.setattr(device_identity, "_linux_machine_id", lambda: "")
    path = tmp_path / "identity" / "device.json"
    first = device_identity.ensure_identity(path)
    assert path.is_file()
    assert first.fingerprint

    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "machine-b")
    again = device_identity.ensure_identity(path)
    assert again.device_id == first.device_id
    assert again.fingerprint == first.fingerprint


def test_compare_device_states(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "machine-a")
    monkeypatch.setattr(device_identity, "_macos_platform_uuid", lambda: "")
    monkeypatch.setattr(device_identity, "_linux_machine_id", lambda: "")
    path = tmp_path / "device.json"
    identity = device_identity.ensure_identity(path)

    assert device_identity.compare_device(identity)[0] == "trusted"
    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "machine-b")
    state, detail = device_identity.compare_device(identity)
    assert state == "mismatch"
    assert "仅提醒" in detail
    assert "machine-" not in detail

    assert device_identity.compare_device(None)[0] == "first_run"


def test_unavailable_platform_is_not_a_failure(monkeypatch) -> None:
    monkeypatch.setattr(device_identity, "_windows_machine_guid", lambda: "")
    monkeypatch.setattr(device_identity, "_macos_platform_uuid", lambda: "")
    monkeypatch.setattr(device_identity, "_linux_machine_id", lambda: "")
    assert device_identity.current_machine_fingerprint() == ("", "unavailable")
