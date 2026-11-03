"""设备身份：这台机器/这个记忆库属于谁。

**随机 ID 与机器指纹分开。**
- ``device_id``：首次运行随机生成，只用来在界面里显示“这台设备”；不是从 MAC/主机名
  派生的，复制一份文件到另一台机器会带着同一个 ID。
- ``fingerprint``：机器标识（Windows MachineGuid / macOS IOPlatformUUID / Linux
  machine-id）的域分离 SHA-256 截断值。**只存哈希，不存原值，也不进日志。**

身份文件放在 ``<data_dir>/identity/device.json``：它跟着记忆库一起搬迁，所以把库
复制到另一台机器时能被发现。指纹只用于**警告**，不做加密、不做启动拦截。

纯 stdlib；不 import Qt。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform as platform_module
import socket
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "DeviceIdentity",
    "compare_device",
    "current_machine_fingerprint",
    "ensure_identity",
    "load_identity",
    "trust_current_device",
]

_FINGERPRINT_SALT = b"axo-device-fingerprint-v1\0"


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    label: str
    platform: str
    fingerprint: str
    source: str
    created_at: str


def _hash(raw: str) -> str:
    return hashlib.sha256(_FINGERPRINT_SALT + raw.encode("utf-8")).hexdigest()[:32]


def _windows_machine_guid() -> str:
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "MachineGuid")
        return str(value).strip()
    except Exception:  # noqa: BLE001 —— 读不到就降级
        return ""


def _macos_platform_uuid() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        done = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in done.stdout.splitlines():
        if "IOPlatformUUID" not in line:
            continue
        parts = line.split("=", 1)
        if len(parts) == 2:
            return parts[1].strip().strip('"')
    return ""


def _linux_machine_id() -> str:
    if not sys.platform.startswith("linux"):
        return ""
    for name in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            value = Path(name).read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            continue
    return ""


def current_machine_fingerprint() -> tuple[str, str]:
    """``(fingerprint, source)``。读不到返回 ``("", "unavailable")``，绝不抛。"""
    try:
        raw = _windows_machine_guid()
        if raw:
            return _hash(raw), "windows-machine-guid"
        raw = _macos_platform_uuid()
        if raw:
            return _hash(raw), "macos-ioplatformuuid"
        raw = _linux_machine_id()
        if raw:
            return _hash(raw), "linux-machine-id"
    except Exception:  # noqa: BLE001
        pass
    return "", "unavailable"


def _label() -> str:
    host = ""
    with contextlib.suppress(OSError):
        host = socket.gethostname().strip()
    system = platform_module.system() or "本机"
    return f"{system} · {host}" if host else system


def _new_identity() -> DeviceIdentity:
    fingerprint, source = current_machine_fingerprint()
    return DeviceIdentity(
        device_id=uuid.uuid4().hex,
        label=_label(),
        platform=platform_module.system() or "unknown",
        fingerprint=fingerprint,
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _atomic_write(path: Path, identity: DeviceIdentity) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(identity), ensure_ascii=False, indent=1) + "\n"
    handle, name = tempfile.mkstemp(prefix=".device-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        with contextlib.suppress(OSError):
            path.chmod(0o600)
    finally:
        with contextlib.suppress(OSError):
            Path(name).unlink(missing_ok=True)


def load_identity(path: Path) -> DeviceIdentity | None:
    """读身份文件；缺失/损坏/字段不对返回 ``None``，不抛。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        identity = DeviceIdentity(
            device_id=str(raw["device_id"]),
            label=str(raw["label"]),
            platform=str(raw["platform"]),
            fingerprint=str(raw["fingerprint"]),
            source=str(raw["source"]),
            created_at=str(raw["created_at"]),
        )
    except (KeyError, TypeError):
        return None
    if not identity.device_id:
        return None
    return identity


def ensure_identity(path: Path) -> DeviceIdentity:
    """首次创建身份文件；已有则原样返回（**不刷新指纹**，否则永远发现不了漂移）。"""
    existing = load_identity(path)
    if existing is not None:
        return existing
    identity = _new_identity()
    with contextlib.suppress(OSError):
        _atomic_write(path, identity)
    return identity


def trust_current_device(path: Path, identity: DeviceIdentity) -> DeviceIdentity:
    """把当前机器指纹写进身份文件（用户点“重新信任此设备”）。"""
    fingerprint, source = current_machine_fingerprint()
    updated = DeviceIdentity(
        device_id=identity.device_id,
        label=identity.label,
        platform=identity.platform,
        fingerprint=fingerprint,
        source=source,
        created_at=identity.created_at,
    )
    _atomic_write(path, updated)
    return updated


def compare_device(identity: DeviceIdentity | None) -> tuple[str, str]:
    """``(state, detail)``；state ∈ trusted / mismatch / first_run / unavailable。"""
    if identity is None:
        return "first_run", "首次运行，将登记本机"
    current, _source = current_machine_fingerprint()
    if not current:
        return "unavailable", "无法读取本机标识（仅提醒）"
    if not identity.fingerprint:
        return "first_run", "本机标识尚未登记（仅提醒）"
    if current == identity.fingerprint:
        return "trusted", "与登记一致"
    return "mismatch", "设备指纹与登记不同（仅提醒，不拦截）"
