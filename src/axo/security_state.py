"""应用锁状态的**权威副本**：``<data_root>/security/state.json``。

为什么不是只放 QSettings（注册表）：注册表按 **Windows 账户**分开，和数据根不是
一回事。删掉注册表键、换一个 Windows 用户、或者把数据根搬到另一台机器，都会让
``load_security()`` 读到空记录——旧行为是 ``StartupGate`` 退回"首次设置"，等于把
锁摘了。所以真正说了算的 verifier 必须跟着数据根走。

**这里只放密码/恢复码的 verifier，不放明文，也不放自动锁定策略。** 策略是每台
机器自己的偏好，丢了只会回到默认值；verifier 丢了会变成"换个人就能重设密码"。

格式::

    {"version": 1, "password_verifier": "pbkdf2-sha256$...",
     "recovery_verifier": "pbkdf2-sha256$...", "updated_at": "..."}

读的规则：

* 文件不存在 → ``None``（首次启动，或从旧版本升级还没迁移）；
* 文件存在但读不动/解析失败/版本不认识/verifier 为空或格式不对 → 抛
  :class:`SecurityStateError`。**调用方必须失败关闭**：这种情况不能当作"没设过
  密码"，否则删坏一个文件就能重设。
* 结构合法 → :class:`SecurityState`。

纯 stdlib；不 import Qt。
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .passwords import password_record_ok

__all__ = [
    "STATE_VERSION",
    "SecurityState",
    "SecurityStateError",
    "load_state",
    "save_state",
]

STATE_VERSION = 1


class SecurityStateError(Exception):
    """状态文件存在但不可用。**永远不要把这件事当成"首次设置"。**"""


@dataclass(frozen=True)
class SecurityState:
    password_verifier: str
    recovery_verifier: str
    updated_at: str


def _valid_verifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_state(path: Path) -> SecurityState | None:
    """见模块文档。文件不存在返回 ``None``；其余异常情况抛 ``SecurityStateError``。"""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        # ValueError 覆盖"不是 UTF-8"这种字节层损坏；不能让它冒到启动路径上。
        raise SecurityStateError(f"读不了安全状态文件：{exc}") from exc
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise SecurityStateError("安全状态文件不是合法 JSON") from exc
    if not isinstance(raw, dict):
        raise SecurityStateError("安全状态文件结构不对")
    if raw.get("version") != STATE_VERSION:
        raise SecurityStateError("安全状态文件版本不认识")
    password = raw.get("password_verifier")
    recovery = raw.get("recovery_verifier", "")
    if not _valid_verifier(password):
        raise SecurityStateError("安全状态文件里没有可用的密码记录")
    # **格式也算结构的一部分。** 只验"非空"的话，把 verifier 写成任意字符串就能让
    # 闸门退回 setup——那正是失败关闭要挡的事。
    if not password_record_ok(password.strip()):
        raise SecurityStateError("安全状态文件里的密码记录格式不对")
    if not isinstance(recovery, str):
        raise SecurityStateError("安全状态文件里的恢复码记录类型不对")
    if recovery.strip() and not password_record_ok(recovery.strip()):
        raise SecurityStateError("安全状态文件里的恢复码记录格式不对")
    updated = raw.get("updated_at")
    return SecurityState(
        password_verifier=password.strip(),
        recovery_verifier=recovery.strip(),
        updated_at=str(updated) if updated else "",
    )


def save_state(path: Path, *, password_verifier: str, recovery_verifier: str) -> None:
    """原子写：同目录临时文件 → fsync → ``os.replace``。失败抛 ``OSError``。

    **不吞异常。** 调用方（启动闸门）要靠它决定"没写进去就不能放行"。
    """
    payload = {
        "version": STATE_VERSION,
        "password_verifier": password_verifier,
        "recovery_verifier": recovery_verifier,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        with contextlib.suppress(OSError):
            path.chmod(0o600)
    finally:
        with contextlib.suppress(OSError):
            Path(name).unlink(missing_ok=True)
