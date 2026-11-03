"""记忆加密的密钥封装接口——**预留，不接当前记忆库**。

当前 `mem.db` 是明文 SQLite；本模块不创建数据库密钥、不改 `MemoryStore`、不引入
加密依赖。它只定义“将来把数据库密钥绑到系统凭据/硬件上”的接缝：

* :class:`SystemKeyWrapper` 走 Windows DPAPI / macOS Keychain（诚实标为
  ``dpapi-user-machine`` / ``keychain-login``，**不是硬件**）；
* :class:`TpmKeyWrapper` / :class:`SecureEnclaveKeyWrapper` 是占位，当前
  ``available=False``；
* :class:`NullKeyWrapper` 永不回退明文，调用方必须 fail-closed。

将来的数据库密钥路径预留在 ``<data_dir>/memory/mem.db.key.v1``，本轮不创建。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from . import secrets

__all__ = [
    "MEMORY_DB_KEY_CONTEXT",
    "KeyWrapper",
    "KeyWrapperUnavailable",
    "NullKeyWrapper",
    "SecureEnclaveKeyWrapper",
    "SystemKeyWrapper",
    "TpmKeyWrapper",
    "current_wrapper",
    "memory_key_path",
    "probe",
]

MEMORY_DB_KEY_CONTEXT = "memory/mem.db.key.v1"
MEMORY_DB_KEY_FILENAME = "mem.db.key.v1"


class KeyWrapperUnavailable(RuntimeError):
    """要求硬件封装但当前机器/构建没有可用实现。"""


class KeyWrapper(Protocol):
    """把一段数据库密钥包成可落盘的 token；解不开时必须返回 ``None``。"""

    kind: str

    def available(self) -> bool: ...

    def wrap(self, secret: bytes, *, context: str) -> str: ...

    def unwrap(self, token: str, *, context: str) -> bytes | None: ...


class SystemKeyWrapper:
    """系统凭据通道（DPAPI / Keychain），带 context 域分离。"""

    @property
    def kind(self) -> str:
        if sys.platform == "win32":
            return "dpapi-user-machine"
        if sys.platform == "darwin":
            return "keychain-login"
        return "unavailable"

    def available(self) -> bool:
        return secrets.is_available()

    def wrap(self, secret: bytes, *, context: str) -> str:
        return secrets.protect_bytes(secret, context=context)

    def unwrap(self, token: str, *, context: str) -> bytes | None:
        return secrets.unprotect_bytes(token, context=context)


class NullKeyWrapper:
    """没有系统通道时的显式空实现：绝不把密钥写成明文。"""

    kind = "null"

    def available(self) -> bool:
        return False

    def wrap(self, secret: bytes, *, context: str) -> str:
        del secret, context
        return ""

    def unwrap(self, token: str, *, context: str) -> bytes | None:
        del token, context
        return None


class TpmKeyWrapper:
    """Windows TPM/CNG 的预留位；本轮不实现。"""

    kind = "tpm-unavailable"

    def available(self) -> bool:
        return False

    def wrap(self, secret: bytes, *, context: str) -> str:
        del secret, context
        return ""

    def unwrap(self, token: str, *, context: str) -> bytes | None:
        del token, context
        return None


class SecureEnclaveKeyWrapper:
    """macOS Secure Enclave 的预留位；本轮不实现。"""

    kind = "secure-enclave-unavailable"

    def available(self) -> bool:
        return False

    def wrap(self, secret: bytes, *, context: str) -> str:
        del secret, context
        return ""

    def unwrap(self, token: str, *, context: str) -> bytes | None:
        del token, context
        return None


def current_wrapper(*, require_hardware: bool = False) -> KeyWrapper:
    """返回系统封装器；要求硬件但不可用时抛异常，**不静默降级**。"""
    if require_hardware:
        raise KeyWrapperUnavailable("当前版本没有可用的 TPM/Secure Enclave 封装实现。")
    wrapper = SystemKeyWrapper()
    return wrapper if wrapper.available() else NullKeyWrapper()


def probe() -> tuple[str, bool]:
    """给设置页显示一行状态：``(kind, available)``。"""
    wrapper = current_wrapper()
    return wrapper.kind, wrapper.available()


def memory_key_path(data_dir: Path) -> Path:
    """未来数据库密钥的落点；**当前不创建**。"""
    return data_dir / "memory" / MEMORY_DB_KEY_FILENAME
