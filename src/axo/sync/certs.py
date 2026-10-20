"""桌面同步服务器的 TLS 身份：自签 ECDSA 证书 + 加密私钥 + pin。

身份靠**证书 pin**，不靠 CA。证书 SAN 固定为 ``axo-sync.invalid``；手机连接
任意 LAN/Tailscale 地址时把这个名字当作 ``server_hostname``，于是主机名校验
仍然生效，而 DHCP/隧道地址变化不会让信任失效。

私钥以 PKCS#8 加密形式落盘，口令由 DPAPI（注入的 ``protect``/``unprotect``）
保护；证书本身是公开的。没有口令就拒绝启动，不偷偷生成第二份身份。
"""

from __future__ import annotations

import contextlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .protocol import b64url_decode, b64url_encode, cert_pin

__all__ = ["CertInfo", "SyncCertError", "ensure_server_cert", "reset_server_cert"]

_FIXED_SAN = "axo-sync.invalid"
_VALID_DAYS = 3650


class SyncCertError(RuntimeError):
    pass


@dataclass(frozen=True)
class CertInfo:
    cert_path: Path
    key_path: Path
    passphrase: bytes
    pin: str
    der: bytes
    cert_pem: str


def _paths(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "server.crt",
        directory / "server.key.enc",
        directory / "passphrase.dpapi",
    )


def _generate(directory: Path, protect: Callable[[str], str]) -> CertInfo:
    # 懒 import：没装 `sync` extra 的安装照样能启动主程序。
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory.mkdir(parents=True, exist_ok=True)
    passphrase = secrets.token_bytes(32)
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, _FIXED_SAN),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Axo MemoryBridge"),
        ]
    )
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=_VALID_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(_FIXED_SAN)]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(passphrase),
    )
    cert_path, key_path, phrase_path = _paths(directory)
    cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    phrase_path.write_text(protect(b64url_encode(passphrase)), encoding="utf-8", newline="\n")
    with contextlib.suppress(OSError):
        key_path.chmod(0o600)
    der = certificate.public_bytes(serialization.Encoding.DER)
    return CertInfo(
        cert_path=cert_path,
        key_path=key_path,
        passphrase=passphrase,
        pin=cert_pin(der),
        der=der,
        cert_pem=cert_pem.decode("ascii"),
    )


def ensure_server_cert(
    directory: Path | str,
    *,
    protect: Callable[[str], str],
    unprotect: Callable[[str], str],
) -> CertInfo:
    directory = Path(directory)
    cert_path, key_path, phrase_path = _paths(directory)
    if not (cert_path.is_file() and key_path.is_file() and phrase_path.is_file()):
        return _generate(directory, protect)
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        phrase = b64url_decode(unprotect(phrase_path.read_text(encoding="utf-8").strip()))
        cert_pem = cert_path.read_bytes()
        certificate = x509.load_pem_x509_certificate(cert_pem)
    except Exception as exc:  # noqa: BLE001 —— 损坏/换账户统一成"不能启动"
        raise SyncCertError(
            "同步证书或口令读不出来（可能换了 Windows 账户，或文件损坏）。"
            "请在设置里「重置信任」后重新配对。"
        ) from exc
    if len(phrase) != 32:
        raise SyncCertError("同步私钥口令长度不对；请重置信任后重新配对。")
    der = certificate.public_bytes(serialization.Encoding.DER)
    return CertInfo(
        cert_path=cert_path,
        key_path=key_path,
        passphrase=phrase,
        pin=cert_pin(der),
        der=der,
        cert_pem=cert_pem.decode("ascii"),
    )


def reset_server_cert(directory: Path | str) -> None:
    directory = Path(directory)
    for path in _paths(directory):
        with contextlib.suppress(OSError):
            path.unlink()
