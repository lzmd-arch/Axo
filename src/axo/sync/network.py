"""本机地址与 Tailscale 识别。应用层防火墙用它算白名单。

**不修改任何系统网络配置。** 只读：查本机网卡、问 Tailscale CLI 要状态。
CLI 不可用时返回空/False，由调用方失败关闭。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

__all__ = [
    "lan_cidrs",
    "lan_ipv4_addresses",
    "tailscale_ipv4_addresses",
    "tailscale_peer_ok",
]


def _creation_flags() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


@lru_cache(maxsize=1)
def lan_ipv4_addresses() -> tuple[str, ...]:
    """本机私有 IPv4 地址（去重、去 loopback）。探测不到返回空元组。"""
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if _private_ipv4(address) and address not in found:
                found.append(address)
    except OSError:
        pass
    # UDP connect 不发送任何包；它只是让内核选一个出口地址。
    for probe in ("192.0.2.1", "198.51.100.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((probe, 9))
            address = sock.getsockname()[0]
            if _private_ipv4(address) and address not in found:
                found.append(address)
        except OSError:
            pass
        finally:
            sock.close()
    return tuple(found)


def _private_ipv4(address: str) -> bool:
    parts = address.split(".")
    if len(parts) != 4:
        return False
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    if first == 10:
        return True
    if first == 172 and 16 <= second <= 31:
        return True
    return bool(first == 192 and second == 168)


def lan_cidrs(addresses: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """把本机地址折成 /24；用户没配白名单时用它做默认。"""
    values = lan_ipv4_addresses() if addresses is None else addresses
    return tuple(f"{address.rsplit('.', 1)[0]}.0/24" for address in values)


def _tailscale_cli() -> str | None:
    candidates = ["tailscale"]
    if sys.platform == "win32":
        candidates += [
            r"C:\Program Files\Tailscale\tailscale.exe",
            r"C:\Program Files (x86)\Tailscale\tailscale.exe",
        ]
    for candidate in candidates:
        if os.path.sep in candidate:
            if Path(candidate).is_file():
                return candidate
        else:
            for folder in os.environ.get("PATH", "").split(os.pathsep):
                path = Path(folder) / (candidate + (".exe" if sys.platform == "win32" else ""))
                if path.is_file():
                    return str(path)
    return None


@lru_cache(maxsize=1)
def tailscale_ipv4_addresses() -> tuple[str, ...]:
    cli = _tailscale_cli()
    if cli is None:
        return ()
    try:
        result = subprocess.run(
            [cli, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=_creation_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(
        line.strip() for line in result.stdout.splitlines() if line.strip().startswith("100.")
    )


_STATUS_CACHE: tuple[float, dict] = (0.0, {})


def _status() -> dict:
    global _STATUS_CACHE
    cached_at, cached = _STATUS_CACHE
    if cached and time.monotonic() - cached_at < 60:
        return cached
    cli = _tailscale_cli()
    if cli is None:
        return {}
    try:
        result = subprocess.run(
            [cli, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=_creation_flags(),
            check=False,
        )
        parsed = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    result = parsed if isinstance(parsed, dict) else {}
    _STATUS_CACHE = (time.monotonic(), result)
    return result


def tailscale_peer_ok(address: str) -> bool:
    """来源 IP 是否是**同一 tailnet 用户**的在线 peer。CLI 不可用 -> False。"""
    status = _status()
    if not status:
        return False
    self_info = status.get("Self") or {}
    self_user = self_info.get("UserID")
    peers = status.get("Peer") or {}
    if not isinstance(peers, dict):
        return False
    for peer in peers.values():
        if not isinstance(peer, dict):
            continue
        addresses = peer.get("TailscaleIPs") or []
        if address not in addresses:
            continue
        return (
            bool(peer.get("Online"))
            and not bool(peer.get("Expired"))
            and peer.get("UserID") == self_user
        )
    return False
