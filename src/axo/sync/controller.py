"""同步服务器的生命周期：起/停监听、配对、设备列表。

不依赖 Qt。UI 层只需要造一个 controller、在设置变化时调用 ``restart``，并在
退出路径上调用 ``stop``。监听线程是 daemon，但 ``stop`` 仍会显式 shutdown/join，
不把"进程退出时碰运气"当成生命周期。
"""

from __future__ import annotations

import contextlib
import ipaddress
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axo.memory.writer import MemoryWriter
from memcore.store import MemoryStore

from .certs import CertInfo
from .network import lan_cidrs, lan_ipv4_addresses, tailscale_ipv4_addresses, tailscale_peer_ok
from .protocol import PairingPayload
from .server import SyncServer, make_tls_context
from .service import SyncConfig, SyncService

__all__ = ["SyncController", "SyncControllerConfig"]


@dataclass(frozen=True)
class SyncControllerConfig:
    server_name: str = "Axo"
    port: int = 47_831
    hosts: tuple[str, ...] = ()
    max_devices: int = 4
    audit_enabled: bool = True
    lan_enabled: bool = True
    lan_cidrs: tuple[str, ...] = ()
    tailscale_enabled: bool = False
    tailscale_same_user_only: bool = True


class SyncController:
    def __init__(
        self,
        store: MemoryStore,
        writer: MemoryWriter,
        data_dir: Path | str,
        *,
        config: SyncControllerConfig,
        protect: Callable[[str], str],
        unprotect: Callable[[str], str],
        pair_approver: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.store = store
        self.writer = writer
        self.data_dir = Path(data_dir)
        self.config = config
        self._protect = protect
        self._unprotect = unprotect
        self._pair_approver = pair_approver
        self.service: SyncService | None = None
        self.cert: CertInfo | None = None
        self._servers: list[SyncServer] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self._started = False
        self._hosts: tuple[str, ...] = ()

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            hosts = self._resolve_hosts()
            self._hosts = hosts
            service = SyncService(
                self.store,
                self.writer,
                self.data_dir,
                config=SyncConfig(
                    server_name=self.config.server_name or "Axo",
                    port=self.config.port,
                    hosts=tuple(hosts),
                    max_devices=self.config.max_devices,
                    audit_enabled=self.config.audit_enabled,
                ),
                protect=self._protect,
                unprotect=self._unprotect,
                pair_approver=self._pair_approver,
            )
            context = make_tls_context(service.cert)
            allow_source = self._source_checker()
            servers: list[SyncServer] = []
            try:
                for host in hosts:
                    server = SyncServer(
                        (host, self.config.port),
                        service,
                        tls_context=context,
                        allow_source=allow_source,
                    )
                    servers.append(server)
            except OSError:
                for server in servers:
                    server.server_close()
                service.close()
                raise
            threads = [
                threading.Thread(
                    target=server.serve_forever,
                    kwargs={"poll_interval": 0.2},
                    daemon=True,
                    name=f"axo-sync-{server.server_address[0]}",
                )
                for server in servers
            ]
            for thread in threads:
                thread.start()
            self.service = service
            self.cert = service.cert
            self._servers = servers
            self._threads = threads
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            servers, threads, service = self._servers, self._threads, self.service
            self._servers = []
            self._threads = []
            self.service = None
            self.cert = None
        for server in servers:
            with contextlib.suppress(Exception):
                server.shutdown()
            with contextlib.suppress(Exception):
                server.server_close()
        for thread in threads:
            thread.join(timeout=3)
        if service is not None:
            service.close()

    def restart(self) -> None:
        self.stop()
        self.start()

    def pairing(self) -> PairingPayload:
        if self.service is None:
            raise RuntimeError("同步服务没有启动")
        return self.service.open_pairing()

    def pairing_svg(self) -> bytes:
        if self.service is None:
            raise RuntimeError("同步服务没有启动")
        return self.service.pairing_svg()

    def devices(self) -> list[Any]:
        return [] if self.service is None else self.service.devices.list_devices()

    def revoke(self, device_id: str) -> bool:
        return False if self.service is None else self.service.devices.revoke(device_id)

    def status(self) -> dict[str, Any]:
        if self.service is None:
            return {"running": False}
        return {
            "running": True,
            "port": self.config.port,
            "hosts": list(self._hosts or self.config.hosts),
            "devices": self.service.devices.device_count(),
            "count": self.store.count(),
            "rev": self.store.change_head(),
            "pin": self.cert.pin if self.cert else "",
        }

    def _resolve_hosts(self) -> tuple[str, ...]:
        hosts: list[str] = ["127.0.0.1"]
        if self.config.hosts:
            for host in self.config.hosts:
                if host not in hosts:
                    hosts.append(host)
            return tuple(hosts)
        if self.config.lan_enabled:
            for address in lan_ipv4_addresses():
                if address not in hosts:
                    hosts.append(address)
        if self.config.tailscale_enabled:
            for address in tailscale_ipv4_addresses():
                if address not in hosts:
                    hosts.append(address)
        return tuple(hosts)

    def _source_checker(self) -> Callable[[str], bool]:
        configured = (
            (tuple(self.config.lan_cidrs) or lan_cidrs()) if self.config.lan_enabled else ()
        )
        networks = []
        for cidr in configured:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                continue
        tailscale = self.config.tailscale_enabled
        same_user = self.config.tailscale_same_user_only

        def allowed(raw: str) -> bool:
            try:
                address = ipaddress.ip_address(raw)
            except ValueError:
                return False
            if address.is_loopback:
                return True
            if tailscale and (
                address in ipaddress.ip_network("100.64.0.0/10")
                or address in ipaddress.ip_network("fd7a:115c:a1e0::/48")
            ):
                return True if not same_user else tailscale_peer_ok(raw)
            return any(address in network for network in networks)

        return allowed
