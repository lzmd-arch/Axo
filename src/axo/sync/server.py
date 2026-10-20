"""TLS 传输层：一个**默认拒绝**的最小 HTTP/1.1 服务器。

不用 ``http.server`` 的解析器：我们要在 accept 后、TLS 前做源地址白名单，
要对请求行/请求头/请求体设硬上限，还要拒绝 chunked、重复 Content-Length 和
任何不在白名单里的方法。``socketserver`` + 自己读三行比在 ``http.server``
上打补丁更容易把边界说清楚。

每个连接只服务一个请求（``Connection: close``），没有 keep-alive、没有流水线，
也没有静态文件。
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import socketserver
import ssl
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .certs import CertInfo
from .protocol import MAX_BODY_BYTES, PROTOCOL_VERSION
from .service import SyncRequest, SyncResponse, SyncService

__all__ = ["SyncServer", "make_tls_context", "source_allowed"]

_LOG = logging.getLogger(__name__)
_MAX_LINE = 8_192
_MAX_HEADERS = 16_384
_IO_TIMEOUT_S = 10.0
_HANDSHAKE_TIMEOUT_S = 5.0
_MAX_CONNECTIONS = 8

_REASONS = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    411: "Length Required",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
    505: "HTTP Version Not Supported",
}


def make_tls_context(cert: CertInfo) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.options |= ssl.OP_NO_COMPRESSION
    context.load_cert_chain(str(cert.cert_path), str(cert.key_path), password=cert.passphrase)
    return context


def source_allowed(ip: str, *, lan_cidrs: tuple[str, ...], tailscale: bool) -> bool:
    """应用层防火墙的第一道：只接受回环、指定局域网、Tailscale 网段。"""
    import ipaddress

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    for cidr in lan_cidrs:
        try:
            if address in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    if tailscale:
        for cidr in ("100.64.0.0/10", "fd7a:115c:a1e0::/48"):
            if address in ipaddress.ip_network(cidr):
                return True
    return False


class SyncServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        service: SyncService,
        *,
        tls_context: ssl.SSLContext,
        allow_source: Callable[[str], bool],
        max_connections: int = _MAX_CONNECTIONS,
    ) -> None:
        self.service = service
        self.tls_context = tls_context
        self.allow_source = allow_source
        self._slots = threading.BoundedSemaphore(max(1, max_connections))
        self._closed = False
        super().__init__(address, SyncHandler)

    def get_request(self) -> tuple[ssl.SSLSocket, tuple[str, int]]:
        while not self._closed:
            raw, address = self.socket.accept()
            if not self.allow_source(address[0]):
                _close_quietly(raw)
                continue
            if not self._slots.acquire(blocking=False):
                _close_quietly(raw)
                continue
            try:
                raw.settimeout(_HANDSHAKE_TIMEOUT_S)
                tls = self.tls_context.wrap_socket(raw, server_side=True)
                tls.settimeout(_IO_TIMEOUT_S)
                return tls, address
            except (ssl.SSLError, OSError):
                _close_quietly(raw)
                self._slots.release()
        raise OSError("server closed")

    def shutdown_request(self, request: socket.socket) -> None:
        try:
            super().shutdown_request(request)
        finally:
            with contextlib.suppress(ValueError):
                self._slots.release()

    def close(self) -> None:
        self._closed = True
        super().close()

    def handle_error(self, _request: Any, client_address: Any) -> None:
        # 解析层已经尽量把错误变成响应；这里只记录类型，不打印请求内容。
        _LOG.debug("sync connection error from %s", client_address)


class SyncHandler(socketserver.StreamRequestHandler):
    server: SyncServer

    def handle(self) -> None:
        try:
            self.connection.settimeout(_IO_TIMEOUT_S)
            request = self._read_request()
            if isinstance(request, SyncResponse):
                self._write(request)
                return
            response = self.server.service.dispatch(request)
            self._write(response)
        except (OSError, ssl.SSLError):
            return
        except Exception:  # noqa: BLE001 —— 连接线程不能把异常泄给进程
            _LOG.exception("sync handler failed")
            self._write(SyncResponse(500, {"ok": False, "error": "服务器内部错误"}))

    def _read_request(self) -> SyncRequest | SyncResponse:
        raw_line = self.rfile.readline(_MAX_LINE + 1)
        if not raw_line or len(raw_line) > _MAX_LINE or not raw_line.endswith(b"\n"):
            return _error(431, "请求行过长")
        try:
            line = raw_line.decode("ascii").rstrip("\r\n")
            method, target, version = line.split(" ")
        except (UnicodeDecodeError, ValueError):
            return _error(400, "请求行不合法")
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            return _error(405, "方法不允许")
        if version not in {"HTTP/1.1", "HTTP/1.0"}:
            return _error(505, "HTTP 版本不支持")
        if len(target) > 2_048:
            return _error(400, "URL 过长")
        parts = urlsplit(target)
        if not parts.path.startswith("/v1/"):
            return _error(404, "没有这个接口")
        query = {
            key: values[-1] for key, values in parse_qs(parts.query, keep_blank_values=True).items()
        }

        headers: dict[str, str] = {}
        total = len(raw_line)
        while True:
            raw_header = self.rfile.readline(_MAX_LINE + 1)
            if not raw_header:
                return _error(400, "请求头不完整")
            total += len(raw_header)
            if total > _MAX_HEADERS or len(raw_header) > _MAX_LINE:
                return _error(431, "请求头过大")
            if raw_header in (b"\r\n", b"\n"):
                break
            try:
                name, value = raw_header.decode("latin-1").split(":", 1)
            except (UnicodeDecodeError, ValueError):
                return _error(400, "请求头不合法")
            name = name.strip().lower()
            value = value.strip()
            if not name or any(ord(char) < 0x20 for char in value):
                return _error(400, "请求头不合法")
            if name in headers:
                if name in {"content-length", "transfer-encoding"}:
                    return _error(400, "重复的请求头")
                headers[name] = headers[name] + ", " + value
            else:
                headers[name] = value
        if "transfer-encoding" in headers:
            return _error(400, "不支持 chunked")
        length_text = headers.get("content-length", "0")
        try:
            length = int(length_text)
        except ValueError:
            return _error(400, "Content-Length 不合法")
        if length < 0 or length > MAX_BODY_BYTES:
            return _error(413, "请求体过大")
        if method == "GET" and length:
            return _error(400, "GET 不应带请求体")
        if method == "POST":
            content_type = headers.get("content-type", "").lower()
            if not content_type.startswith("application/json"):
                return _error(400, "只接受 application/json")
            if "content-length" not in headers:
                return _error(411, "缺少 Content-Length")
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            return _error(400, "请求体不完整")
        return SyncRequest(
            method=method,
            path=parts.path,
            query=query,
            headers=headers,
            body=body,
            source_ip=str(self.client_address[0]),
            source_port=int(self.client_address[1]) if len(self.client_address) > 1 else 0,
        )

    def _write(self, response: SyncResponse) -> None:
        payload = json.dumps(response.body, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        reason = _REASONS.get(response.status, "Error")
        head = (
            f"HTTP/1.1 {response.status} {reason}\r\n"
            f"Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Cache-Control: no-store\r\n"
            f"X-Content-Type-Options: nosniff\r\n"
            f"X-Axo-Version: {PROTOCOL_VERSION}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("ascii")
        try:
            self.wfile.write(head + payload)
            self.wfile.flush()
        except OSError:
            pass


def _error(status: int, message: str) -> SyncResponse:
    return SyncResponse(status, {"ok": False, "error": message})


def _close_quietly(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.close()
