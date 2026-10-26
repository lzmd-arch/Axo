"""测试用最小 LSP 服务器：只实现本仓客户端用到的那几个方法。"""

from __future__ import annotations

import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii", "replace").partition(":")
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def send(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


def main() -> int:
    while True:
        message = read_message()
        if message is None:
            return 0
        method = message.get("method", "")
        request_id = message.get("id")
        params = message.get("params", {})
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "capabilities": {
                            "definitionProvider": True,
                            "referencesProvider": True,
                            "documentSymbolProvider": True,
                        }
                    },
                }
            )
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": request_id, "result": None})
        elif method == "exit":
            return 0
        elif method == "textDocument/didOpen":
            uri = params.get("textDocument", {}).get("uri", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "diagnostics": [
                            {
                                "range": {
                                    "start": {"line": 0, "character": 0},
                                    "end": {"line": 0, "character": 3},
                                },
                                "severity": 2,
                                "source": "fake-lsp",
                                "message": "这是测试诊断",
                            }
                        ],
                    },
                }
            )
        elif method == "textDocument/definition":
            uri = params.get("textDocument", {}).get("uri", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "uri": uri,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 3},
                        },
                    },
                }
            )
        elif method == "textDocument/references":
            uri = params.get("textDocument", {}).get("uri", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 3},
                            },
                        },
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 1, "character": 0},
                                "end": {"line": 1, "character": 3},
                            },
                        },
                    ],
                }
            )
        elif method == "textDocument/documentSymbol":
            uri = params.get("textDocument", {}).get("uri", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "name": "fake_function",
                            "kind": 12,
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 3},
                            },
                        }
                    ],
                }
            )
        elif request_id is not None:
            send({"jsonrpc": "2.0", "id": request_id, "result": None})


if __name__ == "__main__":
    raise SystemExit(main())
