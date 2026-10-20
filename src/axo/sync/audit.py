"""同步审计：追加写、hash 链、轮转。

记录的是"发生过什么"：设备、IP、方法、状态、字节数、耗时、memory_id、内容
sha256/长度。**绝不记录正文、token、配对码、查询原文。** 写失败返回 ``False``，
由服务器决定 fail closed。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from .protocol import canonical_json

__all__ = ["AuditLog"]

_MAX_BYTES = 1024 * 1024
_KEEP_FILES = 5
_GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path | str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 8192))
                lines = handle.read().decode("utf-8", "replace").strip().splitlines()
            for line in reversed(lines):
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                value = record.get("hash")
                if isinstance(value, str) and len(value) == 64:
                    return value
        except OSError:
            pass
        return _GENESIS

    def write(self, record: dict[str, Any]) -> bool:
        if not self.enabled:
            return True
        try:
            payload = dict(record)
            payload.pop("hash", None)
            payload.pop("prev_hash", None)
            payload["prev_hash"] = self._last_hash
            digest = hashlib.sha256(
                (self._last_hash + canonical_json(payload)).encode("utf-8")
            ).hexdigest()
            payload["hash"] = digest
            line = canonical_json(payload) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
                self._last_hash = digest
                self._rotate_if_needed()
            return True
        except OSError:
            return False

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.stat().st_size <= _MAX_BYTES:
                return
        except OSError:
            return
        for index in range(_KEEP_FILES - 2, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            target = self.path.with_name(f"{self.path.name}.{index + 1}")
            if source.is_file():
                with contextlib.suppress(OSError):
                    source.replace(target)
        with contextlib.suppress(OSError):
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))
