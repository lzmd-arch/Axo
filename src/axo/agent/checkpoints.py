"""文件检查点：改文件前留一份原文，可按 id 回滚。

只由工具在**已获用户批准之后**调用；检查点落在应用数据目录内（工具不能直接写那里）。
它不做版本控制，也不替代 git；它只回答"刚才那次写/补丁之前，这几个文件是什么样"。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CheckpointError", "CheckpointStore", "RestoreResult"]


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class RestoreResult:
    checkpoint_id: str
    restored: tuple[str, ...]
    deleted: tuple[str, ...]


class CheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, paths: list[Path] | tuple[Path, ...], label: str = "") -> str:
        cid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        target = self.root / cid
        files = target / "files"
        files.mkdir(parents=True)
        entries: list[dict[str, Any]] = []
        for index, raw in enumerate(sorted({Path(p).resolve() for p in paths})):
            existed = raw.is_file()
            copy = None
            if existed:
                copy = f"files/{index:03d}.bin"
                shutil.copy2(raw, target / copy)
            entries.append({"path": str(raw), "existed": existed, "file": copy})
        manifest = {"id": cid, "label": label, "created": time.time(), "files": entries}
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (self.root / "latest.txt").write_text(cid + "\n", encoding="utf-8", newline="\n")
        return cid

    def _manifest(self, cid: str | None) -> tuple[str, dict[str, Any]]:
        if cid is None:
            latest = self.root / "latest.txt"
            if not latest.is_file():
                raise CheckpointError("还没有任何检查点。")
            cid = latest.read_text(encoding="utf-8").strip()
        if not cid or "/" in cid or "\\" in cid or ".." in cid:
            raise CheckpointError("检查点 id 不合法。")
        path = self.root / cid / "manifest.json"
        if not path.is_file():
            raise CheckpointError(f"找不到检查点 {cid}。")
        return cid, json.loads(path.read_text(encoding="utf-8"))

    def restore(self, cid: str | None = None, *, roots: tuple[Path, ...] = ()) -> RestoreResult:
        cid, manifest = self._manifest(cid)
        if not roots:
            raise CheckpointError("回滚需要至少一个可写目录。")
        allowed = tuple(Path(r).expanduser().resolve() for r in roots)
        restored: list[str] = []
        deleted: list[str] = []
        for entry in manifest.get("files", []):
            raw = Path(entry.get("path", "")).expanduser().resolve()
            if not any(raw.is_relative_to(root) for root in allowed):
                raise CheckpointError(f"检查点里的路径不在可写目录：{raw}")
            if entry.get("existed"):
                copy = self.root / cid / entry.get("file", "")
                if not copy.is_file():
                    raise CheckpointError(f"检查点缺少文件内容：{raw.name}")
                raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(copy, raw)
                restored.append(str(raw))
            else:
                if raw.is_file():
                    raw.unlink()
                    deleted.append(str(raw))
        return RestoreResult(cid, tuple(restored), tuple(deleted))
