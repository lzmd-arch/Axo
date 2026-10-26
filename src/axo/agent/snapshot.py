"""工作区轻量快照：路径 + 大小 + mtime 的摘要，**不读文件内容**。

测试会话用它判断"上次测试前到现在文件动过没有"，仓库地图用它做缓存键。两处要的是
同一条性质：**快、不碰内容、噪声目录不参与**。放在这里而不是其中一边，是为了避免
`repo_map` 反向 import `test_session`（那会把"符号索引"和"测试历史"绑在一起）。

**截断是明确的事实，不是"没变化"。** 文件太多时只扫前 ``_MAX_SNAPSHOT_FILES`` 个，
``truncated=True``；调用方据此拒绝"没有变化"的断言。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Snapshot", "snapshot_tree"]

_NOISE_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "node_modules",
        "target",
        "out",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".idea",
        ".vscode",
    }
)
_MAX_SNAPSHOT_FILES = 4000


@dataclass(frozen=True)
class Snapshot:
    """工作区在某一刻的轻量指纹。

    ``digest`` 为 ``None`` 表示这个目录没扫成（不存在/没权限）；此时对比方不应该说
    "没有变化"。``truncated`` 表示文件太多只扫了前 ``_MAX_SNAPSHOT_FILES`` 个。
    """

    digest: str | None
    files: int
    truncated: bool = False


def snapshot_tree(root: Path, *, max_files: int = _MAX_SNAPSHOT_FILES) -> Snapshot:
    """扫一遍目录，返回大小/mtime 摘要。**不读文件内容。**"""
    try:
        if not root.is_dir():
            return Snapshot(None, 0)
    except OSError:
        return Snapshot(None, 0)
    hasher = hashlib.sha256()
    count = 0
    truncated = False
    try:
        walker = os.walk(root, followlinks=False)
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = sorted(
                name for name in dirnames if name not in _NOISE_DIRS and not name.startswith(".")
            )
            for name in sorted(filenames):
                path = Path(dirpath) / name
                try:
                    stat = path.stat()
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                hasher.update(f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
                count += 1
                if count >= max_files:
                    truncated = True
                    break
            if truncated:
                break
    except OSError:
        return Snapshot(None, count, truncated)
    return Snapshot(hasher.hexdigest(), count, truncated)
