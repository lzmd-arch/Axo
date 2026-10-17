"""agent 的执行轨迹：本机审计用的结构化 JSONL。

**与 ``agent/trace.py`` 分工不同。** 那个进记忆的 ``meta``（8000 字上限、参与召回），
这个只落本机盘、不参与检索：它是"刚才那几轮到底跑了什么"的原始记录，出问题时用来
复盘，不是给模型读的。

按天一个文件 ``<目录>/YYYY-MM-DD.jsonl``。**写入失败只吞掉**：日志坏了不该反过来
弄坏主流程（那会让一次正常的回答因为磁盘满而失败）。

参数要脱敏：``write_file`` 的正文只留前 200 字 + sha1，剪贴板只留长度 + sha1，
URL 只留主机与路径、去掉查询串（查询串里可能有用户粘进来的东西）。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["RunLog", "redact_arguments"]

_CONTENT_PREVIEW = 200


class RunLog:
    """按天追加的 JSONL。**任何异常都不往外抛。**"""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    def write(self, record: Mapping[str, Any], *, now: float | None = None) -> None:
        if not self.enabled:
            return
        stamp = time.time() if now is None else now
        day = time.strftime("%Y-%m-%d", time.localtime(stamp))
        path = self.directory / f"{day}.jsonl"
        payload = {"ts": round(stamp, 3), **dict(record)}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            # 审计日志坏了不影响回答——见模块文档。
            return


def redact_arguments(_tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """把可能很大的/敏感的参数换成长度与摘要。"""
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if not isinstance(value, str):
            cleaned[key] = value
            continue
        if key in ("content", "new_text", "old_text", "text"):
            cleaned[key] = {
                "chars": len(value),
                "sha1": hashlib.sha1(value.encode("utf-8")).hexdigest()[:16],
                "preview": value[:_CONTENT_PREVIEW],
            }
        elif key == "url":
            cleaned[key] = value.split("?", 1)[0]
        else:
            cleaned[key] = value
    return cleaned
