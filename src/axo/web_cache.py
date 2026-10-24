"""联网抓取的**原文缓存 + 台账**：只追加，不进记忆库。

## 为什么工具自己不写记忆

计划书/纪要的边界：**工具面负责获取，daemon 负责入库**。所以 ``fetch_url`` /
``web_search`` 只做三件事——抓、把原文按内容哈希落盘、往台账追加一行。它们**不碰**
:class:`memcore.store.MemoryStore`。入库（摘要、去重、强化已有边）是 daemon 巩固路径
的事；台账就是那条路径的输入。

## 为什么按内容哈希存

同一个 URL 的正文会变，同一个正文也会出现在不同 URL 上。按内容哈希命名把两件事
分开：**对象是"这份字节"**，**台账是"谁在什么时候拿到了它"**。重复抓到的内容因此
不产生第二个对象，台账里记 ``deduped=True``——那正是 daemon 判断"这条记忆是不是
同一份材料"的依据。

## 为什么不拿缓存当读缓存

``fetch_url`` **每次都真去抓**。缓存是审计与去重用的，不是"省一次网络"：用户问
"现在的价格"，拿一份三天前的缓存在语义上就是错的，而这种错误没有症状。

本模块**只用标准库**，可以在没有 httpx 的地方 import。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CacheEntry", "FetchCache"]


@dataclass(frozen=True)
class CacheEntry:
    """一次抓取/搜索的落盘结果。

    Attributes:
        sha256: 内容的 SHA-256（十六进制，全 64 位）。
        path: 对象文件路径。
        at: 台账时刻（Unix 秒）。
        deduped: 这个内容之前已经在对象库里了（**不是**错误，是 daemon 的去重信号）。
    """

    sha256: str
    path: Path
    at: float
    deduped: bool


class FetchCache:
    """``<root>/objects/<sha256>.txt`` + ``<root>/ledger.jsonl``。

    Args:
        root: 缓存根目录（一般是 ``axo.paths.fetch_cache_dir()``）。**不创建**到第一次
            真正写入为止——没用过联网的人不该被凭空建一个空目录。
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    # ------------------------------------------------------------ 写入

    def record_fetch(
        self, *, url: str, title: str, content: str, at: float | None = None
    ) -> CacheEntry:
        """落一份网页正文，并往台账追加一行。"""
        return self._record(
            kind="fetch",
            content=content,
            url=url,
            title=title,
            query="",
            at=at,
        )

    def record_search(self, *, query: str, hits: list[Any], at: float | None = None) -> CacheEntry:
        """落一份搜索响应（结果的 JSON），并往台账追加一行。

        存 JSON 而不是只存摘要：搜索结果里的标题/片段以后要复核"当时它看到了什么"，
        而那与网页正文不是同一份东西。
        """
        payload = [
            {
                "title": getattr(hit, "title", ""),
                "url": getattr(hit, "url", ""),
                "snippet": getattr(hit, "snippet", ""),
                "summary": getattr(hit, "summary", ""),
                "date": getattr(hit, "date", ""),
            }
            for hit in hits
        ]
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return self._record(
            kind="search",
            content=content,
            url="",
            title="",
            query=query,
            at=at,
        )

    def _record(
        self,
        *,
        kind: str,
        content: str,
        url: str,
        title: str,
        query: str,
        at: float | None,
    ) -> CacheEntry:
        moment = float(time.time() if at is None else at)
        blob = content.encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        path = self.objects_dir / f"{digest}.txt"
        deduped = path.is_file()
        if not deduped:
            self.objects_dir.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".txt.tmp")
            temp.write_bytes(blob)
            temp.replace(path)
        self.root.mkdir(parents=True, exist_ok=True)
        line = {
            "at": moment,
            "kind": kind,
            "sha256": digest,
            "url": url,
            "title": title,
            "query": query,
            "bytes": len(blob),
            "deduped": deduped,
        }
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return CacheEntry(sha256=digest, path=path, at=moment, deduped=deduped)

    def latest_for_url(self, url: str) -> dict[str, Any] | None:
        """这个 URL 最近一次抓取的台账行；没有就返回 ``None``。

        **只认 fetch**：搜索结果（``kind="search"``）不产生可入库的原文。坏行跳过，
        与事件通道同一条纪律——一行坏了不该让整个台账读不出来。
        """
        target = url.strip()
        if not target or not self.ledger_path.is_file():
            return None
        latest: dict[str, Any] | None = None
        with self.ledger_path.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(line, dict):
                    continue
                if line.get("kind") == "fetch" and str(line.get("url", "")).strip() == target:
                    latest = line
        return latest

    # ------------------------------------------------------------ 读取

    def read(self, sha256: str) -> str:
        """按哈希取回原文。**给审计/复核用**，工具不拿它当读缓存。"""
        return (self.objects_dir / f"{sha256}.txt").read_text(encoding="utf-8")
