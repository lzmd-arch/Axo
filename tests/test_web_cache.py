"""抓取缓存与台账：按内容哈希去重、只追加、可复核。"""

from __future__ import annotations

import json
from pathlib import Path

from axo.web_cache import FetchCache


def _hits() -> list[object]:
    class _Hit:
        title = "标题"
        url = "https://example.com/1"
        snippet = "片段"
        summary = "摘要"
        date = "2026-10-24"

    return [_Hit()]


def test_record_fetch_writes_one_object_and_one_ledger_line(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fc")

    entry = cache.record_fetch(url="https://example.com", title="标题", content="正文", at=1.0)

    assert entry.path.is_file() and cache.read(entry.sha256) == "正文"
    line = json.loads((tmp_path / "fc" / "ledger.jsonl").read_text(encoding="utf-8"))
    assert line["kind"] == "fetch" and line["sha256"] == entry.sha256 and line["at"] == 1.0


def test_the_same_content_dedupes_to_one_object(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fc")

    first = cache.record_fetch(url="https://a", title="", content="同一段", at=1.0)
    second = cache.record_fetch(url="https://b", title="", content="同一段", at=2.0)

    assert first.sha256 == second.sha256
    assert second.deduped is True
    assert len(list((tmp_path / "fc" / "objects").glob("*.txt"))) == 1
    lines = (tmp_path / "fc" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["deduped"] is True


def test_record_search_stores_the_hits_as_json(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fc")

    entry = cache.record_search(query="最近的新闻", hits=_hits(), at=1.0)

    payload = json.loads(cache.read(entry.sha256))
    assert payload[0]["url"] == "https://example.com/1"
    line = json.loads((tmp_path / "fc" / "ledger.jsonl").read_text(encoding="utf-8"))
    assert line["kind"] == "search" and line["query"] == "最近的新闻"
