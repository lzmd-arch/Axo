"""热点话头 worker：搜索成功变成一条新闻素材，失败交给界面。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from axo.ui.news_worker import NewsWorker

pytestmark = pytest.mark.gui

NOW = datetime(2026, 10, 18, 15, 0)


class FakeWeb:
    def __init__(self, hits: list[Any] | None = None, *, boom: bool = False) -> None:
        self.hits = hits or []
        self.boom = boom
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, count: int = 8) -> list[Any]:
        self.queries.append((query, count))
        if self.boom:
            raise RuntimeError("搜不到")
        return self.hits


def run(web: FakeWeb, query: str = "今天的热点新闻") -> tuple[list[Any], list[str]]:
    worker = NewsWorker(web, query, now=NOW)
    done: list[Any] = []
    failed: list[str] = []
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()
    return done, failed


def test_a_hit_becomes_a_news_material() -> None:
    web = FakeWeb([SimpleNamespace(title="头条", url="https://example.com", summary="摘要")])

    done, failed = run(web)

    assert failed == []
    assert len(done) == 1
    assert done[0].kind == "news"
    assert web.queries == [("今天的热点新闻", 5)]


def test_an_empty_result_is_none_not_an_error() -> None:
    done, failed = run(FakeWeb([]))

    assert failed == []
    assert done == [None]


def test_a_failure_is_reported_not_raised() -> None:
    done, failed = run(FakeWeb(boom=True))

    assert done == []
    assert failed and "搜不到" in failed[0]
