"""第二段重排的钩子：只改顺序，不改去留；默认路径逐位不变。"""

from __future__ import annotations

import pytest

from memcore.ids import new_ulid
from memcore.recall import RecallConfig, recall
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

SPACE = make_space_id("test", "rerank", 4)


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        opened.register_space(Space(space_id=SPACE, dim=4))
        yield opened


def add(store: MemoryStore, content: str, *, weight: float = 0.8) -> MemoryItem:
    item = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content=content,
        taint=Taint.USER,
        weight=weight,
        lex_sdr=encode_lexical(content),
    )
    store.add(item)
    return item


def _recall(store: MemoryStore, **kwargs):
    return recall(
        store,
        query_text="alpha",
        budget_tokens=10_000,
        config=RecallConfig(min_score=0.0, max_items=3),
        settle=False,
        **kwargs,
    )


def test_without_rerank_the_order_and_ledger_are_unchanged(store: MemoryStore) -> None:
    add(store, "alpha one")
    add(store, "alpha two")
    ledger = _recall(store)
    assert [item.scores.rerank for item in ledger.injected] == [0.0, 0.0]
    assert ledger.injected[0].content == "alpha one"


def test_rerank_changes_order_and_records_its_score(store: MemoryStore) -> None:
    add(store, "alpha one")
    add(store, "alpha two")

    def prefer_second(_query: str, items) -> list[float]:
        return [0.0 if item.content.endswith("one") else 1.0 for item in items]

    ledger = _recall(store, rerank=prefer_second)
    assert ledger.injected[0].content == "alpha two"
    assert ledger.injected[0].scores.rerank == 1.0


def test_rerank_sees_public_items_only(store: MemoryStore) -> None:
    add(store, "alpha one")
    seen: list[object] = []

    def hook(query: str, items) -> list[float]:
        seen.append(query)
        for item in items:
            assert hasattr(item, "memory_id") and hasattr(item, "scores")
        return [0.0 for _ in items]

    _recall(store, rerank=hook)
    assert seen == ["alpha"]


def test_a_wrong_length_is_an_error_not_a_silent_fallback(store: MemoryStore) -> None:
    add(store, "alpha one")
    add(store, "alpha two")

    def broken(_query: str, items) -> list[float]:
        return [0.0]

    with pytest.raises(ValueError, match="数量对不上"):
        _recall(store, rerank=broken)
