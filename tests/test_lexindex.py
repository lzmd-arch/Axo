"""词法倒排索引与批量取条目。"""

from __future__ import annotations

from memcore.lexindex import LexicalIndex
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier


def _store_with(*texts: str) -> MemoryStore:
    store = MemoryStore(":memory:")
    for index, text in enumerate(texts):
        store.add(
            MemoryItem(
                memory_id=f"m{index}",
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=text,
                created_at=1_700_000_000.0 + index,
                lex_sdr=encode_lexical(text),
            )
        )
    return store


def test_candidates_come_from_shared_bits_and_respect_limit() -> None:
    store = _store_with("上海自来水来自海上", "上海的路由器", "猫咪在窗台上")
    index = store.lexical_index()
    query = encode_lexical("上海")

    assert index.candidates(query, limit=10)
    assert all(memory_id in index.ids for memory_id in index.candidates(query, limit=10))
    assert len(index.candidates(query, limit=1)) == 1
    assert index.candidates(query, limit=0) == ()


def test_index_is_invalidated_by_writes() -> None:
    store = _store_with("上海自来水")
    before = store.lexical_index()
    store.add(
        MemoryItem(
            memory_id="m-new",
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content="上海的新条目",
            created_at=1_700_000_100.0,
            lex_sdr=encode_lexical("上海的新条目"),
        )
    )
    after = store.lexical_index()

    assert "m-new" not in before.ids
    assert "m-new" in after.ids
    assert store.revision > 0


def test_get_many_preserves_order_and_skips_missing() -> None:
    store = _store_with("第一条", "第二条", "第三条")

    items = store.get_many(["m2", "missing", "m0"])

    assert [item.memory_id for item in items] == ["m2", "m0"]


def test_empty_index_has_no_candidates() -> None:
    index = LexicalIndex.build([])

    assert index.ids == ()
    assert index.candidates(encode_lexical("任意文本"), limit=10) == ()
