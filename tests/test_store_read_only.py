"""`MemoryStore(read_only=True)` 的守卫：写操作必须在**引擎层**被拒。

用户 2026-10-30 明确要求：Axo 侧对记忆库只允许读，不允许任何更改。这里测的是"不可能写"，
而不是"我们不写"——后者靠自觉，前者靠 SQLite 的 `mode=ro`。
"""

from __future__ import annotations

import sqlite3

import pytest

from memcore.sdr import encode_lexical
from memcore.store import MemoryStore, StoreCorruptError
from memcore.types import MemoryItem, MemoryKind, Tier


def _item(memory_id: str, text: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=MemoryKind.FACT,
        tier=Tier.WORKING,
        content=text,
        lex_sdr=encode_lexical(text),
    )


@pytest.fixture()
def seeded_db(tmp_path):
    path = tmp_path / "mem.db"
    store = MemoryStore(path)
    store.add(_item("m1", "spiking working memory"))
    store.close()
    return path


def test_read_only_open_can_read(seeded_db):
    store = MemoryStore(seeded_db, read_only=True)
    try:
        assert store.read_only is True
        assert store.count() == 1
    finally:
        store.close()


def test_every_write_path_is_refused_by_sqlite(seeded_db):
    store = MemoryStore(seeded_db, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.add(_item("m2", "second"))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.update(_item("m1", "changed"))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.delete("m1")
    finally:
        store.close()
    # 关掉之后用可写方式再开：内容必须还是原来那一条
    check = MemoryStore(seeded_db)
    try:
        assert check.count() == 1
        assert check.get("m1").content == "spiking working memory"
    finally:
        check.close()


def test_missing_file_is_an_error_not_a_new_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        MemoryStore(tmp_path / "nope.db", read_only=True)
    assert not (tmp_path / "nope.db").exists()


def test_in_memory_store_cannot_be_read_only():
    with pytest.raises(ValueError, match="memory"):
        MemoryStore(":memory:", read_only=True)


def test_read_only_does_not_create_schema(tmp_path):
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    with pytest.raises(StoreCorruptError, match="只读模式不会建库"):
        MemoryStore(empty, read_only=True)
