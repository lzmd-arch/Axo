"""同步层的存储原语：**只增不改**。

这几条测试守的是同一件事：同步路径上没有任何一个入口能改掉已有记忆的正文。
`MemoryStore.add` 是 UPSERT；如果同步误用它，手机传一个已存在的 id 就会静默
覆盖桌面记忆。同步只允许走 `add_if_absent`。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from axo.memory.policy import MemoryDraft
from axo.memory.writer import MemoryWriter
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def make_item(text: str = "一条记忆", **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.WORKING,
        "content": text,
        "taint": Taint.USER,
        "weight": 0.5,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


def test_add_if_absent_inserts_once(store: MemoryStore) -> None:
    item = make_item("第一次写入")
    inserted, created = store.add_if_absent(item)
    assert created is True
    assert inserted.memory_id == item.memory_id
    assert store.count() == 1


def test_add_if_absent_never_overwrites_existing(store: MemoryStore) -> None:
    original = make_item("原来的正文", weight=0.9)
    store.add_if_absent(original)

    intruder = make_item(
        "手机想改成的正文",
        memory_id=original.memory_id,
        weight=0.01,
        tier=Tier.COLD,
    )
    existing, created = store.add_if_absent(intruder)

    assert created is False
    assert existing.content == "原来的正文"
    assert existing.weight == pytest.approx(0.9)
    assert existing.tier is Tier.WORKING
    assert store.count() == 1
    # 库里的行也必须是原文；这里再读一次，避免只信返回值。
    stored = store.get(original.memory_id)
    assert stored is not None and stored.content == "原来的正文"


def test_update_bookkeeping_cannot_touch_content(store: MemoryStore) -> None:
    item = make_item("正文不许动")
    store.add_if_absent(item)

    assert store.update_bookkeeping(item.memory_id, tier=Tier.LONGTERM, pinned=True)
    stored = store.get(item.memory_id)
    assert stored is not None
    assert stored.content == "正文不许动"
    assert stored.tier is Tier.LONGTERM
    assert stored.pinned is True


def test_update_bookkeeping_rejects_non_tier(store: MemoryStore) -> None:
    item = make_item()
    store.add_if_absent(item)
    with pytest.raises(ValueError):
        store.update_bookkeeping(item.memory_id, tier="longterm")  # type: ignore[arg-type]


def test_change_log_records_only_sync_relevant_changes(store: MemoryStore) -> None:
    store.install_change_log()
    item = make_item("变更账")
    store.add_if_absent(item)

    # 权重/访问计数衰减不是同步事件：手机上自己算自己的衰减。
    store.touch_access([item.memory_id], at=time.time())
    store.update_bookkeeping(item.memory_id, tier=Tier.RECENT)
    store.delete(item.memory_id)

    events = store.changes_since(0, limit=50)
    assert [event.op for event in events] == ["insert", "update", "delete"]
    assert all(event.memory_id == item.memory_id for event in events)


def test_change_log_records_content_edit_for_audit(store: MemoryStore) -> None:
    store.install_change_log()
    item = make_item("旧正文")
    store.add_if_absent(item)
    store.transform(
        item.memory_id,
        lambda old: MemoryItem(
            memory_id=old.memory_id,
            kind=old.kind,
            tier=old.tier,
            content="新正文",
            taint=old.taint,
            created_at=old.created_at,
            weight=old.weight,
            lex_sdr=encode_lexical("新正文"),
            session_id=old.session_id,
            source_ref=old.source_ref,
        ),
    )
    events = store.changes_since(0, limit=50)
    assert [event.op for event in events] == ["insert", "content_edited"]


def test_change_log_limit(store: MemoryStore) -> None:
    store.install_change_log()
    for index in range(5):
        store.add_if_absent(make_item(f"第 {index} 条"))
    assert len(store.changes_since(0, limit=2)) == 2
    assert len(store.changes_since(0, limit=99)) == 5


def test_record_if_absent_uses_client_id_and_keeps_old_created_at(store: MemoryStore) -> None:
    writer = MemoryWriter(store)
    before = time.time()
    memory_id = new_ulid()
    old_created = before - 86400 * 30
    item, created = writer.record_if_absent(
        MemoryDraft(content="从手机同步来的旧记忆"),
        session_id="mobile/dev",
        source_ref="mobile://dev/op",
        memory_id=memory_id,
        created_at=old_created,
    )
    assert created is True
    assert item is not None
    assert item.memory_id == memory_id
    assert item.created_at == pytest.approx(old_created)
    # 旧 created_at 不能让导入时立刻重度衰减。
    assert item.last_access_ts >= before
    assert item.last_decay_ts >= before


def test_record_if_absent_duplicate_returns_existing(store: MemoryStore) -> None:
    writer = MemoryWriter(store)
    first, created = writer.record_if_absent(
        MemoryDraft(content="只写一次"),
        session_id="mobile/dev",
        source_ref="mobile://dev/op1",
    )
    assert created and first is not None
    second, created_again = writer.record_if_absent(
        MemoryDraft(content="不同的正文"),
        session_id="mobile/dev",
        source_ref="mobile://dev/op2",
        memory_id=first.memory_id,
    )
    assert created_again is False
    assert second is not None and second.content == "只写一次"
    assert store.count() == 1


def test_apply_authority_updates_content_but_keeps_local_bookkeeping(store: MemoryStore) -> None:
    original = make_item("桌面旧正文", weight=0.9, access_count=7)
    store.add_if_absent(original)
    revised = make_item(
        "桌面新正文",
        memory_id=original.memory_id,
        weight=0.1,
        access_count=99,
        tier=Tier.LONGTERM,
    )
    store.apply_authority(revised)
    stored = store.get(original.memory_id)
    assert stored is not None
    assert stored.content == "桌面新正文"
    assert stored.tier is Tier.LONGTERM
    # 手机本地的记账不被桌面覆盖。
    assert stored.weight == pytest.approx(0.9)
    assert stored.access_count == 7
