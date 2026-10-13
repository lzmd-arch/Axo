"""记忆库的持久化。

这个文件里最重要的三条，都是**静默数据损坏**的守门人——它们的失效方式不会报错，
只会让数据慢慢变得不对：

* :func:`test_meta_round_trips_verbatim` ——计划书 §3.3 的**接缝 5**。P3 的任务
  记忆把全文与工具轨迹塞在 ``meta`` 里，靠的就是存储层对它不做任何解释。
* :func:`test_add_does_not_cascade_delete_embeddings` ——``INSERT OR REPLACE``
  在语义上是先 DELETE 再 INSERT，会连带触发向量表的 CASCADE。这一条失败意味着
  "重写一条记忆的正文会顺手删掉它的向量"。
* :func:`test_partial_sdr_is_corruption` ——半个 SDR 会以"有编码"的样子参与相似度
  计算。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memcore.errors import (
    DimensionMismatchError,
    SpaceNotFoundError,
    StoreCorruptError,
    StoreError,
)
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import SDR, MemoryItem, MemoryKind, Taint, Tier

SPACE = make_space_id("test", "fake-embed", 8)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


@pytest.fixture
def embed_store(store: MemoryStore) -> MemoryStore:
    store.register_space(Space(space_id=SPACE, dim=8, model_ref="fake"))
    return store


def make_item(text: str = "一条记忆", **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 0.7,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


# ── 条目往返 ────────────────────────────────────────────────────────────────


def test_item_round_trips_all_fields(store: MemoryStore) -> None:
    original = make_item(
        "完整的往返：所有字段都该原样回来。",
        taint=Taint.MODEL,
        tier=Tier.RECENT,
        access_count=3,
        reinforce_count=2,
        pinned=True,
        session_id="20261010-120000-abc123",
        source_ref="turn://01J/2",
    )
    store.add(original)
    loaded = store.get(original.memory_id)

    assert loaded is not None
    assert loaded.kind is original.kind
    assert loaded.tier is original.tier
    assert loaded.taint is original.taint
    assert loaded.content == original.content
    assert loaded.weight == pytest.approx(original.weight)
    assert loaded.access_count == original.access_count
    assert loaded.reinforce_count == original.reinforce_count
    assert loaded.pinned is original.pinned
    assert loaded.session_id == original.session_id
    assert loaded.source_ref == original.source_ref
    assert loaded.lex_sdr == original.lex_sdr


def test_meta_round_trips_verbatim(store: MemoryStore) -> None:
    """**接缝 5**：meta 原样往返，存储层不做任何 schema 校验。

    P3 的任务记忆会把"摘要层 + 全文 + 工具轨迹"塞进 meta。若存储层对它的结构
    加了假设，P3 就必须回头改存储层——而存储层的改动是静默数据损坏的高发区。

    这里刻意用一个**异形**的 meta（嵌套、混合类型、中文键、超出任何预期 schema
    的字段）来测：能原样回来，才说明存储层真的"不认识"它。
    """
    meta = {
        "full_text": "任务全文，" * 50,
        "tool_trace": [
            {"tool": "read_file", "args": {"path": "README.md"}, "ok": True, "ms": 12.5},
            {"tool": "run_command", "args": {"command": ["git", "status"]}, "ok": False},
        ],
        "轮次": 3,
        "interrupted": False,
        "嵌套": {"深": {"更深": [1, 2.5, None, True, "字符串"]}},
    }
    item = make_item("带异形 meta 的记忆", meta=meta)
    store.add(item)

    loaded = store.get(item.memory_id)
    assert loaded is not None
    assert loaded.meta == meta, "meta 没有原样往返——存储层对它做了假设"


def test_meta_rejects_non_json_payload(store: MemoryStore) -> None:
    """不可 JSON 化的 meta 当场报错，而不是在库里留下一个坏值。"""
    item = make_item("bad meta", meta={"vec": np.zeros(3)})
    with pytest.raises(StoreError, match="JSON"):
        store.add(item)


def test_get_missing_returns_none(store: MemoryStore) -> None:
    assert store.get(new_ulid()) is None


def test_update_missing_raises(store: MemoryStore) -> None:
    """``update`` 与 ``add`` 分开，就是为了让"手滑写错 id"当场报错。"""
    with pytest.raises(StoreError, match="不存在"):
        store.update(make_item())


def test_count_and_iter(store: MemoryStore) -> None:
    items = [make_item(f"第 {index} 条") for index in range(5)]
    for item in items:
        store.add(item)
    assert store.count() == 5
    assert [item.memory_id for item in store.iter_items()] == sorted(i.memory_id for i in items)


def test_delete_returns_whether_it_hit(store: MemoryStore) -> None:
    item = make_item()
    store.add(item)
    assert store.delete(item.memory_id) is True
    assert store.delete(item.memory_id) is False


# ── 向量 ────────────────────────────────────────────────────────────────────


def test_add_does_not_cascade_delete_embeddings(embed_store: MemoryStore) -> None:
    """**重写一条记忆的正文不能带走它的向量。**

    ``INSERT OR REPLACE`` 在 SQLite 里是先 DELETE 再 INSERT，而向量表上挂着
    ``ON DELETE CASCADE``。用它的写法，下面第二次 ``add`` 会静默删掉向量——
    没有任何报错，只表现为"召回质量莫名其妙下降了"。
    """
    item = make_item("先写条目，再补向量。")
    embed_store.add(item)
    embed_store.set_embedding(item.memory_id, SPACE, np.arange(8, dtype=np.float32))

    embed_store.add(item.copy(content="改写了正文。"))
    assert embed_store.get_embedding(item.memory_id, SPACE) is not None


def test_delete_cascades_embeddings(embed_store: MemoryStore) -> None:
    item = make_item()
    embed_store.add(item)
    embed_store.set_embedding(item.memory_id, SPACE, np.ones(8, dtype=np.float32))
    assert embed_store.count_embeddings(SPACE) == 1

    embed_store.delete(item.memory_id)
    assert embed_store.count_embeddings(SPACE) == 0, (
        "外键 CASCADE 没有生效——多半是 PRAGMA foreign_keys 没开，而它默认是关的。"
    )


def test_set_embedding_needs_the_item_first(embed_store: MemoryStore) -> None:
    with pytest.raises(StoreError, match="不存在"):
        embed_store.set_embedding(new_ulid(), SPACE, np.ones(8, dtype=np.float32))


def test_set_embedding_checks_dimension_against_registry(embed_store: MemoryStore) -> None:
    """维度以**登记表**为准，不是以传进来的数组为准。"""
    item = make_item()
    embed_store.add(item)
    with pytest.raises(DimensionMismatchError):
        embed_store.set_embedding(item.memory_id, SPACE, np.ones(4, dtype=np.float32))


def test_unregistered_space_is_refused(embed_store: MemoryStore) -> None:
    """往未登记的空间写向量会让"记忆独立于模型"这条承诺失去校验点。

    抛的是 ``SpaceNotFoundError`` 而不是"不匹配"：那个空间从来没登记过，
    谈不上维度对不上。**拒绝而不是自动登记**——自动登记等于让第一个写错的
    向量定义整个空间的维度。
    """
    item = make_item()
    embed_store.add(item)
    other = make_space_id("test", "另一个模型", 8)
    with pytest.raises(SpaceNotFoundError, match="未登记"):
        embed_store.set_embedding(item.memory_id, other, np.ones(8, dtype=np.float32))


def test_missing_embedding_ids(embed_store: MemoryStore) -> None:
    """ "还是空的"必须能被查出来——实时写入是先写条目后补向量的。"""
    items = [make_item(f"第 {index} 条") for index in range(4)]
    for item in items:
        embed_store.add(item)
    embed_store.set_embedding(items[0].memory_id, SPACE, np.ones(8, dtype=np.float32))

    missing = embed_store.missing_embedding_ids(SPACE)
    assert missing == [item.memory_id for item in items[1:]]
    assert embed_store.missing_embedding_ids(SPACE, limit=2) == missing[:2]


def test_embedding_matrix_is_aligned_and_empty_safe(embed_store: MemoryStore) -> None:
    ids, matrix = embed_store.embedding_matrix(SPACE)
    assert ids == []
    assert matrix.shape == (0, 8), "空空间要返回 (0, dim) 的空矩阵，调用方不必写分支"

    items = [make_item(f"第 {index} 条") for index in range(3)]
    for index, item in enumerate(items):
        embed_store.add(item)
        embed_store.set_embedding(item.memory_id, SPACE, np.full(8, index + 1, dtype=np.float32))

    ids, matrix = embed_store.embedding_matrix(SPACE)
    assert ids == sorted(item.memory_id for item in items)
    assert matrix.shape == (3, 8)
    assert matrix[0][0] == pytest.approx(1.0)


def test_embedding_matrix_returns_a_copy(embed_store: MemoryStore) -> None:
    """召回在锁外做 matmul，不能就地改写库里的字节。"""
    item = make_item()
    embed_store.add(item)
    embed_store.set_embedding(item.memory_id, SPACE, np.ones(8, dtype=np.float32))

    _, matrix = embed_store.embedding_matrix(SPACE)
    matrix[0][0] = 999.0
    again = embed_store.get_embedding(item.memory_id, SPACE)
    assert again is not None and again[0] == pytest.approx(1.0)


# ── 空间登记 ────────────────────────────────────────────────────────────────


def test_register_space_is_idempotent(store: MemoryStore) -> None:
    space = Space(space_id=SPACE, dim=8)
    store.register_space(space)
    store.register_space(space)
    assert len(store.spaces()) == 1


def test_registry_survives_reopen(tmp_path: Path) -> None:
    """空间必须在重开之后还在——否则重启一次向量就无处安放。"""
    path = tmp_path / "mem.db"
    with MemoryStore(path) as first:
        first.register_space(Space(space_id=SPACE, dim=8, model_ref="fake"))
    with MemoryStore(path) as second:
        assert second.registry.get(SPACE) is not None
        assert second.registry.require(SPACE, 8).dim == 8


# ── 衰减 ────────────────────────────────────────────────────────────────────


def test_decay_halves_at_one_half_life(store: MemoryStore) -> None:
    item = make_item(weight=1.0, tier=Tier.LONGTERM)
    store.add(item)
    at = item.created_at + 100 * 86400.0
    settled = store.apply_decay(item, at)
    assert settled.weight == pytest.approx(0.5, abs=0.01)
    assert settled.last_decay_ts == pytest.approx(at)

    reloaded = store.get(item.memory_id)
    assert reloaded is not None and reloaded.weight == pytest.approx(0.5, abs=0.01)


def test_small_decay_is_not_written_back(store: MemoryStore) -> None:
    """变化小于 epsilon 就不写库。否则每轮召回都会写几十行浮点末位的差异。"""
    item = make_item(weight=1.0, tier=Tier.LONGTERM)
    store.add(item)
    at = item.created_at + 0.001
    settled = store.apply_decay(item, at)
    assert settled.weight < 1.0  # 调用方拿到更精确的值

    reloaded = store.get(item.memory_id)
    assert reloaded is not None
    assert reloaded.last_decay_ts == pytest.approx(item.created_at), "不该写回"


def test_pinned_items_do_not_decay(store: MemoryStore) -> None:
    item = make_item(weight=0.9, pinned=True)
    store.add(item)
    settled = store.apply_decay(item, item.created_at + 10_000 * 86400.0)
    assert settled.weight == pytest.approx(0.9)


def test_touch_access_is_batched_and_cumulative(store: MemoryStore) -> None:
    items = [make_item(f"第 {index} 条") for index in range(3)]
    for item in items:
        store.add(item)
    store.touch_access([item.memory_id for item in items], store.get(items[0].memory_id).created_at)
    store.touch_access([items[0].memory_id], items[0].created_at + 10)

    assert store.get(items[0].memory_id).access_count == 2
    assert store.get(items[1].memory_id).access_count == 1
    store.touch_access([], 0.0)  # 空列表是合法的无操作


# ── 损坏检测 ────────────────────────────────────────────────────────────────


def _corrupt(path: Path, sql: str, params: tuple[object, ...]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(sql, params)
    connection.commit()
    connection.close()


def test_partial_sdr_is_corruption(tmp_path: Path) -> None:
    """半个 SDR 比没有 SDR 更糟——它会以"有编码"的样子参与相似度计算。"""
    path = tmp_path / "mem.db"
    item = make_item()
    with MemoryStore(path) as store:
        store.add(item)
    _corrupt(path, "UPDATE memories SET lex_w = NULL WHERE memory_id = ?", (item.memory_id,))

    with MemoryStore(path) as store, pytest.raises(StoreCorruptError, match="三列"):
        store.get(item.memory_id)


def test_unknown_tier_is_corruption(tmp_path: Path) -> None:
    """不认识的 tier 值不能静默退回默认值——那条记忆会以错误的层级参与召回。"""
    path = tmp_path / "mem.db"
    item = make_item()
    with MemoryStore(path) as store:
        store.add(item)
    _corrupt(
        path, "UPDATE memories SET tier = 'from_the_future' WHERE memory_id = ?", (item.memory_id,)
    )

    with MemoryStore(path) as store, pytest.raises(StoreCorruptError, match="from_the_future"):
        store.get(item.memory_id)


def test_broken_meta_json_is_corruption(tmp_path: Path) -> None:
    path = tmp_path / "mem.db"
    item = make_item()
    with MemoryStore(path) as store:
        store.add(item)
    _corrupt(
        path, "UPDATE memories SET meta_json = '{不是 JSON' WHERE memory_id = ?", (item.memory_id,)
    )

    with MemoryStore(path) as store, pytest.raises(StoreCorruptError, match="JSON"):
        store.get(item.memory_id)


def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    """**旧版本不许打开新库。** 旧代码不认识新列，会把它们当成不存在而静默丢数据。"""
    path = tmp_path / "mem.db"
    with MemoryStore(path):
        pass
    _corrupt(path, "PRAGMA user_version = 99", ())

    with pytest.raises(StoreCorruptError, match="99"):
        MemoryStore(path)


def test_sdr_width_mismatch_in_matrix_row(tmp_path: Path) -> None:
    """同一空间里混进不同维度的向量 = 库被写坏了。"""
    path = tmp_path / "mem.db"
    with MemoryStore(path) as store:
        store.register_space(Space(space_id=SPACE, dim=8))
        item = make_item()
        store.add(item)
        store.set_embedding(item.memory_id, SPACE, np.ones(8, dtype=np.float32))
    _corrupt(path, "UPDATE embeddings SET dim = 4 WHERE space_id = ?", (SPACE,))

    with MemoryStore(path) as store, pytest.raises(StoreCorruptError, match="4 维"):
        store.embedding_matrix(SPACE)


# ── SDR 列的存取 ────────────────────────────────────────────────────────────


def test_item_without_sdr_round_trips_as_none(store: MemoryStore) -> None:
    """三列同为 NULL 是合法的（尚未编码），不能当成损坏。"""
    item = make_item("还没有编码的记忆", lex_sdr=None)
    store.add(item)
    loaded = store.get(item.memory_id)
    assert loaded is not None and loaded.lex_sdr is None


def test_sdr_weights_survive_blob_round_trip(store: MemoryStore) -> None:
    sdr = SDR(
        idx=np.array([3, 17, 40], dtype=np.int32),
        weights=np.array([1.0, 0.5, 0.25], dtype=np.float32),
        width=64,
    )
    item = make_item("手写的 SDR", lex_sdr=sdr)
    store.add(item)
    loaded = store.get(item.memory_id)
    assert loaded is not None and loaded.lex_sdr == sdr


def test_meta_size_limit(store: MemoryStore) -> None:
    """meta 是给"随条目走的少量附加信息"用的，不是给全文用的。"""
    huge = {"full_text": "x" * (300 * 1024)}
    with pytest.raises(StoreError, match="字节"):
        store.add(make_item("meta 太大", meta=huge))


def test_meta_is_stored_as_readable_json(store: MemoryStore) -> None:
    """库里存的是可读的 JSON，不是二进制——出问题时用 sqlite3 就能看。"""
    item = make_item("可读的 meta", meta={"键": "值"})
    store.add(item)
    connection = sqlite3.connect(store.path)
    raw = connection.execute(
        "SELECT meta_json FROM memories WHERE memory_id = ?", (item.memory_id,)
    ).fetchone()[0]
    connection.close()
    assert json.loads(raw) == {"键": "值"}
    assert "值" in raw, "ensure_ascii=False：中文不该被转义成 \\uXXXX"


def test_items_created_between_is_half_open_and_unfiltered(store: MemoryStore) -> None:
    """按时间区间读：半开区间，而且不做任何语义过滤。

    「这段时间里有什么」是存储层的事实，「哪些该进上下文」是召回层的判断。存储层
    顺手把 tier/pinned 筛了，账本就说不清是谁丢的。
    """
    early = make_item("早的。", created_at=100.0)
    edge = make_item("卡在 until 上的。", created_at=200.0)
    late = make_item("晚的。", created_at=300.0)
    forgotten = make_item("已经忘了的。", created_at=150.0, tier=Tier.COLD)
    for item in (early, edge, late, forgotten):
        store.add(item)

    hits = store.items_created_between(100.0, 200.0)

    assert [item.content for item in hits] == ["早的。", "已经忘了的。"]
    assert edge.memory_id not in {item.memory_id for item in hits}, "until 是开区间"
