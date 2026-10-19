"""词法倒排索引与批量取条目。"""

from __future__ import annotations

import numpy as np

from memcore.lexindex import LexicalIndex
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import SDR, MemoryItem, MemoryKind, Tier


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
    # 新增条目先进 delta，下一次全量合并才进 ids；contains 覆盖两者。
    assert after.contains("m-new")
    assert "m-new" in after.delta_ids
    assert store.revision > 0


def test_get_many_preserves_order_and_skips_missing() -> None:
    store = _store_with("第一条", "第二条", "第三条")

    items = store.get_many(["m2", "missing", "m0"])

    assert [item.memory_id for item in items] == ["m2", "m0"]


def test_empty_index_has_no_candidates() -> None:
    index = LexicalIndex.build([])

    assert index.ids == ()
    assert index.candidates(encode_lexical("任意文本"), limit=10) == ()


def test_candidates_rank_by_jaccard_not_by_rare_bit_order() -> None:
    """公共位上高度重叠的记忆，必须排在只共享一个稀有位的记忆前面。

    旧实现按"稀有位先出现"截断：查询的位 13 只出现在 ``rare`` 里，于是
    ``limit=1`` 会先撞上它，而 ``common`` 与查询共享 3 个位、Jaccard 更高。
    """
    width = 512

    def blob(bits: list[int]) -> bytes:
        return np.asarray(bits, dtype=np.int32).tobytes()

    # 填充项也含 10/11/12（它们是公共位），但额外带 10 个独有位，Jaccard 只有
    # 3/14；这样"公共位重叠最多"的 common 仍然应该赢。
    rows = []
    for index in range(20):
        rows.append(
            (f"f{index:02d}", blob([10, 11, 12, *range(30 + index * 10, 40 + index * 10)]), width)
        )
    rows.append(("common", blob([10, 11, 12, 14]), width))
    rows.append(("rare", blob([11, 13, 15]), width))
    index = LexicalIndex.build(rows)
    query = SDR(idx=np.asarray([10, 11, 12, 13], dtype=np.int32), weights=np.ones(4), width=width)

    assert index.candidates(query, limit=1) == ("common",)
    ranked = index.candidates(query, limit=2)
    assert ranked[0] == "common"
    # rare 的 Jaccard 是 2/5，common 是 3/5；两者都比填充项高。
    assert ranked[1] == "rare"


def test_candidates_match_full_jaccard_ranking() -> None:
    width = 128

    def blob(bits: list[int]) -> bytes:
        return np.asarray(bits, dtype=np.int32).tobytes()

    rows = [
        ("a", blob([1, 2, 3, 4]), width),
        ("b", blob([1, 2, 9]), width),
        ("c", blob([4, 5, 6]), width),
        ("d", blob([1, 2, 3]), width),
        ("e", blob([7, 8]), width),
    ]
    index = LexicalIndex.build(rows)
    query = SDR(idx=np.asarray([1, 2, 3, 4], dtype=np.int32), weights=np.ones(4), width=width)

    by_id = {
        memory_id: SDR(
            idx=np.asarray(bits, dtype=np.int32), weights=np.ones(len(bits)), width=width
        )
        for memory_id, bits in (
            ("a", [1, 2, 3, 4]),
            ("b", [1, 2, 9]),
            ("c", [4, 5, 6]),
            ("d", [1, 2, 3]),
            ("e", [7, 8]),
        )
    }
    expected = sorted(
        (memory_id for memory_id in by_id if by_id[memory_id].jaccard(query) > 0.0),
        key=lambda memory_id: (-by_id[memory_id].jaccard(query), memory_id),
    )

    assert list(index.candidates(query, limit=5)) == expected


def test_warmup_worker_builds_the_index() -> None:
    from axo.ui.memory_worker import LexicalIndexWorker

    store = _store_with("第一条", "第二条", "第三条")
    worker = LexicalIndexWorker(store=store)
    seen: list[int] = []
    worker.ready.connect(seen.append)

    worker.run()

    assert seen == [3]
    assert store.lexical_index().ids


def test_build_does_not_hold_the_store_lock_and_retries_after_a_write(monkeypatch) -> None:
    """索引构建期间必须能写库；构建中新增的条目不能被缓存成"当前索引"。"""
    import memcore.store as store_module

    store = _store_with("上海自来水", "上海的路由器")
    real_build = store_module.LexicalIndex.build
    calls: list[int] = []

    def build(rows):
        items = list(rows)
        if not calls:
            # 旧实现整段持 ``_lock``：这一句会自锁到构建结束（测试会挂住）。
            # 现在它在 ``fetchall`` 之后、缓存写回之前，必须能立即完成。
            store.add(
                MemoryItem(
                    memory_id="m-during-build",
                    kind=MemoryKind.FACT,
                    tier=Tier.RECENT,
                    content="构建期间写进来的条目",
                    created_at=1_700_000_300.0,
                    lex_sdr=encode_lexical("构建期间写进来的条目"),
                )
            )
        calls.append(1)
        return real_build(items)

    monkeypatch.setattr(store_module.LexicalIndex, "build", staticmethod(build))

    index = store.lexical_index()

    assert "m-during-build" in index.ids
    assert len(calls) == 2


def test_add_updates_the_index_without_a_full_rebuild(monkeypatch) -> None:
    import memcore.store as store_module

    store = _store_with("上海自来水", "猫咪在窗台上")
    store.lexical_index()

    def fail(_rows):
        raise AssertionError("新增一条记忆不该触发整库重建")

    monkeypatch.setattr(store_module.LexicalIndex, "build", staticmethod(fail))
    store.add(
        MemoryItem(
            memory_id="m-new",
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content="新条目的正文",
            created_at=1_700_000_200.0,
            lex_sdr=encode_lexical("新条目的正文"),
        )
    )

    updated = store.lexical_index()

    assert updated.contains("m-new")
    assert "m-new" in updated.candidates(encode_lexical("新条目的正文"), limit=10)


def test_delta_merges_after_the_limit(monkeypatch) -> None:
    import memcore.store as store_module

    monkeypatch.setattr(store_module, "_LEX_DELTA_MAX", 2)
    store = _store_with("基线")
    store.lexical_index()
    for index in range(3):
        store.add(
            MemoryItem(
                memory_id=f"m-new-{index}",
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=f"新增条目{index}",
                created_at=1_700_000_300.0 + index,
                lex_sdr=encode_lexical(f"新增条目{index}"),
            )
        )

    merged = store.lexical_index()

    assert merged.delta_ids == ()
    assert all(merged.contains(f"m-new-{index}") for index in range(3))


def test_delete_forces_a_rebuild_without_the_item() -> None:
    store = _store_with("上海自来水", "猫咪在窗台上")
    store.lexical_index()
    store.delete("m0")

    index = store.lexical_index()

    assert not index.contains("m0")
    assert index.delta_ids == ()


def test_update_forces_a_rebuild_with_the_new_code() -> None:
    store = _store_with("上海自来水")
    store.lexical_index()
    item = store.get("m0")
    assert item is not None

    store.update(item.copy(content="猫咪在窗台上", lex_sdr=encode_lexical("猫咪在窗台上")))
    index = store.lexical_index()

    assert index.delta_ids == ()
    assert index.contains("m0")
    assert "m0" in index.candidates(encode_lexical("猫咪在窗台上"), limit=10)


def test_delta_works_on_an_empty_base_and_can_widen_the_width() -> None:
    empty = LexicalIndex.build([])
    query = SDR(idx=np.asarray([1, 2], dtype=np.int32), weights=np.ones(2), width=64)
    assert empty.candidates(query, limit=10) == ()

    grown = empty.with_added("a", np.asarray([1, 5], dtype=np.int32), 64)
    assert grown.candidates(query, limit=10) == ("a",)

    wide = grown.with_added("b", np.asarray([70], dtype=np.int32), 128)
    wide_query = SDR(idx=np.asarray([70], dtype=np.int32), weights=np.ones(1), width=128)
    assert wide.candidates(wide_query, limit=10) == ("b",)
