"""归并落库：**只增不改**。

用户的约束是「已有的记忆不允许被你改动」，而归并天然长着一副要改写的样子（把几句
合成一句）。所以这个文件只守一件事：跑完一轮之后，**既有行逐字节没变**，多出来的
全是带 derived 标记的新行。

比对的是 (memory_id, content, created_at, kind, tier) 这五个字段——**不比
reinforce_count 与访问记账**：那是用户明确说可以继续写的（「正文不动，记账照写」）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from axo.memory import consolidation as entry
from axo.paths import memory_db_path
from memcore.graph import GraphIndex
from memcore.merge import group_fragments
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

AT = 1_760_000_000.0
BASE = 1_760_000_100.0


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把数据目录指到 tmp_path——不铺这一层，用例会往真实数据目录里写。"""
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    return tmp_path


def _item(memory_id: str, content: str, *, at: float, session: str = "s1") -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=MemoryKind.FACT,
        tier=Tier.RECENT,
        content=content,
        created_at=at,
        session_id=session,
        lex_sdr=encode_lexical(content),
    )


def _fragments() -> list[MemoryItem]:
    """两条碎条目（该并）+ 一条隔了 83 分钟的（不该并）。"""
    return [
        _item("m000", "感冒那位回来了。", at=BASE),
        _item("m001", "还带了杯奶茶。", at=BASE + 120),
        _item("m002", "路由器在书房。", at=BASE + 5000),
    ]


def _one_cluster(size: int = 3) -> GraphIndex:
    top_m = size - 1
    nbr_idx = np.full((size, top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, top_m), dtype=np.float32)
    for row in range(size):
        for slot, target in enumerate(index for index in range(size) if index != row):
            nbr_idx[row, slot] = target
            nbr_w[row, slot] = 1.0
    return GraphIndex(
        node_ids=tuple(f"m{index:03d}" for index in range(size)),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="d",
        built_at=0.0,
        top_m=top_m,
        min_weight=0.1,
    )


def _identity(store: MemoryStore) -> dict[str, tuple]:
    """一条记忆的「正文身份」：改没改就看这五个字段。"""
    return {
        item.memory_id: (item.content, item.created_at, item.kind, item.tier)
        for item in store.load_all()
    }


# ── 只增不改 ────────────────────────────────────────────────────────────────


def test_a_merge_round_never_touches_an_existing_row(data_dir: Path) -> None:
    """跑完一轮，既有行的正文身份逐条相等；多出来的全是派生条目。"""
    with MemoryStore(memory_db_path()) as store:
        for item in _fragments():
            store.add(item)
        before = _identity(store)

        run = entry.run_consolidation(store, _one_cluster(), at=AT)

        after = _identity(store)
        for memory_id, row in before.items():
            assert after[memory_id] == row, f"既有行 {memory_id} 被改了：{row} → {after[memory_id]}"

        added = set(after) - set(before)
        assert added, "该有一条派生条目"
        assert len(run.merged) == len(added)
        for memory_id in added:
            item = store.get(memory_id)
            assert item is not None and item.meta.get("derived") is True


def test_the_derived_row_names_its_fragments_and_its_span(data_dir: Path) -> None:
    """派生条目带指针与区间，正文里附着逐字的原话。"""
    with MemoryStore(memory_db_path()) as store:
        for item in _fragments():
            store.add(item)

        run = entry.run_consolidation(store, _one_cluster(), at=AT)

        assert len(run.merged) == 1, "只该并出一组：第三条隔了 83 分钟"
        derived = run.merged[0]
        assert derived.meta["derived_from"] == ["m000", "m001"]
        assert derived.meta["span"] == [BASE, BASE + 120]
        assert derived.created_at == BASE
        assert "感冒那位回来了。" in derived.content
        assert "还带了杯奶茶。" in derived.content


def test_planning_is_idempotent_across_rounds(data_dir: Path) -> None:
    """跑第二遍不该重复造条目——派生条目会把「已经并过」这件事记下来。"""
    with MemoryStore(memory_db_path()) as store:
        for item in _fragments():
            store.add(item)
        entry.run_consolidation(store, _one_cluster(), at=AT)

        assert entry.plan_merges(list(store.iter_items())) == []


def test_the_merge_event_names_its_fragments(data_dir: Path) -> None:
    """归并也进同一本事件账，decision=merge，detail 带 derived_from。"""
    with MemoryStore(memory_db_path()) as store:
        for item in _fragments():
            store.add(item)

        run = entry.run_consolidation(store, _one_cluster(), at=AT)

        rows = [
            json.loads(line)
            for line in run.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merges = [row for row in rows if row.get("decision") == "merge"]
        assert len(merges) == 1
        assert merges[0]["memory_id"] == run.merged[0].memory_id
        assert merges[0]["detail"]["derived_from"] == ["m000", "m001"]


def test_a_library_with_nothing_to_merge_adds_nothing(data_dir: Path) -> None:
    """没有可并的组就一条都不写——不能为了「跑过一轮」而造点东西出来。"""
    with MemoryStore(memory_db_path()) as store:
        store.add(_item("m000", "路由器在书房。", at=BASE))
        before = _identity(store)

        run = entry.run_consolidation(store, _one_cluster(1), at=AT)

        assert not run.merged
        assert set(_identity(store)) == set(before)


# ── 判据在写库这一侧也成立 ──────────────────────────────────────────────────


def test_the_write_side_uses_the_same_grouping_as_the_pure_layer() -> None:
    """写库用的就是 group_fragments，不是另抄一套判据。"""
    items = _fragments()

    groups = group_fragments(items)
    planned = entry.plan_merges(items)

    assert [group.memory_ids for group in planned] == [group.memory_ids for group in groups]
