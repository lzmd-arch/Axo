"""遗忘与恢复：**软遗忘是可逆的，而这正是它区别于删除的全部理由。**

分两半，边界很清楚：

* :mod:`~axo.memory.forget` 的三个纯函数——写 COLD、写回程票、读回程票。
* 端到端那几条：软遗忘之后 ``recall`` 真的不再注入它（且账本里**说得出原因**），
  恢复之后又回来了。

最后一条 ``test_soft_forgetting_a_pinned_item_does_nothing`` 记录的是一个**已知的
坏语义**，不是期望行为：钉住的条目在 tier 判定之前就被跳过，所以对它软遗忘是个
无声的空动作。它留在这里，是因为界面禁掉那个按钮的理由就是它——把这条测试删了，
"钉住的条目禁软遗忘"就成了一条没有依据的约定。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axo.memory.forget import (
    FORGET_KEY,
    ForgetRecord,
    can_soft_forget,
    read_forget_record,
    restored,
    soft_forgotten,
)
from memcore.audit import RejectReason
from memcore.ids import new_ulid
from memcore.recall import recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

NOW = 1_760_000_000.0


def make_item(text: str, **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "weight": 1.0,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def _recall(store: MemoryStore, query: str):
    return recall(store, query_text=query, budget_tokens=4000, settle=False)


# ── can_soft_forget ─────────────────────────────────────────────────────────


def test_can_soft_forget_accepts_an_ordinary_item() -> None:
    assert can_soft_forget(make_item("一条普通事实。")) is None


def test_can_soft_forget_refuses_pinned_items() -> None:
    """理由在界面上要能读出来，而且**要指向删除**——留一个点了没反应的按钮更糟。"""
    reason = can_soft_forget(make_item("一条钉住的事实。", pinned=True))

    assert reason is not None
    assert "钉住" in reason
    assert "彻底删除" in reason


def test_can_soft_forget_refuses_already_cold_items() -> None:
    reason = can_soft_forget(make_item("已经忘了。", tier=Tier.COLD))

    assert reason is not None
    assert "已经" in reason


# ── soft_forgotten ──────────────────────────────────────────────────────────


def test_soft_forgotten_moves_to_cold_and_records_the_old_tier() -> None:
    item = make_item("我记得这个。", tier=Tier.RECENT)

    forgotten = soft_forgotten(item, at=NOW)

    assert forgotten.tier is Tier.COLD
    assert forgotten.content == "我记得这个。"
    assert forgotten.memory_id == item.memory_id
    assert read_forget_record(forgotten) == ForgetRecord(previous_tier=Tier.RECENT, at=NOW)


def test_soft_forgotten_leaves_the_original_alone() -> None:
    item = make_item("我记得这个。", tier=Tier.RECENT)

    soft_forgotten(item, at=NOW)

    assert item.tier is Tier.RECENT
    assert FORGET_KEY not in item.meta


def test_soft_forgotten_keeps_other_meta_keys() -> None:
    """``meta`` 是别人的地盘，写回程票不能顺手清空它。

    P3 的工具轨迹、P4 的归因快照都住在这里——一次遗忘把它们抹掉，是"用户只想让
    她别再提这件事"变成"任务记录没了"的那种错。
    """
    item = make_item("我记得这个。", meta={"trace": [1, 2], "note": "别动我"})

    forgotten = soft_forgotten(item, at=NOW)

    assert forgotten.meta["trace"] == [1, 2]
    assert forgotten.meta["note"] == "别动我"
    assert FORGET_KEY in forgotten.meta


def test_soft_forgetting_twice_is_idempotent() -> None:
    """**第二条回程票不能覆盖第一条。**

    违反它的后果是静默的：``previous_tier`` 被写成 ``COLD`` 之后，"恢复"会把条目
    从冷层恢复到冷层——界面显示成功、``meta`` 也确实少了一个键，但它再也不会回来。
    而用户点两次遗忘是很自然的事。
    """
    item = make_item("我记得这个。", tier=Tier.ARCHIVE)

    once = soft_forgotten(item, at=NOW)
    twice = soft_forgotten(once, at=NOW + 3600)

    assert twice.tier is Tier.COLD
    assert read_forget_record(twice) == ForgetRecord(previous_tier=Tier.ARCHIVE, at=NOW)
    assert restored(twice)[0].tier is Tier.ARCHIVE


# ── restored ────────────────────────────────────────────────────────────────


def test_restored_returns_to_the_recorded_tier() -> None:
    item = make_item("我记得这个。", tier=Tier.LONGTERM)

    back, had_record = restored(soft_forgotten(item, at=NOW))

    assert had_record is True
    assert back.tier is Tier.LONGTERM
    assert FORGET_KEY not in back.meta


def test_restored_falls_back_to_working_without_a_record() -> None:
    """没有回程票就回工作层，**并且如实说"没读到"**。

    P4 的睡眠巩固也会写 ``COLD``，那些条目没有回程票。猜一个层级然后不说，是这里
    唯一不能做的事——用户会以为它恢复了原来的位置。
    """
    item = make_item("本来就在冷层。", tier=Tier.COLD)

    back, had_record = restored(item)

    assert had_record is False
    assert back.tier is Tier.WORKING


@pytest.mark.parametrize(
    "broken",
    [
        {"previous_tier": "cold", "at": NOW},  # 回程票写成了单程票
        {"previous_tier": "不存在的层", "at": NOW},
        {"at": NOW},  # 缺键
        "不是对象",
        None,
    ],
)
def test_read_forget_record_tolerates_garbage(broken: object) -> None:
    """``meta`` 对 memcore 是不透明的，我们没有资格要求它装的一定是我们写的内容。

    读不懂的后果应该只是"恢复后落到默认层"，不是"这条记忆打不开"。
    """
    item = make_item("我记得这个。", tier=Tier.COLD, meta={FORGET_KEY: broken})

    assert read_forget_record(item) is None
    back, had_record = restored(item)
    assert had_record is False
    assert back.tier is Tier.WORKING


def test_restore_clears_a_stale_record() -> None:
    """恢复之后那个键必须删掉。

    留着它，下一次 :func:`read_forget_record` 会读到一个**过期**的回程票——它指向
    的是更早那次遗忘之前的层级，于是第二次「恢复」会把条目送到一个用户从没选过的
    层级去。
    """
    once = soft_forgotten(make_item("我记得这个。", tier=Tier.ARCHIVE), at=NOW)
    back, _ = restored(once)

    assert read_forget_record(back) is None


def test_forget_record_rejects_cold_as_previous_tier() -> None:
    """构造上就挡住单程票——不能只靠"没人会这么写"。"""
    with pytest.raises(ValueError, match="COLD"):
        ForgetRecord(previous_tier=Tier.COLD, at=NOW)


# ── 端到端：遗忘真的让它不再被注入 ───────────────────────────────────────────


def test_forgotten_memory_is_no_longer_injected(store: MemoryStore) -> None:
    item = store.add(make_item("我家的路由器放在书房。"))

    store.transform(item.memory_id, lambda it: soft_forgotten(it, at=NOW))

    ledger = _recall(store, "路由器")
    assert ledger.injected_ids == []


def test_forgotten_memory_says_why_in_the_ledger(store: MemoryStore) -> None:
    """**它必须出现在账本里，且原因是 COLD。**

    不出现的话，界面就无法回答"她为什么忘了这件事"——用户看到的是"库里有这条、
    账本里没有"，而那是"缺席"这个二义状态，只能靠猜。
    """
    item = store.add(make_item("我家的路由器放在书房。"))

    store.transform(item.memory_id, lambda it: soft_forgotten(it, at=NOW))

    rejected = {entry.memory_id: entry.reason for entry in _recall(store, "路由器").rejected}
    assert rejected[item.memory_id] is RejectReason.COLD


def test_restored_memory_is_injected_again(store: MemoryStore) -> None:
    """**软遗忘的全部意义在这里。** 后悔药是能吃的。"""
    item = store.add(make_item("我家的路由器放在书房。", tier=Tier.RECENT))
    store.transform(item.memory_id, lambda it: soft_forgotten(it, at=NOW))
    assert _recall(store, "路由器").injected_ids == []

    store.transform(item.memory_id, lambda it: restored(it)[0])

    ledger = _recall(store, "路由器")
    assert ledger.injected_ids == [item.memory_id]
    reread = store.get(item.memory_id)
    assert reread is not None and reread.tier is Tier.RECENT


def test_soft_forgetting_a_pinned_item_does_nothing(store: MemoryStore) -> None:
    """**已知的坏语义，记录现状而不是期望行为。**

    ``_scan`` 在 tier 判定**之前**就 ``if item.pinned ... : continue``（recall.py:374），
    所以一条钉住的记忆：

    * 软遗忘它**不改变任何事**——它压根不走召回，也就无所谓"不再被召回"；
    * 它还会**从账本里彻底消失**（既不 injected 也不 rejected），于是界面显示
      「已遗忘」而它仍在每一轮的上下文里。

    人格锚是这条的最坏一例：``persona_texts`` 只按 kind 过滤、不看 tier，所以软遗忘
    人格锚连 L0 都拦不住。

    这就是裁决「钉住的条目禁软遗忘」的依据，也是界面必须禁掉那个按钮的原因——
    留按钮 + 只靠提示语，等于让用户亲手把自己骗了。要让它消失只有彻底删除一条路。
    """
    item = store.add(
        make_item("钉住的事实。", pinned=True, tier=Tier.LONGTERM),
    )

    store.transform(item.memory_id, lambda it: soft_forgotten(it, at=NOW))

    ledger = _recall(store, "钉住的事实")
    assert item.memory_id not in ledger.injected_ids
    assert item.memory_id not in [entry.memory_id for entry in ledger.rejected]
    # 而且它没有"不再被召回"——它本来就不在候选里。
    before = _recall(store, "任何东西")
    assert item.memory_id not in before.injected_ids
