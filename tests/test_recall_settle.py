"""``recall(..., settle=False)``：**看一眼不该写下任何东西。**

记忆界面里的「以它为线索召回一次」是探索，不是想起。不关掉结算的话，用户每点
一次那个按钮，被探到的几条就会 ``access_count + 1``、衰减被结算一次——于是

* 它们的权重衰减得比别人慢，在**真实**召回里排得更靠前；
* 而这个差异的起因是一次浏览，没有任何地方能看出来。

这就是"一次观察改变了被观察的东西"。默认 ``settle=True`` 保持不变，所以既有的
调用点与测试一行都不改——这个文件只钉住那个新开关。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memcore.ids import new_ulid
from memcore.recall import recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

NOW = 1_760_000_000.0
DAY = 86400.0


def make_item(text: str, **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "weight": 1.0,
        "created_at": NOW - 100 * DAY,
        "last_decay_ts": NOW - 100 * DAY,
        "last_access_ts": NOW - 100 * DAY,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def test_probe_leaves_access_count_alone(store: MemoryStore) -> None:
    item = store.add(make_item("我家的路由器放在书房。"))

    ledger = recall(store, query_text="路由器", budget_tokens=4000, now=NOW, settle=False)

    assert ledger.injected_ids == [item.memory_id]
    reread = store.get(item.memory_id)
    assert reread is not None
    assert reread.access_count == 0
    assert reread.last_access_ts == item.last_access_ts


def test_probe_leaves_the_weight_alone(store: MemoryStore) -> None:
    """衰减也不能被结算。

    结算之后 ``last_decay_ts`` 会被推到 ``now``——那本身不算错，但它会让这条记忆
    **下一次**真实召回少衰减 100 天。浏览改写了它的衰亡曲线。
    """
    item = store.add(make_item("我家的路由器放在书房。"))

    recall(store, query_text="路由器", budget_tokens=4000, now=NOW, settle=False)

    reread = store.get(item.memory_id)
    assert reread is not None
    assert reread.weight == item.weight
    assert reread.last_decay_ts == item.last_decay_ts


def test_the_default_still_settles(store: MemoryStore) -> None:
    """默认行为**一点没变**——这是 ``settle`` 加得上来的前提。

    它一变，P1 的全部召回测试与每一个生产调用点都要跟着改，而"召回不写库"会
    静默地让权重永远不衰减。
    """
    item = store.add(make_item("我家的路由器放在书房。"))

    recall(store, query_text="路由器", budget_tokens=4000, now=NOW)

    reread = store.get(item.memory_id)
    assert reread is not None
    assert reread.access_count == 1
    assert reread.last_access_ts == NOW
    assert reread.weight < item.weight  # 100 天没被想起，衰减过了


def test_probe_and_settle_return_the_same_selection(store: MemoryStore) -> None:
    """两种调用选出**同一批**记忆。

    不同的话，"线索召回"看到的就是另一套结果，而它与真实注入的差异会被归给
    探索参数（阈值、条数），实际却是结算本身造成的——账本的可解释性就假了。
    """
    for index in range(6):
        store.add(make_item(f"关于路由器的第 {index} 条记录，放在书房。"))

    probed = recall(store, query_text="路由器", budget_tokens=4000, now=NOW, settle=False)
    settled = recall(store, query_text="路由器", budget_tokens=4000, now=NOW, settle=True)

    assert probed.injected_ids == settled.injected_ids
    assert probed.scanned == settled.scanned


def test_probe_writes_nothing_even_for_a_pinned_persona(store: MemoryStore) -> None:
    """钉住的条目本来就不进结算路径，这里确认探针不会从别处漏写它。"""
    anchor = store.add(
        make_item("你是 BioSNN 记忆终端的助手。", kind=MemoryKind.PERSONA_ANCHOR, pinned=True)
    )

    recall(store, query_text="记忆终端", budget_tokens=4000, now=NOW, settle=False)

    reread = store.get(anchor.memory_id)
    assert reread is not None
    assert reread.access_count == 0
