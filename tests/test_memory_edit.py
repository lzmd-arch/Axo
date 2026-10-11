"""编辑正文：**改了字就必须换编码**，且换完还能按新字搜到。

这个文件守护的是一个不会报错的错误。只改 ``content`` 不改 ``lex_sdr`` 之后，条目
在库里的样子完全正常——详情面板显示新正文、字节也确实变了、``store.get`` 读出来
也对——但召回打分与 MMR 去冗用的还是旧内容。症状是「改了正文的那条，用新词搜不到、
用旧词反而搜得到」，而没有任何一处抛异常。

所以最后两条测试是**端到端**的：改完之后真的按新正文召回得到、按旧正文召回不到。
只断言 ``lex_sdr != 旧值`` 是不够的——那证明编码变了，不证明它变成了对的。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biosnn_memterm.memory.edit import can_encode, edited_content
from memcore.errors import SDRCapacityError
from memcore.ids import new_ulid
from memcore.recall import recall
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier


def make_item(text: str, **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 1.0,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


# ── can_encode ──────────────────────────────────────────────────────────────


def test_can_encode_accepts_ordinary_text() -> None:
    assert can_encode("我家的路由器放在书房。") is None


def test_can_encode_rejects_empty_text() -> None:
    reason = can_encode("")
    assert reason is not None
    assert "不能为空" in reason


def test_can_encode_rejects_punctuation_only() -> None:
    """全标点要给人话，不是 ``SDRCapacityError`` 的回溯。

    这正是用户最容易撞上的一种输入——想清空一条记忆时顺手敲一个句号或一串省略号。
    """
    reason = can_encode("。。。！？——")
    assert reason is not None
    assert "可编码" in reason


# ── edited_content ──────────────────────────────────────────────────────────


def test_edited_content_recomputes_the_lexical_sdr() -> None:
    """**这是这个模块唯一的存在理由。** 见文件头。"""
    item = make_item("我家的路由器放在书房。")

    revised = edited_content(item, "今天午饭吃的是番茄炒蛋。")

    assert revised.content == "今天午饭吃的是番茄炒蛋。"
    assert revised.lex_sdr != item.lex_sdr
    assert revised.lex_sdr == encode_lexical("今天午饭吃的是番茄炒蛋。")


def test_edited_content_leaves_the_original_alone() -> None:
    item = make_item("原来的正文。")
    original_sdr = item.lex_sdr

    edited_content(item, "新的正文。")

    assert item.content == "原来的正文。"
    assert item.lex_sdr is original_sdr


def test_edited_content_keeps_everything_else() -> None:
    """编辑正文**不是"重新记一遍"**：年龄、权重、访问史、钉住、层级都不该动。

    动了它们的后果是静默的——一条被改过错字的记忆突然变成"刚写下的、权重 0.5 的
    新记忆"，从长期层掉回工作层，下次衰减就被降级。
    """
    item = make_item(
        "错字版本。",
        weight=0.83,
        access_count=7,
        reinforce_count=3,
        pinned=True,
        session_id="01JABCDEF",
        source_ref="turn://01JABCDEF/2",
        meta={"note": "保留我"},
        created_at=1_700_000_000.0,
    )

    revised = edited_content(item, "改好的版本。")

    assert revised.weight == 0.83
    assert revised.access_count == 7
    assert revised.reinforce_count == 3
    assert revised.pinned is True
    assert revised.tier is item.tier
    assert revised.session_id == "01JABCDEF"
    assert revised.source_ref == "turn://01JABCDEF/2"
    assert revised.meta == {"note": "保留我"}
    assert revised.created_at == item.created_at
    assert revised.memory_id == item.memory_id


def test_edited_content_rejects_empty_and_punctuation() -> None:
    item = make_item("原来的正文。")

    with pytest.raises(ValueError, match="不能为空"):
        edited_content(item, "")
    with pytest.raises(SDRCapacityError):
        edited_content(item, "……")


def test_edited_content_only_touches_content_and_lex_sdr() -> None:
    """它**不碰向量**。丢向量是调用方在 ``transform`` 成功之后单独做的事
    （见 :func:`test_the_edit_path_drops_the_stale_vector`）。

    顺序不能反：先丢向量再改正文，万一改写抛了异常（全标点的输入完全可能），
    向量就白丢了——而它本来是对的。
    """
    item = make_item("原来的正文。", sem_sdr=encode_lexical("旧语义占位"), space_id="x/y@8")

    revised = edited_content(item, "新的正文。")

    assert revised.sem_sdr is item.sem_sdr
    assert revised.space_id == "x/y@8"


# ── 端到端：改完之后要能按新正文召回 ─────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def _recall_texts(store: MemoryStore, query: str) -> list[str]:
    ledger = recall(store, query_text=query, budget_tokens=4000, settle=False)
    return [memory.content for memory in ledger.injected]


def test_edited_memory_is_found_by_its_new_text(store: MemoryStore) -> None:
    item = store.add(make_item("我家的路由器放在书房。", weight=1.0))

    store.transform(item.memory_id, lambda it: edited_content(it, "今天午饭吃的是番茄炒蛋。"))

    assert _recall_texts(store, "番茄炒蛋") == ["今天午饭吃的是番茄炒蛋。"]


def test_edited_memory_is_no_longer_found_by_its_old_text(store: MemoryStore) -> None:
    """旧正文搜不到它——**这一条才是"编码换了"的证据**。

    只改 ``content`` 不改 ``lex_sdr`` 时，这条会失败而上面那条会通过：记忆的正文
    已经是新的，但它的编码还认得旧词。
    """
    item = store.add(make_item("我家的路由器放在书房。", weight=1.0))
    assert _recall_texts(store, "路由器") == ["我家的路由器放在书房。"]

    store.transform(item.memory_id, lambda it: edited_content(it, "今天午饭吃的是番茄炒蛋。"))

    assert _recall_texts(store, "路由器") == []


def test_the_edit_path_drops_the_stale_vector(store: MemoryStore) -> None:
    """界面按下「保存」之后要做的**三件事，按这个顺序**：改正文 → 丢旧向量 → 等回填。

    第三步不需要界面做任何事——丢掉之后该 id 重新出现在
    :meth:`~memcore.store.MemoryStore.missing_embedding_ids` 里，既有的回填管线
    会按**新正文**重新嵌入它。所以编辑路径是纯本地的、不等网络，而向量最终仍然
    是新的。

    留在库里的旧向量是这个流程里唯一的静默错法：向量还在，看起来"语义通道可用"，
    但它编码的是**改之前的**意思。
    """
    space = make_space_id("test", "fake-embed", 8)
    store.register_space(Space(space_id=space, dim=8, model_ref="fake"))
    item = store.add(make_item("我家的路由器放在书房。"))
    store.set_embedding(item.memory_id, space, np.zeros(8, dtype=np.float32))
    assert store.missing_embedding_ids(space) == []

    store.transform(item.memory_id, lambda it: edited_content(it, "今天午饭吃的是番茄炒蛋。"))
    store.drop_embedding(item.memory_id)

    assert store.get_embedding(item.memory_id, space) is None
    assert store.missing_embedding_ids(space) == [item.memory_id]


def test_transform_rolls_back_when_the_new_text_cannot_be_encoded(store: MemoryStore) -> None:
    """全标点的改写**整条不动**——库里不能留下"正文是新的、编码是旧的"。

    ``edited_content`` 在这里抛异常，``transform`` 的锁内流程就此中断、不落盘。
    这条把两个模块接在一起：单看任一个都正确，接起来才是不变量。
    """
    item = store.add(make_item("原来的正文。"))
    before = store.get(item.memory_id)

    with pytest.raises(SDRCapacityError):
        store.transform(item.memory_id, lambda it: edited_content(it, "？？？"))

    after = store.get(item.memory_id)
    assert after is not None and before is not None
    assert after.content == "原来的正文。"
    assert after.lex_sdr == before.lex_sdr
