"""记忆库的**改写路径**：transform / drop_embedding / embedded_ids / summaries。

P1 的存储层只被"写入"与"召回"两条路用过，而这两条路都**不修改既有条目**
（召回只碰权重与访问计数，且是单列 UPDATE）。P2 的记忆界面第一次引入"用户直接
改一条记忆"——这是另一类写入，它有自己的一批静默失效方式，所以这个文件的三组
断言都指向同一件事：**改一条记忆不能顺带改坏别的东西。**

* ``transform`` —— 为什么不是"先 get 再 update"：``update`` 整行覆盖，而召回
  路径会在这个窗口里结算权重/访问计数，后写的整个盖掉先写的。
* ``drop_embedding`` —— 编辑正文之后必须走它，否则向量还是旧意思的（静默）。
  删完之后该 id 必须能被既有的补账管线重新捡起来。
* ``summaries`` —— 列表视图**不读** SDR blob。这一条要能扛住"某一行半个 SDR
  是坏的"：条目的坏数据不该让整个界面打不开。
"""

from __future__ import annotations

import sqlite3
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from memcore.errors import StoreCorruptError, StoreError
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, MemorySummary, Taint, Tier

SPACE = make_space_id("test", "fake-embed", 8)
OTHER_SPACE = make_space_id("test", "other-embed", 8)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


@pytest.fixture
def embed_store(store: MemoryStore) -> MemoryStore:
    store.register_space(Space(space_id=SPACE, dim=8, model_ref="fake"))
    store.register_space(Space(space_id=OTHER_SPACE, dim=8, model_ref="other"))
    return store


def make_item(text: str = "一条记忆", **overrides: object) -> MemoryItem:
    fields_: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 0.7,
        "lex_sdr": encode_lexical(text),
    }
    fields_.update(overrides)
    return MemoryItem(**fields_)  # type: ignore[arg-type]


# ── transform ───────────────────────────────────────────────────────────────


def test_transform_persists_the_change(store: MemoryStore) -> None:
    item = make_item("改之前的正文。")
    store.add(item)

    revised = store.transform(item.memory_id, lambda it: it.copy(content="改之后的正文。"))

    assert revised.content == "改之后的正文。"
    reread = store.get(item.memory_id)
    assert reread is not None
    assert reread.content == "改之后的正文。"
    assert store.count() == 1


def test_transform_missing_id_raises(store: MemoryStore) -> None:
    with pytest.raises(StoreError, match="不存在"):
        store.transform(new_ulid(), lambda it: it)


def test_transform_leaves_the_row_untouched_when_change_raises(store: MemoryStore) -> None:
    """``change`` 抛异常时**整条不动**。

    真实的触发路径不是"测试里故意抛"，而是编辑成一段全标点的正文——
    :func:`memcore.sdr.encode_lexical` 会抛 ``SDRCapacityError``。那时候库里
    必须还是旧的正文与旧的 SDR，不能出现"正文换了、编码没换"的半截状态。
    """
    item = make_item("原来的正文。")
    store.add(item)

    def boom(_: MemoryItem) -> MemoryItem:
        raise RuntimeError("编码失败")

    with pytest.raises(RuntimeError):
        store.transform(item.memory_id, boom)

    reread = store.get(item.memory_id)
    assert reread is not None
    assert reread.content == "原来的正文。"


def test_transform_rejects_a_changed_id(store: MemoryStore) -> None:
    """改 id 等于换一条记忆——调用方以为在改原来那条，实际会写出一条新的。"""
    item = make_item()
    store.add(item)

    with pytest.raises(StoreError, match="memory_id"):
        store.transform(item.memory_id, lambda it: it.copy(memory_id=new_ulid()))

    assert store.count() == 1


def test_transform_does_not_cascade_delete_embeddings(embed_store: MemoryStore) -> None:
    """改写正文**不能带走向量**——这是 ``add`` 那条 UPSERT 纪律在 transform 里的重演。

    ``transform`` 内部走的是 ``add``。若哪天有人把它换成 ``INSERT OR REPLACE``
    （语义上是先 DELETE 再 INSERT），向量表上的 CASCADE 会静默删掉向量：表现只是
    "改过一条记忆之后，它语义上就再也搜不到了"。
    """
    item = make_item("先写条目，再补向量。")
    embed_store.add(item)
    embed_store.set_embedding(item.memory_id, SPACE, np.arange(8, dtype=np.float32))

    embed_store.transform(item.memory_id, lambda it: it.copy(content="改写后的正文。"))

    assert embed_store.get_embedding(item.memory_id, SPACE) is not None


# ── drop_embedding / embedded_ids ───────────────────────────────────────────


def test_embedded_ids_reflects_writes(embed_store: MemoryStore) -> None:
    first, second = make_item("第一条。"), make_item("第二条。")
    embed_store.add(first)
    embed_store.add(second)
    embed_store.set_embedding(first.memory_id, SPACE, np.zeros(8, dtype=np.float32))

    assert embed_store.embedded_ids(SPACE) == {first.memory_id}
    assert embed_store.embedded_ids(OTHER_SPACE) == set()


def test_drop_embedding_removes_every_space_by_default(embed_store: MemoryStore) -> None:
    item = make_item()
    embed_store.add(item)
    for space in (SPACE, OTHER_SPACE):
        embed_store.set_embedding(item.memory_id, space, np.zeros(8, dtype=np.float32))

    removed = embed_store.drop_embedding(item.memory_id)

    assert removed == 2
    assert embed_store.embedded_ids(SPACE) == set()
    assert embed_store.embedded_ids(OTHER_SPACE) == set()


def test_drop_embedding_can_target_one_space(embed_store: MemoryStore) -> None:
    item = make_item()
    embed_store.add(item)
    for space in (SPACE, OTHER_SPACE):
        embed_store.set_embedding(item.memory_id, space, np.zeros(8, dtype=np.float32))

    embed_store.drop_embedding(item.memory_id, SPACE)

    assert embed_store.embedded_ids(SPACE) == set()
    assert embed_store.embedded_ids(OTHER_SPACE) == {item.memory_id}


def test_drop_embedding_returns_zero_when_there_was_nothing(embed_store: MemoryStore) -> None:
    """**0 不是错误。** 编辑一条从未嵌入过的记忆（嵌入模型一直没装）本来就是常态。"""
    item = make_item()
    embed_store.add(item)

    assert embed_store.drop_embedding(item.memory_id) == 0


def test_dropped_id_can_be_picked_up_again(embed_store: MemoryStore) -> None:
    """**这是「丢弃待补」这条裁决的全部机制。**

    编辑正文之后丢掉向量，靠的不是一个新的补账者，而是"该 id 重新出现在
    ``missing_embedding_ids`` 里"——既有的每轮补账管线会按**新正文**重新嵌入它。
    这条断言失败就意味着编辑过的记忆会永久停在"只有词法通道"的状态，
    而召回质量下降不会有任何一处报错。
    """
    item = make_item("改之前的正文。")
    embed_store.add(item)
    embed_store.set_embedding(item.memory_id, SPACE, np.zeros(8, dtype=np.float32))
    assert embed_store.missing_embedding_ids(SPACE) == []

    embed_store.transform(item.memory_id, lambda it: it.copy(content="改之后的正文。"))
    embed_store.drop_embedding(item.memory_id)

    assert embed_store.missing_embedding_ids(SPACE) == [item.memory_id]


def test_dropped_id_sorts_ahead_of_newer_pending_items(embed_store: MemoryStore) -> None:
    """被编辑的老记忆**插队到队首**，不等 FIFO。

    ``missing_embedding_ids`` 按 ``created_at`` 升序。所以一条创建于很久以前的记忆
    在被编辑、丢掉向量之后，排在所有新条目**前面**——下一轮那一小批补账一定含它。
    （若这里反过来，界面就得在编辑路径上接一次网络调用，而那是"编辑要等模型"
    的代价，用户已经明确否决了。）
    """
    old = make_item("很久以前写下的一条。")
    embed_store.add(old)
    embed_store.set_embedding(old.memory_id, SPACE, np.zeros(8, dtype=np.float32))
    for index in range(20):
        fresh = make_item(f"后来写下的第 {index} 条。")
        embed_store.add(fresh)

    embed_store.drop_embedding(old.memory_id)
    pending = embed_store.missing_embedding_ids(SPACE, limit=1)

    assert pending == [old.memory_id]


# ── summaries ───────────────────────────────────────────────────────────────


def test_summary_has_no_sdr_fields() -> None:
    """``MemorySummary`` 里**不能**长出 ``lex_sdr``/``sem_sdr`` 这样的字段。

    有了它，列表就会开始反序列化每条的 SDR（约 6.9 KB/条），而列表一个字节都不
    显示它们——两万条记忆是 130 MB 的纯浪费。
    """
    names = {field.name for field in fields(MemorySummary)}
    assert not [name for name in names if "sdr" in name]


def test_summaries_survive_a_corrupt_sdr_row(store: MemoryStore) -> None:
    """某一行半个 SDR 是坏的，**列表照样能打开**。

    从**另一个连接**把磁盘上的行改坏（不是伸手进 store 的私有属性），模拟的就是
    真实的坏数据。``_row_to_item`` 遇到"三列没有同时有值"会当场抛
    ``StoreCorruptError``——那是刻意的，半个 SDR 会以"有编码"的样子参与相似度
    计算。但浏览列表根本不读那三列，所以一条坏数据不该让整个记忆界面打不开。
    """
    item = make_item("一条 SDR 坏掉的记忆。")
    store.add(item)
    with sqlite3.connect(store.path) as outside:
        outside.execute("UPDATE memories SET lex_w = NULL WHERE memory_id = ?", (item.memory_id,))

    summaries = store.summaries()

    assert [summary.memory_id for summary in summaries] == [item.memory_id]
    assert summaries[0].content == "一条 SDR 坏掉的记忆。"
    with pytest.raises(StoreCorruptError):
        store.get(item.memory_id)


def test_summaries_are_newest_first(store: MemoryStore) -> None:
    ids = [store.add(make_item(f"第 {index} 条。")).memory_id for index in range(5)]

    assert [summary.memory_id for summary in store.summaries()] == list(reversed(ids))


def test_summaries_limit_and_offset(store: MemoryStore) -> None:
    ids = [store.add(make_item(f"第 {index} 条。")).memory_id for index in range(5)]
    newest_first = list(reversed(ids))

    assert [s.memory_id for s in store.summaries(limit=2)] == newest_first[:2]
    assert [s.memory_id for s in store.summaries(limit=2, offset=2)] == newest_first[2:4]


def test_summaries_filter_by_tier_and_kind(store: MemoryStore) -> None:
    cold = store.add(make_item("被软遗忘的。", tier=Tier.COLD))
    preference = store.add(make_item("用户偏好中文。", kind=MemoryKind.PREFERENCE))
    store.add(make_item("一条普通事实。"))

    assert [s.memory_id for s in store.summaries(tier=Tier.COLD)] == [cold.memory_id]
    assert [s.memory_id for s in store.summaries(kind=MemoryKind.PREFERENCE)] == [
        preference.memory_id
    ]


def test_summaries_include_forgotten_items(store: MemoryStore) -> None:
    """``COLD`` 的条目**必须出现在列表里**。

    把它过滤掉的话，"软遗忘"就变成了单程票：界面里再也找不到它，用户也就没有
    任何办法把它恢复回来。
    """
    cold = store.add(make_item("被软遗忘的。", tier=Tier.COLD))

    assert [s.memory_id for s in store.summaries()] == [cold.memory_id]


def test_summaries_query_matches_substring(store: MemoryStore) -> None:
    store.add(make_item("我家的路由器放在书房。"))
    store.add(make_item("今天午饭吃的是番茄炒蛋。"))

    hits = store.summaries(query="路由器")

    assert len(hits) == 1
    assert "路由器" in hits[0].content


def test_summaries_query_treats_wildcards_literally(store: MemoryStore) -> None:
    """``%`` 与 ``_`` 按字面匹配，不当通配符。

    不转义的话，搜「进度 50%」会变成"以进度 50 开头的任意串"，搜「a_b」会命中
    「axb」——两种都是"结果比你要的多"，而多出来的那些看起来完全正常。
    """
    store.add(make_item("完成度 100% 了。"))
    store.add(make_item("完成度 50x 左右。"))
    store.add(make_item("配置项 a_b 要改。"))
    store.add(make_item("配置项 axb 要改。"))

    assert len(store.summaries(query="100%")) == 1
    assert [s.content for s in store.summaries(query="a_b")] == ["配置项 a_b 要改。"]
