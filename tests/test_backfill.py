"""向量回填。

这个文件守的是"语义通道不会悄悄断掉"这件事。写入不调模型（说一句话不该变成一次
模型调用），所以一条记忆进库时只有词法 SDR，向量是事后补的——而**补向量断了是
没有症状的**：召回照常返回结果，只是少了一条通道，答案变差一点点。没有人会因此
报警。所以这里的断言都盯着可查询的那几个量：补了几条、还剩几条没补、空间登记成
什么样。

嵌入后端是假的。这个模块的正确性（补谁、补多少、断了怎么办）与 HTTP 无关，而
接上真 Ollama 的测试最后总会被跳过——那时上面这句话就没有断言了。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from axo.memory import sync_embeddings
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

SPACE_ID = make_space_id("fake", "embedder", 4)


class FakeEmbedder:
    """十行假件。记录它被喂了什么，并可以被要求**少还几行**或**还错宽度**。"""

    def __init__(
        self,
        *,
        space_id: str = SPACE_ID,
        dim: int = 4,
        rows: int | None = None,
        vector_dim: int | None = None,
    ) -> None:
        self.space_id = space_id
        self.dim = dim
        self.batches: list[list[str]] = []
        self._rows = rows
        self._vector_dim = vector_dim if vector_dim is not None else dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        count = len(texts) if self._rows is None else self._rows
        return np.ones((count, self._vector_dim), dtype=np.float32)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def add(store: MemoryStore, content: str, *, created_at: float = 0.0) -> MemoryItem:
    item = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content=content,
        taint=Taint.USER,
        created_at=created_at,
        lex_sdr=encode_lexical(content),
    )
    return store.add(item)


# ── 补齐 ────────────────────────────────────────────────────────────────────


def test_backfill_fills_everything_that_lacks_a_vector(store: MemoryStore) -> None:
    add(store, "我家的路由器放在书房书架顶层。", created_at=1.0)
    add(store, "周三下午三点开组会。", created_at=2.0)
    embedder = FakeEmbedder()

    assert sync_embeddings(store, embedder) == 2
    assert store.missing_embedding_ids(SPACE_ID) == []
    # 一次全给后端，由它自己分片。这里再切一次会让"批大小"有两个出处。
    assert embedder.batches == [["我家的路由器放在书房书架顶层。", "周三下午三点开组会。"]]


def test_the_second_run_writes_nothing(store: MemoryStore) -> None:
    """幂等：判据是"这个空间里没有向量的条目"，不是"上次跑到哪了"。"""
    add(store, "我家的路由器放在书房书架顶层。")
    embedder = FakeEmbedder()

    assert sync_embeddings(store, embedder) == 1
    assert sync_embeddings(store, embedder) == 0
    assert len(embedder.batches) == 1


def test_limit_bounds_one_run_without_losing_the_rest(store: MemoryStore) -> None:
    """每轮只补一小批。

    这是"刚写下的记忆下一轮就能被语义召回"的实现方式——所以剩下的必须还在待补
    列表里，不能因为这一次没补完就被跳过。
    """
    for index in range(5):
        add(store, f"第 {index} 条要记住的事。", created_at=float(index))

    assert sync_embeddings(store, FakeEmbedder(), limit=2) == 2
    assert len(store.missing_embedding_ids(SPACE_ID)) == 3
    assert sync_embeddings(store, FakeEmbedder(), limit=2) == 2
    assert len(store.missing_embedding_ids(SPACE_ID)) == 1


def test_the_oldest_are_filled_first(store: MemoryStore) -> None:
    """按创建时间从早到晚。

    顺序不是审美：每轮只有 8 个名额，而"先补最老的"意味着断过一次的库会从断点
    接着往前补，而不是永远围着最新的几条打转。
    """
    add(store, "最早说的那句话。", created_at=1.0)
    add(store, "后来才说的一句话。", created_at=2.0)
    embedder = FakeEmbedder()

    sync_embeddings(store, embedder, limit=1)
    assert embedder.batches == [["最早说的那句话。"]]


# ── 空间登记 ────────────────────────────────────────────────────────────────


def test_the_space_is_registered_from_the_embedders_own_identity(store: MemoryStore) -> None:
    """空间标识与维度都来自嵌入后端，``model_ref`` 从标识里拆。

    两处各写一遍模型名，迟早会有一处忘了改——而那是记忆浏览器里唯一能告诉用户
    "这些向量是哪个模型留下的"的东西。
    """
    add(store, "我家的路由器放在书房书架顶层。")
    sync_embeddings(store, FakeEmbedder())

    space = store.registry.get(SPACE_ID)
    assert space.dim == 4
    assert space.model_ref == "fake/embedder"


def test_registering_twice_is_idempotent(store: MemoryStore) -> None:
    add(store, "我家的路由器放在书房书架顶层。")
    sync_embeddings(store, FakeEmbedder())
    first = store.registry.get(SPACE_ID)
    sync_embeddings(store, FakeEmbedder())

    # 连 created_at 都不变——说明第二次登记走的是"确认"而不是"覆盖"。
    assert store.registry.get(SPACE_ID) == first


def test_an_unregistered_space_stays_unregistered_when_asked(store: MemoryStore) -> None:
    """``register=False`` 时不登记，未登记的空间就照实报错。

    ``SpaceRegistry.get`` 未登记时抛异常而不是返回 ``None``——这条路径要保证调用方
    拿到的是**异常**，而不是一个静默的空空间（那样补向量会写进一个没有 dim 的地方）。
    """
    from memcore.errors import SpaceNotFoundError

    add(store, "我家的路由器放在书房书架顶层。")
    with pytest.raises(SpaceNotFoundError):
        sync_embeddings(store, FakeEmbedder(), register=False)


# ── 出错时不留下半批 ────────────────────────────────────────────────────────


def test_a_short_matrix_raises_and_writes_nothing(store: MemoryStore) -> None:
    """后端少还了几行就报错，**且一条都不写**。

    按下标硬配的后果是 A 的向量挂到 B 上，而召回照样算得出一个数——那比直接失败
    难发现得多。所以这里同时钉两件事：抛错，以及库里没有留下任何一条。
    """
    add(store, "我家的路由器放在书房书架顶层。", created_at=1.0)
    add(store, "周三下午三点开组会。", created_at=2.0)

    with pytest.raises(ValueError, match="数量对不上"):
        sync_embeddings(store, FakeEmbedder(rows=1))

    assert len(store.missing_embedding_ids(SPACE_ID)) == 2


def test_a_vector_wider_than_the_declared_dimension_is_refused(store: MemoryStore) -> None:
    """后端自报 4 维、却还回 8 维的向量时抛错，且一条都不写。

    这不是假想的场景：``space_id`` 里的维度与实际模型的输出不符时，症状正是这个。
    写进去的后果是库里混着两种维度的向量——而**余弦相似度照样算得出一个数**，
    只是那个数没有意义。所以宁可在这里炸。
    """
    from memcore.errors import DimensionMismatchError

    add(store, "我家的路由器放在书房书架顶层。")
    with pytest.raises(DimensionMismatchError):
        sync_embeddings(store, FakeEmbedder(dim=4, vector_dim=8))

    assert len(store.missing_embedding_ids(SPACE_ID)) == 1


def test_an_empty_library_is_not_an_error(store: MemoryStore) -> None:
    """空库照样登记空间。

    "空间登记了但一直是空的"**不是坏状态**，因为**下一次写入就会补上**——登记在这里
    发生，而写入每轮都在发生。若不补上，语义通道会一直返回零条，看起来和"库里没有
    相关的记忆"一样。
    """
    assert sync_embeddings(store, FakeEmbedder()) == 0
    assert store.registry.get(SPACE_ID).dim == 4
