"""把库里还没向量的条目补上向量 —— 语义通道的唯一补账者。

**为什么这件事必须有一个专门的地方。** 写入是**不调模型**的（见
:mod:`~biosnn_memterm.memory.writer`：说一句话不该变成一次模型调用），所以一条
记忆进库时只有词法 SDR，向量是事后补的。补向量这条路只要断一次，库里就会留下
一批"只有词法通道"的记忆——召回质量下降，**而没有任何一处报错**。

所以它是可查询的（``store.missing_embedding_ids``）、幂等的（补过的不会再补）、
且**每次启动都跑一遍**：断过一次的库不需要人工干预就能自愈。

**登记空间与补向量是同一件事的两半**，所以放在同一个函数里。分开的话就会有一个
"空间登记了但一直是空的"状态，而它的症状——语义通道总是返回零条——看起来和
"库里没有相关的记忆"一模一样。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from memcore.spaces import Space, parse_space_id
from memcore.store import MemoryStore

__all__ = ["Embedder", "sync_embeddings"]


class Embedder(Protocol):
    """补向量需要嵌入后端提供的全部东西。

    刻意不复用 :class:`~biosnn_memterm.llm.base.EmbeddingBackend`：那个协议是
    HTTP 后端的契约（它还要 ``dim`` 之外的若干约定），而这里要的只是"给我一批
    文本，还我一堆向量"。窄协议让测试能塞一个十行的假件进来——而这个模块的正确性
    （补谁、补多少、断了怎么办）与 HTTP 无关。
    """

    @property
    def space_id(self) -> str:
        """这些向量属于哪个记忆空间。"""
        ...

    @property
    def dim(self) -> int:
        """向量维度。"""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本编成 ``(len(texts), dim)`` 的 float32 矩阵。"""
        ...


def sync_embeddings(
    store: MemoryStore,
    embedder: Embedder,
    *,
    limit: int | None = None,
    register: bool = True,
) -> int:
    """给 ``embedder`` 空间里还缺向量的记忆补上，返回补了几条。

    Args:
        store: 记忆库。
        embedder: 嵌入后端。
        limit: 这一次最多补几条。``None`` 表示补完为止。启动时补全部，每轮对话
            后补一小批——后者是为了让刚写下的那条在**下一轮**就能被语义召回，
            而不是等到下次重启。
        register: 是否顺手登记空间。默认要——第一次跑的时候库里没有任何空间，
            而"先登记再写向量"的顺序是 ``set_embedding`` 强制的。

    Returns:
        真正写进去的条数。**已经补过的不会被再算一次**：判据是
        ``missing_embedding_ids``，它问的是"这个空间里没有向量的条目"。

    Raises:
        SpaceMismatchError: 维度与已登记的空间不符。**不吞**：那说明当前用的嵌入
            模型与库里的向量不是同一个，继续补下去会把两个模型的向量倒进同一个
            空间——而那种库里，余弦相似度算得出数，只是那个数没有意义。
    """
    space = _ensure_space(store, embedder) if register else store.registry.get(embedder.space_id)
    pending = store.missing_embedding_ids(space.space_id, limit=limit)
    if not pending:
        return 0

    # 按库里的顺序取正文。**取不到的跳过而不是报错**：扫描与写回之间用户可能在
    # 记忆浏览器里删了一条（P2），而"因为少了一条正文就整批不补"会让剩下几百条
    # 一起停在那里——那正是这个模块存在的意义。
    items = [
        (memory_id, item) for memory_id in pending if (item := store.get(memory_id)) is not None
    ]
    if not items:
        return 0

    # 一次把它们全交给后端，由它自己分片（``OllamaEmbeddingBackend.batch_size``）。
    # 在这里再切一次会让"批大小"有两个出处，而调过其中一个的人不会知道另一个。
    matrix = embedder.embed([item.content for _, item in items])
    if matrix.shape[0] != len(items):
        raise ValueError(
            f"嵌入后端返回了 {matrix.shape[0]} 条向量，而请求的是 {len(items)} 条。"
            "数量对不上说明向量与记忆的对应关系已经断了——这种情况下**不能**按下标"
            "硬配，那会把 A 的向量挂到 B 上，而召回照样算得出一个数。"
        )

    written = 0
    for (memory_id, _), vector in zip(items, matrix, strict=True):
        store.set_embedding(memory_id, space.space_id, vector)
        written += 1
    return written


def _ensure_space(store: MemoryStore, embedder: Embedder) -> Space:
    """登记空间（已登记且维度一致时是幂等的）。

    ``model_ref`` 从 ``space_id`` 里拆出来，而不是让调用方另传一个参数：两处各写
    一遍模型名，迟早会有一处忘了改，而 ``model_ref`` 是记忆浏览器里唯一能告诉
    用户"这些向量是哪个模型留下的"的东西。
    """
    source, model, _ = parse_space_id(embedder.space_id)
    return store.register_space(
        Space(space_id=embedder.space_id, dim=int(embedder.dim), model_ref=f"{source}/{model}")
    )
