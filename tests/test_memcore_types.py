"""``MemoryItem`` 的派生量。

这个文件存在的理由是一个**一读就炸**的拼写缺陷：``has_embedding`` 读的是
``self.sem_srd``，而字段叫 ``sem_sdr``。一个"语义通道可用的前提"从来没被求过值
——任何调用它的地方都会拿到 AttributeError，而不是 ``False``。

顺带把它的**作用域**钉住：库里的条目 ``sem_sdr`` 恒为空（向量在另一张表里，从不
回写 ``memories`` 行），所以"库里这条有没有向量"不能问它。它回答的是内存态那条。
"""

from __future__ import annotations

from memcore.sdr import encode_lexical
from memcore.types import MemoryItem, MemoryKind, Tier


def make_item(text: str = "一条记忆", **overrides: object) -> MemoryItem:
    payload: dict = {
        "memory_id": "01J000000000000000000000",
        "kind": MemoryKind.FACT,
        "tier": Tier.RECENT,
        "content": text,
    }
    payload.update(overrides)
    return MemoryItem(**payload)


def test_has_embedding_reads_the_field_that_exists() -> None:
    """有向量、有空间 → ``True``。以前这一句抛 AttributeError。"""
    item = make_item(space_id="test/fake@8", sem_sdr=encode_lexical("旧语义占位"))

    assert item.has_embedding is True


def test_has_embedding_is_false_without_a_vector() -> None:
    """没向量是 ``False``，不是异常——这一支以前同样抛。"""
    assert make_item(space_id="test/fake@8").has_embedding is False


def test_has_embedding_needs_both_a_space_and_a_vector() -> None:
    """只有向量没有空间也算没有——语义通道要两者都在才启用（同 ``recall`` 的判据）。"""
    item = make_item(sem_sdr=encode_lexical("旧语义占位"))

    assert item.has_embedding is False
