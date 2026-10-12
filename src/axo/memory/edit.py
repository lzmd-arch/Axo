"""改写一条记忆的正文——**以及随之必须重算的词法 SDR**。

这个模块存在，是因为"编辑正文"这件事有一个静默的错法：只改 ``content``，不管
``lex_sdr``。改完之后条目看起来完全正常（详情面板显示新正文、库里的字节也确实
变了），但它带着**旧内容的编码**继续参与打分与 MMR 去冗。表现是：改了正文的那条
记忆，用新词搜不到、用旧词反而搜得到——而没有任何一处报错。

所以重算不是"顺手也做一下"，是这个模块唯一的存在理由：``edited_content`` 的
返回值里，``content`` 与 ``lex_sdr`` **永远同源**。

两个纯函数，都不碰存储：

* :func:`can_encode` —— 能不能改。给界面在动手之前问的。
* :func:`edited_content` —— 改。给 :meth:`memcore.store.MemoryStore.transform`
  当 ``change`` 用（在锁内、要么整条改好要么整条不动）。

**这里不处理语义向量。** 正文变了，旧向量就是旧意思的，必须丢掉交回填管线重算
——但那是 :meth:`~memcore.store.MemoryStore.drop_embedding` 的事，调用方在
``transform`` **成功之后**才做。顺序不能反：先丢向量再改正文，万一 ``transform``
抛了异常，向量就白丢了（而它本来是对的）。
"""

from __future__ import annotations

from memcore import MemoryItem
from memcore.errors import SDRCapacityError
from memcore.sdr import encode_lexical

__all__ = ["can_encode", "edited_content"]


def can_encode(text: str) -> str | None:
    """这段文本能不能当记忆正文。能就返回 ``None``，不能返回一句人能看懂的理由。

    界面在动手之前调它，把两个异常变成人话：空正文（``ValueError``）与全标点
    （``SDRCapacityError``）。**这两种错都不该让用户看到回溯**——"正文里没有任何
    可编码的字符"是一句解释，"SDRCapacityError: 文本 '...' 里没有……"不是。

    判定与 :func:`edited_content` 共用这一处，所以信息框里说的理由，与实际拦下
    它的理由永远是同一条。
    """
    if not text:
        return "正文不能为空。"
    try:
        encode_lexical(text)
    except SDRCapacityError:
        return "正文里没有任何可编码的字符（字、词、数字），只有标点或空白。"
    return None


def edited_content(item: MemoryItem, text: str) -> MemoryItem:
    """返回一条正文已改、**词法 SDR 已按新正文重算**的副本。原对象不动。

    Args:
        item: 被改的那条。
        text: 新正文。

    Returns:
        副本。除 ``content`` 与 ``lex_sdr`` 外一切字段（含 ``weight``、
        ``access_count``、``meta``、``tier``）原样保留——编辑正文不是"重新记一遍"，
        不该动它的年龄、权重与访问史。

    Raises:
        ValueError: 正文为空。
        SDRCapacityError: 正文里没有任何可编码字符。

    两次编码（``can_encode`` 一次、这里一次）是**刻意的**：判据只写在一处，
    代价是几微秒的哈希。换来的是"界面拦下的理由"与"库里抛出的理由"不可能分叉。
    """
    reason = can_encode(text)
    if reason is not None:
        if not text:
            raise ValueError(reason)
        raise SDRCapacityError(reason)
    return item.copy(content=text, lex_sdr=encode_lexical(text))
