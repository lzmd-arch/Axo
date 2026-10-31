"""低置信度时她说的那几句。

**这些是"话术"，不是"回答"**：当用户问的是记忆里的事、而记忆里确实一条都没有时，
由程序挑一句说出来，不再让模型去编。措辞刻意留了"再问一句"的口子（"能再说具体一点
吗"），因为那个场合的真实状态就是"我没找到，但也许是我找的方式不对"。

**只有一条路径会用到它**：`:meth:`MainWindow._should_say_uncertain`——闲聊轮与工具轮
都不走。工具轮不走是因为模型还能自己去查 ``search_memory``；闲聊轮不走是因为"今天有
点累"本来就不该要求有记忆。这条边界的理由写在
``docs/prereg-recall-threshold-v1.md`` 里。
"""

from __future__ import annotations

import random

__all__ = ["UNCERTAIN_PHRASES", "pick_phrase"]

#: 六句，按"最不像机器"到"最直接"排。**不要写"我没有相关记忆"这种系统腔**——
#: 她是在跟人说话，不是在报错。
UNCERTAIN_PHRASES: tuple[str, ...] = (
    "关于这个我有几个不同的印象——能再说具体一点吗？",
    "这个我不太熟悉，能从头讲讲吗？",
    "你之前好像提过类似的事，但细节我记不清了——再提醒一下？",
    "我好像懂一点，但不确定是不是你想说的那个——多讲几句？",
    "这部分我的记忆有点模糊，能确认一下吗？",
    "等等，我好像有几个不同版本的记忆——哪个是对的？",
)


def pick_phrase(rng: random.Random | None = None) -> str:
    """随机挑一句。``rng`` 可注入——测试与复现要的是同一句，而不是"大概是这样"。"""
    return (rng or random).choice(UNCERTAIN_PHRASES)
