"""表达层：后端只负责"说什么"，这里负责"怎么说"。

模块分工：

* :mod:`axo.expression.sir` —— 中间表示与它的一切纯判定（解析、降级、事实核对）；
* :mod:`axo.expression.layer` —— 一次渲染：SIR + 人格 → 最终文本，失败一律退回草稿。

**这一层默认关闭。** 打开它意味着每轮多一次本地模型调用（延迟与内存都涨），换来
"换后端不换说法"；不打开时行为与以前逐字节一致（见 ``tests/test_expression_layer.py``
的直通断言）。
"""

from __future__ import annotations

from .layer import ExpressionLayer, ExpressionOutcome
from .sir import SIR, facts_preserved, parse, render_prompt, style_signature

__all__ = [
    "ExpressionLayer",
    "ExpressionOutcome",
    "SIR",
    "facts_preserved",
    "parse",
    "render_prompt",
    "style_signature",
]
