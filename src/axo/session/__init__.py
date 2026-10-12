"""会话层：逐字稿、token 算术、分层装配、滚动摘要。

四件事，一笔请求的完整来回：

* :mod:`~axo.session.store` —— 会话逐字稿的读写
* :mod:`~axo.session.budget` —— token 算术与额度
* :mod:`~axo.session.assemble` —— 分层装配与注入账本
* :mod:`~axo.session.summarize` —— 被逐出的历史压成摘要
* :mod:`~axo.session.environment` —— L1 那一槽的正文

数据流是单向的，每一段都能单独测：

    逐字稿 ──▶ budget 算额度 ──▶ assemble 分层取舍 ──▶ ChatRequest
                    ▲                    │
                    └── summarize ◀──────┘（被逐出的那些）

**这个包保存的是逐字稿，不是记忆。** 记忆在 :mod:`memcore` 里，带权重、分层与
关联；这里只是"聊天记录"。分开的理由见 :mod:`~axo.session.store` 的
模块文档——简言之，删聊天记录不该有概率删掉长期记忆。
"""

from __future__ import annotations

from .assemble import (
    Assembly,
    AssemblyLedger,
    Layer,
    LayerReport,
    MemoryInjection,
    assemble,
    layer_ceiling,
)
from .budget import CONVERSATION_BASE, PER_MESSAGE_OVERHEAD, TokenBudget, TokenMeter
from .environment import build_environment
from .store import Message, Session, SessionStore, new_session_id
from .summarize import summarize

__all__ = [
    "CONVERSATION_BASE",
    "PER_MESSAGE_OVERHEAD",
    "Assembly",
    "AssemblyLedger",
    "Layer",
    "LayerReport",
    "MemoryInjection",
    "Message",
    "Session",
    "SessionStore",
    "TokenBudget",
    "TokenMeter",
    "assemble",
    "build_environment",
    "layer_ceiling",
    "new_session_id",
    "summarize",
]
