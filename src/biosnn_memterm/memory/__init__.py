"""记忆的写入侧。

* :mod:`~biosnn_memterm.memory.policy` —— 什么算一条记忆（纯决策）
* :mod:`~biosnn_memterm.memory.writer` —— 落到库里（唯一写记忆的地方）
* :mod:`~biosnn_memterm.memory.backfill` —— 事后补向量（唯一补语义通道的地方）
* :mod:`~biosnn_memterm.memory.persona` —— 人格锚（L0 层唯一的内容，内置一条 + 用户一条）

**读的那一侧不在这里**，它在 :mod:`memcore.recall`。分开的理由是这一侧要认识
界面（"一轮对话结束了"这件事只有界面知道），而 :mod:`memcore` 永远不该认识界面
——它是一份能脱离终端单独跑和单独测的库。``tests/test_layering.py`` 钉住了这条。
"""

from __future__ import annotations

from .backfill import Embedder, sync_embeddings
from .persona import (
    BUILTIN_PERSONA_VERSION,
    BUILTIN_REF,
    DEFAULT_PERSONA,
    USER_REF,
    builtin_anchor,
    builtin_anchor_version,
    clear_user_persona,
    ensure_builtin_anchor,
    needs_persona,
    persona_texts,
    read_only_reason,
    set_user_persona,
    user_anchor,
)
from .policy import (
    MAX_TASK_TEXT_CHARS,
    MIN_MEANINGFUL_CHARS,
    MemoryDraft,
    MemoryPolicy,
    PerItemPolicy,
    TaskPolicy,
    Turn,
)
from .writer import MemoryWriter

__all__ = [
    "BUILTIN_PERSONA_VERSION",
    "BUILTIN_REF",
    "DEFAULT_PERSONA",
    "MAX_TASK_TEXT_CHARS",
    "MIN_MEANINGFUL_CHARS",
    "USER_REF",
    "Embedder",
    "MemoryDraft",
    "MemoryPolicy",
    "MemoryWriter",
    "PerItemPolicy",
    "TaskPolicy",
    "Turn",
    "builtin_anchor",
    "builtin_anchor_version",
    "clear_user_persona",
    "ensure_builtin_anchor",
    "needs_persona",
    "persona_texts",
    "read_only_reason",
    "set_user_persona",
    "sync_embeddings",
    "user_anchor",
]
