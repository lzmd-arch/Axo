"""模型后端层。

这一层的存在意义是让"换模型"变成一个**可替换的组件**而不是一次改造：

* :mod:`~biosnn_memterm.llm.base` —— :class:`ChatBackend` / :class:`EmbeddingBackend`
  两个协议，字段集对齐 OpenAI 兼容端点
* :mod:`~biosnn_memterm.llm.ollama` —— 本地推理（HTTP，不依赖 ollama 客户端库）
* :mod:`~biosnn_memterm.llm.openai_compat` —— 云端 OpenAI 兼容端点（DeepSeek 等）。

  **它只实现对话，不实现嵌入**：多数云端厂商没有嵌入端点，而记忆的召回、回填、
  探针整条路都靠向量。所以"对话走云端、嵌入仍走本机 Ollama"是这个模块的前提，
  不是它的一个选项。

这一层**只依赖箭头指向 memcore**，反过来不成立。因此这里可以自由地引入
``httpx``、处理 NDJSON、拼 JSON，而 :mod:`memcore` 对这一切一无所知——它拿到的
永远是文本、向量的 numpy 数组，以及一个 ``space_id``。
"""

from __future__ import annotations

from .base import (
    ChatBackend,
    ChatBackendError,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    EmbeddingBackend,
    EmbeddingBackendError,
)

__all__ = [
    "ChatBackend",
    "ChatBackendError",
    "ChatChunk",
    "ChatMessage",
    "ChatRequest",
    "EmbeddingBackend",
    "EmbeddingBackendError",
]

# 刻意不 import .ollama：它会拖进 httpx。只想用协议类型的地方（例如测试里的
# 假后端）不该被迫装上 HTTP 客户端。
