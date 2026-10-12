"""模型后端的抽象契约。

这个模块是"记忆独立于模型"在代码里的形状。协议里的字段集**刻意对齐 OpenAI
Chat Completions 的最小公共子集**——Ollama、llama.cpp、LM Studio、vLLM、
OpenRouter、以及任何云厂商的兼容端点都能一一映射过来。供应商特有的概念只能
进 :attr:`ChatRequest.extra`，而 ``extra`` **到不了** :mod:`memcore`：记忆核心
拿到的是文本、向量和一个 ``space_id``，仅此而已。

依赖方向是单向的::

    axo.llm  ──▶  memcore
                    ◀── ✗ ──  (memcore 永不 import 本包)

这个方向不能反。反了就意味着记忆核心开始认识具体的模型，而"换模型 = 换读者、
库一字节不变"的承诺随即失效。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ChatBackend",
    "ChatBackendError",
    "ChatChunk",
    "ChatMessage",
    "ChatRequest",
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "Role",
    "ToolCall",
    "ToolSpec",
]

Role = Literal["system", "user", "assistant", "tool"]


class ChatBackendError(RuntimeError):
    """对话后端不可用或返回了错误。

    刻意用 :class:`RuntimeError` 而不是自定义异常树：后端出问题时调用方几乎总是
    只想"告诉用户"（后端没起来、模型不存在、超时），而不是按类型分派处理。
    """


class EmbeddingBackendError(RuntimeError):
    """嵌入后端不可用或返回了错误。"""


# ── 对话 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    """一个可供模型调用的工具。

    ``parameters`` 是 JSON Schema 里的那个 ``object`` 段（``{"type": "object",
    "properties": {...}, "required": [...]}``），**不是完整的工具声明**——外面那层
    ``{"type":"function","function":{...}}`` 外壳是厂商格式，由后端自己套。
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolSpec.name 不能为空——模型就是靠它指定要调哪个工具的。")


@dataclass(frozen=True)
class ToolCall:
    """模型要求调用一个工具。

    **``arguments`` 存的是已解析的对象，不是 OpenAI 那种 JSON 字符串。** 三个理由：

    1. 消费端要 ``dict`` 才能派发，存字符串等于把"每处都要记得 parse"变成一条
       纪律而不是一个类型——迟早有一处忘了解析，而那种错误的表现是"工具收到了
       一个字符串参数"，不会报错。
    2. Ollama 本来就返回对象（实测）。存字符串要先 ``dumps`` 再由后端 ``loads``
       回去，纯往返损耗。
    3. 字符串形态在 OpenAI 那边是**流式碎片累积**的产物，而 Ollama 的
       ``tool_calls`` 整条一次到齐（实测），这个理由在本项目不存在。

    代价是非对象参数无法表示。后端归一化失败时抛 :class:`ChatBackendError`，
    而不是猜一个形状塞进来。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, dict):
            raise TypeError(
                f"ToolCall.arguments 必须是已解析的 dict，收到 {type(self.arguments).__name__}。"
                f"把 JSON 字符串塞进来正是在这个类型上被挡掉的那种混淆。"
            )


@dataclass(frozen=True)
class ChatMessage:
    """一条对话消息。

    ``role`` 用 :data:`Role` 字面量而不是枚举，因为它的取值集合由 OpenAI 兼容
    协议定义、不由本项目定义——多一个枚举层只会让序列化多一次转换。

    Attributes:
        content: 消息正文。**允许为空串**，而且这不是容错而是必需：实测 Ollama
            在给出工具调用时 ``content`` 就是 ``""``。此外
            ``Message(role="assistant", content="")`` 本来就是一条合法的既有状态
            （被中断的空回复，会话存储专门断言过），装配器会把它原样喂回模型。
            所以"assistant 必须有内容"这条校验加不得。
        name: 可选的发言者名。多 agent 场景用，当前单 agent 下恒为 ``None``。
            **与工具无关**：回灌工具结果不需要它（实测只要 ``tool_call_id``），
            所以它的语义保持"多 agent 发言者"不变。
        tool_calls: 模型要求调用的工具。**只在 ``role=="assistant"`` 上合法。**
        tool_call_id: 这条结果对应哪一次调用。**``role=="tool"`` 必须带它**——
            否则模型无法把结果与请求配起来。
    """

    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    images: tuple[str, ...] = ()
    """随这条消息一起送出去的图，**base64 字符串**（不带 ``data:`` 前缀）。

    现在只有一个来源：``look_at_screen`` 的返回行带着一张屏幕截图，模型在这一轮之后的
    请求里能看到它。**它只活在那一轮的工作尾巴里**——那个工具的结果标了
    ``replayable=False``（它是快照），于是不回放、逐字稿里也只剩一行文字，**图永远
    不落盘**。

    用 base64 而不是路径：后端要的就是这个形状（Ollama 的 ``images`` 字段实测收它），
    而"给一条路径让后端自己去读"会把**后端的文件系统可见性**变成一条隐含契约——
    本机后端看得见、云端后端看不见，那种差别不该藏在一条消息里。

    **哪些角色能带**：实测 Ollama 在 ``user`` 与 ``tool`` 行上都收；OpenAI 兼容协议
    只认 ``user`` 的 content 分片，所以那边的后端见到图会**抛**（见
    ``llm/openai_compat.py``），而不是悄悄丢掉。
    """

    def __post_init__(self) -> None:
        # 只校验工具相关的这两条不变量。它们不是风格偏好，是协议层的硬需求：
        # 违反任一条，模型收到的都是一段它无法解释的对话，而失败表现是
        # "模型开始胡说"而不是一个形状错误。
        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                f"只有 assistant 消息能带 tool_calls，这条是 {self.role!r}。"
                f"（``role`` 本身是自由字符串，写错一个字母不会被类型系统挡住。）"
            )
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "role='tool' 的消息必须带 tool_call_id——模型靠它把结果与它发出的那次调用配对。"
            )


@dataclass(frozen=True)
class ChatRequest:
    """一次对话请求。

    Attributes:
        messages: 按时间顺序排列的消息。至少一条。
        temperature: 采样温度。
        max_tokens: 回复的最大 token 数。``None`` 表示用后端默认。
        stop: 停止串。
        tools: 本轮可用的工具。空元组表示不启用工具调用——**这是当前的默认**，
            所以不带工具的请求与加这个字段之前逐字节一致。
        extra: 供应商特有参数的透传槽。**不许把它读进记忆核心的任何逻辑**——
            它是"厂商概念不许逃出后端层"这条纪律的泄压阀。
    """

    messages: Sequence[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stop: Sequence[str] = ()
    tools: Sequence[ToolSpec] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ChatRequest.messages 不能为空——至少要有当前这一轮用户输入。")


@dataclass(frozen=True)
class ChatChunk:
    """流式回复的一片。

    同一个对象承载两种事件：增量文本片（:attr:`text` 非空）与流结束标记
    （:attr:`done` 为真）。之所以不拆成两个类型，是因为消费端（Qt 的
    ``QThread``）通常只关心"这一片有没有字要显示"，多一层类型判断只是噪声。

    Attributes:
        text: 本片的增量文本。可能为空串（例如只剩统计信息的结尾片，以及模型
            改为发出工具调用时的那些片）。
        done: 是否为最后一片。
        prompt_tokens: 输入 token 数。**只有最后一片带**，且由后端在上报时给出；
            服务端不报时为 ``None``。装配预算不能依赖它——预算是发请求**之前**
            的事。
        completion_tokens: 输出 token 数。同上。
        tool_calls: 本片携带的工具调用。**整条一次到齐，不分片**——Ollama 实测
            如此（它不像 OpenAI 那样按 index 累积 JSON 字符串碎片），所以消费端
            **不需要**任何碎片累积逻辑，直接在 ``done`` 那一片上取全部即可。
            刻意不给 :class:`ChatChunk` 加 ``thinking`` 之类的字段：那是厂商概念，
            它的出口是 :attr:`raw` 与 :attr:`ChatRequest.extra`。
        raw: 后端的原始事件，供调试与将来扩展。消费端不应依赖它的结构。
    """

    text: str = ""
    done: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatBackend(Protocol):
    """对话后端。"""

    model_id: str
    """稳定标识，形如 ``"ollama/deepseek-r1:4b"``。

    它同时是**注入账本里的模型指纹**：换了模型，账本里这一栏就变，而记忆库不变。
    前缀是后端类型，冒号后是后端内部的模型名。
    """

    context_window: int
    """上下文窗口（token 数）。装配预算以它为基准。"""

    def count_tokens(self, text: str) -> int | None:
        """尽力给出精确的 token 数；给不出时返回 ``None``，由调用方回退到启发式。

        返回 ``None`` 不是失败，是**契约的一部分**——多数本地后端在发请求前没有
        廉价的分词接口。调用方必须处理这个分支。
        """
        ...

    def stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        """流式生成回复。

        实现必须是**同步生成器**。理由：记忆写入是 CPU 上的同步 numpy 运算
        （SDR 编码、矩阵更新），把它塞进 asyncio 只会引入线程安全的麻烦而不带来
        任何吞吐收益；消费端在 Qt 侧，本来就要用 ``QThread`` + signal。

        出错时抛 :class:`ChatBackendError`——**在生成器第一次迭代时抛**，不是
        在调用 ``stream`` 时抛。这意味着"后端没启动"这类错误会出现在打开流的位置，
        调用方需要把 ``next()`` 也包在 try 里。
        """
        ...


# ── 嵌入 ────────────────────────────────────────────────────────────────────


class EmbeddingBackend(Protocol):
    """嵌入后端。

    这个协议只有三个成员，其中 :attr:`space_id` 是**整个"记忆独立于模型"设计的
    承重墙**：向量的数值含义完全由产生它的模型决定，因此每个向量都必须连同它
    所属的空间标识一起被记账。两个不同模型的向量放在一起算余弦相似度不会报错，
    只会给出无意义的数——所以 :mod:`memcore` 把 ``space_id`` 作为**必需参数**，
    任何写入路径都躲不掉。
    """

    space_id: str
    """记忆空间标识，形如 ``"ollama/nomic-embed-text:v1.5@768"``。

    约定包含三层信息：**来源 / 模型名 / 维度**。维度写进标识是为了让"维度变了"
    这件事在字符串比较层面就暴露，而不必等到矩阵乘法报形状错误。
    """

    dim: int
    """向量维度。"""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本编成 ``(len(texts), dim)`` 的 float32 矩阵，**已 L2 归一化**。

        返回归一化向量是硬约定：下游的"余弦相似度"因此退化成点积，整个召回路径
        少一次逐行除法，也少一处范数为零的除零风险。

        空输入返回 ``(0, dim)`` 的空矩阵，不报错——批量回填时"这批全是空串"是
        合法情形。
        """
        ...
