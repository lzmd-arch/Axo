"""测试替身。

**为什么需要一个会吐工具调用的假后端**：真实的工具调用路径在核心腿里跑不了
（要么联网、要么要 Ollama）。而核心腿要断言的恰恰是循环的逻辑——"工具被调了几次"
"第 2 次请求里有没有那条工具结果""轮数用尽后有没有收尾请求"。这些在真后端上
既慢又不可复现（同一个 prompt 两次可能一次调工具一次不调，2026-10-10 实测多步
用例 5 次里只对 1 次）。

所以：把"模型这一侧"整个换成脚本，循环要断言的每一条都变成确定性的。
真实模型的行为由 ``tools/smoke_agent.py`` 负责量（那是量具，不是测试）。

**它同时也是一台录音机。** :attr:`ScriptedBackend.requests` 记下每一次收到的
:class:`ChatRequest` —— 反空转断言全都依赖它："第 2 次请求里含那条工具结果"
这种话没法靠返回值证明，只能翻请求记录。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from axo.llm.base import ChatBackendError, ChatChunk, ChatRequest, ToolCall

__all__ = ["ScriptedBackend", "text_chunk", "tool_chunk"]


def text_chunk(text: str, *, done: bool = False, **tokens: int | None) -> ChatChunk:
    """一片纯文本。``done=True`` 的那片通常还要带上 token 统计。"""
    return ChatChunk(text=text, done=done, **tokens)  # type: ignore[arg-type]


def tool_chunk(*calls: ToolCall, done: bool = True) -> ChatChunk:
    """一片工具调用。

    **默认 ``done=True``**，因为真实 Ollama 就是这么发的：2026-10-10 的实测字节
    （``tests/fixtures/ollama_tool_calls.json``）里 ``tool_calls`` 与 ``done: true``
    在**同一片**。把它拆成两片会让假替身比真后端更"宽容"，于是循环里那种"只在
    done 片上取工具调用"的写法能过测试却过不了真实后端。
    """
    return ChatChunk(text="", done=done, tool_calls=tuple(calls))


@dataclass
class ScriptedBackend:
    """按脚本逐条吐片的假后端，同时记录收到的每个请求。

    ``script`` 是**每次** ``stream()`` 调用的返回内容：第 N 次调用吐第 N 组片。
    调用次数超出脚本长度时，最后一条**重复**——这样"循环没有按预期停下来"的表现
    是断言失败（比如请求次数对不上），而不是一个 StopIteration 把测试炸成
    与被测逻辑无关的形状。

    ``raise_at`` 用来测错误路径：第几次调用（0 基）在**首次迭代**时抛
    :class:`ChatBackendError`。照 :class:`ChatBackend` 的契约，错误在第一次
    ``next()`` 时抛，不是在 ``stream()`` 时。
    """

    script: Sequence[Sequence[ChatChunk]]
    model_id: str = "fake/scripted"
    context_window: int = 8192
    tokens: int | None = None
    raise_at: int | None = None
    requests: list[ChatRequest] = field(default_factory=list)

    def count_tokens(self, text: str) -> int | None:
        """返回预设值（默认 ``None``，即"给不出精确数"这个合法的契约分支）。"""
        return self.tokens

    def stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        """记下请求，返回它的片。

        **录制是即时的，报错是延迟的**——这个不对称是刻意的：
        ``requests`` 在 ``stream()`` 被调用的当下就追加，于是 :attr:`calls` 数的是
        "循环发了几轮请求"（循环创建了生成器却不迭代，说明它坏了，断言应该看见）；
        而 :class:`ChatBackendError` 留到第一次 ``next()`` 才抛，因为
        :class:`ChatBackend` 的契约就是这么写的（"在生成器第一次迭代时抛"）。
        把两者都做成即时的，会造出一个比真后端更宽松的替身。
        """
        self.requests.append(request)
        return self._pieces(len(self.requests) - 1)

    def _pieces(self, index: int) -> Iterator[ChatChunk]:
        if self.raise_at is not None and index == self.raise_at:
            raise ChatBackendError(f"脚本化的后端错误（第 {index} 次调用）")
        # 脚本用尽后重复最后一条：让"循环没按预期停下"表现为**请求次数对不上**
        # 这种能看懂的断言失败，而不是一个 StopIteration。
        yield from self.script[min(index, len(self.script) - 1)]

    # ── 断言辅助 ──────────────────────────────────────────────────────────

    @property
    def calls(self) -> int:
        """``stream`` 被调用了几次 == 循环发了几轮请求。"""
        return len(self.requests)

    def tools_declared(self, index: int = 0) -> tuple[str, ...]:
        """第 ``index`` 次请求里声明了哪些工具。"""
        return tuple(spec.name for spec in self.requests[index].tools)

    def last_text(self, index: int) -> str:
        """把第 ``index`` 次请求里所有消息的正文拼起来（含 tool 行的结果）。

        反空转断言"第 2 次请求里含那条工具结果的正文"就用它——不检查工具行，
        只检查"工具结果确实被回灌进了上下文"这件事。
        """
        return "\n".join(message.content for message in self.requests[index].messages)

    def roles(self, index: int) -> list[str]:
        """第 ``index`` 次请求里各条消息的角色，按顺序。"""
        return [message.role for message in self.requests[index].messages]
