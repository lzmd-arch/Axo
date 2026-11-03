"""一次渲染：SIR + 人格 → 最终文本。**失败一律退回草稿。**

## 为什么是"退回"不是"报错"

表达层是锦上添花的一层：它坏了、模型没起、事实没保住——用户要的仍然是那句回答。
所以这里所有失败路径都返回 :attr:`ExpressionOutcome.source == "draft"`，并带一句能
显示给人看的 ``reason``。调用方拿到的永远是**可以说的文本**，不是异常。

## 为什么要核事实

提示词里写了"必须原样保留"，但 0.5B 尺寸的模型写顺了手照样会把 "0.898" 写成
"约 0.9"。这不是风格，是错误——而它发生在用户看不见的地方。所以渲染完**逐条核对**
:attr:`~axo.expression.sir.SIR.facts`，少一条就整段退回草稿：宁可这一轮语气没统一，
也不让一个数被悄悄改掉。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..llm.base import ChatBackend, ChatMessage, ChatRequest
from .sir import SIR, facts_preserved, render_prompt

__all__ = ["ExpressionLayer", "ExpressionOutcome"]


@dataclass(frozen=True)
class ExpressionOutcome:
    """渲染结果。

    Attributes:
        text: 要显示的最终文本。**永远非空**（渲染失败时是草稿）。
        source: ``"styled"``（表达层写的）或 ``"draft"``（退回后端原文）。
        reason: 退回时的一句话；``"styled"`` 时为空串。它进状态栏与日志，**不进正文**。
        prompt_tokens: 这一次渲染用掉的输入 token（后端没报时为 ``None``）。
        completion_tokens: 同上，输出那一侧。
        route: ``"model"``（Phase 1 的模型渲染）或 ``"template"``（Phase 2 的模板）。
        template_id: 模板路径时命中的模板 id；模型路径为空串。
    """

    text: str
    source: str
    reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    route: str = "model"
    template_id: str = ""


class ExpressionLayer:
    """持有渲染后端与人格。**无状态以外的一切都留在这里**，主窗口只调 :meth:`render`。

    Args:
        backend: 渲染用的后端。它可以是任何 :class:`~axo.llm.base.ChatBackend`——
            本版推荐一个独立的本地小模型（Ollama 里的 ``qwen2.5:0.5b`` 之类），因为
            表达层要解决的问题正是"主模型换成了谁"，用同一个大模型渲染只是把同一段
            文体再念一遍。
        persona: 人格文本（L0 那份）。**只作为语气来源**，不参与事实判断。
        max_tokens: 渲染输出的上限。``None`` = 后端默认。
        style_note: 可选的一句风格补充（设置页将来给"风格强度"用）。
        think: 请求里要不要带 ``think`` 开关。**默认 ``False``**：本机 ``qwen3`` 这一
            族会思考，而思考与正文共用同一个输出额度——2026-11-03 实测 256 token 的
            预算下它把额度全花在思考上、正文一片空白（与主对话那个"空回复"是同一个
            坑）。Ollama 认这个顶层字段；OpenAI 兼容端点不认它的话传 ``None``。
    """

    def __init__(
        self,
        backend: ChatBackend,
        persona: str,
        *,
        max_tokens: int | None = None,
        style_note: str = "",
        think: bool | None = False,
    ) -> None:
        self._backend = backend
        self._persona = persona or ""
        self._max_tokens = max_tokens
        self._style_note = style_note
        self._think = think

    def request(self, sir: SIR) -> ChatRequest:
        """渲染请求。**单独一个方法是为了让测试能逐字检查提示词。**"""
        system = self._persona.strip() or "你是千岁（阿螈），说话简短、直接。"
        task = render_prompt(sir, "", style_note=self._style_note).strip()
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=task),
            ),
            max_tokens=self._max_tokens,
            extra={} if self._think is None else {"think": bool(self._think)},
        )

    def render(
        self,
        sir: SIR,
        *,
        keep_going: Callable[[], bool] | None = None,
    ) -> ExpressionOutcome:
        """渲染一段 SIR。**不抛异常**——任何失败都退回 :attr:`SIR.draft`。

        ``keep_going`` 每收到一片调用一次；返回假时停止读取并退回草稿（用户按了停止）。
        """
        draft = sir.draft.strip()
        request = self.request(sir)
        pieces: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            for chunk in self._backend.stream(request):
                if keep_going is not None and not keep_going():
                    return ExpressionOutcome(
                        draft, "draft", "渲染被停止", prompt_tokens, completion_tokens
                    )
                if chunk.text:
                    pieces.append(chunk.text)
                if chunk.prompt_tokens is not None:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    completion_tokens = chunk.completion_tokens
                if chunk.done:
                    break
        except Exception as exc:  # noqa: BLE001 —— 这一层的契约就是"失败退回草稿"
            return ExpressionOutcome(
                draft,
                "draft",
                f"表达层不可用（{type(exc).__name__}: {exc}）",
                prompt_tokens,
                completion_tokens,
            )

        rendered = "".join(pieces).strip()
        if not rendered:
            return ExpressionOutcome(
                draft, "draft", "表达层没有给出文本", prompt_tokens, completion_tokens
            )
        missing = facts_preserved(rendered, sir.facts)
        if sir.intent == "ask" and ("？" not in rendered and "?" not in rendered):
            # **问句不能写成陈述句。** 2026-11-03 实测：渲染把"需要我详细说说吗？"
            # 的句末问号吃掉了——字都对，语气全变了，而用户读起来像被通知而不是
            # 被询问。这一条不靠事实表（问号不是"事实"），单独判。
            # 判据只看 intent，不看草稿末尾有没有问号：降级路径的 draft 会削掉句末
            # 标点（见 ``sir.derive``），拿它比对等于把这一条放空。
            return ExpressionOutcome(
                draft,
                "draft",
                "表达层把问句写成了陈述句",
                prompt_tokens,
                completion_tokens,
            )
        if missing:
            lost = "、".join(missing[:3])
            return ExpressionOutcome(
                draft,
                "draft",
                f"表达层改动了事实（丢了 {lost}），已退回草稿",
                prompt_tokens,
                completion_tokens,
            )
        return ExpressionOutcome(rendered, "styled", "", prompt_tokens, completion_tokens)
