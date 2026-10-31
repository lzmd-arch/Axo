"""探针：闲聊轮带上只读工具后，模型会不会**自己伸手**。

2026-10-31 的失败现场：用户问「看看最近有些什么新闻」，路由判成闲聊，请求里连
``tools`` 都没有，模型只能回答"我手上没有能连出去的接口"。修法是把
``CHAT_TOOL_NAMES`` 里的只读工具在闲聊轮也声明出去（见 ``main_window._chat_tools``）。

这个探针只验**模型这一侧**：给了这组工具，它会不会调、调得对不对。工具执行那段
（联网、天气）不在这里跑——那需要 key 与网络，且与"自主调用"是两件事。

用法::

    uv run python tools/probe_chat_tools.py            # 默认 qwen3-4b:latest
    uv run python tools/probe_chat_tools.py --model X

判据：前两条必须分别调出 ``web_search`` 与 ``get_weather``；负对照一条都不该调。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from axo.agent.catalog import CHAT_TOOL_NAMES, default_tools
from axo.agent.tools import ToolRegistry
from axo.llm.base import ChatMessage, ChatRequest
from axo.llm.ollama import OllamaChatBackend


class _Web:
    """``default_tools`` 只问它要 ``available``。探针不真的联网。"""

    available = True


@dataclass(frozen=True)
class _Case:
    label: str
    prompt: str
    expect: tuple[str, ...]


_CASES = (
    _Case("新闻", "看看最近有些什么新闻", ("web_search",)),
    _Case("天气", "今天天气怎么样", ("get_weather",)),
    _Case("时间", "现在几点？", ("get_time",)),
    _Case("负对照", "今天有点累，随便聊两句吧。", ()),
)


def _chat_registry() -> ToolRegistry:
    tools = [
        tool
        for tool in default_tools(weather=object(), web=_Web())
        if tool.spec.name in CHAT_TOOL_NAMES
    ]
    return ToolRegistry(tools)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-4b:latest")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    registry = _chat_registry()
    print("offered:", ", ".join(registry.names))
    backend = OllamaChatBackend(model=args.model)
    failures = 0
    for case in _CASES:
        messages = (
            ChatMessage(role="system", content="你是千岁，说话简短、直接。"),
            ChatMessage(role="user", content=case.prompt),
        )
        request = ChatRequest(
            messages=messages,
            tools=registry.specs(),
            extra={"think": False},
        )
        parts: list[str] = []
        calls = ()
        for piece in backend.stream(request):
            if piece.text:
                parts.append(piece.text)
            if piece.tool_calls:
                calls = piece.tool_calls
        names = tuple(call.name for call in calls)
        ok = names == case.expect if case.expect else not names
        print(f"\n[{case.label}] {case.prompt}")
        print(f"  calls: {names or '（无）'}")
        print(f"  text : {''.join(parts).strip()[:120]}")
        if args.verbose:
            for call in calls:
                print(f"  args[{call.name}]: {call.arguments}")
        print("  PASS" if ok else f"  FAIL（期望 {case.expect or '不调'}）")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
