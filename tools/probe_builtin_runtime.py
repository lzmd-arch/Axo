"""探针：随包的 llama.cpp 运行时**起得来、答得出、报得出用量**。

它验的是"内置 provider"这条链在**这台机器上**的三件事实，不经过界面、不经过
Ollama、也不碰用户的设置：

1. ``llama-server`` 能从 ``build/bundle/llama``（或 ``AXO_LLAMA_DIR``）起来；
2. 用用户自己的 GGUF 真的答出一句话；
3. 端点上给了 ``usage``——没有它，界面那行"输入 … · 输出 … token"就永远是空的。

用法::

    uv run python tools/probe_builtin_runtime.py --model D:/models/qwen3-4b.gguf
    uv run python tools/probe_builtin_runtime.py --model … --context 4096 --with-tools

退出码 0 = 三件都成立。**这不是测试**：它要真加载几个 GB 的模型、占用显存，
所以不进 pytest；机制那一侧由 ``tests/test_local_llama.py`` 用替身钉住。
"""

from __future__ import annotations

import argparse
import sys

from axo.llm.base import ChatMessage, ChatRequest
from axo.llm.llama_server import BuiltinLlamaBackend, LlamaServerError, LlamaServerRuntime


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF 文件路径（用户自己那份）")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prompt", default="用一句话介绍你自己。")
    parser.add_argument("--with-tools", action="store_true", help="同时声明闲聊轮那组只读工具")
    args = parser.parse_args(argv)

    runtime = LlamaServerRuntime(model_path=args.model, context=args.context)
    try:
        base_url = runtime.start()
    except LlamaServerError as exc:
        print(f"起不来：{exc}", file=sys.stderr)
        return 2
    print(f"运行时就绪：{base_url}")
    try:
        tools = ()
        if args.with_tools:
            from axo.agent.catalog import CHAT_TOOL_NAMES, default_tools

            class _Web:
                available = True

            tools = tuple(
                tool.spec
                for tool in default_tools(weather=object(), web=_Web())
                if tool.spec.name in CHAT_TOOL_NAMES
            )
            print("声明工具：" + "、".join(spec.name for spec in tools))
        backend = BuiltinLlamaBackend(
            model="local", base_url=base_url, api_key="local", context_window=args.context
        )
        request = ChatRequest(
            messages=(ChatMessage(role="user", content=args.prompt),),
            tools=tools,
            max_tokens=256,
        )
        parts: list[str] = []
        prompt_tokens = completion_tokens = None
        calls = ()
        for piece in backend.stream(request):
            parts.append(piece.text)
            if piece.tool_calls:
                calls = piece.tool_calls
            prompt_tokens = piece.prompt_tokens or prompt_tokens
            completion_tokens = piece.completion_tokens or completion_tokens
        print("回答：", "".join(parts).strip() or "（空）")
        if calls:
            print("工具调用：" + "、".join(call.name for call in calls))
        print(f"用量：输入 {prompt_tokens} · 输出 {completion_tokens}")
        ok = bool("".join(parts).strip() or calls) and prompt_tokens is not None
        return 0 if ok else 1
    finally:
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
