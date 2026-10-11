"""不经过界面，直接验一遍对话后端的真实链路。

界面出问题时，第一个要回答的问题永远是"是后端的事还是渲染的事"。这个脚本把
后半段完全排除在外：它只调 :mod:`biosnn_memterm.llm`，把流式增量打到 stdout。

用法::

    uv run python tools/smoke_backend.py                    # 用第一个可用的模型
    uv run python tools/smoke_backend.py qwen3-4b           # 指定模型
    uv run python tools/smoke_backend.py --list             # 只列模型
"""

from __future__ import annotations

import sys
import time

from biosnn_memterm.llm.base import ChatBackendError, ChatMessage, ChatRequest
from biosnn_memterm.llm.ollama import (
    OllamaChatBackend,
    list_local_models,
    pick_chat_model,
    probe,
)

PROMPT = "用一句话说明你是谁，再报一下现在的时间——如果不知道时间就直说不知道。"


def main(argv: list[str]) -> int:
    if "--list" in argv:
        models = list_local_models()
        if not models:
            print("本地一个模型都没有。")
            return 1
        for entry in models:
            size_gb = entry["size"] / 1024**3
            name = entry.get("short_name") or entry["name"]
            print(
                f"  {name:<24} {entry['parameter_size']:>6} "
                f"{entry['quantization']:<10} {size_gb:.2f} GB"
            )
        return 0

    version = probe()
    if version is None:
        print("Ollama 没在运行。启动命令：ollama serve")
        return 1
    print(f"Ollama {version}")

    models = list_local_models()
    if not models:
        print("Ollama 在运行，但没有模型。")
        return 1

    wanted = next((arg for arg in argv[1:] if not arg.startswith("-")), None)
    # 按 short_name 匹配：用户打的是 `qwen3-4b`，而 /api/tags 返回 `qwen3-4b:latest`。
    short = {entry.get("short_name") or entry["name"]: entry["name"] for entry in models}
    if wanted is None:
        # **不能取 `models[0]`**：列表按名字排序，队首是 `nomic-embed-text`——一个
        # 不能对话的模型。拿它去发 /api/chat，报回来的错与"这个脚本要验的东西"毫无
        # 关系，而根因是选择本身就不该从这里取。挑不到的说明只有嵌入模型。
        model = pick_chat_model(models)
        if not model:
            print(
                f"Ollama 里有 {len(models)} 个模型，但一个能对话的都没有（全是嵌入模型）。"
                f"\n装一个对话模型，或用 --list 看一眼现有的，再显式传模型名。"
            )
            return 1
    elif wanted in short:
        model = short[wanted]
    elif wanted in {entry["name"] for entry in models}:
        model = wanted
    else:
        print(f"没有模型 {wanted!r}。可用：{', '.join(sorted(short))}")
        return 1

    backend = OllamaChatBackend(model=model)
    print(f"模型 {backend.model_id}，上下文窗口按 {backend.context_window} token 计")
    print(f"用户：{PROMPT}")
    print("-" * 60)

    request = ChatRequest(messages=[ChatMessage(role="user", content=PROMPT)])
    started = time.monotonic()
    first_token_at: float | None = None
    pieces = 0

    try:
        for chunk in backend.stream(request):
            if chunk.text:
                if first_token_at is None:
                    first_token_at = time.monotonic() - started
                pieces += 1
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
            if chunk.done:
                print()
                print("-" * 60)
                elapsed = time.monotonic() - started
                print(
                    f"首字 {first_token_at:.2f}s · 共 {elapsed:.2f}s · "
                    f"{pieces} 片 · 输入 {chunk.prompt_tokens} token · "
                    f"输出 {chunk.completion_tokens} token"
                )
    except ChatBackendError as exc:
        print(f"\n失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
