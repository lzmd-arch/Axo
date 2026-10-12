"""不经过界面，验一遍**云端到底收不收图**。

## 它问的是一件模型编不出来的事

脚本自己当场生成一张**纯色**的 PNG（颜色随机挑、同时打在终端上），然后问它
"这张图是什么颜色？只回一个词"。猜不中的话，那张图就没到它那儿——这一条比"你看得见我的
屏幕吗"可靠得多：后者模型可以顺着上下文编一段像样的描述。

2026-10-12 就是这么撞上的：用户在应用里走了一遍"看屏幕"，工具确实截了图、结果行也写着
"随这条结果一起给你"，而模型先描述得头头是道、下一条又改口说"一张图都没到我这儿"。
两种可能（端点收到没当回事 / 模型在编）用这个脚本一次分得开。

用法::

    uv run python tools/smoke_cloud_vision.py              # 用设置里那一份云端配置
    uv run python tools/smoke_cloud_vision.py --dry-run    # 只打请求形状，不发出去
    uv run python tools/smoke_cloud_vision.py deepseek-flash

**它读的是应用自己那份设置**（地址与 key 都从 ``SettingsStore`` 来，与界面走同一条路），
所以不用把 key 打给它。它也不会把 key 打到屏幕上：任何要显示的文本都过一遍
``openai_compat._redact``。

**它认 ``AXO_ISOLATED_STATE``**（同 ``tools/_smoke_*.py`` 那一族的约定）：设了它就去读
那个目录下的配置，而不是使用者的真注册表。测试靠这一条拿到一份自己的配置。

退出码：**0 = 它认出了颜色**（图到了）· **1 = 它说得出话但认错了**（图没到，或者这个
模型不看图）· **2 = 用法/配置不对**（没配云端、没填 key、连不上）· **3 = 它一个字都没
说**（多半是生成额度被"思考"用完了——那一支与"图没到"是两件事，分开报）。
"""

from __future__ import annotations

import base64
import random
import sys

from axo.llm.base import ChatBackendError, ChatMessage, ChatRequest
from axo.llm.openai_compat import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAICompatChatBackend,
    _redact,
)
from axo.ui.app import configure_isolated_state, make_settings
from axo.ui.settings_store import SettingsStore

#: 三个**名字不会混**的颜色。取纯色是为了让"认出来"这件事没有模糊地带：深浅、明暗、
#: 光照都不参与，认得出就是认得出。
_COLOURS = {
    "红": (220, 30, 30),
    "绿": (30, 180, 60),
    "蓝": (40, 80, 220),
}


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """一张纯色 PNG。**手写而不是调 Qt**：这个脚本要能在"Qt 起不来"的机器上单独跑，
    而它验的正是"我们的请求体对不对"——那件事与 Qt 无关。
    """
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _describe(request: ChatRequest) -> list[str]:
    """把"要发出去的请求"写成几行给人看的字（**不发 key、也不发图本身**）。"""
    lines = []
    for message in request.messages:
        if message.images:
            kinds = ["text", "image_url"] if message.content else ["image_url"]
            lines.append(f"  role={message.role:<9} content=分片{len(kinds)}块 {kinds}")
            for data in message.images:
                lines.append(f"      image_url: data:image/png;base64,…（{len(data)} 字符）")
        else:
            head = message.content.replace("\n", " / ")[:60]
            lines.append(f"  role={message.role:<9} content={head!r}")
    return lines


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    named = [item for item in argv if not item.startswith("-")]
    model = named[0] if named else ""

    # **先隔离、再建 QSettings**（同 ``tools/_smoke_*.py`` 那几个脚本）。平时它什么都不做
    # ——``AXO_ISOLATED_STATE`` 没设就直接返回；设了就把设置挪到那个目录下的 ini 里，
    # 于是测试能拿一份自己的配置跑这个脚本，而不去读使用者真注册表里那份。
    configure_isolated_state()
    handle = make_settings()
    settings = SettingsStore(handle).load_backend()
    if not settings.is_cloud:
        print("设置里现在不是云端后端（后端页选「云端」之后再来）。", file=sys.stderr)
        print(
            "本机 Ollama 那条路不用这个脚本：那边收 tool 行的图，走的是另一条线。", file=sys.stderr
        )
        return 2
    if not settings.api_key:
        print("云端后端还没填 API key。", file=sys.stderr)
        return 2
    base_url = settings.cloud_base_url or DEFAULT_OPENAI_BASE_URL
    # 选中的云端模型**不在 BackendSettings 里**：它住在 ``backend/last_model_cloud``
    # 那一格（两个后端各一格，见 ``MainWindow._last_model_key``）。
    model = model or str(handle.value("backend/last_model_cloud", "") or "")
    if not model:
        print("不知道用哪个模型：设置里没选过云端模型，也没在命令行给一个。", file=sys.stderr)
        return 2

    name = random.choice(sorted(_COLOURS))
    rgb = _COLOURS[name]
    data = base64.b64encode(_png(48, 48, rgb)).decode("ascii")
    request = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content="这张图是什么颜色？只回一个词（红 / 绿 / 蓝）。",
                images=(data,),
            )
        ],
        # **额度要留够，别给几十。** ``deepseek-flash`` 是会思考的模型：思考走
        # ``reasoning_content``，与正文**共用这一个额度**。2026-10-12 实测：
        # 给 32 时思考了 1259 字、正文一个字都没轮到，拿回来的是空串——而那看起来
        # 像"图没送到"，完全是另一个结论。给 768 时思考 118 字、正文就是答案。
        max_tokens=768,
    )

    print(f"端点：{base_url}")
    print(f"模型：{model}")
    print(f"这一张色块是：{name}  RGB{rgb}（模型看不到这一行，它只拿到图）")
    print("要发出去的请求：")
    for line in _describe(request):
        print(line)
    if dry_run:
        print("\n--dry-run：没发出去。")
        return 0

    # 不给 ``context_window``：构造函数会自己去问一次 ``GET /models``（这一条顺带也是
    # "端点通不通、key 认不认"的探测）。界面上那条路不能这么做——它在界面线程上。
    backend = OpenAICompatChatBackend(model=model, base_url=base_url, api_key=settings.api_key)
    print("\n正在发…")
    try:
        pieces = [piece.text for piece in backend.stream(request) if piece.text]
    except ChatBackendError as exc:
        print(_redact(str(exc), settings.api_key), file=sys.stderr)
        return 2
    answer = "".join(pieces).strip()
    print(f"它回的是：{answer!r}")

    if not answer:
        # **空答案单独一支**：它既不是"认出了"也不是"认错了"，而是"它压根没说正文"。
        # 混进"没认出"里的话，下一个用这个脚本的人会去查图、查端点，而真正该查的是
        # 那个额度——这台机器上实测过一次，代价是一整轮排查。
        print("\n\n⚠️ 它一个字都没说（不是认错，是没正文）。")
        print("   多半是**生成额度被思考用完了**：会思考的模型把思考也记在 max_tokens 里。")
        print("   把上面那个 max_tokens 调大一点再跑，或者换个不思考的模型试。")
        return 3
    if name in answer:
        print(f"\n✅ 认出「{name}」了 —— 云端这条路的图**送到了**。")
        print("   那应用里那一次「看不到」就是另一回事（比如模型在编，或者那一轮没走这条路）。")
        return 0
    print(f"\n❌ 没认出「{name}」。图没到它那儿，或者这个模型不看图。")
    print(
        "   先确认模型名（官方支持图片输入的是 deepseek-flash），再看端点有没有把 image_url 当回事。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
