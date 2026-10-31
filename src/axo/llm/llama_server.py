"""随包的 llama.cpp 运行时（``llama-server``）。

## 为什么要有它

本机的对话模型一直走 **Ollama**：那是一个独立安装的服务，软件只是它的客户端。
用户要求"软件自己调用本地模型、不经过 Ollama"，于是这里把 **llama.cpp 官方
Windows CUDA 构建**随包带上，由软件自己拉起一个私有实例：

* 二进制与 CUDA 运行库都在 ``<bundle>/llama/``（构建时从 ``build/bundle/llama`` 复制）；
* 模型是用户现有的 GGUF 文件（默认就是 Ollama 那一份 blob —— **不需要 Ollama 在跑**，
  它只是一个文件）；
* 监听 ``127.0.0.1`` 上**操作系统分配的空闲端口**，只对本机、只对本进程可见；
* 进程随软件退出而结束（``stop`` 在关闭时调用）。

它仍然是"本机 HTTP"，但不再是"经过 Ollama"：没有 Ollama 安装、没有 11434 端口、
没有它的模型清单，端口是每次启动现取的。

## 与 OpenAI 兼容后端的关系

``llama-server`` 暴露的正是 OpenAI 的 ``/v1/chat/completions``（含 tools 与流式），
所以对话协议直接复用 :class:`~axo.llm.openai_compat.OpenAICompatChatBackend`，
只加一条：**思考默认关**。Qwen3 这类模型带思考时会把输出预算烧在 ``<think>`` 上，
而本项目自 2026-10-10 起把 ``think=False`` 定为默认（实测首字 0.06s→1.0–2.0s、
多步准确率更低）。这里用服务端的 ``chat_template_kwargs.enable_thinking`` 实现，
与 Ollama 那条链的 ``think=False`` 是同一个决定。

## 失败必须说人话

``start`` 失败（没有二进制、模型路径不对、端口起不来、CUDA 初始化失败）一律抛
:class:`LlamaServerError`，消息里带服务端日志的最后几行。界面把它贴给用户就是
"本地模型为什么没起来"，而不是"点发送没反应"。
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import bundled_llama_dir, data_dir
from .openai_compat import OpenAICompatChatBackend

__all__ = [
    "BuiltinLlamaBackend",
    "LlamaServerError",
    "LlamaServerRuntime",
    "find_llama_server",
]

#: 进会话 ``model_ref`` 的前缀。与 ``ollama/``、``openai-compat/`` 同一形状：
#: 翻旧会话时一眼看出那句话出自哪种后端。
MODEL_ID_PREFIX = "llama-cpp"

#: 随包目录里的可执行文件名（与 llama.cpp 官方发行包一致）。
SERVER_EXE = "llama-server.exe"

#: 默认上下文。与 Ollama 那条链的 ``qwen3-4b`` 对齐（8k），可从设置改。
DEFAULT_CONTEXT = 8192

#: 全部层都放 GPU。显存不够时 llama.cpp 自己会退回 CPU 层，并在日志里写清楚。
DEFAULT_GPU_LAYERS = 99


class LlamaServerError(RuntimeError):
    """运行时起不来。消息是**能直接显示给用户**的一句话。"""


def find_llama_server() -> Path | None:
    """随包的 ``llama-server.exe``；没有就 ``None``。"""
    root = bundled_llama_dir()
    if root is None:
        return None
    exe = root / SERVER_EXE
    return exe if exe.is_file() else None


def _free_port() -> int:
    """问操作系统要一个空闲端口。

    **不写死端口**：写死的那个要么撞上别的程序，要么在第二次启动时还被上一次的
    僵尸进程占着——而症状是"第二开次打不开"，看起来像随机故障。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, limit: int = 800) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:].strip()


@dataclass
class LlamaServerRuntime:
    """一个由本进程拥有的 ``llama-server`` 实例。"""

    model_path: Path
    context: int = DEFAULT_CONTEXT
    gpu_layers: int = DEFAULT_GPU_LAYERS
    log_path: Path = field(default_factory=lambda: data_dir() / "logs" / "llama-server.log")
    base_url: str = ""
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _log_handle: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.context = max(512, int(self.context))

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def start(self, timeout: float = 180.0) -> str:
        """起服务并等到 ``/health`` 通过。返回 ``http://127.0.0.1:<port>/v1``。"""
        if self._process is not None and self._process.poll() is None:
            return self.base_url
        exe = find_llama_server()
        if exe is None:
            raise LlamaServerError(
                "随包的 llama.cpp 运行时不在了（找不到 llama-server.exe）。重装这个软件可以补回来。"
            )
        if not self.model_path.is_file():
            raise LlamaServerError(
                f"本地模型文件不存在：{self.model_path}。"
                "在设置 → 后端 → 内置（llama.cpp）里重新选一个 .gguf 文件。"
            )
        port = _free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        command = [
            str(exe),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(self.context),
            "-ngl",
            str(self.gpu_layers),
            "--jinja",
            "--no-webui",
            "--parallel",
            "1",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(exe.parent),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.base_url = f"http://127.0.0.1:{port}/v1"
        deadline = time.monotonic() + timeout
        health = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                detail = _tail(self.log_path)
                self.stop()
                raise LlamaServerError(
                    "本地模型启动失败（进程已经退出）。服务端最后几行：\n"
                    + (detail or "（没有日志）")
                )
            try:
                with urllib.request.urlopen(health, timeout=2) as response:
                    if response.status == 200:
                        return self.base_url
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.25)
        detail = _tail(self.log_path)
        self.stop()
        raise LlamaServerError(
            f"本地模型 {timeout:.0f} 秒内没有就绪。服务端最后几行：\n" + (detail or "（没有日志）")
        )

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            # 句柄关不上不该挡住"进程已经停掉"这个更重要的结局。
            with contextlib.suppress(OSError):
                handle.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None


@dataclass
class BuiltinLlamaBackend(OpenAICompatChatBackend):
    """把本进程拉起的 llama-server 当成一个 OpenAI 兼容端点。

    唯一的协议差别是**思考默认关**（见模块文档）。``extra`` 里显式给了
    ``chat_template_kwargs`` 时以调用方为准。
    """

    def _build_payload(self, request):  # type: ignore[override]
        payload = super()._build_payload(request)
        payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return payload
