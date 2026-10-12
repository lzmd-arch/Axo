"""后台工作线程。

界面线程绝不能被网络等待阻塞——哪怕只是半秒，用户也会看到窗口"卡住"。因此所有
后端调用都在这两个 QObject 里跑，通过信号把结果送回界面线程。Qt 的信号跨线程是
队列式的（``Qt.AutoConnection`` 在收发双方属于不同线程时自动变成
``QueuedConnection``），所以这里的槽函数里可以放心更新控件。

**为什么用同步生成器 + QThread，而不是 asyncio**：记忆写入是 CPU 上的同步 numpy
运算（SDR 编码、矩阵更新）。把它塞进事件循环只会引入"这段计算要不要 ``await``"
的麻烦，而不带来任何吞吐收益——瓶颈在本地模型推理，不在 IO 并发。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..llm.base import ChatBackend, ChatBackendError, ChatRequest
from ..llm.ollama import list_local_models
from ..llm.openai_compat import list_remote_models
from .settings_store import BackendSettings

__all__ = ["ChatWorker", "ModelListWorker", "ToolCallsWhereNoneExpected"]


class ToolCallsWhereNoneExpected(RuntimeError):
    """无工具路径收到了工具调用——接线错了。

    独立一个异常类型而不是复用 :class:`ChatBackendError`：这条**不是后端的错**，
    是我们自己把一轮请求交给了错误的 worker。混用会让日志读起来像"模型有问题"，
    而真正要改的是调用点。它会被 :meth:`ChatWorker.run` 转成 ``failed`` 信号。
    """


class ChatWorker(QObject):
    """跑一次流式对话。

    **它跑的是"不带工具"的那一轮。** 带工具的一轮该由 ``AgentWorker`` 处理
    （见 P3a 计划 §二）。这个分工不是风格问题：这里的循环只会转发文本，
    模型要是发了工具调用，它无从执行、也无从回灌结果。
    """

    chunk = Signal(str)
    """一段增量文本。"""

    finished = Signal(str, dict)
    """生成结束：``(完整文本, 统计)``。

    即使中途出错或用户点了停止也会发出——界面据此复位状态。``statistics`` 里
    ``interrupted`` 为真表示是用户停的，``prompt_tokens`` / ``completion_tokens``
    可能为 ``None``（后端未上报）。
    """

    failed = Signal(str)
    """出错，参数是可直接显示给用户的句子。"""

    def __init__(self, backend: ChatBackend, request: ChatRequest) -> None:
        super().__init__()
        self._backend = backend
        self._request = request
        self._stop_requested = False

    @Slot()
    def stop(self) -> None:
        """请求停止。

        **不走信号**：界面线程直接调用这个方法（普通函数调用），所以它就在界面线程里
        执行，只是把一个布尔翻成真。工作线程在每收到一片增量后读它。少了那层排队，
        收尾路径上「事件循环已经在退出、排队的东西未必还能被派发」这个问题也就不存在。

        这意味着停止不是即时的：当前这一片之后才会退出，而且若模型久久不吐字，
        界面看起来像没反应。这是刻意的取舍——``httpx`` 的连接一旦发出去就无法
        从外部安全地掐断，硬杀线程会留下半关闭的连接。用户体验上表现为"等最后
        一句话说完"，可以接受；后端在客户端断开后也会自行停止生成。
        """
        self._stop_requested = True

    @Slot()
    def run(self) -> None:
        """在线程里执行。由 ``QThread.started`` 触发。"""
        parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        interrupted = False

        try:
            for piece in self._backend.stream(self._request):
                if piece.tool_calls:
                    # **绝不静默丢弃。** 这一行以前不存在，于是模型调了工具的那一刻，
                    # 界面把它当空文本：用户看到一句没头没尾的话（或者一片空白），
                    # 而**没有任何地方报错**——比报错更坏，因为它没有症状可查。
                    #
                    # 走到这里只有两种可能，都是接线问题，都必须立刻看得见：
                    #   1. 请求里带了工具，却被交给了不处理工具的 ChatWorker；
                    #   2. 请求里没带工具，模型却仍发出了 tool_calls（协议意外）。
                    # 抛而不是 return：``return`` 会让 try/except/else 的 ``else``
                    # 也执行，于是 ``failed`` 之后还会再发一次 ``finished``——正是
                    # 下面那条注释警告的双发。
                    raise ToolCallsWhereNoneExpected(
                        f"这一轮收到 {len(piece.tool_calls)} 个工具调用"
                        f"（{', '.join(call.name for call in piece.tool_calls)}），"
                        f"但请求里{'带了' if self._request.tools else '没有'}工具声明，"
                        "而 ChatWorker 不处理工具调用。带工具的一轮应当交给 AgentWorker。"
                    )
                if piece.text:
                    parts.append(piece.text)
                    self.chunk.emit(piece.text)
                if piece.done:
                    prompt_tokens = piece.prompt_tokens
                    completion_tokens = piece.completion_tokens
                if self._stop_requested:
                    interrupted = True
                    break
        except ToolCallsWhereNoneExpected as exc:
            self.failed.emit(str(exc))
        except ChatBackendError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — 兜底，绝不把线程炸掉
            # 一个未捕获的异常会静默终止这个线程，界面永远停在"生成中"。
            # 这里宁可把陌生异常的原文交出去，也不要让界面卡死。
            self.failed.emit(f"对话过程中出现未预期的错误：{type(exc).__name__}: {exc}")
        else:
            # 用 try/except/else 而不是在 try 末尾 emit：出错路径**不该**发
            # finished，否则界面会先复位成空闲、再收到 failed，两处都要复位状态。
            # 现在契约很干净：要么 finished，要么 failed，二选一。
            self.finished.emit(
                "".join(parts),
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "interrupted": interrupted,
                },
            )


class ModelListWorker(QObject):
    """拉取模型清单。

    **清单从哪来取决于后端类型。** 本机走 Ollama 的 ``/api/tags``，云端走 OpenAI
    兼容的 ``/models``——两处不只地址不同，返回的形状也不同，由各自的函数归一化到
    同一套键（``name`` / ``short_name`` / ``capabilities`` …），所以这个 worker
    出去的东西对上层是一样的。

    单独一个 worker 而不是直接在主线程调：后端没启动时连接超时是好几秒，而
    "启动时窗口出现慢 5 秒"是最典型的"明明是网络问题却看起来像程序卡死"。
    """

    loaded = Signal(list)
    """成功，参数是 ``[{name, size, parameter_size, ...}, ...]``。"""

    failed = Signal(str)

    def __init__(self, backend: BackendSettings) -> None:
        """收的是**整个后端设置**，不是一个地址。

        两个理由：云端要知道 base_url + key（两个字段装不进一个 ``host``），
        而"这一次清单是从哪儿拉的"与"当前后端是什么"必须是**同一个事实**——
        分成两组参数传，就有"传了云端地址、provider 还是 ollama"这种半路的组合，
        而它的症状是去问一个不存在的本机服务。
        """
        super().__init__()
        self._backend = backend

    @Slot()
    def run(self) -> None:
        try:
            self.loaded.emit(self._fetch())
        except ChatBackendError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"获取模型列表时出现未预期的错误：{type(exc).__name__}: {exc}")

    def _fetch(self) -> list[dict[str, Any]]:
        """按后端类型分派。

        **抽成一个方法而不是写两个 worker**：失败处理、线程收尾、"为什么是空的"
        那一句都由 :meth:`run` 统一负责，分派只决定去问谁。
        """
        if self._backend.is_cloud:
            return list_remote_models(self._backend.cloud_base_url, api_key=self._backend.api_key)
        # ``allow_remote`` **必须一路传到底**。``llm/ollama.py`` 的 ``_client`` 对非
        # 本机地址会直接抛错，所以漏传任何一处，那条路径的症状都是"某个功能莫名说
        # 连不上，而别的功能好好的"。
        return list_local_models(self._backend.host, allow_remote=self._backend.allow_remote)
