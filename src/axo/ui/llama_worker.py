"""在后台线程里拉起随包的 llama-server。

为什么必须离开界面线程：模型加载是几秒到几十秒的磁盘读取 + CUDA 初始化。放在槽里
做，窗口在那段时间**完全无响应**——与"程序卡死"同一个观感（2026-10-31 那条
"卡在检索记忆"刚教过一次）。

与其它 worker 同一套形状：``QThread.started`` → ``run()`` → 一个终态信号。
``run`` 一定返回（异常也收成 ``failed``），线程才不会空转——空转的症状是收尾时
退出码 ``0xC0000409``，见 ``ui/agent_worker.py`` 与 ``_refresh_models`` 的注释。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..llm.llama_server import LlamaServerError, LlamaServerRuntime

__all__ = ["LlamaRuntimeWorker"]


class LlamaRuntimeWorker(QObject):
    """起一次运行时；成功给出 base_url，失败给一句人话。"""

    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, *, model_path: str, context: int) -> None:
        super().__init__()
        self._model_path = model_path
        self._context = context
        #: 成功之后由界面线程取走（信号只带字符串，运行时对象不能过信号）。
        self.runtime: LlamaServerRuntime | None = None

    @Slot()
    def run(self) -> None:
        try:
            runtime = LlamaServerRuntime(
                model_path=Path(self._model_path), context=int(self._context)
            )
            base_url = runtime.start()
        except LlamaServerError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — 兜底，绝不把线程炸掉
            self.failed.emit(f"本地模型启动失败：{type(exc).__name__}: {exc}")
            return
        self.runtime = runtime
        self.ready.emit(base_url)
