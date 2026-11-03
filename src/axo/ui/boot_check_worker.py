"""启动自检的工作线程。

自检要写临时文件、只读打开记忆库、对 Ollama 做最多 0.6 秒的 TCP 连接；这些都在
``pre_roll()`` 之前跑的话，第一幕会僵住。放到 ``QThread`` 里，六幕继续画，结果用
``done`` 信号送回主线程的 ``splash.set_checks()``。

**纯函数不动。** :func:`axo.ui.boot_checks.collect_checks` 仍然不 import Qt；
这个模块只负责线程边界。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal

__all__ = ["BootCheckWorker"]


class BootCheckWorker(QThread):
    """跑一遍 :func:`~axo.ui.boot_checks.collect_checks`，把结果 emit 回主线程。"""

    done = Signal(object)
    """``tuple[BootCheck, ...]``。自检本身异常时发空元组，不把启动拖垮。"""

    def __init__(self, backend: Any, tools: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._tools = tools

    def run(self) -> None:  # noqa: D401 - QThread 入口
        from .boot_checks import collect_checks

        try:
            checks = collect_checks(self._backend, self._tools)
        except Exception:  # noqa: BLE001 —— 自检坏掉不该拦住启动
            checks = ()
        self.done.emit(checks)
