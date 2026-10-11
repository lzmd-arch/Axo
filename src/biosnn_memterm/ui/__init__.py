"""桌面 UI 层（PySide6）。

这个 ``__init__`` 刻意不 import 任何 Qt 子模块。理由与气体工作台同源：Qt 的导入
代价很大，而 ``import biosnn_memterm.ui`` 可能发生在任何地方——包括只想读一下
版本号的场景。把重导入推迟到属性真正被访问的那一刻（PEP 562），是让
``python -c "import biosnn_memterm.ui as u; print(u.__doc__)"`` 不必启动整个
Qt 运行时的唯一办法。

用到的名字按 :data:`_LAZY_FROM_QT` 的表查；表里没有的名字一律 ``AttributeError``，
不静默回退——拼错名字时要立刻发现，而不是拿到 ``None`` 再去猜。
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["ChatView", "MainWindow", "main"]

_LAZY_FROM_QT: dict[str, str] = {
    "ChatView": ".chat_view",
    "MainWindow": ".main_window",
    "main": ".app",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_FROM_QT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    # 缓存回 globals()：PEP 562 的 __getattr__ 每次访问都会重新查表，缓存后
    # 只有第一次付 import 的代价。
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
