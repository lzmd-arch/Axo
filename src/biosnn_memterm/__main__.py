"""``python -m biosnn_memterm`` 的入口，也是**冻结产物的入口**。

存在它的理由只有一条：**相对 import 在文件被当作 ``__main__`` 执行时解析不了。**
:mod:`biosnn_memterm.ui.app` 里那几处 ``from .main_window import ...`` 与
``from .. import __version__`` 依赖 ``__package__``，而 ``python -m`` 会给它设好。
PyInstaller 收的是一份**脚本**——它把入口文件当 ``__main__`` 跑，``__package__``
是空的，那几处 import 于是在**运行时**炸掉。而产物是个无控制台的 exe：炸掉的表
现是弹一个错误框，进程还活着，看起来像"启动了但什么都没有"。

所以冻结的入口必须是**自己不带相对 import** 的那一个。这里只剩一句按名字找包的
绝对 import，两种跑法下都成立。

``run.bat`` 走的仍然是 ``-m biosnn_memterm.ui.app``（同一件事，只是不经这里）。
"""

from __future__ import annotations

from biosnn_memterm.ui.app import main

if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())
