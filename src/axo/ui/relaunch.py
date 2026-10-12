"""重新起一份自己。

只给「更改记忆位置」那一条链用：位置指针写下去之后要重启才能生效（理由见
``ui/data_root.py`` 的模块文档——原地搬需要一条"停线程但别退、并且这次关库"的半停机
路径，而 ``_shutdown`` 刻意不提供它）。

**不 import Qt 到模块级。** 这个模块会被 ``ui/main_window.py`` 间接用到，而
``QProcess`` 只有在真的要重启时才需要——把 Qt 拖进 import 图的代价是它再也进不了
核心腿的测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["restart_application", "restart_command"]


def restart_command() -> tuple[str, list[str], str]:
    """该起什么、带什么参数、在哪个目录起。返回 ``(程序, 参数, 工作目录)``。

    两种形态分开，因为入口不一样：

    * **冻结态**（别人下载的那个 zip 解出来的）：``sys.executable`` 就是
      ``axo.exe``，参数原样带过去，工作目录是它自己所在的那个。
    * **源码态**（``run.bat`` 与 ``python -m axo`` 走的是同一个 ``main()``）：
      ``sys.executable`` 是那个 venv 的解释器，所以直接用它 + ``-m axo.ui.app``——
      **不必知道 uv 在不在、也不必去猜 venv 的路径**。

    ``-m axo.ui.app`` 要求 ``axo`` 可导入。editable 安装满足这一点；而
    "只把 ``src`` 挂进 ``sys.path`` 再跑"那种临时跑法不满足——那时
    :func:`restart_application` 会返回假，界面上走"下次打开时自动完成"那条降级路
    （那条路是通的，因为标记已经写进注册表了）。
    """
    if getattr(sys, "frozen", False):
        program = sys.executable
        return program, list(sys.argv[1:]), str(Path(program).resolve().parent)
    return sys.executable, ["-m", "axo.ui.app"], str(Path.cwd())


def restart_application() -> bool:
    """另起一份自己，成功返回 ``True``。**调用方拿到假时不要退出。**

    起不来不是失败：搬迁的标记已经落在注册表里了，用户下次自己打开程序照样会办完。
    所以这里只回一个布尔，由界面对那个情形说一句实话。
    """
    from PySide6.QtCore import QProcess

    program, arguments, working_directory = restart_command()
    return QProcess.startDetached(program, arguments, working_directory)
