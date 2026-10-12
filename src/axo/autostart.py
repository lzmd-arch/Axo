"""提醒到点时**开机自启的那一条路**：Windows 的 Run 键。

用户在设置页勾上「开机时自动运行」之后，这个模块往
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`` 写一个值，于是下次登录
Windows 会自己把 :mod:`axo.headless` 那一次跑起来。

## 为什么是那个键，不是启动文件夹

两个地方都能做同一件事，选注册表键的理由是**它精确地只有一件事**：

* 可读可验——打开注册表编辑器就看得见程序写了什么，"自启到底设上没有"有一个客观答案；
* 开关就是写一个值 / 删一个值：**幂等、可逆、不留垃圾**（启动文件夹里会留下一个
  ``.lnk``，删漏了就是一个指向不存在路径的死快捷方式）；
* 不需要 PowerShell 或 COM。``tools/build_exe.py`` 为了在桌面上放一个快捷方式已经
  绕过一次 COM（中文路径要过控制台代码页），能不再绕就不绕。

## 不 import Qt

理由与 :mod:`axo.secrets` 一模一样：**平台相关的代码集中在一个模块**，而"它自己不
import Qt"因此是一条可断言的事实（``test_layering`` 钉着）。这一层也不弹任何 UI——
要不要写这个键是设置页的事。

## 非 Windows 一律返回"不支持"，绝不抛

它要在 Linux 的 CI 上被 import 到（``winreg`` 在那儿根本不存在），所以那个 import
是**推迟到函数里**的。返回值一律是"做成了没有"，而不是异常：调用方是设置页，它要把
"这台机器上做不到"显示成一句话，而不是崩一次。
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "REMIND_FLAG",
    "RUN_KEY",
    "VALUE_NAME",
    "apply",
    "command_for",
    "current_exe",
    "is_supported",
    "read_command",
    "remove",
    "sync",
    "unavailable_reason",
]

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
"""自启那条值住在哪儿。**``HKCU``（当前用户）**，不是 ``HKLM``：让程序开机自启是
这台机器上这个用户的选择，改全局的那一份要管理员权限、还会影响别的账户。
"""

VALUE_NAME = "Axo"
"""值的名字。用**界面上的显示名**（同 ``tools/build_exe.py`` 的 ``SHORTCUT_NAME``）：
用户在注册表里看到的那一行要跟托盘提示、窗口标题对得上。"""

REMIND_FLAG = "--remind"
"""命令行上那个开关。**与 ``ui/app.py`` 里那句逐字相同**（``test_packaging.py`` 断言），
因为它写进注册表之后就固定在那儿了：改了它，已经设过自启的机器会在下次登录时把一个
不认识的参数交给程序，而那看起来像"程序自己启动了一个窗口"。
"""


def is_supported() -> bool:
    """这台机器上能不能设自启。非 Windows 是假（那个键不存在）。"""
    return sys.platform == "win32"


def current_exe() -> Path | None:
    """要写进注册表的那个 exe。**只有冻结产物才算**，否则是 ``None``。

    源码检出下 ``sys.executable`` 是 ``python.exe``——把它写进去，"开机自启"启动的
    是一个**没有脚本可跑的 Python**（它会开一个交互解释器，或者什么都没发生，取决于
    怎么被拉起）。而那个症状只会在下次登录时出现一次，且看起来与这个设置毫无关系。
    """
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable)


def unavailable_reason() -> str:
    """这台机器上为什么设不了自启；能设就是空串。

    设置页把这句原样显示出来，并把那个勾选框灰掉——**"勾了却没生效"比"勾不上"坏
    得多**，而这里两种情形都说得清（见 :func:`current_exe` 与 :func:`is_supported`）。
    """
    if not is_supported():
        return "这台机器不是 Windows，没有那个开机自启的注册表项。"
    if current_exe() is None:
        return "现在跑的是源码、不是打包出来的 exe，设不了开机自启——打包之后才有这一项。"
    return ""


def apply(enabled: bool) -> bool:
    """按设置把自启那条值对齐：勾上 → 写，取消 → 删。返回**有没有动过注册表**。

    **"只在用户显式勾选时写"这条对策就落在这一句上**：``enabled`` 为假时走的是
    :func:`remove`（它只会删自己那一个值），而源码检出下 :func:`current_exe` 是
    ``None``，于是这里什么都不写——那台机器上根本没有可自启的东西。

    **不动别的程序的值**，也不删 ``Run`` 这个键（见 :func:`remove`）。
    """
    if not enabled:
        return remove()
    exe = current_exe()
    if exe is None:
        return False
    return sync(command_for(exe))


def command_for(exe: Path | str) -> str:
    """写进注册表的完整命令行。

    **路径带引号**，而且是**全路径**：Run 键里的命令行由 Windows 自己拼，工作目录
    不是程序所在的目录，而用户的安装路径里带空格是常态（``C:\\Program Files\\...``）。
    不带引号的话，Windows 会拿第一个空格去断命令行——症状是"设了自启，登录时什么都没
    发生"，而注册表里那一行看起来完全正常。
    """
    return f'"{Path(exe)}" {REMIND_FLAG}'


def read_command() -> str | None:
    """现在写的是什么。没有、不支持、或读不动都是 ``None``。

    **读不出来不等于没设**——但对调用方来说两者要做的事一样（写一遍），所以这里不
    区分。真正的"没设"与"值坏了"在注册表编辑器里是两回事，那是给人看的，不是给这一层
    看的。
    """
    if not is_supported():
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
    except OSError:
        # ``FileNotFoundError``（那个键或那个值不在，全新机器上都是这样）也是 OSError。
        return None
    return value if isinstance(value, str) else None


def sync(command: str) -> bool:
    """把自启那条值对齐到 ``command``。**值真的一样就什么都不做**，返回有没有动过。

    "不无脑重写"是三条对策里最要紧的一条：每次启动都去动那个键，会让杀软与用户都
    觉得这个程序在做可疑的事（它每天改一次注册表，而界面上什么都没发生）。所以先读、
    一样就返回——**搬家之后的自愈**（exe 换了位置）也走这一条路：值不一样才写。

    **只在用户显式勾选时被调用**（见设置页），这一层不做那个决定。
    """
    if not is_supported():
        return False
    if read_command() == command:
        return False
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
    return True


def remove() -> bool:
    """把自启那条值删掉。删过、本来就没有、不支持都是假，真删掉了是真。

    **只删自己那一个值，不删那个键**：``Run`` 这个键是 Windows 的，里面还住着别的
    程序；删它等于顺手把别人的自启也关掉。
    """
    if not is_supported():
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except OSError:
        # 值不在（没设过、或者刚被别的路径删掉）。**当作成功**：调用方要的结果
        # "它现在不在那儿"已经成立了。
        return False
    return True
