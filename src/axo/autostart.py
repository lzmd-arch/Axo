"""提醒到点时**开机自启的那一条路**：Windows 的 Run 键、macOS 的 LaunchAgent。

用户在设置页勾上「开机时自动运行」之后，这一层把 :mod:`axo.headless` 那一次挂到
登录流程上：

* Windows：``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`` 里一个值。
* macOS：``~/Library/LaunchAgents/com.axo.memoryterminal.reminders.plist``，登录时
  由 launchd 跑一次 ``Axo --remind``。

## 为什么是这两个系统位置，不是启动文件夹

* 可读可验——注册表编辑器 / ``launchctl`` 都看得见程序写了什么，"自启到底设上没有"
  有一个客观答案；
* 开关就是写一个值 / 删一个文件：**幂等、可逆、不留垃圾**（启动文件夹里会留下一个
  ``.lnk``，删漏了就是一个指向不存在路径的死快捷方式）；
* 不需要 PowerShell 或 COM。``tools/build_exe.py`` 为了在桌面上放一个快捷方式已经
  绕过一次 COM（中文路径要过控制台代码页），能不再绕就不绕。

## 不 import Qt

理由与 :mod:`axo.secrets` 一模一样：**平台相关的代码集中在一个模块**，而"它自己不
import Qt"因此是一条可断言的事实（``test_layering`` 钉着）。这一层也不弹任何 UI——
要不要写这一项是设置页的事。

## 非 Windows / 非 macOS 一律返回"不支持"，绝不抛

它要在 Linux 的 CI 上被 import 到（``winreg`` 在那儿根本不存在），所以那个 import
是**推迟到函数里**的。返回值一律是"做成了没有"，而不是异常：调用方是设置页，它要把
"这台机器上做不到"显示成一句话，而不是崩一次。
"""

from __future__ import annotations

import plistlib
import shlex
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
"""Windows 自启那条值住在哪儿。**``HKCU``（当前用户）**，不是 ``HKLM``：让程序开机自启是
这台机器上这个用户的选择，改全局的那一份要管理员权限、还会影响别的账户。
"""

VALUE_NAME = "Axo"
"""Windows 值的名字。用**界面上的显示名**（同 ``tools/build_exe.py`` 的
``SHORTCUT_NAME``）：用户在注册表里看到的那一行要跟托盘提示、窗口标题对得上。
"""

REMIND_FLAG = "--remind"
"""命令行上那个开关。**与 ``ui/app.py`` 里那句逐字相同**（``test_packaging.py`` 断言），
因为它写进系统自启项之后就固定在那儿了：改了它，已经设过自启的机器会在下次登录时把一个
不认识的参数交给程序，而那看起来像"程序自己启动了一个窗口"。
"""

_LAUNCH_AGENT_LABEL = "com.axo.memoryterminal.reminders"
"""macOS LaunchAgent 的标签。反向域名 + 用途，固定不变。"""


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCH_AGENT_LABEL}.plist"


def is_supported() -> bool:
    """这台机器上能不能设自启。"""
    return sys.platform in ("win32", "darwin")


def current_exe() -> Path | None:
    """要写进自启项的那个可执行文件。**只有冻结产物才算**，否则是 ``None``。

    源码检出下 ``sys.executable`` 是 ``python.exe`` / ``python3``——把它写进去，
    "开机自启"启动的是一个**没有脚本可跑的 Python**（它会开一个交互解释器，或者什么都
    没发生）。而那个症状只会在下次登录时出现一次，且看起来与这个设置毫无关系。
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
        return "这台机器既不是 Windows 也不是 macOS，没有可用的开机自启入口。"
    if current_exe() is None:
        return "现在跑的是源码、不是打包出来的程序，设不了开机自启——打包之后才有这一项。"
    return ""


def apply(enabled: bool) -> bool:
    """按设置把自启项对齐：勾上 → 写，取消 → 删。返回**有没有动过系统设置**。

    **"只在用户显式勾选时写"这条对策就落在这一句上**：``enabled`` 为假时走的是
    :func:`remove`（它只会删自己那一项），而源码检出下 :func:`current_exe` 是
    ``None``，于是这里什么都不写——那台机器上根本没有可自启的东西。
    """
    if not enabled:
        return remove()
    exe = current_exe()
    if exe is None:
        return False
    return sync(command_for(exe))


def command_for(exe: Path | str) -> str:
    """写进自启项的那一行命令。

    **路径带引号，而且是全路径。** Windows 的 Run 键里命令行由系统自己拼，工作目录
    不是程序所在的目录；macOS 这一边虽然最终写的是 ``ProgramArguments`` 数组，保留
    同一个字符串让两个平台共用 ``read_command`` 的比较逻辑。
    """
    return f'"{Path(exe)}" {REMIND_FLAG}'


def _launch_agent_record(command: str) -> dict[str, object]:
    args = shlex.split(command)
    return {
        "Label": _LAUNCH_AGENT_LABEL,
        "ProgramArguments": args,
        "RunAtLoad": True,
        "ProcessType": "Background",
    }


def read_command() -> str | None:
    """现在写的是什么。没有、不支持、或读不动都是 ``None``。

    **读不出来不等于没设**——但对调用方来说两者要做的事一样（写一遍），所以这里不
    区分。
    """
    if sys.platform == "darwin":
        path = _launch_agent_path()
        try:
            record = plistlib.loads(path.read_bytes())
        except (OSError, ValueError, plistlib.InvalidFileException):
            return None
        args = record.get("ProgramArguments") if isinstance(record, dict) else None
        if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
            return None
        if len(args) == 2 and args[1] == REMIND_FLAG:
            return command_for(args[0])
        return shlex.join(args)
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
    """把自启项对齐到 ``command``。**值真的一样就什么都不做**，返回有没有动过。

    "不无脑重写"是三条对策里最要紧的一条：每次启动都去改系统设置，会让用户（以及
    安全软件）觉得这个程序在做可疑的事。所以先读、一样就返回——**搬家之后的自愈**
    （程序换了位置）也走这一条路：值不一样才写。

    **只在用户显式勾选时被调用**（见设置页），这一层不做那个决定。
    """
    if not is_supported():
        return False
    if read_command() == command:
        return False
    if sys.platform == "darwin":
        try:
            record = _launch_agent_record(command)
            if not record["ProgramArguments"]:
                return False
            path = _launch_agent_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(plistlib.dumps(record))
            temporary.replace(path)
        except (OSError, ValueError):
            return False
        return True
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
    return True


def remove() -> bool:
    """把自启项删掉。删过、本来就没有、不支持都是假，真删掉了是真。

    **只删自己那一项**：Windows 的 ``Run`` 键里还住着别的程序，删它等于顺手把别人的
    自启也关掉；macOS 只删自己那个 plist。
    """
    if sys.platform == "darwin":
        path = _launch_agent_path()
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True
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
