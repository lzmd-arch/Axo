"""把应用冻结成 Windows 上的 onedir 产物，并当场自检。

**不写 ``.spec``。** ``.spec`` 是 ``.py`` 之外的散件：ruff 扫不到它，测试也读不到
它，等于一块不进任何腿的代码。所以参数在这个文件里拼出来、调 ``python -m
PyInstaller``，构建配置与自检逻辑住在一起，都能被 AST 读到（见
``tests/test_packaging.py``）。

**自检是这一步的正文，不是附注。** 冻结坏掉的方式几乎都不抛异常：资源没带进去、
QtWebEngine 的平台件落在镜像外、某个 import 被漏掉——它们表现为"双击之后什么都没
发生"，或者"窗口开出来了、左上角空着"。而 CI 验不了这一环（六条腿全在 ``ubuntu-latest``，
Windows exe 只能在 Windows 上构建），所以这一环唯一的守卫就是这个脚本自己：构建完
立刻查盘上的事实，再真起一次。

用法::

    python tools/build_exe.py --dry-run     # 只打印要跑的命令
    python tools/build_exe.py               # 构建 + 自检 + 在桌面放一个快捷方式
    python tools/build_exe.py --console     # 排障用的带控制台版本，与上面那个并存
    python tools/build_exe.py --no-shortcut # 只构建，不碰桌面
    python tools/build_exe.py --release     # 上面全跑，再打一个发给别人的 zip
    python tools/build_exe.py --release --skip-build   # 产物已经在 dist/ 里，只跑闸门

``--skip-build`` 跳过的**只有** PyInstaller 那一步，判据一条都不少。它存在的理由不是
省时间，是**让闸门可以被验红**：往 ``dist/`` 里塞一个假 ``memory.db`` 看它红不红，
不该先等十分钟重新冻结一遍（见 README 的「发布」一节）。

**``--release`` 不新开工具，也不另写一份自检。** 它必须等构建完才打得成包，而打包要用
的判据（产物在不在、清不干净、首启是不是空的）与自检用的是同一批事实——两个工具迟早
会漂，本仓已经吃过一次"替身跟丢"的亏。所以它是在同一条流水线后面接了几步：构建 →
自检 → 剔 GPL-only 模块 → 隔离首启探测 → 残留闸门 → 与 NOTICES 对账 → 打 zip。
**任何一步红就不出 zip**：一个"打出来了但内容不对"的压缩包比没有更糟，因为它会被发出去。

**产物藏在 ``dist/<name>/_internal/`` 旁边**，而 ``dist/`` 在 ``.gitignore`` 里（它是构建
输出）——所以在编辑器里它是灰的、在远端它根本不存在，"构建好了但找不到"是常态。自检最后
一步因此包含**在桌面放一个快捷方式**：构建成功才放，失败不放（指向一个坏的 exe 比没有更糟）。
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata
import mmap
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Collection, Sequence
from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parents[1]
_DIST = _ROOT / "dist"
_BUILD = _ROOT / "build"

NAME = "axo"
"""交付物的名字。产物落在 ``dist/axo/axo.exe``。"""

CONSOLE_NAME = "axo-console"
"""``--console`` 那一版的名字。**两个可以并存**：无控制台的是交付物，带控制台的是
排障用的——窗口版没有控制台可看，出问题时唯一的线索就在带控制台那一版的 stderr 上。
"""

SHORTCUT_NAME = "Axo"
"""桌面上那个快捷方式的名字。

用**界面上的显示名**（``ui/app.py`` 的 ``setApplicationDisplayName``），不是包名：桌面
上的图标是给人点的，名字得跟窗口标题、托盘提示一致，而不是 ``axo``。
"""

ENTRY = "src/axo/__main__.py"
"""真入口，而且**必须是这一个**。

实质的入口是 :mod:`axo.ui.app` 的 ``main()``，平时跑的是 ``python -m
axo.ui.app``（``run.bat``）。但 PyInstaller 收的是**脚本**：它把入口文件
当 ``__main__`` 跑，``__package__`` 是空的——``app.py`` 里那几处相对 import
（``from .main_window import ...``）于是在运行时炸掉。第一次构建实测就是这个症状：
**一个项目模块都没被收进去**，exe 弹一个 PyInstaller 的错误框、进程活着，所以活体
探测当时给的是**假绿**。

``__main__.py`` 里只有一句按名字找包的绝对 import，两种跑法下都成立。它同时让
``python -m axo`` 成为可用入口。
"""

DATAS = (
    ("src/axo/ui/theme.qss", "axo/ui"),
    ("src/axo/ui/web", "axo/ui/web"),
)
"""要带进镜像的**非 ``.py`` 资产**（本仓的非 Python 资产就这些）。

目标路径不是随手写的：PyInstaller 给包内模块设的 ``__file__`` 指向 ``_MEIPASS`` 下的
镜像路径，所以只要把文件放在**与源码相同的相对位置**上，``theme.py`` 那句
``Path(__file__).with_name("theme.qss")`` 与 ``chat_view.py`` 那句
``Path(__file__).parent / "web"`` 就自动对——应用代码一个字都不用改。这条假设由自检
第 2 条钉住，不成立就当场红。

``web/`` **整目录**带：以后往 ``ui/web/`` 里加文件自动跟上，不需要维护一张清单，也就
不会出现"新加了一个 js 而产物里没有"这种只在交付物里才看得见的坏法。
"""

ICON_SIZES = (256, 64, 48, 32, 24, 16)
"""手工拼进 ``.ico`` 的档位，从大到小。

不用 Qt 的 ICO 写入器：``qico`` 插件**只写单张图**（本机在 ``PySide6/plugins/
imageformats/qico.dll``），而 Windows 从资源管理器到任务栏会按 DPI 挑不同档位。ICO
容器简单到可以手写：目录头 + 每档一条 PNG 条目（Vista 起 PNG 条目是标准）。大档排
前面是惯例，也是老工具挑"第一张"时的顺序。
"""

_MIRRORED = (
    "axo/ui/theme.qss",
    "axo/ui/web/index.html",
    "axo/ui/web/css/app.css",
    "axo/ui/web/js/app.js",
)
"""自检第 2 条：镜像里必须有这四个（都在 ``_internal/`` 下）。

只抽这四条而不是把 ``web/`` 整个列一遍：这四条各自代表一处 ``Path(__file__)`` 读取
（qss 一处、网页三处不同目录），列全了只是把 ``DATAS`` 抄第二遍。
"""

_QT_REQUIRED = (
    "PySide6/QtWebEngineProcess.exe",
    "PySide6/translations/qtwebengine_locales",
    "PySide6/plugins",
    "PySide6/resources/qtwebengine_resources.pak",
    "PySide6/resources/qtwebengine_devtools_resources.pak",
    "PySide6/resources/icudtl.dat",
    "PySide6/resources/v8_context_snapshot.bin",
)
"""自检第 3 条：QtWebEngine 的运行件。

**按本机实测的布局写**：这里的 PySide6 是**平铺**的（``site-packages/PySide6/
QtWebEngineProcess.exe``），不是某些发行版那样的 ``Qt6/bin/``。少任何一个的表现都
是对话区白屏或进程直接起不来，而这两种都不抛异常。

后四条是**被 :func:`_prune_debug_resources` 删掉的那些调试变体各自的那一份非调试孪生**
——列出来是为了钉住"删的是孪生里错的那一个"。真要哪天把 ``qtwebengine_resources.pak``
也删了，这条会当场红，而不是等到屏幕上白屏。
"""

_DEBUG_RESOURCES = (
    "PySide6/resources/*.debug.pak",
    "PySide6/resources/*.debug.bin",
)
"""构建后删掉的**调试变体**资源（相对 ``_internal/`` 的 glob）。

**这一组是量出来的，不是猜的。** 首次构建的产物 598.4 MB 里 ``PySide6/resources/`` 占
101.4 MB，其中 ``qtwebengine_devtools_resources.debug.pak`` **一个文件 72.3 MB**、
``v8_context_snapshot.debug.bin`` 2.3 MB。这些 ``.debug`` 变体是给 Qt WebEngine 的
**调试版**用的（本产物链的是发布版 DLL），而且**每一个都有同名的非调试孪生一起进了镜像**
——所以删的是"同一份东西的另一版"，不是"猜着用不到的东西"。删完实测 521.2 MB，省下
**77.2 MB**（产物里的数字一律按 1024² 算）。

**其余几块大的不动，理由写在 README 的「打包（P6a）」里**：那个尺寸的构成是 QtWebEngine
自己（194 MB）、它的资源与语言包（154 MB）、软件 OpenGL 兜底（20 MB）——都不是"没被用到
的 Python 模块"，砍它们换来的是**机器验不了的**白屏风险，而这里唯一的机器判据（窗口开没
开出来）根本不走那些代码路径。
"""

_FORBIDDEN_MISSING = ("biosnn_bus", "numpy", "httpx", "PySide6", "memcore", "axo")
"""自检第 4 条：PyInstaller 自己的缺失清单里**不许出现**这些模块。

名单短而具体是刻意的——长尾的 ``posix`` / ``pwd`` / ``org`` 一类在 Windows 上本来就
找不到，收进来只会让这份名单变成一坨噪音，然后被无视。

这六条各指一件事：``biosnn_bus`` 是 ``memcore/types.py`` 里那个**函数内** import（第四
模态唯一的接缝）——分析器照样会走字节码找到它，所以它**不该**出现在缺失清单里；``numpy``
/``httpx`` 是两条运行时依赖；``PySide6`` 缺一个子模块就是白屏或起不来；``memcore`` /
``axo`` 是本仓自己那两个包，"缺失"只可能是收集规则写错了。
"""


# ── 图标 ────────────────────────────────────────────────────────────────────


def _icon_pngs(sizes: Sequence[int]) -> list[tuple[int, bytes]]:
    """每一档**现画**一张 PNG，返回 ``(边长, PNG 字节)``。

    画的是 ``ui/tray.py`` 那个图标本身（托盘用的就是它），只是换个边长重画一次——
    把 64px 那张缩到 16px 会**把中心那个点缩没**，而 16px 恰恰是任务栏和资源管理器最
    常用的那一档。

    ``QT_QPA_PLATFORM=offscreen`` 是必须的：``QPixmap``/``QIcon`` 要求先有一个
    ``QGuiApplication``，而这个脚本可能在没有桌面会话的地方跑。这一句必须在建 app
    **之前**写进环境。
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QGuiApplication

    from axo.ui.theme import DEFAULT_THEME
    from axo.ui.tray import build_icon

    app = QGuiApplication.instance() or QGuiApplication([])

    out: list[tuple[int, bytes]] = []
    for size in sizes:
        image = build_icon(DEFAULT_THEME.accent, size).pixmap(size, size).toImage()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        out.append((size, bytes(buffer.data())))
        buffer.close()
    del app
    return out


def _write_ico(path: Path, pngs: Sequence[tuple[int, bytes]]) -> None:
    """把若干 PNG 拼成一个 ``.ico``。**格式全部小端。**

    头 6 字节 = 保留(0) + 类型(1) + 张数；随后每张一条 16 字节目录项 = 宽、高、调色板
    数、保留、色平面、位深、数据长度、数据偏移；目录项之后是数据本体。

    宽或高为 **256 时要写 0**——那一格是一个字节，装不下 256。这是这个格式里唯一一处
    会静默出错的地方（写 256 溢出成 0 恰好等价，所以看不出来；写别的值则整张图不认）。
    """
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = len(header) + 16 * len(pngs)
    entries = bytearray()
    body = bytearray()
    for size, data in pngs:
        side = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset)
        body += data
        offset += len(data)
    path.write_bytes(bytes(header) + bytes(entries) + bytes(body))


def _draw_icon(path: Path) -> bool:
    """画图标并落盘，返回画没画成。**画不出来不算构建失败。**

    图标是外观，不是功能：为一个 4 KB 的位图让整次构建挂掉，代价明显不成比例。但
    **也不静默降级**——说一句，让人知道这个产物没有自定义图标。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_ico(path, _icon_pngs(ICON_SIZES))
    except Exception as error:  # Qt 起不来、字体缺失、任何绘制路径上的意外
        print(f"图标画不出来，照常构建（产物用 PyInstaller 的默认图标）：{error!r}")
        return False
    print(f"图标：{path.relative_to(_ROOT).as_posix()}（{len(ICON_SIZES)} 档）")
    return True


# ── 构建 ────────────────────────────────────────────────────────────────────


def _command(*, name: str, console: bool, icon: Path | None, clean: bool) -> list[str]:
    """拼出 PyInstaller 的命令行。

    ``--specpath build`` 是为了**不在仓库根掉一个 ``.spec`` 出来**：那个文件一旦出现
    就会被当成"配置"，而它是一份生成物，改了下次构建还会被覆盖。

    ``--add-data`` 的分隔符用 ``os.pathsep``（Windows ``;``、其余 ``:``），源路径一律
    给**绝对路径**：这个脚本可以从任何工作目录被调起来，而相对的源路径会跟着 cwd 变。

    ``--paths src`` 是显式的：入口里那句 ``from axo.ui.app import main`` 要
    能按名字找到包。源码检出时 `src/` 通常在 ``sys.path`` 上（editable 安装），但显式
    写出来这件事就不依赖安装方式了。
    """
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console" if console else "--windowed",
        "--name",
        name,
        "--paths",
        str(_ROOT / "src"),
        "--distpath",
        str(_DIST),
        "--workpath",
        str(_BUILD / name),
        "--specpath",
        str(_BUILD),
    ]
    if clean:
        command.append("--clean")
    for source, target in DATAS:
        command += ["--add-data", f"{_ROOT / source}{os.pathsep}{target}"]
    if icon is not None:
        command += ["--icon", str(icon)]
    command.append(str(_ROOT / ENTRY))
    return command


def _require_pyinstaller() -> None:
    """没装就把**该跑哪一条命令**说清楚，而不是让裸的 ``ModuleNotFoundError`` 冒出去。

    命令是 ``--all-extras`` 而不是 ``--extra pack``：``uv sync`` 是**精确同步**，只给
    ``pack`` 会把 ``PySide6`` 与 ``httpx`` 从环境里摘掉——那会把开发环境弄坏，而症状
    （界面上跑不起来、测试红一片）离原因很远。
    """
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "PyInstaller 不在当前环境里。装上：\n\n"
            "    uv sync --all-extras\n\n"
            "（不是 --extra pack：uv sync 是精确同步，只给 pack 会把 PySide6 与\n"
            "httpx 从环境里摘掉。）",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


# ── 桌面快捷方式 ────────────────────────────────────────────────────────────


def _desktop() -> Path:
    """桌面目录。**问系统，不拼 ``%USERPROFILE%\\Desktop``。**

    桌面可能被 OneDrive 接管（那时它在 ``%USERPROFILE%\\OneDrive\\桌面``），也可能被组策略
    挪到别处。拼字符串在这类机器上会落到一个**存在的、但不是桌面的**目录里，于是快捷方式
    "建成功了、人却看不见"——正是这一步要治的那个症状。``SHGetKnownFolderPath`` 是唯一会
    把这些重定向一并算进去的问法，而它不需要额外依赖。
    """
    import ctypes
    from ctypes import wintypes

    class _Guid(ctypes.Structure):
        _fields_ = (
            ("part1", wintypes.DWORD),
            ("part2", wintypes.WORD),
            ("part3", wintypes.WORD),
            ("part4", ctypes.c_ubyte * 8),
        )

    # FOLDERID_Desktop = {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
    folder = _Guid(
        0xB4BFCC3A,
        0xDB2C,
        0x424C,
        (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
    )
    shell32 = ctypes.windll.shell32
    shell32.SHGetKnownFolderPath.argtypes = (
        ctypes.POINTER(_Guid),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    holder = ctypes.c_wchar_p()
    try:
        code = shell32.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(holder))
        if code == 0 and holder.value:
            return Path(holder.value)
    finally:
        if holder:
            ctypes.windll.ole32.CoTaskMemFree(ctypes.cast(holder, ctypes.c_void_p))
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def _make_shortcut(name: str) -> str | None:
    """在桌面放一个指向产物的快捷方式，返回快捷方式的路径；失败返回 ``None``。

    **走 PowerShell 的 COM，不引 pywin32。** ``.lnk`` 是二进制格式，手写它比手写 :func:`_write_ico`
    复杂一个量级（CLSID 头 + LinkTargetIDList + LinkInfo…），而 Windows 自带一个会写它的对象；
    为这一步往依赖里加一个包，或者在本仓维护一段二进制写入器，都不划算。

    **命令用 ``-EncodedCommand`` 递过去**（UTF-16LE + base64）：路径里有中文，而走命令行参数
    会经过控制台代码页，那个代码页在不同机器上不一样。

    **目标路径是绝对的。** 仓库搬了家，桌面那个快捷方式就指向旧位置（那是快捷方式的固有属性，
    不是这里能修的）；重新构建一次会把它覆盖成新的，README 的排障表里写着这句。

    **失败不算构建失败**，与画图标同一条理由：快捷方式是入口，不是产物本身。但**不静默**——
    说一句，并把绝对目标打出来，让人可以自己建一个。
    """
    exe = _DIST / name / f"{name}.exe"
    shortcut = _desktop() / f"{SHORTCUT_NAME}.lnk"

    def quote(value: object) -> str:
        """PowerShell 单引号字符串：里面的单引号**写两遍**才转义。"""
        return "'" + str(value).replace("'", "''") + "'"

    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut({quote(shortcut)}); "
        f"$link.TargetPath = {quote(exe)}; "
        f"$link.WorkingDirectory = {quote(exe.parent)}; "
        f"$link.IconLocation = {quote(f'{exe},0')}; "
        f"$link.Description = {quote(SHORTCUT_NAME)}; "
        "$link.Save()"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"桌面快捷方式没建成（{error!r}）。目标：{exe}")
        return None
    if done.returncode != 0 or not shortcut.is_file():
        print(f"桌面快捷方式没建成（powershell 退出码 {done.returncode}）。目标：{exe}")
        return None
    return str(shortcut)


# ── 自检 ────────────────────────────────────────────────────────────────────


def _prune_debug_resources(name: str) -> list[str]:
    """删掉 :data:`_DEBUG_RESOURCES` 点名的那些文件，返回"删了什么"（进构建报告）。

    **在自检之前做**，删完由第 6 条回头确认。PyInstaller 没有任何开关能做这件事（它会
    把 Qt 送来的东西照单全收），所以这里是构建之后的一次显式删除——既然是自己删的，就
    得自己证明删掉了。
    """
    internal = _DIST / name / "_internal"
    removed: list[str] = []
    for pattern in _DEBUG_RESOURCES:
        for path in sorted(internal.glob(pattern)):
            size = path.stat().st_size
            path.unlink()
            removed.append(f"{path.relative_to(internal).as_posix()}（{size / 1024**2:.1f} MB）")
    return removed


def _check_pruned(name: str) -> list[str]:
    """第 6 条：调试变体**真的没了**。

    守的是**这次删除本身**：要是哪天 PySide6 改了文件名，glob 一条都匹配不上，构建照常
    成功、报告里也照常写着体积——而盘上什么都没变。删了就要拿得出证据。
    """
    internal = _DIST / name / "_internal"
    left = [
        path.relative_to(internal).as_posix()
        for pattern in _DEBUG_RESOURCES
        for path in sorted(internal.glob(pattern))
    ]
    return [f"调试变体还在：{item}" for item in left]


def _check_layout(name: str) -> list[str]:
    """第 1/2/3 条：产物在不在、镜像里的人对不对、QtWebEngine 的运行件在不在。"""
    root = _DIST / name
    internal = root / "_internal"
    problems: list[str] = []

    exe = root / f"{name}.exe"
    if not exe.is_file():
        problems.append(f"没有产物：{exe.relative_to(_ROOT).as_posix()}")
    for relative in _MIRRORED:
        path = internal / relative
        if not path.is_file():
            problems.append(f"镜像里少了资产：{path.relative_to(_ROOT).as_posix()}")
    for relative in _QT_REQUIRED:
        path = internal / relative
        if not path.exists():
            problems.append(f"QtWebEngine 的运行件不在：{path.relative_to(_ROOT).as_posix()}")
    return problems


def _check_warnings(name: str) -> list[str]:
    """第 4 条：读 PyInstaller 自己写的缺失清单，拒收名单里出现谁就失败。

    清单里绝大多数行是无害的（Windows 上找不到 ``posix`` 是正常的），所以只挑
    ``_FORBIDDEN_MISSING`` 这几个名字比。行长得像
    ``missing module named numpy - imported by ...``，名字是第 4 个词。

    **只比整名，不做前缀匹配。** numpy 会留下一大坨**属性式**记录
    （``numpy._core.atleast_2d``、``numpy._core.amax``…实测 175 行），那些名字根本不是
    模块，是 C 扩展的属性访问被记成了模块名；按前缀比的话这份名单天天在响，然后被无视。
    而"整个包不在"的行长相恰恰是整名：``missing module named numpy - imported by ...``。

    **按 ``rglob`` 找而不是拼死路径**：``--workpath build/<name>`` 之后 PyInstaller
    还会在里面再套一层同名目录，清单落在 ``build/<name>/<name>/warn-<name>.txt``。
    那个层级是实现细节，不值得写死在这里。
    """
    work = _BUILD / name
    files = sorted(work.rglob("warn-*.txt"))
    if not files:
        return [
            f"找不到缺失清单（{work.relative_to(_ROOT).as_posix()}/ 底下没有 warn-*.txt）："
            "PyInstaller 是不是没跑到写它那一步？"
        ]

    problems: list[str] = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("missing module named"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            module = parts[3].strip("'\"")
            if module in _FORBIDDEN_MISSING:
                problems.append(f"{path.name}: {stripped}")
    return problems


_ERROR_DIALOG_CLASS = "#32770"
"""Win32 里对话框的窗口类名。PyInstaller 给无控制台产物弹的错误框就是这个类。"""

_CONSOLE_WINDOW_CLASS = "ConsoleWindowClass"
"""带控制台的产物自己开的那个控制台窗口的类名。"""


def _visible_windows(pid: int) -> list[tuple[int, str, str]]:
    """这个进程的**可见顶层窗口**，每项是 ``(窗口句柄, 窗口类, 标题)``。

    用 ctypes 直接问 Win32：这是"不看屏幕也知道窗口开出来没有"的唯一办法，而冻结
    产物恰恰没有别的话可说——它没有控制台，也不打日志。**句柄也带出来**：图标那条
    判据（``WM_GETICON``）要按句柄问。
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[tuple[int, str, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid or not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        kind = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, kind, 256)
        found.append((int(hwnd), kind.value, title.value))
        return True

    user32.EnumWindows(visit, 0)
    return found


def _app_windows(pid: int) -> list[tuple[int, str, str]]:
    """可见顶层窗口里**能当成"起来了"证据**的那些。

    控制台窗口不算：``--console`` 那一版自己一定会开一个，把它算进去的话这条判据对
    ``--console`` 就恒真了——而那一版恰恰是排障用的，它更需要这条判据是真的。
    """
    return [item for item in _visible_windows(pid) if item[1] != _CONSOLE_WINDOW_CLASS]


_WM_GETICON = 0x007F
"""问一个窗口"你身上挂着图标没有"。返回值是 ``HICON``——**0 就是没有**。"""

_SMTO_ABORTIFHUNG = 0x0002
"""消息送不进去就别等：卡住的窗口不该把构建挂在这里。"""

_ICON_SLOTS = (0, 1, 2)
"""``ICON_SMALL`` / ``ICON_BIG`` / ``ICONSMALL2``——标题栏、任务栏、Alt+Tab 各取
一档，三个都空才算"没有图标"。"""


def _app_window_icons(windows: Sequence[tuple[int, str, str]]) -> tuple[int, int, int]:
    """应用窗口上那三个图标句柄，全 0 就是没有（没有窗口也是全 0）。

    "窗口有、图标没有"是冻结产物特有的一种半成品：``--icon`` 嵌进 exe 的那份只决定
    资源管理器里那个文件长什么样，**窗口自己不设就是空的**。这条判据为什么单列，见
    :func:`_probe`。
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    # 返回值是 HICON（64 位上是个指针），不声明 restype 会被截断——"有没有"照样判得
    # 出来，但读到的句柄是假的，而这个值要打印给人看。
    user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
    user32.SendMessageTimeoutW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_ssize_t),
    )

    for hwnd, kind, _title in windows:
        if kind == _ERROR_DIALOG_CLASS:
            continue  # 错误框不算"起来了"，它的图标也不是这条要看的
        slots: list[int] = []
        for slot in _ICON_SLOTS:
            value = ctypes.c_ssize_t()
            user32.SendMessageTimeoutW(
                hwnd, _WM_GETICON, slot, 0, _SMTO_ABORTIFHUNG, 1000, ctypes.byref(value)
            )
            slots.append(int(value.value))
        return slots[0], slots[1], slots[2]
    return 0, 0, 0


#: 探测用的数据目录前缀。落在系统临时目录里，起完就删。
_PROBE_DATA_PREFIX = "axo-probe-"

#: 窗口出现之后、杀进程之前多留的几秒。
#:
#: 窗口是**建到一半**就可见的（``show()`` 一返回它就在了），而首启那几件事——建库、
#: 写内置人格锚——发生在它之后。不留这段时间的话，读到的可能是一个刚建出来、表还没有
#: 的库，于是下面那几条断言会按一种**看起来像产物坏了**的方式报红。留够时间再杀，
#: 顺带让 SQLite 把事务提交完（半途被杀掉的事务会回滚，那样读到的也不是首启形状）。
_FIRST_RUN_GRACE = 3.0


def _builtin_persona_ref() -> str:
    """内置人格锚在那个库里的 ``source_ref``。

    **从被冻结的那份代码里取，不在这里抄一份。** 抄一份的话，某天改了那个常量，
    这里的判据会开始对着一个过时的值点头——而它点头的意思是"库是干净的"。
    """
    sys.path.insert(0, str(_ROOT / "src"))  # 与其他 tools/ 脚本同一手法
    from axo.memory.persona import BUILTIN_REF

    return BUILTIN_REF


def _first_run_shape(data: Path) -> list[str]:
    """**别人解压之后第一次启动，那个库该长什么样。**

    这是"下载到的是最初始化的版本"的机器判据。只断三件事，每一件都对着一句人话：

    1. **库建出来了。** 建不出来说明首启路径根本没走到，后面两条也就无从谈起。
    2. **库里只有一条记忆，而且它是内置人格锚。** 这一条同时挡两件事：发布前有人拿
       自己的库做过实验（多出来的条目），以及内置锚的写入路径坏掉（零条）。
    3. **没有逐字稿、没有壁纸。** 它们是"用户的东西"里最容易顺手带出来的两类，而
       ``sessions/*.jsonl`` 正是 .gitignore 与仓库卫生守卫反复围堵的那个形状。

    **不查"目录里有没有多余文件"。** 那是一张白名单，而白名单会随程序加功能天天红，
    红到没人看为止。这里查的是内容：库里有什么、有没有那两类东西。
    """
    db = data / "memory" / "mem.db"
    if not db.is_file():
        return [f"隔离的首启数据目录里没有建出库：{db}"]

    try:
        # **只读打开。** 让构建工具自己去建表的话，"产物有没有建库"这条判据就变成了
        # "构建工具建了一次库"——而两者在盘上长得一模一样。
        connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return [f"打不开首启建出来的库（{exc}）"]
    try:
        rows = connection.execute("SELECT memory_id, source_ref FROM memories").fetchall()
    except sqlite3.Error as exc:
        return [f"首启建出来的库读不了 memories 表（{exc}）"]
    finally:
        connection.close()

    problems: list[str] = []
    if len(rows) != 1:
        problems.append(
            f"首启之后库里有 {len(rows)} 条记忆，应该只有 1 条（内置人格锚）。"
            "多出来的条目说明这个产物里夹带了别人的数据。"
        )
    elif rows[0][1] != _builtin_persona_ref():
        problems.append(
            f"首启之后库里那条记忆不是内置人格锚（source_ref={rows[0][1]!r}）——"
            "多半是发布前拿自己的库做过实验。"
        )

    sessions = data / "sessions"
    transcripts = sorted(sessions.glob("*.jsonl")) if sessions.is_dir() else []
    if transcripts:
        problems.append(
            f"首启之后出现了 {len(transcripts)} 份逐字稿（{transcripts[0].name} 等）——"
            "那是使用者自己的东西，不该出现在交付物里。"
        )
    wallpapers = data / "wallpapers"
    files = (
        sorted(item for item in wallpapers.iterdir() if item.is_file())
        if wallpapers.is_dir()
        else []
    )
    if files:
        problems.append(f"首启之后壁纸目录里有 {len(files)} 个文件。")
    return problems


def _probe(name: str, wait: float) -> list[str]:
    """第 5 条：真起一次，然后看**有没有真的开出窗口**。

    **判据不是"进程还在不在"。** 打包坏掉时进程不一定立刻退出：无控制台版本撞上
    未捕获异常会弹一个模态错误框，进程于是活着、等到天荒地老。这条是实测踩出来的
    ——第一次构建时入口的相对 import 解析不了，exe 弹的正是那个框，而"还活着"给了
    **假绿**。

    **子进程的环境里必须没有 ``QT_QPA_PLATFORM``。** 画图标那一步给**本进程**设过
    ``offscreen``（见 :func:`_icon_pngs`），而 ``Popen`` 默认继承环境——被继承的话 Qt
    根本不建 Win32 窗口，于是这条探测**每次**都报"没有任何可见窗口"。同样是实测踩出来
    的：那份报错是这条自己的假症状，不是产物的毛病。

    **等到窗口出现为止，不是等满固定时长。** 冻结产物冷启要十几秒（目录 590 MB 上下、
    首次读盘），固定时长只能靠猜：猜短了误报，猜长了每次构建都干等。这里每 0.25 秒看
    一次，出现即过。

    判据是四条：进程活着，**且**它名下有一个可见的顶层窗口、且那不是个对话框，
    **且**那个窗口上挂着图标，**且**它在隔离的数据目录里首启出来的库是空的
    （见 :func:`_first_run_shape`）。窗口里长得对不对不归这条管——那只能靠眼睛，
    是 README 里交给人做的那一步。

    **数据目录必须隔离。** 这一段原来是漏的：探测只弹掉了 ``QT_QPA_PLATFORM``，
    于是子进程按 ``paths.data_dir()`` 落到了 ``%LOCALAPPDATA%\axo``——
    **每次构建都拿使用者的真库起一遍应用**，往真库里写内置锚、真库里本来就有的东西
    也一并被这条路读到。而这一切在构建日志里一个字都不显示。给子进程一个临时目录
    之后，"这次构建碰过我的库没有"这个问题才有答案，而且答案是"没有"。

    **图标为什么要机器验。** 它是这条探测里唯一"窗口开出来了、看着像好了、其实是半
    成品"的那一档：``--icon`` 嵌进 exe 的那份只影响资源管理器里那个文件的外观，窗口
    自己不设图标，标题栏左上角就空着一格。用户实测报过一次，而三处（标题栏、任务栏、
    Alt+Tab）里只有一处会被顺手注意到。
    """
    exe = _DIST / name / f"{name}.exe"
    if not exe.is_file():
        return [f"没有产物可起：{exe.relative_to(_ROOT).as_posix()}"]

    environment = os.environ.copy()
    environment.pop("QT_QPA_PLATFORM", None)
    # **隔离数据目录。** 理由见这个函数的 docstring：不设这一句，构建自检就是拿
    # 使用者的真库在起应用，而构建日志里一个字都看不出来。
    probe_data = Path(tempfile.mkdtemp(prefix=_PROBE_DATA_PREFIX))
    environment["AXO_DATA"] = str(probe_data)

    process = subprocess.Popen([str(exe)], cwd=str(exe.parent), env=environment)
    code: int | None = None
    windows: list[tuple[int, str, str]] = []
    icons: tuple[int, int, int] = (0, 0, 0)
    deadline = time.monotonic() + wait
    try:
        while True:
            code = process.poll()
            if code is not None:
                break
            windows = _app_windows(process.pid)
            if windows:
                # 图标在**建原生窗口那一步**就交出去了（先装图标、后 ``show()``，
                # ``show()`` 返回时 ``WM_GETICON`` 已经非零——实测过），所以窗口一
                # 出现就可以问，不必留宽限。
                icons = _app_window_icons(windows)
                # 但库里那几条要再等一会儿：窗口可见的时候，建库与写内置人格锚未必跑完。
                time.sleep(_FIRST_RUN_GRACE)
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    try:
        if code is not None:
            return [
                f"起不来：{exe.name} 在 {wait:.0f} 秒内自己退了（退出码 {code}）。"
                "换 --console 那一版重跑，回溯会打在控制台上。"
            ]

        dialogs = [
            f"窗口类 {kind}、标题 {title!r}"
            for _hwnd, kind, title in windows
            if kind == _ERROR_DIALOG_CLASS
        ]
        if dialogs:
            return [
                f"{exe.name} 起来了，但弹的是错误框（{'; '.join(dialogs)}）。"
                "换 --console 那一版重跑，回溯会打在控制台上。"
            ]
        if not windows:
            return [
                f"{exe.name} 进程活着，但 {wait:.0f} 秒里没有任何可见窗口——"
                "多半卡在启动路径上（缺资源或缺平台插件）。换 --console 那一版重跑。"
            ]
        if not any(icons):
            return [
                f"{exe.name} 起来了，窗口上却没有图标（WM_GETICON 三个槽全是 0）——"
                "标题栏左上角、任务栏、Alt+Tab 三处都会是空的。装图标的是 "
                "src/axo/ui/main_window.py 的 _install_window_icon（由 "
                "_apply_theme 调用）。"
            ]
        # 窗口与图标都对上了，最后一条：它在**隔离的**数据目录里首启出来的库长什么样。
        return _first_run_shape(probe_data)
    finally:
        shutil.rmtree(probe_data, ignore_errors=True)


def _size_of(path: Path) -> int:
    """一个文件或一棵目录树的字节数（目录递归）。"""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _print_report(name: str) -> None:
    """打印产物体积与 ``_internal`` 下最占地方的几项。

    **这是"下次要砍体积"的唯一依据**：砍之前得先知道大块在哪，而"体积比上次小了"这句话
    只有在同一处打印出来的数字之间才可比。
    """
    product = _DIST / name
    print(
        f"\n产物：{product.relative_to(_ROOT).as_posix()}（{_size_of(product) / 1024**2:.1f} MB）"
    )
    rows = sorted(
        ((_size_of(item), item.name) for item in (product / "_internal").iterdir()),
        reverse=True,
    )
    for size, item_name in rows[:6]:
        print(f"  {size / 1024**2:9.1f} MB  {item_name}")


# ── 发布 ────────────────────────────────────────────────────────────────────


NOTICES_NAME = "THIRD-PARTY-NOTICES.md"
"""仓根那份第三方清单。**它同时是两个东西**：给人读的文档，和打包时逐条对账的判据
（那张表夹在 ``frozen-set:begin`` / ``frozen-set:end`` 之间）。随 zip 出去的名字是
``THIRD-PARTY-NOTICES.txt``——zip 里一律 ASCII 文件名，理由见 :func:`_make_zip`。"""

RELEASE_SUFFIX = "win64"
"""zip 名字里的平台后缀。产物只有 Windows x64 这一份（PyInstaller 不能交叉编译），
所以这不是"可选值之一"，是在文档里写死的事实。"""

_GPL_ONLY_MODULES = (
    "Qt6Charts.dll",
    "Qt6ChartsQml.dll",
    "Qt6DataVisualization.dll",
    "Qt6DataVisualizationQml.dll",
    "Qt6Graphs.dll",
    "Qt6VirtualKeyboard.dll",
    "Qt6VirtualKeyboardQml.dll",
    "Qt6VirtualKeyboardSettings.dll",
)
"""随产物出去就不该有的那几族 Qt 模块，相对 ``_internal/PySide6/``。

**为什么是这些。** PyInstaller 的 PySide6 钩子把整个 wheel 的 DLL 照单全收（实测
154 个 DLL、464 MB），而 Qt 官方把这四族（Charts / DataVisualization / Graphs /
VirtualKeyboard）历来只在 GPLv3 下提供——本项目按 LGPL-3.0 那一支用 Qt，带上它们就
等于把一个 GPL-only 的部件塞进一份声称只有宽松许可与 LGPLv3 的产物里。

**为什么删了在两种世界里都成立。** Qt 官方那张模块表在这里核不到（``doc.qt.io``
在本机与 WebFetch 都被拦），所以"这几族到底是不是 GPL-only"没有定论。但两种答案下
删除都是对的：真是 GPL-only，删除消掉了这个风险；其实是 LGPLv3，只是少了几个本项目
一次都不 import 的文件。展开写在 ``THIRD-PARTY-NOTICES.md`` 第二节。
"""

_GPL_ONLY_COMPANIONS = (
    "PySide6/qml/QtCharts",
    "PySide6/qml/QtDataVisualization",
    "PySide6/qml/QtGraphs",
    "PySide6/qml/QtQuick/VirtualKeyboard",
    "PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
)
"""**只为加载上面那几族而存在**的文件：四个 QML 插件目录，加一个输入法插件。

它们不在那 8 个 DLL 里，但每一个的导入表里都写着其中一个的名字——实测：3148 个文件
全树扫下来，提到那 8 个名字的**只有这 5 条路径下的文件**。剔了 DLL 却留下它们，产物
里就有几个指向不存在的东西的二进制；一个能加载、加载完就失败的插件，比干脆没有更
难查。所以它们跟着一起走，而 :func:`_check_gpl_pruned` 会回头证明"留下来的文件里没
有一个还提到那 8 个名字"。
"""

_FORBIDDEN_GLOBS = ("*.db", "*.db-wal", "*.db-shm", "*.log", "*.jsonl")
"""产物树里不许出现的文件形状。

每一条都对着一样**使用者的东西**：``*.db`` 是记忆库（含它那两个 SQLite 的旁路文件），
``*.log`` 是运行日志，``*.jsonl`` 是对话逐字稿（``sessions/`` 下面那个形状，也是
``.gitignore`` 与仓库卫生守卫反复围堵的一个）。

**这份名单短是刻意的。** 加进来的每一条都得是"出现了就一定是坏事"的东西；写成
"所有扩展名都在白名单里"就成了另一回事——白名单会随程序加功能天天红，红到没人看。
"""

_FORBIDDEN_DIRS = ("sessions", "wallpapers", "benchmarks", "tests", "tools", "__pycache__", ".git")
"""产物树里不许出现的目录名。

``sessions`` / ``wallpapers`` 是使用者的逐字稿与壁纸；``benchmarks`` / ``tests`` /
``tools`` 是**仓库里**的东西（召回语料、测试、构建脚本），冻结时一个都不该跟过去；
``__pycache__`` 与 ``.git`` 更是"这是从源码目录复制出来的"的痕迹。实测当前产物里
一个都没有。
"""

_FROZEN_MARKERS = ("<!-- frozen-set:begin -->", "<!-- frozen-set:end -->")
"""``NOTICES_NAME`` 里那张表的首尾标记。用 HTML 注释是因为它在渲染出来的页面上不可见，
而这里要的是"人读的文档"和"机器读的表"是同一份东西——两份迟早会分叉。"""

_FROZEN_EXEMPT = ("pyinstaller",)
"""在表里、但不以 Python 模块的形式出现在 PYZ 里的发行包。

``pyinstaller`` 进产物的是它的 **bootloader**（exe 头部那几 MB 的 C 代码），它自己的
Python 包一个模块都没进去。表里列它的理由是"随产物分发"成立，与"它在 PYZ 里"无关
——所以对账时要豁免这一条，而不是把表改小。
"""

_EXPECTED_WITHOUT_LICENSES = ("biosnn-bus",)
"""dist-info 里确实没有许可正文的那几个，**必须是已知的**。

``biosnn-bus`` 的 PyPI 元数据声明 Apache-2.0，但 wheel 里没带正文，所以它的许可是从
Apache 官网取的、来源写在 ``NOTICES_NAME`` 里。这条名单的作用是让"少了一份许可"
永远是个**要解释的**状态：哪天冒出一个新的，构建红。
"""

_RESIDUE_EVERYWHERE = (str(_ROOT), str(_ROOT.parent))
"""全树搜的串：**这次检出所在的绝对路径**（仓根，以及它上一层）。

从 ``_ROOT`` 算出来而不是写死一个 ``X:\\...``：写死的话这条判据只在作者的机器上有
效，别人在别处检出时它会**静默变成一条永不命中的搜索**；而且 ``tools/build_exe.py``
里"一个字面量绝对路径都没有"这条纪律也保住了
（``tests/test_packaging.py::test_no_absolute_path_is_hardcoded`` 查的正是这个，
它按 ``ast`` 遍历所有字符串常量，宽到连搜索用的针也算——所以针也得是算出来的）。

**这条可以全树搜。** 它在产物里没有任何正当来源：数据目录是运行时按
``%LOCALAPPDATA%`` 解析的，而第三方的二进制也不会写它——Qt 的构建机在
``C:\\Users\\qt\\...``、CPython 的在 ``C:\\Users\\Administrator\\...``（这两条实测过，
见 :data:`_OWN_RESIDUE_SOURCES`）。
"""

_OWN_RESIDUE_SOURCES = (str(Path.home().parent), str(Path.home()))
"""**只在本仓自己的文件里**搜的串：用户目录的上一层（Windows 上就是 ``C:\\Users``）
与用户目录本身。

也不能全树搜，原因是实测的：产物 3157 个文件里有 270 个（Qt 的 DLL、CPython 的
``.pyd``）嵌着**它们构建机**的路径，全是 ``C:\\Users\\qt\\work\\...`` 与
``C:\\Users\\Administrator\\AppData\\Local\\Temp\\python-build-...``。那不是我们的
痕迹，收了这条判据就天天在响，然后被无视——而被无视的判据等于没有。

我们自己的文件没有这个豁免：它们只要有这两个串，来源就只可能是写死的绝对路径。
两条都从环境算出来，理由同 :data:`_RESIDUE_EVERYWHERE`。
"""

_RELEASE_README = """Axo {version}（Windows x64）
=======================================

这是什么
--------
一个本机的长期记忆终端。记忆和性格都存在你这台机器上的一个库里，独立于你所调用的
大模型——换模型不丢记忆，断网也能翻自己记过的东西。

怎么开始
--------
1. 把整个压缩包解压到任意一个你有写权限的位置。**注意是解压，不是只把 exe 拖出来**：
   它旁边的 axo 文件夹是程序的一部分，缺了就跑不起来。
2. 双击 axo.exe。
3. Windows 多半会拦一次，说"已保护你的电脑"（这个程序没有购买代码签名证书）。
   点"更多信息"，再点"仍要运行"。
4. 第一次启动是空的，它会问你一次想让它记住的人设。

要能聊天，需要一个模型，下面两条任选一条：

  · 本机跑（免费、不用联网）
      先装 Ollama：https://ollama.com/download
      装完在命令行里拉一个模型：ollama pull qwen3:4b
    程序默认就找本机的 Ollama，不需要额外设置。

  · 用云端
      在程序的"设置"里填一个 OpenAI 兼容端点的地址与 API key（比如 DeepSeek）。
      填进去的 key 会用 Windows 自带的加密接口（DPAPI）加密后存起来，只有当前这个
      Windows 账户解得开。

想让它在记忆里"按意思"找，而不只是按字面匹配，还需要一个嵌入模型：

      ollama pull nomic-embed-text

数据放在哪、怎么卸载
--------------------
你记的东西全部在这个文件夹里：

    %LOCALAPPDATA%\\axo

删掉它，记忆就没了。卸载程序本身只需要删掉解压出来的那个文件夹——程序不往别处写
文件。唯一的例外是界面与模型设置，它们记在注册表里，位置是
HKEY_CURRENT_USER\\Software\\Axo\\axo，删掉即清空。

想让数据放到别处（比如移动硬盘），启动前设一个环境变量：

    set AXO_DATA=<你想要的目录>

先说清楚哪里还不行
------------------
· 没有代码签名证书，所以 SmartScreen 会拦一次。
· 只有 Windows 64 位这一版。
· 还没有 Live2D 形象（素材未到），对话区是普通的网页渲染。
· 既没装 Ollama、又没填云端 key 的话，程序能打开，但没法对话。

许可
----
本程序按 Apache-2.0 发布，全文见 LICENSE。
它里面用了别人的东西，清单在 THIRD-PARTY-NOTICES.txt。其中唯一带 copyleft 的是
Qt / PySide6，按 LGPL-3.0 使用，两份许可全文在 licenses 文件夹下的 GPL-3.0.txt 与
LGPL-3.0.txt；其余第三方的许可文本按发行包分目录放在同一个 licenses 文件夹里。
"""
"""zip 里那份 ``README.txt``：**写给拿到它的人**，不是写给仓库里的自己。

判据是"不出现仓库内部的行话"（由
``tests/test_packaging.py::test_the_release_readme_speaks_to_its_reader`` 机械地查）：
P6a、ADR、腿、守卫、语料这些词在仓库里是精确的，对读者是噪音。

``{version}`` 是个占位符而不是 f-string：这样它保持**模块级字面量常量**，测试能用
``ast.literal_eval`` 读到它（:func:`_declared` 那一套），打包时再替换。
"""


def _canonical(distribution: str) -> str:
    """PEP 503 的规范化名：``-``/``_``/``.`` 一律当 ``-``，小写。

    表里写 ``typing_extensions``、``PySide6_WebEngine``，而 ``importlib.metadata``
    认的是安装时那个拼法——两边都可能出现，所以规范化之后再比。
    """
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _project_metadata() -> dict:
    """``pyproject.toml`` 的 ``[project]`` 段。"""
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _release_version() -> str:
    """版本从 ``pyproject.toml`` 现读——**不在这里抄第二份常量**。

    抄一份的话，"改了版本号而 zip 名字没跟上"的那一刻没有任何东西会红，而 zip 一旦
    发出去就收不回。
    """
    return str(_project_metadata()["version"])


def _declared_frozen_set() -> list[tuple[str, str]]:
    """``NOTICES_NAME`` 里 ``frozen-set`` 之间那张表的 ``(发行包, 版本)``。

    只认以 ``|`` 开头的行，跳过表头与分隔行。找不到标记就抛——**不静默返回空表**：
    空表会让对账变成"产物里多了一堆没登记的包"，报出来的是一串症状而不是原因。
    """
    text = (_ROOT / NOTICES_NAME).read_text(encoding="utf-8")
    start_marker, end_marker = _FROZEN_MARKERS
    try:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker)
    except ValueError:
        raise ValueError(
            f"{NOTICES_NAME} 里找不到 {' / '.join(_FROZEN_MARKERS)} 标记——"
            "对账没有依据，而这份清单的全部意义就是能被对账。"
        ) from None
    rows: list[tuple[str, str]] = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or set(cells[1]) <= {"-", ":"} or cells[0] == "发行包":
            continue
        rows.append((cells[0], cells[1]))
    if not rows:
        raise ValueError(f"{NOTICES_NAME} 的 frozen-set 之间一条都没解析出来。")
    return rows


def _frozen_modules(name: str) -> set[str]:
    """产物里**实际被冻结的**顶层模块名，取自 ``PYZ-*.toc``。

    **不能看 ``_internal/``。** PyInstaller 的 onedir 把全部纯 Python 字节码（本仓的、
    以及 httpx 一类的第三方）压进 exe 体内那个 PYZ 归档，``_internal/`` 只剩二进制与
    数据资产。这个坑实测踩过一次：按 ``_internal/`` 找 httpx 会得出"没带进去"的结论，
    而它就在 exe 里。

    ``.toc`` 是 ``repr`` 出来的 Python 字面量（``(归档路径, [(模块名, 源路径, 类型)])``），
    所以 ``ast.literal_eval`` 直接读得了——不用去解 PyInstaller 的内部格式。
    """
    files = sorted((_BUILD / name).rglob("PYZ-*.toc"))
    if not files:
        raise FileNotFoundError(
            f"找不到 {(_BUILD / name).relative_to(_ROOT).as_posix()}/ 底下的 PYZ-*.toc："
            "PyInstaller 是不是没跑到写它那一步？"
        )
    _archive, entries = ast.literal_eval(files[0].read_text(encoding="utf-8"))
    return {entry[0].split(".")[0] for entry in entries if entry[2] == "PYMODULE"}


def _distribution_license_files(distribution: str) -> list[tuple[str, Path]]:
    """一个发行包随包发的许可文件：``[(相对 licenses/ 的路径, 磁盘上的路径)]``。

    **只认 ``<发行包>-<版本>.dist-info/licenses/`` 这一层。** 放宽成"路径里有
    ``/licenses/``"会捞到 ``packaging/licenses/__init__.py``——那是**代码**，不是许可
    文本。实测踩过：宽松那条筛出 40 份，其中混着两个 ``.py``。
    """
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return []
    marker = ".dist-info/licenses/"
    found: list[tuple[str, Path]] = []
    for entry in dist.files or ():
        posix = str(entry).replace("\\", "/")
        if marker not in posix or posix.endswith("/"):
            continue
        found.append((posix.split(marker, 1)[1], Path(dist.locate_file(entry))))
    return found


def _matching_needles(path: Path, needles: Collection[bytes]) -> set[bytes]:
    """这个文件里出现了 ``needles`` 中的哪几个。

    用 ``mmap`` 而不是 ``read_bytes``：产物里最大的单文件是 QtWebEngineCore（一百多
    MB），而发布闸门要把它扫上几遍；映射进地址空间既不复制，常驻内存也不跟着文件大小
    走。空文件 ``mmap`` 不收（长度 0），直接给一个空结果——空文件里当然什么都没有。
    """
    if path.stat().st_size == 0:
        return set()
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as window,
    ):
        return {needle for needle in needles if window.find(needle) >= 0}


def _gpl_mentions(internal: Path, skip: Collection[Path]) -> dict[str, list[str]]:
    """``DLL 名 -> 提到它的那些文件``，扫描范围是 ``internal`` 全树**减去** ``skip``。

    按字节搜而不是解析导入表：这些文件里有 PE、有 QML、有纯文本，一次覆盖全部；而按
    格式分别解析要写好几套代码，还没有这一条准。名字是 ASCII（``Qt6Charts.dll``），
    PE 的导入名也是 ASCII 字节串，所以直接编码了搜。
    """
    needles = {module.encode("ascii"): module for module in _GPL_ONLY_MODULES}
    found: dict[str, list[str]] = {}
    for path in internal.rglob("*"):
        if not path.is_file() or path in skip:
            continue
        where = path.relative_to(internal).as_posix()
        for needle in _matching_needles(path, set(needles)):
            found.setdefault(needles[needle], []).append(where)
    return found


def _gpl_only_victims(internal: Path) -> list[Path]:
    """这一族要删的文件，展开成具体路径。"""
    victims = [internal / "PySide6" / module for module in _GPL_ONLY_MODULES]
    for relative in _GPL_ONLY_COMPANIONS:
        path = internal / relative
        if path.is_dir():
            victims += [item for item in path.rglob("*") if item.is_file()]
        else:
            victims.append(path)
    return sorted(item for item in victims if item.exists())


def _prune_gpl_only(name: str) -> tuple[list[str], list[str]]:
    """剔掉 GPL-only 那几族，返回 ``(删掉了什么, 拒绝删的理由)``。

    **删之前先证明删得干净。** 判据是：全树扫一遍，除了**这次要删的那些文件自己**，
    没有任何别的文件提到过那 8 个 DLL 名。提到就**一个都不删**并报红——半剔的树比没剔
    的更坏：它既没消掉许可风险，又留下一堆指向空处的插件。

    这是"结果判据"而不是"意图"：不是"我认为这些没人用"，是"扫过了，没有别人提到它们"。
    哪天 Qt 多出一个引用者，这条会当场红，而不是等它变成用户机器上的一次崩溃。
    """
    internal = _DIST / name / "_internal"
    victims = _gpl_only_victims(internal)
    refused = [
        f"要删 {module}，但还有 {len(where)} 个文件提到它：{_summarise(where)}。"
        "这说明剔除清单不全——把提到它的那个文件也写进 _GPL_ONLY_COMPANIONS，"
        "或者确认它到底属不属于 GPL-only 那一族。**这次一个都没删。**"
        for module, where in _gpl_mentions(internal, set(victims)).items()
    ]
    if refused:
        return [], refused

    removed: list[str] = []
    for module in _GPL_ONLY_MODULES:
        path = internal / "PySide6" / module
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            removed.append(f"{path.relative_to(internal).as_posix()}（{size / 1024**2:.1f} MB）")
    for relative in _GPL_ONLY_COMPANIONS:
        path = internal / relative
        if path.is_dir():
            size = _size_of(path)
            shutil.rmtree(path)
            removed.append(f"{relative}/（{size / 1024**2:.1f} MB）")
        elif path.is_file():
            size = path.stat().st_size
            path.unlink()
            removed.append(f"{relative}（{size / 1024**2:.1f} MB）")
    return removed, []


def _check_gpl_pruned(name: str) -> list[str]:
    """**结果判据**：那几族真的没了，而且留下来的文件里没有一个还提到它们。

    两条都要。第一条挡"glob 没匹配上、其实什么都没删"；第二条挡"删漏了某个引用者"
    ——只查第一条的话，``_GPL_ONLY_COMPANIONS`` 里写错一个文件名就会静默漏删，而
    漏掉的那个插件会带着指向不存在 DLL 的导入表一起出门。
    """
    internal = _DIST / name / "_internal"
    problems = [
        f"{module} 还在产物里，剔除没有生效"
        for module in _GPL_ONLY_MODULES
        if (internal / "PySide6" / module).exists()
    ]
    problems += [
        f"剔完之后还有文件提到 {module}：{_summarise(where)}"
        for module, where in _gpl_mentions(internal, frozenset()).items()
    ]
    return problems


def _summarise(items: Sequence[str], limit: int = 3) -> str:
    """把一串路径压成一行：前 ``limit`` 条，多出来的报个数。

    构建日志里的这些行是**给人看的线索**，不是给人读的清单——列全了会淹掉报错本身。
    """
    head = "; ".join(items[:limit])
    return head if len(items) <= limit else f"{head} 等 {len(items)} 个"


def _own_files(name: str) -> list[Path]:
    """产物里**属于这个项目自己**的文件。

    两条：顶层的 exe（本仓全部 Python 字节码都在它体内的 PYZ 里，第三方的一并也在
    ——所以它照样得过"没有用户目录路径"那一关），以及 ``_internal/axo/``
    （``theme.qss`` 与 ``ui/web/`` 那几份文本资产）。

    按**路径**划而不是按扩展名划：按扩展名划要靠一张"什么算文本"的清单，而那张清单
    会在有人加一种新资产的时候悄悄漏掉它。
    """
    root = _DIST / name
    files = [item for item in root.glob("*.exe") if item.is_file()]
    package = root / "_internal" / "axo"
    if package.is_dir():
        files += [item for item in package.rglob("*") if item.is_file()]
    return sorted(files)


def _hits(paths: Sequence[Path], root: Path, texts: Sequence[str]) -> list[str]:
    """在 ``paths`` 里搜 ``texts``，**UTF-8 与 UTF-16LE 两种编码都搜**。

    UTF-16LE 不是凑数：exe 里的字符串资源与 Qt 的翻译件都是宽字符，只搜 UTF-8 会漏掉
    整整一半可能藏东西的地方——而漏掉的那一半恰好是"以为搜过了"的那种漏。
    """
    needles = {
        (text, encoding): encoded
        for text in texts
        for encoded, encoding in ((text.encode(), "utf-8"), (text.encode("utf-16-le"), "utf-16-le"))
    }
    found: list[str] = []
    for path in paths:
        hits = _matching_needles(path, set(needles.values()))
        found += [
            f"{path.relative_to(root).as_posix()} 里有 {text!r}（{encoding}）"
            for (text, encoding), encoded in sorted(needles.items())
            if encoded in hits
        ]
    return found


def _check_residue(name: str) -> list[str]:
    """发布前的最后一道内容闸门：产物里**不许有使用者的东西**。

    两类判据，扫的范围不一样，理由都写在 :data:`_RESIDUE_EVERYWHERE` 与
    :data:`_OWN_RESIDUE_SOURCES` 上——一句话是"全树只搜检出路径，用户目录那两条只搜
    自己的文件"。

    **笔名不进这份名单。** 它在公开源码里合法存在（``tests/test_repo_hygiene.py`` 的
    禁词表里就有它，而那张表本身是公开仓的一部分），产物里有它不算问题。
    """
    root = _DIST / name
    problems: list[str] = []

    for pattern in _FORBIDDEN_GLOBS:
        for path in sorted(root.rglob(pattern)):
            problems.append(f"产物里有不该有的文件：{path.relative_to(root).as_posix()}")
    for item in sorted(root.rglob("*")):
        if item.is_dir() and item.name in _FORBIDDEN_DIRS:
            problems.append(f"产物里有不该有的目录：{item.relative_to(root).as_posix()}/")

    everything = [item for item in root.rglob("*") if item.is_file()]
    problems += _hits(everything, root, _RESIDUE_EVERYWHERE)
    problems += _hits(_own_files(name), root, _OWN_RESIDUE_SOURCES)
    return problems


def _check_frozen_set(name: str) -> list[str]:
    """与 ``NOTICES_NAME`` 那张表逐条对账：多一个、少一个、版本对不上，都红。

    **为什么清单必须是判据而不是文档。** 一份手写的依赖清单，唯一的结局是慢慢与现实
    脱节：某天有人加了一个依赖、清单上没写，而 zip 里带着它出去了。这类失效之所以危险，
    是因为它看起来一切正常——清单还在，读上去很完整。

    两边不一样的时候**不自动决定该改哪边**：工具只负责让差异可见，改哪边是人的事。
    """
    try:
        declared = {_canonical(row[0]): row[1] for row in _declared_frozen_set()}
    except ValueError as error:
        return [str(error)]
    try:
        tops = _frozen_modules(name)
    except FileNotFoundError as error:
        return [str(error)]

    owners = importlib.metadata.packages_distributions()
    installed = {
        _canonical(str(dist.metadata["Name"])): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    mine = _canonical(str(_project_metadata()["name"]))

    actual: dict[str, str] = {}
    for top in sorted(tops):
        for owner in owners.get(top, ()):
            key = _canonical(owner)
            if key != mine:  # 本仓自己那两个包不进这张表，它们不是第三方
                actual[key] = installed.get(key, "（认不出是哪个发行包）")

    problems: list[str] = []
    for key in sorted(set(actual) - set(declared)):
        problems.append(
            f"产物里多了一个没登记的发行包：{key} {actual[key]}。"
            f"要么把它写进 {NOTICES_NAME} 的 frozen-set 表，要么别让它进产物。"
        )
    for key in sorted(set(declared) - set(actual) - {_canonical(item) for item in _FROZEN_EXEMPT}):
        problems.append(
            f"{NOTICES_NAME} 里登记了 {key} {declared[key]}，但产物里没有它的模块。"
            "要么它已经不在产物里了（把表改小），要么它换了名字（两边一起改）。"
        )
    for key in sorted(set(actual) & set(declared)):
        if actual[key] != declared[key]:
            problems.append(
                f"{key} 的版本对不上：产物里是 {actual[key]}，{NOTICES_NAME} 里写的是 {declared[key]}。"
            )
    return problems


def _check_licenses() -> list[str]:
    """随 zip 出去的许可文本**取得到**。

    **判据是"取得到"，不是"仓库里有"。** 这两件事会分开：某天一个依赖改成不随包发
    许可文件，仓库里那份清单照样在，而 zip 里少了一份义务。所以这里现从装好的
    dist-info 里找一遍，找不到就是要报的那一条。

    LGPLv3 与 GPLv3 那两份单列：上游 wheel 里**一份都没有**（PySide6 每个 dist-info
    只带一份 ``LicenseRef-Qt-Commercial.txt``，实测过），所以它们只能自己从 gnu.org
    取一次存进仓，并用 sha256 钉住（``tests/test_licenses.py``）。它们不在了，这里要
    报的是一句人话，而不是一个文件找不到。
    """
    problems: list[str] = []
    for relative in ("licenses/GPL-3.0.txt", "licenses/LGPL-3.0.txt"):
        if not (_ROOT / relative).is_file():
            problems.append(
                f"仓里没有 {relative}。LGPLv3 §4(b) 要的是给每个收件人一份副本，"
                "而 PySide6 的 wheel 里没有这两份全文——只能自己取一次存进仓。"
            )
    exempt = {_canonical(item) for item in _EXPECTED_WITHOUT_LICENSES}
    try:
        declared = _declared_frozen_set()
    except ValueError as error:
        return [*problems, str(error)]
    for distribution, _version in declared:
        if _canonical(distribution) in exempt:
            continue
        if not _distribution_license_files(distribution):
            problems.append(
                f"{distribution} 的 dist-info 里一份许可文件都没有，zip 里会少一份义务。"
                f"若它本来就不带（已知的有 {', '.join(_EXPECTED_WITHOUT_LICENSES)}），"
                "把它加进 _EXPECTED_WITHOUT_LICENSES 并在 NOTICES 里写明来源。"
            )
    return problems


def _dos_text(text: str) -> bytes:
    """CRLF + UTF-8 的字节。

    **显式做，不靠平台默认。** 记事本是 Windows 上打开 ``README.txt`` 的第一个工具，
    它对 LF-only 的容忍度逐版本在变；而 ``Path.write_text`` 的默认换行行为又随平台走
    ——两处"看情况"叠起来，就是一个只在别人机器上复现的连行问题。先归一再转，所以
    源文件是 LF 还是 CRLF 都不影响结果。
    """
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def _file_digest(path: Path) -> str:
    """一个文件的 sha256。

    不用 ``hashlib.file_digest``：它是 3.11 才有的，而本仓声明 ``requires-python
    >= 3.10``——打包脚本不该比它打包的那个程序要求更高的解释器。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_zip(name: str) -> Path:
    """把产物、许可与说明打成 ``dist/<名字>-<版本>-win64.zip``，返回它的路径。

    **归档内文件名一律 ASCII**（``README.txt``，不是 ``使用说明.txt``）：中文文件名在
    zip 里要靠一个通用的 UTF-8 标志位，老解压器会乱码成一串问号——而**正文照旧中文**，
    乱码只发生在文件名上，恰好是最容易被当成"文件坏了"的那一半。所以这条在写之前
    先断，红了就不出包。

    **跑在 ``dist/`` 里**（``.gitignore`` 已经挡住），所以它永远不会进仓——几百 MB 的
    二进制进版本控制就再也删不掉了。
    """
    version = _release_version()
    top = f"{NAME}-{version}"
    target = _DIST / f"{top}-{RELEASE_SUFFIX}.zip"

    product = _DIST / name
    sources: list[tuple[str, Path]] = [
        (f"{top}/{name}/{item.relative_to(product).as_posix()}", item)
        for item in sorted(product.rglob("*"))
        if item.is_file()
    ]
    sources.append((f"{top}/LICENSE", _ROOT / "LICENSE"))
    for item in sorted((_ROOT / "licenses").iterdir()):
        if item.is_file():
            sources.append((f"{top}/licenses/{item.name}", item))
    for distribution, _declared_version in _declared_frozen_set():
        # 按发行包分目录：numpy 一个包就有 17 份许可，平铺会撞名。
        for tail, path in _distribution_license_files(distribution):
            sources.append((f"{top}/licenses/{_canonical(distribution)}/{tail}", path))

    generated = {
        f"{top}/README.txt": _RELEASE_README.replace("{version}", version),
        f"{top}/{Path(NOTICES_NAME).stem}.txt": (_ROOT / NOTICES_NAME).read_text(encoding="utf-8"),
    }

    foreign = sorted({arc for arc, _ in sources if not arc.isascii()})
    if foreign:
        raise SystemExit(
            "归档里的文件名必须全是 ASCII（中文名要靠 UTF-8 标志位，老解压器会乱码）：\n  "
            + "\n  ".join(foreign[:10])
        )

    total = sum(source.stat().st_size for _arc, source in sources)
    print(f"打 {target.name}（{total / 1024**2:.1f} MB 输入，压缩要几分钟）……", flush=True)
    written = 0
    mark = 10
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, source in sources:
            archive.write(source, arcname)
            written += source.stat().st_size
            percent = written * 100 // max(total, 1)
            if percent >= mark:
                print(f"  {percent}%……", flush=True)
                mark = percent + 10
        for arcname, text in generated.items():
            archive.writestr(arcname, _dos_text(text))
    return target


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="只打印要跑的命令，不构建")
    parser.add_argument("--console", action="store_true", help="构建带控制台的排障版本")
    parser.add_argument("--clean", action="store_true", help="构建前让 PyInstaller 清掉缓存")
    parser.add_argument("--no-shortcut", action="store_true", help="构建完不在桌面上放快捷方式")
    parser.add_argument("--wait", type=float, default=15.0, help="活体探测最多等几秒（默认 15）")
    parser.add_argument(
        "--release", action="store_true", help="多跑发布闸门，再打一个发给别人的 zip"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="产物已经在 dist/ 里，跳过 PyInstaller 那一步（其余判据照跑）",
    )
    args = parser.parse_args(argv)

    if args.release and args.console:
        print("--release 不与 --console 同用：发给别人的是那个无控制台的产物。", file=sys.stderr)
        return 2

    name = CONSOLE_NAME if args.console else NAME
    icon_path = _BUILD / f"{name}.ico"

    if args.dry_run:
        # 图标是构建时才画的，所以这里按"会画出来"报路径——dry-run 要看的是命令行。
        print(" ".join(_command(name=name, console=args.console, icon=icon_path, clean=args.clean)))
        return 0

    if args.skip_build:
        print(f"--skip-build：直接用 dist/{name} 里现有的产物，不重新冻结。")
    else:
        _require_pyinstaller()
        icon = icon_path if _draw_icon(icon_path) else None
        command = _command(name=name, console=args.console, icon=icon, clean=args.clean)
        print("$ " + " ".join(command), flush=True)
        code = subprocess.call(command, cwd=str(_ROOT))
        if code != 0:
            print(f"PyInstaller 退出码 {code}。", file=sys.stderr)
            return code

    removed = _prune_debug_resources(name)
    gpl_removed, refused = _prune_gpl_only(name)
    removed += gpl_removed

    problems = _check_layout(name) + _check_warnings(name) + _check_pruned(name)
    if refused:
        # 一棵没剔干净的树：下面两条判据都建立在"剔已经做完"之上，跑出来的只会淹掉
        # 这条真正的原因，而活体探测更没意义——起一个还带着 GPL-only 模块的产物，
        # 验的是一件不打算交付的东西。
        problems += refused
    else:
        problems += _check_gpl_pruned(name)
        problems += _probe(name, args.wait)

    if args.release and not problems:
        problems += _check_residue(name)
        problems += _check_frozen_set(name)
        problems += _check_licenses()

    _print_report(name)
    for item in removed:
        print(f"  删掉：{item}")
    if problems:
        print("自检没过：")
        for problem in problems:
            print(f"  - {problem}")
        if args.release:
            print("没有产出 zip——一个内容不对的压缩包比没有更糟，因为它会被发出去。")
        return 1

    print(
        "自检全过：产物在、镜像里的资产在、QtWebEngine 的运行件在、缺失清单干净、"
        "调试变体与 GPL-only 模块已删、能起来、窗口上带图标、库是首启形状。"
    )
    # 快捷方式放在**自检之后**：指向一个自检没过的产物，比桌面上什么都没有更糟。
    if not (args.console or args.no_shortcut):
        made = _make_shortcut(name)
        if made:
            print(f"桌面快捷方式：{made}")

    if args.release:
        print("发布闸门全过：产物里没有使用者数据、没有本机痕迹，与第三方清单对得上，许可齐。")
        archive = _make_zip(name)
        digest = _file_digest(archive)
        print(f"\n发布包：{archive.relative_to(_ROOT).as_posix()}")
        print(f"  {archive.stat().st_size / 1024**2:.1f} MB")
        print(f"  sha256 {digest}")
        print("  附件上传到 Release，别提交进仓（几百 MB，进了 git 就再也删不掉）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
