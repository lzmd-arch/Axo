"""应用入口。

入口函数只做七件事：**定两处环境变量、建 app、定样式、定记忆位置、建窗口、进事件
循环**。日志、配置、数据库连接、模型探活都不在这里——那些属于各自模块的职责，塞进
入口只会让"启动时到底发生了什么"变得不可读。

那两处环境变量（:func:`configure_isolated_state` 与
:func:`_configure_webengine_environment`）是**唯一必须先于 ``QApplication``** 的
东西，所以它们是入口的职责，而不是某个模块的——Qt 初始化之后再设环境变量对已经
起来的 Chromium 无效，而 ``QStandardPaths`` 的路径问过一次就定死了。

**"定记忆位置"（:func:`~axo.ui.data_root.boot_data_root`）必须排在建窗口之前**，
理由与那两处不同：它不碰 Qt 的全局状态，而是**赶在四个 store 把路径焊死之前**。
待办的搬迁也是在那里真正动盘的——所以不存在"搬到一半、库还开着"那种状态。

``main`` 刻意不 import :mod:`axo.ui.cli`（尚未存在）也不 import
``MainWindow``，后者的导入推迟到 ``QApplication`` 建好之后：PySide6 的窗口类在
构造时会去查 QApplication 的存在与 DPI 设置，先建 app 再 import 是唯一稳妥的顺序。

**样式必须在建窗口之前定好。** 应用级样式表一旦设置，之后建的控件都会按它起手
画第一次；反过来（先建窗口再设样式表）要靠一次全局 repolish 才能追上，而那一步
在有一些原生子控件时会漏。所以 ``setStyle`` 与 ``apply_theme`` 都排在
``MainWindow`` 之前。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .settings_store import SETTINGS_APP, SETTINGS_ORG

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注，运行时不 import Qt
    from PySide6.QtCore import QSettings

__all__ = ["configure_isolated_state", "main", "make_settings"]


#: 设了这个环境变量就强制软件光栅（值不重要，存在即可）
_SOFTWARE_RENDER_ENV = "AXO_SOFTWARE_RENDER"

#: 设了这个环境变量就把**界面设置与 Qt 的缓存**也挪到它指的那个目录里。
#:
#: 只给打包自检与冒烟脚本用（见 ``tools/build_exe.py`` 的活体探测），**不是给用户的
#: 功能**。它与 ``AXO_DATA`` 是两件事、各管各的：那个管记忆与数据放哪，这个管程序
#: 自己往系统里写的那点东西。两个都设上才算真的隔离。
_ISOLATED_STATE_ENV = "AXO_ISOLATED_STATE"


def configure_isolated_state() -> None:
    """把界面设置与 Qt 的缓存重定向到 ``AXO_ISOLATED_STATE`` 那个目录。

    **公开**（不叫 ``_configure_...``）是因为 ``tools/_smoke_*.py`` 那几个脚本自己
    建 ``QApplication``、不走 :func:`main`，而它们在同一条"别碰使用者的真东西"的
    约束下——所以这个函数得能被它们调到。

    **必须早于 ``QApplication``。** ``QStandardPaths.setTestModeEnabled`` 在第一次
    问过路径之后就改不动了，而 ``QSettings`` 的默认格式要在任何 ``QSettings`` 被
    构造之前定下来。

    **为什么需要这一条。** 活体探测原来只设了 ``AXO_DATA``，那只重定向了
    :mod:`axo.paths`——于是每次构建都往**使用者的真注册表**写一次
    ``HKCU\\Software\\Axo\\axo\\backend\\last_model``（拉完模型列表无条件回填那一句），
    并往真的 ``%LOCALAPPDATA%\\Axo\\axo\\cache`` 写 WebEngine 的管线缓存。而这一切在
    构建日志里一个字都不显示——"这次构建碰过我的东西没有"于是没有答案。

    它也**必须先于新手引导落地**：引导那个"走过了"的标记写在同一个注册表里，探测
    一旦把它写成 1，这台机器上引导再也不会出现，而构建日志里同样什么都不显示。

    **只在变量存在时才切。** 无条件切会把所有已有用户的设置从注册表搬到一个空文件
    里，症状是"换了个版本，外观与 API key 全没了"——而 API key 还是 DPAPI 加密的，
    换个位置就解不开。

    缓存那一条用 ``setTestModeEnabled``，它把 ``QStandardPaths`` 那些位置整体挪到
    平台自己的"测试"位置去，而不是挪进下面这个目录。够用（碰不到使用者的真东西），
    代价是留一份在那儿不随构建清理——比"每次构建都改使用者的注册表"轻得多。
    """
    raw = os.environ.get(_ISOLATED_STATE_ENV)
    if not raw:
        return

    from PySide6.QtCore import QSettings, QStandardPaths

    root = Path(raw)
    (root / "settings").mkdir(parents=True, exist_ok=True)

    # **只设 ``setPath``，不设 ``setDefaultFormat``。** 后者看着更直接，实际管不到
    # 我们要用的那个构造——它只影响 ``QSettings(parent)`` 那个单参数形式；而
    # ``QSettings("Axo", "axo")`` 这个两参数的，在 Windows 上**永远**用原生格式
    # （注册表），默认格式说不上话。实测过：设完之后
    # ``QSettings("Axo", "axo").format()`` 仍然是 ``NativeFormat``，于是"隔离"只
    # 成立在日志里。真正让它走 ini 的是 :func:`make_settings`——那个函数显式传格式。
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / "settings"))
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.SystemScope, str(root / "settings")
    )
    QStandardPaths.setTestModeEnabled(True)


def make_settings() -> QSettings:
    """建一个 ``QSettings``。**建 QSettings 的地方都要走这里。**

    两件事只有在这里能同时说清楚：

    1. **隔离模式下显式指定 ini 格式。** 理由见 :func:`configure_isolated_state`
       里那段——两参数的构造在 Windows 上认不出默认格式。少了这一句，
       ``configure_isolated_state`` 就只是在日志里成立。
    2. **组织名/应用名只有一处。** 它们从 ``settings_store`` 来，而这里是唯一把它们
       喂给 Qt 的地方。

    隔离与否**从环境变量现读**，不缓存：调用方可能在任何时候建 QSettings，而缓存会
    让"先建 QSettings 再设环境变量"这种顺序问题变得不可见。
    """
    from PySide6.QtCore import QSettings

    if os.environ.get(_ISOLATED_STATE_ENV):
        return QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, SETTINGS_ORG, SETTINGS_APP
        )
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _configure_webengine_environment() -> None:
    """在 **QApplication 构造之前**处理 Qt WebEngine 的环境变量。

    必须在 Qt 初始化前做完——之后再改环境变量对已经启动的 Chromium 无效。

    **默认不动 GPU 标志。** 硬编码 ``--disable-gpu`` 是一种常见的"保险"，但它
    会砍掉 Live2D 与视频的硬件加速，而本项目**预留了 Live2D 接口**，为了规避一个
    尚未出现的问题先把自己的渲染能力废掉不划算。

    真有显卡驱动让 WebEngine 的 GPU 进程崩溃时，症状是**页面空白但没有报错**
    （日志里只有一行 Fontconfig warning）——这种"静默坏掉"最难排查，所以给一个
    环境变量开关，让排查路径变成一句话：

        set AXO_SOFTWARE_RENDER=1  &  python -m axo.ui.app
    """
    if os.environ.get(_SOFTWARE_RENDER_ENV):
        existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{existing} --disable-gpu".strip()


REMIND_FLAG = "--remind"
"""无头入口那个开关。**与 ``src/axo/autostart.py`` 的同名常量逐字相同**，由
``tests/test_packaging.py`` 断言——因为它写进注册表之后就固定在那儿了：两处改岔了，
已经设过自启的机器会在下次登录时把一个不认识的参数交给程序。

**同一个 exe 分流，不新增产物。** 第二个入口会带出第二份 ~520 MB 的 ``_internal``，
而"把第二个 exe 塞进同一个 ``_internal`` 旁边"依赖 PyInstaller 的实现细节，属于这个仓
最不信任的那类"看起来在工作"。
"""


def _remind_once() -> int:
    """登录时那一次：看一眼有没有到点的提醒，弹一条就退（见 :mod:`axo.headless`）。

    **数据目录必须走同一套四档解析**（``resolve_data_root``）。这一句漏了的症状最阴：
    用户在设置里把数据目录搬到了别处，而无头入口还在读默认位置的那一本**空**提醒簿，
    于是**每天开机什么都不做，界面上一个字都不显示**。
    """
    from ..agent.reminders import CancelledStore, FiredStore, ReminderStore
    from ..headless import run
    from ..paths import reminders_cancelled_path, reminders_fired_path, reminders_path
    from .data_root import resolve_data_root
    from .notify import show_balloon

    resolve_data_root(make_settings())
    return run(
        reminders=ReminderStore(reminders_path()),
        fired=FiredStore(reminders_fired_path()),
        cancelled=CancelledStore(reminders_cancelled_path()),
        now=time.time(),
        notify=show_balloon,
    )


def main(argv: list[str] | None = None) -> int:
    """启动桌面终端，返回进程退出码。**带 ``--remind`` 时走无头那一支。**"""
    args = list(sys.argv[1:] if argv is None else argv)
    # 环境那两件**必须**排在最前：都要在 Qt 初始化之前做完（各自 docstring 里写了
    # 为什么），而 app 一旦建出来就晚了。无头那一支同样要——尤其隔离状态那一条：
    # ``_check_remind`` 起的就是一个带隔离数据目录的无头进程。
    configure_isolated_state()
    _configure_webengine_environment()

    if REMIND_FLAG in args:
        # **在这一句之前不能有任何 Qt 对象**：登录时绝大多数情况下没有到点的提醒，
        # 而那一次不该去建一个应用（见 ``headless.run`` 的 ``notify`` 参数）。
        return _remind_once()

    # 这两个 import 必须在上面那两句之后
    from PySide6.QtWidgets import QApplication

    from .. import __version__

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("axo")
    app.setApplicationDisplayName("Axo")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Axo")

    # **Fusion 是必须的，不是审美选择。** Windows 原生样式（``windowsvista``）
    # 对相当一部分 QSS 规则不认——按钮、输入框的背景与边框会被它忽略，症状是
    # "主色改了但控件还是系统灰"，看起来像样式表写错了。Fusion 是 Qt 自己的跨平台
    # 样式，与 QSS 配合完整。
    #
    # 代价是**全局观感变化**：控件不再是 Windows 原生的样子。这是一处用户可见的
    # 既有外观变更，可逆（删掉这一行），README 的排障表里记了它。
    app.setStyle("Fusion")

    # **关掉"最后一个窗口关了就退出"。** 有了托盘之后，关窗只是把窗口藏起来。严格
    # 说 ``hide()`` 不会触发 ``lastWindowClosed``（那个要 ``close()``），但这条推理
    # 依赖一个 Qt 的细节；它一旦不成立，症状是"关窗之后进程自己没了、托盘图标也没
    # 了"，看起来像崩溃。写出来比依赖那个细节稳。
    #
    # 真正的退出路径本来就只有一个：``_shutdown``——托盘菜单的「退出」、Ctrl+Q、
    # ``aboutToQuit`` 三条路都汇进去，而它是幂等的。
    app.setQuitOnLastWindowClosed(False)

    # ── 记忆放在哪：**必须在建 MainWindow 之前** ────────────────────────────
    # 那个窗口一建好，四个 store 加一张图就把 ``paths`` 现算出来的路径焊死了
    # （见 main_window.py 里那几行）。而"位置"住在注册表里、读它要 Qt，所以这件事
    # 只能在这里做——它是本入口的第七件事，理由与另外六件同级。
    #
    # 这也是**待办搬迁真正动盘的地方**：写指针与复制都发生在打开任何东西之前，所以
    # 不存在"搬到一半、库还开着"那种状态。
    from .data_root import boot_data_root

    move_report = boot_data_root(make_settings())

    from .main_window import MainWindow  # 推迟到 QApplication 之后

    window = MainWindow()
    # 保持引用。父级为 None 的顶层窗口在 Python 侧如果没有别的引用，会被 GC 掉
    # 并连带销毁原生窗口——表现为"窗口一闪而过"。挂在 app 上是最省事的锚点。
    app.main_window = window  # type: ignore[attr-defined]
    window.show()
    # **在 show() 之后说。** 搬迁的结果要有个父窗口才弹得出来，而在建窗口之前既没有
    # 地方说、抛异常又等于"程序打不开"。这一步可能弹一个对话框问"旧的那份删不删"，
    # 所以不能塞进 MainWindow 的构造函数里——那会让它出现在主窗口之前。
    if move_report is not None:
        window.report_data_move(move_report)
    # 首启向导。**必须在 show() 之后**：从 MainWindow 的构造函数里弹会让它出现在主窗口
    # 之前。副作用是几个冒烟脚本不会看到它——正好，它们的断言都是关于主窗口的。
    window.maybe_start_onboarding()
    try:
        return app.exec()
    finally:
        # **兜底的收尾。** 正常路径上这件事已经由 `aboutToQuit` 与 `closeEvent`
        # 做过了（`_shutdown` 幂等，重复调用无害），这里管的是另外两条：
        # `exec()` 之前就抛异常（那时窗口已经建好，四个后台线程已经在跑），
        # 以及事件循环根本没进去的情形。少了这一层，这两种情况下仍在运行的线程
        # 会在解释器收尾时让 Qt 调 qFatal——退出码变成 `0xC0000409`，把一个本该
        # 看得清的 traceback 换成一个没有线索的崩溃码。
        window._shutdown()


if __name__ == "__main__":  # pragma: no cover - 手动入口
    raise SystemExit(main())
