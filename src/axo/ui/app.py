"""应用入口。

入口函数只做四件事：**建 app、定样式、建窗口、进事件循环**。日志、配置、数据库
连接、模型探活都不在这里——那些属于各自模块的职责，塞进入口只会让"启动时到底
发生了什么"变得不可读。

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

__all__ = ["main"]


#: 设了这个环境变量就强制软件光栅（值不重要，存在即可）
_SOFTWARE_RENDER_ENV = "AXO_SOFTWARE_RENDER"


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


def main(argv: list[str] | None = None) -> int:
    """启动桌面终端，返回进程退出码。"""
    _configure_webengine_environment()

    # 这两个 import 必须在 _configure_webengine_environment 之后
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

    from .main_window import MainWindow  # 推迟到 QApplication 之后

    window = MainWindow()
    # 保持引用。父级为 None 的顶层窗口在 Python 侧如果没有别的引用，会被 GC 掉
    # 并连带销毁原生窗口——表现为"窗口一闪而过"。挂在 app 上是最省事的锚点。
    app.main_window = window  # type: ignore[attr-defined]
    window.show()
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
