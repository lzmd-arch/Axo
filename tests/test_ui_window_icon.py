"""窗口图标：**装没装上，以及是谁装的**。

这一份的来源是一次真事：``ui/tray.py`` 的 :func:`~biosnn_memterm.ui.tray.build_icon`
把图标画好了、托盘也一直在用，但**没人把它装到窗口上**——于是标题栏左上角空着一格。
打包后的产物更容易露出来：``--icon`` 嵌进 exe 的那一份只影响资源管理器里那个文件的
外观，窗口自己不设就是空的。

两半分开量，因为坏的方式不同：

* **装没装上**是行为，能直接问 Qt（``windowIcon()`` 是不是 null，以及画出来的颜色对
  不对）。null 图标在 Windows 上不是一个空格子那么无辜——它是一个空白方块，看起来
  像程序坏了。
* **谁装的**在 gui 单元测试里量不到：那要建 ``MainWindow``（连 Ollama、起四个后台
  线程、加载 WebEngine），重得不像单元测试。所以用 AST 把"``_apply_theme`` 里调了
  它"钉住——与 ``tests/test_ui_appearance_push.py`` 末尾那条同一个理由、同一套办法。
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QWidget

from biosnn_memterm.ui.main_window import _install_window_icon
from biosnn_memterm.ui.tray import ICON_SIZE

pytestmark = pytest.mark.gui

_MAIN_WINDOW = (
    Path(__file__).resolve().parent.parent / "src" / "biosnn_memterm" / "ui" / "main_window.py"
)

#: 一个不是任何预设的主色。用它是为了能断言"**这一支**颜色真的画上去了"，而不是
#: 只要不是 null 就算过。
_ACCENT = "#ff8800"


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # 应用级图标是**全局**的，留着会污染同一次 pytest 里别的 gui 用例。
    QGuiApplication.setWindowIcon(QIcon())


# ── 行为：两级都装上了，而且颜色是那一支 ────────────────────────────────────


def test_the_icon_reaches_the_window_and_the_application(qapp: Any) -> None:
    """窗口级与应用级**都要有**，两级管的是不同的窗口。

    窗口级管主窗口的标题栏；应用级管设置窗口与记忆窗口——它们是独立顶层窗口，不挂
    在主窗口上，只设窗口级的话那两个的标题栏照样空着。"""
    window = QWidget()
    _install_window_icon(window, _ACCENT)

    assert window.windowIcon().isNull() is False
    assert QGuiApplication.windowIcon().isNull() is False


def test_the_window_icon_is_drawn_in_the_accent(qapp: Any) -> None:
    """装上去的那一份得**是画出来的那个记号**，不是随便一个非空图标。

    取两个点验色：中心那个白点、以及正上方一米处的方块本体（避开圆角——四个角在
    圆角之外是透明的）。索引用位图自己的宽度算，因为高 DPI 上 ``pixmap()`` 会按
    ``devicePixelRatio`` 交回一张更大的图。"""
    window = QWidget()
    _install_window_icon(window, _ACCENT)

    image = window.windowIcon().pixmap(ICON_SIZE, ICON_SIZE).toImage()
    middle = image.width() // 2
    assert image.pixelColor(middle, middle).name() == "#ffffff", "中心那个点没画出来"
    assert image.pixelColor(middle, 1).name() == _ACCENT, "方块本体不是这一支主色"


def test_a_broken_accent_still_gives_a_window_icon(qapp: Any) -> None:
    """主题坏了（手改的注册表、降级回旧版本）也得有个图标——**绝不抛**。

    这条路的抛点在 ``MainWindow.__init__`` 里，那是启动路径：一个图标颜色不该让程序
    打不开。兜底颜色由 :func:`~biosnn_memterm.ui.tray.build_icon` 决定。"""
    window = QWidget()
    _install_window_icon(window, "不是颜色")

    assert window.windowIcon().isNull() is False


# ── 接线：换外观那条路真的调它 ──────────────────────────────────────────────


def _calls_in(method: str) -> set[str]:
    """``main_window.py`` 里某个方法调到的名字。``self.x()`` 与裸 ``x()`` 都算。"""
    tree = ast.parse(_MAIN_WINDOW.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            found: set[str] = set()
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name):
                    found.add(call.func.id)
                elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    found.add(call.func.attr)
            return found
    raise AssertionError(f"{_MAIN_WINDOW.name} 里找不到 {method}")


def test_the_appearance_entry_installs_the_icon() -> None:
    """**这条是这个文件的重点。** ``_apply_theme`` 是换外观的唯一入口——启动时的首次
    装载、设置窗口的实时预览、系统明暗翻转三条路都走它——所以图标装在里面三条路才
    都带得上。删掉那一句的症状只有一处：标题栏左上角空着，而那是没有人会写测试的
    地方。"""
    calls = _calls_in("_apply_theme")
    assert "_install_window_icon" in calls, (
        f"_apply_theme 里没有 _install_window_icon()：标题栏与任务栏那个图标没人装，"
        f"症状是左上角空着一格。它当前调的是 {sorted(calls)}"
    )
