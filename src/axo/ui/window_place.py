"""把窗口放到"用户正在看的那块屏"上，并按可用区定初始尺寸。

单独一个模块，因为它**必须只写一遍**。主窗口与记忆窗口各自写一次的话，两处会
各自踩进同一个坑，而踩进去的样子都不像几何问题：

* **尺寸不写死。** 写死 1180×760 在 4K 屏上是一个缩在角落的小窗口，在 1366×768
  的笔记本上则**超出屏幕**——超出的是标题栏与右下角，用户拖不动也够不着。
* **按光标所在那块屏算，不按主屏。** 副屏上启动时按主屏算尺寸、再 move 到主屏
  居中，窗口就出现在用户没在看的那块屏上。而记忆窗口是**从主窗口点开的**，
  它更应该出现在主窗口所在的那块屏上，光标的屏是最接近的近似。
* **``availableGeometry()`` 返回的就是逻辑像素**，正是 ``resize()`` 要的单位。
  **绝不再乘 ``devicePixelRatio()``**——乘了之后窗口会按缩放比例超出屏幕，而
  "最大化按钮被顶出屏幕"这种事很难联想到是缩放。

尺寸算术本身在 :mod:`axo.ui.layout` 里（那个文件不 import Qt，可以
拿一堆假分辨率逐条断言）。这里只负责"问屏幕"与"摆位置"。
"""

from __future__ import annotations

from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QWidget

from .layout import initial_window_size

__all__ = ["place_window"]


def place_window(window: QWidget) -> None:
    """按光标所在屏的可用区定尺寸并居中。

    拿不到屏幕时（无头环境、某些远程桌面）用最小尺寸且不移动——这是唯一一个
    不需要屏幕就能给出的合理答案，而"直接 return 不 resize"会让窗口停在 Qt 的
    默认尺寸上，那是另一回事。
    """
    screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    if screen is None:
        window.resize(*initial_window_size(0, 0))
        return
    area = screen.availableGeometry()
    window.resize(*initial_window_size(area.width(), area.height()))
    # 先 resize 再取 frameGeometry：居中要算上标题栏与边框，而它们在 resize 之前
    # 还没有实际尺寸。`frameGeometry()` 在窗口尚未显示时给的是 Qt 按当前尺寸
    # 估出来的框架，正是这里要的。
    frame = window.frameGeometry()
    frame.moveCenter(area.center())
    window.move(frame.topLeft())
