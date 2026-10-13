"""托盘：**一个图标、两项菜单、一条气泡**。

这个文件量的是四件事，每一件对着一种"在真机上才会出现"的失效：

1. **不可用时什么都不做。** 那台机器上拦下关窗等于让应用够不着（见 ``ui/tray.py``
   的模块文档），所以"不可用"这一支必须有测试走到——而它恰好是真人最不容易碰到的
   一支。
2. **图标不是空的。** Windows 上一个 null 图标不是"没图标"，是一个空白方块，看起来
   像程序坏了。
3. **菜单项真的接了信号。** 文字对不算数：``QAction`` 连错了槽的症状是"点了没反应"，
   而托盘菜单点不动时用户没有任何别的路可走。
4. **那句提示只说一次。** 每次关窗都说会变成噪音，一次都不说用户会以为程序退出了。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QPixmap

from axo.ui import tray as tray_module
from axo.ui.tray import ICON_SIZE, TrayController, build_icon

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ── 不可用的那台机器 ───────────────────────────────────────────────────────


def test_an_unavailable_tray_is_inert(qapp: Any) -> None:
    """``available=False`` → :attr:`active` 为假，而且**一个方法都不许炸**。

    调用方（``MainWindow._shutdown``）不先问 ``active`` 就调 ``hide()``——那是刻意的：
    把"托盘在不在"的判断散到每一个调用点上，漏掉一处就是一次 AttributeError，而它
    抛在退出路径上。
    """
    tray = TrayController(available=False)

    assert tray.active is False
    tray.show()
    tray.hide()
    tray.show_message("标题", "正文")


def test_an_unavailable_tray_says_it_did_not_speak(qapp: Any) -> None:
    """返回值是"真的说了没有"。托盘不在时说自己说过了，就是一句谎话——而谎话的
    代价是调用方以为用户已经被告知过。"""
    assert TrayController(available=False).show_message_once("标题", "正文") is False


def test_available_none_asks_qt(qapp: Any) -> None:
    """默认（``None``）是**问 Qt**，不是假设可用。构造不许抛——离屏环境里
    ``QSystemTrayIcon`` 能不能建是平台的事，而这个类要活得下来。"""
    tray = TrayController()
    assert isinstance(tray.active, bool)


# ── 图标 ───────────────────────────────────────────────────────────────────


def test_the_icon_is_drawn_not_null(qapp: Any) -> None:
    """**这条是 Windows 上唯一不能少的一条。** null 图标显示成一个空白方块，
    看起来像程序坏了——而那与"没画"是两回事，用户看不出区别。"""
    icon = build_icon()
    assert icon.isNull() is False
    assert icon.pixmap(ICON_SIZE, ICON_SIZE).isNull() is False


def test_a_missing_asset_still_gives_a_usable_icon(
    qapp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**资产读不到时退回画笔记号，绝不抛**（图标是固定的品牌立绘了，但坏图/缺文件
    仍可能——测试把它指到不存在的路径来走这条路）。

    抛在哪儿：``MainWindow.__init__`` 里，而那是启动路径——一个图标不该让程序打不开。
    """
    monkeypatch.setattr(tray_module, "_ASSET", Path("nope/missing.png"))

    assert build_icon().isNull() is False


def test_the_icon_comes_from_the_asset_not_from_a_procedural_mark(
    qapp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图标**读的就是资产那张图**——2026-10-13 用户给了阿螈立绘，图标是产品的脸、
    不跟主题主色走了。判据是决定性的：**换成一张纯色小图，图标中心就该是那个颜色**
    （整幅像素比对会踩图标容器补边的坑）。"""
    art = QPixmap(64, 64)
    art.fill("#ff00ff")
    art_path = tmp_path / "tmp-art.png"
    art.save(str(art_path))
    monkeypatch.setattr(tray_module, "_ASSET", art_path)

    icon = build_icon(32)

    middle = icon.pixmap(32, 32).toImage()
    assert middle.pixelColor(16, 16).name() == "#ff00ff", "图标不是从资产缩出来的"


# ── 菜单 ───────────────────────────────────────────────────────────────────


def test_an_available_tray_has_three_menu_items(qapp: Any) -> None:
    """**三项，而且次序是有意的**：「立即巩固」夹在「显示」与「退出」之间。

    它是一次真的计算（几秒到几分钟），而分隔线那一边是"结束进程"——夹在中间，误点的
    代价最小（最坏是白跑一轮）。
    """
    tray = TrayController(available=True)
    assert tray.active is True

    actions = [one for one in tray._menu.actions() if not one.isSeparator()]
    assert [one.text() for one in actions] == ["显示主窗口", "立即巩固", "退出"]


def test_the_menu_items_emit_their_signals(qapp: Any) -> None:
    """**文字对不算数，要真的发信号。** 连错槽的症状是"点了没反应"，而托盘菜单
    点不动时用户没有任何别的路可走——窗口已经藏起来了。"""
    tray = TrayController(available=True)
    seen: list[str] = []
    tray.show_requested.connect(lambda: seen.append("show"))
    tray.consolidate_requested.connect(lambda: seen.append("consolidate"))
    tray.quit_requested.connect(lambda: seen.append("quit"))

    actions = [one for one in tray._menu.actions() if not one.isSeparator()]
    for action in actions:
        action.trigger()

    assert seen == ["show", "consolidate", "quit"]


def test_double_clicking_asks_for_the_window(qapp: Any) -> None:
    """双击图标（以及 Linux 面板发的单击）要把窗口叫回来。"""
    from PySide6.QtWidgets import QSystemTrayIcon

    tray = TrayController(available=True)
    seen: list[str] = []
    tray.show_requested.connect(lambda: seen.append("show"))

    tray._on_activated(QSystemTrayIcon.ActivationReason.DoubleClick)
    tray._on_activated(QSystemTrayIcon.ActivationReason.Trigger)
    tray._on_activated(QSystemTrayIcon.ActivationReason.Context)

    assert seen == ["show", "show"], "右键菜单不该走这条路（它走 contextMenu）"


# ── 提示只给一次 ───────────────────────────────────────────────────────────


def test_the_hint_is_said_only_once(qapp: Any) -> None:
    """**第一次说，之后不再说。**

    一次都不说：通知区域在很多机器上默认折着，用户会以为程序退出了。
    每次都说：一天几十次关窗，它变成噪音——而噪音的结局是用户把通知全关掉。
    """
    tray = TrayController(available=True)

    assert tray.show_message_once("还在后台运行", "右键托盘图标选「退出」。") is True
    assert tray.show_message_once("还在后台运行", "右键托盘图标选「退出」。") is False


def test_the_once_flag_is_per_controller_not_per_message(qapp: Any) -> None:
    """它是"这个进程里说过一次了"，不是"这条正文说过一次了"——两个对象各自有
    自己的第一次（``MainWindow`` 只会建一个）。"""
    first, second = TrayController(available=True), TrayController(available=True)
    assert first.show_message_once("甲", "一") is True
    assert second.show_message_once("乙", "二") is True
