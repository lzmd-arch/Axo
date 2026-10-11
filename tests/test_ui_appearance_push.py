"""启动时的外观推送：三件套有没有真的送出去，以及"建完控件要再推一次"这条顺序。

**为什么专门盯它。** :meth:`MainWindow._apply_theme` 排在 ``_build_ui()`` 之前
（应用级样式表必须早于控件，理由在 ``__init__`` 里），所以**启动路径上那一次推送
必然找不到 ``chat_view``**——真正把外观送到网页的是 ``_build_ui()`` 末尾补的那
一次。少了它，网页侧永远停在 CSS 默认值上，而它**只有一种症状会被察觉**：默认预设
与默认主色恰好相同，主题那一半看不出来；可见度只有 18 与 60 的差别，多半也认不出。
刺眼的是"我设的壁纸重启后不见了"，而设置里那个文件名原样留着（那是刻意的），于是
排查会从"图是不是被删了"开始，一路都是错的。

**最后一条用例是 AST 断言，不是行为断言。** 理由是"``_build_ui`` 的末尾"这件事
在行为上测不到：gui 腿不建 ``MainWindow``（它一连 Ollama、起四个线程、加载
WebEngine），而没有 ``MainWindow`` 就没有"末尾"这个东西。真进程那一侧由
``tools/_smoke_memory.py`` 的 ``[0]`` 盯着；这里只把"那个调用点存在"钉在每次提交
上，好让它在 gui 腿里红，而不是等到某次手动验收。``tests/test_layering.py`` 用同
一套办法强制分层，先例在这里。
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

from biosnn_memterm.ui.main_window import MainWindow
from biosnn_memterm.ui.theme import Theme, resolve

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Recorder:
    """对话区的替身。只记调用序列——顺序本身是断言的一部分。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_theme(self, resolved: object) -> None:
        self.calls.append("setTheme")

    def set_wallpaper_opacity(self, percent: int) -> None:
        self.calls.append(f"setWallpaperOpacity({percent})")

    def set_wallpaper(self, path: str | None) -> None:
        self.calls.append(f"setWallpaper({path!r})")

    def set_status(self, text: str) -> None:
        self.calls.append("setStatus")


class Stub:
    """``_push_appearance`` 与它调到的 :meth:`_push_wallpaper`。"""

    _push_appearance = MainWindow._push_appearance
    _push_wallpaper = MainWindow._push_wallpaper

    def __init__(self, *, with_view: bool) -> None:
        #: 壁纸名留空：``wallpaper_file("")`` 会当场返回 ``None``，于是这条用例
        #: 不碰磁盘、也不碰数据目录（那个函数在早退之前不会去建目录）。
        self._theme = Theme()
        self._resolved = resolve(self._theme, "dark")
        if with_view:
            self.chat_view = Recorder()


def test_the_push_survives_a_missing_chat_view() -> None:
    """**启动路径上第一次调它时 ``chat_view`` 必然不存在。** 直接 ``self.chat_view``
    会当场 ``AttributeError``，而那是构造窗口时抛的——窗口根本不会出现。"""
    Stub(with_view=False)._push_appearance()


def test_the_push_sends_theme_opacity_and_wallpaper() -> None:
    stub = Stub(with_view=True)
    stub._push_appearance()
    assert stub.chat_view.calls == [
        "setTheme",
        f"setWallpaperOpacity({stub._theme.wallpaper_visible})",
        "setWallpaper(None)",
    ], "三件里少一件，网页侧就有一块停在 CSS 默认值上"


def _calls_in_build_ui() -> list[str]:
    """``MainWindow._build_ui`` 里所有 ``self.<名字>(...)`` 形式的调用名。"""
    source = Path("src") / "biosnn_memterm" / "ui" / "main_window.py"
    tree = ast.parse((Path(__file__).resolve().parents[1] / source).read_text(encoding="utf-8"))
    # 用 AST 而不是 import 之后看源码：这里要的就是"源码里那个调用点"，而
    # ``inspect.getsource`` 在同一形状出现两次时（比如另一个方法末尾也调它）
    # 分不出是哪一处。
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_ui":
            return [
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ]
    raise AssertionError("main_window.py 里找不到 _build_ui")


def test_the_appearance_is_pushed_at_the_end_of_build_ui() -> None:
    """**这条是整个文件的重点。** ``_apply_theme`` 排在 ``_build_ui()`` 之前，
    所以启动时的外观全靠 ``_build_ui`` 末尾补的那一次推 —— 删掉它，界面上唯一
    的症状是"壁纸重启后不见了"，而设置里那个文件名还在（刻意留着），
    于是排查方向会被它带偏。"""
    calls = _calls_in_build_ui()
    assert "_push_appearance" in calls, (
        f"_build_ui 里没有 _push_appearance()：启动时网页侧拿不到外观，"
        f"症状是壁纸与可见度重启后消失。当前它的调用点是 {calls}"
    )
