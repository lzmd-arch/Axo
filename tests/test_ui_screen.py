"""截图那一层：裁剪、缩放、编码、以及"全黑也是一种收场"。

这一层只认识 Qt（`grabWindow`、`QPixmap`），返回的是纯层的 :class:`Capture`，所以它能在
gui 腿里被逐条断言。**唯一断言不了的恰恰是"真的截到了什么"**：离屏平台的假屏幕恒为一张
800×800 的纯黑图（不是"截不到"，`isNull()` 为假——见 ``docs/agent.md`` 那条实测），
而真实截图只能在真屏幕上手验。所以这里量的是**在那之前的每一步**：涂自己、缩放、判全黑。

> **单独跑这个文件可能把进程带走**（退出码 ``0xC0000409``、没有回溯，而 pytest 可能已经
> 报完了结果）。这是离屏 Qt 的一处已知怪癖，触发与否取决于堆布局：**整条 gui 腿
> （``pytest -m gui``）是绿的**，而 ``pytest tests/test_ui_screen.py`` 单独跑会崩。2026-10-12
> 实测过一轮二分：把测试体换成等价但少一个常量引用的写法就不崩——也就是说它与被测代码
> 无关，纯粹是"这个进程里 Qt 收尾时的运气"。写在这里是为了让下一个单独跑它的人不必再
> 花半小时二分一遍（同 ``tests/test_ui_onboarding.py`` 那段"这个坑踩过一次"）。
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QGuiApplication, QPixmap

from axo.ui.screen import MAX_WIDTH, grab, looks_blank, mask_rect, scale_down

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _painted(width: int, height: int, colour: str = "#3366cc") -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(colour))
    return pixmap


# ── 缩放 ────────────────────────────────────────────────────────────────────


def test_a_big_screenshot_is_scaled_to_the_documented_width() -> None:
    """长边超过 :data:`MAX_WIDTH` 就缩到它。**这是额度那一侧的承诺**：
    ``budget.IMAGE_TOKENS`` 那个先验常数就是按这个尺寸估的，两处对不上，算的就不是
    实际发出去的那张图。"""
    scaled = scale_down(_painted(3000, 1600))

    assert max(scaled.width(), scaled.height()) == MAX_WIDTH
    # 等比：截图上都是文字，拉扁了就没法看。
    assert scaled.width() / scaled.height() == pytest.approx(3000 / 1600, rel=0.01)


def test_a_small_screenshot_is_left_alone() -> None:
    """已经够小就**原样返回**（也绝不放大——放大只会让那张图更大、更糊）。"""
    small = _painted(400, 300)

    assert scale_down(small).size() == small.size()


# ── 涂掉自己那一块 ──────────────────────────────────────────────────────────


def test_masking_paints_the_rect_and_leaves_the_rest() -> None:
    """把给定的矩形涂成一块纯色，**其余像素不动**。

    "看不到自己"是那条决定（用户点「允许」那一下就把这个窗口顶到了最前），涂还是裁只是
    实现；选涂的理由见 ``ui/screen.py`` 的模块文档（裁会在图中间留一道缝）。
    """
    painted = _painted(200, 200)

    masked = mask_rect(painted, QRect(50, 50, 60, 40))

    image = masked.toImage()
    assert image.pixelColor(60, 60).name() != "#3366cc", "该涂的那一块没涂"
    assert image.pixelColor(10, 10).name() == "#3366cc", "不该动的像素被动到了"
    assert masked.size() == painted.size(), "涂不是裁——尺寸不该变"


def test_a_rect_outside_the_screen_is_ignored_not_painted_elsewhere() -> None:
    """给进来的矩形可能部分（甚至全部）落在屏幕外——窗口一半在另一块屏上是常态。

    **求交是必需的**：``fillRect`` 画到画布外面是静默的，于是"涂了一半"看起来与
    "涂干净了"一模一样。
    """
    painted = _painted(100, 100)

    partially = mask_rect(painted, QRect(80, 80, 100, 100))
    assert partially.toImage().pixelColor(85, 85).name() != "#3366cc", "该涂的部分没涂"
    assert partially.toImage().pixelColor(5, 5).name() == "#3366cc"

    outside = mask_rect(painted, QRect(500, 500, 50, 50))
    assert outside.toImage() == painted.toImage(), "整个矩形都在屏幕外，却动了图"


# ── 全黑 ────────────────────────────────────────────────────────────────────


def test_an_all_black_image_is_reported_as_blank() -> None:
    """全黑 → 判"看起来是黑的"。

    锁屏、UAC 安全桌面、独占全屏都给全黑**且 ``isNull()`` 为假**的图（离屏的假屏幕
    也一样）。不判的话，模型会对着黑图编一段描述，而用户看到的是"它说屏幕上有个登录框"。
    """
    assert looks_blank(_painted(300, 200, "#000000")) is True


def test_an_ordinary_screen_is_not_blank() -> None:
    """普通画面不是黑的。**反方向也要钉**：一个恒真的判据会让这个功能永远什么都不看。"""
    assert looks_blank(_painted(300, 200, "#f5f5f5")) is False
    assert looks_blank(_painted(300, 200, "#3366cc")) is False


def test_a_mostly_black_screen_with_something_on_it_still_counts() -> None:
    """黑底上有一点东西（锁屏上的时钟、鼠标）仍然算"黑的"——

    阈值取 0.98 而不是 1.0 就是为了这个。判据要的是"大概是黑的"，而锁屏上几乎总有
    一条提示或一个时间。
    """
    mostly = _painted(400, 400, "#000000")
    painter = _paint(mostly)
    try:
        painter.fillRect(QRect(10, 10, 20, 20), QColor("#ffffff"))
    finally:
        painter.end()

    assert looks_blank(mostly) is True


def _paint(pixmap: QPixmap):  # 小工具：往图上画一笔
    from PySide6.QtGui import QPainter

    return QPainter(pixmap)


# ── 整条路 ──────────────────────────────────────────────────────────────────


def test_the_offscreen_screen_is_reported_as_black_not_as_a_success(qapp: Any) -> None:
    """**离屏下整条路走完得到的是"屏幕看起来是黑的"，不是一个成功。**（这是真的）

    这一条同时钉住两件相反的事：

    * 平台给出的假屏幕**不是**空的（``isNull()`` 为假、800×800），所以"截不到"那一支
      根本走不到——这正是 ``docs/agent.md`` 里那条实测警告说的形状；
    * 而**全黑那条判据接住了它**，于是这条路径在离屏下不会静默地"成功"。

    也就是说：这条路上的真实行为**只能在真屏幕上手验**，测试能证的是"我们不会把一张
    黑图当成看到了屏幕"。
    """
    screen = QGuiApplication.primaryScreen()
    assert screen is not None

    shot = grab(screen)

    assert shot.ok is False
    assert "黑的" in shot.reason
