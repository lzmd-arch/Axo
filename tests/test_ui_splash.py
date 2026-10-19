"""开屏动画：生命周期、动效开关、隔离模式跳过。

这个文件标 ``gui``：它要 ``QApplication`` 与 ``offscreen`` 平台。开屏本身只用
``QPainter`` / ``QPropertyAnimation``，不碰 ``QtWebEngine``——所以离屏跑它是
安全的（与 ``tests/test_ui_*`` 里那些窗口测试不同，它们不加载网页）。

断言的是**行为**而不是像素：动效关掉时不能还有定时器在跑、隔离模式不能开屏、
``finish()`` 必须真的把窗口关掉。像素层面的构图由 ``tools/_smoke_window.py``
的目视检查兜底。
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def make_splash(*, motion: bool):  # noqa: ANN202 — 返回值是 Qt 类
    from axo.ui.splash import SplashScreen
    from axo.ui.theme import MODE_DARK, Theme, resolve

    theme = Theme(mode=MODE_DARK, motion=motion)
    return SplashScreen(theme=theme, resolved=resolve(theme, MODE_DARK))


def test_motion_off_has_no_running_animations(qapp: Any) -> None:
    """关掉动效之后，开屏只是一张静态画。"""
    from PySide6.QtCore import QAbstractAnimation

    splash = make_splash(motion=False)
    splash.show()
    try:
        assert splash._phase_anim.state() == QAbstractAnimation.State.Stopped
        assert splash._intro_anim.state() == QAbstractAnimation.State.Stopped
        assert splash.intro == 1.0
    finally:
        splash.close()


def test_motion_on_starts_the_breathing_loop(qapp: Any) -> None:
    from PySide6.QtCore import QAbstractAnimation

    splash = make_splash(motion=True)
    splash.show()
    try:
        assert splash._phase_anim.state() == QAbstractAnimation.State.Running
        assert splash._intro_anim.state() == QAbstractAnimation.State.Running
    finally:
        splash.close()


def test_finish_closes_the_window_when_motion_is_off(qapp: Any) -> None:
    from PySide6.QtTest import QTest

    splash = make_splash(motion=False)
    splash.show()
    QTest.qWait(20)
    splash.finish()
    QTest.qWait(30)
    assert not splash.isVisible()


def test_stage_line_is_updatable(qapp: Any) -> None:
    splash = make_splash(motion=False)
    splash.set_stage("正在打开记忆库…")
    assert splash._stage == "正在打开记忆库…"
    splash.close()


def test_isolated_state_and_manual_switch_skip_the_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """打包自检 / 冒烟脚本等的是主窗口，多一个顶层窗口会让探测多一个候选。"""
    from axo.ui.splash import splash_enabled

    monkeypatch.delenv("AXO_ISOLATED_STATE", raising=False)
    monkeypatch.delenv("AXO_NO_SPLASH", raising=False)
    assert splash_enabled()

    monkeypatch.setenv("AXO_NO_SPLASH", "1")
    assert not splash_enabled()

    monkeypatch.delenv("AXO_NO_SPLASH")
    monkeypatch.setenv("AXO_ISOLATED_STATE", "D:/tmp/isolated")
    assert not splash_enabled()
