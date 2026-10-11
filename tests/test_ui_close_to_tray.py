"""关窗：**是收起来，还是真的退出**。

这条路径有三种情形，而它们的区别完全在一个布尔上——所以判据抽成了
:meth:`MainWindow._close_should_hide`，这里按仓里的惯例**用未绑定方法 + 假 self**
验它：离屏下构造真 ``MainWindow`` 会让进程无声消失（``0xC0000409``，没有回溯、
没有 stderr），见 ``ui/agent_worker.py`` 与 ``thread_tracking`` 里同一段教训。

三件事：

1. **判据的真值表**：有托盘才收，不可用的托盘不算有，真退出时一律不收。
2. **收起来那一支真的返回了**（没走收尾），而且**给了那句提示**。
3. **``_quit`` 先置位再关**——顺序反了，"退出"会变成又一次"收起来"。
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtGui import QCloseEvent

from biosnn_memterm.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _self(*, tray: Any, quitting: bool = False) -> SimpleNamespace:
    """``_close_should_hide`` 真正读到的全部状态。**只有三个字段。**"""
    return SimpleNamespace(_tray=tray, _quitting=quitting)


def _window(*, tray: Any, quitting: bool = False, **rest: Any) -> SimpleNamespace:
    """够 ``closeEvent`` 跑一个来回的假 self。

    把**真的** ``_close_should_hide`` 显式绑到这个假 self 上（未绑定方法 +
    ``SimpleNamespace``，仓里的惯例）。**不许**在调用处塞一个假的判据：那样改坏真的
    那个，分支测试也不会红——而"分支与判据是同一份代码"正是这组测试要钉的东西。
    """
    window = SimpleNamespace(_tray=tray, _quitting=quitting, **rest)
    window._close_should_hide = lambda: MainWindow._close_should_hide(window)
    return window


# ── 判据 ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tray_active,quitting,expected",
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
        (False, True, False),
    ],
)
def test_the_truth_table(tray_active: bool, quitting: bool, expected: bool) -> None:
    window = _self(tray=SimpleNamespace(active=tray_active), quitting=quitting)
    assert MainWindow._close_should_hide(window) is expected


def test_no_tray_at_all_means_close_still_quits() -> None:
    """**托盘不可用的机器上关窗必须真的退出。**

    反过来的话，窗口关了、图标也没有，应用**变得够不着**——只能去任务管理器里杀。
    比"关窗即退"坏得多。
    """
    assert MainWindow._close_should_hide(_self(tray=None)) is False


def test_a_present_but_dead_tray_does_not_count() -> None:
    """``TrayController`` 在不可用时是一个**空壳对象**（建得出来，``active`` 为假）。
    判"托盘在不在"如果只看"对象不是 None"，那台机器上就会静默地走进"收起来"那一支
    ——而图标根本不存在。"""
    assert MainWindow._close_should_hide(_self(tray=SimpleNamespace(active=False))) is False


# ── 收起来那一支 ───────────────────────────────────────────────────────────


def test_closing_with_a_tray_hides_and_returns(qapp: Any) -> None:
    """走隐藏支时**必须** ``ignore`` + ``hide`` + 说一句，然后返回。

    ``event.ignore()`` 那一下是承重的：Qt 据此不关窗口。只 ``hide()`` 而不
    ``ignore`` 的话，窗口会被真的关掉——而那时进程还在（``setQuitOnLastWindowClosed``
    是假），于是用户得到一个"退不掉也回不来"的状态。

    **这条**不塞一个假的 ``_close_should_hide``：假 self 只带状态，判据走**真的那一个**
    （它与上面真值表共用同一份代码）。塞假的话，把真判据改成恒假这些分支测试也不会红
    ——那就等于没验。
    """
    calls: list[Any] = []
    window = _window(
        tray=SimpleNamespace(
            active=True,
            show_message_once=lambda title, body: calls.append(("message", title, body)) or True,
        ),
        hide=lambda: calls.append("hide"),
        _shutdown=lambda: calls.append("shutdown"),
    )
    event = QCloseEvent()

    MainWindow.closeEvent(window, event)

    assert event.isAccepted() is False, "没 ignore，Qt 会把窗口真的关掉"
    assert calls[0] == "hide"
    assert calls[1][0] == "message"
    # 一定是 ``show_message_once``：``show_message`` 每次关窗都说，而那是噪音。
    assert "托盘" in calls[1][2]
    assert "shutdown" not in calls, "收起来不是退出，不该走收尾"


def test_hiding_does_not_touch_the_shutdown_path(qapp: Any) -> None:
    """再钉一遍那件事，用**不同的证明方式**：``_shutdown`` 一旦被走到，进程就结束
    了——提醒也就永远不会响。这条路径的全部意义就是"它没被走到"。"""
    window = _window(
        tray=SimpleNamespace(active=True, show_message_once=lambda *_args: True),
        hide=lambda: None,
        _shutdown=lambda: pytest.fail("隐藏这一支不该收尾"),
    )
    MainWindow.closeEvent(window, QCloseEvent())


def test_the_branches_and_the_judgement_are_the_same_code(qapp: Any) -> None:
    """把上一条反过来证一次：**判据说收，就必须收。**

    上面两条证的是"收了之后没乱来"，这条证的是"该收的时候真的收了"——只证前者的话，
    一个恒不进入隐藏支的 ``closeEvent`` 也能全绿，而那是"关窗即退"，托盘白加。
    """
    seen: list[str] = []
    window = _window(
        tray=SimpleNamespace(active=True, show_message_once=lambda *_args: seen.append("said")),
        hide=lambda: seen.append("hidden"),
        _shutdown=lambda: seen.append("shutdown"),
    )
    event = QCloseEvent()

    MainWindow.closeEvent(window, event)

    assert seen == ["hidden", "said"], "有托盘时关窗应当收起来并且说一句"
    assert event.isAccepted() is False


# ── 退出那一支 ─────────────────────────────────────────────────────────────


def test_quit_marks_before_closing() -> None:
    """**先置 ``_quitting`` 再 ``close()``。**

    顺序反过来的话，``closeEvent`` 会在置位之前读到假，于是"退出"变成又一次
    "收起来"——用户按了退出，窗口消失了，进程还在，托盘图标还在那儿。
    """
    calls: list[Any] = []
    window = SimpleNamespace(_quitting=False, close=lambda: calls.append(window._quitting))

    MainWindow._quit(window)

    assert window._quitting is True
    assert calls == [True], "close() 时 _quitting 还是假"


def test_quit_from_the_tray_really_quits(qapp: Any) -> None:
    """托盘菜单的「退出」连的是 ``_quit``，不是 ``close``——从菜单点下去的那条路
    也要真的结束进程。"""
    from biosnn_memterm.ui.tray import TrayController

    tray = TrayController(available=True)
    window = SimpleNamespace(_quitting=False, close=lambda: None)
    # 真接线是 `_tray.quit_requested.connect(self._quit)`；这里只验那根线的另一端
    # 指向的是 `_quit` 而不是 `close`。
    tray.quit_requested.connect(lambda: MainWindow._quit(window))

    actions = {one.text(): one for one in tray._menu.actions() if not one.isSeparator()}
    # **按文字取，不按下标。** 菜单会长（P4c 加了「立即巩固」进来），而"下标 1"这种
    # 写法在菜单一变时不会红——它会安安静静地去点另一项，然后这条测试开始报一个
    # 与它本意无关的错。
    actions["退出"].trigger()

    assert window._quitting is True
