"""``_track`` 钉住的两条线程约定——它们是"界面偶尔无声消失"那条病史的疫苗。

背景（实测，不是推演）：worker 若用 ``moveToThread`` 搬进工作线程，那么**在界面
线程里丢掉它的最后一个 Python 引用**就等于在错误的线程里析构一个 C++ 对象，Qt 直接
``abort()``；装了 faulthandler 时退出码是 ``3``，没装时是 ``0xC0000409``（Windows
fast-fail），没有回溯、没有 stderr。``_smoke_recall.py`` 的 remember 相位在这个形状
下 2/6 崩，改成"亲和面留在界面线程 + ``started`` 直连"之后 0/8。

所以 :meth:`MainWindow._track` 之后必须同时成立两件事，缺一件就退化成上面那个病：

1. **``run()`` 在工作线程里跑**——否则每次网络请求都会把界面冻住（这条靠
   ``Qt.DirectConnection``；忘了写它，``AutoConnection`` 会把 ``run()`` 排队回界面
   线程，而症状看起来像 Ollama 慢）；
2. **worker 的 Qt 亲和面仍在界面线程**——否则那个 abort 会回来。

这个文件标 ``gui``：核心腿用 ``--ignore-glob='tests/test_ui_*'`` 排除（**``--ignore``
不认通配符**，写成它一条都排不掉且不报错），与气体仓的分腿纪律一致。
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Job:
    """假 worker：``_track`` 只要求它是个 ``QObject`` 且有 ``run()``。"""


def _make_job():  # noqa: ANN202 - 返回值是动态定义的 QObject 子类实例
    """在**界面线程**里造一个 worker，形状与四个真 worker 完全一致。"""
    from PySide6.QtCore import QObject, Signal

    class Job(QObject):
        done = Signal()

        def run(self) -> None:
            from PySide6.QtCore import QThread

            # 两条都**必须在这里读**：``_track`` 挂在 ``finished`` 上的 deleteLater
            # 一旦被派发，包装器就失效了，事后 ``job.thread()`` 只会抛 RuntimeError。
            self.where = QThread.currentThread()
            self.affinity = self.thread()
            time.sleep(0.05)
            self.done.emit()

    return Job()


def _drive(qapp: Any, job: Any, delivered: list[bool]) -> tuple[Any, list[int]]:
    """跑一轮完整的 ``_track`` 生命周期，返回（线程, 界面 tick 次数）。"""
    from PySide6.QtCore import QThread, QTimer

    from biosnn_memterm.ui.main_window import MainWindow

    # ``_track`` 只碰 ``_live_threads`` 与两个由 slot 拼出来的成员名，不碰任何控件，
    # 所以这里不必造一个真窗口（造了就要连 Ollama，测试会变得又慢又脆）。
    receiver = SimpleNamespace(_live_threads=[])
    thread = QThread()
    delivered.clear()
    job.done.connect(lambda: delivered.append(True))  # 工作线程发射 → 界面线程槽
    job.done.connect(thread.quit)  # 终端信号收掉事件循环，否则线程会一直空转
    MainWindow._track(receiver, thread, job, what="测试线程", slot="probe")

    ticks: list[int] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)

    thread.start()
    deadline = time.monotonic() + 5
    while thread.isRunning() and time.monotonic() < deadline:
        # pytest 里没有 ``app.exec()``，排队送达的东西（``thread.quit`` 也在内）要靠
        # 这一句派发。**不能改成 sleep**：那样测的就不是"界面照常转"了。
        qapp.processEvents()
        time.sleep(0.001)
    # 线程结束后再转几圈，把 ``finished`` 上的 deleteLater 也派发掉。
    for _ in range(50):
        qapp.processEvents()
        time.sleep(0.001)
    timer.stop()
    return thread, ticks


def test_run_executes_off_the_gui_thread(qapp: Any) -> None:
    """约定 1：``run()`` 在工作线程里跑。

    这条断了的表现是**界面在每次请求期间冻住**，而不是报错——最容易被当成 Ollama 慢
    而查错方向。
    """
    job = _make_job()
    _drive(qapp, job, [])
    assert job.where is not qapp.thread()


def test_the_worker_stays_affine_to_the_gui_thread(qapp: Any) -> None:
    """约定 2：worker 的所属线程自始至终是界面线程。

    这条断了就是那次 ``abort()``：在界面线程丢它的最后一个引用 = 跨线程析构。
    """
    job = _make_job()
    _drive(qapp, job, [])
    assert job.affinity is qapp.thread()


def test_the_gui_thread_keeps_ticking_while_run_runs(qapp: Any) -> None:
    """``run()`` 期间界面线程照常转（不是靠"跑得快"掩盖过去的）。"""
    job = _make_job()
    _, ticks = _drive(qapp, job, [])
    assert ticks, "run() 期间界面一次都没转——它多半是在界面线程里跑的"


def test_signals_cross_back_queued(qapp: Any) -> None:
    """工作线程发射的信号照常送达界面线程的槽。"""
    job = _make_job()
    delivered: list[bool] = []
    _drive(qapp, job, delivered)
    assert delivered == [True]


def test_the_worker_is_actually_released(qapp: Any) -> None:
    """``finished`` 上的两个 deleteLater 都真的生效了——**没有泄漏**。

    旧形状下 ``worker.deleteLater`` 是静默失效的：它把 DeferredDelete 排进**工作
    线程**的队列，而那个事件循环已经结束，事件永远不会被派发。这条断言钉的就是"删除
    发生在界面线程的队列里"。用 ``shiboken6.isValid`` 是因为"C++ 对象没了"这件事在
    Python 侧只有这一个可观测的入口——而且它同时说明**析构发生在界面线程**，这正是
    那个 ``abort()`` 的成因。
    """
    import shiboken6

    job = _make_job()
    thread, _ = _drive(qapp, job, [])
    assert not shiboken6.isValid(job)
    assert not shiboken6.isValid(thread)
