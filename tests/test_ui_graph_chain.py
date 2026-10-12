"""联想开关与图，从设置页一路走到召回配置的那条链。

**为什么不建 ``MainWindow``**，同 ``test_ui_open_settings.py``：离屏下 QtWebEngine
一渲染就把整个进程硬杀掉。所以这里一律拿 ``MainWindow`` 的**未绑定方法**配上一个人造
的 ``self``。放进去的属性就是那几个方法真正碰到的全部，长出新的依赖会当场
``AttributeError``，而不是悄悄测了个别的东西。

这个文件的主题是**"开关"与"图"是两件事**——它们住在同一次 ``save_memory`` 里，生效
机理却完全不同：

* 开关只改召回配置，而那个配置**每轮现造**（``_recall_config``），所以翻一下下一轮就
  生效，图一个字节都不动；
* 图只 load/build 一次（``_on_open_memory`` 那一趟），开关决定"用不用它"，不会重建它。

两条链在别处各自都测过（``test_ui_memory_worker.py`` 守 worker 自己，
``test_ui_settings_dialog.py`` 守勾选框发出的对象），这里守的是**中间的传导**：东西
有没有从这一头走到那一头。少了这一层，"纯层全绿"与"界面上真的发生了"之间就是空的。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui import main_window as main_window_module
from axo.ui.main_window import _GRAPH_CHANNEL_WEIGHT, MainWindow
from axo.ui.settings_store import MemorySettings

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


# ── 假件：一个不跑线程的建图 ────────────────────────────────────────────────
#
# ``_start_graph_build`` 会造真的 ``QThread`` 与 ``GraphWorker``。这里两个都换掉：
# 真 worker 会去 **真数据目录** 读写 ``graph.npz``（``graph_path()`` 指的是
# ``%LOCALAPPDATA%``），而真线程起完就在事件循环里空转，测试结束也不会自己退出。
# 换掉之后这一段测的正是它该测的：**主窗口在这条链上接对了没有**。


class _RecordingSignal:
    """``connect`` 只记下接了谁，不真发。

    **不用真的 ``Signal``**：``MainWindow._on_graph_ready`` 带 ``@Slot`` 装饰，而
    PySide6 不给"绑在非 QObject 上的槽"接线（``SystemError``）。换成记名的假信号之后
    反而能断言**接上去的到底是哪个方法**——"纯层全绿 ≠ 接线在"那一条，在这里就是
    "槽写得对 ≠ 接上了"。
    """

    def __init__(self) -> None:
        self.slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        self.slots.append(slot)


class _FakeWorker:
    """形状与 ``GraphWorker`` 一致：两个信号、一个 ``force``、一个 ``run``。"""

    made: list[_FakeWorker] = []

    def __init__(self, *, store: Any, force: bool = False) -> None:
        self.store = store
        self.force = force
        self.ready = _RecordingSignal()
        self.failed = _RecordingSignal()
        _FakeWorker.made.append(self)

    def run(self) -> None:  # pragma: no cover — 不起线程，所以永远不会被调到
        raise AssertionError("测试里不该真的跑建图")


class _FakeThread:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """把线程与 worker 换成假件，返回可以查的那几个。"""
    _FakeWorker.made = []
    threads: list[_FakeThread] = []

    def make_thread(*args: Any, **kwargs: Any) -> _FakeThread:
        thread = _FakeThread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(main_window_module, "GraphWorker", _FakeWorker)
    monkeypatch.setattr(main_window_module, "QThread", make_thread)
    return SimpleNamespace(threads=threads, workers=lambda: list(_FakeWorker.made))


class _FakeDialog:
    def __init__(self) -> None:
        self.busy_calls: list[bool] = []
        self.notes: list[str] = []

    def set_graph_busy(self, busy: bool) -> None:
        self.busy_calls.append(busy)

    def set_graph_note(self, note: str) -> None:
        self.notes.append(note)


class _FakeMemoryWindow:
    def __init__(self) -> None:
        self.graphs: list[tuple[Any, Any]] = []

    def set_graph(self, graph: Any, *, stale: Any = False) -> None:
        self.graphs.append((graph, stale))


class _FakeScheduler:
    """``MainWindow._on_memory_changed`` 会把巩固的两个字段落到调度器上（P4c）。

    这里只需要它**存在**——那个传导的断言在 ``test_ui_consolidation_chain.py`` 里。
    """

    def set_interval_s(self, seconds: int) -> None:
        self.interval_s = seconds

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def window() -> SimpleNamespace:
    """这几个方法是它们碰到的全部状态。**不是 MainWindow。**

    ``_start_graph_build`` 要**绑**上去：``_on_rebuild_graph`` 是 ``self.`` 调它的
    （不是模块级函数），人造 ``self`` 上不绑就会 ``AttributeError``——那正是这个
    文件想要的失败方式。
    """
    fake = SimpleNamespace(
        _graph_enabled=False,
        _graph=None,
        _graph_freshness=None,
        _graph_note="",
        _graph_building=False,
        # P4c 之后 ``_on_graph_ready`` 末尾要多看一眼"有没有挂起的巩固"，
        # ``_on_memory_changed`` 要把两个巩固字段落到调度器上。图这一路测的仍然只有
        # 图，所以这三个字段给一个"什么都没挂"的最小值。
        _consolidating=False,
        _consolidate_pending=False,
        _consolidate_manual=False,
        _consolidation_enabled=True,
        _consolidation_interval_s=86_400,
        _consolidation_scheduler=_FakeScheduler(),
        _memory="这个 store 是哪个不重要，worker 也换掉了",
        _memory_window=None,
        _settings_dialog=None,
        _task_memory=False,
        _writer=SimpleNamespace(policy=None),
        _settings_store=SimpleNamespace(saved=[]),
        _track=lambda *_args, **_kwargs: None,
    )
    fake._start_graph_build = MainWindow._start_graph_build.__get__(fake)
    # 它还要连这两个槽（``worker.ready.connect(self._on_graph_ready)``）。**绑真方法**，
    # 不塞替身——接错对象这一整条链就是空的，而那正是这个文件要验的东西；下面的
    # ``_RecorderSignal`` 会把这个连接记下来供断言。
    fake._on_graph_ready = MainWindow._on_graph_ready.__get__(fake)
    fake._on_graph_failed = MainWindow._on_graph_failed.__get__(fake)
    return fake


def _saved(window: SimpleNamespace) -> list[MemorySettings]:
    store = window._settings_store
    store.saved = []
    store.save_memory = store.saved.append
    return store.saved


# ── 开关 → 召回配置 ────────────────────────────────────────────────────────


def test_the_recall_config_is_off_by_default(window: SimpleNamespace) -> None:
    """默认关，而且是**真的零**（短路，不是"权重很小"）。

    ``graph_weight=0.0`` 在 ``recall`` 里会让扩散根本不算——这是本版唯一一条"不许变坏"
    的硬约束，而它从这里开始。
    """
    assert MainWindow._recall_config(window).graph_weight == 0.0  # type: ignore[arg-type]


def test_turning_the_switch_on_non_zeroes_the_weight(window: SimpleNamespace) -> None:
    window._graph_enabled = True

    weight = MainWindow._recall_config(window).graph_weight  # type: ignore[arg-type]

    assert weight == _GRAPH_CHANNEL_WEIGHT
    assert 0.0 < weight < 1.0


def test_the_config_is_rebuilt_every_round(window: SimpleNamespace) -> None:
    """**每轮现造**，所以翻开关下一轮就生效——这条链上没有"陈旧配置"这回事。

    这正是 :meth:`_on_memory_changed` 那段注释里说的、与 ``writer.policy`` 不同的
    形状。若哪天有人把它改成缓存一个长命的 ``RecallConfig``，这一条会红：拿到手的
    那个对象在翻开关之后仍然说 0.0。
    """
    before = MainWindow._recall_config(window)  # type: ignore[arg-type]
    window._graph_enabled = True
    after = MainWindow._recall_config(window)  # type: ignore[arg-type]

    assert before.graph_weight == 0.0
    assert after.graph_weight > 0.0
    assert before is not after


def test_the_weight_stays_below_one_until_it_is_calibrated() -> None:
    """**未标定的初值，不许往上调。**

    一档扫描里四项指标没有一项变好（AUC 0.9093 → 0.8450），而 ``min_score`` 是按词法
    量纲标的——权重越接近 1，那条阈值就越不对应原来的严格程度。取多少要真实库上的数，
    在拿到之前这里钉一个上界。
    """
    assert 0.0 < _GRAPH_CHANNEL_WEIGHT < 1.0


# ── 设置页 → 主窗口 ────────────────────────────────────────────────────────


def test_the_memory_settings_land_the_switch(window: SimpleNamespace) -> None:
    """设置页发来的对象里的 ``graph_channel`` 要落到 ``_graph_enabled`` 上。

    漏了这一行的话，勾选框会照常打勾、照常存盘，而召回**下一轮什么都不做**——
    没有任何报错，开关像坏了。
    """
    saved = _saved(window)
    old_graph = object()
    window._graph = old_graph

    MainWindow._on_memory_changed(  # type: ignore[arg-type]
        window, MemorySettings(task_memory=False, graph_channel=True)
    )

    assert window._graph_enabled is True
    assert len(saved) == 1 and saved[0].graph_channel is True
    # **图不许跟着动。** 开关只决定"用不用它"；这里重建一次图是四分钟起的事，
    # 而用户只是勾了个框。
    assert window._graph is old_graph


def test_turning_the_switch_off_lands_too(window: SimpleNamespace) -> None:
    """反向也要走通：关得掉才算开关。"""
    window._graph_enabled = True
    _saved(window)

    MainWindow._on_memory_changed(  # type: ignore[arg-type]
        window, MemorySettings(task_memory=False, graph_channel=False)
    )

    assert window._graph_enabled is False


# ── 建图：起、单飞、回来 ────────────────────────────────────────────────────


def _start(window: SimpleNamespace, *, force: bool) -> bool:
    return window._start_graph_build(force=force)


def test_a_build_starts_a_thread_and_marks_itself_busy(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    assert _start(window, force=False) is True

    assert window._graph_building is True
    assert len(fakes.threads) == 1 and fakes.threads[0].started
    assert len(fakes.workers()) == 1
    assert fakes.workers()[0].force is False


def test_the_build_is_wired_to_the_two_return_handlers(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """**接的是谁，这里要钉住。** 方法与槽都写对了却连到别处，是这条链上唯一一种
    "每块都绿、界面上什么都不发生"的错法（P3a 路由做完却零调用就是这一类）。

    两条路径（``ready`` / ``failed``）都要接，而且都还要 ``quit`` 线程——不接那两下
    的话，``run()`` 跑完之后线程自己的事件循环不会结束，每建一次图就留下一个空转线程。
    """
    _start(window, force=False)
    worker = fakes.workers()[0]
    thread = fakes.threads[0]

    assert worker.ready.slots == [window._on_graph_ready, thread.quit]
    assert worker.failed.slots == [window._on_graph_failed, thread.quit]


def test_a_second_build_while_one_is_running_is_refused(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """**同一时刻只跑一趟。** 两趟会同时写同一个 ``graph.npz``，而"哪一趟先落盘"
    决定盘上最后是哪张图——那种错没有任何报错，只有一张悄悄变旧的图。

    两种叠法都要挡：自动那趟还在跑时用户按了重建，以及手动重建还在跑时用户开了记忆
    窗口。
    """
    _start(window, force=False)

    assert _start(window, force=True) is False
    assert len(fakes.threads) == 1, "挡在起线程之前，不是起了再丢"
    assert len(fakes.workers()) == 1


def test_a_manual_rebuild_disables_the_settings_button(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    dialog = _FakeDialog()
    window._settings_dialog = dialog

    _start(window, force=True)

    assert dialog.busy_calls == [True], "手动那趟要先禁用，否则连按会排出好几趟"


def test_an_automatic_build_leaves_the_button_alone(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """自动那趟（开窗触发）**不碰设置页的按钮**——用户没在看那个页面，而按钮本来
    就是放开的。"""
    dialog = _FakeDialog()
    window._settings_dialog = dialog

    _start(window, force=False)

    assert dialog.busy_calls == []


def test_rebuild_while_one_is_running_tells_the_user(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """挡下来之后**要说一句话**。不响的话，用户按了按钮、什么都没发生、按钮也没禁
    用——看起来就是坏了。"""
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    window._graph_building = True

    MainWindow._on_rebuild_graph(window)  # type: ignore[arg-type]

    assert len(dialog.notes) == 1
    assert "还在跑" in dialog.notes[0]
    assert fakes.threads == [], "被挡下的那一次不该起线程"
    assert dialog.busy_calls == [], "也不该把按钮禁用掉——那一趟不是它起的"


def test_ready_hands_the_graph_to_both_windows(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """``ready`` 回来是一条链的三下：清掉"在跑"、推给记忆窗口、放开设置页。

    第二个参数是**三态**，原样往下传——这里传 ``None``（"没比对过"，大库那条路），
    窗口收到的必须还是 ``None``，不能塌成 ``False``。
    """
    dialog = _FakeDialog()
    memory_window = _FakeMemoryWindow()
    window._settings_dialog = dialog
    window._memory_window = memory_window
    window._graph_building = True
    graph = object()

    MainWindow._on_graph_ready(window, graph, None, "已建好：3 个节点、2 条边。")  # type: ignore[arg-type]

    assert window._graph_building is False, "不清掉的话，从此再也起不了第二趟"
    assert window._graph is graph
    assert window._graph_freshness is None
    assert window._graph_note == "已建好：3 个节点、2 条边。"
    assert memory_window.graphs == [(graph, None)]
    assert dialog.busy_calls == [False]
    assert dialog.notes == ["已建好：3 个节点、2 条边。"]


def test_ready_without_the_windows_open_still_stores_the_graph(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """记忆窗口关着、设置页也没开——图照样要存下来。

    存下来才有下一次开窗时那句"先补一次"（``_on_open_memory`` 里的 ``set_graph``）：
    不存的话，用户每次开窗都从零开始建。
    """
    graph = object()

    MainWindow._on_graph_ready(window, graph, False, "图是最新的，不用重建。")  # type: ignore[arg-type]

    assert window._graph is graph
    assert window._graph_freshness is False


def test_a_failed_rebuild_keeps_the_old_graph(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """失败**不动 ``_graph``**：旧的那份照样能用，只是没有新的。

    清掉它的话，一次失败的重建会把用户正在看的邻居面板也弄空——"重建失败"加一个
    空面板，两件坏事一起。而失败原因必须原样说出来，不能换成一句"失败了"。
    """
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    old_graph = object()
    window._graph = old_graph
    window._graph_building = True

    MainWindow._on_graph_failed(window, "建立关联图失败：ValueError：SDR 宽度不一致")  # type: ignore[arg-type]

    assert window._graph is old_graph
    assert window._graph_building is False, "失败也要清掉，否则用户再也重建不了"
    assert dialog.busy_calls == [False], "按钮必须放开"
    assert "SDR 宽度不一致" in dialog.notes[0]
