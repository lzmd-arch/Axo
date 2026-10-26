"""睡眠巩固，从"表响了"一路走到"库改了、然后说了句话"的那条链。

**为什么不建 ``MainWindow``**，同 ``test_ui_graph_chain.py``：离屏下 QtWebEngine 一渲染
就把整个进程硬杀掉。所以这里一律拿 ``MainWindow`` 的**未绑定方法**配上一个人造的
``self``，放进去的属性就是那几个方法真正碰到的全部——长出新的依赖会当场
``AttributeError``，而不是悄悄测了个别的东西。

这个文件的主题是**"表响了"与"跑一轮"是两件事**：

* 表只报时（``ConsolidationScheduler`` 自己的行为在
  ``test_ui_consolidation_scheduler.py`` 里量）；
* ：meth:`_on_consolidation_due` 才是决定跑不跑的地方——它要查单飞、查用户是不是正忙、
  查有没有图，而**三条分支里有两条什么都不会跑**。

同样，**自动与手动不是两条路径**：同一个 worker、同一道单飞守卫、同一段收尾，只差一个
``force`` 与"回来说不说气泡"。这里守的就是这几处传导有没有接对——少了这一层，"纯层
全绿"与"界面上真的发生了"之间是空的。

有一条**只有真的跑起来才发现**的失效特别值得在这里说：`test_a_second_round_is_refused`
那一条在写出来的时候**当场红过**——``ConsolidationScheduler.reschedule()`` 当时排的是
``MIN_DELAY_MS`` 而不是 ``_delay_ms()``，于是这个表退化成每分钟问一次"该不该跑"。
单飞守卫挡得住"同时跑两轮"，挡不住"每 60 秒起一轮"。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui import main_window as main_window_module
from axo.ui.consolidation_scheduler import MIN_DELAY_MS
from axo.ui.main_window import MainWindow
from axo.ui.settings_store import MemorySettings

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


# ── 假件 ───────────────────────────────────────────────────────────────────
#
# 真的 ``QThread`` 起完就在事件循环里空转、测试结束也不会自己退出；真的 worker 会去
# **真数据目录**读写记忆库（``memory_db_path()`` 指的是 ``%LOCALAPPDATA%``），而巩固那
# 一个**会写库**。两个都换掉之后，这一段测的正是它该测的：主窗口在这条链上接对了没有。


class _RecordingSignal:
    """``connect`` 只记下接了谁，不真发。

    **不用真的 ``Signal``**：``MainWindow`` 的那几个槽带 ``@Slot`` 装饰，而 PySide6 不给
    "绑在非 QObject 上的槽"接线（``SystemError``）。换成记名的假信号之后反而能断言
    **接上去的到底是哪个方法**——"纯层全绿 ≠ 接线在"那一条，在这里就是"槽写得对 ≠
    接上了"。
    """

    def __init__(self) -> None:
        self.slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        self.slots.append(slot)


class _FakeConsolidationWorker:
    """形状与 ``ConsolidationWorker`` 一致：三个信号、一个 ``force``。"""

    made: list[_FakeConsolidationWorker] = []

    def __init__(self, *, store: Any, graph: Any, force: bool = False) -> None:
        self.store = store
        self.graph = graph
        self.force = force
        self.ready = _RecordingSignal()
        self.skipped = _RecordingSignal()
        self.failed = _RecordingSignal()
        _FakeConsolidationWorker.made.append(self)

    def run(self) -> None:  # pragma: no cover — 不起线程，所以永远不会被调到
        raise AssertionError("测试里不该真的跑巩固")


class _FakeGraphWorker:
    """同上，只是 ``GraphWorker`` 只有两个信号。"""

    made: list[_FakeGraphWorker] = []

    def __init__(self, *, store: Any, force: bool = False) -> None:
        self.store = store
        self.force = force
        self.ready = _RecordingSignal()
        self.failed = _RecordingSignal()
        _FakeGraphWorker.made.append(self)


class _FakeThread:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True


class _FakeScheduler:
    """只记下被怎么使唤了。**它不是被验的对象**（那是另一个文件），这里是"主窗口有没有
    正确地使唤它"。"""

    def __init__(self) -> None:
        self.interval_s = 86_400
        self.interval_calls: list[int] = []
        self.started = 0
        self.stopped = 0
        self.rescheduled = 0
        self.delays: list[int | None] = []
        self.fulls = 0

    def set_interval_s(self, seconds: int) -> None:
        self.interval_calls.append(seconds)
        self.interval_s = seconds

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def reschedule(self, delay_ms: int | None = None) -> None:
        self.rescheduled += 1
        self.delays.append(delay_ms)

    def reschedule_full_interval(self) -> None:
        self.fulls += 1


class _FakeTray:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def show_message(self, title: str, body: str) -> None:
        self.messages.append((title, body))


class _FakeDialog:
    def __init__(self) -> None:
        self.busy_calls: list[bool] = []
        self.consolidate_notes: list[str] = []
        self.graph_busy_calls: list[bool] = []
        self.graph_notes: list[str] = []

    def set_consolidate_busy(self, busy: bool) -> None:
        self.busy_calls.append(busy)

    def set_consolidate_note(self, note: str) -> None:
        self.consolidate_notes.append(note)

    def set_graph_busy(self, busy: bool) -> None:
        self.graph_busy_calls.append(busy)

    def set_graph_note(self, note: str) -> None:
        self.graph_notes.append(note)


class _FakeChatView:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def append_message(self, role: str, text: str) -> None:
        self.messages.append((role, text))


class _FakeMemoryWindow:
    """记忆窗口的替身。**只有主窗口碰它的那一个方法。**"""

    def __init__(self) -> None:
        self.notified = 0

    def notify_consolidated(self) -> None:
        self.notified += 1


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """把线程与两个 worker 换成假件，返回可以查的那几个。"""
    _FakeConsolidationWorker.made = []
    _FakeGraphWorker.made = []
    threads: list[_FakeThread] = []

    def make_thread(*args: Any, **kwargs: Any) -> _FakeThread:
        thread = _FakeThread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(main_window_module, "ConsolidationWorker", _FakeConsolidationWorker)
    monkeypatch.setattr(main_window_module, "GraphWorker", _FakeGraphWorker)
    monkeypatch.setattr(main_window_module, "QThread", make_thread)
    return SimpleNamespace(
        threads=threads,
        workers=lambda: list(_FakeConsolidationWorker.made),
        graph_workers=lambda: list(_FakeGraphWorker.made),
    )


def _round(**overrides: Any) -> SimpleNamespace:
    """一轮的产出。**形状照 ``ConsolidationRun``**：主窗口只碰 ``.report``，
    而 ``summary_line`` 从它读六个字段。"""

    report = SimpleNamespace(
        run_id="0123456789abcdef",
        replayed=4,
        reinforced=["m1", "m2"],
        skip_reasons={"pinned": 1},
        recognition_rate=0.5,
        missing_from_graph=0,
    )
    report.__dict__.update(overrides)
    return SimpleNamespace(report=report)


@pytest.fixture
def window() -> SimpleNamespace:
    """这几个方法碰到的**全部**状态。**不是 MainWindow。**

    ``_start_consolidation`` 与 ``_start_graph_build`` 都要**绑**上去：调用方是
    ``self.`` 调的（不是模块级函数），人造 ``self`` 上不绑就会 ``AttributeError``
    ——那正是这个文件想要的失败方式。
    """
    fake = SimpleNamespace(
        _consolidating=False,
        _consolidate_pending=False,
        _consolidate_manual=False,
        _consolidation_enabled=True,
        _consolidation_interval_s=86_400,
        _consolidation_scheduler=_FakeScheduler(),
        _graph=None,
        _graph_freshness=None,
        _graph_note="",
        _graph_building=False,
        _streaming=False,
        _preparing=False,
        _memory="这个 store 是哪个不重要，worker 也换掉了",
        _memory_window=None,
        _settings_dialog=None,
        _tray=_FakeTray(),
        _task_memory=False,
        _graph_enabled=False,
        _writer=SimpleNamespace(policy=None),
        _settings_store=SimpleNamespace(saved=[]),
        _track=lambda *_args, **_kwargs: None,
    )
    fake._start_consolidation = MainWindow._start_consolidation.__get__(fake)
    fake._start_graph_build = MainWindow._start_graph_build.__get__(fake)
    # 这几个是 worker 的收场槽，以及相互调用的那几处。**绑真方法**，不塞替身——
    # 接错对象这一整条链就是空的，而那正是这个文件要验的东西。
    for name in (
        "_on_consolidation_ready",
        "_on_consolidation_skipped",
        "_on_consolidation_failed",
        "_on_consolidation_due",
        "_on_manual_consolidation",
        "_note_consolidate",
        "_on_graph_ready",
        "_on_graph_failed",
    ):
        setattr(fake, name, getattr(MainWindow, name).__get__(fake))
    fake.chat_view = _FakeChatView()
    return fake


def _saved(window: SimpleNamespace) -> list[MemorySettings]:
    store = window._settings_store
    store.saved = []
    store.save_memory = store.saved.append
    return store.saved


# ── 起一趟：接的是谁、力不力 ────────────────────────────────────────────────


def test_a_round_starts_a_thread_and_marks_itself_busy(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    window._graph = object()

    assert window._start_consolidation(manual=False) is True

    assert window._consolidating is True
    assert len(fakes.threads) == 1 and fakes.threads[0].started
    assert len(fakes.workers()) == 1
    assert fakes.workers()[0].force is False, "自动那趟要让 worker 自己判断有没有新东西"


def test_the_round_is_wired_to_all_three_endings(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """**接的是谁，这里要钉住。** 方法与槽都写对了却连到别处，是这条链上唯一一种
    "每块都绿、界面上什么都不发生"的错法（P3a 路由做完却零调用就是这一类）。

    **三个收场都要接，而且都要 ``quit`` 线程。** 这个 worker 比 ``GraphWorker`` 多一个
    "让掉"的结局（``skipped``）——漏掉它就是一个**必然发生**的空转线程：让掉的那一轮
    是常态（每天没有新记忆），所以那个泄漏每天都会多一个。
    """
    window._graph = object()

    window._start_consolidation(manual=True)
    worker = fakes.workers()[0]
    thread = fakes.threads[0]

    assert worker.ready.slots == [window._on_consolidation_ready, thread.quit]
    assert worker.skipped.slots == [window._on_consolidation_skipped, thread.quit]
    assert worker.failed.slots == [window._on_consolidation_failed, thread.quit]


def test_a_second_round_is_refused(window: SimpleNamespace, fakes: SimpleNamespace) -> None:
    """**同一时刻只跑一趟。**

    两轮会同时 ``bump_reinforce`` 同一批条目（计数多涨一次），而且事件文件按 ``run_id``
    命名——同一个 ``at`` 秒里起的第二趟会撞上第一趟的文件名然后抛 ``ValueError``。

    两种叠法都要挡：自动那趟还在跑时用户点了「立即巩固」，以及手动那趟还在跑时表又响了。
    """
    window._graph = object()
    window._start_consolidation(manual=False)

    assert window._start_consolidation(manual=True) is False
    assert len(fakes.threads) == 1, "挡在起线程之前，不是起了再丢"
    assert len(fakes.workers()) == 1


def test_only_the_manual_round_disables_the_settings_button(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """手动那趟先把按钮禁掉（否则连按会排出好几趟），自动那趟**不碰它**——用户没在看
    那个页面，而按钮本来就是放开的。"""
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    window._graph = object()

    window._start_consolidation(manual=True)
    assert dialog.busy_calls == [True]

    window._consolidating = False
    window._start_consolidation(manual=False)
    assert dialog.busy_calls == [True], "自动那趟不该多说一句"


# ── 表响了：跑不跑由这里定 ──────────────────────────────────────────────────


def test_a_due_while_one_is_running_only_rearms(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """已经在跑 → **重排一枪、不起第二趟**。这是单飞守卫在"自动那条路"上的入口。"""
    window._consolidating = True
    window._graph = object()

    window._on_consolidation_due()

    assert fakes.threads == []
    assert window._consolidation_scheduler.rescheduled == 1, "那一枪已经打完了，表要有人接"


def test_a_due_while_the_user_is_busy_is_postponed(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """正在生成或装配 → **推迟一分钟再看，不发车**。

    全仓没有真 idle 检测，这是最接近的代理：巩固要读全库、复制整库，挂在用户正等回复
    的那一刻上，表现是"打字卡一下"。推迟**不动** ``last.json``（它只在真跑完之后写），
    所以一分钟之后会重新问一遍"该不该跑"，答案仍然是真的。
    """
    window._graph = object()
    window._streaming = True

    window._on_consolidation_due()

    assert fakes.threads == [], "用户正等着回复，不在这时候起一个读全库的线程"
    assert window._consolidation_scheduler.delays == [MIN_DELAY_MS], "推迟要**明确**给一分钟"
    assert window._consolidate_pending is False, "这不是挂起，是过一会儿再说"


def test_a_due_without_a_graph_hangs_itself_on_the_build(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """没有图 → 把这一轮**挂起**，并请一趟建图。**不静默放弃。**

    图是懒建的（只在打开记忆窗口时建），所以"从不打开记忆窗口的人"是这条路径的常客。
    不挂起的话，默认开的功能对他们就是每天醒一次、每天什么都不做——而那在界面上完全
    看不出来。
    """
    window._on_consolidation_due()

    assert window._consolidate_pending is True
    assert window._consolidate_manual is False
    assert fakes.workers() == [], "没有图，巩固那一趟起不来"
    assert len(fakes.graph_workers()) == 1, "要把建图请起来"
    assert fakes.graph_workers()[0].force is False, "自动那趟不强制重建"


def test_the_pending_round_continues_when_the_graph_arrives(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """图回来了 → **在这里继续**，不在表那里。

    表那一枪当时已经打完了，它唯一能做的就是把这个意图挂起来；而"图来了"是唯一能让
    挂起的那一轮继续的事件，所以继续的落点只能在这里。
    """
    window._consolidate_pending = True
    window._consolidate_manual = True

    window._on_graph_ready(object(), None, "已建好：3 个节点、2 条边。")

    assert len(fakes.workers()) == 1
    assert fakes.workers()[0].force is True, "挂起时是手动要的，续上还是手动那趟"
    assert window._consolidating is True
    assert window._consolidate_pending is False


def test_a_failed_build_ends_the_pending_round_out_loud(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """建图失败 → 挂起的那一轮**跟着了结，而且说一句**。

    留着 ``_consolidate_pending`` 的后果是：手动那一趟会等到下一次"到点"才被人发现没跑
    （最长 24 小时），而那期间设置页那一行什么都不说。
    """
    window._consolidate_pending = True
    window._consolidate_manual = True
    dialog = _FakeDialog()
    window._settings_dialog = dialog

    window._on_graph_failed("建立关联图失败：ValueError：SDR 宽度不一致")

    assert window._consolidate_pending is False
    assert window._consolidate_manual is False
    assert fakes.workers() == []
    assert any("建图失败" in note for note in dialog.consolidate_notes)


# ── 手动按下去 ─────────────────────────────────────────────────────────────


def test_manual_ignores_the_nothing_new_gate(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """手动那趟传 ``force=True``：**用户按它就是不信任"没有新东西"那个判断。**

    漏了这一位的话，「立即巩固」对一个刚跑过的库永远什么都不做——而那正好是用户最可能
    去按它的时刻（他想看一眼刚才那一轮到底做了什么）。
    """
    window._graph = object()

    window._on_manual_consolidation()

    assert len(fakes.workers()) == 1
    assert fakes.workers()[0].force is True


def test_manual_without_a_graph_builds_first_instead_of_refusing(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """没有图**不拒绝**：挂起并请一趟建图。

    这条路径的常客是"从没打开过记忆窗口"的人，而那个按钮不该因为图是懒建的而变成死的。
    """
    window._on_manual_consolidation()

    assert window._consolidate_pending is True
    assert window._consolidate_manual is True
    assert len(fakes.graph_workers()) == 1
    assert window._tray.messages, "按下去了要有个回声，否则看起来像没反应"


@pytest.mark.parametrize(
    ("field", "expected"),
    [("_consolidating", "还在跑"), ("_streaming", "先停止"), ("_preparing", "先停止")],
)
def test_manual_refusals_always_say_why(
    window: SimpleNamespace, fakes: SimpleNamespace, field: str, expected: str
) -> None:
    """三条拒绝，每条都要**说一句为什么**。

    一个按下去没反应的按钮，用户的结论是"它坏了"，然后他会再按一次。而"正在生成"
    那条**不推迟**——推迟是自动那条路的语汇，手动按下的那一趟不该被悄悄排到一个用户
    没看着的时刻上，那正是他按下它的反面。
    """
    setattr(window, field, True)
    window._graph = object()
    dialog = _FakeDialog()
    window._settings_dialog = dialog

    window._on_manual_consolidation()

    assert fakes.threads == []
    assert any(expected in note for note in dialog.consolidate_notes), dialog.consolidate_notes
    assert window._tray.messages, "托盘那一项也要有个回声"


# ── 回来了 ─────────────────────────────────────────────────────────────────


def test_ready_rearms_and_says_it_outside_the_chat(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """跑完了：**清单飞、重排、把结果说给设置页，但对话区一个字都不写**。

    重排走的是 ``reschedule()`` 而不是"隔一整个周期"——这一轮刚把 ``last.json`` 写成
    "现在"，所以那个算法算出来的正是"从现在起一个周期"。

    **对话区那一行是用户要求去掉的**：巩固每天自己跑一次，往对话区写一句"回放了几条、
    认出几条"是噪音。这句话本身还在（``summary_line``），只是落点换成了设置页的状态行
    与手动那趟的托盘气泡。所以这里**两条都断言**——只断言"没往对话区写"的话，将来谁把
    整句话删掉，这一条照样是绿的。
    """
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    window._consolidating = True
    window._consolidate_manual = False

    window._on_consolidation_ready(_round())

    assert window._consolidating is False, "不清掉的话，一次失败就把巩固永久锁死"
    assert window._consolidation_scheduler.rescheduled == 1
    assert window._consolidation_scheduler.fulls == 0
    assert dialog.busy_calls == [False], "按钮必须放开"
    assert window.chat_view.messages == [], "对话区只留用户与它说过的话"
    assert len(dialog.consolidate_notes) == 1
    assert "回放" in dialog.consolidate_notes[0], "结果本身还得说，只是换了落点"


def test_an_automatic_round_does_not_pop_a_bubble(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """自动那趟**不弹气泡**：它每天一次，弹出来就是噪音——它的结果落在热度面板上
    （以及设置页开着时那一行状态）。"""
    window._consolidating = True
    window._consolidate_manual = False

    window._on_consolidation_ready(_round())

    assert window._tray.messages == []


def test_a_finished_round_pushes_the_new_numbers_to_the_memory_window(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """跑完了要**推一下开着的记忆窗口**。

    热度那一块画的是库里的 ``reinforce_count``，而这一轮刚刚改过它。不推的话，用户
    看到的是一份**旧的**热度摘要——而自动那趟在人前**只有这一个落点**（它不弹气泡、
    也不写对话区），所以这一下漏了，那一轮就等于什么都没发生。这条链最容易漏的正是
    这一下：worker 的收场槽接对了、设置页说了、按钮也放开了，只有那一块停在旧数字上。
    """
    memory_window = _FakeMemoryWindow()
    window._memory_window = memory_window
    window._consolidating = True

    window._on_consolidation_ready(_round())

    assert memory_window.notified == 1


def test_no_memory_window_open_means_nothing_to_push(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """窗口没开着（``None``）→ 这一下**跳过，不报错**。它会自己在下一次 ``reload``
    时读库，而那正是打开窗口时发生的事。"""
    window._consolidating = True

    window._on_consolidation_ready(_round())

    assert window._consolidating is False, "别的收尾照做，没有被这一下带崩"
    assert window._consolidation_scheduler.rescheduled == 1


def test_a_manual_round_pops_a_bubble(window: SimpleNamespace, fakes: SimpleNamespace) -> None:
    """手动那趟**弹**：不弹就等于"点了没反应"，而那个动作的全部意义就是这个回声。"""
    window._consolidating = True
    window._consolidate_manual = True

    window._on_consolidation_ready(_round())

    assert len(window._tray.messages) == 1
    title, body = window._tray.messages[0]
    assert title == "睡眠巩固" and "回放" in body
    assert window._consolidate_manual is False, "说过一次就清掉，不然下一趟自动的会替它弹"


def test_a_skipped_round_is_silent_but_rearms_a_full_interval(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """让掉了（空库，或"最新那条还是上次那条"）→ **除了重排，什么都不做**。

    沉默是刻意的：自动那趟每天至少醒一次，而"今天没有新记忆"是常态而不是事件。

    **必须按"隔一整个周期"重排**：让掉的一轮没有写 ``last.json``，所以 ``reschedule()``
    读到的还是上一轮那个旧 ``at``、算出来必然是负数、被夹成一分钟——于是每分钟醒一次、
    每分钟再让掉一次，直到有东西变。
    """
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    window._consolidating = True

    window._on_consolidation_skipped()

    assert window._consolidating is False
    assert window._consolidation_scheduler.fulls == 1
    assert window._consolidation_scheduler.rescheduled == 0
    assert window.chat_view.messages == []
    assert window._tray.messages == []
    assert dialog.busy_calls == [], "自动那趟本来就没禁用过按钮，这里也不该去放开它"


def test_a_failed_round_is_reported_in_the_chat(
    window: SimpleNamespace, fakes: SimpleNamespace
) -> None:
    """失败**要说**，与让掉正相反：让掉是正常的日常，失败是一件本该发生而没发生的事。

    走对话区而不是气泡（除非那趟是手动按的）——用户回看读得到，当下也不打断他。**不
    重试**：失败的原因（图过期、库被占着）多半不会在一分钟后消失。
    """
    dialog = _FakeDialog()
    window._settings_dialog = dialog
    window._consolidating = True
    window._consolidate_manual = False

    window._on_consolidation_failed("ValueError：SDR 宽度不一致")

    assert window._consolidating is False
    assert window._consolidation_scheduler.fulls == 1
    assert len(window.chat_view.messages) == 1
    assert "SDR 宽度不一致" in window.chat_view.messages[0][1], "原因要原样说出来"
    assert dialog.busy_calls == [False]
    assert window._tray.messages == [], "自动那趟失败也不弹"


# ── 设置页 → 调度器 ────────────────────────────────────────────────────────


def test_the_memory_settings_land_on_the_scheduler(window: SimpleNamespace) -> None:
    """设置页发来的那两个字段要**当场落到表上**。

    漏了的话，勾选框会照常打勾、照常存盘，而表照旧按 24 小时排——用户改的是一个
    "每多久跑一次"的设置，而它看起来没反应。

    **这两项与同一次 ``save_memory`` 里的另外两项形状不同**：那两个是"下一轮才生效"
    （一轮的粒度/通道不该在中途变），这两个是"现在就生效"。
    """
    saved = _saved(window)

    MainWindow._on_memory_changed(  # type: ignore[arg-type]
        window,
        MemorySettings(
            task_memory=False,
            graph_channel=False,
            consolidation_enabled=True,
            consolidation_interval_s=3600,
        ),
    )

    assert len(saved) == 1 and saved[0].consolidation_interval_s == 3600
    assert window._consolidation_interval_s == 3600
    assert window._consolidation_scheduler.interval_calls == [3600]
    assert window._consolidation_scheduler.started == 1


def test_turning_the_switch_off_stops_the_timer(window: SimpleNamespace) -> None:
    """关掉 → **停表**，而且顺手把挂起的那一轮清掉。

    不清的话，用户关掉开关之后过了几天又打开，那一轮"挂起的巩固"会突然跑起来——而它
    是几天前那个决定留下的。
    """
    _saved(window)
    window._consolidate_pending = True

    MainWindow._on_memory_changed(  # type: ignore[arg-type]
        window,
        MemorySettings(
            task_memory=False,
            graph_channel=False,
            consolidation_enabled=False,
            consolidation_interval_s=86_400,
        ),
    )

    assert window._consolidation_scheduler.stopped == 1
    assert window._consolidation_scheduler.started == 0
    assert window._consolidate_pending is False
    assert window._consolidation_enabled is False


def test_turning_the_switch_back_on_rearms(window: SimpleNamespace) -> None:
    """反向也要走通：关掉再打开，表要**重新排上**（``stop`` 之后它不会自己活过来）。"""
    _saved(window)
    window._consolidation_enabled = False

    MainWindow._on_memory_changed(  # type: ignore[arg-type]
        window,
        MemorySettings(
            task_memory=False,
            graph_channel=False,
            consolidation_enabled=True,
            consolidation_interval_s=86_400,
        ),
    )

    assert window._consolidation_scheduler.started == 1
    assert window._consolidation_scheduler.stopped == 0
