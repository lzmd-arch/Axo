"""``MainWindow`` 上挂的那两族槽：任务语域的六个（卡片画不画、点击算不算、逐字稿写不写），
以及"到点由她开口"的九个（什么时候开口、开口失败怎么办、叫醒窗口的三档）。

**不构造真的 ``MainWindow``**（它一连 Ollama、起几个线程、加载 WebEngine）。这里
把 ``MainWindow`` 的六个方法**赋成类属性**给一个只带那几个属性的替身——
``test_ui_chat_persona.py`` 与 ``test_ui_context_strip.py`` 用的是同一套办法：替身
缺了哪个属性会当场 ``AttributeError``，而不是悄悄测了别的东西。

**为什么值得单独一个文件。** 这六个槽是"循环吐出的事件"与"用户看见的东西"之间
唯一的一段代码，而它每一处都有**说不出口的失效**：

* 收到 ``approval_needed`` 而不画卡片 → 用户永远看不到这次问询，worker 等满超时后
  按拒绝处理，症状是"助手说它没做"；
* 点击没被采纳却显示成"已允许" → 用户以为他同意了，而闸门记的是没人同意过；
* 收到 ``MessagesAppended`` 而不落盘 → 逐字稿里少了**真的执行过**的几步；
* 吞掉 ``unexecuted`` → 回到 ``ChatWorker`` 被修掉的那个静默丢弃。

每一条都在界面上完全看不出来，所以它们只能靠这里的断言守着。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import pytest
from fakes import ScriptedBackend, text_chunk
from PySide6.QtCore import QCoreApplication, QEvent, QObject

from axo.agent.gate import APPROVAL_TIMEOUT_S
from axo.agent.reminders import Alert, Occurrence, Reminder
from axo.llm.base import ChatMessage, ToolCall
from axo.session.store import Message
from axo.ui.main_window import SPEECH_TIMEOUT_MS, MainWindow, display_rows
from memcore.store import MemoryStore

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Recorder:
    """对话区的替身：只记调用序列（顺序本身是断言的一部分）。"""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def end_message(self) -> None:
        self.calls.append("endMessage")

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append(("appendMessage", role, text, meta))

    def append_tool_confirm(
        self, call_id: str, tool_name: str, summary: str, meta: dict | None = None
    ) -> None:
        self.calls.append(("appendToolConfirm", call_id, tool_name, summary, meta))

    def resolve_tool_confirm(self, call_id: str, *, granted: bool, accepted: bool) -> None:
        self.calls.append(("resolveToolConfirm", call_id, granted, accepted))

    def set_status(self, text: str) -> None:
        self.calls.append(("setStatus", text))


class FakeWorker:
    """``AgentWorker`` 那一半：只实现界面调的那一个方法。"""

    def __init__(self, *, adopted: bool) -> None:
        self.adopted = adopted
        self.clicks: list[tuple[str, bool]] = []

    def provide_approval(self, call_id: str, granted: bool) -> bool:
        self.clicks.append((call_id, granted))
        return self.adopted


class FakeScheduler:
    """``ReminderScheduler`` 那一半：只记"被叫醒过几次"。"""

    def __init__(self) -> None:
        self.reschedules = 0

    def reschedule(self) -> None:
        self.reschedules += 1


class FakeTray:
    """``TrayController`` 那一半：只记气泡。"""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def show_message(self, title: str, body: str) -> None:
        self.messages.append((title, body))


#: 一条消息里标题与正文之间那个空行。写成常量是因为**它出现在断言里**：那一行是
#: 渲染契约的一部分（``**标题**`` 独占一段），拼错了在屏幕上看得出来，在测试里拼错
#: 了却会变成一条"看起来在测格式"的空断言。
_PARA = "\n\n"


class FakeTimer:
    """单发表的替身：只记"排过多久、停过几次"。"""

    def __init__(self) -> None:
        self.started_with: list[int] = []
        self.stops = 0

    def start(self, ms: int) -> None:
        self.started_with.append(ms)

    def stop(self) -> None:
        self.stops += 1


class FakeSpeechWorker:
    """``ChatWorker`` 那一半：只记"被叫停过"。"""

    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


class Stub(QObject):
    """那些槽，以及它们碰得到的东西。

    **它是一个 ``QObject`` 而不是普通类。** 这一版"到点由她开口"要起一个真线程
    （``QThread(self)`` 要求父对象是 QObject），还要在槽里做 ``sender()`` 校验——
    两样都要求替身是个 QObject。仍然**不构造真的 ``MainWindow``**：那个东西会连
    Ollama、起几个线程、加载 WebEngine（见本文件开头那段）。
    """

    _on_tool_round_started = MainWindow._on_tool_round_started
    _on_tool_finished = MainWindow._on_tool_finished
    _on_approval_needed = MainWindow._on_approval_needed
    _on_tool_confirm = MainWindow._on_tool_confirm
    _on_tool_messages = MainWindow._on_tool_messages
    _on_tool_round_finished = MainWindow._on_tool_round_finished
    _on_reminder_fired = MainWindow._on_reminder_fired
    _on_submit = MainWindow._on_submit
    _tell_reminder = MainWindow._tell_reminder
    _wake_up = MainWindow._wake_up
    _start_reminder_speech = MainWindow._start_reminder_speech
    _retire_speech = MainWindow._retire_speech
    _on_speech_finished = MainWindow._on_speech_finished
    _on_speech_failed = MainWindow._on_speech_failed
    _time_out_speech = MainWindow._time_out_speech
    _stop_speech = MainWindow._stop_speech
    _track = MainWindow._track

    def __init__(
        self,
        *,
        streaming: bool = True,
        worker: FakeWorker | None = None,
        backend: Any = None,
        memory: Any = None,
    ) -> None:
        super().__init__()
        self.chat_view = Recorder()
        self._streaming = streaming
        self._agent_worker = worker
        self._scheduler = FakeScheduler()
        self._tray = FakeTray()
        self.stored: list[Any] = []
        #: 她开口那一轮**不许**碰到的东西。用替身挡住而不是省掉：省掉的话，
        #: 一次误调的 `_remember` 会因为 AttributeError 而红在别的地方，而这里要断言
        #: 的是"它根本没被调过"。
        self.remembered: list[Any] = []
        # ── 她开口那一档 ──
        self._backend = backend
        self._memory = memory
        self._speak_thread: Any = None
        self._speak_worker: Any = None
        # 这一轮还算不算数。**让路 = 翻这一位，不是把上面两个成员置空**（那会
        # 析构一个可能还在跑的对象，见 `MainWindow._retire_speech`）。
        self._speak_live = False
        self._speak_alert: Alert | None = None
        self._speak_timeout = FakeTimer()
        self._live_threads: list[Any] = []
        # 窗口状态：默认"正开着"。三档叫醒各由一条测试改这两个值。
        self.visible = True
        self.minimized = False
        self.window_calls: list[str] = []
        self.flashes = 0

    def _append(self, message: Any) -> None:
        self.stored.append(message)

    def _remember(self, reply_text: str, **kwargs: Any) -> None:
        self.remembered.append((reply_text, kwargs))

    # ── 窗口三态与任务栏闪烁（真窗口才有，替身只记） ──

    def isVisible(self) -> bool:
        return self.visible

    def isMinimized(self) -> bool:
        return self.minimized

    def show(self) -> None:
        self.window_calls.append("show")
        self.visible = True

    def showNormal(self) -> None:
        self.window_calls.append("showNormal")
        self.minimized = False

    def raise_(self) -> None:
        self.window_calls.append("raise")

    def activateWindow(self) -> None:
        self.window_calls.append("activateWindow")

    def _flash_taskbar(self) -> None:
        self.flashes += 1


def reminder_alert(*texts: str, late: bool = False) -> Alert:
    """一条（或一堆）到点，**按纯层那个 :class:`Alert` 的形状造**。

    不手搓一个只有 ``heading``/``body`` 的替身：这个槽的全部职责就是"把 Alert 自己
    那两句话原样送到两处"，替身会把"标题究竟怎么来的"一起替掉——而那正是上一版
    （标题由 Qt 那一侧现拼）搬进纯层时要解决的东西。
    """
    items = tuple(
        Occurrence(
            reminder=Reminder(reminder_id=f"r{index}", text=text, due_at=1000.0, created_at=900.0),
            at=1000.0,
        )
        for index, text in enumerate(texts)
    )
    return Alert(items=items, late=late)


def statuses(stub: Stub) -> list[str]:
    return [call[1] for call in stub.chat_view.calls if call[0] == "setStatus"]


# ── 卡片 ───────────────────────────────────────────────────────────────────


def test_a_pending_approval_draws_a_card_in_the_stream(qapp: Any) -> None:
    """收到问询就画卡片，而且**先把上一句助手的话收尾**。

    不收尾的话，模型那句"我先记一下"会和卡片贴在同一条气泡里——用户读到的就是
    "助手说了一段格式奇怪的话"，而不是"有个动作在等我点头"。

    倒计时那项也一起钉：``app.js`` 读的键名是 ``timeoutMs``，写错一个字母的表现是
    卡片上永远不显示还有多久——而它恰好是用户决定"要不要理它"的唯一依据。
    """
    stub = Stub()

    stub._on_approval_needed("c1", "write_memory", "content：用户喜欢美式")

    assert stub.chat_view.calls == [
        "endMessage",
        (
            "appendToolConfirm",
            "c1",
            "write_memory",
            "content：用户喜欢美式",
            {"timeoutMs": int(APPROVAL_TIMEOUT_S * 1000)},
        ),
        ("setStatus", "等待你确认：write_memory"),
    ], "状态栏那一句是唯一一处「现在轮到你」的提示：卡片可能出现在用户翻不到的地方"


def test_a_card_for_an_already_finished_turn_is_not_drawn(qapp: Any) -> None:
    """已经复位过的那一轮不画卡片。

    那一轮的闸门已经被拆了，画出来的卡片点下去只会被丢弃（``provide`` 返回
    ``False``）——用户面对的是一个必然失效的按钮，还要自己猜为什么。
    """
    stub = Stub(streaming=False)

    stub._on_approval_needed("c1", "write_memory", "content：甲")

    assert stub.chat_view.calls == []


def test_a_click_is_passed_to_the_worker_and_answered_on_the_card(qapp: Any) -> None:
    """点击 → 直连 worker → 卡片上写结果。**四层走完才叫"点到了"。**"""
    worker = FakeWorker(adopted=True)
    stub = Stub(worker=worker)

    stub._on_tool_confirm("c1", True)

    assert worker.clicks == [("c1", True)]
    assert stub.chat_view.calls == [("resolveToolConfirm", "c1", True, True)]
    assert statuses(stub) == [], "采纳了就不该再解释一句"


def test_a_click_that_was_not_adopted_says_so(qapp: Any) -> None:
    """没被采纳时必须**与"已拒绝"分开显示**。

    用户点的是「允许」，而闸门记的是没发生。不区分的话，那正是一条"用户以为他同意了、
    系统认为没人同意"的静默分歧——界面上两句话长得一样，而后果差得很远。
    """
    worker = FakeWorker(adopted=False)
    stub = Stub(worker=worker)

    stub._on_tool_confirm("c1", True)

    assert stub.chat_view.calls[0] == ("resolveToolConfirm", "c1", True, False)
    assert any("失效" in text for text in statuses(stub)), "卡片失效了，但没有一句解释"


def test_a_click_with_no_worker_is_not_adopted(qapp: Any) -> None:
    """这一轮已经收摊（``_agent_worker`` 是 ``None``）→ 同样按"没采纳"收场。

    这是最容易写错的一支：顺手 ``if worker: ... else: return`` 的话，卡片会停在
    "已经点过了"的样子上，而没有任何结果被写回去。
    """
    stub = Stub(worker=None)

    stub._on_tool_confirm("c1", False)

    assert stub.chat_view.calls[0] == ("resolveToolConfirm", "c1", False, False)


# ── 工具结果的三种样子 ─────────────────────────────────────────────────────


def payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "call_id": "c1",
        "name": "search_memory",
        "ok": True,
        "denied": False,
        "clipped": False,
        "content": "三条结果",
        "detail": "",
    }
    base.update(overrides)
    return base


def test_a_result_gets_its_own_line_and_a_footer(qapp: Any) -> None:
    """结果单独一条，标题说得出是哪件事，``detail`` 当页脚。

    ``role="tool"``（不是 ``assistant``）：回看时它必须一眼与模型说的话分得开，
    否则那个虚线框里的东西会被读成"助手说了一段格式奇怪的话"。
    """
    stub = Stub()

    stub._on_tool_finished(payload(detail="账本：注入 3 条（召回 2 / 最近 1）"))

    assert stub.chat_view.calls == [
        "endMessage",
        (
            "appendMessage",
            "tool",
            "三条结果",
            {"title": "工具结果 · search_memory", "footer": "账本：注入 3 条（召回 2 / 最近 1）"},
        ),
    ]


def test_a_denial_is_titled_as_such_and_carries_no_footer(qapp: Any) -> None:
    """被拒绝：标题写明，而 ``detail``（这里是"闸门拒绝"）不当页脚。

    页脚的位置是给"这一轮召回为什么是这几条"那种账本的；把拒绝原因塞进去，用户
    读到的是一句重复的话，而真正该看的那句在正文里。
    """
    stub = Stub()

    stub._on_tool_finished(
        payload(ok=False, denied=True, content="用户没有同意执行 search_memory，这一步跳过了。")
    )

    assert stub.chat_view.calls[-1] == (
        "appendMessage",
        "tool",
        "用户没有同意执行 search_memory，这一步跳过了。",
        {"title": "已拒绝 · search_memory"},
    )


def test_a_failure_keeps_the_stack_off_the_card(qapp: Any) -> None:
    """失败：标题是"工具失败"，栈不进页脚。

    失败时 ``detail`` 是完整回溯，几千字——放上去只会把真正该看的那句失败原因挤走。
    它在 ``tool_finished`` 的 payload 里还在（给排查用），只是不占卡片的位置。
    """
    stub = Stub()

    stub._on_tool_finished(
        payload(ok=False, content="search_memory 执行失败（KeyError）：'query'", detail="栈" * 500)
    )

    assert stub.chat_view.calls[-1] == (
        "appendMessage",
        "tool",
        "search_memory 执行失败（KeyError）：'query'",
        {"title": "工具失败 · search_memory"},
    )


def test_a_clipped_result_says_so_in_the_footer(qapp: Any) -> None:
    """进了上下文的那份被截断过 → 页脚要说出来。

    显示的是完整原文（``content``），所以不说的话，用户看到的与实际进模型的不是
    同一份东西——而"模型为什么没看见后半段"就成了一个查不出来的问题。
    """
    stub = Stub()

    stub._on_tool_finished(payload(clipped=True))

    _kind, _role, _text, meta = stub.chat_view.calls[-1]
    assert "已截断" in meta["footer"]


# ── 登记提醒要立刻叫醒表 ───────────────────────────────────────────────────


def test_setting_a_reminder_re_arms_the_timer(qapp: Any) -> None:
    """``set_reminder`` 跑完 → 表立刻改排。

    **这条测的是接线。** :class:`~axo.ui.reminder_scheduler.ReminderScheduler`
    自己有十几条测试，但它们全在验那个对象**自身**的行为——"没有任何东西调
    ``reschedule()``"这件事它们一条都盖不住，而症状与"调度器坏了"看起来一模一样
    （提醒不按时响）。区分开它们的唯一办法就是在这里点一次名。
    """
    stub = Stub()

    stub._on_tool_finished(payload(name="set_reminder"))

    assert stub._scheduler.reschedules == 1


def test_other_tools_do_not_re_arm_the_timer(qapp: Any) -> None:
    """别的工具不许碰表：每调一次工具就重排一次的话，重排的代价变成了"每次工具
    调用都重读一遍提醒文件"，而且那会让"表为什么在这个时候醒"变得查不清。"""
    stub = Stub()

    stub._on_tool_finished(payload(name="search_memory"))
    stub._on_tool_finished(payload(name="read_file"))

    assert stub._scheduler.reschedules == 0


# ── 到点了要说两处 ─────────────────────────────────────────────────────────


def test_a_reminder_is_told_in_both_places(qapp: Any) -> None:
    """**两处都要说**，而且说的是同一句话。

    只说对话区：窗口藏在托盘里时没人看得见——那正是"设了提醒却不会响"。
    只说气泡：气泡会消失，回看对话时那段历史里没有这件事。
    """
    stub = Stub()

    stub._on_reminder_fired(reminder_alert("买牛奶"))

    # 正文的形状（``**标题**`` + 空行 + 正文）在这里**写死**，不从 ``Alert`` 现算：
    # 现算的话两边共用同一个表达式，形状改了这条测试跟着改，等于什么都没钉。
    assert stub.chat_view.calls == [
        ("appendMessage", "system", "**提醒**" + "\n\n" + "买牛奶", None)
    ]
    assert stub._tray.messages == [("提醒", "买牛奶")]


def test_a_backlog_of_reminders_is_one_message_not_many(qapp: Any) -> None:
    """一批错过的**只占一条**。

    调度器已经把它们聚合成一句了（见 :mod:`axo.ui.reminder_scheduler`）；
    这条钉的是"这里没有自作主张拆开"——逐个弹的话，关了一周再打开会变成一场通知
    风暴，而风暴的结局是用户把通知关掉。
    """
    stub = Stub()

    stub._on_reminder_fired(reminder_alert("甲", "乙", "丙"))

    assert len(stub.chat_view.calls) == 1
    assert len(stub._tray.messages) == 1
    _kind, _role, text, _meta = stub.chat_view.calls[-1]
    assert "3 条提醒到点了" in text
    for each in ("甲", "乙", "丙"):
        assert each in text, "聚合的那一句里少了一条"


def test_the_two_places_say_the_same_words(qapp: Any) -> None:
    """两处说的是**同一句**，而且都来自 :class:`Alert` 自己。

    这个槽能犯的错是"顺手改一处"：气泡里写正文、对话区里写标题，或者其中一处自己
    拼一句话。改完两边都还在响，只是说的不是一回事——而它被发现的时机是"用户照着
    气泡去找那句话，在对话区里找不到"。
    """
    stub = Stub()
    one = reminder_alert("该睡了", "交房租")

    stub._on_reminder_fired(one)

    _kind, _role, text, _meta = stub.chat_view.calls[-1]
    title, body = stub._tray.messages[-1]
    assert (title, body) == (one.heading(), one.body())
    assert one.heading() in text and one.body() in text


# ── 逐字稿 ─────────────────────────────────────────────────────────────────


def pair() -> tuple[ChatMessage, ChatMessage]:
    """一对真的从循环里出来的行（见 ``agent/loop.py`` 的 ``MessagesAppended``）。"""
    call = ToolCall(id="c1", name="write_memory", arguments={"content": "甲"})
    return (
        ChatMessage(role="assistant", content="我记一下：", tool_calls=(call,)),
        ChatMessage(role="tool", content="写好了", tool_call_id="c1"),
    )


def test_the_whole_pair_is_written_to_the_transcript(qapp: Any) -> None:
    """一对行落进会话，且 tool 行带着**发起它的那个工具名**。

    ``tool_name`` 是 ``stored_rows`` 补的（事件里没有它）。它只用于回看显示，但少了
    它，逐字稿里那行就是一条没有主语的"写好了"——几个月后回看，谁也说不清那是谁写的。
    """
    stub = Stub()

    stub._on_tool_messages(pair(), ())

    assert [message.role for message in stub.stored] == ["assistant", "tool"]
    assert stub.stored[1].tool_name == "write_memory"
    assert stub.stored[1].tool_call_id == "c1"
    assert stub.stored[0].tool_calls[0]["id"] == "c1", "逐字稿里的调用要能配回它的结果"
    assert stub.stored[1].no_replay is None, "普通结果不带不回放的标记"


def test_the_reason_travels_with_the_rows_into_the_transcript(qapp: Any) -> None:
    """拒绝的原因**和行一起**落进逐字稿。

    这一位是 L5 唯一能据以决定"要不要把这一对搬进后续轮次"的东西（``session/`` 够不到
    ``agent/``，见 ``Message.no_replay``）。丢了它，逐字稿会把一次拒绝记成一次正常
    结果，然后每一轮都把它回放给模型看——症状是模型再也不肯调那个工具。
    """
    stub = Stub()

    stub._on_tool_messages(pair(), (("c1", "denied"),))

    assert stub.stored[1].no_replay == "denied"
    assert stub.stored[0].no_replay is None, "标记只落在 tool 行上"


def test_a_payload_that_is_not_a_pair_is_dropped(qapp: Any) -> None:
    """形状不对就不写。

    ``Signal(object, object)`` 没有静态类型可依，所以这里判一次形状。判错方向的代价
    不对称：少写一对是回看时少几步，而把一段来路不明的行写进逐字稿**就再也删不掉了**。
    """
    stub = Stub()

    stub._on_tool_messages(list(pair()), ())

    assert stub.stored == []


def test_a_malformed_reason_list_still_writes_the_rows(qapp: Any) -> None:
    """原因的形状坏了，**行照样要落**——默认按"都能回放"处理。

    两个参数判错方向的代价不同，所以不能一起丢：行本身是真的（那些调用确实执行过），
    丢了它是一份"做过但没记"的逐字稿，比多回放一对严重得多。
    """
    stub = Stub()

    stub._on_tool_messages(pair(), "denied")

    assert [message.role for message in stub.stored] == ["assistant", "tool"]
    assert stub.stored[1].no_replay is None


# ── 轮次与收尾 ─────────────────────────────────────────────────────────────


def test_each_round_gets_its_own_bubble(qapp: Any) -> None:
    """第 0 轮不收尾（还没有上一轮），之后每一轮都要。

    不收尾的话，几轮的"边想边说"会连成一条气泡，读起来像模型一口气说了三段互相
    矛盾的话——而真正的分界（中间夹着工具结果）在文字里完全看不出来。
    """
    first = Stub()
    first._on_tool_round_started(0, 1)
    assert first.chat_view.calls == [("setStatus", "第 1 轮")]

    later = Stub()
    later._on_tool_round_started(1, 0)
    assert later.chat_view.calls == [
        "endMessage",
        ("setStatus", "第 2 轮（收尾，不再调工具）"),
    ], "收尾那一轮没有工具了，状态栏要说出来——否则用户会以为还能接着调"


def test_unexecuted_calls_are_reported_even_when_the_turn_was_stopped(qapp: Any) -> None:
    """``unexecuted`` **不许被吞掉**，而且停止也要说。

    顺序是刻意的：先说"模型还想调但已经没有工具了"，再让 ``interrupted`` 把状态栏
    让给"已停止"。反过来的话，那几句话会在用户点停止的那一刻消失——而它们恰恰是
    "这一轮没做完什么"的唯一线索。
    """
    stub = Stub()

    stub._on_tool_round_finished(
        "", {"interrupted": True, "unexecuted": ("read_file",), "rounds": 2, "tool_calls": 3}
    )

    assert len(stub.chat_view.calls) == 1
    kind, role, text, _meta = stub.chat_view.calls[0]
    assert kind == "appendMessage" and role == "system"
    assert "read_file" in text


def test_the_status_line_summarises_the_rounds(qapp: Any) -> None:
    """正常收场：状态栏给出"用了几轮、调了几次"。"""
    stub = Stub()

    stub._on_tool_round_finished(
        "", {"interrupted": False, "unexecuted": (), "rounds": 2, "tool_calls": 3}
    )

    assert statuses(stub) == ["2 轮 · 调用工具 3 次"]


def test_a_plain_chat_turn_leaves_the_status_alone(qapp: Any) -> None:
    """一轮工具都没调 → 不写状态（``rounds`` 为 0）。

    写上去的话状态栏会多一句"0 轮 · 调用工具 0 次"，而它和 ``ChatWorker`` 那条路上
    的收场摆在同一个位置——用户会以为这次回答走了工具。
    """
    stub = Stub()

    stub._on_tool_round_finished(
        "", {"interrupted": False, "unexecuted": (), "rounds": 0, "tool_calls": 0}
    )

    assert stub.chat_view.calls == []


# ── 到点由她开口 ───────────────────────────────────────────────────────────
#
# 这一族测的是**"她真的会开口"与"开口失败时提醒不许消失"这两件事同时成立**。
# 前一半要一次真的线程与真的信号（假件替不掉：要证的正是接线本身），后一半全在
# 降级那条路上，逐条可断言。


def _wait_until(qapp: Any, until: Callable[[], bool], *, ms: int = 3000) -> bool:
    """转事件循环直到 ``until()`` 为真或超时。

    真起线程的那几条只能这样等：worker 的 ``finished`` 是从工作线程发回来的**排队**
    信号，不转事件循环它永远送不到。
    """
    deadline = time.monotonic() + ms / 1000
    ok = False
    while time.monotonic() < deadline:
        qapp.processEvents()
        if until():
            ok = True
            break
        time.sleep(0.01)
    # **把排队的延迟删除也排干，再走。** ``track`` 的收尾是三步排队送来的
    # （``worker.deleteLater`` → ``release``（清单在这里变空）→ ``thread.deleteLater``），
    # 所以上面那个判据在**第二步**就为真了。剩下的那两条延迟删除要是留到下一个测试去
    # 处理，它们就会在另一个测试的事件循环里落地——那时这些对象的 Python 包装器早已
    # 回收，Qt 当场 ``abort``（实测：单独的测试全绿，两个连着跑就崩在第二个）。
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    return ok or until()


def _speaking_stub(qapp: Any, **script: Any) -> tuple[Stub, ScriptedBackend]:
    """一个能开口的替身：有后端、有库、并按脚本吐一句。"""
    backend = ScriptedBackend(**script)
    stub = Stub(streaming=False, backend=backend, memory=MemoryStore(":memory:"))
    return stub, backend


def test_she_speaks_when_everything_is_ready(qapp: Any) -> None:
    """条件齐了：她真的开口，而且**只做这一件事**。

    一次开口要落到三处，而每一处漏掉都是一类具体的坏：

    * **托盘气泡**照旧（它不经过模型，是"现在就要知道"唯一可靠的通道）；
    * **对话区一条 ``proactive``**——不是 ``assistant``：那一轮没有 user 行，混进
      助手的气泡里，回看时读到的是"这是你问来的"；
    * **逐字稿一行**（``assistant`` + ``proactive=True``），**且不进记忆库**、
      **不动 ``_streaming``**：走 ``_remember`` 会把提醒正文当成"用户说过的一句话"
      入库，而锁输入框会毁掉"你还在跟我说话，我才能提"这条立论。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("该睡了，早点休息。", done=True)]])
    alert = reminder_alert("该睡了")

    stub._on_reminder_fired(alert)

    assert stub._tray.messages == [(alert.heading(), alert.body())]
    assert _wait_until(qapp, lambda: not stub._live_threads), "那一轮没跑完"
    assert stub._speak_timeout.started_with == [SPEECH_TIMEOUT_MS], "20 秒那块表没排上"

    assert stub.chat_view.calls[-1] == (
        "appendMessage",
        "proactive",
        "该睡了，早点休息。",
        {"title": "提醒"},
    )
    assert len(stub.stored) == 1
    line = stub.stored[0]
    assert (line.role, line.content, line.proactive) == ("assistant", "该睡了，早点休息。", True)
    assert stub.remembered == [], "她主动说的那句进了记忆库——它会变成'用户说过的一句话'"
    assert stub._streaming is False, "她开口那一轮锁住了输入框"
    assert backend.calls == 1
    # 请求是**一条 system + 一条事件行**（见 ``agent/proactive.py`` 模块文档：
    # 只有 system 时这个模型一个字都说不出来），而**不带工具**。
    assert backend.roles(0) == ["system", "user"]
    assert backend.tools_declared(0) == ()


def test_a_late_reminder_is_never_spoken(qapp: Any) -> None:
    """启动时收掉的那一批**一律降级**，一条请求都不发。

    取舍的唯一来由是时序：``scheduler.start()`` 排在 ``__init__`` 最后，而模型列表是
    异步拉的——那一刻后端几乎必然是 ``None``。语义上它本来就是"错过"，由程序说比由她
    补一句"到点了"更诚实。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("不该出现", done=True)]])
    alert = reminder_alert("该睡了", late=True)

    stub._on_reminder_fired(alert)

    assert backend.calls == 0, "错过的那一批也去问模型了"
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert (role, text) == ("system", "**错过了一条提醒**" + _PARA + "该睡了")


def test_it_does_not_speak_over_the_user(qapp: Any) -> None:
    """用户正问到一半：**降级**，不插嘴。

    那时候她的气泡会长在用户这一轮的流里，而用户等的是自己那个问题的答案。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("不该出现", done=True)]])
    stub._streaming = True

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert backend.calls == 0
    assert stub.chat_view.calls[-1][1] == "system"


def test_it_does_not_speak_while_she_is_already_speaking(qapp: Any) -> None:
    """两条提醒前后脚到点：第二条由程序说，**不排队**。

    排队会让"她说了什么"与"现在是几点"对不上——第二句开口时，第一件早就过去了。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("不该出现", done=True)]])
    busy = FakeSpeechWorker()
    stub._speak_worker = busy
    stub._speak_live = True

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert backend.calls == 0
    assert busy.stops == 0, "第二条不该把第一条掐掉"
    assert stub.chat_view.calls[-1][1] == "system"


# ── 开口失败：提醒不许跟着消失 ─────────────────────────────────────────────


def test_an_empty_reply_falls_back_to_the_program(qapp: Any) -> None:
    """模型只吐了空白 → **由程序说**，而且说出原因。

    空回复不是"她说了句空话"，是这一次开口没发生。用户看不见的失效必须留一行字，
    否则"她有时开口、有时不开口"就成了一件没有解释的事。
    """
    stub, _backend = _speaking_stub(qapp, script=[[text_chunk("   ", done=True)]])

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert _wait_until(qapp, lambda: not stub._live_threads)
    assert stub.stored == [], "空回复不该进逐字稿"
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "该睡了" in text and "没有说出话来" in text


def test_a_failed_speech_falls_back_to_the_program(qapp: Any) -> None:
    """后端报错 → 不重试，但**那条提醒照旧要说**，并带上原因。"""
    stub, backend = _speaking_stub(qapp, script=[[]], raise_at=0)

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert _wait_until(qapp, lambda: not stub._live_threads)
    assert backend.calls == 1, "失败之后又试了一次"
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "该睡了" in text and "模型没能开口" in text


def test_the_timeout_tells_and_stops(qapp: Any) -> None:
    """二十秒到了：**叫停那一轮**，并由程序把提醒说出来。

    叫停这件事不是体验问题：那个线程还挂在后端上（读超时 300 秒），而收尾只等 3 秒
    ——它不停下来就是下一次 ``0xC0000409``。
    """
    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    alert = reminder_alert("该睡了")
    worker = FakeSpeechWorker()
    stub._speak_worker = worker
    stub._speak_live = True
    stub._speak_alert = alert

    stub._time_out_speech()

    assert worker.stops == 1
    assert stub._speak_live is False and stub._speak_alert is None
    # **引用留着**：这一轮可能还跑在工作线程里，丢引用会析构一个正在执行的
    # QObject（见 `MainWindow._retire_speech`）。摘引用是 `track` 的 `release`
    # 闭包在 `thread.finished` 之后干的事。
    assert stub._speak_worker is worker
    assert stub._speak_timeout.stops == 1
    assert "等了二十秒还没说完" in stub.chat_view.calls[-1][2]

    # 幂等：表晚到的第二次、或者 finished 刚回来之后的那一次，都不该再说一遍。
    stub._time_out_speech()
    assert len(stub.chat_view.calls) == 1


def test_a_stale_speech_result_is_dropped(qapp: Any) -> None:
    """被让掉的那一轮，**迟到的 ``finished`` 不许落地**。

    它的 ``finished`` 是排队来的，往往在用户那一轮开始之后才送到。收下它就会把上一件事
    的一句话插进这一轮中间，而它看起来像她答非所问——所以那两个槽的第一句是发送者校验
    （同 :meth:`_on_assembly_ready` 那条手法）。
    """
    stub, _backend = _speaking_stub(qapp, script=[[text_chunk("该睡了，早点休息。", done=True)]])

    stub._on_reminder_fired(reminder_alert("该睡了"))
    stub._stop_speech(tell=False)
    assert _wait_until(qapp, lambda: not stub._live_threads), "那一轮没跑完"

    assert stub.stored == [], "被让掉的那一句还是进了逐字稿"
    assert not [call for call in stub.chat_view.calls if call[1:2] == ("proactive",)]


def test_the_user_speaking_makes_her_stop(qapp: Any) -> None:
    """用户开口 = 她让路，而且**由程序把那条提醒补说一遍**。

    这一句补白是必需的：她那一轮被抢话打断了，而那条提醒**真的响过**。不补的话，用户
    这一轮的气泡会把"刚刚发生过什么"整个顶掉，只剩一条会自己消失的托盘气泡。

    ``_stop_speech`` 排在 ``_on_submit`` 的第一行、``_streaming`` 那一支**之前**：
    生成中被忽略的那一次提交，也照样该让她让路。
    """
    stub = Stub(streaming=True, backend=object(), memory=MemoryStore(":memory:"))
    alert = reminder_alert("该睡了")
    worker = FakeSpeechWorker()
    stub._speak_worker = worker
    stub._speak_live = True
    stub._speak_alert = alert

    stub._on_submit("等一下")

    assert worker.stops == 1
    assert stub._speak_worker is worker, "让路把 worker 的引用丢了——那会析构一个还在跑的对象"
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "该睡了" in text
    assert "本该由她开口" not in text, "按设计让路的那一次不该带原因——那是常规状态"


# ── 叫醒窗口的三档 ─────────────────────────────────────────────────────────


def test_an_open_window_is_only_flashed(qapp: Any) -> None:
    """窗口正开着：**绝不抢焦点**，只在任务栏闪一下。

    用户可能正在别的窗口里打字，而这条消息要的是"知道一下"、不是"停下来处理"。
    """
    stub = Stub()

    stub._wake_up()

    assert stub.window_calls == []
    assert stub.flashes == 1


def test_a_minimized_window_comes_back_without_stealing_focus(qapp: Any) -> None:
    """最小化：恢复，但**不抢焦点**——它已经回到屏幕上了，用户看得见。"""
    stub = Stub()
    stub.minimized = True

    stub._wake_up()

    assert stub.window_calls == ["showNormal"]


def test_a_hidden_window_is_really_called_out(qapp: Any) -> None:
    """藏在托盘里：**三样都要**（show + raise + activate），少一样就是"点了没反应"。"""
    stub = Stub()
    stub.visible = False

    stub._wake_up()

    assert stub.window_calls == ["show", "raise", "activateWindow"]


# ── live 与 reload 画得一样 ────────────────────────────────────────────────


def test_the_row_is_the_same_when_read_back(qapp: Any) -> None:
    """当时屏幕上的那一行与回看时算出来的那一行**逐字段相同**。

    这是 :func:`~axo.ui.main_window.display_rows` 存在的全部意义，而 ``proactive``
    是这一版新加的一支：live 传 ``proactive``、reload 传一个别的角色（或者反过来传了
    一个 live 没传的标题），症状是"切一次会话，那条提醒换了副长相"——没有任何测试会红。
    """
    stub, _backend = _speaking_stub(qapp, script=[[text_chunk("该睡了，早点休息。", done=True)]])
    stub._on_reminder_fired(reminder_alert("该睡了"))
    assert _wait_until(qapp, lambda: not stub._live_threads)

    _kind, live_role, live_text, live_meta = stub.chat_view.calls[-1]
    (row,) = display_rows(stub.stored)

    assert (row["role"], row["text"], row["meta"]) == (live_role, live_text, live_meta)


def test_an_ordinary_reply_is_still_an_ordinary_row(qapp: Any) -> None:
    """对照组：普通回答回看时**不长成她主动说的那条**。

    ``proactive`` 那一支要是写成"凡 assistant 都换角色"，整份历史会一夜之间换一副
    长相，而没有任何一条数据变过。
    """
    (row,) = display_rows([Message(role="assistant", content="好，我查一下。")])

    assert row == {"role": "assistant", "text": "好，我查一下。", "meta": {}}
