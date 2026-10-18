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
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedBackend, text_chunk, tool_chunk
from PySide6.QtCore import QCoreApplication, QEvent, QObject

from axo import autostart
from axo.agent.gate import APPROVAL_TIMEOUT_S
from axo.agent.proactive_chat import Material
from axo.agent.reminders import Alert, Occurrence, Reminder
from axo.asr import Transcription
from axo.llm.base import ChatMessage, ToolCall
from axo.session.store import Message
from axo.tts import SpeechReceipt, SpeechResult
from axo.ui import voice as voice_module
from axo.ui.main_window import SPEECH_TIMEOUT_MS, MainWindow, display_rows
from axo.ui.proactive_scheduler import ProactiveScheduler
from axo.ui.settings_store import ProactiveSettings, ReminderSettings, SpeechSettings
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

    def set_voice_state(self, state: str) -> None:
        self.calls.append(("setVoiceState", state))


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


class FakeDialog:
    """``SettingsDialog`` 那一半：只记"被回填了哪句话"。"""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def set_reminder_note(self, note: str) -> None:
        self.notes.append(note)


class FakeSettingsStore:
    """``SettingsStore`` 那一半：只记"存过哪几份"。"""

    def __init__(self) -> None:
        self.saved: list[Any] = []

    def save_reminders(self, settings: Any) -> None:
        self.saved.append(settings)


class FakeVoice:
    """``VoiceRecorder`` 那一半：只记"起过几次、关过几次、在不在录"。

    ``start`` 收一个 ``sink`` 并**回空串**（= 起来了）。回一句人话那一支另有测试
    （麦克风被 Windows 隐私设置拦下），这里要的是"起来了"那一条路。
    """

    def __init__(self, *, refuses: str = "") -> None:
        self.listening = False
        self.starts = 0
        self.stops = 0
        self.refuses = refuses

    def start(self, sink: Any) -> str:
        self.starts += 1
        if self.refuses:
            return self.refuses
        self.listening = True
        return ""

    def stop(self) -> None:
        self.stops += 1
        self.listening = False


class FakePlayer:
    """``SpeechPlayer`` 的替身：只记"念了几段、停过几次、在不在念"。"""

    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stops = 0
        self.playing = False  # 测试要判打断回执时自己拨成 True

    def speak(self, wav: bytes) -> str:
        self.played.append(wav)
        return ""

    def stop(self) -> None:
        self.stops += 1


class FakeTts:
    """``RemoteSpeech`` 的替身：按脚本回收条，记下被念的文本。"""

    def __init__(self, *, wav: bytes = b"WAV", reason: str = "") -> None:
        self._wav = wav
        self._reason = reason
        self.asked: list[str] = []

    def synthesize(self, text: str, *, voice: str = "") -> Any:
        self.asked.append(text)
        from axo.tts import SpeechResult

        return SpeechResult(wav=self._wav, reason=self._reason)


class FakeSpeechWorker:
    """``ChatWorker`` 那一半：只记"被叫停过"。"""

    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


class FakeSession:
    """一个会话的替身。**只有主动搭话用得上它**：``_last_activity`` 看两样东西——
    最后一条 ``user`` 行的时刻、以及它的 id（用来排掉"本会话刚说过的"那条记忆）。
    """

    def __init__(self, session_id: str = "S-now", messages: list[Any] | None = None) -> None:
        self.session_id = session_id
        self.messages = list(messages or [])


class FakeCall:
    """通话控制器的替身。**默认不在通话中**——这些测试量的是提醒/主动那两条路。"""

    def __init__(self) -> None:
        self.live = False
        self.muted = False
        self.started = 0
        self.stopped = 0

    def start(self) -> str:
        self.live = True
        self.started += 1
        return ""

    def stop(self, *, discard: bool = True) -> None:
        self.live = False
        self.stopped += 1

    def shutdown(self) -> None:
        self.stop()

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)

    def interrupt(self) -> None:
        return

    def on_user_turn_started(self) -> None:
        return

    def on_reply_finished(self, *, will_speak: bool) -> None:
        return

    def on_playback_finished(self) -> None:
        return


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
    _on_reminders_changed = MainWindow._on_reminders_changed
    _on_voice_toggled = MainWindow._on_voice_toggled
    _set_voice_state = MainWindow._set_voice_state
    _begin_voice = MainWindow._begin_voice
    _end_voice = MainWindow._end_voice
    _on_voice_finished = MainWindow._on_voice_finished
    _on_voice_done = MainWindow._on_voice_done
    _discard_voice = MainWindow._discard_voice
    _speak_reply = MainWindow._speak_reply
    _on_reply_finished = MainWindow._on_reply_finished
    _on_tts_done = MainWindow._on_tts_done
    _on_speech_played = MainWindow._on_speech_played
    _take_speech_receipt = MainWindow._take_speech_receipt
    _peek_speech_receipt = MainWindow._peek_speech_receipt
    _commit_speech_receipt = MainWindow._commit_speech_receipt
    _format_speech_receipt = MainWindow._format_speech_receipt
    _track = MainWindow._track
    # ── 主动搭话那一档（她先开口的另一条路）──
    _on_proactive_tick = MainWindow._on_proactive_tick
    _roll_over_if_new_day = MainWindow._roll_over_if_new_day
    _last_activity = MainWindow._last_activity
    _pick_material = MainWindow._pick_material
    _start_proactive_speech = MainWindow._start_proactive_speech
    _show_sticker = MainWindow._show_sticker
    # ── 通话 ──
    _on_call_toggled = MainWindow._on_call_toggled
    _start_call = MainWindow._start_call
    _end_call = MainWindow._end_call
    # ── 拖进来的文件 ──
    _on_files_dropped = MainWindow._on_files_dropped
    _on_attach_files = MainWindow._on_attach_files
    _attach_pending_images = MainWindow._attach_pending_images
    # ── 插件 ──
    _tools = MainWindow._tools
    _report_plugin_issues = MainWindow._report_plugin_issues

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
        self._call = FakeCall()
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
        #: 提醒那两个开关（设置 → 提醒）。默认那一份 = 两个都开，与真程序一致。
        self._reminder_settings = ReminderSettings()
        # ── 主动搭话那一档 ──
        #: 会话替身。``_last_activity`` 与"本会话的不当话头"两条都读它。
        self._session = FakeSession()
        #: 主动搭话的设置。**默认把免打扰关掉**（两个数相等）——否则这几条测试在
        #: 23:00 之后跑就会红，而红的原因与它们要验的东西毫无关系。
        self._proactive_settings = ProactiveSettings(quiet_from=0, quiet_to=0, allow_tools=False)
        #: 她这一轮拿的是哪个话头（与 ``_speak_alert`` 互斥，两种"她先开口"共用收尾）。
        self._speak_material: Material | None = None
        #: 她上一次开口（提醒与主动都算）。她说话也算"有动静"。
        self._last_speech_at: datetime | None = None
        #: 今天主动开口几次、用掉哪些话头，以及那个用来跨天清零的日期。
        self._spoken_today = 0
        self._used_materials: list[str] = []
        self._proactive_day: date | None = None
        # ── 拖进来的文件那一档 ──
        self._pending_attachments: tuple[Any, ...] = ()
        self._pending_images: tuple[str, ...] = ()
        self._settings_store = FakeSettingsStore()
        #: 设置窗口（可能还没开）。``_on_reminders_changed`` 往它上面回填结果那一行。
        self._settings_dialog: Any = None
        # ── 语音输入那一档 ──
        self._voice = FakeVoice()
        self._voice_state = ""
        #: ── 语音输出那一族（她开口）──
        self._voice_settings = SpeechSettings()
        self._speech_player = FakePlayer()
        self._speech_backend = FakeTts()
        self._tts_thread: Any = None
        self._tts_worker: Any = None
        self._tts_live = False
        self._tts_serial = 0
        self._spoken_reply = False
        self._speech_receipt = None
        self._speaking_text = ""
        self._voice_worker: Any = None
        self._voice_thread: Any = None
        #: ``_on_voice_done`` 会把识别结果交给 ``_on_submit``——这里换成一个记录器，
        #: **要验的正是"它走的是那扇门"**（真 `_on_submit` 要连 Ollama、起线程）。
        # 窗口状态：默认"正开着"。三档叫醒各由一条测试改这两个值。
        self.visible = True
        self.minimized = False
        self.window_calls: list[str] = []
        self.flashes = 0

    def _reset_generation_state(self) -> None:
        """替身版：只记一次。真件那一半碰的是会话/按钮/三个 worker——那是别处的事。"""
        self.resets = getattr(self, "resets", 0) + 1

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


class VoiceStub(Stub):
    """语音那一族用的替身：**只换掉 `_on_submit`**。

    不给基类换（那会让 C4 那条"用户开口 = 她让路"的测试打到一个假门上——它要的正是真的
    那个 `_on_submit`）。这一族要验的是"识别结果进的是那扇门，与打字同一处"，而真门要连
    Ollama、起线程，所以这里记一笔就够。
    """

    def __init__(self, **fields: Any) -> None:
        super().__init__(**fields)
        self.submitted: list[str] = []
        self.last_spoken = False

    def _on_submit(self, text: str, *, spoken: bool = False, audio: bytes = "") -> None:
        self.submitted.append(text)
        self.last_spoken = spoken
        self.last_audio = audio


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


# ── 设置页那两个开关 ───────────────────────────────────────────────────────
#
# 两条通道各自一个开关：叫不叫醒窗口、让不让模型开口。**关掉任意一个，提醒本身照响**
# ——这正是它们可以默认开着的理由，而这三条测试量的就是这件事没有被"顺手连坐"。


def test_the_speak_switch_off_means_the_program_tells_it(qapp: Any) -> None:
    """关掉「由她说一句」：一条请求都不发，走的还是那条降级。

    它是模型慢或吵时的逃生口（本机 4B 模型实测 3–15 秒），所以关掉之后**一个字都不
    该少**：提醒照旧进对话区。而不带原因——那是常规状态，不是失败。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("不该出现", done=True)]])
    stub._reminder_settings = ReminderSettings(speak_on_alert=False)

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert backend.calls == 0, "关掉了还去问了模型"
    assert stub._live_threads == []
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "该睡了" in text
    assert "本该由她开口" not in text


def test_the_wake_switch_off_means_the_window_stays_put(qapp: Any) -> None:
    """关掉「把窗口叫出来」：窗口一动不动，而提醒照旧。

    这条钉的是"两个开关管的是两件不相干的事"——气泡、叫醒、开口是三条独立通道，
    关掉一条不该顺手关掉别的。**托盘气泡没有开关，也不该有**：它是"现在就要知道"
    唯一一条不经过模型的通道。
    """
    stub = Stub()  # 默认 ``_streaming=True``，于是走降级那一路
    stub._reminder_settings = ReminderSettings(wake_on_alert=False)

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert stub.window_calls == [] and stub.flashes == 0, "叫醒开关关了还把窗口弄动了"
    assert stub._tray.messages, "托盘气泡跟着一起没了"
    assert stub.chat_view.calls[-1][1] == "system"


def test_the_autostart_switch_is_parked_on_the_registry_helper(qapp: Any) -> None:
    """「开机自启」那一格的接线：**它走的是 ``autostart.apply``，而且值原样传过去**。

    这里把那个函数换掉（真跑一遍会往**这台机器**的注册表 ``Run`` 键写一个值——
    那正是本仓最不想发生的一类副作用：不留痕迹、也不报错）。换掉之后要证的只剩接线：
    勾上 → ``apply(True)``、取消 → ``apply(False)``。

    顺带钉住"改完开关就说一句"：成功时回填空串（= 没事），因为那一行本来就写着命令行，
    再写一遍"已设成开机自启"是噪音。
    """
    stub = Stub()
    stub._settings_dialog = FakeDialog()
    applied: list[bool] = []
    monkeypatch = pytest.MonkeyPatch()
    # 替身按"勾上就真的写成了"回话，于是两条分支都走得到：True → 回填那句话，
    # False（取消，本来就没写）→ 回填空串。
    monkeypatch.setattr(autostart, "apply", lambda enabled: applied.append(enabled) or enabled)
    try:
        stub._on_reminders_changed(ReminderSettings(autostart=True))
        stub._on_reminders_changed(ReminderSettings(autostart=False))
    finally:
        monkeypatch.undo()

    assert applied == [True, False]
    assert stub._settings_dialog.notes == ["已设成开机自启。", ""]


def test_a_registry_failure_is_told_on_the_page(qapp: Any) -> None:
    """写不进去要说出来（同 ``_remember`` 那条纪律）：勾了却没生效，而界面上那个框
    仍然勾着——不说的话，用户要等到某天早上的提醒没响才发现。
    """
    stub = Stub()
    stub._settings_dialog = FakeDialog()
    monkeypatch = pytest.MonkeyPatch()

    def boom(_enabled: bool) -> bool:
        raise PermissionError("拒绝访问")

    monkeypatch.setattr(autostart, "apply", boom)
    try:
        stub._on_reminders_changed(ReminderSettings(autostart=True))
    finally:
        monkeypatch.undo()

    assert stub._settings_dialog.notes
    assert "拒绝访问" in stub._settings_dialog.notes[-1]


def test_the_switch_takes_effect_on_the_next_alert_and_is_saved(qapp: Any) -> None:
    """改完开关：**当场落盘，下一个到点就用新的**。

    生效机理是"到点那一枪现读 ``_reminder_settings``"，所以这里只要证"记下来了"
    与"下一次读的是新的"。**不通知调度器**：它管的是"什么时候该看一眼"，与"用户想不想
    被打扰"是两件事——把它搅进来只会让那一层多认识一个它不该认识的设置项。
    """
    stub, backend = _speaking_stub(qapp, script=[[text_chunk("该睡了，早点休息。", done=True)]])
    changed = ReminderSettings(speak_on_alert=False)

    stub._on_reminders_changed(changed)

    assert stub._settings_store.saved == [changed]
    assert stub._reminder_settings == changed
    stub._on_reminder_fired(reminder_alert("该睡了"))
    assert backend.calls == 0, "换了设置，下一个到点还是问了模型"


# ── 语音输入那条链 ──────────────────────────────────────────────────────────
#
# 录音与识别在 Python 侧（网页那个按钮只发"切换"），所以这一族验的是**接线**：
# 按一下会发生什么、识别回来的文字走哪扇门、以及"用不了"时说不说。
#
# **识别器要注入，不能靠本机的状态。** 这两条测试原来直接跑真的 ``DecodeWorker``，
# 于是"库没装 / 模型没放"那一支**取决于这台机器**：装过模型的机器上它不再是那一条路
# 了。2026-10-12 就是这么暴露的——模型一装上，`test_the_mic_button_toggles_recording`
# 当场红（它走进成功那一支，而替身没有 ``start``），而它在此之前一直是绿的，因为那条
# 路根本没被走到过。注入之后两种机器上跑的是同一条路。


class FakeRecognizer:
    """``DecodeWorker`` 里那个识别器：**问题可注入**。"""

    def __init__(self, problem: str = "", text: str = "") -> None:
        self.problem = problem
        self.fed = 0
        self._text = text

    def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    def finish(self) -> Transcription:
        return Transcription(text=self._text)


@pytest.fixture
def recognizer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """把 ``DecodeWorker`` 造出来的识别器换成可控的，**返回一个改它的口子**。

    换的是 ``axo.ui.voice`` 里那个名字（``DecodeWorker.__init__`` 就是在那儿取它的），
    所以线程、信号、队列那一整套还是真的——只把引擎那一段换成假的。
    """
    state: dict[str, Any] = {"problem": "", "text": ""}
    monkeypatch.setattr(
        voice_module,
        "Recognizer",
        lambda *_args, **_kwargs: FakeRecognizer(state["problem"], state["text"]),
    )
    return state


def test_the_mic_button_toggles_recording(qapp: Any, recognizer: dict[str, Any]) -> None:
    """按一下开始、再按一下结束——**同一个入口，看状态分岔**。

    **这一条是 2026-10-12 那个洞的回归测试**：``_begin_voice`` 原来只把状态推给了网页、
    漏了 ``self._voice_state``，于是按第二次时 ``_end_voice`` 读到空状态当场 return——
    录音不停、解码线程不收尾，用户看到的是按钮一直显示"正在听"而永远拿不到文字。

    它原来躲过了这条测试，因为测试是**手写** ``_voice_state = "listening"`` 的：照着
    "本该如此"写，而实现漏了一处。所以现在一个字都不手写——按两下，让代码自己去设。
    """
    stub = VoiceStub(streaming=False)

    # 第一下：没在录 → 开始录，而且**状态真的被设上了**
    stub._on_voice_toggled()
    assert stub._voice.starts == 1, "没在录，按一下应当开始录"
    assert stub._voice.stops == 0, "没在录却按成了停"
    assert stub._voice_state == "listening", "开始录之后状态没设上——按第二次会当场失灵"
    assert ("setVoiceState", "listening") in stub.chat_view.calls

    # 第二下：在录 → 停，并且走到解码线程的收尾
    stub._on_voice_toggled()
    assert stub._voice.stops == 1, "在录的时候按一下应当是停"
    assert stub._voice_state == "recognizing", "停下来了却没进入收尾"

    # 解码线程要有人收掉：`done` 一到它自己 quit（worker 的 finish 是空文本，所以
    # `_on_voice_done` 走的是"没听清"那一支，不碰 `_on_submit`）。
    assert _wait_until(qapp, lambda: not stub._live_threads), "解码线程没收尾"


def test_without_the_library_it_says_so_and_does_not_start(
    qapp: Any, recognizer: dict[str, Any]
) -> None:
    """库没装（或模型没放）→ **说出来，而且不开始录**。

    这一条在路上很要紧：先说再录，与"录三十秒之后再说没有模型"是两种体验——后者用户
    对着屏幕说了半天话，而它一个字都没记下。

    **问题那句话是注入的**（见上面那个夹具）：原来这一条跑的是真的 `DecodeWorker`，
    于是它验的那一支**取决于这台机器装没装模型**——装上了就变成另一条路，而它在此之前
    一直是绿的。
    """
    recognizer["problem"] = "还没有语音模型：把 zip 解压到某个目录就能用。"
    stub = VoiceStub(streaming=False)

    stub._begin_voice()

    told = [call for call in stub.chat_view.calls if call[0] == "appendMessage"]
    assert told and "语音输入用不了" in told[0][2]
    assert "还没有语音模型" in told[0][2], "那句话没被原样带出去"
    assert stub._voice.starts == 0, "说了用不了却还是开始录了"
    assert stub._voice_state == ""


def test_a_refused_microphone_also_says_so_and_cleans_up(
    qapp: Any, recognizer: dict[str, Any]
) -> None:
    """麦克风起不来（Windows 的隐私设置会整个拦下桌面应用）→ **说出来并收掉那一对**。

    这条路与上一条不同：识别器没问题，是 ``_voice.start`` 回了一句人话。那时已经建好了
    线程与 worker（`_track` 记着它们），**不收掉就会留下一个等不到的线程**——而它下一次
    出现在收尾的 `wait_for_all` 里，表现是退出时那三秒白等。
    """
    stub = VoiceStub(streaming=False)
    stub._voice.refuses = "拿不到麦克风。可能是 Windows 的「麦克风隐私设置」拦掉了。"

    stub._begin_voice()

    told = [call for call in stub.chat_view.calls if call[0] == "appendMessage"]
    assert told and "拿不到麦克风" in told[0][2]
    assert stub._voice_state == "", "起不来却把状态设成了在听"
    # **判据是"没有在跑的线程"，不是"清单空了"**：那一对是 `thread.quit()` 收掉的，
    # 而 `quit()` 对一个**从没 start 过**的线程不发 `finished`——于是那个 release 闭包
    # 不会跑，它会一直挂在 `_live_threads` 上。那不要紧（`wait_for_all` 按"在不在跑"
    # 逐个等，没在跑的立刻过），要紧的是**它真的没在跑**。
    assert not any(thread.isRunning() for thread in stub._live_threads), "留下的线程还在跑"


def test_the_recognized_text_goes_through_the_same_door_as_typing(qapp: Any) -> None:
    """识别出来的文字**走 `_on_submit`**——与打字完全同一条路。

    这条是这个功能的设计要点：进逐字稿、进记忆库、走路由、走闸门，一处都不绕。
    换一条"语音专用"的路的话，同一句话从嘴里说出来和从键盘敲出来结果不同——而那是
    最难向用户解释的一类怪。
    """
    stub = VoiceStub()
    stub._voice_state = "listening"

    stub._on_voice_done(Transcription(text="帮我记一下牛奶喝完了"))

    assert stub.submitted == ["帮我记一下牛奶喝完了"]
    assert stub._voice_state == ""
    assert ("setVoiceState", "") in stub.chat_view.calls


def test_nothing_heard_is_told_not_swallowed(qapp: Any) -> None:
    """没听清 → **说出来**，而且**不进** `_on_submit`（那会把一句空话发出去）。"""
    stub = VoiceStub()

    stub._on_voice_done(Transcription(reason="没有听清（离麦克风近一点）"))

    assert stub.submitted == []
    told = [call for call in stub.chat_view.calls if call[0] == "appendMessage"]
    assert told and "没有听清" in told[0][2]


def test_the_user_speaking_discards_the_recording(qapp: Any) -> None:
    """用户开口说了话 → **刚才那段语音不要了**（连结果一起）。

    继续跑的话，识别结果会在几秒后突然作为一条"新消息"发出去——而那时用户已经在说别的
    了。只丢结果不够：那段文字会从别的路冒出来。
    """
    stub = Stub()
    stub._voice.listening = True
    stub._voice_state = "listening"

    stub._on_submit("等一下")

    assert stub._voice.stops == 1, "用户说话了，录音没停"
    assert stub._voice_state == ""
    # 丢弃那一遍**不许闪「正在识别」**（它根本不会去识别），但要**回到空闲**：
    # 原来这里两样都没做，于是网页上那个按钮永远停在"正在听"——用户看着一个在听的
    # 按钮，而它早就没在听了。这条断言原来是 `("setVoiceState", "") not in ...`，
    # 那个写法把"网页那一半没跟上"当成了正确行为。
    assert ("setVoiceState", "recognizing") not in stub.chat_view.calls
    assert ("setVoiceState", "") in stub.chat_view.calls, "丢弃之后按钮没回到空闲"


def test_recording_that_hits_the_cap_still_gets_transcribed(qapp: Any) -> None:
    """录满上限自己停时，**照样走收尾**（那一句不该白说）。

    `_on_voice_finished` 是两种收场共用的那一个槽：用户按停会经由 `_end_voice` 走到它，
    而"录满三十秒"是它**自己**发出来的。前者那时状态已经改过（所以它放行），后者才刚要
    开始收尾——这一条验的就是后者的入口。
    """
    stub = Stub()
    stub._voice_state = "listening"

    stub._on_voice_finished()

    assert stub._voice.stops == 1
    assert stub._voice_state == "recognizing"
    assert ("setVoiceState", "recognizing") in stub.chat_view.calls


# ── 出声（TTS）：谁该出声、谁不该 ────────────────────────────────────────────
#
# 2026-10-13 语音对话闭环。出声的判定**只有一个**：这一轮是用户说出来的（`spoken`）。
# 五条判据各对着一类坏法：错出声（打字也念）、不出声（说出来的也不念）、念半句（被打断的）、
# 翻历史念（回看）、停不下来（让路漏了播放）。


def test_she_reads_her_reply_after_a_spoken_turn(qapp: Any) -> None:
    """用户**说**出来的那一轮：她的回复交给合成（出声的唯一入口）。"""
    stub = VoiceStub()
    stub._spoken_reply = True

    stub._on_reply_finished("记得买牛奶。", {"interrupted": False})

    # 合成在工作线程、落地是排队信号——**等两站都到**（同 `_speaking_stub` 那族的写法）。
    assert _wait_until(qapp, lambda: stub._speech_backend.asked == ["记得买牛奶。"]), (
        "说出来的那一轮没让合成上"
    )
    assert _wait_until(qapp, lambda: stub._speech_player.played == [b"WAV"]), "合成完了没开播"
    assert _wait_until(qapp, lambda: not stub._live_threads)


def test_a_typed_turn_stays_silent(qapp: Any) -> None:
    """**打字问的不出声**——这是"我说话她才开口"那条决定的另一半。

    反空转那一条：判定错成"每轮都念"也能过上一条测试，只有这条能拦住它。
    """
    stub = VoiceStub()
    stub._spoken_reply = False

    stub._on_reply_finished("记得买牛奶。", {"interrupted": False})

    assert stub._speech_backend.asked == [], "打字那轮也去合成了"
    assert stub._speech_player.played == []


def test_an_interrupted_or_empty_reply_is_not_read(qapp: Any) -> None:
    """打断的半句与空回复**都不念**：念一句空的是无声，念半句是念一个被打断的念头。"""
    stub = VoiceStub()
    stub._spoken_reply = True
    stub._on_reply_finished("（半句）", {"interrupted": True})
    stub._on_reply_finished("   ", {"interrupted": False})

    assert stub._speech_backend.asked == []


def test_the_reminder_line_is_read_too(qapp: Any) -> None:
    """到点那句**也念**——「到点由她开口」的本意就是开口（C4 只有文字是因为当时没 TTS）。"""
    stub, _backend = _speaking_stub(qapp, script=[[text_chunk("该睡了，早点休息。", done=True)]])

    stub._on_reminder_fired(reminder_alert("该睡了"))

    assert _wait_until(qapp, lambda: not stub._live_threads), "那一轮没跑完"
    assert stub._speech_backend.asked == ["该睡了，早点休息。"], "提醒句没让合成上"


def test_the_degraded_reminder_is_not_read(qapp: Any) -> None:
    """降级成**程序说**的那句不念：它是 `system` 气泡，不是"她说的话"——
    机器声念一句"本该由她开口"是二次羞辱。"""
    stub = VoiceStub()

    stub._tell_reminder(reminder_alert("该睡了"), note="模型没能开口")

    assert stub._speech_backend.asked == []


def test_stopping_speech_stops_playback_and_drops_late_synthesis(qapp: Any) -> None:
    """让路要收掉**两样**：正在播的，和还没合成完的。

    前者漏了 = 她说到一半停不下来（让路要治的正是这个病）；后者漏了 = 用户已经说别的了，
    她那句才姗姗来迟地响起来——像她插了一句。
    """
    stub = VoiceStub()
    stub._speech_player.played.append(b"EARLIER")  # 字节字面量只收 ASCII
    stub._tts_live = True

    stub._stop_speech(tell=False)

    assert stub._speech_player.stops >= 1, "让路没停播放"
    assert stub._tts_live is False, "迟到的合成还许落地"


def test_a_spoken_user_line_is_marked_when_read_back() -> None:
    """说出来的那句回看时带「语音说的」小标，**而且逐字稿往返丢不掉它**。

    小标是"这句话是说出来的"唯一的痕迹（音频不落盘）。丢法有两种都要堵：display_rows
    忘了映射、或者 Message 往返忘了写这个字段——后者是静默的（老行不受影响，所以
    schema 守卫看不出来）。
    """
    line = Message(role="user", content="帮我记一下", spoken=True)
    back = Message.from_dict(line.to_dict())

    assert back.spoken is True
    rows = display_rows([back])
    assert rows[0]["meta"]["title"] == "语音说的"

    typed = Message(role="user", content="帮我记一下", spoken=False)
    assert display_rows([typed])[0]["meta"] == {}, "打字那句不该带小标"
    assert "spoken" not in typed.to_dict(), "假字段也写了——老行的逐字节不变就没了"


# ── 播出回执：她说的那句出没出声，模型要知道 ────────────────────────────────
#
# 2026-10-13 模型自己提的需求（原话在提交信息里）：回执要**带时间戳**（分得出是哪一
# 句）、**失败优先**（卡住的那次才要紧）、**别做太重**（一行够判断）。进下一轮环境槽、
# 用完即清——不进逐字稿（它是关于播放的，不是对话）。


def test_a_played_sentence_leaves_a_receipt_for_the_next_turn(qapp: Any) -> None:
    """自然播完 → 回执 `ok`，**只在下一轮的环境槽里出现一次，然后清掉**。"""
    stub = VoiceStub()
    stub._speaking_text = "记得买牛奶。"
    stub._speech_player.played.append(b"WAV")

    stub._on_speech_played()

    line = stub._take_speech_receipt()
    assert "已播出" in line and "记得买牛奶" in line
    assert ":" in line or "）" in line, "回执要带时刻——分不出是哪一句的回执没有用"
    assert stub._take_speech_receipt() == "", "用完即清——同一条回执不许出现在每一轮"


def test_a_failed_playback_says_the_reason(qapp: Any) -> None:
    """合成失败/播放失败 → 回执带**原因**（顺利的回执用处有限，卡住的那次才要紧）。"""
    stub = VoiceStub()
    stub._speaking_text = "记得买牛奶。"
    stub._tts_live = True
    stub._tts_serial = 7

    stub._on_tts_done(SpeechResult(reason="语音组件没连上（http://127.0.0.1:8800）"), 7)

    line = stub._take_speech_receipt()
    assert "没播成" in line
    assert "语音组件没连上" in line


def test_an_interruption_is_also_a_receipt(qapp: Any) -> None:
    """打断也是「没播成」的一种——模型要能对上是哪一句没说完。"""
    stub = VoiceStub()
    stub._speaking_text = "她说的上一句。"
    stub._speech_player.played.append(b"WAV")  # 占位：playing 判定看它
    stub._speech_player.stop = lambda: None  # 不真停，保住 playing 供判定

    class _Playing:
        playing = True

        def stop(self) -> None:
            pass

    stub._speech_player = _Playing()
    stub._stop_speech(tell=False)

    line = stub._take_speech_receipt()
    assert "没播成" in line and "打断" in line


# ── 主动搭话：她先开口的另一条路 ────────────────────────────────────────────


def _memory_item(
    memory_id: str = "mA",
    content: str = "感冒那位回来了。",
    *,
    minutes_ago: float = 60.0,
    session: str = "S-then",
) -> Any:
    """一条可当话头的记忆。默认写在**别的会话**里（本会话的要被排掉，见下）。"""
    from memcore.sdr import encode_lexical
    from memcore.types import MemoryItem, MemoryKind, Tier

    return MemoryItem(
        memory_id=memory_id,
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content=content,
        created_at=time.time() - minutes_ago * 60,
        session_id=session,
        lex_sdr=encode_lexical(content),
    )


def _idle_since(stub: Stub, minutes: float) -> None:
    """把"他最后说话"摆到 N 分钟前（主动搭话只看这一条与 ``_last_speech_at``）。"""
    stub._session.messages = [
        Message(role="user", content="帮我看看笔记", at=time.time() - minutes * 60)
    ]


def _proactive_stub(qapp: Any, **script: Any) -> tuple[Stub, ScriptedBackend]:
    """一个能主动开口的替身：有后端、有库、他静了 25 分钟、库里有话头。"""
    backend = ScriptedBackend(
        script=script.pop("script", [[text_chunk("你上次说的那把扳手，借到了吗？", done=True)]]),
        **script,
    )
    stub = Stub(streaming=False, backend=backend, memory=MemoryStore(":memory:"))
    _idle_since(stub, 25)
    stub._memory.add(_memory_item())
    return stub, backend


def test_she_speaks_proactively_when_everything_is_ready(qapp: Any) -> None:
    """条件齐了：她拿一条旧记忆起头，进逐字稿、画进对话区、**念出声**。

    与提醒那条的三处不同都在这条断言里：回看小标是「她主动说的」（不是「提醒」）、
    **不叫醒窗口**、计数与去重在**真正开口时**才落下（查完保持沉默的那次不算）。
    """
    stub, backend = _proactive_stub(qapp)

    stub._on_proactive_tick()

    assert _wait_until(qapp, lambda: not stub._live_threads), "那一轮没跑完"
    assert backend.calls == 1
    _kind, role, text, meta = stub.chat_view.calls[-1]
    assert role == "proactive"
    assert meta == {"title": "她主动说的"}
    assert len(stub.stored) == 1
    line = stub.stored[0]
    assert (line.role, line.proactive) == ("assistant", True)
    assert stub.remembered == [], "她主动说的那句进了记忆库"
    assert stub._tray.messages == [], "主动搭话不该动托盘气泡（那是提醒那条通道）"
    # 计数与去重**真正开口时**落下；沉默（[SILENCE]）的那次不算。
    assert (stub._spoken_today, stub._used_materials) == (1, ["mA"])


def test_the_material_rides_in_the_user_line(qapp: Any) -> None:
    """素材进的是 ``user`` 侧的事件行，而不是 system —— 理由见 `agent/proactive.py`。"""
    stub, backend = _proactive_stub(qapp)

    stub._on_proactive_tick()

    assert _wait_until(qapp, lambda: not stub._live_threads)
    request = backend.requests[0]
    assert [row.role for row in request.messages] == ["system", "user"]
    assert "你记得这件事" in request.messages[1].content
    assert "感冒那位回来了。" in request.messages[1].content
    assert request.messages[0].content.count("你记得这件事") == 0


def test_she_never_starts_when_the_switch_is_off(qapp: Any) -> None:
    stub, backend = _proactive_stub(qapp)
    stub._proactive_settings = ProactiveSettings(
        enabled=False, quiet_from=0, quiet_to=0, allow_tools=False
    )

    stub._on_proactive_tick()

    assert backend.calls == 0
    assert stub.stored == [] and stub._speak_material is None


def test_quiet_hours_keep_her_quiet_every_hour_of_the_day(qapp: Any) -> None:
    """免打扰那一闸的**接线**：把时段设成"整天"（0 → 24），她一次都不开口。

    墙上时钟无关：纯层那张表已经逐格钉过边界，这里要证的只是"设置真的传到了判定里"。
    """
    stub, backend = _proactive_stub(qapp)
    stub._proactive_settings = ProactiveSettings(quiet_from=0, quiet_to=24, allow_tools=False)

    stub._on_proactive_tick()

    assert backend.calls == 0


def test_a_recent_message_keeps_her_quiet(qapp: Any) -> None:
    stub, backend = _proactive_stub(qapp)
    _idle_since(stub, 5)

    stub._on_proactive_tick()

    assert backend.calls == 0


def test_her_own_speech_resets_the_idle_clock(qapp: Any) -> None:
    """她刚说完一句，**"他静下来了"就不成立**——否则那是唠叨的形状。"""
    stub, backend = _proactive_stub(qapp)
    stub._last_speech_at = datetime.now()

    stub._on_proactive_tick()

    assert backend.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("_backend", None), ("_streaming", True), ("_speak_live", True)],
    ids=["没有后端", "他正问到一半", "她已经在说"],
)
def test_the_machine_gates_keep_her_quiet(qapp: Any, field: str, value: Any) -> None:
    """这三条不是"时机"问题，是"这一轮做不成"——同一个 tick 里现问。"""
    stub, backend = _proactive_stub(qapp)
    setattr(stub, field, value)

    stub._on_proactive_tick()

    assert backend.calls == 0
    assert stub.stored == [] and stub._speak_material is None


def test_without_a_material_she_stays_quiet(qapp: Any) -> None:
    """一条候选都没有 → **不开口**，不硬找一个话头。"""
    backend = ScriptedBackend(script=[[text_chunk("好。", done=True)]])
    stub = Stub(streaming=False, backend=backend, memory=MemoryStore(":memory:"))
    _idle_since(stub, 25)

    stub._on_proactive_tick()

    assert backend.calls == 0


def test_what_she_said_this_session_is_not_a_topic(qapp: Any) -> None:
    """本会话写下的不当话头——"刚说过"不是"她还记得"。"""
    backend = ScriptedBackend(script=[[text_chunk("好。", done=True)]])
    stub = Stub(streaming=False, backend=backend, memory=MemoryStore(":memory:"))
    _idle_since(stub, 25)
    stub._memory.add(_memory_item(session=stub._session.session_id))

    stub._on_proactive_tick()

    assert backend.calls == 0


def test_a_failed_proactive_speech_says_nothing_at_all(qapp: Any) -> None:
    """**没有降级。** 提醒那条失败时由程序代念；主动搭话没有那样的正文。

    由程序说一句"我想跟你说话"是荒谬的，所以失败之后**一个字都不该出现**——不是
    不重试，是整件事就这么过去了。
    """
    stub, backend = _proactive_stub(qapp, script=[[]], raise_at=0)

    stub._on_proactive_tick()

    assert _wait_until(qapp, lambda: not stub._live_threads)
    assert backend.calls == 1, "失败之后又试了一次"
    assert stub.stored == []
    assert all(call[1] != "system" for call in stub.chat_view.calls), "程序替她开口了"


def test_an_empty_proactive_reply_also_says_nothing(qapp: Any) -> None:
    stub, _backend = _proactive_stub(qapp, script=[[text_chunk("   ", done=True)]])

    stub._on_proactive_tick()

    assert _wait_until(qapp, lambda: not stub._live_threads)
    assert stub.stored == []
    assert all(call[1] != "system" for call in stub.chat_view.calls)


def test_a_late_proactive_result_is_dropped_like_any_other(qapp: Any) -> None:
    """让路之后迟到的结果不许落地——主动与提醒共用同一条收尾，所以这条也要成立。"""
    stub, _backend = _proactive_stub(qapp)
    stub._on_proactive_tick()
    assert _wait_until(qapp, lambda: not stub._live_threads)
    before = len(stub.stored)

    stub._speak_live = False  # 让路
    stub._on_speech_finished("迟到的一句。", {})

    assert len(stub.stored) == before


def test_the_day_counter_rolls_over(qapp: Any) -> None:
    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    stub._proactive_day = date(2026, 10, 14)
    stub._spoken_today = 3
    stub._used_materials = ["a", "b"]

    stub._roll_over_if_new_day(datetime(2026, 10, 15, 9, 0))

    assert stub._proactive_day == date(2026, 10, 15)
    assert (stub._spoken_today, stub._used_materials) == (0, [])


def test_the_timeout_does_not_make_the_program_speak_for_her(qapp: Any) -> None:
    """超时那条分支：提醒会降级成程序说，主动搭话不会。"""
    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    worker = FakeSpeechWorker()
    stub._speak_worker = worker
    stub._speak_live = True
    stub._speak_material = Material("mA", "感冒那位回来了。", time.time())

    stub._time_out_speech()

    assert worker.stops == 1
    assert all(call[1] != "system" for call in stub.chat_view.calls)
    assert stub._speak_material is None and stub._speak_alert is None


# ── 节拍器 ─────────────────────────────────────────────────────────────────


def test_the_scheduler_ticks_and_stops(qapp: Any) -> None:
    scheduler = ProactiveScheduler(tick_ms=1)
    seen: list[int] = []
    scheduler.tick.connect(lambda: seen.append(1))

    assert not scheduler.running
    scheduler.start()
    assert scheduler.running
    scheduler.start()  # 幂等：重复起不会把周期重启一遍
    assert _wait_until(qapp, lambda: len(seen) >= 2), "表没走"

    scheduler.stop()
    scheduler.stop()  # 幂等
    assert not scheduler.running


def test_the_tick_carries_no_payload(qapp: Any) -> None:
    """载荷为空：判定与取材都不在节拍器里，所以它只有"时间到了"这一件事可说。

    用 ``lambda *args`` 接：它什么都收得下，于是"收到几个"成了一个能断言的事实。
    """
    scheduler = ProactiveScheduler(tick_ms=1)
    seen: list[tuple[Any, ...]] = []
    scheduler.tick.connect(lambda *args: seen.append(args))

    scheduler.start()
    assert _wait_until(qapp, lambda: bool(seen)), "表没走"
    scheduler.stop()

    assert seen and all(args == () for args in seen), f"信号带了载荷：{seen[:2]}"


# ── 接线守卫（AST） ────────────────────────────────────────────────────────


def _attribute_calls(name: str) -> set[str]:
    """``MainWindow`` 某个方法里**代码上**出现过的 ``self.xxx`` 名字。

    **走 AST 而不是在源码文本里搜字符串**：文档字符串里解释"这里为什么不调
    ``_tell_reminder``"是应该的，而它不该让守卫变红——第一次写这条守卫就是这么红的。
    """
    import ast
    import importlib

    module = importlib.import_module("axo.ui.main_window")
    source = Path(module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
    raise AssertionError(f"没找到 MainWindow.{name}")


def test_the_proactive_path_has_no_fallback_by_construction() -> None:
    """**没有降级是设计，不是漏写。**

    提醒那条说不出来时由程序代念（``_tell_reminder``）——那条气泡的正文本来就是提醒。
    主动搭话没有那样的正文，所以从取话头到起那一轮，整条路上不许出现 ``_tell_reminder``。
    这条守卫盯的是"谁调它"：纯层全绿不等于接线在，而接线在也不等于**只**接了该接的。
    """
    for name in ("_on_proactive_tick", "_start_proactive_speech"):
        assert "_tell_reminder" not in _attribute_calls(name), f"{name} 里出现了降级"


def test_she_really_is_read_aloud_on_this_path_too() -> None:
    """出声那条接线的守卫：主动搭话与提醒共用收尾，而收尾里确实调了 ``_speak_reply``。

    "用户说话她才开口"那条规则说的是**打字回合不念**；她主动开口属于"开口"。
    """
    assert "_speak_reply" in _attribute_calls("_on_speech_finished")


def test_the_table_is_actually_started_and_stopped() -> None:
    """**表起没起**是一条静默失效的接线：不起，她一次都不会主动开口，而界面上没有任何
    异常（连日志都没有）。

    这一层测试碰不到真的 ``__init__``（构造 ``MainWindow`` 会连 Ollama、起线程、加载
    WebEngine，见本文件开头那段），所以退一步钉源码：构造、起、停三处都在。
    """
    import ast
    import importlib

    module = importlib.import_module("axo.ui.main_window")
    source = Path(module.__file__).read_text(encoding="utf-8")
    window = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    # **只取 MainWindow 自己的**：``ast.walk`` 会把嵌套类里同名的 ``__init__`` 也收进来，
    # 而它后访问到、会把真的那一份盖掉——那种守卫会在某次重构之后悄悄失效。
    bodies = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in window.body
        if isinstance(node, ast.FunctionDef)
    }

    assert "ProactiveScheduler(" in bodies["__init__"], "没建那块表"
    assert "self._proactive.tick.connect(" in bodies["__init__"], "建了但没接 tick"
    assert "self._proactive.start()" in bodies["__init__"], "建了但没起表"
    assert "self._proactive.stop()" in bodies["_shutdown"], "收尾没停表"


# ── 表情包：她发的那张图 ────────────────────────────────────────────────────


def test_a_sticker_lands_as_her_own_row(qapp: Any, tmp_path: Path) -> None:
    """她发一张表情包：落成**她自己的一行**，不是一张工具卡片。

    工具卡片的标题会写着「工具结果 · send_sticker」，而用户看到的是一张图挂在一个
    技术名词下面——而那张图是她说的话，不是她查到的结果。
    """
    picture = tmp_path / "开心.png"
    picture.write_bytes(bytes([137, 80, 78, 71]))
    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))

    stub._on_tool_finished({"name": "send_sticker", "ok": True, "preview": str(picture)})

    _kind, role, text, meta = stub.chat_view.calls[-1]
    assert role == "assistant"
    assert text == ""
    assert meta["image"].startswith("file:") and meta["image"].endswith(".png")
    assert len(stub.stored) == 1
    assert stub.stored[0].content == ""
    assert stub.stored[0].image.endswith("开心.png")


def test_a_normal_tool_result_is_not_treated_as_a_sticker(qapp: Any) -> None:
    """``preview`` 为空时照旧画工具卡片——这条判据是"有没有那张图"，不是工具名。"""
    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))

    stub._on_tool_finished({"name": "send_sticker", "ok": True, "content": "已发送。"})

    _kind, role, _text, meta = stub.chat_view.calls[-1]
    assert role == "tool"
    assert "image" not in meta
    assert stub.stored == []


def test_display_rows_hands_the_image_to_the_web_view() -> None:
    """回看时那一行带的是 ``file:`` URL——与当时看到的是同一张。

    逐字稿里存的是**相对数据根的路径**，所以这里要能看到它被还原成一个绝对 URL。
    """
    rows = display_rows([Message(role="assistant", content="", image="stickers/开心.png")])

    assert rows[0]["meta"]["image"].startswith("file:")
    assert rows[0]["meta"]["image"].endswith(".png")


def test_display_rows_leaves_a_text_row_alone() -> None:
    """没有图的行**一个键都不多**——老历史的长相不变。"""
    rows = display_rows([Message(role="assistant", content="好。")])

    assert rows[0]["meta"] == {}


# ── 拖进来的文件 ────────────────────────────────────────────────────────────


def _drop_stub(qapp: Any, *, cloud: bool = False, can_see: bool = True) -> Stub:
    """一个能收附件的替身。

    后端只用 ``SimpleNamespace``：``_attach_pending_images`` 只读 ``images_ride_on``
    这一个属性，为它造一个假后端类不值得。
    """
    from types import SimpleNamespace

    stub = Stub(
        streaming=False,
        backend=SimpleNamespace(images_ride_on="user"),
        memory=MemoryStore(":memory:"),
    )
    stub._backend_settings = SimpleNamespace(is_cloud=cloud)
    stub._model_can_see = lambda: can_see  # type: ignore[method-assign]
    return stub


def test_dropping_a_file_holds_it_for_the_next_message(qapp: Any, tmp_path: Path) -> None:
    """拖进来**不立刻发**：多半还要配一句话（"帮我看看这个"）。"""
    note = tmp_path / "笔记.md"
    note.write_text("路由器在书房。", encoding="utf-8")
    stub = _drop_stub(qapp)

    stub._on_attach_files([str(note)])

    assert [item.name for item in stub._pending_attachments] == ["笔记.md"]
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "笔记.md" in text and "下一条消息" in text


def test_the_cloud_gets_a_warning_before_a_file_goes(qapp: Any, tmp_path: Path) -> None:
    """用云端模型时**拖之前就说清内容会发出去**——与截图那次同一个姿态。"""
    note = tmp_path / "笔记.md"
    note.write_text("x", encoding="utf-8")
    stub = _drop_stub(qapp, cloud=True)

    stub._on_attach_files([str(note)])

    assert "云端" in stub.chat_view.calls[-1][2]


def test_a_dropped_thing_that_is_not_a_file_says_so(qapp: Any, tmp_path: Path) -> None:
    """拖一个目录进来（很常见）不该静默——也不该炸。"""
    stub = _drop_stub(qapp)

    stub._on_attach_files([str(tmp_path), str(tmp_path / "没有这个文件.txt")])

    assert stub._pending_attachments == ()
    assert "读不了" in stub.chat_view.calls[-1][2]


def test_an_attached_image_rides_on_the_user_row(qapp: Any, tmp_path: Path) -> None:
    """附件图贴到**本轮那条 user 行**上——它没有工具结果那一行可挂。"""
    from axo.session.assemble import assemble
    from axo.session.budget import TokenBudget

    stub = _drop_stub(qapp)
    stub._pending_images = ("data:image/png;base64,AAAA",)
    assembly = assemble([], current="看看这张。", budget=TokenBudget(window=8192))

    attached = stub._attach_pending_images(assembly)

    assert attached.messages[-1].images == ("data:image/png;base64,AAAA",)
    assert stub._pending_images == (), "贴完要清——它只活在这一轮"


def test_an_image_the_model_cannot_see_says_why(qapp: Any) -> None:
    """看不见图的模型：**说清原因**（同 ``_attach_pending_audio`` 那条纪律）。"""
    from axo.session.assemble import assemble
    from axo.session.budget import TokenBudget

    stub = _drop_stub(qapp, can_see=False)
    stub._pending_images = ("data:image/png;base64,AAAA",)
    assembly = assemble([], current="看看这张。", budget=TokenBudget(window=8192))

    attached = stub._attach_pending_images(assembly)

    assert attached.messages[-1].images == ()
    _kind, role, text, _meta = stub.chat_view.calls[-1]
    assert role == "system" and "没有发出去" in text


def test_a_backend_that_hangs_images_elsewhere_says_it_is_us_not_them(qapp: Any) -> None:
    """本机那条链把图挂在工具结果行上——**那是「我们还没接过去」，不是「对面不支持」**。

    说成后者的代价是用户去反复换模型，而换哪个都一样（那条纪律是上一批用真事换来的）。
    """
    from types import SimpleNamespace

    from axo.session.assemble import assemble
    from axo.session.budget import TokenBudget

    stub = Stub(
        streaming=False,
        backend=SimpleNamespace(images_ride_on="tool"),
        memory=MemoryStore(":memory:"),
    )
    stub._backend_settings = SimpleNamespace(is_cloud=False)
    stub._model_can_see = lambda: True  # type: ignore[method-assign]
    stub._pending_images = ("data:image/png;base64,AAAA",)
    assembly = assemble([], current="看看这张。", budget=TokenBudget(window=8192))

    stub._attach_pending_images(assembly)

    text = stub.chat_view.calls[-1][2]
    assert "还没接过去" in text
    assert "不支持" not in text


def test_a_file_uri_becomes_a_path_and_walks_the_same_door(qapp: Any, tmp_path: Path) -> None:
    """网页送来的是 ``file:`` URI，主窗口解成路径——**下游只有一条路**（_on_attach_files）。"""
    note = tmp_path / "笔记.md"
    note.write_text("路由器在书房。", encoding="utf-8")
    stub = _drop_stub(qapp)

    stub._on_files_dropped([note.as_uri()])

    assert [item.name for item in stub._pending_attachments] == ["笔记.md"]


def test_a_uri_that_is_not_a_file_is_dropped_here_too(qapp: Any) -> None:
    """网页那边滤过一道，这里是第二道——**桥是公开给网页的**，而"只信网页送来的
    东西"这条纪律在别处也写着（拖一段网页上的图片进来，URI 里是一个 http 地址）。"""
    stub = _drop_stub(qapp)

    stub._on_files_dropped(["https://example.com/a.png", "javascript:alert(1)"])

    assert stub._pending_attachments == ()
    assert "读不了" in stub.chat_view.calls[-1][2]


# ── 插件 ────────────────────────────────────────────────────────────────────


def test_a_broken_plugin_is_told_once_not_every_round(qapp: Any) -> None:
    """装不上的插件要说一句（否则用户只能自己猜），但**同一个错只说一次**。

    ``_tools()`` 每轮重建，所以每轮都会看到同一个错——每轮都说一遍是噪音，而噪音的
    结局是用户开始忽略这一整类消息（同提醒那条"通知风暴"的教训）。
    """
    from axo.plugins import PluginIssue

    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    stub._plugin_issues = set()

    stub._report_plugin_issues([PluginIssue("坏的", "RuntimeError: boom")])

    assert "坏的" in stub.chat_view.calls[-1][2]
    told = len(stub.chat_view.calls)
    stub._report_plugin_issues([PluginIssue("坏的", "RuntimeError: boom")])
    assert len(stub.chat_view.calls) == told, "同一句话被说了第二遍"


def test_a_new_plugin_problem_is_still_told(qapp: Any) -> None:
    """记住的是「说过什么」，不是「说过了没有」——新出现的问题照样要报。"""
    from axo.plugins import PluginIssue

    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    stub._plugin_issues = {"坏的：RuntimeError: boom"}

    stub._report_plugin_issues(
        [PluginIssue("坏的", "RuntimeError: boom"), PluginIssue("新的", "没有 plugin.py")]
    )

    text = stub.chat_view.calls[-1][2]
    assert "新的" in text and "坏的" not in text


def test_the_tool_table_asks_the_plugin_loader(qapp: Any, monkeypatch: Any) -> None:
    """**接线守卫**：工具表真的去插件目录装了一趟，并且收下了它给的工具。

    纯层全绿不等于接线在——这条纪律在这一批里已经应验过两次。所以这里不验"插件能不能
    装"（那是 ``tests/test_plugins.py`` 的事），只验**它问过没有、收没收下**：替掉加载器，
    看它被调用的参数与结果去了哪。
    """
    from types import SimpleNamespace

    import axo.ui.main_window as window_module
    from axo.plugins import PluginLoad

    seen: list[tuple[Any, tuple[str, ...]]] = []
    plugin_tool = SimpleNamespace(
        spec=SimpleNamespace(name="weather", description="d", parameters={}),
        requires_approval=False,
        run=lambda *_: None,
    )

    def fake_load(root: Any, *, taken: Sequence[str] = ()) -> PluginLoad:
        seen.append((root, tuple(taken)))
        return PluginLoad(tools=(plugin_tool,), issues=())

    monkeypatch.setattr(window_module, "load_plugins", fake_load)
    monkeypatch.setattr(window_module, "plugins_dir", lambda: Path("插件目录"))
    monkeypatch.setattr(window_module, "stickers_dir", lambda: Path("没有表情包"))

    stub = Stub(streaming=False, backend=object(), memory=MemoryStore(":memory:"))
    stub._read_roots = ()
    stub._web_backend = lambda: None
    stub._plugin_issues = set()
    stub._model_can_see = lambda: False  # type: ignore[method-assign]

    registry = stub._tools()

    assert len(seen) == 1, "工具表没去装插件"
    root, taken = seen[0]
    assert str(root) == "插件目录"
    assert "get_time" in taken, "内置工具的名字要先占上——撞名的插件工具会被挡掉"
    assert "weather" in [spec.name for spec in registry.specs()]
    assert stub.chat_view.calls == [], "装上了就不该报问题"


# ── 通话开关 ────────────────────────────────────────────────────────────────


def test_the_call_button_starts_and_hangs_up(qapp: Any) -> None:
    """一个按钮两件事；第二次按就是挂断。"""
    stub = Stub(streaming=False)

    stub._on_call_toggled()
    assert stub._call.live is True
    assert stub._call.started == 1

    stub._on_call_toggled()
    assert stub._call.live is False
    assert stub._call.stopped == 1


def test_a_generating_turn_refuses_to_start_a_call(qapp: Any) -> None:
    """正在生成时开通话会把两轮搅在一起——先说完再开始。"""
    stub = Stub(streaming=True)

    stub._on_call_toggled()

    assert stub._call.live is False
    assert stub._call.started == 0


def test_a_receipt_is_only_consumed_after_the_turn_really_ran(qapp: Any) -> None:
    """**peek/commit**：组装失败或请求没发出去时，回执不能凭空消失。"""
    import time as _time

    stub = Stub(streaming=False)
    stub._speech_receipt = SpeechReceipt(at=_time.time(), text="你好", ok=True)

    first = stub._peek_speech_receipt()
    assert first and stub._peek_speech_receipt() == first, "peek 不许清"

    stub._commit_speech_receipt()
    assert stub._peek_speech_receipt() == ""


def test_a_proactive_turn_can_check_tools_and_stay_silent(qapp: Any) -> None:
    """允许她查之后：主动回合走工具循环，查完回 ``[SILENCE]`` 就什么都不发生。"""
    from axo.agent.catalog import GetTimeTool
    from axo.agent.tools import ToolRegistry

    stub, backend = _proactive_stub(
        qapp,
        script=[
            [tool_chunk(ToolCall(id="c1", name="get_time"))],
            [text_chunk("[SILENCE]", done=True)],
        ],
    )
    stub._proactive_settings = ProactiveSettings(
        quiet_from=0, quiet_to=0, allow_tools=True, weather_city="杭州"
    )
    stub._tools = lambda: ToolRegistry([GetTimeTool()])
    stub._writer = None
    stub._reminders = None
    stub._reminders_fired = None
    stub._reminders_cancelled = None
    stub._read_roots = ()
    stub._web_backend = lambda: None

    stub._on_proactive_tick()

    assert _wait_until(qapp, lambda: not stub._live_threads), "那一轮没跑完"
    assert backend.calls == 2, "先走了一轮工具，再走收尾"
    assert "杭州" in backend.requests[0].messages[0].content
    assert "[SILENCE]" in backend.requests[0].messages[0].content
    assert stub.stored == [], "沉默不该进逐字稿"
    assert (stub._spoken_today, stub._used_materials) == (0, []), "沉默不该消耗次数"
