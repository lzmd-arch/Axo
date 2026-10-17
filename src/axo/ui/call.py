"""通话的接线层：把状态机、采集、识别与界面的信号接起来。**不做判断。**

判断在 :mod:`axo.call`（纯层，可以逐格断言）；这里只负责"什么时候开麦、什么时候
关麦、把哪条信号交给谁"。这一层刻意薄，是因为它的每一行都要碰 Qt 的线程与设备，
而那一类代码在 CI 上只能靠替身验证——薄一点，替身才盖得住。

## 半双工

* ``LISTENING`` / ``CAPTURING``：麦克风开；
* ``THINKING`` / ``SPEAKING``：麦克风关；
* 插话 = ``interrupt()``：先叫停播放，再开麦。Qt 没有回声消除，这是唯一确定的
  做法（见 :mod:`axo.call` 的模块文档）。

## 识别线程只起一次

``SegmentWorker`` 在通话开始时起一次，整场复用：模型只加载一次，麦克风开开关关
不影响它。挂断时才 ``stop`` 与 ``quit``。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..call import DEFAULT_IDLE_HANGUP_S, CallSession, CallState
from .thread_tracking import track, wait_for_all
from .voice import SegmentWorker, VoiceRecorder

__all__ = ["CallController"]

#: 电平最多多久推一次。网页那边一次推送就是一次 ``runJavaScript``，50 Hz 太吵。
_LEVEL_INTERVAL_S = 0.1


class CallController(QObject):
    """一次通话的控制器。

    Signals:
        state_changed: ``(状态字符串, 附加信息)``。附加信息目前只有 ``muted`` 与
            ``detail``，给状态条用。
        level_changed: 0–100 输入电平（已合帧到 10 Hz）。
        utterance: 一段转写（``Transcription``）。**调用方负责提交**。
        problem: 麦克风/模型不可用。通话已自行收掉。
        interrupt_requested: 用户在 ``SPEAKING`` 时点了插话；调用方应停播。
    """

    state_changed = Signal(str, dict)
    level_changed = Signal(int)
    utterance = Signal(object)
    problem = Signal(str)
    interrupt_requested = Signal()

    def __init__(
        self,
        *,
        idle_hangup_s: float = DEFAULT_IDLE_HANGUP_S,
        recorder: Any | None = None,
        worker_factory: Callable[[], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = CallSession()
        self._idle_hangup_s = idle_hangup_s
        self._recorder = (
            recorder if recorder is not None else VoiceRecorder(max_seconds=None, parent=self)
        )
        self._worker_factory = worker_factory or (lambda: SegmentWorker(one_shot=False))
        self._worker: Any = None
        self._thread: QThread | None = None
        #: ``track`` 的契约（见 ``ui/thread_tracking.py``）。
        self._live_threads: list[QThread] = []
        self._last_level_at = 0.0
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick)

    # ── 查询 ────────────────────────────────────────────────────────────────

    @property
    def live(self) -> bool:
        return self._session.live

    @property
    def state(self) -> CallState:
        return self._session.state

    @property
    def muted(self) -> bool:
        return self._session.muted

    # ── 开/关 ───────────────────────────────────────────────────────────────

    def start(self) -> str:
        """开始通话。**空串 = 成了**，否则是一句可以直接显示的人话。"""
        if self.live:
            return ""
        worker = self._worker_factory()
        problem = getattr(getattr(worker, "recognizer", None), "problem", "")
        if problem:
            return problem
        thread = QThread(self)
        worker.utterances.connect(self._on_utterance)
        worker.problem.connect(self._on_worker_problem)
        worker.level.connect(self._on_level)
        worker.speech_started.connect(self._on_speech_started)
        worker.speech_ended.connect(self._on_speech_ended)
        worker.finished.connect(thread.quit)
        track(self, thread, worker, what="通话识别线程", slot="call")
        self._worker = worker
        thread.start()

        self._session.start(time.monotonic())
        self._ticker.start()
        note = self._listen()
        if note:
            self.stop(discard=True)
            return note
        self._emit_state()
        return ""

    def stop(self, *, discard: bool = True) -> None:
        """挂断。**幂等**；先关麦、再停线程（同 ``_shutdown`` 的纪律）。"""
        if not self.live:
            return
        self._ticker.stop()
        self._pause()
        if self._worker is not None:
            self._worker.stop(discard=discard)
        if self._thread is not None:
            self._thread.quit()
        self._session.finish(time.monotonic())
        self._emit_state()

    def shutdown(self) -> None:
        """退出程序：挂断并**等识别线程停下**（不等的话就是下一次 ``0xC0000409``）。"""
        self.stop(discard=True)
        wait_for_all(self._live_threads)

    # ── 用户动作 ────────────────────────────────────────────────────────────

    def set_muted(self, muted: bool) -> None:
        if not self.live:
            return
        self._session.set_muted(bool(muted), time.monotonic())
        if muted:
            self._pause()
        else:
            note = self._listen()
            if note:
                self.problem.emit(note)
        self._emit_state()

    def interrupt(self) -> None:
        """用户在 ``SPEAKING`` 时插话：叫停播放并开麦。"""
        if not self.live:
            return
        if self._session.state is CallState.SPEAKING:
            self.interrupt_requested.emit()
            self._session.move(CallState.LISTENING, time.monotonic())
            note = self._listen()
            if note:
                self.problem.emit(note)
            self._emit_state()
            return
        # 不在说话时点一下等于"确保在听"（例如看门狗复位之后）。
        if self._session.state is CallState.LISTENING and not self._recorder.listening:
            note = self._listen()
            if note:
                self.problem.emit(note)
            self._emit_state()

    # ── 一轮的收尾 ──────────────────────────────────────────────────────────

    def on_user_turn_started(self) -> None:
        """用户这一轮开始了（识别出来的、或者打字的）：**关麦，等她回答**。"""
        if not self.live:
            return
        if self._session.state in (CallState.LISTENING, CallState.CAPTURING):
            self._pause()
            self._session.move(CallState.THINKING, time.monotonic())
            self._emit_state()

    def on_playback_started(self) -> None:
        """第一句开始播了：进入 ``SPEAKING``（即使回复还在生成）。"""
        if self.live and self._session.state is CallState.THINKING:
            self._session.move(CallState.SPEAKING, time.monotonic())
            self._emit_state()

    def on_reply_finished(self, *, will_speak: bool) -> None:
        """回复生成完了：接着说话，或者直接回到聆听。"""
        if not self.live:
            return
        now = time.monotonic()
        target = CallState.SPEAKING if will_speak else CallState.LISTENING
        if not self._session.move(target, now):
            # 例如用户在她"想"的时候点了插话，状态已经是 LISTENING。
            self._session.move(CallState.LISTENING, now)
        if not will_speak:
            note = self._listen()
            if note:
                self.problem.emit(note)
        self._emit_state()

    def on_playback_finished(self) -> None:
        """她说完（或合成失败）：回到聆听。"""
        if not self.live:
            return
        if self._session.state is CallState.SPEAKING:
            self._session.move(CallState.LISTENING, time.monotonic())
            note = self._listen()
            if note:
                self.problem.emit(note)
            self._emit_state()

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _listen(self) -> str:
        if not self._session.can_listen():
            return ""
        if self._recorder.listening:
            return ""
        return self._recorder.start(self._worker.push)

    def _pause(self) -> None:
        if self._recorder.listening:
            self._recorder.stop()

    def _on_utterance(self, result: Any) -> None:
        if not self.live:
            return
        # **先关麦再提交**：她要想、要说，而这期间开着的麦克风会把她的声音录进来。
        self._pause()
        self._session.move(CallState.THINKING, time.monotonic())
        self._emit_state()
        self.utterance.emit(result)

    def _on_speech_started(self) -> None:
        if self._session.state is CallState.LISTENING:
            self._session.move(CallState.CAPTURING, time.monotonic())
            self._emit_state()

    def _on_speech_ended(self) -> None:
        if self._session.state is CallState.CAPTURING:
            self._session.move(CallState.LISTENING, time.monotonic())
            self._emit_state()

    def _on_worker_problem(self, problem: str) -> None:
        self.problem.emit(problem)
        self.stop(discard=True)

    def _on_level(self, level: int) -> None:
        now = time.monotonic()
        if now - self._last_level_at < _LEVEL_INTERVAL_S:
            return
        self._last_level_at = now
        self.level_changed.emit(int(level))

    def _tick(self) -> None:
        verdict = self._session.tick(time.monotonic(), idle_hangup_s=self._idle_hangup_s)
        if verdict == "idle":
            self.stop(discard=True)
            return
        if verdict == "watchdog":
            now = time.monotonic()
            if self._session.move(CallState.LISTENING, now):
                note = self._listen()
                if note:
                    self.problem.emit(note)
                self.state_changed.emit(CallState.LISTENING.value, {"detail": "状态已复位"})

    def _emit_state(self) -> None:
        detail = ""
        if self._session.state is CallState.LISTENING and self._session.muted:
            detail = "已静音"
        self.state_changed.emit(
            self._session.state.value,
            {"muted": self._session.muted, "detail": detail},
        )
