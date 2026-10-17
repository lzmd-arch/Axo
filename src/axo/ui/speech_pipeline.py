"""边生成边念：一句完整了就合成、就播，不等整段回复写完。

现在的出声路径是"整段写完 → 整段合成 → 整段播放"，首字延迟等于**整段**合成时间。
电话式的体验等不了：一句话说完，下一句就该在播了。

这里做三件事：

1. **分句**交给纯层 :mod:`axo.speech_text`（代码不念、超长在逗号处断）；
2. **串行合成、预取一句**：同一时刻只有一路合成（组件第一次要加载模型，并发只会
   更慢）；播放第 N 句时，第 N+1 句已经在合成；
3. **一条回复只产生一条回执**：多句各写一条会互相覆盖（``_speech_receipt`` 是单个
   槽），而"她刚才那句出没出声"是关于整条回复的一句话。

取消（插话/挂断）时清空队列、停播、丢掉在飞的合成结果——迟到的 `done` 由序号挡掉。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..speech_text import split_sentences
from .speech_player import SpeechPlayer, TtsWorker
from .thread_tracking import track, wait_for_all

__all__ = ["SpeechPipeline"]


class SpeechPipeline(QObject):
    """把一段流式文本变成一句句音频。

    Signals:
        started: 第一句开始播。
        finished: 所有句子都播完。
        failed: 合成或播放失败（一句人话）。**之后不会再出声。**
    """

    started = Signal()
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        backend: Any,
        player: SpeechPlayer,
        *,
        voice: str = "",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._player = player
        self._voice = voice
        self._buffer = ""
        self._queue: list[str] = []
        self._wav: bytes | None = None
        self._playing = False
        self._synth = False
        self._finished_input = False
        self._done = False
        self._announced = False
        self._serial = 0
        self._live_threads: list[QThread] = []
        self._player.finished.connect(self._on_played)

    def reconfigure(self, backend: Any, *, voice: str = "") -> None:
        """设置页改了组件地址/音色：下一句开始用新的。**正在播的那句不重来。**"""
        self._backend = backend
        self._voice = voice

    @property
    def active(self) -> bool:
        """还有东西没播完。"""
        return self._playing or self._synth or self._wav is not None or bool(self._queue)

    # ── 流式入口 ────────────────────────────────────────────────────────────

    def feed(self, delta: str) -> None:
        """喂一段增量文本。完整句子立刻入队。"""
        if not delta or self._done:
            return
        self._buffer += delta
        sentences, self._buffer = split_sentences(self._buffer, final=False)
        self._enqueue(sentences)

    def finish(self) -> None:
        """文本结束：把尾巴也交出去。**之后可以再开一轮**（先 :meth:`reset`）。"""
        if self._done:
            return
        sentences, self._buffer = split_sentences(self._buffer, final=True)
        self._enqueue(sentences)
        self._finished_input = True
        self._pump()

    def speak(self, text: str) -> None:
        """整段说。``feed`` + ``finish`` 的便利入口。"""
        self.feed(text)
        self.finish()

    def reset(self) -> None:
        """准备下一轮。**不动正在播的东西**（那是 :meth:`cancel` 的事）。"""
        self._buffer = ""
        self._queue.clear()
        self._wav = None
        self._finished_input = False
        self._done = False
        self._announced = False

    def cancel(self) -> None:
        """插话/挂断：停播、清空、丢掉在飞的合成。"""
        self._serial += 1
        self._playing = False
        self._synth = False
        self._wav = None
        self._queue.clear()
        self._buffer = ""
        self._finished_input = False
        self._done = True
        self._player.stop()
        worker = getattr(self, "_pipeline_worker", None)
        if worker is not None:
            worker.stop()
        thread = getattr(self, "_pipeline_thread", None)
        if thread is not None:
            thread.quit()

    def shutdown(self) -> None:
        self.cancel()
        wait_for_all(self._live_threads)

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _enqueue(self, sentences: list[str]) -> None:
        for sentence in sentences:
            if sentence.strip():
                self._queue.append(sentence.strip())
        self._pump()

    def _pump(self) -> None:
        if self._done:
            return
        if self._playing:
            # 播第 N 句时预取第 N+1 句。**串行**合成：组件并发只会互相拖慢。
            if self._wav is None and not self._synth and self._queue:
                self._synth_one()
            return
        if self._wav is not None:
            self._play()
            return
        if not self._synth:
            if self._queue:
                self._synth_one()
                return
            if self._finished_input:
                self._done = True
                self.finished.emit()

    def _synth_one(self) -> None:
        text = self._queue.pop(0)
        thread = QThread(self)
        worker = TtsWorker(self._backend, text, self._voice)
        self._synth = True
        self._serial += 1
        serial = self._serial
        worker.done.connect(lambda result, serial=serial: self._on_synth(result, serial))
        worker.done.connect(thread.quit)
        track(self, thread, worker, what="语音合成线程", slot="pipeline")
        thread.start()

    @Slot(object)
    def _on_synth(self, result: Any, serial: int) -> None:
        if serial != self._serial or self._done:
            return
        self._synth = False
        if not getattr(result, "ok", False):
            self._fail(getattr(result, "reason", "") or "语音组件没有返回可播放的音频")
            return
        self._wav = result.wav
        self._pump()

    def _play(self) -> None:
        wav, self._wav = self._wav, None
        problem = self._player.speak(wav)
        if problem:
            self._fail(problem)
            return
        self._playing = True
        if not self._announced:
            self._announced = True
            self.started.emit()
        # 开播之后立刻预取下一句：**播放与合成并行**才是分句流式的意义。
        self._pump()

    @Slot()
    def _on_played(self) -> None:
        if self._done:
            return
        self._playing = False
        self._pump()

    def _fail(self, reason: str) -> None:
        if self._done:
            return
        self._done = True
        self._playing = False
        self._synth = False
        self._wav = None
        self._queue.clear()
        self._buffer = ""
        self.failed.emit(reason)
