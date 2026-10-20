"""Qt 播放器：``QMediaPlayer`` + ``QAudioOutput`` 的薄包装。**只在 UI 线程用。**

为什么不用语音那条 ``QAudioSink``：那条链吃的是整段 PCM，而音乐要的是文件解码、
拖动进度与"播完下一首"；``QMediaPlayer`` 自带这些，打断延迟在这里也不是立论。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from ..music.models import PlayerState, Track

__all__ = ["MusicPlayer"]


class MusicPlayer(QObject):
    state_changed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, parent: QObject | None = None, *, volume: int = 80) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._track: Track | None = None
        self._volume = max(0, min(100, int(volume)))
        self._audio.setVolume(self._volume / 100.0)
        self._error = ""
        self._player.positionChanged.connect(lambda _value: self._emit())
        self._player.durationChanged.connect(lambda _value: self._emit())
        self._player.playbackStateChanged.connect(lambda _state: self._emit())
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.errorOccurred.connect(self._on_error)

    # ── 状态 ────────────────────────────────────────────────────────────────

    @property
    def track(self) -> Track | None:
        return self._track

    @property
    def position_s(self) -> float:
        return self._player.position() / 1000.0

    @property
    def duration_s(self) -> float:
        duration = self._player.duration() / 1000.0
        if duration > 0:
            return duration
        return self._track.duration_s if self._track is not None else 0.0

    @property
    def playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def paused(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PausedState

    @property
    def volume(self) -> int:
        return self._volume

    def state(self) -> PlayerState:
        return PlayerState(
            track=self._track,
            playing=self.playing,
            paused=self.paused,
            position_s=self.position_s,
            duration_s=self.duration_s,
            volume=self._volume,
            error=self._error,
        )

    # ── 动作 ────────────────────────────────────────────────────────────────

    def play(self, track: Track, *, position_s: float = 0.0) -> None:
        self._track = track
        self._error = ""
        url = QUrl.fromLocalFile(str(Path(track.path)))
        self._player.setSource(url)
        if position_s > 0:
            self._player.setPosition(int(position_s * 1000))
        self._player.play()
        self._emit()

    def pause(self) -> None:
        self._player.pause()
        self._emit()

    def resume(self) -> None:
        self._player.play()
        self._emit()

    def stop(self) -> None:
        self._player.stop()
        self._emit()

    def seek(self, position_s: float) -> None:
        self._player.setPosition(max(0, int(position_s * 1000)))
        self._emit()

    def set_volume(self, volume: int) -> None:
        self._volume = max(0, min(100, int(volume)))
        self._audio.setVolume(self._volume / 100.0)
        self._emit()

    def set_muted(self, muted: bool) -> None:
        self._audio.setMuted(bool(muted))
        self._emit()

    def shutdown(self) -> None:
        """退出前调：停播并断开输出。**不删对象**——Qt 自己收。"""
        self._player.stop()
        self._player.setSource(QUrl())

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _emit(self) -> None:
        self.state_changed.emit(self.state())

    def _on_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.finished.emit()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._error = "这个文件播不了（格式或编码不支持）"
            self.failed.emit(self._error)
            self._emit()

    def _on_error(self, _error: QMediaPlayer.Error, message: str) -> None:
        self._error = (message or "").strip() or "播放失败"
        self.failed.emit(self._error)
        self._emit()

    def describe(self) -> dict[str, Any]:
        """给工具面的一行状态；避免把整个 Qt 枚举交给模型。"""
        current = self._track
        return {
            "playing": self.playing,
            "paused": self.paused,
            "position_s": round(self.position_s, 1),
            "duration_s": round(self.duration_s, 1),
            "volume": self._volume,
            "track": current.display if current is not None else "",
            "error": self._error,
        }
