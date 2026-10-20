"""音乐窗口：搜歌、看歌单、播放控制与歌词。**只显示与转发，不拥有播放器。**

播放器归 :class:`~axo.ui.music_service.MusicService`：关掉这个窗口音乐继续放，
退出程序才停。窗口自己不起线程，``closeEvent`` 里等的是服务交过来的那批
（目前没有，但保留同一形状，免得下次加线程时忘掉）。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..music.library import format_duration, read_cover
from ..music.models import LibraryIndex, Lyrics, PlayerState, Track
from .glass import GlassHost, fade_window
from .thread_tracking import wait_for_all
from .window_place import place_window

__all__ = ["MusicWindow"]


class MusicWindow(QMainWindow):
    settings_requested = Signal()
    volume_changed = Signal(int)

    def __init__(self, service: Any, *, accent: str = "", motion: bool = True) -> None:
        super().__init__()
        self._service = service
        self._accent = accent or "#6b8afd"
        self._motion = bool(motion)
        self._live_threads: list[Any] = []
        self._tracks: tuple[Track, ...] = ()
        self._current: Track | None = None
        self._lyrics: Lyrics = Lyrics()
        self._dragging = False
        self._pending_volume = 0
        self.setWindowTitle("音乐")
        place_window(self)
        self._build_ui()
        self._wire()
        self._glass = GlassHost(self, accent=self._accent, animated=self._motion)
        index = getattr(service, "index", None)
        if isinstance(index, LibraryIndex):
            self._on_library(index)
        state = service.state()
        self._on_state(state)

    # ── 界面 ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("musicRoot")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        header = QHBoxLayout()
        self.search_edit = QLineEdit(root)
        self.search_edit.setObjectName("musicSearch")
        self.search_edit.setPlaceholderText("搜歌名、艺术家或专辑…")
        header.addWidget(self.search_edit, 1)
        self.scan_button = QPushButton("重新扫描", root)
        header.addWidget(self.scan_button)
        self.folder_button = QPushButton("打开文件夹", root)
        header.addWidget(self.folder_button)
        self.settings_button = QPushButton("设置", root)
        header.addWidget(self.settings_button)
        outer.addLayout(header)

        self.status_label = QLabel("", root)
        self.status_label.setObjectName("musicStatus")
        outer.addWidget(self.status_label)

        split = QSplitter(Qt.Orientation.Horizontal, root)
        outer.addWidget(split, 1)

        left = QWidget(split)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.playlist_combo = QComboBox(left)
        self.playlist_combo.addItem("全部曲目", "")
        left_layout.addWidget(self.playlist_combo)
        self.track_table = QTableWidget(0, 4, left)
        self.track_table.setHorizontalHeaderLabels(["标题", "艺术家", "专辑", "时长"])
        self.track_table.verticalHeader().setVisible(False)
        self.track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.track_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.track_table.horizontalHeader().setStretchLastSection(False)
        self.track_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(self.track_table, 1)
        split.addWidget(left)

        right = QWidget(split)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.cover = QLabel(right)
        self.cover.setObjectName("musicCover")
        self.cover.setFixedSize(160, 160)
        self.cover.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.cover)
        meta = QVBoxLayout()
        self.title_label = QLabel("还没有在播放", right)
        self.title_label.setObjectName("musicTitle")
        self.title_label.setWordWrap(True)
        self.artist_label = QLabel("", right)
        self.artist_label.setObjectName("musicArtist")
        self.album_label = QLabel("", right)
        self.album_label.setObjectName("musicAlbum")
        meta.addWidget(self.title_label)
        meta.addWidget(self.artist_label)
        meta.addWidget(self.album_label)
        meta.addStretch(1)
        top.addLayout(meta, 1)
        right_layout.addLayout(top)
        self.lyrics_view = QListWidget(right)
        self.lyrics_view.setObjectName("musicLyrics")
        right_layout.addWidget(self.lyrics_view, 1)
        split.addWidget(right)
        split.setSizes([560, 520])

        transport = QHBoxLayout()
        self.prev_button = QPushButton("⏮", root)
        self.play_button = QPushButton("▶", root)
        self.stop_button = QPushButton("⏹", root)
        self.next_button = QPushButton("⏭", root)
        self.shuffle_button = QPushButton("随机", root)
        self.shuffle_button.setCheckable(True)
        self.repeat_button = QPushButton("顺序", root)
        for button in (
            self.prev_button,
            self.play_button,
            self.stop_button,
            self.next_button,
            self.shuffle_button,
            self.repeat_button,
        ):
            transport.addWidget(button)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal, root)
        self.seek_slider.setRange(0, 0)
        transport.addWidget(self.seek_slider, 1)
        self.time_label = QLabel("0:00 / 0:00", root)
        transport.addWidget(self.time_label)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal, root)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(110)
        transport.addWidget(self.volume_slider)
        outer.addLayout(transport)
        self.setCentralWidget(root)

    def _wire(self) -> None:
        self.search_edit.textChanged.connect(self._apply_filter)
        self.scan_button.clicked.connect(self._service.scan)
        self.folder_button.clicked.connect(self._open_folder)
        self.settings_button.clicked.connect(self.settings_requested)
        self.playlist_combo.currentIndexChanged.connect(self._on_playlist_changed)
        self.track_table.cellDoubleClicked.connect(self._play_row)
        self.prev_button.clicked.connect(lambda: self._control("prev"))
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button.clicked.connect(lambda: self._control("stop"))
        self.next_button.clicked.connect(lambda: self._control("next"))
        self.shuffle_button.toggled.connect(lambda _on: self._control("shuffle"))
        self.repeat_button.clicked.connect(lambda: self._control("repeat"))
        self.seek_slider.sliderPressed.connect(self._start_drag)
        self.seek_slider.sliderReleased.connect(self._finish_drag)
        self.volume_slider.valueChanged.connect(self._on_volume)
        self.volume_slider.sliderReleased.connect(self._emit_volume)
        self._service.library_changed.connect(self._on_library)
        self._service.scan_started.connect(lambda: self.status_label.setText("正在扫描音乐库…"))
        self._service.scan_progress.connect(
            lambda done, total: self.status_label.setText(f"正在扫描：{done}/{total}")
        )
        self._service.scan_finished.connect(self._on_scan_finished)
        self._service.scan_failed.connect(lambda reason: self.status_label.setText(reason))
        self._service.state_changed.connect(self._on_state)
        self._service.current_changed.connect(self._on_current)
        self._service.lyrics_changed.connect(self._on_lyrics)
        self._service.notice.connect(self.status_label.setText)
        self.volume_slider.setValue(int(self._service.state().volume))

    # ── 外部 ────────────────────────────────────────────────────────────────

    def set_glass(self, accent: str, *, motion: bool) -> None:
        self._accent = accent
        self._motion = bool(motion)
        glass = getattr(self, "_glass", None)
        if glass is not None:
            glass.set_theme(accent, animated=motion)

    def showEvent(self, event: Any) -> None:  # noqa: N802 — Qt 命名
        super().showEvent(event)
        if self._motion:
            fade_window(self)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt 命名
        wait_for_all(list(self._live_threads))
        super().closeEvent(event)

    def refresh(self) -> None:
        self._on_library(self._service.index)
        self._on_state(self._service.state())

    # ── 事件 ────────────────────────────────────────────────────────────────

    def _on_library(self, index: LibraryIndex) -> None:
        self._tracks = index.tracks
        self.playlist_combo.blockSignals(True)
        current = str(self.playlist_combo.currentData() or "")
        self.playlist_combo.clear()
        self.playlist_combo.addItem("全部曲目", "")
        for playlist in index.playlists:
            self.playlist_combo.addItem(
                f"{playlist.name}（{len(playlist.track_ids)}）", playlist.name
            )
        position = self.playlist_combo.findData(current)
        self.playlist_combo.setCurrentIndex(max(0, position))
        self.playlist_combo.blockSignals(False)
        self._apply_filter()

    def _on_scan_finished(self, index: LibraryIndex) -> None:
        self.status_label.setText(f"音乐库：{len(index.tracks)} 首")

    def _on_playlist_changed(self) -> None:
        self._apply_filter()

    def _visible_tracks(self) -> tuple[Track, ...]:
        playlist = str(self.playlist_combo.currentData() or "")
        if playlist:
            chosen = next(
                (one for one in self._service.index.playlists if one.name == playlist), None
            )
            ids = set(chosen.track_ids) if chosen is not None else set()
            base = tuple(track for track in self._tracks if track.track_id in ids)
        else:
            base = self._tracks
        query = self.search_edit.text().strip()
        if not query:
            return base
        return (
            self._service.search(query, limit=500)
            if not playlist
            else tuple(
                track
                for track in base
                if query.casefold() in f"{track.display} {track.album}".casefold()
            )
        )

    def _apply_filter(self) -> None:
        tracks = self._visible_tracks()
        if len(tracks) > 2000:
            # QTableWidget 不是虚拟列表；一次塞几万行会把界面卡住几秒。
            self.status_label.setText(f"曲目太多，只显示前 2000 首（共 {len(tracks)} 首）")
            tracks = tracks[:2000]
        self.track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            values = (track.title, track.artist, track.album, format_duration(track.duration_s))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, track.track_id)
                self.track_table.setItem(row, column, item)

    def _play_row(self, row: int, _column: int) -> None:
        item = self.track_table.item(row, 0)
        if item is None:
            return
        try:
            self.status_label.setText(
                self._service.play(track_id=str(item.data(Qt.ItemDataRole.UserRole)))
            )
        except Exception as exc:  # noqa: BLE001 — 服务的失败是一句人话
            self.status_label.setText(str(exc))

    def _toggle_play(self) -> None:
        state = self._service.state()
        if state.track is None:
            tracks = self._visible_tracks()
            if tracks:
                self._play_row(0, 0)
            return
        self._control("playpause")

    def _control(self, action: str, value: Any = None) -> None:
        try:
            self.status_label.setText(self._service.control(action, value))
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(str(exc))

    def _on_state(self, state: PlayerState) -> None:
        if state.track is not None:
            self._on_current(state.track)
        self.play_button.setText("⏸" if state.playing else "▶")
        self.shuffle_button.blockSignals(True)
        self.shuffle_button.setChecked(state.shuffle)
        self.shuffle_button.blockSignals(False)
        self.repeat_button.setText(
            {"off": "顺序", "all": "列表循环", "one": "单曲循环"}[state.repeat.value]
        )
        duration = max(0, int(round(state.duration_s)))
        if not self._dragging:
            self.seek_slider.setRange(0, duration)
            self.seek_slider.setValue(min(duration, int(round(state.position_s))))
        self.time_label.setText(
            f"{format_duration(state.position_s)} / {format_duration(state.duration_s)}"
        )
        self._highlight_lyrics(state.position_s)

    def _on_current(self, track: Track) -> None:
        if track.track_id == getattr(self._current, "track_id", ""):
            return
        self._current = track
        self.title_label.setText(track.title)
        self.artist_label.setText(track.artist)
        self.album_label.setText(track.album)
        self._set_cover(track)

    def _set_cover(self, track: Track) -> None:
        data = read_cover(track.path)
        pixmap = QPixmap()
        if data and pixmap.loadFromData(data):
            self.cover.setPixmap(
                pixmap.scaled(
                    160,
                    160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.cover.setText("♪")

    def _on_lyrics(self, lyrics: Lyrics) -> None:
        self._lyrics = lyrics
        self.lyrics_view.clear()
        if not lyrics.ok:
            self.lyrics_view.addItem(QListWidgetItem("（没有歌词）"))
            return
        for line in lyrics.lines:
            self.lyrics_view.addItem(QListWidgetItem(line.text))
        self.lyrics_view.setCurrentRow(-1)

    def _highlight_lyrics(self, position_s: float) -> None:
        if not self._lyrics.ok or not self._lyrics.synced:
            return
        active = -1
        for index, line in enumerate(self._lyrics.lines):
            if line.at_s <= position_s + 0.25:
                active = index
            else:
                break
        if active != self.lyrics_view.currentRow():
            self.lyrics_view.setCurrentRow(active)
            if active >= 0:
                item = self.lyrics_view.item(active)
                if item is not None:
                    self.lyrics_view.scrollToItem(
                        item, QAbstractItemView.ScrollHint.PositionAtCenter
                    )

    def _start_drag(self) -> None:
        self._dragging = True

    def _finish_drag(self) -> None:
        self._dragging = False
        self._control("seek", float(self.seek_slider.value()))

    def _on_volume(self, value: int) -> None:
        """拖动时立即改音量；**落盘只在松手时**（``_emit_volume``）。

        每一格都写一次 QSettings 的话，一次拖动会在注册表里留下几十次写入。
        """
        self._pending_volume = int(value)
        if not getattr(self, "_volume_ready", False):
            self._volume_ready = True
            return
        with contextlib.suppress(Exception):
            self._service.control("volume", int(value))

    def _emit_volume(self) -> None:
        self.volume_changed.emit(int(self._pending_volume))

    def _open_folder(self) -> None:
        root = self._service.settings.library_root
        if root is None:
            self.settings_requested.emit()
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(root))))
