"""播放队列：顺序、随机、单曲循环/列表循环。**纯逻辑，不碰音频。**

窗口与工具面共用这一份顺序，所以"下一首是什么"只有一个事实源。随机顺序用
``random.Random`` 记录种子——测试里给固定种子就能逐条断言，运行时用系统熵。
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from .models import RepeatMode, Track

__all__ = ["PlayQueue"]


class PlayQueue:
    def __init__(self, *, seed: int | None = None) -> None:
        self._tracks: list[Track] = []
        self._order: list[int] = []
        self._pos = 0
        self._shuffle = False
        self._repeat = RepeatMode.OFF
        self._rng = random.Random(seed)

    # ── 读取 ────────────────────────────────────────────────────────────────

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    @property
    def length(self) -> int:
        return len(self._order)

    @property
    def index(self) -> int:
        return self._pos if self._order else -1

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def repeat(self) -> RepeatMode:
        return self._repeat

    def current(self) -> Track | None:
        if not self._order:
            return None
        return self._tracks[self._order[self._pos]]

    def upcoming(self, limit: int = 5) -> tuple[Track, ...]:
        if not self._order or limit <= 0:
            return ()
        start = min(self._pos + 1, len(self._order))
        return tuple(self._tracks[self._order[one]] for one in self._order[start : start + limit])

    def by_id(self, track_id: str) -> Track | None:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    # ── 装载 ────────────────────────────────────────────────────────────────

    def set_tracks(self, tracks: Sequence[Track], *, start_id: str = "") -> Track | None:
        self._tracks = list(tracks)
        if not self._tracks:
            self._order = []
            self._pos = 0
            return None
        start = 0
        if start_id:
            for index, track in enumerate(self._tracks):
                if track.track_id == start_id:
                    start = index
                    break
        self._order = list(range(len(self._tracks)))
        if self._shuffle:
            rest = [one for one in self._order if one != start]
            self._rng.shuffle(rest)
            self._order = [start, *rest]
            self._pos = 0
        else:
            self._pos = start
        return self.current()

    # ── 移动 ────────────────────────────────────────────────────────────────

    def next(self, *, auto: bool = False) -> Track | None:
        """下一首。``auto=True`` 表示"当前这首自然播完"，单曲循环只在这一支生效。"""
        if not self._order:
            return None
        if auto and self._repeat is RepeatMode.ONE:
            return self.current()
        if self._pos + 1 < len(self._order):
            self._pos += 1
            return self.current()
        if self._repeat is RepeatMode.ALL:
            first = self._order[0]
            if self._shuffle:
                self._rng.shuffle(self._order)
                first = self._order[0]
            self._pos = 0
            self._order = [first, *[one for one in self._order if one != first]]
            return self.current()
        return None

    def prev(self) -> Track | None:
        if not self._order:
            return None
        if self._pos > 0:
            self._pos -= 1
        elif self._repeat is RepeatMode.ALL:
            self._pos = len(self._order) - 1
        return self.current()

    def jump(self, track_id: str) -> Track | None:
        for position, index in enumerate(self._order):
            if self._tracks[index].track_id == track_id:
                self._pos = position
                return self.current()
        return None

    # ── 开关 ────────────────────────────────────────────────────────────────

    def set_shuffle(self, enabled: bool) -> None:
        if enabled == self._shuffle:
            return
        current = self.current()
        self._shuffle = enabled
        if not self._order:
            return
        current_index = self._order[self._pos]
        if enabled:
            rest = [one for one in self._order if one != current_index]
            self._rng.shuffle(rest)
            self._order = [current_index, *rest]
            self._pos = 0
        else:
            self._order = sorted(self._order)
            self._pos = self._order.index(current_index)
        if current is None:  # pragma: no cover — 空队列上面已经返回
            self._pos = 0

    def set_repeat(self, mode: RepeatMode | str) -> None:
        self._repeat = mode if isinstance(mode, RepeatMode) else RepeatMode(str(mode))

    def cycle_repeat(self) -> RepeatMode:
        order = (RepeatMode.OFF, RepeatMode.ALL, RepeatMode.ONE)
        self._repeat = order[(order.index(self._repeat) + 1) % len(order)]
        return self._repeat

    def remove(self, track_id: str) -> Track | None:
        """摘掉一条；当前项还在就留在原地，摘掉的正是当前项就顺延到下一首。"""
        current = self.current()
        kept = [track for track in self._tracks if track.track_id != track_id]
        if len(kept) == len(self._tracks):
            return current
        start = ""
        if current is not None and current.track_id != track_id:
            start = current.track_id
        return self.set_tracks(kept, start_id=start)

    def drop_missing(self, valid_ids: Iterable[str]) -> None:
        """扫描后曲目消失：把它从队列里摘掉，当前项尽量留在原位。"""
        valid = set(valid_ids)
        current = self.current()
        kept = [track for track in self._tracks if track.track_id in valid]
        kept_ids = {track.track_id for track in kept}
        start_id = current.track_id if current is not None and current.track_id in kept_ids else ""
        self.set_tracks(kept, start_id=start_id)
