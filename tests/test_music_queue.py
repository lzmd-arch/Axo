"""播放队列：顺序、随机、循环、跳转与文件消失。**不碰音频。**"""

from __future__ import annotations

from pathlib import Path

from axo.music.models import RepeatMode, Track
from axo.music.queue import PlayQueue


def _tracks(count: int = 4) -> tuple[Track, ...]:
    return tuple(
        Track(str(index), Path(f"/music/{index}.mp3"), f"歌{index}") for index in range(count)
    )


def test_next_and_prev_walk_in_order() -> None:
    queue = PlayQueue(seed=1)
    first = queue.set_tracks(_tracks())
    assert first is not None and first.track_id == "0"
    assert [queue.next().track_id for _ in range(3)] == ["1", "2", "3"]  # type: ignore[union-attr]
    assert queue.next() is None
    assert queue.prev().track_id == "2"  # type: ignore[union-attr]


def test_repeat_one_only_applies_to_auto_advance() -> None:
    queue = PlayQueue(seed=1)
    queue.set_tracks(_tracks(2))
    queue.set_repeat(RepeatMode.ONE)
    assert queue.next(auto=True).track_id == "0"  # type: ignore[union-attr]
    assert queue.next(auto=False).track_id == "1"  # type: ignore[union-attr]


def test_repeat_all_wraps() -> None:
    queue = PlayQueue(seed=1)
    queue.set_tracks(_tracks(2))
    queue.set_repeat(RepeatMode.ALL)
    assert queue.next().track_id == "1"  # type: ignore[union-attr]
    assert queue.next().track_id == "0"  # type: ignore[union-attr]


def test_shuffle_keeps_current_first_and_is_reproducible() -> None:
    one = PlayQueue(seed=7)
    one.set_tracks(_tracks(6), start_id="3")
    one.set_shuffle(True)
    order_one = [one.current().track_id] + [one.next().track_id for _ in range(5)]  # type: ignore[union-attr]
    two = PlayQueue(seed=7)
    two.set_tracks(_tracks(6), start_id="3")
    two.set_shuffle(True)
    order_two = [two.current().track_id] + [two.next().track_id for _ in range(5)]  # type: ignore[union-attr]
    assert order_one == order_two
    assert order_one[0] == "3"


def test_jump_and_cycle_repeat() -> None:
    queue = PlayQueue()
    queue.set_tracks(_tracks(3))
    assert queue.jump("2").track_id == "2"  # type: ignore[union-attr]
    assert queue.cycle_repeat() is RepeatMode.ALL
    assert queue.cycle_repeat() is RepeatMode.ONE
    assert queue.cycle_repeat() is RepeatMode.OFF


def test_drop_missing_keeps_current_when_it_survives() -> None:
    queue = PlayQueue()
    queue.set_tracks(_tracks(3), start_id="1")
    queue.drop_missing({"0", "1"})
    assert queue.current().track_id == "1"  # type: ignore[union-attr]
    assert queue.length == 2
    queue.drop_missing({"0"})
    assert queue.current().track_id == "0"  # type: ignore[union-attr]
    queue.drop_missing(set())
    assert queue.current() is None
