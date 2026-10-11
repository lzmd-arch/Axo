"""ULID 生成。

主键即排序键是本项目的一个结构性假设（``iter_items`` 直接 ``ORDER BY
memory_id``，关联图的节点也用它）。这个文件钉住的就是那条假设：**字典序等于
时间序**，包括同一毫秒内与系统时钟被校准的那一刻。
"""

from __future__ import annotations

import pytest

from memcore.ids import ULID_LENGTH, is_ulid, new_ulid, ulid_timestamp_ms


def test_shape() -> None:
    value = new_ulid()
    assert len(value) == ULID_LENGTH == 26
    assert value == value.upper()
    assert is_ulid(value)


def test_lexicographic_order_matches_time_order() -> None:
    """这是 :class:`memcore.types.MemoryItem.memory_id` 被当作排序键的前提。"""
    ids = [new_ulid() for _ in range(500)]
    assert ids == sorted(ids), "生成的 ULID 序列不是严格递增的"


def test_same_millisecond_still_strictly_increasing() -> None:
    """同一毫秒内必须严格递增，不能靠随机部分决定先后。

    500 次连调几乎必然落在同一毫秒里；若随机部分不回绕递增，这里就会乱序。
    """
    at = 1_700_000_000.0
    ids = [new_ulid(at=at) for _ in range(500)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids), "同一毫秒内出现了重复 id"


def test_clock_going_backwards_keeps_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """系统时钟被往回调（NTP 校准、用户手改）时 id 仍须单调。

    不处理的话，"id 序 == 时间序"会在时钟校准的那一刻断裂，而断裂之后写入的
    记忆会排到已有记忆的前面。

    **必须走"此刻"那条路径**（``at=None``）才测得到这个保护。用 ``at=`` 测不到
    ——显式时间戳被刻意豁免了保护，理由见 :func:`memcore.ids.new_ulid`。
    """
    clock = iter([1_700_000_100.0, 1_699_999_000.0, 1_699_999_001.0])  # 往回调了 110 秒
    monkeypatch.setattr("memcore.ids.time.time", lambda: next(clock))

    first = new_ulid()
    second = new_ulid()
    third = new_ulid()
    assert first < second < third, "时钟倒流后 id 不再单调"


def test_timestamp_round_trip() -> None:
    """显式传入的时间戳必须**原样**出现在 id 里。

    这条曾经失败过：时钟倒流保护把 9 年前的 ``at`` 顶成了"当前时间 + 1ms"，于是
    回填历史记录与测试都无法指定时间。一个被静默改写的显式参数比报错难查得多。
    """
    at = 1_700_000_000.0
    value = new_ulid(at=at)
    assert ulid_timestamp_ms(value) == 1_700_000_000_000


def test_explicit_timestamps_may_go_backwards() -> None:
    """显式回填允许产生比已有 id 更早的 id——那正是"回填"的含义。"""
    new_ulid(at=1_700_000_100.0)
    backfilled = new_ulid(at=1_699_999_000.0)
    assert ulid_timestamp_ms(backfilled) == 1_699_999_000_000


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-ulid",
        "0" * 25,  # 太短
        "0" * 27,  # 太长
        "0" * 25 + "I",  # Crockford 字母表里没有 I
        "0" * 25 + "L",
        "0" * 25 + "U",
        "8" + "0" * 25,  # 时间戳超出 48 位（首位最大只能是 7）
        None,
        12345,
    ],
)
def test_invalid_shapes_rejected(value: object) -> None:
    assert not is_ulid(value)


def test_uniqueness_at_scale() -> None:
    ids = {new_ulid() for _ in range(5000)}
    assert len(ids) == 5000


def test_ulid_timestamp_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="太短"):
        ulid_timestamp_ms("ABC")
