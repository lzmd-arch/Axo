"""记忆 id 的生成：ULID。

**为什么不用 uuid4**：ULID 的前 48 位是毫秒时间戳，按字典序排列就等于按时间
排列。本项目里这一个性质同时撑起三件事——``memories`` 的主键即排序键（
:meth:`memcore.store.MemoryStore.iter_items` 直接 ``ORDER BY memory_id``）、
关联图的节点 id 天然有序、以及"这条记忆比那条早"不需要额外查一列。

**同一毫秒内必须严格递增。** 纯随机的话，同一毫秒里写入的两条记忆谁先谁后是
随机的，于是"按 id 排序"与"按时间排序"在毫秒边界上不一致——而排序的一致性
是上面那三件事的前提。做法沿用 ULID 规范的建议：时间戳相同时，把随机部分当
80 位整数加一。

Crockford base32（去掉 I/L/O/U 四个易混字母），26 个字符，字典序与数值序一致。
"""

from __future__ import annotations

import os
import threading
import time

__all__ = ["ULID_LENGTH", "is_ulid", "new_ulid", "ulid_timestamp_ms"]

#: Crockford base32 字母表。**顺序不能动**：字符的序号就是它编码的数值，换顺序
#: 会让所有既有 id 的解析结果改变。也正因为如此，它的字典序 == 数值序。
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {char: value for value, char in enumerate(_ALPHABET)}

ULID_LENGTH = 26
_TIME_CHARS = 10  # 48 位毫秒时间戳
_RANDOM_CHARS = 16  # 80 位随机
_RANDOM_BITS = 80  # 16 字符 × 5 位

_MAX_TIMESTAMP_MS = (1 << 48) - 1
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1

# 同毫秒递增的状态。加锁的理由：写入发生在界面线程，回填发生在工作线程，
# 两者都可能在同一毫秒里新建记忆——而"读随机数 → 加一 → 写回"不是原子的。
_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def new_ulid(at: float | None = None) -> str:
    """生成一个新的 ULID。

    Args:
        at: Unix 秒。**原样使用，不做任何"修顺"**——它的用途正是回填历史记录
            与在测试里造确定的序列。传了它就不再跑时钟倒流保护（见下），但仍
            保证同一毫秒内的严格递增。

    "时钟倒流保护"只对 ``at=None``（也就是"给我一个此刻的 id"）生效：那才是
    "新写入的记忆必须排在已有记忆之后"这条要求的适用场景。若把保护也套到显式
    时间戳上，``new_ulid(at=九年前)`` 会被静默地顶到当前时间——**一个显式传入
    的参数被悄悄改掉，比直接报错难查得多**。
    """
    global _last_ms, _last_random

    explicit = at is not None
    millis = int((time.time() if at is None else at) * 1000.0)
    millis = max(0, min(millis, _MAX_TIMESTAMP_MS))

    with _lock:
        if millis == _last_ms:
            if _last_random >= _MAX_RANDOM:
                # 一毫秒内用光了 80 位随机数——现实中不可能（每秒两万亿亿条），
                # 但静默回绕会让 id 重复，进而让主键冲突。宁可就地推进一毫秒，
                # 让顺序仍然单调：id 的时间戳比真实时间晚 1 ms 无害，重复有害。
                millis += 1
                _last_random = 0
            else:
                _last_random += 1
        else:
            if not explicit and _last_ms >= 0 and millis <= _last_ms:
                # 时钟被往回调了（NTP 校准或用户手改）。新的 id 必须仍排在旧的
                # 之后，否则那一刻之后写入的记忆会排到前面去。推进到 +1 ms 即
                # 可：时间戳不同时，顺序由时间戳决定，随机部分不影响。
                millis = _last_ms + 1
            # 换毫秒就换一批随机数。时间戳不同时顺序已由它决定，随机部分只负责
            # 避免不同机器／不同进程在同一毫秒内撞车。
            _last_random = int.from_bytes(os.urandom(10), "big")
        # 推进 1 ms 之后可能顶出 48 位。裁回去是**丢失单调性**，但在公元 10889
        # 年才会发生的这件事上，让 id 形状不合法（于是 is_ulid 判否、解析报错）
        # 是更糟的结果。
        millis = min(millis, _MAX_TIMESTAMP_MS)
        _last_ms = millis
        random_bits = _last_random

    return _encode(millis, _TIME_CHARS) + _encode(random_bits, _RANDOM_CHARS)


def _encode(value: int, width: int) -> str:
    """把整数编成定宽 Crockford base32，高位在前。"""
    chars = ["0"] * width
    for position in range(width - 1, -1, -1):
        chars[position] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def is_ulid(value: object) -> bool:
    """是否是形状合法的 ULID。

    只查形状（长度、字母表、时间戳不溢出），**不查它是否真的存在**——这个函数
    的用途是在边界上挡住手写坏的 id，而不是做外键检查。
    """
    if not isinstance(value, str) or len(value) != ULID_LENGTH:
        return False
    if any(char not in _DECODE for char in value):
        return False
    return int(_decode(value[:_TIME_CHARS])) <= _MAX_TIMESTAMP_MS


def ulid_timestamp_ms(value: str) -> int:
    """从 ULID 里读出毫秒时间戳。排障时用（"这条记忆是什么时候写的"）。"""
    if len(value) < _TIME_CHARS:
        raise ValueError(f"{value!r} 太短，不是 ULID。")
    return _decode(value[:_TIME_CHARS])


def _decode(text: str) -> int:
    value = 0
    for char in text:
        try:
            value = (value << 5) | _DECODE[char]
        except KeyError as exc:
            raise ValueError(f"{char!r} 不在 Crockford base32 字母表里。") from exc
    return value
