"""记忆核心的异常类型。

所有异常都继承 :class:`MemCoreError`，调用方可以只捕获基类——这是照抄
``biosnn_bus.errors`` 的写法，让两个库的错误处理手感一致。

约定：每个子类在继承 :class:`MemCoreError` 之外，**再继承一个最贴近语义的内建
异常**。这样调用方既可以用 ``except MemCoreError`` 一把兜住，也可以用
``except ValueError`` 走通用的参数校验路径。
"""

from __future__ import annotations

__all__ = [
    "MemCoreError",
    "SpaceError",
    "SpaceNotFoundError",
    "SpaceMismatchError",
    "DimensionMismatchError",
    "SDRCapacityError",
    "StoreError",
    "StoreCorruptError",
]

# 本模块刻意只 import 标准库之外什么都不 import —— 它是异常的公共出口，
# 任何子模块都可能 import 它，因此它必须是依赖图里最底层的节点之一。


class MemCoreError(Exception):
    """记忆核心所有异常的基类。"""


# ── 记忆空间（embedding space） ───────────────────────────────────────────────


class SpaceError(MemCoreError):
    """记忆空间相关错误的基类。"""


class SpaceNotFoundError(SpaceError, KeyError):
    """按 space_id 查找记忆空间时未找到。"""

    def __str__(self) -> str:  # KeyError 的 __str__ 会加引号，改回可读信息
        return self.args[0] if self.args else ""


class SpaceMismatchError(SpaceError, ValueError):
    """一个向量被送进了不属于它的记忆空间。

    最典型的触发方式：把 embedding 模型 A 产生的向量写进模型 B 的空间。两者的
    数值含义完全无关，混在一起不会报错、只会让召回质量静默劣化——所以宁可在
    写入点直接拒绝。这是"记忆独立于模型"承诺的守门人：**换模型时该换的是空间，
    不是把新向量倒进旧空间**。
    """


class DimensionMismatchError(SpaceError, ValueError):
    """向量维度与空间声明的维度不符。"""


# ── 稀疏分布表示 ─────────────────────────────────────────────────────────────


class SDRCapacityError(MemCoreError, ValueError):
    """SDR 的活跃位数非法：k <= 0、k > N，或权重数组长度与索引不一致。"""


# ── 存储 ────────────────────────────────────────────────────────────────────


class StoreError(MemCoreError):
    """存储层错误的基类。"""


class StoreCorruptError(StoreError, ValueError):
    """库文件损坏或与代码期望的 schema 版本不符。"""
