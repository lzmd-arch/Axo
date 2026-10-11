"""记忆空间（embedding space）的登记。

**这个模块为什么值得存在**：两个不同模型产生的向量放在一起算余弦相似度，不会
报错，只会给出一个无意义的数。这是本项目最危险的一类失败——没有异常、没有
日志、召回质量静默下降，而用户看到的是"它好像不太记得我说过什么"。

所以每个向量都必须连同它的空间标识一起被记账，而空间标识里**必须带维度**：

    ollama/nomic-embed-text@768
    └─来源─┘ └────模型────┘ └维度┘

把维度写进标识，是为了让"维度变了"这件事在**字符串比较**层面就暴露，而不必
等到矩阵乘法报形状错误——那时你只知道形状不对，不知道是哪个模型换过了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import DimensionMismatchError, SpaceNotFoundError
from .types import now_ts

__all__ = ["Space", "SpaceRegistry", "make_space_id", "parse_space_id"]

#: 空间标识的形状：``来源/模型@维度``。
#:
#: 用白名单正则而不是"切开看看"：空间标识会进 SQLite 主键、会出现在账本里、
#: 将来还会成为导出文件的目录名。与其在各个使用点各写一遍防御，不如在这里
#: 定死形状——与 ``session/store.py`` 里 ``_SESSION_ID_RE`` 同一条纪律。
_SPACE_ID_RE = re.compile(r"^(?P<source>[a-z0-9_.-]+)/(?P<model>[^/@\s]+)@(?P<dim>[1-9][0-9]*)$")


def make_space_id(source: str, model: str, dim: int) -> str:
    """拼一个空间标识，并校验结果符合 :data:`_SPACE_ID_RE`。"""
    if dim < 1:
        raise DimensionMismatchError(f"维度必须 >= 1，收到 {dim!r}。")
    space_id = f"{source}/{model}@{dim}"
    if not _SPACE_ID_RE.match(space_id):
        raise ValueError(
            f"拼出的空间标识 {space_id!r} 不合法。来源与模型名只能含字母数字与 . _ - ，"
            f"且不能含 / @ 与空白。"
        )
    return space_id


def parse_space_id(space_id: str) -> tuple[str, str, int]:
    """拆开空间标识，返回 ``(来源, 模型, 维度)``。

    Raises:
        ValueError: 形状不合法。
    """
    match = _SPACE_ID_RE.match(space_id or "")
    if not match:
        raise ValueError(
            f"空间标识 {space_id!r} 不合法。合法形状为「来源/模型名@维度」，"
            f"例如「本地/某嵌入模型@768」——来源与模型名只能用字母数字与 . _ - ，"
            f"维度必须是正整数。"
        )
    return match.group("source"), match.group("model"), int(match.group("dim"))


@dataclass(frozen=True)
class Space:
    """一个记忆空间。

    Attributes:
        space_id: 见 :func:`make_space_id`。
        dim: 向量维度。与 ``space_id`` 里记的那个数**必须一致**——两处都存是
            刻意的冗余，为了能在写入时交叉校验，而不是信任字符串解析。
        model_ref: 产生这些向量的模型的人类可读标识，例如
            ``"ollama/nomic-embed-text:latest"``。它**不参与**任何计算，只是
            让人在账本与调试输出里认出"这是哪个模型的空间"。
        created_at: 空间首次登记的时间。
    """

    space_id: str
    dim: int
    model_ref: str = ""
    created_at: float = field(default_factory=now_ts)

    def __post_init__(self) -> None:
        _, _, parsed_dim = parse_space_id(self.space_id)
        if parsed_dim != int(self.dim):
            raise DimensionMismatchError(
                f"space_id {self.space_id!r} 里写的维度是 {parsed_dim}，"
                f"而 Space.dim 是 {self.dim}。两者必须一致——这个交叉校验存在的"
                f"理由就是不让一个手滑的维度悄悄生效。"
            )


class SpaceRegistry:
    """内存里的空间登记表。

    刻意**不做持久化**：持久的那份在 :class:`memcore.store.MemoryStore` 的
    ``spaces`` 表里。这里只负责"进程内的一致性检查"，因为检查必须发生在写入
    向量**之前**，而那时还没有理由碰磁盘。

    一个进程通常只有一个 registry，由 store 在打开时把已登记的空间灌进来。
    """

    def __init__(self) -> None:
        self._spaces: dict[str, Space] = {}

    def register(self, space: Space) -> Space:
        """登记一个空间。已存在且维度一致时是幂等的，不一致则报错。

        不一致时**报错而不是覆盖**：同一个 space_id 对应两个维度，说明标识的
        构造方式已经错了（多半是模型换了版本却没换标识），此时覆盖等于把旧向量
        变成垃圾数据。
        """
        existing = self._spaces.get(space.space_id)
        if existing is not None:
            if existing.dim != space.dim:
                raise DimensionMismatchError(
                    f"空间 {space.space_id!r} 已登记为 {existing.dim} 维，"
                    f"现在却要用 {space.dim} 维登记它。若模型的输出维度真的变了，"
                    f"那是一个**新空间**，请给它一个新的标识。"
                )
            return existing
        self._spaces[space.space_id] = space
        return space

    def get(self, space_id: str) -> Space:
        """查找。未登记时抛 :class:`SpaceNotFoundError`。"""
        try:
            return self._spaces[space_id]
        except KeyError:
            known = ", ".join(sorted(self._spaces)) or "（一个都没有）"
            raise SpaceNotFoundError(
                f"空间 {space_id!r} 未登记。已登记的空间：{known}。"
                f"往一个未登记的空间写向量会让「记忆独立于模型」这条承诺失去校验点——"
                f"所以这里是拒绝，不是自动登记。"
            ) from None

    def require(self, space_id: str, dim: int) -> Space:
        """查找并校验维度。写入向量前的唯一入口。"""
        space = self.get(space_id)
        if space.dim != int(dim):
            raise DimensionMismatchError(
                f"空间 {space_id!r} 登记的是 {space.dim} 维，收到 {dim} 维的向量。"
                f"这通常意味着 space_id 里的维度与实际模型不符——该向量不得写入。"
            )
        return space

    def __contains__(self, space_id: object) -> bool:
        return space_id in self._spaces

    def __len__(self) -> int:
        return len(self._spaces)

    def all(self) -> list[Space]:
        return sorted(self._spaces.values(), key=lambda space: space.space_id)
