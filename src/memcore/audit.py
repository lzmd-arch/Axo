"""注入账本：这一轮往模型里放了什么，以及**为什么是这几条**。

这是本终端区别于普通聊天窗口的地方。一个普通的 RAG 系统，你只能看到"答案变了"；
这里你能逐条核对模型的输入，包括**哪些候选被淘汰了、因为什么被淘汰**。

两条纪律，写在这里因为它们定义了整个模块的形状：

1. **归因必须与数值一致。** :meth:`RecalledMemory.explain` 是从
   :class:`ChannelScores` 现算出来的模板文本，不是一个独立的字段。若允许调用方
   传一句人写的解释进来，解释就会与分数脱钩，而脱钩的解释比没有解释更糟——它
   看起来可信。

2. **禁止用 LLM 生成解释。** 一个语言模型写出的"我之所以想起这条，是因为……"
   读起来完全合理，但它描述的是**语言模型的猜测**，不是召回算法的实际行为。
   本模块只提供数据结构和纯函数模板；调用方拿不到任何可以接 LLM 的钩子。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ChannelScores",
    "RecallLedger",
    "RecalledMemory",
    "RejectReason",
    "RejectedCandidate",
]

#: 低于这个值的通道得分在解释文本里不再列出。
#:
#: 不列出来不是为了好看：把 0.004 也写成「联想 0.00」，会让人以为联想通道
#: 真的找到了什么，只是很弱。**没找到就是没找到**，这两件事在可解释性上不能混。
_ATTRIBUTION_FLOOR = 0.005

#: 原始余弦与归一化后的值相差不超过这个量时，不再单独标注原始值。低于它
#: 说明对比度归一化基本没起作用（库太小、或查询与本库普遍不相似），此时多写
#: 一个括号只是噪声。
_RAW_DISPLAY_EPSILON = 0.01

_CHANNEL_NAMES: tuple[tuple[str, str], ...] = (
    ("semantic", "语义"),
    ("lexical", "词法"),
    ("graph", "联想"),
)

_LABELS: dict[str, str] = dict(_CHANNEL_NAMES)


class RejectReason(str, Enum):
    """一条候选没有进入上下文的原因。"""

    BELOW_THRESHOLD = "below_threshold"
    """融合得分低于阈值。大多数记忆每轮都死在这里，这是正常的。"""

    REDUNDANT = "redundant"
    """与已入选的某条高度重复（MMR 判定）。"""

    OVER_BUDGET = "over_budget"
    """分数够、也不重复，但剩下的 token 预算装不下它。"""

    COLD = "cold"
    """已遗忘，不再参与召回。

    文案与记忆浏览器里的层级名一致（原本写的是「已归档」）。现在 ``COLD`` 唯一的
    生产者就是界面上的软遗忘，账本里说「已归档」会让用户认不出自己刚忘掉的那条。
    """

    NO_ENCODING = "no_encoding"
    """没有任何可用的编码（既没有词法 SDR，也没有本空间的向量）。

    这一条与其它几条性质不同：它指向的是**数据问题**而不是排序结果。出现它
    通常意味着向量回填断过——所以它在账本里必须显眼，而不是混在"分数不够"里。
    """

    MAX_ITEMS = "max_items"
    """本轮注入条数已达上限。"""

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        return _REJECT_LABELS[self]


_REJECT_LABELS: dict[RejectReason, str] = {
    RejectReason.BELOW_THRESHOLD: "分数不够",
    RejectReason.REDUNDANT: "与已入选的重复",
    RejectReason.OVER_BUDGET: "装不进剩余预算",
    RejectReason.COLD: "已遗忘",
    RejectReason.NO_ENCODING: "没有可用编码（向量可能没回填上）",
    RejectReason.MAX_ITEMS: "已达条数上限",
}


@dataclass(frozen=True)
class ChannelScores:
    """各通道**参与融合的那个值**，以及融合后的结果。

    保留分通道的分数（而不是只留融合值），是账本能回答"为什么是这几条"的前提：
    融合后 0.42 的两条记忆，可能一条是语义找到的、另一条是词法找到的。

    :attr:`semantic` 是**归一化之后**的值（见 :func:`memcore.recall._contrast`），
    不是原始余弦——账本里显示的必须是**真正决定去留的那个数**，否则解释与行为
    脱钩。原始余弦如果需要，在 :attr:`semantic_raw` 里。

    :attr:`graph` 在 P1 恒为 0——关联基质要到 P4 才建。
    """

    lexical: float = 0.0
    semantic: float = 0.0
    graph: float = 0.0
    fused: float = 0.0
    semantic_raw: float = 0.0
    """语义通道的原始余弦。只在它被对比度归一化过、且与 :attr:`semantic` 明显
    不同时才值得看——它回答的是"嵌入模型认为这两段话有多像"，而不是"相对本库
    的其它条目有多突出"。"""

    def channel(self, field_name: str) -> float:
        """按字段名取通道得分。"""
        return float(getattr(self, field_name))

    def active_fields(self) -> list[str]:
        """有实际贡献的通道的**字段名**，按贡献降序。"""
        fired = [field for field, _ in _CHANNEL_NAMES if self.channel(field) > _ATTRIBUTION_FLOOR]
        return sorted(fired, key=lambda field: -self.channel(field))

    def active(self) -> list[tuple[str, float]]:
        """有实际贡献的通道 ``(中文名, 得分)``，按贡献降序。"""
        return [(_LABELS[field], self.channel(field)) for field in self.active_fields()]

    def note(self, field_name: str) -> str:
        """某个通道后面该附的补充说明。目前只有语义通道有。"""
        if field_name != "semantic":
            return ""
        if abs(self.semantic_raw - self.semantic) <= _RAW_DISPLAY_EPSILON:
            return ""
        return f"（余弦 {self.semantic_raw:.2f}）"


@dataclass(frozen=True)
class RecalledMemory:
    """一条被注入上下文的记忆。"""

    memory_id: str
    content: str
    kind: str
    tier: str
    weight: float
    """参与打分时的权重（**已结算惰性衰减**）。"""

    tokens: int
    scores: ChannelScores
    pinned: bool = False
    session_id: str = ""

    @property
    def explain(self) -> str:
        """一句话说明这条为什么在这里。**模板生成**，不经过任何模型。

        形如「因语义 0.87 · 词法 0.31 被注入，184 token；权重 0.62」。
        """
        fields = self.scores.active_fields()
        if not fields:
            # 归因全 0 却仍然被注入，只有两种情形：pinned 条目走 L0 通道进来，
            # 或者代码里出了问题。两种都必须**如实显示**，不能编一个理由。
            lead = "钉住条目，不参与打分" if self.pinned else "归因不可用（各通道得分均为 0）"
            return f"{lead}；{self.tokens} token"
        phrase = " · ".join(
            f"{_LABELS[field]} {self.scores.channel(field):.2f}{self.scores.note(field)}"
            for field in fields
        )
        return f"因{phrase} 被注入，{self.tokens} token；权重 {self.weight:.2f}"


@dataclass(frozen=True)
class RejectedCandidate:
    """一条被淘汰的候选。"""

    memory_id: str
    content_preview: str
    reason: RejectReason
    score: float
    detail: str = ""

    @property
    def explain(self) -> str:
        suffix = f"（{self.detail}）" if self.detail else ""
        return f"{self.reason.label}{suffix} · 得分 {self.score:.2f}"


@dataclass(frozen=True)
class RecallLedger:
    """一轮召回的完整记录。"""

    query_preview: str
    """查询的前几十个字。**截断**——账本会留在内存里跟着会话，不该把整段用户
    输入再复制一份。"""

    scanned: int = 0
    """扫过多少条候选（全库减掉 pinned 与排除项）。"""

    channels: tuple[str, ...] = ()
    """实际启用的通道名。没有嵌入模型时这里只有 ``("lexical",)``。"""

    space_id: str = ""
    injected: tuple[RecalledMemory, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    tokens_used: int = 0
    tokens_budget: int = 0
    elapsed_ms: float = 0.0
    notes: tuple[str, ...] = field(default=())
    """装配过程中的异常情况，例如"回填了 32 条向量"。"""

    @property
    def injected_ids(self) -> list[str]:
        return [memory.memory_id for memory in self.injected]

    def summary(self) -> str:
        """给侧栏用的一行摘要。"""
        channel_text = "+".join(self.channels) if self.channels else "无通道"
        return (
            f"扫过 {self.scanned} 条 · 注入 {len(self.injected)} 条 · "
            f"{self.tokens_used}/{self.tokens_budget} token · {channel_text} · "
            f"{self.elapsed_ms:.0f} ms"
        )
