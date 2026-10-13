"""把「同一件事的碎条目」并成一条带时间的记录。

用户的要求是两句话：「巩固与召回也需要大幅优化」+「已有的记忆不允许被你改动」。
前一句指向碎条目——「感冒那位回来了」「还带了杯奶茶」一句一条，读起来是散的；
后一句把「改写头部、吞并成员」那条路封死了。所以并出来的是**新的一行**，原句
一个字都不动。

**头部是并列，不是合成的因果断言。** 写成「感冒那位回来了；还带了杯奶茶」，它
说的是「这几句是这段时间里说的」——这永远是真的；而「她回来时带了杯奶茶」是新的
因果，合错了就等于往库里写一句假话。所以这里**只按时间序把原话连起来，不改写
用词**，原话另起一行附在后面。

**判据是纯算法的，不调任何模型（本地的也不调）。** 三件事：同一个会话、时间挨着、
要么有共同的字组要么有一边是碎句。宁可少并，不能并错——并错了原话还在、原句也还
在库里的，用户对得出来；但模型读到的那条整理条目会把两件事说成一件事。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .text import tokenize
from .timefmt import format_clock
from .types import MemoryItem

__all__ = [
    "BURST_GAP_SECONDS",
    "FragmentGroup",
    "MAX_GAP_SECONDS",
    "MAX_GROUP",
    "SHORT_CHARS",
    "group_fragments",
    "render_group",
    "span_of",
]

#: 同一件事的碎条目之间最多隔多久。30 分钟——比一轮对话长，比一顿饭短。
MAX_GAP_SECONDS = 30.0 * 60.0

#: **没有共同字组**时只并「紧接着说的」：3 分钟内。共同字组说明在说同一样东西，
#: 没有它就只剩时间相邻这一条线索，而那条线索只能撑住一个话头。
BURST_GAP_SECONDS = 3.0 * 60.0

#: 算「碎句」的长度上限（字符数）。短句多半是半截话，长句各自是完整事实，挨着说
#: 不代表是一件事。
#:
#: **12 是量出来的，不是猜的**：库里完整的事实句在 12–20 字（「医保报销材料交到
#: 街道服务中心二楼窗口」19 字），碎句在 8 字以内（「还带了杯奶茶」7 字）。早先
#: 取 24，一条 19 字的完整事实句就被当成了碎句，与隔壁一句无关的话并成了一条
#: ——那正是「并错了」的形状。
SHORT_CHARS = 12

#: 一组最多几条。再多就不是一件事，是一整段对话——而「一整段对话」正是碎条目
#: 本来的样子，把它并成一条等于把归并做成了打包。
MAX_GROUP = 8


@dataclass(frozen=True)
class FragmentGroup:
    """一组碎条目。:attr:`items` 按 ``created_at`` 升序。"""

    items: tuple[MemoryItem, ...]

    @property
    def start(self) -> float:
        return self.items[0].created_at

    @property
    def end(self) -> float:
        return self.items[-1].created_at

    @property
    def span(self) -> tuple[float, float]:
        """(起, 止)——给 :attr:`MemoryItem.meta` 的 ``span`` 与渲染用的区间。"""
        return (self.start, self.end)

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(item.memory_id for item in self.items)

    @property
    def session_id(self) -> str:
        return self.items[0].session_id

    def __len__(self) -> int:
        return len(self.items)


def span_of(item: MemoryItem) -> tuple[float, float] | None:
    """从 ``meta["span"]`` 读区间。不是归并产物、或脏数据，都返回 ``None``。

    与 :func:`memcore.recall._span_of` 同一口径：``meta`` 是自由格式 JSON，读不出
    来就当没有，不能让一轮装配因此失败。
    """
    raw = item.meta.get("span")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        start, end = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if start > 0 else None


def group_fragments(items: Iterable[MemoryItem]) -> list[FragmentGroup]:
    """把碎条目分组。**只并成 2 条以上**——单条没什么可并的。

    分组是**链式**的：按时间排序后一条条往前接，接不上就封口。所以「同一件事」
    在这里的定义是「一串彼此相邻、两两都合得上的话」，而不是「全体两两都合得上」
    ——后者在三条以上时会因为中间那条的措辞而整组散掉。

    不参与的：钉住条目、人格锚、已经归并过的派生条目（再并一遍会得到「整理条目的
    整理条目」）、以及没有 ``session_id`` 的（跨会话的那些不是「这一场里说的」）。
    """
    candidates = sorted(
        (item for item in items if _eligible(item)),
        key=lambda item: (item.created_at, item.memory_id),
    )
    groups: list[FragmentGroup] = []
    chain: list[MemoryItem] = []
    for item in candidates:
        if chain and len(chain) < MAX_GROUP and _compatible(chain[-1], item):
            chain.append(item)
            continue
        if len(chain) >= 2:
            groups.append(FragmentGroup(tuple(chain)))
        chain = [item]
    if len(chain) >= 2:
        groups.append(FragmentGroup(tuple(chain)))
    return groups


def render_group(group: FragmentGroup) -> str:
    """派生条目的正文：按时间序连成的一句 + 原话附在后面。

    原话带自己的时刻（不带日期——这一条本身挂着完整日期，见
    :func:`memcore.timefmt.format_clock`）。**原话逐字照抄**：它是用户核对的依据，
    也是「不许改动已有记忆」这条约束在派生条目上的表现形式。
    """
    head = "；".join(_bare(item.content) for item in group.items) + "。"
    quotes = "；".join(
        f"{format_clock(item.created_at)}「{item.content.strip()}」" for item in group.items
    )
    return head + "\n" + "    ↳ 原话：" + quotes


def _eligible(item: MemoryItem) -> bool:
    if item.pinned or item.is_persona_anchor:
        return False
    if item.meta.get("derived"):
        return False
    return bool(item.session_id)


def _compatible(left: MemoryItem, right: MemoryItem) -> bool:
    if left.session_id != right.session_id:
        return False
    gap = right.created_at - left.created_at
    if not 0.0 <= gap <= MAX_GAP_SECONDS:
        return False
    if _shares_bigrams(left.content, right.content):
        return True
    # 没有共同字组就只剩时间相邻：收窄到「紧接着说的」，并且要求有一边是碎句。
    return gap <= BURST_GAP_SECONDS and min(len(left.content), len(right.content)) <= SHORT_CHARS


def _shares_bigrams(left: str, right: str) -> bool:
    """两句话有没有共同的二字组。

    **用二字组而不是单字**：中文没有词边界，单字重合几乎必然（的/了/我），拿它当
    「在说同一件事」会让所有碎条目并成一堆。二字组是 :func:`memcore.text.tokenize`
    里那个 ``b:`` 前缀的 token——「感冒」这样的词才留得下来。
    """
    return bool(_bigrams(left) & _bigrams(right))


def _bigrams(text: str) -> set[str]:
    return {token for token in tokenize(text) if token.startswith("b:")}


def _bare(content: str) -> str:
    """去掉尾部的句读——几段拼成一句时，中间那些句号会把并列读成几句独立的话。"""
    return content.strip().rstrip("。！？；，、 ")
