"""把一次请求要发给模型的全部内容拼出来，并记账。

**这是"上下文长度限制"那个原点问题落地的地方。** 不靠把窗口塞满，靠三层动作：
按层分配额度 → 装不下的压成摘要 → 每轮产出一份"注入了什么、为什么"的账本。

层的编号是**优先级**，不是位置：

======  =================  ==========================================
层      内容               能不能被驱逐
======  =================  ==========================================
L0      人格锚             不能。它就是"助手是谁"
L1      环境               不能。小，且每一轮都要
L2      记忆召回           能。它是这层设计存在的理由，但整层丢得起
L3      近期对话（逐字）   能。丢掉的进 L4
L4      历史摘要           能。它是 L3 的压缩，最便宜
L5      工具结果           P3
======  =================  ==========================================

**当前这一轮的用户输入不在这张表里，因为它永远不被驱逐。** 用户刚说的话优先级
高于一切——一条"很久以前也许相关"的记忆挤掉用户当前的问题，是这类系统最典型的
失态。

输入是**完整的** :class:`~axo.session.store.Message` 序列，而不是文本
列表：按"文本列表"设计的装配器喂不了工具行，要加就得整个重写。

**所有进模型的东西都要经过这里。** 绕过它的后果不是"少了一次记账"，而是预算失去
意义：某一层偷偷多发 200 token，代价由别层承担，而承担方式是服务端静默截断最早的
消息——那是人格锚的位置。

这个模块不碰 Qt、不碰网络、不知道模型是谁。它只做算术和排版，所以能单测。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memcore.merge import FragmentGroup, group_fragments, render_group
from memcore.timefmt import format_span, format_when, has_time

from ..llm.base import ChatMessage, ToolCall
from .budget import GENERATION_PROMPT, PER_MESSAGE_OVERHEAD, TokenBudget, TokenMeter
from .store import Message
from .summarize import summarize

__all__ = [
    "Assembly",
    "AssemblyLedger",
    "Layer",
    "LayerReport",
    "MemoryInjection",
    "assemble",
    "stored_rows",
]

#: **L3 认识的**对话角色。两类工具相关的行不在里面：``tool`` 行，以及带
#: ``tool_calls`` 的 assistant 行。它们整对归 L5（见 :func:`_scan_tool_history`），留在 L3 里
#: 会同时破坏两件事——L3 的"连续逐字稿"语义（工具行插在中间就是挖洞），以及配对：
#: L3 是**从最近往前收到额度为止**的，很容易收下调用声明而把紧随其后的结果行挡在外面，
#: 于是模型收到一句没人答复的调用。**而且它会时灵时不灵**，取决于当时剩多少钱。
_ROLES = frozenset({"user", "assistant", "system"})

_RECALL_HEADER = "以下是与当前问题相关的长期记忆："


class Layer(str, Enum):
    """上下文的层。编号是优先级，不是位置。"""

    PERSONA = "L0_persona"
    ENVIRONMENT = "L1_environment"
    RECALL = "L2_recall"
    RECENT = "L3_recent"
    SUMMARY = "L4_summary"
    TOOLS = "L5_tools"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        """账本侧栏里的名字。"""
        return _LAYER_LABELS[self]


_LAYER_LABELS: dict[Layer, str] = {
    Layer.PERSONA: "人格锚",
    Layer.ENVIRONMENT: "环境",
    Layer.RECALL: "记忆召回",
    Layer.RECENT: "近期对话",
    Layer.SUMMARY: "历史摘要",
    Layer.TOOLS: "工具结果",
}

#: **没有份额了。** 以前这里有一张 L2=0.25 / L3=0.50 / L5=0.20 的分配表和一个
#: :func:`layer_ceiling`，那是自设的额度——用户的要求是「除了官方的上下文限制之外，
#: 不要出现任何其他的预算、额度、上下文限制」。现在装箱只有**一个**上限：窗口本身。
#:
#: 顺序仍然有意义（记忆 → 近处对话 → 工具结果 → 摘要），但那是**谁先拿**，不是
#: **谁最多拿多少**。前一层用剩多少，后一层就拿多少；全部装得下时顺序完全不影响
#: 结果——而云端 128k 的窗口装下全部记忆与整段对话是常态。


@dataclass(frozen=True)
class MemoryInjection:
    """一条准备注入上下文的记忆。

    **只有装配需要的四个字段。** 装配器不该认识记忆的权重、分层、SDR、通道得分
    ——那些是 :mod:`memcore.recall` 的事，而它已经把结论写进 :attr:`explain` 了。
    把 :class:`memcore.audit.RecalledMemory` 整个传进来，等于让装配器开始依赖召回
    的实现细节；P3 的任务记忆来自另一条路径，那时这个结构就是两条路径的公共形状。
    """

    memory_id: str
    content: str
    tokens: int
    explain: str
    score: float = 0.0
    """融合得分。**只在装不下时用来决定先丢谁**（从低到高丢）——它不参与排版，
    上下文里的顺序是按时间的。"""

    created_at: float = 0.0
    """写下这条记忆的时刻。进上下文的那一行要带它——没有时间戳的原话只是一串
    没有先后的句子，而用户要的正是"模型能形成时间观念"。``<= 0`` 表示不知道，
    渲染时省掉方括号（见 :mod:`memcore.timefmt`）。"""

    span: tuple[float, float] | None = None
    """归并产物的（起、止）时刻。单条记忆没有区间，记 ``None``。"""

    session_id: str = ""
    """写下这条记忆的会话。归并要看它——跨会话的两句不是「这一场里说的」。"""

    derived_from: tuple[str, ...] = ()
    """非空就说明这一条是**整理条目**，它代表这些原句。

    正文里已经带着逐字原话，所以原句不再单独列——同一段话在上下文里出现两次，
    第二次还占着记忆层的位置。"""


@dataclass(frozen=True)
class LayerReport:
    """账本里的一层。"""

    layer: Layer
    tokens: int
    ceiling: int | None = None
    """这一层装箱时**手上还剩多少**。不是分配给它的份额（份额已经拆掉了），而是
    "前面几层花完之后剩下的全部"——所以它随层递减。不可驱逐的层（L0/L1）记
    ``None``。"""

    count: int = 0
    """这一层装了几条（L5 数的是**行**，不是对）。摘要层恒为 0——它覆盖了多少条
    由账本的"压掉"那一行说。"""

    note: str = ""


@dataclass(frozen=True)
class AssemblyLedger:
    """这一轮往模型里注入了什么。**给人看的，能被核对。**

    §3.4 的验收要求侧栏能显示「因语义 0.87 被注入，184 token」这样的逐条归因——
    那句话由 :attr:`MemoryInjection.explain` 原样带过来，**模板生成，不经过任何
    模型**。让模型解释"我为什么记得这件事"是幻觉的标准配方。
    """

    budget: TokenBudget
    layers: tuple[LayerReport, ...] = ()
    injected: tuple[MemoryInjection, ...] = ()
    kept_messages: int = 0
    dropped_messages: int = 0
    total_tokens: int = 0
    """整个请求的估算代价，**从发出去的消息算出来**（``TokenMeter.messages``），
    不是各层手工相加。手工相加总会漏掉某一项，而漏掉的那一项正是"账本对不上"的
    来源——这个字段是给人核对的，它必须度量实际发出去的东西。

    **不含** :attr:`tools_tokens`：那部分不在 ``messages`` 里，混进来会让这个数
    不再是"消息值多少"。真正整个请求的代价是这两者之和。"""

    tools_tokens: int = 0
    """本轮 ``ChatRequest.tools`` 的**声明开销**，由调用方量好传进 :func:`assemble`。

    **它必须单独记一笔，因为 ``TokenMeter.messages`` 量不到它**——工具声明走
    ``ChatRequest.tools``，是一个与 ``messages`` 并列的字段。Step 0 实测 3 个工具
    = 净增 311 token/轮（且每轮都要付），要上的七个约是它的两倍。这笔钱不属于任何
    一层，但它确实挤占同一个窗口，所以把它藏起来就等于账本少了一行。
    """

    over_budget: int = 0
    """超出 ``usable`` 的估算量（含 :attr:`tools_tokens`）。正常为 0；非 0 只有三种
    可能：当前这轮用户输入本身就太长、人格锚大到装不下别的东西，或者工具声明把
    余量吃掉了。"""

    notes: tuple[str, ...] = ()

    @property
    def injected_ids(self) -> tuple[str, ...]:
        return tuple(item.memory_id for item in self.injected)

    def layer(self, layer: Layer) -> LayerReport | None:
        for report in self.layers:
            if report.layer is layer:
                return report
        return None

    def render(self) -> str:
        """侧栏里的多行文本。

        刻意用**定宽两列**而不是 Markdown 表格：它显示在 Qt 控件里，而对齐靠的是
        字符宽度——中文占两格，Python 的 ``ljust`` 不知道这件事，所以这里自己按
        显示宽度补齐。
        """
        lines = [
            f"装配账本 · 估算 {self.total_tokens} / {self.budget.usable} token"
            f"（窗口 {self.budget.window}，留 {self.budget.reserve_output} 给生成）",
            "─" * 34,
        ]
        for report in self.layers:
            lines.append(
                f"{_padded(report.layer.label)} {report.tokens:>6} token"
                + (f"  {report.count} 条" if report.count else "")
                + (f"  {report.note}" if report.note else "")
            )
        # 工具声明**不是一层**（它不在 messages 里），但它在同一个窗口里，所以照样
        # 单独占一行。名字刻意与 L5 的"工具结果"分开写：一个是每轮都要付的声明开销，
        # 一个是回放的旧结果，混成一个词会让"这轮为什么超了"没法追。
        if self.tools_tokens:
            lines.append(
                f"{_padded('工具声明')} {self.tools_tokens:>6} token  （不在 messages 里）"
            )
        if self.dropped_messages:
            lines.append(f"（{self.dropped_messages} 条更早的消息已被逐出上下文）")
        if self.injected:
            lines.append("")
            for item in self.injected:
                stamp = _stamp_of(item)
                prefix = f"[{stamp}] " if stamp else ""
                lines.append(f"  · {prefix}{item.explain}")
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


def _padded(label: str, width: int = 10) -> str:
    """按**显示宽度**补空格。

    ``str.ljust`` 数的是字符数，而中文在等宽字体里占两格——直接用 ``ljust`` 会让
    中文标签那一列参差不齐，而参差不齐的账本比没有账本更难看懂。
    """
    shown = sum(2 if _is_wide(char) else 1 for char in label)
    return label + " " * max(1, width - shown)


def _is_wide(char: str) -> bool:
    code = ord(char)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
        or 0x20000 <= code <= 0x3FFFD
    )


@dataclass(frozen=True)
class Assembly:
    """装配的产物：发给模型的消息 + 它的账本。"""

    messages: tuple[ChatMessage, ...]
    ledger: AssemblyLedger


def _render_recall(items: Sequence[MemoryInjection]) -> str:
    """L2 那一段正文。**每行都带时间**，碎条目当场并成一条。

    时间写在方括号里、紧挨正文：模型扫一行就同时拿到「什么时候」和「什么事」，
    而「昨天我说了什么」全靠这两样挨在一起。没时间的条目不打方括号——编一个时间
    比没有时间更坏，它会被模型当事实读（见 :mod:`memcore.timefmt`）。

    **归并在视图这一层做，零写库。** 两条规则：有整理条目代表的原句不再单独列
    （它的正文里带着逐字原话）；还没被整理过的新碎条目当场打包。判据与
    :func:`memcore.merge.group_fragments` 是同一套——两层各写一套就会长出
    「库里的整理条目与上下文里的整理条目不是一回事」。

    **按时间升序排**，时间不可知的排最后：它们不属于这条时间线，而排在最前会
    让模型以为那是最早发生的事。
    """
    if not items:
        return ""
    ordered = sorted(items, key=_order_key)
    covered = {memory_id for item in ordered for memory_id in item.derived_from}
    plain = [item for item in ordered if not item.derived_from and item.memory_id not in covered]
    derived = [item for item in ordered if item.derived_from]

    blocks: list[tuple[float, str]] = [(_sort_time(item), _recall_line(item)) for item in derived]
    grouped: set[str] = set()
    for group in group_fragments(plain):
        if len(group) < 2:
            continue
        blocks.append((group.start, _merged_line(group)))
        grouped.update(group.memory_ids)
    blocks.extend(
        (_sort_time(item), _recall_line(item)) for item in plain if item.memory_id not in grouped
    )
    blocks.sort(key=lambda block: block[0])
    return "\n".join([_RECALL_HEADER, *(text for _, text in blocks)])


def _sort_time(item: MemoryInjection) -> float:
    """排时间线用的时刻。**时间不可知的丢到最后**（记无穷大，不是 0）。

    记 0 会让它排在最前，而模型会把「没有时间」读成「最早」——那正是这批要治的
    「模型形成错误的时间观念」。
    """
    return item.created_at if has_time(item.created_at) else float("inf")


def _order_key(item: MemoryInjection) -> tuple[float, str]:
    return (_sort_time(item), item.memory_id)


def _merged_line(group: FragmentGroup[MemoryInjection]) -> str:
    """一组碎条目 → 一行：方括号挂**区间**，正文第一行接在后面，原话另起一行。"""
    start, end = group.span
    stamp = format_span(start, end) if end > start else format_when(start)
    head, _, tail = render_group(group).partition("\n")
    return f"· [{stamp}] {head}" + (f"\n{tail}" if tail else "")


def _recall_line(item: MemoryInjection) -> str:
    stamp = _stamp_of(item)
    return f"· [{stamp}] {item.content}" if stamp else f"· {item.content}"


def _stamp_of(item: MemoryInjection) -> str:
    """这一行该挂哪个时间戳。区间优先于单点，都没有就空串。"""
    if item.span is not None:
        return format_span(*item.span)
    return format_when(item.created_at) if has_time(item.created_at) else ""


def _to_chat(message: Message) -> ChatMessage | None:
    """**L3 用**的一行转换：只认识纯对话行。``None`` 表示"这一行不进 L3"。

    三类被挡在外面，理由各不相同：

    * **带 ``tool_calls`` 的 assistant 行**：它的结果行整对归 L5，留在这里就是一句
      没人答复的调用声明——服务端通常直接拒掉整个请求，而症状是一个"最近用过工具"
      的会话之后每一轮都失败，离原因很远。
    * **``tool`` 行**：同上。而且它必须**紧跟**自己那条 assistant 行，L3 是收到额度
      为止的，跟着跟着就可能把它俩拆开。
    * **本版本不认识的角色**：跳过而不是抛。存储层出现一行新东西，不该让整个装配
      失败、进而让界面卡在"生成中"。
    """
    if message.role == "assistant" and message.tool_calls:
        return None
    if message.role not in _ROLES:
        return None
    return ChatMessage(role=message.role, content=message.content)  # type: ignore[arg-type]


def _read_call(raw: dict[str, Any]) -> ToolCall:
    """把存下来的一条调用读回 :class:`ToolCall`。

    id 与 name 都要求非空：空 id 会让"配对"这件事本身失去意义（两条空 id 的调用
    看起来是同一个），而那是靠下面 :func:`_tool_pair` 把这一对整对丢掉来处理的——
    所以这里抛，由调用方接住并整对放弃。
    """
    call_id = raw.get("id")
    name = raw.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"工具调用的 id 为空或不是字符串：{call_id!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"工具调用的 name 为空或不是字符串：{name!r}")
    arguments = raw.get("arguments")
    return ToolCall(
        id=call_id,
        name=name,
        arguments=dict(arguments) if isinstance(arguments, dict) else {},
    )


def _write_call(call: ToolCall) -> dict[str, Any]:
    """把一条调用写成落盘的那个字典。**:func:`_read_call` 的逆。**"""
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}


def stored_rows(
    rows: Sequence[ChatMessage],
    *,
    no_replay: Sequence[tuple[str, str]] = (),
) -> list[Message]:
    """把工具循环产出的一对（或几对）行落成逐字稿里的行。

    这是 :func:`_read_call` 那一侧的逆运算，**放在同一个文件里**是为了让这两个方向
    挨着——它们各自能过测试、却互相读不懂的那种错误，只有并排看才看得出来。

    Args:
        rows: 循环产出的那些行。
        no_replay: ``(tool_call_id, 原因)``，由 :class:`~axo.agent.loop.
            MessagesAppended` 原样带过来。标在对应的 tool 行上，L5 据此整对跳过。
            **默认空**：不带这个参数的老调用点（以及没有工具轮的普通对话）行为一字不变。

    ## 为什么 ``no_replay`` 是参数而不是从行里推

    它是**循环才知道的事**：那次调用是被闸门拒了，还是结果本身不耐放。到了这里只剩
    行，从行里推不出来——除非去认那几句文案，而那是把一句话的措辞变成判据，改一个字
    就静默失效。所以它随事件过来，这里只负责贴上。

    ## 为什么 ``tool_name`` 在这里补，而不是让循环带上

    :class:`~axo.llm.base.ChatMessage` 上**没有**这个字段，而它是协议层的
    形状：回灌工具结果只需要 ``tool_call_id``（实测如此，见 ``llm/base.py``），为了
    让逐字稿好看一点去给协议加一个字段，是把存储的偏好写进协议。

    所以名字在这里从**同一批里前一条 assistant 行**的 ``tool_calls`` 里查出来。这个
    查法是可靠的，理由不是"一般是这么配的"，而是**这一批由循环产出、必然完整**：
    每个 ``tool`` 行的 ``tool_call_id`` 都来自它前面那条 assistant 行。真配不上时
    （不该发生）留 ``None``——名字只用于回看显示，缺了不该让整行写不进去。

    ## 为什么校验和 ``_read_call`` 不对称

    ``_read_call`` 对空 id 抛，因为读到的东西可能来自**任何一个**旧版本写下的文件；
    这里不抛，因为输入是自己刚产出的、形状是构造出来的。多一道校验只会在真出问题时
    把一条**已经发生过**的对话整段挡住——那正好反了：记录宁可残缺也要留下。
    """
    stored: list[Message] = []
    names: dict[str, str] = {}
    reasons = dict(no_replay)
    for row in rows:
        if row.role == "assistant":
            for call in row.tool_calls:
                names[call.id] = call.name
            stored.append(
                Message(
                    role=row.role,
                    content=row.content,
                    tool_calls=tuple(_write_call(call) for call in row.tool_calls),
                )
            )
        elif row.role == "tool":
            stored.append(
                Message(
                    role=row.role,
                    content=row.content,
                    tool_call_id=row.tool_call_id,
                    tool_name=names.get(row.tool_call_id or ""),
                    no_replay=reasons.get(row.tool_call_id or ""),
                )
            )
        else:
            stored.append(Message(role=row.role, content=row.content))
    return stored


def _tool_pair(
    history: Sequence[Message], index: int
) -> tuple[list[ChatMessage], int, str | None] | None:
    """从 ``index`` 起读**一对**：一条带 ``tool_calls`` 的 assistant + 答复它的全部 tool 行。

    读不出完整的一对时返回 ``None``（由调用方跳过这一条、继续往后找）。返回
    ``(行, 下一处下标, 不回放的原因)``。

    **"完整"是这里的全部内容**：声明了几个调用就要有几个结果，必须紧随其后、id 对得
    上、而且一一对应（一条结果答复两次同一次调用同样不成对）。判据刻意用集合相等而
    不是"至少有一个结果"——半个配对是模型无法解释的对话，而它的失败表现是服务端拒掉
    整个请求，不是任何能追到这里的症状。

    **第三项与"完整"无关**：那一对是**好好的**，只是不该在后续轮次里回放（见
    :attr:`~axo.session.store.Message.no_replay`）。它在这里返回而不是在
    调用方另查一遍，是因为读这些行只有这一处——多一处读法就多一处能读漏。
    一次调用里有两个 tool 行、原因还不一样时（一个被拒、一个过期）取**第一个**：
    整对是原子回放的，两个原因指向同一个动作——跳过。
    """
    head = history[index]
    if head.role != "assistant" or not head.tool_calls:
        return None
    try:
        calls = tuple(_read_call(raw) for raw in head.tool_calls)
    except (TypeError, ValueError):
        # 落盘的调用形状坏了。整对丢掉而不是修一修：猜出来的参数可能对应一次完全
        # 不同的动作，而它会看起来和真的一样。
        return None
    wanted = {call.id for call in calls}
    if len(wanted) != len(calls):
        return None  # id 重复 → 没法判定哪条结果答复哪次调用

    rows: list[ChatMessage] = []
    reasons: list[str] = []
    cursor = index + 1
    while cursor < len(history) and history[cursor].role == "tool":
        row = history[cursor]
        if row.tool_call_id not in wanted:
            return None  # 结果答复了一次不在声明里的调用
        rows.append(ChatMessage(role="tool", content=row.content, tool_call_id=row.tool_call_id))
        if row.no_replay:
            reasons.append(row.no_replay)
        cursor += 1
    if len(rows) != len(calls) or {row.tool_call_id for row in rows} != wanted:
        return None  # 有声明没被答复，或同一次调用被答复了两次
    pair = [ChatMessage(role="assistant", content=head.content, tool_calls=calls), *rows]
    return pair, cursor, (reasons[0] if reasons else None)


def _scan_tool_history(history: Sequence[Message]) -> tuple[list[list[ChatMessage]], int, int]:
    """切出**可回放的完整工具对**，并数出另外两种行。

    返回 ``(可回放的对, 坏行数, 按策略不回放的对数)``。切不出对的行（孤儿 ``tool``
    行、形状坏掉的声明）不出现在返回值的第一项里——它们不是"被丢掉的一对"，而是本就
    不构成一对。于是"绝不半留"这件事是**结构上**保证的，不靠任何一处记得去检查。

    **三种行分成三个出口**（能回放的 / 坏了 / 按策略不回放），因为它们是三个不同的
    诊断：坏了指向落盘的数据有问题，按策略不回放指向一次当轮就该过去的事或一个过期
    的快照，而装不下（额度不够）由调用方另算。混成一个数，账本上就分不出该修哪儿。

    三件事放在同一个遍历里，是因为它们是同一个判断的三个出口。分成几个函数各扫一遍
    的话，后面的必须重新判断"这一行被收走了吗"，而它判错的方式是把某行算两次或一次
    都不算——那些计数从此不对，且没有人会发现，它们只是提示数。
    """
    pairs: list[list[ChatMessage]] = []
    broken = 0
    skipped = 0
    index = 0
    while index < len(history):
        message = history[index]
        relevant = message.role == "tool" or (
            message.role == "assistant" and bool(message.tool_calls)
        )
        found = _tool_pair(history, index) if relevant else None
        if found is None:
            if relevant:
                broken += 1
            index += 1
            continue
        rows, cursor, no_replay = found
        if no_replay:
            skipped += 1
        else:
            pairs.append(rows)
        index = cursor
    return pairs, broken, skipped


@dataclass(frozen=True)
class _ToolTail:
    """L5 装进去的东西，以及没装进去的东西。

    **对与行分开记**：``count`` 那一列画的是"条"（行），而"装不下几对"必须以对为单位
    说才有意义——把两个单位混进同一个数，账本上会是一个谁也核对不出来的数字。
    """

    rows: tuple[ChatMessage, ...] = ()
    tokens: int = 0
    pairs: int = 0
    """装进去的**对数**，也就是回放了几轮工具调用。"""
    dropped: int = 0
    """装不下而整对丢掉的**对数**。"""
    broken: int = 0
    """形状不完整、根本不成对的**行数**。与"装不下"分开记：一个指向额度太小，另一个
    指向落盘的数据坏了，混成一个数就分不出该修哪儿。"""
    skipped: int = 0
    """**成对但按策略不回放**的**对数**（闸门拒绝、或结果是个过期的快照）。

    第三种，理由见 :func:`_scan_tool_history`。它与 ``dropped`` 分开记的方向刚好相反：
    ``dropped`` 多说明额度该调大，``skipped`` 多说明这一轮里拒了几次、或查了几次
    "当前值"——两个都不该让账本上的数字看起来像同一件事。"""

    @property
    def note(self) -> str:
        parts = []
        if self.skipped:
            parts.append(f"{self.skipped} 对按策略不回放")
        if self.dropped:
            parts.append(f"{self.dropped} 对装不下")
        return f"另有{'、'.join(parts)}" if parts else ""


def _collect_tool_tail(history: Sequence[Message], ceiling: int, meter: TokenMeter) -> _ToolTail:
    """从新到旧装工具对，直到装不下为止。

    **装不下就停，不跳过。** 与 L2 的"跳过这一条继续看下一条"相反，理由与 L3 相同：
    工具对是一条时间线，中间留个洞等于对模型说"中间那步没发生过"。而 L2 的记忆条目
    彼此独立，跳过一条不影响别的。

    **"按策略不回放"不违反上面这条**：那些对在 :func:`_scan_tool_history` 里就已经
    出局了，它们从来没进过这条时间线——留洞是把一条本该完整的时间线挖空，而这是
    **根本不把那一步算进时间线**（"这一步在当轮就已经过去了"）。
    """
    pairs, broken, skipped = _scan_tool_history(history)
    kept: list[list[ChatMessage]] = []
    tokens = 0
    for pair in reversed(pairs):
        price = sum(meter.message(row) for row in pair)
        if tokens + price > ceiling:
            break
        kept.append(pair)
        tokens += price
    kept.reverse()
    return _ToolTail(
        rows=tuple(row for pair in kept for row in pair),
        tokens=tokens,
        pairs=len(kept),
        dropped=len(pairs) - len(kept),
        broken=broken,
        skipped=skipped,
    )


def assemble(
    history: Sequence[Message],
    *,
    current: str,
    budget: TokenBudget,
    persona: Sequence[str] = (),
    environment: str = "",
    recalled: Sequence[MemoryInjection] = (),
    meter: TokenMeter | None = None,
    tools_tokens: int = 0,
) -> Assembly:
    """把一次请求拼出来。

    Args:
        history: **之前的**消息，按时间顺序。不含当前这一轮的用户输入。
        current: 当前这一轮的用户输入。永远不被驱逐，也不受任何层额度的约束。
        budget: 额度。``usable`` 是装配器唯一能支配的数。
        persona: 人格锚的正文（通常是记忆库里 ``pinned`` 的那条）。**原样发出，
            一个字节都不改**——"任何模型读到的完全一样"这句话的落点就是它。
        environment: 环境说明（时间、模型、工作目录之类）。由调用方拼，因为只有它
            知道该说什么；这个模块不认识时钟。
        recalled: 召回结果，按得分从高到低。
        meter: 量具。传进来是为了让调用方与账本用**同一个**量具——两处各造一个的
            后果是账本上写着 184、实际发出去 240。
        tools_tokens: 本轮工具声明的开销，由调用方量好传进来（见
            :attr:`AssemblyLedger.tools_tokens`）。**带工具的那一轮必须传**：它不在
            ``messages`` 里，量具看不到它，默认 0 就是少算一笔已经确定要付的钱。
            Step 0 实测 3 个工具净增 311 token/轮，七个约两倍——而 ``usable`` 与
            ``input_limit`` 之间那道余量（``tolerance`` 0.10 ≈ 1400 token）本来只
            留给估计误差，被这笔钱吃掉一半之后就不再是余量了。
    """
    if not current:
        raise ValueError(
            "current 不能为空——这一轮没有用户输入就没有请求。"
            "（``ChatRequest`` 也不接受空消息序列，这个校验只是把错误提前到能看清"
            "调用点的地方。）"
        )

    meter = meter or TokenMeter()
    tools_tokens = max(0, tools_tokens)
    persona_text = "\n\n".join(text.strip() for text in persona if text.strip())
    environment_text = environment.strip()
    current_message = ChatMessage(role="user", content=current)

    # ── 不可驱逐的部分先扣 ────────────────────────────────────────────────
    # 顺序有意义：当前输入排在最前，因为它连"装不下"这个状态都不该进入。
    #
    # 系统消息的包装与生成提示**无条件预留**。调用方总会给人格锚（persona 为空
    # 只可能是配置还没读出来），而让这两个常数跟着内容的有无浮动，会让账本里的
    # 总额和各层之和对不上——那种差额没有任何人能对照明白。
    pool = (
        budget.usable
        - meter.message(current_message)
        - GENERATION_PROMPT
        - PER_MESSAGE_OVERHEAD
        - meter.text(persona_text)
        - meter.text(environment_text)
        - tools_tokens
    )

    notes: list[str] = []
    if pool <= 0:
        # 光是把不可驱逐的部分放进去就超了。**照样发出去**——用户的问题永远要发，
        # 而"什么都不发"比"发一条超额的请求"更糟。服务端会截断，账本如实记。
        notes.append(
            "不可驱逐的部分（人格锚 + 环境 + 本轮输入）已超出额度，记忆召回与历史摘要这一轮全部放弃"
        )
        pool = 0

    # ── L2 记忆（全部、带时间；装不下按得分从低到高丢） ──────────────────────
    # **没有份额。** 记忆以前被关在 usable 的 25% 里，那是自设的额度；现在它拿的是
    # 扣掉不可驱逐部分之后剩下的全部，装得下几条是几条。装不下时按得分从低到高丢，
    # 账本如实写「N 条没装下」——丢是**窗口真的满了**（官方限制），不是我们的配额。
    #
    # 装不下就**跳过它继续看下一条**，不退出循环：记忆是互相独立的条目，中间留个
    # 洞无所谓，而"一条很长的记忆把后面所有短的一起挡在门外"代价更大。
    # 账本那一栏记"这一层装箱时手上还剩多少"，所以要在这里取一次。
    recall_ceiling = pool
    injected: list[MemoryInjection] = []
    for item in sorted(recalled, key=lambda entry: -entry.score):
        trial = [*injected, item]
        if meter.text(_render_recall(trial)) > pool:
            continue
        injected = trial
    recall_cost = meter.text(_render_recall(injected))
    pool -= recall_cost

    # ── L3 近期对话（从最近往前收） ────────────────────────────────────
    # 排在记忆之后。**这是与旧版相反的取舍**：旧版把近处对话排在记忆之前，理由是
    # "记忆丢得起、近处对话丢了就真没了"；而用户要的是「无论哪一个对话都要能够使用
    # 完整的记忆」，记忆被丢掉就是这一轮真的不知道。丢掉的旧对话不是没了——它们进
    # L4 摘要，连续性还在。
    #
    # 这里**断掉就停，不跳过**——与 L2 相反。这一层是连续的对话，挖洞会让它读起来
    # 不再是一条序列（半句话在答一个已经不在上下文里的问题）。装不下的那些不是被
    # 丢掉，是被 L4 接管了。
    #
    # 工具行与带 ``tool_calls`` 的助手行在这里**被跳过**（``_to_chat`` 返回 ``None``），
    # 那不是挖洞：它们整对归 L5。**不能在这里停车**——最近一轮若调过工具，停车会让
    # 整个最近的工具轮次从 L3 消失，而 L3 正是"最近发生了什么"的那一层。
    #
    # 同时记下切分点 ``cut``：保留段从 ``history`` 的哪一条开始。**不能用"保留了几
    # 条"反推**——``history`` 里可能有工具行被跳过，也可能有开头的助手消息被清掉，
    # 两种都会让反推出来的切分点往前漂，把本该保留的对话又压进摘要。
    recent_ceiling = pool
    kept: list[tuple[int, ChatMessage]] = []
    recent_cost = 0
    for index in range(len(history) - 1, -1, -1):
        chat = _to_chat(history[index])
        if chat is None:
            continue
        cost = meter.message(chat)
        if recent_cost + cost > recent_ceiling:
            break
        kept.append((index, chat))
        recent_cost += cost
    kept.reverse()

    # 去掉开头连续的助手消息：反向遍历会在任意一条消息处断掉，于是最老的那条常常
    # 是助手回复，而它对应的用户那句话已经被压进摘要了。留着它，模型看到的第一条
    # 历史是一句没有问题的回答——这类错位**不会报错**，只会让模型把上一轮的回答
    # 当成开场白，然后顺着一个不存在的上下文往下说。不去查"它前面那条是不是被丢
    # 了"：开头是助手就是错的，无论前因是什么。
    while kept and kept[0][1].role == "assistant":
        kept.pop(0)
    cut = kept[0][0] if kept else len(history)
    kept_chat = [chat for _, chat in kept]
    recent_cost = sum(meter.message(chat) for chat in kept_chat)
    pool -= recent_cost

    # ── L5 工具结果（整对，从新到旧） ───────────────────────────────────
    # 额度是**剩下的全部**：它不再占固定份额（旧版 0.20），也就不需要再解释"这 0.20
    # 是从谁那儿拿的"。工具结果是这一轮任务的现场，自包含、当下最相关。
    tools_ceiling = pool
    tool_tail = _collect_tool_tail(history, tools_ceiling, meter)
    pool -= tool_tail.tokens

    # ── L4 历史摘要（拿剩下的全部） ─────────────────────────────────────
    # **这不是额度，是"前面花剩下的"。** 它的输入本来就是装不下的那些，前面省下来
    # 的钱正好该用在多覆盖几轮历史上。pool 归零时它就是空的——那是窗口真的满了。
    summary_ceiling = max(0, pool)
    # 工具行不进摘要（``_to_chat`` 对它们返回 ``None``）：它们要么在 L5 里原样回放，
    # 要么已经因为装不下而整对丢掉，压成一段散文两种都不沾。**这是取舍**——被丢掉的
    # 工具轮次因此不会以任何形式留下痕迹，只有 L5 那行的"另有 N 对装不下"提到它。
    dropped = [message for message in history[:cut] if _to_chat(message) is not None]
    summary_text = summarize(dropped, budget_tokens=summary_ceiling, meter=meter) if dropped else ""
    summary_cost = meter.text(summary_text)

    # ── 拼装 ──────────────────────────────────────────────────────────────
    system_text = "\n\n".join(
        part
        for part in (persona_text, environment_text, _render_recall(injected), summary_text)
        if part
    )
    messages: list[ChatMessage] = [ChatMessage(role="system", content=system_text)]
    # L5 排在 system 之后、L3 之前。这是相对真实时间顺序的**重排**（工具历史整体
    # 提前），**选它是因为**工具对是自包含的，而按时间把工具行插进 L3 会破坏 L3 的
    # 连续性，也会撞上"开头不能是 assistant"那条不变量。残余风险见计划书 §十一：
    # 探针里专门观察模型有无因此困惑，兜底方案是"只回放最近一对"、不动 L3。
    messages.extend(tool_tail.rows)
    messages.extend(kept_chat)
    messages.append(current_message)

    total = meter.messages(messages)
    if tool_tail.broken:
        notes.append(
            f"有 {tool_tail.broken} 行工具记录不构成完整的对（缺 tool_call_id、"
            f"结果对不上、或同一次调用被答复两次），已丢弃"
        )
    # 超出额度要连工具声明一起算：它不在 ``messages`` 里，但它花的是同一个窗口。
    over = max(0, total + tools_tokens - budget.usable)
    if over:
        notes.append(
            f"本轮估算 {total + tools_tokens} token（含工具声明 {tools_tokens}），"
            f"超出额度 {over}——服务端可能会截断最早的消息（人格锚的位置）"
        )

    ledger = AssemblyLedger(
        budget=budget,
        layers=(
            LayerReport(Layer.PERSONA, meter.text(persona_text)),
            LayerReport(Layer.ENVIRONMENT, meter.text(environment_text)),
            LayerReport(
                Layer.RECALL,
                recall_cost,
                recall_ceiling,
                count=len(injected),
                note=_dropped_note(len(recalled), len(injected)),
            ),
            LayerReport(Layer.RECENT, recent_cost, recent_ceiling, count=len(kept_chat)),
            LayerReport(Layer.SUMMARY, summary_cost, summary_ceiling),
            LayerReport(
                Layer.TOOLS,
                tool_tail.tokens,
                tools_ceiling,
                count=len(tool_tail.rows),
                note=tool_tail.note,
            ),
        ),
        injected=tuple(injected),
        kept_messages=len(kept_chat),
        dropped_messages=len(dropped),
        total_tokens=total,
        tools_tokens=tools_tokens,
        over_budget=over,
        notes=tuple(notes),
    )
    return Assembly(messages=tuple(messages), ledger=ledger)


def _dropped_note(available: int, injected: int) -> str:
    """被额度挡在外面的召回条数。

    **要写出来。** 不写的话，账本看起来像是"库里只有这几条相关"，而不是"还有几条
    装不下"——前者会让人去改召回算法，后者才指向该改的额度。
    """
    gap = available - injected
    return f"另有 {gap} 条装不下" if gap > 0 else ""
