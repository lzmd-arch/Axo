"""七个具体工具的内容。

前四个（``get_time`` / ``search_memory`` / ``read_memory`` / ``read_file``）
**结构上不可能造成损失**：读时钟、读记忆、读文件，没有一条会改变任何东西。

后三个（``write_memory`` / ``forget_memory`` / ``set_reminder``）都会改变状态，
所以 :attr:`requires_approval` 为真，每次调用都要用户点一次头。它们与另外四个的
分界写在 ``requires_approval`` 这一个字段上、由 :mod:`~biosnn_memterm.agent.gate`
读——**不在这里再写一张"哪些要确认"的表**，两张表迟早会不一致。

## 改状态的三个都**不是**删除，也不是承诺

- ``write_memory`` 只往库里加一条（还带着 ``via``/``call_id``，回看时认得出它
  是模型写的）。
- ``forget_memory`` 是**软遗忘**：改 ``tier``、留回程票，用户随时能恢复。真正的
  删除在本版里没有工具——它不可逆，而一次误点不该是不可逆的。
- ``set_reminder`` 只**记下来**，到点不会弹（触发是 P3b 的事）。所以它的工具描述
  里明说了"不要说「到点我会提醒你」"——用户是照着模型的措辞建立期待的，而
  "我记下了"和"我会叫你"在用户听来是两件不同的事，只有一个是真的。

## ``description`` 是提示词，不是注释

它唯一的读者是模型，而模型据此决定要不要调这个工具。所以每一条都写成"**什么时候
用它**"，并且尽量带一句"什么时候别用"——后者不是客套：Step 0 探针里模型最常犯的
错就是该闲聊的时候去查记忆，而工具描述是它唯一的信息来源。

## 每个工具的返回都必须是**有界**的

一次 ``search_memory`` 若把二十条记忆的全文倒进上下文，工具循环就会因为"用了工具"
而**比不用更差**——逐字稿被挤出去，用户刚说的话反而看不见了。所以每个工具都自带
一个上限，且**超限时明说**（见 :func:`_clip`）。

写口的那两个方向上相反：**写不下就拒绝，不截断**。截断一条记忆等于把半句话长期
存进库，而它以后会被当成完整的一句读出来——那个错误没有任何症状。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from memcore import MemoryKind, Taint
from memcore.recall import RecallConfig, recall

from ..llm.base import ToolSpec
from ..memory.forget import can_soft_forget, soft_forgotten
from ..memory.policy import MemoryDraft
from .fs import ReadFileError, read_text, resolve_within
from .tools import (
    Tool,
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    argument_int,
    argument_str,
)

__all__ = [
    "ForgetMemoryTool",
    "GetTimeTool",
    "ReadFileTool",
    "ReadMemoryTool",
    "SearchMemoryTool",
    "SetReminderTool",
    "WriteMemoryTool",
    "default_tools",
]

SEARCH_BUDGET_TOKENS = 800
"""``search_memory`` 允许把多少 token 的记忆放进这次工具结果。

**先验值**，与 ``max_rounds``、L5 的 0.20 份额同属待标定的一类（计划书 §八）。
取舍的方向是明确的：小了只是少给模型几条（它可以再查一次），大了会挤掉逐字稿，
而逐字稿是用户刚说的话。所以先取偏小的数。

它和 :data:`MAX_ITEMS` 是两道独立的闸：条目多但每条很短时先撞条数，反之先撞预算。
"""

MAX_ITEMS = 8
"""单次最多返回几条。与 :data:`~memcore.recall.RecallConfig.max_items` 的默认值
一致——模型可以显式要更少，但不该要更多：召回本身是贪心装满预算的，条目数翻倍
只会让每条都变短。"""

_ITEM_PREVIEW_CHARS = 160
_MEMORY_BODY_CHARS = 4000

MAX_WRITE_CHARS = 2000
"""一条工具写入的记忆最长多少字。**超了就拒绝，不截断。**

**先验值**（同 ``SEARCH_BUDGET_TOKENS``），但它挡的是一个具体的坏形态：模型把整段
对话抄进 ``content``。那样写的记忆在召回时与别的条目抢同一份预算，而它自己内部
又有九成是与本次无关的寒暄。2000 字够装下一个段落级别的事实——真需要更长的场合，
该分成几条写，而这正是拒绝时的回灌要引导它做的事。

**为什么不截断**：截断会让库里存着半句话，而它以后被当成完整的一句读出来。这条
错误没有症状，所以只能在门口拦住。
"""

MAX_REMINDER_CHARS = 200
"""一条提醒最长多少字。理由同上，尺度更小：提醒是"到点要看见的那一句"，
超过一句话就不是提醒了，是一个待办清单。"""

MAX_DELAY_MINUTES = 366 * 24 * 60
"""``delay_minutes`` 的上限（约一年）。

**它拦不住量纲错误，这一点要说清楚，免得后人以为这条线管用。**用户说「一小时后」，
模型若把 3600 秒当成分给过来，落在上限之内——收到的是一条两天半以后的提醒，而用户
要到明天才会发现。这条上界真正保证的只是：不会有提醒被静默地约到几十年后（那种值
几乎一定是一次算错或一次幻觉）。值是先验的。
"""


def _clip(text: str, limit: int) -> str:
    """截断并**留下痕迹**。

    与 :mod:`~biosnn_memterm.agent.fs` 的截断标记是同一条纪律：让模型读到半个东西
    而不告诉它，它会基于半个东西下结论，而那个结论在界面上看起来与完整的一样确定。
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（原文共 {len(text)} 字，已截断）"


def _weekday_cn(moment: datetime) -> str:
    return "一二三四五六日"[moment.weekday()]


class GetTimeTool:
    """现在几点。

    看起来最没用的一个工具，但它是**探针里的单工具正对照**：模型会不会为一句
    "现在几点"发出一次格式良好的工具调用，是这个整条链路能不能跑的第一个证据。
    保留它也是在真实使用里最便宜的一次工具往返——用户问时间时，模型答的日期必须
    来自这里，而不是来自它训练时的语料。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="get_time",
        description=(
            "取当前日期与时间。用户问「现在几点」「今天几号」「今天星期几」，"
            "或者你需要知道今天是哪一天时用它。**不要凭记忆回答日期。**"
        ),
        parameters={"type": "object", "properties": {}},
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        moment = ctx.clock()
        # ``replayable=False``：这个结果是**那一刻**的快照，而这个工具的全部语义就是
        # "当前值"。回放它比不回放更坏——模型会拿着十分钟前（或昨天）的时间戳回答
        # "现在几点"，而且看不出有什么不对。
        return ToolOutcome.success(
            f"{moment:%Y-%m-%d} 星期{_weekday_cn(moment)} {moment:%H:%M}（本地时间）",
            replayable=False,
        )


class SearchMemoryTool:
    """在长期记忆里检索。

    ``settle=False`` 是**选出来的**，不是默认值顺手带上的：``recall`` 的入参文档
    写着，结算会让"被看到的条目显得更常用、衰减得更慢"，而模型的一次试探性调用
    不该改变用户的记忆权重排序。代价是这次召回不进访问统计——如果模型真的因此
    想起来了什么并说给用户听，那次"想起来"不会被计分。**这是已知的取舍，标注为
    待标定**（计划书 §十一 风险 7）。

    **只走词法通道**（不传 ``query_vector``）。理由不是省事，是**有界性**：
    ``Tool.run`` 跑在 worker 线程里，而收尾只等 3 秒；算一次查询向量的 embedding
    是一次网络往返，把它放进工具里就等于把网络延迟放进了工具的时限。而
    ``recall`` 的入参文档明确写着"没有向量是一种正常状态，不是降级"——没装嵌入
    模型时系统本来就要照常可用。要加语义通道的那天，正确的位置是**循环之前**
    （用户那一轮本来就已经算过 embedding），不是工具内部。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="search_memory",
        description=(
            "在用户过往的记忆里检索相关条目。用户提到「我以前说过」「你还记得」"
            "「上次那个」，或者你在回答前需要知道用户之前告诉过你什么时用它。"
            "**闲聊和寒暄时不要用**；用户只是在问一句普通的话时也不要用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的内容，用自然语言写，不要只给关键词。",
                },
                "limit": {
                    "type": "integer",
                    "description": f"最多返回几条，默认 5，上限 {MAX_ITEMS}。",
                },
            },
            "required": ["query"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        query = argument_str(arguments, "query")
        limit = argument_int(arguments, "limit", default=5, minimum=1, maximum=MAX_ITEMS)
        assert limit is not None  # default 给了值，argument_int 不会返回 None

        ledger = recall(
            ctx.store,
            query_text=query,
            budget_tokens=SEARCH_BUDGET_TOKENS,
            config=RecallConfig(max_items=limit),
            settle=False,
        )
        if not ledger.injected:
            # 空结果必须**明说**。让模型自己去推断"没有输出"等于"没有找到"，就是
            # 在给它留出编一条的空间——Step 0 探针里最主要的失败形态正是编。
            return ToolOutcome.success(
                f"没有找到与「{_clip(query, 60)}」相关的记忆。",
                detail=f"召回账本：{ledger.summary()}",
            )

        lines = [
            f"{index}. [{memory.memory_id}] {_clip(memory.content, _ITEM_PREVIEW_CHARS)}"
            f"（{memory.kind}，权重 {memory.weight:.2f}）"
            for index, memory in enumerate(ledger.injected, start=1)
        ]
        header = f"找到 {len(ledger.injected)} 条相关记忆（用 read_memory 看完整内容）："
        return ToolOutcome.success("\n".join([header, *lines]), detail=ledger.summary())


class ReadMemoryTool:
    """按 id 读一条记忆的全文。"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="read_memory",
        description=(
            "按 id 读取一条记忆的完整内容。id 来自 search_memory 的结果。"
            "需要看某条记忆的全文而不是摘要时用它。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "search_memory 返回的方括号里的 id。",
                },
            },
            "required": ["memory_id"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        memory_id = argument_str(arguments, "memory_id")
        item = ctx.store.get(memory_id)
        if item is None:
            # 这条回灌要能指导下一步：告诉它"可能传错了"比只说"没有"有用，因为
            # 模型最常见的错因是**把摘要正文里的某个词当成 id**。
            return ToolOutcome.failure(
                f"没有 id 为 {memory_id!r} 的记忆。id 必须是 search_memory 返回的"
                "方括号里的那串字符，不能是正文里的词。"
            )
        created = datetime.fromtimestamp(item.created_at).strftime("%Y-%m-%d")
        return ToolOutcome.success(
            f"[{item.memory_id}]（{item.kind.value} · {item.tier.value} · "
            f"记于 {created}）\n{_clip(item.content, _MEMORY_BODY_CHARS)}"
        )


class ReadFileTool:
    """读一个文本文件。

    **没配可读目录时它不该出现在工具表里**——见 :func:`default_tools`。留一个
    "配置为空则返回失败"的分支是给硬调兜底：模型看不到它就不会调，而万一路径
    从别处来了，它该拿到一句解释而不是一个异常。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="read_file",
        description=(
            "读取用户允许目录里的一个文本文件。用户说「看看某个文件里写了什么」"
            "「那个文件的内容」时用它。路径要写**相对于允许目录**的路径。"
            "读不了会告诉你原因，按提示改路径再试。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对于允许目录的文件路径。"},
                "start_line": {
                    "type": "integer",
                    "description": "从第几行开始读，从 1 开始。文件很长时用它翻页。",
                },
                "max_lines": {"type": "integer", "description": "最多读多少行。"},
            },
            "required": ["path"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = argument_str(arguments, "path")
        start_line = argument_int(arguments, "start_line", default=1, minimum=1)
        max_lines = argument_int(arguments, "max_lines", default=200, minimum=1, maximum=2000)
        assert start_line is not None and max_lines is not None
        try:
            path = resolve_within(ctx.read_root, raw)
            body = read_text(path, start_line=start_line, max_lines=max_lines)
        except ReadFileError as exc:
            # ``ReadFileError`` 的消息本来就是写给模型看的，直接回灌。
            # 注意它**不是** ToolArgumentError：路径不合法是"这个世界的事实"
            # （文件不在那儿），不是"参数写错了"——两者的回灌语气不同。
            return ToolOutcome.failure(str(exc))
        # ``replayable=False``：读到的是**此刻**的文件内容，而文件随时会被改。回放它
        # 的代价是模型拿着旧正文下结论，且它没有任何线索知道那是旧的——比"重新读一次"
        # 贵得多（重读只是一次工具往返，而这里是悄悄答错）。
        return ToolOutcome.success(f"{path.name}：\n{body}", replayable=False)


def _tool_source_ref(ctx: ToolContext) -> str:
    """工具写入的溯源指针：``tool://<会话>/<轮次>/<call_id>``。

    与 ``MemoryWriter.source_ref`` 的 ``turn://…`` 分开，因为这条记忆不是"某一轮的
    产物"，是模型的一次显式请求。**同一轮里可以调两次 ``write_memory``**，所以轮次
    不够用，最后一截必须是 ``call_id``——而它由派发口绑定（见
    ``ToolRegistry.dispatch``），工具从参数里是拿不到的。
    """
    return f"tool://{ctx.session_id}/{ctx.turn_seq}/{ctx.call_id}"


class WriteMemoryTool:
    """按模型的显式请求写一条记忆。**过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="write_memory",
        description=(
            "把用户明确要求记住的一件事写进长期记忆。用户说「记住……」「帮我记一下」"
            "「以后别忘了」这一类话时用它。**内容会被长期保存并在以后的对话里被想起**，"
            "所以只在用户真的在让你记的时候用——不要自作主张地把聊天内容存下来。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "要记住的那件事，写成一句**自足**的话——将来单独看到它也要能看懂。"
                        "不要用「这个」「那个」指代对话里的东西，也不要把整段对话抄进来。"
                    ),
                },
            },
            "required": ["content"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        """写一条。**kind 固定 FACT、taint 固定 MODEL，都不让模型选。**

        ``kind`` 不给模型选，最硬的理由是 :attr:`~memcore.MemoryKind.PERSONA_ANCHOR`：
        那一类按 ``pinned=True`` 永远进上下文、**连遗忘都拦不住它**（见
        ``memory/forget.py`` 的说明）。让一次工具调用能把东西放进那一层，等于把
        人格锚的写入权交给模型。其余的 kind 差别没那么致命，但"选哪个 kind"是一条
        会影响召回排序的杠杆，而 P3a 没有任何量具能测它选得对不对——**先不给**，
        标注为待标定（计划书 §六）。

        ``taint=MODEL`` 是这条路径的重点：它记下"这句话是模型写的"，而"用户点过头"
        这件事由 ``meta`` 里的 ``via`` 与闸门的记录背书。两者都要有——只有 taint 的话
        看不出它经过了确认，只有 via 的话看不出内容是谁组织的。
        """
        content = argument_str(arguments, "content")
        if len(content) > MAX_WRITE_CHARS:
            raise ToolArgumentError(
                f"content 最多 {MAX_WRITE_CHARS} 字，收到 {len(content)} 字。"
                "一条记忆只装一件事——请挑出真正要记住的那一句，分几次写。"
            )
        if ctx.writer is None:
            # 装配期就缺了 writer（只跑只读工具的运行、或是接线漏了）。回灌一句话
            # 让模型如实告诉用户，而不是让它以为写成功了。
            return ToolOutcome.failure(
                "这次运行没有配置记忆写入器，写不了。请告诉用户这一条**没有**记下来。"
            )

        draft = MemoryDraft(
            content=content,
            kind=MemoryKind.FACT,
            taint=Taint.MODEL,
            meta={"via": "write_memory", "call_id": ctx.call_id},
        )
        item = ctx.writer.record_direct(
            draft, session_id=ctx.session_id, source_ref=_tool_source_ref(ctx)
        )
        if item is None:
            # ``record_direct`` 用 ``None`` 表示"没写进去"（词法指纹编不出来，
            # 见 ``_build``）。**这里必须说实话**：它的另一个调用方
            # （``MemoryWriter.record``）拿 ``None`` 当"跳过这一条"，照抄那个态度的话
            # 模型会说"记下了"而库里什么都没有——这个项目里最坏的一种失效。
            return ToolOutcome.failure(
                f"这条内容里没有可以编码的字符（全是标点或空白），**没有**记住：{content!r}。"
                "请换成一句有实际内容的话再试。"
            )
        return ToolOutcome.success(
            f"已记住 [{item.memory_id}]：{item.content}",
            # id 给模型是**有用的**：用户下一句说「刚才那条不算」，它要能接上
            # ``forget_memory``。（``set_reminder`` 那边就不给 id——本版没有取消提醒
            # 的工具，给了只会诱它编一个。）
            detail=f"source_ref={item.source_ref} kind={item.kind.value} taint={item.taint.value}",
        )


class ForgetMemoryTool:
    """软遗忘一条记忆。**过闸门。**

    **不是删除。** 改 ``tier=COLD`` 并留一张写在 ``meta`` 里的回程票，用户在记忆
    界面里随时能恢复。彻底删除在本版里**没有工具**：它不可逆，而一次误点（或者一次
    模型理解错）不该是不可逆的。用户真要删，界面里有那条路，而且要二次确认。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="forget_memory",
        description=(
            "让一条记忆不再被想起来。用户说「忘掉……」「别记着这个了」「以后别再提」时用。"
            "**它是可以撤销的**：记忆还在库里，用户随时能恢复——答应用户时说「不再用它"
            "回答你」，不要说「已经删了」。需要 id，从 search_memory 的结果里拿。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要忘掉的记忆 id，即 search_memory 返回的方括号里的那串字符。",
                },
            },
            "required": ["memory_id"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        memory_id = argument_str(arguments, "memory_id")
        item = ctx.store.get(memory_id)
        if item is None:
            # 与 ``read_memory`` 同一句提示：模型最常见的错因是把正文里的词当成了 id。
            return ToolOutcome.failure(
                f"没有 id 为 {memory_id!r} 的记忆。id 必须是 search_memory 返回的"
                "方括号里的那串字符，不能是正文里的词。"
            )

        reason = can_soft_forget(item)
        if reason is not None:
            # ``can_soft_forget`` 的消息是写给**界面**的（它末尾会叫人去按「彻底删除」
            # 那个按钮）。模型没有那个按钮，所以要把话补完：告诉它这件事得由用户做。
            # 不补的话，模型会以为是自己参数不对而反复重试。
            return ToolOutcome.failure(
                f"{reason}这一步要用户在记忆界面里操作，你做不到——请把情况如实告诉用户。"
            )

        at = ctx.clock().timestamp()
        # ``transform`` 而不是 ``get`` + ``update``：后者是整行覆盖，会与同一时刻的
        # 召回结算（衰减、访问计数）互相盖掉，见 ``memcore.store`` 里那条说明。
        forgotten = ctx.store.transform(memory_id, lambda it: soft_forgotten(it, at=at))
        return ToolOutcome.success(
            f"已不再使用 [{forgotten.memory_id}]：{_clip(forgotten.content, _ITEM_PREVIEW_CHARS)}"
            "。它还在库里，用户可以随时恢复。",
            detail=f"tier={forgotten.tier.value} at={at:.0f}",
        )


def _resolve_due(arguments: dict[str, Any], *, now: datetime) -> float:
    """把 ``due_at`` / ``delay_minutes`` 归成一个 epoch 秒。**两个都要有人管。**

    为什么要两种写法：本地小模型算日期很吃力，而「明天早上八点」和「半小时后」是
    用户真正常说的两类。前者只能靠模型自己算（先 ``get_time`` 拿到今天），后者是一个
    整数、几乎不会错。只留一种就是把一半的用例做成它做不好的形状。

    为什么**必须**给一个：一条没有时间的提醒不是提醒，是一句存下来的话——而它会顶着
    "提醒"的名字躺在文件里，P3b 的托盘扫到它时无事可做。
    """
    raw = arguments.get("due_at")
    has_text = isinstance(raw, str) and bool(raw.strip())
    has_delay = arguments.get("delay_minutes") is not None
    if has_text and has_delay:
        raise ToolArgumentError(
            "due_at 与 delay_minutes 只能给一个。相对时间用 delay_minutes，绝对时间用 due_at。"
        )
    if not has_text and not has_delay:
        raise ToolArgumentError(
            "要给一个时间：due_at（本地时间，如 2026-10-11 08:00）"
            "或 delay_minutes（从现在起多少分钟）。"
        )
    if has_delay:
        minutes = argument_int(arguments, "delay_minutes", minimum=1, maximum=MAX_DELAY_MINUTES)
        assert minutes is not None  # minimum/maximum 都给了，不是 None
        return now.timestamp() + minutes * 60.0

    assert isinstance(raw, str)
    text = raw.strip()
    if ":" not in text:
        # 只给日期的输入会被 ``fromisoformat`` **静默**读成那天的零点——一个我们替
        # 用户猜的时间。提醒是这个项目里唯一一次"到点"有后果的东西，不给它猜。
        raise ToolArgumentError(
            f"due_at {text!r} 里没有时分。请写成「2026-10-11 08:00」这样带时间的"
            "本地时间；今天是几号可以先用 get_time 查。"
        )
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ToolArgumentError(
            f"due_at {text!r} 读不出来（{exc}）。请写成「2026-10-11 08:00」这样带时分"
            "的本地时间，时分补足两位。"
        ) from exc
    # ``.timestamp()`` 对带时区与不带时区的输入都成立（不带时区按本地时间解释），
    # 于是下面与 ``now`` 的比较可以统一在 epoch 上做，不必去管两者是不是同一种
    # datetime——那种比较在混用时会抛 ``TypeError``，而它抛在工具里就是一次
    # "工具执行失败"的栈，谁看了都不知道是模型多写了个 +08:00。
    return moment.timestamp()


class SetReminderTool:
    """登记一条提醒。**过闸门。**

    **只登记，不触发。** 到点弹出来要有一个常驻进程，那是 P3b 的托盘；本版连
    ``paths.reminders_path()`` 那个文件都还没有人读它。所以这个工具的价值现在是
    "用户说过的事不会散落在聊天记录里"，而不是"到点叫你"。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="set_reminder",
        description=(
            "替用户记下一件**将来**要做的事。用户说「提醒我……」「晚点叫我」"
            "「别忘了……」时用它。\n"
            "**它只是把这件事记下来，到时间不会自己弹出来。**所以答应用户的时候"
            "说「我记下了」，**不要说「到点我会提醒你」或「到时候我叫你」**——你现在"
            "做不到这件事，说了就是骗他。\n"
            "时间二选一：绝对时间用 due_at（先用 get_time 查今天几号再算），"
            "相对时间用 delay_minutes。两个都给会被拒绝。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要提醒的事，一句话。将来单独看到它也要能看懂。",
                },
                "due_at": {
                    "type": "string",
                    "description": (
                        "什么时候提醒，本地时间，形如 2026-10-11 08:00（时分必须写全）。"
                        "**不要凭记忆写日期**，先用 get_time 查。"
                    ),
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": "从现在起多少分钟之后。适合「半小时后」这类相对时间。",
                },
            },
            "required": ["text"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        text = argument_str(arguments, "text")
        if len(text) > MAX_REMINDER_CHARS:
            raise ToolArgumentError(
                f"text 最多 {MAX_REMINDER_CHARS} 字，收到 {len(text)} 字。"
                "提醒是到点要看见的那一句，请压到一句话。"
            )
        if ctx.reminders is None:
            return ToolOutcome.failure(
                "这次运行没有配置提醒存储，记不了。请告诉用户这一条**没有**记下来。"
            )

        now = ctx.clock()
        now_ts = now.timestamp()
        due_ts = _resolve_due(arguments, now=now)
        if due_ts <= now_ts:
            # 已经过去的时间只有两种来源：模型算错了日期，或者它根本不知道今天几号。
            # 两种都**不能**收下——一条生下来就过期的提醒永远不会被触发，而用户会
            # 以为约好了。拒绝并把"现在几点"一起回灌，它下次就能算对。
            when = datetime.fromtimestamp(due_ts)
            raise ToolArgumentError(
                f"这个时间（{when:%Y-%m-%d %H:%M}）已经过去了，现在是 {now:%Y-%m-%d %H:%M}。"
                "提醒要约在未来。先用 get_time 查一下现在的日期和时间，再算一次。"
            )

        reminder = ctx.reminders.add(
            text,
            due_at=due_ts,
            created_at=now_ts,
            session_id=ctx.session_id,
            turn_seq=ctx.turn_seq,
            call_id=ctx.call_id,
        )
        when = datetime.fromtimestamp(reminder.due_at)
        return ToolOutcome.success(
            f"已记下：{reminder.text}（{when:%Y-%m-%d %H:%M}）。"
            "告诉用户时请说明这只是记下来，到时间不会自动弹出。",
            # 这里**不给** id：本版没有取消提醒的工具，给一个模型用不上的 id 只会
            # 诱它编一个「已经帮你取消了」。id 在 detail 里，人看得见。
            detail=f"reminder_id={reminder.reminder_id}",
        )


def default_tools(*, read_root: Sequence[Path] = ()) -> tuple[Tool, ...]:
    """P3a 的七个工具。

    ``read_file`` **只在配了可读目录时才登记**。这不只是安全考虑，也是一笔实用账：
    工具声明每一轮都要重发（Step 0 实测三个工具约 311 token/轮），把一个注定会
    失败的工具有效地告诉模型"你可以读文件"，只会换来它去试一次然后被拒。

    另外六个不受这个开关影响——它们要的东西（``ctx.writer`` / ``ctx.reminders``）
    是**运行时装配**的，不是用户配置。装配期真的缺了（只跑只读工具的运行）时，
    工具自己会在 ``run`` 里回一句"没有配置"（见各自的实现），而不是在这里被悄悄
    摘掉：那种"有时在有时不在"的工具表会让模型的失败经验互相矛盾。
    """
    return (
        GetTimeTool(),
        SearchMemoryTool(),
        ReadMemoryTool(),
        WriteMemoryTool(),
        ForgetMemoryTool(),
        SetReminderTool(),
        *((ReadFileTool(),) if read_root else ()),
    )
