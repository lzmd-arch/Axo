"""十个具体工具的内容。

读的那一族（``get_time`` / ``search_memory`` / ``read_memory`` / ``read_file`` /
``list_reminders``）**结构上不可能造成损失**：读时钟、读记忆、读文件、读那本提醒账，
没有一条会改变任何东西。

改状态的那一族（``write_memory`` / ``forget_memory`` / ``set_reminder`` /
``cancel_reminder``）都会改变状态，所以 :attr:`requires_approval` 为真，每次调用都要
用户点一次头。``look_at_screen`` **不改任何状态，也一样要点头**——它的判据不是"改不改
状态"，而是"这一步之后有没有东西离开了用户的机器"：整块屏幕的内容会交给模型看一次，
那是一次**不可撤回**的信息披露，而文件沙箱那条路在这里用不上（屏幕没有路径，见
``docs/agent.md``）。由此推出一条硬规矩：**永远不许出现"记住这次授权"的选项**。它们与另外四个的
分界写在 ``requires_approval`` 这一个字段上、由 :mod:`~axo.agent.gate`
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

import base64
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from memcore import MemoryKind, Taint
from memcore.recall import RecallConfig, recall

from ..llm.base import ToolSpec
from ..memory.forget import can_soft_forget, soft_forgotten
from ..memory.policy import MemoryDraft
from ..net import DEFAULT_COUNT, WebError
from ..stickers import Sticker, catalog_line, find_in
from .fs import (
    MAX_LIST_ENTRIES,
    MAX_SEARCH_RESULTS,
    ReadFileError,
    WriteFileError,
    list_entries,
    read_text,
    resolve_for_write,
    resolve_within,
    search_text,
    write_text_atomic,
)
from .reminders import (
    REPEAT_DAILY,
    REPEAT_ONCE,
    REPEAT_WEEKLY,
    Reminder,
    next_occurrence,
    repeat_label,
    weekday_name,
)
from .shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, run_command
from .tools import (
    Capture,
    Tool,
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    argument_int,
    argument_str,
)

__all__ = [
    "EditFileTool",
    "FetchUrlTool",
    "ForgetMemoryTool",
    "ListFilesTool",
    "ListRemindersTool",
    "CancelReminderTool",
    "GetTimeTool",
    "ReadFileTool",
    "ReadMemoryTool",
    "RunCommandTool",
    "SearchFilesTool",
    "SearchMemoryTool",
    "SetReminderTool",
    "SystemInfoTool",
    "UpdatePlanTool",
    "WebSearchTool",
    "WriteFileTool",
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

    与 :mod:`~axo.agent.fs` 的截断标记是同一条纪律：让模型读到半个东西
    而不告诉它，它会基于半个东西下结论，而那个结论在界面上看起来与完整的一样确定。
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（原文共 {len(text)} 字，已截断）"


def _weekday_cn(moment: datetime) -> str:
    return "一二三四五六日"[moment.weekday()]


class SendStickerTool:
    """发一张表情包。

    她自己找上来之外的另一条主动：在一轮回答里顺手丢一张图。表情包从数据根下的
    ``stickers/`` 来（一个文件一个、名字就是文件名）——**目录空着的时候这个工具根本
    不登记**（同 ``read_file`` 在没有可读根时的姿态：做不到的事不该出现在工具表里）。

    **它不产出给模型看的图**（``images`` 为空）：那张是给用户看的，走 ``preview``。
    模型知道自己发过哪一张就够了，没必要把图回灌给它"看"——那还会白烧一笔图 token。

    ``replayable=True``：与 ``get_time`` 那种"某一刻的快照"不同，"我发过一张开心的
    表情包"这件事在世界里没有变过，回放它是对的。
    """

    def __init__(self, stickers: Sequence[Sticker]) -> None:
        self._stickers = tuple(stickers)
        self.spec = ToolSpec(
            name="send_sticker",
            description=(
                "发一张表情包给对方。语气合适的时候（玩闹、安慰、道谢、无话可说）配一张，"
                "**不要每句话都发**。可用的表情包："
                f"{catalog_line(self._stickers)}。"
                "参数 name 必须是上面列出的名字之一。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "表情包的名字"},
                },
                "required": ["name"],
            },
        )

    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        wanted = arguments.get("name")
        name = wanted if isinstance(wanted, str) else ""
        sticker = find_in(self._stickers, name)
        if sticker is None:
            # **失败也要把可用的名单再说一遍**：模型多半是把名字记错了，而它手上
            # 没有别的办法知道这个名字不存在（工具描述里那份清单已经过去好几轮了）。
            return ToolOutcome.failure(
                f"没有叫「{name}」的表情包。可用的有：{catalog_line(self._stickers)}。"
            )
        return ToolOutcome.success(
            f"已发送表情包「{sticker.name}」。",
            preview=str(sticker.path),
        )


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

    **登记，而且到点真的会被触发**——``ui/reminder_scheduler.py`` 到点会弹一条托盘
    气泡、并在对话区留一条。

    这个 docstring 与下面 :attr:`spec` 的描述原先写着「只登记，不触发……本版连
    ``paths.reminders_path()`` 那个文件都还没有人读它」与「到时间不会自己弹出来……
    说了就是骗他」。那是 P3a 的话：那时确实没有东西在读那个文件。**P3b 把触发实现
    之后，没有人回来改这三处文案**（还有 ``run`` 的回灌），于是模型一直照着一句过期
    的话对用户说"到点它不会自己弹出来"——而它其实早就会了。这一版把它们一起改成
    实话，并**只承诺它真做得到的那一件**。

    重复见 :attr:`spec` 的 ``repeat`` 参数；取消与列出是另外两个工具
    (:class:`CancelReminderTool` / :class:`ListRemindersTool`)。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="set_reminder",
        description=(
            "替用户记下一件**将来**要做的事。用户说「提醒我……」「晚点叫我」"
            "「别忘了……」时用它。\n"
            "**到点了它真的会弹出来提醒他**——所以可以说「到点我叫你」「我会提醒你」。"
            "但**不要承诺别的**：它弹出来的是一句提醒，不是你去替他把那件事做了。\n"
            "要每天或每周重复时给 repeat。\n"
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
                        "**要重复时，这里写第一次发生的那一次**——重复的星期几由它决定。"
                    ),
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": "从现在起多少分钟之后。适合「半小时后」这类相对时间。",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly"],
                    "description": (
                        "要不要重复。once（默认）= 只响一次；daily = 每天这个时候；"
                        "weekly = 每周这个时候。**weekly 时 due_at 写第一次那一天就够了**"
                        "（「每周三 22:30」就写下一个周三的 22:30），星期几由它推出来，"
                        "不必自己算——算错星期几是这里最常见的一种错。"
                    ),
                },
                "weekdays": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "只有「每周一三五」这种一次要定好几个星期几时才给"
                        "（0=周一 … 6=周日），而且**必须包含 due_at 那一天**，否则会被拒绝。"
                        "平时不要给。"
                    ),
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

        repeat = _resolve_repeat(arguments)
        weekdays = _resolve_weekdays(arguments, repeat=repeat, due_ts=due_ts)

        reminder = ctx.reminders.add(
            text,
            due_at=due_ts,
            created_at=now_ts,
            session_id=ctx.session_id,
            turn_seq=ctx.turn_seq,
            call_id=ctx.call_id,
            repeat=repeat,
            weekdays=weekdays,
        )
        when = datetime.fromtimestamp(reminder.due_at)
        label = repeat_label(reminder)
        if label:
            head = f"已记下：{reminder.text}（{label}，下一次 {when:%Y-%m-%d %H:%M}）"
        else:
            head = f"已记下：{reminder.text}（{when:%Y-%m-%d %H:%M}）"
        return ToolOutcome.success(
            f"{head}。到点会提醒他。",
            # **id 也放进正文里**（早先刻意不给，因为那时没有取消的工具，给一个用不上
            # 的 id 只会诱模型编一个"已经帮你取消了"）。现在有 ``cancel_reminder`` 了，
            # 而用户下一句很可能就是「刚才那条取消掉」——不给她 id，她只能先去
            # ``list_reminders`` 里翻一遍。
            detail=f"reminder_id={reminder.reminder_id}",
        )


def _resolve_repeat(arguments: dict[str, Any]) -> str:
    """读 ``repeat``。缺省与未知值都当一次性——**与 :meth:`Reminder.from_dict` 同向**：
    这一格只能往"少响"那一侧落。

    取值校验在这里做而不是靠 JSON Schema 的 ``enum``：``ToolRegistry.dispatch`` 只查
    ``required``（``_missing_required``），``enum`` 没人管——这是既有约定。
    """
    raw = arguments.get("repeat")
    if raw is None or raw == "":
        return REPEAT_ONCE
    if not isinstance(raw, str) or raw not in (REPEAT_DAILY, REPEAT_WEEKLY):
        raise ToolArgumentError(
            f"repeat 只能是 once / daily / weekly，收到 {raw!r}。不想重复就不要给这个参数。"
        )
    return raw


def _resolve_weekdays(arguments: dict[str, Any], *, repeat: str, due_ts: float) -> tuple[int, ...]:
    """读 ``weekdays``。**只在 weekly 时才有意义**，而且必须包含 ``due_at`` 那一天。

    「必须包含」这条判据不是洁癖：不包含的话，用户说「每周三 22:30」而模型给了
    ``[0]``，得到的是"每周一 22:30"——一条**从第一次就不对**的提醒，而它到点会真的
    弹出来。那时用户看到的是一个他没约过的时间。
    """
    raw = arguments.get("weekdays")
    if raw is None:
        return ()
    if repeat != REPEAT_WEEKLY:
        raise ToolArgumentError("weekdays 只在 repeat=weekly 时有意义，去掉它。")
    if not isinstance(raw, (list, tuple)):
        raise ToolArgumentError("weekdays 要写成数组，例如 [0, 2, 4]。")
    days: list[int] = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, int) or not 0 <= entry <= 6:
            raise ToolArgumentError(
                f"weekdays 里每一项都要是 0..6 的整数（0=周一 … 6=周日），收到 {entry!r}。"
            )
        if entry not in days:
            days.append(entry)
    if not days:
        raise ToolArgumentError(
            "weekdays 是空的。要么去掉它（就按 due_at 那一天的星期几），要么给至少一个。"
        )
    first = time.localtime(due_ts).tm_wday
    if first not in days:
        names = "、".join(weekday_name(day) for day in sorted(days))
        raise ToolArgumentError(
            f"due_at 那一天是{weekday_name(first)}，而 weekdays 是 {names}——"
            "第一次发生的时间必须落在 weekdays 里。要么把 due_at 改成其中一天的同一个"
            "时刻，要么把那一天加进 weekdays。"
        )
    return tuple(days)


MAX_LISTED = 20
"""``list_reminders`` 一次最多列几条。超出的折成"还有 N 条"——与
``reminders.missed_summary`` 同一条纪律：**不静默丢，也不必一次说二十遍**。"""


class ListRemindersTool:
    """列出还没被提醒的事。**不过闸门——它一个字节都不改。**

    按**下一次**发生的时间排，不按登记顺序：用户此刻想的是"接下来有什么事"，而那与
    "我哪天约的"没有关系（同 ``due_now`` 那条排序的理由）。

    **它是"某一刻那本账里有什么"的快照**，所以 ``replayable=False``：账随时会变
    （用户在大厅里跟她说一句"那条取消了"就变了），而回放它会让她拿着一份旧清单说
    "你还有三条"——一句**看不出是旧话的假话**。代价是每轮重读一次文件，那是几十行
    的解析。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="list_reminders",
        description=(
            "列出用户现在**还没被提醒**的那些事。用户问「我约了什么」「有哪些提醒」"
            "「都记了些什么事」时用它。\n"
            "**在 cancel_reminder 之前也要先用它拿到 id**——那个 id 只能从这里来，"
            "不要自己编。\n"
            "它只读，不改任何东西。"
        ),
        parameters={"type": "object", "properties": {}},
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        del arguments
        if ctx.reminders is None or ctx.fired is None or ctx.cancellations is None:
            return ToolOutcome.failure(
                "这次运行没有配置提醒存储，看不到。请告诉用户这一条**没有**列出来。"
            )
        fired = ctx.fired.fired_keys()
        cancelled = ctx.cancellations.cancelled_ids()

        rows: list[tuple[float, Reminder]] = []
        for one in ctx.reminders.all():
            if one.reminder_id in cancelled:
                continue
            at = next_occurrence(one, fired)
            if at is None:
                continue
            rows.append((at, one))
        rows.sort(key=lambda pair: (pair[0], pair[1].reminder_id))

        if not rows:
            # **"别再去找了"那句是必需的。** 让模型自己从"空输出"推"没有找到"，就是
            # 给它留出编一条的空间（同 ``search_memory`` 那条防空转）。
            return ToolOutcome.success(
                "现在没有任何待办的提醒。**别再去找了**——直接告诉用户没有。",
                replayable=False,
            )

        lines = []
        for index, (at, one) in enumerate(rows[:MAX_LISTED], start=1):
            when = datetime.fromtimestamp(at)
            label = repeat_label(one)
            prefix = f"{label}，" if label else ""
            lines.append(
                f"{index}. [{one.reminder_id}] {prefix}{one.text}（下一次 {when:%Y-%m-%d %H:%M}）"
            )
        body = "\n".join(lines)
        if len(rows) > MAX_LISTED:
            body += f"\n（还有 {len(rows) - MAX_LISTED} 条没列出来）"
        return ToolOutcome.success(
            f"现在有 {len(rows)} 条待办：\n{body}",
            detail=f"共 {len(rows)} 条",
            replayable=False,
        )


class CancelReminderTool:
    """撤掉一条提醒。**过闸门。**

    过闸门的判据不是"它改状态"这么笼统，而是：**它取消的是一个未来的动作**，而误取消
    的后果用户要等到错过才发现——那正是最该有一个人点头的一类。它同时让
    ``ui/agent_worker.py`` 那条推理（"改状态的工具一律过闸门，所以停止之后那一批里
    剩下的最多是几个只读查询"）继续成立。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="cancel_reminder",
        description=(
            "撤掉一条还没到点的提醒。用户说「别提醒我了」「那条取消掉」「不用了」时用它。\n"
            "**需要 id**：先用 list_reminders 查出来，不要自己编。\n"
            "**它是永久的**——撤掉之后要重新约，得再说一次。\n"
            "告诉用户时说「不会再提醒你了」，**不要说你把记录删掉了**。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "reminder_id": {
                    "type": "string",
                    "description": "list_reminders 列出来的方括号里的那串字符。",
                }
            },
            "required": ["reminder_id"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        reminder_id = argument_str(arguments, "reminder_id")
        if ctx.reminders is None or ctx.cancellations is None:
            return ToolOutcome.failure(
                "这次运行没有配置提醒存储，取消不了。请告诉用户这一条**没有**撤掉。"
            )
        if reminder_id in ctx.cancellations.cancelled_ids():
            return ToolOutcome.success(
                f"id 为 {reminder_id} 的那条**已经取消过了**（列表里已经没有它）。"
                "告诉用户它不会再响，不必再撤一次。"
            )
        target = next((one for one in ctx.reminders.all() if one.reminder_id == reminder_id), None)
        if target is None:
            return ToolOutcome.failure(
                f"没有 id 为 {reminder_id} 的待办提醒。"
                "先用 list_reminders 看一眼当前有哪些，再原样传它方括号里的那串字符。"
            )

        ctx.cancellations.mark_cancelled([reminder_id], ctx.clock().timestamp())
        label = repeat_label(target)
        if label:
            # **这一条是这个工具最要紧的一句。** 取消一条重复提醒，用户的理解是
            # "以后都没有了"，而实现的形状是"这一条再也不会响"——两者一致，但回灌
            # 必须让模型说对，否则它会说「今天先不提醒，明天再说」这种它做不到的话。
            return ToolOutcome.success(
                f"已取消：{label} 的「{target.text}」以后都不会再提醒他了。"
                "这条取消是永久的——要重新约得再说一次。"
            )
        when = datetime.fromtimestamp(target.due_at)
        return ToolOutcome.success(
            f"已取消：{when:%Y-%m-%d %H:%M} 的「{target.text}」不会再到点提醒他。"
        )


class LookAtScreenTool:
    """看一眼用户**当前的屏幕**。**过闸门，而且一次一授权。**

    为什么是"一次一授权"而不是"记一次就一直能看"：这一次点头是**全部的约束**。文件
    那条路上还有沙箱（``agent/fs.py`` 把请求解析到可读根之内），而屏幕没有文件名、
    没有目录、没有边界——内容里有密码、有别人的窗口、有用户正打到一半的那句话。
    所以这个工具**不允许**有任何"以后别再问了"的选项：一旦有，这道闸门就从"每次都要
    人做的一个动作"退化成"填过一次的表单"。

    **它的结果一定是 ``replayable=False``**：截图是某一刻的快照（同 ``get_time``/
    ``read_file``），而更要紧的是**图本身回放不了**——逐字稿只存正文，L5 重建行的
    时候也造不出图。不标的话，下一轮模型会收到一条"你看过屏幕"却没有图的记录，然后
    开始编。

    **一轮里只带得进一张图**：图在工具尾巴的额度里占约六成（``budget.IMAGE_TOKENS``），
    两张就装不下。第二张不会报错，只是**不会进上下文**（循环那边会说明）——所以描述里
    明说了"不要连着调两次"。
    """

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="look_at_screen",
        description=(
            "看一眼用户**当前屏幕上是什么**（整块屏幕，Axo 自己的窗口会被裁掉）。"
            "用户明确让你看屏幕（「你看一下我的屏幕上写了什么」）、或者你确实需要屏幕上的"
            "信息才能回答时用它。**不要用它去确认你从对话里就能知道的事**，也**不要连着"
            "调两次**——一轮里只有第一张图带得进上下文。"
        ),
        parameters={"type": "object", "properties": {}},
    )
    requires_approval: ClassVar[bool] = True
    #: 卡片上那句话。这个工具**没有参数**，模板生成不出任何有用的话
    #: （``gate.summarize`` 那边用它，见那里"没有参数的工具"那一段）。
    approval_summary: ClassVar[str] = (
        "把整块屏幕的内容交给模型看一次（Axo 自己的窗口会被裁掉，其余一切都看得见）"
    )

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        capture = ctx.capture
        if capture is None:
            return ToolOutcome.failure(
                "这次运行没有接上截图通道，看不了屏幕。请如实告诉用户这一条**没有**看成。",
                replayable=False,
            )
        shot: Capture = capture()
        if not shot.ok:
            return ToolOutcome.failure(
                f"这一次没有截到屏幕：{shot.reason}。请如实告诉用户，不要猜屏幕上有什么。",
                replayable=False,
            )
        assert shot.png is not None  # ok 为真就一定有图
        encoded = base64.b64encode(shot.png).decode("ascii")
        size = f"{shot.width}×{shot.height}"
        return ToolOutcome.success(
            f"截了一张屏幕图（{size}），随这条结果一起给你。",
            detail=f"屏幕截图 {size}",
            replayable=False,
            images=(encoded,),
        )


class ListFilesTool:
    """列一个目录。**只读，不过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="list_files",
        description=(
            "列出允许目录里某个文件夹的内容。用户说「那个文件夹里有什么」「都有哪些文件」"
            "时用它。不传 path 就列第一个允许目录的根。目录名带结尾斜杠。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于允许目录的文件夹路径；不传 = 根。",
                },
                "limit": {"type": "integer", "description": "最多列多少项（默认 200）。"},
            },
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = argument_str(arguments, "path", required=False) or "."
        limit = argument_int(
            arguments, "limit", default=MAX_LIST_ENTRIES, minimum=1, maximum=MAX_LIST_ENTRIES
        )
        assert limit is not None
        try:
            path = resolve_within(ctx.read_root, raw)
            names, total = list_entries(path, limit=limit)
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        body = "\n".join(names) if names else "（空目录）"
        if total > len(names):
            body += f"\n\n…还有 {total - len(names)} 项没列出来（可以调大 limit 或在子目录里列）"
        return ToolOutcome.success(f"{path}（共 {total} 项）：\n{body}", replayable=False)


class SearchFilesTool:
    """在允许目录里按正则搜文本。**只读，不过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="search_files",
        description=(
            "在允许目录里按正则搜索文件内容，返回命中的文件、行号与那一行。用户说"
            "「哪里写了……」「帮我找一下包含……的文件」时用它。文件很多时会提前停下并说明。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式。"},
                "path": {
                    "type": "string",
                    "description": "搜索起点（相对路径）；默认第一个允许目录。",
                },
                "glob": {"type": "string", "description": "只搜匹配这个通配符的文件，例如 *.py。"},
                "max_results": {
                    "type": "integer",
                    "description": "最多返回多少条命中（默认 20）。",
                },
            },
            "required": ["pattern"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pattern = argument_str(arguments, "pattern")
        raw = argument_str(arguments, "path", required=False) or "."
        glob_pattern = argument_str(arguments, "glob", required=False) or "*"
        limit = argument_int(
            arguments, "max_results", default=20, minimum=1, maximum=MAX_SEARCH_RESULTS
        )
        assert limit is not None
        try:
            base = resolve_within(ctx.read_root, raw)
            hits = search_text(base, pattern, glob_pattern=glob_pattern, max_results=limit)
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if not hits:
            return ToolOutcome.success(f"没有找到匹配 {pattern!r} 的内容。", replayable=False)
        lines = [f"{hit.path}:{hit.line_no}: {hit.line}" for hit in hits]
        body = "\n".join(lines)
        if len(hits) >= limit:
            body += f"\n\n（只给了前 {limit} 条；要更多请换更具体的关键词或调大 max_results）"
        return ToolOutcome.success(body, replayable=False)


class SystemInfoTool:
    """这台机器的基本情况。**只读，不过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="system_info",
        description=(
            "查看这台机器的大致情况：操作系统、Python 版本、当前目录、磁盘剩余空间。"
            "用户问「你跑在什么机器上」「磁盘还有多少」时用它。**不要用它代替告诉用户"
            "具体某个文件的内容。**"
        ),
        parameters={"type": "object", "properties": {}},
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        lines = [
            f"系统：{platform.platform()}",
            f"Python：{sys.version.split()[0]}",
            f"当前目录：{Path.cwd()}",
        ]
        root = ctx.data_root or Path.cwd()
        try:
            usage = shutil.disk_usage(root)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            lines.append(f"数据盘（{root.anchor}）：剩余 {free_gb:.1f} GB / 共 {total_gb:.1f} GB")
        except OSError:
            pass
        return ToolOutcome.success("\n".join(lines), replayable=False)


class WebSearchTool:
    """联网搜索（DeepSeek 原生）。**只读，但会让关键词离开本机。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="web_search",
        description=(
            "联网搜索最新信息。用户问「最近」「现在」「查一下」这类需要网络的问题时用它。"
            "搜索会把关键词发给搜索服务；结果里带来源网址，回答时要说明来源。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索词（最多 128 字）。"},
                "count": {"type": "integer", "description": "要几条结果（默认 8，最多 50）。"},
            },
            "required": ["query"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        query = argument_str(arguments, "query")
        count = argument_int(arguments, "count", default=DEFAULT_COUNT, minimum=1, maximum=50)
        assert count is not None
        if ctx.net is None:
            return ToolOutcome.failure("这一轮没有联网搜索通道。")
        try:
            hits = ctx.net.search(query, count=count)
        except WebError as exc:
            return ToolOutcome.failure(str(exc))
        if not hits:
            return ToolOutcome.success("搜索没有返回结果。", replayable=False)
        lines = []
        for index, hit in enumerate(hits, start=1):
            summary = hit.summary or hit.snippet
            lines.append(f"{index}. {hit.title}\n   {hit.url}\n   {summary}")
        return ToolOutcome.success("\n".join(lines), replayable=False)


class FetchUrlTool:
    """抓一页正文（DeepSeek 原生 Fetch）。**只读。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="fetch_url",
        description=(
            "抓取一个网页的正文（Markdown）。需要读某篇具体页面时用它；只知道要查什么时"
            "先用 web_search。正文可能被截断，会明确说明。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http/https 地址。"},
            },
            "required": ["url"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        url = argument_str(arguments, "url")
        if ctx.net is None:
            return ToolOutcome.failure("这一轮没有联网抓取通道。")
        try:
            title, content = ctx.net.fetch(url)
        except WebError as exc:
            return ToolOutcome.failure(str(exc))
        head = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
        return ToolOutcome.success(head + content, replayable=False)


class UpdatePlanTool:
    """把多步任务的计划摆出来。**不改任何外部状态。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="update_plan",
        description=(
            "当一件事需要三步以上时，先用它列出步骤；之后每完成一步再更新一次状态。"
            "步骤状态只有 pending / in_progress / completed。用户会看到这份清单，"
            "它不会被写进长期记忆。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "这一步做什么。"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["step", "status"],
                    },
                    "description": "按顺序排列的步骤，最多 12 步。",
                },
            },
            "required": ["steps"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = arguments.get("steps")
        if not isinstance(raw, list) or not raw:
            raise ToolArgumentError("steps 必须是一个非空数组。")
        if len(raw) > 12:
            raise ToolArgumentError("steps 最多 12 步，请合并。")
        steps: list[dict[str, str]] = []
        for index, entry in enumerate(raw, start=1):
            if not isinstance(entry, dict):
                raise ToolArgumentError(f"第 {index} 步不是对象。")
            step = str(entry.get("step") or "").strip()
            status = str(entry.get("status") or "").strip()
            if not step or status not in ("pending", "in_progress", "completed"):
                raise ToolArgumentError(f"第 {index} 步缺少 step，或 status 不在允许值里。")
            steps.append({"step": step[:80], "status": status})
        lines = [f"{index}. [{one['status']}] {one['step']}" for index, one in enumerate(steps, 1)]
        return ToolOutcome.success(
            "计划已更新：\n" + "\n".join(lines),
            replayable=False,
            plan=tuple(steps),
        )


def _argument_flag(arguments: dict[str, Any], key: str) -> bool:
    """取一个布尔参数。**不猜**：字符串、数字都算参数写错。"""
    value = arguments.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ToolArgumentError(
            f"{key} 必须是布尔值，收到的是 {type(value).__name__}（{value!r}）。"
        )
    return value


class WriteFileTool:
    """写一个文本文件。**过闸门**：它会改变磁盘上的东西。"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="write_file",
        description=(
            "把一段文本写进用户允许目录里的文件（覆盖或追加）。用户说「把这个存下来」"
            "「写到那个文件里」时用它。**会改磁盘上的内容**，所以每次都会先问用户。"
            "路径要相对于允许目录；要新建子目录时把 create_dirs 设为 true。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对于允许目录的文件路径。"},
                "content": {"type": "string", "description": "要写入的完整文本。"},
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "overwrite（默认）覆盖，append 追加。",
                },
                "create_dirs": {"type": "boolean", "description": "父目录不存在时是否新建。"},
            },
            "required": ["path", "content"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = argument_str(arguments, "path")
        content = argument_str(arguments, "content", required=False)
        mode = argument_str(arguments, "mode", required=False) or "overwrite"
        create_dirs = _argument_flag(arguments, "create_dirs")
        if mode not in ("overwrite", "append"):
            raise ToolArgumentError(f"mode 只能是 overwrite 或 append，收到 {mode!r}。")
        if not ctx.write_root:
            return ToolOutcome.failure("写入工具没有配置可写目录。")
        deny = (ctx.data_root,) if ctx.data_root is not None else ()
        try:
            path = resolve_for_write(ctx.write_root, raw, create_dirs=create_dirs, deny=deny)
            written = write_text_atomic(
                path, content, append=(mode == "append"), create_dirs=create_dirs
            )
        except WriteFileError as exc:
            return ToolOutcome.failure(str(exc))
        verb = "追加" if mode == "append" else "写入"
        return ToolOutcome.success(f"已{verb} {path}（{written} 字节）。", replayable=False)


class EditFileTool:
    """按精确文本替换改一个文件。**过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="edit_file",
        description=(
            "在文件里把一段**精确文本**替换成另一段。适合小改动；要整份重写时用"
            " write_file。old_text 必须在文件里出现且（默认）只出现一次——出现零次或"
            "多次会拒绝并说明，不会猜你想改哪一处。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对于允许目录的文件路径。"},
                "old_text": {"type": "string", "description": "要替换掉的原文（必须精确匹配）。"},
                "new_text": {"type": "string", "description": "替换成什么；空串 = 删除。"},
                "replace_all": {"type": "boolean", "description": "出现多次时是否全部替换。"},
            },
            "required": ["path", "old_text"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = argument_str(arguments, "path")
        old_text = argument_str(arguments, "old_text")
        new_text = argument_str(arguments, "new_text", required=False)
        replace_all = _argument_flag(arguments, "replace_all")
        if not ctx.write_root:
            return ToolOutcome.failure("写入工具没有配置可写目录。")
        deny = (ctx.data_root,) if ctx.data_root is not None else ()
        try:
            path = resolve_for_write(ctx.write_root, raw, deny=deny)
        except WriteFileError as exc:
            return ToolOutcome.failure(str(exc))
        if not path.is_file():
            return ToolOutcome.failure(f"文件不存在：{path}")
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolOutcome.failure(f"读不了 {path.name}：{exc}")
        count = original.count(old_text)
        if count == 0:
            return ToolOutcome.failure(f"在 {path.name} 里没有找到那段原文，文件没有改动。")
        if count > 1 and not replace_all:
            return ToolOutcome.failure(
                f"那段原文在 {path.name} 里出现了 {count} 次。请给更长的上下文让它唯一，"
                "或者把 replace_all 设为 true 全部替换。"
            )
        revised = original.replace(old_text, new_text, -1 if replace_all else 1)
        try:
            write_text_atomic(path, revised)
        except WriteFileError as exc:
            return ToolOutcome.failure(str(exc))
        return ToolOutcome.success(
            f"已修改 {path.name}（替换 {count if replace_all else 1} 处）。", replayable=False
        )


class RunCommandTool:
    """在本机跑一条命令。**过闸门，且只有设置里打开才登记。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="run_command",
        description=(
            "在允许目录里执行一条 shell 命令，拿回输出与退出码。查版本、跑测试、"
            "看目录树这类事用它。**会在这台机器上真正执行**，所以每次都会先问用户；"
            "超时默认 30 秒（最多 120），用户随时可以停止。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令。"},
                "cwd": {
                    "type": "string",
                    "description": "工作目录（相对路径）；默认第一个允许目录。",
                },
                "timeout_s": {"type": "integer", "description": "超时秒数，默认 30，最多 120。"},
            },
            "required": ["command"],
        },
    )
    requires_approval: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        command = argument_str(arguments, "command")
        raw_cwd = argument_str(arguments, "cwd", required=False) or "."
        timeout_s = argument_int(
            arguments,
            "timeout_s",
            default=DEFAULT_TIMEOUT_S,
            minimum=1,
            maximum=MAX_TIMEOUT_S,
        )
        assert timeout_s is not None
        if not ctx.allow_commands:
            return ToolOutcome.failure("命令执行没有打开（设置 → 工具）。")
        if not ctx.write_root:
            return ToolOutcome.failure("命令执行需要至少一个允许目录。")
        try:
            cwd = resolve_within(ctx.write_root, raw_cwd)
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if not cwd.is_dir():
            return ToolOutcome.failure(f"{cwd} 不是目录。")
        result = run_command(command, cwd=cwd, timeout_s=timeout_s, cancel=ctx.cancel)
        head = f"退出码 {result.exit_code}，用时 {result.duration_ms} ms"
        if result.timed_out:
            head = f"**超时**（超过 {timeout_s} 秒），已终止。{head}"
        elif result.cancelled:
            head = f"**被用户停止**。{head}"
        body = result.output.strip() or "（没有输出）"
        text = f"{head}\n\n{body}"
        if result.ok:
            return ToolOutcome.success(text, replayable=False)
        return ToolOutcome.failure(text, replayable=False)


def default_tools(
    *,
    read_root: Sequence[Path] = (),
    vision: bool = False,
    stickers: Sequence[Sticker] = (),
    plugins: Sequence[Tool] = (),
    web: Any | None = None,
    write_root: Sequence[Path] = (),
    allow_commands: bool = False,
) -> tuple[Tool, ...]:
    """十个工具。

    **两个工具按条件登记**，判据是"这个环境里它根本不可能成功"：

    * ``read_file`` 要有可读目录。这不只是安全考虑，也是一笔实用账：工具声明每一轮
      都要重发（Step 0 实测三个工具约 311 token/轮），把一个注定会失败的工具有效地
      告诉模型"你可以读文件"，只会换来它去试一次然后被拒。
    * ``look_at_screen`` 要**当前的模型看得见图**（``vision``）。
    插件的工具由 ``plugins`` 传进来（装在插件目录里的那些，见 :mod:`axo.plugins`）
    ——**它们与内置工具同一套闸门**，这一层不为它们额外加权限判断。同一条账，而且多一层：
      模型看不见图时登记它，除了白烧一段 prompt，还会**弹一张注定白点的确认卡片**——
      用户点完头才发现什么都不会发生。``vision`` 由调用方从后端报的 ``capabilities``
      里算（见 ``ui/main_window.py`` 的 ``_tools()``）；**问不到能力时不登记**：
      "看不到工具"是可见且可解释的，而"登记了但每次必败"是把一次静默失败叠在提示词上。

    另外八个不受这两个开关影响——它们要的东西（``ctx.writer`` / ``ctx.reminders`` /
    ``ctx.fired`` / ``ctx.cancellations``）是**运行时装配**的，不是用户配置。装配期
    真的缺了（只跑只读工具的运行）时，工具自己会在 ``run`` 里回一句"没有配置"
    （见各自的实现），而不是在这里被悄悄摘掉：那种"有时在有时不在"的工具表会让模型
    的失败经验互相矛盾。

    **顺序就是声明顺序**，决定请求里那个数组的排列——稳定前缀能命中后端的提示缓存，
    所以新工具一律**追加**，不插队。
    """
    return (
        GetTimeTool(),
        SearchMemoryTool(),
        ReadMemoryTool(),
        WriteMemoryTool(),
        ForgetMemoryTool(),
        SetReminderTool(),
        ListRemindersTool(),
        CancelReminderTool(),
        *((ReadFileTool(),) if read_root else ()),
        # 目录与搜索与 read_file 同一个前提：**没有可读目录时它们注定失败**，
        # 登记它们只会让模型去试一次然后被拒。
        *((ListFilesTool(), SearchFilesTool()) if read_root else ()),
        SystemInfoTool(),
        UpdatePlanTool(),
        # 联网读**默认不登记**（出网行为）；调用方只在用户打开开关且配了 key 时传
        # ``web``。两个工具与其它只读工具一样不过闸门——是否联网由总开关决定。
        *(
            (WebSearchTool(), FetchUrlTool())
            if web is not None and getattr(web, "available", False)
            else ()
        ),
        # 写与执行**追加在最后**（提示缓存靠稳定前缀）。三个前提：
        # 有可写根、用户没关命令执行；``edit_file`` 与 ``write_file`` 同一前提。
        *((WriteFileTool(), EditFileTool()) if write_root else ()),
        *((RunCommandTool(),) if write_root and allow_commands else ()),
        # 追加，不插队（见上面那条"顺序就是声明顺序"）。
        *((LookAtScreenTool(),) if vision else ()),
        # 没有表情包就不登记它——同 ``read_file`` 与 ``look_at_screen`` 那条账。
        *((SendStickerTool(stickers),) if stickers else ()),
        # **插件一律追加在最后。** 它们由 ``axo.plugins`` 从插件目录装上来，装不上
        # 的那些不在这里（各自记了一句人话，见 `MainWindow._tools`）。
        *plugins,
    )
