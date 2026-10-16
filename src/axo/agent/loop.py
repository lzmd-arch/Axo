"""工具循环：**事件生成器**，不是阻塞函数。

形状照着 :meth:`axo.llm.base.ChatBackend.stream` 来——一个同步生成器，
一次吐一个事件。这样换来两件事：

1. **核心腿里能逐事件断言。** 塞一个假后端进去，就能问"第二轮请求里到底有没有
   那条工具结果"、"上限到顶之后有没有再发一次收尾请求"。这些是"循环真的在循环"
   的证据，而它们全都**不需要 Qt、不需要网络**。
2. **UI 侧不用等整轮结束。** 文本片到一片发一片，与现在 ``ChatWorker`` 的手感一致。

## 取消由谁负责：都不是这一层

``await_approval`` 是**注入的回调**，循环自己不碰 Qt、不碰线程。取消与超时的处理
在 :mod:`axo.agent.gate` 与 ``ui/agent_worker.py`` 里：闸门的握手槽被
``cancel()`` 唤醒，``await_approval`` 于是返回 ``None``，:func:`resolve_decision`
把它归成拒绝。**循环因此不需要一个"检查是否已取消"的分支**——那种分支在真正需要
它的那天一定会被某个新加的步骤忘掉。

## 工具尾巴的额度：两条规则，缺一条就有一种静默失效

工具结果进的是**同一个上下文窗口**，所以它能挤掉人格锚与用户刚说的话。上限由
``tools_ceiling`` 给出（默认是"装配完之后还剩多少"——份额已经拆掉了）。

* **单条结果不许超过整个尾巴的额度**（超出就截断，并留痕）。少了这条，一次
  ``read_file``（上限 64 KiB，约两万 token）会**单独一条**就撑爆窗口——而服务端
  的反应是**静默截断最早的消息**，也就是人格锚。那是一种没有任何报错的失效。
* **尾巴整体超了就从最老的整对开始驱逐。** 只驱逐**整对**（一条带 ``tool_calls``
  的 assistant 连同它后面所有的 tool 行）：留一半的配对是模型无法解释的对话，
  而"模型开始胡说"不是一条能追到这里的症状。

两条合起来还保证了一件事：**最新的一对永远不会被驱逐**——第一条规则已经把它压到
额度以内了。所以驱逐循环的条件里有一个 ``len(pairs) > 1``。

**"驱逐"只作用于工作尾巴，不作用于记录。** 循环里因此有两份东西：``tail``（会被
驱逐，进下一次请求）与每产出一对就立刻吐出去的 :class:`MessagesAppended`（只增不减，
调用方拿它写逐字稿）。只留 ``tail`` 的话，被驱逐的那几轮工具调用会从逐字稿里消失
——而逐字稿是"这次对话真的发生过什么"的记录（与 ``_WRAP_UP_NOTE`` 不进记录是同一条
理由）。这个区别**没有任何症状**：界面上一切正常，只是回看的时候少了几步。

**为什么记录是"边产出边吐"，而不是攒到 :class:`LoopDone` 里一次给。** 因为调用方
**会中途停止消费**：用户点了停止，``ui/agent_worker.py`` 就跳出这个生成器（生成器
随即被关闭，后端那条流也跟着断）。攒到 LoopDone 的话，那些**已经真的执行过**的工具
调用——包括已经落了库的写——一行记录都不会留下。一份"做过但没记"的逐字稿比少记几步
严重得多：它是这个项目唯一不能有的那种失效。

**由此推出调用方的一条纪律**：退出点不能选在 :class:`ToolFinished` 上。那一对事件
之间夹着一次已经发生的副作用（``dispatch`` 就在两次 ``yield`` 中间），在它那里退出
正好落进上面那种失效。``ui/agent_worker.py`` 的停止判据因此显式跳过这个事件，并
在模块文档里说明了代价。**加新事件时要照这条查一遍。**

**已知边界**：一轮里如果模型一次发了五个大结果，那一对整体仍可能超过额度（最新
的一对不驱逐）。修它要在一对内部做更细的取舍，而"模型一轮发五个大结果"在实测里
没出现过（Step 0 的分布：每轮 1–2 个）。写在这里是因为不写就会被当成"已经处理了"。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from ..llm.base import ChatBackend, ChatMessage, ChatRequest, ToolCall
from ..session.budget import IMAGE_TOKENS, TokenBudget, TokenMeter
from .gate import (
    ApprovalDecision,
    ApprovalRequest,
    build_request,
    resolve_decision,
)
from .tools import ToolContext, ToolRegistry

__all__ = [
    "DEFAULT_LIMITS",
    "ApprovalRequested",
    "Event",
    "LoopDone",
    "LoopLimits",
    "MessagesAppended",
    "RoundStarted",
    "TextDelta",
    "ToolFinished",
    "run_agent_loop",
]


@dataclass(frozen=True)
class LoopLimits:
    """循环的边界。

    Attributes:
        max_rounds: 最多几轮工具调用。到顶之后**再发一次收尾请求**
            （``tools=()``，请模型基于已有信息作答）——不给这次收尾，用户可能在
            工具调完之后什么都拿不到。

            4 这个数是 **Step 0 探针量出来的**：实测收敛所需的工具轮次分布最大为 2
            （``tools/smoke_agent.py`` 的输出留档）。取 4 是给"多一步"留余量，
            不是拍的——但这个余量对不对，要在真实用法上再看一次。
    """

    max_rounds: int = 4


DEFAULT_LIMITS = LoopLimits()


# ── 事件 ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoundStarted:
    """一轮开始。界面据此知道"又在动词源了"。"""

    index: int
    tools_offered: int
    wrapping_up: bool = False


@dataclass(frozen=True)
class TextDelta:
    """一段增量文本。与 ``ChatChunk.text`` 同义，只是换了条路径。"""

    text: str


@dataclass(frozen=True)
class ApprovalRequested:
    """有一个改状态的动作在等用户点头。

    **界面必须为它画一张卡片**——收到这个事件而没有画，用户就永远看不到这次问询，
    而 worker 会一直等到超时然后按拒绝处理。症状是"助手说它没做"，看不出是界面
    少画了一张卡。
    """

    request: ApprovalRequest


@dataclass(frozen=True)
class ToolFinished:
    """一次工具调用结束（成功、失败、或被拒绝）。

    :attr:`content` 是**工具真正产出的完整内容**，不是回灌给模型的那份。
    :attr:`clipped` 标出两者的差别：超长时回灌的是截断版，而界面上应当显示完整
    的那份。用户要看的是真相，模型要的是能放得下的那份。
    """

    call_id: str
    name: str
    ok: bool
    content: str
    detail: str = ""
    denied: bool = False
    clipped: bool = False
    preview: str = ""
    """工具产出的、要**显示给用户**的那张图的路径（空串表示没有）。

    从 :attr:`~axo.agent.tools.ToolOutcome.preview` 原样带过来——这一层不认识「表情包」
    是什么，它只负责不让一个已知的事实（工具有一张要给人看的图）在传话的路上丢掉。
    """


@dataclass(frozen=True)
class MessagesAppended:
    """一对工具消息进了对话，**调用方要把它写进逐字稿**。

    :attr:`rows` 是一条带 ``tool_calls`` 的 assistant 行，连同答复它的全部 tool 行，
    按顺序。**成对给，不成行给**：调用方按行落盘就可能落在半个配对上（写第一行时
    进程退出），而半个配对是模型无法解释的一段对话。

    ``ui/agent_worker.py`` 收到它就原样收下，**不从 :class:`LoopDone` 里拿**——理由
    见模块文档"为什么记录是边产出边吐"：用户中途停止时，这个事件已经出去了，而
    :class:`LoopDone` 永远不会来。

    **最新一对永远在这里，哪怕它后来被驱逐**（驱逐只动工作尾巴，见模块文档）。

    **记的是"进过对话的那份"。** 超长结果在 :attr:`ToolFinished.content` 里是完整的
    原文，而这里（以及任何进 ``messages`` 的地方）是**截断后**的那份。理由：逐字稿
    记的是**对话里真的有什么**，而截断标记本身已经把"这是片段、原文多少 token"说
    清楚了；把 64 KiB 的原文写进每一行 JSONL，下一轮 ``assemble()`` 读回来还是会被
    截掉一次，只是让逐字稿白长了几兆。
    """

    rows: tuple[ChatMessage, ...]
    """``(assistant 行, tool 行, ...)``，至少两条。"""

    no_replay: tuple[tuple[str, str], ...] = ()
    """``(tool_call_id, 原因)``：这一批里**不该在后续轮次回放**的调用。

    空元组 = 这一对正常回放。非空时由 :func:`~axo.session.assemble.stored_rows`
    落成 :attr:`~axo.session.store.Message.no_replay`，L5 再据此整对跳过。

    **为什么不塞进 :class:`ChatMessage`。** 那个类型是**协议层**的形状，回灌一次工具
    结果只需要 ``tool_call_id``；给协议加一个只有逐字稿用得上的字段，是把存储的偏好写
    进协议——``stored_rows`` 的 docstring 里已经为 ``tool_name`` 记过同一条取舍。
    所以它跟着事件走，由调用方在落盘那一步交给 ``stored_rows``。

    **为什么是 ``(id, 原因)`` 而不是一个 id 集合。** 原因在循环里就已经知道（闸门拒绝
    还是结果不耐放），而它到不了装配器——那里只有行，没有事件。丢掉它就等于在逐字稿
    里留下一句"这一对没被回放"而说不出为什么。
    """


@dataclass(frozen=True)
class LoopDone:
    """循环结束。

    Attributes:
        final_text: 最后一轮模型给出的文本。**这是要展示给用户的那个回答**——
            前面几轮的文本是"边想边说"，它们已经作为 :class:`TextDelta` 流走了。
        rounds: 用掉的工具轮数。
        tool_calls: 一共执行了几次工具调用（含被拒绝的）。
        wrapped_up: 最后一轮是不是那次收尾请求。
        unexecuted: 收到了但**没有执行**的调用名。正常路径上恒为空；只在上限之后
            的收尾轮里模型仍然要求调工具时非空——那时没有工具声明可用，所以只能
            不执行。**调用方不许把它吞掉**：吞掉它就等于回到了"静默丢弃"。

    **逐字稿要写的那几行不在这里**，在每产出一对就发出的 :class:`MessagesAppended`
    里（见那条的理由）。这里只剩"怎么收场"。
    """

    final_text: str
    rounds: int
    tool_calls: int
    wrapped_up: bool
    unexecuted: tuple[str, ...] = ()


Event = RoundStarted | TextDelta | ApprovalRequested | ToolFinished | MessagesAppended | LoopDone


# ── 截断与尾巴管理 ─────────────────────────────────────────────────────────

_CLIP_NOTE = "\n\n[…工具结果过长，已截断：完整内容约 {tokens} token。需要更多请缩小范围再调一次。…]"

_IMAGE_DROPPED_NOTE = (
    "\n\n（这次截到的图没有带进来：它自己就占了工具结果的全部额度。"
    "请如实告诉用户你看不到屏幕，不要凭猜描述。）"
)

_IMAGE_CARRIER_NOTE = (
    "（这是上一步 look_at_screen 截到的屏幕，见上面那一条工具结果。"
    "这一行不是用户说的话，是程序把那张图捎过来的。）"
)
"""图不能挂在工具结果那一行时，用来捎它的那条 ``user`` 消息的正文。

**为什么需要一条单独的消息**：OpenAI 兼容协议（含 DeepSeek）只认 ``user`` 行的
content 分片数组，``tool`` 行的 content 是字符串——图在那儿没有位置。Ollama 那边
不需要它（图直接挂在工具结果行上，实测收）。

**它不是用户说的，也不是模型说的**，所以它只进这一轮的工作尾巴、**不进**
:class:`MessagesAppended`（同下面 ``_WRAP_UP_NOTE`` 的道理）：逐字稿是"这次对话真的
发生过什么"的记录，把一条程序拼的话写进去，回看时它会看起来像一次真实发言——而这
一条还带着一整张屏幕。

正文写全"这一行不是用户说的话"，是因为模型的读者只有它：让它把这张图当成用户新发的
一条消息的话，它会开始回答一条没人提过的请求。
"""


_WRAP_UP_NOTE = ChatMessage(
    role="system",
    content=(
        "已经达到本轮的工具调用上限，接下来的请求里没有工具可用了。"
        "请基于已经拿到的信息回答，并说明还有什么没做完。"
    ),
)
"""到顶那一次收尾请求里追加的一句。

**它是这一轮临时的脚手架，不进逐字稿**（只进 ``working``，不进
:class:`MessagesAppended`）。理由：它不是用户说的、也不是模型说的，而逐字稿是
"这次对话真的发生过什么"的记录——把一句系统指令写进去，回放时它会看起来像一次
真实发言。

不给这句话也能跑（模型手上有工具结果，多半会直接作答），但"多半"不够：一次
没有交代的收尾会让模型重新开始调工具、或者干脆答一句"我需要更多信息"。"""


def _clip_to_tokens(text: str, limit: int, meter: TokenMeter) -> tuple[str, bool]:
    """把一段正文压到 ``limit`` token 以内，返回 ``(正文, 是否截断)``。

    用**二分**而不是"按字数估一个比例"：估计器对中文和拉丁文的系数不同（见
    ``memcore.tokens``），一个统一比例会在混合文本上偏掉，而偏大的方向正是这里
    要防的那个方向。二分的每一次探测都是 O(长度)，但一共只有 log₂(长度) 次。
    """
    total = meter.text(text)
    if total <= limit:
        return text, False

    note = _CLIP_NOTE.format(tokens=total)
    # 标记本身也占位置，所以先把它从额度里扣掉——否则"压到 limit 以内"会稳定地
    # 超出一点点，而超出的那一点点正好落在没有余量的时候。
    budget = max(0, limit - meter.text(note))
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if meter.text(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low] + note, True


def _tail_tokens(tail: Sequence[ChatMessage], meter: TokenMeter) -> int:
    """这条工具尾巴值多少 token。assistant 行也要算——它的 ``tool_calls`` 是
    JSON 形状的正文，与 ``content`` 一样占位置（``TokenMeter.message`` 已经算进去了）。"""
    return sum(meter.message(message) for message in tail)


def _fit_tool_tail(
    tail: list[ChatMessage],
    pairs: list[tuple[int, int]],
    *,
    ceiling: int,
    meter: TokenMeter,
) -> int:
    """按**整对**把尾巴压进 ``ceiling``，返回驱逐了几对。

    ``pairs`` 里每个元素是 ``(起始下标, 长度)``，长度含那条 assistant 行。驱逐最老
    的一对之后，后面所有对的下标都要往前挪——**这一步漏了会删错行**，而症状是
    "模型收到了两段对不上的工具结果"，一种看起来像模型变笨的问题。

    **只动 ``tail``，不动记录。** 被驱逐的行仍然在这次回合的逐字稿里（见模块文档）。
    """
    evicted = 0
    while len(pairs) > 1 and _tail_tokens(tail, meter) > ceiling:
        start, length = pairs.pop(0)
        del tail[start : start + length]
        pairs[:] = [(begin - length, size) for begin, size in pairs]
        evicted += 1
    return evicted


# ── 主循环 ─────────────────────────────────────────────────────────────────


def run_agent_loop(
    *,
    backend: ChatBackend,
    messages: Sequence[ChatMessage],
    registry: ToolRegistry,
    ctx: ToolContext,
    await_approval: Callable[[ApprovalRequest], ApprovalDecision | None],
    budget: TokenBudget,
    meter: TokenMeter,
    tools_ceiling: int | None = None,
    limits: LoopLimits = DEFAULT_LIMITS,
) -> Iterator[Event]:
    """跑一轮任务语域：可能要调几次工具，然后给出回答。

    Args:
        backend: 对话后端。**注入进来**，所以这一层在核心腿里能用假后端逐事件断言。
        messages: 已经装配好的上下文（``assemble()`` 的产物）。**只是读取**——
            循环不会改动它，也不重新装配（理由见下）。
        registry: 工具注册表。它同时是工具声明的来源。
        ctx: 工具能碰到的东西。
        await_approval: 问用户要一次裁决。返回 ``None`` 表示超时/取消 → 按拒绝。
        budget: 装配预算。只用于取 ``reserve_output`` 传给 ``max_tokens``。
        meter: token 估计器。截断与驱逐都用它，**与装配器同一个**——两处用不同的
            算法算同一段文本，是这个项目里账对不上的唯一根因。
        tools_ceiling: 工具尾巴的额度。**省略就按"装配完之后还剩多少"算**——
            份额已经拆掉，没有第二个数可传；显式传是为了测试要钉住一个具体形状。
        limits: 循环边界。

    Yields:
        :data:`Event`，最后一个是 :class:`LoopDone`。

    **一个用户回合只装配一次。** 循环只在尾部追加工具消息，不重新装配。三条理由，
    按重要性排：重新装配会把**刚拿到、正要用的工具结果**当成可驱逐的旧历史重算一遍；
    它会对同一句输入再跑一次嵌入与召回（多一次网络往返，结果逐条相同）；
    而工具结果必须紧跟发起它的那条 assistant 消息（协议），重排会切断配对。
    """
    # ``tail`` 会被驱逐，进下一次请求；记录不在这里——每产出一对就作为
    # :class:`MessagesAppended` 发出去，因为调用方可能中途停止消费（见模块文档）。
    if tools_ceiling is None:
        tools_ceiling = max(0, budget.usable - meter.messages(messages))
    tail: list[ChatMessage] = []
    pairs: list[tuple[int, int]] = []
    rounds = 0
    total_calls = 0
    final_text = ""
    unexecuted: tuple[str, ...] = ()
    wrapped_up = False

    while True:
        wrap = rounds >= limits.max_rounds
        offered = () if wrap else registry.specs()
        yield RoundStarted(index=rounds, tools_offered=len(offered), wrapping_up=wrap)

        # **重排一遍而不是维护一个 working 列表**：驱逐会把 tail 里的整段删掉，
        # 而下标是驱逐算法唯一的抓手。每次从头拼，``turn`` 与 ``tail`` 就不可能
        # 不一致——那种不一致的表现是模型收到两份对不上的历史。
        turn: list[ChatMessage] = [*messages, *tail]
        if wrap:
            turn.append(_WRAP_UP_NOTE)

        # ``reserve_output`` 是 0 就**一个字都不传**（云端）：输出长度走官方默认，
        # 我们不再设任何上限。非 0 只出现在 Ollama 那条链上——它不传 ``num_predict``
        # 就会拿同一个 num_ctx 生成，把输入挤掉（见 ``session/budget.py``）。
        request = ChatRequest(
            messages=tuple(turn),
            tools=offered,
            max_tokens=budget.reserve_output or None,
        )

        parts: list[str] = []
        calls: tuple[ToolCall, ...] = ()
        for piece in backend.stream(request):
            if piece.text:
                parts.append(piece.text)
                yield TextDelta(piece.text)
            # Ollama 的 tool_calls **整条一次到齐**（实测，见 ``llm/base.py``），
            # 所以这里直接覆盖而不累积：不需要碎片状态机，也就不需要处理
            # "一个调用分了两片"这种在本项目里不存在的情形。
            if piece.tool_calls:
                calls = piece.tool_calls
        text = "".join(parts)

        if not calls:
            final_text = text
            wrapped_up = wrap
            break

        if wrap:
            # 收尾轮里**没有工具声明**，所以这些调用执行不了。
            #
            # 为什么是"不执行 + 报告"而不是抛异常：收尾轮的文本是用户这一轮唯一
            # 的答案，抛出去它就没了。但**也不能静默丢掉**——那正是
            # ``ui/chat_worker.py`` 被修掉的那个缺陷。于是它进
            # :attr:`LoopDone.unexecuted`，由调用方显示出来。
            unexecuted = tuple(call.name for call in calls)
            final_text = text
            wrapped_up = True
            break

        assistant = ChatMessage(role="assistant", content=text, tool_calls=calls)
        tail.append(assistant)
        pair_start = len(tail) - 1
        #: 这一对里**要进逐字稿**的那几行。捎图那条消息不进（见 ``_IMAGE_CARRIER_NOTE``），
        #: 所以这份名单是显式攒的，不再拿 ``tail[pair_start:]`` 切片——切片会把捎图的
        #: 那条一起带出去，而它会以"用户说的一句话"的身份落进对话记录。
        storable: list[ChatMessage] = [assistant]

        pair_no_replay: list[tuple[str, str]] = []
        for index, tool_call in enumerate(calls):
            total_calls += 1
            approved = True
            if registry.needs_approval(tool_call.name):
                # 把工具一起递过去：**没有参数的工具只能自己说清范围**
                # （``gate.summarize``），而这句话的读者是正要点头的那个人。
                ask = build_request(tool_call, round_index=index, tool=registry.get(tool_call.name))
                yield ApprovalRequested(ask)
                # ``None`` = 超时/取消/通道没了 → 拒绝。fail-closed 的理由见 gate.py。
                approved = resolve_decision(await_approval(ask)) is ApprovalDecision.GRANT

            outcome = registry.dispatch(tool_call, ctx, approved=approved)
            # **"拒绝"以循环自己的判断为准，不看 ``outcome.replayable``。** 那个字段
            # 说的是"这个结果值不值得回放"，而这里要知道的是**为什么**不值得——两种
            # 原因在逐字稿里是两个不同的失效，混成一个就分不出该修哪儿。顺序也刻意：
            # 一次拒绝永远是不可回放的（派发口在拒绝分支上标了 False），先判拒绝才不会
            # 把 `denied` 说成 `stale`。
            if not approved:
                pair_no_replay.append((tool_call.id, "denied"))
            elif not outcome.replayable:
                pair_no_replay.append((tool_call.id, "stale"))
            # **先给图留位置，再裁正文。** 图裁不动（``_clip_to_tokens`` 只处理文本），
            # 所以顺序反过来的话正文会先把额度占满、图被挤掉，而模型读到的正文却自称
            # 带着图——那正是这一条链上最容易发生的一种编。
            images = outcome.images
            if images and IMAGE_TOKENS * len(images) >= tools_ceiling:
                # 这一轮剩下的窗口装不下这张图（只在小窗口上才会发生）。**不静默丢**
                # ——丢的条件是"窗口真的满了"，不是某一层的份额；份额已经拆掉了。
                images = ()
                clipped_by_budget = True
            else:
                clipped_by_budget = False
            fed, clipped = _clip_to_tokens(
                outcome.content, max(0, tools_ceiling - IMAGE_TOKENS * len(images)), meter
            )
            if clipped_by_budget:
                fed += _IMAGE_DROPPED_NOTE
            yield ToolFinished(
                call_id=tool_call.id,
                name=tool_call.name,
                ok=outcome.ok,
                content=outcome.content,
                detail=outcome.detail,
                denied=not approved,
                clipped=clipped,
                preview=outcome.preview,
            )
            # **图只活在这一次往返里。** 落逐字稿的那条路只存 ``content``、L5 回放
            # 重建行时也不带 ``images``（``session/assemble.py`` 的 ``_tool_pair``），
            # 所以带图的工具**必须**标 ``replayable=False``——那一条由工具自己保证，
            # 这里只是把图交出去。
            # **图挂哪一行由后端说了算**（``chat_message.images`` 那条注释里写了
            # 两个协议的差别）。``getattr`` 而不是进协议：``ChatBackend`` 是
            # ``@runtime_checkable`` 的，往里加成员等于要求每个后端都长这个属性，
            # 而测试里那一堆假后端根本不需要它。
            if images and getattr(backend, "images_ride_on", "tool") == "user":
                row = ChatMessage(role="tool", content=fed, tool_call_id=tool_call.id)
                tail.append(row)
                storable.append(row)
                tail.append(ChatMessage(role="user", content=_IMAGE_CARRIER_NOTE, images=images))
            else:
                row = ChatMessage(
                    role="tool", content=fed, tool_call_id=tool_call.id, images=images
                )
                tail.append(row)
                storable.append(row)

        # **整对发出去，发在驱逐之前。** 顺序是刻意的：先发再驱逐，所以"记录里有
        # 这一对"不取决于这一对有没有被挤出工作尾巴。反过来的话，被挤掉的那一对
        # 会连同它的记录一起消失，而调用方无从知道少了什么。
        yield MessagesAppended(rows=tuple(storable), no_replay=tuple(pair_no_replay))

        pairs.append((pair_start, len(tail) - pair_start))
        _fit_tool_tail(tail, pairs, ceiling=tools_ceiling, meter=meter)
        rounds += 1

    yield LoopDone(
        final_text=final_text,
        rounds=rounds,
        tool_calls=total_calls,
        wrapped_up=wrapped_up,
        unexecuted=unexecuted,
    )
