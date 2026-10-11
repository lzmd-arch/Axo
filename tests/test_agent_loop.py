"""工具循环。**核心腿：不碰 Qt、不碰网络。**

真实模型在这条路上不可复现——Step 0 实测同一个多步 prompt 五次里只对一次。所以
"模型这一侧"整个换成 ``ScriptedBackend`` 的脚本，要断言的每一条都变成确定性的；
真实模型的行为由 ``tools/smoke_agent.py`` 量（那是量具，不是测试）。

## 反空转：四条必须一起钉

单条断言都能被假绿通过。"最终有文本"这种断言，一个**从不执行工具**的循环也过——
它直接返回第一轮的回答就行了。所以
:func:`test_single_tool_call_executes_feeds_back_and_finishes` 在同一个测试里钉四件事：

1. **副作用计数**：spy 工具被调用**恰好 1 次**（``== 1``，不是 ``>= 1``）；
2. **回灌证明**：第 2 次请求的正文里含那条工具结果；
3. **声明下发证明**：第 1 次请求的 ``tools`` 非空且条数 == 注册表条数；
4. **循环证明**：``backend.stream`` 被调用 **≥ 2 次**。

拆成四条测试的话，每条都能被一个不同的实现满足，而合起来才是"这是个循环"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedBackend, text_chunk, tool_chunk

from biosnn_memterm.agent.gate import ApprovalDecision, ApprovalRequest, summarize
from biosnn_memterm.agent.loop import (
    DEFAULT_LIMITS,
    ApprovalRequested,
    LoopDone,
    LoopLimits,
    MessagesAppended,
    RoundStarted,
    TextDelta,
    ToolFinished,
    run_agent_loop,
)
from biosnn_memterm.agent.tools import (
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    ToolRegistry,
)
from biosnn_memterm.llm.base import (
    ChatBackendError,
    ChatMessage,
    ToolCall,
    ToolSpec,
)
from biosnn_memterm.session.budget import TokenBudget, TokenMeter
from memcore.store import MemoryStore

#: 一个"够大"的额度——正常路径的用例不想被截断或驱逐干扰。
ROOMY = 100_000


class SpyTool:
    """记录被怎么调用的工具。

    ``render`` 从参数算出返回正文，于是**同一个替身能在不同轮返回不同内容**——
    驱逐那条测试要靠这个区分"哪一对还在上下文里"。
    """

    def __init__(
        self,
        *,
        name: str = "spy",
        requires_approval: bool = False,
        render: Any = None,
        raiser: BaseException | None = None,
        parameters: dict[str, Any] | None = None,
        replayable: bool = True,
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description="测试用替身",
            parameters=parameters or {"type": "object", "properties": {}},
        )
        self.requires_approval = requires_approval
        self.calls: list[dict[str, Any]] = []
        self._render = render
        self._raiser = raiser
        self._replayable = replayable

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        self.calls.append(arguments)
        if self._raiser is not None:
            raise self._raiser
        if self._render is not None:
            return ToolOutcome.success(self._render(arguments), replayable=self._replayable)
        return ToolOutcome.success("替身工具的结果", replayable=self._replayable)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def call(name: str = "spy", call_id: str = "call_1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def drive(
    backend: ScriptedBackend,
    registry: ToolRegistry,
    store: MemoryStore,
    *,
    ceiling: int = ROOMY,
    limits: LoopLimits = DEFAULT_LIMITS,
    answer: ApprovalDecision | None = ApprovalDecision.GRANT,
    messages: list[ChatMessage] | None = None,
) -> tuple[list[Any], LoopDone]:
    """跑一遍循环，返回 ``(所有事件, 最后一个事件)``。"""
    events = list(
        run_agent_loop(
            backend=backend,
            messages=messages or [ChatMessage(role="user", content="帮我看看我的笔记")],
            registry=registry,
            ctx=ToolContext(store=store, session_id="S1", turn_seq=1),
            await_approval=lambda _request: answer,
            budget=TokenBudget(window=8192),
            meter=TokenMeter(),
            tools_ceiling=ceiling,
            limits=limits,
        )
    )
    assert isinstance(events[-1], LoopDone), "最后一个事件必须是 LoopDone"
    return events, events[-1]


def kinds(events: list[Any]) -> list[type]:
    return [type(event) for event in events]


def appended_rows(events: list[Any]) -> list[ChatMessage]:
    """把逐对发出的 :class:`MessagesAppended` 摊平成一份记录，按顺序。

    记录**不挂在 ``LoopDone`` 上**（见 :class:`MessagesAppended` 的理由：调用方可能
    中途停止消费）。所以这里也照调用方的方式收——测试与真接线走同一条路，否则
    "事件发得对不对"这件事在测试里绿着、在界面上丢掉。
    """
    rows: list[ChatMessage] = []
    for event in events:
        if isinstance(event, MessagesAppended):
            rows.extend(event.rows)
    return rows


def appended_reasons(events: list[Any]) -> list[tuple[str, str]]:
    """收齐"这一对怎么结束的"。与 :func:`appended_rows` 同一条路——**测试与真接线走
    同一次事件**，否则"标记发得对不对"在测试里绿着、在界面上丢掉。"""
    reasons: list[tuple[str, str]] = []
    for event in events:
        if isinstance(event, MessagesAppended):
            reasons.extend(event.no_replay)
    return reasons


def assert_tool_pairs_intact(
    messages: tuple[ChatMessage, ...] | list[ChatMessage],
) -> None:
    """对话里每一对"调用—结果"必须完整，**两个方向都查**。

    孤儿 ``tool`` 行是"驱逐只删了一半"的一种症状；**只剩 assistant、结果行被删**是
    另一种，而它同样不合法——``role="assistant"`` 声明了 ``tool_calls`` 却没有对应的
    结果，服务端的反应是拒掉整个请求。只查一个方向的助手会让第二种情况绿着过去。

    值得单独写出来是因为它**没有症状**：模型收到一段它无法解释的对话，表现是"模型
    开始胡说"，而那不是一条能追到驱逐逻辑的症状。
    """
    pending: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            assert not pending, f"上一轮的调用没有结果：{sorted(pending)}"
            pending = {tool_call.id for tool_call in message.tool_calls}
        elif message.role == "tool":
            assert message.tool_call_id in pending, f"孤儿 tool 行：{message.tool_call_id}"
            pending.discard(message.tool_call_id)
        else:
            assert not pending, f"{message.role} 行之前还有没答复的调用：{sorted(pending)}"
    assert not pending, f"最后一条 assistant 的调用没有结果：{sorted(pending)}"


# ── 反空转四条 ─────────────────────────────────────────────────────────────


def test_single_tool_call_executes_feeds_back_and_finishes(store: MemoryStore) -> None:
    """**四条一起钉**，见模块文档。"""
    spy = SpyTool(render=lambda _args: "查到了：备用钥匙在鞋柜里")
    registry = ToolRegistry([spy])
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", key="钥匙"))],
            [text_chunk("钥匙在鞋柜里。", done=True)],
        ]
    )

    _, done = drive(backend, registry, store)

    # ① 副作用计数：恰好一次。``>= 1`` 会放过"每一轮都重调一遍"的实现。
    assert len(spy.calls) == 1
    # ② 回灌证明：第 2 次请求里含那条工具结果的正文。
    assert "备用钥匙在鞋柜里" in backend.last_text(1)
    # ③ 声明下发证明：第 1 次请求声明了工具，且条数 == 注册表条数。
    assert backend.tools_declared(0) == ("spy",) == registry.names
    # ④ 循环证明。
    assert backend.calls >= 2

    assert done.final_text == "钥匙在鞋柜里。"
    assert done.rounds == 1
    assert done.tool_calls == 1


def test_plain_answer_is_a_single_request(store: MemoryStore) -> None:
    """不需要工具时只有一次请求。

    反方向的反空转：一个"总是先发一轮工具声明、等不到工具调用又发一轮"的实现在
    这里会红，而在上面那条测试里完全看不出来。
    """
    spy = SpyTool()
    backend = ScriptedBackend(script=[[text_chunk("今天挺好。", done=True)]])

    events, done = drive(backend, ToolRegistry([spy]), store)

    assert backend.calls == 1
    assert spy.calls == []
    assert done.rounds == 0
    assert done.tool_calls == 0
    assert appended_rows(events) == []
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["今天挺好。"]


def test_assistant_message_carries_the_tool_calls(store: MemoryStore) -> None:
    """回灌的那条 assistant 行必须带 ``tool_calls``。

    协议要求：``role="tool"`` 的消息靠 ``tool_call_id`` 与发起它的调用配对，而
    那个 id 只存在于 assistant 行上。少了它，第 2 次请求就是一段模型无法解释的
    对话——``ChatMessage.__post_init__`` 只挡住了反过来的那个方向。
    """
    registry = ToolRegistry([SpyTool()])
    backend = ScriptedBackend(
        script=[[tool_chunk(call("spy", call_id="c9"))], [text_chunk("好了", done=True)]]
    )

    events, _done = drive(backend, registry, store)

    rows = appended_rows(events)
    assistant = rows[0]
    assert assistant.role == "assistant"
    assert [tool_call.id for tool_call in assistant.tool_calls] == ["c9"]
    assert rows[1].tool_call_id == "c9"
    assert_tool_pairs_intact(rows)


def test_two_rounds_on_the_way_to_the_answer(store: MemoryStore) -> None:
    """连续两轮工具调用——"循环"这个词的第二个证据。"""
    spy = SpyTool(render=lambda args: f"结果{args['n']}")
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", call_id="c1", n=1))],
            [tool_chunk(call("spy", call_id="c2", n=2))],
            [text_chunk("都查完了。", done=True)],
        ]
    )

    _, done = drive(backend, ToolRegistry([spy]), store)

    assert [arguments["n"] for arguments in spy.calls] == [1, 2]
    assert backend.calls == 3
    assert done.tool_calls == 2
    assert "结果1" in backend.last_text(2) and "结果2" in backend.last_text(2)


# ── 事件流 ─────────────────────────────────────────────────────────────────


def test_events_arrive_in_order(store: MemoryStore) -> None:
    backend = ScriptedBackend(
        script=[
            [text_chunk("让我查一下。"), tool_chunk(call("spy"))],
            [text_chunk("查到了。", done=True)],
        ]
    )

    events, _ = drive(backend, ToolRegistry([SpyTool()]), store)

    assert kinds(events) == [
        RoundStarted,
        TextDelta,
        ToolFinished,
        # 记录紧跟它记的那一对，**在同一轮里**：这样调用方中途停止时，前面几轮的
        # 记录已经到手了（见 MessagesAppended）。
        MessagesAppended,
        RoundStarted,
        TextDelta,
        LoopDone,
    ]


def test_text_streams_before_the_tool_runs(store: MemoryStore) -> None:
    """文本片**先**流出去，工具才执行。

    顺序反了的话，界面上会是"卡片先出现、模型的话后出现"，而用户在卡片弹出时
    还不知道模型打算干什么——确认卡片的意义正是"在动作之前给用户上下文"。
    """
    backend = ScriptedBackend(
        script=[
            [
                text_chunk(
                    "先查",
                ),
                text_chunk("一下。"),
                tool_chunk(call("spy")),
            ],
            [text_chunk("好了", done=True)],
        ]
    )

    events, _ = drive(backend, ToolRegistry([SpyTool()]), store)

    order = [type(event) for event in events]
    assert order.index(TextDelta) < order.index(ToolFinished)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["先查", "一下。", "好了"]


# ── 闸门 ───────────────────────────────────────────────────────────────────


def test_gated_tool_asks_first_and_runs_on_grant(store: MemoryStore) -> None:
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[[tool_chunk(call("write_memory", content="用户喜欢美式"))], [text_chunk("记好了")]]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store, answer=ApprovalDecision.GRANT)

    asks = [event for event in events if isinstance(event, ApprovalRequested)]
    assert len(asks) == 1
    assert asks[0].request.tool_name == "write_memory"
    assert spy.calls == [{"content": "用户喜欢美式"}]


def test_gated_tool_denied_is_not_run_and_the_denial_is_fed_back(store: MemoryStore) -> None:
    """拒绝要**回灌**，不只是不执行。

    只不执行的话，模型不知道自己被拒了——它会照着"我已经写好了"的语气继续，而那
    正是 Step 0 测到的主要失败形态（声称做了却没做）。回灌的那句话因此明确禁止了
    重试与假装完成。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("write_memory", content="用户喜欢美式"))],
            [text_chunk("那我不记了")],
        ]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store, answer=ApprovalDecision.DENY)

    assert spy.calls == [], "被拒绝的工具还是被执行了"
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished[0].ok is False and finished[0].denied is True
    # 回灌的是 refusal：第 2 次请求里能看到它，而不是一条空的工具结果。
    assert "没有同意" in backend.last_text(1)


def test_a_denied_pair_is_marked_denied(store: MemoryStore) -> None:
    """**被拒绝的那一对要标上"别回放"。**

    它与上面那条不矛盾，因为两者管的是**两个不同的时刻**：回灌是**当轮**告诉模型
    "这次跳过了、别死磕同一次调用"；不回放是让那句话**不要**在下一轮还挂在那儿——
    挂着的话它在模型眼里就从"这一次"变成一条仍然生效的规矩，一次拒绝于是成了该会话
    内对该工具的永久封禁，而用户只会觉得"这模型怎么不肯查了"。

    单独一条而不是并进上面那条：两者断的是不同的东西（回灌 vs 标记），失败时的
    修法也不同，而并在一起时名字里看不出后者的存在。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("write_memory", content="用户喜欢美式"))],
            [text_chunk("那我不记了")],
        ]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store, answer=ApprovalDecision.DENY)

    denied = [event for event in events if isinstance(event, ToolFinished) and event.denied]
    assert len(denied) == 1, "夹具本身没造出一次拒绝，下面的断言是空转的"
    assert appended_reasons(events) == [(denied[0].call_id, "denied")]


def test_a_normal_result_is_not_marked_for_no_replay(store: MemoryStore) -> None:
    """反空转：普通结果不带标记。

    没有这条，"给所有工具对都盖上 no_replay"也能过上面那条——而那等于把 L5 废掉，
    代价是模型每轮重查它已经查过的东西。
    """
    spy = SpyTool(name="search_memory")
    backend = ScriptedBackend(
        script=[[tool_chunk(call("search_memory", query="钥匙"))], [text_chunk("在鞋柜里。")]]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store)

    assert appended_reasons(events) == []


def test_a_snapshot_result_is_marked_stale(store: MemoryStore) -> None:
    """结果自己说不耐放（``ToolOutcome.replayable=False``）→ 标 ``stale``。

    这一条是 :class:`~biosnn_memterm.agent.catalog.GetTimeTool` 那类工具的唯一守卫：
    ``replayable`` 默认是 ``True``，所以一个忘了标的快照工具**不会有任何症状**——
    它只是悄悄地把过期的时间戳递给模型。``tests/test_agent_tools.py`` 里另有一条
    直接盯着 ``get_time``/``read_file`` 的。（见那两条测试的 docstring。）
    """
    spy = SpyTool(name="get_time", replayable=False)
    backend = ScriptedBackend(script=[[tool_chunk(call("get_time"))], [text_chunk("14:32。")]])

    events, _ = drive(backend, ToolRegistry([spy]), store)

    assert appended_reasons(events) == [(appended_rows(events)[-1].tool_call_id, "stale")]


def test_no_answer_is_a_denial(store: MemoryStore) -> None:
    """``None``（超时 / 用户按了停止 / 通道没了）→ 拒绝。

    fail-closed 是唯一安全的默认：把"没答"当成"同意"，用户会在一次卡顿之后发现
    记忆里多了一条他从没批准过的东西，而没有任何地方记着那件事发生过。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[[tool_chunk(call("write_memory", content="x"))], [text_chunk("好")]]
    )

    _, done = drive(backend, ToolRegistry([spy]), store, answer=None)

    assert spy.calls == []
    assert done.tool_calls == 1  # 计了数：它确实发生过，只是没执行


def test_approval_summary_comes_from_arguments_not_from_the_model(store: MemoryStore) -> None:
    """确认文案**模板生成**，模型写的那句绝不许进去。

    用户是照着这句话点「允许」的，所以"让模型解释自己准备做什么"在这里的后果比在
    ``assemble.py`` 里还重。把模型当轮写的文本设成一句谎话，再断言它一个字都没
    出现在问询里——「我什么都没干」这句话若出现，就是幻觉直接进了确认卡片。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[
            [text_chunk("我什么都没干"), tool_chunk(call("write_memory", content="用户喜欢美式"))],
            [text_chunk("好")],
        ]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store)
    request = next(e.request for e in events if isinstance(e, ApprovalRequested))

    assert "用户喜欢美式" in request.summary
    assert "我什么都没干" not in request.summary


def test_summarize_renders_structure_instead_of_a_flattering_sentence() -> None:
    """结构化参数按 JSON 渲染。

    给一个"看起来像那么回事"的中文摘要会藏掉真正的形状，而用户要看的就是形状——
    一个 ``{"tier": ["a", "b"]}`` 被写成"把 a 和 b 分层"之后，他没法判断到底会发生
    什么。
    """
    text = summarize(call("forget_memory", ids=["a", "b"]))
    assert '["a","b"]' in text.replace(" ", "")


# ── 上限与收尾 ─────────────────────────────────────────────────────────────


def test_round_limit_sends_a_wrap_up_request_without_tools(store: MemoryStore) -> None:
    """到顶之后**再发一次** ``tools=()`` 的收尾请求。

    不给这次收尾，用户可能在工具调完之后什么都拿不到——而"什么都没拿到"与"模型
    没话说"在界面上是同一件事。
    """
    spy = SpyTool()
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", call_id="c1"))],
            [text_chunk("这是收尾。", done=True)],
        ]
    )

    events, done = drive(backend, ToolRegistry([spy]), store, limits=LoopLimits(max_rounds=1))

    assert len(spy.calls) == 1, "上限之后还在执行工具"
    assert backend.calls == 2, "到顶之后没有再发收尾请求"
    assert backend.requests[-1].tools == (), "收尾请求里还声明着工具"
    assert done.wrapped_up is True
    assert done.final_text == "这是收尾。"

    rounds = [event for event in events if isinstance(event, RoundStarted)]
    assert rounds[-1].wrapping_up is True and rounds[-1].tools_offered == 0


def test_wrap_up_round_still_asking_for_tools_is_reported_not_swallowed(store: MemoryStore) -> None:
    """收尾轮里模型仍然调工具 → **报告出来，但不执行、也不抛**。

    * **不执行**：那一轮没有工具声明，执行它等于绕过上限。
    * **不抛**：收尾轮的文本是用户这一轮唯一的答案，抛出去它就没了。
    * **不吞**：静默丢掉正是 ``ui/chat_worker.py`` 被修掉的那个缺陷。它进
      :attr:`LoopDone.unexecuted`，由调用方显示。
    """
    spy = SpyTool()
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", call_id="c1"))],
            [text_chunk("我再查一次"), tool_chunk(call("spy", call_id="c2"))],
        ]
    )

    _, done = drive(backend, ToolRegistry([spy]), store, limits=LoopLimits(max_rounds=1))

    assert len(spy.calls) == 1
    assert done.unexecuted == ("spy",)
    assert done.final_text == "我再查一次"


def test_backend_error_propagates(store: MemoryStore) -> None:
    """后端的错误**不许被循环吞掉**。

    工具派发口吃异常（那是为了回灌给模型自纠），但后端不可用不是"模型该改一改"的
    事——那是这一轮根本没法开始。吞掉它，界面会停在"正在输入"，而用户看不到任何
    解释。
    """
    backend = ScriptedBackend(script=[[text_chunk("x")]], raise_at=0)
    with pytest.raises(ChatBackendError):
        drive(backend, ToolRegistry([SpyTool()]), store)


# ── 工具尾巴的额度 ─────────────────────────────────────────────────────────

_LONG = "这是一条很长的工具结果。" * 40


def test_oversized_result_is_clipped_with_a_marker(store: MemoryStore) -> None:
    """单条结果不许超过**整个尾巴**的额度，超出了要留痕。

    少了这条，一次 ``read_file`` 就能单独撑爆窗口，而服务端的反应是**静默截断最早
    的消息**——那是人格锚所在的位置。没有任何报错，只有助手某天开始"失忆"。
    """
    meter = TokenMeter()
    ceiling = 120
    spy = SpyTool(render=lambda _args: _LONG)
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("好了")]])

    events, done = drive(backend, ToolRegistry([spy]), store, ceiling=ceiling)

    finished = next(event for event in events if isinstance(event, ToolFinished))
    # 界面上看到的是**完整**的那份：用户要看真相，模型要的是能放得下的那份。
    assert finished.content == _LONG
    assert finished.clipped is True

    fed = next(
        message.content for message in backend.requests[-1].messages if message.role == "tool"
    )
    assert fed != _LONG
    assert "已截断" in fed
    assert meter.text(fed) <= ceiling
    assert_tool_pairs_intact(appended_rows(events))

    # 逐字稿里记的是**进过对话的那份**（截断版），不是完整原文。这条断言钉的是一个
    # 会被后人"顺手修好"的决定：把 ``outcome.content`` 写进行里看起来更"完整"，
    # 代价是每次 ``read_file`` 都往 JSONL 里塞最多 64 KiB，而下一轮读回来仍会被
    # 截掉一次。界面上看到的完整正文走的是 ``ToolFinished.content``，那条路不受影响。
    assert appended_rows(events)[1].content == fed


def test_small_result_is_not_clipped(store: MemoryStore) -> None:
    """反空转：不是"永远截断"。"""
    spy = SpyTool(render=lambda _args: "很短")
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("好了")]])

    events, _ = drive(backend, ToolRegistry([spy]), store, ceiling=ROOMY)

    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.clipped is False
    assert finished.content == "很短"


def test_oldest_pair_is_evicted_when_the_tail_overflows(store: MemoryStore) -> None:
    """尾巴整体超了就驱逐**最老的整对**，且绝不留孤儿。

    三条断言缺一不可：最新一对还在（驱逐不能把刚拿到的东西删掉）、最老一对没了、
    并且上下文里没有半条配对。**只删 assistant 行或只删 tool 行的实现**能过前两条，
    却会在第三条上红——而它的症状是模型收到两段对不上的历史。
    """
    spy = SpyTool(render=lambda args: f"{args['tag']}" * 300)
    registry = ToolRegistry([spy])
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", call_id="c1", tag="甲"))],
            [tool_chunk(call("spy", call_id="c2", tag="乙"))],
            [tool_chunk(call("spy", call_id="c3", tag="丙"))],
            [text_chunk("三条都查完了", done=True)],
        ]
    )

    events, _done = drive(backend, registry, store, ceiling=200, limits=LoopLimits(max_rounds=3))

    final = backend.last_text(3)
    assert "丙" in final, "最新的一对被驱逐了"
    assert "甲" not in final, "最老的一对没有被驱逐"

    assert_tool_pairs_intact(backend.requests[-1].messages)
    # 被驱逐的仍然留在记录里（调用方要把它写进逐字稿）——驱逐只是"不进这次请求"，
    # 不是"这件事没发生过"。它必须还在，否则逐字稿会永远丢掉那几步。
    rows = appended_rows(events)
    assert_tool_pairs_intact(rows)
    assert len(rows) == 6  # 三对，一对不多一对不少


def test_tail_management_never_evicts_the_only_pair(store: MemoryStore) -> None:
    """只有一对时**不驱逐**。

    那条把最新一对也删掉的实现会让工具结果**永远进不了上下文**——于是循环看起来
    在跑、工具看起来在调，而模型始终看不到结果。这个失效没有任何症状。
    """
    spy = SpyTool(
        name="spy",
        render=lambda _args: "长" * 500,
        parameters={"type": "object", "properties": {}, "required": []},
    )
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("好了")]])

    events, _done = drive(backend, ToolRegistry([spy]), store, ceiling=50)

    assert appended_rows(events), "唯一的一对被驱逐了"
    roles = backend.roles(1)
    assert "tool" in roles, "工具结果没进上下文"


def test_stopping_early_still_leaves_a_record_of_what_ran(store: MemoryStore) -> None:
    """中途停止消费：**已经执行过的调用必须留下记录**。

    这是把记录从 ``LoopDone`` 挪到逐对事件的全部理由。调用方（``ui/agent_worker.py``）
    在用户点停止时就跳出这个生成器，生成器随即被关闭、后端那条流也跟着断——而
    ``LoopDone`` 永远不会来。记录若只挂在它上面，那一刻之前**真的执行过**的调用
    （包括已经落库的写）会一行都不剩。一份"做过但没记"的逐字稿是这个项目唯一
    不能有的那种失效，而且它在界面上完全看不出来。

    这条测试自己就是那个中途停下的调用方：收到第一对就跳出循环。
    """
    spy = SpyTool()
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("spy", call_id="c1"))],
            [tool_chunk(call("spy", call_id="c2"))],
            [text_chunk("都查完了", done=True)],
        ]
    )
    stream = run_agent_loop(
        backend=backend,
        messages=[ChatMessage(role="user", content="帮我看看")],
        registry=ToolRegistry([spy]),
        ctx=ToolContext(store=store, session_id="S1", turn_seq=1),
        await_approval=lambda _request: ApprovalDecision.GRANT,
        budget=TokenBudget(window=8192),
        meter=TokenMeter(),
        tools_ceiling=ROOMY,
    )

    seen: list[Any] = []
    for event in stream:
        seen.append(event)
        if isinstance(event, MessagesAppended):
            break
    stream.close()

    assert len(spy.calls) == 1, "夹具本身没造出「跑了一轮就停下」这个形状"
    rows = appended_rows(seen)
    assert [row.role for row in rows] == ["assistant", "tool"]
    assert_tool_pairs_intact(rows)


# ── 参数校验的失败要回灌 ───────────────────────────────────────────────────


def test_bad_arguments_are_fed_back_for_the_model_to_fix(store: MemoryStore) -> None:
    """工具自己抛的参数错要变成一句"怎么改"，并进下一次请求。

    这是"派发口吃异常"那个设计在循环这一层的可见证据——少了它，模型拿不到任何
    反馈，只能重试同样的调用。
    """
    spy = SpyTool(raiser=ToolArgumentError("content 不能为空"))
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("那我换个说法")]])

    events, _ = drive(backend, ToolRegistry([spy]), store)

    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.ok is False
    assert "content 不能为空" in backend.last_text(1)


def test_messages_are_not_mutated(store: MemoryStore) -> None:
    """循环**不改动**传进来的消息列表。

    它是调用方的上下文（``assemble()`` 的产物），而调用方还要用它去记账、去显示。
    原地追加会让"这一次发了什么"变成"最后一次看它是什么"——账本从此对不上。
    """
    messages = [ChatMessage(role="user", content="帮我查")]
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("好了")]])

    drive(backend, ToolRegistry([SpyTool()]), store, messages=messages)

    assert len(messages) == 1
    assert len(backend.requests[1].messages) > 1


def test_approval_request_round_index_counts_within_the_round(store: MemoryStore) -> None:
    """一轮里两次要确认的调用 → 两个问询，``round_index`` 分别 0 和 1。"""
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[
            [
                tool_chunk(
                    call("write_memory", call_id="c1", content="甲"),
                    call("write_memory", call_id="c2", content="乙"),
                )
            ],
            [text_chunk("好了")],
        ]
    )

    events, _ = drive(backend, ToolRegistry([spy]), store)

    ask = [event.request for event in events if isinstance(event, ApprovalRequested)]
    assert [request.round_index for request in ask] == [0, 1]
    assert [request.call_id for request in ask] == ["c1", "c2"]


def test_approval_request_is_a_plain_dataclass() -> None:
    """``ApprovalRequest`` 不带任何界面或线程对象。

    它要跨线程送到界面线程去画卡片，所以它必须是**可安全传过去**的纯数据。
    这条测试拦的是"顺手把 worker 或者信号对象塞进去"。
    """
    request = ApprovalRequest(call_id="c", tool_name="t", summary="s")
    assert request.arguments == {}
    assert repr(request).startswith("ApprovalRequest(")
