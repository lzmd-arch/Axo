"""分层装配。

这个模块是"上下文长度限制"那个原点问题的落点，所以它最不能出错的地方不是"装得下
吗"——装不下只是少注入一条记忆——而是**装不下的时候它先丢谁**。丢错顺序的症状不是
报错，是模型突然变得不像它自己，而用户看到的只是一个"今天有点不对"的助手。

于是这里的核心断言都是关于**优先级**的：当前输入在不在、人格锚在不在、丢的确实是
最老的那一头。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from axo.llm.base import ChatMessage, ToolCall
from axo.session.assemble import (
    Assembly,
    Layer,
    MemoryInjection,
    _collect_tool_tail,
    _is_wide,
    _padded,
    _read_call,
    _scan_tool_history,
    assemble,
    stored_rows,
)
from axo.session.budget import TokenBudget, TokenMeter
from axo.session.store import Message

METER = TokenMeter()
#: 一个固定的窗口，够大以至于默认情况下什么都不用丢。具体的数不重要，
#: 重要的是每个测试自己决定要不要把它调小。
ROOMY = TokenBudget(window=16384, reserve_output=2048)


def msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def history(turns: int = 3) -> list[Message]:
    out: list[Message] = []
    for index in range(turns):
        out.append(msg("user", f"第 {index} 个问题。" * 8))
        out.append(msg("assistant", f"第 {index} 个回答。" * 8))
    return out


def memory(
    memory_id: str,
    content: str = "",
    *,
    tokens: int | None = None,
    created_at: float = 0.0,
    session_id: str = "",
    derived_from: tuple[str, ...] = (),
) -> MemoryInjection:
    body = content or f"这是一条记忆 {memory_id}。" * 4
    return MemoryInjection(
        memory_id=memory_id,
        content=body,
        tokens=tokens if tokens is not None else METER.text(body),
        explain=f"因语义 0.87 被注入，{tokens or 0} token",
        created_at=created_at,
        session_id=session_id,
        derived_from=derived_from,
    )


def system_of(assembly: Assembly) -> str:
    first = assembly.messages[0]
    assert first.role == "system", "第一条永远是系统消息"
    return first.content


# ── 消息序列的形状 ──────────────────────────────────────────────────────────


def test_the_current_input_is_always_last() -> None:
    """当前输入永远在最后一条。它在不在，比这一轮装了多少东西都重要。"""
    result = assemble(history(), current="现在几点了？", budget=ROOMY)
    assert result.messages[-1].role == "user"
    assert result.messages[-1].content == "现在几点了？"


def test_an_empty_current_input_is_refused() -> None:
    with pytest.raises(ValueError, match="current 不能为空"):
        assemble(history(), current="", budget=ROOMY)


def test_the_history_is_kept_in_order_between_the_system_and_the_input() -> None:
    result = assemble(history(3), current="继续。", budget=ROOMY)
    roles = [message.role for message in result.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in result.messages[1:3]] == [
        history(3)[0].content,
        history(3)[1].content,
    ]


def test_it_takes_the_stored_message_sequence_not_a_text_list() -> None:
    """接缝 3 的断言。

    输入必须能是完整的 :class:`Message`——因为 P3 要往历史里喂工具行，而工具行带
    结构（``tool_call_id``）。一个只吃 ``list[str]`` 的装配器到那时得整个重写。
    这里只要它能接收带额外字段的真实 ``Message`` 而不出错就够了。
    """
    rows = [
        Message(role="user", content="帮我读一下 README。", at=1.0),
        Message(
            role="assistant", content="读到了。", at=2.0, prompt_tokens=120, completion_tokens=44
        ),
    ]
    result = assemble(rows, current="继续。", budget=ROOMY)
    assert [m.content for m in result.messages[1:3]] == ["帮我读一下 README。", "读到了。"]


def test_the_ledger_total_is_measured_from_what_is_actually_sent() -> None:
    """**账本对不上就等于没有账本。**

    所以总额不是各层相加出来的，是对发出去的那串消息量的。这条断言把两者绑在
    一起：改任何一处让它俩脱钩，这里就红。
    """
    result = assemble(
        history(),
        current="问一句。",
        budget=ROOMY,
        persona=["你是 Axo。"],
        environment="模型：qwen3-4b",
        recalled=[memory("m1"), memory("m2")],
    )
    assert result.ledger.total_tokens == METER.messages(result.messages)
    assert result.ledger.over_budget == 0


def test_the_budget_is_paid_once_not_per_layer() -> None:
    """所有槽拼进**同一条**系统消息。

    分开发的话每一层都要付一次消息包装，而且服务端看到的系统提示被切成几段——
    "人格锚"就不再是那条不可动摇的首条消息了。
    """
    result = assemble(
        history(),
        current="问一句。",
        budget=ROOMY,
        persona=["你是 Axo。"],
        environment="模型：qwen3-4b",
        recalled=[memory("m1")],
    )
    assert sum(1 for m in result.messages if m.role == "system") == 1


# ── 六层各自归位 ────────────────────────────────────────────────────────────


def test_every_layer_lands_in_the_system_message() -> None:
    result = assemble(
        history(),
        current="路由器放哪了？",
        budget=ROOMY,
        persona=["你是 Axo，记忆独立于模型。"],
        environment="模型：qwen3-4b",
        recalled=[memory("m1", "路由器放在书房书架顶层。")],
    )
    system = system_of(result)
    assert "记忆独立于模型" in system, "L0"
    assert "qwen3-4b" in system, "L1"
    assert "路由器放在书房书架顶层" in system, "L2"


def test_the_system_message_is_first_even_when_it_is_empty() -> None:
    """没有系统内容时仍然发一条空的。

    无条件的包装开销换来的是账本里"总额 = 各层之和"这个恒等式。让 5 个 token 跟着
    内容的有无浮动，差额就无法对照了。
    """
    result = assemble(history(), current="问一句。", budget=ROOMY)
    assert result.messages[0].role == "system"
    assert result.messages[0].content == ""


def test_the_recall_block_has_a_header_so_it_is_not_read_as_dialogue() -> None:
    """不加表头的话，注入的记忆和用户说的话长得一样。"""
    result = assemble(
        history(), current="问。", budget=ROOMY, recalled=[memory("m1", "路由器在书房。")]
    )
    system = system_of(result)
    assert "以下是与当前问题相关的长期记忆：" in system
    assert "· 路由器在书房。" in system


def test_the_summary_is_injected_with_its_own_loss_statement() -> None:
    rows = history(6)
    result = assemble(rows, current="继续。", budget=TokenBudget(window=380, reserve_output=100))
    assert result.ledger.dropped_messages > 0, "窗口小到这个程度应该压掉一些"
    assert "已压缩" in system_of(result)


# ── 优先级：丢谁 ────────────────────────────────────────────────────────────


def test_the_persona_survives_eviction() -> None:
    """**核心承诺。** 窗口小到什么都装不下时，人格锚必须还在。

    它是"任何模型读到的完全一样"的唯一载体；它被挤掉，助手就不再是同一个助手，
    而这一轮看起来完全正常——只是答得不太像它。
    """
    for window in (300, 500, 1000, 2000, 4000):
        result = assemble(
            history(20),
            current="你记得我是谁吗？",
            budget=TokenBudget(window=window, reserve_output=100),
            persona=["你是 Axo。"],
        )
        assert "你是 Axo。" in system_of(result), f"window={window} 下人格锚被挤掉了"


def test_the_current_input_survives_eviction() -> None:
    """比人格锚更强的只有这一条：用户刚说的话永远发得出去。"""
    question = "我刚才说的那件事，你帮我再想想？" * 6
    result = assemble(
        history(20), current=question, budget=TokenBudget(window=300, reserve_output=100)
    )
    assert result.messages[-1].content == question


def test_the_oldest_history_is_dropped_first() -> None:
    """驱逐从最老的一头开始。丢最近的那头会让模型忘掉刚发生的事。"""
    rows = [msg("user", f"第 {i} 句。" + "填充" * 40) for i in range(8)]
    result = assemble(
        rows, current="最后一句。", budget=TokenBudget(window=420, reserve_output=100)
    )
    contents = [m.content for m in result.messages[1:-1]]
    assert contents, "至少该留下一句"
    assert "第 7 句" in contents[-1], "最近的那句一定在"
    assert "第 0 句" not in "".join(contents), "最老的那句先走"


def test_when_the_pool_is_squeezed_memory_wins() -> None:
    """池子被挤小时**先服务记忆**。

    **优先级与旧版相反。** 旧版把近处对话排在记忆之前，理由是"记忆丢得起、近处对话
    丢了就真没了"；而用户要的是「无论哪一个对话都要能够使用完整的记忆」——记忆被
    丢掉就是这一轮真的不知道，丢掉的旧对话还有 L4 摘要兜着。
    """
    budget = TokenBudget(window=16384, reserve_output=2048)
    anchor = "很啰嗦的人格锚。" * 1800
    rows = [msg("user", f"第 {i} 个问题。" * 60) for i in range(10)]
    memories = [memory(f"m{i}", f"第 {i} 条记忆。" * 60) for i in range(6)]
    assert METER.text(anchor) > budget.usable / 2, "前提：人格锚把池子压小了"

    result = assemble(rows, current="问一句。", budget=budget, persona=[anchor], recalled=memories)
    assert result.ledger.injected, (
        "池子被挤小时先服务记忆。顺序反过来（近处对话在前）的话这里会是 0 条，"
        "而账本上看不出区别——它只会显示「这一轮没注入任何记忆」，指向完全错误的排查方向。"
    )
    assert result.ledger.over_budget == 0


def test_a_memory_that_does_not_fit_does_not_block_smaller_ones() -> None:
    """L2 里装不下就跳过，不退出循环。

    不这么做的话，一条很长的记忆会把排在它后面所有短的一起挡在门外——而"挡在门外"
    在账本上看起来只是"没召回"，没人会想到是长度造成的。
    """
    huge = memory("huge", "长" * 3000)
    small = memory("small", "路由器在书房。")
    budget = TokenBudget(window=900, reserve_output=100)
    result = assemble(history(1), current="问一句。", budget=budget, recalled=[huge, small])
    assert result.ledger.injected_ids == ("small",), "大的装不下，小的仍然该进去"


def test_what_the_budget_kept_out_is_stated_in_the_ledger() -> None:
    """被额度挡住的条数要写出来。

    不写的话，账本看起来像"库里只有这几条相关"，而不是"还有几条装不下"——前者会
    让人去改召回算法，后者才指向该改的额度。
    """
    budget = TokenBudget(window=900, reserve_output=100)
    result = assemble(
        history(1),
        current="问一句。",
        budget=budget,
        recalled=[memory("huge", "长" * 3000), memory("small", "路由器在书房。")],
    )
    assert "另有 1 条装不下" in result.ledger.render()


# ── 一致性：额度是真约束 ────────────────────────────────────────────────────


def test_it_never_exceeds_the_usable_budget_when_it_can_help_it() -> None:
    """**核心断言。** 超出 ``usable`` 意味着真实值可能超出 ``input_limit``，而服务端
    的处理是**静默**截断最早的消息——那是人格锚的位置。

    这里拿一段比窗口长得多的历史去逼它：该丢的全丢完，剩下的自然就装得下了。
    """
    for window in (400, 700, 1200, 3000, 8000):
        budget = TokenBudget(window=window, reserve_output=100)
        result = assemble(
            history(30),
            current="问一句。",
            budget=budget,
            persona=["你是 Axo。"],
            environment="模型：qwen3-4b",
            recalled=[memory(f"m{i}") for i in range(10)],
        )
        assert result.ledger.over_budget == 0, f"window={window} 下超了：{result.ledger.render()}"


def test_a_current_input_longer_than_the_window_is_sent_anyway_and_reported() -> None:
    """用户的问题永远要发。发出去、然后**如实记在账本上**。

    "什么都不发"比"发一条超额的请求"更糟：前者用户看到的是一次无声的失败，后者
    至少会得到一句话，而账本解释了他为什么没得到更多。
    """
    budget = TokenBudget(window=300, reserve_output=100)
    result = assemble([], current="很长的问题。" * 100, budget=budget)
    assert result.messages[-1].content == "很长的问题。" * 100
    assert result.ledger.over_budget > 0
    assert any("超出额度" in note for note in result.ledger.notes)
    assert "⚠" in result.ledger.render()


def test_there_are_no_layer_shares_left() -> None:
    """**没有份额这回事了。** 以前有一张 L2=0.25 / L3=0.50 / L5=0.20 的分配表。

    用户的要求是「除了官方的上下文限制之外，不要出现任何其他的预算、额度、上下文
    限制」。这条断言守的是那个符号**不许回来**：装箱的唯一上限是窗口本身，谁先装谁
    优先，前一层花剩多少后一层拿多少。
    """
    # **不能用 ``import axo.session.assemble as module``**：``axo.session.__init__``
    # 把同名的函数 ``assemble`` 重导出成了 ``axo.session.assemble`` 这个属性，于是
    # 那句话拿到的是函数而不是模块（第一次写就是这么炸的）。
    import importlib

    module = importlib.import_module("axo.session.assemble")
    # 判据是**符号在不在**，不是源码里提没提它（注释里当然会提到"以前有这么个东西"）。
    assert not hasattr(module, "layer_ceiling"), "layer_ceiling 回来了"
    assert not hasattr(module, "_SHARES"), "_SHARES 回来了"
    assert "layer_ceiling" not in module.__all__, "它又出现在导出表里了"


def test_the_ceiling_column_shrinks_from_layer_to_layer() -> None:
    """账本那一栏现在记的是「手上还剩多少」，不是「分到多少」。

    随层递减是它唯一的形状：记忆先拿、拿完剩下的归近处对话、再剩下的归工具结果。
    """
    budget = TokenBudget(window=16384, reserve_output=2048)
    result = assemble(
        history(2),
        current="问。",
        budget=budget,
        recalled=[memory("m1", "路由器在书房。")],
    )
    ceilings = [
        result.ledger.layer(layer).ceiling for layer in (Layer.RECALL, Layer.RECENT, Layer.TOOLS)
    ]
    assert all(value is not None for value in ceilings), ceilings
    assert ceilings[0] >= ceilings[1] >= ceilings[2], f"没有随层递减：{ceilings}"


# ── 跨层不变量 ──────────────────────────────────────────────────────────────


def test_nothing_is_invented_and_nothing_is_lost_silently() -> None:
    """每一条保留下来的历史都**逐字**出现在消息序列里，顺序不变。

    装配器做的是取舍，不是改写。一旦它开始"顺手整理"历史，账本就再也对不上逐字稿。
    """
    rows = history(6)
    result = assemble(rows, current="继续。", budget=TokenBudget(window=700, reserve_output=100))
    kept = [m.content for m in result.messages if m.role in ("user", "assistant")]
    assert kept[-1] == "继续。"
    body = kept[:-1]
    assert body, "窗口 700 不至于一句历史都留不下"
    # 保留的是原历史的一个后缀，逐字相同
    expected = [m.content for m in rows][-len(body) :]
    assert body == expected


def test_the_kept_run_starts_at_a_user_message() -> None:
    """开头不能是一句没有问题的回答。

    没有这个清理，模型看到的第一条历史是助手在回答一个已经被压进摘要的问题，它会
    把上一轮的回答当成开场白顺着往下说——不报错，只是从头就错位。
    """
    rows = [msg("user", "很早的问题。" * 30), msg("assistant", "很早的回答。" * 30)]
    rows += [msg("user", f"第 {i} 句。" * 10) for i in range(3)]
    for window in range(400, 1400, 60):
        result = assemble(
            rows, current="最后一句。", budget=TokenBudget(window=window, reserve_output=100)
        )
        body = list(result.messages[1:-1])
        if body:
            assert body[0].role == "user", f"window={window} 下开头是 {body[0].role}"


def test_the_input_history_is_not_mutated() -> None:
    """装配器会在驱逐之前调 ``summarize``，之后还要拿同一批消息做判断。"""
    rows = history(5)
    snapshot = [(m.role, m.content) for m in rows]
    assemble(rows, current="继续。", budget=TokenBudget(window=700, reserve_output=100))
    assert [(m.role, m.content) for m in rows] == snapshot


def test_rows_this_version_does_not_understand_are_skipped_not_fatal() -> None:
    """存储层出现一条本版本不认识的记录（P3 的 ``tool`` 行），装配不该整个失败
    ——那会让界面卡在"生成中"。"""
    rows = [
        msg("user", "帮我读一下 README。"),
        Message(role="tool", content="工具结果", at=2.0),
        msg("assistant", "读到了。"),
    ]
    result = assemble(rows, current="继续。", budget=ROOMY)
    assert [m.content for m in result.messages[1:-1]] == ["帮我读一下 README。", "读到了。"]


def test_the_same_input_always_gives_the_same_assembly() -> None:
    """确定性。"这一轮为什么注入这段文字"要可复现，账本才有意义。"""
    rows = history(5)
    kwargs = {
        "current": "继续。",
        "budget": TokenBudget(window=700, reserve_output=100),
        "persona": ["你是 Axo。"],
        "recalled": [memory("m1"), memory("m2")],
    }
    first = assemble(rows, **kwargs)
    for _ in range(5):
        again = assemble(rows, **kwargs)
        assert again.ledger.render() == first.ledger.render()
        assert [m.content for m in again.messages] == [m.content for m in first.messages]


# ── 账本给人看的那一面 ──────────────────────────────────────────────────────


def test_the_rendered_ledger_names_every_layer() -> None:
    result = assemble(
        history(),
        current="问一句。",
        budget=ROOMY,
        persona=["你是 Axo。"],
        environment="模型：qwen3-4b",
        recalled=[memory("m1")],
    )
    text = result.ledger.render()
    for label in ("人格锚", "环境", "记忆召回", "近期对话", "历史摘要"):
        assert label in text, f"账本里少了 {label}"


def test_the_rendered_ledger_carries_the_per_memory_attribution() -> None:
    """§3.4 的验收那一行。**模板生成，不经过模型。**

    让模型解释"我为什么记得这件事"是幻觉的标准配方——解释本身会被读成事实。
    """
    result = assemble(
        history(),
        current="路由器放哪了？",
        budget=ROOMY,
        recalled=[
            MemoryInjection(
                memory_id="m1",
                content="路由器在书房。",
                tokens=184,
                explain="因语义 0.87 · 词法 0.31 被注入，184 token；权重 0.62",
            )
        ],
    )
    assert "因语义 0.87 · 词法 0.31 被注入，184 token" in result.ledger.render()


def test_the_rendered_ledger_lines_up() -> None:
    """中文标签按**显示宽度**补齐，不是按字符数。

    用 ``ljust`` 会让"人格锚"和"记忆召回"差出两格——而参差不齐的账本比没有账本更
    难看懂，因为读者会以为那个错位本身有含义。
    """
    labels = ("人格锚", "环境", "记忆召回", "近期对话", "历史摘要")
    widths = {sum(2 if _is_wide(c) else 1 for c in _padded(label)) for label in labels}
    assert len(widths) == 1, f"补齐后的显示宽度不一致：{widths}"
    assert _padded("人") == "人" + " " * 8
    assert _padded("记忆召回") == "记忆召回" + " " * 2


def test_the_ledger_is_json_able_through_its_budget() -> None:
    """额度那一层要能被记进日志——否则"这一轮为什么只注入了三条"事后无法复盘。"""
    ledger = assemble(history(), current="问。", budget=ROOMY).ledger
    record = ledger.budget.to_dict()
    assert record["usable"] == ROOMY.usable
    assert record["window"] == 16384


# ── L5 工具结果 ────────────────────────────────────────────────────────────
#
# L5 的形状与别层都不同：它的单位是**对**（一条带 ``tool_calls`` 的 assistant 加
# 紧随其后、答复它的结果行），而"绝不半留"是它唯一的硬不变量。半个配对不是"少了一点
# 上下文"——它是模型无法解释的一段对话，而服务端的反应通常是**拒掉整个请求**：一次
# "最近调过工具"的会话之后每一轮都失败，症状离原因很远。
#
# 所以这里的负面测试比正面多：每一种半对都各有一条断言。


def tool_call(call_id: str, name: str = "search_memory", **arguments: object) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def tool_round(
    index: int,
    *,
    call_id: str | None = None,
    result: str | None = None,
    no_replay: str | None = None,
    tool_name: str = "search_memory",
) -> list[Message]:
    """一轮"模型要调工具 → 拿到结果"。

    ``no_replay`` 是那一对被标上的"别在后续轮次回放"的原因（见 ``Message.no_replay``）。
    """
    cid = call_id or f"call_{index}"
    return [
        Message(
            role="assistant", content=f"第 {index} 轮我先查一下。", tool_calls=(tool_call(cid),)
        ),
        Message(
            role="tool",
            content=result if result is not None else f"第 {index} 轮的结果。" * 4,
            tool_call_id=cid,
            tool_name=tool_name,
            no_replay=no_replay,
        ),
    ]


def tool_rows(assembly: Assembly) -> list:
    return [message for message in assembly.messages if message.role == "tool"]


def assert_tool_pairs_wellformed(messages) -> None:
    """两个方向都查：孤儿 ``tool`` 行、以及没人答复的 ``tool_calls`` 声明。"""
    pending: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            assert not pending, f"上一轮的调用没有结果：{sorted(pending)}"
            pending = {call.id for call in message.tool_calls}
        elif message.role == "tool":
            assert message.tool_call_id in pending, f"孤儿 tool 行：{message.tool_call_id}"
            pending.discard(message.tool_call_id)
        else:
            assert not pending, f"{message.role} 行之前还有没答复的调用：{sorted(pending)}"
    assert not pending, f"最后一条 assistant 的调用没有结果：{sorted(pending)}"


def test_tool_rows_go_to_L5_and_not_into_L3() -> None:
    """工具行整对归 L5，L3 里一条都不许有。

    ``kept_messages`` 是这里的关键断言：它数的是 L3 装了几条。工具行漏进 L3 的话，
    这个数会从 4 变成 6——而**界面上的表现完全正常**，只是模型有时会在一个已经答复过
    的调用上再问一次。
    """
    result = assemble(
        [
            msg("user", "帮我查一下路由器放哪了。"),
            *tool_round(0),
            msg("assistant", "路由器在书房。"),
            msg("user", "那备用钥匙呢？"),
            msg("assistant", "在鞋柜最上面那格。"),
        ],
        current="谢谢。",
        budget=ROOMY,
    )

    assert len(tool_rows(result)) == 1
    tools = result.ledger.layer(Layer.TOOLS)
    # 一条结果行 + 它那条 assistant 行 = L5 的 2 条
    assert tools.count == 2
    assert tools.tokens > 0
    # L3 只收下四行纯对话（两个问题 + 两个回答），工具那两行不在里面
    assert result.ledger.kept_messages == 4, result.ledger.render()


def test_L5_sits_between_the_system_message_and_the_recent_turns() -> None:
    """位置：system → L5 → L3 → 本轮输入。

    这是相对真实时间的**重排**（工具历史整体提前）。钉住它是因为"位置"在这里是件
    有后果的事：工具对按时间插进 L3 会破坏 L3 的连续性、也会撞上"开头不能是 assistant"
    那条不变量，而错位的表现是模型把上一轮的回答当成开场白。
    """
    result = assemble(
        [*tool_round(0), msg("user", "那备用钥匙呢？"), msg("assistant", "在鞋柜里。")],
        current="谢谢。",
        budget=ROOMY,
    )
    roles = [message.role for message in result.messages]
    assert roles[0] == "system"
    assert roles[1:3] == ["assistant", "tool"], roles
    assert roles[-1] == "user"


def test_an_orphan_tool_row_is_dropped_and_reported() -> None:
    """缺 ``tool_call_id`` 的孤儿行**丢掉并记账**。

    丢是必须的（它不成对），**说**也是必须的：不说的话，一次"工具结果没进上下文"会被
    读成"工具没被调用"，而这两件事要修的地方完全不同。
    """
    result = assemble(
        [
            msg("user", "查一下。"),
            Message(role="tool", content="没有配对的结果。", tool_call_id=None),
            msg("assistant", "查到了。"),
        ],
        current="好。",
        budget=ROOMY,
    )
    assert tool_rows(result) == []
    assert result.ledger.layer(Layer.TOOLS).count == 0
    assert any("不构成完整的对" in note for note in result.ledger.notes), result.ledger.notes


def test_the_tool_rows_survive_a_round_trip_through_the_store() -> None:
    """从逐字稿读回来的工具行照样成对。

    落在 ``Message`` 上的三个字段（``tool_calls`` / ``tool_call_id`` / ``tool_name``）是
    "逐字稿能回放工具轮次"的全部依据，而它们的**写出去**与**读回来**是两段代码——只测
    其中一段的话，字段名写错一个字母不会有任何东西变红。
    """
    written = [Message.from_dict(message.to_dict()) for message in tool_round(0)]

    result = assemble(
        [msg("user", "查一下。"), *written, msg("assistant", "查到了。")],
        current="好。",
        budget=ROOMY,
    )
    assert len(tool_rows(result)) == 1
    assert result.messages[1].tool_calls[0].id == "call_0"
    assert result.messages[1].tool_calls[0].name == "search_memory"
    assert result.messages[1].tool_calls[0].arguments == {}
    assert tool_rows(result)[0].tool_call_id == "call_0"


# ── 反方向：循环产出的 ChatMessage 落成逐字稿的行 ──────────────────────────
#
# 工具循环手上是 ``ChatMessage``（协议形状），逐字稿要的是 ``Message``（存储形状）。
# 这一步是 P3a 第 8 步才接上的那一段：没有它，工具在界面上真的跑了、结果也回灌了，
# 而**回看时那几轮不存在**——一种没有任何报错的失效。


def test_a_tool_row_gets_its_name_from_the_call_it_answers() -> None:
    """``tool_name`` 是从**前一条 assistant 行**的 ``tool_calls`` 里查出来的。

    因为 ``ChatMessage`` 上没有这个字段——它是回看显示用的，协议只需要 ``tool_call_id``
    （回灌工具结果时）。名字查不到的后果不是报错，是回看时那一行只显示一个 32 字符的 id。
    """
    rows = stored_rows(
        [
            ChatMessage(
                role="assistant",
                content="我看一下那个文件。",
                tool_calls=(ToolCall(id="call_9", name="read_file", arguments={"path": "a.txt"}),),
            ),
            ChatMessage(role="tool", content="文件内容。", tool_call_id="call_9"),
        ]
    )

    assert [row.role for row in rows] == ["assistant", "tool"]
    assert rows[1].tool_call_id == "call_9"
    assert rows[1].tool_name == "read_file"


def test_a_call_survives_the_write_read_round_trip() -> None:
    """``stored_rows`` 写出去的那份，``_read_call`` 原样读回来。

    这两个函数在源文件里挨着放，就是为了这条断言能挨着写：字段名差一个字母时，两边
    各自都"正常"，而症状要到"模型说它没收到那次工具结果"才显形——那时离这里已经很远。
    """
    call = ToolCall(
        id="call_3", name="set_reminder", arguments={"when": "明天九点", "what": "吃药"}
    )
    written = stored_rows([ChatMessage(role="assistant", content="", tool_calls=(call,))])

    assert _read_call(written[0].tool_calls[0]) == call


def test_a_whole_pair_lands_so_the_transcript_can_be_replayed() -> None:
    """一整对落下去之后，``assemble()`` 读回来仍然成对。

    单独测转换函数是不够的：转换对了、但落盘少了字段或者顺序反了，照样换不回一对
    完整的对话。所以这里走完**产出 → 落盘 → 读回 → 装配**整条路。
    """
    produced = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="get_time", arguments={}),),
        ),
        ChatMessage(role="tool", content="现在是十点。", tool_call_id="c1"),
    ]
    written = [Message.from_dict(row.to_dict()) for row in stored_rows(produced)]

    result = assemble(
        [msg("user", "几点了？"), *written, msg("assistant", "十点。")],
        current="谢谢。",
        budget=ROOMY,
    )

    assert [row.role for row in tool_rows(result)] == ["tool"]
    assert tool_rows(result)[0].content == "现在是十点。"
    assert result.messages[1].tool_calls[0].name == "get_time"
    assert_tool_pairs_wellformed(result.messages)


def test_a_broken_pair_is_still_recorded_rather_than_thrown_away() -> None:
    """配不上名字的 tool 行**照样落盘**，只是 ``tool_name`` 为空。

    这里刻意与 :func:`_read_call` 的严格相反：那一边读到的东西可能来自任何一个旧版本，
    所以抛；这一边的输入是自己刚产出的。多一道校验只会在真出问题时把一条**已经发生过**
    的对话整段挡住——记录宁可残缺也要留下。
    """
    rows = stored_rows([ChatMessage(role="tool", content="没有前文的结果。", tool_call_id="cX")])

    assert len(rows) == 1
    assert rows[0].role == "tool"
    assert rows[0].tool_call_id == "cX"
    assert rows[0].tool_name is None


def test_plain_rows_come_through_untouched() -> None:
    """user/assistant/system 行不该被这一步改出任何东西——工具轮次前后的正文都在里面。"""
    rows = stored_rows(
        [
            ChatMessage(role="user", content="你好。"),
            ChatMessage(role="assistant", content="你也好。"),
            ChatMessage(role="system", content="（环境）"),
        ]
    )

    assert [(row.role, row.content) for row in rows] == [
        ("user", "你好。"),
        ("assistant", "你也好。"),
        ("system", "（环境）"),
    ]
    assert all(not row.tool_calls and row.tool_call_id is None for row in rows)


def test_no_tool_rows_means_L5_is_empty_and_changes_nothing_else() -> None:
    """没有工具行时，L5 是 0 token 的一层——**别层的额度逐值不变**。

    这是回归为零的那一条：``_SHARES`` 多了一项 0.20，若它写成"先扣再算"，L4 的池子会
    凭空少掉五分之一，而账本上看不出是它拿走的。这里同时钉住 L2/L3 的上限没被 L5 影响。
    """
    result = assemble(history(), current="问。", budget=ROOMY)
    tools = result.ledger.layer(Layer.TOOLS)
    assert (tools.tokens, tools.count) == (0, 0)
    # 别层该拿多少照旧——工具层为空时不该顺手改掉别人的账
    assert result.ledger.layer(Layer.RECALL).count == 0


def test_the_tool_declaration_cost_comes_off_the_pool_and_is_recorded() -> None:
    """工具声明的开销单独记一笔，并从池子里扣掉。

    它**不在** ``messages`` 里（走 ``ChatRequest.tools``，一个与 ``messages`` 平行的
    字段），所以量具看不到它。Step 0 实测 3 个工具净增 311 token/轮，而**每轮都要付**。
    不在装配时扣掉的后果与"少算一层"一模一样：服务端静默截断最早的消息，也就是人格锚。
    """
    declared = 311
    recalled = [memory(f"m{index}") for index in range(20)]
    free = assemble(history(), current="问。", budget=ROOMY, recalled=recalled)
    charged = assemble(
        history(), current="问。", budget=ROOMY, recalled=recalled, tools_tokens=declared
    )

    assert free.ledger.tools_tokens == 0
    assert charged.ledger.tools_tokens == declared
    # 真的从池子里扣了：L4 可供支配的余量少了一整笔声明的钱
    assert charged.ledger.layer(Layer.SUMMARY).ceiling == (
        free.ledger.layer(Layer.SUMMARY).ceiling - declared
    )
    # 账本读得出来，而且是单独一行——与 L5 的"工具结果"分得开
    rendered = charged.ledger.render()
    assert "工具声明" in rendered and str(declared) in rendered


def test_an_oversized_declaration_is_counted_as_over_budget() -> None:
    """声明把余额吃光时，``over_budget`` 要算上它。

    只按 ``messages`` 算的话，账本会在"其实已经超了"的时候报 0——而这个字段是唯一一个
    会说"服务端可能要截断人格锚"的地方。
    """
    result = assemble(
        [],
        current="问。",
        budget=TokenBudget(window=400, reserve_output=100),
        tools_tokens=10_000,
    )
    assert result.ledger.over_budget > 0
    assert any("超出额度" in note for note in result.ledger.notes)


def test_a_negative_declaration_cost_cannot_inflate_the_budget() -> None:
    """负的声明开销按 0 处理。

    这个参数最终来自一次量测。量测出错、或者调用方传 ``-1`` 当"没有"时，**不能**反过来
    把池子撑大——那是"一个坏输入让上下文超发"的最短路径。
    """
    plain = assemble([], current="问。", budget=ROOMY)
    result = assemble([], current="问。", budget=ROOMY, tools_tokens=-5000)
    assert result.ledger.tools_tokens == 0
    assert result.ledger.layer(Layer.SUMMARY).ceiling == plain.ledger.layer(Layer.SUMMARY).ceiling


def test_an_empty_declaration_list_is_not_a_pair() -> None:
    """``tool_calls`` 为空的 assistant 行就是一条普通消息，不是"一对"。

    反空转：不写这条的话，一个把"但凡 role 是 assistant"都当头部去凑对的实现也能过上面
    全部测试，而它会把普通对话行卷进 L5、从 L3 里挖走。
    """
    plain = [msg("assistant", "就是一句话。")]
    assert _scan_tool_history(plain) == ([], 0, 0)

    result = assemble([msg("user", "问。"), *plain], current="再问。", budget=ROOMY)
    assert result.ledger.layer(Layer.TOOLS).tokens == 0
    # 它留在 L3 里，没有被 L5 挖走
    assert result.ledger.kept_messages == 2


# ── 整对原子：收不下的那对整对丢 ────────────────────────────────────────────


def test_a_pair_that_does_not_fit_is_dropped_whole_never_half() -> None:
    """装不下就**整对**丢，绝不留半个。

    三条断言一起看才有意义：留下的是最新那一对、丢的是最老的两对、而且没有孤儿
    ``tool`` 行（半个配对的这一半）。
    """
    rounds = [row for index in range(3) for row in tool_round(index)]
    pairs, broken, skipped = _scan_tool_history(rounds)
    assert len(pairs) == 3 and broken == 0 and skipped == 0, (
        "夹具本身没造出三对，下面的断言是空转的"
    )

    # 额度刚好等于一对的价钱：最新那对进得去，再往前一对就差一个 token
    one = sum(METER.message(row) for row in pairs[-1])
    tail = _collect_tool_tail(rounds, one, METER)

    assert tail.pairs == 1 and tail.dropped == 2
    assert [row.content for row in tail.rows if row.role == "tool"] == [rounds[-1].content]
    assert tail.rows[0].role == "assistant"


def test_the_tail_stops_at_the_first_pair_that_does_not_fit() -> None:
    """装不下**就停**，不跳过。

    与 L2 的"跳过这一条继续看下一条"相反：工具对是一条时间线，中间留个洞等于对模型说
    "中间那步没发生过"。所以这里造一个"最老一对很小、中间那对很大"的历史，断言结果里
    **只有最新的一对**——一个跳过式的实现会把那对小的也装回来。

    额度是这里唯一的技术活：它必须**装得下最新一对加最老那一对**（否则两种实现都只留
    一对，测试分不出它们），同时**装不下中间那对**。第一版把额度取成"刚好一对的价钱"，
    于是池子每次都在最新那对之后正好用光——注入"改成 ``continue``"时它绿着过去了。
    """
    rounds = [
        *tool_round(0, result="甲"),
        *tool_round(1, result="很长的结果。" * 200),
        *tool_round(2, result="丙"),
    ]
    pairs = _scan_tool_history(rounds)[0]
    assert len(pairs) == 3, "夹具本身没造出三对"
    small = sum(METER.message(row) for row in pairs[0])
    huge = sum(METER.message(row) for row in pairs[1])
    ceiling = small * 2 + 1
    assert huge > ceiling, f"中间那对（{huge}）没大到能让两种实现分开（额度 {ceiling}）"

    tail = _collect_tool_tail(rounds, ceiling, METER)

    assert tail.pairs == 1, "跳过式的实现会把中间那对挡不住的、更老的短对也装回来"
    assert [row.content for row in tail.rows if row.role == "tool"] == ["丙"]


def test_every_broken_shape_is_rejected() -> None:
    """五种坏形状，一种都不许成对。

    表驱动，因为每一种单独看都像"大概没事"：只丢结果行、只丢声明行、多一条没人要的
    结果、同一次调用被答复两次、id 撞车——而它们各自的后果都是服务端拒请求。
    """
    head = Message(role="assistant", content="查一下。", tool_calls=(tool_call("c1"),))
    cases: dict[str, list[Message]] = {
        "结果答复了不在声明里的调用": [
            head,
            Message(role="tool", content="x", tool_call_id="别的"),
        ],
        "声明了却没有结果": [head],
        "同一次调用被答复两次": [
            head,
            Message(role="tool", content="x", tool_call_id="c1"),
            Message(role="tool", content="y", tool_call_id="c1"),
        ],
        "落盘的调用形状坏了": [
            Message(role="assistant", content="查一下。", tool_calls=({"name": "x"},)),
            Message(role="tool", content="x", tool_call_id="c1"),
        ],
        "两条调用的 id 撞车": [
            Message(
                role="assistant",
                content="查一下。",
                tool_calls=(tool_call("c1"), tool_call("c1")),
            ),
            Message(role="tool", content="x", tool_call_id="c1"),
        ],
    }
    for why, rows in cases.items():
        pairs, broken, skipped = _scan_tool_history(rows)
        assert pairs == [], f"{why}：这本该不成对"
        assert broken > 0, f"{why}：坏了却没有记下来"
        # **坏 ≠ 按策略不回放。** 这两种都让 ``pairs`` 为空，所以只断言"没成对"的话，
        # 一个把所有对都算进 ``skipped`` 的实现也能过——而那样账本上"数据坏了"会
        # 伪装成"这一轮拒了几次"。
        assert skipped == 0, f"{why}：这是坏形状，不是按策略不回放"


def test_a_pair_never_leaves_an_orphan_behind_in_the_assembled_request() -> None:
    """被额度挤掉的工具轮次**整个**不出现，不留下任何一半。

    对着 ``assemble()`` 端到端断言，因为"L5 丢了一半"这件事只有在拼好的消息序列上才
    看得见——:func:`_collect_tool_tail` 的返回值本身是良构的。

    额度刻意卡在"一对装得下、两对装不下"之间：全丢光的话这条测试是空转的（没有任何
    半对可留），全留下也一样。所以下面那段前提断言不是装饰——**估计器一旦重标定**，
    卡住的那个窗口就会漂出这个区间，而漂移之后这条测试仍然会绿，只是什么都没在测。
    """
    rows = [
        msg("user", "先查三样东西。"),
        *tool_round(0, result="第一条结果。" * 40),
        *tool_round(1, result="第二条结果。" * 40),
        *tool_round(2, result="第三条结果。" * 40),
        msg("assistant", "都查完了。"),
    ]
    # 窗口收到「一对装得下、两对装不下」。份额拆掉之后工具层拿的是剩下的全部，
    # 所以这条前提要靠窗口本身来造，不能再指望 0.20 那个份额。
    budget = TokenBudget(window=700, reserve_output=200)
    price = sum(METER.message(row) for row in _scan_tool_history(rows)[0][-1])

    result = assemble(rows, current="好。", budget=budget)
    ceiling = result.ledger.layer(Layer.TOOLS).ceiling or 0
    assert price <= ceiling < 2 * price, (
        f"额度没卡在一对与两对之间（一对 {price}，额度 {ceiling}），这条测试会变成空转"
    )
    tools = result.ledger.layer(Layer.TOOLS)
    # 丢掉的对数只出现在 note 里（见 ``_ToolTail``：对与行分开记）
    assert "对装不下" in tools.note, result.ledger.render()
    assert tools.count == 2, f"应当恰好回放一对；{result.ledger.render()}"
    assert_tool_pairs_wellformed(result.messages)


# ── 按策略不回放：闸门拒绝与过期快照 ────────────────────────────────────────
#
# 这一组守的是两个**安静的**失效，两个都只在"回放得太多"这个方向上出错：
#
# * 一次闸门拒绝被搬进后续轮次的上下文，于是在模型眼里从"这一次调用被跳过了"变成
#   一条**仍在生效的规矩**——症状是模型此后再也不肯调那个工具，而用户看不出为什么；
# * ``get_time``/``read_file`` 的结果是**某一刻的快照**，回放它等于把过期的时间戳和
#   旧的文件正文当成当前值递过去——模型会照它回答，且没有任何线索知道那是旧的。
#
# 两条都断言**端到端**（对着 ``assemble()`` 拼出来的消息序列），而不是对着
# ``_scan_tool_history`` 的返回值：后者是良构的，"这一对被搬进去了"这件事只有在
# 拼好的序列上才看得见。


def test_a_denied_pair_is_never_replayed_into_a_later_turn() -> None:
    """被拒绝的那一对**留在逐字稿里，但不进后续轮次**。

    两件事都要断言，因为它们是一对：**留着**是因为用户回看时该看到"这一步被跳过了"，
    **不回放**是因为那句话是当轮说的。只断言后一半的话，一个"干脆不落盘"的实现也能过
    ——而它会让逐字稿少掉一整步。
    """
    rows = [
        msg("user", "帮我记一下钥匙在鞋柜里。"),
        *tool_round(
            0,
            result="用户没有同意执行 write_memory，这一次调用跳过了。",
            no_replay="denied",
            tool_name="write_memory",
        ),
        msg("user", "那算了。"),
    ]

    result = assemble(rows, current="还有别的吗？", budget=ROOMY)

    assert result.ledger.layer(Layer.TOOLS).tokens == 0, (
        "被拒绝的那一对被回放了——那句『不要重试』会变成一条长期规矩"
    )
    assert not any("这一次调用跳过了" in message.content for message in result.messages)


def test_a_stale_snapshot_is_never_replayed_into_a_later_turn() -> None:
    """``get_time`` 那种快照同样不回放——回放它比不回放更坏。

    不回放的代价只是"模型要重新调一次"；回放的代价是它拿着十分钟前（或昨天）的时间戳
    回答"现在几点"，而且看不出有什么不对。
    """
    rows = [
        msg("user", "现在几点？"),
        *tool_round(
            0, result="2026-10-10 星期六 14:32（本地时间）", no_replay="stale", tool_name="get_time"
        ),
        msg("user", "好。"),
    ]

    result = assemble(rows, current="那再过一小时是几点？", budget=ROOMY)

    assert result.ledger.layer(Layer.TOOLS).tokens == 0
    assert not any("14:32" in message.content for message in result.messages)


def test_normal_pairs_are_still_replayed() -> None:
    """反空转：**普通的工具结果照旧回放**。

    没有这条，"把所有工具对都不回放"也能过上面两条——而那等于把 L5 整个删掉，
    代价是模型每轮都要把它已经查过的东西重查一遍（而 L5 存在的全部理由就是这个）。
    """
    rows = [
        msg("user", "钥匙在哪？"),
        *tool_round(0, result="钥匙在鞋柜里。", tool_name="search_memory"),
        msg("user", "那伞呢？"),
    ]

    result = assemble(rows, current="谢谢。", budget=ROOMY)

    assert result.ledger.layer(Layer.TOOLS).tokens > 0, "普通结果没被回放——L5 等于没了"
    assert any("钥匙在鞋柜里" in message.content for message in result.messages)
    assert_tool_pairs_wellformed(result.messages)


def test_a_skipped_pair_is_not_counted_as_dropped_or_broken() -> None:
    """账本上三种情形分列：**按策略不回放** ≠ 装不下 ≠ 数据坏了。

    混成一个数就分不出该修哪儿：``dropped`` 多指向额度该调大，``broken`` 多指向落盘的
    数据坏了，``skipped`` 多只是说明这一会话里被拒过几次、查过几次"当前值"。
    """
    rows = [
        msg("user", "问。"),
        *tool_round(0, no_replay="denied"),
        *tool_round(1, result="正常结果。" * 4),
    ]

    tail = _collect_tool_tail(rows, ceiling=10_000, meter=METER)

    assert (tail.pairs, tail.dropped, tail.broken, tail.skipped) == (1, 0, 0, 1), tail
    assert "1 对按策略不回放" in tail.note
    # 一个被跳过、一个被留下，而且留下的那对是**完整**的
    assert [row.content for row in tail.rows if row.role == "tool"] == ["正常结果。" * 4]


# ── 记忆行的时间戳 ──────────────────────────────────────────────────────────

WHEN = datetime(2026, 10, 13, 22, 41).timestamp()
SPAN = (
    datetime(2026, 10, 13, 13, 2).timestamp(),
    datetime(2026, 10, 13, 13, 5).timestamp(),
)


def test_every_recall_line_carries_its_time() -> None:
    """注入行必须带时间戳——这是「模型能形成时间观念」的唯一来源。

    环境行给了「现在几点」，记忆行不给「那件事是几点」，模型就只能猜先后。用户
    的原话是「记忆没有时间戳，模型无法形成时间观念」，落点就是这一行。
    """
    result = assemble(
        history(),
        current="昨天我说了什么？",
        budget=ROOMY,
        recalled=[memory("m1", "感冒那位回来了。", created_at=WHEN)],
    )

    assert "· [2026-10-13 22:41] 感冒那位回来了。" in system_of(result)


def test_a_merged_entry_shows_its_span_instead_of_one_point() -> None:
    """归并产物挂**区间**，不挂单点——区间才是「这件事从几点到几点」。"""
    merged = MemoryInjection(
        memory_id="g1",
        content="感冒那位回来了，还带了杯奶茶。",
        tokens=20,
        explain="整理产物",
        created_at=SPAN[0],
        span=SPAN,
    )
    result = assemble(history(), current="问。", budget=ROOMY, recalled=[merged])

    assert "· [2026-10-13 13:02–13:05] 感冒那位回来了，还带了杯奶茶。" in system_of(result)


def test_a_memory_without_a_time_gets_no_bracket() -> None:
    """没时间就不编时间。``created_at <= 0`` 渲染成没有方括号，绝不能变 1970。"""
    result = assemble(
        history(), current="问。", budget=ROOMY, recalled=[memory("m1", "路由器在书房。")]
    )
    system = system_of(result)

    assert "· 路由器在书房。" in system
    assert "1970" not in system
    assert "· [" not in system


def test_the_ledger_shows_the_same_stamp() -> None:
    """账本与上下文用同一个时间戳来源（``_stamp_of``）。

    两处各写一套就会出现「账本说 22:41、正文写 22:40」——而账本对不上就等于没有
    账本（同 ``layer_ceiling`` 那条注释的理由）。
    """
    result = assemble(
        history(),
        current="问。",
        budget=ROOMY,
        recalled=[memory("m1", "路由器在书房。", created_at=WHEN)],
    )

    assert "[2026-10-13 22:41] 因语义 0.87 被注入" in result.ledger.render()


# ── 注入时的临时归并（视图层，零写库） ──────────────────────────────────────


def test_fragments_of_one_event_merge_on_the_way_out() -> None:
    """还没被巩固过的新碎条目，在渲染那一层当场并成一条。

    这一层归并**零写库**：库里的形状完全不变，变的只是发给模型的那几行。
    """
    result = assemble(
        history(),
        current="问。",
        budget=ROOMY,
        recalled=[
            memory("a", "感冒那位回来了。", created_at=WHEN, session_id="s1"),
            memory("b", "还带了杯奶茶。", created_at=WHEN + 120, session_id="s1"),
        ],
    )
    system = system_of(result)

    assert "· [2026-10-13 22:41–22:43] 感冒那位回来了；还带了杯奶茶。" in system
    assert "原话：" in system
    assert system.count("感冒那位回来了") == 2, "头部一次、原话一次"


def test_a_derived_entry_replaces_the_fragments_it_stands_for() -> None:
    """整理条目代表它的那几条原句——原句不再单独列。

    它的正文里已经带着逐字原话，再列一遍就是同一段话在上下文里出现两次。
    """
    result = assemble(
        history(),
        current="问。",
        budget=ROOMY,
        recalled=[
            memory(
                "g1",
                "感冒那位回来了；还带了杯奶茶。",
                created_at=WHEN,
                session_id="s1",
                derived_from=("a", "b"),
            ),
            memory("a", "感冒那位回来了。", created_at=WHEN, session_id="s1"),
            memory("b", "还带了杯奶茶。", created_at=WHEN + 120, session_id="s1"),
        ],
    )
    system = system_of(result)

    assert system.count("· [") == 1, "只该有一行记忆：" + chr(10) + system
    assert "还带了杯奶茶。" in system


def test_the_timeline_runs_in_time_order_and_unknown_times_go_last() -> None:
    """按时间升序；时间不可知的排最后——排在最前会让模型以为那是最早的事。"""
    later = datetime(2026, 10, 13, 23, 59).timestamp()
    result = assemble(
        history(),
        current="问。",
        budget=ROOMY,
        recalled=[
            memory("late", "晚的。", created_at=later, session_id="s1"),
            memory("unknown", "没时间的。", session_id="s1"),
            memory("early", "早的。", created_at=WHEN, session_id="s1"),
        ],
    )
    system = system_of(result)

    body = system.split("长期记忆：", 1)[1]
    assert body.index("早的。") < body.index("晚的。") < body.index("没时间的。")


def test_fragments_from_different_sessions_stay_apart() -> None:
    """跨会话的两句不是「这一场里说的」，哪怕时间挨着也不并。"""
    result = assemble(
        history(),
        current="问。",
        budget=ROOMY,
        recalled=[
            memory("a", "感冒那位回来了。", created_at=WHEN, session_id="s1"),
            memory("b", "还带了杯奶茶。", created_at=WHEN + 60, session_id="s2"),
        ],
    )
    system = system_of(result)

    assert system.count("· [") == 2
    assert "原话：" not in system


def test_every_memory_gets_in_when_the_window_has_room() -> None:
    """库里 N 条 → 上下文里 N 条全在。**这是「完整记忆」的落点。**

    以前一次最多 8 条、还挤在 usable 的 25% 隔间里——「能回想起的记忆少」就死在那两处。
    现在条数只受窗口约束；窗口装得下就一条都不丢。
    """
    memories = [
        memory(f"m{index}", f"盐度探头第 {index} 次标定的读数各不相同。", created_at=WHEN + index)
        for index in range(20)
    ]

    result = assemble(history(), current="问。", budget=ROOMY, recalled=memories)

    assert len(result.ledger.injected) == 20, f"只进了 {len(result.ledger.injected)} 条"
    system = system_of(result)
    for index in range(20):
        assert f"盐度探头第 {index} 次标定" in system, f"第 {index} 条没进上下文"


def test_the_ledger_does_not_say_it_kept_zero_for_generation() -> None:
    """云端（不预留）时账本写「输出走后端默认」，不是「留 0 给生成」。

    「留 0」读着像 bug，而实际是"我们不设任何输出上限"——两件事差得很远，而账本是
    用户判断"这一轮为什么这样"的唯一入口。
    """
    cloud = TokenBudget(window=16384, reserve_output=0)
    local = TokenBudget(window=16384, reserve_output=2048)

    assert "输出走后端默认" in assemble(history(1), current="问。", budget=cloud).ledger.render()
    assert "留 2048 给生成" in assemble(history(1), current="问。", budget=local).ledger.render()


def test_a_row_without_any_text_never_reaches_the_model() -> None:
    """**没有正文的行不进 L3。** 表情包那一行就是这种：它是一张图，正文是空的。

    图是给**用户**看的（它压根不进上下文），所以这一行对模型来说没有任何东西。
    留着它的后果是请求里多一条空消息——而多数服务端会为此拒掉整个请求，症状离
    原因很远。
    """
    from axo.session.assemble import _to_chat

    sticker = Message(role="assistant", content="", image="stickers/开心.png")

    assert _to_chat(sticker) is None
    # 有正文的那一行照旧进（图只是搭在它上面）
    assert _to_chat(Message(role="assistant", content="看这个", image="x")).content == "看这个"
