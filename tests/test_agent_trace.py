"""工具轨迹：**做过什么，压成能装进 ``meta`` 的形状**。

这个文件量的是四件事，每一件对着一种"看不出来"的失效：

1. **参数要来另一路**。:class:`~biosnn_memterm.agent.loop.ToolFinished` 里没有
   参数，参数只在 :class:`~biosnn_memterm.agent.loop.MessagesAppended` 那一行
   assistant 的 ``tool_calls`` 里。配不上就写空——**配不上不等于这一步没发生**，
   所以那一步必须还在。
2. **截断**。``meta`` 逐字进 SQLite，超了 ``_MAX_META_BYTES`` 会让整条记忆变成一句
   "记忆写入失败"。而 ``fs.MAX_READ_BYTES`` 是 64 KiB——这不是假想的输入。
3. **步数永不丢**。被均分掉的是摘录，不是步；把"调了三次工具"压成"调了一次"是
   静默丢弃。
4. **摘录说得出口自己是摘录**。``result_chars`` 记原文长度，于是"600 字"与"原文只有
   600 字"在数据里分得开。
"""

from __future__ import annotations

import json

from biosnn_memterm.agent.loop import MessagesAppended, ToolFinished
from biosnn_memterm.agent.trace import (
    MAX_STEP_ARG_CHARS,
    MAX_STEP_RESULT_CHARS,
    MAX_TRACE_CHARS,
    build_trace,
)
from biosnn_memterm.llm.base import ChatMessage, ToolCall

# ── 造事件 ─────────────────────────────────────────────────────────────────
#
# 用真的 dataclass，不用替身：轨迹的全部输入就是这两类事件，而替身会把"字段改名"
# 这类改动挡在测试之外——那正是将来最可能发生的一种。


def finished(
    call_id: str = "c1",
    name: str = "read_file",
    *,
    ok: bool = True,
    content: str = "结果。",
    denied: bool = False,
    clipped: bool = False,
) -> ToolFinished:
    return ToolFinished(
        call_id=call_id,
        name=name,
        ok=ok,
        content=content,
        denied=denied,
        clipped=clipped,
    )


def appended(*calls: tuple[str, str, dict]) -> MessagesAppended:
    """一对工具消息。``(call_id, 工具名, 参数)`` 三元组，按执行顺序。"""
    rows = (
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=tuple(
                ToolCall(id=cid, name=name, arguments=args) for cid, name, args in calls
            ),
        ),
        *(ChatMessage(role="tool", content="", tool_call_id=cid) for cid, _, _ in calls),
    )
    return MessagesAppended(rows=rows)


def size_of(trace: object) -> int:
    """这条轨迹序列化之后多少字符。**``ensure_ascii=False``**：默认的转义会把
    汉字算成 6 个字符，于是量出来的东西不是存进库的那份。"""
    return len(json.dumps(trace, ensure_ascii=False))


# ── 配对 ───────────────────────────────────────────────────────────────────


def test_a_step_carries_the_arguments_the_model_gave() -> None:
    """参数从**配对事件**里来。拿不到它，轨迹就只剩"调过一次 read_file"——
    而"读的是哪个文件"才是这条记忆的来历。"""
    trace = build_trace(
        [finished("c1", "read_file")],
        [appended(("c1", "read_file", {"path": "README.md"}))],
    )
    (step,) = trace
    assert step["name"] == "read_file"
    assert step["arguments"] == {"path": "README.md"}


def test_arguments_survive_a_pair_with_several_calls() -> None:
    """一次 yield 带回来的一对里可以有好几个调用，按 ``call_id`` 各归各的。

    按下标硬配的话，第二个调用会拿到第一个的参数——而那种错配**看起来完全正常**，
    只是某个工具的参数悄悄变成了别人的。
    """
    trace = build_trace(
        [finished("c1", "read_file"), finished("c2", "search_memory")],
        [
            appended(
                ("c1", "read_file", {"path": "a.md"}),
                ("c2", "search_memory", {"query": "记忆"}),
            )
        ],
    )
    assert [step["arguments"] for step in trace] == [{"path": "a.md"}, {"query": "记忆"}]


def test_a_step_whose_pair_never_arrived_is_still_a_step() -> None:
    """**配不上 ≠ 没发生。**

    把配不上的那一步删掉，轨迹就成了一份"看不出少了什么"的记录——而它记的正是
    "这次对话真的发生过什么"。所以参数写空，步数照旧。
    """
    trace = build_trace([finished("c1"), finished("c2")], [])
    assert len(trace) == 2, "配不上参数的一步被整个删掉了"
    assert trace[0]["arguments"] == {}


def test_a_partial_trace_is_still_a_trace() -> None:
    """**被中断的那一轮**：循环被 break，配对事件可能一个都没到。

    那正是这个函数不放在 ``LoopDone`` 上的理由——用户按了停止，他做过的那些事
    仍然该留下。
    """
    trace = build_trace(
        [finished("c1", "read_file", content="半个结果")],
        [],
    )
    assert [step["name"] for step in trace] == ["read_file"]
    assert trace[0]["result"] == "半个结果"


def test_nothing_happened_gives_an_empty_trace() -> None:
    """没调过工具的一轮没有轨迹。**空元组而不是 ``None``**：调用方（``writer``）
    把它原样塞进 ``Turn``，而那个字段的类型是元组。"""
    assert build_trace((), ()) == ()


# ── 标志 ───────────────────────────────────────────────────────────────────


def test_the_flags_are_passed_through() -> None:
    """``ok`` / ``denied`` / ``clipped`` 一个不改。

    ``denied`` 尤其：一条"被拒绝过"的任务记忆和一条"做成了"的记忆，回看时
    是两件完全不同的事——而它们的正文可能一模一样。
    """
    trace = build_trace(
        [
            finished("c1", ok=True),
            finished("c2", ok=False, denied=True),
            finished("c3", clipped=True),
        ],
        [],
    )
    assert [(s["ok"], s["denied"], s["clipped"]) for s in trace] == [
        (True, False, False),
        (False, True, False),
        (True, False, True),
    ]


def test_the_step_shape_is_fixed() -> None:
    """每一步的键集合是**定死的**。

    它是 ``meta`` 的一部分，而 ``meta`` 是不透明的 JSON——键名写错一个字母不会有
    任何东西变红，只是那一项永远读不出来。
    """
    (step,) = build_trace([finished()], [appended(("c1", "read_file", {"path": "a"}))])
    assert set(step) == {
        "call_id",
        "name",
        "arguments",
        "ok",
        "denied",
        "clipped",
        "result",
        "result_chars",
    }


# ── 截断 ───────────────────────────────────────────────────────────────────


def test_a_long_result_becomes_an_excerpt_that_admits_it() -> None:
    """``result`` 是摘录，``result_chars`` 是原文长度。

    没有后者的话，"这个工具只返回了 600 字"与"原文有 6 万字、这里只留了 600 字"
    在读数据时是分不开的——而前者会让一次任务看起来无事发生。
    """
    content = "字" * 50_000
    (step,) = build_trace([finished(content=content)], [])
    assert len(step["result"]) <= MAX_STEP_RESULT_CHARS
    assert step["result_chars"] == 50_000
    assert step["result"] != content


def test_a_short_result_is_kept_whole() -> None:
    """短的**一字不砍**。全砍成摘录会让"这次调用到底返回了什么"永远答不全。"""
    (step,) = build_trace([finished(content="就一句话。")], [])
    assert step["result"] == "就一句话。"
    assert step["result_chars"] == 5


def test_a_sixty_four_kib_read_does_not_blow_up_the_trace() -> None:
    """**这条是整组截断存在的理由。** ``agent/fs.py`` 的 ``MAX_READ_BYTES`` 就是
    64 KiB，而 ``meta`` 超了 ``memcore/store.py`` 的 256 KiB 上限时
    ``MemoryStore.add`` 会抛——那条路一路走到对话里的一句「记忆写入失败」。
    """
    huge = "很长的文件内容。" * 8_000  # 64 KiB 那一档
    trace = build_trace([finished("c1", "read_file", content=huge)], [])
    assert size_of(trace) < 4_000, f"64 KiB 的一次读取压出了 {size_of(trace)} 字符的轨迹"


def test_the_budget_shrinks_the_excerpts_instead_of_dropping_steps() -> None:
    """**步数永不丢。**

    超预算时该缩的是每一格的摘录，不是"后面那几步不要了"——把"调了二十次工具"
    记成"调了八次"是静默丢弃，而这个项目唯一不能有的失效就是那个。
    """
    steps = 40
    trace = build_trace(
        [finished(f"c{i}", content="结" * 5_000) for i in range(steps)],
        [],
    )
    assert len(trace) == steps, "超预算时丢了步"
    assert size_of(trace) < MAX_TRACE_CHARS * 2, "预算没兜住"


def test_the_per_step_caps_hold_when_there_are_few_steps() -> None:
    """步数少时，兜底的是**单格上限**而不是均分。

    均分只在步数多到单格上限够不着时才起作用；反过来（只有单格上限、没有均分）
    才是步数一多就爆的那种。两条都要在。
    """
    trace = build_trace(
        [finished("c1", content="结" * 9_000)],
        [appended(("c1", "read_file", {"path": "路" * 9_000}))],
    )
    (step,) = trace
    assert len(step["result"]) <= MAX_STEP_RESULT_CHARS
    assert len(step["arguments"]["path"]) <= MAX_STEP_ARG_CHARS


def test_a_long_tool_name_is_cut() -> None:
    """工具名有上限——它存在的唯一理由是让"步数永不丢"这句话有一个有限的代价。"""
    (step,) = build_trace([finished(name="tool" * 100)], [])
    assert len(step["name"]) == 64


# ── 形状保不住的参数 ───────────────────────────────────────────────────────


def test_flat_arguments_keep_their_keys() -> None:
    """参数的**形状尽量留住**：字典还是字典，键还是原来那些键。"""
    (step,) = build_trace(
        [finished()],
        [appended(("c1", "read_file", {"path": "a.md", "limit": 10}))],
    )
    assert step["arguments"] == {"path": "a.md", "limit": 10}


def test_arguments_that_cannot_be_trimmed_fall_back_to_json() -> None:
    """值都不是字符串、或者键多到截不完时，退回一整段截断的 JSON。

    那一段没有原来的键，但它仍然是**这次调用真正传了什么**。反过来（返回 ``{}``）
    会让"参数丢了"与"模型没给参数"变成同一件事。
    """
    many = {f"键{index}": index for index in range(500)}
    (step,) = build_trace([finished()], [appended(("c1", "t", many))])
    arguments = step["arguments"]
    assert set(arguments) == {"_json"}, "既没留住形状，也没留下原文"
    assert len(arguments["_json"]) <= MAX_STEP_ARG_CHARS


def test_arguments_are_json_serializable() -> None:
    """``meta`` 只能装 JSON 认得的类型（``memcore/store.py`` 会在那里抛）。

    轨迹里混进一个 ``Path`` 或 ``set`` 的症状不是这里报错，是**整条记忆写不进去**。
    """
    trace = build_trace(
        [finished()],
        [appended(("c1", "read_file", {"path": "a.md", "limit": 10}))],
    )
    assert json.dumps(trace, ensure_ascii=False)
