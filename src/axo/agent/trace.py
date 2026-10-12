"""一次任务**做过什么** —— 压成能装进 ``meta`` 的形状。

:class:`~axo.memory.policy.TaskPolicy` 的正文是**回复**（用户读到的那
份），于是轨迹是唯一说明"这个回答是怎么来的"的东西：查了哪个文件、搜了什么词、
哪一步被拒绝了。它进 :attr:`~axo.memory.policy.MemoryDraft.meta`，
跟着那条记忆走。

## 为什么是一个纯函数，不是循环里攒的一个列表

累加点是 :class:`~axo.ui.agent_worker.AgentWorker`——**它是唯一活得过
``stop()`` 的东西**：循环被 break 之后 ``LoopDone`` 永不产出，而"被中断的那一轮"
恰恰最该留下它做过的那些事。但累加只是"把两类事件各收一份"，判断全在这里，于是
它能在**核心腿里用真的 dataclass 逐条钉**，一行 Qt 都不用碰。

## 为什么必须截断，而且截得这么狠

``meta`` 是**逐字**进 SQLite 的，而 ``memcore/store.py`` 的 ``_MAX_META_BYTES``
是 256 KiB，**超了 ``MemoryStore.add`` 抛 ``StoreError``，而
:class:`~axo.memory.writer.MemoryWriter` 有意不吞**（那条路径的纪律是
"丢了却不知道比丢了更糟"）。于是不截断的后果不是"多占点地方"，是**读一次 64 KiB
的文件就把整条记忆变成对话里的一句「记忆写入失败」**——
:data:`~axo.agent.fs.MAX_READ_BYTES` 正是 64 KiB，这不是假想的输入。

**步数永不丢，长度按步数均分。** 每个执行过的调用都占一格：``name`` 有上限、标志
是 bool，所以即使一次任务调了两百次工具，那一部分也只有几十 KB。被均分掉的是
``arguments`` 与 ``result`` 的**摘录**——它们短了只是不够详细，
:data:`_RESULT_CHARS` 那个键说明了原文有多长。反过来（超了就丢步）是**静默丢弃**，
而"已经执行过的调用一定留有记录"是这个项目唯一不能破的约定
（见 ``ui/agent_worker.py`` 模块文档）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .loop import MessagesAppended, ToolFinished

__all__ = [
    "MAX_STEP_ARG_CHARS",
    "MAX_STEP_NAME_CHARS",
    "MAX_STEP_RESULT_CHARS",
    "MAX_TRACE_CHARS",
    "build_trace",
]

MAX_TRACE_CHARS = 8000
"""整条轨迹的字符预算。

按 UTF-8 最坏情况（汉字 3 字节）折算约 24 KB，连同下面那两个单格上限，最坏的一条
``meta`` 离 256 KiB 的上限还有一个数量级——**这个余量是给 ``full_text`` 和将来往
``meta`` 里加东西留的**，不是拍的。
"""

MAX_STEP_NAME_CHARS = 64
"""工具名的上限。库里最长的名字是 ``set_reminder`` 这一档，64 是量级上的余量，
不是量出来的——它唯一的职责是让"步数永不丢"这句话有一个有限的代价。"""

MAX_STEP_ARG_CHARS = 300
"""单格参数的摘录上限。``read_file`` 的路径、``search_memory`` 的查询词都在这个
长度以内；``write_memory`` 的正文会被砍——砍掉的部分在逐字稿里。"""

MAX_STEP_RESULT_CHARS = 600
"""单格结果的摘录上限。"""

_ARG_SHARE = 3
"""单格预算里参数占几分之一。

**参数比结果值得留。** 结果通常是"这个文件里有这些字"，而参数是"用户让你做什么"
——后者才是那条记忆的来历。所以宁可把结果砍短一点。
"""


def build_trace(
    finished: Sequence[ToolFinished],
    appended: Sequence[MessagesAppended],
) -> tuple[dict[str, Any], ...]:
    """把两类事件拼成轨迹。**纯的**：不读时钟、不碰盘、不 import Qt。

    Args:
        finished: 按执行顺序收集的 :class:`~axo.agent.loop.ToolFinished`。
        appended: 同样按顺序收集的 :class:`~axo.agent.loop.MessagesAppended`。

    Returns:
        每一步一个字典，键是 ``call_id`` / ``name`` / ``arguments`` / ``ok`` /
        ``denied`` / ``clipped`` / ``result`` / ``result_chars``。**没有执行过的
        调用不会出现在这里**——那是事实，不是遗漏（收尾轮里模型还要调工具时，
        那几个名字走 :attr:`~axo.agent.loop.LoopDone.unexecuted`，
        由界面显示）。

    **参数要另找一路。** :class:`ToolFinished` 里没有它（``loop.py`` 只吐
    id/name/ok/content），参数只在 :class:`MessagesAppended` 那一行 assistant 的
    ``tool_calls`` 里。按 ``call_id`` 反查；**查不到就留一个空字典，但那一步照样
    留着**——为了参数缺失而删掉整步，等于把"这个工具跑过"这件事也一起删了。
    """
    if not finished:
        return ()

    arguments = _arguments_by_call(appended)
    per_step = max(0, MAX_TRACE_CHARS // len(finished))
    arg_budget = min(MAX_STEP_ARG_CHARS, per_step // _ARG_SHARE)
    result_budget = min(MAX_STEP_RESULT_CHARS, max(0, per_step - arg_budget))

    return tuple(
        _step(one, arguments.get(one.call_id, {}), arg_budget, result_budget) for one in finished
    )


def _arguments_by_call(appended: Sequence[MessagesAppended]) -> dict[str, Mapping[str, Any]]:
    """``call_id`` → 模型当初给的那份参数。

    **遍历每一行、不假定它在第 0 位**：配对事件按 "assistant 行 + 若干 tool 行"
    成对给，而带 ``tool_calls`` 的只有 assistant 那一行。写死下标会让形状一变就
    静默取空——而静默取空的轨迹看起来和"模型没给参数"一模一样。
    """
    found: dict[str, Mapping[str, Any]] = {}
    for event in appended:
        for row in event.rows:
            for call in row.tool_calls:
                found[call.id] = call.arguments
    return found


def _step(
    finished: ToolFinished,
    arguments: Mapping[str, Any],
    arg_budget: int,
    result_budget: int,
) -> dict[str, Any]:
    content = finished.content
    return {
        "call_id": finished.call_id,
        "name": finished.name[:MAX_STEP_NAME_CHARS],
        "arguments": _bound_arguments(arguments, arg_budget),
        "ok": finished.ok,
        "denied": finished.denied,
        "clipped": finished.clipped,
        "result": content[:result_budget] if result_budget > 0 else "",
        "result_chars": len(content),
    }


def _bound_arguments(arguments: Mapping[str, Any], budget: int) -> dict[str, Any]:
    """参数的摘录。**形状尽量留住——字典还是字典。**

    先按值截断（工具造出来的参数字典是平的：``path`` / ``query`` / ``text``），
    再**量一遍**序列化后的长度；超了就按超出量再缩一轮。缩不动（值都不是字符串，
    比如一张 500 个键的整数表）时才退回一整段截断的 JSON——那一段没有原来的键，
    但它仍然是**这次调用真正传了什么**，比丢掉它好。

    **为什么要量第二遍而不是一次算准**：键名、引号、花括号也占地方，而它们随参数
    的形状变。算准等于在这里重写一遍 JSON 的长度公式，而那个公式一旦和
    :func:`_dumps` 对不上，症状是"形状莫名其妙地退化了"——量一遍不会有这个问题。

    与 :attr:`~axo.agent.loop.ToolFinished.clipped` 的区别：那个说的是
    *回灌给模型的那份结果*被截了，这里说的是**存进记忆的摘录**被截了。两者互不
    蕴含：一条 600 字的工具结果会被这里截、却不会被循环截。
    """
    if budget <= 0 or not arguments:
        return {}
    room = budget
    for _ in range(2):
        trimmed: dict[str, Any] = {
            key: (value[:room] if isinstance(value, str) else value)
            for key, value in arguments.items()
        }
        encoded = _dumps(trimmed)
        if len(encoded) <= budget:
            return trimmed
        if not any(isinstance(value, str) for value in arguments.values()):
            break
        room = max(0, room - (len(encoded) - budget))
    return {"_json": _dumps(arguments)[:budget]}


def _dumps(value: Any) -> str:
    """紧凑序列化，**只为量长度**。``ensure_ascii=False`` 才是真实的字节量——
    默认的 ``\\uXXXX`` 转义会把汉字算成 6 个字符，于是预算凭空宽了一倍。"""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        # 参数里混进了 JSON 不认的东西（``memcore/store.py`` 那边的报错就是这个
        # 形状）。**不抛**：轨迹是附加信息，为它炸掉整条记忆是本末倒置。
        # ``ToolCall.arguments`` 来自后端的 JSON 解析，正常路径上到不了这里。
        return str(value)
