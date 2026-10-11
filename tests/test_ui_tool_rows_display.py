"""回看时，工具行怎么变成对话流里的一个气泡。

守的是一件真发生过的事：**被闸门拒绝的那一行，正文是写给模型的当轮指令**
（"这一次调用跳过了……不要重试同一次调用……做完你还能做的部分"），而它跟别的工具行
一样落了盘，在打开一个旧会话时被 :func:`display_rows` 原样渲染出来——于是用户在自己的
对话记录里读到一段**对他下达的命令**，标题还写着「工具结果」，可那一步根本没有执行过。

**那段文案从真实派发口取，不在这个文件里手抄一份**（:func:`_denial`）。抄一份的话，
改了 ``agent/tools.py`` 的措辞、这个文件照样绿——而它要守的正是"**那一段话**不许出现在
用户的对话流里"，句子换了，要守的东西就换了。顺带把两条腿接上：核心腿
``test_gate_denial_skips_the_tool`` 钉的是"回灌里**必须有**『不要重试』"，这里钉的是
"回看里**不许有**"。同一个标记，管子的两头。

**为什么不建 ``MainWindow``。** 与 ``test_ui_tools_wire`` 同一条理由：离屏下
QtWebEngine 一渲染就把整个进程硬杀掉。``display_rows`` 是个模块级纯函数，直接调就行
——这个文件不碰任何窗口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from biosnn_memterm.agent.tools import ToolContext, ToolOutcome, ToolRegistry
from biosnn_memterm.llm.base import ToolCall, ToolSpec
from biosnn_memterm.session.store import Message
from biosnn_memterm.ui.main_window import display_rows
from memcore.store import MemoryStore

pytestmark = pytest.mark.gui


class _Gated:
    """一个过闸门的工具替身。**``run`` 一旦被调用就炸**——

    这条测试的全部前提是"那一步没有执行"，所以"执行了"必须在现场就响，而不是靠
    后面某条断言去推。
    """

    def __init__(self, name: str = "write_memory") -> None:
        self.spec = ToolSpec(name=name, description="测试用替身")
        self.requires_approval = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raise AssertionError("被拒绝的工具还是被执行了")


def _denial(tmp_path: Path, name: str = "write_memory") -> str:
    """走一遍真实派发口，取回**真的会落进逐字稿**的那段拒绝文案。"""
    with MemoryStore(tmp_path / "mem.db") as store:
        outcome = ToolRegistry([_Gated(name)]).dispatch(
            ToolCall(id="call_1", name=name, arguments={}),
            ToolContext(store=store, session_id="S1", turn_seq=1),
            approved=False,
        )
    assert outcome.ok is False
    assert outcome.replayable is False, "拒绝的结果必须是不可回放的，否则 L5 会逐轮搬它"
    return outcome.content


def _row(message: Message) -> dict[str, Any]:
    """一行逐字稿 → 唯一那个气泡。多于一行说明夹具写错了，当场炸比默默取第一个好。"""
    rows = display_rows([message])
    assert len(rows) == 1, f"夹具应当只产出一个气泡，实际 {len(rows)} 个"
    return rows[0]


def _denied_row(tmp_path: Path, name: str = "write_memory") -> tuple[Message, str]:
    content = _denial(tmp_path, name)
    return (
        Message(
            role="tool",
            content=content,
            tool_call_id="call_1",
            tool_name=name,
            no_replay="denied",
        ),
        content,
    )


def test_a_denied_row_is_not_rendered_as_a_model_instruction(tmp_path: Path) -> None:
    """拒绝行换成给人看的说法，而那段给模型的话**一个字都不许漏出来**。

    断言拿的是整段原文（而不是"气泡里没有『不要重试』四个字"）：只盯一个短语的话，
    措辞改一次这个守卫就失效了，而它看起来还是绿的。
    """
    message, content = _denied_row(tmp_path)

    row = _row(message)

    assert row["role"] == "tool"
    assert row["meta"]["title"] == "已跳过 · write_memory"
    assert content not in row["text"]
    # 与核心腿共用同一个标记：那边钉"回灌必须有它"，这边钉"回看不许有它"
    assert "不要重试" not in row["text"]
    # 标题也不该说"结果"：什么都没跑，没有结果
    assert "结果" not in row["meta"]["title"]


def test_the_title_says_skipped_rather_than_denied(tmp_path: Path) -> None:
    """``denied`` 是一个合并了三种收场的值，所以标题不能替用户认下"我拒绝了"。

    用户点了拒绝、超时、或者卡片挂着时按了停止——``gate.py`` 把三者都归成
    ``ApprovalDecision.DENY``，逐字稿里都是 ``"denied"``。写"已拒绝"对后两种是
    **替用户认一件他没做的事**。
    """
    message, _ = _denied_row(tmp_path)

    title = _row(message)["meta"]["title"]

    assert "拒绝" not in title
    assert "跳过" in title


def test_a_normal_tool_row_still_shows_its_result() -> None:
    """反空转：普通的工具行照旧原样显示。

    没有这一条的话，"把所有工具行都换成跳过文案"也能让上面两条通过——而那会把
    用户真正查到的内容一并抹掉。
    """
    message = Message(
        role="tool",
        content="钥匙在鞋柜里。",
        tool_call_id="call_2",
        tool_name="search_memory",
    )

    row = _row(message)

    assert row["meta"]["title"] == "工具结果 · search_memory"
    assert row["text"] == "钥匙在鞋柜里。"


def test_a_stale_result_is_still_a_result() -> None:
    """反空转的另一半：``stale`` 是真跑过的结果，它是**快照**不是**没执行**。

    这一条挡住的是"把所有带 ``no_replay`` 的行都当成拒绝"——那只差一个
    ``== "denied"`` 写成 ``is not None``，而 ``get_time`` / ``read_file`` 的内容
    正是用户回看时最想看的。
    """
    message = Message(
        role="tool",
        content="2026-10-10 15:20:00（本地时间）",
        tool_call_id="call_3",
        tool_name="get_time",
        no_replay="stale",
    )

    row = _row(message)

    assert row["meta"]["title"] == "工具结果 · get_time"
    assert row["text"] == "2026-10-10 15:20:00（本地时间）"


def test_a_tool_row_without_a_name_still_renders() -> None:
    """``tool_name`` 缺失时不能炸。schema-2 之前的老行就是这样读回来的。"""
    row = _row(Message(role="tool", content="老的结果", tool_call_id="call_9"))

    assert row["meta"]["title"] == "工具结果 · 未知工具"
    assert row["text"] == "老的结果"


def test_an_assistant_row_that_only_called_tools_is_still_dropped() -> None:
    """既有行为不许被这次改动带坏：只有 ``tool_calls``、没有正文的助手行仍是空气泡。

    它跟"拒绝行换说法"是同一个函数里的两个分支，改一个最容易顺手弄坏另一个。
    """
    assert (
        display_rows(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=({"id": "c1", "name": "get_time", "arguments": {}},),
                )
            ]
        )
        == []
    )
