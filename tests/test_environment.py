"""L1 环境槽的正文。

这一层只有两三行字，但它挡掉的是整类**错得理直气壮**的回答：4B 模型的训练数据
截止在某一天，它会拿那一天当"今天"。所以测两件事——时间行写得对（包括星期几，
它是这一层唯一一处需要查表的计算），以及模型标识在不在里面。

时钟是注入的。真实时钟会让"同样的输入给出同样的输出"这条断言写不出来，而这一行
每轮都进上下文。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from axo.session.environment import build_environment


def test_time_line_reads_as_chinese_and_states_the_weekday() -> None:
    text = build_environment(now=datetime(2026, 10, 10, 15, 4))
    assert text == "当前时间：2026-10-10 星期六 15:04。"


@pytest.mark.parametrize(
    ("moment", "weekday"),
    [
        (datetime(2026, 10, 12, 9, 0), "星期一"),
        (datetime(2026, 10, 10, 15, 4), "星期六"),
        (datetime(2026, 10, 11, 23, 59), "星期日"),
        (datetime(2024, 2, 29, 0, 0), "星期四"),
    ],
)
def test_the_weekday_table_lines_up_with_the_calendar(moment: datetime, weekday: str) -> None:
    """星期几是这一层唯一一处查表计算，而查表算错不会有任何症状。

    闰日那条不是凑数：``strftime('%A')`` 在 Windows 上跟着系统区域设置走，同一个
    日期在不同机器上会给出不同的语言。这张固定的中文表刻意绕开了它——这几条就是
    在钉"绕开了"这件事。
    """
    assert weekday in build_environment(now=moment)


def test_the_model_id_is_how_it_knows_who_is_reading() -> None:
    """同一个记忆库会被不同的模型读到，上下文里要说得出**此刻是谁**。"""
    text = build_environment(model_id="ollama/qwen3-4b", now=datetime(2026, 10, 10, 15, 4))
    lines = text.splitlines()
    assert len(lines) == 2
    assert "ollama/qwen3-4b" in lines[1]
    assert "记忆不属于你" in lines[1]


def test_without_a_model_id_only_the_time_is_said() -> None:
    """没给模型标识时**不编**一句。

    测试与命令行会走这条路。补一句"你由未知模型驱动"看起来无害，但它会让模型
    把注意力花在一个没有信息量的句子上。
    """
    text = build_environment(model_id="", now=datetime(2026, 10, 10, 15, 4))
    assert text.splitlines() == ["当前时间：2026-10-10 星期六 15:04。"]


def test_arguments_are_keyword_only() -> None:
    """四个参数都是关键字参数，因为其中两个是字符串——位置调用会把它们悄悄调换。"""
    with pytest.raises(TypeError):
        build_environment("ollama/qwen3-4b")  # type: ignore[misc]


# ── 这一轮的能力：工具与歧义 ────────────────────────────────────────────────


def test_no_tools_is_said_out_loud() -> None:
    """**这一轮没有工具，就要说出来。**

    2026-10-10 的实测：一轮闲聊语域里用户说"记住：我最喜欢靛蓝"，模型回"已通过
    ``write_memory`` 工具保存"——而那一轮请求里根本没有 ``tools`` 字段。模型读得到
    自己收到了什么，读不到自己**没**收到什么，所以这件事只能由上下文告诉它。

    钉的是"说了什么能力"，不是"说了什么指令"：这层只写事实（见模块文档）。
    """
    text = build_environment(now=datetime(2026, 10, 10, 15, 4), tools_declared=False)
    assert "没有工具可用" in text
    assert "不会有任何动作被执行" in text


def test_no_tools_line_carries_no_recitable_inventory() -> None:
    """**"做不到"的清单不许回来。**

    问题不在"说了事实"，在**那个清单可以被原样背回去**：2026-10-11 实测，用户打
    「晚上好」，模型据它回了一句"我无法使用工具查看记忆"，用户于是以为工具坏了——
    而那一轮本来是闲聊，走的是路由那条路（见 ``test_router.py`` 里"去看已经存下来的
    东西"那一族）。

    所以这条钉的是**形状**而不是措辞：一句描述"这一轮发生了什么"的事件，和一句
    "我做不到 A、B、C"的能力清单，说的是同一件事实，但只有后者能被反手当成免责声明。
    加回任何一个能力名词，这条就红。
    """
    text = build_environment(now=datetime(2026, 10, 10, 15, 4), tools_declared=False)
    for capability in ("保存", "删除", "提醒", "读文件", "做不到"):
        assert capability not in text, f"「{capability}」是可被背回去的能力名词，不该回来"


def test_tools_declared_says_nothing() -> None:
    """有声明的那些轮**一个字都不加**。

    工具是靠请求的 ``tools`` 字段声明的，那是契约所在；在环境里再列一遍名字，等于
    让模型有两处关于"我能调什么"的说法，而它们迟早会不一致。
    """
    assert build_environment(now=datetime(2026, 10, 10, 15, 4), tools_declared=True) == (
        build_environment(now=datetime(2026, 10, 10, 15, 4))
    )


def test_ambiguity_is_stated_as_a_fact() -> None:
    """歧义那一路：**只陈述"没判准"**，请求确认的措辞留给模型（见
    ``RoutedTurn.wants_clarification`` 的说明——路由不产文案）。
    """
    text = build_environment(now=datetime(2026, 10, 10, 15, 4), needs_clarification=True)
    assert "没判准" in text


def test_the_two_turn_slots_are_independent() -> None:
    """两个槽各说各的，不互相顶替——歧义那一路**两个都为真**（歧义必然不给工具）。"""
    both = build_environment(
        now=datetime(2026, 10, 10, 15, 4), tools_declared=False, needs_clarification=True
    )
    assert "没判准" in both
    assert "没有工具可用" in both
    only_clarify = build_environment(now=datetime(2026, 10, 10, 15, 4), needs_clarification=True)
    assert "没有工具可用" not in only_clarify


def test_the_speech_receipt_is_a_fresh_line_each_turn() -> None:
    """播出回执进 L1 环境槽，**给的是那一行字本身**（格式由 `SpeechReceipt.describe` 管）；
    空串时什么都不会加——那一轮本来就没出声这件事，不该在上下文里冒一行空话。"""
    with_receipt = build_environment(speech_receipt="播出回执（23:41）：已播出「记得买牛奶。」")
    without = build_environment()

    assert "播出回执（23:41）" in with_receipt
    assert "播出回执" not in without
