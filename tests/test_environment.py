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
    # **钉第一行**：这一条测的是时间行的措辞；环境槽其余的行各有自己的测试
    # （出不出声那一行、回执行……），整串全等会把它们的变化混进这条的失败里。
    assert text.splitlines()[0] == "当前时间：2026-10-10 星期六 15:04。"


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
    assert len(lines) == 3
    assert "ollama/qwen3-4b" in lines[1]
    assert "记忆不属于你" in lines[1]
    assert "只出文字" in lines[2], "出不出声那行也该在（默认不出声）"


def test_without_a_model_id_only_the_time_is_said() -> None:
    """没给模型标识时**不编**一句。

    测试与命令行会走这条路。补一句"你由未知模型驱动"看起来无害，但它会让模型
    把注意力花在一个没有信息量的句子上。
    """
    text = build_environment(model_id="", now=datetime(2026, 10, 10, 15, 4))
    assert text.splitlines() == [
        "当前时间：2026-10-10 星期六 15:04。",
        "注意：你这一轮的话**只出文字，不出声**。",
    ]


def test_arguments_are_keyword_only() -> None:
    """**全部都是关键字参数**——位置调用会把几个字符串悄悄调换，而那不会报错。"""
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


def test_the_clarify_line_says_ask_back_not_cannot() -> None:
    """歧义那一轮，环境要说的是**反问**而不是「做不到」。

    2026-10-14 真机事故：用户说「两分钟后测试语音提醒」（判歧义，这一轮没工具），
    模型答「我没有定时能力」——一句**伤信任的假结论**（App 明明有 set_reminder）。
    设计上的收场是反问确认；这条钉住那句指令真在环境里。
    """
    text = build_environment(needs_clarification=True)

    assert "反问" in text
    assert "别直接说做不到" in text
    assert "没有工具只是暂时的" in text


def test_she_knows_whether_this_turn_will_be_spoken() -> None:
    """**这一轮的话出不出声，她要提前知道**——否则她会在打字那轮问「听到了吗」
    （2026-10-14 实测就是这么发生的：那一轮压根不出声）。"""
    speaking = build_environment(will_speak=True)
    typing = build_environment(will_speak=False)

    assert "会被念出声" in speaking
    assert "只出文字" in typing
    assert "念出声" not in typing


# ── 模态真相：转写不是原声 ─────────────────────────────────────────────────


def test_a_spoken_turn_says_the_model_only_got_text() -> None:
    """**模型会以为自己听到了原声**，进而声称听出了语气——它手里只有转写。"""
    text = build_environment(spoken_input=True, now=datetime(2026, 10, 10, 15, 4))

    assert "语音转写" in text
    assert "原声没有发给你" in text


def test_forwarded_audio_is_stated_as_such() -> None:
    text = build_environment(
        spoken_input=True, audio_forwarded=True, now=datetime(2026, 10, 10, 15, 4)
    )

    assert "附了原声" in text
    assert "原声没有发给你" not in text


def test_call_mode_is_a_fact_in_the_environment() -> None:
    text = build_environment(call_mode=True, now=datetime(2026, 10, 10, 15, 4))

    assert "免按键通话" in text
    assert "打断" in text


def test_alerts_that_arrived_during_a_call_are_reported_next_turn() -> None:
    text = build_environment(pending_alerts=("该睡了", "交水费"), now=datetime(2026, 10, 10, 15, 4))

    assert "该睡了" in text and "交水费" in text

    quiet = build_environment(now=datetime(2026, 10, 10, 15, 4))
    assert "该睡了" not in quiet


def test_workspace_context_is_in_environment_only_when_given() -> None:
    """L1 只带事实；没给工作区时逐字节不变。"""
    base = build_environment(now=datetime(2026, 10, 10, 15, 4))
    with_ws = build_environment(
        now=datetime(2026, 10, 10, 15, 4), workspace="工作区：D:/repo\ngit：main，干净"
    )
    assert "工作区：" not in base
    assert "工作区：D:/repo" in with_ws and "git：main，干净" in with_ws
    assert with_ws.startswith(base.splitlines()[0])
