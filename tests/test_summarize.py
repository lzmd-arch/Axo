"""被挤出去的历史怎么压成摘要。

这个模块的产出会被**当成事实**注入上下文，接着被写进记忆库，所以它的每一条性质
都得是能断言的：放得下（额度是真约束）、够诚实（漏掉了几条必须写出来）、可复现
（同样的输入永远同样的输出）。三条缺一条，摘要就从"保险"变成"看起来完全合理
的错误"——那是最难发现的一类错，因为它在逐字稿里没有对应物可供核对。
"""

from __future__ import annotations

import pytest

from biosnn_memterm.session.budget import TokenMeter
from biosnn_memterm.session.store import Message
from biosnn_memterm.session.summarize import MAX_LINE_CHARS, first_sentence, summarize

METER = TokenMeter()


def msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def history() -> list[Message]:
    """一段像真会话的历史：前提 + 若干轮 + 若干轮。"""
    pairs = [
        ("我家的路由器放在书房书架顶层。", "好的，记下了。"),
        ("每周三晚上八点要开组会，会议纪要在共享盘上。", "明白。"),
        ("记忆终端用 SQLite 存记忆库，WAL 模式。", "这个取舍的理由是什么？"),
        ("因为写入是逐条的，读多写少。", "收到。"),
        ("明天记得提醒我买牛奶。", "好的。"),
        ("另外把 num_ctx 改成 16384。", "已经改好了。"),
    ]
    messages = [msg("user", "我在做 BioSNN 记忆终端，记忆要独立于所调用的模型。")]
    for user_text, assistant_text in pairs:
        messages.append(msg("user", user_text))
        messages.append(msg("assistant", assistant_text))
    return messages


# ── 分句 ────────────────────────────────────────────────────────────────────


def test_the_first_sentence_stops_at_chinese_punctuation() -> None:
    assert first_sentence("路由器在书房。另外还有别的。") == "路由器在书房。"
    assert first_sentence("这样对吗？我觉得不对！") == "这样对吗？"


def test_the_first_sentence_does_not_split_on_dots() -> None:
    """**这条是分句规则里唯一容易写错的地方。**

    ``0.40.2``、``recall.py``、``README.md``、``127.0.0.1`` 里全是句点。按句点
    切会把版本号和路径切成碎片——而碎片拼起来的"摘要"读起来像一句完整的话，
    内容却已经错了。它不会报错，只会让模型记住一个不存在的版本号。
    """
    assert first_sentence("版本 0.40.2 的 Ollama 有问题。") == "版本 0.40.2 的 Ollama 有问题。"
    assert first_sentence("改一下 src/memcore/recall.py。") == "改一下 src/memcore/recall.py。"
    # 整句没有中文句读时全部保留，不是切到第一个句点为止
    assert first_sentence("README.md 里写了 0.40.2") == "README.md 里写了 0.40.2"


def test_the_first_sentence_collapses_whitespace_and_marks_truncation() -> None:
    assert first_sentence("第一行\n第二行") == "第一行 第二行"
    long_text = "很长的文本，" * 100
    cut = first_sentence(long_text, limit=20)
    assert len(cut) == 21, "截断后是 limit 个字符加一个省略号"
    assert cut.endswith("…")


def test_an_empty_message_has_no_line() -> None:
    assert first_sentence("") == ""
    assert first_sentence("   \n  ") == ""


# ── 摘要 ────────────────────────────────────────────────────────────────────


def test_the_summary_always_fits_the_budget() -> None:
    """**核心断言。** 额度是真约束，不是建议。

    装配层已经按额度分好了每一层的钱；某一层偷偷超一点，代价由**别层**承担，
    而承担的方式是服务端静默截断最早的消息——那是人格锚的位置。
    """
    for budget in (0, 1, 5, 20, 40, 60, 80, 120, 200, 400, 2000):
        result = summarize(history(), budget_tokens=budget)
        assert METER.text(result) <= budget, f"额度 {budget} 下摘要超了：{result!r}"


def test_the_premise_survives_a_tight_budget() -> None:
    """第一条被挤出上下文，后面全都在答一个没人问过的问题。"""
    premise = "我在做 BioSNN 记忆终端，记忆要独立于所调用的模型。"
    result = summarize(history(), budget_tokens=60)
    assert result, "60 token 不至于连表头加一句都放不下"
    assert "BioSNN 记忆终端" in result
    assert result.splitlines()[1].startswith("用户：")
    assert premise[:10] in result


def test_recent_turns_are_preferred_over_older_ones() -> None:
    """额度不够时从最近往前收。"""
    result = summarize(history(), budget_tokens=90)
    assert "num_ctx" in result, "最近那一轮该在"
    assert "会议纪要" not in result, "最老的那一轮该被让出去"


def test_what_was_left_out_is_stated() -> None:
    """**漏掉这件事必须写出来。**

    不写的话摘要读起来是完整的，模型会把"没提到"当成"没发生过"——而它接着会
    据此回答，还答得很自信。
    """
    result = summarize(history(), budget_tokens=90)
    assert "未列出" in result
    # 条数是像样的：写在页脚里的数不该是 0，也不该超过总条数
    tail = result.splitlines()[-1]
    assert tail.startswith("…另有 ")
    count = int(tail.removeprefix("…另有 ").removesuffix(" 条未列出"))
    assert 0 < count < len(history())


def test_a_generous_budget_lists_everything_and_claims_no_loss() -> None:
    result = summarize(history(), budget_tokens=4000)
    assert "未列出" not in result, "全都放下了就不该有遗漏声明"
    for message in history():
        assert first_sentence(message.content) in result


def test_empty_messages_do_not_become_lines() -> None:
    """被中断的空回复是合法状态，但它不该在摘要里占一行"助手："。"""
    messages = [msg("user", "第一句。"), msg("assistant", ""), msg("user", "第二句。")]
    result = summarize(messages, budget_tokens=200)
    assert result.count("助手：") == 0
    assert "（2 条，已压缩）" in result, "空消息不该被算进条数（三条消息，两条有内容）"


def test_nothing_fits_returns_empty_not_a_butchered_header() -> None:
    """一段被砍掉头的摘要比没有摘要更糟：它会把自己读成"历史就这么长"。"""
    assert summarize(history(), budget_tokens=3) == ""
    assert summarize(history(), budget_tokens=0) == ""
    assert summarize(history(), budget_tokens=-10) == ""
    assert summarize([], budget_tokens=500) == ""


def test_the_same_input_always_gives_the_same_output() -> None:
    """确定性。有这一步，账本里"这一轮为什么注入这段文字"才是可复现的。"""
    messages = history()
    first = summarize(messages, budget_tokens=120)
    for _ in range(5):
        assert summarize(messages, budget_tokens=120) == first


def test_it_does_not_depend_on_message_timestamps() -> None:
    """换掉所有时间戳（等于"同一段对话换了一天再算"），摘要逐字不变。"""
    messages = history()
    before = summarize(messages, budget_tokens=120)
    for index, message in enumerate(messages):
        message.at = 1_700_000_000.0 + index * 3600
    assert summarize(messages, budget_tokens=120) == before


def test_the_input_is_not_mutated() -> None:
    """装配器会在驱逐之前调它，之后还要拿同一批消息做别的判断。"""
    messages = history()
    snapshot = [(message.role, message.content) for message in messages]
    summarize(messages, budget_tokens=120)
    assert [(message.role, message.content) for message in messages] == snapshot


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_every_speaker_has_a_label(role: str) -> None:
    """工具行在 P1 不出现，但标签表里先留着——P3 的轨迹行会走同一条路。"""
    result = summarize([msg(role, "内容。")], budget_tokens=200)
    assert "：" in result


def test_long_messages_do_not_eat_the_whole_budget() -> None:
    """一条很长的回复只按 :data:`MAX_LINE_CHARS` 计价。

    摘要按"覆盖了多少轮"计价，不是按"某一段写得多细"。没有这个上限，逐字稿里
    第一条超长回复就会把摘要额度吃光，后面全都进了"未列出"。
    """
    messages = [
        msg("user", "很短的前提。"),
        msg("assistant", "长" * 5000),
        msg("user", "最后一句。"),
    ]
    result = summarize(messages, budget_tokens=150)
    assert "最后一句" in result
    assert "长" * (MAX_LINE_CHARS + 1) not in result, "超过上限的部分不该出现"
