"""SIR 与表达层：解析、降级、事实核对、退回草稿。

**这一层测的是"不许坏的那几件事"**，不是"文笔好不好"：写得像不像千岁交给人工与
风格指标（见 ``docs/expression.md`` 的验收），这里只钉住确定性行为——解析能容忍模型
漏写标签、降级能覆盖没有 SIR 的后端、事实丢了必须整段退回、任何失败都不许把这一轮
变成空回复。
"""

from __future__ import annotations

import pytest
from fakes import ScriptedBackend, text_chunk

from axo.expression.layer import ExpressionLayer
from axo.expression.sir import (
    SIR,
    derive,
    extract_facts,
    facts_preserved,
    parse,
    render_prompt,
    sir_instruction,
    style_signature,
)

PERSONA = "你是千岁（阿螈）。说话简短、直接，不用感叹号。"


# ── 解析：完整 SIR ──────────────────────────────────────────────────────────


def test_a_complete_sir_block_is_read_verbatim() -> None:
    text = """<think>
意图: answer
要点:
- BioSNN-Plug 在推进 Paper C 的基线实验
- 问用户要不要展开
语气指示: casual
长度: short
必须保留的事实:
- Paper C
- 0.898
</think>
<content>BioSNN-Plug 最近在推进 Paper C 的基线实验，需要我详细说说吗？</content>"""
    sir = parse(text)
    assert sir.intent == "answer"
    assert sir.key_points == ("BioSNN-Plug 在推进 Paper C 的基线实验", "问用户要不要展开")
    assert sir.tone == "casual"
    assert sir.length == "short"
    assert "Paper C" in sir.facts and "0.898" in sir.facts
    assert sir.draft.startswith("BioSNN-Plug 最近")


def test_chinese_labels_are_accepted_too() -> None:
    text = "<think>\n意图：拒绝\n语气：直接\n长度：短\n</think>\n不知道。"
    sir = parse(text)
    assert (sir.intent, sir.tone, sir.length) == ("refuse", "direct", "short")
    assert sir.draft == "不知道。"


def test_a_missing_closing_tag_does_not_swallow_the_whole_answer() -> None:
    """模型漏写 ``</think>`` 是常态。**丢掉整段回复**是比"多解析一行"贵得多的事故。"""
    text = "<think>\n意图: answer\n要点:\n- 第一条\n语气指示: casual\n长度: short\n<content>正文在这。</content>"
    sir = parse(text)
    assert sir.intent == "answer"
    assert sir.key_points == ("第一条",)
    assert sir.draft == "正文在这。"


def test_missing_fields_fall_back_to_the_draft_instead_of_being_empty() -> None:
    text = "<think>\n意图: answer\n</think>\n这句话没有要点，也没有长度。"
    sir = parse(text)
    assert sir.key_points, "要点缺了要从草稿里补，不能是空元组"
    assert sir.length in ("short", "medium", "long")


# ── 解析：没有 SIR 的普通文本 ───────────────────────────────────────────────


def test_plain_text_is_derived_and_keeps_every_sentence() -> None:
    """降级是生产路径的常态，不是异常路径。"""
    sir = derive("先看日志。再看配置。最后重启。")
    assert sir.key_points == ("先看日志", "再看配置", "最后重启")
    assert sir.intent == "answer"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("不知道。", "refuse"),
        ("我上周说的那个项目怎么样了？", "ask"),
        ("哈哈，真的吗", "emotion"),
        ("已经记下了。", "answer"),
    ],
)
def test_intent_is_guessed_from_shape_not_from_a_model(text: str, intent: str) -> None:
    assert derive(text).intent == intent


def test_the_draft_is_kept_byte_for_byte() -> None:
    """退回草稿时显示的是原字：全角标点、句末问号都不许被"顺手整理"。"""
    text = "我上周说：１.２倍，然后呢？"
    assert derive(text).draft == text


def test_length_buckets_are_by_character_count() -> None:
    assert derive("嗯。").length == "short"
    assert derive("这是一段中等长度的回答。" * 8).length == "medium"
    assert derive("长" * 200).length == "long"


# ── 事实：抽什么、怎么核 ────────────────────────────────────────────────────


def test_facts_cover_numbers_urls_paths_and_quotes() -> None:
    text = "分数从 0.699 涨到 0.898，见 https://example.com/a，文件在 D:\\tmp\\x.log，项目叫「BioSNN」。"
    facts = extract_facts(text)
    for want in ("0.699", "0.898", "https://example.com/a", "D:\\tmp\\x.log", "BioSNN"):
        assert want in facts, (want, facts)


def test_a_lone_digit_is_not_a_fact() -> None:
    """单个数字在正文里常是"第三点"的一部分；抽成事实只会让表达层不敢用自己话。"""
    assert "3" not in extract_facts("第一步做完，第二步检查。")


def test_fact_check_normalizes_width_and_whitespace() -> None:
    assert facts_preserved("得分是 ０.８９８ 。", ("0.898",)) == ()
    assert facts_preserved("得分是 0.9 。", ("0.898",)) == ("0.898",)


# ── 提示词 ──────────────────────────────────────────────────────────────────


def test_the_render_prompt_carries_persona_points_and_facts() -> None:
    sir = SIR(
        intent="answer",
        key_points=("甲", "乙"),
        tone="direct",
        length="short",
        facts=("0.898",),
        draft="原草稿。",
    )
    prompt = render_prompt(sir, PERSONA)
    for want in (PERSONA, "- 甲", "- 乙", "0.898", "不分点", "不用 emoji"):
        assert want in prompt


def test_the_sir_instruction_asks_for_both_blocks() -> None:
    text = sir_instruction()
    assert "<think>" in text and "<content>" in text
    assert "必须保留的事实" in text


def test_style_signature_is_deterministic_and_reads_the_usual_markers() -> None:
    first = style_signature("好！真的吗？哈哈😄因为这样，所以那样。")
    second = style_signature("好！真的吗？哈哈😄因为这样，所以那样。")
    assert first == second
    assert first["exclamations"] == 1
    assert first["questions"] == 1
    assert first["emoji"] == 1
    assert first["connectives"] == 2


# ── 表达层：渲染与退回 ──────────────────────────────────────────────────────


def _layer(script, **kwargs) -> ExpressionLayer:
    return ExpressionLayer(ScriptedBackend(script, **kwargs), PERSONA)


def test_a_clean_render_is_used() -> None:
    sir = SIR(intent="answer", key_points=("甲",), draft="草稿。", facts=())
    layer = _layer([[text_chunk("好", done=True)]])
    outcome = layer.render(sir)
    assert outcome.source == "styled"
    assert outcome.text == "好"
    assert outcome.reason == ""


def test_losing_a_fact_falls_back_to_the_draft() -> None:
    """**宁可语气不统一，也不让一个数被悄悄改掉。**"""
    sir = SIR(intent="answer", key_points=("分数",), draft="分数是 0.898。", facts=("0.898",))
    layer = _layer([[text_chunk("分数大概是 0.9。", done=True)]])
    outcome = layer.render(sir)
    assert outcome.source == "draft"
    assert outcome.text == "分数是 0.898。"
    assert "0.898" in outcome.reason


def test_a_question_rewritten_as_a_statement_falls_back() -> None:
    """问号不是"事实"，但它决定这句话是不是在问。2026-11-03 实测被吃掉过一次。"""
    sir = SIR(intent="ask", key_points=("甲",), draft="需要我详细说说吗？")
    outcome = _layer([[text_chunk("需要我详细说说", done=True)]]).render(sir)
    assert outcome.source == "draft"
    assert outcome.text == "需要我详细说说吗？"
    assert "问句" in outcome.reason


def test_a_question_that_keeps_its_mark_is_fine() -> None:
    sir = SIR(intent="ask", key_points=("甲",), draft="需要我详细说说吗？")
    outcome = _layer([[text_chunk("要不要展开讲讲？", done=True)]]).render(sir)
    assert outcome.source == "styled"


def test_url_case_is_significant_but_chinese_is_not() -> None:
    assert facts_preserved("见 https://Example.com/A", ("https://example.com/a",)) == (
        "https://example.com/a",
    )
    assert facts_preserved("见 https://example.com/a", ("https://example.com/a",)) == ()


def test_the_render_request_turns_thinking_off_by_default() -> None:
    layer = _layer([[text_chunk("好", done=True)]])
    assert layer.request(SIR(draft="草稿")).extra == {"think": False}


def test_a_backend_error_falls_back_without_raising() -> None:
    sir = SIR(intent="answer", key_points=("甲",), draft="草稿正文。")
    layer = _layer([[text_chunk("不该被读到")]], raise_at=0)
    outcome = layer.render(sir)
    assert outcome.source == "draft"
    assert outcome.text == "草稿正文。"
    assert "ChatBackendError" in outcome.reason


def test_an_empty_draft_is_never_rendered() -> None:
    """空草稿只剩规则文本，模型会回一句"要点：无"——那不能被当成正文。"""
    sir = SIR(intent="answer", key_points=(), draft="")
    backend = ScriptedBackend([[text_chunk("不该被调用")]])
    outcome = ExpressionLayer(backend, PERSONA).render(sir)
    assert outcome.source == "draft"
    assert outcome.text == ""
    assert "空草稿" in outcome.reason
    assert backend.calls == 0


def test_an_empty_render_falls_back() -> None:
    sir = SIR(intent="answer", key_points=("甲",), draft="草稿正文。")
    outcome = _layer([[text_chunk("   ", done=True)]]).render(sir)
    assert outcome.source == "draft"
    assert outcome.text == "草稿正文。"


def test_stopping_midway_returns_the_draft() -> None:
    sir = SIR(intent="answer", key_points=("甲",), draft="草稿正文。")
    calls = {"n": 0}

    def keep() -> bool:
        calls["n"] += 1
        return calls["n"] < 2

    outcome = _layer([[text_chunk("第一片"), text_chunk("第二片", done=True)]]).render(
        sir, keep_going=keep
    )
    assert outcome.source == "draft"
    assert outcome.text == "草稿正文。"
