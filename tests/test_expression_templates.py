"""模板库与混合渲染：命中、填槽、事实与问句核对、以及"命中就不调模型"。

模板最容易出的两种错都在这里钉住：

1. **拼错的槽位**——加载时拒绝，而不是运行时永远不命中；
2. **模板悄悄丢事实**——命中之后仍走 ``facts_preserved``，不过就继续走模型。
"""

from __future__ import annotations

import json

import pytest
from fakes import ScriptedBackend, text_chunk

from axo.expression.layer import ExpressionLayer
from axo.expression.mix import MixedRenderer
from axo.expression.sir import SIR
from axo.expression.templates import BUILTIN_TEMPLATES, Template, TemplateBank


class _NeverCalled:
    """命中模板时**不该被调用**的渲染器。"""

    def render(self, sir, *, keep_going=None):
        raise AssertionError("模板命中时不该调用模型")


def _bank(*templates: Template) -> TemplateBank:
    return TemplateBank(templates)


# ── 加载与校验 ──────────────────────────────────────────────────────────────


def test_the_builtin_bank_loads() -> None:
    bank = TemplateBank.builtin()
    assert len(bank.templates) == len(BUILTIN_TEMPLATES) >= 5


def test_an_unknown_slot_is_rejected_at_load_time() -> None:
    """留到运行时的症状是"模板永远不命中"，而那看起来像模板库太小。"""
    with pytest.raises(ValueError, match="未知槽位"):
        TemplateBank([Template("bad", "{bogus}")])


def test_a_bad_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        TemplateBank([Template("bad", "好。", kind="poem")])


def test_loading_jsonl_rejects_a_broken_line(tmp_path) -> None:
    path = tmp_path / "templates.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        TemplateBank.load(path)


def test_loading_jsonl_roundtrips(tmp_path) -> None:
    path = tmp_path / "templates.jsonl"
    rows = [
        {"id": "refuse", "pattern": "不知道。", "kind": "exact", "intents": ["refuse"]},
        {"id": "answer", "pattern": "{key_point:0}。", "intents": ["answer"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    bank = TemplateBank.load(path)
    assert [template.id for template in bank.templates] == ["refuse", "answer"]


# ── 匹配与填槽 ──────────────────────────────────────────────────────────────


def test_an_exact_template_renders_verbatim() -> None:
    bank = _bank(Template("r", "不知道。", kind="exact", intents=("refuse",)))
    hit = bank.render(SIR(intent="refuse", draft="不知道。"))
    assert hit is not None and hit.text == "不知道。" and hit.template_id == "r"


def test_a_skeleton_fills_key_points() -> None:
    bank = _bank(Template("a", "{key_point:0}。{key_point:1}。", intents=("answer",)))
    hit = bank.render(SIR(intent="answer", key_points=("甲", "乙"), draft="甲乙"))
    assert hit is not None and hit.text == "甲。乙。"


def test_a_missing_slot_means_no_hit() -> None:
    bank = _bank(Template("a", "{key_point:3}。", intents=("answer",)))
    assert bank.render(SIR(intent="answer", key_points=("甲",), draft="甲")) is None


def test_matching_is_deterministic() -> None:
    bank = TemplateBank.builtin()
    sir = SIR(intent="refuse", tone="direct", length="short", draft="不知道。")
    assert bank.match(sir) == bank.match(sir)


def test_the_trigger_filters_by_intent_tone_and_length() -> None:
    bank = _bank(
        Template("direct", "不知道。", kind="exact", intents=("refuse",), tones=("direct",)),
        Template("casual", "没印象。", kind="exact", intents=("refuse",), tones=("casual",)),
    )
    hit = bank.render(SIR(intent="refuse", tone="casual", draft="不知道。"))
    assert hit is not None and hit.template_id == "casual"


# ── 事实与问句 ──────────────────────────────────────────────────────────────


def test_a_template_that_drops_a_fact_is_not_used() -> None:
    bank = _bank(Template("a", "{key_point:0}。", intents=("answer",)))
    sir = SIR(intent="answer", key_points=("分数",), facts=("0.898",), draft="分数 0.898")
    hit = bank.render(sir)
    assert hit is None, "0.898 不在模板正文里，应该判定为丢事实"


def test_a_template_that_includes_the_fact_is_used() -> None:
    bank = _bank(Template("a", "{key_point:0}（{facts}）。", intents=("answer",)))
    sir = SIR(intent="answer", key_points=("分数",), facts=("0.898",), draft="分数 0.898")
    hit = bank.render(sir)
    assert hit is not None and "0.898" in hit.text


def test_an_ask_template_must_keep_the_question_mark() -> None:
    bank = _bank(Template("a", "{key_point:0}", intents=("ask",)))
    hit = bank.render(SIR(intent="ask", key_points=("要不要展开",), draft="要不要展开？"))
    assert hit is None

    bank = _bank(Template("a", "{key_point:0}？", intents=("ask",)))
    hit = bank.render(SIR(intent="ask", key_points=("要不要展开",), draft="要不要展开？"))
    assert hit is not None and hit.text.endswith("？")


# ── 混合渲染：命中就不调模型 ────────────────────────────────────────────────


def test_a_template_hit_never_calls_the_model() -> None:
    sir = SIR(intent="refuse", tone="direct", length="short", draft="不知道。")
    renderer = MixedRenderer(TemplateBank.builtin(), _NeverCalled())
    outcome = renderer.render(sir)
    assert outcome.source == "styled"
    assert outcome.route == "template"
    assert outcome.template_id


def test_a_template_miss_falls_through_to_the_model() -> None:
    sir = SIR(intent="answer", length="long", key_points=("甲",), draft="这是一段很长的回答。")
    layer = ExpressionLayer(ScriptedBackend([[text_chunk("模型写的。", done=True)]]), "人设")
    outcome = MixedRenderer(_bank(), layer).render(sir)
    assert outcome.route == "model"
    assert outcome.text == "模型写的。"


def test_a_failing_model_still_falls_back_to_the_draft() -> None:
    sir = SIR(intent="answer", length="long", key_points=("甲",), draft="草稿正文。")
    layer = ExpressionLayer(ScriptedBackend([[text_chunk("x")]], raise_at=0), "人设")
    outcome = MixedRenderer(_bank(), layer).render(sir)
    assert outcome.source == "draft"
    assert outcome.text == "草稿正文。"
