"""召回门槛（弱记忆不注入）与"直说不确定"那条路。

这批改动有两个风险点，各钉一条：

1. **门槛要真的进到召回配置里。** 只写一个设置字段、没人读它，就是本项目反复出现的
   "纯层全绿但没接线"——界面上滑了、下一轮召回照旧按 0.10 塞进来六十条。
2. **"不确定"路径的边界。** 它有工具轮与闲聊轮两条不该走的边：工具轮模型还能自己去
   ``search_memory``，闲聊轮（"今天有点累"）本来就不该要求有记忆。越界一次，用户就会
   看到"你好"被回一句"我不太熟悉"。

`_should_say_uncertain` 用真实方法 + 只带必要字段的替身跑（同 ``test_ui_agent_slots``
的手法）；不建 ``MainWindow``（离屏下 QtWebEngine 一渲染就把进程硬杀掉）。

**文件名必须带 ``test_ui_`` 前缀**：CI 的核心腿用 ``--ignore-glob='tests/test_ui_*'``
跳过这一族，而本模块在顶层 import ``main_window``——它会拉进 QtWebEngine，Linux 的
核心腿上没有 libEGL，**收集阶段就会 ImportError**。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from axo.agent.router import MEMORY_QUERY_MARKERS, is_memory_question
from axo.phrases import UNCERTAIN_PHRASES, pick_phrase
from axo.ui.main_window import MainWindow
from memcore.audit import ChannelScores, RecalledMemory, RecallLedger
from memcore.recall import RecallConfig

pytestmark = pytest.mark.gui


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append(("appendMessage", role, text, meta))

    def set_status(self, text: str) -> None:
        self.calls.append(("setStatus", text))


def _ledger(injected: int = 0) -> RecallLedger:
    items = tuple(
        RecalledMemory(
            memory_id=f"m{index}",
            content="正文",
            kind="fact",
            tier="working",
            weight=1.0,
            tokens=10,
            scores=ChannelScores(lexical=0.5, fused=0.5),
        )
        for index in range(injected)
    )
    return RecallLedger(query_preview="x", scanned=10, injected=items)


def _window(*, text: str, tools: bool = False, ambiguous: bool = False) -> SimpleNamespace:
    fake = SimpleNamespace(
        _turn_uses_tools=tools,
        _turn_needs_clarification=ambiguous,
        _pending_text=text,
        chat_view=_Recorder(),
        finished=[],
    )
    fake._on_reply_finished = lambda reply, stats: fake.finished.append((reply, stats))
    fake._should_say_uncertain = MainWindow._should_say_uncertain.__get__(fake)
    return fake


# ── 接线：门槛真的进了召回配置 ──────────────────────────────────────────────


def test_recall_config_carries_the_threshold() -> None:
    body = inspect.getsource(MainWindow._recall_config)
    assert "min_score=self._recall_min_score" in body, "门槛没进召回配置——滑杆只是个摆设"
    assert hasattr(RecallConfig(), "min_score")


def test_the_window_reads_the_threshold_from_settings() -> None:
    """设置改了要落到主窗口那个字段上，下一轮才生效（与 ``_graph_enabled`` 同形）。"""
    body = inspect.getsource(MainWindow._on_memory_changed)
    assert "self._recall_min_score = settings.recall_min_score" in body


# ── 边界：什么时候才直说不确定 ──────────────────────────────────────────────


def test_an_empty_recall_on_a_memory_question_says_so() -> None:
    window = _window(text="我上次说的那个项目怎么样了")
    assert window._should_say_uncertain(_ledger(0)) is True
    spoken, stats = window.finished[0]
    assert spoken in UNCERTAIN_PHRASES
    assert stats["no_grounding"] is True
    assert ("appendMessage", "assistant", spoken, None) in window.chat_view.calls


def test_a_tool_turn_never_refuses() -> None:
    """有工具时模型可以自己去 ``search_memory``——替它拒答等于把路堵上。"""
    window = _window(text="我上次说的那个项目怎么样了", tools=True)
    assert window._should_say_uncertain(_ledger(0)) is False
    assert window.finished == []


def test_an_ambiguous_turn_never_refuses() -> None:
    window = _window(text="我上次说的那个项目怎么样了", ambiguous=True)
    assert window._should_say_uncertain(_ledger(0)) is False


def test_chit_chat_without_memories_is_not_refused() -> None:
    """「今天有点累」没有记忆是正常的，不该回一句"我不太熟悉"。"""
    window = _window(text="今天有点累，随便聊两句吧")
    assert window._should_say_uncertain(_ledger(0)) is False


def test_when_something_was_recalled_it_still_answers() -> None:
    window = _window(text="我上次说的那个项目怎么样了")
    assert window._should_say_uncertain(_ledger(2)) is False
    assert window.finished == []


def test_a_failed_recall_ledger_does_not_trigger_it() -> None:
    """召回失败（``ledger is None``）时不能假装"没有记忆"——那是两件事。"""
    window = _window(text="我上次说的那个项目怎么样了")
    assert window._should_say_uncertain(None) is False


# ── 判据表与话术 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["我上次说的那个项目", "记忆里有没有这件事", "帮我回想一下"])
def test_memory_question_markers(text: str) -> None:
    assert is_memory_question(text) is True


@pytest.mark.parametrize("text", ["今天天气怎么样", "帮我看看新闻", ""])
def test_not_memory_questions(text: str) -> None:
    assert is_memory_question(text) is False


def test_the_marker_table_is_shared_with_the_router() -> None:
    """路由与界面读**同一张表**——两张表迟早一边改了另一边没改。

    2026-11-01 起路由不再把它拼进 ``_TASK_MARKERS``，而是由 ``rule_register`` 直接
    调 ``is_memory_question``（同一个常量、同一个函数）；这一条跟着钉新的接线。
    """
    source = inspect.getsource(__import__("axo.agent.router", fromlist=["rule_register"]))
    assert "is_memory_question(stripped)" in source
    assert set(MEMORY_QUERY_MARKERS) >= {"上次", "记忆里"}


def test_pick_phrase_is_reproducible_with_a_seed() -> None:
    import random

    first = pick_phrase(random.Random(1))
    second = pick_phrase(random.Random(1))
    assert first == second
    assert first in UNCERTAIN_PHRASES
