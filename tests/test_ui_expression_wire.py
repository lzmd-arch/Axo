"""表达层的接线：只在不带工具的那一轮生效，且**关着时一个字节都不多走**。

这里守三件事：

1. ``_start_after_assembly`` 里那条分支真的存在——打开时把增量攒进草稿、关着时仍
   直连 ``push_token``（关着时的行为必须与加这一层之前逐位相同）；
2. ``_on_draft_ready`` 的三条早退：被打断、没有渲染后端、空正文——都要把草稿交出去；
3. ``_on_styled_ready`` 把两次调用（主模型 + 渲染）的 token **加在一起**再收尾。

不建 ``MainWindow``：理由同 ``test_ui_tools_wire``（离屏 QtWebEngine 一渲染就硬杀
进程）。这里用未绑定方法配一个人造 ``self``。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from axo.expression.layer import ExpressionOutcome
from axo.expression.sir import SIR
from axo.ui.expression_worker import ExpressionWorker
from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


class _ChatView:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.messages.append((role, text))


def _fake(**overrides):
    base = SimpleNamespace(
        _expression_worker=None,
        chat_view=_ChatView(),
        delivered=[],
    )
    base._deliver_reply = lambda text, stats, note="": base.delivered.append((text, stats, note))
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# ── 1. 接线：分支存在，关着时直连 ──────────────────────────────────────────


def test_the_chat_path_buffers_only_when_enabled() -> None:
    body = inspect.getsource(MainWindow._start_after_assembly)
    assert "if self._expression_settings.enabled:" in body, "表达层的分支没接上"
    assert "worker.chunk.connect(self._collect_draft)" in body
    assert "worker.chunk.connect(self.chat_view.push_token)" in body, (
        "关着时必须仍然直连流式文本——否则这一层的开关等于把所有人都拖慢"
    )


def test_the_task_path_is_not_touched() -> None:
    """任务轮（带工具）这一版**不经过**表达层：中间几轮的"边想边说"与工具结果
    是一体的，重写它会把"她做了什么"和"她怎么说"混在一起。这条钉住"没接"。"""
    body = inspect.getsource(MainWindow._start_agent_turn)
    assert "_collect_draft" not in body
    assert "expression" not in body


# ── 2. 早退：三种情况都要把草稿交出去 ──────────────────────────────────────


def test_an_interrupted_turn_never_reaches_the_expression_backend() -> None:
    fake = _fake(_draft_parts=["半句"])

    def boom():
        raise AssertionError("被打断的一轮不该去构建渲染后端")

    fake._expression_backend = boom
    MainWindow._on_draft_ready(fake, "半句", {"interrupted": True})

    assert fake.delivered == [("半句", {"interrupted": True}, "")]


def test_a_missing_renderer_delivers_the_draft_with_a_reason() -> None:
    fake = _fake(_draft_parts=["草稿正文"], _expression_backend=lambda: None)
    MainWindow._on_draft_ready(fake, "草稿正文", {})

    text, _stats, note = fake.delivered[0]
    assert text == "草稿正文"
    assert "没有可用的渲染模型" in note


def test_an_empty_draft_still_finishes_the_turn() -> None:
    fake = _fake(_draft_parts=[])
    MainWindow._on_draft_ready(fake, "", {})

    assert fake.delivered == [("", {}, "")]


# ── 3. 收尾：token 相加、退回时说一句 ──────────────────────────────────────


def test_styled_output_is_delivered_with_both_calls_counted() -> None:
    fake = _fake()
    outcome = ExpressionOutcome("改好的句子", "styled", "", 10, 5)
    MainWindow._on_styled_ready(
        fake, outcome, "草稿", {"prompt_tokens": 100, "completion_tokens": 20}
    )

    text, stats, note = fake.delivered[0]
    assert text == "改好的句子"
    assert stats["prompt_tokens"] == 110
    assert stats["completion_tokens"] == 25
    assert note == ""
    assert fake.chat_view.messages == []


def test_a_fallback_says_why_and_delivers_the_text() -> None:
    fake = _fake()
    outcome = ExpressionOutcome("草稿", "draft", "表达层改动了事实（丢了 0.898）")
    MainWindow._on_styled_ready(fake, outcome, "草稿", {})

    assert fake.delivered[0][0] == "草稿"
    assert fake.chat_view.messages[0][0] == "system"
    assert "0.898" in fake.chat_view.messages[0][1]


def test_tokens_are_not_invented_when_one_side_is_missing() -> None:
    fake = _fake()
    outcome = ExpressionOutcome("好", "styled", "", None, 5)
    MainWindow._on_styled_ready(fake, outcome, "草稿", {})
    assert fake.delivered[0][1]["completion_tokens"] == 5
    assert fake.delivered[0][1]["prompt_tokens"] is None


# ── 4. 工作线程：成功与异常都必须发一次 finished ────────────────────────────


class _Layer:
    def __init__(self, outcome=None, error: Exception | None = None) -> None:
        self._outcome = outcome
        self._error = error
        self.calls = 0

    def render(self, sir, *, keep_going=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._outcome


def test_the_worker_emits_the_outcome(qapp: QApplication) -> None:
    outcome = ExpressionOutcome("好", "styled")
    worker = ExpressionWorker(_Layer(outcome), SIR(draft="草稿"))
    got: list[object] = []
    worker.finished.connect(got.append)
    worker.run()
    assert got == [outcome]


def test_the_worker_turns_an_exception_into_a_fallback(qapp: QApplication) -> None:
    worker = ExpressionWorker(_Layer(error=RuntimeError("炸了")), SIR(draft="草稿正文"))
    got: list[object] = []
    worker.finished.connect(got.append)
    worker.run()
    assert got and got[0].source == "draft"
    assert got[0].text == "草稿正文"
    assert "RuntimeError" in got[0].reason
