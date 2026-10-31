"""每一轮的 token 用量：**留在对话里**，而不是只在状态栏闪一下。

2026-10-31 用户的原话里有两条：``会出现对话区没了的情况`` 与 ``现在看不到每次对话
的 token 用量``。前者在这个文件里只守"恢复路径存在且接上了线"，后者守两件事：

1. 收尾时那行小字真的被递给 ``end_message``（不是又写进状态栏）；
2. 回看时逐字稿里的数字也变成同一行小字——**用同一个格式化函数**，两处各写
   一份的话，现场与回看迟早对不上。

**不建 ``ChatView``**：离屏下 QtWebEngine 一执行脚本就把进程硬杀掉（见
``test_ui_chat_tools_wire`` 的同一条理由）。所以接线那两条用 ``inspect.getsource``
守"某一句话在不在"，行为那几条用纯函数。
"""

from __future__ import annotations

import inspect

import pytest

from axo.session.store import Message
from axo.ui.chat_view import ChatView
from axo.ui.main_window import MainWindow, display_rows, usage_footer

pytestmark = pytest.mark.gui


# ── 那行小字长什么样 ────────────────────────────────────────────────────


def test_both_numbers_are_shown() -> None:
    assert usage_footer(2259, 580) == "输入 2,259 · 输出 580 token"


def test_one_number_is_enough() -> None:
    """被中断的那一轮可能只有一半的数——**照样写出来，比空着诚实**。"""
    assert usage_footer(1200, None) == "输入 1,200 token"
    assert usage_footer(None, 7) == "输出 7 token"


def test_nothing_known_means_no_footer() -> None:
    assert usage_footer(None, None) == ""


def test_zero_is_a_number_not_a_missing_value() -> None:
    """0 与 ``None`` 不是一回事：前者是"确实用了 0"，后者是"这块不给数"。"""
    assert usage_footer(0, 0) == "输入 0 · 输出 0 token"


# ── 回看：逐字稿里的账变成同款小字 ───────────────────────────────────────


def test_history_row_carries_the_same_footer() -> None:
    (row,) = display_rows(
        [Message(role="assistant", content="好。", prompt_tokens=12, completion_tokens=5)]
    )
    assert row["meta"]["footer"] == usage_footer(12, 5)


def test_history_row_without_numbers_has_no_footer() -> None:
    (row,) = display_rows([Message(role="assistant", content="好。")])
    assert row["meta"] == {}


def test_proactive_line_shows_its_usage_too() -> None:
    """她主动开口那一轮也是模型请求，账要能对上。"""
    (row,) = display_rows(
        [
            Message(
                role="assistant",
                content="到点了。",
                proactive=True,
                prompt_tokens=800,
                completion_tokens=20,
            )
        ]
    )
    assert row["role"] == "proactive"
    assert row["meta"]["footer"] == "输入 800 · 输出 20 token"


# ── 接线：收尾那一句真的把数递下去了 ─────────────────────────────────────


def test_reply_finished_hands_the_footer_to_the_chat_view() -> None:
    body = inspect.getsource(MainWindow._on_reply_finished)
    assert "end_message(" in body, "收尾还在用不带页脚的那版 end_message"
    assert "usage_footer(" in body, "收尾没有把 token 数做成那行小字"
    assert 'statistics.get("prompt_tokens")' in body, "prompt_tokens 没被取出来"


def test_chat_view_replays_history_after_the_page_loads() -> None:
    """页面是异步加载的：``load_history`` 可能落在空页面上。

    那一次调用会被 ``window.ChatHost &&`` 静默吞掉——用户选了一个有内容的会话却
    看到空白。所以它必须**先缓存、再在加载完成时补推**。
    """
    load = inspect.getsource(ChatView.load_history)
    finished = inspect.getsource(ChatView._on_load_finished)
    assert "self._history" in load, "load_history 没有缓存那份消息"
    assert "loadHistory" in finished, "页面加载完成后没有补推历史"


def test_a_dead_renderer_is_reloaded_and_reported() -> None:
    """渲染进程死了：重载一次、写日志，而且**不许无限重启**。"""
    crash = inspect.getsource(ChatView._on_renderer_crashed)
    assert "_view.reload()" in crash
    assert "log_ui_error" in crash
    assert "_crash_count" in crash, "没有次数上限就会变成每两秒重启一次的循环"
    init = inspect.getsource(ChatView.__init__)
    assert "renderProcessTerminated.connect" in init, "崩了没人知道——信号没接"


def test_recovery_repushes_the_current_transcript() -> None:
    """重载后的那份缓存可能已经落后（这一轮的字只活在网页里）——用逐字稿重推。"""
    body = inspect.getsource(MainWindow._on_chat_recovered)
    assert "display_rows(self._session.messages)" in body
    assert "load_history" in body
    wire = inspect.getsource(MainWindow._wire)
    assert "recovered.connect" in wire, "恢复信号没有接线，重推永远不会发生"
