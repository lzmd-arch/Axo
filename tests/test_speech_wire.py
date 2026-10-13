"""出声那几处**真的接上了**：纯源码分析，核心腿里跑（不碰 Qt）。

「纯层全绿 ≠ 接线在」那条纪律（`memterm-pure-layer-not-wired`）在语音这一族尤其要紧：
`_speak_reply` 自己测得再全，也不证明"`_on_reply_finished` 真的调了它"。2026-10-13 的
实现里就有过一次差点：朗读挂在了错误的分支上，行为测试全绿而按钮按下去无声。

钉四条线：

1. 语音回合的回复要过 `_speak_reply`（`_on_reply_finished`）；
2. 提醒句也要过它（`_on_speech_finished`）——「到点由她开口」的完成；
3. **降级那条路不过它**（`_tell_reminder`）——机器声念「本该由她开口」是二次羞辱；
4. 让路要收播放（`_stop_speech` 里 `_speech_player.stop()` 在 early-return **之前**）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "axo" / "ui" / "main_window.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"main_window.py 里找不到 {name}")


def _calls(node: ast.AST, attr: str) -> list[int]:
    """这个函数体里 ``self.<attr>(...)`` 的行号。"""
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ]


def test_a_spoken_reply_goes_through_speak_reply(tree: ast.Module) -> None:
    lines = _calls(_method(tree, "_on_reply_finished"), "_speak_reply")
    assert lines, "_on_reply_finished 没接朗读——说出来的那一轮她不会开口（无声）"
    source = (
        ast.get_source_segment(
            SOURCE.read_text(encoding="utf-8"), _method(tree, "_on_reply_finished")
        )
        or ""
    )
    assert "_spoken_reply" in source, "朗读没有被'这一轮是说出来的'那条判定框住——打字也会念"


def test_the_reminder_line_goes_through_speak_reply(tree: ast.Module) -> None:
    assert _calls(_method(tree, "_on_speech_finished"), "_speak_reply"), (
        "_on_speech_finished 没接朗读——「到点由她开口」只出字不出声"
    )


def test_the_degraded_reminder_stays_silent(tree: ast.Module) -> None:
    assert _calls(_method(tree, "_tell_reminder"), "_speak_reply") == [], (
        "_tell_reminder 接了朗读——机器声念「本该由她开口」是二次羞辱"
    )


def test_yielding_stops_playdown_before_any_early_return(tree: ast.Module) -> None:
    """``_stop_speech`` 里 ``_speech_player.stop()`` 必须在 ``_speak_worker is None``
    那句 early-return **之前**——纯播放的打断走的就是那条早路。"""

    node = _method(tree, "_stop_speech")
    player_stops = [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "stop"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "_speech_player"
    ]
    assert player_stops, "_stop_speech 没停播放——她说到一半停不下来"
    returns = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Return) and not n.value]
    assert returns, "early-return 没了？那这条判据的语义变了，回来看这段"
    assert all(line < returns[0] for line in player_stops), (
        f"停播在第一个裸 return 之后（stop 在 {player_stops}，return 在 {returns}）——"
        "纯播放的打断会被 early-return 吞掉"
    )
