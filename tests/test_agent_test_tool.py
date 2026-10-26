"""测试运行工具：失败摘要必须真的从输出里摘出来，不能只说"失败了"。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo.agent.catalog import default_tools
from axo.agent.test_tools import RunTestsTool, _failure_lines
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def test_failure_lines_prefer_test_ids_over_assertion_detail() -> None:
    output = (
        "E   assert 1 == 2\nFAILED tests/test_a.py::test_a - assert 1 == 2\nE   assert 3 == 4\n"
    )
    assert _failure_lines(output) == ("FAILED tests/test_a.py::test_a - assert 1 == 2",)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "--- FAIL: TestAdd (0.00s)\n    calc_test.go:10: got 1 want 2\nFAIL\tpkg\t0.1s\n",
            "--- FAIL: TestAdd",
        ),
        ("test add ... FAILED\n\nfailures:\n    add\n", "test add ... FAILED"),
        ("● calc › adds\n\n  expect(received).toBe(5)\n", "● calc › adds"),
        ("✕ adds numbers\n", "✕ adds numbers"),
        ("✖ add (0.7ms)\n", "✖ add"),
        ("[-] fails\nTests Passed: 0, Failed: 1\n", "[-] fails"),
        ("[  FAILED  ] CalcTest.Adds\n", "[  FAILED  ] CalcTest.Adds"),
        ("Tests run: 3, Failures: 1, Errors: 0\n[ERROR] Failures:\n", "[ERROR] Failures:"),
        (
            "1) test adds (CalcTest)\n   ** (ExUnit.AssertionError) boom\n",
            "1) test adds (CalcTest)",
        ),
        ("  ***Failed  test_add\n", "***Failed  test_add"),
        ("Failures:\n  1) Calc adds\n     Failure/Error: expect(1).to eq(2)\n", "Failures:"),
        ("FAILURES!\nTests: 2, Assertions: 3, Failures: 1.\n", "FAILURES!"),
        ("Failed CalcTest.Adds [1 ms]\nFailed! - Failed: 1\n", "Failed CalcTest.Adds [1 ms]"),
    ],
)
def test_failure_markers_cross_language(output: str, expected: str) -> None:
    assert any(expected in line for line in _failure_lines(output))


def _ctx(root: Path) -> ToolContext:
    return ToolContext(store=MemoryStore(":memory:"), write_root=(root,), allow_commands=True)


def test_failing_test_returns_excerpt_and_next_step(tmp_path: Path) -> None:
    (tmp_path / "fail.py").write_text(
        "print('FAILED tests/test_x.py::test_foo')\nraise SystemExit(1)\n", encoding="utf-8"
    )
    outcome = RunTestsTool().run(
        {"command": f'"{sys.executable}" fail.py', "timeout_s": 20}, _ctx(tmp_path)
    )
    assert not outcome.ok
    assert "FAILED tests/test_x.py::test_foo" in outcome.content
    assert "下一步" in outcome.content
    assert "退出码 1" in outcome.content


def test_passing_test_is_success(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("print('ok')\n", encoding="utf-8")
    outcome = RunTestsTool().run(
        {"command": f'"{sys.executable}" ok.py', "timeout_s": 20}, _ctx(tmp_path)
    )
    assert outcome.ok and "通过" in outcome.content


def test_no_command_needs_explicit_override(tmp_path: Path) -> None:
    outcome = RunTestsTool().run({}, _ctx(tmp_path))
    assert not outcome.ok and "没有探测到测试命令" in outcome.content


def test_run_tests_registered_only_with_commands(tmp_path: Path) -> None:
    without = {t.spec.name for t in default_tools(read_root=(tmp_path,), write_root=(tmp_path,))}
    with_commands = {
        t.spec.name
        for t in default_tools(read_root=(tmp_path,), write_root=(tmp_path,), allow_commands=True)
    }
    assert "run_tests" not in without
    assert "run_tests" in with_commands
