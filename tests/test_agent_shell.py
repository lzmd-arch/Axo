"""命令执行：超时、取消、输出上限、环境白名单。**只用本机跑得动的短命令。**"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from axo.agent.shell import _minimal_env, run_command


def python_snippet(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_a_short_command_returns_its_output(tmp_path: Path) -> None:
    result = run_command(python_snippet("print('hello')"), cwd=tmp_path, timeout_s=20)

    assert result.ok is True
    assert result.exit_code == 0
    assert "hello" in result.output


def test_a_failing_command_is_reported_with_its_exit_code(tmp_path: Path) -> None:
    result = run_command(python_snippet("raise SystemExit(3)"), cwd=tmp_path, timeout_s=20)

    assert result.ok is False
    assert result.exit_code == 3


def test_a_timeout_kills_the_command(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_command(python_snippet("import time; time.sleep(30)"), cwd=tmp_path, timeout_s=1)

    assert result.timed_out is True
    assert result.ok is False
    assert time.monotonic() - started < 15, "超时之后没有尽快返回"


def test_the_cancel_event_stops_a_long_command(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()

    result = run_command(
        python_snippet("import time; time.sleep(30)"),
        cwd=tmp_path,
        timeout_s=60,
        cancel=cancel,
    )

    assert result.cancelled is True
    assert result.ok is False


def test_output_is_clipped_with_a_visible_marker(tmp_path: Path) -> None:
    result = run_command(
        python_snippet("print('x' * 100000)"), cwd=tmp_path, timeout_s=20, output_cap=1000
    )

    assert result.truncated is True
    assert "中间省略" in result.output
    assert len(result.output.encode("utf-8")) < 3000


def test_the_environment_is_a_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """**模型跑的命令不该顺手拿到模型自己的凭据。**"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DEEPSEEK_KEY", "should-not-leak")

    env = _minimal_env()

    assert "OPENAI_API_KEY" not in env
    assert "DEEPSEEK_KEY" not in env
    assert "PATH" in env or "SYSTEMROOT" in env


def test_an_empty_command_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="空"):
        run_command("   ", cwd=tmp_path)
