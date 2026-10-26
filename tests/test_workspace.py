"""工作区上下文：自动进 L1 的那几行，必须短、真、可截断。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from axo.agent.workspace import build_workspace_context


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def test_git_workspace_includes_branch_changes_and_agents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "AGENTS.md").write_text("先读 AGENTS，再改代码。\n第二行。\n", encoding="utf-8")
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")

    text = build_workspace_context((repo,))
    assert "工作区：" in text
    assert "git：" in text and "1 个未提交改动" in text
    assert "init" in text
    assert "先读 AGENTS，再改代码。" in text
    assert "src" in text


def test_no_roots_is_empty() -> None:
    assert build_workspace_context(()) == ""


def test_non_git_directory_still_gives_directory_context(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    text = build_workspace_context((tmp_path,))
    assert "不是仓库" in text
    assert "notes" in text


def test_test_command_is_detected_for_python_project(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    text = build_workspace_context((tmp_path,))
    assert f'测试命令（自动探测）："{sys.executable}" -m pytest -q' in text


def test_test_files_are_listed(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    text = build_workspace_context((tmp_path,))
    assert "测试文件：" in text and "tests/test_a.py" in text
