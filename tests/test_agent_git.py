"""工作区工具：git 只读三件、仓库概览与 apply_patch。"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from axo.agent.catalog import RollbackTool, default_tools
from axo.agent.checkpoints import CheckpointStore
from axo.agent.gate import build_request
from axo.agent.git_tools import (
    ApplyPatchTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    RepoOverviewTool,
)
from axo.agent.tools import ToolContext
from axo.llm.base import ToolCall
from memcore.store import MemoryStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("old\n", encoding="utf-8")
    (root / "b.txt").write_text("old\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("先读 AGENTS，再动手。\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _ctx(repo: Path) -> ToolContext:
    return ToolContext(store=MemoryStore(":memory:"), read_root=(repo,), write_root=(repo,))


def test_status_diff_and_log(repo: Path) -> None:
    (repo / "a.txt").write_text("new\n", encoding="utf-8")
    ctx = _ctx(repo)

    status = GitStatusTool().run({}, ctx)
    assert status.ok and "a.txt" in status.content
    diff = GitDiffTool().run({}, ctx)
    assert diff.ok and "-old" in diff.content and "+new" in diff.content
    log = GitLogTool().run({"limit": 3}, ctx)
    assert log.ok and "init" in log.content


def test_repo_overview_includes_agents(repo: Path) -> None:
    out = RepoOverviewTool().run({}, _ctx(repo))
    assert out.ok
    assert "仓库根" in out.content and "AGENTS.md" in out.content and "先读 AGENTS" in out.content
    assert "init" in out.content


def test_apply_patch_modifies_a_file(repo: Path) -> None:
    patch = """--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
"""
    out = ApplyPatchTool().run({"patch": patch}, _ctx(repo))
    assert out.ok, out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_rejects_outside_path(repo: Path) -> None:
    patch = """--- a/a.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""
    out = ApplyPatchTool().run({"patch": patch}, _ctx(repo))
    assert not out.ok and ".." in out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"


def test_non_repo_is_a_clear_failure(tmp_path: Path) -> None:
    ctx = ToolContext(store=MemoryStore(":memory:"), read_root=(tmp_path,), write_root=(tmp_path,))
    out = GitStatusTool().run({}, ctx)
    assert not out.ok and "不是 git 仓库" in out.content


def test_apply_patch_is_registered_only_with_write_root_and_asks(repo: Path) -> None:
    read_only = {t.spec.name: t for t in default_tools(read_root=(repo,))}
    writable = {t.spec.name: t for t in default_tools(read_root=(repo,), write_root=(repo,))}
    assert "apply_patch" not in read_only
    assert "apply_patch" in writable and writable["apply_patch"].requires_approval


def test_apply_patch_can_be_rolled_back(repo: Path, tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "cp")
    ctx = ToolContext(
        store=MemoryStore(":memory:"), read_root=(repo,), write_root=(repo,), checkpoints=store
    )
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
    out = ApplyPatchTool().run({"patch": patch}, ctx)
    assert out.ok and "检查点" in out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "new\n"
    rb = RollbackTool().run({}, ctx)
    assert rb.ok
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"


def test_apply_patch_approval_summary_shows_diff() -> None:
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
    req = build_request(
        ToolCall(id="c1", name="apply_patch", arguments={"patch": patch}),
        tool=ApplyPatchTool(),
    )
    assert "a.txt" in req.summary
    assert "--- a/a.txt" in req.summary
    assert req.files == ("a.txt",)


def _two_file_patch(repo: Path) -> str:
    (repo / "a.txt").write_text("new\n", encoding="utf-8")
    (repo / "b.txt").write_text("new\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(repo), "diff"], capture_output=True, text=True, check=True
    ).stdout
    _git(repo, "checkout", "--", ".")
    return patch


def test_apply_patch_files_filter_applies_only_selected(repo: Path) -> None:
    patch = _two_file_patch(repo)
    out = ApplyPatchTool().run({"patch": patch, "files": ["a.txt"]}, _ctx(repo))
    assert out.ok, out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "new\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "old\n"
    assert "b.txt" not in out.content


def test_apply_patch_uses_the_cards_file_selection(repo: Path) -> None:
    patch = _two_file_patch(repo)
    ctx = replace(_ctx(repo), call_id="c1", approval_selection=lambda _cid: ("b.txt",))
    out = ApplyPatchTool().run({"patch": patch}, ctx)
    assert out.ok, out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_with_no_selected_files_changes_nothing(repo: Path) -> None:
    patch = _two_file_patch(repo)
    ctx = replace(_ctx(repo), call_id="c1", approval_selection=lambda _cid: ())
    out = ApplyPatchTool().run({"patch": patch}, ctx)
    assert not out.ok and "没有选择任何文件" in out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "old\n"


def test_apply_patch_rejects_unknown_file_filter(repo: Path) -> None:
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
    out = ApplyPatchTool().run({"patch": patch, "files": ["missing.txt"]}, _ctx(repo))
    assert not out.ok and "没有这些文件" in out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"


def test_preview_patch_can_limit_to_selected_files(repo: Path) -> None:
    from axo.agent.git_tools import PreviewPatchTool

    patch = _two_file_patch(repo)
    out = PreviewPatchTool().run({"patch": patch, "files": ["a.txt"]}, _ctx(repo))
    assert out.ok
    assert "a.txt" in out.content
    assert "b.txt" not in out.content


def test_approval_files_respect_model_file_filter() -> None:
    patch = (
        "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/b.txt b/b.txt\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-old\n+new\n"
    )
    req = build_request(
        ToolCall(
            id="c1",
            name="apply_patch",
            arguments={"patch": patch, "files": ["a.txt"]},
        ),
        tool=ApplyPatchTool(),
    )
    assert req.files == ("a.txt",)
