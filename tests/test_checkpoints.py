"""文件检查点与回滚：写/编辑/补丁之前的原文要能恢复。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from axo.agent.catalog import RollbackTool, WriteFileTool
from axo.agent.checkpoints import CheckpointError, CheckpointStore
from axo.agent.git_tools import PreviewPatchTool
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def test_capture_restore_existing_and_new(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    old = root / "old.txt"
    old.write_text("before\n", encoding="utf-8")
    new = root / "new.txt"
    store = CheckpointStore(tmp_path / "cp")
    cid = store.capture([old, new], "test")

    old.write_text("after\n", encoding="utf-8")
    new.write_text("created\n", encoding="utf-8")
    result = store.restore(cid, roots=(root,))

    assert old.read_text(encoding="utf-8") == "before\n"
    assert not new.exists()
    assert str(old.resolve()) in result.restored
    assert str(new.resolve()) in result.deleted


def test_restore_refuses_path_outside_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    store = CheckpointStore(tmp_path / "cp")
    cid = store.capture([outside], "escape")

    with pytest.raises(CheckpointError):
        store.restore(cid, roots=(allowed,))


def test_write_and_rollback_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("before\n", encoding="utf-8")
    store = CheckpointStore(tmp_path / "cp")
    ctx = ToolContext(store=MemoryStore(":memory:"), write_root=(tmp_path,), checkpoints=store)
    out = WriteFileTool().run({"path": "a.txt", "content": "after\n"}, ctx)
    assert out.ok and "检查点" in out.content
    assert target.read_text(encoding="utf-8") == "after\n"

    rollback = RollbackTool().run({}, ctx)
    assert rollback.ok
    assert target.read_text(encoding="utf-8") == "before\n"


def test_preview_patch_checks_without_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("old\n", encoding="utf-8")
    ctx = ToolContext(store=MemoryStore(":memory:"), read_root=(repo,), write_root=(repo,))
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
    out = PreviewPatchTool().run({"patch": patch}, ctx)
    assert out.ok and "可以应用" in out.content
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old\n"
