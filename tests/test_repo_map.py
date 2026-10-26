"""仓库地图：符号要摘对、噪声目录要跳过、截断要说明。"""

from __future__ import annotations

from pathlib import Path

import pytest

from axo.agent import repo_map as repo_map_module
from axo.agent.catalog import default_tools
from axo.agent.repo_map import RepoMapCache, RepoMapTool, build_repo_map
from axo.agent.snapshot import Snapshot
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def _ctx(root: Path) -> ToolContext:
    return ToolContext(store=MemoryStore(":memory:"), read_root=(root,))


def test_lists_files_and_python_symbols(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text(
        "class Greeter:\n"
        "    def greet(self, name: str) -> str:\n"
        "        return name\n"
        "\n"
        "def main() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    text = build_repo_map(tmp_path)
    assert "src/app.py" in text
    assert "tests/test_app.py" in text
    assert "class Greeter (L1)" in text
    assert "def greet(self, name: str) -> str (L2)" in text
    assert "def main() -> None (L5)" in text
    assert "def test_ok() (L1)" in text


def test_skips_noise_directories(tmp_path: Path) -> None:
    for noise in ("__pycache__", ".venv", "node_modules", ".git"):
        directory = tmp_path / noise
        directory.mkdir()
        (directory / "hidden.py").write_text("def hidden():\n    pass\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("def visible():\n    pass\n", encoding="utf-8")

    text = build_repo_map(tmp_path)
    assert "visible.py" in text
    for noise in ("__pycache__", ".venv", "node_modules", ".git", "hidden.py"):
        assert noise not in text


def test_reports_syntax_error_and_lists_binary(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(\n", encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02\xff")

    text = build_repo_map(tmp_path)
    assert "broken.py" in text and "语法错误" in text
    assert "data.bin" in text


def test_truncates_files_and_symbols_explicitly(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"m{index}.py").write_text(
            "def one():\n    pass\n\ndef two():\n    pass\n", encoding="utf-8"
        )

    files_clipped = build_repo_map(tmp_path, max_files=2)
    assert "文件列表已截断" in files_clipped
    assert "还有文件没有列出" in files_clipped

    symbols_clipped = build_repo_map(tmp_path, max_files=10, max_symbols=1)
    assert "符号已达到上限" in symbols_clipped


def test_output_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    assert build_repo_map(tmp_path) == build_repo_map(tmp_path)


def test_output_respects_char_cap(tmp_path: Path) -> None:
    for index in range(120):
        name = f"file_{index:03d}_" + "x" * 60 + ".py"
        (tmp_path / name).write_text("def f():\n    pass\n", encoding="utf-8")
    text = build_repo_map(tmp_path, max_files=400)
    assert len(text) <= 6000
    assert "仓库地图已截断" in text


def test_max_symbols_zero_lists_files_only(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def visible_symbol():\n    pass\n", encoding="utf-8")
    text = build_repo_map(tmp_path, max_symbols=0)
    assert "a.py" in text
    assert "visible_symbol" not in text


def test_max_symbols_zero_does_not_parse_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    calls: list[Path] = []
    monkeypatch.setattr(repo_map_module, "_symbols", lambda path: calls.append(path) or [])
    build_repo_map(tmp_path, max_symbols=0)
    assert calls == []


def test_tool_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def secret():\n    pass\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    outcome = RepoMapTool().run({"path": ".."}, _ctx(root))
    assert not outcome.ok
    assert "secret" not in outcome.content


def test_symlink_escaping_root_is_not_parsed(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.py"
    outside.write_text("def leaked_symbol():\n    pass\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "leak.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("这个环境不允许创建符号链接")

    text = build_repo_map(root)
    assert "leaked_symbol" not in text
    assert "leak.py" not in text


def test_tool_returns_map_for_authorized_directory(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def run():\n    pass\n", encoding="utf-8")
    outcome = RepoMapTool().run({}, _ctx(tmp_path))
    assert outcome.ok
    assert "def run() (L1)" in outcome.content
    assert not outcome.replayable


def test_registered_only_with_read_root(tmp_path: Path) -> None:
    names = {tool.spec.name for tool in default_tools()}
    assert "repo_map" not in names
    with_read = {tool.spec.name for tool in default_tools(read_root=(tmp_path,))}
    assert "repo_map" in with_read


def test_cache_avoids_rebuild_until_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake(root: Path, *, max_files: int, max_symbols: int) -> str:
        calls.append(1)
        return f"map-{len(calls)}"

    monkeypatch.setattr(repo_map_module, "build_repo_map", fake)
    cache = RepoMapCache()
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        read_root=(tmp_path,),
        repo_map_cache=cache,
    )
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")

    first = RepoMapTool().run({}, ctx)
    second = RepoMapTool().run({}, ctx)
    assert first.content == second.content == "map-1"
    assert len(calls) == 1

    (tmp_path / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    third = RepoMapTool().run({}, ctx)
    assert third.content == "map-2"
    assert len(calls) == 2


def test_cache_key_includes_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake(root: Path, *, max_files: int, max_symbols: int) -> str:
        calls.append(1)
        return f"map-{len(calls)}"

    monkeypatch.setattr(repo_map_module, "build_repo_map", fake)
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        read_root=(tmp_path,),
        repo_map_cache=RepoMapCache(),
    )
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    RepoMapTool().run({}, ctx)
    RepoMapTool().run({"max_files": 50}, ctx)
    assert len(calls) == 2


def test_truncated_snapshot_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake(root: Path, *, max_files: int, max_symbols: int) -> str:
        calls.append(1)
        return f"map-{len(calls)}"

    monkeypatch.setattr(repo_map_module, "build_repo_map", fake)
    monkeypatch.setattr(
        repo_map_module, "snapshot_tree", lambda _root: Snapshot("same", 4000, True)
    )
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        read_root=(tmp_path,),
        repo_map_cache=RepoMapCache(),
    )
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    assert RepoMapTool().run({}, ctx).content == "map-1"
    assert RepoMapTool().run({}, ctx).content == "map-2"
    assert len(calls) == 2
