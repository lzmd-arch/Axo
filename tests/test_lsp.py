"""LSP 桥：协议、四个动作、无服务器回退与进程收尾。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo.agent import lsp as lsp_module
from axo.agent.lsp import LspError, LspManager, LspTool
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore

_FAKE = Path(__file__).parent / "fakes" / "fake_lsp.py"


def _ctx(root: Path, manager: LspManager) -> ToolContext:
    return ToolContext(
        store=MemoryStore(":memory:"),
        read_root=(root,),
        lsp_manager=manager,
    )


@pytest.fixture()
def fake_python(monkeypatch: pytest.MonkeyPatch) -> None:
    command = f"{Path(sys.executable).as_posix()} {_FAKE.as_posix()}"
    monkeypatch.setenv("AXO_LSP_PYTHON", command)


def test_definition_references_symbols_and_diagnostics(tmp_path: Path, fake_python: None) -> None:
    (tmp_path / "a.py").write_text("foo()\n", encoding="utf-8")
    manager = LspManager()
    try:
        ctx = _ctx(tmp_path, manager)
        definition = LspTool().run(
            {"action": "definition", "path": "a.py", "line": 1, "symbol": "foo"}, ctx
        )
        assert definition.ok, definition.content
        assert "a.py:1:1" in definition.content

        references = LspTool().run(
            {"action": "references", "path": "a.py", "line": 1, "symbol": "foo"}, ctx
        )
        assert references.ok, references.content
        assert references.content.count("a.py:") == 2

        symbols = LspTool().run({"action": "symbols", "path": "a.py"}, ctx)
        assert symbols.ok
        assert "function fake_function" in symbols.content

        diagnostics = LspTool().run({"action": "diagnostics", "path": "a.py"}, ctx)
        assert diagnostics.ok
        assert "这是测试诊断" in diagnostics.content
        assert "警告" in diagnostics.content
    finally:
        manager.close_all()


def test_no_server_falls_back_with_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AXO_LSP_PYTHON", raising=False)
    monkeypatch.setattr(lsp_module, "_DEFAULT_SERVERS", {})
    (tmp_path / "a.py").write_text("foo()\n", encoding="utf-8")
    manager = LspManager()
    try:
        out = LspTool().run(
            {"action": "definition", "path": "a.py", "line": 1}, _ctx(tmp_path, manager)
        )
        assert not out.ok
        assert "没有可用的 LSP 服务器" in out.content
        assert "find_symbol" in out.content
    finally:
        manager.close_all()


def test_unknown_suffix_and_path_escape_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x\n", encoding="utf-8")
    manager = LspManager()
    try:
        ctx = _ctx(tmp_path, manager)
        out = LspTool().run({"action": "symbols", "path": "notes.txt"}, ctx)
        assert not out.ok and "回退" in out.content
        out = LspTool().run({"action": "symbols", "path": "../outside.py"}, ctx)
        assert not out.ok
    finally:
        manager.close_all()


def test_missing_line_for_definition_is_a_clear_failure(tmp_path: Path, fake_python: None) -> None:
    (tmp_path / "a.py").write_text("foo()\n", encoding="utf-8")
    manager = LspManager()
    try:
        out = LspTool().run({"action": "definition", "path": "a.py"}, _ctx(tmp_path, manager))
        assert not out.ok
        assert "需要 line" in out.content
    finally:
        manager.close_all()


def test_manager_reuses_one_client_and_closes_it(tmp_path: Path, fake_python: None) -> None:
    (tmp_path / "a.py").write_text("foo()\n", encoding="utf-8")
    manager = LspManager()
    try:
        first = manager.client_for(tmp_path, "python", "python")
        second = manager.client_for(tmp_path, "python", "python")
        assert first is second
    finally:
        manager.close_all()
    assert first._proc.poll() is not None


def test_client_only_reports_diagnostics_after_publish(tmp_path: Path, fake_python: None) -> None:
    (tmp_path / "a.py").write_text("foo()\n", encoding="utf-8")
    manager = LspManager()
    try:
        client = manager.client_for(tmp_path, "python", "python")
        diags = client.diagnostics(tmp_path / "a.py", timeout=1.0)
        assert diags is not None and diags[0]["message"] == "这是测试诊断"
    finally:
        manager.close_all()


def test_bad_server_command_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXO_LSP_PYTHON", "definitely-not-a-real-lsp-server --stdio")
    monkeypatch.setattr(lsp_module, "_DEFAULT_SERVERS", {})
    manager = LspManager()
    try:
        with pytest.raises(LspError) as exc:
            manager.client_for(tmp_path, "python", "python")
        assert "没有可用的 LSP 服务器" in str(exc.value)
    finally:
        manager.close_all()


def test_lsp_is_gated_and_summary_names_the_server() -> None:
    from axo.agent.gate import build_request
    from axo.llm.base import ToolCall

    assert LspTool.requires_approval is True
    request = build_request(
        ToolCall(
            id="c1",
            name="lsp",
            arguments={"action": "definition", "path": "a.py", "line": 1},
        ),
        tool=LspTool(),
    )
    assert "a.py" in request.summary
    assert "语言服务器" in request.summary
