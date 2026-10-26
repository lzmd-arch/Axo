"""跨语言符号搜索：名字要能定位到文件/行，大小写不敏感，结果有上限。"""

from __future__ import annotations

from pathlib import Path

from axo.agent.catalog import default_tools
from axo.agent.repo_map import FindSymbolTool, iter_symbols
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def _ctx(root: Path) -> ToolContext:
    return ToolContext(store=MemoryStore(":memory:"), read_root=(root,))


def test_finds_symbols_across_languages(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def normalize_name():\n    pass\n", encoding="utf-8")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "app.js").write_text(
        "export function normalizeName() {}\n", encoding="utf-8"
    )
    (tmp_path / "main.go").write_text("package main\nfunc NormalizeName() {}\n", encoding="utf-8")

    outcome = FindSymbolTool().run({"name": "normalize"}, _ctx(tmp_path))
    assert outcome.ok
    assert "pkg/a.py:1" in outcome.content
    assert "web/app.js:1" in outcome.content
    assert "main.go:2" in outcome.content
    assert "找到 3 个符号" in outcome.content
    assert not outcome.replayable


def test_symbol_search_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def NormalizeName():\n    pass\n", encoding="utf-8")
    out = FindSymbolTool().run({"name": "normalizename"}, _ctx(tmp_path))
    assert out.ok and "a.py:1" in out.content


def test_symbol_search_caps_results_and_says_so(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"m{index}.py").write_text(
            f"def target_{index}():\n    pass\n", encoding="utf-8"
        )
    out = FindSymbolTool().run({"name": "target", "max_results": 2}, _ctx(tmp_path))
    assert out.ok
    assert "找到 5 个符号" in out.content
    assert "还有 3 个" in out.content


def test_no_match_is_a_successful_empty_answer(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def other():\n    pass\n", encoding="utf-8")
    out = FindSymbolTool().run({"name": "missing"}, _ctx(tmp_path))
    assert out.ok
    assert "没有找到" in out.content


def test_iter_symbols_reports_line_numbers() -> None:
    # 纯函数的形状由 find_symbol 直接消费；这里单独钉返回结构。
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.py").write_text("\n\ndef later():\n    pass\n", encoding="utf-8")
        hits, truncated = iter_symbols(root)
        assert ("a.py", 3, "def later()") in hits
        assert truncated is False


def test_registered_with_read_root_only(tmp_path: Path) -> None:
    assert "find_symbol" not in {tool.spec.name for tool in default_tools()}
    assert "find_symbol" in {tool.spec.name for tool in default_tools(read_root=(tmp_path,))}
