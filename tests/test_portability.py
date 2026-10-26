"""可移植性：运行代码不依赖本机的盘符路径，评测结果不带本机路径。"""

from __future__ import annotations

import ast
from pathlib import Path

_LOCAL_MARKERS = (
    "D:\\Projects",
    "D:/Projects",
    "/d/Projects",
    "BioSNN-MemoryTerminal",
    "D:\\Tools",
    "D:/Tools",
)


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_runtime_strings_do_not_hardcode_local_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for directory in (root / "src" / "axo", root / "tools"):
        for path in directory.rglob("*.py"):
            checked += 1
            for text in _string_constants(path):
                for marker in _LOCAL_MARKERS:
                    assert marker not in text, f"{path} 的字符串里硬编码了本机路径：{marker}"
    assert checked > 50


def test_test_command_uses_the_running_interpreter() -> None:
    import sys

    from axo.agent.workspace import detect_test_command

    root = Path(__file__).resolve().parents[1]
    command = detect_test_command(root)
    if command.startswith('"'):
        assert sys.executable in command
