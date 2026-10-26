"""编码评测任务本身要能跑通：隐藏测试不是摆设，修好之后必须真的变绿。"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "coding_eval_tasks", _ROOT / "tools/coding_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["coding_eval_tasks"] = module
    spec.loader.exec_module(module)
    return module


def _task(module, name: str):
    return next(task for task in (*module.TASKS, *module._extra_tasks()) if task.name == name)


def _prepare_with_hidden(module, task, root: Path, fix: dict[str, str]) -> tuple[bool, str]:
    module._prepare(task, root)
    for name, text in fix.items():
        (root / name).write_text(text, encoding="utf-8", newline="\n")
    for name, text in task.hidden.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return module._run_tests(root, task.test_command)


def test_multi_file_locate_hidden_test_turns_green(tmp_path: Path) -> None:
    module = _module()
    task = _task(module, "multi_file_locate")
    root = tmp_path / "repo"
    root.mkdir()

    before, _ = _prepare_with_hidden(module, task, root, {})
    assert before is False, "未修复时隐藏测试不该通过"

    fixed = tmp_path / "repo-fixed"
    fixed.mkdir()
    passed, output = _prepare_with_hidden(
        module,
        task,
        fixed,
        {"app/util/text.py": "def normalize(name):\n    return name.strip().lower()\n"},
    )
    assert passed, output


@pytest.mark.skipif(shutil.which("node") is None, reason="本机没有 node")
def test_js_calc_hidden_test_turns_green(tmp_path: Path) -> None:
    module = _module()
    task = _task(module, "js_calc")
    root = tmp_path / "repo"
    root.mkdir()

    before, _ = _prepare_with_hidden(module, task, root, {})
    assert before is False, "未修复时隐藏测试不该通过"

    fixed = tmp_path / "repo-fixed"
    fixed.mkdir()
    passed, output = _prepare_with_hidden(
        module,
        task,
        fixed,
        {"calc.js": "export function add(a, b) {\n  return a + b;\n}\n"},
    )
    assert passed, output


def test_extra_tasks_are_well_formed() -> None:
    module = _module()
    tasks = module._extra_tasks()
    names = [task.name for task in tasks]
    assert len(names) == len(set(names))
    for task in tasks:
        assert task.prompt.strip()
        for name in task.protected:
            assert name in task.files, f"{task.name}: 保护文件 {name} 不在初始仓库里"
        for name in task.hidden:
            assert name not in task.files, f"{task.name}: 隐藏测试 {name} 提前出现在仓库里"
