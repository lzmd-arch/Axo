"""固定编码任务基准：通过率、轮数、工具调用与耗时。

任务集是**固定文件 + 固定测试**，不参与开发调试；每个任务在临时 git 仓库里从干净
提交开始，自动批准写/命令（临时沙箱）。测试文件被保护：改测试来"通过"记失败。

默认跑本机 Ollama 的 qwen3-4b。它是当前可用基线，不代表 Codex 水平；换更强模型时
只换 --model，任务与判据不变。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axo.agent.background import CommandJobs  # noqa: E402
from axo.agent.catalog import default_tools  # noqa: E402
from axo.agent.checkpoints import CheckpointStore  # noqa: E402
from axo.agent.gate import ApprovalDecision  # noqa: E402
from axo.agent.loop import LoopDone, LoopLimits, ToolFinished, run_agent_loop  # noqa: E402
from axo.agent.shell import _minimal_env  # noqa: E402
from axo.agent.tools import ToolContext, ToolRegistry  # noqa: E402
from axo.llm.base import ChatMessage  # noqa: E402
from axo.llm.ollama import OllamaChatBackend  # noqa: E402
from axo.llm.openai_compat import DEFAULT_OPENAI_BASE_URL, OpenAICompatChatBackend  # noqa: E402
from axo.session.budget import TokenBudget, TokenMeter, reserve_for  # noqa: E402
from memcore.store import MemoryStore  # noqa: E402

SYSTEM = """你是编码代理。任务开始先探索仓库（repo_overview / repo_map / git_status / read_file），
仓库大时先用 repo_map 看目录与符号、再决定读哪些文件。然后做最小修改；改完运行 run_tests
或 run_command。测试失败要读失败摘要、继续改，不要只解释、不要把测试改成能过。
最后用一句话总结改了什么、测试结果。"""


@dataclass(frozen=True)
class Task:
    name: str
    files: dict[str, str]
    prompt: str
    protected: tuple[str, ...] = ()
    hidden: dict[str, str] = field(default_factory=dict)
    test_command: str = ""
    """最终判据跑的测试命令。空串 = Python pytest（仓库自带 tests/ 时）。"""


TASKS: tuple[Task, ...] = (
    Task(
        name="bugfix_add",
        files={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        prompt="仓库里的 test_calc.py 失败。修复实现，让测试通过；不要修改测试文件。",
        protected=("test_calc.py",),
    ),
    Task(
        name="feature_slugify",
        files={
            "strings.py": 'def greet(name):\n    return f"Hello, {name}!"\n',
            "test_strings.py": (
                "from strings import slugify\n\n"
                "def test_slugify():\n    assert slugify('Hello World') == 'hello-world'\n"
                "    assert slugify('  A  B ') == 'a-b'\n"
            ),
        },
        prompt="test_strings.py 需要一个 slugify 函数。实现它并让测试通过；不要修改测试文件。",
        protected=("test_strings.py",),
    ),
    Task(
        name="bugfix_cli",
        files={
            "cli.py": (
                "def parse_port(text):\n"
                "    port = int(text)\n"
                "    if port < 1 or port > 80:\n"
                "        raise ValueError('port out of range')\n"
                "    return port\n"
            ),
            "test_cli.py": (
                "import pytest\nfrom cli import parse_port\n\n"
                "def test_valid():\n    assert parse_port('8080') == 8080\n\n"
                "def test_bad_range():\n    with pytest.raises(ValueError):\n        parse_port('70000')\n"
            ),
        },
        prompt="test_cli.py 失败：parse_port 的合法端口范围写错了。修复实现；不要修改测试文件。",
        protected=("test_cli.py",),
    ),
)


class _NoThink:
    """把 qwen3 的 think 关掉；云端后端不使用这个包装。"""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def stream(self, request):
        return self._inner.stream(replace(request, extra={**request.extra, "think": False}))


class _Counting:
    """包一层后端，累计 prompt/completion tokens；不改请求。"""

    def __init__(self, inner):
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def stream(self, request):
        for chunk in self._inner.stream(request):
            if chunk.prompt_tokens is not None:
                self.prompt_tokens += int(chunk.prompt_tokens)
            if chunk.completion_tokens is not None:
                self.completion_tokens += int(chunk.completion_tokens)
            yield chunk


def _extra_tasks() -> list[Task]:
    path = Path(__file__).resolve().parents[1] / "benchmarks/coding/tasks-extra.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(task: Task, root: Path) -> None:
    for name, text in task.files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "eval@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Eval"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "initial"], check=True)


def _run_tests(root: Path, command: str = "") -> tuple[bool, str]:
    # **不要继承整个环境**：DEEPSEEK_API_KEY 等凭据不该进入 agent 生成代码的子进程。
    env = _minimal_env()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    # 评测的临时仓库不需要宿主项目的 pytest 插件。CI 核心腿里 PySide6 没装，
    # 而 pytest-qt 被自动加载时会 ImportError: libEGL.so.1——那不是任务失败。
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    resolved = command or f'"{sys.executable}" -m pytest -q'
    proc = subprocess.run(
        resolved,
        shell=True,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    text = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, text[-4000:]


def run_task(task: Task, *, backend, keep: bool) -> dict[str, Any]:
    base = Path(tempfile.mkdtemp(prefix="axo-coding-"))
    root = base / "repo"
    root.mkdir()
    try:
        _prepare(task, root)
        protected_before = {name: _sha(root / name) for name in task.protected}
        if hasattr(backend, "reset"):
            backend.reset()
        budget = TokenBudget(window=backend.context_window, reserve_output=reserve_for(backend))
        ctx = ToolContext(
            store=MemoryStore(":memory:"),
            read_root=(root,),
            write_root=(root,),
            allow_commands=True,
            checkpoints=CheckpointStore(base / "checkpoints"),
            jobs=CommandJobs(base / "jobs"),
        )
        registry = ToolRegistry(
            default_tools(read_root=(root,), write_root=(root,), allow_commands=True)
        )
        messages = [
            ChatMessage(role="system", content=SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"{task.prompt}\n\n仓库根：{root}\n"
                    "先探索；改动后运行测试。最终回答要说明测试结果。"
                ),
            ),
        ]
        started = time.time()
        events = list(
            run_agent_loop(
                backend=backend,
                messages=messages,
                registry=registry,
                ctx=ctx,
                await_approval=lambda _req: ApprovalDecision.GRANT,
                budget=budget,
                meter=TokenMeter(),
                limits=LoopLimits(fuse_rounds=12, no_progress_rounds=3),
            )
        )
        elapsed = round(time.time() - started, 2)
        done = next((e for e in events if isinstance(e, LoopDone)), None)
        tool_events = [e for e in events if isinstance(e, ToolFinished)]
        tools = [e.name for e in tool_events]
        # 只留名字/成败；工具正文可能含临时绝对路径，不进结果文件。
        tool_outcomes = [{"name": e.name, "ok": e.ok} for e in tool_events]
        hidden_paths = []
        for name, text in task.hidden.items():
            hp = root / name
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(text, encoding="utf-8", newline="\n")
            hidden_paths.append(hp)
        try:
            passed, test_output = _run_tests(root, task.test_command)
        finally:
            for hp in hidden_paths:
                hp.unlink(missing_ok=True)
        protected_after = {name: _sha(root / name) for name in task.protected}
        protected_ok = protected_before == protected_after
        # 结果文件可能被提交/公开：把临时绝对路径换成占位符，别让本机用户名进仓库。
        test_output = test_output.replace(str(root), "<repo>").replace(str(base), "<tmp>")
        final_text = (
            (done.final_text[-1200:] if done else "")
            .replace(str(root), "<repo>")
            .replace(str(base), "<tmp>")
        )
        return {
            "task": task.name,
            "passed": bool(passed and protected_ok),
            "tests_passed": passed,
            "hidden_tests_passed": passed,
            "protected_ok": protected_ok,
            "rounds": done.rounds if done else None,
            "tool_calls": len(tools),
            "tools": tools,
            "tool_outcomes": tool_outcomes,
            "elapsed_s": elapsed,
            "prompt_tokens": getattr(backend, "prompt_tokens", None),
            "completion_tokens": getattr(backend, "completion_tokens", None),
            "stop_reason": done.stop_reason if done else "exception",
            "final_text": final_text,
            "test_output": test_output,
            "repo": "<repo>" if keep else "",
        }
    except Exception as exc:  # noqa: BLE001 — 一条任务失败不应终止整轮评测
        detail = f"{type(exc).__name__}: {exc}".replace(str(base), "<tmp>")
        return {"task": task.name, "passed": False, "error": detail}
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai", "axo-settings"], default="ollama")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--task", default=None)
    ap.add_argument("--keep", action="store_true", help="保留临时仓库用于排查")
    ap.add_argument("--out", default=str(ROOT / "benchmarks/coding/results.json"))
    a = ap.parse_args()
    model = a.model
    if a.backend == "ollama":
        backend = _NoThink(OllamaChatBackend(model=model))
    elif a.backend == "openai":
        key = os.environ.get(a.api_key_env, "")
        if not key:
            raise SystemExit(f"缺少 API key 环境变量 {a.api_key_env}")
        backend = OpenAICompatChatBackend(
            model=model, base_url=a.base_url or DEFAULT_OPENAI_BASE_URL, api_key=key
        )
    else:
        # 用 Axo 里已经配置的云端 key（DPAPI）；**只在进程内交给后端，不打印**。
        # 需要 PySide6；模型名不持久化，默认按 DeepSeek 聊天模型。
        from PySide6.QtCore import QSettings

        from axo.ui.settings_store import SETTINGS_APP, SETTINGS_ORG, SettingsStore
        from axo.ui.theme_qt import QSettingsBackend

        configured = SettingsStore(
            QSettingsBackend(QSettings(SETTINGS_ORG, SETTINGS_APP))
        ).load_backend()
        if not configured.is_cloud or not configured.api_key:
            raise SystemExit("Axo 设置里没有配置云端 DeepSeek key。")
        if model == "qwen3-4b":
            model = "deepseek-chat"
        backend = OpenAICompatChatBackend(
            model=model, base_url=configured.cloud_base_url, api_key=configured.api_key
        )
    backend = _Counting(backend)
    tasks = [t for t in (*TASKS, *_extra_tasks()) if a.task is None or t.name == a.task]
    results = [run_task(t, backend=backend, keep=a.keep) for t in tasks]
    for r in results:
        print(
            f"{r['task']:<16} passed={r.get('passed')} tools={r.get('tool_calls')} "
            f"rounds={r.get('rounds')} elapsed={r.get('elapsed_s')}s"
        )
    payload = {
        "backend": a.backend,
        "model": model,
        "tasks": results,
        "passed": sum(1 for r in results if r.get("passed")),
        "total": len(results),
        "privacy": "结果不含 API key；临时绝对路径已去除/替换。",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"passed {payload['passed']}/{len(results)}; wrote {a.out}")


if __name__ == "__main__":
    main()
