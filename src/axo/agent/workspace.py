"""本轮工作区上下文：把仓库身份、分支、最近提交和 AGENTS 摘要压成 L1 的几行。

与 ``git_tools.repo_overview`` 的区别：这是**自动注入**的，不占一次工具往返；因此
必须短、稳定、可截断。它只读，不修改任何东西。

**不在这里读 README。** README 往往很长，而且多数轮次用不上；需要时模型可以
``read_file``。AGENTS.md 不同——它是仓库对 agent 的常驻约定，Codex 类工具会在会话
开始自动带上它。
"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .shell import _minimal_env

__all__ = ["build_workspace_context", "detect_test_command"]

_MAX_CHARS = 3000
_AGENTS_HEAD_LINES = 40
_NOISE_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "node_modules",
        "target",
        "out",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
    }
)


def _git(cwd: Path, args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            env=_minimal_env(),
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", "replace").strip()


def _clip(text: str, cap: int = _MAX_CHARS) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + "\n…（工作区上下文已截断）"


def detect_test_command(root: Path) -> str:
    """探测一条测试命令；只写事实，不执行。

    覆盖常用生态的项目级入口：Python、Node/Deno、Rust、Go、Maven/Gradle、.NET、
    Ruby、PHP、Swift、Elixir、Dart/Flutter、CMake/CTest、Make。**探测不到就返回空串**，
    由调用方要求模型显式给命令——不猜一个注定失败的默认值。
    """
    # Python 生态：先认最常见的组合，再看专用 runner。
    if (root / "pyproject.toml").is_file() and (root / "tests").is_dir():
        return f'"{sys.executable}" -m pytest -q'
    if (root / "tox.ini").is_file():
        return f'"{sys.executable}" -m tox'
    if (root / "noxfile.py").is_file():
        return f'"{sys.executable}" -m nox'
    if (root / "pytest.ini").is_file() or (root / "conftest.py").is_file():
        return f'"{sys.executable}" -m pytest -q'

    # Node / Deno。
    if (root / "deno.json").is_file() or (root / "deno.jsonc").is_file():
        return "deno test"
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if isinstance(scripts, dict) and scripts.get("test"):
            if (root / "pnpm-lock.yaml").is_file():
                return "pnpm test"
            if (root / "yarn.lock").is_file():
                return "yarn test"
            if (root / "bun.lockb").is_file() or (root / "bun.lock").is_file():
                return "bun run test"
            return "npm test"

    if (root / "Cargo.toml").is_file():
        return "cargo test"
    if (root / "go.mod").is_file():
        return "go test ./..."

    # JVM。
    if (root / "pom.xml").is_file():
        return "mvn -q test"
    if (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
        wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        if wrapper.is_file():
            return f'"{wrapper}" test'
        return "gradle test"

    # .NET。
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        return "dotnet test"

    # Ruby / PHP / Swift / Elixir / Dart。
    if (root / "Gemfile").is_file():
        return "bundle exec rspec" if (root / "spec").is_dir() else "bundle exec rake test"
    if (root / "Rakefile").is_file():
        return "rake test"
    if (root / "phpunit.xml").is_file() or (root / "phpunit.xml.dist").is_file():
        phpunit = root / "vendor" / "bin" / ("phpunit.bat" if os.name == "nt" else "phpunit")
        return f'"{phpunit}"' if phpunit.is_file() else "phpunit"
    if (root / "Package.swift").is_file():
        return "swift test"
    if (root / "mix.exs").is_file():
        return "mix test"
    if (root / "pubspec.yaml").is_file():
        try:
            manifest = (root / "pubspec.yaml").read_text(encoding="utf-8", errors="replace")
        except OSError:
            manifest = ""
        return (
            "flutter test" if "sdk: flutter" in manifest or "flutter:" in manifest else "dart test"
        )

    # 其它常见构建/测试入口。
    if (root / "build.sbt").is_file():
        return "sbt test"
    if (root / "stack.yaml").is_file():
        return "stack test"
    if any(root.glob("*.cabal")):
        return "cabal test"
    if (root / "deps.edn").is_file():
        return "clojure -X:test"
    if (root / "project.clj").is_file():
        return "lein test"
    if (root / "build.zig").is_file():
        return "zig build test"
    if (
        (root / "MODULE.bazel").is_file()
        or (root / "WORKSPACE").is_file()
        or (root / "WORKSPACE.bazel").is_file()
    ):
        return "bazel test //..."
    if (root / "meson.build").is_file():
        return (
            "meson test -C build"
            if (root / "build").is_dir()
            else "meson setup build && meson test -C build"
        )
    if (root / "tests" / "testthat").is_dir():
        return "Rscript -e \"testthat::test_dir('tests/testthat')\""
    if (root / "Makefile.PL").is_file() or (root / "Build.PL").is_file():
        return "prove -lr t"
    if any(root.glob("*.Tests.ps1")):
        return "pwsh -NoProfile -Command Invoke-Pester"

    # C/C++：先看已经配置好的 CTest，再看需要构建目录的 CMake。
    if (root / "CTestTestfile.cmake").is_file():
        return "ctest --output-on-failure"
    if (root / "CMakeLists.txt").is_file() and (root / "build").is_dir():
        return "cmake --build build && ctest --test-dir build --output-on-failure"

    if (root / "Makefile").is_file():
        return "make test（如果 Makefile 有 test 目标）"
    return ""


_TEST_GLOBS = (
    "test_*.py",
    "*_test.py",
    "*.test.js",
    "*.spec.js",
    "*.test.ts",
    "*.spec.ts",
    "*.test.tsx",
    "*.test.jsx",
    "*_test.go",
    "*Test.java",
    "*Tests.java",
    "*Tests.cs",
    "*_spec.rb",
    "*Test.php",
    "*Tests.swift",
    "*Test.kt",
    "*_test.dart",
    "*_test.exs",
    "*.Tests.ps1",
    "*.t",
)
_TEST_DIRS = ("tests", "test", "spec", "__tests__", "src/test", "t")
_TEST_SUFFIXES = (
    "*.py",
    "*.js",
    "*.jsx",
    "*.ts",
    "*.tsx",
    "*.go",
    "*.rs",
    "*.java",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.kt",
    "*.dart",
    "*.exs",
    "*.ps1",
    "*.pl",
    "*.pm",
    "*.vue",
    "*.svelte",
)


def _walk_matching(root: Path, pattern: str, *, limit: int = 2) -> list[Path]:
    """按文件名模式找文件，**剪掉噪声目录**。

    ``Path.rglob`` 会走进 ``node_modules``/``.venv`` 再过滤，在真实仓库上等于每轮
    白扫几万个文件。这里用 ``os.walk`` 在进入之前就剪枝，找到 ``limit`` 个就停。
    """
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _NOISE_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames):
            if fnmatch.fnmatch(name, pattern):
                matches.append(Path(dirpath) / name)
                if len(matches) >= limit:
                    return matches
    return matches


def _test_files(root: Path) -> list[str]:
    """列前 8 个测试文件。**多语言按文件名/目录约定**，不解析内容。"""
    files: list[Path] = []
    try:
        for pattern in _TEST_GLOBS:
            files.extend(_walk_matching(root, pattern, limit=2))
        for dirname in _TEST_DIRS:
            directory = root / dirname
            if not directory.is_dir():
                continue
            for pattern in _TEST_SUFFIXES:
                files.extend(_walk_matching(directory, pattern, limit=2))
    except OSError:
        return []
    relative = [path.relative_to(root).as_posix() for path in sorted(set(files))]
    return relative[:8]


def _one(root: Path) -> str:
    resolved = root.expanduser().resolve()
    inside = _git(resolved, ["rev-parse", "--show-toplevel"])
    repo = Path(inside).resolve() if inside else resolved
    if inside and not repo.is_relative_to(resolved):
        # 用户授权的是子目录：不把仓库根以外的内容带进上下文。
        repo = resolved
    lines = [f"工作区：{repo}"]
    branch = _git(repo, ["status", "--short", "--branch"])
    if branch:
        first = branch.splitlines()[0].lstrip("# ").strip()
        changes = [line for line in branch.splitlines()[1:] if line.strip()]
        lines.append(f"git：{first}" + (f"，{len(changes)} 个未提交改动" if changes else "，干净"))
        log = _git(repo, ["log", "-3", "--oneline"])
        if log:
            lines.append("最近提交：" + "；".join(log.splitlines()))
    else:
        lines.append("git：不是仓库（只有目录上下文）")
    try:
        tops = sorted(
            p.name
            for p in resolved.iterdir()
            if not p.name.startswith(".") and p.name not in _NOISE_DIRS and p.is_dir()
        )[:12]
    except OSError:
        tops = []
    if tops:
        lines.append("顶层目录：" + "、".join(tops))
    test_cmd = detect_test_command(resolved)
    if test_cmd:
        lines.append(f"测试命令（自动探测）：{test_cmd}")
    test_files = _test_files(resolved)
    if test_files:
        lines.append("测试文件：" + "、".join(test_files))
    agents = repo / "AGENTS.md"
    if not agents.is_file():
        agents = resolved / "AGENTS.md"
    if agents.is_file():
        try:
            text = agents.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if text:
            head = "\n".join(text.splitlines()[:_AGENTS_HEAD_LINES])
            lines.append(f"AGENTS.md（前 {_AGENTS_HEAD_LINES} 行）：\n{head}")
    return "\n".join(lines)


def build_workspace_context(roots: Sequence[Path]) -> str:
    """返回最多 3000 字符的工作区上下文；没有可读根/失败时返回空串。"""
    if not roots:
        return ""
    blocks = []
    for root in list(roots)[:2]:
        try:
            blocks.append(_one(Path(root)))
        except (OSError, ValueError):
            continue
    return _clip("\n\n".join(blocks))
