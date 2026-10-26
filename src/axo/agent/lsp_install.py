"""语言服务器自动安装器：检测、计划、执行、复核。

**默认只打印计划**。真正安装要显式 ``--yes``；它不会自动同意任何许可证、不会提权、
不会改系统 PATH。能用的安装通道只有：

* ``npm install -g``（Python/TS/JS/PHP/Shell/Vue/Svelte）
* ``rustup component add rust-analyzer``
* ``go install .../gopls``
* ``dotnet tool install csharp-ls``（用户目录 ``~/.dotnet/tools``）
* ``brew install``（macOS：LuaLS、Dart、LLVM/clangd、gopls）
* ``winget install --scope user``（Windows：LuaLS、Dart SDK、LLVM/clangd；可能触发 UAC）

装不上的语言会明确列出“需要什么运行时/手工配置”，不会假装成功。装完用
``python tools/language_doctor.py`` 复核；新开的 Axo 进程才会读到新 PATH。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .lsp import _launch_command, _resolve_executable

__all__ = ["Method", "Plan", "build_plans", "main"]


@dataclass(frozen=True)
class Method:
    kind: str
    packages: tuple[str, ...] = ()
    component: str = ""
    package: str = ""
    version: str = ""
    note: str = ""


@dataclass(frozen=True)
class Plan:
    language: str
    label: str
    provides: tuple[str, ...]
    methods: tuple[Method, ...]
    manual: str = ""


_PLANS: tuple[Plan, ...] = (
    Plan(
        "python",
        "Python",
        ("pyright-langserver",),
        (Method("npm", packages=("pyright",)),),
    ),
    Plan(
        "typescript",
        "TypeScript / JavaScript",
        ("typescript-language-server",),
        (Method("npm", packages=("typescript-language-server", "typescript")),),
    ),
    Plan("php", "PHP", ("intelephense",), (Method("npm", packages=("intelephense",)),)),
    Plan(
        "shell",
        "Shell",
        ("bash-language-server",),
        (Method("npm", packages=("bash-language-server",)),),
    ),
    Plan(
        "vue",
        "Vue",
        ("vue-language-server",),
        (Method("npm", packages=("@vue/language-server",)),),
    ),
    Plan(
        "svelte",
        "Svelte",
        ("svelteserver",),
        (Method("npm", packages=("svelte-language-server",)),),
    ),
    Plan(
        "rust",
        "Rust",
        ("rust-analyzer",),
        (Method("rustup", component="rust-analyzer"),),
    ),
    Plan(
        "go",
        "Go",
        ("gopls",),
        (
            Method("go", package="golang.org/x/tools/gopls@latest"),
            Method("brew", packages=("gopls",), note="没有 go 时的 macOS 安装通道"),
        ),
    ),
    Plan(
        "csharp",
        "C#",
        ("csharp-ls",),
        (Method("dotnet", package="csharp-ls", version="0.13.0"),),
        manual="先装 .NET SDK（macOS 可 brew install --cask dotnet-sdk），"
        "再 dotnet tool install --global csharp-ls。",
    ),
    Plan(
        "lua",
        "Lua",
        ("lua-language-server",),
        (
            Method("brew", packages=("lua-language-server",), note="macOS"),
            Method("winget", package="LuaLS.lua-language-server"),
        ),
    ),
    Plan(
        "dart",
        "Dart",
        ("dart",),
        (
            Method("brew", packages=("dart",), note="macOS"),
            Method("winget", package="Google.DartSDK"),
        ),
    ),
    Plan(
        "c",
        "C / C++",
        ("clangd",),
        (
            Method(
                "brew",
                packages=("llvm",),
                note="macOS：keg-only，clangd 在 opt/llvm/bin（已加入 LSP 搜索路径）",
            ),
            Method("winget", package="LLVM.LLVM", note="可能触发 UAC/管理员安装"),
        ),
    ),
    Plan(
        "java",
        "Java",
        ("jdtls",),
        (),
        manual="下载 jdtls 后设 AXO_JDTLS_HOME（便携目录需含 plugins/ 与 config_win/"
        "config_mac/config_linux 中对应的那个）。",
    ),
    Plan(
        "ruby",
        "Ruby",
        ("solargraph",),
        (),
        manual="装 Ruby 后 gem install solargraph；Windows 需要 MSYS2/Devkit。",
    ),
    Plan(
        "kotlin",
        "Kotlin",
        ("kotlin-language-server",),
        (),
        manual="Kotlin LS 需要 JDK 17/21；当前 Java 25 不兼容。",
    ),
    Plan("swift", "Swift", ("sourcekit-lsp",), (), manual="安装 Swift toolchain。"),
    Plan("scala", "Scala", ("metals",), (), manual="用 Coursier 安装 Metals。"),
    Plan("r", "R", ("R",), (), manual="安装 R 后 install.packages('languageserver')。"),
    Plan(
        "powershell",
        "PowerShell",
        ("PowerShellEditorServices",),
        (),
        manual="安装 PowerShellEditorServices 并配置其启动命令。",
    ),
    Plan("elixir", "Elixir", ("language_server.sh",), (), manual="安装 ElixirLS 与 Erlang。"),
    Plan(
        "haskell",
        "Haskell",
        ("haskell-language-server-wrapper",),
        (),
        manual="用 ghcup 安装 haskell-language-server。",
    ),
    Plan("clojure", "Clojure", ("clojure-lsp",), (), manual="下载 clojure-lsp 原生包。"),
)


def _prereq(kind: str) -> str:
    return {
        "npm": "npm",
        "rustup": "rustup",
        "go": "go",
        "dotnet": "dotnet",
        "brew": "Homebrew",
        "winget": "winget",
    }.get(kind, kind)


def _command(method: Method) -> list[str] | None:
    if method.kind == "npm":
        return _launch_command(("npm", "install", "-g", *method.packages))
    if method.kind == "rustup":
        rustup = _resolve_executable("rustup")
        return [rustup, "component", "add", method.component] if rustup else None
    if method.kind == "go":
        go = _resolve_executable("go")
        return [go, "install", method.package] if go else None
    if method.kind == "dotnet":
        root = os.environ.get("DOTNET_ROOT", "").strip()
        portable = Path(root) / ("dotnet.exe" if os.name == "nt" else "dotnet") if root else None
        dotnet = (
            str(portable)
            if portable is not None and portable.is_file()
            else _resolve_executable("dotnet")
        )
        if not dotnet:
            return None
        tool_dir = Path.home() / ".dotnet" / "tools"
        return [
            dotnet,
            "tool",
            "install",
            method.package,
            "--tool-path",
            str(tool_dir),
            "--version",
            method.version,
        ]
    if method.kind == "brew":
        brew = _resolve_executable("brew")
        return [brew, "install", *method.packages] if brew else None
    if method.kind == "winget":
        winget = _resolve_executable("winget")
        if not winget:
            return None
        return [
            winget,
            "install",
            "--id",
            method.package,
            "-e",
            "--scope",
            "user",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    return None


def _installed(plan: Plan) -> bool:
    return any(_resolve_executable(name) for name in plan.provides)


def build_plans(languages: list[str] | None = None) -> list[dict[str, object]]:
    wanted = set(languages or [plan.language for plan in _PLANS])
    rows: list[dict[str, object]] = []
    for plan in _PLANS:
        if plan.language not in wanted:
            continue
        installed = _installed(plan)
        chosen: Method | None = None
        command: list[str] | None = None
        reason = ""
        if not installed:
            for method in plan.methods:
                command = _command(method)
                if command is not None:
                    chosen = method
                    reason = ""
                    break
                reason = f"缺少 {_prereq(method.kind)}"
            if not plan.methods:
                reason = plan.manual or "没有自动安装通道"
        rows.append(
            {
                "language": plan.language,
                "label": plan.label,
                "installed": installed,
                "provides": list(plan.provides),
                "method": chosen.kind if chosen else "",
                "command": command or [],
                "manual": plan.manual,
                "reason": reason,
            }
        )
    return rows


def _run(command: list[str]) -> tuple[bool, str]:
    print("$ " + " ".join(command), flush=True)
    try:
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        return False, str(exc)
    output = (proc.stdout + proc.stderr).strip()
    if output:
        print("\n".join(output.splitlines()[-12:]), flush=True)
    return proc.returncode == 0, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装 Axo 可自动获取的语言服务器")
    parser.add_argument("--only", default="", help="逗号分隔的语言 id，例如 python,go,rust")
    parser.add_argument("--yes", action="store_true", help="真正执行；默认只打印计划")
    parser.add_argument("--json", action="store_true", help="输出计划 JSON")
    args = parser.parse_args(argv)
    languages = [part.strip() for part in args.only.split(",") if part.strip()] or None
    rows = build_plans(languages)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    print(f"{'语言':<14} {'状态':<12} {'动作'}")
    for row in rows:
        if row["installed"]:
            status = "已安装"
            action = ""
        elif row["command"]:
            status = "可自动安装"
            action = " ".join(row["command"])  # type: ignore[arg-type]
        else:
            status = "需手工"
            action = str(row["reason"] or row["manual"])
        print(f"{row['language']:<14} {status:<12} {action}")
    if not args.yes:
        print("\n只打印计划；确认后加 --yes 执行。")
        return 0
    failures: list[str] = []
    for row in rows:
        command = row["command"]
        if row["installed"] or not command:
            continue
        ok, _ = _run(command)  # type: ignore[arg-type]
        # 复核：有些安装器把 bin 放在新 PATH，当前进程未必看得到。
        if ok or _installed(next(plan for plan in _PLANS if plan.language == row["language"])):
            print(f"✓ {row['language']}")
        else:
            failures.append(str(row["language"]))
    if failures:
        print("未完成：" + "、".join(failures))
        print(
            "新开一个终端后运行 python tools/language_doctor.py 复核；必要时手工配置 AXO_LSP_BIN_DIRS。"
        )
        return 1
    print("安装完成。新开 Axo/终端后再运行 language_doctor 复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
