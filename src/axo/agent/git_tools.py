"""工作区感知与补丁工具：让 agent 能像编码代理一样先看仓库、再改、再验证。

只加两件 Codex 类代理最基础的东西：

* 只读的 ``git_status`` / ``git_diff`` / ``git_log`` / ``repo_overview``；
* 需要用户点头的 ``apply_patch``（统一 diff，多文件，先 ``git apply --check``）。

这一层是纯层：只用标准库与 ``axo.agent`` 自己的 ``fs``/``shell``/``tools``，
不 import Qt、httpx 或 UI。所有 git 调用都限定在用户授权的目录里；仓库根落在
授权目录之外时直接拒绝，避免从子目录逃到整个仓库。
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..llm.base import ToolSpec
from .fs import ReadFileError, resolve_within
from .shell import _minimal_env
from .tools import (
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    argument_int,
    argument_str,
    argument_text,
)

__all__ = [
    "ApplyPatchTool",
    "GitDiffTool",
    "GitLogTool",
    "GitStatusTool",
    "PreviewPatchTool",
    "RepoOverviewTool",
]

_GIT_TIMEOUT_S = 10
_OUTPUT_CAP = 32 * 1024
_PATCH_CAP = 256 * 1024


def _clip(text: str, cap: int = _OUTPUT_CAP) -> str:
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    omitted = len(text) - cap
    return text[:head] + f"\n\n[…中间省略 {omitted} 字符…]\n\n" + text[-tail:]


def _git(
    cwd: Path, args: Sequence[str], *, input_text: str | None = None, clip: bool = True
) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            input=input_text.encode("utf-8") if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=_GIT_TIMEOUT_S,
            env=_minimal_env(),
            check=False,
        )
    except FileNotFoundError:
        return None, "找不到 git 可执行文件。"
    except subprocess.TimeoutExpired:
        return None, f"git 命令超过 {_GIT_TIMEOUT_S} 秒，已终止。"
    out = proc.stdout.decode("utf-8", "replace")
    return proc.returncode, (_clip(out) if clip else out)


def _workspace(ctx: ToolContext, raw: str, *, write: bool) -> Path:
    roots = ctx.write_root if write else ctx.read_root
    if not roots:
        raise ReadFileError("没有配置%s目录。" % ("可写" if write else "可读"))
    path = resolve_within(roots, raw or ".")
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise ReadFileError(f"不是目录：{path}")
    return path


def _repo_root(ctx: ToolContext, raw: str, *, write: bool) -> Path:
    path = _workspace(ctx, raw, write=write)
    roots = ctx.write_root if write else ctx.read_root
    code, out = _git(path, ["rev-parse", "--show-toplevel"])
    if code != 0:
        raise ReadFileError(f"{path} 不是 git 仓库（或 git 不可用）：{out.strip()}")
    repo = Path(out.strip()).resolve()
    for root in roots:
        try:
            if repo.is_relative_to(Path(root).expanduser().resolve()):
                return repo
        except (OSError, ValueError):
            continue
    raise ReadFileError(f"git 仓库根 {repo} 不在授权目录里，拒绝操作。")


def _flag(arguments: dict[str, Any], key: str) -> bool:
    value = arguments.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ToolArgumentError(f"{key} 必须是布尔值。")
    return value


class GitStatusTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git_status",
        description=(
            "查看允许目录所在 git 仓库的当前状态（分支、修改、未跟踪文件）。"
            "开始一个编码任务时先看它，确认工作区是否干净。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "仓库内路径；默认第一个可读目录。"}
            },
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        try:
            repo = _repo_root(
                ctx, argument_str(arguments, "path", required=False) or ".", write=False
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        code, out = _git(repo, ["status", "--short", "--branch", "--untracked-files=normal"])
        if code != 0:
            return ToolOutcome.failure("git status 失败：\n" + out)
        body = out.strip() or "工作区干净。"
        return ToolOutcome.success(f"仓库 {repo}\n{body}", replayable=False)


class GitDiffTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git_diff",
        description=(
            "查看允许目录所在 git 仓库的未提交改动（unified diff）。"
            "改完文件后用它自查；staged=true 看已暂存部分，stat=true 只看统计。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "只显示这个路径的差异。"},
                "staged": {"type": "boolean", "description": "看已暂存（git diff --cached）。"},
                "stat": {"type": "boolean", "description": "只显示 --stat 统计。"},
            },
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw = argument_str(arguments, "path", required=False) or "."
        try:
            repo = _repo_root(ctx, raw, write=False)
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        args = ["diff", "--no-ext-diff", "--no-color"]
        if _flag(arguments, "staged"):
            args.append("--cached")
        if _flag(arguments, "stat"):
            args.append("--stat")
        path = argument_str(arguments, "path", required=False)
        if path:
            args += ["--", path]
        code, out = _git(repo, args)
        if code != 0:
            return ToolOutcome.failure("git diff 失败：\n" + out)
        return ToolOutcome.success(out.strip() or "没有差异。", replayable=False)


class GitLogTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="git_log",
        description="查看允许目录所在 git 仓库的最近提交（oneline），用于理解改动历史。",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "最多几条，1–20，默认 10。"},
                "path": {"type": "string", "description": "只看这个路径的历史。"},
            },
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        try:
            repo = _repo_root(
                ctx, argument_str(arguments, "path", required=False) or ".", write=False
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        limit = argument_int(arguments, "limit", default=10, minimum=1, maximum=20)
        assert limit is not None
        args = ["log", "-n", str(limit), "--oneline", "--decorate"]
        path = argument_str(arguments, "path", required=False)
        if path:
            args += ["--", path]
        code, out = _git(repo, args)
        if code != 0:
            return ToolOutcome.failure("git log 失败：\n" + out)
        return ToolOutcome.success(out.strip() or "没有提交。", replayable=False)


def _read_head(path: Path, limit: int = 120) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[:limit])


class RepoOverviewTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="repo_overview",
        description=(
            "一次看清 git 仓库：根目录、分支/工作区状态、最近提交、跟踪文件数、顶层目录、"
            "关键清单和 AGENTS/README 开头。开始编码任务时先调用它，比逐个 list/read 省轮次。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "仓库内路径；默认第一个可读目录。"}
            },
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        try:
            repo = _repo_root(
                ctx, argument_str(arguments, "path", required=False) or ".", write=False
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        code, status = _git(repo, ["status", "--short", "--branch"])
        if code != 0:
            return ToolOutcome.failure("git status 失败：\n" + status)
        _, log = _git(repo, ["log", "-n", "5", "--oneline", "--decorate"])
        _, files = _git(repo, ["ls-files"], clip=False)
        tracked = [line for line in files.splitlines() if line.strip()]
        tops: dict[str, int] = {}
        for line in tracked:
            top = line.split("/", 1)[0] if "/" in line else "(根文件)"
            tops[top] = tops.get(top, 0) + 1
        top_text = "\n".join(
            f"  {name} ({count})"
            for name, count in sorted(tops.items(), key=lambda x: (-x[1], x[0]))[:25]
        )
        key_names = [
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "requirements.txt",
            "Makefile",
        ]
        key_lines = [name for name in key_names if (repo / name).is_file()]
        key_text = "、".join(key_lines) or "（无）"
        body = (
            f"仓库根：{repo}\n\n## 状态\n{status.strip() or '干净'}"
            f"\n\n## 最近提交\n{log.strip() or '（无）'}"
            f"\n\n跟踪文件：{len(tracked)}"
            f"\n\n## 顶层\n{top_text or '（无）'}"
            f"\n\n## 关键文件\n{key_text}"
        )
        agents = _read_head(repo / "AGENTS.md") if (repo / "AGENTS.md").is_file() else ""
        readme = _read_head(repo / "README.md", 60) if (repo / "README.md").is_file() else ""
        if agents:
            body += "\n\n## AGENTS.md（前 120 行）\n" + agents
        if readme:
            body += "\n\n## README.md（前 60 行）\n" + readme
        return ToolOutcome.success(_clip(body), replayable=False)


def _patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith(("a/", "b/")):
            raw = raw[2:]
        if not raw or raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
            raise ToolArgumentError(f"补丁路径必须是仓库内相对路径：{raw!r}")
        if ".." in Path(raw).parts:
            raise ToolArgumentError(f"补丁路径不能包含 ..：{raw!r}")
        paths.append(raw)
    return paths


def _patch_sections(patch: str) -> list[str]:
    """把多文件补丁切成段。优先按 ``diff --git``，没有就按 ``--- ``。"""
    lines = patch.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts:
        starts = [index for index, line in enumerate(lines) if line.startswith("--- ")]
    if not starts:
        return [patch]
    sections: list[str] = []
    if starts[0] > 0:
        sections.append("".join(lines[: starts[0]]))
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        sections.append("".join(lines[start:end]))
    return [section for section in sections if section.strip()]


def _filter_patch(patch: str, selected: set[str]) -> tuple[str, set[str], set[str]]:
    """只保留选中文件的补丁段。

    返回 ``(过滤后的补丁, 实际保留的路径, 选中但补丁里没有的路径)``。一个文件都没
    选中时返回空补丁；调用方负责拒绝，而不是把空补丁交给 ``git apply``。
    """
    sections = _patch_sections(patch)
    kept: list[str] = []
    kept_paths: set[str] = set()
    all_paths: set[str] = set()
    for section in sections:
        paths = set(_patch_paths(section))
        all_paths |= paths
        if paths and paths <= selected:
            kept.append(section)
            kept_paths |= paths
    return "".join(kept), kept_paths, selected - all_paths


def _argument_files(arguments: dict[str, Any]) -> list[str] | None:
    raw = arguments.get("files")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise ToolArgumentError("files 必须是仓库内相对路径的字符串数组。")
    return [item.strip() for item in raw]


class PreviewPatchTool:
    """只检查/预览补丁，不写盘。**不过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="preview_patch",
        description=(
            "预览一个 unified diff：检查能否应用，返回 --stat 与补丁前几行。"
            "不改任何文件；确认后再调 apply_patch。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "unified diff 全文。"},
                "cwd": {"type": "string", "description": "仓库内路径；默认第一个可写目录。"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "只预览这些文件（可选）；省略=整个补丁。",
                },
            },
            "required": ["patch"],
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        patch = argument_text(arguments, "patch")
        if len(patch) > _PATCH_CAP:
            return ToolOutcome.failure(f"补丁太大（{len(patch)} 字节，上限 {_PATCH_CAP}）。")
        try:
            selected = _argument_files(arguments)
            paths = _patch_paths(patch)
        except ToolArgumentError as exc:
            return ToolOutcome.failure(str(exc))
        if not paths:
            return ToolOutcome.failure("补丁里没有找到 ---/+++ 文件路径。")
        if selected is not None:
            patch, kept, missing = _filter_patch(patch, set(selected))
            if missing:
                return ToolOutcome.failure(f"补丁里没有这些文件：{'、'.join(sorted(missing))}。")
            if not kept:
                return ToolOutcome.failure("选中的文件在补丁里没有可应用的段落。")
            paths = sorted(kept)
        try:
            repo = _repo_root(
                ctx, argument_str(arguments, "cwd", required=False) or ".", write=True
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        code, out = _git(repo, ["apply", "--check", "--whitespace=nowarn", "-"], input_text=patch)
        if code != 0:
            return ToolOutcome.failure("补丁无法应用：\n" + out)
        _, stat = _git(repo, ["apply", "--stat", "--whitespace=nowarn", "-"], input_text=patch)
        head = "\n".join(patch.splitlines()[:40])
        return ToolOutcome.success(
            f"补丁可以应用（{len(set(paths))} 个文件）：{'、'.join(sorted(set(paths)))}\n\n{stat.strip()}\n\n## 补丁前 40 行\n{head}",
            replayable=False,
        )


class ApplyPatchTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="apply_patch",
        description=(
            "把一个 unified diff（可多文件）应用到允许目录所在的 git 仓库。"
            "先 git apply --check 再应用；失败不改文件。适合多处、跨文件修改；"
            "单点小改仍可用 edit_file。应用后立刻用 git_diff 自查、run_command 跑测试。"
            "**会改磁盘，所以每次都要你确认。**"
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "unified diff 全文（---/+++/@@）。"},
                "cwd": {"type": "string", "description": "仓库内路径；默认第一个可写目录。"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "只应用这些文件（可选）；省略=整个补丁。用户在卡片上勾选的结果优先。",
                },
            },
            "required": ["patch"],
        },
    )
    requires_approval: ClassVar[bool] = True
    parallel_safe: ClassVar[bool] = False

    def approval_summary_for(self, call) -> str:
        """确认卡片上展示文件列表与补丁开头，而不是 80 字的截断参数。"""
        patch = call.arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return "应用补丁（补丁为空）"
        try:
            paths = _patch_paths(patch)
        except ToolArgumentError:
            return "应用补丁（补丁路径不合法；应用前还会再检查）"
        if not paths:
            return "应用补丁（没有找到文件路径）"
        requested = call.arguments.get("files")
        if isinstance(requested, list):
            allowed = {str(item) for item in requested if isinstance(item, str)}
            paths = [path for path in paths if path in allowed]
            if not paths:
                return "应用补丁（模型指定的 files 在补丁里没有匹配）"
        head = "\n".join(patch.splitlines()[:14])[:1200]
        names = "、".join(sorted(set(paths)))
        return f"应用补丁（{len(set(paths))} 个文件）：{names}\n\n{head}"

    def approval_files_for(self, call) -> tuple[str, ...]:
        """确认卡片按文件勾选用。路径不合法时不给列表（应用前还会再检查）。

        模型自己给了 ``files`` 时只列那几项：卡片不该让用户勾一个模型没打算碰的文件。
        """
        patch = call.arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ()
        try:
            paths = sorted(set(_patch_paths(patch)))
        except ToolArgumentError:
            return ()
        requested = call.arguments.get("files")
        if isinstance(requested, list):
            allowed = {str(item) for item in requested if isinstance(item, str)}
            paths = [path for path in paths if path in allowed]
        return tuple(paths)

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        patch = argument_text(arguments, "patch")
        if len(patch) > _PATCH_CAP:
            return ToolOutcome.failure(
                f"补丁太大（{len(patch)} 字节，上限 {_PATCH_CAP}）。请拆分。"
            )
        try:
            model_files = _argument_files(arguments)
            paths = _patch_paths(patch)
        except ToolArgumentError as exc:
            return ToolOutcome.failure(str(exc))
        if not paths:
            return ToolOutcome.failure("补丁里没有找到 ---/+++ 文件路径。")
        if model_files is not None:
            patch, kept, missing = _filter_patch(patch, set(model_files))
            if missing:
                return ToolOutcome.failure(f"补丁里没有这些文件：{'、'.join(sorted(missing))}。")
            if not kept:
                return ToolOutcome.failure("模型指定的文件在补丁里没有可应用的段落。")
            paths = sorted(kept)
        user_files = None
        if callable(ctx.approval_selection) and ctx.call_id:
            user_files = ctx.approval_selection(ctx.call_id)
        if user_files is not None:
            if not user_files:
                return ToolOutcome.failure("用户在确认卡片上没有选择任何文件，未改动。")
            patch, kept, missing = _filter_patch(patch, set(user_files))
            if missing:
                return ToolOutcome.failure(
                    f"用户选中的文件不在当前补丁里：{'、'.join(sorted(missing))}。"
                )
            if not kept:
                return ToolOutcome.failure("用户选中的文件在补丁里没有可应用的段落。")
            paths = sorted(kept)
        try:
            repo = _repo_root(
                ctx, argument_str(arguments, "cwd", required=False) or ".", write=True
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        code, out = _git(repo, ["apply", "--check", "--whitespace=nowarn", "-"], input_text=patch)
        if code != 0:
            return ToolOutcome.failure("补丁无法应用（未改动任何文件）：\n" + out)
        files = [repo / name for name in sorted(set(paths))]
        cid = ctx.checkpoints.capture(files, "apply_patch") if ctx.checkpoints else ""
        code, out = _git(repo, ["apply", "--whitespace=nowarn", "-"], input_text=patch)
        if code != 0:
            return ToolOutcome.failure("补丁检查通过但应用失败：\n" + out)
        names = "、".join(sorted(set(paths)))
        note = f"（检查点 {cid}）" if cid else ""
        return ToolOutcome.success(
            f"已应用补丁（{len(set(paths))} 个文件）：{names}{note}\n{out.strip()}",
            replayable=False,
        )
