"""测试运行工具：自动探测项目测试命令，跑完把失败摘要回灌给模型。

它不替模型决定"要不要修"；它只把"跑了什么、退出码、失败在哪"变成一条结构化结果。
下一轮模型据此读文件、改、再调用——这就是测试失败→修复循环的承重件。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ..llm.base import ToolSpec
from .fs import ReadFileError, resolve_within
from .shell import DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, run_command
from .snapshot import snapshot_tree
from .tools import ToolContext, ToolOutcome, argument_int, argument_str
from .workspace import detect_test_command

__all__ = ["RunTestsTool"]


_TEST_LEVEL_MARKERS = (
    # pytest / unittest
    "FAILED ",
    "FAILED:",
    "ERROR ",
    "FAIL:",
    "test result: FAILED",
    # Go
    "--- FAIL:",
    "FAIL\t",
    "panic:",
    # Rust / Cargo
    " ... FAILED",
    "failures:",
    # Jest / Mocha / Vitest / Node TAP
    "● ",
    "✕ ",
    "✗ ",
    "✖ ",
    "× ",
    "not ok ",
    # JUnit / Gradle / Maven
    "[ERROR]",
    " FAILED",
    "FAILED",
    # GoogleTest / CTest / Catch2
    "[  FAILED  ]",
    "***Failed",
    # RSpec / minitest / PHPUnit / .NET / ExUnit / XCTest
    "Failures:",
    "Failure:",
    "rspec ./",
    "FAILURES!",
    "Failed ",
    "Failed: ",
    "Failed!",
    "[-] ",
    "1) test ",
    "** (",
    " failed",
)
"""测试级失败行的标记。**一个测试一行**，用来做跨次集合差。"""

_DETAIL_MARKERS = ("AssertionError", "E   ", "panic:", "Error:", "error:", "expected", "actual")
"""没有测试级标识时，退回断言行。它们只用来给人看摘要，不参与逐项对比。"""


def _failure_lines(output: str, limit: int = 40) -> tuple[str, ...]:
    """从输出里摘出可逐项对比的失败行。空输出返回空元组。

    **先取测试级标识，再退回断言行。** pytest 的 ``FAILED tests/x.py::test_y`` 一个
    测试一行，拿它做集合差才是"哪几个测试变了"；``E   ...`` 是同一个测试内部的断言
    细节，混进来会把一项失败数成三项。标记覆盖 pytest、Go、Rust、Jest、JUnit、
    GoogleTest、RSpec、PHPUnit、.NET、ExUnit 等常见输出。
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    picked = [line for line in lines if any(key in line for key in _TEST_LEVEL_MARKERS)]
    if not picked:
        picked = [line for line in lines if any(key in line for key in _DETAIL_MARKERS)]
    if not picked:
        picked = lines[-limit:]
    return tuple(picked[-limit:])


def _failure_excerpt(failures: Sequence[str], limit: int = 18) -> str:
    return "\n".join(failures[-limit:]).strip() or "（没有可摘出的失败行，见输出尾部）"


class RunTestsTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="run_tests",
        description=(
            "运行项目测试（自动探测 pytest/npm/pnpm/yarn/cargo/go/mvn/gradle/dotnet/"
            "rspec/phpunit/swift/mix/dart/ctest/make；可用 command 覆盖）。"
            "返回退出码与失败摘要。改完代码用它验证；失败时读摘要后继续修，不要只重跑。"
            "**会在本机执行命令，所以每次都要你确认。**"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "覆盖自动探测的测试命令。"},
                "cwd": {"type": "string", "description": "相对可写目录的工作目录。"},
                "timeout_s": {
                    "type": "integer",
                    "description": f"超时秒数，默认 {DEFAULT_TIMEOUT_S}，最多 {MAX_TIMEOUT_S}。",
                },
            },
        },
    )
    requires_approval: ClassVar[bool] = True
    parallel_safe: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if not ctx.allow_commands:
            return ToolOutcome.failure("命令执行没有打开（设置 → 工具）。")
        if not ctx.write_root:
            return ToolOutcome.failure("测试需要至少一个可写目录。")
        raw_cwd = argument_str(arguments, "cwd", required=False) or "."
        try:
            cwd = resolve_within(ctx.write_root, raw_cwd)
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if cwd.is_file():
            cwd = cwd.parent
        if not cwd.is_dir():
            return ToolOutcome.failure(f"{cwd} 不是目录。")
        command = argument_str(arguments, "command", required=False) or detect_test_command(cwd)
        if not command:
            return ToolOutcome.failure(
                "没有探测到测试命令。请显式给 command（例如 `python -m pytest -q`）。"
            )
        timeout_s = argument_int(
            arguments,
            "timeout_s",
            default=DEFAULT_TIMEOUT_S,
            minimum=1,
            maximum=MAX_TIMEOUT_S,
        )
        assert timeout_s is not None
        # 快照在**跑之前**取：它要回答的是"上次跑完到现在，用户/模型改过文件吗"，
        # 而不是这次测试自己写出的缓存/覆盖率文件。
        before = snapshot_tree(cwd) if ctx.test_session is not None else None
        result = run_command(command, cwd=cwd, timeout_s=timeout_s, cancel=ctx.cancel)
        head = f"测试命令：{command}\n退出码 {result.exit_code}，用时 {result.duration_ms} ms"
        if result.timed_out:
            head = f"**超时**（超过 {timeout_s} 秒），已终止。\n" + head
        elif result.cancelled:
            head = "**被用户停止**。\n" + head
        output = result.output.strip() or "（没有输出）"
        # 通过时不要从输出里“找失败”——.NET 的通过摘要也会写 Failed: 0。
        failures = () if result.ok else _failure_lines(output)
        summary = ""
        if ctx.test_session is not None and before is not None:
            summary = ctx.test_session.finish(
                command=command,
                cwd=cwd,
                ok=result.ok,
                exit_code=result.exit_code,
                failures=failures,
                before=before,
            )
        if result.ok:
            body = f"{head}\n\n通过。\n"
            if summary:
                body += f"\n{summary}\n"
            return ToolOutcome.success(f"{body}\n{output}", replayable=False)
        excerpt = _failure_excerpt(failures)
        hint = (
            "下一步：先读失败摘要指向的文件/断言，做最小修改；不要只重跑同一个命令。"
            "如果失败与你的改动无关，说明它在改动前是否已经存在。"
        )
        body = f"{head}\n\n失败摘要：\n{excerpt}\n\n"
        if summary:
            body += f"{summary}\n\n"
        body += f"{hint}\n\n--- 输出 ---\n{output}"
        return ToolOutcome.failure(body, replayable=False)
