"""后台命令工具：启动、查状态、停止。启动/停止过闸门，查状态只读。"""

from __future__ import annotations

from typing import Any, ClassVar

from ..llm.base import ToolSpec
from .background import JobError
from .fs import ReadFileError, resolve_within
from .tools import ToolContext, ToolOutcome, argument_int, argument_str

__all__ = [
    "CommandListTool",
    "CommandStatusTool",
    "CommandStopTool",
    "StartCommandTool",
]


class StartCommandTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="start_command",
        description=(
            "在允许目录里**后台**启动一条命令，立刻返回 job_id；用 command_status 查增量日志，"
            "command_stop 停止。适合测试、构建等长任务，不阻塞对话。**会真正执行，需确认。**"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令。"},
                "cwd": {"type": "string", "description": "相对可写目录的工作目录。"},
            },
            "required": ["command"],
        },
    )
    requires_approval: ClassVar[bool] = True
    parallel_safe: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if not ctx.allow_commands:
            return ToolOutcome.failure("命令执行没有打开（设置 → 工具）。")
        if not ctx.write_root:
            return ToolOutcome.failure("后台命令需要至少一个可写目录。")
        if ctx.jobs is None:
            return ToolOutcome.failure("这一轮没有后台作业管理器。")
        command = argument_str(arguments, "command")
        try:
            cwd = resolve_within(
                ctx.write_root, argument_str(arguments, "cwd", required=False) or "."
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if cwd.is_file():
            cwd = cwd.parent
        try:
            job = ctx.jobs.start(command, cwd)
        except (OSError, JobError) as exc:
            return ToolOutcome.failure(f"启动失败：{exc}")
        return ToolOutcome.success(
            f"已启动后台作业 {job.job_id}（cwd {cwd}）。"
            "用 command_status 查日志/退出码；长任务不要用 run_command 阻塞等待。",
            replayable=False,
        )


class CommandStatusTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="command_status",
        description=(
            "查看后台作业的状态、退出码与输出。默认读日志尾部；给 cursor（上一次返回的 "
            "next_cursor）就只读新增部分，has_more=true 时用新的 next_cursor 继续追。"
            "长任务建议第一次 cursor=0，之后逐次追增量，避免重复读整段日志。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "start_command 返回的 id。"},
                "cursor": {
                    "type": "integer",
                    "description": "字节偏移；从 0 开始读全量，或用上次的 next_cursor 读增量。",
                },
            },
            "required": ["job_id"],
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if ctx.jobs is None:
            return ToolOutcome.failure("这一轮没有后台作业管理器。")
        job_id = argument_str(arguments, "job_id")
        cursor = argument_int(arguments, "cursor", default=None, minimum=0)
        try:
            status = ctx.jobs.status(job_id, cursor=cursor)
        except JobError as exc:
            return ToolOutcome.failure(str(exc))
        state = "运行中" if status["running"] else f"已结束（退出码 {status['exit_code']}）"
        lines = [
            f"作业 {job_id}：{state}，已运行 {status['elapsed_s']} s。",
            f"命令：{status['command']}",
            f"日志：已读到 {status['next_cursor']} / {status['size']} 字节"
            + ("（还有更多：用这个 next_cursor 再查）" if status["has_more"] else ""),
        ]
        if status.get("reset"):
            lines.append("（游标超过了日志长度，已从头读取；日志可能被换过。）")
        lines.append("--- 输出 ---")
        lines.append(status["tail"].strip() or "（还没有输出）")
        return ToolOutcome.success("\n".join(lines), replayable=False)


class CommandListTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="command_list",
        description=(
            "列出这次运行里启动过的后台作业（id、命令、是否还在跑、退出码）。"
            "忘记 job_id 时先调它；已结束的旧作业可能已被清理。"
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, _arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if ctx.jobs is None:
            return ToolOutcome.failure("这一轮没有后台作业管理器。")
        jobs = ctx.jobs.list_jobs()
        if not jobs:
            return ToolOutcome.success("没有后台作业。", replayable=False)
        lines = []
        for job in jobs:
            state = "运行中" if job["running"] else f"结束（{job['exit_code']}）"
            lines.append(f"{job['id']}  {state}  {job['elapsed_s']}s  {job['command']}")
        return ToolOutcome.success("\n".join(lines), replayable=False)


class CommandStopTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="command_stop",
        description="停止一个后台作业（杀进程树）。会改变机器状态，需要确认。",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string", "description": "要停止的作业 id。"}},
            "required": ["job_id"],
        },
    )
    requires_approval: ClassVar[bool] = True
    parallel_safe: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if ctx.jobs is None:
            return ToolOutcome.failure("这一轮没有后台作业管理器。")
        job_id = argument_str(arguments, "job_id")
        try:
            status = ctx.jobs.stop(job_id)
        except JobError as exc:
            return ToolOutcome.failure(str(exc))
        return ToolOutcome.success(
            f"已停止作业 {job_id}（退出码 {status['exit_code']}）。\n"
            f"--- 输出尾部 ---\n{status['tail'].strip() or '（没有输出）'}",
            replayable=False,
        )
