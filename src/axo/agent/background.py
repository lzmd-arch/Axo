"""后台命令作业：启动后立刻返回 id，用状态工具查增量日志/退出码/停止。

这不是终端复用器：输出落到作业目录的一个文件，``command_status`` 读尾部。好处是
不阻塞对话线程，也不需要把工具调用改成流式事件。
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shell import _kill_tree, _minimal_env

__all__ = ["CommandJob", "CommandJobs", "JobError"]


class JobError(RuntimeError):
    pass


@dataclass
class CommandJob:
    job_id: str
    command: str
    cwd: Path
    log_path: Path
    process: subprocess.Popen[bytes]
    started: float

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.process.poll()


def _trim_incomplete_utf8(data: bytes) -> bytes:
    """去掉末尾不完整的 UTF-8 序列。

    游标是**字节**偏移，分块读时很容易把「中」的三字节切成两半。切在中间的那半截
    如果直接 ``decode(errors="replace")``，调用方会看到两个替换符，而下一块又从
    字符中间开始。这里把不完整的尾巴留给下一次读，``next_cursor`` 因此落在字符边界上。
    中间本来就坏的字节不动——那交给 ``replace``，不是这条的职责。
    """
    if not data:
        return data
    try:
        data.decode("utf-8")
        return data
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data":
            return data
    for cut in range(1, 4):
        if cut > len(data):
            break
        try:
            data[:-cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return data[:-cut]
    return data


class CommandJobs:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, CommandJob] = {}
        self._lock = threading.Lock()

    def start(self, command: str, cwd: Path) -> CommandJob:
        if not command.strip():
            raise JobError("命令是空的。")
        if not cwd.is_dir():
            raise JobError(f"{cwd} 不是目录。")
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        log_path = self.root / f"{job_id}.log"
        flags = 0
        extra: dict[str, Any] = {}
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            extra["creationflags"] = flags
        else:
            extra["start_new_session"] = True
        stream = log_path.open("wb")
        try:
            proc = subprocess.Popen(  # noqa: S602 — 审批在工具层
                command,
                shell=True,
                cwd=str(cwd),
                env=_minimal_env(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **extra,
            )
        except Exception:
            stream.close()
            raise
        job = CommandJob(job_id, command, cwd, log_path, proc, time.time())
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> CommandJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError(f"没有后台作业 {job_id}。")
        return job

    def status(
        self,
        job_id: str,
        *,
        cursor: int | None = None,
        tail_bytes: int = 16 * 1024,
        chunk_bytes: int = 64 * 1024,
    ) -> dict[str, Any]:
        """读作业状态。

        ``cursor=None`` 是旧行为：读日志最后 ``tail_bytes`` 字节。
        给了 ``cursor`` 就**从该字节偏移往后读一块**，返回 ``next_cursor`` 与
        ``has_more``；调用方拿 ``next_cursor`` 再问一次就能追上。游标是字节偏移而不是
        行号——日志是二进制追加的，按字节永远不会因为半行/编码错位而丢内容。
        """
        job = self.get(job_id)
        size = 0
        reset = False
        has_more = False
        next_cursor = 0
        try:
            with job.log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if cursor is None:
                    start = max(0, size - max(0, tail_bytes))
                    limit = size - start
                else:
                    start = max(0, int(cursor))
                    if start > size:
                        # 日志被换掉/截短了：从头再来，并明确告诉调用方。
                        start = 0
                        reset = True
                    limit = min(max(4, chunk_bytes), size - start)
                    has_more = start + limit < size
                f.seek(start)
                raw = f.read(limit)
            if cursor is not None and has_more:
                # 别把多字节字符切两半：半截留给下一次，next_cursor 落在字符边界。
                raw = _trim_incomplete_utf8(raw)
            tail = raw.decode("utf-8", "replace")
            next_cursor = start + len(raw)
        except OSError as exc:
            tail = f"（读不到日志：{exc}）"
            next_cursor = 0
        return {
            "id": job.job_id,
            "command": job.command,
            "cwd": str(job.cwd),
            "running": job.running,
            "exit_code": job.exit_code,
            "elapsed_s": round(time.time() - job.started, 1),
            "tail": tail,
            "next_cursor": next_cursor,
            "size": size,
            "has_more": has_more,
            "reset": reset,
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        """当前进程记得的作业，按启动时间排序。**不读日志正文。**"""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.started)
        return [
            {
                "id": job.job_id,
                "command": job.command,
                "running": job.running,
                "exit_code": job.exit_code,
                "elapsed_s": round(time.time() - job.started, 1),
            }
            for job in jobs
        ]

    def prune(self, *, keep_finished: int = 20) -> int:
        """丢掉旧作业的**内存记录**，保留最近 ``keep_finished`` 个已结束作业。

        运行中的作业永远保留；日志文件不删（出问题要复盘）。返回移除的记录数。
        被移除的作业再用 ``command_status`` 查会得到"没有这个作业"——这是有意的：
        记住无限多个 job_id 只会让一次会话越跑越重。
        """
        keep_finished = max(0, keep_finished)
        with self._lock:
            running = [job for job in self._jobs.values() if job.running]
            finished = sorted(
                (job for job in self._jobs.values() if not job.running),
                key=lambda job: job.started,
                reverse=True,
            )
            keep = {job.job_id for job in running}
            keep.update(job.job_id for job in finished[:keep_finished])
            removed = [job_id for job_id in self._jobs if job_id not in keep]
            for job_id in removed:
                del self._jobs[job_id]
        return len(removed)

    def stop(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.running:
            _kill_tree(job.process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                job.process.wait(timeout=5)
        return self.status(job_id)

    def stop_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.running:
                _kill_tree(job.process)
        with self._lock:
            self._jobs.clear()
