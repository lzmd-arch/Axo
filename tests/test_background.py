"""后台命令：启动立刻返回、状态读增量日志、停止杀进程树。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from axo.agent.background import CommandJobs
from axo.agent.background_tools import (
    CommandListTool,
    CommandStatusTool,
    CommandStopTool,
    StartCommandTool,
)
from axo.agent.catalog import default_tools
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def _wait(job_id: str, jobs: CommandJobs, *, timeout: float = 5.0):
    deadline = time.time() + timeout
    status = jobs.status(job_id)
    while status["running"] and time.time() < deadline:
        time.sleep(0.05)
        status = jobs.status(job_id)
    return status


def test_background_process_and_tail(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    script = tmp_path / "job.py"
    script.write_text(
        "import time; print('one', flush=True); time.sleep(0.2); print('two', flush=True)",
        encoding="utf-8",
    )
    job = jobs.start(f'"{sys.executable}" "{script.name}"', tmp_path)
    first = jobs.status(job.job_id)
    assert first["running"] or first["exit_code"] == 0
    done = _wait(job.job_id, jobs)
    assert done["exit_code"] == 0 and "two" in done["tail"]
    assert (tmp_path / "jobs" / (job.job_id + ".log")).is_file()


def test_stop_kills_process(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    job = jobs.start(f'"{sys.executable}" -c "import time; time.sleep(5)"', tmp_path)
    assert jobs.status(job.job_id)["running"]
    stopped = jobs.stop(job.job_id)
    assert stopped["exit_code"] is not None


def test_tools_start_status_stop(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        write_root=(tmp_path,),
        allow_commands=True,
        jobs=jobs,
    )
    started = StartCommandTool().run({"command": f'"{sys.executable}" -c "print(123)"'}, ctx)
    assert started.ok and "job_id" not in started.content  # 只回 id，不回字段名
    job_id = started.content.split("作业 ")[1].split("（")[0]
    status = CommandStatusTool().run({"job_id": job_id}, ctx)
    assert status.ok and "123" in status.content
    stopped = CommandStopTool().run({"job_id": job_id}, ctx)
    assert stopped.ok


def test_background_tools_registered_only_with_commands(tmp_path: Path) -> None:
    without = {
        t.spec.name
        for t in default_tools(read_root=(tmp_path,), write_root=(tmp_path,), allow_commands=False)
    }
    with_commands = {
        t.spec.name
        for t in default_tools(read_root=(tmp_path,), write_root=(tmp_path,), allow_commands=True)
    }
    assert "start_command" not in without
    assert {"start_command", "command_status", "command_stop", "command_list"} <= with_commands


def test_incremental_cursor_reads_only_new_bytes(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    job = jobs.start(f'"{sys.executable}" -c "print(\'abcdef\')"', tmp_path)
    _wait(job.job_id, jobs)

    first = jobs.status(job.job_id, cursor=0, chunk_bytes=4)
    assert first["tail"] == "abcd"
    assert first["has_more"] is True
    assert first["next_cursor"] == 4

    second = jobs.status(job.job_id, cursor=first["next_cursor"], chunk_bytes=64)
    assert "ef" in second["tail"]
    assert "abcd" not in second["tail"]
    assert second["has_more"] is False
    assert second["next_cursor"] == second["size"]


def test_cursor_never_splits_multibyte_characters(tmp_path: Path) -> None:
    """字节游标切在「中」中间时，半截要留到下一次读，不能变成两个替换符。"""
    script = tmp_path / "cn.py"
    script.write_text("print('中文')\n", encoding="utf-8")
    jobs = CommandJobs(tmp_path / "jobs")
    job = jobs.start(f'"{sys.executable}" "{script.name}"', tmp_path)
    _wait(job.job_id, jobs)

    first = jobs.status(job.job_id, cursor=0, chunk_bytes=4)
    assert "�" not in first["tail"]
    assert first["tail"] == "中"

    chunks = [first["tail"]]
    cursor = first["next_cursor"]
    while True:
        status = jobs.status(job.job_id, cursor=cursor, chunk_bytes=4)
        assert "�" not in status["tail"]
        chunks.append(status["tail"])
        if not status["has_more"]:
            break
        cursor = status["next_cursor"]
    assert "".join(chunks).replace("\r\n", "\n") == "中文\n"


def test_cursor_past_log_end_resets_explicitly(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    job = jobs.start(f'"{sys.executable}" -c "print(123)"', tmp_path)
    _wait(job.job_id, jobs)

    status = jobs.status(job.job_id, cursor=9999)
    assert status["reset"] is True
    assert "123" in status["tail"]


def test_list_jobs_and_prune_keeps_running(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    finished = jobs.start(f'"{sys.executable}" -c "print(1)"', tmp_path)
    running = jobs.start(f'"{sys.executable}" -c "import time; time.sleep(5)"', tmp_path)
    _wait(finished.job_id, jobs)

    listed = {item["id"] for item in jobs.list_jobs()}
    assert {finished.job_id, running.job_id} <= listed

    removed = jobs.prune(keep_finished=0)
    assert removed == 1
    remaining = {item["id"] for item in jobs.list_jobs()}
    assert running.job_id in remaining
    assert finished.job_id not in remaining
    jobs.stop(running.job_id)


def test_command_list_tool_reports_jobs(tmp_path: Path) -> None:
    jobs = CommandJobs(tmp_path / "jobs")
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        write_root=(tmp_path,),
        allow_commands=True,
        jobs=jobs,
    )
    job = jobs.start(f'"{sys.executable}" -c "print(1)"', tmp_path)
    _wait(job.job_id, jobs)

    outcome = CommandListTool().run({}, ctx)
    assert outcome.ok
    assert job.job_id in outcome.content
    assert "结束" in outcome.content
