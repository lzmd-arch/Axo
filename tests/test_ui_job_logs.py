"""后台作业日志轮询：只推增量、结束后只收一次、出错不炸界面。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from axo.ui.main_window import MainWindow


class FakeJobs:
    def __init__(self, statuses: dict[str, list[dict[str, Any]]], running: bool) -> None:
        self.statuses = statuses
        self.running = running
        self.calls: list[tuple[str, int]] = []
        self.pruned = 0

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": job_id,
                "command": "build",
                "running": self.running,
                "exit_code": 0 if not self.running else None,
                "elapsed_s": 1.0,
            }
            for job_id in self.statuses
        ]

    def status(self, job_id: str, *, cursor: int, chunk_bytes: int) -> dict[str, Any]:
        self.calls.append((job_id, cursor))
        queue = self.statuses[job_id]
        index = 0 if cursor == 0 else min(len(queue) - 1, cursor)
        return queue[index]

    def prune(self, *, keep_finished: int) -> int:
        self.pruned += 1
        return 0


class FakeView:
    def __init__(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.finishes: list[tuple[str, str]] = []

    def append_job_log(self, job_id: str, text: str) -> None:
        self.logs.append((job_id, text))

    def finish_job_log(self, job_id: str, status: str) -> None:
        self.finishes.append((job_id, status))


def _stub(jobs: FakeJobs) -> SimpleNamespace:
    return SimpleNamespace(
        _agent_jobs=jobs,
        _job_log_cursors={},
        _job_log_done=set(),
        chat_view=FakeView(),
    )


def _status(text: str, next_cursor: int, *, more: bool, running: bool, code: int | None) -> dict:
    return {
        "tail": text,
        "next_cursor": next_cursor,
        "has_more": more,
        "running": running,
        "exit_code": code,
    }


def test_running_job_is_read_incrementally() -> None:
    jobs = FakeJobs(
        {
            "j1": [
                _status("one", 3, more=True, running=True, code=None),
                _status("two", 6, more=False, running=True, code=None),
            ]
        },
        running=True,
    )
    stub = _stub(jobs)

    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]

    assert stub.chat_view.logs == [("j1", "one"), ("j1", "two")]
    assert stub._job_log_cursors["j1"] == 6
    assert stub._job_log_done == set()
    assert jobs.calls == [("j1", 0), ("j1", 3)]


def test_large_finished_job_waits_for_the_last_chunk() -> None:
    """作业已结束但日志还有一块没追完时，不能提前收场。

    一个 tick 最多追四块；第五块留到下一次，收场也等到那里。
    """
    jobs = FakeJobs(
        {
            "j1": [
                _status("c1", 1, more=True, running=False, code=0),
                _status("c2", 2, more=True, running=False, code=0),
                _status("c3", 3, more=True, running=False, code=0),
                _status("c4", 4, more=True, running=False, code=0),
                _status("c5", 5, more=False, running=False, code=0),
            ]
        },
        running=False,
    )
    stub = _stub(jobs)

    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]
    assert stub.chat_view.logs == [
        ("j1", "c1"),
        ("j1", "c2"),
        ("j1", "c3"),
        ("j1", "c4"),
    ]
    assert stub.chat_view.finishes == []

    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]
    assert stub.chat_view.logs[-1] == ("j1", "c5")
    assert stub.chat_view.finishes == [("j1", "已结束（退出码 0）")]


def test_finished_job_is_closed_once() -> None:
    jobs = FakeJobs(
        {"j1": [_status("done", 4, more=False, running=False, code=0)]},
        running=False,
    )
    stub = _stub(jobs)

    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]
    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]

    assert stub.chat_view.logs == [("j1", "done")]
    assert stub.chat_view.finishes == [("j1", "已结束（退出码 0）")]
    assert stub._job_log_done == {"j1"}
    assert jobs.calls == [("j1", 0)]


def test_status_error_does_not_break_the_timer() -> None:
    class Broken(FakeJobs):
        def status(self, job_id: str, *, cursor: int, chunk_bytes: int) -> dict[str, Any]:
            raise RuntimeError("gone")

    jobs = Broken({"j1": []}, running=True)
    stub = _stub(jobs)

    MainWindow._poll_job_logs(stub)  # type: ignore[arg-type]

    assert stub.chat_view.logs == []
    assert stub.chat_view.finishes == []
