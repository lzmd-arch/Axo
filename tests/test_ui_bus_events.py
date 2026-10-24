"""UI 侧的事件接收与子进程启停。

这两块是"单向、不依赖"的可执行形式：轮询器只认文件，启动器只给句柄。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from axo.busd.events import append_event
from axo.busd.launcher import launch_blocked_reason, spawn_daemon, stop_daemon
from axo.ui.bus_events import BusEventPoller

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_poller_emits_each_event_once(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    append_event(events, "daemon_start", {}, at=1.0)
    poller = BusEventPoller(events_path=events, cursor_path=tmp_path / "cursor.json", poll_ms=60000)
    seen: list[Any] = []
    poller.event.connect(seen.append)

    assert poller.poll() == 1
    assert [e.kind for e in seen] == ["daemon_start"]
    assert poller.poll() == 0, "同一条事件不该在第二次轮询里重放"

    append_event(events, "greeting_suggestion", {"novelty": 0.9}, at=2.0)
    assert poller.poll() == 1
    assert [e.kind for e in seen] == ["daemon_start", "greeting_suggestion"]


def test_cursor_survives_a_restart(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    cursor = tmp_path / "cursor.json"
    append_event(events, "daemon_start", {}, at=1.0)

    first = BusEventPoller(events_path=events, cursor_path=cursor)
    seen: list[Any] = []
    first.event.connect(seen.append)
    first.poll()

    second = BusEventPoller(events_path=events, cursor_path=cursor)
    seen2: list[Any] = []
    second.event.connect(seen2.append)
    assert second.poll() == 0, "重启后不该把旧建议再说一遍"
    assert seen2 == []


def test_missing_file_and_bad_cursor_are_quiet(tmp_path: Path) -> None:
    poller = BusEventPoller(
        events_path=tmp_path / "nope.jsonl", cursor_path=tmp_path / "cursor.json"
    )
    assert poller.poll() == 0

    (tmp_path / "cursor.json").write_text("{半截", encoding="utf-8")
    again = BusEventPoller(
        events_path=tmp_path / "nope.jsonl", cursor_path=tmp_path / "cursor.json"
    )
    assert again.line == 0


def test_frozen_build_refuses_to_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert launch_blocked_reason() is not None
    assert spawn_daemon(db_path=tmp_path / "m.db", data_dir=tmp_path) is None


def test_stop_is_idempotent_on_nothing() -> None:
    stop_daemon(None)  # 不该抛
