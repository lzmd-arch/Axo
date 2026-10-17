"""agent 执行轨迹：按天 JSONL、参数脱敏、**坏了不影响主流程**。"""

from __future__ import annotations

import json
from pathlib import Path

from axo.agent.runlog import RunLog, redact_arguments


def test_a_record_lands_in_the_daily_file(tmp_path: Path) -> None:
    log = RunLog(tmp_path)

    log.write({"kind": "agent_turn", "rounds": 2}, now=1_760_000_000.0)

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    line = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert line["rounds"] == 2
    assert "ts" in line


def test_a_broken_directory_does_not_raise(tmp_path: Path) -> None:
    """日志坏了不该反过来弄坏一次正常的回答。"""
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    log = RunLog(blocker / "sub")

    log.write({"kind": "x"})  # 不抛


def test_a_disabled_log_writes_nothing(tmp_path: Path) -> None:
    RunLog(tmp_path, enabled=False).write({"kind": "x"})

    assert list(tmp_path.glob("*.jsonl")) == []


def test_big_content_is_replaced_by_length_and_hash() -> None:
    cleaned = redact_arguments("write_file", {"path": "a.txt", "content": "秘" * 1000})

    assert cleaned["path"] == "a.txt"
    assert cleaned["content"]["chars"] == 1000
    assert len(cleaned["content"]["sha1"]) == 16
    assert len(cleaned["content"]["preview"]) == 200
    assert "秘" * 300 not in json.dumps(cleaned, ensure_ascii=False)


def test_url_query_is_dropped() -> None:
    cleaned = redact_arguments("fetch_url", {"url": "https://example.com/a?token=secret"})

    assert cleaned["url"] == "https://example.com/a"
