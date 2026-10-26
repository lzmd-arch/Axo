"""测试会话记忆：失败变化要说清楚，原地重跑要拦住。"""

from __future__ import annotations

import sys
from pathlib import Path

from axo.agent.test_session import Snapshot, TestSession, snapshot_tree
from axo.agent.test_tools import RunTestsTool
from axo.agent.tools import ToolContext
from memcore.store import MemoryStore


def _ctx(root: Path, session: TestSession | None = None) -> ToolContext:
    return ToolContext(
        store=MemoryStore(":memory:"),
        write_root=(root,),
        allow_commands=True,
        test_session=session,
    )


def test_snapshot_ignores_caches_and_detects_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 1\n", encoding="utf-8")
    first = snapshot_tree(tmp_path)
    assert first.digest is not None and first.files == 1

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.pyc").write_bytes(b"noise")
    assert snapshot_tree(tmp_path).digest == first.digest

    source.write_text("x = 2\n", encoding="utf-8")
    assert snapshot_tree(tmp_path).digest != first.digest


def test_snapshot_reports_truncation(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"f{index}.py").write_text("x = 1\n", encoding="utf-8")
    snap = snapshot_tree(tmp_path, max_files=2)
    assert snap.digest is not None
    assert snap.files == 2
    assert snap.truncated


def test_first_run_has_no_comparison(tmp_path: Path) -> None:
    session = TestSession()
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a",),
        before=Snapshot("d1", 1),
    )
    assert "第 1 次" in text
    assert "没有上次" in text


def test_brief_is_empty_before_any_run(tmp_path: Path) -> None:
    assert TestSession().brief() == ""


def test_brief_summarizes_last_run_for_l1(tmp_path: Path) -> None:
    session = TestSession()
    session.finish(
        command="python -m pytest -q",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED tests/test_a.py::test_a", "FAILED tests/test_b.py::test_b"),
        before=Snapshot("d1", 2),
    )
    brief = session.brief()
    assert "上次测试" in brief
    assert "第 1 次" in brief
    assert "python -m pytest -q" in brief
    assert "FAILED tests/test_a.py::test_a" in brief
    assert len(brief) <= 500


def test_history_is_capped_but_index_keeps_counting(tmp_path: Path) -> None:
    session = TestSession()
    snap = Snapshot("same", 1)
    for _ in range(51):
        session.finish(
            command="pytest",
            cwd=tmp_path,
            ok=True,
            exit_code=0,
            failures=(),
            before=snap,
        )
    assert len(session.runs) == 50
    assert session.runs[-1].index == 51
    assert "第 51 次" in session.brief()


def test_identical_failure_without_changes_is_called_out(tmp_path: Path) -> None:
    session = TestSession()
    snap = Snapshot("same", 1)
    session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a",),
        before=snap,
    )
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a",),
        before=snap,
    )
    assert "仍失败 1 项" in text
    assert "与上次测试前相同" in text
    assert "原地重跑" in text


def test_failure_diff_reports_new_and_fixed(tmp_path: Path) -> None:
    session = TestSession()
    session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a", "FAILED b"),
        before=Snapshot("d1", 2),
    )
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED b", "FAILED c"),
        before=Snapshot("d2", 2),
    )
    assert "仍失败 1 项" in text and "FAILED b" in text
    assert "新增失败 1 项" in text and "FAILED c" in text
    assert "已消失 1 项" in text and "FAILED a" in text
    assert "与上次测试前不同" in text
    assert "原地重跑" not in text


def test_all_failures_disappear(tmp_path: Path) -> None:
    session = TestSession()
    session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a", "FAILED b"),
        before=Snapshot("d1", 2),
    )
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=True,
        exit_code=0,
        failures=(),
        before=Snapshot("d2", 2),
    )
    assert "都消失了" in text
    assert "与上次测试前不同" in text


def test_empty_failures_do_not_claim_identical(tmp_path: Path) -> None:
    session = TestSession()
    snap = Snapshot("same", 1)
    session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=None,
        failures=(),
        before=snap,
    )
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=None,
        failures=(),
        before=snap,
    )
    assert "没有摘出失败行" in text
    assert "失败与上次完全相同" not in text
    assert "原地重跑" not in text


def test_truncated_snapshot_never_claims_unchanged(tmp_path: Path) -> None:
    session = TestSession()
    snap = Snapshot("same", 4000, True)
    session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a",),
        before=snap,
    )
    text = session.finish(
        command="pytest",
        cwd=tmp_path,
        ok=False,
        exit_code=1,
        failures=("FAILED a",),
        before=snap,
    )
    assert "不能据此断言" in text
    assert "原地重跑" not in text


def _state_script(tmp_path: Path) -> None:
    (tmp_path / "run.py").write_text(
        "from pathlib import Path\n"
        "state = Path('state.txt').read_text(encoding='utf-8').strip()\n"
        "if state == 'a':\n"
        "    print('FAILED tests/test_a.py::test_a')\n"
        "    raise SystemExit(1)\n"
        "if state == 'b':\n"
        "    print('FAILED tests/test_b.py::test_b')\n"
        "    raise SystemExit(1)\n"
        "print('all passed')\n",
        encoding="utf-8",
    )


def _run(root: Path, session: TestSession):
    return RunTestsTool().run(
        {"command": f'"{sys.executable}" run.py', "timeout_s": 20}, _ctx(root, session)
    )


def test_run_tests_reports_cross_run_diff(tmp_path: Path) -> None:
    _state_script(tmp_path)
    session = TestSession()
    (tmp_path / "state.txt").write_text("a\n", encoding="utf-8")
    first = _run(tmp_path, session)
    assert not first.ok
    assert "第 1 次" in first.content

    (tmp_path / "state.txt").write_text("b\n", encoding="utf-8")
    second = _run(tmp_path, session)
    assert not second.ok
    assert "第 2 次" in second.content
    assert "新增失败 1 项" in second.content
    assert "已消失 1 项" in second.content

    (tmp_path / "state.txt").write_text("ok\n", encoding="utf-8")
    third = _run(tmp_path, session)
    assert third.ok
    assert "都消失了" in third.content


def test_run_tests_without_session_keeps_old_shape(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("print('ok')\n", encoding="utf-8")
    outcome = RunTestsTool().run(
        {"command": f'"{sys.executable}" ok.py', "timeout_s": 20}, _ctx(tmp_path)
    )
    assert outcome.ok
    assert "第 1 次" not in outcome.content
