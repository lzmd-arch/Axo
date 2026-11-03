"""界面错误日志的轮转：不封顶会长期吃掉磁盘。"""

from __future__ import annotations

from pathlib import Path

from axo.ui import errorlog


def test_the_log_rotates_at_the_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    monkeypatch.setattr(errorlog, "_MAX_BYTES", 40)
    for index in range(4):
        errorlog.log_ui_error(f"第 {index} 条" + "x" * 30)
    path = errorlog.ui_error_log_path()
    assert path.exists()
    assert path.with_name(path.name + ".1").exists()
    assert path.stat().st_size < 80
