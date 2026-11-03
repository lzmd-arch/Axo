"""启动路径：闸门必须在 MainWindow 之前，主窗口全屏。"""

from __future__ import annotations

import inspect

from axo.ui import app


def test_the_security_gate_runs_before_the_main_window() -> None:
    source = inspect.getsource(app.main)
    assert "StartupGate(" in source
    assert source.index("StartupGate(") < source.index("MainWindow(")
    assert "_isolated_gate_may_be_skipped()" in source


def test_the_main_window_is_shown_fullscreen() -> None:
    assert "window.showFullScreen()" in inspect.getsource(app.main)


def test_the_isolated_bypass_requires_a_temp_data_root(monkeypatch, tmp_path) -> None:
    from axo.ui.app import _isolated_gate_may_be_skipped

    monkeypatch.setenv("AXO_ISOLATED_STATE", str(tmp_path / "state"))
    monkeypatch.delenv("AXO_DATA", raising=False)
    assert _isolated_gate_may_be_skipped() is False

    monkeypatch.setenv("AXO_DATA", str(tmp_path / "probe"))
    assert _isolated_gate_may_be_skipped() is True

    monkeypatch.setenv("AXO_DATA", "D:/AxoData")
    assert _isolated_gate_may_be_skipped() is False
