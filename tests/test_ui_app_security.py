"""启动路径：闸门必须在 MainWindow 之前，主窗口全屏。"""

from __future__ import annotations

import inspect

from axo.ui import app


def test_the_security_gate_runs_before_the_main_window() -> None:
    source = inspect.getsource(app.main)
    assert "StartupGate(" in source
    assert source.index("StartupGate(") < source.index("MainWindow()")
    assert 'not os.environ.get("AXO_ISOLATED_STATE")' in source


def test_the_main_window_is_shown_fullscreen() -> None:
    assert "window.showFullScreen()" in inspect.getsource(app.main)
