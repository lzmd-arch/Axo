"""六幕启动：幕表校验、提示音、以及开屏里的状态机。

**不碰 QtWebEngine**（与 ``test_ui_splash.py`` 同一条纪律）；确认/跳过用直接调用
事件处理函数，不依赖真实键盘焦点。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ── 幕表 ────────────────────────────────────────────────────────────────────


def test_the_builtin_sequence_has_six_acts() -> None:
    from axo.ui.boot import BOOT_ACTS, builtin_sequence

    assert len(BOOT_ACTS) == 6
    assert len(builtin_sequence().acts) == 6


def test_a_good_json_sequence_loads(tmp_path) -> None:
    from axo.ui.boot import load_boot_sequence

    path = tmp_path / "sequence.json"
    path.write_text(
        json.dumps(
            {
                "acts": [
                    {"id": "a", "title": "甲", "duration_ms": 400, "kind": "fade"},
                    {"id": "b", "title": "乙", "confirm": True, "kind": "hero"},
                ]
            }
        ),
        encoding="utf-8",
    )
    sequence = load_boot_sequence(path)
    assert [act.id for act in sequence.acts] == ["a", "b"]
    assert sequence.acts[1].confirm is True


@pytest.mark.parametrize(
    "payload",
    [
        "{not json}",
        json.dumps({"acts": []}),
        json.dumps({"acts": [{"kind": "poem"}]}),
        json.dumps({"acts": [{"duration_ms": -1}]}),
        json.dumps({"acts": [{"surprise": 1}]}),
        json.dumps({"acts": [{"duration_ms": 400}] * 25}),
    ],
)
def test_bad_sequences_fall_back_to_the_builtin(tmp_path, payload: str) -> None:
    from axo.ui.boot import BOOT_ACTS, load_boot_sequence

    path = tmp_path / "sequence.json"
    path.write_text(payload, encoding="utf-8")
    assert load_boot_sequence(path).acts == BOOT_ACTS


# ── 提示音 ──────────────────────────────────────────────────────────────────


def test_the_chime_is_a_deterministic_wav() -> None:
    from axo.ui.boot_sound import chime_bytes

    first = chime_bytes()
    assert first[:4] == b"RIFF" and first[8:12] == b"WAVE"
    assert 44 < len(first) < 200_000
    assert chime_bytes() == first


def test_ensure_chime_writes_once(tmp_path, monkeypatch) -> None:
    from axo.ui import boot_sound

    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    path = boot_sound.ensure_chime()
    assert path is not None and path.is_file()
    stamp = path.stat().st_mtime_ns
    assert boot_sound.ensure_chime() == path
    assert path.stat().st_mtime_ns == stamp


# ── 开屏状态机 ──────────────────────────────────────────────────────────────


def _splash(*, motion: bool, acts=None, allow_skip: bool = True, confirm: bool = True):
    from axo.ui.boot import BootAct, BootSequence
    from axo.ui.splash import SplashScreen
    from axo.ui.theme import MODE_DARK, Theme, resolve

    if acts is None:
        acts = (
            BootAct("one", "甲", duration_ms=200, kind="fade"),
            BootAct("two", "乙", duration_ms=200, kind="type"),
            BootAct("three", "丙", duration_ms=200, confirm=True, kind="hero"),
        )
    theme = Theme(mode=MODE_DARK, motion=motion)
    return SplashScreen(
        theme=theme,
        resolved=resolve(theme, MODE_DARK),
        boot=BootSequence(acts),
        allow_skip=allow_skip,
        require_confirm=confirm,
        speed=2.0,
    )


def test_motion_off_never_runs_the_sequence(qapp: Any) -> None:
    splash = _splash(motion=False)
    splash.show()
    try:
        assert splash._boot is None
        assert splash.has_pending_boot() is False
    finally:
        splash.close()


def test_pre_roll_plays_every_act_before_returning(qapp: Any) -> None:
    """用户要"每一幕结束之后才进主界面"：pre_roll 返回时序列必须已经放完。"""
    splash = _splash(motion=True, confirm=False)
    splash.show()
    try:
        splash.pre_roll()
        assert splash._boot_done is True
        assert splash.has_pending_boot() is False
        assert splash._boot_running is False
    finally:
        splash.close()


def test_a_confirm_act_returns_as_soon_as_the_key_arrives(qapp: Any) -> None:
    from PySide6.QtCore import QTimer

    splash = _splash(motion=True)
    splash.show()
    try:
        splash._boot_running = True
        splash._boot_index = 2  # 默认幕表里第 3 幕才会确认
        QTimer.singleShot(60, splash._confirm_boot)
        started = time.monotonic()
        splash._wait_boot_act(splash._boot.acts[2], confirm=True)
        assert time.monotonic() - started < 2.0, "按键之后不该再等满 5 秒"
    finally:
        splash.close()


def test_skip_ends_the_sequence_early(qapp: Any) -> None:
    from PySide6.QtCore import QTimer

    splash = _splash(motion=True, confirm=False)
    splash.show()
    try:
        QTimer.singleShot(60, splash._skip_boot)
        started = time.monotonic()
        splash.pre_roll()
        assert time.monotonic() - started < 2.0, "跳过之后不该再播完三幕"
        assert splash._boot_running is False
    finally:
        splash.close()


def test_disabling_skip_ignores_the_request(qapp: Any) -> None:
    splash = _splash(motion=True, allow_skip=False, confirm=False)
    splash.show()
    try:
        splash._boot_running = True
        splash._skip_boot()
        assert splash._boot_skipped is False
    finally:
        splash.close()
