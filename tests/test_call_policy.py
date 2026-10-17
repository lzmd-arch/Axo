"""通话状态机：什么时候开麦、什么时候复位、什么时候挂断。"""

from __future__ import annotations

import pytest

from axo.call import DEFAULT_IDLE_HANGUP_S, WATCHDOG_S, CallSession, CallState


def live_session(now: float = 0.0) -> CallSession:
    session = CallSession()
    session.start(now)
    return session


def test_start_is_idempotent_and_leaves_off() -> None:
    session = live_session()
    session.note_activity(5.0)

    session.start(10.0)

    assert session.state is CallState.LISTENING
    assert session.started_at == 0.0
    assert session.last_activity == 5.0


def test_the_happy_path_listen_capture_think_speak_listen() -> None:
    session = live_session()

    assert session.move(CallState.CAPTURING, 1.0)
    assert session.move(CallState.THINKING, 2.0)
    assert session.move(CallState.SPEAKING, 3.0)
    assert session.move(CallState.LISTENING, 4.0)

    assert session.state is CallState.LISTENING
    assert session.history == (
        CallState.LISTENING,
        CallState.CAPTURING,
        CallState.THINKING,
        CallState.SPEAKING,
        CallState.LISTENING,
    )


def test_an_illegal_transition_is_refused_and_changes_nothing() -> None:
    """非法迁移**不抛异常、不改状态**——调用方检查返回值并说一句。"""
    session = live_session()
    session.move(CallState.CAPTURING, 1.0)

    assert session.move(CallState.SPEAKING, 2.0) is False
    assert session.state is CallState.CAPTURING
    assert session.state_since == 1.0


def test_moving_to_the_same_state_is_a_noop() -> None:
    session = live_session()

    assert session.move(CallState.LISTENING, 1.0) is True
    assert session.state_since == 0.0


def test_the_microphone_is_closed_while_she_thinks_or_speaks() -> None:
    """半双工的全部含义就在这一句里。"""
    session = live_session()
    assert session.can_listen() is True

    session.move(CallState.CAPTURING, 1.0)
    assert session.can_listen() is True

    session.move(CallState.THINKING, 2.0)
    assert session.can_listen() is False

    session.move(CallState.SPEAKING, 3.0)
    assert session.can_listen() is False

    session.move(CallState.LISTENING, 4.0)
    assert session.can_listen() is True


def test_muting_closes_the_microphone_but_keeps_the_call_alive() -> None:
    session = live_session()

    session.set_muted(True, 1.0)

    assert session.live is True
    assert session.can_listen() is False

    session.set_muted(False, 2.0)
    assert session.can_listen() is True


def test_idle_hangup_only_applies_to_a_quiet_listening_call() -> None:
    session = live_session()

    assert session.tick(DEFAULT_IDLE_HANGUP_S - 1) == ""
    assert session.tick(DEFAULT_IDLE_HANGUP_S + 1) == "idle"

    session.set_muted(True, DEFAULT_IDLE_HANGUP_S + 2)
    assert session.tick(DEFAULT_IDLE_HANGUP_S * 2) == "", "静音不该把通话挂掉"


def test_activity_keeps_the_call_alive() -> None:
    session = live_session()

    session.note_activity(DEFAULT_IDLE_HANGUP_S * 0.9)

    assert session.tick(DEFAULT_IDLE_HANGUP_S * 1.5) == ""


def test_the_watchdog_covers_capture_and_thinking_only() -> None:
    """长回复念一分钟是正常的；卡在"正在想"才是信号丢了。"""
    session = live_session()
    session.move(CallState.CAPTURING, 1.0)
    assert session.tick(1.0 + WATCHDOG_S - 0.1) == ""
    assert session.tick(1.0 + WATCHDOG_S + 0.1) == "watchdog"

    session.move(CallState.LISTENING, 20.0)
    session.move(CallState.THINKING, 21.0)
    assert session.tick(21.0 + WATCHDOG_S + 0.1) == "watchdog"

    session.move(CallState.SPEAKING, 40.0)
    assert session.tick(40.0 + WATCHDOG_S * 10) == "", "播放没有看门狗"


def test_error_can_fall_back_to_listening_or_hang_up() -> None:
    session = live_session()
    assert session.move(CallState.ERROR, 1.0)

    assert session.move(CallState.LISTENING, 2.0)
    assert session.move(CallState.ERROR, 3.0)
    assert session.move(CallState.OFF, 4.0)

    assert session.live is False
    assert session.can_listen() is False


def test_finish_is_idempotent() -> None:
    session = live_session()
    session.finish(1.0)
    session.finish(2.0)

    assert session.state is CallState.OFF
    assert session.state_since == 1.0


def test_a_dead_call_ignores_ticks() -> None:
    session = CallSession()

    assert session.tick(10_000) == ""
    assert session.can_listen() is False


_PATHS: dict[CallState, tuple[CallState, ...]] = {
    CallState.LISTENING: (),
    CallState.CAPTURING: (CallState.CAPTURING,),
    CallState.THINKING: (CallState.CAPTURING, CallState.THINKING),
    CallState.SPEAKING: (CallState.CAPTURING, CallState.THINKING, CallState.SPEAKING),
    CallState.ERROR: (CallState.ERROR,),
}


@pytest.mark.parametrize("state", list(_PATHS))
def test_every_live_state_can_be_left(state: CallState) -> None:
    """**每个阶段都必须有出口**：否则某一次异常之后用户只能关程序。"""
    session = live_session()
    for index, step in enumerate(_PATHS[state], start=1):
        assert session.move(step, float(index)), f"到 {step} 没有路径"
    assert session.state is state

    assert session.move(CallState.OFF, 10.0)
    assert session.state is CallState.OFF
