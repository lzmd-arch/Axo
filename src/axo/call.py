"""一次通话的状态与边界。**纯函数，不认识 Qt，也不碰音频。**

电话式对话真正难的不是"怎么录"，而是"现在到底在哪个阶段"：她在想、她在说、
麦克风该不该开、用户按了挂断、她十秒没动是不是卡住了。这些判断如果散在界面槽
里，每加一条路径就会多一处状态不一致——而症状通常是**麦克风开着但按钮显示空闲**，
或者**她还在说、麦克风又开了**（后者会把她自己的声音录成用户输入）。

所以这里把状态机单独拿出来：

```text
OFF ──start──▶ LISTENING ──speech──▶ CAPTURING ──utterance──▶ THINKING
                   ▲                     │                        │
                   │                     └────drop────▶ LISTENING  │
                   │                                              ▼
                   └──────────playback done────────────── SPEAKING
                   │                                              │
                   └──────────────── interrupt ───────────────────┘
```

**半双工**：``THINKING`` 与 ``SPEAKING`` 期间麦克风是关的。Qt 没有回声消除，
她说话时开麦会把她的声音录进去；插话做成"点一下先停播、再开麦"，行为确定。

两个时间边界也在这里，因为它们与状态有关而与被谁驱动无关：

* **空闲挂断**：``LISTENING`` 且这么久没有动静 → 挂断（默认 90 秒）；
* **看门狗**：``CAPTURING``/``THINKING`` 停在原地太久 → 说明某个信号丢了 →
  回 ``LISTENING`` 复活，而不是让界面永远停在"正在想"。

``SPEAKING`` **不设看门狗**：一段长回复念一分钟是正常的，而它有自己的完成信号
（``playback_done``）与用户的插话。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "DEFAULT_IDLE_HANGUP_S",
    "WATCHDOG_S",
    "CallSession",
    "CallState",
]

DEFAULT_IDLE_HANGUP_S = 90.0
"""听着一句话都没有多久算挂断。**不是隐私边界**（麦克风只在 LISTENING 开），
只是别让一个忘了挂的通话永远占着麦克风与识别线程。"""

WATCHDOG_S = 8.0
"""CAPTURING/THINKING 停在原地多久算信号丢了。"""


class CallState(str, Enum):
    OFF = "off"
    LISTENING = "listening"
    CAPTURING = "capturing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


#: 允许的迁移 (from, to)。**显式表而不是一堆 if**：漏掉一条时测试能直接看出是哪
#: 一条，而 if 链漏掉的那条只会表现为"某个按法没反应"。
_TRANSITIONS: frozenset[tuple[CallState, CallState]] = frozenset(
    {
        (CallState.OFF, CallState.LISTENING),
        (CallState.LISTENING, CallState.CAPTURING),
        (CallState.LISTENING, CallState.OFF),
        (CallState.LISTENING, CallState.THINKING),
        (CallState.LISTENING, CallState.ERROR),
        (CallState.CAPTURING, CallState.LISTENING),
        (CallState.CAPTURING, CallState.THINKING),
        (CallState.CAPTURING, CallState.OFF),
        (CallState.CAPTURING, CallState.ERROR),
        (CallState.THINKING, CallState.SPEAKING),
        (CallState.THINKING, CallState.LISTENING),
        (CallState.THINKING, CallState.OFF),
        (CallState.THINKING, CallState.ERROR),
        (CallState.SPEAKING, CallState.LISTENING),
        (CallState.SPEAKING, CallState.CAPTURING),
        (CallState.SPEAKING, CallState.OFF),
        (CallState.SPEAKING, CallState.ERROR),
        (CallState.ERROR, CallState.LISTENING),
        (CallState.ERROR, CallState.OFF),
    }
)


@dataclass
class CallSession:
    """一次通话的时钟与状态。**所有方法都收 ``now``**，所以测试不睡真时间。

    Attributes:
        state: 当前阶段。
        started_at: 这一次通话开始的时刻。
        last_activity: 最近一次"有动静"（开口、提交、播放完）。闲置计时用它。
        state_since: 进入当前阶段的时刻。看门狗用它。
        muted: 用户静音。静音期间不开麦，但通话仍然活着。
    """

    state: CallState = CallState.OFF
    started_at: float = 0.0
    last_activity: float = 0.0
    state_since: float = 0.0
    muted: bool = False
    _history: list[CallState] = field(default_factory=list, repr=False)

    @property
    def live(self) -> bool:
        return self.state is not CallState.OFF

    @property
    def history(self) -> tuple[CallState, ...]:
        return tuple(self._history)

    def start(self, now: float) -> None:
        """开始通话。**已经是通话中就什么都不做**（幂等）。"""
        if self.live:
            return
        self.state = CallState.LISTENING
        self.started_at = now
        self.last_activity = now
        self.state_since = now
        self.muted = False
        self._history = [CallState.LISTENING]

    def move(self, state: CallState, now: float) -> bool:
        """迁移。**非法迁移返回假、不改任何东西**（调用方据此说一句）。

        这里不抛异常：状态机跑在界面线程的槽里，一次非法迁移不该把整个窗口带
        下去；而"静默地按非法路径走"正是要防的事，所以返回值必须被检查。
        """
        if state is self.state:
            return True
        if (self.state, state) not in _TRANSITIONS:
            return False
        self.state = state
        self.state_since = now
        self._history.append(state)
        if state in (CallState.LISTENING, CallState.CAPTURING):
            self.last_activity = now
        return True

    def note_activity(self, now: float) -> None:
        self.last_activity = now

    def set_muted(self, muted: bool, now: float) -> None:
        self.muted = bool(muted)
        self.note_activity(now)

    def finish(self, now: float) -> None:
        """挂断。**幂等**。"""
        if not self.live:
            return
        self.state = CallState.OFF
        self.state_since = now
        self._history.append(CallState.OFF)

    def tick(self, now: float, *, idle_hangup_s: float = DEFAULT_IDLE_HANGUP_S) -> str:
        """定时检查。返回 ``""`` / ``"idle"`` / ``"watchdog"``。

        **只报告，不改状态**：怎么收场（挂断还是复位）是调用方的决定，而这一层
        不该替它做。
        """
        if not self.live:
            return ""
        if (
            self.state is CallState.LISTENING
            and not self.muted
            and now - self.last_activity >= idle_hangup_s
        ):
            return "idle"
        if (
            self.state in (CallState.CAPTURING, CallState.THINKING)
            and now - self.state_since >= WATCHDOG_S
        ):
            return "watchdog"
        return ""

    def can_listen(self) -> bool:
        """现在该不该开麦。**半双工的所有约束都收在这一句里。**"""
        return (
            self.live
            and not self.muted
            and self.state
            in (
                CallState.LISTENING,
                CallState.CAPTURING,
            )
        )
