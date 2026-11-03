"""对话页加载完成前的消息/状态，要在页面起来之后补推。

不构造真的 ``ChatView``：离屏下 QtWebEngine 第一次渲染会让进程无声消失。
这里用未绑定方法 + 假 self，只验缓存与补推这两条纯逻辑。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axo.ui.chat_view import ChatView

pytestmark = pytest.mark.gui


def _self(calls: list[tuple[str, tuple[Any, ...]]]) -> SimpleNamespace:
    fake = SimpleNamespace(
        _history=None,
        _status="",
        _page=SimpleNamespace(runJavaScript=lambda *_args: None),
        _theme=None,
        _wallpaper_opacity=-1,
        _motion=False,
        _wallpaper="",
        _banner_text="",
        _banner_opacity=0,
        _recovering=False,
        _persona=None,
        _voice_state="",
        _call_state="",
        _call_level=-1,
        _call_partial="",
        _stickers=[],
    )
    fake._call = lambda function, *args: calls.append((function, args))
    return fake


def test_a_message_appended_before_load_is_cached_and_replayed() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    view = _self(calls)

    ChatView.append_message(view, "system", "提醒：该喝水了")  # type: ignore[arg-type]
    assert view._history == [{"role": "system", "text": "提醒：该喝水了", "meta": {}}]

    ChatView._on_load_finished(view, True)  # type: ignore[arg-type]
    history_calls = [args for name, args in calls if name == "loadHistory"]
    assert history_calls == [(view._history,)]


def test_the_last_status_is_replayed_after_load() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    view = _self(calls)

    ChatView.set_status(view, "正在探测本地模型服务…")  # type: ignore[arg-type]
    ChatView._on_load_finished(view, True)  # type: ignore[arg-type]
    status_calls = [args for name, args in calls if name == "setStatus"]
    assert status_calls[-1] == ("正在探测本地模型服务…",)


def test_the_same_wallpaper_is_not_pushed_twice() -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    view = _self(calls)

    ChatView.set_wallpaper(view, "file:///wall.webm")  # type: ignore[arg-type]
    ChatView.set_wallpaper(view, "file:///wall.webm")  # type: ignore[arg-type]
    assert [name for name, _ in calls].count("setWallpaper") == 1

    ChatView.set_banner(view, "阿螈", 12)  # type: ignore[arg-type]
    ChatView.set_banner(view, "阿螈", 12)  # type: ignore[arg-type]
    assert [name for name, _ in calls].count("setBanner") == 1
    assert [args for name, args in calls if name == "setBanner"] == [("阿螈", 12)]
