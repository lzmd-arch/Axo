"""只改横幅/可见度时，外观走轻量路径，不重建整套主题。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axo.ui.main_window import MainWindow
from axo.ui.theme import Theme, resolve

pytestmark = pytest.mark.gui


class _Store:
    def __init__(self) -> None:
        self.saved: Theme | None = None

    def save(self, theme: Theme) -> None:
        self.saved = theme


class _View:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def set_banner(self, text: str, opacity: int) -> None:
        self.calls.append(("banner", (text, opacity)))

    def set_wallpaper_opacity(self, percent: int) -> None:
        self.calls.append(("opacity", (percent,)))


def _fake(old: Theme, view: _View, store: _Store) -> SimpleNamespace:
    fake = SimpleNamespace(
        _theme=old,
        _resolved=resolve(old, "dark"),
        _settings_store=store,
        chat_view=view,
        _settings_dialog=None,
    )
    fake._apply_theme = lambda _theme: (_ for _ in ()).throw(AssertionError("横幅改动不该重建主题"))
    fake._push_wallpaper = lambda: view.calls.append(("wallpaper", ()))
    return fake


def test_a_banner_change_only_updates_the_banner() -> None:
    old = Theme(banner_text="旧", banner_opacity=10)
    new = Theme(banner_text="新", banner_opacity=20)
    view, store = _View(), _Store()
    fake = _fake(old, view, store)

    MainWindow._on_theme_changed(fake, new)  # type: ignore[arg-type]

    assert view.calls == [("banner", ("新", 20))]
    assert fake._theme is new
    assert store.saved is new


def test_an_opacity_change_only_updates_opacity() -> None:
    old = Theme(wallpaper_visible=18)
    new = Theme(wallpaper_visible=60)
    view, store = _View(), _Store()
    fake = _fake(old, view, store)

    MainWindow._on_theme_changed(fake, new)  # type: ignore[arg-type]

    assert view.calls == [("opacity", (60,))]
