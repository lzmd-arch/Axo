"""可读目录那根线：设置窗口 → 主窗口存的 ``QSettings`` → 下一轮的工具集。

这个文件守的是一条**横跨三层**的约定，而每一层单独看都正常：设置窗口把新列表发
出来、主窗口把它存进设置、下一轮重建工具集时按它决定登不登记 ``read_file``。任何
一处断了，症状都是同一句话——"我加了目录，它还是读不了文件"——而那句话指不到断在
哪一层。

**为什么不建 ``MainWindow``。** 与 ``test_ui_open_settings`` 同一条理由：离屏下
QtWebEngine 一渲染就把整个进程硬杀掉。这里要验的两个方法各自只碰 ``self`` 上的
一个属性，所以拿**未绑定方法**配一个人造的 ``self`` 来调，塞进去的那几个就是它们
碰到的全部；哪天它们长出新的依赖，这个文件会以 ``AttributeError`` 的方式告诉你，
而不是悄悄测了个别的东西。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from axo.ui.main_window import MainWindow
from axo.ui.settings_store import SettingsStore, ToolSettings

pytestmark = pytest.mark.gui


def _fake_store() -> tuple[SettingsStore, dict[str, object]]:
    """一个 dict 后端。**不用真 ``QSettings``**：那会写到真实注册表、用例之间互相
    串——当初把设置读写抽成 ``SettingsBackend`` 就是这个理由（见
    ``test_settings_store`` 的模块文档）。"""
    data: dict[str, object] = {}

    class _Backend:
        def value(self, key: str, default: object = None) -> object:
            return data.get(key, default)

        def setValue(self, key: str, value: object) -> None:  # noqa: N802 (Qt 命名)
            data[key] = value

    return SettingsStore(_Backend()), data


def _tool_names(window: SimpleNamespace) -> list[str]:
    return [spec.name for spec in MainWindow._tools(window).specs()]  # type: ignore[arg-type]


def test_changing_the_directories_is_saved_right_away(tmp_path: Path) -> None:
    """**改一次就存一次。**

    窗口里没有「确定」键（只有「完成」与「撤销修改」），而不存的话，用户加完目录
    直接关掉窗口，下次启动就又没有了——一个只有"下次启动"才暴露的丢失。
    """
    store, _data = _fake_store()
    window = SimpleNamespace(_read_roots=(), _settings_store=store)

    MainWindow._on_tools_changed(window, (tmp_path,))  # type: ignore[arg-type]

    assert window._read_roots == (tmp_path,)
    assert store.load_tools() == ToolSettings(read_roots=(tmp_path,)), "没落到设置里"


def test_the_directories_reach_the_tool_set_after_a_restart(tmp_path: Path) -> None:
    """存下来的目录要真的走到工具集里——**这一条是整根线的出口**。

    "下次启动"在这里就是重新从设置里读一份 ``_read_roots``（与
    ``MainWindow.__init__`` 里那一行同一件事），所以这条同时验了"写进去的读得回来"
    与"读回来的真的被用上"。
    """
    store, _data = _fake_store()
    window = SimpleNamespace(_read_roots=(), _settings_store=store)
    MainWindow._on_tools_changed(window, (tmp_path,))  # type: ignore[arg-type]

    restarted = SimpleNamespace(_read_roots=store.load_tools().read_roots)

    assert "read_file" in _tool_names(restarted)


def test_no_directories_means_the_tool_is_not_advertised() -> None:
    """**空元组 = 那个工具被禁用**，不是"给个默认目录"。

    这一条在 ``test_agent_tools`` 里已经有一条（``default_tools`` 收空元组时不登记
    ``read_file``），这里再钉一遍是因为**另一处会坏**：``_tools()`` 忘了把
    ``self._read_roots`` 传下去的话，上面那条测试照样绿（它直接调 ``default_tools``），
    而用户配了目录也没用。
    """
    assert "read_file" not in _tool_names(SimpleNamespace(_read_roots=()))
