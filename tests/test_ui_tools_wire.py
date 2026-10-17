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
from axo.ui.settings_store import BackendSettings, SettingsStore, ToolSettings

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


def _combo(model: str) -> SimpleNamespace:
    """模型下拉框那一半：``_model_can_see`` 只问它要 ``currentData()``。"""
    return SimpleNamespace(currentData=lambda: model)


def _window(**fields: object) -> SimpleNamespace:
    """一个只带 ``_tools()`` 碰得到的那几样的替身。

    ``_model_caps`` 与 ``model_combo`` 是这一版新加的两样（能力门），默认给"看不见图的
    模型"：**默认值取最保守的那个**，这样漏配的用例不会意外把截图工具登记上去。
    """
    base: dict[str, object] = {
        "_read_roots": (),
        # 插件那一趟的结果只是一句提示（见 MainWindow._report_plugin_issues）。
        # 替身这里收下不作声：这些用例验的是"哪些工具进了表"，不是"怎么报问题"。
        "_report_plugin_issues": lambda *_: None,
        "model_combo": _combo(""),
        "_model_caps": {},
        # 后端那一份：**默认本机**。``_model_can_see`` 现在先问"是不是云端"——云端
        # 那条路问不到能力，判据是用户声明的那一格（``_cloud_vision``）；本机照旧读
        # ``_model_caps``。不给这一位的话，漏配的用例会当场 AttributeError 而不是
        # 悄悄测成云端那一支。
        "_backend_settings": BackendSettings(),
        "_cloud_vision": False,
        "_audio_in": False,
    }
    base.update(fields)
    window = SimpleNamespace(**base)
    # `_tools()` 内部会调 `self._model_can_see()`，而替身不是一个 ``MainWindow``。
    # 照这个文件开头那条办法把**未绑定方法**接上去（包一层是为了让 ``self`` 是替身
    # 本身）——用替身重写一遍那道判断的话，测的就不是真的那一行了。
    window._model_can_see = lambda: MainWindow._model_can_see(window)  # type: ignore[attr-defined]
    # 同一族的另一样：``_on_tools_changed`` 现在从**一处**拼出那一份设置
    # （``_current_tool_settings``，两个构造点共用它）。不给的话这里会当场
    # AttributeError——而漏给的那一半（视觉那一格）会在"设置窗口里勾了、主窗口不认"
    # 那条路上静默地丢，正是这个替身要防的。
    window._current_tool_settings = lambda: MainWindow._current_tool_settings(window)  # type: ignore[attr-defined]
    return window


def _tool_names(window: SimpleNamespace) -> list[str]:
    return [spec.name for spec in MainWindow._tools(window).specs()]  # type: ignore[arg-type]


def test_changing_the_directories_is_saved_right_away(tmp_path: Path) -> None:
    """**改一次就存一次。**

    窗口里没有「确定」键（只有「完成」与「撤销修改」），而不存的话，用户加完目录
    直接关掉窗口，下次启动就又没有了——一个只有"下次启动"才暴露的丢失。
    """
    store, _data = _fake_store()
    window = _window(_settings_store=store)

    MainWindow._on_tools_changed(window, (tmp_path,))  # type: ignore[arg-type]

    assert window._read_roots == (tmp_path,)
    # **写根跟着读根走**（用户拍板：写入与执行限现有读取根），所以它也该存下来。
    assert store.load_tools() == ToolSettings(read_roots=(tmp_path,), write_roots=(tmp_path,)), (
        "没落到设置里"
    )


def test_the_directories_reach_the_tool_set_after_a_restart(tmp_path: Path) -> None:
    """存下来的目录要真的走到工具集里——**这一条是整根线的出口**。

    "下次启动"在这里就是重新从设置里读一份 ``_read_roots``（与
    ``MainWindow.__init__`` 里那一行同一件事），所以这条同时验了"写进去的读得回来"
    与"读回来的真的被用上"。
    """
    store, _data = _fake_store()
    window = _window(_settings_store=store)
    MainWindow._on_tools_changed(window, (tmp_path,))  # type: ignore[arg-type]

    restarted = _window(_read_roots=store.load_tools().read_roots)

    assert "read_file" in _tool_names(restarted)


def test_no_directories_means_the_tool_is_not_advertised() -> None:
    """**空元组 = 那个工具被禁用**，不是"给个默认目录"。

    这一条在 ``test_agent_tools`` 里已经有一条（``default_tools`` 收空元组时不登记
    ``read_file``），这里再钉一遍是因为**另一处会坏**：``_tools()`` 忘了把
    ``self._read_roots`` 传下去的话，上面那条测试照样绿（它直接调 ``default_tools``），
    而用户配了目录也没用。
    """
    assert "read_file" not in _tool_names(_window())


# ── 第二道门：当前模型看不看得见图 ──────────────────────────────────────────


def test_the_screen_tool_needs_a_model_that_can_see() -> None:
    """``look_at_screen`` **只在当前模型看得见图时**登记——这一条也是接线。

    ``default_tools(vision=...)`` 自己那条测试证明不了 ``_tools()`` 真的把 ``vision``
    传下去了：忘了传的话，一个看不见图的模型也会看到这个工具，于是用户点完「允许」
    才发现什么都不会发生——**白弹一张确认卡片**，而那正是这道门要省掉的东西。

    能力来自后端自报的 ``capabilities``（Ollama 的 ``/api/show`` 会带 ``vision``），
    由 ``_on_models_loaded`` 留下来、按**模型名**查（用户切模型不会重新拉清单，所以
    存单值会把第一次那个模型的答案钉死在所有模型上）。
    """
    blind = _window(
        model_combo=_combo("qwen3-4b:latest"),
        _model_caps={"qwen3-4b:latest": ("tools", "thinking", "completion")},
    )
    seeing = _window(
        model_combo=_combo("qwen2.5vl:7b"),
        _model_caps={"qwen2.5vl:7b": ("vision", "completion")},
    )

    assert "look_at_screen" not in _tool_names(blind)
    assert "look_at_screen" in _tool_names(seeing)


def test_an_unknown_model_does_not_get_the_screen_tool() -> None:
    """**问不到能力就不登记**：看不到工具是可见且可解释的，而"登记了但每次必败"会把
    一次静默失败叠在提示词上（同 ``read_file`` 那条"宁可不给"）。"""
    assert "look_at_screen" not in _tool_names(_window(model_combo=_combo(""), _model_caps={}))
