"""「设置」按钮打不开时，要说一句话，而不是变成一个点不动的按钮。

这个文件只有一个主题，但它守的是一件真发生过的事：设置窗口的构造器末尾就调
``set_state``，而它一旦抛异常，异常就抛在 ``show()`` **之前**——窗口不出现，
下一次点击走同一条路照样抛。Qt 只在控制台留回溯，于是一个 bug 表现成"按钮坏了"。

**为什么不建 ``MainWindow``。** 仓里的 gui 测试一律不建它：离屏下 QtWebEngine
一渲染就把整个进程硬杀掉（见 README「别在离屏下跑渲染路径」），而这里要验的只是
`_on_open_settings` 这一个方法。所以拿 `MainWindow` 的**未绑定方法**配上一个人造
的 ``self`` 来调——放进去的那几个属性就是那个方法碰到的全部。哪天它长出了新的依赖，
这个文件会以 ``AttributeError`` 的方式告诉你，而不是悄悄测了个别的东西。

`SettingsDialog` 与 `QMessageBox.warning` 都换掉：一个要炸，一个本来是模态框，
真弹出来会把测试挂住。
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui import main_window as main_window_module
from axo.ui.main_window import MainWindow
from axo.ui.settings_store import BackendSettings
from axo.ui.theme import Theme, resolve

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    return QApplication.instance() or QApplication([])


def _window() -> SimpleNamespace:
    """`_on_open_settings` 真正读到的全部状态。**不是 MainWindow。**

    后面那几个 ``_on_*`` 不是它读的，是它**递给别人**的：真窗口里那几个是绑定方法，
    此处只需要有个东西能塞进假窗口的 ``connect``。``_persona_texts`` 则是它**真的
    调**的（回填人设页那一步），所以这里给一个返回两段正文的替身。

    **加了一个新依赖就加一个字段**，这正是这个文件想要的失败方式：漏了会当场
    ``AttributeError``，而不是悄悄测了个别的东西。
    """
    theme = Theme()
    return SimpleNamespace(
        _settings_dialog=None,
        _theme=theme,
        _resolved=resolve(theme, "dark"),
        # 后端那一份：**整份**传进设置窗口（``provider`` / ``cloud_base_url`` /
        # ``api_key`` 也住在里面）。``_host`` / ``_allow_remote`` 只是
        # :class:`~axo.ui.settings_store.BackendSettings` 上的只读属性
        # ——替身是个 ``SimpleNamespace``，属性不会自己长出来，所以给的是
        # 真对象。默认值就是「本机 Ollama、不允许远程」，与刚装好的终端一模一样。
        _backend_settings=BackendSettings(),
        # 工具页那一份：可读目录**空着**（= read_file 禁用），与一个刚装好的终端
        # 一模一样。
        _read_roots=(),
        # 记忆页那一份：**任务记忆关着**，与一个刚装好的终端一模一样。
        _task_memory=False,
        # 记忆页的另一项：**联想通道也是关的**（它是量出来的负数，默认关）。
        _graph_enabled=False,
        # 记忆页剩下的两项（P4c）：巩固默认开、周期一天。这里不必与默认值一致——
        # 它是"主窗口手里那份状态"，而设置窗口只是把它显示出来。
        _consolidation_enabled=True,
        _consolidation_interval_s=86_400,
        # 重开设置窗口时要把上一次建图的结果补回去——图状态住在主窗口上，
        # 对话框每次都是新的。
        _graph_note="",
        # 数据页那一份：**这个进程实际在用哪个目录**，以及它听谁的。真源在 ``paths``
        # 里（不是 QSettings、也不是记忆库），所以每次打开都现读一遍。
        _data_root=Path(r"C:\some\where\axo"),
        _data_source="default",
        _on_theme_changed=lambda *_args: None,
        _on_backend_changed=lambda *_args: None,
        _on_tools_changed=lambda *_args: None,
        _on_memory_changed=lambda *_args: None,
        _on_rebuild_graph=lambda *_args: None,
        _on_manual_consolidation=lambda *_args: None,
        _on_persona_save_requested=lambda *_args: None,
        _on_persona_clear_requested=lambda *_args: None,
        _on_open_data_folder=lambda *_args: None,
        _on_change_data_location=lambda *_args: None,
        _on_restore_data_location=lambda *_args: None,
        _persona_texts=lambda: ("内置的那段", "我写的那段"),
    )


@pytest.fixture(autouse=True)
def _canned_last_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """把「上次巩固是什么时候」换成一句固定的。

    ``_on_open_settings`` 每次打开都要现读一遍 ``memory/consolidation/last.json``
    （见 ``_last_run_note``），而那个文件的真路径在 ``%LOCALAPPDATA%`` 下——测试去读
    用户的真数据目录，结果就会随这台机器上跑没跑过巩固而变。**注入一个固定的输出**，
    这一文件里其余的东西都不依赖它。

    替身之所以放在**模块属性**上而不是 ``self`` 上：那个函数是模块级的纯函数，
    与"主窗口手里那份状态"无关；这一整个文件的主题正是"哪些东西属于主窗口"。
    """
    monkeypatch.setattr(main_window_module, "_last_run_note", lambda: "最近一次：测试。")


def _consolidate_signals() -> SimpleNamespace:
    """``SettingsDialog`` 上那几个 ``Signal`` 的替身。多一个信号就多一行。"""
    names = (
        "theme_changed",
        "backend_changed",
        "tools_changed",
        "memory_changed",
        "graph_rebuild_requested",
        "consolidation_requested",
        "persona_save_requested",
        "persona_clear_requested",
        "open_data_folder_requested",
        "change_data_location_requested",
        "restore_data_location_requested",
        "onboarding_requested",
    )
    return SimpleNamespace(**{name: SimpleNamespace(connect=lambda _slot: None) for name in names})


def _catch_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把 ``QMessageBox.warning`` 换成收信箱。第三个参数是正文。"""
    seen: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, _title, text, *_rest: seen.append(text),
    )
    return seen


def test_a_settings_window_that_cannot_be_built_says_so(
    qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """构造抛异常 → 一句看得见的话，且**不留**一个坏掉的对象。

    留着的后果是：这个方法的下一次调用会走进 ``else`` 分支去给一个半成品
    ``set_state``，而那正是刚才炸掉的那一步——按钮从此每次都炸一遍，且因为
    对象不是 None，连"重新构造一次"这条路都没了。
    """

    class Boom:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("缩略图画不出来")

    monkeypatch.setattr(main_window_module, "SettingsDialog", Boom)
    warned = _catch_warnings(monkeypatch)
    window = _window()

    MainWindow._on_open_settings(window)  # type: ignore[arg-type]

    assert len(warned) == 1
    assert "缩略图画不出来" in warned[0], "没把真正的原因带出来，用户无从下手"
    assert "控制台" in warned[0], "没告诉他去哪儿看详情"
    assert window._settings_dialog is None


def test_a_settings_window_that_cannot_be_synced_says_so(
    qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重开时 ``set_state`` 抛异常 → 同一条路。

    这条与上一条分开，因为走的是两个分支（新建 vs 复用），而**真发生过的那个
    bug 恰好在复用这一支**：对象早就建好了，炸的是每次重开都要走的那次同步。
    """
    broken = SimpleNamespace(
        set_state=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("同步炸了"))
    )
    warned = _catch_warnings(monkeypatch)
    window = _window()
    window._settings_dialog = broken

    MainWindow._on_open_settings(window)  # type: ignore[arg-type]

    assert len(warned) == 1
    assert "同步炸了" in warned[0]


def test_the_settings_window_is_shown_when_nothing_goes_wrong(
    qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正常路径仍然要 ``show()`` 出来——否则上面那个 ``try`` 就白包了。

    **这条不是多余的**：把 ``show()`` 缩进到 ``try`` 里面写错一次，前两条照样绿。
    """
    calls: list[str] = []
    fake = SimpleNamespace(
        **_consolidate_signals().__dict__,
        set_persona=lambda *_args: calls.append("setPersona"),
        set_data=lambda *_args: calls.append("setData"),
        set_graph_note=lambda *_args: None,
        set_consolidate_note=lambda *_args: calls.append("setConsolidateNote"),
        show=lambda: calls.append("show"),
        raise_=lambda: calls.append("raise_"),
        activateWindow=lambda: calls.append("activateWindow"),
    )
    monkeypatch.setattr(main_window_module, "SettingsDialog", lambda *_args, **_kwargs: fake)
    warned = _catch_warnings(monkeypatch)
    window = _window()

    MainWindow._on_open_settings(window)  # type: ignore[arg-type]

    assert warned == []
    # 人设页与巩固那一行**都在 show() 之前**就要填好：它们各自的真源都不在这个
    # 窗口里（人设在库里、巩固的"上次什么时候跑的"在盘上），漏了的话窗口会带着
    # 两个空框开出来，而用户看到的是"我写的人设没了""它到底跑没跑过"。
    assert calls == [
        "setConsolidateNote",
        "setPersona",
        "setData",
        "show",
        "raise_",
        "activateWindow",
    ]
    assert window._settings_dialog is fake


def test_the_catch_all_is_deliberate_and_marked(qapp: Any) -> None:
    """钉住"这个宽 except 是写下来的、有出处的"，不是顺手抄来的。

    裸 ``except Exception`` 在这个仓里是稀罕东西，所以它旁边必须有一句说明它
    为什么不吞异常（它把异常变成一句人话，回溯仍然打到控制台）。这条断言很轻，
    但它让"哪天有人把 except 改成 pass"变成一次有意的改动。
    """
    source = (Path(__file__).resolve().parent.parent / "src/axo/ui/main_window.py").read_text(
        encoding="utf-8"
    )
    block = source.split("def _on_open_settings", 1)[1].split("def _on_theme_changed", 1)[0]
    assert re.search(r"except Exception as exc:\s*# noqa: BLE001", block)
    assert "QMessageBox.warning" in block
