"""闲聊轮的工具面：只读白名单 + 接线。

2026-10-31 实测的失败形态：用户说「看看最近有些什么新闻」，规则路由把它判成闲聊，
那一轮请求里连 ``tools`` 字段都没有，模型只能如实回答"我手上没有能连出去的接口"。
这个文件守两件事：

1. :meth:`MainWindow._chat_tools` 只让白名单里的只读工具通过——写记忆、建提醒、
   跑命令这些改状态的工具**不在闲聊轮的工具表里**；
2. ``_on_assembly_ready`` 在闲聊轮**真的调它**——只写一个方法而没人调，正是这个
   项目反复出现的"纯层全绿但没接线"（见 ``test_ui_router_wire`` 的教训）。

不建 ``MainWindow``：理由同 ``test_ui_tools_wire``（离屏下 QtWebEngine 一渲染就把
进程硬杀掉）。这里用未绑定方法配一个只带 ``_tools`` 的替身。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from axo.agent.catalog import CHAT_TOOL_NAMES, default_tools
from axo.agent.tools import ToolRegistry
from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui


class _Web:
    """``default_tools`` 只问它要 ``available``（见工具面那段"联网读默认不登记"）。"""

    available = True


def _window(tools):
    return SimpleNamespace(_tools=lambda: ToolRegistry(tools))


def _tool_set():
    return default_tools(weather=object(), web=_Web())


def test_chat_tools_are_the_read_only_whitelist() -> None:
    registry = MainWindow._chat_tools(_window(_tool_set()))
    names = registry.names
    # 这一条是 2026-10-31 那个失败的直接回归：新闻/天气必须能被模型自己伸手拿到。
    assert "web_search" in names
    assert "fetch_url" in names
    assert "get_weather" in names
    assert "get_time" in names
    # 改状态的一律不进闲聊轮。
    assert "write_memory" not in names
    assert "forget_memory" not in names
    assert "set_reminder" not in names
    assert "cancel_reminder" not in names
    assert "run_command" not in names
    assert "write_file" not in names
    # 白名单是封闭集合：名字不在表里就不该出现。
    assert set(names) <= set(CHAT_TOOL_NAMES)


def test_every_chat_tool_is_approval_free() -> None:
    """闲聊轮没有"等用户点头"的语境——白名单里混进一个要确认的工具，那一轮会卡到超时。"""
    tools = _tool_set()
    by_name = {tool.spec.name: tool for tool in tools}
    for name in CHAT_TOOL_NAMES:
        tool = by_name.get(name)
        if tool is not None:
            assert not tool.requires_approval, f"{name} 要确认，不能进闲聊轮"


def test_assembly_ready_wires_chat_tools() -> None:
    """接线守卫：闲聊轮的分支里必须真的构造并交给 agent 循环。"""
    source = inspect.getsource(MainWindow._on_assembly_ready)
    assert "_chat_tools()" in source, "闲聊轮没有构造只读工具集"
    assert "_start_agent_turn(assembly, registry)" in source, "工具集构造了却没有交给循环"


def test_empty_tool_set_falls_back_to_plain_chat() -> None:
    registry = MainWindow._chat_tools(_window(()))
    assert len(registry) == 0


def test_turn_tools_declared_follows_the_same_registry() -> None:
    """装配器那句"这一轮没有工具可用"必须跟着**同一份**结论走。

    只说 ``_turn_uses_tools`` 的话，闲聊轮的上下文会告诉模型没有工具，而请求里
    明明声明了 ``tools``——模型要么不敢调，要么像 2026-10-10 那样凭空声称调过。
    """
    tools = _tool_set()
    window = SimpleNamespace(
        _turn_uses_tools=False,
        _turn_needs_clarification=False,
        _chat_tools=lambda: MainWindow._chat_tools(_window(tools)),
    )
    assert MainWindow._turn_tools_declared(window) is True
    window._turn_needs_clarification = True
    assert MainWindow._turn_tools_declared(window) is True, "没判准的一轮也该有只读工具"
    window._turn_needs_clarification = False
    window._turn_uses_tools = True
    assert MainWindow._turn_tools_declared(window) is True
