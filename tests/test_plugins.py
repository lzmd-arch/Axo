"""插件目录（``axo.plugins``）。

这一层会**执行别人写的代码**，所以断言分两族：装得对的装上了，以及**装不上的每一种
都只影响它自己**（一个插件的 import 抛了，后面的照装——那正是这个模块存在的理由）。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from axo.plugins import PLUGIN_MODULE, load_plugins, plugins_in

GOOD = """
class _Spec:
    name = "weather"
    description = "查天气"
    parameters = {"type": "object", "properties": {}}


class WeatherTool:
    spec = _Spec()
    requires_approval = False

    def run(self, arguments, ctx):
        return "晴"
"""

FACTORY = """

def tools():
    return [WeatherTool()]
"""


def write_plugin(root: Path, name: str, source: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PLUGIN_MODULE).write_text(textwrap.dedent(source), encoding="utf-8")
    return folder


def renamed(tool_name: str) -> str:
    return GOOD.replace('name = "weather"', 'name = "' + tool_name + '"') + FACTORY


# ── 装得上的 ────────────────────────────────────────────────────────────────


def test_a_good_plugin_lands(tmp_path: Path) -> None:
    write_plugin(tmp_path, "天气", GOOD + FACTORY)

    load = load_plugins(tmp_path)

    assert [tool.spec.name for tool in load.tools] == ["weather"]
    assert load.issues == ()


def test_two_plugins_both_land(tmp_path: Path) -> None:
    """两个插件里那个文件**都叫** ``plugin.py``。

    加载时模块名带一层前缀（``axo_plugin_<目录名>``），否则 ``sys.modules`` 里会互相
    覆盖——症状是"只有最后一个插件的工具是对的"，而它看起来完全正常。
    """
    write_plugin(tmp_path, "甲", GOOD + FACTORY)
    write_plugin(tmp_path, "乙", renamed("calendar"))

    load = load_plugins(tmp_path)

    assert sorted(tool.spec.name for tool in load.tools) == ["calendar", "weather"]


def test_the_order_is_deterministic(tmp_path: Path) -> None:
    """按目录名排序：工具表的排列要稳定（提示词缓存吃得着）。"""
    for name, tool_name in (("乙", "b"), ("甲", "a"), ("丙", "c")):
        write_plugin(tmp_path, name, renamed(tool_name))

    assert [folder.name for folder in plugins_in(tmp_path)] == ["丙", "乙", "甲"]
    first = [tool.spec.name for tool in load_plugins(tmp_path).tools]
    second = [tool.spec.name for tool in load_plugins(tmp_path).tools]
    assert first == second


def test_a_missing_directory_is_no_plugins_at_all(tmp_path: Path) -> None:
    """**还没放插件是常态**，不是错误。"""
    load = load_plugins(tmp_path / "根本没有这个目录")

    assert load.tools == () and load.issues == ()


# ── 装不上的：每一种都只影响它自己 ──────────────────────────────────────────


def test_a_folder_without_the_module_is_reported(tmp_path: Path) -> None:
    (tmp_path / "空壳").mkdir()

    load = load_plugins(tmp_path)

    assert load.tools == ()
    assert len(load.issues) == 1 and "plugin.py" in load.issues[0].reason


def test_a_module_without_the_factory_is_reported(tmp_path: Path) -> None:
    write_plugin(tmp_path, "半成品", "x = 1")

    load = load_plugins(tmp_path)

    assert "tools()" in load.issues[0].reason


def test_a_plugin_that_raises_does_not_take_the_others_down(tmp_path: Path) -> None:
    """**这个模块存在的全部理由。** 插件是别人写的代码，它抛的时候后面的照装。"""
    write_plugin(tmp_path, "坏的", 'raise RuntimeError("boom")')
    write_plugin(tmp_path, "好的", GOOD + FACTORY)

    load = load_plugins(tmp_path)

    assert [tool.spec.name for tool in load.tools] == ["weather"]
    assert len(load.issues) == 1
    assert "坏的" in load.issues[0].describe() and "RuntimeError" in load.issues[0].reason


def test_a_factory_that_raises_is_reported_too(tmp_path: Path) -> None:
    write_plugin(tmp_path, "工厂坏了", "def tools():" + chr(10) + '    raise ValueError("no")')

    load = load_plugins(tmp_path)

    assert "ValueError" in load.issues[0].reason


def test_a_factory_that_returns_nothing_is_reported(tmp_path: Path) -> None:
    write_plugin(tmp_path, "空手", "def tools():" + chr(10) + "    return []")

    load = load_plugins(tmp_path)

    assert len(load.issues) == 1 and "一个工具都没返回" in load.issues[0].reason


def test_the_empty_report_does_not_blame_the_previous_plugin(tmp_path: Path) -> None:
    """上一个插件报过错，不该让这一个被误记成"返回了空"。

    ``issues`` 是整趟共享的——拿它判空是最容易写错的一处（第一版就是这么写的）。
    """
    write_plugin(tmp_path, "坏的", 'raise RuntimeError("boom")')
    write_plugin(tmp_path, "空手", "def tools():" + chr(10) + "    return []")

    load = load_plugins(tmp_path)

    reasons = {issue.plugin: issue.reason for issue in load.issues}
    assert "RuntimeError" in reasons["坏的"]
    assert "一个工具都没返回" in reasons["空手"]


def test_a_tool_without_a_name_is_skipped(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "没名字",
        "class T:"
        + chr(10)
        + "    spec = None"
        + chr(10)
        + "    def run(self, arguments, ctx):"
        + chr(10)
        + "        return ''"
        + chr(10)
        + "def tools():"
        + chr(10)
        + "    return [T()]",
    )

    load = load_plugins(tmp_path)

    assert load.tools == () and "spec.name" in load.issues[0].reason


def test_a_tool_without_run_is_skipped(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "没手",
        GOOD.replace(
            "    def run(self, arguments, ctx):" + chr(10) + '        return "晴"' + chr(10), ""
        )
        + FACTORY,
    )

    load = load_plugins(tmp_path)

    assert load.tools == () and "run()" in load.issues[0].reason


def test_a_name_that_is_already_taken_is_skipped(tmp_path: Path) -> None:
    """**与内置工具撞名要挡住。** 两个同名工具会让模型看到一份自相矛盾的工具表，
    而它没法自己发现这件事——它只看到两个都叫 ``weather`` 的东西。"""
    write_plugin(tmp_path, "撞名", GOOD + FACTORY)

    load = load_plugins(tmp_path, taken=["weather"])

    assert load.tools == ()
    assert "已经被占了" in load.issues[0].reason


def test_two_plugins_fighting_over_one_name_keep_the_first(tmp_path: Path) -> None:
    """**谁先到由目录名的排序决定**（不是文件系统的返回次序）——所以这条断言什么名字
    丢了，是按规范序算出来的：``乙`` (U+4E59) 排在 ``甲`` (U+7532) 前面。"""
    write_plugin(tmp_path, "甲", GOOD + FACTORY)
    write_plugin(tmp_path, "乙", GOOD + FACTORY)

    load = load_plugins(tmp_path)

    assert len(load.tools) == 1
    assert [issue.plugin for issue in load.issues] == ["甲"]
