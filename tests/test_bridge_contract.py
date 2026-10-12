"""网页桥那一层的契约：**谁被连、谁被发、谁写在表里**。纯源码分析，核心腿里跑。

## 为什么值得一条测试

2026-10-12 的语音那一批里，``WebBridge`` 上加了 :meth:`voice_toggle` 这个槽，槽里写
``self.voice_toggled.emit()``——**而 `voice_toggled` 只声明在 `ChatView` 上**。
两个 ``self`` 是两个不同的对象，于是构造 ``ChatView`` 的那一刻抛 ``AttributeError``：
**应用根本起不来**。

它躲过了全部四条腿，原因是这份仓里**没有任何一条测试构造真的 ``ChatView``**（离屏下
QtWebEngine 一渲染就把进程硬杀掉，所以 gui 腿一律用替身，见
``tests/test_ui_capture_chain.py`` 开头那段）。而"起不来"这件事在本仓没有机器判据：
构建自检会拦（那是唯一一条真起产物的路），但它要四分钟、还要在有 Windows 的机器上。

**所以这里补一条便宜的、纯文本的判据。** 它不验行为，只验"名字对不对得上"——而这一
类错误恰恰只由名字构成。三条：

1. ``ChatView`` 连的每一个 ``self._bridge.X`` 都得在 ``WebBridge`` 上存在。
2. ``WebBridge`` 里发的每一个 ``self.X.emit()`` 都得是它自己声明的信号。
3. ``WebBridge`` 上每一个 ``@Slot`` 都得写在模块文档那张契约表里——**那张表漏过一次**
   （``setWallpaper``），而它是"改哪边要动哪边"的唯一索引（见 ``chat_view.py`` 开头）。

**不构造任何 Qt 对象**，所以这条腿在没装 PySide6 的机器上照跑。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "axo" / "ui" / "chat_view.py"

BRIDGE = "WebBridge"
VIEW = "ChatView"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def _klass(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"chat_view.py 里找不到 class {name}")


def _members(node: ast.ClassDef) -> set[str]:
    """类体里声明的名字：赋值（信号、类变量）与方法名。"""
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(item.name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
    return names


def _touched(node: ast.AST, root: str) -> set[str]:
    """``self.<root>.<attr>`` 里所有的 ``attr``（本文件的两个类都只写一层）。"""
    found: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Attribute):
            continue
        inner = sub.value
        if (
            isinstance(inner, ast.Attribute)
            and inner.attr == root
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "self"
        ):
            found.add(sub.attr)
    return found


def _emitted(node: ast.AST) -> set[str]:
    """``self.<name>.emit(...)`` 里所有的 ``name``。"""
    found: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "emit"
            and isinstance(sub.func.value, ast.Attribute)
            and isinstance(sub.func.value.value, ast.Name)
            and sub.func.value.value.id == "self"
        ):
            found.add(sub.func.value.attr)
    return found


def test_everything_the_view_connects_exists_on_the_bridge(tree: ast.Module) -> None:
    """``ChatView`` 连的每一个 ``self._bridge.X`` 都得在 ``WebBridge`` 上真的存在。

    这一条正是 2026-10-12 那次"应用起不来"的精确形状：连的那一侧有、被连的那一侧
    没有。**四个槽里漏一个**就够——而 ``submitted`` / ``persona_answered`` /
    ``tool_confirm_decided`` 三个当时都在，所以它看上去完全正常。
    """
    bridge = _members(_klass(tree, BRIDGE))
    view = _klass(tree, VIEW)

    missing = sorted(_touched(view, "_bridge") - bridge)

    assert missing == [], (
        f"ChatView 连了 WebBridge 上没有的东西：{missing}。"
        "少声明一个信号/槽的症状是构造 ChatView 时 AttributeError——应用起不来。"
    )


def test_the_bridge_only_emits_its_own_signals(tree: ast.Module) -> None:
    """``WebBridge`` 里 ``self.X.emit()`` 的 X 必须是它**自己**声明的信号。

    **反向的那一半。** 只查"连的那一侧"的话，把信号声明到 ``ChatView`` 上、槽里发
    ``self.voice_toggled``，两边各自都"存在"，而它们在两个对象上——一样是
    AttributeError。两个方向合起来才等于"名字对得上"。
    """
    node = _klass(tree, BRIDGE)
    declared = _members(node)

    strangers = sorted(_emitted(node) - declared)

    assert strangers == [], (
        f"WebBridge 里发了不属于它自己的信号：{strangers}。"
        "信号声明在哪个类上、就在哪个 self 上发——差一层就是两个对象。"
    )


def test_every_slot_is_written_down_in_the_contract_table() -> None:
    """每个 ``@Slot`` 都要出现在模块文档那张 ``window.bridge.*`` 表里。

    **那张表漏过一次**（``setWallpaper``），而它的作用是"改哪边要动哪边"的唯一索引：
    漏一项就意味着下一次改签名时没人会想到去改对面，而 QWebChannel 找不到槽位时的
    表现是**静默丢弃**——不是报错。
    """
    source = SOURCE.read_text(encoding="utf-8")
    table = source[: source.index("**这份清单漏过一次")]
    node = _klass(ast.parse(source, filename=str(SOURCE)), BRIDGE)

    slots = [
        item.name
        for item in node.body
        if isinstance(item, ast.FunctionDef)
        and any(
            (isinstance(d, ast.Name) and d.id == "Slot")
            or (isinstance(d, ast.Attribute) and d.attr == "Slot")
            or (
                isinstance(d, ast.Call)
                and getattr(d.func, "attr", getattr(d.func, "id", "")) == "Slot"
            )
            for d in item.decorator_list
        )
    ]

    assert slots, "WebBridge 上一个 @Slot 都没找到——装饰器写法变了？"
    missing = [name for name in slots if f"window.bridge.{name}(" not in table]
    assert missing == [], f"契约表里没写这几个槽：{missing}。表在 chat_view.py 的模块文档里。"
