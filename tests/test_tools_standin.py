"""冒烟脚本里那个替身与真件对齐：**掉队的症状是冒烟脚本在离屏下抛异常。**

``tools/tools_support.py`` 的 ``RecordingChatView`` 是一份**手抄**的 ``ChatView``
接口，三个冒烟脚本（``_smoke_e2e`` / ``_smoke_recall`` / ``_smoke_memory``）靠它把真
窗口建起来而不碰 QtWebEngine。手抄的接口会掉队，而且掉得很安静：真件加一个方法、
``MainWindow`` 里接上线，替身少了那一格——于是那个冒烟脚本在那一行抛
``AttributeError``。三个脚本都是**手动** oracle，平时不跑，所以这一抛可以几个月没人
碰；等真正需要它的时候（比如刚改完启动路径，正想拿真窗口验一次）它坏在最不该坏的
时机。

**这不是假想的。** ``tool_confirm_decided``（确认闸门那张卡片）进 ``MainWindow``
之后，替身就跟丢了，``_smoke_memory.py`` 一直坏到有人去跑它。

判据取最窄的那一条：``MainWindow`` 里出现的每一个 ``chat_view.<名字>``，**若真件
``ChatView`` 自己定义了它**，替身就必须也定义。真件没定义、靠继承来的那些（``QWidget``
的）不管——替身也是 ``QWidget`` 的子类，那一半本来就成立，而"它继承到了没有"不经
import 判不了。

**只读源码的 AST，不 import 任何一边。** 那两个文件都要 Qt（``chat_view`` 还拖着
QtWebEngine），而这条守卫要留在核心 pytest 里跑。
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_VIEW = _REPO_ROOT / "src" / "axo" / "ui" / "chat_view.py"
_MAIN_WINDOW = _REPO_ROOT / "src" / "axo" / "ui" / "main_window.py"
_STANDIN = _REPO_ROOT / "tools" / "tools_support.py"

_REAL_CLASS = "ChatView"
_STANDIN_CLASS = "RecordingChatView"
#: 窗口里握着对话区的那个属性名——两侧对齐的就是它的成员。
_HOLDER = "chat_view"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _own_members(path: Path, class_name: str) -> set[str]:
    """类**自己**写的名字：方法、以及 ``Signal`` 一类的类级赋值。

    不含继承来的——那正是这个判据要区分的那一半。
    """
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    members.update(
                        target.id for target in item.targets if isinstance(target, ast.Name)
                    )
            return members
    raise AssertionError(f"{path.name} 里找不到类 {class_name}")


def _asked_of_the_holder(tree: ast.Module) -> set[str]:
    """``self.chat_view.<名字>`` 里出现的那些名字。

    只认 ``self.<_HOLDER>.<名字>`` 这个形状：``from .chat_view import ChatView`` 那行
    里的 ``chat_view.ChatView`` 是模块名不是属性，不能混进来。
    """
    asked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        holder = node.value
        if not isinstance(holder, ast.Attribute) or holder.attr != _HOLDER:
            continue
        if isinstance(holder.value, ast.Name) and holder.value.id == "self":
            asked.add(node.attr)
    return asked


def test_the_standin_covers_everything_the_window_asks_of_the_view() -> None:
    real = _own_members(_REAL_VIEW, _REAL_CLASS)
    standin = _own_members(_STANDIN, _STANDIN_CLASS)
    asked = _asked_of_the_holder(_tree(_MAIN_WINDOW))

    assert asked, f"一个 self.{_HOLDER}.<名字> 都没扫到——是判据的取法坏了，不是代码对了"
    assert real, f"{_REAL_VIEW.name} 里的 {_REAL_CLASS} 一个成员都没读到"
    assert standin, f"{_STANDIN.name} 里的 {_STANDIN_CLASS} 一个成员都没读到"

    # 真件没定义的那些（继承来的）不要求替身显式定义，所以取交集而不是整份 asked。
    missing = sorted((asked & real) - standin)
    assert missing == [], (
        f"{_STANDIN.name} 的 {_STANDIN_CLASS} 少了 {missing}——真件有、窗口在用，"
        "替身没有的话冒烟脚本会在那一行抛 AttributeError。补法是把真件的签名照抄过来。"
    )
