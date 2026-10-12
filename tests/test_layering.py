"""分层纯净度。

这个测试守护的是整个项目最关键的一条设计约束：

    **memcore 不许知道世界上有 Qt、有 HTTP、有大模型。**

它为什么值得一个专门的测试文件，而不是靠代码评审：这条约束的失效方式是**静默**
的。某天有人为了图方便，在 ``memcore/recall.py`` 顶上写一句
``from ..llm.ollama import OllamaEmbeddingBackend`` 来"顺手"取个向量——代码照样
跑，测试照样绿，但记忆核心从此绑定了一个具体的模型后端，"换模型 = 换读者"的
承诺在那一刻就已经破了，而且没有任何症状。

所以这里用 ``ast`` 遍历真实的 import 语句来判定，而不是正则搜字符串：正则会把
文档里提到的名字也算成违规（本文件自己就在反复讨论这些名字），而 ast 只认真正的
导入。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
_MEMCORE = _SRC / "memcore"

#: memcore 允许 import 的顶层模块。**加东西进来必须有理由**，因为每加一个
#: 就多一份"记忆核心要靠它才能跑"的依赖。
_ALLOWED_TOP_LEVEL = {
    "numpy",  # 唯一的重依赖，且它是纯计算的
    "biosnn_bus",  # 骨架库：脉冲类型与插件协议，见 ADR-0002
    # 以下全是标准库
    "dataclasses",
    "enum",
    "errors",
    "functools",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "sqlite3",
    "threading",  # 记忆库的连接要跨线程用，那把可重入锁是它的前提
    "time",
    "typing",
    "uuid",
    "warnings",
    "collections",
    "contextlib",
    "hashlib",
    "random",
    "shutil",
    "tempfile",
    "textwrap",
    "re",
    "struct",
    "sys",
    "abc",
    "copy",
    "io",
    "logging",
    "operator",
    "string",
    "types",
    "weakref",
    "base64",
    "binascii",
    "unicodedata",
    "zlib",
    "gzip",
    "glob",
    "traceback",
    "heapq",
    "bisect",
    "array",
    "statistics",
    "datetime",
    "secrets",
    "__future__",
    "memcore",
}

#: 出现在 memcore 源码里的禁用词（代码部分，注释与文档字符串不算）。
_FORBIDDEN = (
    "PySide6",
    "PyQt",
    "httpx",
    "requests",
    "aiohttp",
    "ollama",
    "openai",
    "anthropic",
    "torch",
    "transformers",
)

#: 必须保持"没有界面、没有网络"的部分。路径相对 ``src/``，可以是包也可以是单个
#: 文件——``llm/`` 整包不能进来，因为 ``ollama.py`` 本来就该 import httpx，而
#: ``base.py``（纯类型）不该。
#:
#: **这不是洁癖，是测试速度与确定性。** 一旦 ``session/`` 里出现一次 httpx 调用，
#: 装配的测试就会需要网络与一个跑着的模型，而需要网络的测试最后总是被跳过——那时
#: "装配器在超额时先丢谁"就再也没有断言了。``assemble``/``budget`` 的模块文档里
#: 都写着"不碰 Qt、不碰网络，所以能单测"，这一条就是那句承诺的执法者。
#:
#: ``agent/`` 从 P3a 起也在这里（此前这段注释预告的是"P3 的 agent 要加进来"）。
#: 它比 ``session/`` 宽松一点——要用 ``datetime``、要读文件——但**边界同样是
#: "不 import PySide6 / httpx"**：Ollama 后端靠注入（循环收一个 ``ChatBackend``），
#: 于是工具循环能在核心腿里用假后端逐事件断言，一行网络都不用碰。这一层一旦
#: import 了 ``httpx``，那些断言会集体变成"需要 Ollama 在跑"，而需要网络的测试
#: 最后总是被跳过。
#:
#: 下面四个 ``ui/`` 下的**单个文件**（不是整个 ``ui/`` 包）是另一种情况：
#: ``ui/`` 当然会 import Qt，``main_window.py`` 不 import 才怪。但这四个搬出来
#: 就是为了不碰 Qt——配色数学、主题数据、设置存取、尺寸算术，它们要能在无 Qt 的
#: 测试里逐条断言。列成单个文件而不是整包，正是为了让"哪些文件必须干净"是一个
#: 可以逐行读的事实，而不是一句口号。
_PURE_LAYERS = (
    "axo/agent",
    "axo/bench",
    "axo/llm/base.py",
    "axo/memory",
    # 密钥加解密（Windows DPAPI）。放进来是为了让「平台相关的代码集中在一个模块」
    # 是一条**可断言**的事实：它只 import ctypes/base64，不碰 Qt、也不碰 httpx。
    "axo/secrets.py",
    "axo/session",
    "axo/ui/color.py",
    "axo/ui/theme.py",
    "axo/ui/settings_store.py",
    "axo/ui/layout.py",
    # 搬迁算法。列进来是因为它**是这一族里唯一会动用户盘的东西**：复制整棵数据根、
    # 按哈希逐条校验、失败时回滚。这种代码必须能在无 Qt 的测试里把每一种拒绝与每
    # 一种失败都摆出来，而"它自己不 import Qt"是可断言的前提。
    "axo/ui/data_move.py",
)

_PURE_FORBIDDEN = ("PySide6", "PyQt5", "PyQt6", "httpx", "requests")


def _layer_files(relative: str) -> list[Path]:
    target = _SRC / relative
    if target.is_dir():
        return sorted(target.rglob("*.py"))
    return [target] if target.is_file() else []


def _memcore_files() -> list[Path]:
    return sorted(_MEMCORE.rglob("*.py"))


def test_memcore_has_files() -> None:
    """自检：遍历不到文件说明路径错了，后面的断言会全部空转通过。"""
    assert _memcore_files(), f"在 {_MEMCORE} 下没找到任何 .py——测试自身失效了。"


@pytest.mark.parametrize("path", _memcore_files(), ids=lambda p: p.name)
def test_imports_are_within_whitelist(path: Path) -> None:
    """memcore 的每一条 import 都在白名单里。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_TOP_LEVEL:
                    offenders.append(f"第 {node.lineno} 行：import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue  # 相对导入（from .errors import ...）总是包内，天然合法
            root = (node.module or "").split(".")[0]
            if root and root not in _ALLOWED_TOP_LEVEL:
                offenders.append(f"第 {node.lineno} 行：from {node.module} import ...")

    assert not offenders, (
        f"{path.relative_to(_SRC)} 引入了白名单之外的模块：\n  "
        + "\n  ".join(offenders)
        + "\n\nmemcore 只能依赖 numpy、biosnn-bus 与标准库。若确实需要新依赖，"
        "先想清楚它会不会让记忆核心从此绑上某个具体实现——多数情况下，答案是"
        "这行代码应该写在 axo 层，然后把结果作为参数传进来。"
    )


@pytest.mark.parametrize("path", _memcore_files(), ids=lambda p: p.name)
def test_no_forbidden_words_in_code(path: Path) -> None:
    """源码里不出现禁用词。

    只查**代码**，不查注释与文档字符串。理由：``memcore`` 的文档里不得不提到
    "embedding 来自某个模型"，那是概念层面的说明，不是依赖。要区分这两者，
    最省事的做法是逐行判断该行是否落在某个字符串字面量之外——这里用
    ``ast`` 收集所有字符串字面量的行号，然后只检查其余的行。
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    # 只豁免**真正的 docstring**（模块/类/函数的首条字符串），不豁免所有字符串。
    #
    # 这个区别是关键。docstring 里不得不提到"embedding 来自某个模型"——那是概念
    # 说明，不是依赖。但**其它**字符串字面量是代码：``importlib.import_module
    # ("ollama")`` 这句话落在普通字符串里，它和 import 语句一样致命。把两者一起
    # 豁免，等于给"绕过这条测试"留了一扇后门。
    literal_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    excused_lines: set[int] = set()
    for node in literal_nodes:
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = getattr(first.value, "end_lineno", first.value.lineno)
            excused_lines.update(range(first.value.lineno, end + 1))

    offenders: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if number in excused_lines:
            continue
        code_only = line.split("#", 1)[0]
        for word in _FORBIDDEN:
            if word in code_only:
                offenders.append(f"第 {number} 行出现了 {word!r}：{code_only.strip()}")

    assert not offenders, f"{path.relative_to(_SRC)} 的代码里出现了禁用词：\n  " + "\n  ".join(
        offenders
    )


# ── 纯层：不碰界面、不碰网络 ────────────────────────────────────────────────


def _pure_files() -> list[tuple[str, Path]]:
    return [(relative, path) for relative in _PURE_LAYERS for path in _layer_files(relative)]


def test_every_pure_layer_exists() -> None:
    """自检：路径写错的话下面的断言会全部空转通过，而空转的守卫比没有守卫更糟
    ——它会让人以为有人在看。"""
    for relative in _PURE_LAYERS:
        assert _layer_files(relative), f"{relative} 下没有任何 .py，这条规则在空转。"


@pytest.mark.parametrize("relative,path", _pure_files(), ids=lambda v: str(v).replace("\\", "/"))
def test_pure_layers_do_not_import_qt_or_http(relative: str, path: Path) -> None:
    """``session/``、``memory/``、``llm/base.py`` 不 import Qt 也不 import HTTP。

    它们的模块文档都写着"不碰 Qt、不碰网络，所以能单测"——**没被执法的承诺会烂
    掉**，而烂掉的方式很安静：某天为了"顺手"取个模型名，``assemble.py`` 顶上多一句
    ``import httpx``，装配的测试从此需要网络，然后在某个 CI 上被整体跳过。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in _PURE_FORBIDDEN:
                offenders.append(f"第 {node.lineno} 行：{name}")

    assert not offenders, (
        f"{path.relative_to(_SRC)} 引入了界面或网络依赖：\n  "
        + "\n  ".join(offenders)
        + "\n\n这一层要能脱离 PySide6 与 httpx 单独测。真需要模型或界面时，把结果"
        "作为参数传进来——``assemble`` 收的是 ``MemoryInjection``，正是为此。"
    )
