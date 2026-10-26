"""仓库地图：目录结构 + 多语言符号索引，一次看清"仓库长什么样"。

``search_files`` 回答的是"某段文字在哪"，这里回答的是"有哪些模块、入口和测试"。
大仓库里模型先要的是后者——先有地图，再决定读哪几个文件。

**多语言是启发式，不是编译器。** Python 用 ``ast``（精确到语法）；其余常用语言按
声明行的结构正则摘类/函数/方法。正则不会做类型推断，也不保证覆盖宏、模板、装饰器
生成的名字；它只负责"让模型知道大概去哪找"。解析不了的文件仍然列在目录里，不会
被藏掉。

**只读、有界、确定性。** 跳过噪声目录；单个文件超过 512 KiB 不解析；文件数与符号数
都有上限，截断时明确写出"还有没列的"，不假装看全。符号只取声明级，不做跨文件调用图。
"""

from __future__ import annotations

import ast
import os
import re
import threading
from pathlib import Path
from typing import Any, ClassVar

from ..llm.base import ToolSpec
from .fs import ReadFileError, resolve_within
from .snapshot import snapshot_tree
from .tools import ToolContext, ToolOutcome, argument_int, argument_str

__all__ = ["RepoMapCache", "RepoMapTool", "build_repo_map"]

_NOISE_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "node_modules",
        "target",
        "out",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".idea",
        ".vscode",
    }
)
_MAX_SCAN_FILES = 5000
"""最多扫多少个文件就停。防止在一个几十万文件的目录里把一次工具调用拖成分钟级。"""

_MAX_PARSE_BYTES = 512 * 1024
"""超过这么大的 Python 文件不解析符号（仍然列在目录里）。"""

_MAX_CHARS = 6000
"""输出上限。工具结果会进上下文，地图再全也不该挤掉人格锚。"""


def _visible_dir(name: str) -> bool:
    return name not in _NOISE_DIRS and not name.startswith(".")


def _scan(root: Path, max_files: int) -> tuple[list[Path], int, bool]:
    """按稳定顺序收集文件。返回 (列出的文件, 扫描总数, 是否截断)。"""
    listed: list[Path] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if _visible_dir(d))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            # ``read_file`` 的沙箱判据是"解析后仍在根内"（``resolve_within`` 会
            # 跟随符号链接）。这里照同一条：一个指向根外的链接文件不能被解析出符号。
            try:
                if path.is_symlink() and not path.resolve().is_relative_to(root):
                    continue
            except (OSError, ValueError):
                continue
            scanned += 1
            if len(listed) < max_files:
                listed.append(path)
            if scanned >= _MAX_SCAN_FILES:
                return listed, scanned, True
    return listed, scanned, len(listed) < scanned


# ── 多语言符号：启发式声明匹配 ──────────────────────────────────────────────
#
# 这里不做语法分析，只认"一行以什么形状开头"。代价是宏/模板/生成代码可能漏，
# 好处是零依赖、对任意大小仓库都能跑。Python 走上面的 AST，其余语言走这张表。

_SUFFIX_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".lua": "lua",
    ".r": "r",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "shell",
    ".vue": "javascript",
    ".svelte": "javascript",
    ".pl": "perl",
    ".pm": "perl",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
}

_CONTROL_WORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "foreach",
        "while",
        "switch",
        "case",
        "default",
        "catch",
        "finally",
        "try",
        "do",
        "return",
        "throw",
        "throws",
        "new",
        "delete",
        "sizeof",
        "typeof",
        "instanceof",
        "in",
        "of",
        "await",
        "yield",
        "with",
        "using",
        "lock",
        "fixed",
        "unsafe",
        "checked",
        "unchecked",
        "when",
        "match",
        "loop",
        "select",
        "defer",
        "go",
    }
)

_JS_PATTERNS = (
    (
        "class",
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)",
    ),
    (
        "type",
        r"^\s*(?:export\s+)?(?:default\s+)?(?:interface|type|enum|namespace)\s+([A-Za-z_$][\w$]*)",
    ),
    (
        "function",
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(",
    ),
    (
        "function",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
    ),
    (
        "method",
        r"^\s*(?:(?:public|private|protected|static|async|get|set|readonly|override|abstract)\s+)+"
        r"([A-Za-z_$][\w$]*)\s*\(",
    ),
    ("method", r"^\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[^{]+)?\{"),
)

_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "javascript": _JS_PATTERNS,
    "typescript": _JS_PATTERNS,
    "go": (
        ("function", r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
        ("type", r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b"),
    ),
    "rust": (
        (
            "function",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?"
            r"fn\s+([A-Za-z_]\w*)",
        ),
        (
            "type",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|union)\s+([A-Za-z_]\w*)",
        ),
        ("impl", r"^\s*impl(?:<[^>]*>)?\s+([A-Za-z_]\w*)"),
        ("mod", r"^\s*(?:pub\s+)?mod\s+([A-Za-z_]\w*)"),
    ),
    "java": (
        (
            "type",
            r"^\s*(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed|"
            r"strictfp)\s+)*\b(?:class|interface|enum|record|@interface)\s+([A-Za-z_]\w*)",
        ),
        (
            "constructor",
            r"^\s*(?:public|private|protected)\s+([A-Z]\w*)\s*\([^;]*\)\s*\{",
        ),
        (
            "method",
            r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|native|"
            r"default|strictfp)\s+)*[\w<>\[\].?,]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*"
            r"(?:throws [\w, .]+)?\{",
        ),
    ),
    "kotlin": (
        (
            "type",
            r"^\s*(?:(?:public|private|protected|internal|open|abstract|sealed|data|value|"
            r"annotation|inline|enum|inner|companion)\s+)*\b(?:class|interface|object)\s+"
            r"([A-Za-z_]\w*)",
        ),
        (
            "function",
            r"^\s*(?:(?:public|private|protected|internal|open|abstract|inline|suspend|"
            r"operator|override|tailrec|external|expect|actual|const|lateinit)\s+)*fun\s+"
            r"(?:<[^>]+>\s*)?(?:[\w.]+\.)?([A-Za-z_]\w*)\s*\(",
        ),
    ),
    "csharp": (
        (
            "type",
            r"^\s*(?:(?:public|private|protected|internal|static|sealed|abstract|partial|"
            r"readonly|record)\s+)*\b(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)",
        ),
        (
            "constructor",
            r"^\s*(?:public|private|protected|internal)\s+([A-Z]\w*)\s*\([^;]*\)\s*\{",
        ),
        (
            "method",
            r"^\s*(?:(?:public|private|protected|internal|static|virtual|override|abstract|"
            r"async|sealed|new|partial|extern|unsafe|readonly)\s+)*[\w<>\[\].?,]+\s+"
            r"([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:where [^{]+)?\{",
        ),
    ),
    "c": (
        ("namespace", r"^\s*namespace\s+([A-Za-z_]\w*)"),
        ("type", r"^\s*(?:class|struct|union|enum)\s+([A-Za-z_]\w*)"),
        (
            "function",
            r"^\s*(?:template\s*<[^>]*>\s*)?(?:[\w:*&<>,~\[\]]+\s+)+"
            r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)\s*\([^;]*\)\s*(?:const)?\s*(?:\{|$)",
        ),
    ),
    "cpp": (
        ("namespace", r"^\s*namespace\s+([A-Za-z_]\w*)"),
        ("type", r"^\s*(?:class|struct|union|enum)\s+([A-Za-z_]\w*)"),
        (
            "function",
            r"^\s*(?:template\s*<[^>]*>\s*)?(?:[\w:*&<>,~\[\]]+\s+)+"
            r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)\s*\([^;]*\)\s*(?:const)?\s*(?:\{|$)",
        ),
    ),
    "ruby": (
        ("method", r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)"),
        ("type", r"^\s*(?:class|module)\s+([A-Z]\w*)"),
    ),
    "php": (
        (
            "type",
            r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+([A-Za-z_]\w*)",
        ),
        (
            "function",
            r"^\s*(?:(?:public|private|protected|static|final|abstract|async)\s+)*function\s+"
            r"([A-Za-z_]\w*)\s*\(",
        ),
    ),
    "swift": (
        (
            "type",
            r"^\s*(?:(?:public|private|fileprivate|internal|open|static|class|final|override|"
            r"mutating|nonmutating|async)\s+)*\b(?:class|struct|enum|protocol|extension|actor)\s+"
            r"([A-Za-z_]\w*)",
        ),
        (
            "function",
            r"^\s*(?:(?:public|private|fileprivate|internal|open|static|class|final|override|"
            r"mutating|nonmutating|async)\s+)*func\s+([A-Za-z_]\w*)",
        ),
    ),
    "scala": (
        (
            "declaration",
            r"^\s*(?:private|protected|override|final|implicit|inline|case|\s)*\b(?:def|class|"
            r"trait|object|enum)\s+([A-Za-z_]\w*)",
        ),
    ),
    "perl": (
        ("function", r"^\s*sub\s+([A-Za-z_]\w*)"),
        ("package", r"^\s*package\s+([A-Za-z_]\w*)"),
    ),
    "lua": (
        ("function", r"^\s*(?:local\s+)?function\s+([A-Za-z_][\w.:]*)"),
        ("function", r"^\s*([A-Za-z_]\w*)\s*=\s*function\s*\("),
    ),
    "r": (("function", r"^\s*([A-Za-z_.][\w.]*)\s*<-\s*function\s*\("),),
    "shell": (
        ("function", r"^\s*function\s+([A-Za-z_][\w-]*)\s*\{?"),
        ("function", r"^\s*([A-Za-z_]\w*)\s*\(\)\s*\{?"),
    ),
    "dart": (
        (
            "type",
            r"^\s*(?:abstract\s+|base\s+|final\s+|sealed\s+|interface\s+|mixin\s+)?"
            r"(?:class|enum|mixin|extension|typedef)\s+([A-Za-z_]\w*)",
        ),
        (
            "function",
            r"^\s*(?:[\w<>?\[\]]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:async\s*)?\{",
        ),
    ),
    "elixir": (
        ("module", r"^\s*defmodule\s+([A-Z]\w*)"),
        (
            "function",
            r"^\s*def(?:p|macro|macrop)?\s+([A-Za-z_]\w*[?!]?)",
        ),
    ),
    "haskell": (
        ("type", r"^\s*(?:data|newtype|type|class)\s+([A-Z]\w*)"),
        ("function", r"^\s*([a-z_]\w*)\s*::"),
    ),
    "clojure": (
        ("function", r"^\s*\(defn-?\s+([\w\-?!*+<>=]+)"),
        (
            "declaration",
            r"^\s*\(def(?:macro|protocol|record|multi|once|struct)?\s+([\w\-?!*+<>=]+)",
        ),
    ),
}

_COMPILED = {
    language: tuple((label, re.compile(pattern)) for label, pattern in patterns)
    for language, patterns in _PATTERNS.items()
}


def _regex_symbols(source: str, language: str) -> list[str]:
    """按语言表逐行匹配声明。**只认行首形状**，不做语法分析。"""
    compiled = _COMPILED.get(language, ())
    if not compiled:
        return []
    lines: list[str] = []
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "--")):
            continue
        for label, pattern in compiled:
            match = pattern.match(line)
            if not match:
                continue
            name = (match.group(1) or "").strip()
            if not name or name in _CONTROL_WORDS:
                continue
            lines.append(f"  {label} {name} (L{number})")
            break
    return lines


def _python_symbols(source: str, path: Path) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        where = f"第 {exc.lineno} 行" if exc.lineno else "未知位置"
        return [f"（语法错误，{where}）"]
    except (ValueError, RecursionError) as exc:
        return [f"（无法解析：{exc}）"]

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"  {_signature(node)} (L{node.lineno})")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"  {_class_line(node)} (L{node.lineno})")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append(f"    {_signature(child)} (L{child.lineno})")
    return lines


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except (ValueError, RecursionError):
        args = "..."
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = ""
    if node.returns is not None:
        try:
            returns = " -> " + ast.unparse(node.returns)
        except (ValueError, RecursionError):
            returns = ""
    return f"{keyword} {node.name}({args}){returns}"


def _class_line(node: ast.ClassDef) -> str:
    bases = ""
    if node.bases:
        try:
            bases = "(" + ", ".join(ast.unparse(base) for base in node.bases) + ")"
        except (ValueError, RecursionError):
            bases = ""
    return f"class {node.name}{bases}"


def _symbols(path: Path) -> list[str]:
    """按后缀选语言，摘声明行。读不了/超限/语法错都返回一句说明。"""
    language = _SUFFIX_LANGUAGE.get(path.suffix.lower())
    if language is None:
        return []
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"（读不到：{exc}）"]
    if size > _MAX_PARSE_BYTES:
        return [f"（文件 {size} 字节，超过解析上限）"]
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"（读不到：{exc}）"]
    if language == "python":
        return _python_symbols(source, path)
    return _regex_symbols(source, language)


def _size_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _clip(text: str, limit: int = _MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    note = "\n…（仓库地图已截断）"
    return text[: max(0, limit - len(note))] + note


def build_repo_map(
    root: Path,
    *,
    max_files: int = 120,
    max_symbols: int = 120,
) -> str:
    """返回一张有界的仓库地图；``root`` 必须是已授权的目录。"""
    root = root.expanduser().resolve()
    max_files = max(1, min(max_files, 400))
    max_symbols = max(0, min(max_symbols, 400))
    files, scanned, truncated = _scan(root, max_files)

    lines = [
        f"仓库地图：{root.name or root}",
        f"目录：{root}",
        f"扫描 {scanned} 个文件，列出 {len(files)} 个"
        + ("（文件列表已截断）" if truncated else ""),
        "",
    ]
    symbol_count = 0
    stopped_symbols = False
    for path in files:
        try:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        lines.append(f"{relative} ({_size_text(size)})")
        if _SUFFIX_LANGUAGE.get(path.suffix.lower()) is None or stopped_symbols or max_symbols == 0:
            # max_symbols=0 是"只看目录"：连源码都不读，省得为一个必被丢掉的结果付 IO。
            continue
        symbols = _symbols(path)
        for symbol in symbols:
            if symbol_count >= max_symbols:
                stopped_symbols = True
                break
            lines.append(symbol)
            symbol_count += 1
        if stopped_symbols:
            lines.append(f"  …（符号已达到上限 {max_symbols}，后续文件未解析）")
    if truncated:
        lines.append(f"…（还有文件没有列出：共扫描 {scanned} 个）")
    return _clip("\n".join(lines))


_LINE_SUFFIX = re.compile(r"\(L(\d+)\)\s*$")


def iter_symbols(
    root: Path, *, max_files: int = 400, max_symbols: int = 2000
) -> tuple[list[tuple[str, int, str]], bool]:
    """把符号索引摊平成 ``(相对路径, 行号, 声明)``，供 ``find_symbol`` 检索。

    与 :func:`build_repo_map` 共用同一套语言表和上限；不生成展示文本，所以
    ``max_symbols`` 可以比地图大。返回 ``(符号, 是否截断)``。
    """
    root = root.expanduser().resolve()
    files, _scanned, truncated = _scan(root, max(1, min(max_files, 400)))
    hits: list[tuple[str, int, str]] = []
    for path in files:
        if _SUFFIX_LANGUAGE.get(path.suffix.lower()) is None:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        for line in _symbols(path):
            match = _LINE_SUFFIX.search(line)
            if not match:
                continue
            declaration = _LINE_SUFFIX.sub("", line).strip()
            hits.append((relative, int(match.group(1)), declaration))
            if len(hits) >= max_symbols:
                return hits, True
    return hits, truncated


class RepoMapCache:
    """一个小 LRU：同一根、同一参数、同一工作区快照就复用上次的地图。

    缓存键里带**快照摘要**而不是时间：文件没变就命中，改一个字就失效。快照被截断
    （文件太多只看了一部分）时**不写缓存**——那时摘要不能证明"没变"，复用会给出旧地图。
    """

    def __init__(self, max_entries: int = 8) -> None:
        self._entries: dict[tuple[str, int, int], tuple[str, str]] = {}
        self._max = max(1, max_entries)
        self._lock = threading.Lock()

    def get(self, root: Path, max_files: int, max_symbols: int, digest: str) -> str | None:
        key = (str(root), max_files, max_symbols)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None or entry[0] != digest:
                return None
            self._entries[key] = entry  # 最近用过的放回末尾（LRU）
            return entry[1]

    def put(self, root: Path, max_files: int, max_symbols: int, digest: str, text: str) -> None:
        key = (str(root), max_files, max_symbols)
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (digest, text)
            while len(self._entries) > self._max:
                oldest = next(iter(self._entries))
                del self._entries[oldest]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class RepoMapTool:
    """只读工具：把目录树与 Python 符号压成一张有界地图。**不过闸门。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="repo_map",
        description=(
            "查看允许目录的仓库地图：目录、文件大小、常见语言的类/函数/方法与行号"
            "（Python 用 AST，JS/TS/Go/Rust/Java/C/C++/C#/Ruby/PHP/Swift/Kotlin/Scala/"
            "Lua/R/Shell/Dart/Elixir/Haskell/Clojure 用声明行匹配）。"
            "开始不熟悉的仓库时先用它决定读哪些文件，比逐个 list/read 省轮次。"
            "截断时会写明；需要某个文件全文仍用 read_file。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对可读目录的路径；默认第一个。"},
                "max_files": {
                    "type": "integer",
                    "description": "最多列几个文件，1–400，默认 120。",
                },
                "max_symbols": {
                    "type": "integer",
                    "description": "最多列几个符号，0–400，默认 120；0 表示只看目录。",
                },
            },
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if not ctx.read_root:
            return ToolOutcome.failure("没有配置可读目录。")
        try:
            root = resolve_within(
                ctx.read_root, argument_str(arguments, "path", required=False) or "."
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            return ToolOutcome.failure(f"{root} 不是目录。")
        max_files = argument_int(arguments, "max_files", default=120, minimum=1, maximum=400)
        max_symbols = argument_int(arguments, "max_symbols", default=120, minimum=0, maximum=400)
        assert max_files is not None and max_symbols is not None
        # 只有装了缓存才付这次扫描：没缓存时多走一遍纯属浪费，而大仓库上它很明显。
        cache = ctx.repo_map_cache
        snapshot = snapshot_tree(root) if cache is not None else None
        cacheable = bool(
            cache is not None
            and snapshot is not None
            and snapshot.digest is not None
            and not snapshot.truncated
        )
        if cacheable and cache is not None and snapshot is not None:
            assert snapshot.digest is not None
            cached = cache.get(root, max_files, max_symbols, snapshot.digest)
            if cached is not None:
                return ToolOutcome.success(cached, replayable=False)
        try:
            body = build_repo_map(root, max_files=max_files, max_symbols=max_symbols)
        except OSError as exc:
            return ToolOutcome.failure(f"生成仓库地图失败：{exc}")
        if cacheable and cache is not None and snapshot is not None:
            assert snapshot.digest is not None
            cache.put(root, max_files, max_symbols, snapshot.digest, body)
        return ToolOutcome.success(body, replayable=False)


class FindSymbolTool:
    """按名字搜符号索引。**只读、不过闸门；不是 LSP。**"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="find_symbol",
        description=(
            "在允许目录的符号索引里按名字搜索类/函数/方法（多语言，大小写不敏感），"
            "返回文件、行号与声明。适合“某个函数在哪定义”。它是静态索引，不是 LSP；"
            "找不到时再用 search_files 搜文本。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "要搜索的符号名或其中一段。"},
                "path": {"type": "string", "description": "相对可读目录的路径；默认第一个。"},
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条，1–50，默认 20。",
                },
            },
            "required": ["name"],
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if not ctx.read_root:
            return ToolOutcome.failure("没有配置可读目录。")
        query = argument_str(arguments, "name").strip()
        if not query:
            return ToolOutcome.failure("name 是空的。")
        max_results = argument_int(arguments, "max_results", default=20, minimum=1, maximum=50)
        assert max_results is not None
        try:
            root = resolve_within(
                ctx.read_root, argument_str(arguments, "path", required=False) or "."
            )
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            return ToolOutcome.failure(f"{root} 不是目录。")
        try:
            hits, truncated = iter_symbols(root)
        except OSError as exc:
            return ToolOutcome.failure(f"搜索符号失败：{exc}")
        needle = query.casefold()
        matched = [hit for hit in hits if needle in hit[2].casefold()]
        if not matched:
            return ToolOutcome.success(
                f"没有找到包含 {query!r} 的符号（索引了 {len(hits)} 个）。"
                "可以换更短的词，或用 search_files 搜文本。",
                replayable=False,
            )
        shown = matched[:max_results]
        lines = [f"找到 {len(matched)} 个符号（显示前 {len(shown)} 个）："]
        lines.extend(f"{path}:{line}: {text}" for path, line, text in shown)
        if len(matched) > len(shown):
            lines.append(f"…（还有 {len(matched) - len(shown)} 个；缩小 name 或加 path）")
        if truncated:
            lines.append("（符号索引已截断；结果不是全仓库全集。）")
        return ToolOutcome.success("\n".join(lines), replayable=False)
