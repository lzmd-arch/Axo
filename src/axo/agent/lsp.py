"""LSP 桥：可选的语义代码智能，没有服务器时明确回退。

== 它解决什么 ==

`repo_map` / `find_symbol` 是**声明索引**：按名字和行首形状找符号。它们不知道
import、别名、重载、泛型或作用域，所以三个文件都有 `normalize` 时会全列出来。
LSP（Language Server Protocol）让真正的语言服务器回答“这一行这个位置指向哪个
定义”“谁引用了它”“编译器看到了什么诊断”。

== 为什么是可选层 ==

LSP 服务器不在本仓里，也不该被自动安装：它们要单独装、占内存、可能执行构建脚本。
本模块只做三件事：按语言找到用户**已经装好**的服务器、用 stdio JSON-RPC 问它、
把结果压成模型能读的短文本。找不到服务器就返回一句“用 find_symbol/repo_map 回退”，
不假装有语义能力。

服务器命令默认从 PATH 找；也可以用环境变量覆盖：

    AXO_LSP_PYTHON="pyright-langserver --stdio"
    AXO_LSP_TYPESCRIPT="typescript-language-server --stdio"

键是语言 id 大写、连字符换下划线。**没有设置界面**：这是给愿意自己装服务器的人用的
高级层；默认行为不变。

== 安全边界 ==

* 只启动白名单里的服务器命令；环境变量覆盖是用户显式写的，不是模型能改的。
* 子进程环境是白名单（`_lsp_env`），不继承 API key。
* 只读操作：definition/references/documentSymbol/diagnostics，不调用 rename/codeAction。
* 服务器本身仍以当前用户权限运行；不要配置不可信服务器。
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from ..llm.base import ToolSpec
from .fs import ReadFileError, resolve_within
from .shell import _minimal_env
from .tools import ToolContext, ToolOutcome, argument_int, argument_str

__all__ = ["LspClient", "LspError", "LspManager", "LspTool"]

_TIMEOUT_S = 10.0
_DIAGNOSTIC_TIMEOUT_S = 8.0
_MAX_DOC_BYTES = 2 * 1024 * 1024
_MAX_RESULTS = 50

_LANGUAGE_BY_SUFFIX = {
    ".py": ("python", "python"),
    ".pyi": ("python", "python"),
    ".js": ("javascript", "javascript"),
    ".mjs": ("javascript", "javascript"),
    ".cjs": ("javascript", "javascript"),
    ".jsx": ("javascript", "javascriptreact"),
    ".ts": ("typescript", "typescript"),
    ".tsx": ("typescript", "typescriptreact"),
    ".mts": ("typescript", "typescript"),
    ".cts": ("typescript", "typescript"),
    ".go": ("go", "go"),
    ".rs": ("rust", "rust"),
    ".java": ("java", "java"),
    ".kt": ("kotlin", "kotlin"),
    ".kts": ("kotlin", "kotlin"),
    ".cs": ("csharp", "csharp"),
    ".c": ("c", "c"),
    ".h": ("c", "c"),
    ".cc": ("cpp", "cpp"),
    ".cpp": ("cpp", "cpp"),
    ".cxx": ("cpp", "cpp"),
    ".hpp": ("cpp", "cpp"),
    ".hh": ("cpp", "cpp"),
    ".hxx": ("cpp", "cpp"),
    ".rb": ("ruby", "ruby"),
    ".php": ("php", "php"),
    ".swift": ("swift", "swift"),
    ".scala": ("scala", "scala"),
    ".sc": ("scala", "scala"),
    ".lua": ("lua", "lua"),
    ".dart": ("dart", "dart"),
    ".ex": ("elixir", "elixir"),
    ".exs": ("elixir", "elixir"),
    ".vue": ("vue", "vue"),
    ".svelte": ("svelte", "svelte"),
    ".sh": ("shell", "shellscript"),
    ".bash": ("shell", "shellscript"),
    ".zsh": ("shell", "shellscript"),
    ".r": ("r", "r"),
    ".pl": ("perl", "perl"),
    ".pm": ("perl", "perl"),
    ".hs": ("haskell", "haskell"),
    ".clj": ("clojure", "clojure"),
    ".cljs": ("clojure", "clojure"),
    ".cljc": ("clojure", "clojure"),
    ".ps1": ("powershell", "powershell"),
}

_DEFAULT_SERVERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "python": (("pyright-langserver", "--stdio"), ("pylsp",)),
    "javascript": (("typescript-language-server", "--stdio"),),
    "typescript": (("typescript-language-server", "--stdio"),),
    "go": (("gopls",),),
    "rust": (("rust-analyzer",),),
    "java": (("jdtls",),),
    "kotlin": (("kotlin-language-server",),),
    "csharp": (("csharp-ls",), ("omnisharp",)),
    "c": (("clangd",),),
    "cpp": (("clangd",),),
    "ruby": (("solargraph", "stdio"),),
    "php": (("intelephense", "--stdio"), ("phpactor", "language-server")),
    "swift": (("sourcekit-lsp",),),
    "scala": (("metals",),),
    "lua": (("lua-language-server",),),
    "dart": (("dart", "language-server"),),
    "elixir": (("language_server.sh",),),
    "haskell": (("haskell-language-server-wrapper", "--lsp"),),
    "r": (("R", "--no-save", "-e", "languageserver::run()"),),
    "perl": (("perl", "-MPerl::LanguageServer", "-e", "Perl::LanguageServer::run"),),
    "shell": (("bash-language-server", "start"),),
    "vue": (("vue-language-server", "--stdio"),),
    "svelte": (("svelteserver", "--stdio"),),
    "clojure": (("clojure-lsp",),),
    "powershell": (("PowerShellEditorServices",),),
}

_LSP_ENV_KEYS = (
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "NODE_PATH",
    "JAVA_HOME",
    "GOROOT",
    "GOPATH",
    "RUSTUP_HOME",
    "CARGO_HOME",
    "DOTNET_ROOT",
)


class LspError(RuntimeError):
    """语言服务器启动、超时或返回错误。"""


def _lsp_env() -> dict[str, str]:
    env = _minimal_env()
    env.update({key: os.environ[key] for key in _LSP_ENV_KEYS if key in os.environ})
    return env


def _user_bin_dirs() -> tuple[Path, ...]:
    """常见用户级安装目录。服务器不一定要进系统 PATH。"""
    home = Path.home()
    dirs = [
        home / ".cargo" / "bin",
        home / "go" / "bin",
        home / ".dotnet" / "tools",
        home / ".local" / "bin",
        Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / "Programs",
    ]
    if platform.system() == "Darwin":
        # .app 从 Finder 启动时 PATH 里没有 Homebrew；语言服务器基本都装在这里。
        dirs.extend(
            (
                Path("/opt/homebrew/bin"),
                Path("/opt/homebrew/sbin"),
                Path("/opt/homebrew/opt/llvm/bin"),
                Path("/usr/local/bin"),
                Path("/usr/local/sbin"),
                Path("/usr/local/opt/llvm/bin"),
            )
        )
    # 便携工具链不一定在系统 PATH；用分号/冒号分隔的显式目录列表补进来。
    for raw in os.environ.get("AXO_LSP_BIN_DIRS", "").split(os.pathsep):
        if raw.strip():
            dirs.append(Path(raw.strip()))
    return tuple(dirs)


def _resolve_executable(token: str) -> str:
    candidate = Path(token).expanduser()
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(token)
    if found:
        return found
    suffixes = (".exe", ".cmd", ".bat", "") if os.name == "nt" else ("", ".exe", ".cmd")
    for directory in _user_bin_dirs():
        for suffix in suffixes:
            path = directory / f"{token}{suffix}"
            if path.is_file():
                return str(path)
    return ""


def _launch_command(parts: tuple[str, ...] | list[str]) -> list[str] | None:
    if not parts:
        return None
    resolved = _resolve_executable(parts[0])
    if not resolved:
        return None
    command = [resolved, *parts[1:]]
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        command = [comspec, "/c", *command]
    return command


def _configured_command(language: str) -> list[str] | None:
    key = "AXO_LSP_" + language.upper().replace("-", "_")
    configured = os.environ.get(key, "").strip()
    if not configured:
        return None
    try:
        parts = [
            token.strip().strip('"').strip("'") for token in shlex.split(configured, posix=False)
        ]
    except ValueError:
        return None
    return _launch_command([part for part in parts if part])


@functools.lru_cache(maxsize=1)
def _npm_global_root() -> Path | None:
    npm = _resolve_executable("npm")
    if not npm:
        return None
    try:
        proc = subprocess.run(
            f'"{npm}" root -g',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=_minimal_env() if platform.system() == "Darwin" else None,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip())


def _jdtls_command() -> list[str] | None:
    """便携 jdtls：AXO_JDTLS_HOME 指向解压目录，直接调 java，绕开 .bat 的 pause。"""
    home = os.environ.get("AXO_JDTLS_HOME", "").strip()
    if not home:
        return None
    base = Path(home).expanduser()
    jars = sorted((base / "plugins").glob("org.eclipse.equinox.launcher_*.jar"))
    if not jars:
        return None
    system = platform.system()
    config_name = (
        "config_win"
        if system == "Windows"
        else "config_mac"
        if system == "Darwin"
        else "config_linux"
    )
    config = base / config_name
    if not config.is_dir():
        return None
    return [
        "java",
        "-Declipse.application=org.eclipse.jdt.ls.core.id1",
        "-Dosgi.bundles.defaultStartLevel=4",
        "-Declipse.product=org.eclipse.jdt.ls.core.product",
        "-Dosgi.checkConfiguration=true",
        f"-Dosgi.sharedConfiguration.area={config}",
        "-Dosgi.sharedConfiguration.area.readOnly=true",
        "-Dosgi.configuration.cascaded=true",
        "-Xms256m",
        "--add-modules=ALL-SYSTEM",
        "--add-opens",
        "java.base/java.util=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.lang=ALL-UNNAMED",
        "-jar",
        str(jars[0]),
        "-data",
        "{jdtls_data}",
    ]


def _default_command(language: str) -> list[str] | None:
    if language == "java":
        portable = _jdtls_command()
        if portable is not None:
            return portable
    for parts in _DEFAULT_SERVERS.get(language, ()):
        command = _launch_command(parts)
        if command is not None:
            return command
    return None


def _uri_for(path: Path) -> str:
    return path.resolve().as_uri()


def _path_for(uri: str) -> Path:
    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def _utf16_column(line: str, character: int) -> int:
    """把 Python 字符串下标换算成 LSP 的 UTF-16 列号。"""
    return len(line[:character].encode("utf-16-le")) // 2


def _read_document(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LspError(f"读不到 {path.name}：{exc}") from exc
    if size > _MAX_DOC_BYTES:
        raise LspError(f"{path.name} 超过 LSP 单文件上限（{_MAX_DOC_BYTES} 字节）。")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LspError(f"读不到 {path.name}：{exc}") from exc


class LspClient:
    """一个语言服务器进程。**长生命周期**，由 :class:`LspManager` 缓存。"""

    def __init__(self, command: list[str], root: Path, language_id: str) -> None:
        self.command = command
        self.root = root
        self.language_id = language_id
        self._proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_lsp_env(),
            **(
                {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                if os.name == "nt"
                else {}
            ),
        )
        self._write_lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._next_id = 0
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._diagnostic_events: dict[str, threading.Event] = {}
        self._documents: dict[str, tuple[int, str]] = {}
        self._stderr: deque[str] = deque(maxlen=80)
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()
        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    # ── 协议 ────────────────────────────────────────────────────────────────

    def _stderr_loop(self) -> None:
        assert self._proc.stderr is not None
        for raw in iter(self._proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._stderr.append(line)

    def _read_message(self) -> dict[str, Any] | None:
        assert self._proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            key, _, value = line.decode("ascii", "replace").partition(":")
            headers[key.strip().lower()] = value.strip()
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            return None
        if length <= 0:
            return None
        body = self._proc.stdout.read(length)
        if not body:
            return None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _read_loop(self) -> None:
        while not self._closed:
            message = self._read_message()
            if message is None:
                break
            if "id" in message and ("result" in message or "error" in message):
                pending = self._pending.pop(int(message["id"]), None)
                if pending is not None:
                    event, holder = pending
                    holder.update(message)
                    event.set()
                continue
            method = message.get("method")
            if "id" in message and method:
                result: Any = None
                if method == "workspace/configuration":
                    items = message.get("params", {}).get("items", [])
                    result = [None] * (len(items) if isinstance(items, list) else 0)
                elif method == "workspace/workspaceFolders":
                    result = [{"uri": self.root.as_uri(), "name": self.root.name}]
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})
                continue
            if method == "textDocument/publishDiagnostics":
                params = message.get("params", {})
                uri = str(params.get("uri", ""))
                diagnostics = params.get("diagnostics", [])
                if uri:
                    self._diagnostics[uri] = diagnostics if isinstance(diagnostics, list) else []
                    event = self._diagnostic_events.get(uri)
                    if event is not None:
                        event.set()
        for event, _holder in list(self._pending.values()):
            event.set()

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc.poll() is not None:
            raise LspError("语言服务器进程已经退出。")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        assert self._proc.stdin is not None
        with self._write_lock:
            # 不用 BufferedWriter.flush：某些服务器（TS/jdtls）在 Windows 上会让它
            # 抛 EINVAL；直接写 fd，避免缓冲层。
            awaitable = memoryview(frame)
            while awaitable:
                written = os.write(self._proc.stdin.fileno(), awaitable)
                if written <= 0:
                    raise LspError("写 LSP 请求失败（管道已关闭）。")
                awaitable = awaitable[written:]

    def request(self, method: str, params: dict[str, Any], *, timeout: float = _TIMEOUT_S) -> Any:
        self._next_id += 1
        request_id = self._next_id
        event = threading.Event()
        holder: dict[str, Any] = {}
        self._pending[request_id] = (event, holder)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if not event.wait(timeout):
            self._pending.pop(request_id, None)
            raise LspError(f"LSP 请求超时（{method}，{timeout:.0f}s）。")
        if not holder and self._proc.poll() is not None:
            raise LspError(f"LSP 进程在请求 {method} 期间退出。")
        if "error" in holder:
            error = holder["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise LspError(f"LSP 返回错误（{method}）：{message}")
        return holder.get("result")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _initialize(self) -> None:
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "clientInfo": {"name": "Axo", "version": "0.1"},
            "rootUri": self.root.as_uri(),
            "workspaceFolders": [{"uri": self.root.as_uri(), "name": self.root.name}],
            "capabilities": {
                "workspace": {"workspaceFolders": True, "configuration": True},
                "textDocument": {
                    "definition": {"linkSupport": False},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "publishDiagnostics": {},
                    "synchronization": {"didSave": False},
                },
            },
        }
        if self.language_id in (
            "javascript",
            "javascriptreact",
            "typescript",
            "typescriptreact",
        ):
            npm_root = _npm_global_root()
            tsserver = npm_root / "typescript" / "lib" / "tsserver.js" if npm_root else None
            if tsserver is not None and tsserver.is_file():
                # typescript-language-server 6 不再接受 --tsserver-path，改走
                # initializationOptions.tsserver.path。
                params["initializationOptions"] = {"tsserver": {"path": str(tsserver)}}
        self.request("initialize", params, timeout=15.0)
        self.notify("initialized", {})

    def _ensure_open(self, path: Path) -> str:
        uri = _uri_for(path)
        text = _read_document(path)
        old = self._documents.get(uri)
        if old is None:
            self._documents[uri] = (1, text)
            self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
        elif old[1] != text:
            version = old[0] + 1
            self._documents[uri] = (version, text)
            self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        return uri

    def definition(self, path: Path, line: int, character: int) -> Any:
        uri = self._ensure_open(path)
        return self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )

    def references(self, path: Path, line: int, character: int) -> Any:
        uri = self._ensure_open(path)
        return self.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": True},
            },
        )

    def document_symbols(self, path: Path) -> Any:
        uri = self._ensure_open(path)
        return self.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})

    def diagnostics(
        self, path: Path, *, timeout: float = _DIAGNOSTIC_TIMEOUT_S
    ) -> list[dict[str, Any]] | None:
        uri = _uri_for(path)
        # 事件必须在 didOpen 之前建好：服务器可能在 didOpen 后立刻 publish，
        # 晚一步建 Event 会永远等不到那一次通知。
        event = self._diagnostic_events.setdefault(uri, threading.Event())
        event.clear()
        previous = self._documents.get(uri)
        self._ensure_open(path)
        # LSP 3.17 的拉取诊断：rust-analyzer 等服务器不会主动 push，必须问。
        try:
            pulled = self.request(
                "textDocument/diagnostic", {"textDocument": {"uri": uri}}, timeout=2.0
            )
            if isinstance(pulled, dict) and isinstance(pulled.get("items"), list):
                return list(pulled["items"])
        except LspError:
            pass
        if (
            previous is not None
            and previous[1] == _read_document(path)
            and uri in self._diagnostics
        ):
            # 文件没变，服务器不会重新 publish；缓存就是当前诊断。
            return list(self._diagnostics[uri])
        if not event.wait(timeout):
            return None
        return list(self._diagnostics.get(uri, []))

    def stderr_tail(self, limit: int = 8) -> str:
        return "\n".join(list(self._stderr)[-limit:])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for uri in list(self._documents):
            with contextlib.suppress(Exception):
                self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        with contextlib.suppress(Exception):
            self.request("shutdown", {}, timeout=2.0)
        with contextlib.suppress(Exception):
            self.notify("exit", {})
        if self._proc.poll() is None:
            self._proc.terminate()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=3)
        if self._proc.poll() is None:
            self._proc.kill()
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()


class LspManager:
    """按 ``(服务器命令, 工作区根)`` 缓存客户端。**一个进程里共享。**"""

    def __init__(self) -> None:
        self._clients: dict[tuple[tuple[str, ...], str], LspClient] = {}
        self._lock = threading.Lock()

    def server_for(self, language: str) -> list[str] | None:
        key = "AXO_LSP_" + language.upper().replace("-", "_")
        if os.environ.get(key, "").strip():
            # 显式配置优先，而且**不回落**：配错了要能看出来，不能被默认命令掩盖。
            return _configured_command(language)
        return _default_command(language)

    def client_for(self, root: Path, language: str, language_id: str) -> LspClient:
        command = self.server_for(language)
        if command is None:
            raise LspError(
                f"没有可用的 LSP 服务器（{language}）。装一个或用 find_symbol/repo_map 回退。"
            )
        if "{jdtls_data}" in command:
            digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
            data_dir = Path(tempfile.gettempdir()) / f"axo-jdtls-{digest}"
            data_dir.mkdir(parents=True, exist_ok=True)
            command = [part.replace("{jdtls_data}", str(data_dir)) for part in command]
        key = (tuple(command), str(root))
        with self._lock:
            client = self._clients.get(key)
            if client is not None and client._proc.poll() is None:
                return client
            if client is not None:
                client.close()
            try:
                client = LspClient(command, root, language_id)
            except OSError as exc:
                raise LspError(f"启动 LSP 服务器失败：{exc}") from exc
            self._clients[key] = client
            return client

    def close_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

    def __enter__(self) -> LspManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close_all()


class LspTool:
    """只读 LSP 查询：definition / references / symbols / diagnostics。"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="lsp",
        description=(
            "用语言服务器做语义查询：definition（定义）、references（引用）、"
            "symbols（文件符号）、diagnostics（诊断）。需要本机已安装对应语言的 LSP "
            "服务器；没有时用 find_symbol/repo_map 回退。line 是 1 基，character 是 "
            "0 基；给了 symbol 时会自动在该行找列。只读，不改文件。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["definition", "references", "symbols", "diagnostics"],
                    "description": "要做的查询。",
                },
                "path": {"type": "string", "description": "仓库内文件路径。"},
                "line": {
                    "type": "integer",
                    "description": "1 基行号（definition/references 用）。",
                },
                "symbol": {
                    "type": "string",
                    "description": "可选：该行里的符号文本；用它定位列，比手填 character 稳。",
                },
                "character": {
                    "type": "integer",
                    "description": "0 基 UTF-16 列号；给了 symbol 时忽略。",
                },
            },
            "required": ["action", "path"],
        },
    )
    requires_approval: ClassVar[bool] = True
    parallel_safe: ClassVar[bool] = False

    def approval_summary_for(self, call) -> str:
        action = call.arguments.get("action", "查询")
        path = call.arguments.get("path", "")
        return (
            f"LSP {action}：{path}\n"
            "会启动/查询本机语言服务器（以当前用户权限运行，可能执行构建脚本）。"
        )

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        if not ctx.read_root:
            return ToolOutcome.failure("没有配置可读目录。")
        action = argument_str(arguments, "action").strip()
        if action not in ("definition", "references", "symbols", "diagnostics"):
            return ToolOutcome.failure(f"未知 action：{action}")
        try:
            path = resolve_within(ctx.read_root, argument_str(arguments, "path"))
        except ReadFileError as exc:
            return ToolOutcome.failure(str(exc))
        if not path.is_file():
            return ToolOutcome.failure(f"{path} 不是文件。")
        mapped = _LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
        if mapped is None:
            return ToolOutcome.failure(
                f"没有为 {path.suffix or '这个后缀'} 配置 LSP；用 find_symbol/repo_map 回退。"
            )
        language, language_id = mapped
        manager = ctx.lsp_manager
        if manager is None:
            return ToolOutcome.failure("这一轮没有 LSP 管理器。")
        root = next(
            (
                candidate.expanduser().resolve()
                for candidate in ctx.read_root
                if path.is_relative_to(candidate.expanduser().resolve())
            ),
            Path(ctx.read_root[0]).expanduser().resolve(),
        )
        try:
            client = manager.client_for(root, language, language_id)
        except LspError as exc:
            return ToolOutcome.failure(str(exc))

        try:
            if action == "symbols":
                return ToolOutcome.success(
                    _format_symbols(client.document_symbols(path), path), replayable=False
                )
            if action == "diagnostics":
                diagnostics = client.diagnostics(path)
                if diagnostics is None:
                    return ToolOutcome.success(
                        "服务器没有在超时内发布诊断（可能不支持 push diagnostics）。",
                        replayable=False,
                    )
                return ToolOutcome.success(_format_diagnostics(diagnostics, path), replayable=False)
            position = _position_for(path, arguments)
            if position is None:
                return ToolOutcome.failure(
                    "definition/references 需要 line（1 基），或给 symbol 让它自动找行。"
                )
            line, character = position
            result: Any = None
            for attempt in range(3):
                try:
                    result = (
                        client.definition(path, line, character)
                        if action == "definition"
                        else client.references(path, line, character)
                    )
                except LspError as exc:
                    if "content modified" in str(exc) and attempt < 2:
                        time.sleep(0.3)
                        continue
                    raise
                if result not in (None, []):
                    break
                # 服务器可能还在索引：空结果重试两次，别让第一次空答复骗过模型。
                if attempt < 2:
                    time.sleep(0.5)
            return ToolOutcome.success(_format_locations(result, root), replayable=False)
        except LspError as exc:
            detail = client.stderr_tail()
            return ToolOutcome.failure(f"LSP 查询失败：{exc}", detail=detail)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolOutcome.failure(f"LSP 返回了看不懂的结果：{exc}")


def _position_for(path: Path, arguments: dict[str, Any]) -> tuple[int, int] | None:
    symbol = argument_str(arguments, "symbol", required=False)
    line = argument_int(arguments, "line", default=None, minimum=1)
    text = _read_document(path)
    lines = text.splitlines()
    if line is None:
        if not symbol:
            return None
        for index, raw in enumerate(lines, 1):
            if symbol in raw:
                line = index
                break
        if line is None:
            return None
    if line > len(lines):
        return None
    raw = lines[line - 1]
    if symbol:
        column = raw.find(symbol)
        if column < 0:
            return None
        character = _utf16_column(raw, column)
    else:
        character = argument_int(arguments, "character", default=0, minimum=0) or 0
    return line - 1, character


def _display_path(uri: str, base: Path) -> str:
    try:
        path = _path_for(uri)
    except (ValueError, OSError):
        return uri
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _format_locations(result: Any, base: Path) -> str:
    if result is None:
        return "没有找到结果。"
    items = result if isinstance(result, list) else [result]
    lines: list[str] = []
    for item in items[:_MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri") or item.get("targetUri")
        range_ = item.get("range") or item.get("targetRange") or {}
        if not uri or not isinstance(range_, dict):
            continue
        start = range_.get("start", {})
        line = int(start.get("line", 0)) + 1
        char = int(start.get("character", 0)) + 1
        lines.append(f"{_display_path(str(uri), base)}:{line}:{char}")
    if not lines:
        return "没有找到结果。"
    more = f"\n…（还有 {len(items) - _MAX_RESULTS} 个）" if len(items) > _MAX_RESULTS else ""
    return "\n".join(lines) + more


_KIND_NAMES = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    21: "object",
    22: "key",
    23: "null",
    24: "enum member",
    25: "struct",
    26: "event",
    27: "operator",
    28: "type parameter",
}


def _flatten_symbols(items: Any, path: Path, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if not isinstance(items, list):
        return lines
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        kind = _KIND_NAMES.get(int(item.get("kind", 0)), "symbol")
        range_ = item.get("range") or item.get("location", {}).get("range") or {}
        start = range_.get("start", {}) if isinstance(range_, dict) else {}
        line = int(start.get("line", 0)) + 1 if start else 0
        lines.append(f"{path.name}:{line}: {prefix}{kind} {name}")
        children = item.get("children")
        if isinstance(children, list):
            lines.extend(_flatten_symbols(children, path, prefix + "  "))
    return lines


def _format_symbols(result: Any, path: Path) -> str:
    lines = _flatten_symbols(result, path)
    if not lines:
        return "没有找到符号。"
    return "\n".join(lines[:_MAX_RESULTS])


_SEVERITY = {1: "错误", 2: "警告", 3: "信息", 4: "提示"}


def _format_diagnostics(items: list[dict[str, Any]], path: Path) -> str:
    if not items:
        return "没有诊断。"
    lines: list[str] = []
    for item in items[:_MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        start = item.get("range", {}).get("start", {})
        line = int(start.get("line", 0)) + 1
        char = int(start.get("character", 0)) + 1
        severity = _SEVERITY.get(int(item.get("severity", 3)), "信息")
        message = " ".join(str(item.get("message", "")).split())
        source = str(item.get("source", "")).strip()
        suffix = f"（{source}）" if source else ""
        lines.append(f"{path.name}:{line}:{char}: {severity}：{message}{suffix}")
    return "\n".join(lines) or "没有诊断。"
