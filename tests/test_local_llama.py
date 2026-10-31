"""随包的 llama.cpp 运行时：**软件自己拉起本地模型**，不再经过 Ollama。

用户 2026-10-31 的要求原话是「我希望软件自身就可以调用本地模型而不是经过一次
ollama」。这条路能成立靠三件事，这个文件各钉一条：

1. **找得到**运行时（``paths.bundled_llama_dir`` 只认带 ``llama-server.exe`` 的目录，
   空目录不算）；
2. **起得来**（``LlamaServerRuntime.start`` 拼出来的命令行就是那几条关键参数，
   尤其 ``--jinja`` 与 ``-ngl``——少了它们不是慢，是工具调用不工作 / 全走 CPU）；
3. **想得关**（思考默认关：``BuiltinLlamaBackend`` 往请求体里补
   ``chat_template_kwargs``）。

**不真的起进程**：``subprocess.Popen`` 与 ``urlopen`` 都换成替身。真起一次要加载
4 GB 模型、占 8 GB 显存里的一大块，而这里要验的是"命令怎么拼、失败怎么说人话"。
"""

from __future__ import annotations

import sys

import pytest

from axo import paths
from axo.llm import llama_server
from axo.llm.base import ChatMessage, ChatRequest
from axo.llm.llama_server import (
    BuiltinLlamaBackend,
    LlamaServerError,
    LlamaServerRuntime,
    find_llama_server,
)

# ── 找得到 ──────────────────────────────────────────────────────────────────


def test_env_dir_is_used_when_it_holds_the_server(tmp_path, monkeypatch) -> None:
    (tmp_path / "llama-server.exe").write_bytes(b"MZ")
    monkeypatch.setenv("AXO_LLAMA_DIR", str(tmp_path))
    assert paths.bundled_llama_dir() == tmp_path


def test_a_dir_without_the_exe_is_not_a_runtime(tmp_path, monkeypatch) -> None:
    """只判目录会让一个空目录骗过所有人——判据是那个 exe 在不在。"""
    monkeypatch.setenv("AXO_LLAMA_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths, "_bundled_base_dirs", lambda: ())
    assert paths.bundled_llama_dir() is None


def test_find_returns_none_without_a_bundle(monkeypatch) -> None:
    monkeypatch.setattr(llama_server, "bundled_llama_dir", lambda: None)
    assert find_llama_server() is None


# ── 起得来 ──────────────────────────────────────────────────────────────────


class _FakeProcess:
    """``Popen`` 的替身：活着、能停、叫得醒。"""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.killed = True


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def runtime_pieces(tmp_path, monkeypatch):
    """把一次 ``start`` 要用到的外部件全换成替身，返回 (runtime, 记录)。"""
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"MZ")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    process = _FakeProcess()
    recorded: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return process

    monkeypatch.setattr(llama_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_server, "find_llama_server", lambda: exe)
    monkeypatch.setattr(llama_server, "_free_port", lambda: 51234)
    monkeypatch.setattr(
        llama_server.urllib.request, "urlopen", lambda _url, **_kwargs: _FakeResponse()
    )
    runtime = LlamaServerRuntime(model_path=model, context=8192, log_path=tmp_path / "llama.log")
    return runtime, recorded, process


def test_start_builds_the_command_that_makes_tools_work(runtime_pieces) -> None:
    """四个参数各自挡一类失效：

    * ``--jinja``：不带给不出模型的工具调用模板，工具调用直接不工作（不是慢）；
    * ``-ngl 99``：不给就全在 CPU 上跑，8 GB 显存白搭；
    * ``-c <上下文>``：与设置里那个数同一个值，用户改它必须真的生效；
    * ``--parallel 1``：多槽会把上下文切成几份，而这份应用只有一个会话在跑。
    """
    runtime, recorded, _process = runtime_pieces
    base_url = runtime.start(timeout=5)

    assert base_url == "http://127.0.0.1:51234/v1"
    command = recorded["command"]
    assert command[0].endswith("llama-server.exe")
    assert "-m" in command and command[command.index("-m") + 1].endswith("model.gguf")
    assert "--jinja" in command
    assert command[command.index("-ngl") + 1] == "99"
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("--parallel") + 1] == "1"
    # 端口是现问系统要的（``_free_port`` 被换成 51234），不是写死的 8080。
    assert command[command.index("--port") + 1] == "51234"


def test_stop_really_ends_the_process(runtime_pieces) -> None:
    """关窗要收得掉：这个进程是软件自己拉起的，没人替它收尸。"""
    runtime, _recorded, process = runtime_pieces
    runtime.start(timeout=5)
    runtime.stop()
    assert process.terminated
    assert runtime.running is False


def test_a_missing_model_file_says_which_file(tmp_path, monkeypatch) -> None:
    """启动失败要说人话：哪一个文件不在，去哪儿改。"""
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(llama_server, "find_llama_server", lambda: exe)
    runtime = LlamaServerRuntime(model_path=tmp_path / "nope.gguf", log_path=tmp_path / "llama.log")
    with pytest.raises(LlamaServerError) as caught:
        runtime.start(timeout=1)
    assert "nope.gguf" in str(caught.value)
    assert "设置" in str(caught.value)


def test_a_missing_runtime_says_reinstall(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llama_server, "find_llama_server", lambda: None)
    runtime = LlamaServerRuntime(
        model_path=tmp_path / "model.gguf", log_path=tmp_path / "llama.log"
    )
    with pytest.raises(LlamaServerError) as caught:
        runtime.start(timeout=1)
    assert "llama-server.exe" in str(caught.value)


# ── 想得关 ──────────────────────────────────────────────────────────────────


def _payload(request: ChatRequest) -> dict:
    backend = object.__new__(BuiltinLlamaBackend)
    backend.model = "local"
    return BuiltinLlamaBackend._build_payload(backend, request)


def test_thinking_is_off_by_default() -> None:
    """Qwen3 带思考时会把输出额度烧在 ``<think>`` 上（2026-10-12 实测正文是空串）。"""
    request = ChatRequest(messages=(ChatMessage(role="user", content="你好"),))
    assert _payload(request)["chat_template_kwargs"] == {"enable_thinking": False}


def test_an_explicit_choice_wins() -> None:
    """调用方显式给了就听它的——这一层不做二次判断。"""
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="你好"),),
        extra={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert _payload(request)["chat_template_kwargs"] == {"enable_thinking": True}
