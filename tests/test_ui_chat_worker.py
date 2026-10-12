"""``ChatWorker`` 撞见工具调用时必须**出声**。

这条病史很短但很难查：``ChatWorker.run`` 以前只读 ``piece.text`` / ``piece.done``
/ token 数，**从不读 ``piece.tool_calls``**。于是模型决定调工具的那一刻，那些片
的正文是空串，循环拼出一个空回复；用户看到的是一句没头没尾的话，或者一片空白，
而**没有任何地方报错**——没有异常、没有日志、状态照常复位成空闲。

静默丢弃比报错更坏：报错有症状可查，静默丢弃只能靠"模型有时候会答非所问"这种
印象去追。所以现在它抛 :class:`ToolCallsWhereNoneExpected`，由既有的错误路径
转成 ``failed`` 信号。

标 ``gui`` 是因为要构造 ``QObject``（信号要有 Qt）。用 offscreen，不需要显示器，
也不碰 QtWebEngine——那个在离屏下会让进程无声消失（见 ``_smoke_recall`` 的病史）。
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# 裸导入而不是 ``from .fakes``：``tests/`` 没有 ``__init__.py``，pytest 在默认的
# prepend 导入模式下会把测试文件所在目录塞进 ``sys.path``，所以同目录的模块就是这样
# 引的。加 ``__init__.py`` 会把所有测试改成包内模块、动到全仓的收集方式，为一行
# 导入不值当。
from fakes import ScriptedBackend, text_chunk, tool_chunk

from axo.llm.base import (
    ChatBackendError,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolSpec,
)
from axo.ui import chat_worker
from axo.ui.chat_worker import ChatWorker, ModelListWorker
from axo.ui.settings_store import (
    PROVIDER_CLOUD,
    PROVIDER_OLLAMA,
    BackendSettings,
)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _run(worker: ChatWorker) -> tuple[list[str], list[tuple[str, dict]], list[str]]:
    """跑完 worker，收齐三路信号的产物。"""
    chunks: list[str] = []
    finished: list[tuple[str, dict]] = []
    failed: list[str] = []
    worker.chunk.connect(chunks.append)
    worker.finished.connect(lambda text, stats: finished.append((text, stats)))
    worker.failed.connect(failed.append)
    worker.run()
    return chunks, finished, failed


def _request(*, with_tools: bool = False) -> ChatRequest:
    tools = (ToolSpec(name="get_time", description="查时间"),) if with_tools else ()
    return ChatRequest(messages=[ChatMessage(role="user", content="现在几点？")], tools=tools)


# ── 正常路径：先确认改动没伤到它 ───────────────────────────────────────────


def test_plain_reply_still_streams_and_finishes(qapp: Any) -> None:
    """纯文本回复照旧：逐片转发、发 finished、不发 failed。

    放在最前面是因为下面几条测的都是"出事"的路径；没有这一条，一个**永远**发
    ``failed`` 的实现也能让它们全绿。
    """
    backend = ScriptedBackend(
        [
            [
                text_chunk("你好"),
                text_chunk("，世界", done=True, prompt_tokens=12, completion_tokens=5),
            ]
        ]
    )
    worker = ChatWorker(backend, _request())

    chunks, finished, failed = _run(worker)

    assert chunks == ["你好", "，世界"]
    assert failed == []
    assert len(finished) == 1
    text, stats = finished[0]
    assert text == "你好，世界"
    assert stats["prompt_tokens"] == 12
    assert stats["completion_tokens"] == 5
    assert stats["interrupted"] is False


def test_stop_mid_stream_marks_interrupted(qapp: Any) -> None:
    """用户停：标记 interrupted，仍然发 finished（不是 failed）。

    这一条是既有行为的护栏——新增的工具调用分支就在同一个循环里，改错位置很容易
    把停止路径也一起改成 failed。
    """
    backend = ScriptedBackend([[text_chunk("第一片"), text_chunk("第二片", done=True)]])
    worker = ChatWorker(backend, _request())
    worker.chunk.connect(lambda _: worker.stop())

    chunks, finished, failed = _run(worker)

    assert chunks == ["第一片"]
    assert failed == []
    assert len(finished) == 1
    assert finished[0][1]["interrupted"] is True


# ── 出事：模型发了工具调用 ─────────────────────────────────────────────────


def test_tool_calls_raise_instead_of_being_silently_dropped(qapp: Any) -> None:
    """核心断言：**不发 finished、发 failed，且消息里点出工具名**。

    三条一起钉，缺一条这个守卫就可能是假的：

    * 只断言"发了 failed" —— 一个对所有输入都发 failed 的实现也能过；
    * 不断言"没发 finished" —— 双发（failed 之后又 finished）会让界面先报错
      再复位成"回复完成"，而用户拿到的是一个空回复；
    * 不断言工具名 —— 消息里没有可查的线索，等于换了个地方静默。
    """
    call = ToolCall(id="abc123", name="get_time", arguments={})
    backend = ScriptedBackend([[tool_chunk(call)]])
    worker = ChatWorker(backend, _request())

    chunks, finished, failed = _run(worker)

    assert chunks == []
    assert finished == [], "双发：failed 之后不该再有 finished"
    assert len(failed) == 1
    assert "get_time" in failed[0]
    assert "AgentWorker" in failed[0]


def test_message_distinguishes_declared_tools_from_undeclared(qapp: Any) -> None:
    """两种接线错的诊断信息不同。

    请求里带了工具却走了 ``ChatWorker``（路由错），与没带工具却收到 ``tool_calls``
    （协议意外）是两件要分别去查的事。消息里说清是哪一种，能省掉一次复现。
    """
    call = ToolCall(id="x", name="search_memory", arguments={})

    undeclared = _run(ChatWorker(ScriptedBackend([[tool_chunk(call)]]), _request()))[2]
    declared = _run(ChatWorker(ScriptedBackend([[tool_chunk(call)]]), _request(with_tools=True)))[2]

    assert "没有工具声明" in undeclared[0]
    assert "带了工具声明" in declared[0]


def test_tool_calls_after_some_text_still_fail_loudly(qapp: Any) -> None:
    """先吐了字、再发工具调用：已经显示出去的半句留着，但仍然报错。

    少见的形状，但它说明这条守卫放在**每一片**上，不是"回合一开头检查一次"。
    若只在开头检查，模型中途改主意去调工具就会被漏掉。
    """
    call = ToolCall(id="x", name="write_memory", arguments={"content": "x"})
    backend = ScriptedBackend([[text_chunk("我查一下"), tool_chunk(call)]])
    worker = ChatWorker(backend, _request())

    chunks, finished, failed = _run(worker)

    assert chunks == ["我查一下"]
    assert finished == []
    assert "write_memory" in failed[0]


def test_backend_error_still_reported_as_before(qapp: Any) -> None:
    """后端的错照旧走 ``ChatBackendError`` 那条路——新增分支没把老路径挡住。"""
    backend = ScriptedBackend([[text_chunk("x", done=True)]], raise_at=0)
    worker = ChatWorker(backend, _request())

    _chunks, finished, failed = _run(worker)

    assert finished == []
    assert len(failed) == 1
    assert "脚本化的后端错误" in failed[0]


# ── 模型清单：从哪个后端拉 ──────────────────────────────────────────────────


def _listed(worker: Any) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """跑完 worker，收齐「清单」与「报错」两路信号。"""
    loaded: list[list[dict[str, Any]]] = []
    failed: list[str] = []
    worker.loaded.connect(loaded.append)
    worker.failed.connect(failed.append)
    worker.run()
    return loaded, failed


def _spy(monkeypatch: Any) -> list[tuple[str, ...]]:
    """把两个真函数换成记录调用参数的替身。

    记录**参数**而不只是"被调过"：``allow_remote`` 漏传的症状是"某个功能莫名说
    连不上，而别的功能好好的"，只记"调过哪个函数"抓不住它。
    """
    calls: list[tuple[str, ...]] = []

    def fake_local(host: str, *, allow_remote: bool = False) -> list[dict[str, Any]]:
        calls.append(("local", host, str(allow_remote)))
        return [{"name": "qwen3-4b"}]

    def fake_remote(base_url: str, *, api_key: str = "") -> list[dict[str, Any]]:
        calls.append(("cloud", base_url, api_key))
        return [{"name": "deepseek-chat"}]

    monkeypatch.setattr(chat_worker, "list_local_models", fake_local)
    monkeypatch.setattr(chat_worker, "list_remote_models", fake_remote)
    return calls


def test_a_local_backend_never_touches_the_cloud_function(qapp: Any, monkeypatch: Any) -> None:
    """**这一条是分派的守卫。**

    走错方向的后果不是一条错误，是"启动时去问一个不存在的云端地址"：模型下拉框空着、
    状态栏写着一段与用户当前配置无关的话，而用户根本没配过云端。
    """
    calls = _spy(monkeypatch)
    worker = ModelListWorker(
        BackendSettings(provider=PROVIDER_OLLAMA, host="http://127.0.0.1:11434", allow_remote=True)
    )

    loaded, failed = _listed(worker)

    assert calls == [("local", "http://127.0.0.1:11434", "True")], "allow_remote 必须一路传到"
    assert loaded == [[{"name": "qwen3-4b"}]]
    assert failed == []


def test_a_cloud_backend_asks_the_cloud(qapp: Any, monkeypatch: Any) -> None:
    """云端要知道 base_url + key——两个字段装不进一个 ``host``，这正是这个 worker
    收整个 :class:`BackendSettings` 的原因。"""
    calls = _spy(monkeypatch)
    worker = ModelListWorker(
        BackendSettings(
            provider=PROVIDER_CLOUD, cloud_base_url="https://api.deepseek.com/v1", api_key="sk-x"
        )
    )

    loaded, failed = _listed(worker)

    assert calls == [("cloud", "https://api.deepseek.com/v1", "sk-x")]
    assert loaded == [[{"name": "deepseek-chat"}]]
    assert failed == []


def test_a_cloud_failure_is_reported_with_its_reason(qapp: Any, monkeypatch: Any) -> None:
    """**不在 worker 里吞掉错误返回兜底预设。**

    ``loaded`` 只有一个 ``list``，装不下"我用了兜底，因为……"——那就成了静默。
    照常发 ``failed``，由主窗口决定要不要同时摆一份预设出来。
    """

    def boom(base_url: str, *, api_key: str = "") -> list[dict[str, Any]]:
        raise ChatBackendError("对话失败：HTTP 401（API key 无效、过期，或者根本没填。）")

    monkeypatch.setattr(chat_worker, "list_remote_models", boom)
    worker = ModelListWorker(BackendSettings(provider=PROVIDER_CLOUD))

    loaded, failed = _listed(worker)

    assert loaded == []
    assert len(failed) == 1
    assert "API key" in failed[0]


def test_an_unexpected_error_still_becomes_a_sentence(qapp: Any, monkeypatch: Any) -> None:
    """兜底那一支：一个未捕获的异常会静默终止线程，界面永远停在"正在加载"。"""

    def boom(host: str, *, allow_remote: bool = False) -> list[dict[str, Any]]:
        raise RuntimeError("谁知道呢")

    monkeypatch.setattr(chat_worker, "list_local_models", boom)
    worker = ModelListWorker(BackendSettings(provider=PROVIDER_OLLAMA))

    loaded, failed = _listed(worker)

    assert loaded == []
    assert "未预期的错误" in failed[0]
    assert "谁知道呢" in failed[0]
