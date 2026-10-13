"""Ollama 后端。

走 Ollama 的 HTTP API，不 import 任何 ollama 客户端库——理由是那层封装在流式
路径上什么都没省，却多一个要跟版本走的依赖。直接对着 ``/api/chat`` 的 NDJSON
流写二十行，比读一个客户端的 API 文档更快，也更容易排查。

用到的四个端点::

    GET  /api/tags            列出本地已有模型（供 UI 的下拉框）
    GET  /api/version         探活（比 /api/tags 便宜）
    POST /api/chat            流式对话，NDJSON
    POST /api/embed           批量嵌入（Ollama >= 0.1.36）

**默认只连本机。** 这不是可选项：本项目的记忆库设计假设对话内容不出设备。把
``host`` 指向非本机地址时必须显式传 ``allow_remote=True``，代码才会放行——一道
廉价的护栏，让"手滑把 localhost 写成某个内网 IP"变成一次明确的确认。
"""

from __future__ import annotations

import ipaddress
import json
import ssl
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np

from .base import (
    ChatBackendError,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    EmbeddingBackendError,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "MODEL_ID_PREFIX",
    "OllamaChatBackend",
    "OllamaEmbeddingBackend",
    "check_host_is_local",
    "is_chat_capable",
    "list_local_models",
    "pick_chat_model",
    "pick_embedding_model",
    "probe",
    "probe_embedding_dim",
    "query_context_window",
    "short_model_name",
]

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

#: ``model_id`` 的前缀，也是会话 ``model_ref`` 上留下的来源名。
#: **它属于后端类型，不属于模型**（对比 :data:`~.openai_compat.MODEL_ID_PREFIX`）：
#: 同一个模型名从两个来源来就是两回事，翻旧会话时要能一眼分开。主窗口拼
#: ``model_ref`` 也读这一个常量，不另抄一份字面量。
MODEL_ID_PREFIX = "ollama"

# 常见对话模型的上下文窗口（token）。**这不是权威值**——真实值由模型的
# Modelfile 里的 num_ctx 决定，而 Ollama 会为每个请求按它自己的默认值截断。
# 所以这里取保守值：宁可预算算小了少注入几条记忆，也不要因为高估而在服务端被
# 静默截断（截断丢的是最早的消息，正是人格锚所在的位置，症状极具迷惑性）。
_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # (模型名前缀, 上下文窗口)
    ("deepseek-r1", 32768),
    ("deepseek-v3", 32768),
    ("deepseek", 16384),
    ("qwen3", 32768),
    ("qwen2.5", 32768),
    ("qwen2", 32768),
    ("llama3.2", 32768),
    ("llama3.1", 32768),
    ("llama3", 8192),
    ("mistral", 8192),
    ("gemma3", 32768),
    ("gemma2", 8192),
    ("phi4", 16384),
    ("phi3", 4096),
    ("glm4", 32768),
)
_FALLBACK_CONTEXT_WINDOW = 8192

# 请求超时：连接要快失败（后端没起来时不该让 UI 卡住），读取要长超时
# （本地模型在 8 GB 显存上吐第一个 token 可能要好几秒）。
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)


# ── 护栏 ────────────────────────────────────────────────────────────────────


def check_host_is_local(host: str) -> bool:
    """判断 ``host`` 是否指向本机。

    认定为本机的情形：``localhost``、回环 IP 字面量（``127.0.0.0/8``、``::1``）、
    以及 ``0.0.0.0``（Ollama 监听这个地址时本机也在其中）。
    """
    try:
        parsed = urlparse(host)
        netloc = parsed.hostname or ""
    except ValueError:
        return False
    if not netloc:
        return False
    if netloc.lower() in {"localhost", "localhost.localdomain"}:
        return True
    try:
        address = ipaddress.ip_address(netloc)
    except ValueError:
        return False  # 是个域名，解析与否都不当本机处理
    return address.is_loopback or address.is_unspecified


def _client(host: str, allow_remote: bool) -> httpx.Client:
    if not allow_remote and not check_host_is_local(host):
        raise ChatBackendError(
            f"后端地址 {host!r} 不是本机地址。本终端的记忆库设计假设对话内容不出设备，"
            f"把它指向远程主机会让这个假设失效。若确实要这么做（例如自建的内网推理机），"
            f"显式传 allow_remote=True。"
        )
    return httpx.Client(base_url=host, timeout=_TIMEOUT)


def _raise_for_status(response: httpx.Response, *, what: str) -> None:
    """把 httpx 的 HTTP 错误翻译成能直接显示给用户的句子。

    **取正文之前必须先确认它被读出来了。** 流式请求（``client.stream(...)``）返回
    时正文还在网络上，此刻取 ``response.text`` 抛的是 ``httpx.ResponseNotRead``
    ——于是本该是一条干净错误（``model 'x' not found, try pulling it first`` 那种
    能直接给用户看的句子）的场合，用户看到一个 httpx 回溯。触发条件是**流式请求撞上
    非 2xx**：模型名写错、Ollama 没起、上下文超限，全是常见情形。

    ``read()`` 对已经读过的响应是幂等的（直接返回缓存），所以不用先判断要不要读；
    写成"先试再补"是为了**只在这一种情形下多走一步**，而不是对每条错误路径都改行为。
    """
    if response.is_success:
        return
    try:
        body = response.text.strip()
    except httpx.ResponseNotRead:
        response.read()
        body = response.text.strip()
    # Ollama 的错误体形如 {"error":"model 'x' not found, try pulling it first"}
    try:
        detail = json.loads(body).get("error", body)
    except (json.JSONDecodeError, AttributeError):
        detail = body
    detail = (detail or "无附加信息")[:500]
    hint = ""
    if response.status_code == 404:
        hint = "（模型未下载？先跑 ollama pull <模型名>）"
    elif response.status_code >= 500:
        hint = "（Ollama 服务端自身报错，查看它的终端输出）"
    raise ChatBackendError(f"{what}失败：HTTP {response.status_code} {detail}{hint}")


def _wrap_transport_error(exc: Exception, host: str, *, what: str) -> None:
    """把传输层的异常收成一句人话。**TLS 那一类要单独说**（同云端那一份）。

    对 ``http://127.0.0.1:11434`` 来说 TLS 失败很罕见，但 ``allow_remote`` 打开之后
    连的是别人的机器，那一类就真会到。说"Ollama 是否在运行"在那时是**误导**——
    它会给用户一个改不动它的动作。
    """
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ssl.SSLError) or isinstance(exc.__cause__, ssl.SSLError):
        hint = (
            "TLS 握手没成功。这通常不是服务没起来，而是**这台机器上的代理或网络中间设备**拦了它。"
        )
    else:
        hint = "Ollama 是否在运行？（启动命令：ollama serve）"
    raise ChatBackendError(f"{what}失败：连不上 {host}。{hint} 原始错误：{detail}") from exc


def short_model_name(name: str) -> str:
    """去掉 Ollama 的默认标签：``qwen3-4b:latest`` → ``qwen3-4b``。

    ``:latest`` 只在"没写标签"时由 Ollama 自动补上，所以脱掉它是**恒等变换**，
    不是有损截断——``qwen3-4b`` 与 ``qwen3-4b:latest`` 在服务端指向同一个模型。
    非 ``latest`` 的标签（``qwen3:4b`` 里的 ``4b``、``llama3.2:3b-instruct-q4_K_M``
    里的量化标记）都是**实义**的，必须原样保留。

    界面与命令行都按这个名字工作，因为人要打的是 ``qwen3-4b`` 而不是
    ``qwen3-4b:latest``；把它当作"名字里的一部分"显示出来只会制造
    "我明明有这个模型，为什么说没有"的困惑。
    """
    stem = name[: -len(":latest")]
    # 退化输入 `":latest"`：脱完是空串。它本身是个畸形名字而非"默认标签的模型"，
    # 还它原样比还一个空名字好——空名字在下拉框里既选不中也报不清楚错。
    if name.endswith(":latest") and stem:
        return stem
    return name


# ── 探活与模型清单 ───────────────────────────────────────────────────────────


def probe(host: str = DEFAULT_OLLAMA_HOST, *, allow_remote: bool = False) -> str | None:
    """探活。活着返回版本号字符串，没起来返回 ``None``。

    **不抛异常**：这个函数的调用者是 UI 的启动流程，它要的是"能连上吗"这个
    布尔答案，而不是一个需要在每个调用点包 try 的异常。
    """
    try:
        with _client(host, allow_remote) as client:
            response = client.get("/api/version")
            if not response.is_success:
                return None
            return str(json.loads(response.text).get("version", "") or "unknown")
    except (httpx.RequestError, json.JSONDecodeError, ChatBackendError, ValueError):
        return None


def list_local_models(
    host: str = DEFAULT_OLLAMA_HOST, *, allow_remote: bool = False
) -> list[dict[str, Any]]:
    """列出本地已有模型。

    返回 ``[{"name": ..., "short_name": ..., "size": 字节数, "capabilities": (...)}, ...]``，
    按名字排序。``name`` 是 Ollama 的原样返回（可能带 ``:latest``），``short_name``
    是去掉默认标签后的名字（见 :func:`short_model_name`）——**界面与命令行显示的是
    ``short_name``**，但 ``name`` 一并给出，因为它是服务端的权威标识。

    连不上时抛 :class:`ChatBackendError`——与 :func:`probe` 相反，
    调用这个函数的地方（模型下拉框）确实需要知道"为什么是空的"。
    """
    with _client(host, allow_remote) as client:
        try:
            response = client.get("/api/tags")
        except (httpx.RequestError, ssl.SSLError) as exc:
            _wrap_transport_error(exc, host, what="列出本地模型")
        _raise_for_status(response, what="列出本地模型")

    raw = json.loads(response.text).get("models", [])
    # capabilities 要逐个问 /api/show。并发取是因为串行会让每个模型都付出一次
    # 往返——装十几个模型时足以让下拉框"卡一下"，而它本该是瞬间的。
    capabilities: list[tuple[str, ...]] = [() for _ in raw]
    if raw:
        with ThreadPoolExecutor(max_workers=min(8, len(raw))) as pool:
            futures = [
                (index, pool.submit(_fetch_capabilities, entry.get("name", ""), host, allow_remote))
                for index, entry in enumerate(raw)
            ]
            for index, future in futures:
                capabilities[index] = future.result()

    models: list[dict[str, Any]] = []
    for entry, caps in zip(raw, capabilities, strict=True):
        detail = entry.get("details") or {}
        name = entry.get("name", "")
        models.append(
            {
                "name": name,
                "short_name": short_model_name(name),
                "size": int(entry.get("size") or 0),
                "parameter_size": detail.get("parameter_size", ""),
                "quantization": detail.get("quantization_level", ""),
                "family": detail.get("family", ""),
                "modified_at": entry.get("modified_at", ""),
                "capabilities": caps,
            }
        )
    models.sort(key=lambda item: item["name"])
    return models


def _fetch_capabilities(name: str, host: str, allow_remote: bool) -> tuple[str, ...]:
    """问 ``/api/show`` 这个模型能干什么。

    存在的理由只有一个：**嵌入模型和对话模型在 ``/api/tags`` 里长得一模一样**，
    都有名字、大小、参数量。把 ``nomic-embed-text`` 选进对话下拉框的后果是发出去
    的请求报一个与服务端语义无关的错，而根因是选择列表本身就不该包含它。

    返回 ``()`` 表示**问不到**（老版 Ollama 没这个字段、模型刚删、服务抖动）。
    空元组与"确实什么都不支持"在调用方是同一处理：[`is_chat_capable`] 会把
    未知一律当作可用——宁可让用户遇到一个可解释的失败，也不要因为一次探询失败
    就让模型从列表里凭空消失。
    """
    try:
        with _client(host, allow_remote) as client:
            response = client.post("/api/show", json={"model": name})
            if not response.is_success:
                return ()
            payload = json.loads(response.text)
    except (httpx.RequestError, json.JSONDecodeError, ChatBackendError, ValueError):
        return ()
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        return ()
    return tuple(str(item) for item in capabilities)


def is_chat_capable(entry: dict[str, Any]) -> bool:
    """这个模型能不能用来对话。

    判据是 Ollama 自报的 ``capabilities`` 里有没有 ``"completion"``。本机实测：
    ``qwen3-4b`` 报 ``['tools', 'thinking', 'completion']``，而
    ``nomic-embed-text`` 只报 ``['embedding']``。

    ``capabilities`` 缺失或为空时**当作可用**：那是"不知道"，不是"不行"。
    把它当"不行"会让老版本 Ollama 上的模型全部消失。
    """
    capabilities = entry.get("capabilities") or ()
    if not capabilities:
        return True
    return "completion" in capabilities


def pick_embedding_model(models: list[dict[str, Any]]) -> str:
    """挑一个嵌入模型，挑不到返回空串。

    优先 :data:`DEFAULT_EMBEDDING_MODEL`——这两个是同一个决定的两半（"默认用哪个"
    与"从装好的模型里挑哪个"），所以共用同一个常量：分成两个名字的话，改了其中一个
    就会出现"默认按 A 建后端，却把 B 的向量写进库"。

    而"究竟用哪个嵌入模型"是会影响**全库向量含义**的决定：``space_id`` 里带模型名，
    换了就是另一个空间，已有的向量一条都用不上。所以顺序是"先认定的那个，再退而
    求其次"，而不是"随便挑一个能嵌入的"——后者会让换一台机器、或库里多出一个模型，
    就悄悄换掉整个库的向量空间。

    判据是 ``capabilities`` 而不是名字里有没有 ``embed``：与 :func:`is_chat_capable`
    同一条纪律，因为名字是用户随手起的，而能力是服务端自报的。

    比 ``short_name`` 而不是 ``name``：``:latest`` 是 Ollama 在没写标签时自动补的，
    库里存的到底是哪个写法不由我们决定。
    """
    candidates = [
        entry for entry in models if "embedding" in tuple(entry.get("capabilities") or ())
    ]
    for entry in candidates:
        if (entry.get("short_name") or entry["name"]) == DEFAULT_EMBEDDING_MODEL:
            return str(entry["name"])
    return str(candidates[0]["name"]) if candidates else ""


def pick_chat_model(models: list[dict[str, Any]]) -> str:
    """挑一个能对话的模型，挑不到返回空串。

    :func:`pick_embedding_model` 的另一半。存在的理由是一个具体的坑：
    ``list_local_models`` 的返回值**按名字排序**，而 ``nomic-embed-text`` 排在最前
    ——照列表顺序取第一个，拿到的是一个**不能对话**的模型。症状是请求报一个与服务端
    语义无关的错，根因却在"选择列表本身就不该从这里取"。

    **判据交给 :func:`is_chat_capable`，不按名字。** 这里最容易顺手写的是"名字里没有
    ``embed`` 就行"，但那与 :func:`pick_embedding_model` 的纪律正好相反（它明说了
    "判据是 capabilities 而不是名字：名字是用户随手起的，而能力是服务端自报的"）。
    两处各写一套判据的话，某天它们会对同一个模型给出不同答案。

    ``is_chat_capable`` 把"问不到能力"当作可用，所以这里的返回值在那种情况下是列表
    里的第一个——那是"不知道"，不是"不行"，与下拉框里模型不会凭空消失是同一个取舍。

    界面上不调这个函数：那边的需求是"列出全部可对话的"，不是"挑一个"，所以它直接用
    :func:`is_chat_capable` 过滤。两者共用同一条判据。
    """
    for entry in models:
        if is_chat_capable(entry):
            return str(entry["name"])
    return ""


# ── 对话后端 ────────────────────────────────────────────────────────────────


@dataclass
class OllamaChatBackend:
    #: 图挂在**工具结果那一行**上（实测收 ``user`` 与 ``tool`` 两处，见
    #: :attr:`~axo.llm.base.ChatMessage.images`）。这是那个约定属性的一半：
    #: OpenAI 兼容那个后端声明的是 ``"user"``。
    images_ride_on = "tool"

    #: **输出会挤掉输入。** ``num_predict`` 不传时，Ollama 拿同一个 ``num_ctx`` 生成，
    #: 输出可以一直涨到把最早的消息（人格锚的位置）挤出窗口。所以这一条链要给生成留
    #: 一笔预留（``session/budget.py`` 的 ``DEFAULT_RESERVE_OUTPUT``），并把它真的传进
    #: 请求。云端没有这个怪癖，一个字都不传——这个属性就是那个分叉的判据。
    output_squeezes_input = True

    """Ollama 上的一个对话模型。

    Attributes:
        model: Ollama 内部的模型名，如 ``"deepseek-r1:4b"``。
        host: Ollama 服务地址。
        context_window: 上下文窗口。``None`` 时按 :data:`_CONTEXT_WINDOWS` 猜。
        allow_remote: 允许连接非本机地址（默认否）。
        keep_alive: 传给 Ollama 的驻留时长。默认 5 分钟——比默认的 5 分钟长一点，
            让连续对话不必反复把模型从显存里换进换出。
    """

    model: str
    host: str = DEFAULT_OLLAMA_HOST
    context_window: int | None = None
    allow_remote: bool = False
    keep_alive: str = "5m"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("OllamaChatBackend.model 不能为空。")
        if self.context_window is None:
            # 先问服务端，问不到才猜。**这个顺序不能反**：猜错的代价是预算算大，
            # 而算大的后果是 Ollama 在服务端静默截断——截掉的是最早的消息，
            # 正是人格锚所在的位置。表面上一切正常，只是模型"忘了自己是谁"。
            self.context_window = query_context_window(self.model, self.host, self.allow_remote)
        if self.context_window < 512:
            raise ValueError(
                f"context_window={self.context_window} 太小，装配预算无法工作（人格锚本身就要 800 token）。"
            )

    # ------------------------------------------------------------ 协议实现

    @property
    def model_id(self) -> str:
        """``"ollama/<模型名>"``。进注入账本的就是它。"""
        return f"{MODEL_ID_PREFIX}/{self.model}"

    def count_tokens(self, text: str) -> int | None:  # noqa: ARG002 — 协议签名，刻意忽略
        """返回 ``None``：Ollama 在发请求前没有廉价的分词端点。

        这是契约允许的分支，调用方必须回退到启发式。将来若 Ollama 提供
        ``/api/tokenize``，在这里接上即可——调用方不需要改。
        """
        return None

    def stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        """流式对话。异常在**第一次迭代**时抛出（见 :class:`ChatBackend`）。"""
        payload = self._build_payload(request)

        with _client(self.host, self.allow_remote) as client:
            try:
                with client.stream("POST", "/api/chat", json=payload) as response:
                    _raise_for_status(response, what=f"对话（{self.model}）")
                    yield from _iter_ndjson_chunks(response)
            except (httpx.RequestError, ssl.SSLError) as exc:
                _wrap_transport_error(exc, self.host, what=f"对话（{self.model}）")

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """把 :class:`ChatRequest` 映射成 ``/api/chat`` 的请求体。

        抽成一个方法而不是留在 :meth:`stream` 里，是为了让"请求体长什么样"能
        被单独断言。工具调用与 ``think`` 都属于**只在请求体这一层存在**的东西：
        它们进去、出来，中间的记忆核心永远不知道它们来过。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_build_message(message) for message in request.messages],
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": _build_options(request),
        }
        if request.tools:
            payload["tools"] = [_build_tool(spec) for spec in request.tools]
        # 顶层参数从 extra 里提上来。不这么做的话它会被 _build_options 塞进
        # options（那里已经跳过），于是"设了却没生效"——见 _TOP_LEVEL_EXTRA。
        for key in _TOP_LEVEL_EXTRA:
            if key in request.extra:
                payload[key] = request.extra[key]
        return payload


def _build_tool(spec: ToolSpec) -> dict[str, Any]:
    """一个工具声明，套上 OpenAI 的 ``{"type":"function", ...}`` 外壳。

    注意方向：**入参**（我们发给 Ollama 的）要这个外壳，而**出参**（Ollama 返回的
    ``tool_calls``）没有它。两者不对称是 Ollama 的实际形状，不是笔误——实测
    Ollama 0.40.2 收下这个外壳并正确派发了调用，返回时却把外壳去掉了。
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters or {"type": "object", "properties": {}},
        },
    }


def _build_message(message: ChatMessage) -> dict[str, Any]:
    """一条消息。工具调用在这里被序列化成 Ollama 认的形状。

    ``tool_calls`` 回灌时**带上 ``id``**：工具结果靠 ``tool_call_id`` 找到它对应的
    那次调用，id 丢了配对就断了。**不带 Ollama 自己在 ``function`` 里加的那个
    ``index``**——实测把不带 ``index`` 的版本回灌给它，它照样把工具结果与那次调用
    配上了。``index`` 是它输出时的内部编号，回灌时带上等于把内部编号当成了协议。
    """
    if message.audio:
        # **明说，不静默丢**。Ollama 的 ChatMessage 只有 ``images``、没有 ``audio``
        # （官方 OpenAI 规格实测，2026-10-13）——丢掉它的话模型会收到一条"用户说了话"
        # 却听不见声音的记录。走到这里说明能力门没拦住（那边对 Ollama 恒为假）。
        raise ChatBackendError(
            "这条消息带着原声，而 Ollama 的接口收不了音频（只有 images 字段）。"
            "这一轮该走转写文字那条路——能力门见 MainWindow._model_can_hear。"
        )
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    # **非空才写**（同上面那三个字段）：不带图的请求与加这个字段之前逐字节一致。
    # 实测过 Ollama 收 ``tool`` 行上的 ``images``——1×1 PNG 试下来报的是"这个模型
    # 不支持图像输入"，那是能力问题，不是字段问题。
    if message.images:
        payload["images"] = list(message.images)
    return payload


#: ``extra`` 里属于**请求体顶层**而非 ``options`` 成员的键。
#:
#: ``think`` 落进 ``options`` 会被 Ollama **静默忽略**，症状是"明明关了思考，
#: token 账还是开着思考的数"——一个不报错的错。这个集合就是那件事的唯一落点，
#: 改这里等于改所有请求的行为，所以它只有这一处。
_TOP_LEVEL_EXTRA: frozenset[str] = frozenset({"think"})


def _build_options(request: ChatRequest) -> dict[str, Any]:
    """把 :class:`ChatRequest` 映射成 Ollama 的 ``options``。

    只有能一一对应的参数才进来；``extra`` 里的键原样并入（除了
    :data:`_TOP_LEVEL_EXTRA` 里的那几个，它们的落点在请求体顶层）——这是供应商
    特有参数的出口，也是它们唯一被允许存在的位置。
    """
    options: dict[str, Any] = {"temperature": float(request.temperature)}
    if request.max_tokens is not None:
        options["num_predict"] = int(request.max_tokens)
    if request.stop:
        options["stop"] = list(request.stop)
    for key, value in request.extra.items():
        if key in options:
            # 静默覆盖会让人以为设的是 ChatRequest.temperature 而实际生效的是 extra
            raise ChatBackendError(
                f"ChatRequest.extra 里的 {key!r} 与显式字段冲突。请只在一处设置它。"
            )
        if key in _TOP_LEVEL_EXTRA:
            continue
        options[key] = value
    return options


def _iter_ndjson_chunks(response: httpx.Response) -> Iterator[ChatChunk]:
    """把 NDJSON 流翻译成 :class:`ChatChunk`。

    Ollama 每行一个 JSON 对象：``{"message":{"role":"assistant","content":"..."},"done":false}``，
    最后一行 ``done=true`` 并附带 ``prompt_eval_count`` / ``eval_count``。

    **不用 ``response.iter_lines()`` 直接切**：它内部按 ``iter_text`` 分块再缓冲，
    在流式路径上多一层字符串拷贝。直接在字节层按 ``\\n`` 切，省掉中间那层文本
    解码——反正每个 JSON 对象都很小，逐行解码的代价可以忽略。
    """
    buffer = b""
    for raw in response.iter_bytes():
        buffer += raw
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            chunk = _parse_line(line)
            if chunk is not None:
                yield chunk
    # 流可能不以换行结尾（HTTP 分块边界不保证对齐行边界）
    if buffer.strip():
        chunk = _parse_line(buffer)
        if chunk is not None:
            yield chunk


def _parse_line(line: bytes) -> ChatChunk | None:
    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        # 单行坏了不该打断整段回复——本地模型偶尔会吐出半个 JSON 的统计行
        return None

    if error := event.get("error"):
        raise ChatBackendError(f"Ollama 在流中报错：{error}")

    message = event.get("message") or {}
    return ChatChunk(
        text=str(message.get("content") or ""),
        done=bool(event.get("done", False)),
        prompt_tokens=event.get("prompt_eval_count"),
        completion_tokens=event.get("eval_count"),
        tool_calls=_parse_tool_calls(message),
        raw=event,
    )


def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """把 Ollama 的 ``tool_calls`` 归一化成 :class:`ToolCall`。

    **实测（Ollama 0.40.2 / qwen3-4b），它与 OpenAI 差三处**，全部在这里吃掉::

        {"message": {"role": "assistant", "content": "",
          "tool_calls": [{"id": "vyXmwpr3JNmTY5B9MXvrklqkf6r9TBSC",
                          "function": {"index": 0, "name": "get_weather",
                                       "arguments": {"city": "北京"}}}]}}

    1. 没有 ``"type": "function"`` 外壳；
    2. ``function.arguments`` 是**对象**，不是 OpenAI 那种 JSON 字符串；
    3. ``function`` 里多一个 ``index``。

    第 2 条正是 :class:`ToolCall` 的 ``arguments`` 存 ``dict`` 而非字符串的原因——
    在这里做归一化只需要一个 ``isinstance`` 判断，而不是一次 ``json.loads`` 往返。

    第 3 条被丢弃：``index`` 是 Ollama 输出时的内部编号，不是协议的一部分，
    回灌时带上它等于把它当成接口。

    **id 缺失时在这里合成。** 老版本 Ollama 与部分兼容端点不返回 id，而"工具结果
    必须带 tool_call_id"是协议层的硬需求——放任 id 为 ``None`` 逃到上层，就是让每个
    消费端各写一遍兜底，并且各自写得不一样。合成用随机后缀而不是 ``call-{index}``：
    同一段对话里两条 assistant 消息都会有 ``index=0`` 的那次调用，那种 id 会在
    服务端把结果配到错误的请求上，而**配错不会报错**。
    """
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return ()

    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        if not name:
            # 没有名字就无从派发。跳过而不是抛异常：一条畸形的 tool_call 不该让
            # 整段回复作废，而"模型想调一个不存在的工具"是模型的问题不是协议的问题。
            continue
        call_id = str(entry.get("id") or "").strip()
        calls.append(
            ToolCall(
                id=call_id or f"call_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=_as_arguments(function.get("arguments"), name),
            )
        )
    return tuple(calls)


def _as_arguments(value: Any, name: str) -> dict[str, Any]:
    """把 ``function.arguments`` 归一化成 dict。

    接受三种：对象（Ollama，正常路径）、JSON 字符串（OpenAI 兼容端点）、``None``
    （无参调用）。**其余一律抛错**——尤其是"确实不是 JSON 的字符串"和数组：那
    说明端点的形状与我们的假设不符，此时猜一个 dict 塞进去，工具会收到一个它
    看不懂的参数然后在内部炸掉，而错误信息会把锅甩给工具实现。
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError as exc:
            raise ChatBackendError(
                f"工具 {name!r} 的 arguments 是字符串但不是合法 JSON：{value[:200]!r}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise ChatBackendError(
            f"工具 {name!r} 的 arguments 解析出来是 {type(parsed).__name__}，不是对象。"
        )
    raise ChatBackendError(
        f"工具 {name!r} 的 arguments 是 {type(value).__name__}，既不是对象也不是 JSON 字符串。"
    )


def _guess_context_window(model: str) -> int:
    """按模型名猜上下文窗口。猜不中给保守的 :data:`_FALLBACK_CONTEXT_WINDOW`。

    **这是备选路径，不是首选。** 首选 :func:`query_context_window`——它问的是
    服务端实际会用的值。
    """
    name = model.lower()
    for prefix, window in _CONTEXT_WINDOWS:
        if name.startswith(prefix):
            return window
    return _FALLBACK_CONTEXT_WINDOW


def query_context_window(
    model: str, host: str = DEFAULT_OLLAMA_HOST, allow_remote: bool = False
) -> int:
    """问 Ollama 这个模型真正的上下文窗口是多少。

    权威答案是 ``/api/show`` 返回的 ``parameters`` 里的 ``num_ctx``——**那是运行时
    生效的值**，也就是 Ollama 实际会用来截断的那个数。

    为什么不能用 ``model_info`` 里的 ``<架构>.context_length``：那是能力上限
    （架构支持多长），不是运行配置。本机的 qwen3-4b 两者差了一倍多——架构报 40960，
    而 Modelfile 写的是 ``num_ctx 16384``，所以真实可用的是 16384。按上限算预算
    等于把窗口想大了一倍，多注入的记忆会在服务端被无声丢掉。

    问不到（服务没起来、模型不在、响应不合预期）时退回 :func:`_guess_context_window`：
    宁可用一个可能偏保守的猜测，也不要因为一次探询失败就拒绝构造后端。
    """
    try:
        with _client(host, allow_remote) as client:
            response = client.post("/api/show", json={"model": model})
            if not response.is_success:
                return _guess_context_window(model)
            payload = json.loads(response.text)
    except (httpx.RequestError, json.JSONDecodeError, ChatBackendError, ValueError):
        return _guess_context_window(model)

    found = _parse_num_ctx(payload.get("parameters") or "")
    if found is None:
        return _guess_context_window(model)
    return found


def _parse_num_ctx(parameters: str) -> int | None:
    """从 ``/api/show`` 的 ``parameters`` 块里取 ``num_ctx``。

    那个字段是个多行字符串，每行形如 ``num_ctx                        16384``
    （中间是空格对齐，不是等号）。没写 ``num_ctx`` 的模型很常见——那就该退回去猜，
    因为"没写"意味着 Ollama 用它自己的默认值（通常 4096），而 4096 这个数我们
    无从得知，只能靠保守估计。
    """
    for line in parameters.splitlines():
        key, _, value = line.strip().partition(" ")
        if key == "num_ctx":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


# ── 嵌入后端 ────────────────────────────────────────────────────────────────

#: 探测维度用的文本。**内容无关紧要**——任何非空输入都会走完同一条前向，返回的
#: 向量长度只由模型结构决定。取一个汉字而不是 ``"x"``，是为了让某些按 token 数
#: 早早退出的实现也走完正常路径。
_DIM_PROBE_TEXT = "维度"


def probe_embedding_dim(
    model: str, host: str = DEFAULT_OLLAMA_HOST, *, allow_remote: bool = False
) -> int:
    """问 Ollama 这个嵌入模型吐出几维的向量。

    **为什么非问不可**：:attr:`OllamaEmbeddingBackend.dim` 是必填的，而它不是可有
    可无的元数据——它进了 ``space_id``（``ollama/nomic-embed-text@768``），而
    ``space_id`` 是库里每个向量的身份。**写错的维度不会被任何一处发现**：向量照样
    存进去，余弦照样算出一个数，只是那个数没有意义。

    Ollama 没有"只查维度"的端点，``/api/show`` 里那条 ``embedding_length`` 是按
    架构名分键的（``nomic-bert.embedding_length``、``bert.embedding_length``…），
    按它取值等于把 Ollama 的内部命名当接口使。所以这里**实际嵌一次**：一次前向，
    一百毫秒左右，而它换来的是一段不会说谎的字节数。

    Raises:
        EmbeddingBackendError: 连不上、模型不存在、或响应形状不对。**没有兜底值**
            ——猜一个维度比报错糟得多（见上）。
    """
    with _client(host, allow_remote) as client:
        try:
            response = client.post("/api/embed", json={"model": model, "input": [_DIM_PROBE_TEXT]})
        except (httpx.RequestError, ssl.SSLError) as exc:
            _wrap_transport_error(exc, host, what=f"探测嵌入模型（{model}）")
        if not response.is_success:
            body = response.text.strip()[:300]
            hint = (
                "（模型未下载？先跑 ollama pull <模型名>）" if response.status_code == 404 else ""
            )
            raise EmbeddingBackendError(
                f"探测嵌入模型（{model}）失败：HTTP {response.status_code} {body}{hint}"
            )
        text = response.text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmbeddingBackendError(f"嵌入模型的响应不是合法 JSON：{text[:200]!r}") from exc

    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or not vectors or not isinstance(vectors[0], list):
        raise EmbeddingBackendError(
            f"嵌入模型的响应里没有可用的 'embeddings'（Ollama 版本过旧？需要 >= 0.1.36 "
            f"才有 /api/embed）：{text[:200]!r}"
        )
    if not vectors[0]:
        raise EmbeddingBackendError(f"嵌入模型（{model}）返回了一个零维向量，它不能用来建空间。")
    return len(vectors[0])


@dataclass
class OllamaEmbeddingBackend:
    """Ollama 上的一个嵌入模型。

    Attributes:
        model: 如 ``"nomic-embed-text"``。
        dim: 向量维度。**必填**——Ollama 没有"只查维度"的端点，而空间标识里
            必须带维度（见 :class:`~axo.llm.base.EmbeddingBackend`）。
            调用方用 :func:`probe_embedding_dim` 当场量出它再传进来：这个数进了
            ``space_id``，也就是库里每个向量的身份；猜错的结果是"存得进、算得出，
            只是那个数没有意义"。
        space_id: 记忆空间标识。``None`` 时按 ``ollama/<model>@<dim>`` 生成。
            允许显式覆盖是为了表达"同一个模型的不同权重版本"——那种情况下模型名
            相同而向量空间不同，是这个设计必须能区分的头号情形。
    """

    model: str
    dim: int
    host: str = DEFAULT_OLLAMA_HOST
    space_id: str | None = None
    allow_remote: bool = False
    batch_size: int = 32
    """单次请求的文本条数上限。批量回填时用它分片，避免一次 POST 塞进几千条文本。"""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("OllamaEmbeddingBackend.model 不能为空。")
        if self.dim < 1:
            raise ValueError(f"OllamaEmbeddingBackend.dim 必须 >= 1，收到 {self.dim!r}。")
        if self.space_id is None:
            self.space_id = f"ollama/{self.model}@{self.dim}"
        # 显式覆盖时必须自带维度信息，否则一个手写的 space_id 会让维度检查形同虚设
        if f"@{self.dim}" not in self.space_id:
            raise ValueError(
                f"space_id={self.space_id!r} 里没有 '@{self.dim}'。空间标识必须带上维度，"
                f"这样维度变化能在字符串比较层面被发现，而不必等到矩阵乘法报形状错误。"
            )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """批量嵌入，返回 ``(len(texts), dim)`` 的 float32 矩阵，已 L2 归一化。

        Ollama 的 ``/api/embed`` 一次可收多条；这里再按 :attr:`batch_size` 分片，
        是为了让"首次全库回填"这种大批量任务的内存占用与请求体大小都可控。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        batches: list[np.ndarray] = []
        with _client(self.host, self.allow_remote) as client:
            for start in range(0, len(texts), self.batch_size):
                window = list(texts[start : start + self.batch_size])
                try:
                    response = client.post(
                        "/api/embed",
                        json={"model": self.model, "input": window},
                    )
                except (httpx.RequestError, ssl.SSLError) as exc:
                    _wrap_transport_error(exc, self.host, what=f"嵌入（{self.model}）")
                _raise_for_status(response, what=f"嵌入（{self.model}）")
                batches.append(_parse_embeddings(response.text, expected=len(window), dim=self.dim))

        matrix = np.concatenate(batches, axis=0) if batches else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(matrix)


def _parse_embeddings(body: str, *, expected: int, dim: int) -> np.ndarray:
    """解析 ``/api/embed`` 的响应，并当场校验形状。"""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise EmbeddingBackendError(f"嵌入响应不是合法 JSON：{body[:200]!r}") from exc

    vectors = payload.get("embeddings")
    if vectors is None:
        raise EmbeddingBackendError(
            f"嵌入响应里没有 'embeddings' 字段（Ollama 版本过旧？需要 >= 0.1.36 才有 /api/embed）："
            f"{body[:200]!r}"
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (expected, dim):
        raise EmbeddingBackendError(
            f"嵌入响应的形状是 {matrix.shape}，期望 {(expected, dim)}。"
            f"维度不符通常意味着 space_id 里记的 dim 与实际模型不匹配——"
            f"这正是空间标识要带维度的原因，此向量**不得**写进该空间。"
        )
    return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。零向量保持零向量，不产生 NaN。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # where= 避免了 0/0；范数为 0 的行结果是 0，而不是 NaN
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
