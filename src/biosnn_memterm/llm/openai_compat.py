"""云端 OpenAI 兼容后端（DeepSeek / OpenRouter / vLLM / LM Studio ...）。

**它只实现对话，不实现嵌入。** 这不是省事：多数云端厂商（包括 DeepSeek）根本没有
嵌入端点，而记忆的召回、回填、探针整条路都靠向量。所以"对话走云端、嵌入仍走本机
Ollama"是这个模块的**前提**而不是它的一个选项——``MemorySyncWorker`` 与
``AssemblyWorker`` 因此一个字都不用改，库里的 ``space_id`` 也原样不变。

**这个模块存在的理由是"形状对不上"，不是"多一个 HTTP 客户端"。** 我们自己的
:class:`~biosnn_memterm.llm.base.ChatChunk` 契约在 Ollama 那边是成立的，到了云端
有三处必须在这里吃掉：

1. **SSE 而不是 NDJSON**：``data: {...}`` 行、空行分隔、``[DONE]`` 收尾。
2. **``delta.tool_calls`` 按 index 分片累积**——``id`` 与 ``function.name`` 只在
   第一片，``arguments`` 是**一段一段拼起来的 JSON 字符串**。这是整块代码里唯一
   真正新的逻辑，单独一个 :class:`_ToolCallAccumulator`。
3. **回灌方向的两处不对称**：``function.arguments`` 必须是 **JSON 字符串**（Ollama
   收对象），且要补 ``{"type": "function"}`` 外壳。

**第 2 条是"契约不变"的关键。** ``base.ChatChunk`` 上写着"``tool_calls`` 整条一次
到齐，消费端不需要任何碎片累积逻辑"，``agent/loop.py`` 也据此**覆盖而不累积**。
碎片在**这个模块内部**拼完，出去的东西与 Ollama 给的一模一样，所以那句话至今成立
——它是这个模块的验收标准，不是一句注释。

**key 绝不进任何给用户看的句子。** 见 :func:`_redact`：它进错误串之后会出现在状态栏、
日志和用户的截图里，那是凭据泄露最常见的一条路。
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from .base import (
    ChatBackendError,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolSpec,
)
from .ollama import check_host_is_local

__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "FALLBACK_CLOUD_MODELS",
    "MODEL_ID_PREFIX",
    "OpenAICompatChatBackend",
    "endpoint_problem",
    "fallback_models",
    "list_remote_models",
    "pick_cloud_chat_model",
    "probe",
    "query_context_window",
]

DEFAULT_OPENAI_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_OPENAI_MODEL = "deepseek-chat"

#: ``model_id`` 的前缀，也是 :meth:`~.settings_store.BackendSettings` 里
#: :data:`~.settings_store.PROVIDER_CLOUD` 那一侧在会话 ``model_ref`` 上留下的来源名。
#:
#: **它是后端类型，不是厂牌。** 写 ``deepseek/`` 的话，同一段代码接 OpenRouter
#: 或自建 vLLM 时那个前缀就在撒谎。主窗口拼会话的 ``model_ref`` 也用这一个常量——
#: 两处各写一遍字面量，改一处漏一处时翻旧会话会看不出那条回复出自哪一类后端。
MODEL_ID_PREFIX = "openai-compat"

#: ``/models`` 拉不到时摆在用户面前的几个名字。
#:
#: **它现在只是数据，不是兜底行为。** 拉取失败时 worker 照常发 ``failed``——让
#: worker 自己吞掉错误返回这一份，症状就是"列表里为什么是这几个"没有任何地方
#: 说得出来（``loaded`` 信号只有一个 ``list``，装不下"我用了兜底，因为……"）。
#: 什么时候用它、以及要不要同时把那句原因写进状态栏，由主窗口决定。
FALLBACK_CLOUD_MODELS: tuple[str, ...] = ("deepseek-chat", "deepseek-reasoner")

#: 端点不报上下文窗口时的兜底值（token），128K。
#:
#: **这不是权威值**（取数日期 2026-10，**待核实**：同一个模型名在两代之间差一倍
#: 是常事）。首选 :func:`query_context_window` 去问端点——DeepSeek 的 ``/models``
#: 就回 ``context_window``。猜错的代价只在"算大"这个方向上：预算算大 → 服务端静默
#: 截断，而截掉的是最早的消息，正是人格锚所在的位置。
_DEFAULT_CLOUD_CONTEXT_WINDOW = 131072

#: 连接要快失败（地址写错时不该让设置页卡住），读取要长超时（云端排队 + 长回复）。
#: connect 比 Ollama 那份（5 秒）宽一点：这里多一次 DNS 与 TLS 握手。
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)

#: SSE 的数据行前缀。字段名后面那个空格可写可不写，所以取前缀再 strip。
_DATA_PREFIX = b"data:"

#: OpenAI 流的结束哨兵。
_DONE = b"[DONE]"

#: key 的形状。见 :func:`_redact` 的第二道。
_SK_KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


# ── 护栏：地址必须可信 ──────────────────────────────────────────────────────


def endpoint_problem(base_url: str) -> str:
    """这个地址可不可信。可信返回空串，否则返回一句能直接显示的话。

    **方向与 :func:`~biosnn_memterm.llm.ollama.check_host_is_local` 正好相反。**
    那边管的是"对话内容别离开这台设备"，所以拦的是*非本机*；这边管的是"API key
    会跟着每个请求发出去"，所以要求 **https，或者本机的 http**——自建推理机
    （vLLM、LM Studio）走明文 http 是正常的，别人机器上的明文 http 则是把 key
    摆在路上。
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return f"服务地址 {base_url!r} 解析不了。"
    scheme = parsed.scheme.lower()
    if not scheme or not parsed.netloc:
        return (
            f"服务地址 {base_url!r} 不完整——它要像 https://api.deepseek.com/v1 这样，"
            f"带上协议头与主机名。"
        )
    if scheme == "https":
        return ""
    if scheme == "http" and check_host_is_local(base_url):
        return ""
    return (
        f"服务地址 {base_url!r} 既不是 https，也不是本机地址。API key 会跟着每一个"
        f"请求发出去，明文 http 意味着它在这条路上可以被任何人读到。自建的推理机请"
        f"用 https，或者就填本机地址（127.0.0.1）。"
    )


def _client(base_url: str, api_key: str) -> httpx.Client:
    problem = endpoint_problem(base_url)
    if problem:
        raise ChatBackendError(problem)
    # 没填 key 时不发那个头。发一个空的 ``Bearer `` 比不发更糟：它把"没配"变成一个
    # 语法上合法、语义上无意义的凭据，而 401 的排查方向就此从"忘了填"偏到"填错了"。
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=_TIMEOUT)


# ── 错误：一律翻译成人话，且不带 key ────────────────────────────────────────


def _redact(text: str, api_key: str = "") -> str:
    """把 key 从任何要给用户看的句子里擦掉。

    两道。第一道按我们手里那把 key 精确替换——它抓得住"服务端把 key 原样回显到
    错误体里"这种情况。第二道按 ``sk-`` 的形状兜住剩下那半：被引号、前缀或者日志
    格式动过手脚的回显，第一道匹配不上，形状还认得出来。
    """
    if api_key:
        text = text.replace(api_key, "***")
    return _SK_KEY_SHAPE.sub("sk-***", text)


#: 各状态码那句话。**与 Ollama 那份不能共用**：它给 404 的提示是
#: "先跑 ollama pull <模型名>"（``ollama.py:152``），云端下这句荒谬。
_STATUS_HINTS: dict[int, str] = {
    400: "（请求被拒。多半是模型名或参数不对。）",
    401: "（API key 无效、过期，或者根本没填。）",
    402: "（账户余额不足。）",
    403: "（这个 key 没有访问该模型的权限。）",
    404: "（地址或模型名不对。服务地址通常要以 /v1 结尾——用「测试连接」核实一下。）",
    422: "（参数不合这个端点的口味。）",
    429: "（触发限流或配额用尽，等一会儿再试。）",
}


def _error_detail(body: str) -> str:
    """从错误体里取出那句话。

    OpenAI 兼容端点把话藏在 ``{"error": {"message": "..."}}`` 里——**比 Ollama 的
    ``{"error": "..."}`` 多一层嵌套**。反代返回 HTML 错误页（502）时整段原样带出来：
    "解析不了就退化成一句无附加信息"会把唯一有用的线索丢掉。
    """
    if not body:
        return "无附加信息"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if not isinstance(payload, dict):
        return body
    error = payload.get("error", payload)
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error or body)
    return str(error)


def _raise_for_status(response: httpx.Response, *, what: str, api_key: str = "") -> None:
    """把 HTTP 状态码翻译成能直接显示给用户的句子。

    **取正文之前必须先确认它被读出来了。** 流式请求（``client.stream(...)``）返回时
    正文还在网络上，此刻取 ``response.text`` 抛的是 ``httpx.ResponseNotRead``——
    于是一条干净的错误（"API key 无效"）变成一个 httpx 回溯。触发条件是**流式请求
    撞上非 2xx**：key 过期、限流、模型名写错，全是用户最常撞上的那几种。
    """
    if response.is_success:
        return
    try:
        body = response.text.strip()
    except httpx.ResponseNotRead:
        response.read()
        body = response.text.strip()
    detail = _redact(_error_detail(body), api_key)[:500]
    if response.status_code >= 500:
        hint = "（云端服务端报错，通常等一会儿就好——不是本机的问题。）"
    else:
        hint = _STATUS_HINTS.get(response.status_code, "")
    raise ChatBackendError(f"{what}失败：HTTP {response.status_code} {detail}{hint}")


def _wrap_transport_error(
    exc: httpx.RequestError, base_url: str, *, what: str, api_key: str = ""
) -> None:
    raise ChatBackendError(
        _redact(
            f"{what}失败：连不上 {base_url}。原始错误：{type(exc).__name__}: {exc}",
            api_key,
        )
    ) from exc


# ── 模型清单 ────────────────────────────────────────────────────────────────


def list_remote_models(
    base_url: str = DEFAULT_OPENAI_BASE_URL, *, api_key: str = ""
) -> list[dict[str, Any]]:
    """列出这个端点上的模型。

    形状与 :func:`~biosnn_memterm.llm.ollama.list_local_models` **逐键对齐**
    （``name`` / ``short_name`` / ``size`` / ``parameter_size`` / ``capabilities`` …），
    因为消费它的那几处（下拉框、``pick_cloud_chat_model``、主窗口的装配）只认这一套键。
    云端多一个 ``context_window``，取值可能为 ``None``（端点不报，多数都不报）。

    连不上时抛 :class:`ChatBackendError`——调用方需要知道"为什么是空的"。
    """
    with _client(base_url, api_key) as client:
        try:
            response = client.get("/models")
        except httpx.RequestError as exc:
            _wrap_transport_error(exc, base_url, what="列出云端模型", api_key=api_key)
        _raise_for_status(response, what="列出云端模型", api_key=api_key)
    return _parse_models(response.text)


def _parse_models(text: str) -> list[dict[str, Any]]:
    """``GET /models`` 的响应 → 模型条目。**纯解析，不碰网络**——核心里直接喂字节测。

    ``capabilities`` **显式打上 ``("completion",)``**。不能指望
    :func:`~biosnn_memterm.llm.ollama.is_chat_capable` 里"空 capabilities 当可用"
    那个巧合：将来那条语义一收紧，云端模型会**整批**从下拉框里消失，而症状是
    "明明配好了，列表却是空的"。
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChatBackendError(f"模型清单不是合法 JSON：{text[:200]!r}") from exc
    raw = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ChatBackendError(f"模型清单的形状不认识（顶层没有 data 数组）：{text[:200]!r}")

    models: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("id") or "").strip()
        if not name:
            continue
        models.append(
            _cloud_entry(
                name,
                context_window=_as_int(entry.get("context_window", entry.get("context_length"))),
                modified_at=str(entry.get("created") or ""),
            )
        )
    models.sort(key=lambda item: item["name"])
    return models


def _cloud_entry(
    name: str, *, context_window: int | None = None, modified_at: str = ""
) -> dict[str, Any]:
    """一个云端模型条目。**字段与 Ollama 那份逐键对齐**，空的那几个只是它没有。"""
    return {
        "name": name,
        # 云端的 id 不带 ``:latest`` 那种默认标签，所以脱标签是恒等变换——
        # 这里不做 ``short_model_name`` 那一步，是因为它要处理的是 Ollama 的约定。
        "short_name": name,
        "size": 0,
        "parameter_size": "",
        "quantization": "",
        "family": "",
        "modified_at": modified_at,
        "capabilities": ("completion",),
        "context_window": context_window,
    }


def fallback_models() -> list[dict[str, Any]]:
    """兜底预设，形状与 :func:`list_remote_models` 的返回值一致。

    **只有 ``context_window`` 这一格与"服务端没报"不同：这里明写兜底值。**
    理由是这一份只出现在 :func:`list_remote_models` **已经失败之后**（见
    :data:`FALLBACK_CLOUD_MODELS` 的说明），而那正是主窗口在界面线程上构造后端的
    时候——``context_window=None`` 会让构造函数再发一次 ``GET /models``，那一次
    必然以同一个原因失败，只是先卡满 ``connect`` 那 10 秒。同一个数，少一次注定
    失败的往返。

    填进去的确实是猜测（见 :data:`_DEFAULT_CLOUD_CONTEXT_WINDOW`），但这与
    :func:`query_context_window` 问不到时的返回值**是同一个数**——差别只在于
    这一次连问都没问。
    """
    return [
        _cloud_entry(name, context_window=_DEFAULT_CLOUD_CONTEXT_WINDOW)
        for name in FALLBACK_CLOUD_MODELS
    ]


def pick_cloud_chat_model(models: list[dict[str, Any]]) -> str:
    """挑一个云端模型，挑不到返回空串。

    优先 :data:`DEFAULT_OPENAI_MODEL`——与
    :func:`~biosnn_memterm.llm.ollama.pick_embedding_model` 同一个理由：默认值与
    "从清单里挑哪个"是同一个决定的两半，分成两个名字就会出现"默认按 A 建后端，
    却把 B 记进了账本"。
    """
    names = [str(entry.get("name") or "") for entry in models]
    if DEFAULT_OPENAI_MODEL in names:
        return DEFAULT_OPENAI_MODEL
    return names[0] if names else ""


def query_context_window(
    model: str, base_url: str = DEFAULT_OPENAI_BASE_URL, *, api_key: str = ""
) -> int:
    """问端点这个模型的上下文窗口，问不到就猜。

    **首选问服务端**：窗口是随版本变的，写死的数只会在"预算算大"这个方向上出错，
    而算大的后果是服务端静默截断——截掉的是最早的消息，正是人格锚所在的位置。

    问不到（端点不报、地址不对、没联网）时退回 :data:`_DEFAULT_CLOUD_CONTEXT_WINDOW`：
    宁可用一个可能偏保守的猜测，也不要因为一次探询失败就拒绝构造后端。
    """
    try:
        with _client(base_url, api_key) as client:
            response = client.get("/models")
            if not response.is_success:
                return _DEFAULT_CLOUD_CONTEXT_WINDOW
            entries = _parse_models(response.text)
    except (httpx.RequestError, ChatBackendError, ValueError):
        return _DEFAULT_CLOUD_CONTEXT_WINDOW

    for entry in entries:
        if entry["name"] == model:
            window = entry.get("context_window")
            # 小于 512 的窗口装配预算根本没法工作（人格锚本身就要 800 token），
            # 所以那种值一定是解析错了，当作"没报"处理。
            if isinstance(window, int) and window >= 512:
                return window
            break
    return _DEFAULT_CLOUD_CONTEXT_WINDOW


def probe(base_url: str = DEFAULT_OPENAI_BASE_URL, *, api_key: str = "") -> tuple[bool, str]:
    """探活。``(通不通, 一句能显示的话)``。

    **签名与 :func:`~biosnn_memterm.llm.ollama.probe` 不同，是故意的。** 那边返回
    ``str | None``（版本号或"没起来"），因为 Ollama 只有"在不在"这一种失败；云端
    最常撞上的却是 **401——key 不对**。把 401 也压成"连不上"，会让人去查地址，
    而地址根本没问题：指错方向的提示比没有提示更费时间。

    **不抛**——调用方是设置页的「测试连接」，它要的是"通不通"这个答案。
    """
    try:
        with _client(base_url, api_key) as client:
            response = client.get("/models")
            if not response.is_success:
                _raise_for_status(response, what="测试连接", api_key=api_key)
            entries = _parse_models(response.text)
    except ChatBackendError as exc:
        return False, str(exc)
    except (httpx.RequestError, ValueError) as exc:
        return False, _redact(f"连不上 {base_url}：{type(exc).__name__}: {exc}", api_key)
    if not api_key:
        return True, f"连上了，列出 {len(entries)} 个模型——但还没填 API key，对话会被拒。"
    return True, f"连上了，列出 {len(entries)} 个模型"


# ── 对话后端 ────────────────────────────────────────────────────────────────


@dataclass
class OpenAICompatChatBackend:
    """云端 OpenAI 兼容端点上的一个对话模型。

    Attributes:
        model: 端点的模型 id，如 ``"deepseek-chat"``。
        base_url: 端点根，**带 ``/v1``**（不带的话请求会打到 404）。
        api_key: 明文 key。盘上那份是 DPAPI 密文，解密发生在
            :mod:`~biosnn_memterm.ui.settings_store` 里，到这里的已经是明文。
        context_window: 上下文窗口。``None`` 时去问端点（见
            :func:`query_context_window`）——**那是一次网络请求**，所以
            ``__post_init__`` 会联网，构造它请放在工作线程里。
    """

    model: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    api_key: str = ""
    context_window: int | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("OpenAICompatChatBackend.model 不能为空。")
        # 先校验地址再联网：地址不可信时不必把请求（以及 key）发出去。
        problem = endpoint_problem(self.base_url)
        if problem:
            raise ChatBackendError(problem)
        if self.context_window is None:
            self.context_window = query_context_window(
                self.model, self.base_url, api_key=self.api_key
            )
        if self.context_window < 512:
            raise ValueError(
                f"context_window={self.context_window} 太小，装配预算无法工作"
                f"（人格锚本身就要 800 token）。"
            )

    # ------------------------------------------------------------ 协议实现

    @property
    def model_id(self) -> str:
        """``"openai-compat/<模型名>"``。进注入账本的就是它。

        **前缀是后端类型，不是厂牌**（见 :data:`MODEL_ID_PREFIX`）。
        """
        return f"{MODEL_ID_PREFIX}/{self.model}"

    def count_tokens(self, text: str) -> int | None:  # noqa: ARG002 — 协议签名，刻意忽略
        """返回 ``None``：发请求之前没有廉价的分词端点。

        这是契约允许的分支，调用方（装配预算）必须回退到启发式。
        """
        return None

    def stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        """流式对话。异常在**第一次迭代**时抛出（见 :class:`ChatBackend`）。"""
        payload = self._build_payload(request)

        with _client(self.base_url, self.api_key) as client:
            try:
                with client.stream("POST", "/chat/completions", json=payload) as response:
                    _raise_for_status(response, what=f"对话（{self.model}）", api_key=self.api_key)
                    yield from _iter_sse_chunks(response)
            except httpx.RequestError as exc:
                _wrap_transport_error(
                    exc, self.base_url, what=f"对话（{self.model}）", api_key=self.api_key
                )

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """把 :class:`ChatRequest` 映射成 ``/chat/completions`` 的请求体。

        抽成方法（而不是留在 :meth:`stream` 里）是为了让"请求体长什么样"能被单独
        断言——``__post_init__`` 会联网，所以测试是拿 ``SimpleNamespace`` 非绑定调用
        这个函数的，不构造真实例。

        **与 Ollama 的差别**：那边除了几个固定字段，其余全塞进 ``options``；这里
        OpenAI 的请求体是**平的**，所以 ``extra`` 直接进顶层。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_build_message(message) for message in request.messages],
            "stream": True,
            # **统计数字要显式要，云端才会给。** 不带这个的话 ``usage`` 根本不会出现，
            # 而它是上下文指示器唯一的数据源（预算本身不依赖它，见 ``base.ChatChunk``）。
            # 代价：不认识这个字段的端点会回 400——那种端点本来就跑不了这一套。
            "stream_options": {"include_usage": True},
            "temperature": float(request.temperature),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = int(request.max_tokens)
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.tools:
            payload["tools"] = [_build_tool(spec) for spec in request.tools]
        for key, value in request.extra.items():
            if key in payload:
                # 静默覆盖会让人以为设的是 ChatRequest 上的字段，而实际生效的是 extra。
                raise ChatBackendError(
                    f"ChatRequest.extra 里的 {key!r} 与显式字段冲突。请只在一处设置它。"
                )
            payload[key] = value
        return payload


def _build_tool(spec: ToolSpec) -> dict[str, Any]:
    """一个工具声明，套上 ``{"type":"function", ...}`` 外壳。

    这个方向与 Ollama 一致（**入参**要外壳，返回的 ``tool_calls`` 不要）。
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
    """一条消息。工具调用在这里序列化成 OpenAI 认的形状。

    **回灌方向与 Ollama 差两处**（对着 ``ollama._build_message`` 看）::

        Ollama：{"id": ..., "function": {"name": ..., "arguments": {...}}}   # 对象
        这里：  {"id": ..., "type": "function",
                 "function": {"name": ..., "arguments": "{...}"}}            # JSON 字符串

    1. ``arguments`` 必须是 **JSON 字符串**（OpenAI 的原文是 "a JSON string"）。
       发一个对象过去，多数端点会回 400 或者干脆忽略整条调用——而"忽略了整条调用"
       的表现是模型收到一段它无法解释的对话，然后开始胡说。
    2. 外壳要补 ``"type": "function"``。

    这两处不对称是**协议本身的**，所以它只能活在后端层。
    """
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


# ── SSE 流 ──────────────────────────────────────────────────────────────────


def _sse_data(line: bytes) -> bytes | None:
    """从一行 SSE 里取出 ``data:`` 后面的内容；不是数据行就返回 ``None``。

    只认 ``data:``：``event:`` / ``id:`` / ``retry:`` 我们都不用，而 ``:`` 开头的
    **注释行**（有些反代拿它当心跳，一路发 ``: ping``）必须跳过——不跳的话它会被
    当成一个解析不了的负载，而"每隔几秒丢一行"这种症状没人会怀疑到心跳头上。
    """
    line = line.strip()
    if not line or line.startswith(b":"):
        return None
    if not line.startswith(_DATA_PREFIX):
        return None
    return line[len(_DATA_PREFIX) :].strip()


def _iter_sse_chunks(response: httpx.Response) -> Iterator[ChatChunk]:
    """把 SSE 流翻译成 :class:`ChatChunk`。

    ``tool_calls`` **只在收尾那一片出去**（``[DONE]``、或流就这么断了时的兜底）。
    理由有两个，都很实际：

    * 中途发出去的话，同一个 index 的后续碎片会让上层看到两次调用，而
      ``agent/loop.py`` 是**覆盖**取值的——它拿到的会是半截参数。
    * 用户点停止时生成器被 GC，``build()`` 不执行，于是截断的 arguments 不会走到
      :func:`_as_arguments` 那条会抛错的路上。停止路径因此不需要任何特判。
    """
    buffer = b""
    accumulator = _ToolCallAccumulator()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    last_usage: dict[str, Any] = {}

    def _finalize() -> ChatChunk:
        return ChatChunk(
            done=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=accumulator.build(),
            raw=dict(last_usage),
        )

    for raw in response.iter_bytes():
        buffer += raw
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            data = _sse_data(line)
            if data is None:
                continue
            if data == _DONE:
                yield _finalize()
                return
            event = _parse_event(data)
            if event is None:
                continue
            if error := _stream_error(event):
                raise ChatBackendError(error)
            usage = event.get("usage")
            if isinstance(usage, dict):
                last_usage = usage
                prompt_tokens = _as_int(usage.get("prompt_tokens"))
                completion_tokens = _as_int(usage.get("completion_tokens"))
            delta = _first_delta(event)
            if delta is None:
                continue
            accumulator.feed(delta.get("tool_calls"))
            text = _delta_text(delta)
            if text:
                yield ChatChunk(text=text, raw=event)

    # **没收到 ``[DONE]`` 也要收尾。** 有些反代在连接收尾时把它吃掉，而少发这一片
    # 的后果不是"少一段字"——是**工具调用彻底消失**：模型明明调了，闸门那边一个都
    # 没看见，于是这一轮静默地结束。宁可补一片，也不要让它变成"模型今天不听话"。
    yield _finalize()


def _parse_event(data: bytes) -> dict[str, Any] | None:
    """一片负载 → dict。坏 JSON 跳过，与 ``ollama._parse_line`` 同一条纪律：
    一行坏了不该打断整段回复。"""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _stream_error(event: dict[str, Any]) -> str:
    """流中间的 ``error`` 字段 → 一句要抛出去的话。没有则返回空串。

    后端明确报错了就不能装作没看见：那时连接通常是好的，用户看到的是"回复说着
    说着停了"，而原因只在这一行里。
    """
    error = event.get("error")
    if not error:
        return ""
    if isinstance(error, dict):
        detail = str(error.get("message") or error.get("code") or error)
    else:
        detail = str(error)
    return _redact(f"云端在流中报错：{detail}")


def _first_delta(event: dict[str, Any]) -> dict[str, Any] | None:
    """这一片里的 ``choices[0].delta``，没有就返回 ``None``。

    **``choices`` 是空列表是正常的**：开了 ``include_usage`` 之后，统计信息单独占
    一片，那一片的 ``choices`` 就是 ``[]``。把它当成畸形会丢掉 token 数。
    """
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    return delta if isinstance(delta, dict) else None


def _delta_text(delta: dict[str, Any]) -> str:
    """这一片的增量文本。

    **只认 ``content``，不认 ``reasoning_content``。** DeepSeek 的 reasoner 把思考
    过程放在后一个字段里——把它拼进 ``text`` 的话，思考过程会显示在对话区、也会被
    写进逐字稿。厂商概念不进 :class:`ChatChunk` 的字段，它唯一的出口是 ``raw``。
    """
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 少数端点回分块数组形式（``[{"type": "text", "text": "..."}]``）。
        return "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    return ""


class _ToolCallAccumulator:
    """把 ``delta.tool_calls`` 的碎片按 index 拼回完整的调用。

    **这是整个模块唯一真正新的逻辑。** OpenAI 兼容端点的工具调用不像 Ollama 那样
    整条到齐，它按 index 分片：第一片带 ``id`` 与 ``function.name``，之后每片只带
    一段 ``arguments`` 的 **JSON 字符串碎片**，拼起来才是完整参数。::

        {"index":0,"id":"call_a","function":{"name":"set_reminder","arguments":"{\\"te"}}
        {"index":0,"function":{"arguments":"xt\\":\\"买牛奶\\"}"}}

    **累加期间绝不能 ``json.loads``**：碎片不是合法 JSON 是常态，不是异常。解析只在
    :meth:`build` 里做，而且只做一次。
    """

    def __init__(self) -> None:
        #: 第一次见到某个 index 的顺序。用它（而不是 ``sorted``）是因为工具调用的
        #: 先后顺序是模型给的语义，按数字排会把并行调用排成另一个次序。
        self._order: list[int] = []
        self._ids: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self._args: dict[int, list[str]] = {}
        #: 上一片用过的 index。给"不返回 index"的端点用——那种端点每片都是同一个
        #: 调用的续片，所以默认接在上一片后面；从头数会把它切成 N 个调用。
        self._current = 0

    def feed(self, fragments: Any) -> None:
        """吃下一片 ``delta.tool_calls``。不是列表就什么都不做。"""
        if not isinstance(fragments, list):
            return
        for entry in fragments:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                index = self._current
            self._current = index
            if index not in self._args:
                self._order.append(index)
                self._args[index] = []
            call_id = entry.get("id")
            if isinstance(call_id, str) and call_id.strip():
                self._ids[index] = call_id.strip()
            function = entry.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                self._names[index] = name
            piece = function.get("arguments")
            if isinstance(piece, str):
                self._args[index].append(piece)
            elif isinstance(piece, dict):
                # 有的端点不分片、直接给对象（与 Ollama 同形）。序列化回字符串，
                # 让下面那条"拼起来再解析一次"的路成为唯一一条路。
                self._args[index].append(json.dumps(piece, ensure_ascii=False))

    def build(self) -> tuple[ToolCall, ...]:
        """拼好的调用。**只在流收尾时调一次。**"""
        calls: list[ToolCall] = []
        for index in self._order:
            name = self._names.get(index, "")
            if not name:
                # 没有名字就无从派发。跳过而不是抛：一条畸形的调用不该让整段回复
                # 作废（与 ``ollama._parse_tool_calls`` 同一条纪律）。
                continue
            calls.append(
                ToolCall(
                    # id 缺失时合成。合成用随机后缀而不是 ``call-{index}``：同一段
                    # 对话里两次 assistant 消息都会有 ``index=0`` 的那次调用，那种
                    # id 会让服务端把工具结果配到错误的请求上，而**配错不报错**。
                    id=self._ids.get(index) or f"call_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=_as_arguments("".join(self._args[index]), name),
                )
            )
        return tuple(calls)


def _as_arguments(raw: str, name: str) -> dict[str, Any]:
    """拼完的 arguments 字符串 → dict。

    **拼出来不是合法 JSON 就抛**，不猜一个 ``{}`` 塞进去——照
    :mod:`~biosnn_memterm.llm.ollama` 里写下的同一条纪律：猜的形状会让工具收到一个
    它看不懂的参数、然后在内部炸掉，而错误信息把锅甩给工具实现。

    ``""`` 是**例外**：无参工具的 ``arguments`` 就是空串，而"空"正是它要的意思
    （与 ``ollama._as_arguments`` 收 ``None`` 同理）。
    """
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChatBackendError(
            f"工具 {name!r} 的 arguments 拼起来不是合法 JSON：{text[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ChatBackendError(
            f"工具 {name!r} 的 arguments 解析出来是 {type(parsed).__name__}，不是对象。"
        )
    return parsed


def _as_int(value: Any) -> int | None:
    """能当整数就当，不能就 ``None``。

    ``None`` 不是"零"：统计信息缺失时消费端要能区分"后端没报"与"报了 0 个 token"。
    """
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
