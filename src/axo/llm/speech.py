"""语音组件的调用端：把一段文字交给独立组件合成成 wav。**这一族唯一的网络层。**

## 为什么它住在 ``llm/``

它与 ``ollama.py`` / ``openai_compat.py`` 是同一类东西（httpx、localhost 服务、
OpenAI 形状的接口），而"HTTP 客户端住 llm/"是这一层的既有秩序（分层测试对
``axo/llm/`` 放行 httpx，对纯层禁它）。协议与形状在 :mod:`axo.tts`（纯），这里只有
发请求与把失败收成人话。

## 契约：``POST {base_url}/v1/audio/speech``

``base_url`` 是**组件的根地址**（设置里填的就是它，默认 ``http://127.0.0.1:8800``），
版本前缀由调用端补——用户看到的那一格越少东西越好。请求体是
:func:`axo.tts.speech_payload`，回一段 wav 字节。**引擎对不上这个形状时改的是组件
那一侧的适配**（它归用户装的那一半），调用端一个字都不用动——那正是把协议单独抽出来
的理由。

## 失败一律是收条里的一句话

组件没在跑（最常见）、回了非 2xx、回了空体——三类各一句人话（同 ``asr.py`` 的纪律：
这个方法会被界面线程与后台线程都调到，抛出去没有人接得住）。**不带鉴权**：它默认只
听 localhost（见 ``tts.DEFAULT_SPEECH_ENDPOINT`` 的理由）。
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from ..tts import DEFAULT_SPEECH_ENDPOINT, SpeechResult, speech_payload

__all__ = ["RemoteSpeech"]

_TIMEOUT_S = 30.0
"""合成超时。**比生成宽**：克隆 TTS 第一次合成要加载参考音（几秒到十几秒），而
卡在"正在说"上的体验比慢一点差得多；超时之后照样回一句人话。"""


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"})


def _is_loopback(base_url: str) -> bool:
    """这个地址是不是本机。**本机的请求不许经过代理。**

    判据只看主机名，不解析域名：`localhost` 与 IP 字面量都在名单里，其余一律当远程
    （远程经不经代理由环境变量说了算，那是用户自己的网络配置）。
    """
    return (urlsplit(base_url).hostname or "").lower() in _LOOPBACK_HOSTS


class RemoteSpeech:
    """贴 ``/v1/audio/speech`` 的调用端。"""

    def __init__(
        self,
        base_url: str = DEFAULT_SPEECH_ENDPOINT,
        *,
        model: str = "",
        timeout: float = _TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        #: 测试缝：塞一个 ``httpx.MockTransport`` 就能在核心腿里断言请求形状。
        #: 真调用方不传，走网络（与 ``openai_compat`` 各后端同姿态）。
        self._transport = transport

    def probe(self, *, timeout: float = 0.5) -> bool:
        """端口上有没有东西在答。**只看它活着，不看它答得对不对。**

        自动启动之前先问这一句：已经有东西在响应就不该再拉一个（否则会起第二个、
        抢不到端口，然后在组件的日志里留下一串与用户无关的错）。
        """
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=timeout, trust_env=not _is_loopback(self.base_url)
            ) as client:
                return client.get("/v1/audio/speech").status_code < 500
        except httpx.HTTPError:
            return False

    def synthesize(self, text: str, *, voice: str = "") -> SpeechResult:
        """合成一段。**不抛**（见模块文档）。"""
        body = speech_payload(text.strip(), voice=voice, model=self.model)
        if not body["input"]:
            return SpeechResult(reason="没有要念的文字（这一句是空的）")
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
                # **本机地址不走代理。** httpx 默认认 HTTP_PROXY/HTTPS_PROXY，而
                # 一个 127.0.0.1 的请求被送去代理的结局是代理回一个 502（**空的**）
                # ——用户看到的是「组件回了 HTTP 502（没有正文）」，而真相是组件
                # 根本没在跑。2026-10-17 实测踩到。
                trust_env=not _is_loopback(self.base_url),
            ) as client:
                response = client.post("/v1/audio/speech", json=body)
        except httpx.HTTPError as exc:
            # httpx 的传输错误族（连接失败、超时、TLS）。**收窄到 HTTPError**：
            # ValueError 一类的编程错误不该被说成"组件没在跑"。
            return SpeechResult(
                reason=(
                    f"语音组件没连上（{self.base_url}）。它是不是没启动？"
                    f" 原始错误：{type(exc).__name__}: {exc}"
                )
            )
        if response.status_code != 200:
            detail = response.text.strip()[:200]
            return SpeechResult(
                reason=f"语音组件回了 HTTP {response.status_code}"
                + (f"：{detail}" if detail else "（没有正文）")
            )
        if not response.content:
            return SpeechResult(reason="语音组件回了个空结果（没有声音）")
        return SpeechResult(wav=bytes(response.content))
