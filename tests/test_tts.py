"""``axo.tts``（纯）与 ``axo.llm.speech``（网络）：请求形状、音色扫描、三条失败。

**形状的测试全部不碰网络**：客户端收一个可注入的 ``transport``（``httpx.MockTransport``），
断言打出去的 URL 与 JSON 就是契约本身——契约漂了，Axo 与组件两边各改各的，
症状是"按钮按了没声"，而两边的测试都还绿着。

三条失败各一句人话（组件没在跑 / 回了非 2xx / 回了空体），以及"没有要念的文字"——
四条都不抛（同 ``asr`` 的纪律：这些路径上抛出去没有人接得住）。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from axo.llm.ollama import _client as _ollama_client
from axo.llm.speech import RemoteSpeech
from axo.tts import (
    SpeechReceipt,
    SpeechResult,
    Voice,
    speech_payload,
    voices_in,
)

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 26  # 够长就行：客户端不解析它


def _client(handler) -> RemoteSpeech:
    """一个走 ``MockTransport`` 的客户端（测试缝，见 ``RemoteSpeech.__init__``）。"""
    return RemoteSpeech("http://speech.test:8800", transport=httpx.MockTransport(handler))


# ── 纯：请求形状与音色扫描 ─────────────────────────────────────────────────


def test_the_payload_is_the_openai_shape() -> None:
    """请求体贴 ``/v1/audio/speech`` 的 OpenAI 形状，``response_format`` 固定 wav。

    wav 是**定死的**：播放直推 PCM、预算按 wav 头算秒数，换 mp3 的省流量在这个
    本机组件形态下换不来什么，还会让两个消费者各认各的容器。
    """
    body = speech_payload("你好呀", voice="she")

    assert body == {"input": "你好呀", "voice": "she", "response_format": "wav"}


def test_an_empty_model_is_not_sent() -> None:
    """``model`` 非空才写：OpenAI 的 API 要它，自建组件多数不看——空着不发，
    免得那一侧把一个空模型名当真。"""
    assert "model" not in speech_payload("你好")
    assert speech_payload("你好", model="axo-voice")["model"] == "axo-voice"


def test_voices_are_directories_that_contain_a_sample(tmp_path: Path) -> None:
    """一个目录 = 一个音色，判据是"里面至少有一个参考音"。

    空目录与放错东西的目录**都不算**：下拉框少一项比多一项好——多的那一项用户点了
    之后是引擎报错，而那个错看起来像我们坏了。
    """
    (tmp_path / "her").mkdir()
    (tmp_path / "her" / "ref.wav").write_bytes(_WAV)
    (tmp_path / "him").mkdir()
    (tmp_path / "him" / "ref.mp3").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "ref.txt").write_text("不是音频", encoding="utf-8")
    (tmp_path / "loose.wav").write_bytes(_WAV)  # 目录外的散文件不算音色

    found = voices_in(tmp_path)

    assert [v.name for v in found] == ["her", "him"]
    assert all(isinstance(v, Voice) for v in found)
    assert found[0].sample == "ref.wav"


def test_a_missing_voices_directory_is_just_empty(tmp_path: Path) -> None:
    """目录不存在 = 空元组（**不创建**，同 ``voice_model_dir`` 的纪律）。"""
    assert voices_in(tmp_path / "nope") == ()


# ── 网络：打出去的形状与三条失败 ───────────────────────────────────────────


def test_the_client_posts_to_v1_audio_speech() -> None:
    """打到 ``{base_url}/v1/audio/speech``——**版本前缀由调用端补**，
    用户在设置里填的那格是组件根地址（越少东西越好）。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, content=_WAV)

    result = _client(handler).synthesize("你好", voice="she")

    assert result.ok is True
    assert result.wav == _WAV
    assert seen["path"] == "/v1/audio/speech"
    # **解析回来断言**，不按字节比：httpx 的 JSON 是紧凑格式（冒号后没空格），
    # 按字节比会在"格式化方式变了"上红，而不是契约变了。
    body = json.loads(seen["body"])  # type: ignore[arg-type]
    assert body["response_format"] == "wav"
    assert body["voice"] == "she"
    assert body["input"] == "你好"  # 你好


def test_a_component_that_is_not_running_says_so() -> None:
    """连不上（组件没起）→ 一句**指得出动作**的话，不是 traceback。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = _client(handler).synthesize("你好")

    assert result.ok is False
    assert "语音组件没连上" in result.reason
    assert "http://speech.test:8800" in result.reason, "没说清是哪个地址——用户要去起它"


def test_a_non_200_carries_the_status_and_the_body() -> None:
    result = _client(lambda _request: httpx.Response(500, text="引擎炸了")).synthesize("你好")

    assert result.ok is False
    assert "HTTP 500" in result.reason
    assert "引擎炸了" in result.reason


def test_an_empty_body_is_not_silently_ok() -> None:
    """回了 200 但空体 → **当失败**（"合成出来是空的"是第三条人话）。

    收下空 wav 的话，调用方会去播一个 0 秒的音频然后什么都不发生——
    那正是"按钮按了没声"这种查不动的症状。
    """
    result = _client(lambda _request: httpx.Response(200, content=b"")).synthesize("你好")

    assert result.ok is False
    assert "空结果" in result.reason


def test_empty_text_never_leaves_the_house() -> None:
    """空文本连请求都不发——念一句空的，用户听到的是无声，而她"说了"。"""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_WAV)

    result = _client(handler).synthesize("   ")

    assert seen == []
    assert result.ok is False
    assert "空的" in result.reason


def test_the_receipt_shape_matches_the_other_two() -> None:
    """``SpeechResult`` 与 ``Transcription`` / ``Capture`` 同形：成功与失败都是它。"""
    assert SpeechResult(wav=b"x").ok is True
    assert SpeechResult(reason="没成").ok is False
    assert SpeechResult().ok is False


def test_the_receipt_is_a_one_line_signal_with_a_timestamp() -> None:
    """回执渲染成**一行**：带时刻、失败带原因、正文截短——模型提的三条（"回执别做太重"）。"""
    ok_line = SpeechReceipt(at=0.0, text="记得买牛奶。", ok=True).describe(at="23:41")
    bad_line = SpeechReceipt(
        at=0.0, text="记得买牛奶。", ok=False, reason="语音组件没连上"
    ).describe(at="23:41")

    assert ok_line == "播出回执（23:41）：已播出「记得买牛奶。」"
    assert "**没播成**" in bad_line
    assert "语音组件没连上" in bad_line, "失败必须带原因——顺利的回执用处有限，卡住的那次才要紧"
    assert "23:41" in bad_line

    long_text = SpeechReceipt(at=0.0, text="一" * 100, ok=True).describe(at="00:00")
    assert len(long_text) < 80, "回执要最小——整句塞进上下文是浪费（模型自己有全文）"


# ── 本机地址不走代理 ────────────────────────────────────────────────────────


def test_a_local_endpoint_never_goes_through_a_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """**本机的请求不许经过代理。**

    httpx 默认认 HTTP_PROXY/HTTPS_PROXY，而一个 127.0.0.1 的请求被送去代理的结局是
    代理回一个 502（**空的**）——用户看到的是「组件回了 HTTP 502（没有正文）」，
    而真相是组件根本没在跑。2026-10-17 实测踩到。
    """
    seen: dict[str, object] = {}
    real = httpx.Client

    class Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", Spy)
    speech = RemoteSpeech(
        "http://127.0.0.1:8800",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"RIFF")),
    )

    speech.synthesize("你好")

    assert seen.get("trust_env") is False


def test_a_remote_endpoint_leaves_the_proxy_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """远程地址经不经代理由用户自己的网络配置说了算——这里不替它决定。"""
    seen: dict[str, object] = {}
    real = httpx.Client

    class Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", Spy)
    speech = RemoteSpeech(
        "http://tts.example.com:8800",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"RIFF")),
    )

    speech.synthesize("你好")

    assert seen.get("trust_env") is True


@pytest.mark.parametrize(
    ("host", "allow_remote", "expected"),
    [
        ("http://127.0.0.1:11434", False, False),
        ("http://localhost:11434", False, False),
        ("http://10.0.0.5:11434", True, True),
    ],
)
def test_the_ollama_client_only_trusts_the_environment_when_it_may_go_remote(
    host: str, allow_remote: bool, expected: bool
) -> None:
    """同一条：本机不走代理，而"允许远程"意味着用户自己掌控那条链路。"""
    client = _ollama_client(host, allow_remote)
    try:
        assert client.trust_env is expected
    finally:
        client.close()
