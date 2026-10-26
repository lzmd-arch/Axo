"""F5-TTS 可选组件的契约守卫：不 import torch，也能验 HTTP 与音色解析。

F5 本身的推理要 CUDA/权重，不能进核心腿；这里只钉三件事：``/v1/audio/speech``
的请求/响应形状、错误的人话、以及 ``models/voices/<名字>/`` 的解析规则。
"""

from __future__ import annotations

import importlib.util
import io
import json
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SERVER = _ROOT / "components" / "f5-tts" / "server.py"


def _load():
    spec = importlib.util.spec_from_file_location("_f5_component_server", _SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def voices_available(self) -> list[str]:
        return ["qiansui"]

    def synthesize(self, text: str, voice: str) -> bytes:
        self.calls.append((text, voice))
        return _wav()


def test_voice_directories_need_audio_and_transcript(tmp_path: Path) -> None:
    server = _load()
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "ref.wav").write_bytes(b"x")
    (tmp_path / "good" / "ref.txt").write_text("你好", encoding="utf-8")
    (tmp_path / "no-text").mkdir()
    (tmp_path / "no-text" / "ref.wav").write_bytes(b"x")
    (tmp_path / "no-audio").mkdir()
    (tmp_path / "no-audio" / "ref.txt").write_text("你好", encoding="utf-8")

    engine = server.Engine(tmp_path)
    assert engine.voices_available() == ["good"]
    assert engine._resolve("")[0].name == "ref.wav"
    with pytest.raises(RuntimeError, match="没有音色"):
        engine._resolve("nope")


def test_http_contract_and_errors() -> None:
    server_module = _load()
    engine = FakeEngine()
    httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), server_module.make_handler(engine))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1/audio/speech"
    try:
        with urllib.request.urlopen(base, timeout=5) as response:
            assert json.loads(response.read()) == {"voices": ["qiansui"]}

        request = urllib.request.Request(
            base,
            data=json.dumps({"input": "你好", "voice": "qiansui"}).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.headers["content-type"] == "audio/wav"
            assert response.read()[:4] == b"RIFF"
        assert engine.calls == [("你好", "qiansui")]

        bad = urllib.request.Request(
            base, data=b'{"input": ""}', headers={"content-type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(bad, timeout=5)
        assert caught.value.code == 400
        assert "input" in json.loads(caught.value.read())["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_engine_does_not_import_torch_at_module_import() -> None:
    server = _load()
    engine = server.Engine(Path(), "auto")
    assert isinstance(engine, server.Engine)  # 构造不该触发 F5/torch 的 import
