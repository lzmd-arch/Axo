"""Axo 的 F5-TTS 语音组件：``POST /v1/audio/speech`` → wav。

**它是可选的第二条 TTS 路线。** 分发包默认用随包的 ZipVoice（CPU、零安装）；
这个组件给愿意用 GPU / F5 音质的用户。Axo 只在设置里关掉「优先使用随包千岁语音」
时才会来连它，契约仍是 ``http://127.0.0.1:8800``。

## 契约

    POST /v1/audio/speech
    {"input": "<要说的中文>", "voice": "<音色名，空=默认>", "response_format": "wav"}
    → 200 audio/wav
      4xx/5xx + JSON {"error": "一句人话"}

引擎是 F5-TTS（零样本克隆：参考音定音色，不需要训练）。换引擎时改这个文件，
Axo 一行不用改。

## 音色

``<voices-root>/<名字>/ref.wav``（10–30 秒参考音）+ ``ref.txt``（它的逐字稿）。
与 Axo 数据根里的 ``models/voices/`` 同一形状；``--voices`` 指过去即可。

用法::

    .venv/Scripts/python server.py --voices <数据根>/models/voices --port 8800 --device cuda
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VOICE_REF = "ref.txt"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


@dataclass(frozen=True)
class Reply:
    status: int
    body: bytes
    content_type: str


def _json_reply(status: int, message: str) -> Reply:
    return Reply(
        status,
        json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"),
        "application/json",
    )


class Engine:
    """加载一次 F5-TTS，反复合成。**延迟 import torch**，服务先起来。"""

    def __init__(self, voices_root: Path, device: str = "auto") -> None:
        self.voices = Path(voices_root)
        self.device = None if device == "auto" else device
        self._model: object | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ── 音色 ────────────────────────────────────────────────────────────────

    def voices_available(self) -> list[str]:
        if not self.voices.is_dir():
            return []
        found: list[str] = []
        for item in sorted(self.voices.iterdir()):
            if not item.is_dir() or not (item / VOICE_REF).is_file():
                continue
            if any(
                child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
                for child in item.iterdir()
            ):
                found.append(item.name)
        return found

    def _resolve(self, name: str) -> tuple[Path, str]:
        available = self.voices_available()
        if not available:
            raise RuntimeError(f"一个音色都没有：{self.voices} 下要有 <名字>/ref.wav 与 ref.txt")
        requested = name.strip()
        pick = requested or available[0]
        if pick not in available:
            raise RuntimeError(f"没有音色 {pick!r}；可用：{', '.join(available)}")
        folder = self.voices / pick
        ref_text = (folder / VOICE_REF).read_text(encoding="utf-8-sig").strip()
        if not ref_text:
            raise RuntimeError(f"{folder / VOICE_REF} 是空的——零样本克隆要参考音的逐字稿")
        audio = next(
            child
            for child in sorted(folder.iterdir())
            if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
        )
        return audio, ref_text

    # ── 引擎 ────────────────────────────────────────────────────────────────

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            self._patch_audio_loading()
            from f5_tts.api import F5TTS

            kwargs = {} if self.device is None else {"device": self.device}
            self._model = F5TTS(**kwargs)
            return self._model

    @staticmethod
    def _patch_audio_loading() -> None:
        """把 ``torchaudio.load`` 换成 soundfile 的实现。

        torchaudio 2.11 的 ``load`` 走 torchcodec，在部分 Windows 机器上加载
        FFmpeg 共享库失败；F5-TTS 只用它读参考音。只换这一个绑定，不碰 F5 的
        其他调用点。
        """
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        def _load_via_soundfile(uri, *_args, **_kwargs):
            data, rate = sf.read(uri, dtype="float32", always_2d=True)
            return torch.from_numpy(np.ascontiguousarray(data.T)), rate

        torchaudio.load = _load_via_soundfile

    def synthesize(self, text: str, voice: str) -> bytes:
        model = self._load()
        ref_audio, ref_text = self._resolve(voice)
        with self._infer_lock:
            wav, sample_rate, _stats = model.infer(  # type: ignore[attr-defined]
                ref_file=str(ref_audio), ref_text=ref_text, gen_text=text
            )
        import numpy as np

        pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())
        return buffer.getvalue()


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, reply: Reply) -> None:
            self.send_response(reply.status)
            self.send_header("content-type", reply.content_type)
            self.send_header("content-length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

        def do_GET(self) -> None:  # noqa: N802 (HTTP 方法名)
            if self.path.rstrip("/").endswith("/v1/audio/speech"):
                payload = json.dumps(
                    {"voices": engine.voices_available()}, ensure_ascii=False
                ).encode("utf-8")
                self._send(Reply(200, payload, "application/json"))
                return
            self._send(_json_reply(404, "只有 /v1/audio/speech"))

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/v1/audio/speech"):
                self._send(_json_reply(404, "只有 /v1/audio/speech"))
                return
            try:
                size = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(size) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(_json_reply(400, "请求体不是合法 JSON"))
                return
            text = str(payload.get("input") or "").strip()
            if not text:
                self._send(_json_reply(400, "input 是空的——没有要念的文字"))
                return
            try:
                wav = engine.synthesize(text, str(payload.get("voice") or ""))
            except Exception as exc:  # noqa: BLE001 — 服务端错误统一成一句人话
                self._send(_json_reply(500, f"{type(exc).__name__}: {exc}"))
                return
            self._send(Reply(200, wav, "audio/wav"))

        def log_message(self, fmt: str, *args: object) -> None:
            print("[speech]", fmt % args)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Axo 的 F5-TTS 语音组件")
    parser.add_argument("--voices", required=True, help="音色根目录（Axo 的 models/voices）")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--host", default="127.0.0.1", help="默认只听本机")
    args = parser.parse_args(argv)

    engine = Engine(Path(args.voices), args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"F5-TTS 语音组件在 http://{args.host}:{args.port}/v1/audio/speech")
    print(f"音色根：{args.voices}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
