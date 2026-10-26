"""冻结产物自己的自检入口（``Axo --bundle-smoke``）。

**为什么需要它。** Windows 的构建脚本能直接查盘、起窗口、读注册表；macOS 的 CI
runner 没有显示器，``.app`` 又是无控制台的，`print` 出去的东西看不到。一个"构建成功
但双击没反应"的包在这两件事叠加时是最难发现的：PyInstaller 的日志全绿，DMG 也在。

所以入口里留一条**不建 QApplication、不碰屏幕**的分支：它 import 真实的 Qt 模块
（含 QtWebEngine 与 QtMultimedia）、加载 vosk 的 dylib、读包里那几份资产，把结果写到
构建脚本给的 JSON 文件。它验的是"冻结后的 import 图与资产在不在"，不是界面像素——后者
只能在真 Mac 上手动看。

``--bundle-smoke`` 只在构建流水线里用，不是用户功能。没有参数时照常启动界面。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "run_checks"]

_RESOURCES = (
    "axo/ui/theme.qss",
    "axo/ui/web/index.html",
    "axo/ui/web/css/app.css",
    "axo/ui/web/js/app.js",
    "axo/ui/assets/stickers/manifest.json",
)

_MODULES = (
    "PySide6.QtCore",
    "PySide6.QtWidgets",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtMultimedia",
    "vosk",
    "axo.agent.lsp_install",
    "axo.agent.language_doctor",
    "axo.ui.main_window",
    "axo.llm.local_embed",
    "axo.llm.local_speech",
    "onnxruntime",
    "tokenizers",
    "sherpa_onnx",
    "soundfile",
)


def _resource_root() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parent.parent


def run_checks() -> dict[str, object]:
    from axo import __version__
    from axo.paths import bundled_embedding_dir, bundled_tts_dir, bundled_voice_model_dir

    root = _resource_root()
    errors: list[str] = []
    resources: dict[str, bool] = {}
    for relative in _RESOURCES:
        present = (root / relative).is_file()
        resources[relative] = present
        if not present:
            errors.append(f"资产不在：{relative}")

    modules: dict[str, bool] = {}
    for name in _MODULES:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 — 自检要把每一种失败收进报告
            modules[name] = False
            errors.append(f"import 失败：{name}: {type(exc).__name__}: {exc}")
        else:
            modules[name] = True

    model = bundled_voice_model_dir()
    model_ok = model is not None and any(model.iterdir())
    if not model_ok:
        errors.append("随包语音模型不在（构建时应传 --bundle-vosk-model）")

    embedding = bundled_embedding_dir()
    embedding_ok = bool(embedding and (embedding / "model_optimized.onnx").is_file())
    if not embedding_ok:
        errors.append("随包嵌入模型不在（构建前跑 tools/fetch_bundle_assets.py）")
    embedding_inference = False
    if embedding_ok:
        try:
            from axo.llm.local_embed import LocalEmbeddingBackend

            vector = LocalEmbeddingBackend(embedding).embed(["测试"])[0]
            embedding_inference = vector.shape == (512,)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"随包嵌入推理失败：{type(exc).__name__}: {exc}")
        if not embedding_inference:
            errors.append("随包嵌入模型没有产出 512 维向量")

    tts = bundled_tts_dir()
    tts_ok = bool(tts and (tts / "zipvoice" / "decoder.int8.onnx").is_file())
    if not tts_ok:
        errors.append("随包 TTS 模型不在（构建前跑 tools/fetch_bundle_assets.py）")
    tts_inference = False
    if tts_ok:
        try:
            from axo.llm.local_speech import LocalSpeech

            spoken = LocalSpeech(tts, tts / "voices").synthesize("你好")
            tts_inference = bool(spoken.ok and spoken.wav[:4] == b"RIFF")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"随包 TTS 推理失败：{type(exc).__name__}: {exc}")
        if not tts_inference:
            errors.append("随包 TTS 没有产出 WAV")

    return {
        "ok": not errors,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
        "version": __version__,
        "executable": sys.executable,
        "resource_root": str(root),
        "resources": resources,
        "modules": modules,
        "voice_model": str(model) if model is not None else "",
        "voice_model_ok": model_ok,
        "embedding_model": str(embedding) if embedding is not None else "",
        "embedding_model_ok": embedding_ok,
        "embedding_inference": embedding_inference,
        "tts_model": str(tts) if tts is not None else "",
        "tts_model_ok": tts_ok,
        "tts_inference": tts_inference,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="Axo --bundle-smoke")
    parser.add_argument("--out", required=True, help="把 JSON 报告写到这个文件")
    args = parser.parse_args(argv)
    report = run_checks()
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if sys.stdout is not None:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1
