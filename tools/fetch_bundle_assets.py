"""把分发版要带的离线模型**与本地推理运行时**抓到 ``build/bundle/``。

这个脚本只做构建准备：**产物不进仓**（``build/`` 在 .gitignore 里），模型来源与
SHA256 写死在这里。两条来源都已固定：

* 嵌入：``Qdrant/bge-small-zh-v1.5`` 的 HF snapshot（MIT），只取五个文件；
* TTS：``sherpa-onnx-zipvoice-distill-int8-zh-en-emilia``（CC-BY-NC-4.0）里的
  encoder/decoder/token/lexicon，加上 vocos 24 kHz vocoder；**不取 espeak-ng-data**
  ——那是 GPL 数据，而中文/英文这两条路只用 lexicon（已实测）。
* 本地模型运行时：llama.cpp 官方的 **Windows CUDA 13.4** 发行包（MIT）与它的
  cudart 包。软件用它自己拉起 ``llama-server``，**不再依赖 Ollama**（见
  ``src/axo/llm/llama_server.py``）。模型本身不在这里抓——它是用户自己的 GGUF
  文件，设置里选。

用法::

    uv run python tools/fetch_bundle_assets.py
    uv run python tools/fetch_bundle_assets.py --dest build/bundle
    uv run python tools/fetch_bundle_assets.py --embedding-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "build" / "bundle"

EMBEDDING_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
EMBEDDING_BASE = f"https://huggingface.co/Qdrant/bge-small-zh-v1.5/resolve/{EMBEDDING_REVISION}"
EMBEDDING_FILES = {
    "model_optimized.onnx": "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    "config.json": "9088751d39abbf86ec3d19ffca92ad62ad19075f7e59712e6c71217fa125d1d3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a",
}

ZIPVOICE_ARCHIVE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2"
)
ZIPVOICE_FILES = {
    "encoder.int8.onnx": "f2de9a761a85e5ddd125dee6e05bad1c7ee92c11b83b4d775dab216a6aa41379",
    "decoder.int8.onnx": "3cc2e08a96610d7ea1b227398e97cdbbe0414499741d3aec0b8113db2a2ab251",
    "tokens.txt": "ce98c1afc5f7a20c2484dffdd68a1fff0a4a2cc707328833750c4476c37cdbda",
    "lexicon.txt": "c94b049d970e971acdc2a7e5e7b23ca594222a9d640e417c711f2ad1e7bb7d46",
}
VOCAB_FILES = {
    "lexicon-en-us.txt": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt",
        "8b67ae549a465eee78f92ecb74644ad90c3c58ce65f19c68baf85878a10ef73a",
    ),
    "vocos_24khz.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx",
        "bcb3b970e384161c4d634f0bb9e999ff1c471b34c9bc0b1049a5014065ed3cc0",
    ),
}
VOICE_SOURCE = ROOT / "src" / "axo" / "ui" / "assets" / "tts" / "qiansui"
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
VOSK_NAME = "vosk-model-small-cn-0.22"

#: llama.cpp 的发行版本号。**写死版本而不是 latest**：``llama-server`` 的命令行与
#: ``/v1`` 行为都随版本变，而"今天能起、下个月不能起"正是这类运行时最坏的一种变化。
LLAMA_RELEASE = "b12020"
LLAMA_ARCHIVES = (
    (
        "llama-b12020-bin-win-cuda-13.4-x64.zip",
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}"
        f"/llama-{LLAMA_RELEASE}-bin-win-cuda-13.4-x64.zip",
        "0ddf55522212365415efe8409353c2c72e3e1dd43a5fd014e314ae1bfc9e961f",
    ),
    (
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}"
        f"/cudart-llama-bin-win-cuda-13.4-x64.zip",
        "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    ),
)
LLAMA_SERVER = "llama-server.exe"
"""随包运行时里那个可执行文件。**构建与冒烟都按这个名字找**（见 ``axo.paths``）。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path, expected: str | None) -> None:
    if expected is not None and target.is_file() and _sha256(target) == expected:
        print(f"已有：{target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"下载：{url}")
    request = urllib.request.Request(url, headers={"User-Agent": "axo-build/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    digest = _sha256(temporary)
    if expected is not None and digest != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 对不上：{target.name} 得到 {digest}，期望 {expected}")
    temporary.replace(target)


def _embedding(dest: Path) -> None:
    folder = dest / "embedding" / "bge-small-zh-v1.5"
    for name, digest in EMBEDDING_FILES.items():
        _download(f"{EMBEDDING_BASE}/{name}", folder / name, digest)


def _vosk(dest: Path) -> None:
    """Windows 也带识别模型；macOS 构建那条路仍然由 ``--bundle-vosk-model`` 传入。"""
    folder = dest / "vosk" / VOSK_NAME
    if (folder / "am" / "final.mdl").is_file():
        print("已有：vosk 模型")
        return
    with tempfile.TemporaryDirectory(prefix="axo-vosk-") as temporary:
        archive = Path(temporary) / "vosk.zip"
        _download(VOSK_URL, archive, None)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if not name.startswith(VOSK_NAME + "/"):
                    continue
                target = dest / "vosk" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def _llama(dest: Path) -> None:
    """llama.cpp 运行时 → ``build/bundle/llama/``。

    **两个包都是平铺的**（实测：列表里没有统一的顶层目录），所以直接按条目名落地。
    它们合计约 732 MiB，其中 ``cublasLt64_13.dll`` 一个就 470 MiB——这就是
    "8 GB 显存跑得动、但包里装得下"的那笔账。

    CUDA 版而不是 CPU 版：本机是 RTX 5060（sm_120），实测首字 0.13 s、约 114 字/秒；
    CPU 版在同一台机上慢一个数量级。显存不够时 llama.cpp 自己退回 CPU 层并在日志里
    写清楚，所以带 CUDA 版不会让没有显卡的机器起不来——它只是没得加速。
    """
    folder = dest / "llama"
    if (folder / LLAMA_SERVER).is_file() and (folder / "cublas64_13.dll").is_file():
        print("已有：llama.cpp 运行时")
        return
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="axo-llama-") as temporary:
        for name, url, digest in LLAMA_ARCHIVES:
            archive = Path(temporary) / name
            _download(url, archive, digest)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if member.is_dir():
                        continue
                    target = folder / Path(member.filename).name
                    with bundle.open(member) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)


def _zipvoice(dest: Path) -> None:
    folder = dest / "tts" / "zipvoice"
    with tempfile.TemporaryDirectory(prefix="axo-zipvoice-") as temporary:
        archive = Path(temporary) / "model.tar.bz2"
        _download(ZIPVOICE_ARCHIVE, archive, None)
        with tarfile.open(archive, "r:bz2") as tar:
            for name, digest in ZIPVOICE_FILES.items():
                member = next(
                    (
                        item
                        for item in tar.getmembers()
                        if item.isfile() and item.name.endswith("/" + name)
                    ),
                    None,
                )
                if member is None:
                    raise SystemExit(f"ZipVoice archive 里没有 {name}")
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                assert source is not None
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                got = _sha256(target)
                if got != digest:
                    raise SystemExit(f"SHA256 对不上：{name} 得到 {got}，期望 {digest}")
    for name, (url, digest) in VOCAB_FILES.items():
        _download(url, folder / name, digest)
    voice = dest / "tts" / "voices" / "qiansui"
    voice.mkdir(parents=True, exist_ok=True)
    for item in VOICE_SOURCE.iterdir():
        if item.is_file():
            shutil.copyfile(item, voice / item.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--tts-only", action="store_true")
    parser.add_argument("--llama-only", action="store_true", help="只抓 llama.cpp 运行时")
    args = parser.parse_args(argv)
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if not args.tts_only and not args.llama_only:
        _embedding(dest)
    if not args.embedding_only and not args.llama_only:
        _zipvoice(dest)
        _vosk(dest)
    if not args.embedding_only and not args.tts_only:
        _llama(dest)
    manifest = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(dest).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
    (dest / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    total = sum(item["size"] for item in manifest.values())
    print(f"完成：{dest}（{len(manifest)} 个文件，{total / 1024**2:.1f} MiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
