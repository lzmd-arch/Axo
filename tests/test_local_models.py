"""随包嵌入与 TTS 的资产边界。

**真模型不进仓**，所以这些用例分两层：文件形状/拒绝路径永远跑；真正推理只在
``build/bundle`` 已经抓过资产时跑（macOS 构建流水线会跑这一层，核心腿不会下载
95 MB + 170 MB 的模型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from axo.llm.local_embed import LocalEmbeddingBackend
from axo.llm.local_speech import LocalSpeech
from axo.llm.local_speech import is_available as speech_available
from axo.paths import EMBEDDING_MODEL_NAME, bundled_embedding_dir
from memcore.store import MemoryStore

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "build" / "bundle"


def test_bundled_embedding_dir_finds_the_frozen_layout(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "embedding" / EMBEDDING_MODEL_NAME
    model.mkdir(parents=True)
    (model / "model_optimized.onnx").write_bytes(b"x")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("AXO_EMBEDDING_DIR", raising=False)

    assert bundled_embedding_dir() == model


def test_local_embedding_refuses_an_incomplete_model_dir(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="不完整"):
        LocalEmbeddingBackend(tmp_path)


def test_local_speech_refuses_a_missing_asset_dir(tmp_path: Path) -> None:
    assert speech_available(tmp_path) is False
    with pytest.raises(ValueError, match="资产不完整"):
        LocalSpeech(tmp_path, tmp_path / "voices")


@pytest.mark.skipif(not (_BUNDLE / "embedding").is_dir(), reason="没有抓随包嵌入模型")
def test_memory_sync_prefers_the_bundled_embedding(tmp_path: Path) -> None:
    from axo.ui.memory_worker import MemorySyncWorker

    store = MemoryStore(tmp_path / "mem.db")
    worker = MemorySyncWorker(
        store=store,
        model="nomic-embed-text",
        local_dir=_BUNDLE / "embedding" / EMBEDDING_MODEL_NAME,
    )
    received: list[object] = []
    worker.ready.connect(received.append)
    worker.run()

    assert received
    setup = received[0]
    assert setup.backend is not None
    assert setup.dim == 512
    assert "随包嵌入模型" in setup.note
    assert "ollama/" not in setup.backend.space_id


@pytest.mark.skipif(not (_BUNDLE / "embedding").is_dir(), reason="没有抓随包嵌入模型")
def test_the_bundled_embedding_produces_normalized_512_vectors() -> None:
    backend = LocalEmbeddingBackend(_BUNDLE / "embedding" / EMBEDDING_MODEL_NAME)
    matrix = backend.embed(["羽毛球馆很热", "记忆库与语义召回"])

    assert matrix.shape == (2, 512)
    assert matrix.dtype.name == "float32"
    np.testing.assert_allclose((matrix * matrix).sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.skipif(not speech_available(_BUNDLE / "tts"), reason="没有抓随包 TTS 模型")
def test_the_bundled_tts_synthesizes_a_wav() -> None:
    speech = LocalSpeech(_BUNDLE / "tts", _BUNDLE / "tts" / "voices")
    result = speech.synthesize("你好，我是千岁。")

    assert result.ok, result.reason
    assert result.seconds > 0.2
    assert result.wav[:4] == b"RIFF"
