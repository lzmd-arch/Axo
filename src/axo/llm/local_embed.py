"""随包的中文嵌入模型（BGE-small-zh-v1.5，ONNX）。**不联网、不依赖 Ollama。**

## 为什么是它

语义召回原来只认 Ollama 的 ``nomic-embed-text``。那在开发机上没问题，但分发版里
Ollama 可能没装、模型可能没 pull，而这条链的失败方式是**界面一直停在"正在检索记忆…"**
（嵌入请求挂在 300 秒的读超时上），用户看到的是整个程序卡死——不是"没有语义召回"。
所以分发版把一个小嵌入模型直接带进来：

* ``BAAI/bge-small-zh-v1.5``：中文检索专用、512 维、MIT 许可；
* 量化后的 ONNX 约 95 MB，CPU 上 4 条短文本约 7 毫秒；
* 推理只用 ``onnxruntime`` + ``tokenizers``，没有 torch。

## 空间标识

``space_id`` 固定为 ``local/bge-small-zh-v1.5@512``。它和旧的
``ollama/nomic-embed-text:latest@768`` 是两个空间——旧向量一条都不会被误用，
启动时 :func:`~axo.memory.backfill.sync_embeddings` 会把缺向量的条目全部补到新空间。
**不要**为了省一次回填把新模型的向量写进旧空间：两个模型的余弦算得出数，但那个
数没有意义（见 ``memcore.errors.SpaceMismatchError`` 的说明）。

## 为什么不用 fastembed / sentence-transformers

fastembed 也能用，但它把 HuggingFace 的缓存布局、模型下载与一堆 Python 依赖带进来；
我们只需要"读两个文件、跑一次 ONNX"。直接读 ``model_optimized.onnx`` 与
``tokenizer.json`` 的实现更小，也能由构建脚本逐文件校验 SHA256。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import EmbeddingBackendError

__all__ = ["MODEL_FILES", "LocalEmbeddingBackend", "is_available"]

MODEL_FILES = (
    "model_optimized.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
"""随包模型的五个文件。缺任何一个都不装懂——见 :func:`is_available`。"""

MAX_TOKENS = 512
"""BGE 的位置编码上限。超过的文本截断；这不是"悄悄丢内容"——召回本来只用前 512
个 token，而记忆条目自身另有长度上限。"""


def is_available(model_dir: Path | str | None) -> bool:
    """这个目录里有没有一套完整的模型文件。坏目录当没有，不抛。"""
    if model_dir is None:
        return False
    root = Path(model_dir)
    return root.is_dir() and all((root / name).is_file() for name in MODEL_FILES)


@dataclass
class LocalEmbeddingBackend:
    """一个随包的 BGE 中文嵌入后端。

    **延迟加载。** 构造这个对象只查文件；真正载入 ONNX 与 tokenizer 发生在第一次
    :meth:`embed`。这样 UI 可以先把它挂上，启动时不会为了 95 MB 的模型卡一下。
    """

    model_dir: Path
    threads: int = 2
    dim: int = 512
    space_id: str = "local/bge-small-zh-v1.5@512"
    _session: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if not is_available(self.model_dir):
            raise EmbeddingBackendError(
                f"随包嵌入模型不完整：{self.model_dir}。需要 {', '.join(MODEL_FILES)}。"
            )
        if self.dim < 1:
            raise ValueError("LocalEmbeddingBackend.dim 必须 >= 1")

    def _ensure(self) -> tuple[object, object]:
        if self._session is not None and self._tokenizer is not None:
            return self._session, self._tokenizer
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return self._session, self._tokenizer
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - 分发版一定带
                raise EmbeddingBackendError(
                    "随包嵌入模型需要 onnxruntime 与 tokenizers；当前环境缺它们。"
                ) from exc
            try:
                tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=MAX_TOKENS)
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                session = ort.InferenceSession(
                    str(self.model_dir / "model_optimized.onnx"),
                    providers=["CPUExecutionProvider"],
                    sess_options=_session_options(ort, self.threads),
                )
            except Exception as exc:  # noqa: BLE001 - 模型坏/平台不支持都归一句人话
                raise EmbeddingBackendError(
                    f"载入随包嵌入模型失败（{type(exc).__name__}）：{exc}"
                ) from exc
            self._tokenizer = tokenizer
            self._session = session
            return session, tokenizer

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码，返回 ``(len(texts), 512)`` 的 float32、已 L2 归一化矩阵。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        session, tokenizer = self._ensure()
        encoded = tokenizer.encode_batch([str(text) for text in texts])  # type: ignore[attr-defined]
        ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        types = np.zeros_like(ids, dtype=np.int64)
        try:
            output = session.run(  # type: ignore[attr-defined]
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingBackendError(f"随包嵌入模型推理失败：{exc}") from exc
        cls = np.asarray(output[:, 0, :], dtype=np.float32)
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        return np.divide(cls, norms, out=np.zeros_like(cls), where=norms > 0)


def _session_options(ort: object, threads: int) -> object:
    """把线程数压低。嵌入是短文本、低延迟任务，默认抢满所有核只会让界面抢不到 CPU。"""
    options = ort.SessionOptions()  # type: ignore[attr-defined]
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
    return options
