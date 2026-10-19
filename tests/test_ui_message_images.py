"""用户附图的编码：**裸 base64、统一 PNG**。

2026-10-19 用户拖一张 JPEG 进来，DeepSeek 回 400 ``Invalid base64 data``：
`_image_data_uri` 返回了完整的 ``data:image/jpeg;base64,…``，而 OpenAI 兼容后端
又拼了一次前缀。这里钉住两件事：不带前缀、且 JPEG 会被转成 PNG（后端那条路把
MIME 写死成 image/png）。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _write_jpeg(path: Path) -> None:
    from PySide6.QtGui import QColor, QImage

    image = QImage(32, 32, QImage.Format.Format_RGB32)
    image.fill(QColor("#cc3355"))
    assert image.save(str(path), "JPEG")


def test_image_base64_has_no_data_prefix_and_is_png(qapp: Any, tmp_path: Path) -> None:
    from axo.attachments import Attachment
    from axo.ui.main_window import _image_base64

    path = tmp_path / "photo.jpg"
    _write_jpeg(path)
    encoded = _image_base64(Attachment(path=path, kind="image", size=path.stat().st_size))

    assert encoded
    assert not encoded.startswith("data:"), "ChatMessage.images 收的是裸 base64"
    raw = base64.b64decode(encoded)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n"), "JPEG 必须先转成 PNG，后端 MIME 写死为 png"


def test_unreadable_image_returns_empty_instead_of_bad_base64(qapp: Any, tmp_path: Path) -> None:
    from axo.attachments import Attachment
    from axo.ui.main_window import _image_base64

    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    assert _image_base64(Attachment(path=path, kind="image", size=13)) == ""
