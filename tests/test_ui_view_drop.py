"""拖进对话区的文件：Qt 这一层只收本地文件。"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QUrl

from axo.ui.chat_view import _local_file_uris

pytestmark = pytest.mark.gui


def test_only_local_files_are_taken() -> None:
    mime = QMimeData()
    mime.setUrls(
        [
            QUrl.fromLocalFile(r"C:\work\笔记.txt"),
            QUrl("https://example.com/pic.png"),
        ]
    )

    uris = _local_file_uris(mime)

    assert len(uris) == 1
    assert uris[0].startswith("file:")
    assert "笔记" in uris[0] or "%E7%AC%94%E8%AE%B0" in uris[0]


def test_text_only_drag_has_no_files() -> None:
    mime = QMimeData()
    mime.setText("一段话")

    assert _local_file_uris(mime) == []
