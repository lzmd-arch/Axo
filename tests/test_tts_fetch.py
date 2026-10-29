"""可选 TTS 权重下载的守卫测试（不联网）。"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from axo.tts_fetch import TtsFetchError, fetch_weights

PAYLOAD = b"fake-tts-weights"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _opener(data: bytes = PAYLOAD):
    return lambda _request, **_kwargs: _Response(data)


def test_downloads_and_verifies(tmp_path):
    path = fetch_weights(
        "https://example.com/weights.zip", sha256=SHA, dest_dir=tmp_path, opener=_opener()
    )
    assert path.read_bytes() == PAYLOAD
    assert path.name == "weights.zip"
    assert not list(tmp_path.glob(".tts-*"))


def test_checksum_mismatch_removes_temp(tmp_path):
    with pytest.raises(TtsFetchError):
        fetch_weights(
            "https://example.com/w.zip",
            sha256="0" * 64,
            dest_dir=tmp_path,
            opener=_opener(),
        )
    assert not list(tmp_path.glob(".tts-*"))
    assert not (tmp_path / "w.zip").exists()


def test_rejects_plain_http():
    with pytest.raises(TtsFetchError):
        fetch_weights("http://example.com/w.zip", sha256=SHA, dest_dir=Path())


def test_allows_localhost_http(tmp_path):
    path = fetch_weights(
        "http://127.0.0.1:8000/w.bin", sha256=SHA, dest_dir=tmp_path, opener=_opener()
    )
    assert path.exists()


def test_rejects_bad_sha_and_name(tmp_path):
    with pytest.raises(TtsFetchError):
        fetch_weights("https://example.com/w.zip", sha256="xyz", dest_dir=tmp_path)
    with pytest.raises(TtsFetchError):
        fetch_weights("https://example.com/w.zip", sha256=SHA, dest_dir=tmp_path, filename="../x")


def test_cli_reports_error(tmp_path):
    from axo.tts_fetch import main

    assert main(["--url", "http://example.com/w", "--sha256", SHA, "--dest", str(tmp_path)]) == 1
