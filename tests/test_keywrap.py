"""预留的密钥封装接口：空实现绝不回落明文。"""

from __future__ import annotations

import pytest

from axo import keywrap


def test_null_wrapper_never_returns_plaintext() -> None:
    wrapper = keywrap.NullKeyWrapper()
    assert wrapper.available() is False
    assert wrapper.wrap(b"secret", context="x") == ""
    assert wrapper.unwrap("anything", context="x") is None


def test_hardware_placeholders_are_unavailable() -> None:
    assert keywrap.TpmKeyWrapper().available() is False
    assert keywrap.SecureEnclaveKeyWrapper().available() is False


def test_requiring_hardware_fails_closed() -> None:
    with pytest.raises(keywrap.KeyWrapperUnavailable):
        keywrap.current_wrapper(require_hardware=True)


@pytest.mark.skipif(not keywrap.SystemKeyWrapper().available(), reason="no system wrapper")
def test_system_wrapper_roundtrip_and_context_separation() -> None:
    wrapper = keywrap.SystemKeyWrapper()
    token = wrapper.wrap(b"secret", context="ctx-a")
    assert token
    assert wrapper.unwrap(token, context="ctx-a") == b"secret"
    assert wrapper.unwrap(token, context="ctx-b") is None
