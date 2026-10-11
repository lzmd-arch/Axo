"""密钥的加解密（Windows DPAPI）。

**这个文件的核心断言只有一条：解不开的时候返回空串、不抛。** "解不开"在真实世界里
不是异常路径而是常态——换了台机器、换了 Windows 账户、注册表被手改、密文被截断，
这四种情况下用户要做的是"重填一个 key"，而这只有在程序还能起来的时候才做得到。

round-trip 那几条**只在 Windows 上跑**（CI 是无头 Linux）：DPAPI 是系统调用，非 Windows
上 ``protect`` 按设计就返回空串——那是降级，不是缺陷。所以"非 Windows 上什么都不做"
也单独有一条断言，免得降级路径将来被人改成抛异常。
"""

from __future__ import annotations

import base64
import sys

import pytest

from biosnn_memterm import secrets

_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="DPAPI 是 Windows 的系统调用")

#: 形状取自 DeepSeek 的 key（``sk-`` 开头）。**只用假 key**。
_FAKE_KEY = "sk-0123456789abcdef0123456789abcdef"


@_WINDOWS_ONLY
def test_a_key_survives_a_round_trip() -> None:
    stored = secrets.protect(_FAKE_KEY)

    assert stored, "Windows 上保护一个非空明文不该得到空串"
    assert secrets.unprotect(stored) == _FAKE_KEY


@_WINDOWS_ONLY
def test_the_stored_form_is_not_the_key() -> None:
    """**这是这个模块存在的全部理由**：写进设置里的那一串不能是明文。

    只断言"不相等"是不够的——base64 一遍也会不相等，而 base64 不是加密。所以还断言
    密文里**找不到原串的任何一段**。
    """
    stored = secrets.protect(_FAKE_KEY)

    assert stored != _FAKE_KEY
    assert _FAKE_KEY not in stored
    assert "sk-0123" not in stored


@_WINDOWS_ONLY
def test_two_rounds_of_the_same_key_differ() -> None:
    """DPAPI 每次都会重新生成盐，所以同样的明文两次得到不同的密文。

    **这条不是洁癖**：如果两次相同，说明有人把它换成了确定性加密，而确定性加密会把
    "两个安装用了同一个 key"这件事泄漏到注册表里。
    """
    assert secrets.protect(_FAKE_KEY) != secrets.protect(_FAKE_KEY)


@_WINDOWS_ONLY
def test_is_available_says_yes_on_windows() -> None:
    assert secrets.is_available() is True


@pytest.mark.skipif(sys.platform == "win32", reason="测的是非 Windows 上的降级")
def test_off_windows_everything_degrades_to_empty() -> None:
    assert secrets.is_available() is False
    assert secrets.protect(_FAKE_KEY) == ""
    assert secrets.unprotect(_FAKE_KEY) == ""


def test_an_empty_key_needs_no_encryption() -> None:
    """空串是"没配"，不是"配了一个空的"。两个方向都原样返回，也不去碰系统调用。"""
    assert secrets.protect("") == ""
    assert secrets.unprotect("") == ""


@pytest.mark.parametrize(
    "stored",
    [
        "not base64 at all!!",
        "c2hvcnQ=",  # 合法 base64，但不是密文（解不开）
        "AAAA",  # 合法 base64 的零字节
        "@@@@",
        "c2hvcn",  # 长度不是 4 的倍数
    ],
)
def test_garbage_in_the_settings_never_raises(stored: str) -> None:
    """设置里那一格被手改成任何东西，都只能得到空串。

    **不抛是这条的全部意义**：注册表是可以被手改的，而 ``load_backend`` 在启动路径上
    ——那里抛一次，程序就打不开了，用户连重填 key 的界面都进不去。
    """
    assert secrets.unprotect(stored) == ""


@_WINDOWS_ONLY
def test_a_truncated_ciphertext_is_refused() -> None:
    """把真密文砍掉一半——它仍是合法 base64，但 DPAPI 必须拒绝它而不是解出垃圾。"""
    stored = secrets.protect(_FAKE_KEY)
    truncated = base64.b64encode(base64.b64decode(stored)[: len(stored) // 4]).decode("ascii")

    assert secrets.unprotect(truncated) == ""


@_WINDOWS_ONLY
def test_a_ciphertext_from_another_machine_is_refused() -> None:
    """换机器 / 换账户是**同一种**失败：DPAPI 解的密钥不在。

    没法在测试里真的换机器，所以拿 entropy 不同的密文当替身——用一个随机的、
    和本机 master key 无关的字节串冒充"别的机器上产出的密文"。
    """
    foreign = base64.b64encode(b"\x01\x02\x03" * 40).decode("ascii")

    assert secrets.unprotect(foreign) == ""
