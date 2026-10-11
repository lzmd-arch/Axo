"""Windows DPAPI 的一层薄封装：把一小段明文（云端 API key）变成能写进设置里的一串。

**为什么需要它。** 云端后端的 key 是一笔能直接花钱的凭据，而它的落点是 Windows 注册表
（``QSettings`` 的原生格式）。注册表不是保险柜：同机的另一个账户、一次"导出设置"、
一份被拷走的注册表分支，都能把它读走。DPAPI（``CryptProtectData``）把密文绑在
**当前 Windows 账户 + 当前机器**上——拷到别处就是一堆解不开的字节。它随系统提供，
所以这一层不需要任何新依赖。

**为什么不是自己写加密。** 自己拼一套对称加密就得自己管密钥，而密钥又得存在某处，
于是问题只是换了个地方——DPAPI 把"密钥存哪"整个问题交给了操作系统。

**为什么不在 ``ui/settings_store.py`` 里。** 不是因为它是纯层（``ctypes`` 与 ``base64``
都是标准库，不在 ``test_layering._PURE_FORBIDDEN`` 里），而是因为**平台相关的代码该集中
在一个模块**：``paths.py`` 管操作系统的目录，这里管操作系统的凭据，两者并列。

**它只做加解密，不做"要不要加密"的决定。** 调用方（``ui/settings_store.py``）是唯一知道
"盘上那一串是密文、内存里这个字段是明文"的地方——两处解密点迟早会漂。

**降级一律是空串，绝不抛。** 换机器、换 Windows 账户、注册表被手改、密文损坏、非
Windows——这五种情况下正确的行为都是"当作没配"，让用户重填一个，而不是让整个程序起不来。
"""

from __future__ import annotations

import base64
import ctypes
import sys
from functools import lru_cache
from typing import Any

__all__ = ["is_available", "protect", "unprotect"]

#: ``CRYPTPROTECT_UI_FORBIDDEN``——**绝不让系统弹 UI**。没有它，某些配置下加密会弹一个
#: 授权对话框；对一个后台线程里跑的设置保存来说，那就是一次无声的挂起。
_CRYPTPROTECT_UI_FORBIDDEN = 0x01

_WINDOWS = sys.platform == "win32"


class _DataBlob(ctypes.Structure):
    """``DATA_BLOB``。用定宽的 ``c_uint32``/``c_void_p`` 而不是 ``ctypes.wintypes``
    ——这个类在**非 Windows** 上也会被定义（模块要被 import），而 ``wintypes`` 里有些
    名字在别的平台上是空的。"""

    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.c_void_p)]


@lru_cache(maxsize=1)
def _crypt32() -> tuple[Any, Any] | None:
    """载入 ``crypt32``/``kernel32`` 并把**每个参数类型都显式写死**。

    **这一步不能省。** 没声明 ``argtypes`` 时 ctypes 把出参按 C ``int`` 处理，64 位下
    指针会被截成 32 位——那种错只在别人的机器上炸（本地分配器恰好把内存放在低地址时
    看着一切正常）。载不到（非 Windows、系统被裁剪）时返回 ``None``，调用方一律降级。
    """
    if not _WINDOWS:
        return None
    try:
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (OSError, AttributeError):
        return None
    blob_p = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [
        blob_p,  # pDataIn
        ctypes.c_wchar_p,  # szDataDescr（给"解密时能认出这是谁写的"，我们不用）
        blob_p,  # pOptionalEntropy
        ctypes.c_void_p,  # pvReserved（必须 NULL）
        ctypes.c_void_p,  # pPromptStruct
        ctypes.c_uint32,  # dwFlags
        blob_p,  # pDataOut
    ]
    crypt32.CryptProtectData.restype = ctypes.c_int
    crypt32.CryptUnprotectData.argtypes = [
        blob_p,  # pDataIn
        ctypes.POINTER(ctypes.c_wchar_p),  # ppszDataDescr（出参，传 NULL 即可）
        blob_p,  # pOptionalEntropy
        ctypes.c_void_p,  # pvReserved
        ctypes.c_void_p,  # pPromptStruct
        ctypes.c_uint32,  # dwFlags
        blob_p,  # pDataOut
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _blob_from(data: bytes) -> _DataBlob:
    """一个指向 ``data`` 的 ``DATA_BLOB``。

    ``_keepalive`` 是**必须的**：``create_string_buffer`` 是那块内存唯一的持有者，
    函数一返回它就被回收，而 DPAPI 要到真正调用时才去读那个指针——少这一行，得到的是
    一种"本地看着好、换个分配器就崩"的错。
    """
    buffer = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.c_void_p))
    blob._keepalive = buffer  # type: ignore[attr-defined]
    return blob


def is_available() -> bool:
    """这台机器上能不能用（非 Windows 一律 False）。"""
    return _crypt32() is not None


def protect(plaintext: str) -> str:
    """明文 → 写进设置里的一串（base64）。空串进、空串出；失败也是空串。"""
    if not plaintext:
        return ""
    loaded = _crypt32()
    if loaded is None:
        return ""
    crypt32, kernel32 = loaded
    source = _blob_from(plaintext.encode("utf-8"))
    out = _DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out),
    )
    if not ok:
        return ""
    try:
        cipher = ctypes.string_at(out.pbData, out.cbData)
    finally:
        # **DPAPI 分配的内存只能用它自己的 LocalFree 还**，不能用 Python 的释放器。
        kernel32.LocalFree(out.pbData)
    return base64.b64encode(cipher).decode("ascii")


def unprotect(stored: str) -> str:
    """设置里那一串 → 明文。**任何一步不对都返回空串，绝不抛**（见模块 docstring）。"""
    if not stored:
        return ""
    loaded = _crypt32()
    if loaded is None:
        return ""
    crypt32, kernel32 = loaded
    try:
        cipher = base64.b64decode(stored, validate=True)
    except (ValueError, TypeError):
        return ""
    if not cipher:
        return ""
    source = _blob_from(cipher)
    out = _DataBlob()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out),
    )
    if not ok:
        return ""
    try:
        raw = ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
