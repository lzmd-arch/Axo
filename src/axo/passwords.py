"""应用密码与恢复码：只做哈希/校验，不碰 Qt、不碰设置存储。

格式固定为 ``pbkdf2-sha256$1$<iterations>$<salt>$<digest>``（salt/digest 用
URL-safe base64，不带 padding）。PBKDF2 是标准库能力，不引入第三方 KDF；默认
600,000 次，解析时把可接受范围夹在 100k–2M，防止被手改的超大值拖死启动。

**这里从不保存明文。** 密码/恢复码只在调用栈里；返回值是 verifier 字符串。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

__all__ = [
    "DEFAULT_ITERATIONS",
    "MAX_ITERATIONS",
    "MIN_ITERATIONS",
    "VerifierParams",
    "display_recovery_code",
    "hash_password",
    "hash_recovery_code",
    "needs_rehash",
    "new_recovery_code",
    "normalize_recovery_code",
    "parse_verifier",
    "password_record_ok",
    "self_test",
    "verify_password",
    "verify_recovery_code",
]

DEFAULT_ITERATIONS = 600_000
MIN_ITERATIONS = 100_000
MAX_ITERATIONS = 2_000_000
SALT_BYTES = 16
DIGEST_BYTES = 32
RECOVERY_BYTES = 20
_ALGORITHM = "pbkdf2-sha256"
_VERSION = "1"

_RECOVERY_ALPHABET = re.compile(r"^[A-Z2-7]{32}$")


@dataclass(frozen=True)
class VerifierParams:
    """解析后的 verifier 参数。``salt``/``digest`` 是原始字节。"""

    iterations: int
    salt: bytes
    digest: bytes


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes | None:
    try:
        padded = text + "=" * (-len(text) % 4)
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError):
        return None


def _derive(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, DIGEST_BYTES)


def hash_password(
    password: str,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """生成 verifier。密码长度 8–1024 字符；空/超长抛 ``ValueError``。"""
    if len(password) < 8:
        raise ValueError("密码至少需要 8 个字符。")
    if len(password) > 1024:
        raise ValueError("密码太长。")
    if iterations < MIN_ITERATIONS or iterations > MAX_ITERATIONS:
        raise ValueError("迭代次数超出允许范围。")
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    if len(salt) != SALT_BYTES:
        raise ValueError("salt 长度不对。")
    digest = _derive(password, salt, iterations)
    return f"{_ALGORITHM}${_VERSION}${iterations}${_b64e(salt)}${_b64e(digest)}"


def parse_verifier(text: str) -> VerifierParams | None:
    """严格解析；任何不合格返回 ``None``，绝不抛。"""
    try:
        algorithm, version, iterations_text, salt_text, digest_text = text.split("$")
    except (AttributeError, ValueError):
        return None
    if algorithm != _ALGORITHM or version != _VERSION:
        return None
    try:
        iterations = int(iterations_text)
    except ValueError:
        return None
    if iterations < MIN_ITERATIONS or iterations > MAX_ITERATIONS:
        return None
    salt = _b64d(salt_text)
    digest = _b64d(digest_text)
    if salt is None or digest is None:
        return None
    if len(salt) != SALT_BYTES or len(digest) != DIGEST_BYTES:
        return None
    return VerifierParams(iterations=iterations, salt=salt, digest=digest)


def password_record_ok(text: str) -> bool:
    """启动自检用：只验记录格式，不验任何候选密码。"""
    return parse_verifier(text) is not None


def verify_password(password: str, verifier: str) -> bool:
    params = parse_verifier(verifier)
    if params is None or not password:
        return False
    candidate = _derive(password, params.salt, params.iterations)
    return hmac.compare_digest(candidate, params.digest)


def needs_rehash(verifier: str) -> bool:
    params = parse_verifier(verifier)
    return params is not None and params.iterations < DEFAULT_ITERATIONS


def new_recovery_code() -> str:
    """20 字节随机 → 32 个 Base32 字符（无分组，显示时再分组）。"""
    raw = secrets.token_bytes(RECOVERY_BYTES)
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def normalize_recovery_code(text: str) -> str | None:
    """大写、去空格/连字符；格式不合法返回 ``None``。"""
    value = re.sub(r"[\s\-]+", "", (text or "").upper())
    if not _RECOVERY_ALPHABET.fullmatch(value):
        return None
    return value


def display_recovery_code(code: str) -> str:
    normalized = normalize_recovery_code(code) or ""
    return " ".join(normalized[index : index + 4] for index in range(0, len(normalized), 4))


def hash_recovery_code(code: str) -> str:
    normalized = normalize_recovery_code(code)
    if normalized is None:
        raise ValueError("恢复码格式不对。")
    return hash_password(normalized, iterations=DEFAULT_ITERATIONS)


def verify_recovery_code(code: str, verifier: str) -> bool:
    normalized = normalize_recovery_code(code)
    return bool(normalized) and verify_password(normalized, verifier)


def self_test() -> bool:
    """固定已知答案向量；证明当前机器的 PBKDF2 实现可用。

    **不走 `parse_verifier`**：这里的 1000 次是测试向量，低于生产下限，解析器
    会按设计拒绝它。直接比对 KDF 输出，才能既快又真正证明实现可用。
    """
    expected = _b64d("wqoOZE6t69o_DhhTy5szbbISWVlO111DUgJNTfJ5Z2w")
    if expected is None:
        return False
    return hmac.compare_digest(_derive("axo-self-test", bytes(range(16)), 1000), expected)
