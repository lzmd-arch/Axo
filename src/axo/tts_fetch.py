"""TTS 权重的**可选**下载（用户显式触发，不自动、不后台）。

发布产物**不带** TTS 权重（那是几百 MB 的 torch 资产，见 ``axo/tts.py`` 的说明）。
本模块只提供一件事：给定 URL 与 sha256，把权重下载到数据根下的目标目录，校验后原子落盘。

纪律：
* **只接受 https**（``http://127.0.0.1`` / ``localhost`` 例外，本机自托管）；
* **sha256 必填**，校验不过就删除临时文件并报错；
* 失败抛 :class:`TtsFetchError`，错误信息是能直接显示的一句话；
* 不读环境变量里的 token —— 公开资产才走这条路；需要鉴权的分发请自带 opener。
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

__all__ = ["TtsFetchError", "fetch_weights", "main"]

_CHUNK = 1 << 20


class TtsFetchError(RuntimeError):
    """下载或校验失败。消息可直接给用户看。"""


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return
    raise TtsFetchError(f"只允许 https（本机 http 例外），收到：{url}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_weights(
    url: str,
    *,
    sha256: str,
    dest_dir: Path,
    filename: str | None = None,
    timeout: float = 60.0,
    opener: Callable | None = None,
) -> Path:
    """下载并校验一个权重文件，返回最终路径。

    Args:
        url: https 地址（本机 http 例外）。
        sha256: 期望的十六进制摘要（大小写不敏感）。
        dest_dir: 目标目录（通常是数据根下的 ``tts/``）。
        filename: 落盘文件名；缺省取 URL 的最后一段。
        timeout: 网络超时（秒）。
        opener: 可注入的 opener（测试用）。

    Raises:
        TtsFetchError: URL 不合规、网络失败、摘要不符。
    """
    _check_url(url)
    expected = sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise TtsFetchError("sha256 必须是 64 位十六进制。")
    name = filename or Path(urllib.parse.urlparse(url).path).name
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise TtsFetchError(f"文件名不合法：{name!r}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    open_ = opener or urllib.request.build_opener().open
    request = urllib.request.Request(url, headers={"User-Agent": "axo-tts-fetch/1"})

    handle_fd, temp_name = tempfile.mkstemp(prefix=".tts-", dir=dest_dir)
    temp = Path(temp_name)
    try:
        with os.fdopen(handle_fd, "wb") as out:
            try:
                with open_(request, timeout=timeout) as response:
                    for block in iter(lambda: response.read(_CHUNK), b""):
                        out.write(block)
            except OSError as exc:
                raise TtsFetchError(f"下载失败：{exc}") from exc
        actual = _sha256(temp)
        if actual != expected:
            raise TtsFetchError(f"sha256 不符（期望 {expected}，实际 {actual}）。")
        temp.replace(target)
    finally:
        # Windows 上杀毒/索引器可能短暂持有刚关掉的文件；清临时文件失败不应盖掉
        # 真正的结果（成功时文件已改名，失败时错误已经抛出）。
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="可选 TTS 权重下载（https + sha256）")
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--filename", default=None)
    args = parser.parse_args(argv)
    try:
        path = fetch_weights(
            args.url, sha256=args.sha256, dest_dir=args.dest, filename=args.filename
        )
    except TtsFetchError as exc:
        print(f"下载失败：{exc}")
        return 1
    print(f"已下载并校验：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
