"""在本机执行一条命令。**纯标准库，不认识 Qt。**

这是整个工具面里权限最大的一件，所以每一条边界都单独写：

* **工作目录**由调用方给（限定在用户授权的根里），这里不猜；
* **环境变量白名单**：绝不继承 ``OPENAI_API_KEY`` / ``DEEPSEEK_KEY`` 之类——模型
  能跑的命令不该顺手拿到模型自己的凭据；
* **超时**（默认 30 秒、硬顶 600 秒）与**取消**：用户点停止/关窗时要能杀掉整棵
  进程树。Windows 上只杀 ``cmd`` 会留下 python 子进程，而它会一直占着管道；
* **输出上限**：保留首尾各一段并明说中间省略了多少——模型看到半截输出却不知道
  是半截，会基于半截下结论。

**不按元字符过滤命令。** 那是一种假安全：``;``、``|``、``$(...)`` 与 cmd 的各种
等价写法列不完，而漏掉的那一个会让"过滤了"变成一句安慰。真正的边界是审批、
工作目录、环境与超时。
"""

from __future__ import annotations

import contextlib
import functools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_TIMEOUT_S", "MAX_TIMEOUT_S", "ShellResult", "run_command"]

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 600

#: 输出总上限（字节）。首尾各留一半。
OUTPUT_CAP = 32 * 1024

#: 子进程拿得到的变量。**白名单，不继承。**
_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "SYSTEMROOT",
    "windir",
    "TEMP",
    "TMP",
    "COMSPEC",
    "USERPROFILE",
    "HOME",
    "LANG",
    "PYTHONIOENCODING",
)


@dataclass(frozen=True)
class ShellResult:
    ok: bool
    exit_code: int | None
    output: str
    truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    duration_ms: int = 0


def run_command(
    command: str,
    *,
    cwd: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    cancel: Any = None,
    output_cap: int = OUTPUT_CAP,
) -> ShellResult:
    """跑一条命令，返回结果。**不抛**（除了参数明显不对）。"""
    if not command.strip():
        raise ValueError("命令是空的。")
    timeout_s = max(1, min(MAX_TIMEOUT_S, int(timeout_s)))
    started = time.monotonic()

    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

    extra: dict[str, Any] = {}
    if sys.platform == "win32":
        extra["creationflags"] = flags
    else:
        extra["start_new_session"] = True
    try:
        process = subprocess.Popen(  # noqa: S602 — 见模块文档"不按元字符过滤"
            command,
            shell=True,
            cwd=str(cwd),
            env=_minimal_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **extra,
        )
    except OSError as exc:
        return ShellResult(
            ok=False, exit_code=None, output=f"命令起不来：{exc}", duration_ms=_ms(started)
        )

    timed_out = False
    cancelled = False
    while True:
        try:
            raw, _ = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                cancelled = True
            elif time.monotonic() - started >= timeout_s:
                timed_out = True
            if cancelled or timed_out:
                _kill_tree(process)
                try:
                    raw, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    raw = b""
                break

    text = raw.decode("utf-8", "replace") if raw else ""
    text, truncated = _clip_output(text, output_cap)
    code = process.returncode
    return ShellResult(
        ok=(code == 0 and not timed_out and not cancelled),
        exit_code=code,
        output=text,
        truncated=truncated,
        timed_out=timed_out,
        cancelled=cancelled,
        duration_ms=_ms(started),
    )


_MSVC_KEYS = (
    "PATH",
    "LIB",
    "INCLUDE",
    "LIBPATH",
    "VCINSTALLDIR",
    "VCTOOLSINSTALLDIR",
    "VCTOOLSVERSION",
    "VSINSTALLDIR",
    "WINDOWSSDKDIR",
    "WINDOWSSDKVERSION",
    "UCRTVERSION",
    "VSCMD_ARG_TGT_ARCH",
    "VSCMD_VER",
)


@functools.lru_cache(maxsize=1)
def _msvc_env() -> dict[str, str]:
    """Windows 上把 MSVC 编译环境补进白名单。找不到就返回空字典。

    为什么需要它：Rust 的 MSVC target 要 ``link.exe``/``LIB``/``INCLUDE``；Build Tools
    装好但不在 PATH 里时，``cargo test`` 会误抓到 Git 自带的 GNU ``link`` 然后报
    ``missing operand``。这里用 ``vswhere`` 找安装路径，再跑一次 ``vcvarsall.bat``
    只取编译相关变量——**不继承完整用户环境**，更不碰 API key。缓存一次，避免每条
    命令都付一次 ``vcvarsall`` 的启动成本。
    """
    if os.name != "nt":
        return {}
    program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return {}
    try:
        found = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    installation = found.stdout.strip().splitlines()[0].strip() if found.stdout.strip() else ""
    if not installation:
        return {}
    vcvars = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvars.is_file():
        return {}
    try:
        shell = subprocess.run(
            f'call "{vcvars}" x64 >nul && set',
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if shell.returncode != 0:
        return {}
    env: dict[str, str] = {}
    for line in shell.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.upper() in _MSVC_KEYS:
            env[key] = value
    return env


def _minimal_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    env.update(_msvc_env())
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """杀掉整棵进程树。Windows 用 ``taskkill /T``；POSIX 杀进程组。"""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
            return
        except OSError:
            pass
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), 9)
    except (OSError, ProcessLookupError):
        with contextlib.suppress(OSError):
            process.kill()


def _clip_output(text: str, cap: int) -> tuple[str, bool]:
    if len(text.encode("utf-8")) <= cap:
        return text, False
    half = max(1, cap // 2)
    head = text.encode("utf-8")[:half].decode("utf-8", "ignore")
    tail = text.encode("utf-8")[-half:].decode("utf-8", "ignore")
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[…]（中间省略约 {omitted} 字）\n\n{tail}", True


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
