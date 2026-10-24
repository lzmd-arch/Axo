"""从 UI 侧起停常驻总线进程。

**UI 只拿到一个 ``Popen`` 句柄**：它不知道 daemon 里有什么，也不 import 那边的
运行时依赖。这与"单向"是同一条边界：UI 读事件文件、发 SIGTERM，仅此而已。

**打包版不启动**：冻结后的 ``sys.executable`` 是 Axo 自己，``-m axo.busd`` 跑不起来。
与其在打包版里悄悄失败，不如明确地不启动，并把原因写在状态里（见
:func:`launch_blocked_reason`）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["launch_blocked_reason", "spawn_daemon", "stop_daemon"]


def launch_blocked_reason() -> str | None:
    """不能启动的理由；能启动返回 ``None``。"""
    if getattr(sys, "frozen", False):
        return "打包版没有可用的 Python 解释器来跑 -m axo.busd"
    return None


def spawn_daemon(
    *, db_path: Path, data_dir: Path, interval_s: int = 60, log_path: Path | None = None
) -> subprocess.Popen[bytes] | None:
    """起一个后台 daemon。返回句柄；不能启动时返回 ``None``。

    不设 ``daemon=True`` 的语义（Windows 没有 fork）：子进程独立于父进程存在，
    父进程退出**不会**自动带走它——这正是"常驻"要的，也意味着 UI 退出时必须
    显式调用 :func:`stop_daemon`，否则会攒下多个 daemon。

    **Windows 上 ``terminate()`` 是硬杀。** 它走 ``TerminateProcess``，daemon 里
    那个 ``SIGTERM`` 处理器不会跑，所以收尾时不会写 ``daemon_stop`` 事件。这不是
    数据风险：状态与 provenance **每一轮都落盘**，被杀掉的只是"这一次日志少一行"。
    真要优雅停就在控制台按 Ctrl+C（那一路会跑处理器）。
    """
    if launch_blocked_reason() is not None:
        return None
    cmd = [
        sys.executable,
        "-m",
        "axo.busd",
        "--db",
        str(db_path),
        "--data-dir",
        str(data_dir),
        "--interval",
        str(int(interval_s)),
    ]
    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["stdout"] = log_path.open("ab")  # noqa: SIM115 — 生命周期跟着子进程
        kwargs["stderr"] = kwargs["stdout"]
    try:
        return subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]
    except OSError:
        return None


def stop_daemon(proc: subprocess.Popen[bytes] | None, *, timeout: float = 5.0) -> None:
    """叫停并等它退。**幂等**；进程早就没了也不抛。"""
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass
