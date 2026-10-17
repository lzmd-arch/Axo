"""随程序拉起语音组件：**替你按一下启动，不是把组件塞进来**。

组件仍然是独立进程（torch 不进本仓那条纪律没变），这一层只做三件事：程序起来时看一眼
端口，没在跑就按用户配的那条命令把它拉起来；程序退出时**只收自己拉起来的那个**；
拉不起来时说一句人话。

三条判据，每一条都对应一种真实的坏结果：

* **先探再拉**：端口上已经有东西在响应就不拉——否则会起第二个、抢不到端口，然后在
  组件的日志里留下一串与用户无关的错。
* **只收自己拉的**：``stop()`` 只杀自己 ``Popen`` 的那个进程树。用户手动开着的那一个
  我们不知道从哪儿来的，也不该动。
* **拉不起来要说**：命令写错了、路径不在了——沉默会让用户以为"自动启动开了"，而
  下一次开口还是没声。

这个模块**不认识 Qt**：它收一个"探活"函数与一个"拉起"函数，所以两条分支（探活着 / 探
不着）在核心腿里都能用替身逐条断言。
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

__all__ = ["CREATE_FLAGS", "SpeechLauncher"]

#: 拉起时用的 Windows 标志：**不弹控制台**（`CREATE_NO_WINDOW`）且**自成一个进程组**
#: （`CREATE_NEW_PROCESS_GROUP`）——后者是 `stop()` 能整棵树收掉的前提。
CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
)


class SpeechLauncher:
    """管"这一趟是不是我们拉起来的"。

    Args:
        command: 用户配的启动命令（一个 ``.bat`` / ``.cmd`` / ``.exe`` 的路径）。
        probe: 探活。返回真表示端口上已经有东西在答。
        spawn: 拉起进程。**注入是为了测试**：核心腿里不该真的起一个 TTS。
    """

    def __init__(
        self,
        command: str,
        *,
        probe: Callable[[], bool],
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.command = command.strip()
        self._probe = probe
        self._spawn = spawn
        self._process: subprocess.Popen | None = None

    @property
    def started(self) -> bool:
        """这一个是不是我们拉起来的（且还活着）。"""
        return self._process is not None and self._process.poll() is None

    def ensure(self) -> str:
        """看一眼，需要就拉起来。返回一句人话（空串 = 没事）。**不抛。**"""
        if not self.command:
            return ""
        if self._probe():
            # 已经有东西在答——**别拉第二个**。
            return ""
        try:
            self._process = self._spawn(self._argv(), **self._spawn_kwargs())
        except OSError as exc:
            # 路径写错、没有权限、文件被删——每一种都要说出来，否则"自动启动开了"
            # 与"它其实没起来"在界面上长得一模一样。
            return f"语音组件没能自动启动（{self.command}）：{exc}"
        return ""

    def stop(self) -> None:
        """收掉**自己拉起来的**那个进程树。幂等；没拉过就什么都不做。"""
        process = self._process
        if process is None or process.poll() is not None:
            return
        # 组件是一只 ``.bat``，它再起 python——所以要收**整棵树**，不能只杀 cmd。
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        except OSError:
            # 收不掉时退回普通终止：至少别让 cmd 挂着。
            process.terminate()
        self._process = None

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _argv(self) -> list[str]:
        """``.bat`` / ``.cmd`` 要走 ``cmd /c``——``CreateProcess`` 直接执行它们时
        参数与工作目录的处理与用户双击时不一样。"""
        suffix = Path(self.command).suffix.lower()
        if suffix in (".bat", ".cmd"):
            return ["cmd", "/c", self.command]
        return [self.command]

    def _spawn_kwargs(self) -> dict[str, object]:
        return {
            "creationflags": CREATE_FLAGS,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
