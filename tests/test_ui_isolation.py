"""隔离**真的成立吗**——这一条是被一次假绿逼出来的。

原来只有一条判据：`_probe` 给子进程设了 `AXO_ISOLATED_STATE`（见
`test_packaging.py`）。那条是绿的，而机制是坏的——应用侧当时写的是
`QSettings.setDefaultFormat(IniFormat)`，可**那个函数管不到两参数的构造**：它只影响
`QSettings(parent)` 那一个形式，而 `QSettings("Axo", "axo")` 在 Windows 上永远用原生
格式。实测的结果是设完之后 `format()` 仍然是 `NativeFormat`、设置照样写进真注册表
——"隔离"只成立在日志里，而 `test_packaging` 那条看着一切正常。

所以这里量的是**行为**：进程里建出来的那个 `QSettings` 到底是文件还是注册表、落在哪。
顺带把"不设变量时行为不变"也钉住——隔离是给自检用的，不该顺手改掉正常启动的存储。

**每条都另起一个进程。** `QSettings.setPath` 与 `QStandardPaths.setTestModeEnabled`
是进程级的、不可逆的静态设置，在同一个进程里先开隔离再"关掉"是做不出来的——那会让
这条测试污染它后面跑的每一个用例。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

_PROBE = """
import json, os, sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from axo.ui.app import configure_isolated_state, make_settings

configure_isolated_state()
settings = make_settings()
if len(sys.argv) > 1 and sys.argv[1] == "write":
    settings.setValue("backend/__isolation_probe_do_not_keep__", "1")
    settings.sync()
cache = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
local = os.environ.get("LOCALAPPDATA")
print(json.dumps({
    "format": str(settings.format()),
    "file": str(settings.fileName()),
    "exists": Path(settings.fileName()).is_file(),
    "cache": str(cache),
    "real_cache": str(Path(local) / "Axo" / "axo" / "cache") if local else "",
}))
"""


def _probe(state: Path | None, *, write: bool = False) -> dict[str, str | bool]:
    """在**另一个进程**里跑一次 `configure_isolated_state` + `make_settings`。

    `write` 只在隔离那一条上用：不隔离时写一次就会落到使用者的真存储里，而这条测试
    不该留下任何东西。

    ⚠️ **机制坏掉时它一定会漏。** 这正是它要抓的事——所以那个键名写成
    ``backend/__isolation_probe_do_not_keep__``：万一哪次真漏进了注册表，"这是测试
    留下的"一眼能看出来（而不是一个叫 ``probe`` 的、看着像真设置的键）。
    """
    environment = os.environ.copy()
    environment.pop("AXO_ISOLATED_STATE", None)
    if state is not None:
        environment["AXO_ISOLATED_STATE"] = str(state)
    command = [sys.executable, "-c", _PROBE]
    if write:
        command.append("write")
    done = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", env=environment, check=True
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_isolation_sends_the_settings_to_a_file_under_the_state_directory(tmp_path: Path) -> None:
    """**这条是核心。** 隔离开着时，设置必须落成一个**文件**、而且落在那个目录里。

    三条缺一不可：格式是 ini（不是注册表）、文件真的写出来了、它在那棵临时目录底下。
    只查前两条的话，`setPath` 写错一个目录名照样能过——而那时隔离漏到了别的地方。
    """
    state = tmp_path / "state"

    result = _probe(state, write=True)

    assert result["format"] == "Format.IniFormat", "还是原生格式（注册表），隔离没生效"
    assert result["exists"] is True, "没有真的写出来"
    assert Path(str(result["file"])).resolve().is_relative_to(state.resolve())


def test_isolation_moves_qt_cache_away_from_the_real_location(tmp_path: Path) -> None:
    """Qt 缓存也要挪走——那是另一半泄漏（WebEngine 的管线缓存）。

    Windows 上才有意义：这条比的是"不再等于 `%LOCALAPPDATA%\\Axo\\axo\\cache`"，
    而别的平台上那个路径本来就不存在，比不出东西来。
    """
    if not os.environ.get("LOCALAPPDATA"):  # pragma: no cover - 只在 Windows 上跑
        pytest.skip("这个判据比的是 %LOCALAPPDATA% 那条路径，别的平台没有它")

    result = _probe(tmp_path / "state")

    assert Path(str(result["cache"])).resolve() != Path(str(result["real_cache"])).resolve()


def test_without_the_variable_nothing_changes(tmp_path: Path) -> None:
    """不设变量时行为**一个字不变**。

    这条挡的是"顺手把所有人都切到 ini"——那会把已有用户的设置留在一个空文件旁边，
    症状是"换了个版本，外观与 API key 全没了"（API key 还是 DPAPI 加密的，换个位置
    就解不开）。

    不写，只问"它打算落在哪"。
    """
    result = _probe(None)

    assert result["format"] != "Format.IniFormat"
    assert not Path(str(result["file"])).resolve().is_relative_to(tmp_path.resolve())
