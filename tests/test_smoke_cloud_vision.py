"""``tools/smoke_cloud_vision.py`` 的进程级契约。

**为什么是子进程而不是 import 进来调 ``main()``**：仓里对 ``tools/`` 的态度是"测试不该
import 工具"（同 ``test_sleep_once_cli.py`` 开头那条）——而这里要验的三件事里有两件本来
就只存在于进程边界上：**退出码**、以及"它真的去读了应用那份设置"。

**为什么要给 ``AXO_ISOLATED_STATE`` 一个临时目录**：这个脚本读的是应用自己那份
``QSettings``，不隔离它就会去读**这台机器上真的那份配置**——于是测试会随"这台机器配没配
云端"变绿变红，而那是仓里反复防的那种假绿。给一个临时目录之后，配置由测试自己写。

（**别想着用 ``APPDATA`` 环境变量**：Qt 走的是 Windows 的"已知文件夹"，实测把 ``APPDATA``
指到别处之后 ``QSettings`` 仍然落在真的 ``%APPDATA%`` 里。这条路是量出来不通的。）
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from axo.secrets import protect

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "smoke_cloud_vision.py"
_FAKE_KEY = "sk-fake-for-test-only"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        # **隔离开关收的是"目录"不是布尔**：``configure_isolated_state`` 把 ini 指到
        # ``<它>/settings`` 下面。所以这里给一个临时目录，这个脚本读的就是测试自己那份
        # 配置，与使用者注册表里那份无关。
        "AXO_ISOLATED_STATE": str(tmp_path / "iso"),
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONIOENCODING": "utf-8",
    }
    env.pop("AXO_DATA", None)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(_REPO),
        timeout=120,
    )


def _seed_cloud(tmp_path: Path, *, key: str = _FAKE_KEY) -> None:
    """手写一份隔离的 ini。**用文本写而不是调 QSettings**：测试不该为了造一份配置而
    先把 Qt 拉起来，而这个格式就这三层。

    **``api_key`` 那一格要写成 DPAPI 密文**（``axo.secrets.protect``）。写明文的话
    ``load_backend`` 解不开、读回空串，而脚本报的是「还没填 API key」——这个测试会红在
    一个**与它要验的东西无关**的原因上。（第一版就是这么红的：那条报错看着像「隔离没
    生效」，实际是密文格式不对。``axo.secrets`` 是纯 ctypes，不 import Qt，这一层留得住。）
    """
    folder = tmp_path / "iso" / "settings" / "Axo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "axo.ini").write_text(
        "[backend]\n"
        "provider=cloud\n"
        "cloud_base_url=https://api.deepseek.com/v1\n"
        f"api_key={protect(key)}\n"
        "last_model_cloud=deepseek-flash\n",
        encoding="utf-8",
        newline="\n",
    )


def test_without_a_cloud_config_it_refuses_and_says_why(tmp_path: Path) -> None:
    """没配云端 → **退 2 并说清**，而不是拿一个空地址去发请求。

    这条路的读者是"拿它当探针使"的人：他需要知道"是我没配"还是"配置不通"。
    """
    done = _run(tmp_path, "--dry-run")

    assert done.returncode == 2, done.stdout + done.stderr
    assert "云端" in done.stderr


_NEEDS_DPAPI = pytest.mark.skipif(
    sys.platform != "win32",
    reason=(
        "key 走 DPAPI：``secrets.protect`` 在没有 crypt32 的平台上退化成空串，"
        "所以那里根本种不进一份带 key 的云端配置（而应用本身也只有 Windows 版）"
    ),
)


@_NEEDS_DPAPI
def test_the_dry_run_shows_the_shape_and_never_the_key(tmp_path: Path) -> None:
    """配好之后 ``--dry-run`` 要把**请求形状**打出来，而 key 一个字都不出现。

    形状那一行正是这个脚本存在的理由：它让"图到底挂在哪一行、是不是分片"变成看得见的
    东西，而不是靠猜。key 不出现是硬要求——它的输出经常被人整段贴到别处去问。
    """
    _seed_cloud(tmp_path)

    done = _run(tmp_path, "--dry-run")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "image_url" in done.stdout, "没把图那一块打出来"
    assert "分片" in done.stdout, "没说清它是分片数组"
    assert _FAKE_KEY not in done.stdout + done.stderr, "key 漏到输出里了"
    assert "deepseek-flash" in done.stdout


@_NEEDS_DPAPI
def test_the_colour_it_asks_about_is_one_of_the_three(tmp_path: Path) -> None:
    """色块是**当场随机挑的三种之一**，而且它把答案打在屏幕上。

    这一条防的是"随机挑了个模型认不出的颜色"——那会让这个探针本身变成噪声源
    （认不出时到底是图没到，还是颜色太难认？）。
    """
    _seed_cloud(tmp_path)

    done = _run(tmp_path, "--dry-run")

    assert done.returncode == 0, done.stdout + done.stderr
    picks = [name for name in ("红", "绿", "蓝") if f"这一张色块是：{name}" in done.stdout]
    assert len(picks) == 1, f"色块那一行不对：{done.stdout}"
