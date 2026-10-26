"""macOS 交付物的声明守卫：构建入口、bundle id、权限文案、CI runner。

**不跑 PyInstaller。** 这条腿在 Linux/Windows 的核心测试里也要过，而 macOS 的 .app 只能
在 macOS 上生成。能在这里守的是那些"搬了名字/漏了 Info.plist 键"的错：它们不会让
Windows 构建红，却会让 Mac 用户第一次按麦克风/截图时被系统直接杀掉进程。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_build_macos():
    # build_exe/build_macos 用 tomllib；Python 3.10 没有它。构建工具本身按 3.12 用，
    # 但核心测试矩阵里有 3.10——跳过这一条，而不是让整个文件在收集阶段红。
    pytest.importorskip("tomllib", reason="build_exe.py 要 tomllib（3.11+）")
    path = _ROOT / "tools" / "build_macos.py"
    spec = importlib.util.spec_from_file_location("_build_macos_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_builds_an_app_bundle_with_macos_separators(capsys) -> None:
    module = _load_build_macos()
    assert module.main(["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "--osx-bundle-identifier" in text
    assert "com.axo.memoryterminal" in text
    assert "--add-data" in text
    assert ":axo/ui" in text, "macOS 的 --add-data 分隔符是冒号"
    assert "Axo.app" in text


def test_vosk_model_is_installed_under_its_own_directory(tmp_path: Path) -> None:
    """``--add-data 目录:目标`` 拷的是目录内容，所以目标要带模型目录名。

    目标写成 ``models`` 时，模型会散成 ``models/am``、``models/conf``，而
    ``paths.bundled_voice_model_dir()`` 找的是 ``models/vosk-model-small-cn-0.22``。
    上一轮 macOS 构建正是红在这里。
    """
    module = _load_build_macos()
    command = module._build_command(model=tmp_path, icon=None)
    target = f"{tmp_path}:models/vosk-model-small-cn-0.22"
    assert target in command


def test_info_plist_declares_the_permissions_macos_requires() -> None:
    text = (_ROOT / "tools" / "build_macos.py").read_text(encoding="utf-8")
    for key in (
        "NSMicrophoneUsageDescription",
        "NSScreenCaptureUsageDescription",
        "NSAppleEventsUsageDescription",
        "LSMinimumSystemVersion",
    ):
        assert key in text, f"Info.plist 少了 {key}"


def test_workflow_runs_on_apple_silicon_and_uploads_the_installer() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "macos-release.yml").read_text(encoding="utf-8")
    assert "runs-on: macos-15" in workflow
    assert "tools/fetch_bundle_assets.py" in workflow
    assert "vosk-model-small-cn-0.22" in (_ROOT / "tools" / "fetch_bundle_assets.py").read_text(
        encoding="utf-8"
    )
    assert "tools/build_macos.py --release" in workflow.replace("\n", " ")
    assert "actions/upload-artifact@v4" in workflow


def test_macos_wheel_version_has_a_darwin_wheel() -> None:
    """vosk 0.3.45 的 PyPI 文件里**没有** macOS 轮子；0.3.44 才有 universal2。

    这条不是版本号洁癖：锁到 0.3.45，macOS 的 ``uv sync --locked`` 会在构建第一步
    直接失败，而 Windows 的测试完全看不到。
    """
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "vosk>=0.3.44,<0.3.45" in pyproject
    assert "vosk-0.3.44-py3-none-macosx_10_6_universal2.whl" in lock
