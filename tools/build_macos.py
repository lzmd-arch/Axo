"""把应用冻结成 macOS 的 ``.app``，并打成 zip / dmg。

**为什么单独一个脚本，而不是在 ``build_exe.py`` 里加分支。** Windows 那条流水线的
自检是活体探测（起窗口、读 Win32 窗口类、查注册表、弹提醒），每一步都长在 Windows
API 上；把 macOS 塞进同一组 ``if`` 会让两边都读不懂。共用的部分（入口、资产表、hidden
imports、图标源）从 ``build_exe`` 取，平台自己的构建与自检留在这里。

**它不是交叉编译器。** PyInstaller 只能为目标平台原生构建：Windows 上跑这个脚本只会
得到一句"去 macOS 上跑"。本仓的做法是由 GitHub Actions 的 ``macos-15`` runner 构建，
产物再下载回来（见 ``.github/workflows/macos-release.yml``）。

用法::

    python tools/build_macos.py --dry-run
    python tools/build_macos.py --release --bundle-vosk-model /path/to/vosk-model-small-cn-0.22
    python tools/build_macos.py --release --skip-build   # 只重打包/自检现有的 .app
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_exe import (  # noqa: E402
    _BUILD,
    _BUNDLE_ASSETS,
    _DIST,
    DATAS,
    ENTRY,
    HIDDEN_IMPORTS,
    ICON_SOURCE,
    _bundle_datas,
    _bundle_missing,
)

APP_NAME = "Axo"
BUNDLE_ID = "com.axo.memoryterminal"
APP = _DIST / f"{APP_NAME}.app"
ICNS = _BUILD / "Axo.icns"

_VOSK_MODEL = "vosk-model-small-cn-0.22"
_ICON_SPECS = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def _version() -> str:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _arch() -> str:
    return "arm64" if platform.machine() in ("arm64", "aarch64") else "x86_64"


def _run(
    command: list[str], *, timeout: int = 1800, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command) + (f"  (cwd={cwd})" if cwd else ""), flush=True)
    done = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"命令失败（退出码 {done.returncode}）：{command[0]}")
    return done


def _make_icns() -> Path | None:
    """从品牌立绘做一份 ``.icns``。失败不中断构建（同 Windows 的图标策略）。"""
    if not ICON_SOURCE.is_file():
        print(f"没有图标源：{ICON_SOURCE}（用 PyInstaller 默认图标）")
        return None
    iconset = _BUILD / "Axo.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    try:
        for name, size in _ICON_SPECS:
            _run(
                [
                    "sips",
                    "-z",
                    str(size),
                    str(size),
                    str(ICON_SOURCE),
                    "--out",
                    str(iconset / name),
                ],
                timeout=120,
            )
        _run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(ICNS)],
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError, SystemExit) as exc:
        print(f"图标没做出来，照常构建：{exc!r}")
        return None
    print(f"图标：{ICNS.relative_to(_ROOT).as_posix()}")
    return ICNS


def _vosk_library() -> Path | None:
    """macOS 轮子里那个原生库叫 ``libvosk.dyld``（不是 ``.dylib``）。

    ``--collect-binaries vosk`` 按 ``.dylib``/``.so``/``.dll`` 扫，收不到这个后缀，
    所以必须显式 ``--add-data`` 到 ``vosk/``——那正是 ``vosk/__init__.py`` 拼出来的
    ``os.path.dirname(__file__)`` 指向的地方。
    """
    try:
        import vosk
    except ImportError:
        return None
    candidate = Path(vosk.__file__).with_name("libvosk.dyld")
    return candidate if candidate.is_file() else None


def _build_command(*, model: Path | None, icon: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--osx-bundle-identifier",
        BUNDLE_ID,
        "--paths",
        str(_ROOT / "src"),
        "--distpath",
        str(_DIST),
        "--workpath",
        str(_BUILD / "Axo-macos"),
        "--specpath",
        str(_BUILD),
        "--collect-binaries",
        "vosk",
    ]
    for package in ("sherpa_onnx", "onnxruntime", "soundfile", "tokenizers"):
        command += ["--collect-binaries", package]
    for module in HIDDEN_IMPORTS:
        command += ["--hidden-import", module]
    for source, target in DATAS:
        command += ["--add-data", f"{_ROOT / source}:{target}"]
    for source, target in _bundle_datas():
        command += ["--add-data", f"{source}:{target}"]
    vosk_library = _vosk_library()
    if vosk_library is not None:
        command += ["--add-data", f"{vosk_library}:vosk"]
    if model is not None:
        # PyInstaller 的 --add-data 对**目录**是把内容拷进目标，不是把目录本身放进去；
        # 目标写成 models 会得到 models/am、models/conf……，而 paths 找的是
        # models/vosk-model-small-cn-0.22。所以目标要带上模型目录名。
        command += ["--add-data", f"{model}:models/{_VOSK_MODEL}"]
    if icon is not None:
        command += ["--icon", str(icon)]
    command.append(str(_ROOT / ENTRY))
    return command


def _write_info_plist(app: Path, version: str) -> None:
    import plistlib

    path = app / "Contents" / "Info.plist"
    payload = plistlib.loads(path.read_bytes())
    payload.update(
        {
            "CFBundleDisplayName": APP_NAME,
            "CFBundleName": APP_NAME,
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "CFBundleDevelopmentRegion": "zh_CN",
            "LSMinimumSystemVersion": "14.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Axo / 千岁（阿螈）",
            "NSMicrophoneUsageDescription": (
                "Axo 用麦克风做离线语音输入与语音通话；识别在本机完成，录音不离开这台 Mac。"
            ),
            "NSScreenCaptureUsageDescription": (
                "Axo 只在你点「允许」后截一张屏幕图，交给当前模型看一次；图片不落盘。"
            ),
            "NSAppleEventsUsageDescription": "Axo 只在你明确要求时打开外部程序或网页。",
        }
    )
    path.write_bytes(plistlib.dumps(payload))
    print(f"Info.plist：{path.relative_to(_ROOT).as_posix()}")


def _sign(app: Path) -> None:
    # 没有 Apple Developer ID，只能 ad-hoc 签名。它让 Apple Silicon 接受这个二进制，
    # 但**不等于公证**——Gatekeeper 仍会要求用户右键打开（README 里写清楚）。
    _run(["codesign", "--force", "--deep", "--sign", "-", str(app)], timeout=900)
    _run(["codesign", "--verify", "--deep", "--strict", str(app)], timeout=300)


def _check_bundle(app: Path, version: str, *, require_model: bool) -> list[str]:
    import plistlib

    problems: list[str] = []
    executable = app / "Contents" / "MacOS" / APP_NAME
    if not executable.is_file():
        problems.append(f"没有可执行文件：{executable}")
    plist_path = app / "Contents" / "Info.plist"
    try:
        plist = plistlib.loads(plist_path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        problems.append("Info.plist 读不出来")
        plist = {}
    for key in ("CFBundleIdentifier", "CFBundleShortVersionString", "CFBundleExecutable"):
        if not plist.get(key):
            problems.append(f"Info.plist 缺 {key}")
    if plist.get("CFBundleIdentifier") != BUNDLE_ID:
        problems.append(f"Bundle id 不是 {BUNDLE_ID}")
    if plist.get("CFBundleShortVersionString") != version:
        problems.append("Info.plist 的版本与 pyproject 不一致")
    for key in (
        "NSMicrophoneUsageDescription",
        "NSScreenCaptureUsageDescription",
        "NSAppleEventsUsageDescription",
    ):
        if not plist.get(key):
            problems.append(f"Info.plist 缺 {key}（macOS 会在第一次用到时直接杀掉进程）")
    resources = app / "Contents" / "Resources"
    for relative in (
        "axo/ui/theme.qss",
        "axo/ui/web/index.html",
        "axo/ui/assets/stickers/manifest.json",
    ):
        if not (resources / relative).exists() and not any(app.rglob(Path(relative).name)):
            problems.append(f"资产不在：{relative}")
    if not any(app.rglob("libvosk.dyld")):
        problems.append("vosk 的 libvosk.dyld 不在（--add-data vosk 失效）")
    if require_model and not any(app.rglob(_VOSK_MODEL)):
        problems.append(f"随包语音模型不在：{_VOSK_MODEL}")
    return problems


def _smoke(app: Path) -> dict[str, object]:
    executable = app / "Contents" / "MacOS" / APP_NAME
    with tempfile.TemporaryDirectory(prefix="axo-macos-smoke-") as temporary:
        report = Path(temporary) / "report.json"
        data = Path(temporary) / "data"
        state = Path(temporary) / "state"
        env = dict(os.environ)
        env.update(
            {
                "AXO_DATA": str(data),
                "AXO_ISOLATED_STATE": str(state),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        done = subprocess.run(
            [str(executable), "--bundle-smoke", "--out", str(report)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=180,
            env=env,
            check=False,
        )
        if done.stdout:
            print(done.stdout.rstrip())
        if done.stderr:
            print(done.stderr.rstrip(), file=sys.stderr)
        if done.returncode != 0 or not report.is_file():
            raise SystemExit(f"bundle 自检失败（退出码 {done.returncode}）")
        payload = json.loads(report.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise SystemExit("bundle 自检报告不通过：" + json.dumps(payload, ensure_ascii=False))
    print("bundle 自检通过：Qt / vosk / 资产 / 语音模型都在。")
    return payload if isinstance(payload, dict) else {}


_README = """Axo（千岁 / 阿螈）macOS 包

安装
----
把 Axo.app 拖进「应用程序」。首次打开请右键点 Axo.app →「打开」。
这个包没有 Apple Developer ID，也没有做 Apple 公证；直接双击时 Gatekeeper 会拦一次，
右键打开之后系统会记住。若提示"已损坏"，在终端执行：
    xattr -dr com.apple.quarantine /Applications/Axo.app

里面有什么
----------
- 完整的 Axo 桌面程序：对话、记忆库、编码 agent、工具面、语音输入、语音通话、
  屏幕识别、提醒、表情包、音乐库、记忆同步。
- 中文离线语音模型 vosk-model-small-cn-0.22 已随包带；不需要再下载。
- 语言服务器不随包带（它们体积大、更新快）。设置 → 工具 → 语言服务器 可以体检、
  预览并自动安装；macOS 上走 npm / rustup / go / dotnet / Homebrew。
- 记忆库、设置与用户文件都在首次启动时新建，包本身是干净的初始状态。
- 语音输出（TTS）随包带千岁的 ZipVoice（CPU，不需要 torch 或外部组件）；
  设置里可以关掉它改用外部 F5-TTS 组件。
- 语义召回随包带 bge-small-zh-v1.5（ONNX），不要求 Ollama 嵌入模型。
- 许可全文在 licenses/，第三方组件清单在 THIRD-PARTY-NOTICES.txt。

第一次启动要知道
----------------
- 对话需要一个模型后端：本机 Ollama，或在设置里填云端 API key。
  macOS 的 key 存在系统钥匙串，不写进 plist。
- 语音输入/通话第一次会请求麦克风；屏幕识别第一次会请求屏幕录制。
  没有权限时对应工具会明确说"没看成/没听到"，而不是编一个结果。
- 数据默认在 ~/Library/Application Support/axo；设置里可以搬走。
"""


def _build_mcp_servers() -> None:
    """冻结两个 MCP 控制台程序到 ``dist/_mcp/``。

    与 Windows 侧用**同一份入口清单**（``build_exe.MCP_ENTRIES``）：桌面包不带这两个
    程序的话，用户勾了"允许外部提议写入"也等不到任何提议，而且不会报错。
    """
    from tools.build_exe import MCP_ENTRIES

    out = _DIST / "_mcp"
    out.mkdir(parents=True, exist_ok=True)
    for exe, entry in MCP_ENTRIES.items():
        _run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--onefile",
                "--console",
                "--name",
                exe,
                "--paths",
                str(_ROOT / "src"),
                "--distpath",
                str(out),
                "--workpath",
                str(_BUILD / exe),
                "--specpath",
                str(_BUILD),
                "--collect-binaries",
                "numpy",
                str(_ROOT / entry),
            ],
            timeout=1800,
            cwd=_ROOT,
        )


def _package(version: str, arch: str, *, make_dmg: bool = True) -> list[Path]:
    stage = _BUILD / "Axo-macos-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(APP, stage / f"{APP_NAME}.app", symlinks=True)
    notices = _ROOT / "THIRD-PARTY-NOTICES.md"
    if notices.is_file():
        shutil.copyfile(notices, stage / "THIRD-PARTY-NOTICES.txt")
    license_file = _ROOT / "LICENSE"
    if license_file.is_file():
        shutil.copyfile(license_file, stage / "LICENSE")
    if (_ROOT / "licenses").is_dir():
        shutil.copytree(_ROOT / "licenses", stage / "licenses")
    if not (stage / "LICENSE").is_file() or not (stage / "licenses" / "LGPL-3.0.txt").is_file():
        raise SystemExit("安装包里少了 LICENSE 或 LGPL-3.0.txt：LGPL §4(b) 的副本义务")
    for exe in ("axo-mcp", "axo-mcp-write"):
        binary = _DIST / "_mcp" / exe
        if not binary.is_file():
            raise SystemExit(f"缺 {exe}：MCP 入口没进包，写档位等于空转")
        shutil.copyfile(binary, stage / exe)
        (stage / exe).chmod(0o755)
    (stage / "README-macOS.txt").write_text(_README, encoding="utf-8")

    stem = f"{APP_NAME}-{version}-macos-{arch}"
    zip_path = _DIST / f"{stem}.zip"
    if zip_path.exists():
        zip_path.unlink()
    _run(["zip", "-qry", str(zip_path), "."], timeout=900, cwd=stage)

    artifacts = [zip_path]
    if make_dmg:
        dmg_path = _DIST / f"{stem}.dmg"
        if dmg_path.exists():
            dmg_path.unlink()
        dmg_stage = _BUILD / "Axo-macos-dmg"
        if dmg_stage.exists():
            shutil.rmtree(dmg_stage)
        shutil.copytree(stage, dmg_stage, symlinks=True)
        (dmg_stage / "Applications").symlink_to("/Applications")
        _run(
            [
                "hdiutil",
                "create",
                "-volname",
                APP_NAME,
                "-srcfolder",
                str(dmg_stage),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            timeout=900,
        )
        artifacts.append(dmg_path)
    for path in artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {path.stat().st_size} 字节  sha256 {digest}")
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true", help="出 zip 与 dmg（默认也出）")
    parser.add_argument("--skip-build", action="store_true", help="复用 dist/Axo.app")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令")
    parser.add_argument("--no-dmg", action="store_true", help="只出 zip")
    parser.add_argument("--bundle-vosk-model", default="", help="随包带的中文语音模型目录")
    args = parser.parse_args(argv)

    version = _version()
    arch = _arch()
    model = Path(args.bundle_vosk_model).expanduser().resolve() if args.bundle_vosk_model else None
    if model is not None and not model.is_dir():
        raise SystemExit(f"--bundle-vosk-model 不是目录：{model}")
    # 构建资产里已经有 vosk 模型时，不再要一个重复的 --bundle-vosk-model。
    bundled_vosk = _BUNDLE_ASSETS / "vosk" / _VOSK_MODEL
    model_expected = model is not None or bundled_vosk.is_dir()
    if args.release and not model_expected:
        raise SystemExit("--release 需要 --bundle-vosk-model（macOS 包的语音输入要开箱可用）")
    if args.release:
        missing = _bundle_missing()
        if missing:
            raise SystemExit(
                "发布版缺随包模型，先跑 uv run python tools/fetch_bundle_assets.py；缺："
                + "、".join(missing)
            )

    if not args.skip_build:
        _build_mcp_servers()

    icon = ICNS if ICNS.is_file() else None
    command = _build_command(model=model, icon=icon)
    if args.dry_run:
        print("$ " + " ".join(command))
        print(f"# 产物：{APP}")
        print(f"# 安装包：{_DIST / f'{APP_NAME}-{version}-macos-{arch}'}.zip / .dmg")
        return 0
    if sys.platform != "darwin":
        raise SystemExit(
            "PyInstaller 不能交叉编译：macOS 的 .app 只能在 macOS 上构建。\n"
            "本仓用 .github/workflows/macos-release.yml 在 macos-15 上构建。"
        )

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit("PyInstaller 不在当前环境：先 uv sync --all-extras") from None

    if not args.skip_build:
        icon = _make_icns()
        command = _build_command(model=model, icon=icon)
        _run(command, timeout=3600)
    if not APP.is_dir():
        raise SystemExit(f"没有产物：{APP}")

    _write_info_plist(APP, version)
    _sign(APP)
    problems = _check_bundle(APP, version, require_model=model_expected)
    if problems:
        raise SystemExit("产物检查不通过：\n- " + "\n- ".join(problems))
    _smoke(APP)
    artifacts = _package(version, arch, make_dmg=not args.no_dmg)
    print("\n完成：")
    for path in artifacts:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
