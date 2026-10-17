"""发版：**一条命令走完手工做过三遍的那五步**。

手动发版长这样（2026-10-15/16 我连着做了三遍）：

1. 改 ``pyproject.toml`` 的版本号
2. ``git commit`` —— **会被 ``memcore-layering`` 钩子拦一次**：那个钩子会重建工程并同步
   ``uv.lock``，而"文件被钩子改了"会让提交失败。所以这一层**带一次自动重试**。
3. ``tools/build_exe.py --release``（约十分钟，含活体探测与发布闸门）
4. 取 zip 的大小与 sha256、跑一遍**分项扫描**（按类别报数，不是"0 命中"一句话）
5. ``gh release create``

这一支把五步收成一条命令，并把第 4 步的扫描从仓外搬进来（``--scan-only`` 也能单跑）。

**它不替你决定版本号与说明。** 版本号要显式给（``--version``），说明要显式给
（``--notes``，一个 Markdown 文件）——那两样是产品判断，不是脚本该猜的。

用法::

    python tools/release.py --check                  # 只看能不能发（不写任何东西）
    python tools/release.py --version 0.6.0 --notes release.md
    python tools/release.py --scan-only              # 只跑分项扫描（扫 dist/ 里现成的 zip）
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PYPROJECT = ROOT / "pyproject.toml"

#: 只扫这些后缀——二进制里搜字符串既慢又没意义（要查二进制得换工具）。
TEXT_EXT = (
    ".py",
    ".md",
    ".txt",
    ".ini",
    ".json",
    ".jsonl",
    ".spec",
    ".yml",
    ".yaml",
    ".toml",
    ".html",
    ".js",
    ".css",
    ".bat",
    ".cfg",
    ".conf",
)

#: 分项扫描的四类。**按类别报数**：只报"那几个路径 0 命中"回答的不是"有没有我的东西"。
CATEGORIES: dict[str, tuple[bytes, ...]] = {
    "本机路径（用户目录/项目盘/数据根）": (
        b"C:" + bytes([92]) + b"Users" + bytes([92]),
        b"D:" + bytes([92]) + b"Projects" + bytes([92]),
        b"D:" + bytes([92]) + b"AxoData",
        "离中梦蝶".encode(),
        b"AppData" + bytes([92]) + b"Roaming",
    ),
    "真名/身份（可指到人的字符串）": ("离中梦蝶".encode(), b"PERSONAL_DENYLIST"),
    "密钥与凭据": (b"sk-", b"api_key", b"api-key", b"Bearer ", b"DPAPI", b"dpapi"),
    "使用者数据（记忆/会话/配置）": (
        b"memory_id",
        b"sessions" + bytes([92]),
        b"axo.ini",
        b".jsonl",
        b"mem.db",
        b"persona_anchor",
    ),
}


def run(command: list[str], *, cwd: Path = ROOT, check: bool = True) -> int:
    print("$ " + " ".join(command), flush=True)
    code = subprocess.call(command, cwd=str(cwd))
    if check and code != 0:
        raise SystemExit(f"这一步失败了（退出码 {code}）：{' '.join(command)}")
    return code


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8"
    )
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败：{result.stderr.strip()}")
    return result.stdout.strip()


def current_version() -> str:
    import re as _re

    match = _re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), _re.M)
    if not match:
        raise SystemExit("pyproject.toml 里找不到 version。")
    return match.group(1)


# ── 预检 ────────────────────────────────────────────────────────────────────


def preflight() -> None:
    """**发版前必须成立的四件事。** 每一条都对应一次真实的失败。"""
    problems: list[str] = []

    if git("status", "--porcelain"):
        problems.append("工作树不干净——发版要打的是已提交的那棵树")

    running = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-Process axo -ErrorAction SilentlyContinue | Measure-Object).Count",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if running.stdout.strip() not in ("", "0"):
        problems.append("还有 axo.exe 在跑——构建会删 dist/，它锁着里面的文件")

    if not (ROOT / "THIRD-PARTY-NOTICES.md").is_file():
        problems.append("缺 THIRD-PARTY-NOTICES.md——发布闸门要对账")

    if problems:
        raise SystemExit("发版前有问题：\n" + "\n".join("· " + one for one in problems))
    print(f"预检通过：工作树干净、没有在跑的 axo、当前版本 {current_version()}")


# ── 分项扫描 ────────────────────────────────────────────────────────────────


def scan(zip_path: Path) -> None:
    """按类别报数。**与构建自带的那道闸门各自独立**：那边验"声明与产物对得上"，
    这边验"里面有没有不该出去的东西"。"""
    print(f"\n=== 发布包：{zip_path.name} ===")
    archive = zipfile.ZipFile(zip_path)
    blobs = [
        (info.filename, archive.read(info))
        for info in archive.infolist()
        if info.filename.lower().endswith(TEXT_EXT) and info.file_size < 2_000_000
    ]
    _scan_blobs(f"zip 内文本文件（{len(blobs)} 个，其余为二进制）", blobs)

    tracked = git("ls-files").splitlines()
    repo = [
        (rel, (ROOT / rel).read_bytes())
        for rel in tracked
        if (ROOT / rel).suffix.lower() in TEXT_EXT
        and (ROOT / rel).is_file()
        and (ROOT / rel).stat().st_size < 2_000_000
    ]
    _scan_blobs(f"仓内 git 跟踪的文本文件（{len(repo)} 个）", repo)


def _scan_blobs(label: str, blobs: list[tuple[str, bytes]]) -> None:
    print(f"\n--- {label} ---")
    for category, needles in CATEGORIES.items():
        total = 0
        samples: list[str] = []
        for name, data in blobs:
            for needle in needles:
                count = data.count(needle)
                if not count:
                    continue
                total += count
                if len(samples) < 3:
                    at = data.find(needle)
                    context = data[max(0, at - 40) : at + 50].decode("utf-8", "replace")
                    samples.append(f"{name[:46]}: …{context.replace(chr(10), ' ')[:100]}…")
        print(f"  [{category}] {'—' if total == 0 else '有命中'}，共 {total} 处")
        for sample in samples:
            print(f"      {sample}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── 主流程 ──────────────────────────────────────────────────────────────────


def bump(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    revised = re.sub(r'^version = "[^"]+"', f'version = "{version}"', text, count=1, flags=re.M)
    if revised == text:
        raise SystemExit(f"pyproject.toml 里的版本已经是 {version}？")
    PYPROJECT.write_text(revised, encoding="utf-8", newline="\n")
    print(f"版本 {current_version()} → {version}")


def commit_version(version: str) -> None:
    git("add", "-A")
    code = subprocess.call(["git", "commit", "-m", f"版本 {version}"], cwd=str(ROOT))
    if code != 0:
        # **钩子改过文件**（``memcore-layering`` 会重建工程并同步 ``uv.lock``）。
        # 那次提交会因此失败，而改动已经在工作树里——再 add 一次、再提交一次。
        print("第一次提交被钩子拦下（它改了 uv.lock）——补上再提一次")
        git("add", "-A")
        git("commit", "-m", f"版本 {version}")


def push_main(attempts: int = 5) -> None:
    """**先推 main，再建 Release。**

    ``gh release create`` 在远端没有这个 tag 时，是按**默认分支的远端 HEAD** 建
    标签的——版本号那次提交还留在本地的话，标签就指向上一个提交，而"发出去的
    版本"与"仓里那个版本的源码"从此对不上。推送是间歇性的（同
    ``github-push-network-flakiness`` 那条），所以重试几轮再放弃。
    """
    for attempt in range(1, attempts + 1):
        if subprocess.call(["git", "push", "origin", "main"], cwd=str(ROOT)) == 0:
            print("main 已推送。")
            return
        print(f"推送失败（第 {attempt} 次），重试…", flush=True)
        time.sleep(3)
    raise SystemExit("推送 main 失败。**别继续发版**——标签会指向旧的提交。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只做预检，不写任何东西")
    parser.add_argument("--scan-only", action="store_true", help="只跑分项扫描")
    parser.add_argument("--version", help="新版本号，如 0.6.0")
    parser.add_argument("--title", help="Release 标题（默认 v<版本>）")
    parser.add_argument("--notes", type=Path, help="Release 说明（Markdown 文件）")
    parser.add_argument("--zip", dest="zip_file", type=Path, help="指定 zip（默认按版本号找）")
    args = parser.parse_args(argv)

    if args.scan_only:
        target = args.zip_file or _latest_zip()
        if target is None:
            raise SystemExit("dist/ 里没有 zip。")
        scan(target)
        return 0

    preflight()
    if args.check:
        return 0

    if not args.version:
        raise SystemExit("要发版就给 --version（版本号是产品判断，脚本不猜）")
    if not args.notes or not args.notes.is_file():
        raise SystemExit("要发版就给 --notes（一个 Markdown 文件）")

    bump(args.version)
    commit_version(args.version)
    push_main()
    run([sys.executable, "tools/build_exe.py", "--release"])
    # 提交与构建之后再看工作树：构建会往 dist/ 写东西，那不是"没提交"。
    zip_path = args.zip_file or DIST / f"axo-{args.version}-win64.zip"
    if not zip_path.is_file():
        raise SystemExit(f"构建没产出 {zip_path}")
    size = zip_path.stat().st_size
    digest = sha256_of(zip_path)
    print(f"\n产物：{zip_path.name}  {size:,} 字节  sha256 {digest}")

    scan(zip_path)

    title = args.title or f"v{args.version}"
    run(
        [
            "gh",
            "release",
            "create",
            f"v{args.version}",
            str(zip_path),
            "--title",
            title,
            "--notes-file",
            str(args.notes),
        ]
    )
    print(f"\n发完了：https://github.com/lzmd-arch/Axo/releases/tag/v{args.version}")
    print("**别忘了**：说明里的 sha256 与大小要换成上面那两个数（脚本不替你改文件）。")
    return 0


def _latest_zip() -> Path | None:
    found = sorted(DIST.glob("axo-*-win64.zip"), key=lambda path: path.stat().st_mtime)
    return found[-1] if found else None


if __name__ == "__main__":
    raise SystemExit(main())
