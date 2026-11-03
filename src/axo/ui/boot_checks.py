"""启动时的真实自检：六幕里显示的是这一台机器刚跑出来的结果。

**为什么不是装饰**：之前的六幕是一段按进度打勾的动画——它永远全绿，用户看两次
就不信了。这里每一项都真的去碰一下：数据目录写一个临时文件、记忆库**只读**打开
数一遍、人格锚数一遍、后端按配置判"能不能用"、工具沙箱看授权了几条。

**全部是本地、快速的检查**：
- 唯一的网络动作是对 Ollama 做一次 **TCP 连接**（0.6 秒上限，只问端口在不在听），
  不发 HTTP 请求、不等模型加载；
- 打不开的库报 ✗ 而不是抛——启动装饰不该让程序起不来；
- 只读打开：检查不改库、不建表、不迁移。

结论给 :class:`axo.ui.splash.SplashScreen` 的档案面板用；纯层，不 import Qt。
"""

from __future__ import annotations

import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["BootCheck", "collect_checks"]


@dataclass(frozen=True)
class BootCheck:
    """一项检测。

    Attributes:
        key: ``data`` / ``archive`` / ``persona`` / ``identity`` / ``password`` /
            ``backend`` / ``scope`` / ``tools``。
        label: 面板上的中文名。
        ok: 通过与否。**首次启动的"还没有库"不算失败**；打不开才算。
        detail: 一行事实（条数、路径、模型名……）。
    """

    key: str
    label: str
    ok: bool
    detail: str


def _data_dir_check() -> BootCheck:
    from ..paths import data_dir

    root = data_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".boot-probe-", dir=root)
        os.close(handle)
        Path(name).unlink(missing_ok=True)
        return BootCheck("data", "数据目录", True, str(root))
    except OSError as exc:
        return BootCheck("data", "数据目录", False, f"不可写：{exc}")


def _memory_checks() -> tuple[BootCheck, BootCheck]:
    from ..paths import memory_db_path

    path = memory_db_path()
    if not path.is_file():
        return (
            BootCheck("archive", "记忆库", True, "首次启动，将新建"),
            BootCheck("persona", "人格锚", True, "首次启动，会问你"),
        )
    try:
        from memcore.store import MemoryStore

        store = MemoryStore(path, read_only=True)
    except Exception as exc:  # noqa: BLE001 —— 打不开要说出来，不能抛
        return (
            BootCheck("archive", "记忆库", False, f"打不开：{type(exc).__name__}"),
            BootCheck("persona", "人格锚", False, "库打不开，未检查"),
        )
    try:
        count = store.count()
        from ..memory.persona import persona_texts

        anchors = len(persona_texts(store))
        persona_ok = anchors > 0
        return (
            BootCheck("archive", "记忆库", True, f"{count} 条"),
            BootCheck(
                "persona",
                "人格锚",
                True,
                f"{anchors} 条" if persona_ok else "还没有，会问你",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            BootCheck("archive", "记忆库", False, f"读取失败：{type(exc).__name__}"),
            BootCheck("persona", "人格锚", False, "读取失败"),
        )
    finally:
        store.close()


def _probe_ollama(host: str, timeout: float = 0.6) -> tuple[bool, str]:
    """对 Ollama 做一次 **TCP 连接**，不是 HTTP 请求：只回答"端口在不在听"。

    这一项是启动路径上唯一的网络动作，所以给 0.6 秒上限；连不上也只是一行 ✗，
    启动照常往下走（主窗口会给出真正的错误）。空 host 按默认本机地址算。
    """
    raw = (host or "http://127.0.0.1:11434").strip()
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    name = parts.hostname or "127.0.0.1"
    port = parts.port or 11434
    try:
        with socket.create_connection((name, port), timeout=timeout):
            return True, f"{name}:{port}"
    except OSError:
        return False, f"{name}:{port} 未响应"


def _backend_check(backend) -> BootCheck:
    provider = getattr(backend, "provider", "")
    if provider == "builtin":
        model = Path(str(getattr(backend, "local_model_path", "") or ""))
        if not model.is_file():
            return BootCheck("backend", "本地模型", False, "还没选 .gguf 模型")
        # 与 ``llm.llama_server.find_llama_server`` 查的是同一个东西；这里只碰
        # ``paths``，不 import 那个模块（它会拖进 httpx，而自检在启动路径上）。
        from ..paths import bundled_llama_dir

        runtime = bundled_llama_dir()
        if runtime is None or not (runtime / "llama-server.exe").is_file():
            return BootCheck("backend", "本地模型", False, "运行时缺失（重装可补）")
        return BootCheck("backend", "本地模型", True, f"{model.name}（随包 llama.cpp）")
    if provider == "cloud":
        key = str(getattr(backend, "api_key", "") or "")
        base = str(getattr(backend, "cloud_base_url", "") or "")
        return BootCheck(
            "backend",
            "云端后端",
            bool(key),
            "key 已配置（未探测）" if key else f"未填 key（{base}）",
        )
    host = str(getattr(backend, "host", "") or "")
    alive, detail = _probe_ollama(host)
    return BootCheck("backend", "本机后端", alive, f"Ollama {detail}")


def _scope_check(backend) -> BootCheck:
    """数据边界：这一台机器上的东西会不会出去。

    它只陈述**配置事实**，不替用户做判断：云端后端填了 key 就是"可以发往云端"，
    即便这一轮其实没发。这条放在「权限核验」幕里，替代一句写死的 ``LOCAL ONLY``。
    """
    provider = getattr(backend, "provider", "")
    if provider == "cloud":
        key = str(getattr(backend, "api_key", "") or "")
        if key:
            return BootCheck("scope", "数据边界", True, "云端已启用，对话按需发送")
        return BootCheck("scope", "数据边界", True, "云端未配置，仅本机")
    host = str(getattr(backend, "host", "") or "")
    parts = urlsplit(host if "://" in host else "http://" + host)
    name = (parts.hostname or "127.0.0.1").lower()
    if name not in ("127.0.0.1", "localhost", "::1"):
        return BootCheck("scope", "数据边界", True, f"Ollama 在非本机（{name}）")
    return BootCheck("scope", "数据边界", True, "仅本机")


def _tools_check(tools) -> BootCheck:
    roots = getattr(tools, "read_roots", ())
    writable = getattr(tools, "write_roots", ())
    missing = sum(1 for root in (*roots, *writable) if not Path(root).is_dir())
    flags = []
    if bool(getattr(tools, "allow_commands", False)):
        flags.append("可执行命令")
    if bool(getattr(tools, "web_enabled", False)):
        flags.append("可联网")
    tail = " · " + " · ".join(flags) if flags else ""
    if not roots and not writable:
        detail = "未授权目录（文件工具不登记）" + tail
        ok = not missing
    else:
        detail = f"{len(roots)} 个只读 / {len(writable)} 个可写{tail}"
        ok = missing == 0
        if missing:
            detail = f"{detail}（{missing} 个目录不存在）"
    return BootCheck("tools", "工具沙箱", ok, detail)


def _identity_check(device) -> BootCheck:
    """设备一致性；**只警告不拦截**。不显示原始机器标识。"""
    from ..device_identity import compare_device

    state, detail = compare_device(device)
    return BootCheck("identity", "设备识别", state != "mismatch", detail)


def _password_check(security) -> BootCheck:
    """只验 verifier 记录格式，不验任何候选密码；真实密码由启动闸门输入。"""
    from ..passwords import password_record_ok

    record = getattr(security, "password_verifier", "")
    if not record:
        return BootCheck("password", "启动口令", False, "未设置，将进入设置流程")
    if not password_record_ok(record):
        return BootCheck("password", "启动口令", False, "记录损坏，请用恢复码重设")
    return BootCheck("password", "启动口令", True, "已设置（PBKDF2-SHA256）")


def _run(factory, key: str, label: str) -> BootCheck:
    """跑一项检测；**单项坏掉只影响这一项**，不把整张面板清空。"""
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001
        return BootCheck(key, label, False, f"检测失败：{type(exc).__name__}")


def collect_checks(backend=None, tools=None, security=None, device=None) -> tuple[BootCheck, ...]:
    """跑一遍启动自检。**任何一项失败都不抛**，只把 ok 置假。

    顺序是刻意的：与内置六幕的推进同序，启动账每幕多亮一行。
    """
    checks = [_data_dir_check()]
    checks.extend(_memory_checks())
    if device is not None:
        checks.append(_run(lambda: _identity_check(device), "identity", "设备识别"))
    if security is not None:
        checks.append(_run(lambda: _password_check(security), "password", "启动口令"))
    if backend is not None:
        checks.append(_run(lambda: _backend_check(backend), "backend", "后端"))
        checks.append(_run(lambda: _scope_check(backend), "scope", "数据边界"))
    if tools is not None:
        checks.append(_run(lambda: _tools_check(tools), "tools", "工具沙箱"))
    return tuple(checks)
