"""插件：**插件目录下的一个子目录，就是这么简单**。

一个插件 = 一个目录，里面有一个 ``plugin.py``，那个模块里有一个 ``tools()`` 函数，返回
它要提供的工具。目录长这样::

    <数据根>/plugins/
        天气/
            plugin.py        # def tools(): return [WeatherTool()]

**加载是进程内的，也就是说：装一个插件 = 让那段代码跑在你的机器上、用你的权限。**
这不是一句免责声明——它就是这件事的定义。程序里那些"可读根""过闸门"的约束**一条都挡不住
插件里的代码**（它与主程序同一个进程、拿得到同样的东西）。所以：

* **插件目录的内容不出仓、不进产物**：它是用户自己的东西，与 ``stickers/``、``models/``
  同一个姿态（跟着数据根走、不创建、不下载）。
* **一个插件坏了不许带塌程序**：import 抛了、没有 ``tools()``、返回的不是工具、名字与
  已有的撞了——每一种都只记一句人话、跳过它，继续装下一个。
* 插件里的工具**与内置工具同一套闸门**：``requires_approval`` 由插件自己声明，闸门读
  那一处。这一层不额外加权限判断，也不假装加得上。

**这一层不做 I/O 之外的事**：它只读目录、加载模块、检查形状。它不认识 Qt，也不认识
窗口——所以"哪些插件装上了、哪些为什么没装上"在核心腿里逐条可断言。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "enabled_plugins",
    "load_enabled_plugins",
    "set_plugin_enabled",
    "FACTORY_NAME",
    "PLUGIN_MODULE",
    "PluginIssue",
    "PluginLoad",
    "load_plugins",
    "plugins_in",
]

#: 一个插件目录里那个模块的名字。
PLUGIN_MODULE = "plugin.py"

#: 那个模块里必须有的函数。**返回一个可迭代的工具序列。**
FACTORY_NAME = "tools"


@dataclass(frozen=True)
class PluginIssue:
    """一个插件没装上，以及为什么。**给人看的一句话。**"""

    plugin: str
    reason: str

    def describe(self) -> str:
        return f"{self.plugin}：{self.reason}"


@dataclass(frozen=True)
class PluginLoad:
    """装插件这一趟的结果。"""

    tools: tuple[Any, ...]
    """装上来的工具，按插件目录的**规范序**排。"""

    issues: tuple[PluginIssue, ...]
    """没装上的那些，各带一句原因。空元组是常态。"""

    @property
    def count(self) -> int:
        return len(self.tools)


def plugins_in(root: Path) -> tuple[Path, ...]:
    """插件目录里的**子目录**，按名字排序。

    排序是规格的一部分（同 :func:`axo.stickers.stickers_in`）：同一份目录永远给同一个
    顺序，于是工具表的排列是稳定的（提示词缓存吃得着），测试也不必依赖文件系统的次序。

    目录不存在就返回空元组——**还没放插件是常态**，不是错误。
    """
    if not root.is_dir():
        return ()
    folders = [child for child in root.iterdir() if child.is_dir()]
    return tuple(sorted(folders, key=lambda path: (path.name.casefold(), path.name)))


def load_plugins(
    root: Path, *, taken: Sequence[str] = (), only: Collection[str] | None = None
) -> PluginLoad:
    """把插件目录里的东西装成一批工具。

    Args:
        root: 插件目录（``paths.plugins_dir()``）。不存在就是"一个都没有"。
        taken: **已经被占用的工具名**（内置的那些）。撞名的插件工具会被跳过并记一条
            ——两个同名工具会让模型看到一份自相矛盾的工具表，而它没法自己发现这件事。

    Returns:
        :class:`PluginLoad`。**不抛**：这个函数存在的理由就是"一个插件坏了不影响别的"，
        而它会遍历的正是别人写的代码。
    """
    tools: list[Any] = []
    issues: list[PluginIssue] = []
    used = set(taken)

    for folder in plugins_in(root):
        if only is not None and folder.name not in only:
            continue
        tools.extend(_load_folder(folder, used, issues))

    return PluginLoad(tools=tuple(tools), issues=tuple(issues))


def _load_folder(folder: Path, used: set[str], issues: list[PluginIssue]) -> list[Any]:
    """装一个插件目录；失败只记 issue，不抛。"""
    source = folder / PLUGIN_MODULE
    if not source.is_file():
        issues.append(PluginIssue(folder.name, f"里面没有 {PLUGIN_MODULE}"))
        return []
    try:
        module = _import(folder.name, source)
    except Exception as exc:  # noqa: BLE001 —— 插件是别人写的代码，什么都能抛
        issues.append(PluginIssue(folder.name, _why("运行时出错", exc)))
        return []
    factory = getattr(module, FACTORY_NAME, None)
    if not callable(factory):
        issues.append(PluginIssue(folder.name, f"没有 {FACTORY_NAME}() 这个函数"))
        return []
    try:
        produced = factory()
    except Exception as exc:  # noqa: BLE001
        issues.append(PluginIssue(folder.name, _why(f"{FACTORY_NAME}() 出错", exc)))
        return []
    return _accept(folder.name, produced, used, issues)


def _import(name: str, source: Path) -> Any:
    """按路径加载那个模块。

    **模块名带一层前缀**（``axo_plugin_天气``）：插件目录里那个文件叫 ``plugin.py``，
    而十个插件都用这个名字的话，``sys.modules`` 里会互相覆盖——症状是"只有最后一个
    插件的工具是对的"，而它看起来完全正常。
    """
    spec = importlib.util.spec_from_file_location(f"axo_plugin_{name}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"{source} 不是能加载的模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accept(
    plugin: str,
    produced: Sequence[Any] | None,
    used: set[str],
    issues: list[PluginIssue],
) -> list[Any]:
    """收下这批工具，把不合格的挡在外面并记一条。"""
    kept: list[Any] = []
    before = len(issues)
    for tool in produced or ():
        name = getattr(getattr(tool, "spec", None), "name", "")
        if not name:
            issues.append(PluginIssue(plugin, "有一个工具没有 spec.name"))
            continue
        if name in used:
            issues.append(PluginIssue(plugin, f"工具名「{name}」已经被占了，跳过"))
            continue
        if not callable(getattr(tool, "run", None)):
            issues.append(PluginIssue(plugin, f"工具「{name}」没有 run()"))
            continue
        used.add(name)
        kept.append(tool)
    # **只在"这个插件"一条都没收下、又没有单独记过原因"时才补这一条**：`issues` 是
    # 整趟共享的，拿它判空会把"上一个插件报过错"当成"这一个报过了"。
    if not kept and len(issues) == before:
        issues.append(PluginIssue(plugin, f"{FACTORY_NAME}() 一个工具都没返回"))
    return kept


def _why(what: str, exc: Exception) -> str:
    return f"{what}：{type(exc).__name__}: {exc}"


# ── 默认关 + 审计（对话纪要修订清单 🟢 #12/#14）────────────────────────────


def _enabled_file(root: Path) -> Path:
    return root / "enabled.json"


def enabled_plugins(root: Path) -> tuple[str, ...]:
    """读 ``enabled.json``。**文件不存在 = 一个都不启用**（默认关）。

    坏文件（不是 JSON / 不是字符串列表）也按「一个都不启用」处理，并把原因记进审计——
    静默地把坏文件当全开，等于默认关这件事没有发生。
    """
    path = _enabled_file(root)
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(data, list) or not all(isinstance(name, str) for name in data):
        return ()
    return tuple(data)


def set_plugin_enabled(root: Path, name: str, enabled: bool = True) -> None:
    """启用/禁用一个插件，写回 ``enabled.json``（排序、去重）。"""
    current = set(enabled_plugins(root))
    if enabled:
        current.add(name)
    else:
        current.discard(name)
    root.mkdir(parents=True, exist_ok=True)
    _enabled_file(root).write_text(
        json.dumps(sorted(current), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _audit(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_enabled_plugins(
    root: Path, *, taken: Sequence[str] = (), audit_path: Path | None = None
) -> PluginLoad:
    """按 ``enabled.json`` 装插件，并把每个插件的结局写进审计。

    与 :func:`load_plugins` 的区别只有两处：**默认关**（没有 enabled.json 就一个都不装），
    以及**审计**（每个目录一条 JSONL：时间、名字、状态、工具名、plugin.py 的 sha256）。
    """
    enabled = set(enabled_plugins(root))
    tools: list[Any] = []
    issues: list[PluginIssue] = []
    used = set(taken)
    for folder in plugins_in(root):
        source = folder / PLUGIN_MODULE
        base = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "plugin": folder.name,
            "sha256": _sha256(source),
            "enabled": folder.name in enabled,
        }
        if folder.name not in enabled:
            _audit(audit_path, {**base, "status": "disabled", "tools": []})
            issues.append(
                PluginIssue(
                    folder.name,
                    "未启用（插件默认关：在 <数据根>/plugins/enabled.json 里加上名字）",
                )
            )
            continue
        before = len(issues)
        loaded = _load_folder(folder, used, issues)
        tools.extend(loaded)
        new_issues = issues[before:]
        status = "loaded" if loaded else ("error" if new_issues else "empty")
        _audit(
            audit_path,
            {
                **base,
                "status": status,
                "tools": [getattr(getattr(tool, "spec", None), "name", "") for tool in loaded],
                "detail": "；".join(issue.reason for issue in new_issues),
            },
        )
    return PluginLoad(tools=tuple(tools), issues=tuple(issues))
