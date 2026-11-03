"""六幕启动序列：**纯数据**，渲染与交互在 :mod:`axo.ui.splash`。

## 为什么单独一个模块

幕表要能被校验、被测试、将来还能被用户替换（``data_dir()/boot/sequence.json``），
而它**不该携带任何可执行内容**。DSH 的启动插件里有一条 ``engine:"iframe"`` 的
代码模板路径——模板目录里的 HTML/JS 会被执行，上游连 sandbox 都没加。Axo 不做
那条路：``BootAct`` 只有白名单字段，JSON 里的未知键在加载时就被拒绝。

## 与"开屏"的分工

开屏（``SplashScreen``）负责"让程序看起来在醒来"；本模块负责**六幕的叙事与
节奏**。幕表播完就回到开屏的呼吸/加载状态，主窗口构造期间它仍在那里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BOOT_ACTS",
    "MAX_ACTS",
    "BootAct",
    "BootSequence",
    "builtin_sequence",
    "load_boot_sequence",
]

#: 与 DSH 的上限同值：再多就不是"启动"了，是一段短片。
MAX_ACTS = 24

#: 每幕可选的绘制版式。**白名单**：未知值在加载时拒绝，不做"猜一个最像的"。
KINDS = ("fade", "type", "glass", "glitch", "hero")


@dataclass(frozen=True)
class BootAct:
    """一幕。

    Attributes:
        id: 稳定标识（报告与测试用）。
        title: 中文主字；空串表示只显示副标题。
        subtitle: 英文等宽副标题；空串表示不显示。
        duration_ms: 自动推进前的停留（``confirm`` 幕是"等待确认"的最短时间）。
        confirm: 是否等一次 Enter/空格/点击；**最多等 5 秒**后自动前进。
        kind: 绘制版式，取值见 :data:`KINDS`。
    """

    id: str
    title: str = ""
    subtitle: str = ""
    duration_ms: int = 1100
    confirm: bool = False
    kind: str = "fade"


@dataclass(frozen=True)
class BootSequence:
    """一组幕。``acts`` 至少一幕。"""

    acts: tuple[BootAct, ...]

    def __post_init__(self) -> None:
        if not self.acts:
            raise ValueError("启动序列至少需要一幕。")
        if len(self.acts) > MAX_ACTS:
            raise ValueError(f"启动序列最多 {MAX_ACTS} 幕，收到 {len(self.acts)}。")


#: 内置六幕。文案自写，不含任何第三方台词；与 DSH 的六幕是"结构参照"不是复制。
BOOT_ACTS: tuple[BootAct, ...] = (
    BootAct("wake", "正在醒来", "AXO / MEMORY TERMINAL", 900, kind="fade"),
    BootAct(
        "identity",
        "身份核验",
        "IDENTITY · QIANSUI",
        1200,
        confirm=True,
        kind="type",
    ),
    BootAct(
        "archive",
        "读取记忆库",
        "ARCHIVE · LOCAL SQLITE",
        1100,
        kind="glass",
    ),
    BootAct(
        "permission",
        "记忆只属于这台机器",
        "PERMISSION · LOCAL ONLY",
        1200,
        confirm=True,
        kind="type",
    ),
    BootAct("unveil", "揭幕", "UNVEIL", 900, kind="glitch"),
    BootAct("welcome", "欢迎回来", "WELCOME BACK", 1100, kind="hero"),
)


def builtin_sequence() -> BootSequence:
    return BootSequence(BOOT_ACTS)


_FIELDS = {"id", "title", "subtitle", "duration_ms", "confirm", "kind"}


def load_boot_sequence(path: Path | None = None) -> BootSequence:
    """从 JSON 读幕表；**任何一处不合格都整体回退内置幕表**。

    回退而不是报错：启动装饰不该让程序起不来。坏文件由调用方记日志。
    """
    if path is None or not path.is_file():
        return builtin_sequence()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return builtin_sequence()
    rows = raw.get("acts") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows:
        return builtin_sequence()
    acts: list[BootAct] = []
    for row in rows:
        if not isinstance(row, dict):
            return builtin_sequence()
        if set(row) - _FIELDS:
            return builtin_sequence()
        kind = str(row.get("kind", "fade"))
        if kind not in KINDS:
            return builtin_sequence()
        try:
            duration = int(row.get("duration_ms", 1100))
        except (TypeError, ValueError):
            return builtin_sequence()
        if duration <= 0 or duration > 30_000:
            return builtin_sequence()
        acts.append(
            BootAct(
                id=str(row.get("id") or f"act-{len(acts) + 1}"),
                title=str(row.get("title", "")),
                subtitle=str(row.get("subtitle", "")),
                duration_ms=duration,
                confirm=bool(row.get("confirm", False)),
                kind=kind,
            )
        )
    try:
        return BootSequence(tuple(acts))
    except ValueError:
        return builtin_sequence()
