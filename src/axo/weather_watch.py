"""天气基线：**记住上一次看到的天气，才能说"变了"**。

这一层不进记忆库，也不进模型上下文——它是本机的一个小状态文件（数据根下的
``weather_baseline.json``）。主动回合每查一次天气：

* 没有基线 → 只记下来，不算"变化"（第一次看到不是变化）；
* 与基线比出了变化 → 把变化写进话头，让她决定要不要提醒；
* 没变化 → 基线推进到这一次，避免下次还把同一件旧事当新闻。

**只有真正说出口时才算"提醒过"**：调用方在模型说了话之后才 :meth:`WeatherBaseline.put`；
模型选择沉默时基线不动，下一次还会看见同一个变化（而不是悄悄吞掉）。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .weather import WeatherReport

__all__ = [
    "TEMP_CHANGE_C",
    "RAIN_CHANGE_PP",
    "WeatherBaseline",
    "WeatherSnapshot",
    "WeatherWatchResult",
    "changed_lines",
    "material_text",
    "snapshot_from_report",
    "summary_line",
]

#: 气温变化多少度才值得提。5°C 是"出门要加件衣服"的量级，不是每度都报。
TEMP_CHANGE_C = 5.0

#: 降水概率变化多少个百分点才值得提。
RAIN_CHANGE_PP = 40.0


@dataclass(frozen=True)
class WeatherSnapshot:
    city: str
    observed_at: float
    condition: str
    temp: float | None
    high: float | None
    low: float | None
    rain: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> WeatherSnapshot | None:
        try:
            return cls(
                city=str(row["city"]),
                observed_at=float(row["observed_at"]),
                condition=str(row.get("condition") or ""),
                temp=_number(row.get("temp")),
                high=_number(row.get("high")),
                low=_number(row.get("low")),
                rain=_number(row.get("rain")),
            )
        except (KeyError, TypeError, ValueError):
            return None


def snapshot_from_report(report: WeatherReport, *, city: str, now: float) -> WeatherSnapshot:
    """把一次查询压成可比较的一行。第一天就是"今天"。"""
    today = report.days[0] if report.days else {}
    return WeatherSnapshot(
        city=city.strip(),
        observed_at=now,
        condition=report.condition,
        temp=report.current_temp,
        high=today.get("high"),
        low=today.get("low"),
        rain=today.get("rain"),
    )


def summary_line(snapshot: WeatherSnapshot) -> str:
    """给话头用的一句当前天气。"""
    parts = [f"{snapshot.city}：现在"]
    if snapshot.temp is not None:
        parts.append(f" {snapshot.temp:g}°C")
    if snapshot.condition:
        parts.append(f"，{snapshot.condition}")
    if snapshot.low is not None and snapshot.high is not None:
        parts.append(f"；今天 {snapshot.low:g}~{snapshot.high:g}°C")
    if snapshot.rain is not None:
        parts.append(f"，降水概率 {snapshot.rain:g}%")
    return "".join(parts) + "。"


@dataclass(frozen=True)
class WeatherWatchResult:
    """一次天气检查的收成：当前快照 + 与基线相比值得提的变化。"""

    snapshot: WeatherSnapshot
    changes: tuple[str, ...] = ()

    @property
    def noteworthy(self) -> bool:
        return bool(self.changes)

    def describe(self) -> str:
        return material_text(self)


def material_text(result: WeatherWatchResult) -> str:
    """进话头的一段话：当前天气 + 变化（如果有）。"""
    text = summary_line(result.snapshot)
    if result.changes:
        text += "变化：" + "；".join(result.changes) + "。"
    return text


def changed_lines(previous: WeatherSnapshot | None, current: WeatherSnapshot) -> tuple[str, ...]:
    """与上一次相比，**值得提**的变化。没有基线或没有变化 → 空元组。"""
    if previous is None:
        return ()
    lines: list[str] = []
    if previous.condition and current.condition and previous.condition != current.condition:
        lines.append(f"天气从「{previous.condition}」变成「{current.condition}」")
    if previous.temp is not None and current.temp is not None:
        delta = current.temp - previous.temp
        if abs(delta) >= TEMP_CHANGE_C:
            direction = "降" if delta < 0 else "升"
            lines.append(
                f"气温{direction}了 {abs(delta):g}°C（{previous.temp:g} → {current.temp:g}°C）"
            )
    if previous.rain is not None and current.rain is not None:
        delta = current.rain - previous.rain
        if abs(delta) >= RAIN_CHANGE_PP:
            direction = "升" if delta > 0 else "降"
            lines.append(
                f"降水概率{direction}了 {abs(delta):g} 个百分点"
                f"（{previous.rain:g}% → {current.rain:g}%）"
            )
    if previous.high is not None and current.high is not None:
        delta = current.high - previous.high
        if abs(delta) >= TEMP_CHANGE_C and not any("气温" in line for line in lines):
            direction = "降" if delta < 0 else "升"
            lines.append(f"最高温{direction}了 {abs(delta):g}°C")
    return tuple(lines)


class WeatherBaseline:
    """数据根下的一份 JSON：``{城市小写: 快照}``。**坏了就当空的。**"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, WeatherSnapshot]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        snapshots: dict[str, WeatherSnapshot] = {}
        for key, row in raw.items():
            if isinstance(row, dict):
                snapshot = WeatherSnapshot.from_dict(row)
                if snapshot is not None:
                    snapshots[str(key)] = snapshot
        return snapshots

    def get(self, city: str) -> WeatherSnapshot | None:
        return self.load().get(city.strip().casefold())

    def put(self, snapshot: WeatherSnapshot) -> None:
        """原子地写回（同目录临时文件 + ``os.replace``）。失败**不抛**。"""
        data = self.load()
        data[snapshot.city.strip().casefold()] = snapshot
        payload = {key: value.to_dict() for key, value in data.items()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".weather-", suffix=".tmp"
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            Path(temporary).replace(self.path)
        except OSError:
            return


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
