"""内置天气：**一个城市名进，一段人话出**。

用 Open-Meteo：不要 key、两步（先按名字找坐标，再取预报）。这里不引入第三方
天气库——两段 JSON、几个字段，手写解析比多一个依赖便宜，也更容易在测试里用假
响应钉住。

**放在 ``axo/`` 而不是 ``axo/agent/``**：后者是纯层，网络请求不能出现在那里
（同 ``axo/net.py`` 的理由）。``httpx`` 在方法里 import，模块导入本身不联网。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_DAYS",
    "MAX_DAYS",
    "OpenMeteoWeather",
    "WeatherError",
    "WeatherReport",
    "format_report",
    "weather_label",
]

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_DAYS = 1
MAX_DAYS = 7

_TIMEOUT = 15.0

#: WMO 天气代码 → 中文。没列到的一律"说不清的天气"——**不猜**。
_LABELS: dict[int, str] = {
    0: "晴",
    1: "大致晴",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴小冰雹",
    99: "雷阵雨伴大冰雹",
}


class WeatherError(RuntimeError):
    """查不到。消息是**写给模型看的**——它会被原样回灌。"""


@dataclass(frozen=True)
class WeatherReport:
    """一次查询的收成。字段都可能是 ``None``（服务端没给那一项）。"""

    place: str
    country: str
    admin: str
    latitude: float
    longitude: float
    current_temp: float | None
    feels_like: float | None
    humidity: float | None
    wind: float | None
    condition: str
    days: tuple[dict[str, Any], ...] = ()


def weather_label(code: Any) -> str:
    """WMO 代码 → 中文。读不出来就是"说不清的天气"。"""
    try:
        return _LABELS.get(int(code), "说不清的天气")
    except (TypeError, ValueError):
        return "说不清的天气"


def format_report(report: WeatherReport) -> str:
    """拼成给模型看的一段话。**没有的字段不编**。"""
    where = "，".join(part for part in (report.country, report.admin) if part)
    head = f"{report.place}" + (f"（{where}）" if where else "")
    parts: list[str] = []
    if report.current_temp is not None:
        parts.append(f"现在 {report.current_temp:g}°C")
    if report.feels_like is not None:
        parts.append(f"体感 {report.feels_like:g}°C")
    parts.append(report.condition)
    if report.humidity is not None:
        parts.append(f"湿度 {report.humidity:g}%")
    if report.wind is not None:
        parts.append(f"风速 {report.wind:g} km/h")
    lines = [head + "：" + "，".join(parts) + "。"]
    for index, day in enumerate(report.days):
        label = "今天" if index == 0 else "明天" if index == 1 else f"{index + 1} 天后"
        low = day.get("low")
        high = day.get("high")
        span = f"{low:g}~{high:g}°C" if low is not None and high is not None else ""
        parts = [part for part in (span, day.get("condition", "")) if part]
        if day.get("rain") is not None:
            parts.append(f"降水概率 {day['rain']:g}%")
        if parts:
            lines.append(f"{label}：" + "，".join(parts) + "。")
    return "\n".join(lines)


class OpenMeteoWeather:
    """Open-Meteo 的薄封装。**不抛网络异常**，统一成 :class:`WeatherError`。"""

    def __init__(self, *, timeout: float = _TIMEOUT) -> None:
        self.timeout = timeout

    def lookup(self, location: str, *, days: int = DEFAULT_DAYS) -> WeatherReport:
        name = location.strip()
        if not name:
            raise WeatherError("location 是空的，请给出城市名。")
        days = max(1, min(MAX_DAYS, int(days)))
        place = self._geocode(name)
        data = self._get(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": days,
            },
            what="天气预报",
        )
        return _report_from(place, data)

    def _geocode(self, name: str) -> dict[str, Any]:
        data = self._get(
            GEOCODE_URL,
            {"name": name, "count": 1, "language": "zh", "format": "json"},
            what="找城市",
        )
        results = data.get("results") or []
        if not results or not isinstance(results[0], dict):
            raise WeatherError(f"没找到「{name}」这个城市。请换一个更常见的名字。")
        return results[0]

    def _get(self, url: str, params: dict[str, Any], *, what: str) -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params)
        except httpx.RequestError as exc:
            raise WeatherError(f"{what}失败：连不上天气服务（{type(exc).__name__}）。") from exc
        if not response.is_success:
            raise WeatherError(f"{what}失败：HTTP {response.status_code}。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherError(f"{what}返回的不是 JSON。") from exc
        if not isinstance(payload, dict):
            raise WeatherError(f"{what}返回的形状不认识。")
        return payload


def _report_from(place: dict[str, Any], data: dict[str, Any]) -> WeatherReport:
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    days: list[dict[str, Any]] = []
    codes = daily.get("weather_code") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rains = daily.get("precipitation_probability_max") or []
    for index in range(min(len(codes), len(highs), len(lows))):
        days.append(
            {
                "condition": weather_label(codes[index]),
                "high": _number(highs[index]),
                "low": _number(lows[index]),
                "rain": _number(rains[index]) if index < len(rains) else None,
            }
        )
    return WeatherReport(
        place=str(place.get("name") or "").strip(),
        country=str(place.get("country") or "").strip(),
        admin=str(place.get("admin1") or "").strip(),
        latitude=float(place.get("latitude") or 0.0),
        longitude=float(place.get("longitude") or 0.0),
        current_temp=_number(current.get("temperature_2m")),
        feels_like=_number(current.get("apparent_temperature")),
        humidity=_number(current.get("relative_humidity_2m")),
        wind=_number(current.get("wind_speed_10m")),
        condition=weather_label(current.get("weather_code")),
        days=tuple(days),
    )


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
