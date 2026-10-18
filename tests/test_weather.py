"""内置天气：解析、格式化、失败都写成人话。**不发网络请求。**"""

from __future__ import annotations

from typing import Any

import pytest

from axo.weather import (
    OpenMeteoWeather,
    WeatherError,
    WeatherReport,
    format_report,
    weather_label,
)

GEO = {
    "results": [
        {"name": "北京", "country": "中国", "admin1": "北京", "latitude": 39.9, "longitude": 116.4}
    ]
}
FORECAST = {
    "current": {
        "temperature_2m": 18.2,
        "apparent_temperature": 17.0,
        "relative_humidity_2m": 55,
        "wind_speed_10m": 12,
        "weather_code": 2,
    },
    "daily": {
        "weather_code": [3, 61],
        "temperature_2m_max": [21, 19],
        "temperature_2m_min": [12, 11],
        "precipitation_probability_max": [10, 70],
    },
}


def test_weather_codes_become_chinese() -> None:
    assert weather_label(0) == "晴"
    assert weather_label(61) == "小雨"
    assert weather_label(999) == "说不清的天气"
    assert weather_label(None) == "说不清的天气"


def test_the_report_reads_like_a_sentence_and_keeps_missing_fields_out() -> None:
    report = WeatherReport(
        place="北京",
        country="中国",
        admin="北京",
        latitude=39.9,
        longitude=116.4,
        current_temp=18.2,
        feels_like=None,
        humidity=55,
        wind=None,
        condition="多云",
        days=({"condition": "阴", "high": 21, "low": 12, "rain": 10},),
    )

    text = format_report(report)

    assert "北京（中国，北京）" in text
    assert "现在 18.2°C" in text
    assert "体感" not in text
    assert "湿度 55%" in text
    assert "风速" not in text
    assert "今天：12~21°C，阴，降水概率 10%。" in text


def test_lookup_does_two_requests_and_builds_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    weather = OpenMeteoWeather()
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, params: dict[str, Any], *, what: str) -> dict[str, Any]:
        seen.append((url, params))
        return GEO if "geocoding" in url else FORECAST

    weather._get = fake_get  # type: ignore[method-assign]

    report = weather.lookup("北京", days=2)

    assert report.place == "北京"
    assert report.current_temp == 18.2
    assert report.condition == "多云"
    assert len(report.days) == 2
    assert len(seen) == 2
    assert seen[1][1]["forecast_days"] == 2


def test_an_unknown_city_is_a_sentence() -> None:
    weather = OpenMeteoWeather()
    weather._get = lambda *_a, **_k: {"results": []}  # type: ignore[method-assign]

    with pytest.raises(WeatherError, match="没找到"):
        weather.lookup("不存在的地方")


def test_an_empty_location_is_refused() -> None:
    with pytest.raises(WeatherError, match="城市"):
        OpenMeteoWeather().lookup("   ")
