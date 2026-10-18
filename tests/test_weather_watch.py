"""天气基线：什么算"变了"、怎么存、第一次看到不算变化。"""

from __future__ import annotations

import json
from pathlib import Path

from axo.weather import WeatherReport
from axo.weather_watch import (
    RAIN_CHANGE_PP,
    TEMP_CHANGE_C,
    WeatherBaseline,
    WeatherSnapshot,
    WeatherWatchResult,
    changed_lines,
    material_text,
    snapshot_from_report,
    summary_line,
)


def snap(**overrides: object) -> WeatherSnapshot:
    values: dict[str, object] = {
        "city": "杭州",
        "observed_at": 1_760_000_000.0,
        "condition": "晴",
        "temp": 20.0,
        "high": 25.0,
        "low": 15.0,
        "rain": 10.0,
    }
    values.update(overrides)
    return WeatherSnapshot(**values)  # type: ignore[arg-type]


def test_the_first_observation_is_not_a_change() -> None:
    assert changed_lines(None, snap()) == ()


def test_a_real_temperature_drop_is_reported() -> None:
    lines = changed_lines(snap(temp=20.0), snap(temp=20.0 - TEMP_CHANGE_C))

    assert lines and "气温降了" in lines[0]


def test_a_small_temperature_wobble_is_ignored() -> None:
    assert changed_lines(snap(temp=20.0), snap(temp=19.0)) == ()


def test_a_condition_change_is_reported() -> None:
    lines = changed_lines(snap(condition="晴"), snap(condition="小雨"))

    assert any("晴" in line and "小雨" in line for line in lines)


def test_a_big_rain_probability_jump_is_reported() -> None:
    lines = changed_lines(snap(rain=10.0), snap(rain=10.0 + RAIN_CHANGE_PP))

    assert any("降水概率" in line for line in lines)


def test_an_identical_observation_has_no_changes() -> None:
    assert changed_lines(snap(), snap(observed_at=1_760_001_000.0)) == ()


def test_the_summary_names_the_city_and_today_range() -> None:
    text = summary_line(snap())

    assert text.startswith("杭州：")
    assert "20°C" in text
    assert "15~25°C" in text


def test_material_text_appends_the_changes() -> None:
    result = WeatherWatchResult(
        snapshot=snap(condition="小雨", temp=12.0), changes=("天气从「晴」变成「小雨」",)
    )

    text = material_text(result)

    assert "杭州" in text
    assert "变化：" in text
    assert "小雨" in text


def test_snapshot_from_report_uses_today() -> None:
    report = WeatherReport(
        place="杭州",
        country="中国",
        admin="浙江",
        latitude=0.0,
        longitude=0.0,
        current_temp=18.0,
        feels_like=17.0,
        humidity=50.0,
        wind=5.0,
        condition="多云",
        days=({"condition": "阴", "high": 20.0, "low": 10.0, "rain": 30.0},),
    )

    snapshot = snapshot_from_report(report, city="杭州", now=123.0)

    assert snapshot.city == "杭州"
    assert snapshot.condition == "多云"
    assert (snapshot.high, snapshot.low, snapshot.rain) == (20.0, 10.0, 30.0)


def test_the_baseline_roundtrips_and_is_keyed_case_insensitively(tmp_path: Path) -> None:
    baseline = WeatherBaseline(tmp_path / "weather_baseline.json")

    baseline.put(snap(city="Hangzhou"))

    loaded = baseline.get("hangzhou")
    assert loaded is not None and loaded.city == "Hangzhou"
    assert (
        "Hangzhou"
        in json.loads((tmp_path / "weather_baseline.json").read_text("utf-8"))["hangzhou"]["city"]
    )


def test_a_corrupt_baseline_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "weather_baseline.json"
    path.write_text("{ not json", encoding="utf-8")

    assert WeatherBaseline(path).load() == {}


def test_a_missing_baseline_reads_as_empty(tmp_path: Path) -> None:
    assert WeatherBaseline(tmp_path / "nope.json").load() == {}
