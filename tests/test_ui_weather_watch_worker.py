"""天气检查 worker：查一次、跟基线比一次，把变化交回界面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from axo.ui.weather_watch_worker import WeatherWatchWorker
from axo.weather import WeatherReport
from axo.weather_watch import WeatherBaseline, WeatherSnapshot

pytestmark = pytest.mark.gui


class FakeWeather:
    def __init__(self, *, temp: float = 12.0, boom: bool = False) -> None:
        self.temp = temp
        self.boom = boom

    def lookup(self, city: str, *, days: int = 1) -> WeatherReport:
        if self.boom:
            raise RuntimeError("天气服务挂了")
        assert days == 1
        return WeatherReport(
            place=city,
            country="中国",
            admin="",
            latitude=0.0,
            longitude=0.0,
            current_temp=self.temp,
            feels_like=None,
            humidity=None,
            wind=None,
            condition="小雨",
            days=({"condition": "小雨", "high": 15.0, "low": 8.0, "rain": 80.0},),
        )


def run(worker: Any) -> tuple[list[Any], list[str]]:
    done: list[Any] = []
    failed: list[str] = []
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()
    return done, failed


def test_a_first_look_has_no_changes(tmp_path: Path) -> None:
    baseline = WeatherBaseline(tmp_path / "b.json")
    done, failed = run(WeatherWatchWorker(FakeWeather(temp=12.0), "杭州", baseline))

    assert failed == []
    assert done[0].snapshot.city == "杭州"
    assert done[0].changes == ()


def test_it_reports_a_temperature_drop_against_the_baseline(tmp_path: Path) -> None:
    baseline = WeatherBaseline(tmp_path / "b.json")
    baseline.put(
        WeatherSnapshot(
            city="杭州",
            observed_at=1.0,
            condition="晴",
            temp=22.0,
            high=25.0,
            low=15.0,
            rain=10.0,
        )
    )

    done, _failed = run(WeatherWatchWorker(FakeWeather(temp=12.0), "杭州", baseline))

    result = done[0]
    assert result.noteworthy is True
    assert any("气温降了" in line for line in result.changes)


def test_a_failure_is_a_sentence(tmp_path: Path) -> None:
    done, failed = run(
        WeatherWatchWorker(FakeWeather(boom=True), "杭州", WeatherBaseline(tmp_path / "b.json"))
    )

    assert done == []
    assert failed and "天气服务挂了" in failed[0]
