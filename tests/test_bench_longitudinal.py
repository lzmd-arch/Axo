"""纵向基准 v0 的记录形状守卫（不跑真语料）。"""

from __future__ import annotations

from axo.bench import longitudinal


def test_snapshot_record_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(longitudinal, "load_goldset", lambda _directory=None: object())
    monkeypatch.setattr(longitudinal, "run", lambda _goldset, **_kwargs: [])
    monkeypatch.setattr(
        longitudinal, "corpus_digests", lambda _directory: {"memories": {"count": 0}}
    )
    record = longitudinal.snapshot(gold_dir=tmp_path, ages=(0, 1))
    assert record["kind"].startswith("longitudinal-v0")
    assert set(record["metrics_by_age"]) == {"0", "1"}
    for metrics in record["metrics_by_age"].values():
        # 空结果时指标未定义，落盘为 None（不是 nan）。
        assert metrics["hit@1"] is None and metrics["mrr"] is None
    assert record["gold"]["memories"]["count"] == 0


def test_cli_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(longitudinal, "load_goldset", lambda _directory=None: object())
    monkeypatch.setattr(longitudinal, "run", lambda _goldset, **_kwargs: [])
    monkeypatch.setattr(
        longitudinal, "corpus_digests", lambda _directory: {"memories": {"count": 0}}
    )
    out = tmp_path / "snap.json"
    assert longitudinal.main(["--ages", "0", "--out", str(out)]) == 0
    assert out.exists()
