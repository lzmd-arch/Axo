"""响应元数据日志：只追加、没有就不写、空 meta 不建文件。"""

from __future__ import annotations

import json
from pathlib import Path

from axo.llm_meta import append_response_meta


def test_appends_one_json_line_per_response(tmp_path: Path) -> None:
    path = tmp_path / "llm-meta.jsonl"
    append_response_meta(path, model_ref="openai-compat/kimi-k2", meta={"model": "kimi-k2"}, at=1.0)
    append_response_meta(
        path, model_ref="openai-compat/deepseek-flash", meta={"build": "b"}, at=2.0
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["meta"] == {"model": "kimi-k2"}
    assert json.loads(lines[1])["model_ref"] == "openai-compat/deepseek-flash"
    assert json.loads(lines[1])["at"] == 2.0


def test_empty_meta_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "llm-meta.jsonl"
    append_response_meta(path, model_ref="x", meta={})
    assert not path.exists()
