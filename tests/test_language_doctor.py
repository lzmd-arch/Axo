"""语言体检的纯函数：26 个后缀、JSON、缺件提示。"""

from __future__ import annotations

import json

from axo.agent import language_doctor


def test_rows_cover_all_common_suffixes() -> None:
    rows = language_doctor.build_rows()
    assert len(rows) == 26
    suffixes = {row["suffix"] for row in rows}
    assert {".py", ".rs", ".go", ".tsx", ".vue", ".svelte"} <= suffixes
    assert all(row["symbol_index"] for row in rows)


def test_table_and_json_are_produced(capsys) -> None:
    assert language_doctor.main([]) == 0
    text = capsys.readouterr().out
    assert "后缀" in text and ".py" in text

    assert language_doctor.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 26
