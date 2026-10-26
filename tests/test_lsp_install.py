"""语言服务器安装器：计划、手工提示、JSON、默认不执行。"""

from __future__ import annotations

import json

from axo.agent import lsp_install


def test_plan_covers_common_languages_and_manual_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(lsp_install, "_installed", lambda _plan: False)
    monkeypatch.setattr(lsp_install, "_command", lambda method: ["echo", method.kind])
    rows = lsp_install.build_plans()
    assert len(rows) >= 20
    by_language = {row["language"]: row for row in rows}
    assert by_language["python"]["command"] == ["echo", "npm"]
    assert by_language["rust"]["command"] == ["echo", "rustup"]
    assert by_language["java"]["command"] == []
    assert "AXO_JDTLS_HOME" in str(by_language["java"]["manual"])


def test_only_filters_languages(monkeypatch) -> None:
    monkeypatch.setattr(lsp_install, "_installed", lambda _plan: False)
    monkeypatch.setattr(lsp_install, "_command", lambda _method: ["echo"])
    assert [row["language"] for row in lsp_install.build_plans(["python"])] == ["python"]


def test_json_output_is_machine_readable(capsys) -> None:
    code = lsp_install.main(["--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item["language"] == "python" for item in payload)


def test_default_mode_does_not_run_commands(monkeypatch, capsys) -> None:
    called: list[list[str]] = []
    monkeypatch.setattr(lsp_install, "_installed", lambda _plan: False)
    monkeypatch.setattr(lsp_install, "_command", lambda _method: ["echo", "should-not-run"])
    monkeypatch.setattr(lsp_install, "_run", lambda command: called.append(command) or (True, ""))
    code = lsp_install.main(["--only", "python"])
    assert code == 0
    assert called == []
    assert "只打印计划" in capsys.readouterr().out
