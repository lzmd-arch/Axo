"""启动自检：每一项都必须真的碰到东西，失败必须显示为失败。

这些测试**不构造 Qt**——``boot_checks`` 是纯层，开屏只负责把它算出来的结果画出来。
"""

from __future__ import annotations

import socket
from pathlib import Path

from memcore.store import MemoryStore


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AXO_DATA", str(tmp_path))


def test_the_data_directory_probe_actually_writes(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks

    _isolate(monkeypatch, tmp_path)
    checks = {check.key: check for check in collect_checks()}
    assert checks["data"].ok is True
    assert checks["data"].detail == str(tmp_path)
    # 探针文件不留下来
    assert not list(tmp_path.glob(".boot-probe-*"))


def test_a_first_run_without_a_store_is_not_a_failure(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks

    _isolate(monkeypatch, tmp_path)
    checks = {check.key: check for check in collect_checks()}
    assert checks["archive"].ok is True
    assert checks["archive"].detail == "首次启动，将新建"
    assert checks["persona"].ok is True
    # 自检不能顺手把库建出来
    assert not (tmp_path / "memory" / "mem.db").exists()


def test_the_memory_checks_read_the_real_store(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks

    _isolate(monkeypatch, tmp_path)
    db = tmp_path / "memory" / "mem.db"
    db.parent.mkdir(parents=True)
    with MemoryStore(db):
        pass

    checks = {check.key: check for check in collect_checks()}
    assert checks["archive"].ok is True
    assert checks["archive"].detail == "0 条"
    assert checks["persona"].detail == "还没有，会问你"


def test_a_corrupt_store_is_reported_as_a_failure(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks

    _isolate(monkeypatch, tmp_path)
    db = tmp_path / "memory" / "mem.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"not a sqlite database" * 32)

    checks = {check.key: check for check in collect_checks()}
    assert checks["archive"].ok is False
    assert checks["archive"].detail.startswith("打不开")
    assert checks["persona"].ok is False


def test_the_backend_check_reports_a_missing_gguf(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    checks = {
        check.key: check for check in collect_checks(backend=BackendSettings(provider="builtin"))
    }
    assert checks["backend"].ok is False
    assert "还没选" in checks["backend"].detail


def test_the_backend_check_reports_a_missing_cloud_key(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    missing = {
        check.key: check
        for check in collect_checks(backend=BackendSettings(provider="cloud", api_key=""))
    }
    assert missing["backend"].ok is False
    assert "未填 key" in missing["backend"].detail

    ready = {
        check.key: check
        for check in collect_checks(backend=BackendSettings(provider="cloud", api_key="sk-test"))
    }
    assert ready["backend"].ok is True
    assert "已配置" in ready["backend"].detail


def test_the_scope_check_says_where_data_can_go(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    cloud = {
        check.key: check
        for check in collect_checks(backend=BackendSettings(provider="cloud", api_key="sk-test"))
    }
    assert "云端已启用" in cloud["scope"].detail

    local = {
        check.key: check for check in collect_checks(backend=BackendSettings(provider="ollama"))
    }
    assert local["scope"].detail == "仅本机"


def test_an_unreachable_ollama_port_is_marked_down(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    # 拿一个刚刚还听着、现在已关闭的本机端口：连接会被立刻拒绝，不靠 0.6 秒超时。
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    checks = {
        check.key: check
        for check in collect_checks(
            backend=BackendSettings(provider="ollama", host=f"http://127.0.0.1:{port}")
        )
    }
    assert checks["backend"].ok is False
    assert "未响应" in checks["backend"].detail


def test_the_tools_check_summarizes_real_permissions(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import ToolSettings

    root = tmp_path / "notes"
    root.mkdir()
    checks = {
        check.key: check
        for check in collect_checks(
            tools=ToolSettings(
                read_roots=(root,),
                write_roots=(root,),
                allow_commands=True,
                web_enabled=False,
            )
        )
    }
    detail = checks["tools"].detail
    assert checks["tools"].ok is True
    assert "1 个只读" in detail
    assert "1 个可写" in detail
    assert "可执行命令" in detail
    assert "可联网" not in detail


def test_a_deleted_tool_root_is_reported(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import ToolSettings

    _isolate(monkeypatch, tmp_path)
    checks = {
        check.key: check
        for check in collect_checks(tools=ToolSettings(read_roots=(tmp_path / "gone",)))
    }
    assert checks["tools"].ok is False
    assert "1 个目录不存在" in checks["tools"].detail


def test_a_builtin_model_without_the_runtime_is_a_failure(monkeypatch, tmp_path: Path) -> None:
    from axo import paths
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    _isolate(monkeypatch, tmp_path)
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"fake")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(paths, "bundled_llama_dir", lambda: runtime)

    checks = {
        check.key: check
        for check in collect_checks(
            backend=BackendSettings(provider="builtin", local_model_path=str(model))
        )
    }
    assert checks["backend"].ok is False
    assert "运行时缺失" in checks["backend"].detail


def test_one_broken_check_does_not_wipe_the_panel(monkeypatch, tmp_path: Path) -> None:
    from axo.ui.boot_checks import collect_checks
    from axo.ui.settings_store import BackendSettings

    _isolate(monkeypatch, tmp_path)
    checks = {
        check.key: check
        for check in collect_checks(
            backend=BackendSettings(provider="ollama", host="http://127.0.0.1:notaport")
        )
    }
    assert checks["backend"].ok is False
    assert "检测失败" in checks["backend"].detail
    # 在它之前查到的结果还在
    assert checks["data"].ok is True
