"""常驻总线 daemon：真 ``SpikeBus``、真 ``step()``、可复现的落盘。

这些测试**不用替身总线**。理由与 ``memcore/substrate.py`` 那段话对着来：说"接上
总线了"必须有一条能被跑出来的证据——这里就是"一条记忆进来，``(T, bus_dim)`` 出去，
输入哈希与参数落在 provenance 里"。用替身总线测只能证明"我们调用了自己的函数"。
"""

from __future__ import annotations

import ast
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from axo.busd.config import BusDaemonConfig
from axo.busd.curiosity import RetrievalIntent
from axo.busd.daemon import BusDaemon, TickResult
from axo.busd.events import append_event, read_new_events
from axo.busd.plugin import MemorySpikePlugin
from axo.busd.state import read_state, write_state
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier


def make_item(text: str = "一条记忆") -> MemoryItem:
    return MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content=text,
        taint=Taint.USER,
        weight=0.7,
        lex_sdr=encode_lexical(text),
    )


@pytest.fixture
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def make_daemon(
    store: MemoryStore,
    tmp_path: Path,
    *,
    curiosity: object = None,
    **overrides: object,
) -> BusDaemon:
    kwargs: dict[str, object] = {"dream_enabled": False}  # 默认关：tick 的测试不该真跑巩固
    kwargs.update(overrides)
    config = BusDaemonConfig(**kwargs)  # type: ignore[arg-type]
    return BusDaemon(
        store, tmp_path / "data", config=config, clock=lambda: 1000.0, curiosity=curiosity
    )


# ── 插件契约 ────────────────────────────────────────────────────────────────


def test_plugin_satisfies_the_bus_contract() -> None:
    plugin = MemorySpikePlugin()
    plugin.validate()  # 不合法会抛 PluginContractError
    assert plugin.modality_name == "memory"
    assert plugin.spike_dim == 4096
    assert plugin.fusion_channel == "semantic"


def test_plugin_puts_one_memory_per_time_step() -> None:
    plugin = MemorySpikePlugin(width=4096)
    sdr = encode_lexical("水底第一粒微光")
    train = plugin.encode([sdr, sdr, sdr])
    assert train.data.shape == (3, 4096)
    assert train.data.dtype == np.float64

    rows = plugin.decode(train)
    assert len(rows) == 3
    assert all(len(r) <= 64 for r in rows)
    assert all(np.all(np.diff(r) > 0) for r in rows)  # 升序下标


def test_plugin_rejects_the_wrong_width() -> None:
    plugin = MemorySpikePlugin(width=128)
    with pytest.raises(ValueError):
        plugin.encode(np.zeros((2, 64), dtype=np.float32))


# ── 真总线 ──────────────────────────────────────────────────────────────────


def test_tick_puts_a_batch_through_a_real_bus(store: MemoryStore, tmp_path: Path) -> None:
    items = [store.add(make_item(f"第{i}条记忆")) for i in range(5)]
    daemon = make_daemon(store, tmp_path, bus_dim=64)

    result = daemon.tick()

    assert result.processed == 5
    assert result.idle is False
    # ADR-0012：第二维恒为 bus_dim，与模态的 4096 无关
    assert result.out_shape == (5, 64)
    assert result.cursor == items[-1].memory_id
    assert daemon.state.processed == 5

    row = json.loads(daemon.provenance_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["ids"] == [it.memory_id for it in items]
    assert row["out_shape"] == [5, 64]
    assert row["bus_dim"] == 64 and row["seed"] == 0
    assert len(row["input_sha256"]) == 64
    assert daemon.state.cursor == read_state(daemon.state_path).cursor  # type: ignore[union-attr]


def test_second_tick_without_new_memories_is_idle(store: MemoryStore, tmp_path: Path) -> None:
    store.add(make_item("只有一条"))
    daemon = make_daemon(store, tmp_path)
    daemon.tick()
    before = daemon.provenance_path.read_text(encoding="utf-8").count("\n")

    result = daemon.tick()

    assert result.idle is True and result.processed == 0
    after = daemon.provenance_path.read_text(encoding="utf-8").count("\n")
    assert after == before, "空转不该写 provenance：那会让日志里的每一轮都像真的跑过"


def test_restart_resumes_from_the_cursor(store: MemoryStore, tmp_path: Path) -> None:
    store.add(make_item("第一条"))
    first = make_daemon(store, tmp_path)
    first.tick()

    store.add(make_item("第二条"))
    second = make_daemon(store, tmp_path)  # 新实例：只能从磁盘状态里知道游标
    result = second.tick()

    assert result.processed == 1
    row = json.loads(second.provenance_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["n"] == 1


def test_higher_batch_limit_does_not_resend_old_memories(
    store: MemoryStore, tmp_path: Path
) -> None:
    for i in range(3):
        store.add(make_item(f"老{i}"))
    daemon = make_daemon(store, tmp_path, batch=1)
    daemon.tick()  # 只吃第一条
    result = daemon.tick()  # 批次放大之前，剩下的还在游标后面
    assert result.processed == 1


# ── 失败与生命周期 ──────────────────────────────────────────────────────────


def test_store_failure_is_recorded_not_raised(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = make_daemon(store, tmp_path)

    def boom(**_: object) -> None:
        raise RuntimeError("库炸了")

    monkeypatch.setattr(store, "items_page", boom)
    result = daemon.tick()

    assert "库炸了" in result.error
    assert "库炸了" in (read_state(daemon.state_path).last_error)  # type: ignore[union-attr]


def test_run_stops_promptly_on_the_event(store: MemoryStore, tmp_path: Path) -> None:
    store.add(make_item("一条"))
    daemon = make_daemon(store, tmp_path, interval_s=60.0)
    stop = threading.Event()
    stop.set()  # 进循环之前就已经要求停
    daemon.run(stop, max_ticks=5)
    assert daemon.state.ticks == 0, "已置位的 stop 不该再跑一轮"


# ── 涌现与事件 ──────────────────────────────────────────────────────────────


def test_first_tick_never_emerges(store: MemoryStore, tmp_path: Path) -> None:
    store.add(make_item("第一条"))
    daemon = make_daemon(store, tmp_path, emergence_novelty=0.0, emergence_floor=0.0)
    result = daemon.tick()
    assert result.emerged is False, "没有基线就没有'新'——别把第一轮算成涌现"


def test_a_different_batch_emerges_and_writes_a_greeting_event(
    store: MemoryStore, tmp_path: Path
) -> None:
    store.add(make_item("关于潮汐的第一条记忆"))
    daemon = make_daemon(store, tmp_path, emergence_novelty=0.0, emergence_floor=0.0)
    daemon.tick()

    store.add(make_item("完全不同的：今天修好了吉他音箱"))
    result = daemon.tick()

    assert result.emerged is True
    events, lines = read_new_events(daemon.events_path)
    assert lines == 1
    assert events[0].kind == "greeting_suggestion"
    assert "novelty" in events[0].payload and "activity" in events[0].payload


def test_event_reader_skips_corrupt_lines_and_still_advances(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_event(path, "daemon_start", {}, at=1.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{这不是 JSON\n")
    append_event(path, "greeting_suggestion", {"novelty": 0.9}, at=2.0)

    events, lines = read_new_events(path, 0)
    assert [e.kind for e in events] == ["daemon_start", "greeting_suggestion"]
    assert lines == 3

    events2, lines2 = read_new_events(path, lines)
    assert events2 == [] and lines2 == 3


def test_missing_event_file_is_not_an_error(tmp_path: Path) -> None:
    events, lines = read_new_events(tmp_path / "nope.jsonl")
    assert events == [] and lines == 0


# ── 巩固（做梦）────────────────────────────────────────────────────────────


def test_dream_is_skipped_when_recent(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []
    daemon = make_daemon(store, tmp_path, dream_enabled=True)
    daemon._dream_runner = lambda now: calls.append(now) or {}  # type: ignore[method-assign]
    monkeypatch.setattr(daemon, "_last_run", lambda: {"at": 999.0})  # 1 秒前刚跑过
    assert daemon.dream_if_due(1000.0) is None
    assert calls == []


def test_dream_runs_when_due(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []
    daemon = make_daemon(store, tmp_path, dream_enabled=True)
    daemon._dream_runner = lambda now: calls.append(now) or {"reinforced": 0}  # type: ignore[method-assign]
    monkeypatch.setattr(daemon, "_last_run", lambda: {"at": 1000.0 - 25 * 3600})
    assert daemon.dream_if_due(1000.0) == {"reinforced": 0}
    assert calls == [1000.0]


def test_dream_can_be_disabled(store: MemoryStore, tmp_path: Path) -> None:
    daemon = make_daemon(store, tmp_path, dream_enabled=False)
    daemon._dream_runner = lambda *_: pytest.fail("关掉了还跑")  # type: ignore[method-assign]
    assert daemon.dream_if_due(1000.0) is None


# ── 边界与约束 ──────────────────────────────────────────────────────────────


def test_state_round_trips_and_rejects_garbage(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_state(
        path,
        __import__("axo.busd.state", fromlist=["BusState"]).BusState(
            cursor="01ABC", processed=7, ticks=3, last_out_shape=(7, 128)
        ),
    )
    loaded = read_state(path)
    assert loaded is not None and loaded.cursor == "01ABC" and loaded.last_out_shape == (7, 128)

    path.write_text("{半截", encoding="utf-8")
    assert read_state(path) is None


def test_ui_side_event_reader_has_no_heavy_imports() -> None:
    """``events.py`` 是 UI 唯一依赖的那一块——它必须只用标准库。

    UI 不认识 numpy、biosnn_bus、memcore。这条用 AST 钉住，因为"顺手 import 一下
    numpy"在代码评审里看起来无害，却会让"UI 只接收、不依赖"这句话失效。
    """
    path = Path(__file__).resolve().parents[1] / "src" / "axo" / "busd" / "events.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    heavy = imported & {"numpy", "biosnn_bus", "memcore", "PySide6", "httpx", "torch"}
    assert not heavy, f"events.py 引入了重依赖：{sorted(heavy)}"


def test_the_daemon_does_not_import_qt() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "axo" / "busd" / "daemon.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "PySide6" not in names
    assert "axo.ui" not in {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }


# ── L3：涌现触发的接口桩 ────────────────────────────────────────────────────


class _Proposer:
    def __init__(self, intent: RetrievalIntent | None) -> None:
        self.intent = intent
        self.calls: list[tuple[float, float, tuple[str, ...]]] = []

    def propose(
        self, *, novelty: float, activity: float, memories: tuple[str, ...]
    ) -> RetrievalIntent | None:
        self.calls.append((novelty, activity, memories))
        return self.intent


def _emerged_tick(store: MemoryStore, tmp_path: Path, **kwargs: object) -> TickResult:
    store.add(make_item("关于潮汐的第一条记忆"))
    daemon = make_daemon(store, tmp_path, emergence_novelty=0.0, emergence_floor=0.0, **kwargs)
    daemon.tick()
    store.add(make_item("完全不同的：今天修好了吉他音箱"))
    return daemon.tick()


def test_curiosity_is_off_by_default(store: MemoryStore, tmp_path: Path) -> None:
    result = _emerged_tick(store, tmp_path)

    assert result.emerged is True
    assert result.intent == ""
    events, _ = read_new_events(tmp_path / "data" / "bus" / "events.jsonl")
    assert [event.kind for event in events] == ["greeting_suggestion"]


def test_enabled_curiosity_records_an_intent_without_touching_memory(
    store: MemoryStore, tmp_path: Path
) -> None:
    proposer = _Proposer(RetrievalIntent(query="查一下潮汐周期", reason="涌现"))
    before = len(store.load_all())

    result = _emerged_tick(store, tmp_path, curiosity_enabled=True, curiosity=proposer)

    assert result.intent == "查一下潮汐周期"
    events, _ = read_new_events(tmp_path / "data" / "bus" / "events.jsonl")
    assert [event.kind for event in events] == ["greeting_suggestion", "retrieval_intent"]
    intent = events[1].payload
    assert intent["query"] == "查一下潮汐周期" and intent["memories"]
    assert len(store.load_all()) == before + 2, "测试加的两条之外，daemon 不许写库"
    assert proposer.calls, "开启后必须真的调用 proposer"


def test_a_proposer_that_declines_adds_no_event(store: MemoryStore, tmp_path: Path) -> None:
    result = _emerged_tick(store, tmp_path, curiosity_enabled=True, curiosity=_Proposer(None))

    assert result.intent == ""
    events, _ = read_new_events(tmp_path / "data" / "bus" / "events.jsonl")
    assert [event.kind for event in events] == ["greeting_suggestion"]


def test_enabling_curiosity_without_a_proposer_is_rejected(
    store: MemoryStore, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="EmergentRetrieval"):
        BusDaemon(
            store,
            tmp_path / "data",
            config=BusDaemonConfig(curiosity_enabled=True, dream_enabled=False),
        )
