"""总线 daemon 的本体：拉新记忆 → 编码 → ``SpikeBus.step()`` → 落盘 → 必要时做梦。

一轮（:meth:`BusDaemon.tick`）只做五件事，顺序固定：

1. 按 ``memory_id`` 游标取一批**新**条目（只读，不改库）；
2. 每条的关联码编码成一行 → ``(T, width)``；
3. ``bus.step({"memory": rows})`` —— 真的走融合与稀疏随机投影；
4. 把这一轮的输入哈希、参数、输出形状、发放率写进 ``bus/provenance.jsonl``；
5. 游标前进。

**已知的粗**：游标只看"新条目"。改一条、删一条不会被看见——与巩固那道门同一条
取舍（``memory_worker.ConsolidationWorker`` 的文档里写过）。要看见它们得比对全库
指纹，而那是每轮一次全表扫描。新记忆会把 daemon 拉回来，那时也只会处理新的。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from biosnn_bus import SpikeBus

from memcore.sdr import encode_assoc
from memcore.store import MemoryStore

from .config import BusDaemonConfig
from .curiosity import EmergentRetrieval
from .events import append_event
from .plugin import MemorySpikePlugin
from .state import BusState, read_state, write_state

__all__ = ["BusDaemon", "TickResult"]

K_ASSOC = 64


@dataclass(frozen=True)
class TickResult:
    """一轮的结果。**足以在日志里复述"这一轮发生了什么"**，不含整张矩阵。"""

    processed: int
    idle: bool
    cursor: str
    out_shape: tuple[int, int]
    activity: float
    novelty: float
    emerged: bool
    error: str = ""
    dream: dict[str, Any] | None = None
    intent: str = ""
    """本轮产生的 L3 检索意图（查询原文）；空串 = 没有提议。**只是记录，未执行。**"""

    def to_json(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "idle": self.idle,
            "cursor": self.cursor,
            "out_shape": list(self.out_shape),
            "activity": round(self.activity, 6),
            "novelty": round(self.novelty, 4),
            "emerged": self.emerged,
            "error": self.error,
            "dream": self.dream,
            "intent": self.intent,
        }


class BusDaemon:
    """常驻总线进程的本体。**不 import Qt**，可以被单独跑起来。"""

    def __init__(
        self,
        store: MemoryStore,
        root: Path | str,
        *,
        config: BusDaemonConfig | None = None,
        clock: Callable[[], float] = time.time,
        dream_runner: Callable[[float], dict[str, Any] | None] | None = None,
        curiosity: EmergentRetrieval | None = None,
    ) -> None:
        self.store = store
        self.config = config or BusDaemonConfig()
        self.bus_dir = Path(root) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self.bus = SpikeBus(
            self.config.bus_dim,
            dt=self.config.dt_ms,
            projection_density=self.config.projection_density,
            seed=self.config.seed,
        )
        self.modality = self.bus.register(
            MemorySpikePlugin(width=self.config.width, dt=self.config.dt_ms, k=K_ASSOC)
        )
        self.state: BusState = read_state(self.state_path) or BusState()
        self._prev_rates: np.ndarray | None = self._load_rates()
        self._dream_runner = dream_runner or self._default_dream
        if self.config.curiosity_enabled and curiosity is None:
            raise ValueError("curiosity_enabled=True 时必须注入一个 EmergentRetrieval。")
        self._curiosity = curiosity

    # ------------------------------------------------------------ 路径

    @property
    def state_path(self) -> Path:
        return self.bus_dir / "state.json"

    @property
    def events_path(self) -> Path:
        return self.bus_dir / "events.jsonl"

    @property
    def provenance_path(self) -> Path:
        return self.bus_dir / "provenance.jsonl"

    @property
    def rates_path(self) -> Path:
        return self.bus_dir / "last_rates.npy"

    # ------------------------------------------------------------ 一轮

    def tick(self) -> TickResult:
        now = float(self._clock())
        try:
            items, _ = self.store.items_page(
                after_id=self.state.cursor,
                limit=max(1, min(1000, self.config.batch)),
                descending=False,
                exclude_persona=False,
            )
        except Exception as exc:  # noqa: BLE001 — 读库失败要如实记，不该打死 daemon
            return self._fail(now, f"读库失败：{type(exc).__name__}: {exc}")
        items = [it for it in items if it.memory_id > self.state.cursor]
        if not items:
            self.state = replace(
                self.state, ticks=self.state.ticks + 1, last_tick_at=now, last_error=""
            )
            write_state(self.state_path, self.state)
            return TickResult(
                0,
                True,
                self.state.cursor,
                self.state.last_out_shape,
                self.state.last_activity,
                self.state.last_novelty,
                False,
            )

        t0 = time.perf_counter()
        try:
            rows = np.stack(
                [
                    np.asarray(encode_assoc(it.lex_sdr, k=K_ASSOC).to_dense(), dtype=np.float64)
                    for it in items
                ]
            )
            out = self.bus.step({self.modality: rows})
        except Exception as exc:  # noqa: BLE001
            return self._fail(now, f"总线一步失败：{type(exc).__name__}: {exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rates = np.asarray(out.data, dtype=np.float64).mean(axis=0)
        # 融合输出是带 ±1 权重的稀疏随机投影，**均值可以是负的**；活动量取 |·| 的均值
        activity = float(np.abs(rates).mean())
        novelty = self._novelty(rates)
        emerged = (
            self._prev_rates is not None
            and novelty >= self.config.emergence_novelty
            and activity >= self.config.emergence_floor
        )
        ids = [it.memory_id for it in items]
        self._write_provenance(now, ids, rows, out.data.shape, rates, novelty, elapsed_ms)
        np.save(self.rates_path, rates)
        self._prev_rates = rates

        cursor = ids[-1]
        self.state = replace(
            self.state,
            cursor=cursor,
            processed=self.state.processed + len(items),
            ticks=self.state.ticks + 1,
            last_tick_at=now,
            last_out_shape=(int(out.data.shape[0]), int(out.data.shape[1])),
            last_activity=activity,
            last_novelty=novelty,
            last_error="",
        )
        write_state(self.state_path, self.state)

        intent = ""
        if emerged:
            append_event(
                self.events_path,
                "greeting_suggestion",
                {
                    "novelty": round(novelty, 4),
                    "activity": round(activity, 6),
                    "memories": ids[-5:],
                    "bus_dim": self.config.bus_dim,
                    "seed": self.config.seed,
                },
                at=now,
            )
            if self.config.curiosity_enabled:
                assert self._curiosity is not None
                proposal = self._curiosity.propose(
                    novelty=novelty,
                    activity=activity,
                    memories=tuple(ids[-5:]),
                )
                if proposal is not None:
                    intent = proposal.query
                    append_event(
                        self.events_path,
                        "retrieval_intent",
                        {
                            "query": proposal.query,
                            "reason": proposal.reason,
                            "novelty": round(novelty, 4),
                            "activity": round(activity, 6),
                            "memories": ids[-5:],
                        },
                        at=now,
                    )

        dream = None
        try:
            dream = self.dream_if_due(now)
        except Exception as exc:  # noqa: BLE001 — 做梦失败不该让总线停
            self.state = replace(self.state, last_error=f"巩固失败：{type(exc).__name__}: {exc}")
            write_state(self.state_path, self.state)
        return TickResult(
            len(items),
            False,
            cursor,
            (int(out.data.shape[0]), int(out.data.shape[1])),
            activity,
            novelty,
            emerged,
            dream=dream,
            intent=intent,
        )

    # ------------------------------------------------------------ 循环

    def run(
        self,
        stop: threading.Event,
        *,
        on_tick: Callable[[TickResult], None] | None = None,
        max_ticks: int | None = None,
    ) -> None:
        """跑到 ``stop`` 被置位。**用 ``Event.wait`` 睡眠**，所以停得很干脆。"""
        done = 0
        while not stop.is_set():
            result = self.tick()
            if on_tick is not None:
                on_tick(result)
            done += 1
            if max_ticks is not None and done >= max_ticks:
                return
            if stop.wait(max(0.1, self.config.interval_s)):
                return

    # ------------------------------------------------------------ 做梦

    def dream_if_due(self, now: float) -> dict[str, Any] | None:
        """到点就跑一轮巩固。判定读 `last.json`，与 UI 调度器**同一个账本**。"""
        if not self.config.dream_enabled:
            return None
        last = self._last_run()
        last_at = last.get("at") if isinstance(last, dict) else None
        if last_at is not None and now - float(last_at) < self.config.dream_interval_s:
            return None
        return self._dream_runner(now)

    def _default_dream(self, now: float) -> dict[str, Any] | None:
        from axo.memory.consolidation import run_consolidation
        from axo.paths import graph_path
        from memcore.graph import GraphIndex

        items = self.store.load_all()
        if not items:
            return None
        path = graph_path()
        graph = GraphIndex.load(path) if Path(path).exists() else GraphIndex.build(items)
        run = run_consolidation(self.store, graph, at=now)
        append_event(
            self.events_path,
            "dream_done",
            {"reinforced": int(run.reinforced), "events": int(run.events_written)},
            at=now,
        )
        return {"reinforced": int(run.reinforced), "events": int(run.events_written)}

    @staticmethod
    def _last_run() -> dict[str, Any] | None:
        try:
            from axo.memory.consolidation import read_last_run

            return read_last_run()
        except Exception:  # noqa: BLE001 — 读不到就把这一轮让掉，别猜
            return None

    # ------------------------------------------------------------ 内部

    def _novelty(self, rates: np.ndarray) -> float:
        """这一次的发放向量与上一次有多不像：``1 - cos``。

        选它的理由：稀疏随机投影把每条记忆打到同一组 ``bus_dim`` 根线上，所以
        "一个新东西进来了"在总线上表现为**方向变了**，而不是"活动变多了"——后者
        在库里条目变多时也会上升，是规模而不是新颖度。上一轮不存在时返回 0
        （没有基线就没有"新"，不编一个数）。
        """
        if self._prev_rates is None:
            return 0.0
        a = np.asarray(rates, dtype=np.float64)
        b = np.asarray(self._prev_rates, dtype=np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.clip(1.0 - float(np.dot(a, b)) / (na * nb), 0.0, 2.0))

    def _load_rates(self) -> np.ndarray | None:
        try:
            arr = np.load(self.rates_path)
        except (OSError, ValueError):
            return None
        if arr.shape != (self.config.bus_dim,):
            return None
        return arr.astype(np.float64)

    def _write_provenance(
        self,
        now: float,
        ids: list[str],
        rows: np.ndarray,
        shape: tuple[int, int],
        rates: np.ndarray,
        novelty: float,
        elapsed_ms: float,
    ) -> None:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(rows, dtype=np.float32).tobytes())
        line = {
            "at": now,
            "modality": self.modality,
            "width": self.config.width,
            "bus_dim": self.config.bus_dim,
            "dt_ms": self.config.dt_ms,
            "projection_density": self.config.projection_density,
            "seed": self.config.seed,
            "n": int(rows.shape[0]),
            "ids": ids,
            "input_sha256": digest.hexdigest(),
            "out_shape": [int(shape[0]), int(shape[1])],
            "mean_rate": float(rates.mean()),
            "p95_rate": float(np.percentile(rates, 95)),
            "max_rate": float(rates.max()),
            "novelty": float(novelty),
            "elapsed_ms": float(elapsed_ms),
        }
        with self.provenance_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _fail(self, now: float, message: str) -> TickResult:
        self.state = replace(
            self.state, ticks=self.state.ticks + 1, last_tick_at=now, last_error=message
        )
        write_state(self.state_path, self.state)
        return TickResult(
            0,
            False,
            self.state.cursor,
            self.state.last_out_shape,
            self.state.last_activity,
            self.state.last_novelty,
            False,
            error=message,
        )
