"""daemon 的持久状态：游标、计数、上一次的发放摘要。

**只放小东西**。整张发放矩阵不进这里——它每轮都是新的、体积随 bus_dim 增长，
写进来只会让"状态"变成又一个数据库。要复现某一轮就去看
``bus/provenance.jsonl`` 里那一行的输入哈希与种子。
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["BusState", "read_state", "write_state"]

STATE_VERSION = 1


@dataclass(frozen=True)
class BusState:
    """一轮跑完之后仍然成立的事实。

    ``cursor`` 是**最后一条被送进总线的 memory_id**（ULID 单调，所以它同时是排序键
    与游标）。它只在新条目真的走过 ``step()`` 之后才前进——先写游标再算，丢一次
    比重复算一次更难发现。
    """

    cursor: str = ""
    processed: int = 0
    ticks: int = 0
    last_tick_at: float = 0.0
    last_out_shape: tuple[int, int] = (0, 0)
    last_activity: float = 0.0
    last_novelty: float = 0.0
    last_error: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["version"] = STATE_VERSION
        d["last_out_shape"] = list(self.last_out_shape)
        return d


def read_state(path: Path) -> BusState | None:
    """读状态；**没有、坏了、版本不对，一律 ``None``**。

    坏的游标不该让 daemon 从"空库"重新开始（那会把整库再送一遍总线），也不该
    抛异常把进程打死——返回 ``None``，由调用方决定怎么起步（当前是从头，并如实
    在状态里记一条 ``last_error``）。
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        return None
    try:
        shape = tuple(int(x) for x in payload.get("last_out_shape", (0, 0)))
        return BusState(
            cursor=str(payload.get("cursor", "")),
            processed=int(payload.get("processed", 0)),
            ticks=int(payload.get("ticks", 0)),
            last_tick_at=float(payload.get("last_tick_at", 0.0)),
            last_out_shape=(shape[0], shape[1]) if len(shape) == 2 else (0, 0),
            last_activity=float(payload.get("last_activity", 0.0)),
            last_novelty=float(payload.get("last_novelty", 0.0)),
            last_error=str(payload.get("last_error", "")),
        )
    except (TypeError, ValueError):
        return None


def write_state(path: Path, state: BusState) -> None:
    """原子写：先写同目录临时文件再 ``os.replace``。

    半截 JSON 会让下一次启动把游标读成 ``None``，于是整库重送一遍——那是"看起来
    在工作"的最坏形态。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(state.to_json(), fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise
