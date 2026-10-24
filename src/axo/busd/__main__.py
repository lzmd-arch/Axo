"""``python -m axo.busd`` —— 常驻总线进程的入口。

用法::

    python -m axo.busd --once              # 跑一轮就退（冒烟 / 登录时用）
    python -m axo.busd --interval 60       # 常驻，60 秒醒一次
    python -m axo.busd --status            # 只打印状态，不碰总线

退出码：``0`` 正常；``1`` 这一轮有错（已写进 state，进程不炸）；``2`` 库打不开。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import threading
import time
from pathlib import Path

from memcore.store import MemoryStore

from .config import BusDaemonConfig
from .daemon import BusDaemon
from .events import append_event


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m axo.busd", description="Axo 常驻脉冲总线 daemon")
    p.add_argument("--db", type=Path, default=None, help="记忆库路径（默认走 axo.paths）")
    p.add_argument("--data-dir", type=Path, default=None, help="数据根目录（默认走 axo.paths）")
    p.add_argument("--once", action="store_true", help="跑一轮就退")
    p.add_argument("--status", action="store_true", help="只打印状态")
    p.add_argument("--interval", type=float, default=60.0, help="醒一次的间隔（秒）")
    p.add_argument("--bus-dim", type=int, default=128, help="总线维度")
    p.add_argument("--batch", type=int, default=64, help="一轮最多吃多少条新记忆")
    p.add_argument("--seed", type=int, default=0, help="投影矩阵种子")
    p.add_argument("--no-dream", action="store_true", help="关掉巩固（做梦）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.db is None or args.data_dir is None:
        from axo import paths  # 延迟 import：测试可以直接给 --db / --data-dir

        args.db = args.db or paths.memory_db_path()
        args.data_dir = args.data_dir or paths.data_dir()

    try:
        store = MemoryStore(Path(args.db))
    except Exception as exc:  # noqa: BLE001 — 库打不开是 2，不是崩溃
        print(f"打不开记忆库：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    with store:
        config = BusDaemonConfig(
            bus_dim=args.bus_dim,
            batch=args.batch,
            interval_s=args.interval,
            seed=args.seed,
            dream_enabled=not args.no_dream,
        )
        daemon = BusDaemon(store, Path(args.data_dir), config=config)
        if args.status:
            print(json.dumps(daemon.state.to_json(), ensure_ascii=False, indent=1))
            return 0
        if args.once:
            result = daemon.tick()
            print(json.dumps(result.to_json(), ensure_ascii=False))
            return 0 if not result.error else 1

        stop = threading.Event()

        def _stop(_signum: int, _frame: object) -> None:
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, _stop)

        append_event(
            daemon.events_path,
            "daemon_start",
            {
                "pid": __import__("os").getpid(),
                "bus_dim": config.bus_dim,
                "interval_s": config.interval_s,
            },
            at=time.time(),
        )
        print(
            f"busd 启动：db={args.db} bus_dim={config.bus_dim} interval={config.interval_s}s",
            flush=True,
        )

        def _report(result: object) -> None:
            print(json.dumps(result.to_json(), ensure_ascii=False), flush=True)  # type: ignore[attr-defined]

        daemon.run(stop, on_tick=_report)
        append_event(
            daemon.events_path, "daemon_stop", {"ticks": daemon.state.ticks}, at=time.time()
        )
        return 0 if not daemon.state.last_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
