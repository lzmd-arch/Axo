"""纵向记忆基准 v0：同一金标准、不同「记忆年龄」下的召回质量。

护城河假设是「记忆质量随积累月数变化」。要验证它，不能等 12 个月——金标准语料的
时间戳是冻结的（``CREATED_AT``），而召回用传入的 ``now`` 结算惰性衰减。所以我们在
**同一份库**上把 ``now`` 往后拨 0/1/3/6/12 个月，量召回质量随年龄的变化。

这是**合成年龄**：它模拟「如果这些记忆真的放了 N 个月，召回会怎样」，不是真实积累。
v0 只建立仪表：每月用同一命令、同一金标准跑一次，快照落盘，逐月比较。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .goldset import CREATED_AT, corpus_digests, default_gold_dir, load_goldset
from .harness import run
from .metrics import (
    false_injection_rate,
    hit_at_k,
    mrr,
    no_answer_injections,
    recall_at_k,
)

__all__ = ["AGES_MONTHS", "snapshot", "main"]

#: 默认的合成年龄（月）。0 是基线，其余用于看衰减。
AGES_MONTHS: tuple[int, ...] = (0, 1, 3, 6, 12)

_SECONDS_PER_MONTH = 30 * 24 * 3600


def _num(value: float) -> float | None:
    """空结果时指标是 nan；落盘为 None，别让 JSON 里出现 nan。"""
    number = float(value)
    return number if math.isfinite(number) else None


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def snapshot(
    *,
    gold_dir: Path | None = None,
    ages: tuple[int, ...] = AGES_MONTHS,
) -> dict:
    """跑一组合成年龄，返回可落盘的快照。"""
    directory = Path(gold_dir) if gold_dir is not None else default_gold_dir()
    goldset = load_goldset(directory)
    per_age: dict[str, dict[str, float | None]] = {}
    for months in ages:
        outcomes = run(goldset, now=CREATED_AT + months * _SECONDS_PER_MONTH)
        per_age[str(months)] = {
            "hit@1": _num(hit_at_k(outcomes, 1)),
            "hit@3": _num(hit_at_k(outcomes, 3)),
            "recall@3": _num(recall_at_k(outcomes, 3)),
            "mrr": _num(mrr(outcomes)),
            "false_injection_rate": _num(false_injection_rate(outcomes)),
            "no_answer_injections": _num(no_answer_injections(outcomes)),
        }
    return {
        "schema_version": 1,
        "kind": "longitudinal-v0 (synthetic ages)",
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "gold": corpus_digests(directory),
        "ages_months": list(ages),
        "metrics_by_age": per_age,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="纵向记忆基准 v0（合成年龄）")
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--ages", default="0,1,3,6,12", help="逗号分隔的月数")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ages = tuple(int(part) for part in args.ages.split(",") if part.strip())
    record = snapshot(gold_dir=args.gold_dir, ages=ages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(record["metrics_by_age"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
