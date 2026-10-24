"""云端响应的版本元数据 → 一次只追加的运行日志。

**为什么需要它**：云端模型名是**滚动别名**（``deepseek-flash`` 这种）。同一个名字
在不同日期背后可能是不同的权重/构建，而请求里写的就是那个别名——于是"这条回复是
谁答的"在事后无法从别名本身回答。响应元数据里偶尔带 ``model`` /
``system_fingerprint`` / ``build`` / ``version``，那是我们能拿到的最接近构建戳的东西。

**没有就如实记没有。** 多数 OpenAI 兼容端点只回 ``model`` 与 ``id``；本模块不编造
版本号，也不把 ``model`` 当构建戳——它只是别名本身。

日志是一行一个 JSON，字段固定：``at`` / ``model_ref``（请求侧标识）/ ``meta``
（响应侧原样字段）。**只追加**，与 ``sweep_results`` 的保留策略同一条理由：哪一天
是什么版本，不该被后一次运行覆盖掉。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["append_response_meta"]


def append_response_meta(
    path: Path | str,
    *,
    model_ref: str,
    meta: dict[str, Any],
    at: float | None = None,
) -> None:
    """把一次响应的元数据追加进日志。

    Raises:
        OSError: 写不进去。调用方决定怎么让用户知道——**不在这里吞**，静默丢一行
            正是这条日志最不该有的失效方式。
    """
    if not meta:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "at": float(time.time() if at is None else at),
        "model_ref": str(model_ref),
        "meta": dict(meta),
    }
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
