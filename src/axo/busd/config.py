"""daemon 的配置。全部有默认值，命令行只为覆盖。"""

from __future__ import annotations

from dataclasses import dataclass

from memcore.types import DEFAULT_SDR_WIDTH

__all__ = ["BusDaemonConfig"]


@dataclass(frozen=True)
class BusDaemonConfig:
    """一次 daemon 的全部可调参数。

    Attributes:
        bus_dim: 总线（融合后）的维度。下游认知核心按它建维度，所以它是契约不是
            优选——见 ADR-0012 的"第二维恒为 bus_dim"。
        dt_ms: 总线时间步。单条记忆占一个时间步，所以这个值目前只影响契约字段。
        projection_density: 稀疏随机投影的连接率。
        seed: 投影矩阵种子。**固定它**，否则同一批记忆两次跑出的发放不同，provenance
            就不可复现。
        batch: 一次 ``step()`` 吃多少条新记忆。上限来自 ``items_page`` 的 1000。
        interval_s: 醒一次的间隔。
        width: 编码器的 SDR 宽度（活跃位是回忆键的宽度，与库里的词法 SDR 一致）。
        dream_enabled / dream_interval_s: 巩固（"做梦"）开关与最小间隔。判定读的是
            ``consolidation/last.json``——与 UI 侧调度器**同一个账本**，所以两边都开着
            也不会同一晚跑两轮。
        emergence_novelty / emergence_floor: 涌现判定的两个阈值，见
            :meth:`~axo.busd.daemon.BusDaemon._novelty`。**这两个数没有实验依据**，
            是"先让它可观测"的初值，不要当成标定过的结论。
        curiosity_enabled: L3「涌现触发」的接口桩开关。**默认关**；打开后 daemon 只把
            proposer 的意图写进事件，不抓取、不写记忆（见 :mod:`axo.busd.curiosity`）。
    """

    bus_dim: int = 128
    dt_ms: float = 1.0
    projection_density: float = 0.1
    seed: int = 0
    batch: int = 64
    interval_s: float = 60.0
    width: int = DEFAULT_SDR_WIDTH
    dream_enabled: bool = True
    dream_interval_s: float = 24 * 3600.0
    emergence_novelty: float = 0.35
    emergence_floor: float = 0.0
    curiosity_enabled: bool = False
