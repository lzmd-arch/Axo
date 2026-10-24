"""把一条记忆的关联码编码成脉冲序列的插件。

**它编码什么**：``memcore.sdr.encode_assoc(lex_sdr)`` 的结果——一个 64 活跃位的
关联码（模式补全用的那一档，不是 573 位的词法 SDR）。一批新记忆就是 ``(T, width)``
的一张表，每个时间步一条记忆。

**它不做什么**：不学习、不改权重、不把"发放"解释成语义。``SpikeBus`` 的输出是
稀疏随机投影后的浮点电流（ADR-0012），这里出去的也是浮点——要 0/1 就显式
``.binary()``，别把浮点电流说成"脉冲发放"。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from biosnn_bus import ModalityPlugin, SpikeTrain
from biosnn_bus.membrane import PassThroughMembrane

from memcore.sdr import encode_assoc
from memcore.types import DEFAULT_SDR_WIDTH, SDR

__all__ = ["MemorySpikePlugin"]


class MemorySpikePlugin(ModalityPlugin):
    """记忆关联码 → ``(T, width)`` 的 ``SpikeTrain``。

    ``T`` 是本批的记忆条数：一条记忆占一个时间步。这是一个**约定**，不是时间
    仿真——本插件不展开 STDP 窗口、不做编码时间常数。把它当真时间轴来用会误读。

    Args:
        width: SDR 宽度，必须等于总线上注册时声明的 ``spike_dim``。
        dt: 交给 ``SpikeTrain`` 的时间步（毫秒）。
        k: 关联码的活跃位数。
    """

    def __init__(self, *, width: int = DEFAULT_SDR_WIDTH, dt: float = 1.0, k: int = 64) -> None:
        self._width = int(width)
        self._dt = float(dt)
        self._k = int(k)

    # ------------------------------------------------------------ 契约

    @property
    def modality_name(self) -> str:
        return "memory"

    @property
    def spike_dim(self) -> int:
        return self._width

    @property
    def fusion_channel(self) -> str:
        # 关联码是"这条记忆像什么"，不是"什么时候发生的"——走语义通道。
        return "semantic"

    @property
    def temporal_scale(self) -> float:
        return 10.0

    def get_membrane(self) -> PassThroughMembrane:
        """恒等膜。骨架库不定义学习规则，这里也不假装有。"""
        return PassThroughMembrane()

    # ------------------------------------------------------------ 编码/解码

    def encode(self, raw_input: Any) -> SpikeTrain:
        """``SDR`` / ``SDR`` 序列 / ``(T, width)`` 数组 → ``SpikeTrain``。"""
        rows = self._rows(raw_input)
        return SpikeTrain(
            data=rows, dt=self._dt, temporal_scale=self.temporal_scale, channel=self.channel
        )

    def decode(self, spike_output: SpikeTrain) -> list[np.ndarray]:
        """取每个时间步最强的 ``k`` 个位（升序下标）。

        与 :meth:`encode` 不必互逆：关联码经投影/融合之后已经不在原空间里了。
        """
        data = np.asarray(spike_output.data)
        k = min(self._k, data.shape[1])
        order = np.argsort(data, axis=1)[:, -k:]
        return [np.sort(row) for row in order]

    # ------------------------------------------------------------ 内部

    def _rows(self, raw: Any) -> np.ndarray:
        if isinstance(raw, SDR):
            return np.asarray(encode_assoc(raw, k=self._k).to_dense(), dtype=np.float64)[None, :]
        if isinstance(raw, np.ndarray):
            arr = raw.astype(np.float64, copy=False)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[1] != self._width:
                raise ValueError(f"数组形状必须是 (T, {self._width})，收到 {arr.shape}。")
            return np.ascontiguousarray(arr)
        if isinstance(raw, Sequence):
            rows = [
                encode_assoc(s, k=self._k).to_dense() if isinstance(s, SDR) else np.asarray(s)
                for s in raw
            ]
            if not rows:
                raise ValueError("不能编码空批次：step() 至少需要一个时间步。")
            arr = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
            if arr.shape[1] != self._width:
                raise ValueError(f"每条记忆的宽度必须是 {self._width}，收到 {arr.shape[1]}。")
            return arr
        raise TypeError(f"不认识的输入类型：{type(raw).__name__}。")
