"""神经元级基质 W —— 关联码上的自联想记忆。

W 是 Hebbian 外积和 ``W = Σ_m a_m a_mᵀ``，其中 ``a_m`` 是第 m 条记忆的**关联码**
（:func:`memcore.sdr.encode_assoc`，k=64）的稠密形态。它的用途是**模式补全**：
给一个残缺的关联码，看网络的一步迭代会把哪一条记忆激活得最强。

## 为什么不物化那个 N×N 矩阵

``N = 4096`` 时稠密 float32 是 64 MiB，而 ``pyproject.toml`` 里"matvec 约 2ms"
那句是照着它的规模写的。但物化是纯浪费：

* ``W = AᵀA``，所以 ``rank(W) ≤ n``（n 是记忆条数）。2 万条时秩远小于 4096，
  那个矩阵的绝大部分自由度是零。
* ``W`` 的行空间 ⊆ 活跃位张成的空间，而活跃位是稀疏的。
* 真正要算的只有 ``W p = Aᵀ(A p)``，它只需要 ``A`` 的两个 ``(n, k)`` 数组：
  n=2e4 时约 10 MB，n=5e4 时约 26 MB。

所以这个类**只存 A**，``respond`` 现算那两步。需要真的做 N×N matvec 时，这个
隐含表示与物化形态在数学上完全等价——只是前者不付那 64 MiB。

## 一段必须先说的负面预期

经典 Hopfield 网络的容量约 ``0.138·N``。N=4096 → **约 570 条模式**。而本仓的
关联码**相互相关**（相似文本共享位），相关模式正是把吸引子容量压得更低的东西。
所以在 5 万条的真实规模上，**全局 Hebbian W 的一步补全大概率不可靠**。

这不是放弃 W 的理由，而是**先把它量出来**的理由——``tools/bench_substrate.py``
的 ``--scale`` 扫描就是干这个的：把容量悬崖的位置看出来，别只在 200 条的语料上
看到一个乐观到骗人的数。计划书 §4.4 那些数字（突触从 10,843 涨到 201,394、
知识图谱 64 节点 124 边）描述的是"一次对话 + 一夜睡眠"的规模，与本终端的 5 万条
不是一个量级。**这个量级落差就是这个模块存在的前提。**

## 与 ``biosnn_bus`` 的关系，说实话

:meth:`Substrate.to_spike` 是骨架库在本仓的**唯一运行时消费者**——探针在
:meth:`Substrate._check_probe` 里过一次 ``SpikeTrain``（校验形状、回读 ``rates``）。
但要说清楚它是什么：
``biosnn_bus`` 里只有 :class:`~biosnn_bus.SpikeTrain`、``Membrane``、
``FusionChannel`` 三个**类型**，没有任何网络、没有任何计算可以"接线"上去。
所以 ``to_spike`` 的产物是一个**类型级的边界对象**，不是一次仿真，也不代表有
什么东西在真的跑脉冲。把它说成"接上总线了"就是骗人。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import SDRCapacityError
from .sdr import ASSOC_K, encode_assoc
from .types import DEFAULT_SDR_WIDTH, SDR, MemoryItem

__all__ = ["Substrate", "SubstrateConfig"]


@dataclass(frozen=True)
class SubstrateConfig:
    """建基质时的参数。跟着关联码的编码参数走。

    Attributes:
        k: 关联码的活跃位数。默认 :data:`~memcore.sdr.ASSOC_K`。
        mode: 关联码的选择规则，透传给 :func:`~memcore.sdr.encode_assoc`。
    """

    k: int = ASSOC_K
    mode: str = "hash"


_DEFAULT_CONFIG = SubstrateConfig()
"""``build`` 的默认参数。**做成模块级单例而不是写在签名里**（``config=SubstrateConfig()``）：
ruff 的 B008 会拦后者，而按它的建议每次调用现建一个也没什么好处——这个类是 frozen
的，共享一个实例不会串。"""


@dataclass(frozen=True, eq=False)
class Substrate:
    """按关联码建起来的自联想基质。**只存 A，不存 W。**

    用 ``eq=False`` 是为了自己定义 :meth:`__eq__`——dataclass 生成的版本会比较
    numpy 数组并返回一个数组，那是个只在 ``if`` 里才炸的坑。

    Attributes:
        ids: ``(n,)`` 的 memory_id，**顺序与 :attr:`idx` / :attr:`weights` 的行
            一一对应**。模式补全的返回值就是从这里取的。
        idx: ``(n, k)`` 的关联码位索引。短于 k 的条目在尾部补 0。
        weights: ``(n, k)`` 的关联码权重。**补齐位的权重是 0.0**，所以在内积与
            累加里都不贡献——这让变长的关联码可以装进一个矩形数组。
        width: 表示宽度 N。
    """

    ids: tuple[str, ...]
    idx: np.ndarray
    weights: np.ndarray
    width: int = DEFAULT_SDR_WIDTH

    def __post_init__(self) -> None:
        idx = np.ascontiguousarray(self.idx, dtype=np.int32)
        weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        if idx.ndim != 2 or weights.ndim != 2:
            raise SDRCapacityError(
                f"基质的 idx 与 weights 都必须是 2 维 (n, k)，收到 {idx.shape} 与 {weights.shape}。"
            )
        if idx.shape != weights.shape:
            raise SDRCapacityError(
                f"基质的 idx 与 weights 形状必须一致，收到 {idx.shape} 与 {weights.shape}。"
            )
        if idx.shape[0] != len(self.ids):
            raise SDRCapacityError(
                f"基质有 {idx.shape[0]} 行，却给了 {len(self.ids)} 个 id——两者必须一一对应。"
            )
        if int(self.width) < 1:
            raise SDRCapacityError(f"基质宽度必须 >= 1，收到 {self.width!r}。")
        if idx.size and (idx.min() < 0 or idx.max() >= int(self.width)):
            raise SDRCapacityError(
                f"关联码的位索引越界：值域应为 [0, {int(self.width)})，"
                f"收到 [{int(idx.min())}, {int(idx.max())}]。"
            )
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "width", int(self.width))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Substrate):
            return NotImplemented
        return (
            self.ids == other.ids
            and self.width == other.width
            and np.array_equal(self.idx, other.idx)
            # 并列的位权重在补齐处是 0，逐位比较比 allclose 更适合"可复现"这个用途。
            and np.array_equal(self.weights, other.weights)
        )

    def __len__(self) -> int:
        """基质里有多少条模式。"""
        return len(self.ids)

    @property
    def n(self) -> int:
        """``len(self)`` 的别名，写公式时更顺手。"""
        return len(self.ids)

    # ── 构建 ────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        items: Iterable[MemoryItem],
        *,
        config: SubstrateConfig = _DEFAULT_CONFIG,
    ) -> Substrate:
        """从记忆条目建基质。

        **关联码在这里现算，不从条目上读。** :attr:`~memcore.types.MemoryItem.assoc_sdr`
        若有值就用它（调用方可能已经算过），否则由 ``item.lex_sdr`` 现推——因为关联码
        是词法码的纯函数，存一份派生值是在浪费磁盘，而重算的代价是 64 次哈希。

        **没有 ``lex_sdr`` 的条目被跳过**，不是报错：还没回填编码的条目在实时写入
        路径上是常态（embedding 异步补），让它们把整个基质建不起来是本末倒置。

        宽度不一致会抛错：不同宽度的关联码拼不成一个矩阵，而"悄悄取最小的那个宽度"
        会让越界的位静默消失。

        Raises:
            SDRCapacityError: 条目的关联码宽度不一致。
        """
        ids: list[str] = []
        rows: list[SDR] = []
        width = 0
        for item in items:
            sdr = item.assoc_sdr
            if sdr is None:
                if item.lex_sdr is None:
                    continue
                sdr = encode_assoc(item.lex_sdr, k=config.k, mode=config.mode)
            if width == 0:
                width = sdr.width
            elif sdr.width != width:
                raise SDRCapacityError(
                    f"条目 {item.memory_id} 的关联码宽度是 {sdr.width}，"
                    f"而前面的是 {width}——不同宽度的关联码拼不成一个矩阵。"
                )
            ids.append(item.memory_id)
            rows.append(sdr)

        if not rows:
            empty = np.zeros((0, 0), dtype=np.int32)
            return cls(
                ids=(),
                idx=empty,
                weights=empty.astype(np.float32),
                width=width or DEFAULT_SDR_WIDTH,
            )

        # 补齐到最宽的那一行。补齐位的权重是 0.0，所以内积与累加都不受影响——
        # 这比维护一个变长列表便宜得多，也让整个类保持纯数组运算。
        k_max = max(row.k for row in rows)
        idx = np.zeros((len(rows), k_max), dtype=np.int32)
        weights = np.zeros((len(rows), k_max), dtype=np.float32)
        for row_index, row in enumerate(rows):
            if row.k:
                idx[row_index, : row.k] = row.idx
                weights[row_index, : row.k] = row.weights
        return cls(ids=tuple(ids), idx=idx, weights=weights, width=width)

    # ── 推理 ────────────────────────────────────────────────────────────────

    def _check_probe(self, probe: SDR) -> np.ndarray:
        """宽度校验 + **从骨架库的类型边界进来**。

        ``SpikeTrain`` 在构造时校验 ``(T, N)`` 形状，``rates`` 回读逐神经元发放率
        ——与 :meth:`~memcore.types.SDR.to_dense` 是同一组数（实测逐位相等，只有
        dtype 不同，转回 float32 即可）。

        **这是 ``biosnn_bus`` 在本仓唯一的运行时消费者，而它带来的是一次校验、
        不是一次计算**——那个库里没有网络（见模块文档「与 biosnn_bus 的关系」）。
        把它放在探针这一侧是因为一次 ``complete`` 只过一条探针，而不是逐行过。
        """
        if probe.width != self.width:
            raise SDRCapacityError(
                f"探针的宽度是 {probe.width}，基质的是 {self.width}——"
                f"不同宽度的位索引不是同一套坐标。"
            )
        return np.asarray(self.to_spike(probe).rates, dtype=np.float32)

    def respond(self, probe: SDR) -> np.ndarray:
        """一步迭代 ``W p``，返回 ``(N,)`` 的稠密激活。

        做法就是 ``W p = Aᵀ(A p)`` 那两步，全程只用 ``(n, k)`` 的两个数组：

        1. ``s = A p``：每条模式与探针的内积，``einsum`` 一次算完。
        2. ``Aᵀ s``：把每条模式按它的强度加回到它自己的那些位上。

        第 2 步用 :func:`numpy.bincount` 而不是 ``np.add.at``：两者都正确处理
        "同一个位被多行重复累加"，但前者快一个量级，而 ``--scale 20000`` 那一档
        会把这个函数调到 2e4×64 次写入。

        **注意返回值是稠密的 (N,)**：这一步不引入稀疏性，也没有阈值化——判读
        "谁被激活了"是调用方的事（见 :meth:`complete`）。
        """
        dense = self._check_probe(probe)
        if self.n == 0:
            return np.zeros(self.width, dtype=np.float32)

        strengths = np.einsum("ij,ij->i", dense[self.idx], self.weights)
        contributions = self.weights * strengths[:, None]
        return np.bincount(
            self.idx.ravel(),
            weights=contributions.ravel().astype(np.float64),
            minlength=self.width,
        ).astype(np.float32)

    def complete(self, probe: SDR, *, top: int = 5) -> list[tuple[str, float]]:
        """模式补全：哪几条记忆最能解释这个探针。按得分降序。

        **得分不是相似度，是重叠能量** ``a_m · (W p)``——它同时受"这条有多像探针"
        与"这条有多强"影响，量纲随语料规模增长。所以它只该用来**排序**，不要拿
        绝对值去设阈值。

        Args:
            probe: 探针 SDR。典型用法是把某条记忆的关联码确定性丢掉一部分活跃位
                （``tools/bench_substrate.py`` 的 ``--drop``）。
            top: 返回几条。

        Returns:
            ``[(memory_id, score), ...]``，可能短于 ``top``（基质里条目不够时）。

        Raises:
            ValueError: ``top < 1``。
        """
        if top < 1:
            raise ValueError(f"top 必须 >= 1，收到 {top!r}。")
        activated = self.respond(probe)
        scores = np.einsum("ij,ij->i", activated[self.idx], self.weights)
        # 并列时按行号（即 ids 的顺序）稳定地打破，保证同一输入永远给同一排名。
        order = np.argsort(-scores, kind="stable")[: min(top, self.n)]
        return [(self.ids[int(m)], float(scores[int(m)])) for m in order]

    def to_spike(self, pattern: SDR) -> Any:
        """把一个 SDR 转成 ``biosnn_bus.SpikeTrain``。**这是类型边界，不是仿真。**

        实现就是转发 :meth:`~memcore.types.SDR.to_spike_train`，没有加任何基质
        语义。它出现在这里，是因为基质的输出（一个位模式）最终要以骨架库认得的
        形态交出去——而在此之前，那个接口在本仓**一个调用者都没有**。

        读这段代码的人容易把"能转成 SpikeTrain"读成"接上脉冲网络了"。**没有。**
        骨架库里没有网络，转出来的只是一个携带 ``(1, N)`` 数据的类型对象；真正
        的仿真在它里面还没有实现。要在这条路径上做仿真，先得有一个网络。
        """
        return pattern.to_spike_train()
