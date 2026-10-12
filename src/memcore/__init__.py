"""memcore —— 记忆核心。

这个包负责三件事，且**只**负责这三件事：

1. **表示**：一条记忆长什么样（:mod:`memcore.types`）、怎么编码成稀疏分布表示
   （:mod:`memcore.sdr`）；
2. **存储与生命周期**：落库、分层、惰性衰减（:mod:`memcore.store`），以及睡眠巩固
   （:mod:`memcore.consolidation`——分簇 + 簇内局部 W 的回放。它和这一层别的东西一样
   **是纯的**：不碰库、不碰盘，改库那一步在 ``axo.memory.consolidation``）；
3. **召回**：混合检索与归因（:mod:`memcore.recall`）——**两条通道**（词法 + 语义），
   加上一条**默认关着**的（沿关联图扩散）。P4b 把第三条接进来了，但
   ``RecallConfig.graph_weight`` 仍然是 0.0，所以"三通道"这句话只在打开那个开关之后
   才是事实——见 README 的「联想通道与邻居面板（P4b）」。

它**不**知道世界上有 Qt、有 HTTP、有大模型。模型侧的 embedding 只能以
``(n, dim)`` 的 numpy 数组形式进来，连同它的 ``space_id`` 一起——:mod:`memcore`
永远不从"模型对象"上取维度或取标识。这条边界由 ``tests/test_layering.py`` 用
``ast`` 遍历 import 语句强制，改不动。

分层依赖（箭头方向 = 允许 import 的方向）::

    axo  ──▶  memcore  ──▶  numpy, biosnn_bus
         (Qt/HTTP)        (纯计算)

反过来不成立。因此 ``memcore`` 可以脱离整个 UI 单独跑基准与单元测试——这正是
"一份召回基准不该需要装 Qt 才能跑"（ADR-0002 的同一理由）。
"""

from __future__ import annotations

from .errors import (
    DimensionMismatchError,
    MemCoreError,
    SDRCapacityError,
    SpaceError,
    SpaceMismatchError,
    SpaceNotFoundError,
    StoreCorruptError,
    StoreError,
)
from .types import (
    DEFAULT_SDR_WIDTH,
    SDR,
    ConsolidationEvent,
    MemoryItem,
    MemoryKind,
    MemorySummary,
    Taint,
    Tier,
    now_ts,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SDR_WIDTH",
    "ConsolidationEvent",
    "DimensionMismatchError",
    "MemCoreError",
    "MemoryItem",
    "MemoryKind",
    "MemorySummary",
    "SDR",
    "SDRCapacityError",
    "SpaceError",
    "SpaceMismatchError",
    "SpaceNotFoundError",
    "StoreCorruptError",
    "StoreError",
    "Taint",
    "Tier",
    "__version__",
    "now_ts",
]

# 本文件刻意只 import 本包内的轻量模块（errors/types 只依赖 numpy），不在包级
# 触发 sdr/store/recall 的导入。这些模块已经存在，用它们的调用方按模块路径显式
# import（``from memcore.store import MemoryStore``）——这样"一次召回的开销"
# 在 import 语句里就是看得见的，而不是藏在一个 ``import memcore`` 后面。
#
# 模块地图（改这里时请一并更新——包文档是索引，索引过期比没有索引更误导）::
#
#     types.py     SDR / MemoryItem / Tier / Taint / MemoryKind / ConsolidationEvent
#     errors.py    全部异常类型
#     text.py      字符分类与切分（编码器与预算器共用）
#     sdr.py       词法 SDR 编码（模型无关）
#     ids.py       ULID 生成（主键 == 排序键）
#     spaces.py    记忆空间的空间标识与登记表
#     store.py     SQLite 持久化：条目、向量、惰性衰减
#     tokens.py    token 估算（系数是实测标定的）
#     audit.py     注入账本的数据结构（含模板化的 explain）
#     recall.py    混合召回：词法 + 语义 → 融合 → MMR → 预算贪心
#     graph.py     关联图 A：关联码的 Jaccard + top-m 邻居（派生缓存，可重建）
#     substrate.py 神经元级 W：AᵀA 的隐含表示（模式补全算子，不物化稠密矩阵）
#     consolidation.py 睡眠巩固：分簇 + 簇内局部 W 回放 + 事件行（纯；不碰库不碰盘）
#
# 规划中：projection.py（稀疏投影）。``persona.py`` 原来也挂在这一行，但人格锚最后
# 落在 ``axo.memory.persona``——它是内容，不是记忆核心的机制。
#
# 上面最后三个（graph / substrate / consolidation）现在**都有消费者**了：图被那条
# 默认关着的召回通道、记忆窗口的邻居面板、以及巩固的分簇读；W 被巩固的簇内回放读。
# 这一行是这几件事里唯一容易被读错的：**"纯层全绿"从来不等于"接线在"**（这个仓吃过
# 一次亏），所以上面那张表只说明"谁在读它"，不说明"它有没有用"——第三通道值不值，
# P4b 量过了，答案是默认不上（README 有一节）。
