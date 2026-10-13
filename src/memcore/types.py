"""记忆核心的基础数据类型。

这个模块是 ``memcore`` 的词汇表：一条记忆长什么样、它属于哪一层、是谁说的、
巩固时发生了什么、稀疏分布表示怎么装。**不含任何算法**——编码器在
:mod:`memcore.sdr`，存储层在 :mod:`memcore.store`。

设计约束（见 ``docs/ADR-0001`` 与实施计划书第二节）：

* 只依赖 numpy 与 biosnn-bus，不 import Qt、不 import HTTP 客户端、不 import torch；
* 除 :class:`SDR` 与 :class:`MemoryItem` 外全部不可变——召回路径是只读的，
  任何"修改"都应该是返回新对象；
* 时间统一用 **Unix 秒（float）**，不用 datetime。理由：衰减公式 ``exp(-Δt/τ)``
  要的是秒差，datetime 每处都要转一次且引入时区语义；而落库时再格式化成可读
  时间即可。唯一例外是 :attr:`MemoryItem.created_at` 的首选来源 ``time.time()``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from .errors import SDRCapacityError

__all__ = [
    "ConsolidationEvent",
    "MemoryItem",
    "MemoryKind",
    "MemorySummary",
    "SDR",
    "Taint",
    "Tier",
    "now_ts",
]

# SDR 的默认宽度。4096 是计划书 §4.3 的选择，两个尺度（词法/语义）共用同一 N，
# 这样它们的 Jaccard 才有可比性、MMR 去冗才能跨通道工作。
DEFAULT_SDR_WIDTH = 4096


def now_ts() -> float:
    """当前 Unix 时间戳（秒，浮点）。

    单点封装是为了让测试可以 monkeypatch 它来"快进时间"，而不必去动
    :mod:`time` 模块本身——衰减与巩固的测试都需要时间旅行。
    """
    return time.time()


# ── 枚举 ────────────────────────────────────────────────────────────────────


class Tier(str, Enum):
    """记忆的分层。

    四层不是四份存储，而是同一张表上的 ``tier`` 列 + 权重动态（计划书 §3.3）。
    继承 :class:`str` 让 SQLite 可以直接存、YAML 可以直接读，不必写转换表。

    层级顺序有意义：:attr:`WORKING` 最新鲜，:attr:`COLD` 是最冷的终态——不删除，
    但不再参与召回。
    """

    WORKING = "working"
    RECENT = "recent"
    LONGTERM = "longterm"
    ARCHIVE = "archive"
    COLD = "cold"

    def __str__(self) -> str:
        return self.value

    @property
    def half_life_days(self) -> float:
        """该层的记忆半衰期（天）。用于惰性衰减 ``w ← w·exp(-Δt/τ)``。

        这些是**默认值**，真实的半衰期由 ``preference.yaml`` 的
        ``forgetting.half_life_days`` 覆盖——把可调的遗忘策略放在纯数据文件里，
        而不是散落在代码常量中。
        """
        return _HALF_LIFE_DAYS[self]


_HALF_LIFE_DAYS: dict[Tier, float] = {
    Tier.WORKING: 0.08,  # 约 2 小时：工作记忆只活一次会话
    Tier.RECENT: 10.0,
    Tier.LONGTERM: 100.0,
    Tier.ARCHIVE: 1000.0,
    Tier.COLD: float("inf"),  # 不再衰减，也不再被召回
}


class Taint(str, Enum):
    """记忆的来源污点。

    命名借自信息流安全：它标记的是"这段内容从哪来"，而不是内容本身好坏。用途
    有两个——salience 规则按来源打分（用户长陈述比他自己的工具输出更值得记），
    以及在装配上下文时区分可信度。

    继承 :class:`str` 的理由同 :class:`Tier`。
    """

    USER = "user"
    MODEL = "model"
    TOOL = "tool"
    SYSTEM = "system"

    def __str__(self) -> str:
        return self.value


class MemoryKind(str, Enum):
    """记忆的语义类别。

    与 :class:`Taint` 正交：taint 回答"谁说的"，kind 回答"这是什么"。salience
    规则表同时用这两个维度来打分。
    """

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    CONCLUSION = "conclusion"
    OPEN_QUESTION = "open_question"
    TOOL_RESULT = "tool_result"
    PERSONA_ANCHOR = "persona_anchor"
    """人格锚：内容是纯文本，``pinned=True``，永远进上下文，不参与常规召回。"""

    def __str__(self) -> str:
        return self.value


# ── 稀疏分布表示 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class SDR:
    """一次稀疏分布表示（Sparse Distributed Representation）。

    **存索引而不是存稠密数组**，这是本类存在的全部理由。一条记忆的
    ``lex_sdr`` 有 573 个活跃位，稠密 float32 是 16 KB；存成 int16 索引 + float32
    权重是 3.4 KB。全库 2 万条记忆时，差额是 320 MB 对 69 MB——而召回时真正
    要算的是集合交并，索引数组比稠密向量更快。

    相等性按值比较（``idx`` 与 ``weights`` 逐元素），但**按身份哈希**，因此不能
    用作字典键或放进 ``set``——与 :class:`biosnn_bus.SpikeTrain` 同样的取舍，
    理由也一样：numpy 数组给不出稳定的值哈希。

    Attributes:
        idx: ``(k,)`` 的活跃位索引，int32，**升序且无重复**。升序是硬要求：
            集合交并用 :func:`numpy.intersect1d` 之类的有序算法，乱序会静默
            算错交集的势。
        weights: ``(k,)`` 的响应强度，float32，落在 ``(0, 1]``。
        width: 表示宽度 ``N``（总位数）。
    """

    idx: np.ndarray
    weights: np.ndarray
    width: int = DEFAULT_SDR_WIDTH

    def __post_init__(self) -> None:
        idx = np.asarray(self.idx)
        weights = np.asarray(self.weights)

        if idx.ndim != 1 or weights.ndim != 1:
            raise SDRCapacityError(
                f"SDR.idx 与 SDR.weights 都必须是 1 维，收到 {idx.shape} 与 {weights.shape}。"
            )
        if idx.shape != weights.shape:
            raise SDRCapacityError(
                f"SDR.idx 与 SDR.weights 长度必须一致，收到 {idx.size} 与 {weights.size}。"
            )
        width = int(self.width)
        if width < 1:
            raise SDRCapacityError(f"SDR.width 必须 >= 1，收到 {self.width!r}。")
        if idx.size > width:
            raise SDRCapacityError(
                f"SDR 活跃位数 {idx.size} 超过了表示宽度 {width}——同一位不可能出现两次。"
            )
        if idx.size == 0:
            raise SDRCapacityError("SDR 不允许全零：没有任何活跃位的表示无法参与任何相似度计算。")
        if idx.min() < 0 or idx.max() >= width:
            raise SDRCapacityError(
                f"SDR.idx 越界：值域应为 [0, {width})，收到 [{int(idx.min())}, {int(idx.max())}]。"
            )
        if not np.all(np.diff(idx) > 0):
            raise SDRCapacityError(
                "SDR.idx 必须严格升序且无重复。乱序会让集合交并按有序算法算错——"
                "这是静默错误，所以在这里就拦掉。"
            )

        object.__setattr__(self, "idx", idx.astype(np.int32, copy=False))
        object.__setattr__(self, "weights", weights.astype(np.float32, copy=False))
        object.__setattr__(self, "width", width)

    # ---------------------------------------------------------------- 几何信息

    @property
    def k(self) -> int:
        """活跃位数。"""
        return int(self.idx.size)

    @property
    def density(self) -> float:
        """活跃位占比 ``k / N``。"""
        return self.k / self.width

    def to_dense(self) -> np.ndarray:
        """展开成 ``(N,)`` 的 float32 稠密向量。

        只在必须走线性代数的地方用（例如与投影矩阵相乘），不要用它来存或比。
        """
        dense = np.zeros(self.width, dtype=np.float32)
        dense[self.idx] = self.weights
        return dense

    def to_spike_train(self, dt: float = 1.0) -> Any:
        """转成 ``biosnn_bus.SpikeTrain``，形状 ``(1, N)``。

        这是与骨架库的接口点：SDR 是记忆核心内部的紧凑表示，跨出到总线时展开
        成脉冲序列。**注意不要去调 ``.binary()`` 的默认阈值 0.5**——SDR 的权重
        值域是 ``(0, 1]``，0.5 会砍掉大半；要二值就直接用 ``idx``。
        """
        from biosnn_bus import SpikeTrain  # 延迟 import：只有真的跨界时才付这个代价

        return SpikeTrain(data=self.to_dense()[None, :], dt=dt)

    # ---------------------------------------------------------------- 相似度

    def jaccard(self, other: SDR) -> float:
        """活跃位集合的 Jaccard 相似度 ``|A∩B| / |A∪B|``。

        **刻意忽略权重**。权重承载的是"这个位被激活得多强"，在比较两条记忆是否
        在说同一件事时是噪声；而且语义 SDR 的权重依赖当前 embedding 模型，用了
        它就等于把相似度重新绑回模型。用集合运算则词法通道与语义通道的读法一致。
        """
        if self.width != other.width:
            raise SDRCapacityError(
                f"两个 SDR 的宽度不同（{self.width} 与 {other.width}），无法比较。"
            )
        # idx 已保证升序，用 sorted 版交集算法，O(k_A + k_B)
        inter = np.intersect1d(self.idx, other.idx, assume_unique=True).size
        union = self.k + other.k - inter
        return float(inter / union) if union else 0.0

    # ---------------------------------------------------------------- 展示

    def __eq__(self, other: object) -> bool:
        """按值相等。手写而不用 dataclass 自动生成的版本，理由同
        :class:`biosnn_bus.SpikeTrain`：自动生成的会返回数组而不是布尔值。"""
        if not isinstance(other, SDR):
            return NotImplemented
        return (
            self.width == other.width
            and self.idx.shape == other.idx.shape
            and bool(np.array_equal(self.idx, other.idx))
            and bool(np.array_equal(self.weights, other.weights))
        )

    def __repr__(self) -> str:
        return (
            f"SDR(N={self.width}, k={self.k}, density={self.density:.4g}, "
            f"w=[{float(self.weights.min()):.3g}, {float(self.weights.max()):.3g}])"
        )


# ── 记忆条目的摘要 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemorySummary:
    """列表视图用的一条记忆——**刻意不带 SDR 与 meta**。

    存在的理由是一个具体的度量：:meth:`MemoryStore.iter_items` 会把每条记忆的
    ``lex_idx``/``lex_w`` 反序列化成两个 numpy 数组（各 573 个元素，约 6.9 KB）。
    两万条就是 130 MB 的 Python 对象，而**记忆界面的列表一个字节都不显示它们**。
    浏览是"找到那一条"，不是"算相似度"——后者是召回的事，有自己的路径。

    所以这个类只装列表与详情标题需要的东西。要看完整条目（meta、SDR、来源）
    就按 id 走 :meth:`MemoryStore.get`：那是单行读，代价可忽略。
    """

    memory_id: str
    kind: MemoryKind
    tier: Tier
    content: str
    weight: float
    pinned: bool
    created_at: float
    last_access_ts: float
    access_count: int
    session_id: str
    source_ref: str

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 24 else self.content[:24] + "…"
        return (
            f"MemorySummary({self.memory_id[:8]}…, {self.kind.value}, {self.tier.value}, "
            f"w={self.weight:.3f}, {preview!r})"
        )


# ── 记忆条目 ────────────────────────────────────────────────────────────────


@dataclass
class MemoryItem:
    """一条记忆。

    与 :class:`SDR` 不同，这个类是**可变的**（``frozen=False``）：权重更新与访问
    计数是每轮都在发生的高频操作，每次造一个新对象会让召回路径产生大量短命
    对象。可变的责任由 :mod:`memcore.store` 承担——它是唯一允许原地改这些字段
    的地方，外部拿到的是拷贝。

    Attributes:
        memory_id: ULID 字符串。**构造上按时间单调递增且随机**，所以它同时是
            主键、排序键和"零串扰的联想图节点 id"。
        kind: 语义类别。
        tier: 当前分层。
        content: 记忆正文。人格锚也在里面（``kind=PERSONA_ANCHOR``）。
        taint: 来源。
        created_at: 创建时间（Unix 秒）。
        last_access_ts: 最后一次被召回的时间。衰减以它为起点。
        last_decay_ts: 最后一次结算衰减的时间。惰性衰减的记账点——**只在读该条
            时按这两个时间戳补算权重**，因此不需要任何后台守护进程。
        access_count: 被召回的次数。
        reinforce_count: 在睡眠巩固中被"回放且认得出来"的次数。与
            ``access_count`` 分开记：前者是被外部问到时想起来，后者是自己睡着
            时还想得起来——后者才是长期稳固的证据。
        weight: 综合权重 ``[0, 1]``。晋升/降级/召回的调制都看它。
        pinned: 钉住。跳过衰减与降级，永远参与召回（人格锚必然是 pinned）。
        lex_sdr: 词法 SDR。**模型无关**，是关联基质与 MMR 去冗的地基。
        assoc_sdr: 关联用的低 k 稀疏 SDR，只喂神经元级基质 W 与关联图 A。它是
            :attr:`lex_sdr` 的**纯函数**（:func:`~memcore.sdr.encode_assoc`）——
            与 embedding 空间无关，所以写路径留空它、由消费者现算，比存一份派生值
            便宜（理由在 ``memory/writer.py`` 的写入处）。
        sem_sdr: 语义 SDR，属于某个 embedding space。换模型后这一项会失效，
            而 :attr:`lex_sdr` 不会——**这是"模型依赖性"那条轴**，别与
            :attr:`assoc_sdr` 那档"同源不同稀疏度"混起来：后者不依赖任何模型。
        space_id: :attr:`sem_sdr` 所属的记忆空间。空字符串表示尚未回填 embedding
            （实时写入时 embedding 是异步补的）。
        session_id: 产生该记忆的会话。空字符串表示跨会话的（如人格锚）。
        source_ref: 溯源指针，如 ``"turn://01J.../2"`` 或文件路径。
        meta: 其它结构化附加信息（归因快照、偏好命中的规则等）。**不进
            :meth:`__eq__`**——它是自由格式的，比较它没有意义。
    """

    memory_id: str
    kind: MemoryKind
    tier: Tier
    content: str
    taint: Taint = Taint.USER
    created_at: float = field(default_factory=now_ts)
    last_access_ts: float = 0.0
    last_decay_ts: float = 0.0
    access_count: int = 0
    reinforce_count: int = 0
    weight: float = 0.5
    pinned: bool = False
    lex_sdr: SDR | None = None
    assoc_sdr: SDR | None = None
    sem_sdr: SDR | None = None
    space_id: str = ""
    session_id: str = ""
    source_ref: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("MemoryItem.memory_id 不能为空。")
        if not self.content:
            # 空内容记忆在上下文装配里表现为"注入了一条空白"，既费 token 又难排查
            raise ValueError(f"MemoryItem {self.memory_id} 的 content 不能为空。")
        self.weight = _clip01(self.weight)
        # 新建时把两个时间戳对齐到 created_at：否则惰性衰减第一次结算时会算出
        # 一个巨大的 Δt（从 epoch 到 now），把权重直接衰减到 0。
        if self.last_decay_ts <= 0.0:
            self.last_decay_ts = self.created_at
        if self.last_access_ts <= 0.0:
            self.last_access_ts = self.created_at

    # ---------------------------------------------------------------- 派生量

    @property
    def is_persona_anchor(self) -> bool:
        """是否是人格锚。人格锚跳过常规召回，只走 L0 槽。"""
        return self.kind is MemoryKind.PERSONA_ANCHOR

    @property
    def has_embedding(self) -> bool:
        """语义通道可用的前提——**指的是内存里这条**。

        历史上这里写的是 ``self.sem_srd``（字段叫 ``sem_sdr``），一读就是
        AttributeError：一个"语义通道可用的前提"从来没被求过值。拼写修了，但要
        知道它的**作用域**：库里读出来的条目 ``sem_sdr`` 恒为空（向量存在另一张
        ``embeddings`` 表里，从不回写 ``memories`` 行），所以"库里这条有没有向量"
        仍然不能问它——那要问
        :meth:`memcore.store.MemoryStore.missing_embedding_ids`。它回答的是写入侧
        手里那条（内存态）有没有向量。
        """
        return bool(self.space_id) and self.sem_sdr is not None

    def decayed_weight(self, at: float, *, half_life_days: float) -> float:
        """结算惰性衰减后的权重（只算，不写回——写回是 store 的事）。

        用半衰期而不是时间常数 τ：写 ``half_life_days`` 时不必解释"τ 与半衰期差
        ln2"，调参时也不会忘掉那个因子。
        """
        if self.pinned or half_life_days == float("inf"):
            return self.weight
        if at <= self.last_decay_ts:
            return self.weight
        elapsed_days = (at - self.last_decay_ts) / 86400.0
        factor = float(np.exp(-np.log(2.0) * elapsed_days / half_life_days))
        return _clip01(self.weight * factor)

    def copy(self, **changes: Any) -> MemoryItem:
        """返回替换了部分字段的副本。"""
        return replace(self, **changes)

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 24 else self.content[:24] + "…"
        return (
            f"MemoryItem({self.memory_id[:8]}…, {self.kind.value}, {self.tier.value}, "
            f"w={self.weight:.3f}, {preview!r})"
        )


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


# ── 巩固事件 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsolidationEvent:
    """睡眠期巩固时的一条决策记录。

    这**不是日志**，是可重放的证据。每次巩固有一个 ``run_id``，把该轮所有事件
    按序落库；``tools/sleep_once.py --replay <run_id>`` 能用同一个 ``run_seed``
    重跑一遍并逐条比对。做不到这一点的话，"巩固"就是一个不可审计的黑盒——而
    本项目的整个卖点就是可审计。

    Attributes:
        run_id: 该轮巩固的标识，由 ``blake2b(base_seed|sleep|run_ts)`` 派生。
        memory_id: 被回放的记忆。
        decision: 决策类型，取值见下表。

            ======================  ==========================================
            ``reinforce``           吸引子收敛到自身，加固
            ``sleep_replay_miss``   收敛到了空或别处，吸引域太窄
            ``sleep_confusion``     收敛到了**另一条已存在的记忆**——串扰
            ``skip``                抽样命中但状态不允许回放（如已 cold）
            ``promote`` / ``demote`` 越过层级阈值
            ======================  ==========================================

        at: 事件时间（Unix 秒）。
        epoch: 属于该轮的哪个 epoch（``0..n_epochs-1``）。
        jaccard: 回放结果与原表示的 Jaccard 相似度；不适用时为 ``None``。
        confused_with: ``decision == "sleep_confusion"`` 时指向那条被混淆的记忆。
        detail: 其它证据（加固前后权重、命中规则名等）。
    """

    run_id: str
    memory_id: str
    decision: str
    at: float
    epoch: int = 0
    jaccard: float | None = None
    confused_with: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
