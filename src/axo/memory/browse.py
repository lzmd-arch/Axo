"""记忆浏览器的**数据与文本**部分：列表怎么筛、详情怎么排版、线索召回怎么读。

这个模块不 import Qt——它拿 :class:`MemorySummary` 与 :class:`MemoryItem` 进来，
吐字符串出去。分开的理由是界面里最难测的就是字符串拼装：一旦它住在 ``QWidget``
里，「这句提示到底写没写对」就只能靠人眼看。放这里就能断言。

四条纪律，写在它们各自会发生的地方：

1. **没数据就说没数据，有数据就给真数。** 这是**同一条**纪律的两面，而它们各自
   会坏在不同的方向上：

   * :func:`render_heat_placeholder` 里的「未启用」占位不许出现任何像数字的东西
     ——一个 0.87 会让它看起来像真的算过。
   * :func:`neighbor_label` 里的边权与记忆权重是**真的算出来的**，所以照实写，
     不许为了"看起来干净"换成「较强」这种词：那会丢掉"它到底多强"这个唯一能
     拿来判断的信息。

   两面合起来才是那个边界：**无数据 → 零数字；有数据 → 真数字。** 只钉一面的话，
   另一面的失效（占位里塞 0.00，或者把真数抹成形容词）没有任何测试会红。
2. **归因不编。** 一条记忆不在本轮账本里时，:func:`render_memory_detail` 只说
   「不在本轮账本里」并列出**可能的**原因，不挑一个说得通的当结论。而且钉住的
   条目与本会话条目要给**不同**的说明：它们在 recall 里是两条不同的早退路径
   （``pinned`` 与 ``excluded``），合并成一句话就等于编。
3. **探索与生产要分开标注。** 线索召回用的是另一组参数（阈值更低、条数更多，见
   :data:`PROBE_CONFIG`），它回答的是「以它为线索，附近还有谁」，**不是**「它当初
   为什么被注入」。不标注的话，用户会把探索排名读成生产排名，账本的可解释性承诺
   就自己破了。
4. **图是缓存，界面要说出它的年纪。** :func:`association_header` 与
   :class:`NeighborRow` 报的是**建图那一刻**的邻居：库里删过东西之后，图里还留着
   指向已删除节点的边。所以这里既不过滤掉那些边（过滤掉就看不出"图旧了"），也不
   把它们画成正常邻居（画成正常就等于指向一条不存在的记忆）——标出来、说清楚。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from memcore import MemoryItem, MemoryKind, MemorySummary, Tier
from memcore.audit import ChannelScores, RecallLedger
from memcore.graph import GraphIndex
from memcore.recall import RecallConfig
from memcore.spaces import Space

from .forget import read_forget_record

__all__ = [
    "PROBE_BUDGET_TOKENS",
    "PROBE_CONFIG",
    "MemoryFilter",
    "MemoryStatus",
    "NeighborRow",
    "association_header",
    "association_rows",
    "filter_summaries",
    "format_ts",
    "kind_label",
    "neighbor_label",
    "probe_position",
    "render_heat_placeholder",
    "render_heat_summary",
    "render_memory_detail",
    "summary_label",
    "tier_label",
    "used_in_ledger",
]

PROBE_CONFIG = RecallConfig(max_items=20, min_score=0.02, candidate_limit=400)
"""「以它为线索召回一次」的参数。**与生产参数刻意不同。**

生产是 ``max_items=8, min_score=0.10, candidate_limit=200``（见
:class:`~memcore.recall.RecallConfig` 各字段的标定链）。探索放宽到 20 条、
0.02 的门槛，为的是看见「差一点就进来」的那些——只把当初那 8 条再列一遍的话，
这个按钮什么也没回答。

**代价是这组排名不能当生产排名读。** 阈值降到 0.02 之后，一堆在生产里根本不会
被考虑的条目会出现在靠前的位置。所以界面必须把「探索参数」写出来（见
:func:`probe_position` 的第一行），而这条参数与那句标注是这个模块里的一对。
"""

PROBE_BUDGET_TOKENS = 8_000
"""线索召回的 token 额度。

**它不回答"装不装得下"，所以刻意给得宽。** 生产里那个额度是上下文预算的一部分，
而探索问的是"以它为线索，附近还有谁"——额度卡在中间会让结果里混进一堆
``OVER_BUDGET`` 淘汰项，读起来像"相关性不够"，实际只是"这一轮上下文小"。
给宽之后留在结果里的淘汰只有一个含义：分数不够。

它仍然要有一个值：``recall`` 的贪心循环靠它收敛。
"""

_TIER_LABELS: dict[Tier, str] = {
    Tier.WORKING: "工作",
    Tier.RECENT: "近期",
    Tier.LONGTERM: "长期",
    Tier.ARCHIVE: "归档",
    # COLD 单独一行的理由：它容易与上面那个 ARCHIVE 混成同一个意思，而两者无关
    # ——COLD 是「你忘掉了它」，ARCHIVE 是「它自动沉下去了」。界面上的软遗忘按钮
    # 是 COLD 唯一的生产者，所以这一格的名字必须与按钮说的话一模一样。
    # （召回账本里那一格已改成同一个词，见 audit.py 的 RejectReason.COLD。）
    Tier.COLD: "已遗忘",
}

_KIND_LABELS: dict[MemoryKind, str] = {
    MemoryKind.FACT: "事实",
    MemoryKind.PREFERENCE: "偏好",
    MemoryKind.DECISION: "决定",
    MemoryKind.CONCLUSION: "结论",
    MemoryKind.OPEN_QUESTION: "未决问题",
    MemoryKind.TOOL_RESULT: "工具结果",
    MemoryKind.PERSONA_ANCHOR: "人格锚",
}

_STATUS_LABELS: dict[str, str] = {
    "all": "全部",
    "active": "在用",
    "forgotten": "已遗忘",
    "pinned": "钉住",
}


class MemoryStatus(str, Enum):
    """列表的「状态」这一维过滤。"""

    ALL = "all"
    ACTIVE = "active"
    FORGOTTEN = "forgotten"
    PINNED = "pinned"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        return _STATUS_LABELS[self.value]


@dataclass(frozen=True)
class MemoryFilter:
    """记忆列表的筛选条件。四维，与 :func:`filter_summaries` 一一对应。"""

    query: str = ""
    tier: Tier | None = None
    kind: MemoryKind | None = None
    status: MemoryStatus = MemoryStatus.ALL


def filter_summaries(
    summaries: Iterable[MemorySummary],
    flt: MemoryFilter | None = None,
) -> list[MemorySummary]:
    """按四维过滤，保持传入的顺序（``summaries()`` 给的是最新在前）。

    **``query`` 这一维只服务于「在当前这批里再筛一层」**；全库搜索走
    :meth:`~memcore.store.MemoryStore.summaries` 的 SQL ``LIKE``。两者必须给出一致
    的结果，所以匹配方式对齐到 SQL ``LIKE`` 的语义：ASCII 大小写不敏感，其余原样。
    ``.lower()`` 两边同降就是它——CJK 上 ``lower()`` 是恒等变换，所以中文既不会被
    降坏，也与 SQL 一致。

    （分层的理由：列表一屏只装最近若干条，而「我明明记过这件事」要找的很可能是
    更早的那条。用户一旦输入搜索，界面就改走 SQL，超出那一页的东西才找得回来。）
    """
    criteria = flt if flt is not None else MemoryFilter()
    needle = criteria.query.lower()
    found: list[MemorySummary] = []
    for summary in summaries:
        if needle and needle not in summary.content.lower():
            continue
        if criteria.tier is not None and summary.tier is not criteria.tier:
            continue
        if criteria.kind is not None and summary.kind is not criteria.kind:
            continue
        if criteria.status is MemoryStatus.ACTIVE and summary.tier is Tier.COLD:
            continue
        if criteria.status is MemoryStatus.FORGOTTEN and summary.tier is not Tier.COLD:
            continue
        if criteria.status is MemoryStatus.PINNED and not summary.pinned:
            continue
        found.append(summary)
    return found


def with_filter(flt: MemoryFilter, **changes: object) -> MemoryFilter:
    """换掉部分筛选条件。单列一个函数而不是给 :class:`MemoryFilter` 加方法，
    是因为 ``with`` 是关键字、方法名得写成 ``with_``，读起来反而绕。"""
    return replace(flt, **changes)  # type: ignore[arg-type]


def tier_label(tier: Tier) -> str:
    """分层的中文名。COLD 是「已遗忘」，不是「已归档」。"""
    return _TIER_LABELS[tier]


def kind_label(kind: MemoryKind) -> str:
    """类别的中文名。"""
    return _KIND_LABELS[kind]


def summary_label(summary: MemorySummary, *, limit: int = 48) -> str:
    """列表里一行显示什么。

    **为什么这在纯模块里而不是控件的 ``addItem`` 旁边**：这一行要同时说清三件事
    ——它是什么类别、在哪一层、以及**用户能不能看见自己刚做过的那个动作**
    （钉住 / 已遗忘）。字符串拼装住在 ``QWidget`` 里就只能靠人眼看，而"软遗忘完
    列表里那一条有没有变样"正是这个界面最需要能被断言的几件事之一。

    正文压成一行并截断（换行会让 ``QListWidget`` 的行高参差、也会让相邻行的
    类别标记看不见）；``tier_label`` 已经把「已遗忘」写在里面，所以不再另加标记。
    """
    marks = ["钉住"] if summary.pinned else []
    marks += [tier_label(summary.tier), kind_label(summary.kind)]
    return f"[{' · '.join(marks)}] {_preview(summary.content, limit)}"


def format_ts(ts: float, *, fallback: str = "—") -> str:
    """Unix 秒 → 本地时间字符串。

    ``ts <= 0`` 返回占位符而不是 1970 年：``last_access_ts`` 在一条从未被召回过的
    记忆上是 0，显示成「1970-01-01 08:00」会被读成「它的数据坏了」。
    """
    if ts <= 0:
        return fallback
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def used_in_ledger(ledger: RecallLedger | None, memory_id: str) -> str:
    """这条记忆在本轮账本里的归因；不在账本里返回空串。

    **账本为 ``None`` 与「不在账本里」是两件事**：前者是这一轮还没发生（没有会话、
    或首次装配还没回来），后者是发生过了但它没进候选。混成一句话会让用户以为
    系统在敷衍他。
    """
    if ledger is None:
        return "还没有装配过一轮（当前没有会话，或这一轮还没发出请求）。"
    for injected in ledger.injected:
        if injected.memory_id == memory_id:
            return f"已注入：{injected.explain}"
    for rejected in ledger.rejected:
        if rejected.memory_id == memory_id:
            return f"被淘汰：{rejected.explain}"
    return ""


def _absent_reason(item: MemoryItem, *, current_session_id: str) -> str:
    """不在账本里时，**列出可能的**原因——不挑一个当结论。

    ``pinned`` 与本会话条目在 :func:`memcore.recall.recall` 里是两条不同的早退
    路径（``if item.pinned or item.memory_id in excluded: continue``），两条都发生在
    扫描**之前**，所以它们既不进 ``injected`` 也不进 ``rejected``。用户看到的是
    「它明明在库里，账本里却没有」——这里必须解释这件事，且两种情形分开说。
    """
    if item.pinned:
        return (
            "不在本轮账本里：它是钉住的条目，召回在扫描之前就跳过它。"
            "钉住不是「参与打分但分高」，是「根本不在候选里」——它只可能通过 L0 "
            "进上下文，而 L0 只有人格锚会走。"
        )
    if current_session_id and item.session_id == current_session_id:
        return (
            "不在本轮账本里：它属于当前这个会话，召回把它当作「逐字稿里已经有了的"
            "内容」排除了。这不是说它不重要，是说它已经在这一轮的上下文里。"
        )
    return "不在本轮账本里：它没有进入候选（分数不够、与已入选的重复、或装不进预算）。"


def render_memory_detail(
    item: MemoryItem,
    *,
    now: float,
    spaces: Sequence[Space] = (),
    embedded: Mapping[str, Collection[str]] | None = None,
    ledger: RecallLedger | None = None,
    current_session_id: str = "",
) -> str:
    """详情面板的正文：条目本身、本轮归因、向量状态。

    Args:
        item: 按 id 从库里取出来的完整条目（含 ``meta`` 与 SDR）。
        now: 结算惰性衰减用的「现在」。
        spaces: 库里登记过的全部空间，用来逐空间标注有没有向量。
        embedded: **空间 id → 该空间里已有向量的记忆 id 集合**（每个空间的集合
            来自 :meth:`~memcore.store.MemoryStore.embedded_ids`）。是一条映射
            而不是一个集合：换过嵌入模型之后库里会有多个空间，而一条记忆在某
            空间有向量、在另一空间没有是常态。一个集合会让所有空间共用同一个
            「有/无」，那是**逐空间标注的全部意义所在**被抹掉。``None`` 等同于
            空映射。
        ledger: 本轮账本。``None`` 表示还没装配过。
        current_session_id: 当前会话 id，用来识别「本会话条目」那条早退路径。

    **「有没有语义向量」只能查 ``embeddings`` 表。** ``item.space_id`` 与
    ``item.sem_sdr`` 在库里恒为空——写入侧留空、回填侧只写 ``embeddings`` 表、
    从不回写 ``memories`` 行——所以 ``item.has_embedding`` 永远是 ``False``。
    信它会得到「每一条都没有向量」这个看起来正常、实际全错的结论。
    """
    decayed = item.decayed_weight(now, half_life_days=item.tier.half_life_days)
    record = read_forget_record(item)
    by_space: Mapping[str, Collection[str]] = embedded if embedded is not None else {}

    weight_line = f"权重 {item.weight:.3f}"
    if abs(decayed - item.weight) > 1e-6:
        weight_line += f" · 结算衰减后 {decayed:.3f}"

    lines = [
        item.memory_id,
        f"类别 {kind_label(item.kind)} · 层级 {tier_label(item.tier)}"
        f" · 来源 {item.taint.value} · 钉住 {'是' if item.pinned else '否'}",
        f"{weight_line} · 被召回 {item.access_count} 次 · 巩固 {item.reinforce_count} 次",
        f"创建 {format_ts(item.created_at)} · 最近召回 {format_ts(item.last_access_ts)}",
        f"来源指针 {item.source_ref or '（无）'}",
        f"会话 {item.session_id or '（跨会话）'}",
    ]
    if record is not None:
        lines.append(
            f"遗忘记录：原层级 {tier_label(record.previous_tier)} · 于 {format_ts(record.at)}"
        )
    if item.meta.get("derived"):
        # **整理条目**要说清它是谁的合并视图。用户的要求是「已有的记忆不允许被改动」，
        # 所以原句永远在库里——这一行让人对得出来：它由哪几条派生、覆盖哪段时间。
        who = [str(each) for each in item.meta.get("derived_from") or ()]
        span = item.meta.get("span")
        head = f"整理产物 · 由 {len(who)} 条原句归并"
        if isinstance(span, (list, tuple)) and len(span) == 2:
            # 脏的区间就只报条数，不编时间（f-string 先求值，抛了 head 不会被改半截）
            with contextlib.suppress(TypeError, ValueError):
                head += f" · {format_ts(float(span[0]))} – {format_ts(float(span[1]))}"
        lines.append(head)
        for memory_id in who:
            lines.append(f"  · {memory_id}")
        lines.append("原句一字不动地留在库里——这条只是它们的合并视图，正文末尾附着逐字原话。")
    if item.meta:
        # meta 对 memcore 是不透明的，这里也**不做解释**，只把键列出来。里面的
        # 东西（工具轨迹、归因快照）由写它的那一层负责说明。
        lines.append(f"meta 键：{', '.join(sorted(item.meta))}")

    lines += ["", "── 正文 ──", item.content, "", "── 本轮归因 ──"]
    lines.append(
        used_in_ledger(ledger, item.memory_id)
        or _absent_reason(item, current_session_id=current_session_id)
    )

    lines += ["", "── 语义向量 ──"]
    if not spaces:
        lines.append("库里还没有登记过任何记忆空间（没装过嵌入模型）。")
    else:
        for space in spaces:
            mark = "有" if item.memory_id in by_space.get(space.space_id, ()) else "无"
            lines.append(f"{mark} · {space.model_ref or space.space_id}")
        lines.append(
            "没有向量不等于这条记忆不可用：词法通道与它无关，召回会照常用新正文"
            "匹配它。补向量由回填管线在后台做。"
        )
    return "\n".join(lines)


def probe_position(ledger: RecallLedger, focus_id: str) -> str:
    """线索召回的结果：这条排第几、附近是谁、各自哪个通道出的分。

    账本按融合得分从高到低装 ``injected``，所以「排名」就是它的下标 + 1。被淘汰
    的候选也在账本里（``rejected``），它们排在入选者**之后**——这一点要写出来，
    否则「第 9 位」会被读成「它在入选名单里」。

    **这条召回的参数与生产不同**（见 :data:`PROBE_CONFIG`），所以排名只回答
    「以它为线索，附近还有谁」，不回答「它当初为什么被注入」。那句话留在输出的
    第一行。
    """
    lines = [
        "线索召回（探索参数，与真实注入的阈值、条数不同）",
        f"扫过 {ledger.scanned} 条 · {len(ledger.injected)} 条入选",
    ]
    if ledger.notes:
        lines += [f"注：{note}" for note in ledger.notes]

    rows: list[str] = []
    found = False
    for index, injected in enumerate(ledger.injected):
        if injected.memory_id == focus_id:
            found = True
        marker = " ← 这一条" if injected.memory_id == focus_id else ""
        rows.append(
            f"{index + 1:>3}. [{injected.scores.fused:.2f}] "
            f"{_channels(injected.scores)}  {_preview(injected.content)}{marker}"
        )
    for rejected in ledger.rejected:
        if rejected.memory_id == focus_id:
            found = True
            rows.append(f"  —. [淘汰] {rejected.explain}  {rejected.content_preview} ← 这一条")

    # **焦点缺席时无论如何都要说。** 只在"一行都没有"时才说的写法有个更坏的失效
    # 形态：列表里列着别的记忆、唯独没有用户点的那一条，而面板一声不吭——用户会
    # 以为它在下面、或者以为按钮坏了。
    if not found:
        lines.append(
            "这次线索召回里**没有它**。钉住的条目必然如此——召回在扫描之前就跳过"
            "钉住与已排除的记忆，包括这一条探索路径；所以对一条钉住的人格锚按这个"
            "按钮，结果永远是这个。"
        )
    elif ledger.injected and ledger.injected[0].memory_id == focus_id:
        lines.append("它就是线索本身——与自己的相似度恒为满分，排第一是构造上的必然。")

    if not rows:
        return "\n".join(lines)

    lines += ["", *rows]
    if ledger.rejected:
        lines.append("（「淘汰」的行排在入选者之后，不在上面的排名里。）")
    return "\n".join(lines)


# ── 关联图：邻居面板 ────────────────────────────────────────────────────────

#: 边权条占几格。5 格是「一眼能分出档、又不至于把行挤到换行」的那个宽度。
_ASSOC_BAR_WIDTH = 5

_ASSOC_NO_GRAPH = (
    "关联图还没有建。小库会在打开时自动建；库大了不会自动建——去「设置 → 记忆」点「重建联想」。"
)
_ASSOC_NOT_IN_GRAPH = "这条记忆还不在关联图里（可能是建图之后才写的）。重建一次就进去了。"
_ASSOC_NO_NEIGHBORS = "关联图里它没有邻居（可能是边权都低于门槛，也可能是邻居已被删除）。"


@dataclass(frozen=True)
class NeighborRow:
    """联想面板里的一行。

    Attributes:
        memory_id: 邻居的 id。行上不显示它（显示串里放 ULID 只会挤掉正文预览），
            但界面靠它做点击跳转。
        edge_weight: 建图那一刻的 Jaccard 边权。**真数，照实显示。**
        label: 显示串，由 :func:`neighbor_label` 拼。
        present: 这条邻居现在还在库里吗。``False`` = 边指向一条已删除的记忆
            ——行**不可点**（点一条不存在的记忆是什么都不该发生的事）。
    """

    memory_id: str
    edge_weight: float
    label: str
    present: bool


def neighbor_label(
    edge_weight: float,
    summary: MemorySummary | MemoryItem | None,
    *,
    width: int = _ASSOC_BAR_WIDTH,
) -> str:
    """邻居一行的显示串。``summary`` 为 ``None`` 表示这条已经不在库里了。

    ``summary`` 收两种类型（``MemorySummary`` 与 ``MemoryItem``）：这里只读
    ``tier``/``weight``/``content`` 三个字段，而两种类型都有。**不为它专门写一个
    转换函数**——调用方手上是哪种就用哪种，多一层转换只是多一处会漂移的抄写。

    条长按 ``edge_weight`` 分档（``█████`` = 满格）。**边权与记忆权重都照实写**：
    它们是真的，而"它到底多强"正是用户看这一行唯一能拿来判断的信息。抹成「较强」
    之类的形容词就把它换成了编的。

    **``memory_id`` 不在显示串里**：ULID 有 26 个字符
    却什么也说明不了，放在行首会把正文预览挤出可视区。它的去处是
    :attr:`NeighborRow.memory_id`，界面拿它做跳转。
    """
    bar = max(0, min(width, round(edge_weight * width)))
    marks = "█" * bar + "░" * (width - bar)
    if summary is None:
        return f"{marks} {edge_weight:.2f} · （这条邻居已经不在库里了）"
    return (
        f"{marks} {edge_weight:.2f} · "
        f"[{tier_label(summary.tier)}·权重{summary.weight:.2f}] "
        f"{_preview(summary.content, 32)}"
    )


def association_rows(
    node_id: str,
    *,
    graph: GraphIndex | None,
    lookup: Callable[[str], MemorySummary | MemoryItem | None],
    width: int = _ASSOC_BAR_WIDTH,
) -> tuple[NeighborRow, ...]:
    """当前这条记忆的邻居行。没有图、或它不在图里时返回空元组。

    Args:
        node_id: 选中的那条记忆。
        graph: 关联图。``None`` = 还没建。
        lookup: ``memory_id → 摘要``，查不到返回 ``None``。界面传的是
            :meth:`~memcore.store.MemoryStore.get`——**每个邻居一次单行读**，
            邻居最多 ``top_m``（8）条，代价可忽略。
        width: 边权条宽度，透传给 :func:`neighbor_label`。

    **刻意不传 ``GraphIndex.neighbors`` 的 ``live=``**（这是个决定，不是漏了）。
    ``live=`` 会把指向已删除节点的边**静默过滤掉**，于是「图是旧的——库里删过东西」
    这件事在界面上完全看不见，而用户看到的是一份"看起来很正常"的邻居表。改为：
    不过滤，逐条查一次库，查不到的标成 :attr:`NeighborRow.present` 为假、不可点，
    由 :func:`association_header` 说出「另有 N 条已被删除」。**图的年纪要说出来，
    不能靠少显示几行来掩盖。**
    """
    if graph is None:
        return ()
    rows: list[NeighborRow] = []
    for neighbor_id, edge_weight in graph.neighbors(node_id):
        found = lookup(neighbor_id)
        rows.append(
            NeighborRow(
                memory_id=neighbor_id,
                edge_weight=edge_weight,
                label=neighbor_label(edge_weight, found, width=width),
                present=found is not None,
            )
        )
    return tuple(rows)


def association_header(
    *,
    has_graph: bool,
    focus_in_graph: bool = False,
    rows: Sequence[NeighborRow] = (),
    built_at: float = 0.0,
    stale: bool | None = False,
) -> str:
    """联想面板顶栏那一句。六种状态，各自把「为什么看不到东西」说出来。

    Args:
        has_graph: 有没有图。
        focus_in_graph: 选中的这条在不在图里。
        rows: :func:`association_rows` 的结果。
        built_at: 建图时刻，用来在过期时说出图的年纪。
        stale: **三态**——``True`` 确定过期、``False`` 刚比对过是最新的、``None``
            **没比对过**（库太大，见 ``AUTO_GRAPH_MAX``）。**不在这一层算**：
            ``is_stale`` 是一次全量重编码（5 万条约 63 秒），由主窗口在建/载入
            那一刻算一次传进来。

    **``None`` 与 ``False`` 必须说不同的话。** 两者都意味着"没发现过期"，但原因
    不一样：一个是查过了，一个是根本没查。把 ``None`` 当 ``False`` 显示，等于替
    一份没验过的图打包票——而这个面板的全部价值就在于它说的话可信。

    **``count`` / ``dropped`` 不由调用方传**：它们
    只能从 ``rows`` 数出来，而"调用方自己数一遍"正是两个数会对不上的地方（比如把
    不可点的行也算进可点的那一栏）。数一次，在这里数。

    **``count`` 为 0 时不另说 ``dropped``**：那时这一句的括号里已经把「邻居已被
    删除」列为可能原因了，再说一遍是同一句话讲两次。
    """
    if not has_graph:
        # 这一句里**一个数字都没有**，而且它必须解释"为什么没有"：小库会自动建、
        # 大库不会——不说清楚的话，用户会以为这个面板坏了。
        return _ASSOC_NO_GRAPH
    if not focus_in_graph:
        return _ASSOC_NOT_IN_GRAPH

    dropped = sum(1 for row in rows if not row.present)
    parts: list[str] = []
    if rows:
        parts.append(f"联想：建图时的 {len(rows)} 个邻居，按边权降序。点一条跳到它。")
        if dropped:
            parts.append(f"另有 {dropped} 条邻居已被删除，图还没重建。")
    else:
        parts.append(_ASSOC_NO_NEIGHBORS)
    if stale is True:
        parts.append(
            f"（这份图是 {format_ts(built_at)} 建的，之后库变过；重建请去「设置 → 记忆」。）"
        )
    elif stale is None:
        # **"没查过"要自己说出口。** 它与"查过、没过期"在当前这份图上可能一模一样，
        # 唯一的差别就是这个面板敢不敢替它担保——所以这句不是啰嗦，是这句话的全部
        # 内容。（三态用 ``is True`` / ``is None`` 判，不用真值——``None`` 与
        # ``False`` 都是假值，混掉的正是这里要分开的那两种。）
        parts.append(
            f"（这份图是 {format_ts(built_at)} 建的，但没有跟库比对过——库太大了。要确定就重建。）"
        )
    return "".join(parts)


# ── SNN 热度 ────────────────────────────────────────────────────────────────

_HEAT_HEAD = (
    "SNN 热度：未启用——它要画的是每条记忆一个「活跃度」，而这个数现在没有来源。\n"
    "  基质已经建好了，但它一步算出来的是「哪条被激活得最强」——一个 argmax，"
    "不是逐条分数；而且那是速率级的活动，不是在跑脉冲。\n"
    "  现成的「活跃」信号是权重、被召回次数与巩固次数；前两项在详情面板里逐条列着，"
    "巩固这一项看下面这行。\n"
)
"""那一块的开头三句，**两个分支逐字相同**。

写一份常量而不是各写一遍：这三句说的是同一件事（热度图为什么还画不出来、现在能看
的是什么），而"两个分支各写一遍"迟早会分叉成两种说法——那时同一块面板会随数据多少
而改变它对**功能本身**的陈述，读的人只会以为其中一句是旧的。

两个说法在这里是**禁词**，各自都会把读者引到错的地方：

* **不许说"脉冲活跃度"。** 这块面板能给的数（``reinforce_count``）确实出自那个基质，
  可基质算的是**速率级**的一步补全，跟脉冲没有关系。写"脉冲活跃度"会让人以为这里
  已经有东西在发放；"能转成 ``SpikeTrain``"与"在跑脉冲"是两件事，见
  :mod:`memcore.substrate` 的模块文档。
* **不许说"还没建"。** 基质是建好了的，缺的是字段的形状，真实的理由见
  :func:`render_heat_placeholder`。

两个都是"混在真话里"的形状：读的人不会去核对基质建没建、跑没跑脉冲，只会顺下来
理解成"这块还没到时候"。
"""


def render_heat_placeholder() -> str:
    """「SNN 热度」那一块**什么都还没有时**的文案：没有数字，也不编一个。

    **绝不做假数据。** 这一段里**不许出现任何数字**，测试直接断言这一条——一个 ``0``
    就足以让它看起来像"算过了，就是这个值"。

    **"为什么还没有"要说准。** 真实的理由是**这块面板要的那个字段不存在**：基质一步
    算出来的是"哪条被激活得最强"，一个 argmax，没有"每条记忆一个标量"这个形状。

    别把"数据源不够好"混进这句。那个源本身值不值得用是另一件事：P4a 量到的数是它在
    目标规模上塌掉（``n=500`` → 0.2250、``n=2000`` → 0.0300，同一份实测表）。

    底下那行说的是**现状**（巩固还没跑过），不是**去哪实现**——面板上不写路线图。
    """
    return (
        f"{_HEAT_HEAD}"
        "  巩固还没跑过。默认每天自动跑一次；也可以从托盘菜单或「设置 → 记忆」里现在跑一轮。\n"
    )


def render_heat_summary(
    *,
    reinforced_memories: int,
    total_reinforcements: int,
    last_run_at: float | None,
) -> str:
    """「SNN 热度」那一块的两态渲染。

    Args:
        reinforced_memories: 库里有几条记忆的 ``reinforce_count > 0``。
        total_reinforcements: 这些计数之和。
        last_run_at: 上一轮跑完的时刻；不知道（没有 / 坏了）时 ``None``。

    **两态的分界线是"有没有可写的数"，不是"跑没跑过"。** 三个分支：

    * ``reinforced_memories <= 0`` 且 ``last_run_at is None`` → 交给
      :func:`render_heat_placeholder`，**零数字**。
    * ``reinforced_memories <= 0`` 但**跑过**（``last_run_at`` 有一个真时刻）→ 同样
      零数字，但**不能说"还没跑过"**：那一轮真的跑了，只是库里要么全是新的（还没
      进图）、要么一条都没通过探针。把这两件事说成同一句，就是在一张跑过的库上
      说假话——而假话的代价是用户以为定时任务没生效，跑去反复点「立即巩固」。
    * 有数 → **真数字**，包括时间。

    ``last_run_at`` 为 ``None`` 而那三个数不为零（事件在、``last.json`` 被删了）时
    **只省掉时间那一截**，不编一个时刻出来。
    """
    if reinforced_memories <= 0 or total_reinforcements <= 0:
        if last_run_at is None:
            return render_heat_placeholder()
        return (
            f"{_HEAT_HEAD}"
            "  巩固跑过了，但上一次一条都没被回放认出——新的还没进关联图，"
            "或者都没通过探针。\n"
        )
    tail = f" · 最近一次 {format_ts(last_run_at)}" if last_run_at is not None else ""
    return (
        f"{_HEAT_HEAD}"
        f"  巩固：{reinforced_memories} 条被回放认出，累计 {total_reinforcements} 次{tail}。\n"
    )


def _channels(scores: ChannelScores) -> str:
    """账本一行里的通道摘要，与 :attr:`RecalledMemory.explain` 同一套口径。"""
    active = scores.active()
    if not active:
        return "（无通道得分）"
    return " · ".join(f"{name} {value:.2f}" for name, value in active)


def _preview(text: str, limit: int = 40) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
