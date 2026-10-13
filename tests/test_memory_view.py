"""记忆浏览器的文本层：筛、排版、读账本。

这些函数全是纯的（进对象、出字符串），所以这个文件**不碰 SQLite、不碰 Qt**，
只造对象。界面上最难测的就是字符串拼装——一旦它住在 ``QWidget`` 里，"这句提示
到底写没写对"就只能靠人眼看。

三条断言比别处要紧：

* ``test_the_heat_panel_without_data_carries_no_numbers`` / ``test_the_heat_panel_with_data_shows_the_real_numbers``
  —— 热度那一块的两态：**没数据一个数字都不许有，有数据就得是真数字**。这两面是
  同一条纪律，而它们会朝相反的方向坏：往零数据分支里塞个 0，或者把真计数抹成
  「若干条」。只钉一面，另一面就没有任何测试会红。
* 联想面板那四条（``test_a_header_*`` / ``test_a_header_with_neighbors_*``）——
  **无数据 → 零数字；有数据 → 真数字**。这两面是同一条纪律，而它们会朝相反的
  方向坏：往占位里塞 0.00，或者把真实边权抹成「较强」。只钉一面，另一面就没有
  任何测试会红。
* ``test_absent_reason_distinguishes_pinned_from_session_items`` —— 一条记忆不在
  账本里时，钉住与本会话是**两条不同的早退路径**，说明必须不同。合并成一句
  「它没进候选」就是编了一个理由。
"""

from __future__ import annotations

import re

import pytest

from axo.memory.browse import (
    MemoryFilter,
    MemoryStatus,
    NeighborRow,
    association_header,
    association_rows,
    filter_summaries,
    format_ts,
    kind_label,
    neighbor_label,
    probe_position,
    render_heat_placeholder,
    render_heat_summary,
    render_memory_detail,
    summary_label,
    tier_label,
    used_in_ledger,
    with_filter,
)
from memcore.audit import (
    ChannelScores,
    RecalledMemory,
    RecallLedger,
    RejectedCandidate,
    RejectReason,
)
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.types import MemoryItem, MemoryKind, MemorySummary, Tier

NOW = 1_760_000_000.0


def make_summary(text: str = "一条记忆", **overrides: object) -> MemorySummary:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "weight": 0.7,
        "pinned": False,
        "created_at": NOW,
        "last_access_ts": NOW,
        "access_count": 0,
        "session_id": "01JSESSION",
        "source_ref": "turn://01JSESSION/2",
    }
    fields.update(overrides)
    return MemorySummary(**fields)  # type: ignore[arg-type]


def make_item(text: str = "一条记忆", **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "weight": 1.0,
        "created_at": NOW,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


def scored(
    memory_id: str, text: str, *, fused: float, lexical: float = 0.0, **over
) -> RecalledMemory:
    fields: dict[str, object] = {
        "memory_id": memory_id,
        "content": text,
        "kind": "fact",
        "tier": "longterm",
        "weight": 0.7,
        "tokens": 40,
        "scores": ChannelScores(lexical=lexical, fused=fused),
    }
    fields.update(over)
    return RecalledMemory(**fields)  # type: ignore[arg-type]


# ── filter_summaries ────────────────────────────────────────────────────────


def test_filter_by_query_is_case_insensitive_like_sql() -> None:
    """对齐 SQL ``LIKE`` 的语义——否则本地筛与全库搜会给出不同的结果。

    ASCII 大小写不敏感（SQLite 的 ``LIKE`` 默认如此），CJK 上 ``lower()`` 是恒等
    变换，也一致。
    """
    summaries = [make_summary("The Router is in the study."), make_summary("今天午饭是番茄炒蛋。")]

    assert [s.content for s in filter_summaries(summaries, MemoryFilter(query="router"))] == [
        "The Router is in the study."
    ]


def test_filter_by_tier() -> None:
    cold = make_summary("忘了的。", tier=Tier.COLD)
    live = make_summary("记着的。")
    summaries = [cold, live]

    assert filter_summaries(summaries, MemoryFilter(tier=Tier.COLD)) == [cold]


def test_filter_by_kind() -> None:
    anchor = make_summary("人格锚。", kind=MemoryKind.PERSONA_ANCHOR, pinned=True)
    summaries = [anchor, make_summary("事实。")]

    assert filter_summaries(summaries, MemoryFilter(kind=MemoryKind.PERSONA_ANCHOR)) == [anchor]


def test_filter_by_status() -> None:
    """``ACTIVE`` / ``FORGOTTEN`` 按 ``tier`` 分，``PINNED`` 按 ``pinned`` 分。

    这是唯一一个用户看得见、也唯一一个用户改得动的状态位——所以"在用"必须真的
    等于"不在冷层"，而不是"权重高"之类的算法内部量。
    """
    cold = make_summary("忘了的。", tier=Tier.COLD)
    pinned = make_summary("钉住的。", pinned=True)
    plain = make_summary("普通的。")
    summaries = [cold, pinned, plain]

    assert filter_summaries(summaries, MemoryFilter(status=MemoryStatus.ACTIVE)) == [pinned, plain]
    assert filter_summaries(summaries, MemoryFilter(status=MemoryStatus.FORGOTTEN)) == [cold]
    assert filter_summaries(summaries, MemoryFilter(status=MemoryStatus.PINNED)) == [pinned]
    assert filter_summaries(summaries) == summaries
    # PINNED 与 FORGOTTEN 可以同时命中一条——两个维度不互斥。
    both = make_summary("又钉又忘。", pinned=True, tier=Tier.COLD)
    assert filter_summaries([both], MemoryFilter(status=MemoryStatus.FORGOTTEN)) == [both]
    assert filter_summaries([both], MemoryFilter(status=MemoryStatus.PINNED)) == [both]


def test_filter_keeps_the_incoming_order() -> None:
    """``summaries()`` 给的是最新在前，筛完不能把顺序弄丢。"""
    texts = [f"第 {index} 条。" for index in range(6)]
    summaries = [make_summary(text) for text in texts]

    assert [s.content for s in filter_summaries(summaries, MemoryFilter(query="第"))] == texts


def test_filter_combines_dimensions() -> None:
    target = make_summary("番茄炒蛋的做法。", kind=MemoryKind.FACT)
    summaries = [
        make_summary("番茄炒蛋的做法。", kind=MemoryKind.PREFERENCE),
        target,
        make_summary("别的。"),
    ]

    found = filter_summaries(summaries, MemoryFilter(query="番茄", kind=MemoryKind.FACT))

    assert found == [target]


def test_with_filter_replaces_only_what_it_is_given() -> None:
    original = MemoryFilter(query="路由器", tier=Tier.RECENT)

    changed = with_filter(original, status=MemoryStatus.FORGOTTEN)

    assert changed.query == "路由器"
    assert changed.tier is Tier.RECENT
    assert changed.status is MemoryStatus.FORGOTTEN


# ── 排版 ────────────────────────────────────────────────────────────────────


def test_labels_are_chinese() -> None:
    assert tier_label(Tier.COLD) == "已遗忘"
    assert tier_label(Tier.LONGTERM) == "长期"
    assert kind_label(MemoryKind.PERSONA_ANCHOR) == "人格锚"
    assert MemoryStatus.FORGOTTEN.label == "已遗忘"


def test_cold_is_labelled_forgotten_not_archived() -> None:
    """``COLD`` 是「已遗忘」，不能写成「归档」。

    容易混的是同一张表里真实存在的 :attr:`Tier.ARCHIVE`（"自动沉下去了"）。
    ``COLD`` 只有一个含义——"你点了遗忘"——而界面上的软遗忘按钮就是这个动作的
    出处。用"归档"的话，用户刚点过遗忘、转头在列表里找不到"已遗忘"这个筛选，
    会以为操作没生效。召回账本里那一格现在是同一个词（audit.py）。
    """
    assert "归档" not in tier_label(Tier.COLD)
    assert RejectReason.COLD.label == tier_label(Tier.COLD)


def test_summary_label_shows_kind_tier_and_a_one_line_preview() -> None:
    summary = make_summary("我家的路由器\n放在书房书架顶层。", kind=MemoryKind.FACT)

    label = summary_label(summary)

    # 顺序是「状态标记 · 层级 · 类别」：层级是用户最常筛的那一维，所以排在类别前面。
    assert label.startswith("[长期 · 事实] ")
    # 正文压成一行：换行会让列表行高参差，也会把后面的行挤出去。
    assert "\n" not in label
    assert "我家的路由器 放在书房书架顶层。" in label


def test_summary_label_marks_the_two_states_the_user_can_change() -> None:
    """钉住与已遗忘都要**在列表行上就看得见**。

    这两个是用户唯一改得动的状态位，也是他做完动作之后回头核对的东西。只写在
    详情里的话，软遗忘完列表看起来毫无变化，用户会以为操作没生效。
    """
    assert "已遗忘" in summary_label(make_summary(tier=Tier.COLD))
    assert "钉住" in summary_label(make_summary(pinned=True))
    assert "钉住" not in summary_label(make_summary())


def test_summary_label_truncates_a_long_body() -> None:
    label = summary_label(make_summary("很长的一段正文。" * 40), limit=12)

    assert label.endswith("…")
    assert len(label) < 60


def test_format_ts_uses_a_placeholder_for_missing_times() -> None:
    """``last_access_ts`` 在从未被召回过的记忆上是 0——显示成 1970 会被读成"数据坏了"。"""
    assert format_ts(0.0) == "—"
    assert format_ts(-1.0) == "—"
    assert format_ts(NOW).startswith("20")


# ── render_memory_detail ────────────────────────────────────────────────────


def test_detail_shows_source_session_and_pin_state() -> None:
    item = make_item("我家的路由器放在书房。", source_ref="turn://01JABC/2", session_id="01JABC")

    text = render_memory_detail(item, now=NOW)

    assert "turn://01JABC/2" in text
    assert "01JABC" in text
    assert "我家的路由器放在书房。" in text


def test_detail_shows_the_decayed_weight_when_it_differs() -> None:
    """惰性衰减是**算出来的**，不是存在库里的——所以详情必须显示结算后的值。

    只显示 ``item.weight`` 的话，用户在界面上看到的是一个已经过期的数：一条一年
    没被想起的记忆会显示"权重 0.83"，而它实际参与打分时是 0.08。
    """
    fresh = make_item("刚刚写的。", weight=0.8, created_at=NOW - 10)
    stale = make_item(
        "一年没想起来过。",
        weight=0.8,
        tier=Tier.LONGTERM,
        created_at=NOW - 365 * 86400,
        last_decay_ts=NOW - 365 * 86400,
    )

    fresh_text = render_memory_detail(fresh, now=NOW)
    stale_text = render_memory_detail(stale, now=NOW)

    assert "结算衰减后" not in fresh_text
    assert "结算衰减后" in stale_text
    assert "0.800" in stale_text


def test_detail_marks_a_forgotten_item() -> None:
    item = make_item("忘了的。", tier=Tier.COLD)

    text = render_memory_detail(item, now=NOW)

    assert "已遗忘" in text


def test_detail_does_not_trust_item_space_id() -> None:
    """**「有没有向量」只能查 ``embeddings`` 表。**

    ``item.space_id`` 与 ``item.sem_sdr`` 在库里恒为空（写入侧留空、回填侧只写
    ``embeddings`` 表、从不回写 ``memories`` 行），所以 ``item.has_embedding``
    永远是 False。信它会得到「每一条都没有向量」这个看起来正常、实际全错的结论。
    这条测试给出一条 ``space_id`` 非空但 ``embeddings`` 里没有它的条目：详情必须
    说「无」。
    """
    space = Space(space_id=make_space_id("test", "fake-embed", 8), dim=8, model_ref="fake")
    lying = make_item("正文没变过。", space_id="test/other@8")

    text = render_memory_detail(lying, now=NOW, spaces=[space], embedded={space.space_id: set()})

    assert "无 · fake" in text


def test_detail_reports_the_vector_per_space() -> None:
    """**逐空间**标注——一个集合表示不了这件事，所以入参是一条映射。

    换过嵌入模型之后库里会有多个空间，而一条记忆在新空间里有向量、在旧空间里
    没有是常态。所有空间共用一个「有/无」等于把逐空间标注的意义抹掉。
    """
    first = Space(space_id=make_space_id("test", "one", 8), dim=8, model_ref="one")
    second = Space(space_id=make_space_id("test", "two", 8), dim=8, model_ref="two")
    item = make_item("只在新空间里有向量。")

    text = render_memory_detail(
        item,
        now=NOW,
        spaces=[first, second],
        embedded={first.space_id: set(), second.space_id: {item.memory_id}},
    )

    assert "无 · one" in text
    assert "有 · two" in text


def test_detail_without_embedded_data_marks_nothing_present() -> None:
    """没传 ``embedded`` 时全部标「无」——**不猜**。

    标错方向的代价不对称：把"有"说成"无"会让用户以为补向量坏了；而把"无"说成
    "有"会让他以为切换模型之后接得上。
    """
    space = Space(space_id=make_space_id("test", "one", 8), dim=8, model_ref="one")

    text = render_memory_detail(make_item(), now=NOW, spaces=[space])

    assert "无 · one" in text


def test_detail_explains_missing_vectors_are_not_fatal() -> None:
    """没有向量不等于这条记忆不可用——这一句必须写出来。

    不写的话，用户看到"无"会以为它坏了，而它其实照常参与词法通道的打分。
    """
    space = Space(space_id=make_space_id("test", "one", 8), dim=8, model_ref="one")

    text = render_memory_detail(
        make_item(), now=NOW, spaces=[space], embedded={space.space_id: set()}
    )

    assert "词法通道" in text


# ── 归因 ────────────────────────────────────────────────────────────────────


def test_detail_quotes_the_ledger_for_an_injected_item() -> None:
    item = make_item("被用上了。")
    ledger = RecallLedger(
        query_preview="路由器",
        channels=("lexical",),
        injected=(scored(item.memory_id, item.content, fused=0.87, lexical=0.9),),
        tokens_used=40,
        tokens_budget=4000,
    )

    text = render_memory_detail(item, now=NOW, ledger=ledger)

    assert "已注入" in text
    assert "词法 0.90" in text


def test_detail_quotes_the_ledger_for_a_rejected_item() -> None:
    item = make_item("没选上。")
    ledger = RecallLedger(
        query_preview="路由器",
        channels=("lexical",),
        rejected=(
            RejectedCandidate(
                memory_id=item.memory_id,
                content_preview="没选上。",
                reason=RejectReason.BELOW_THRESHOLD,
                score=0.04,
            ),
        ),
    )

    text = render_memory_detail(item, now=NOW, ledger=ledger)

    assert "被淘汰" in text
    assert "分数不够" in text


def test_no_ledger_is_not_the_same_as_absent_from_the_ledger() -> None:
    """**「还没装配过」与「装配过了但它没进候选」是两件事。**"""
    item = make_item("随便。")

    assert "还没有装配过" in used_in_ledger(None, item.memory_id)
    assert "不在本轮" in render_memory_detail(item, now=NOW, ledger=RecallLedger(query_preview="x"))


def test_absent_reason_distinguishes_pinned_from_session_items() -> None:
    """两条早退路径在 recall 里是分开的（``pinned`` 与 ``excluded``），说明也要分开。

    钉住：它根本不在候选里，且**永远不会**因为分数变化而进来。
    本会话：它已经在这一轮的逐字稿里了，不是"不重要"。

    合并成一句"它没进候选"，用户就没法区分"我的记忆被系统忽略了"与"它就在
    聊天记录里"——而这两件事该做的下一步完全不同。
    """
    empty = RecallLedger(query_preview="x")
    pinned = make_item("钉住的。", pinned=True, session_id="")
    in_session = make_item("本会话写的。", session_id="01JCURRENT")

    pinned_text = render_memory_detail(
        pinned, now=NOW, ledger=empty, current_session_id="01JCURRENT"
    )
    session_text = render_memory_detail(
        in_session, now=NOW, ledger=empty, current_session_id="01JCURRENT"
    )

    assert pinned_text != session_text
    assert "扫描之前就跳过它" in pinned_text
    assert "扫描之前就跳过它" not in session_text
    assert "当前这个会话" in session_text
    assert "当前这个会话" not in pinned_text


def test_a_pinned_item_in_a_foreign_session_still_gets_the_pinned_reason() -> None:
    """钉住优先判——它是**先于一切**的早退路径，与属于哪个会话无关。"""
    pinned = make_item("钉住的。", pinned=True, session_id="01JOTHER")

    text = render_memory_detail(
        pinned, now=NOW, ledger=RecallLedger(query_preview="x"), current_session_id="01JCURRENT"
    )

    assert "钉住" in text


# ── probe_position ──────────────────────────────────────────────────────────


def _probe_ledger(focus_id: str, *, fused: float = 0.93, rejected: bool = False) -> RecallLedger:
    others = [
        scored(new_ulid(), "别的一条。", fused=0.41, lexical=0.46),
        scored(new_ulid(), "再一条。", fused=0.22, lexical=0.25),
    ]
    return RecallLedger(
        query_preview="线索",
        scanned=12,
        channels=("lexical",),
        injected=(scored(focus_id, "就是它。", fused=fused, lexical=0.99), *others),
        rejected=(
            (
                RejectedCandidate(
                    memory_id=focus_id,
                    content_preview="就是它。",
                    reason=RejectReason.BELOW_THRESHOLD,
                    score=0.03,
                ),
            )
            if rejected
            else ()
        ),
    )


def test_probe_labels_itself_as_exploratory() -> None:
    """**第一行必须是这句。** 不标注的话，用户会把探索排名读成生产排名。"""
    focus = new_ulid()

    text = probe_position(_probe_ledger(focus), focus)

    assert text.splitlines()[0].startswith("线索召回")
    assert "探索参数" in text
    assert "不同" in text


def test_probe_marks_the_focus_row_and_its_rank() -> None:
    focus = new_ulid()

    text = probe_position(_probe_ledger(focus), focus)

    assert "← 这一条" in text
    assert "1. [0.93]" in text
    assert "3. [0.22]" in text
    assert "扫过 12 条" in text


def test_probe_says_when_the_focus_is_the_query_itself() -> None:
    """它是自己的线索时排第一是**构造上的必然**，不是"它最相关"。

    不说这一句，用户会以为第 1 名承载了信息量。
    """
    focus = new_ulid()

    text = probe_position(_probe_ledger(focus), focus)

    assert "线索本身" in text


def test_probe_reports_a_rejected_focus() -> None:
    """被淘汰的候选也出现在账本里——要说清它**不在**上面的排名里。"""
    focus = new_ulid()
    ledger = RecallLedger(
        query_preview="线索",
        scanned=12,
        channels=("lexical",),
        injected=(scored(new_ulid(), "别的一条。", fused=0.5, lexical=0.5),),
        rejected=(
            RejectedCandidate(
                memory_id=focus,
                content_preview="就是它。",
                reason=RejectReason.BELOW_THRESHOLD,
                score=0.03,
            ),
        ),
    )

    text = probe_position(ledger, focus)

    assert "[淘汰]" in text
    assert "分数不够" in text
    assert "不在上面的排名里" in text


def test_probe_says_so_when_the_focus_is_missing() -> None:
    """焦点缺席时**必须说**，哪怕列表里列着别的记忆。

    只在「一行都没有」时才说的写法有个更坏的失效形态：面板列着别的条目、唯独没有
    用户点的那一条，而它一声不吭——用户会以为它在下面，或者以为按钮坏了。缺席的
    最常见原因恰恰是最需要解释的那个：钉住的条目压根不参与召回。
    """
    focus = new_ulid()

    text = probe_position(_probe_ledger(new_ulid()), focus)

    assert "没有它" in text
    assert "钉住" in text
    assert "← 这一条" not in text
    # 别的行照常列出来——缺席说的是焦点，不是整份结果为空。
    assert "别的一条" in text


# ── SNN 热度那一块的两态 ────────────────────────────────────────────────────


def _digits(text: str) -> bool:
    """这段文字里有没有数字。"""
    return re.search(r"\d", text) is not None


@pytest.mark.parametrize(
    "text",
    [
        # 三种"没有可写的数"，走的是两条分支：
        # ① 占位（连一轮都没跑过）；
        # ② 显式喂零（调用方手上有两个 0）；
        # ③ 零条但**跑过**——它走第三条分支，而那条同样一个数字都不许有。
        render_heat_placeholder(),
        render_heat_summary(reinforced_memories=0, total_reinforcements=0, last_run_at=None),
        render_heat_summary(reinforced_memories=0, total_reinforcements=0, last_run_at=NOW),
    ],
)
def test_the_heat_panel_without_data_carries_no_numbers(text: str) -> None:
    """**「绝不做假数据」这条裁决落到这个断言上：没有可写的数就一个数字都不许有。**

    一个 ``0`` 就足以让整段看起来像"算过了、就是这个值"——而这块面板要的那个字段
    **没有来源**：基质一步算出来的是"哪条被激活得最强"（一个 argmax），没有"每条记忆
    一个标量"这个形状。

    别跟另一个问题混起来：基质这个数据源**本身**在目标规模上塌掉（P4a 实测
    ``n=500`` → 0.2250、``n=2000`` → 0.0300），那是"值不值得用"，和"东西有没有建"
    不是同一句话。文案层对这两种说法都有守卫，见下面那条断言。

    **三个输入一起参数化，是因为它们走两条不同的分支**，而"零数字"这条纪律必须两条
    都守。其中"零条但跑过"最容易被顺手写进一个 ``0``。
    """
    assert not _digits(text)


def test_the_heat_panel_says_what_is_missing_and_what_is_there() -> None:
    """它要说清两件事：**热度图为什么还没有**，以及**现在能看的是什么**。

    只说前一半（"未启用"）会留下一句"那我该干嘛"；只说后一半则会让用户以为这就是
    热度图本身。这份摘要里的数字量纲完全不是"热度"，它是**巩固**的计数——所以那两句
    必须同时在场。
    """
    text = render_heat_placeholder()

    assert "未启用" in text, "热度图本身还没画出来，这件事要说"
    assert "热度" in text
    assert "巩固还没跑过" in text, "零数据时说的是现状，不是'去哪实现'"
    # **阶段代号不许再出现。** 这一段说的是**现状**，不是**去哪实现**：写上"依赖 P4c"
    # 会让已经做好的东西看起来还没做。
    assert "P4" not in text
    # 联想链不再是这一块的一部分——留着它会让"这一块没有数字"这条断言在改动联想
    # 面板时被顺带弄红，而两件事没有关系。
    assert "联想链" not in text
    # **"为什么还没有"这条理由不许说错。** 基质是建好了的，不许被说成"还没建"——
    # 真实的理由是这块面板要的字段没有那个形状。
    assert "还没建" not in text
    # 同理，**"脉冲活跃度"这个说法不许出现**：那些数字来自速率级的一步完成，
    # 不是在跑脉冲。
    assert "脉冲活跃度" not in text


def test_a_run_that_reinforced_nothing_does_not_claim_it_never_ran() -> None:
    """跑过、但一条都没加固 → **不许说"还没跑过"**。

    这一轮真的跑了：库里要么全是新写的（还没进关联图），要么都没通过探针。把它说成
    "还没跑过"，用户会以为定时任务没生效，然后去反复点「立即巩固」——而每一次点都
    真的拍一份整库快照。
    """
    text = render_heat_summary(reinforced_memories=0, total_reinforcements=0, last_run_at=NOW)

    assert "还没跑过" not in text
    assert "跑过" in text
    assert not _digits(text), "说清这件事不需要任何数字"


def test_the_heat_panel_with_data_shows_the_real_numbers() -> None:
    """有数据 → **真数字，包括时间**。

    这是与上一条成对的另一半：两条各管一段输入，**不能放在同一段文本上**——一条"有
    数据就有数字"的断言与一条"无数据就没数字"的断言合起来测，只会互相抵消。
    """
    text = render_heat_summary(reinforced_memories=12, total_reinforcements=34, last_run_at=NOW)

    assert "12 条" in text
    assert "34 次" in text
    assert format_ts(NOW) in text, "什么时候跑的要说，否则'累计 34 次'没有时间坐标"


def test_the_time_is_left_out_when_it_is_unknown() -> None:
    """有计数、但**不知道上一轮什么时候跑的** → 只省掉时间那一截，不编一个时刻。

    这条路径是真的：事件文件与 ``last.json`` 是两份东西，后者丢了（手删、盘满、
    换个数据目录）而库里那些计数还在。编一个时刻出来比省掉它坏得多——用户会拿那个
    时间去对别的事。
    """
    text = render_heat_summary(reinforced_memories=12, total_reinforcements=34, last_run_at=None)

    assert "12 条" in text and "34 次" in text
    assert "最近一次" not in text
    assert "1970" not in text, "拿不到时间时最容易出现的那个假值"


# ── 关联图的邻居面板 ────────────────────────────────────────────────────────


def make_graph(entries: dict[str, list[tuple[str, float]]]):
    """手搓一张只含邻接表的 :class:`GraphIndex`。**不给邻居的节点写成空列表。**

    直接构造而不是走 ``GraphIndex.build``：这里要测的是**排版与措辞**，而真建图
    要编码全部正文、还要受 ``min_weight`` 之类的门槛摆布——那样"手上有几个邻居"
    就不再是测试说了算。行序必须是 ``node_ids`` 的字典序（``index_of`` 是二分
    查找），所以这里排一下。

    **"已删除的邻居"仍然是一个节点**（只是没有出边）：图是建图那一刻的快照，库
    里删掉一条记忆之后，它的 id 照样在图里、照样被别人指着。这正是
    :attr:`NeighborRow.present` 存在的原因。
    """
    import numpy as np

    from memcore.graph import GraphIndex

    node_ids = tuple(sorted(entries))
    top_m = max((len(v) for v in entries.values()), default=1)
    nbr_idx = np.full((len(node_ids), max(1, top_m)), -1, dtype=np.int32)
    nbr_w = np.zeros((len(node_ids), max(1, top_m)), dtype=np.float32)
    for row, node_id in enumerate(node_ids):
        for slot, (target, weight) in enumerate(entries[node_id]):
            nbr_idx[row, slot] = node_ids.index(target)
            nbr_w[row, slot] = weight
    return GraphIndex(
        node_ids=node_ids,
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="0" * 64,
        built_at=NOW,
        top_m=max(1, top_m),
        min_weight=0.0,
    )


def lookup_from(*summaries: MemorySummary):
    """把几条摘要做成一个 ``memory_id → 摘要`` 的查询函数。"""
    by_id = {summary.memory_id: summary for summary in summaries}
    return by_id.get


def test_a_header_without_a_graph_carries_no_numbers() -> None:
    """**没有图 → 零数字。** 同一条纪律的另一半在下面的"有邻居"那条里。

    这一句还必须解释"为什么没有"：小库会自动建、大库不会。只说「未启用」的话，
    用户会去开关里找原因——而开关管的是召回，跟图建没建是两件事。
    """
    text = association_header(has_graph=False)

    assert "还没有建" in text
    assert re.search(r"\d", text) is None


def test_a_header_for_a_memory_outside_the_graph_carries_no_numbers() -> None:
    text = association_header(has_graph=True, focus_in_graph=False)

    assert "还不在关联图里" in text
    assert re.search(r"\d", text) is None


def test_a_header_with_no_neighbors_carries_no_numbers() -> None:
    """在图里但一条邻居都没有。**这一句同样不许有数字**——包括不许写"0 个邻居"。

    ``0`` 在这里会被读成"算过了、结果是零"，而真实原因可能只是边权没过门槛。
    """
    text = association_header(has_graph=True, focus_in_graph=True, rows=())

    assert "没有邻居" in text
    assert re.search(r"\d", text) is None


def test_a_header_with_neighbors_reports_them_and_allows_numbers() -> None:
    """**有数据 → 真数字。** 与上面三条合起来才钉住那条边界。"""
    rows = (
        NeighborRow("m-1", 0.62, "l1", True),
        NeighborRow("m-2", 0.31, "l2", True),
    )

    text = association_header(has_graph=True, focus_in_graph=True, rows=rows)

    assert "2 个邻居" in text
    assert "已被删除" not in text


def test_a_stale_header_says_when_the_graph_was_built() -> None:
    rows = (NeighborRow("m-1", 0.62, "l1", True),)

    text = association_header(
        has_graph=True, focus_in_graph=True, rows=rows, stale=True, built_at=NOW
    )

    assert "库变过" in text
    assert format_ts(NOW) in text


def test_an_unverified_header_admits_it_never_compared() -> None:
    """``stale=None`` 是**第三态**，与 ``False`` 不是一回事：一个说"确认没过期"，
    一个说"不知道"。

    大库走的就是这条（``GraphWorker`` 不遍历、不判过期）。写成 ``False`` 的话，用户
    看到的是一份语气笃定的邻居表，而它可能是几周前建的——他于是没有任何理由去点那个
    重建按钮，那个按钮存在的全部意义也就没了。
    """
    rows = (NeighborRow("m-1", 0.62, "l1", True),)
    common = {
        "has_graph": True,
        "focus_in_graph": True,
        "rows": rows,
        "built_at": NOW,
    }

    unknown = association_header(**common, stale=None)
    verified = association_header(**common, stale=False)

    assert "没有跟库比对过" in unknown
    assert "库太大" in unknown, "还得说出**为什么**没比对——否则像是我漏做了一步"
    assert format_ts(NOW) in unknown, "什么时候建的仍然要说，那是唯一确定的事"
    # 两个方向都不许冒充：既不能说"库变过"（那是 True 的话），也不能一声不吭
    # （那是 False 的话）。
    assert "库变过" not in unknown
    assert "比对" not in verified and unknown != verified


def test_the_header_counts_dropped_neighbors_from_the_rows() -> None:
    """指向已删除节点的边**不静默消失**，而是被数出来说出去。

    这正是 ``association_rows`` 不传 ``live=`` 的那个决定的下游：过滤掉的话，
    用户看到的是一份"看起来很正常"的邻居表，而图其实是旧的。
    """
    rows = (
        NeighborRow("m-1", 0.62, "l1", True),
        NeighborRow("m-2", 0.31, "l2", False),
    )

    text = association_header(has_graph=True, focus_in_graph=True, rows=rows)

    assert "另有 1 条邻居已被删除" in text
    # 行数说的是图里那一份（2），不是"现在还能点的那些"（1）——图是建图那一刻的
    # 快照，把它写成 1 会让人以为图刚刚重建过。
    assert "2 个邻居" in text


def test_a_row_for_a_living_neighbor_is_clickable_and_shows_real_numbers() -> None:
    summary = make_summary("客厅的路由器信号到卧室很弱。", weight=0.42)

    label = neighbor_label(0.6, summary)

    assert "0.60" in label
    assert "0.42" in label  # 记忆自身的权重，与边权是两个数
    assert "长期" in label
    assert "客厅的路由器" in label


def test_a_row_for_a_deleted_neighbor_says_so_instead_of_showing_a_preview() -> None:
    """``summary is None``：行**不可点**，而且不许编一条正文出来。

    **边权照旧显示。** 那个数是图里真的存着的（建图那一刻的 Jaccard），它没有
    因为邻居被删就变成假的；抹掉它反而丢掉"这条边当时有多强"这个唯一的信息。
    消失的是**记忆那一半**（层级、权重、正文预览）——那些正是查不到的东西。
    """
    label = neighbor_label(0.6, None)

    assert "已经不在库里了" in label
    assert "0.60" in label
    assert "[" not in label  # 没有 [层级·权重] 那一段——它要查库才知道


def test_the_edge_weight_bar_fills_from_zero_to_full() -> None:
    """边权条是显示层唯一的"相对强弱"表达，所以两端都要钉住。"""
    summary = make_summary("随便一条。")

    assert "█" * 5 in neighbor_label(1.0, summary)
    assert "█" not in neighbor_label(0.0, summary)
    assert "░" * 5 in neighbor_label(0.0, summary)
    # 中间档：0.5 → 向下取整到 2 格（round(2.5) 在 Python 里是 2，银行家舍入）。
    assert "██░░░" in neighbor_label(0.5, summary) or "███░░" in neighbor_label(0.5, summary)


def test_rows_look_up_every_neighbor_and_keep_the_graph_order() -> None:
    """**每个邻居查一次库**，查得到的可点、查不到的标出来——而且顺序照图来。

    顺序是图的边权降序（``GraphIndex.neighbors`` 保证），这里不重排：界面上那一条
    条是从强到弱的，重排会让"第一条最像"这个读法失效。
    """
    alive = make_summary("还在的一条。")
    graph = make_graph(
        {
            "focus": [("已删除的", 0.9), (alive.memory_id, 0.4)],
            "已删除的": [],
            alive.memory_id: [],
        }
    )

    rows = association_rows("focus", graph=graph, lookup=lookup_from(alive))

    assert [row.memory_id for row in rows] == ["已删除的", alive.memory_id]
    assert [row.present for row in rows] == [False, True]
    assert "已经不在库里了" in rows[0].label
    assert "还在的一条" in rows[1].label


def test_rows_are_empty_without_a_graph_or_for_an_unknown_node() -> None:
    """图没建、或这条不在图里：返回空，**不抛**。

    图是缓存，一条刚写进去还没重建的记忆在图里查不到是常态——那不是错误。
    """
    summary = make_summary("随便一条。")
    graph = make_graph({"focus": [(summary.memory_id, 0.4)], summary.memory_id: []})

    assert association_rows("focus", graph=None, lookup=lookup_from(summary)) == ()
    assert association_rows("不在图里", graph=graph, lookup=lookup_from(summary)) == ()


def test_detail_marks_a_derived_entry_and_names_its_fragments() -> None:
    """整理条目要能被认出来，并列出它由哪几条派生。

    用户的约束是「已有的记忆不允许被改动」，所以原句永远在库里——这一行让人对得出来：
    这条是合并视图，不是新事实。
    """
    item = make_item(
        "感冒那位回来了；还带了杯奶茶。",
        meta={"derived": True, "derived_from": ["01A", "01B"], "span": [NOW, NOW + 180]},
    )

    text = render_memory_detail(item, now=NOW)

    assert "整理产物" in text
    assert "由 2 条原句归并" in text
    assert "01A" in text and "01B" in text


def test_a_plain_item_is_not_labelled_as_derived() -> None:
    item = make_item("路由器在书房。")

    text = render_memory_detail(item, now=NOW)

    assert "整理产物" not in text


def test_a_dirty_span_still_says_how_many_fragments() -> None:
    """区间是脏的就只报条数，不编时间——编出来的时间会被当事实读。"""
    item = make_item(
        "并出来的一条。",
        meta={"derived": True, "derived_from": ["01A"], "span": "显然是坏的"},
    )

    text = render_memory_detail(item, now=NOW)

    assert "由 1 条原句归并" in text
