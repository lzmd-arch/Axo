"""什么算一条记忆，以及它怎么进库。

这一层要钉住的是**行为**：哪句话会被记住、记成什么、失败时会不会把整轮吞掉。
它同时也是 P3 粒度切换的接缝，所以最后那组测试不是关于 P1 的行为，而是关于
"换一个策略实现真的能改变入库的东西吗"——接缝空有其形是最糟的结果：它看起来
留了口子，P3 到那天才发现换不动。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from biosnn_memterm.agent.loop import ToolFinished
from biosnn_memterm.agent.trace import build_trace
from biosnn_memterm.memory import (
    MAX_TASK_TEXT_CHARS,
    MemoryDraft,
    MemoryWriter,
    PerItemPolicy,
    TaskPolicy,
    Turn,
)
from memcore import MemoryKind, Taint, Tier
from memcore.store import MemoryStore
from memcore.text import is_word, normalize, tokenize


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def turn(text: str, *, turn_seq: int = 1, **kwargs) -> Turn:
    return Turn(user_text=text, session_id="20261010-120000-abcdef", turn_seq=turn_seq, **kwargs)


# ── 什么该被记住 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "路由器放在书房书架顶层。",
        "每周三晚上八点要开组会。",
        "记忆终端用 SQLite 存记忆库，WAL 模式。",
        "明天记得提醒我买牛奶。",
        "num_ctx 改成 16384 了。",
    ],
)
def test_a_statement_is_remembered(text: str) -> None:
    assert len(PerItemPolicy().drafts(turn(text))) == 1


@pytest.mark.parametrize(
    "text",
    [
        "好的",
        "嗯",
        "继续",
        "谢谢",
        "OK",
        "你好",
        "？",
        "。。。",
        "    ",
        "",
        "嗯嗯嗯嗯",
        "哈哈哈哈",
        "路由器",
    ],
)
def test_a_murmur_is_not_a_memorY(text: str) -> None:
    """垃圾记忆不只是占位置，**它参与竞争**。

    库会退化成一份逐字稿副本，而每一次召回都要为每一句废话算一次余弦，把真正该
    被想起来的那条挤出前 8 名。
    """
    assert PerItemPolicy().drafts(turn(text)) == (), f"{text!r} 不该被记住"


def test_an_acknowledgement_inside_a_real_statement_is_kept() -> None:
    """**只做整句相等，不做子串匹配。**

    子串匹配会把这句话判成附和，而它正是最该被记住的那一类。
    """
    text = "好的，路由器放在书房书架顶层。"
    assert len(PerItemPolicy().drafts(turn(text))) == 1


def test_the_threshold_counts_meaningful_characters_not_length() -> None:
    """ "？？？" 有 3 个字符、0 个实义字符；"会议周三" 4 个字符、4 个实义字符。

    按 ``len()`` 判会让一串标点比一句话更"够格"。
    """
    assert sum(1 for c in "？？？" if is_word(c)) == 0
    assert PerItemPolicy().drafts(turn("？？？")) == ()
    assert len(PerItemPolicy().drafts(turn("会议周三"))) == 1


# ── 记成什么 ────────────────────────────────────────────────────────────────


def test_a_fact_is_a_fact() -> None:
    draft = PerItemPolicy().drafts(turn("路由器放在书房书架顶层。"))[0]
    assert draft.kind is MemoryKind.FACT
    assert draft.taint is Taint.USER


@pytest.mark.parametrize(
    "text",
    ["我喜欢深色主题。", "以后都先跑 ruff 再提交。", "不要再自动展开侧栏了。"],
)
def test_a_preference_is_marked_as_one(text: str) -> None:
    assert PerItemPolicy().drafts(turn(text))[0].kind is MemoryKind.PREFERENCE


@pytest.mark.parametrize("text", ["就用 SQLite 吧。", "决定用 WAL 模式了。", "改成 16384。"])
def test_a_decision_is_marked_as_one(text: str) -> None:
    assert PerItemPolicy().drafts(turn(text))[0].kind is MemoryKind.DECISION


def test_a_question_is_not_stored_as_a_fact() -> None:
    """**问题没有断言任何东西。**

    记成 FACT 会让模型把"用户问过路由器放哪"读成"路由器在某处"——而它接着会
    据此回答，还答得很自信。
    """
    assert PerItemPolicy().drafts(turn("路由器放在哪了？"))[0].kind is MemoryKind.OPEN_QUESTION


def test_the_question_rule_can_be_turned_off() -> None:
    """这条规则有真实的反对意见：一个**被回答了的**问题并不是"待办问题"。

    P1 判断不了一轮有没有被回答，所以只能保守地按形状标；开关留给 P2 的偏好表。
    """
    policy = PerItemPolicy(question_kind=False)
    assert policy.drafts(turn("路由器放在哪了？"))[0].kind is MemoryKind.FACT


@pytest.mark.parametrize(
    "text",
    [
        "帮我查一下我之前记过什么",  # 报告出来的那一句
        "路由器放在哪儿",
        "你吃了吗",
        "你吃过了么",
        "我们是不是该走了",
        "这个方案行不行",
        "你还记不记得上次说的那件事",
        "那个文件怎么读",
        "现在几号",
    ],
)
def test_a_question_without_a_question_mark_is_not_stored_as_a_fact(text: str) -> None:
    """**口语化的输入本来就不爱打问号。**

    只认"句末是不是问号"一条判据的话，中文里不带标点的问句**会全部落进 ``FACT``**
    ——"帮我查一下我之前记过什么"就是一句。这不是边界情况：打字的对话里问号本来就
    可有可无，而这一档要防的正是"用户问过 X"被读成"X 成立"。

    这里同时钉住四种形状：句末语气词、正反问、疑问词、以及它们各自的常见写法。
    """
    assert PerItemPolicy().drafts(turn(text))[0].kind is MemoryKind.OPEN_QUESTION


def test_a_self_disclosure_marker_still_beats_the_question_shape() -> None:
    """**具体压过宽泛**——偏好表命中的是用户的一句陈述，问句那一档判的是形状。

    两者同时命中时以更具体的那张表为准（理由见 ``_kind_of`` 的 docstring）。这条
    句子两边都真的命中了：``以后都`` 是偏好标记，``什么`` 是疑问词。
    """
    draft = PerItemPolicy().drafts(turn("我以后都想知道你什么时候回来"))[0]
    assert draft.kind is MemoryKind.PREFERENCE


# ── 问句那一档的两条反空转守卫 ──────────────────────────────────────────────
#
# "把所有含疑问词的句子都判成问句"也能让上面那组测试通过，而它会把日常陈述成片地
# 贴上"未决问题"。下面两组各挡一边：一张反例词表，和一个语法形状的**紧度**。


@pytest.mark.parametrize(
    "text",
    [
        "我今天什么都不想干",
        "我没什么好说的",
        "谁都不知道这件事",
        "哪儿都不想去",
    ],
)
def test_an_indefinite_question_word_is_still_a_statement(text: str) -> None:
    """任指/虚指的写法：同一个词在这里是"任何/某个"，不是在问。

    ``_QUESTION_FALSE_FRIENDS`` 这张表一定不全（黑名单在汉语里补不完），但漏掉的
    代价只是某条记忆的标签不准——所以它值得存在，不值得追求完备。
    """
    assert PerItemPolicy().drafts(turn(text))[0].kind is MemoryKind.FACT


@pytest.mark.parametrize(
    "text",
    [
        "我不得不说两句",
        "不知不觉就天亮了",
        "他放得不多不少",
        "他做事不管不顾",
    ],
)
def test_the_reduplication_shape_does_not_fire_on_lookalike_idioms(text: str) -> None:
    """正反问是**形状**判的（``A 不 A``），所以这里要证明那个形状够紧。

    上面这四个高频四字格中间都夹着一个 ``不``，看起来像正反问，实际不匹配——
    它们那个 ``不`` 两侧的字并不相同。把形状放宽（比如只找 ``不`` 并看左右有没有
    重复字）就会开始误伤它们，而这一类词在日常话里出现得一点都不少。
    """
    assert PerItemPolicy().drafts(turn(text))[0].kind is MemoryKind.FACT


# ── 进库 ────────────────────────────────────────────────────────────────────


def test_a_recorded_turn_is_in_the_library(store: MemoryStore) -> None:
    written = MemoryWriter(store).record(turn("路由器放在书房书架顶层。"))
    assert len(written) == 1
    assert store.count() == 1
    stored = store.get(written[0].memory_id)
    assert stored is not None
    assert stored.content == "路由器放在书房书架顶层。"
    assert stored.tier is Tier.WORKING
    assert stored.taint is Taint.USER
    assert stored.kind is MemoryKind.FACT


def test_it_is_searchable_the_moment_it_is_written(store: MemoryStore) -> None:
    """**"每轮实时写入"要的就是这个**，不是"迟早会写进去"。

    用户会立刻做的那件事是：说一句事实、把程序关掉、重开、问它。异步写会让这一
    条丢掉——而它恰好是最容易被测试的那一条。
    """
    written = MemoryWriter(store).record(turn("路由器放在书房书架顶层。"))
    assert written[0].lex_sdr is not None
    assert written[0].lex_sdr.idx.size > 0, "词法 SDR 是 MMR 去冗与关联基质的地基，不能空"


def test_the_model_side_of_the_memory_is_left_empty_for_the_backfill(store: MemoryStore) -> None:
    """写入**不调模型**。

    嵌入是异步补的：``space_id`` 为空就是"还没回填"的标记，回填器按
    ``missing_embedding_ids`` 找它们。同步嵌入会让"说一句话"变成一次模型调用，
    而那正是 P0 花了一整个验收周期才拆掉的那种耦合。
    """
    written = MemoryWriter(store).record(turn("路由器放在书房书架顶层。"))
    assert written[0].space_id == ""
    assert written[0].sem_sdr is None


def test_the_association_sdr_is_not_computed_yet(store: MemoryStore) -> None:
    """它的消费者是 P4 的神经元级基质 W。现在算出来只会在库里留一列没人读的数
    ——而没人读的列会和它的含义一起烂掉。"""
    written = MemoryWriter(store).record(turn("路由器放在书房书架顶层。"))
    assert written[0].assoc_sdr is None


def test_the_source_pointer_says_which_turn_it_came_from(store: MemoryStore) -> None:
    """形状 ``turn://<会话 id>/<轮次>``。

    **它同时是将来去重的键**：P1 不会产生重复（一轮只写一次，没有"重新生成"这条
    路），但 P3 的 agent 循环会因为重试再次落盘同一轮。现在把它写对，那一刻就只是
    加一个查询。
    """
    written = MemoryWriter(store).record(turn("路由器放在书房书架顶层。", turn_seq=7))
    assert written[0].source_ref == "turn://20261010-120000-abcdef/7"


def test_the_reply_is_not_stored(store: MemoryStore) -> None:
    """回复是从记忆和这一轮输入推出来的。存它等于同一件事存两遍，而第二遍还带着
    模型的幻觉。P3 的任务记忆是例外且仅此一例。"""
    written = MemoryWriter(store).record(
        turn("路由器放在书房书架顶层。", reply_text="好的，我记住了。")
    )
    assert [item.content for item in written] == ["路由器放在书房书架顶层。"]


def test_a_turn_with_nothing_to_remember_writes_nothing(store: MemoryStore) -> None:
    assert MemoryWriter(store).record(turn("好的")) == []
    assert store.count() == 0


def test_an_interrupted_turn_is_still_recorded(store: MemoryStore) -> None:
    """ "用户打断了"本身有信息量。丢掉它等于丢掉"这件事没说完"这件事。"""
    written = MemoryWriter(store).record(turn("路由器放在书房书", interrupted=True))
    assert len(written) == 1
    assert written[0].meta["interrupted"] is True


def test_a_text_the_encoder_cannot_handle_does_not_kill_the_turn(store: MemoryStore) -> None:
    """编不出 SDR 的记忆在库里是一颗哑弹——参与不了去冗，也几乎召不回来。

    跳过它。**但必须是跳过而不是让整轮失败**：这一轮可能还有别的草稿值得记，
    而"因为一条坏草稿整轮没写"是最难排查的那类丢失。
    """

    class BadPolicy:
        def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
            return [
                MemoryDraft(content="。。。", kind=MemoryKind.FACT),
                MemoryDraft(content="路由器放在书房书架顶层。"),
            ]

    written = MemoryWriter(store, policy=BadPolicy()).record(turn("随便"))
    assert len(written) == 1, "好的那条该进去"
    assert written[0].content == "路由器放在书房书架顶层。"


# ── 接缝 4：粒度真的能换 ────────────────────────────────────────────────────


def test_swapping_the_policy_changes_what_lands_in_the_library(store: MemoryStore) -> None:
    """**接缝 4 的验收。** 这正是 P3 的 agent 开关要做的事。

    一个"一次任务 = 一条记忆"的策略：把这一轮的用户输入和回复合成**一条**记忆，
    摘要层是回复，全文进 ``meta``（§4.8）。存储层对此一无所知——它只知道
    ``content`` 和 ``meta``，这正是 ``memcore`` 永远不需要认识工具的原因。

    如果这条测试写不出来，说明接缝是假的：P3 到那天会发现写入路径里有一处硬编码
    的判断，而那条路径已经被 P1 的测试覆盖住了。

    **这里的 ``StandInTaskPolicy`` 故意是个自造的策略、不是真的
    :class:`TaskPolicy`。** 它证的是"接缝认的是协议、不认具体是谁"，那是另一件
    事——真那个的行为在下面那节里钉。
    """

    class StandInTaskPolicy:
        def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
            return [
                MemoryDraft(
                    content=turn.reply_text,
                    kind=MemoryKind.TOOL_RESULT,
                    taint=Taint.TOOL,
                    meta={"full_text": turn.user_text, "tool_trace": list(turn.tool_trace)},
                )
            ]

    writer = MemoryWriter(store, policy=StandInTaskPolicy())
    written = writer.record(
        Turn(
            user_text="读一下 README 然后总结。",
            reply_text="README 说明了三层装配。",
            session_id="20261010-120000-abcdef",
            turn_seq=1,
            tool_trace=({"tool": "read_file", "args": {"path": "README.md"}},),
        )
    )
    assert len(written) == 1, "一次任务一条，不是一轮一条"
    assert written[0].content == "README 说明了三层装配。", "被召回的是摘要层"
    assert written[0].kind is MemoryKind.TOOL_RESULT
    assert written[0].meta["full_text"] == "读一下 README 然后总结。"
    assert written[0].meta["tool_trace"][0]["tool"] == "read_file"


def test_the_two_policies_are_mutually_exclusive_by_construction(store: MemoryStore) -> None:
    """同一个写手、同一轮输入，换策略就换结果——不需要重建写手。

    开关在**每轮写入时**读，所以中途切换开关时前期按条目、后期按任务，各自带
    ``meta`` 说明来源模式（§4.8）。重建写手也能做到这件事，但那样"切换"就变成了
    一个必须被记住的纪律，而纪律会漏。
    """

    class OneBigDraft:
        def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
            return [MemoryDraft(content=f"整轮合并：{turn.user_text}")]

    writer = MemoryWriter(store)
    writer.record(turn("路由器放在书房书架顶层。"))
    writer.policy = OneBigDraft()
    writer.record(turn("每周三晚上八点要开组会。"))

    contents = {item.content for item in store.load_all()}
    assert contents == {
        "路由器放在书房书架顶层。",
        "整轮合并：每周三晚上八点要开组会。",
    }


def test_the_policy_is_a_runtime_checkable_protocol() -> None:
    """`isinstance` 能用，是为了让"这个对象到底能不能当策略"在装配处就能问出来，
    而不是等第一轮写入时炸在一条 AttributeError 上。"""
    from biosnn_memterm.memory import MemoryPolicy

    assert isinstance(PerItemPolicy(), MemoryPolicy)


# ── 接缝 4 的兑现：TaskPolicy ────────────────────────────────────────────────
#
# 上面那条接缝证的是"换得动"，这一节证的是**换上去的那个真的对**。判据只有一条：
# 这一轮有没有工具轨迹。**没轨迹的一轮必须逐字段等于条目模式**——那是"开关是加法"
# 这句话在代码里的唯一落点，也是这一节里最容易悄悄回潮的一处。


def a_step(
    call_id: str = "c1",
    name: str = "read_file",
    *,
    content: str = "文件内容。",
) -> ToolFinished:
    """一次执行完的工具调用。**真的 dataclass**：字段改名这类改动要能在这里红。"""
    return ToolFinished(
        call_id=call_id,
        name=name,
        ok=True,
        content=content,
        denied=False,
        clipped=False,
    )


def task_turn(user: str, *, reply: str = "做完了。", **kwargs) -> Turn:
    """一轮**调过工具**的对话。轨迹走真的 :func:`build_trace`，不手搓字典——
    手搓的话，"轨迹的形状变了"这件事在这一层就看不出来。"""
    return turn(user, reply_text=reply, tool_trace=build_trace([a_step()], []), **kwargs)


def size_of(value: object) -> int:
    """序列化之后多少字符。**``ensure_ascii=False``**：默认转义会把汉字算成 6 个
    字符，于是量出来的不是真正进库的那一份。"""
    return len(json.dumps(value, ensure_ascii=False))


def test_a_task_is_remembered_as_the_answer_not_as_the_request() -> None:
    """**同一轮，两种读法。**

    两种粒度都是"一轮一条"，变的不是条数而是**那一条说的是什么**：条目模式记用户的
    断言，任务模式记那个回答。用户过后要问的是"上次让它总结的那份 README 说了什么"，
    不是"我上次提了个什么要求"。

    所以这条测试拿**同一个** :class:`Turn` 喂两个策略——两边各自量一遍。只断言任务
    模式那一边的话，一个"两种模式其实没差别"的实现照样会绿。
    """
    user = "读一下 README 然后总结。"
    asked = task_turn(user, reply="README 说了三层装配。")

    (plain,) = PerItemPolicy().drafts(asked)
    assert plain.content == user, "条目模式记的是用户那句"
    assert plain.taint is Taint.USER

    (draft,) = TaskPolicy().drafts(asked)
    assert draft.content == "README 说了三层装配。", "被召回的是回答，不是用户那句要求"
    assert draft.kind is MemoryKind.TOOL_RESULT
    assert draft.taint is Taint.TOOL, "正文是模型的产物，不该拿 USER 的那份权重"


def test_a_turn_without_a_trace_is_field_for_field_the_per_item_one() -> None:
    """**开关是加法，不是替换。**

    打开"任务记忆"却让一部分对话从此不再被记住，是那种界面上完全看不出来的副作用。
    所以没调工具的那些轮次要**逐字段相同**地落在条目模式手里。

    **交错着量**：输入里既有会被记住的陈述，也有会被挡掉的附和与问句。只比"两边都
    返回空"的那几条是量不出东西的——最后那行断言保证上面比过的至少有一组非空。
    """
    texts = [
        "路由器放在书房书架顶层。",
        "每周三晚上八点要开组会。",
        "我喜欢喝黑咖啡，不加糖。",
        "你还记得路由器放哪儿了吗？",
        "好的。",
    ]
    for text in texts:
        plain = PerItemPolicy().drafts(turn(text))
        under_task = TaskPolicy().drafts(turn(text))
        assert [(d.content, d.kind, d.taint, d.weight, d.meta) for d in under_task] == [
            (d.content, d.kind, d.taint, d.weight, d.meta) for d in plain
        ], f"没调工具的一轮在任务模式下变了：{text!r}"

    assert PerItemPolicy().drafts(turn(texts[0])), "上面比的是两组空，什么都没证明"


def test_the_task_memory_carries_the_trace_and_the_original_words() -> None:
    """``meta`` 的键集合是**定死的**。

    它是逐字进 SQLite 的 JSON，键名写错一个字母不会有任何东西变红——只是那一项永远
    读不出来，而"这次任务是怎么做的"恰好只存在这一处。
    """
    draft = TaskPolicy().drafts(
        task_turn("读一下 README 然后总结。", reply="README 说了三层装配。", interrupted=True)
    )[0]
    assert set(draft.meta) == {
        "mode",
        "turn_seq",
        "interrupted",
        "tool_trace",
        "full_text",
        "full_text_chars",
    }
    assert draft.meta["mode"] == "task"
    assert draft.meta["turn_seq"] == 1
    assert draft.meta["interrupted"] is True
    assert [step["name"] for step in draft.meta["tool_trace"]] == ["read_file"]
    assert draft.meta["full_text"] == "读一下 README 然后总结。"
    assert draft.meta["full_text_chars"] == len("读一下 README 然后总结。")


def test_an_interrupted_task_that_said_nothing_falls_back_to_the_user_sentence() -> None:
    """被打断、一个字都没吐出来时，正文回落成用户那句。

    **这一轮仍然发生过。** 丢掉它的后果是：用户记得自己让助手做过一件事，而记忆库
    里查无此事——恰好是最容易被当成"它忘了"的那一类。
    """
    (draft,) = TaskPolicy().drafts(
        task_turn("读一下 README 然后总结。", reply="", interrupted=True)
    )
    assert draft.content == "读一下 README 然后总结。"
    assert draft.meta["interrupted"] is True


def test_a_task_with_nothing_written_anywhere_gives_nothing() -> None:
    """连用户那句都是空的（或全是空白）时返回空，**不是**一条正文为空的记忆。

    正文为空的条目在库里是一行读不出内容的死数据，而 ``content`` 是它的全部。
    """
    assert TaskPolicy().drafts(task_turn("   ", reply="  ")) == ()


def test_the_original_words_are_cut_at_the_limit_but_their_length_is_kept() -> None:
    """``full_text`` 截到 :data:`MAX_TASK_TEXT_CHARS`，``full_text_chars`` 记原长。

    开了任务模式之后，用户那句话在库里**没有第二处落脚**（正文是回复），所以这一格
    是它的来历。没有原长那一格，"用户只说了 2000 字"与"本来有 5 万字、这里只留了
    2000"在读数据时是分不开的。
    """
    long_text = "字" * (MAX_TASK_TEXT_CHARS + 500)
    (draft,) = TaskPolicy().drafts(task_turn(long_text))
    assert len(draft.meta["full_text"]) == MAX_TASK_TEXT_CHARS
    assert draft.meta["full_text_chars"] == MAX_TASK_TEXT_CHARS + 500


def test_the_task_policy_is_pure() -> None:
    """同样的输入，同样的输出。不读时钟、不碰库、不看开关。

    "用哪套粒度"是装配层的选择（``MainWindow`` 按设置装），策略自己一旦去读设置，
    "两轮相同的输入该不该得到同样的记忆"就再也测不了了。
    """
    policy = TaskPolicy()
    first = policy.drafts(task_turn("读一下 README 然后总结。"))
    for _ in range(5):
        again = policy.drafts(task_turn("读一下 README 然后总结。"))
        assert [d.meta for d in again] == [d.meta for d in first]


def test_a_sixty_four_kib_task_still_lands_in_the_library(store: MemoryStore) -> None:
    """**整组截断的兑现。** ``agent/fs.py`` 的 ``MAX_READ_BYTES`` 就是 64 KiB，而
    ``meta`` 超了 ``memcore`` 的硬上限时 ``MemoryStore.add`` 抛，:class:`MemoryWriter`
    **有意不吞**——那条路的终点是对话里的一句「记忆写入失败」。

    所以这条测的是两件事一起：读一次 64 KiB 的文件**照样写得进去**（``record`` 不抛、
    返回一条），而且存下来的那份**是有界的**（正文摘录、``result_chars`` 说明原文多长）。
    """
    huge = "很长的文件内容。" * 8_000  # 64 KiB 那一档
    trace = build_trace([a_step(content=huge)], [])
    writer = MemoryWriter(store, policy=TaskPolicy())

    written = writer.record(
        turn("读一下那个大文件然后总结。", reply_text="它很长。", tool_trace=trace)
    )

    assert len(written) == 1, "一次任务一条"
    (step,) = written[0].meta["tool_trace"]
    assert step["result_chars"] == len(huge), "摘录说得出口原文有多长"
    assert len(step["result"]) < 1_000
    assert size_of(written[0].meta) < 16_000, "一条 ``meta`` 不该有 64 KiB 的读取那么大"


# ── 纯决策 ──────────────────────────────────────────────────────────────────


def test_the_policy_is_pure() -> None:
    """同样的输入，同样的输出。不读时钟、不碰库、不看上一个会话。"""
    policy = PerItemPolicy()
    first = policy.drafts(turn("路由器放在书房书架顶层。"))
    for _ in range(5):
        again = policy.drafts(turn("路由器放在书房书架顶层。"))
        assert [(d.content, d.kind, d.weight) for d in again] == [
            (d.content, d.kind, d.weight) for d in first
        ]


def test_the_draft_carries_no_identifier_and_no_timestamp() -> None:
    """**这是策略能保持纯的原因。**

    ULID 与时间戳由写手给（它才认识时钟和库）；策略一旦自己造 id，它就再也测不了
    "两轮相同的输入该不该得到两条记忆"这个问题。
    """
    draft = PerItemPolicy().drafts(turn("路由器放在书房书架顶层。"))[0]
    assert not hasattr(draft, "memory_id")
    assert not hasattr(draft, "created_at")


def test_the_two_policies_must_not_share_a_draft_shape_shortcut() -> None:
    """草稿的字段是**两条路径的公共形状**。

    如果 P1 的草稿带了只有条目模式才有的东西（比如 ``lex_sdr``），任务模式的策略
    就得编一个它根本没有的值——而那个值会被存进库、再被召回读到。
    """
    fields = set(MemoryDraft.__dataclass_fields__)
    assert fields == {"content", "kind", "taint", "weight", "meta"}


def test_normalize_and_tokenize_are_the_memcore_ones() -> None:
    """策略用的是**库自己的**字符分类与归一化，不是另写一套。

    另写一套的后果是：策略判"这是附和"、而 SDR 编码看到的是另一串字符——两处对
    同一句话的理解不一致，且没有任何地方会报错。
    """
    assert normalize("ＯＫ") == "ok"
    assert tokenize("。。。") == [], "全标点在库的分词里就没有 token——和实义字符数为 0 是同一件事"
    assert [c for c in "？？？" if is_word(c)] == []


def test_the_murmur_rule_does_not_depend_on_the_tokenizer() -> None:
    """**这条规则踩过一次坑，所以单独钉住。**

    第一版查的是"分词后只有一个不同的 token"，而它永远不会触发：库的分词器同时
    产出单字 token 和二元组 token（``'哈哈哈哈'`` → ``c:哈`` 与 ``b:c:哈:c:哈``）。
    静默失效的过滤器看起来和"没有这种输入"一模一样。
    """
    assert len(set(tokenize("哈哈哈哈"))) == 2, "分词给的是 2 个不同的 token，不是 1 个"
    assert PerItemPolicy().drafts(turn("哈哈哈哈")) == (), "但规则照样要挡住它"
