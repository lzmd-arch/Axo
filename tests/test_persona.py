"""人格锚的两条角色：内置那条（程序写的，不可改不可删）与用户那条（可改可删）。

这个文件守的第一件事是**"用户写了一条"不能冒充"内置那条已经在了"**。旧实现按
``kind`` 判存在，在只有一个角色时是对的；现在库里故意有两条 ``PERSONA_ANCHOR``，
再按 kind 判就会漏写内置那条——症状是库里的身份约束凭空少了一半，而没有任何一处
报错（``persona_texts`` 只是少返回一条字符串）。

第二件事是用户那条的**改**要跟记忆窗口的改走同一条路：重算词法编码、丢掉旧向量。
只改 ``content`` 不改 ``lex_sdr`` 的后果见 test_memory_edit.py 的文件头——那条路
是静默的，所以这里再验一次。

第三件事是**内置那条的版本与升级**。加版本号之前，改 ``DEFAULT_PERSONA`` 碰不到任何
已经建过锚的库：新装的库有那句、老库没有，而"任何模型读到的身份完全一样"随即变成
"同一个库上一样、跨库不一样"。这里钉住升级的四个性质：**没有版本键就是第 1 版**
（真实的库全是这一种）、升级是**原地改**（id 不变、条数不变）、**比当前版新就不动**
（不降级）、以及**不碰用户那条**。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biosnn_memterm.memory.edit import can_encode
from biosnn_memterm.memory.persona import (
    BUILTIN_PERSONA_VERSION,
    BUILTIN_REF,
    DEFAULT_PERSONA,
    USER_REF,
    builtin_anchor,
    builtin_anchor_version,
    clear_user_persona,
    ensure_builtin_anchor,
    needs_persona,
    persona_texts,
    read_only_reason,
    set_user_persona,
    user_anchor,
)
from memcore.errors import SDRCapacityError
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

SPACE = make_space_id("ollama", "nomic-embed-text", 768)


@pytest.fixture
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


def _handmade(text: str, *, source_ref: str, created_at: float = 0.0) -> MemoryItem:
    """直接往库里塞一条人格锚。``created_at`` 显式给，好让"谁先建的"是构造出来的。"""
    fields = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.PERSONA_ANCHOR,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 1.0,
        "pinned": True,
        "lex_sdr": encode_lexical(text),
        "source_ref": source_ref,
    }
    if created_at:
        fields["created_at"] = created_at
    return MemoryItem(**fields)  # type: ignore[arg-type]


def _anchors(store: MemoryStore) -> list[MemoryItem]:
    return [item for item in store.pinned_items() if item.kind is MemoryKind.PERSONA_ANCHOR]


# ── 内置那条 ────────────────────────────────────────────────────────────────


def test_a_fresh_library_gets_the_builtin_anchor(store: MemoryStore) -> None:
    created = ensure_builtin_anchor(store)

    assert created is not None
    assert created.source_ref == BUILTIN_REF
    assert created.content == DEFAULT_PERSONA
    assert created.pinned is True
    assert created.tier is Tier.LONGTERM
    assert created.weight == 1.0


def test_the_builtin_anchor_is_written_only_once(store: MemoryStore) -> None:
    """第二次调用什么都不做——返回 ``None`` 且条数不变。

    只看返回值不够：``ensure_*`` 的语义是"库里只有一条"，所以条数才是断言。
    """
    ensure_builtin_anchor(store)

    assert ensure_builtin_anchor(store) is None
    assert len(_anchors(store)) == 1


# ── 版本与升级 ──────────────────────────────────────────────────────────────


def test_a_new_anchor_is_written_with_the_current_version(store: MemoryStore) -> None:
    """新写的那条带着版本号——它是下一轮"要不要升"的全部依据。"""
    ensure_builtin_anchor(store)

    assert builtin_anchor_version(store) == BUILTIN_PERSONA_VERSION


def test_an_anchor_written_before_versioning_counts_as_the_first_version(
    store: MemoryStore,
) -> None:
    """**升级路径的入口就是"没有那个键"。** 加版本号之前写进库里的锚没有它。

    这一条不是构造出来的边角：**真实的库现在全都是这一种**（用户那台机器上那条
    就建在加版本号之前）。所以"没有键"必须读成"第 1 版"，而不是 0、也不是"未知"
    ——读成 0 会让每次启动都判成"要升"，读成未知就永远升不上来。
    """
    old = _handmade("第 1 版的内置人格锚。", source_ref=BUILTIN_REF)
    store.add(old)

    upgraded = ensure_builtin_anchor(store)

    assert upgraded is not None, "没有版本键的锚必须被判成旧版"
    assert builtin_anchor_version(store) == BUILTIN_PERSONA_VERSION


def test_an_upgrade_rewrites_in_place_and_keeps_the_bookkeeping(
    store: MemoryStore,
) -> None:
    """升级是**原地改**：id 不变、条数不变、权重与访问史不变。

    删了重写也能把正文换对，代价是三样都不报错地变掉：关联图的节点换了 id（旧节点
    留在图上指向一条不存在的记忆）、记忆界面上那条的选中状态丢了、``created_at``
    变成今天（读起来像"今天新加了一条人设"）。
    """
    old = _handmade("第 1 版的内置人格锚。", source_ref=BUILTIN_REF)
    store.add(old)
    store.touch_access([old.memory_id], at=1.0)

    upgraded = ensure_builtin_anchor(store)

    assert upgraded is not None
    assert upgraded.memory_id == old.memory_id
    assert len(_anchors(store)) == 1
    assert upgraded.content == DEFAULT_PERSONA
    # 词法编码必须跟着正文走——只改 content 不改 lex_sdr 是这一族里最典型的静默错。
    assert upgraded.lex_sdr == encode_lexical(DEFAULT_PERSONA)
    assert upgraded.pinned is True
    # 访问史也留着：``edited_content`` 一字不改地把 ``access_count`` 带过来，换成
    # "删了重写"这条路它会归零，而"改过正文的那条不再是被召回过的那条"没人看得出来。
    assert upgraded.access_count == 1
    # 升完就到当前版，第二次调用什么都不做。
    assert ensure_builtin_anchor(store) is None


def test_an_upgrade_does_not_touch_the_users_own_anchor(store: MemoryStore) -> None:
    """内置那条升级**不碰**用户写的那条——两条的角色不同，升级只针对前者。"""
    store.add(_handmade("第 1 版的内置人格锚。", source_ref=BUILTIN_REF))
    set_user_persona(store, "说话简短一点。")

    ensure_builtin_anchor(store)

    assert user_anchor(store) is not None
    assert user_anchor(store).content == "说话简短一点。"  # type: ignore[union-attr]
    assert len(_anchors(store)) == 2


def test_an_anchor_from_a_later_version_is_not_downgraded(store: MemoryStore) -> None:
    """比当前版**新**就什么都不做。

    降级会把用户已经读到的正文换回旧的，而且换的是"身份"那一份——这一次改动没有
    任何界面痕迹。版本号存在的意义一半在这里：比较有方向。
    """
    future = BUILTIN_PERSONA_VERSION + 1
    ensure_builtin_anchor(store, text="从未来版本写下来的正文。", version=future)

    assert ensure_builtin_anchor(store) is None
    assert builtin_anchor_version(store) == future
    assert builtin_anchor(store) is not None
    assert builtin_anchor(store).content == "从未来版本写下来的正文。"  # type: ignore[union-attr]


def test_an_upgrade_drops_the_now_stale_embedding(store: MemoryStore) -> None:
    """正文换了，盘上那份语义向量就旧了——而**"旧"不是"缺"**。

    回填那条路只看 :meth:`~memcore.store.MemoryStore.missing_embedding_ids`，所以
    旧向量不会自己好：它会一直按第 1 版的正文去回答"某条记忆与人格锚有多像"。
    顺手钉住顺序（改成功了才丢），与 :func:`set_user_persona` 一致。
    """
    old = _handmade("第 1 版的内置人格锚。", source_ref=BUILTIN_REF)
    store.add(old)
    # 往未登记的空间写向量会被拒（那条拒绝有它自己的理由），所以先登记。
    store.register_space(Space(space_id=SPACE, dim=768, model_ref="nomic-embed-text"))
    store.set_embedding(old.memory_id, SPACE, np.arange(768, dtype=np.float32))

    ensure_builtin_anchor(store)

    assert store.get_embedding(old.memory_id, SPACE) is None


def test_the_persona_forbids_claiming_a_tool_call_that_never_happened(
    store: MemoryStore,
) -> None:
    """第 2 版补的那句必须真的在**发出去的正文**里。

    这条断言故意是弱的（查一句话在不在），因为强的那一条断言不了：模型会不会照做
    只有实测能回答，而它是一句要求、不是闸门（真正挡住"说了没做"的是界面里那条
    真实工具轨迹与确认闸门）。

    弱断言在这里仍然有用：它挡的是"有人重写 DEFAULT_PERSONA 时把这句弄丢了"——
    那种丢法是静默的，而症状（模型声称做过没做的事）离这段正文很远。

    断言打在 :func:`persona_texts` 上而不是常量上：常量对而**没进库**（比如升级那条
    路忘了写正文）时，模型读到的还是没有这句。
    """
    ensure_builtin_anchor(store)

    assert "只有真的调用过工具才算用过工具" in persona_texts(store)[0]


def test_a_user_anchor_does_not_count_as_the_builtin_one(store: MemoryStore) -> None:
    """**本版最容易写错的一处。** 库里只有用户那条时，内置那条仍然要被写进去。

    按 ``kind`` 判（旧实现）在这里会返回 ``None``：用户写了一条 → "已经有人格锚了"
    → 内置那条永远不补。于是库里只有用户那句"说话简短点"，而"你的记忆不属于任何
    模型"这句身份约束不见了，全程无声。
    """
    store.add(_handmade("说话简短一点。", source_ref=USER_REF))

    created = ensure_builtin_anchor(store)

    assert created is not None, "用户那条把内置那条骗过去了"
    assert created.source_ref == BUILTIN_REF
    assert len(_anchors(store)) == 2


def test_a_second_builtin_is_never_added_beside_a_user_anchor(store: MemoryStore) -> None:
    """两条都在之后，启动多少次都不会长出第三条。"""
    ensure_builtin_anchor(store)
    set_user_persona(store, "说话简短一点。")

    assert ensure_builtin_anchor(store) is None
    assert len(_anchors(store)) == 2


# ── 用户那条：写与改 ────────────────────────────────────────────────────────


def test_writing_the_user_persona_creates_a_pinned_longterm_anchor(store: MemoryStore) -> None:
    item = set_user_persona(store, "叫我小林，说话直接一点，不要客套话。")

    assert item.source_ref == USER_REF
    assert item.pinned is True, "没钉住就进不了 persona_texts（它只看 pinned_items）"
    assert item.tier is Tier.LONGTERM
    assert item.weight == 1.0


def test_writing_twice_keeps_exactly_one_user_anchor(store: MemoryStore) -> None:
    first = set_user_persona(store, "说话简短。")
    second = set_user_persona(store, "说话简短，另外叫我老离。")

    assert second.memory_id == first.memory_id, "应该改那一条，不是再建一条"
    assert second.content == "说话简短，另外叫我老离。"
    assert len([item for item in _anchors(store) if item.source_ref == USER_REF]) == 1


def test_editing_the_user_persona_recomputes_the_lexical_sdr(store: MemoryStore) -> None:
    """改了字就必须换编码——理由见 test_memory_edit.py 的文件头。"""
    first = set_user_persona(store, "说话简短。")
    second = set_user_persona(store, "说话详细一点，多举例子。")

    assert second.content != first.content
    reference = encode_lexical(second.content)
    assert np.array_equal(second.lex_sdr.idx, reference.idx)
    assert np.array_equal(second.lex_sdr.weights, reference.weights)
    assert not np.array_equal(second.lex_sdr.idx, first.lex_sdr.idx)


def test_editing_the_user_persona_drops_the_stale_vector(store: MemoryStore) -> None:
    """旧向量是旧意思的，必须丢掉交回填管线重算。

    留着它的后果是静默的：召回会按旧意思命中这条，而详情面板显示的是新正文。
    """
    store.register_space(Space(space_id=SPACE, dim=768, model_ref="nomic-embed-text"))
    item = set_user_persona(store, "说话简短。")
    store.set_embedding(item.memory_id, SPACE, np.ones(768, dtype=np.float32))
    assert store.get_embedding(item.memory_id, SPACE) is not None

    set_user_persona(store, "说话简短，另外叫我老离。")

    assert store.get_embedding(item.memory_id, SPACE) is None


def test_writing_the_same_text_again_changes_nothing(store: MemoryStore) -> None:
    """一个字都没变时**不写库、不丢向量**。

    否则每次打开设置窗口点一次保存，都会把一条好端端的向量白丢掉，然后在后台
    重算一遍。
    """
    store.register_space(Space(space_id=SPACE, dim=768, model_ref="nomic-embed-text"))
    item = set_user_persona(store, "说话简短。")
    store.set_embedding(item.memory_id, SPACE, np.ones(768, dtype=np.float32))

    again = set_user_persona(store, "说话简短。")

    assert again.memory_id == item.memory_id
    assert store.get_embedding(item.memory_id, SPACE) is not None


def test_empty_text_is_refused_with_the_reason_the_editor_gives(store: MemoryStore) -> None:
    reason = can_encode("")

    with pytest.raises(ValueError) as caught:
        set_user_persona(store, "")

    assert reason is not None
    assert str(caught.value) == reason, "界面拦下的理由与库里抛的理由必须同源"
    assert user_anchor(store) is None


def test_punctuation_only_is_refused(store: MemoryStore) -> None:
    with pytest.raises(SDRCapacityError):
        set_user_persona(store, "。。。！？")


# ── 用户那条：删 ────────────────────────────────────────────────────────────


def test_clearing_removes_only_the_user_anchor(store: MemoryStore) -> None:
    ensure_builtin_anchor(store)
    set_user_persona(store, "说话简短一点。")

    assert clear_user_persona(store) is True

    assert user_anchor(store) is None
    assert builtin_anchor(store) is not None, "删人设不该把身份也抹了"
    assert len(_anchors(store)) == 1


def test_clearing_when_there_is_nothing_to_clear_says_so(store: MemoryStore) -> None:
    assert clear_user_persona(store) is False


def test_needs_persona_flips_with_the_user_anchor(store: MemoryStore) -> None:
    assert needs_persona(store) is True

    set_user_persona(store, "说话简短。")
    assert needs_persona(store) is False

    clear_user_persona(store)
    assert needs_persona(store) is True


# ── 装配顺序与空态 ──────────────────────────────────────────────────────────


def test_persona_texts_puts_the_builtin_one_first(store: MemoryStore) -> None:
    """**内置在前，且与谁先建无关。**

    这一条不是洁癖：返回值按顺序拼进 system 消息，身份那句在风格那句之前读起来
    才是"你是谁，你要怎么说话"。而插入序（``pinned_items`` 按 ``created_at``）在
    这里被刻意反过来——用户那条先建——所以这条断言真的在测显式排序。
    """
    store.add(_handmade("说话简短一点。", source_ref=USER_REF, created_at=1_700_000_000.0))
    store.add(_handmade(DEFAULT_PERSONA, source_ref=BUILTIN_REF, created_at=1_700_000_060.0))

    assert persona_texts(store) == [DEFAULT_PERSONA, "说话简短一点。"]


def test_persona_texts_skips_blank_entries(store: MemoryStore) -> None:
    """``"   "`` 过得了 ``MemoryItem`` 的非空检查，但它是空的。

    ``lex_sdr`` 只能是 ``None``：全空白的正文连词法 SDR 都编码不出来（这也正是
    ``set_user_persona`` 走不到这条路径的原因——它拦在前面）。所以这条守的是别的
    写入路径进来的条目，比如将来的导入。
    """
    ensure_builtin_anchor(store)
    store.add(
        MemoryItem(
            memory_id=new_ulid(),
            kind=MemoryKind.PERSONA_ANCHOR,
            tier=Tier.LONGTERM,
            content="   ",
            pinned=True,
            source_ref=USER_REF,
        )
    )

    assert persona_texts(store) == [DEFAULT_PERSONA]


def test_persona_texts_is_empty_when_both_anchors_are_gone(store: MemoryStore) -> None:
    """一条都没有**不是异常状态**：装配器照常工作，只是模型没有身份约束。"""
    ensure_builtin_anchor(store)
    set_user_persona(store, "说话简短。")
    for item in _anchors(store):
        store.delete(item.memory_id)

    assert persona_texts(store) == []


def test_an_unknown_source_is_still_carried_along(store: MemoryStore) -> None:
    """来源既不是内置也不是用户的那种，**不丢**——将来多一种角色时它就是那个人格。"""
    store.add(_handmade("说话用文言文。", source_ref=""))

    assert persona_texts(store) == ["说话用文言文。"]


# ── 界面上不许改不许删的那条 ────────────────────────────────────────────────


def test_read_only_reason_protects_only_the_builtin_anchor(store: MemoryStore) -> None:
    created = ensure_builtin_anchor(store)
    assert created is not None

    assert read_only_reason(created) is not None

    written = set_user_persona(store, "说话简短。")
    assert read_only_reason(written) is None, "用户自己写的必须能改"

    handmade = store.add(_handmade("说话用文言文。", source_ref=""))
    assert read_only_reason(handmade) is None, "来源不明的按用户的理解，不能锁死"


def test_read_only_reason_says_nothing_about_other_kinds(store: MemoryStore) -> None:
    """别的东西哪怕来源指针写成了内置那个值，也不该被锁——判定要看两样。"""
    item = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.FACT,
        tier=Tier.LONGTERM,
        content="我家的路由器放在书房。",
        source_ref=BUILTIN_REF,
    )
    assert read_only_reason(store.add(item)) is None
