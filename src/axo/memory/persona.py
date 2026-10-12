"""人格锚：L0 层唯一的内容，也是"换模型人格不变"的唯一载体。

**它是一条普通的记忆条目**——``kind=PERSONA_ANCHOR``、``pinned=True``，其余字段
与别的记忆没有区别。这样做的理由见计划书 §3.2：人格不是一套独立机制，它只是
"永远不会被驱逐的那条记忆"。多建一套机制，就多一处会与记忆库不一致的地方。

**内容一个字都不许改写。** 装配器把它原样发出去（:func:`assemble` 的 ``persona``
参数），没有任何模板包装、没有任何模型参与加工。哪天它变成"根据当前对话润色过的
人格"，"任何模型读到的完全一样"这句话就不再成立——而失效方式是完全静默的：每个
模型看到的还是"一段人格"，只是各不相同。

**两条锚，不是一个角色分两半。** :data:`BUILTIN_REF` 那条是程序写的（"这套系统是
什么"），用户改不了也删不掉；:data:`USER_REF` 那条是用户自己写的（"你想让我怎么
说话"），可改可删、可以是空的。两者的区别**只由 ``source_ref`` 承载**，不新增
:class:`~memcore.types.MemoryKind`：那要动 ``memcore`` 的词表与界面上的中文标签表，
只为区分两条同一个 ``kind`` 的条目，不划算。而 ``source_ref`` 本来就是"这条是从
哪儿来的"，``persona://default`` 一开始就是按这个意思用的。
"""

from __future__ import annotations

from memcore import MemoryItem, MemoryKind, Tier
from memcore.errors import SDRCapacityError
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore

from .edit import can_encode, edited_content

__all__ = [
    "BUILTIN_PERSONA_VERSION",
    "BUILTIN_REF",
    "DEFAULT_PERSONA",
    "USER_REF",
    "builtin_anchor",
    "clear_user_persona",
    "ensure_builtin_anchor",
    "needs_persona",
    "persona_texts",
    "read_only_reason",
    "set_user_persona",
    "user_anchor",
]

BUILTIN_REF = "persona://default"
"""内置那条的来源指针。

名字是这套设计还是"一条锚"的时候留下的，那时它就是"程序给的默认值"。现在语义
变成了"内置的、不可改"，``persona://builtin`` 更准——但改它要**改写用户库里已有
的那一个字段**，为一个纯粹的措辞收益去动用户的库不值得。所以字符串留着，代码里
给它一个名字：判断"这是不是内置那条"的地方只有这一个常量，措辞的债不会扩散。
"""

USER_REF = "persona://user"
"""用户自己写的那条的来源指针。用户看得见它（记忆详情的「来源指针」那一行）。"""

BUILTIN_PERSONA_VERSION = 3
"""内置人格锚的**第几版正文**。改 :data:`DEFAULT_PERSONA` 就要把它加一。

**为什么会需要这个数。** 只按"缺了就写"落地的话，改 ``DEFAULT_PERSONA`` 碰不到任何
**已经建过锚**的库：新装的库有这句、老库没有，而"任何模型读到的身份完全一样"那句话
立刻变成"同一个库上一样、跨库不一样"——比"大家都没有"更糟，因为它把一条不该有的差异
藏进两份都不知道自己不同的字节里。

加一版就要**升一次级**，而升级会改写用户库里那条锚的正文。这是允许的，因为那条锚
本来就只有程序能改（:func:`read_only_reason` 拦住界面上的编辑与删除），所以"正文与
常量不一致"只有一个解释：它是旧版写的。

**版本号是给方向用的，不是给"变没变"用的。** 没有它也能写成"常量与正文不同就覆盖"，
但那句话在"新版本把措辞改回去"时会往**旧**的方向走，而且它读起来像是要覆盖用户手改
过的内容——后来的人会因此不敢动这里。有了这个数，比较就是一句话：旧了就升，不旧不动。

第 1 版是**没有这个键**的那些（加版本号之前写进库里的锚），见 :func:`_anchor_version`。

第 3 版只是把首句的产品名换成了 ``Axo（阿螈）``（0.2.0 改名），**没有一个字的语义变化**。
照样要加这个数，理由就是上面那两段：已经建过锚的库不会因为"常量变了"而自己更新，除非
版本号指了方向。
"""

#: 版本号存在 :attr:`~memcore.types.MemoryItem.meta` 里的键名。
_VERSION_KEY = "builtin_persona_version"

DEFAULT_PERSONA = (
    "你是 Axo（阿螈）的助手。\n"
    "你的长期记忆存在这台机器的一个本地文件里，不属于任何模型——换一个模型来读，"
    "读到的还是同一份。\n"
    "回答时优先使用注入给你的记忆。记忆里没有的事，就说不知道，不要猜。\n"
    "只说你真的做过的事：只有真的调用过工具才算用过工具，没调用就不要说保存了、"
    "记住了或查过了。"
)
"""内置那条人格锚的正文。

**刻意只写"关于这个系统的事实"，不写性格。** 性格是用户的事——所以现在首次启动
会拿这个问题去问用户一次（对话区的那张卡片），答案落在 :data:`USER_REF` 那条里。
这里继续不替用户决定。

每一句都要同时满足"是事实"和"有用"两件事：说明它是谁（第一句）、说明它的记忆
不属于某个模型（第二句，这是整个项目的立论）、说明记忆和猜测的边界（第三句，
4B 模型最容易越界的地方）、说明**说与做的边界**（第四句）。

**第四句补的是一个真实的缺口。** 这条要求归人格锚，不归 L1：环境槽只写"这一轮发生了
什么"，而"不许声称你调用了工具"是一条指令（L1 只放事实）。缺口的表现是模型**声称做过
没做的事**——"我已经帮你记住了"而那一轮根本没调 ``write_memory``。

**它是一句要求，不是一道闸门。** 真正挡住"说了没做"的是界面里那条**真实工具轨迹**
与动作的确认闸门；这句话只是让模型不必靠猜来决定怎么说。别把这次升级当成那个问题
的修复。
"""


def _anchors(store: MemoryStore) -> list[MemoryItem]:
    """库里全部人格锚，**内置在前、用户写的在后**。

    显式排序而不是靠 ``pinned_items()`` 的插入序：插入序看起来"自然"是因为内置
    那条总是先建（它的 ULID 更小），但那是构造上的巧合，不是规则。顺序进了
    :func:`persona_texts` 的返回值，也就是进了发给模型的 system 消息，所以它该由
    这里说清楚。

    剩下的（来源指针既不是内置那条也不是用户那条的）按原样跟在最后——**不丢**。
    将来多一种角色时，它是"还没被这个模块认识的人格内容"，漏掉它比多带一条糟。
    """
    anchors = [item for item in store.pinned_items() if item.kind is MemoryKind.PERSONA_ANCHOR]
    builtin = [item for item in anchors if item.source_ref == BUILTIN_REF]
    user = [item for item in anchors if item.source_ref == USER_REF]
    rest = [item for item in anchors if item.source_ref not in (BUILTIN_REF, USER_REF)]
    return builtin + user + rest


def builtin_anchor(store: MemoryStore) -> MemoryItem | None:
    """内置那条。没有就返回 ``None``（只可能发生在 :func:`ensure_builtin_anchor`
    还没被调用过的库上）。"""
    return next((item for item in _anchors(store) if item.source_ref == BUILTIN_REF), None)


def user_anchor(store: MemoryStore) -> MemoryItem | None:
    """用户写的那条。用户没写过（或删掉了）就返回 ``None``。"""
    return next((item for item in _anchors(store) if item.source_ref == USER_REF), None)


def needs_persona(store: MemoryStore) -> bool:
    """要不要问用户一次人设。

    问的是**问题**而不是形状：界面说"还没人设，问一次"时不必自己写成
    ``user_anchor(store) is None``——那句话里的判据就得跟着这一层一起改，而它本来
    只是这个模块的实现细节。
    """
    return user_anchor(store) is None


def _anchor_version(item: MemoryItem) -> int:
    """这条内置锚的正文是**第几版**。

    **没有那个键就是第 1 版**——不是 0、也不是"未知"。加版本号之前写进库里的锚，
    正文就是第 1 版的正文，而**真实的库现在全都是这一种**。所以这条默认值不是兜底，
    它就是升级路径的入口：老库靠它被判成"旧"，然后被升上来。

    键在但值读不成整数时也按第 1 版算（``meta`` 是自由格式的，别处写进去什么都可能
    发生）：那会让它被升一次级，而升级是幂等的——正文与版本号都会被写对。
    """
    try:
        return int(item.meta.get(_VERSION_KEY, 1))
    except (TypeError, ValueError):
        return 1


def builtin_anchor_version(store: MemoryStore) -> int | None:
    """库里那条内置锚是第几版；**没有内置锚**时返回 ``None``。

    返回值分得开"没有"与"第 1 版"，是因为这两种在调用方眼里不是一件事：前者要写、
    后者要升（见 :func:`ensure_builtin_anchor`）。
    """
    existing = builtin_anchor(store)
    return None if existing is None else _anchor_version(existing)


def ensure_builtin_anchor(
    store: MemoryStore,
    *,
    text: str = DEFAULT_PERSONA,
    version: int = BUILTIN_PERSONA_VERSION,
) -> MemoryItem | None:
    """把库里那条内置锚**放到 ``version`` 版上**：没有就建，旧了就升，已经在就返回 ``None``。

    三个分支，返回 ``None`` 只有一个意思——**什么都没写**（已经在当前版上，或者比
    当前版还新）。

    **按 :data:`BUILTIN_REF` 判存在，不按 ``kind`` 判。** 这一条在本版改了：以前
    人格锚只有一个角色，"按 kind 判"就够；现在库里**故意**有两条 ``PERSONA_ANCHOR``
    （内置一条、用户一条），按 kind 判会让"用户写了一条"把内置那条骗过去——于是
    每次启动都再写一条内置的进去，两条内置的在库，而它本该是唯一的。

    **升级是原地改，不是"删了重写"。** 保住 ``memory_id`` 有三件实在的好处：关联图
    的节点、记忆界面上那条的选中状态、以及用户可能已经建立的印象（"第一条钉住的那
    个就是它"）。而"删了重写"还会让 :func:`~memcore.types.MemoryItem.created_at`
    变成今天，读起来像"今天新加了一条人设"。

    要写成 ``lex_sdr`` 也一并算出来：人格锚虽然不参与常规召回，但它要能出现在
    记忆浏览器的相似度视图里，而且"某条记忆与人格锚有多像"是个有用的信号。升级那条
    路走 :func:`~axo.memory.edit.edited_content`，与用户在界面上改正文
    是**同一条**路（重算词法 SDR、保留权重与访问史）。

    Raises:
        StoreError: 并发下那条锚刚好被删掉了（``transform`` 的保证）。启动路径上
            单线程，走不到；写在这里是因为它是 ``transform`` 的契约，不该被猜。
    """
    existing = builtin_anchor(store)
    if existing is not None:
        if _anchor_version(existing) >= version:
            return None
        revised = store.transform(
            existing.memory_id,
            lambda item: edited_content(item, text).copy(meta={**item.meta, _VERSION_KEY: version}),
        )
        # **正文变了，盘上那份语义向量就旧了，而"旧"不是"缺"。** 回填那条路只看
        # ``missing_embedding_ids``，所以旧向量不会自己好——它会一直按第 1 版的正文
        # 去比"某条记忆与人格锚有多像"。顺序照 ``set_user_persona``：改成功了才丢。
        store.drop_embedding(existing.memory_id)
        return revised

    # ``tier`` 取长期层而不是工作层：工作层的新条目会随权重衰减被降级，而人格锚不
    # 该经历那个过程。``pinned=True`` 已经让它跳过衰减了，把 tier 也写对是为了另外
    # 两件事——记忆浏览器里的分层视图不该显示"人格锚（工作层）"，以及将来若有人去掉
    # 了 pinned（唯一能让它衰退的方式），它至少从一个合理的位置开始。
    #
    # ``weight=1.0`` 而不是新记忆默认的 0.5：这个数在这里不是"它多重要"，而是
    # "它不该被任何基于权重的调制压下去"。
    anchor = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.PERSONA_ANCHOR,
        tier=Tier.LONGTERM,
        content=text,
        weight=1.0,
        pinned=True,
        lex_sdr=encode_lexical(text),
        source_ref=BUILTIN_REF,
        meta={_VERSION_KEY: version},
    )
    return store.add(anchor)


def set_user_persona(store: MemoryStore, text: str) -> MemoryItem:
    """写下（或改写）用户自己那条人设。返回落库后的那一条。

    没有就建、有就改——**库里始终最多一条** :data:`USER_REF`。改的时候照
    :meth:`~axo.ui.memory_window.MemoryWindow._on_edit` 的顺序：
    ``transform`` 成功之后才 ``drop_embedding``。顺序反了的话，万一 ``transform``
    抛异常，旧向量就白丢了（而它本来是对的，见 :mod:`axo.memory.edit`）。

    Raises:
        ValueError: 正文为空。
        SDRCapacityError: 正文里没有任何可编码字符。

    两种异常的理由字符串与界面拦下它时说的是同一句（都出自
    :func:`~axo.memory.edit.can_encode`）。
    """
    reason = can_encode(text)
    if reason is not None:
        error = ValueError if not text else SDRCapacityError
        raise error(reason)

    existing = user_anchor(store)
    if existing is not None:
        if existing.content == text:
            return existing  # 一个字都没变：不写库，也不丢向量
        revised = store.transform(existing.memory_id, lambda item: edited_content(item, text))
        store.drop_embedding(existing.memory_id)
        return revised

    created = MemoryItem(
        memory_id=new_ulid(),
        kind=MemoryKind.PERSONA_ANCHOR,
        tier=Tier.LONGTERM,
        content=text,
        weight=1.0,
        pinned=True,
        lex_sdr=encode_lexical(text),
        source_ref=USER_REF,
    )
    return store.add(created)


def clear_user_persona(store: MemoryStore) -> bool:
    """删掉用户写的那条。返回是否真删了（没有那条时 ``False``）。

    **只删这一条。** 内置那条不动，所以"删掉人设"的结果是回到"只有事实、没有
    说话方式"，而不是把身份也抹了。
    """
    existing = user_anchor(store)
    if existing is None:
        return False
    store.delete(existing.memory_id)
    return True


def read_only_reason(item: MemoryItem) -> str | None:
    """这条记忆是不是界面上不许改也不许删的。是就返回理由，不是返回 ``None``。

    形状照 :func:`~axo.memory.forget.can_soft_forget`：规则住在核心腿里，
    界面照着它置灰并显示理由（**置灰必须带理由**——一个点不动的按钮不说明任何事）。

    判定的极性是"**只有我们自己写的那条受保护**"，不是"其余都是内置的"。明确写出
    来源指针才受保护，来源为空或将来别的来源的那种人格锚一律落在"可改可删"那一侧：
    拒绝改用户自己的东西，比误放开一条更糟。
    """
    if item.kind is not MemoryKind.PERSONA_ANCHOR or item.source_ref != BUILTIN_REF:
        return None
    return (
        "这是终端内置的那条人设，它保证任何模型读到的身份都是同一份字节，"
        "所以不在这里改也不在这里删。要定制说话方式，改「设置 → 人设」里你自己写的那条。"
    )


def persona_texts(store: MemoryStore) -> list[str]:
    """组装上下文时用的 ``persona`` 参数。

    可能多于一条——内置的事实一条、用户写的风格一条——也可能只有内置那条，或者
    （用户把两条都删了）一条都没有。**空不是异常状态**：装配器照常工作，只是模型
    没有任何身份约束。这比"发现没有锚就自动写一条"好：后者会让"删掉人设"这个操作
    永远无效，而用户删不掉的东西比没有更让人困惑。

    空白正文的条目跳过（:meth:`~memcore.types.MemoryItem.__post_init__` 拦得住空串，
    但拦不住 ``"   "``）。
    """
    return [item.content for item in _anchors(store) if item.content.strip()]
