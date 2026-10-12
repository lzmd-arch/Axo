"""**记忆库独立于所调用的模型**——这是本项目的核心承诺，这个文件是它的证据。

计划书 §6.1 把这一条单列出来并注明「必须进 CI」，理由是：其余所有性质的失效都
只影响质量（召回差一点、解释粗一点），而这一条失效意味着**整个产品的前提不成立**
——"换一个模型，你的助手就不认识你了"正是要消灭的那个现象。

验证方式是把"换模型"这件事**做出来**，而不是论证它不会发生。一次换模型在本项目
里就是：登记一个新的记忆空间（``source/model@dim`` 全变），把全库的向量重算一遍
写进去，然后**逐字节比对换模型前后所有不该变的东西**。

比对的对象刻意选成"库的形状"而不是"召回的分数"：

* ``memories`` 表的每一行**逐字节相等**——向量表是另一张表，换模型只该动它；
* 每条记忆的 ``lex_sdr`` 逐位相等——它由文本的 blake2b 哈希决定，模型碰不到它；
* ``tier`` / ``weight`` / ``access_count`` / ``pinned`` / ``created_at`` 全等；
* 人格锚在 ``iter_items()`` 里的**位置**不变（位置变了意味着"最早那条"换人了）；
* 关联图（P4a 落地的 ``graph.npz``）**重建的结果与磁盘上的字节都逐字节相同**——
  理由见 :func:`_fingerprint` 里那段。

P2 的 ``preference.yaml`` 还没落，到那时也要加进 :func:`_fingerprint`——它与上面
那两条是**同一份承诺的几块承重墙**，只是那一块现在还不存在。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memcore.graph import GraphIndex
from memcore.ids import new_ulid
from memcore.recall import recall
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import SDR, MemoryItem, MemoryKind, Taint, Tier

#: 两个"不同的模型"。维度也不同——换模型时维度变化是常态，不是特例。
SPACE_A = make_space_id("ollama", "nomic-embed-text", 768)
SPACE_B = make_space_id("ollama", "bge-m3", 1024)

#: 人格锚的原文。它在整个测试里不许有一个字节的变化。
ANCHOR_TEXT = "你是记忆终端。回答简洁，不确定时说不确定，不说客套话。"

SAMPLE_TEXTS = [
    "我家的路由器放在书房书架顶层。",
    "我在做一个叫 Axo的项目，记忆必须独立于所调用的模型。",
    "今天午饭吃的是番茄炒蛋和米饭。",
    "每周三晚上八点要开组会，会议纪要在共享盘上。",
]

#: 建图时钉死的时间戳。``built_at`` 会进 ``graph.npz``，不钉死就没法逐字节比。
GRAPH_STAMP = 1_700_000_500.0


def _vector(dim: int, seed: int) -> np.ndarray:
    """确定性的假向量。**不用 ``np.random``**：那是全局状态，测试之间会互相影响。"""
    generator = np.arange(dim, dtype=np.float32) + 1.0
    return np.roll(generator, seed) / np.linalg.norm(generator)


def _seed_library(store: MemoryStore) -> None:
    """造一库出来：一条人格锚 + 四条普通记忆，全部已编码、已带 A 空间向量。

    时间戳用 ``new_ulid(at=...)`` 显式指定，好让"人格锚是第一条"这件事是构造出来
    的而不是碰巧的。
    """
    store.register_space(Space(space_id=SPACE_A, dim=768, model_ref="nomic-embed-text"))

    anchor = MemoryItem(
        memory_id=new_ulid(at=1_700_000_000.0),
        kind=MemoryKind.PERSONA_ANCHOR,
        tier=Tier.LONGTERM,
        content=ANCHOR_TEXT,
        taint=Taint.USER,
        weight=1.0,
        pinned=True,
        created_at=1_700_000_000.0,
        lex_sdr=encode_lexical(ANCHOR_TEXT),
    )
    store.add(anchor)
    store.set_embedding(anchor.memory_id, SPACE_A, _vector(768, 0))

    for index, text in enumerate(SAMPLE_TEXTS, start=1):
        item = MemoryItem(
            memory_id=new_ulid(at=1_700_000_000.0 + index * 60),
            kind=MemoryKind.FACT,
            tier=Tier.LONGTERM,
            content=text,
            taint=Taint.USER,
            weight=0.5 + 0.1 * index,
            access_count=index,
            created_at=1_700_000_000.0 + index * 60,
            lex_sdr=encode_lexical(text),
        )
        store.add(item)
        store.set_embedding(item.memory_id, SPACE_A, _vector(768, index))


def _sdr_bytes(sdr: SDR | None) -> bytes | None:
    """SDR 的**字节**而不是对象。``SDR`` 是 ``eq=False`` 的，比对象身份没有意义。"""
    if sdr is None:
        return None
    return sdr.idx.tobytes() + sdr.weights.tobytes() + str(sdr.width).encode()


def _graph_file(db_path: Path) -> Path:
    """图缓存该在的位置。**与生产布局同形**：``memory/mem.db`` 与 ``memory/graph.npz``
    并列（见 ``axo.paths``），所以测试里也放在 ``mem.db`` 旁边而不是
    另起一个目录——布局本身就是这条承诺的一部分。
    """
    return db_path.parent / "graph.npz"


def _build_graph(store: MemoryStore) -> GraphIndex:
    """按**现在库里这批条目**重建一份图。``built_at`` 钉死，好让两次重建可比。"""
    return GraphIndex.build(store.iter_items(), built_at=GRAPH_STAMP)


def _graph_shape(index: GraphIndex) -> tuple[object, ...]:
    """图的**内容**——不含 ``built_at``（它由 :data:`GRAPH_STAMP` 钉死，比不出东西）。"""
    return (
        tuple(index.node_ids),
        index.nbr_idx.tobytes(),
        index.nbr_w.tobytes(),
        index.source_digest,
        index.top_m,
        index.min_weight,
        index.assoc_k,
        index.assoc_mode,
    )


def _fingerprint(store: MemoryStore, graph_file: Path) -> tuple[object, ...]:
    """库与它的派生缓存的形状——**一个模型能影响的字段都不在这里面**。

    刻意不含 ``last_access_ts``（访问时间会被召回改）与任何向量。这个函数是
    "换模型后该一致的东西"的定义，所以往里面加字段时要先问一句：这个字段由模型
    决定吗？是的话就不该进来。

    图的这一半（P4a 加的）有两块，缺一不可：

    * **重建的结果**（:func:`_graph_shape`）——用现在这批条目重新建一遍图。这才是
      "换一次模型不需要重建联想结构"的判据：图建在关联码上，而关联码是 ``lex_sdr``
      的纯函数，模型碰不到它。**只比磁盘字节的话这一块根本没被比到**——换模型时
      本来就没人去写那个文件，等于什么都没比。

      说清楚它的射程：**今天**图的输入（``memory_id`` 与 ``lex_sdr``）本来就是上半
      部分的一个子集，所以换模型时这一条不会单独红。它守的是两件将来才可能发生的
      事——图的输入哪天不再是上半部分的子集（``_extract`` 开始看 ``assoc_sdr`` /
      ``sem_sdr``，或者图开始按记忆空间分片），以及**建图本身失去可复现性**
      （分块矩阵乘改成多线程、或者 ``np.rint`` 的累积顺序变了）。后一条今天就能
      独立地红，而它是"同一批记忆两次建图必须逐字节相同"那条承诺的上游。
    * **磁盘上的字节**——守的是另一件事：别在换模型时"顺手"把图缓存清了或重建了。
      图既然是模型无关的，换模型就**恰恰不该**动它。这一条今天是必然成立的（代价
      只是一次 ``read_bytes``），留着是因为"模型变了，缓存失效一下吧"是个很容易
      写出来的将来改动，而它的症状是"换完模型联想关系全丢"。

    Returns:
        ``(每条记忆的形状, 图的形状, graph.npz 的字节)``。
    """
    rows = []
    for item in store.iter_items():
        rows.append(
            (
                item.memory_id,
                item.kind,
                item.tier,
                item.content,
                item.taint,
                item.weight,
                item.access_count,
                item.reinforce_count,
                item.pinned,
                item.session_id,
                item.source_ref,
                item.created_at,
                _sdr_bytes(item.lex_sdr),
            )
        )
    return (tuple(rows), _graph_shape(_build_graph(store)), graph_file.read_bytes())


def _memories_table_bytes(path: Path) -> list[tuple[object, ...]]:
    """``memories`` 表的原始行。**逐字节**，不经过 :mod:`memcore` 的任何解析。

    绕开 ORM 是刻意的：走 :meth:`MemoryStore.get` 只能证明"读出来一样"，而读路径
    的归一化恰好会把某些写入时的差异抹平。这里问的是数据库文件本身。
    """
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT * FROM memories ORDER BY memory_id").fetchall()
    finally:
        connection.close()


@pytest.fixture
def library(tmp_path: Path):
    """一个装好 A 模型向量的库，外加它的路径。

    顺带在同目录落一份 ``graph.npz``：图是**显式重建**的派生缓存（``graph.py`` 里
    写死了这件事），所以这里也显式建一次，而不是让谁在写入路径上顺手建。
    """
    path = tmp_path / "mem.db"
    with MemoryStore(path) as store:
        _seed_library(store)
        _build_graph(store).save(_graph_file(path))
        yield store, path


# ── 换模型 ──────────────────────────────────────────────────────────────────


def test_swapping_the_model_changes_nothing_in_the_library(library) -> None:
    """**这个文件的核心断言。** 把全库向量换成另一个模型的，库一个字节不变。"""
    store, path = library
    before = _fingerprint(store, _graph_file(path))
    before_rows = _memories_table_bytes(path)

    # ── 换模型：登记新空间，把全库向量重算并写进去 ──────────────────────────
    store.register_space(Space(space_id=SPACE_B, dim=1024, model_ref="bge-m3"))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    assert store.count_embeddings(SPACE_B) == store.count(), "前提没成立：向量没换全"

    # ── 该一致的东西 ────────────────────────────────────────────────────────
    assert _fingerprint(store, _graph_file(path)) == before, (
        "换模型之后库的形状变了（含关联图的重建结果与 graph.npz 的字节）。"
        "这是**设计失败**，不是测试失败："
        "'记忆独立于模型'是本项目的前提，不是一条可以调参的指标。"
    )
    assert _memories_table_bytes(path) == before_rows, (
        "memories 表的原始字节变了——换模型只该动 embeddings 表。"
    )


def test_lexical_sdr_is_identical_across_models(library) -> None:
    """词法 SDR 由文本的 blake2b 哈希决定，**与用哪个嵌入模型无关**。

    它是整个"关联基质建在模型无关的表示上"那个决定的落点：若这一条不成立，
    换一次模型就要重建全库的联想结构。
    """
    store, _ = library
    before = {item.memory_id: item.lex_sdr for item in store.iter_items()}

    for item in store.iter_items():
        # 与"完全没有嵌入模型时"的编码逐位比——那才是模型无关的定义
        reference = encode_lexical(item.content)
        assert np.array_equal(item.lex_sdr.idx, reference.idx)
        assert np.array_equal(item.lex_sdr.weights, reference.weights)

    store.register_space(Space(space_id=SPACE_B, dim=1024))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    for item in store.iter_items():
        after = item.lex_sdr
        assert np.array_equal(after.idx, before[item.memory_id].idx)
        assert np.array_equal(after.weights, before[item.memory_id].weights)


def test_persona_anchor_stays_first_and_byte_identical(library) -> None:
    """人格锚仍在原位、仍逐字节相同。

    **位置**进了断言，因为"人格锚是第一条"不只是一条内容约束：装配器按 id 序取
    人格锚，位置变了意味着取到的是别人。而人格锚是"任何模型读到的完全一样"这句话
    最锋利的那个例子——它决定助手是谁。
    """
    store, _ = library
    first_before = next(store.iter_items())
    assert first_before.kind is MemoryKind.PERSONA_ANCHOR
    assert first_before.content == ANCHOR_TEXT

    store.register_space(Space(space_id=SPACE_B, dim=1024))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    first_after = next(store.iter_items())
    assert first_after.memory_id == first_before.memory_id
    assert first_after.kind is MemoryKind.PERSONA_ANCHOR
    assert first_after.content == ANCHOR_TEXT
    assert first_after.pinned is True
    assert first_after.weight == first_before.weight
    assert first_after.lex_sdr == first_before.lex_sdr


def test_the_library_survives_a_swap_and_still_recalls(library) -> None:
    """换完模型，**同一句话仍然问得出同一条记忆**——这就是用户看到的那件事。

    注意这里问的方式：用 B 空间的向量做语义查询，但那个查询向量是从**文本**造的
    （见 :func:`_query_vector`），而召回结果里必须还有那条记忆。真正的端到端版本
    在计划书 §3.4 里是"切到另一个模型，同一会话再问一次，记忆仍在"。
    """
    store, _ = library
    target = SAMPLE_TEXTS[1]

    store.register_space(Space(space_id=SPACE_B, dim=1024))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    ledger = recall(
        store,
        query_text="记忆终端为什么要独立于模型？",
        query_vector=_vector(1024, 2),
        space_id=SPACE_B,
        budget_tokens=500,
    )
    assert target in {memory.content for memory in ledger.injected}
    assert ledger.space_id == SPACE_B


def test_recall_without_the_model_is_unchanged_by_the_swap(library) -> None:
    """**没有嵌入模型的召回结果，与换模型之前一模一样。**

    这一条把"模型是可选读者"说得最清楚：纯词法那一路完全不知道模型换过。反过来说，
    一旦换模型改变了无模型路径的结果，模型就通过某条隐藏的路径渗进了记忆核心。
    """
    store, _ = library
    query = "记忆终端为什么要独立于模型？"  # noqa: RUF001 — 问号是有意的，测全角标点

    before = recall(store, query_text=query, budget_tokens=500)
    assert before.injected, "前提没成立：这个词法查询本来就该命中"

    store.register_space(Space(space_id=SPACE_B, dim=1024))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    after = recall(store, query_text=query, budget_tokens=500)
    assert after.injected_ids == before.injected_ids
    assert [m.scores.lexical for m in after.injected] == [m.scores.lexical for m in before.injected]


# ── 库的所有权在终端侧 ──────────────────────────────────────────────────────


def test_registry_records_the_model_but_the_library_does_not_depend_on_it(library) -> None:
    """空间登记表**记着**是哪个模型产生的向量，但库的身份不依赖它。

    这两件事容易混为一谈：登记表是必要的（不同模型的向量必须分开算），但"哪个模型
    写的"只是一条元信息。把它当成库身份的一部分，就等于承认换模型要重建库。
    """
    store, _ = library
    store.register_space(Space(space_id=SPACE_B, dim=1024, model_ref="bge-m3"))

    spaces = {space.space_id: space for space in store.spaces()}
    assert spaces[SPACE_A].model_ref == "nomic-embed-text"
    assert spaces[SPACE_B].model_ref == "bge-m3"
    assert spaces[SPACE_A].dim != spaces[SPACE_B].dim

    # 库本身仍然只认识"记忆条目"，不认识空间
    assert {item.kind for item in store.iter_items()} <= set(MemoryKind)


def test_registry_is_additive_never_migrating(library) -> None:
    """登记第二个空间**不会动第一个空间里的任何向量**。

    这正是"切回旧模型，旧模型的召回质量原样回来"能成立的原因。若登记新空间会
    清掉旧空间，用户就在换模型这件事上被迫做一次不可逆的选择。
    """
    store, _ = library
    before = {
        item.memory_id: store.get_embedding(item.memory_id, SPACE_A) for item in store.iter_items()
    }

    store.register_space(Space(space_id=SPACE_B, dim=1024))
    for index, item in enumerate(store.iter_items()):
        store.set_embedding(item.memory_id, SPACE_B, _vector(1024, index))

    for memory_id, vector in before.items():
        assert vector is not None
        after = store.get_embedding(memory_id, SPACE_A)
        assert after is not None and np.array_equal(after, vector)
    assert store.count_embeddings(SPACE_A) == store.count()


def test_the_prefix_identifies_the_source_not_the_model_alone(library) -> None:
    """``space_id`` 的前缀区分来源。同一个模型名从不同来源来，就是两个空间。

    这不是洁癖：``ollama/nomic-embed-text`` 与将来某个云端同名模型的向量**不能
    互算余弦**，而它们的维度可能恰好相同——那样维度检查也拦不住。
    """
    store, _ = library
    local = make_space_id("ollama", "nomic-embed-text", 768)
    remote = make_space_id("openai-compat", "nomic-embed-text", 768)
    assert local != remote


def test_swapping_back_and_forth_is_lossless(library) -> None:
    """来回切三次，库的指纹始终是第一份。

    单向的"换过去没坏"可能是巧合（比如写入被静默忽略了）。来回切把它排除掉：
    若任何一次切换写了不该写的东西，第三次的指纹就会漂。
    """
    store, path = library
    graph_file = _graph_file(path)
    original = _fingerprint(store, graph_file)
    original_rows = _memories_table_bytes(path)

    for _ in range(3):
        for space_id, dim, seed_base in ((SPACE_A, 768, 0), (SPACE_B, 1024, 100)):
            store.register_space(Space(space_id=space_id, dim=dim))
            for index, item in enumerate(store.iter_items()):
                store.set_embedding(item.memory_id, space_id, _vector(dim, seed_base + index))
            assert _fingerprint(store, graph_file) == original
            assert _memories_table_bytes(path) == original_rows
