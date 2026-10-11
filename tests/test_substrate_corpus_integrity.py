"""基质语料的完整性：指纹、簇普查、以及"里面没有他本人的东西"。

:mod:`tests.test_goldset_integrity` 与 :mod:`tests.test_router_corpus_integrity`
的同位体，守的是同一条纪律——**语料是量具，不是文档**。改一个字，:func:`neighbor_auc`
的读数就变了，而这个读数是以**门槛**的形式进 CI 的：改一个词的后果从"数字对不上
文档"升级成"CI 悄悄放宽了"。

三个 sha256 断言的分工与另外两份语料一样：``freeze.json`` 是**语料自己的**冻结记录，
``baseline.json`` 是**基线比较的**那一把尺子（后者的语料指纹断言在
:mod:`tests.test_substrate_gate`）。两份都要对，因为改语料的人可能只更新了其中一个。

**这一份独有的两段**：

* **簇普查的上下限**——见 :data:`_CLUSTER_FLOOR`。一个空掉的簇与一个只剩一条的簇
  都不会让任何断言变红，却会让 ``neighbor_auc`` 的分母悄悄少一截（同簇对数在那里
  是 ``C(n, 2)``，n=1 时是 0）。空转的守卫比没有守卫更糟：它让人以为有人在看。
* **``expect`` 必须同簇且存在**——那不是"语料写得好不好"，它是 ``graph_hit`` 的
  定义域。它由 :func:`~biosnn_memterm.bench.substrate.substrate_structural_violations`
  机械检查，断言在 gate 那份文件里（那里才有建好的图）；这里只查不依赖图的静态
  部分：id 存在、非空、且不重复。
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from biosnn_memterm.bench import (
    default_substrate_dir,
    file_sha256,
    load_substrate_corpus,
    load_substrate_memories,
    load_substrate_queries,
    personal_identifier_hits,
    substrate_digests,
)

SUBSTRATE = default_substrate_dir()
MEMORIES = SUBSTRATE / "memories.jsonl"
QUERIES = SUBSTRATE / "queries.jsonl"

#: 冻结失败时给出的话。**写死在这里而不是让每条断言各说各的**：读到时该做什么是同一件
#: 事，而"该做什么"比"哪里错了"更容易被忽略。
_FREEZE_HINT = (
    "语料是量具，不是文档。改它等于换了一把尺子：请在**同一次提交**里显式更新 "
    "benchmarks/substrate/freeze.json，并重跑 `python tools/bench_substrate.py "
    "--freeze` 更新基线。"
)

#: 每个簇的**条数下限**。实测每簇 20 条，下限取一半。
#:
#: 为什么需要它：``freeze.json`` 的簇普查只保证"记录与实际一致"，一份把某个簇削到
#: 一条的语料完全一致，而且**每一条断言都还是绿的**——因为那一簇在
#: :func:`~biosnn_memterm.bench.substrate.neighbor_pairs` 里贡献的同簇对数是
#: ``C(1, 2) = 0``，它只是从 AUC 的分母里消失，不产生任何违例。**一个空转的簇让
#: 守卫变成空转**，这一条挡的就是那件事。下限拦的是"削薄"，不是"调整"。
_CLUSTER_FLOOR = 10

#: 每个簇的**查询条数下限**。实测每簇 4 条。
#:
#: 一个没有查询的簇在 ``graph_hit`` 上不投票，而 ``graph_hit`` 是 P4b 那个
#: "第三通道该不该上"的裁决依据——少一个簇的投票不会让它变红，只会让那个裁决
#: 建立在一份更薄的证据上。
_QUERY_CLUSTER_FLOOR = 2

#: 总条数下限。10 簇 × 10 条是"每簇都够算出一条 AUC"的最小规模。
_TOTAL_FLOOR = 100


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads((SUBSTRATE / "freeze.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def memories():
    return load_substrate_memories(MEMORIES)


@pytest.fixture(scope="module")
def queries():
    return load_substrate_queries(QUERIES)


# ── 指纹 ────────────────────────────────────────────────────────────────────


def test_sha256_matches_freeze(freeze: dict) -> None:
    """两份 JSONL 的原始字节指纹都要与 ``freeze.json`` 对上。

    逐字节而不是逐条比较：逐条比较会漏掉"改了行尾""改了字段顺序""多了一个空格"
    这一类改动——而它们都会让 :data:`~biosnn_memterm.bench.substrate.GATED` 里那
    两项的读数变掉（内容不同了，编码自然不同）。
    """
    for name, path in (("memories", MEMORIES), ("queries", QUERIES)):
        actual = file_sha256(path)
        assert actual == freeze[name]["sha256"], (
            f"{path.name} 的 sha256 与 freeze.json 记录的不一致。\n{_FREEZE_HINT}"
        )


def test_counts_match_freeze(freeze: dict) -> None:
    """条数单列一条断言：加一行而不改指纹时，条数是唯一会先露头的地方。"""
    actual = substrate_digests(SUBSTRATE)
    for name in ("memories", "queries"):
        assert actual[name]["count"] == freeze[name]["count"], (
            f"{name}.jsonl 有 {actual[name]['count']} 条，freeze.json 记的是 "
            f"{freeze[name]['count']} 条。\n{_FREEZE_HINT}"
        )


def test_memory_clusters_match_freeze(freeze: dict, memories) -> None:
    """簇普查也要对。

    它是一份**给人看的**说明（"这套语料覆盖了哪十个词场"），而给人看的数字最容易
    腐烂——没人会因为它错了而失败。让它失败。
    """
    counted = dict(sorted(Counter(m.cluster for m in memories).items()))
    assert counted == freeze["clusters"], (
        f"freeze.json 的簇计数与实际不符。\n实际：{counted}\n"
        f"记录：{dict(sorted(freeze['clusters'].items()))}\n{_FREEZE_HINT}"
    )


def test_query_clusters_match_freeze(freeze: dict, queries) -> None:
    """查询侧的簇普查。

    :func:`~biosnn_memterm.bench.substrate.substrate_digests` 只数记忆侧，所以
    ``query_clusters`` 没有任何机械消费者——**没有消费者的数字会腐烂**，这一条
    就是它的消费者。
    """
    counted = dict(sorted(Counter(q.cluster for q in queries).items()))
    assert counted == freeze["query_clusters"], (
        f"freeze.json 的查询簇计数与实际不符。\n实际：{counted}\n"
        f"记录：{dict(sorted(freeze['query_clusters'].items()))}\n{_FREEZE_HINT}"
    )


# 「基线文件记的语料指纹必须与现在的一致」那一条在
# ``tests/test_substrate_gate.py::test_baseline_was_frozen_on_this_corpus``——与另外两份
# 语料同一条分工（理由见 test_goldset_integrity.py 的对应注释）。


# ── 自洽 ────────────────────────────────────────────────────────────────────


def test_ids_are_unique(memories, queries) -> None:
    """id 唯一，**两份语料各自内部**都要查。

    重复 id 的症状不是报错：``SubstrateCorpus.by_id`` 是字典，后一条会静默顶替
    前一条——而顶替之后图里少一个节点、探针少一条，看起来只是"数掉了一点"。
    """
    for label, ids in (
        ("memories", [m.memory_id for m in memories]),
        ("queries", [q.query_id for q in queries]),
    ):
        duplicated = sorted({i for i, n in Counter(ids).items() if n > 1})
        assert not duplicated, f"{label}.jsonl 里有重复的 id：{duplicated}"


def test_contents_are_unique(memories) -> None:
    """同一句记忆不许出现两次。

    重复的语料不是"多量了一遍"——**它抬高同簇相似度**：两条一模一样的条目之间的
    Jaccard 是 1.0，于是它们互相是对方的最近邻，``neighbor_precision@m`` 被抬起来
    而 ``completion_top1`` 塌在"分不清自己和自己的复制品"上。量到的是语料的重复度，
    不是基质的性质。
    """
    counted = Counter(memory.content for memory in memories)
    duplicated = sorted(text for text, times in counted.items() if times > 1)
    assert not duplicated, f"memories.jsonl 里有重复的正文：{duplicated}"


def test_every_field_is_filled(memories, queries) -> None:
    for memory in memories:
        assert memory.memory_id.strip(), "有一条记忆没有 id。"
        assert memory.content.strip(), f"{memory.memory_id} 的正文是空的。"
        assert memory.cluster.strip(), f"{memory.memory_id} 的簇是空的。"
    for query in queries:
        assert query.query_id.strip(), "有一条查询没有 id。"
        assert query.query.strip(), f"{query.query_id} 的查询是空的。"
        assert query.cluster.strip(), f"{query.query_id} 的簇是空的。"


def test_expect_is_not_empty_and_points_at_something_real(memories, queries) -> None:
    """``expect`` 非空、不重复、且指向语料里真实存在的 id。

    **这一条与"同簇"分开查**：同簇要建好图才查得动（在 gate 那份文件里），而
    "指着一个不存在的 id"是纯静态的——它在那边会被报成"期望的 X 不在语料里"，
    与图无关，所以在这里更早地挡住它。

    ``expect`` 空了会让那条查询从 ``hit@*`` 与 ``mrr`` 的分母里**整个消失**
    （没有正确答案，就永远排不到），而报告上只会看到"查询数没变、分子低了"。
    """
    known = {m.memory_id for m in memories}
    for query in queries:
        assert query.expect, f"{query.query_id} 没有期望集。"
        assert len(set(query.expect)) == len(query.expect), (
            f"{query.query_id} 的 expect 里有重复项：{query.expect}——重复项会把那条"
            f"记忆在 MRR 里投两票。"
        )
        missing = sorted(set(query.expect) - known)
        assert not missing, f"{query.query_id} 期望了语料里没有的 {missing}。"


def test_expect_is_not_the_whole_cluster(memories, queries) -> None:
    """``expect`` 不许把整个簇都写上。

    写满的话 ``graph_hit`` 的分母还是 40，分子却几乎白送——那是一条**恒真的**指标，
    它不会报错，只会让"第三通道值不值"这个问题得到一个没有信息量的答案。
    期望集只该写"这条查询该够到的那几条"，实测每条 3 条。
    """
    per_cluster = Counter(m.cluster for m in memories)
    for query in queries:
        assert len(query.expect) < per_cluster[query.cluster], (
            f"{query.query_id} 的 expect 覆盖了整个 {query.cluster} 簇"
            f"（{len(query.expect)}/{per_cluster[query.cluster]}）——那样 hit 会变成恒真。"
        )


def test_every_cluster_has_a_floor(memories, queries) -> None:
    """每一簇都够几条——记忆侧与查询侧都要。

    见 :data:`_CLUSTER_FLOOR` 与 :data:`_QUERY_CLUSTER_FLOOR`：被削薄的那一簇不会
    让任何断言变红，它只是从 AUC 的分母里消失。
    """
    mem = Counter(m.cluster for m in memories)
    thin = [
        f"记忆：{name} 有 {count} 条，下限 {_CLUSTER_FLOOR}"
        for name, count in sorted(mem.items())
        if count < _CLUSTER_FLOOR
    ]
    assert not thin, "这几簇被削薄到下限以下了：\n  " + "\n  ".join(thin) + f"\n{_FREEZE_HINT}"

    que = Counter(q.cluster for q in queries)
    thin_queries = [
        f"查询：{name} 有 {que.get(name, 0)} 条，下限 {_QUERY_CLUSTER_FLOOR}"
        for name in sorted(mem)
        if que.get(name, 0) < _QUERY_CLUSTER_FLOOR
    ]
    assert not thin_queries, (
        "这几簇没有（或几乎没有）查询：\n  "
        + "\n  ".join(thin_queries)
        + "\n一簇没有查询，它在 graph_hit 上就不投票——而 graph_hit 是 P4b 那个"
        "「第三通道该不该上」的裁决依据。\n" + _FREEZE_HINT
    )

    assert sum(mem.values()) >= _TOTAL_FLOOR, (
        f"语料一共只有 {sum(mem.values())} 条，低于 {_TOTAL_FLOOR} 条的下限。"
    )


def test_no_cluster_exists_only_on_the_query_side(queries, memories) -> None:
    """查询的簇名必须都能在记忆侧找到。

    写错一个簇名（``"netwrok"``）的症状是：那条查询的 ``expect`` 全是异簇的，
    于是它在 gate 那份文件里被报成"期望必须是同簇的"——报错没错，但原因看起来
    像"期望写错了"，而实际是**簇名打错了**。在这里先挡住，报错就指向真正的原因。
    """
    known = {m.cluster for m in memories}
    unknown = sorted({q.cluster for q in queries} - known)
    assert not unknown, (
        f"queries.jsonl 里有记忆侧不存在的簇名：{unknown}。这通常是拼写错误——"
        f"它会让那些查询的 expect 全部变成异簇，而症状看起来像期望写错了。"
    )


# ── 不含私人信息 ────────────────────────────────────────────────────────────


def test_corpus_has_no_personal_identifiers() -> None:
    """两份 JSONL 里都不许出现关于作者本人/本人机器/本项目其它仓库的痕迹。

    见 :data:`biosnn_memterm.bench.PERSONAL_DENYLIST`——**不在这个文件里再写一份**：
    三份语料是同一份风险的三个入口，各存一份的话，某天补一个词只会补在其中一边，
    而另一边看起来仍然"有守卫"。

    这条是**建公开仓之前**唯一一个不可逆问题的机械守卫：推上去就收不回来了，而人眼
    扫 200 行中文是扫不干净的。
    """
    for path in (MEMORIES, QUERIES):
        hits = personal_identifier_hits(path.read_text(encoding="utf-8"))
        assert not hits, (
            f"{path.name} 里出现了 {hits}。基质语料是虚构的日常记忆，"
            f"不该夹带私人的东西——这个仓库要公开。"
        )


# ── 字节层 ──────────────────────────────────────────────────────────────────


def test_files_are_lf_and_end_with_newline() -> None:
    """没有 ``\\r`` 且以 ``\\n`` 收尾。

    冻结判据是 ``sha256(原始字节)``，而行尾是 Git 最容易被自动改动的东西
    （``core.autocrlf`` 是 Git for Windows 的系统默认）。``.gitattributes`` 里
    ``*.jsonl text eol=lf`` 那条是上游的守卫，这一条是下游的：**即使 .gitattributes
    被改坏**，坏掉的字节也会在这里被挡住，而不是让两个平台上的指纹不一致。
    """
    for path in (MEMORIES, QUERIES):
        raw = path.read_bytes()
        assert b"\r" not in raw, f"{path.name} 里有 CR——行尾归一那一步漏了它，指纹会漂。"
        assert raw.endswith(b"\n"), f"{path.name} 没有以换行收尾，最后一行可能被吃掉。"


def test_reader_is_explicitly_utf8(tmp_path) -> None:
    """读盘必须显式 UTF-8。

    Windows 上 :func:`open` 的默认编码是 cp936，而坏掉的样子**不是**报错——是
    某些记忆突然编成了另一个码、图上的邻居换了一批，看起来像"参数变差了"。
    这里用一份真 UTF-8 字节的文件钉住它：去掉 ``encoding="utf-8"``，这个测试会红。
    """
    path = tmp_path / "memories.jsonl"
    path.write_bytes(
        json.dumps(
            {"id": "s-0001", "cluster": "network", "content": "路由器的信号只有两格。"},
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    (loaded,) = load_substrate_memories(path)
    assert loaded.content == "路由器的信号只有两格。"


def test_missing_corpus_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus-dir"):
        load_substrate_memories(tmp_path / "nope.jsonl")


def test_a_missing_cluster_raises_with_the_reason(tmp_path) -> None:
    """少了 ``cluster`` 字段要在**读盘时**当场炸，并把理由说出来。

    症状本来会是"某些条目编不出码"或者"它不属于任何簇"——那看起来像量具的问题。
    真正的原因是这一行坏了，而这份语料**没有簇标签就没有判据**（同簇/异簇是
    ``neighbor_auc`` 的全部分母）。
    """
    path = tmp_path / "memories.jsonl"
    path.write_text(
        json.dumps({"id": "s-0001", "content": "路由器"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="cluster"):
        load_substrate_memories(path)


def test_corpus_is_loadable_as_a_whole(freeze: dict) -> None:
    """``load_substrate_corpus`` 能一次读完，且条数、簇、目录都对。

    单看它是一条自检，但它挡住的是"语料里某一行只有 ``load_substrate_memories``
    读得动"这类半坏状态——而 ``tools/bench_substrate.py`` 走的是
    ``load_substrate_corpus``。
    """
    corpus = load_substrate_corpus()
    assert len(corpus.memories) == freeze["memories"]["count"]
    assert len(corpus.queries) == freeze["queries"]["count"]
    assert corpus.clusters == dict(sorted(freeze["clusters"].items()))
    assert corpus.corpus_dir == SUBSTRATE
