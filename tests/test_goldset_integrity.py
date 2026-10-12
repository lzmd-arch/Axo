"""金标准语料的完整性：指纹、自洽、以及"里面没有他本人的东西"。

**这个文件守的是"语料是量具，不是文档"那条纪律。** 它原本写在
``tools/bench_space.py:53-55``：「别跟着阶段代号一起改。这些句子是量具，不是文档
——改一个字，``--json`` 里的数就变了」。那句提醒在那里是一句注释；在这里是一条
断言，因为金标准集的数字会以**门槛**的形式进 CI——改一个字的后果从"数字对不上
文档"升级成"CI 悄悄放宽了"。

三个 sha256 断言的分工：``freeze.json`` 是**语料自己的**冻结记录，
``baseline_lexical.json`` 是**基线比较的**那一把尺子。两份都要对，因为改语料的人
可能只更新了其中一个。
"""

from __future__ import annotations

import json

import pytest

from axo.bench import (
    corpus_digests,
    default_gold_dir,
    file_sha256,
    load_memories,
    load_queries,
    personal_identifier_hits,
)
from memcore.types import MemoryKind, Taint, Tier

GOLD = default_gold_dir()

#: 冻结失败时给出的话。**写死在这里而不是让每条断言各说各的**：读到时该做什么
#: 是同一件事，而"该做什么"比"哪里错了"更容易被忽略。
_FREEZE_HINT = (
    "语料是量具，不是文档（tools/bench_space.py:53-55）。改它等于换了一把尺子："
    "请在**同一次提交**里显式更新 benchmarks/gold/freeze.json，"
    "并重跑 `python tools/bench_recall.py --no-semantic --freeze` 更新基线。"
)

#: 语料里**不许出现**的东西。表在 :data:`axo.bench.PERSONAL_DENYLIST`
#: ——**不在这里再写一份**：路由语料是同一份风险的第二个入口，两处各存一份的话，
#: 某天补一个词只会补在其中一边，而另一边看起来仍然"有守卫"。
#:
#: 这条为什么是机械断言而不是一句承诺：建公开仓时"有没有夹带私人信息"是唯一一个
#: **不可逆**的问题——推上去就收不回来了，而人眼扫 200 行中文是扫不干净的。


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads((GOLD / "freeze.json").read_text(encoding="utf-8"))


# ── 指纹 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["memories.jsonl", "queries.jsonl"])
def test_sha256_matches_freeze(name: str, freeze: dict) -> None:
    key = name.removesuffix(".jsonl")
    actual = file_sha256(GOLD / name)
    assert actual == freeze[key]["sha256"], (
        f"{name} 的 sha256 与 freeze.json 记录的不一致。\n{_FREEZE_HINT}"
    )


def test_counts_match_freeze(freeze: dict) -> None:
    """条数单列一条断言：加一行而不改指纹时，条数是唯一会先露头的地方。"""
    actual = corpus_digests(GOLD)
    for key in ("memories", "queries"):
        assert actual[key]["count"] == freeze[key]["count"], (
            f"{key}.jsonl 有 {actual[key]['count']} 条，freeze.json 记的是 "
            f"{freeze[key]['count']} 条。\n{_FREEZE_HINT}"
        )


def test_categories_match_freeze(freeze: dict) -> None:
    """freeze.json 里的类别普查也要对。

    它是一份**给人看的**说明（"这套语料覆盖了哪十类难例"），而给人看的数字最容易
    腐烂——没人会因为它错了而失败。让它失败。
    """
    counted: dict[str, int] = {}
    for query in load_queries(GOLD / "queries.jsonl"):
        counted[query.category] = counted.get(query.category, 0) + 1
    assert counted == freeze["categories"], (
        f"freeze.json 的类别计数与实际不符。\n实际：{dict(sorted(counted.items()))}\n"
        f"记录：{dict(sorted(freeze['categories'].items()))}\n{_FREEZE_HINT}"
    )


# 「基线文件记的语料指纹必须与现在的一致」那一条**不在这里**，在
# ``tests/test_recall_gate.py::test_baseline_row_was_frozen_on_this_corpus``。
# 两处都写的话，一份漂移会在两个文件里各红一次，而修的人要先判断它们是不是同一件
# 事；而且这里那份只能在基线存在时跑（缺了就 skip），恰恰是"缺基线"那种情形下
# 最需要有人红着——那一半由 gate 文件负责，因为它的 fixture 要求基线必须存在。


# ── 自洽 ────────────────────────────────────────────────────────────────────


def test_ids_are_unique() -> None:
    memories = load_memories(GOLD / "memories.jsonl")
    queries = load_queries(GOLD / "queries.jsonl")
    memory_ids = [m.memory_id for m in memories]
    query_ids = [q.query_id for q in queries]
    assert len(set(memory_ids)) == len(memory_ids), "memories.jsonl 里有重复的 id。"
    assert len(set(query_ids)) == len(query_ids), "queries.jsonl 里有重复的 id。"


def test_fields_are_within_their_enums() -> None:
    """枚举与值域都要合法。

    ``Tier("cold")`` 之类会在**读盘时**就抛，所以这一条真正防的是数值：一个
    ``weight=7.0`` 会让那条记忆在权重调制里拿到远超别人的分数，而它看起来完全
    正常——只是那条永远排第一。
    """
    for memory in load_memories(GOLD / "memories.jsonl"):
        MemoryKind(memory.kind)
        Tier(memory.tier)
        Taint(memory.taint)
        assert 0.0 <= memory.weight <= 1.0, (
            f"{memory.memory_id} 的 weight={memory.weight} 不在 [0, 1] 里。"
        )
        assert memory.content.strip(), f"{memory.memory_id} 的正文是空的。"


def test_every_referenced_id_exists() -> None:
    known = {m.memory_id for m in load_memories(GOLD / "memories.jsonl")}
    missing: list[str] = []
    for query in load_queries(GOLD / "queries.jsonl"):
        for field in ("expect", "optional", "never", "exclude"):
            for memory_id in getattr(query, field):
                if memory_id not in known:
                    missing.append(f"{query.query_id}.{field} → {memory_id}")
    assert not missing, "查询引用了不存在的记忆 id：\n  " + "\n  ".join(missing)


def test_expect_and_never_are_disjoint() -> None:
    """一条记忆不能既是期望答案又是干扰项。

    这不是"不好"，是**没有定义**：它进了分子也进了分母，那条查询的误注入率
    在数学上就是错的，而错误的方向取决于实现细节。
    """
    for query in load_queries(GOLD / "queries.jsonl"):
        overlap = set(query.expect) & set(query.never)
        assert not overlap, f"{query.query_id} 的 expect 与 never 重叠：{sorted(overlap)}"


def test_expected_memories_are_actually_recallable() -> None:
    """期望答案不能是 pinned 或 cold 的。

    ``_scan`` 会跳过 pinned、直接淘汰 cold，所以那种查询**永远不可能命中**——
    它在 hit@K 里恒为 0，把聚合拉垮，而原因藏在语料里。金标准自相矛盾时应当
    在语料层报错，而不是让召回替它背锅。
    """
    by_id = {m.memory_id: m for m in load_memories(GOLD / "memories.jsonl")}
    bad: list[str] = []
    for query in load_queries(GOLD / "queries.jsonl"):
        for memory_id in query.expect:
            gold = by_id[memory_id]
            if gold.pinned or gold.tier == "cold":
                bad.append(f"{query.query_id} 期望 {memory_id}（pinned={gold.pinned}）")
    assert not bad, "不可召回的条目被当成了期望答案：\n  " + "\n  ".join(bad)


def test_no_answer_queries_have_no_ground_truth() -> None:
    """``expect`` 为空的查询是合法的，但它不能同时有 ``optional``。

    有 optional 就意味着"命中它算对"——那这条查询其实**有**答案，而它却被算进
    "库里确实没有答案"的那一组（:func:`~axo.bench.metrics.
    no_answer_injections` 的分母），于是那一组里混进了一个本该得分的查询。
    """
    for query in load_queries(GOLD / "queries.jsonl"):
        assert query.expect or not query.optional, (
            f"{query.query_id} 的 expect 是空的却有 optional={query.optional}——"
            f"它到底有没有答案，这是两说。"
        )


# ── 不含私人信息 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["memories.jsonl", "queries.jsonl"])
def test_corpus_has_no_personal_identifiers(name: str) -> None:
    """语料里不许出现关于作者本人/本人机器/本项目其它仓库的痕迹。

    见 :data:`axo.bench.PERSONAL_DENYLIST`。这条是**建公开仓之前**唯一
    一个不可逆问题的机械守卫：推上去就收不回来了。
    """
    text = (GOLD / name).read_text(encoding="utf-8")
    hits = personal_identifier_hits(text)
    assert not hits, (
        f"{name} 里出现了 {hits}。金标准语料是**虚构的**海洋观测站数据，"
        f"不该夹带私人的东西——这个仓库要公开。"
    )


# ── 字节层 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["memories.jsonl", "queries.jsonl"])
def test_files_are_lf_and_end_with_newline(name: str) -> None:
    """没有 ``\\r`` 且以 ``\\n`` 收尾。

    冻结判据是 ``sha256(原始字节)``，而行尾是 Git 最容易被自动改动的东西
    （``core.autocrlf`` 是 Git for Windows 的系统默认）。``.gitattributes`` 里
    ``*.jsonl text eol=lf`` 那条是上游的守卫，这一条是下游的：**即使 .gitattributes
    被改坏**，坏掉的字节也会在这里被挡住，而不是让两个平台上的指纹不一致。
    """
    raw = (GOLD / name).read_bytes()
    assert b"\r" not in raw, f"{name} 里有 CR——行尾归一那一步漏了它，指纹会漂。"
    assert raw.endswith(b"\n"), f"{name} 没有以换行收尾，最后一行可能被吃掉。"


def test_reader_is_explicitly_utf8(tmp_path) -> None:
    """读盘必须显式 UTF-8。

    Windows 上 :func:`open` 的默认编码是 cp936，而坏掉的样子**不是**报错——是
    某些查询突然召不回东西，看起来像算法问题。这里用一份真 UTF-8 字节的文件来
    钉住它：去掉 ``encoding="utf-8"``，这个测试会红。
    """
    path = tmp_path / "memories.jsonl"
    path.write_bytes(
        json.dumps({"id": "m-1", "content": "盐度探头每月标定一次"}, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    (loaded,) = load_memories(path)
    assert loaded.content == "盐度探头每月标定一次"


def test_bad_line_reports_its_number(tmp_path) -> None:
    """坏行当场炸、并且报出行号。

    静默跳过一行的症状是"某个查询突然没有正确答案了"，而那看起来像召回出了问题。
    行号是唯一能把它拉回"语料坏了"的线索。
    """
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"id": "q-1", "query": "x", "category": "exact"}\n这不是 JSON\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="第 2 行"):
        load_queries(path)


def test_missing_corpus_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus-dir"):
        load_memories(tmp_path / "nope.jsonl")
