"""路由语料的完整性：指纹、自洽、以及"里面没有他本人的东西"。

:mod:`tests.test_goldset_integrity` 的同位体，守的是同一条纪律——**语料是量具，不是
文档**。改一个字，:func:`~biosnn_memterm.bench.routeeval.decide_turn` 的读数就变了，
而这个读数是以**门槛**的形式进 CI 的：改一个词的后果从"数字对不上文档"升级成"CI 悄悄
放宽了"。

三个 sha256 断言的分工与金标准那边一样：``freeze.json`` 是**语料自己的**冻结记录，
``baseline.json`` 是**基线比较的**那一把尺子。两份都要对，因为改语料的人可能只更新了
其中一个。

最后一段是这一份独有的：**类别普查的上下限**。理由见那两条的说明——一个空掉的类别会
让一条结构断言变成空转，而空转的守卫比没有守卫更糟。
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from biosnn_memterm.agent.router import Register
from biosnn_memterm.bench import (
    REGISTERS,
    default_routes_dir,
    file_sha256,
    load_corpus,
    load_utterances,
    personal_identifier_hits,
    routes_digests,
)

ROUTES = default_routes_dir()
UTTERANCES = ROUTES / "utterances.jsonl"

#: 冻结失败时给出的话。**写死在这里而不是让每条断言各说各的**：读到时该做什么是同一件
#: 事，而"该做什么"比"哪里错了"更容易被忽略。
_FREEZE_HINT = (
    "语料是量具，不是文档。改它等于换了一把尺子：请在**同一次提交**里显式更新 "
    "benchmarks/routes/freeze.json，并重跑 `python tools/bench_router.py --no-model "
    "--freeze` 更新基线。"
)

#: 各类别的**条数下限**。低于它就是语料被削薄了，即使指纹与普查都"自洽"。
#:
#: 为什么需要它：``freeze.json`` 的普查断言只保证"记录与实际一致"，一份把 ``slash``
#: 删空的语料完全一致，而且**每一条断言都还是绿的**——因为
#: :func:`~biosnn_memterm.bench.routeeval.route_structural_violations` 里那条循环
#: ``if category != "slash": continue`` 一次都不进，一个违例都产生不出来。
#: **一个空掉的类别让守卫变成空转**，这一条挡的就是那件事。下限取当前实测值的一半上下，
#: 它拦的是"削薄"，不是"调整"。
_CATEGORY_FLOORS = {
    "slash": 5,
    "imperative": 15,
    "question": 10,
    "chitchat": 15,
    "ambiguous": 8,
    "negation": 6,
    "emoji_punct": 5,
    "long_task": 8,
    "hard": 5,
}

#: 总条数下限。120 条是"每个类别都够分几条"的最小规模。
_TOTAL_FLOOR = 100

#: ``hard`` 类最多占多少。
#:
#: 这一类的准确率**恒为 0**（它就是"今天还判错的那些"），所以它不影响门槛的松紧——
#: 但它**掩盖归因**：把一条判错的 ``negation`` 挪进 ``hard``，准确率一个数位都不动，
#: 而"哪一类在退步"这个信息就没了。超过这个比例说明这张表整体不够用，该补表而不是
#: 往这一类里堆。首次实测是 8/120 ≈ 6.7%。
_HARD_MAX_SHARE = 0.15


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads((ROUTES / "freeze.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def utterances():
    return load_utterances(UTTERANCES)


# ── 指纹 ────────────────────────────────────────────────────────────────────


def test_sha256_matches_freeze(freeze: dict) -> None:
    actual = file_sha256(UTTERANCES)
    assert actual == freeze["utterances"]["sha256"], (
        f"utterances.jsonl 的 sha256 与 freeze.json 记录的不一致。\n{_FREEZE_HINT}"
    )


def test_counts_match_freeze(freeze: dict) -> None:
    """条数单列一条断言：加一行而不改指纹时，条数是唯一会先露头的地方。"""
    actual = routes_digests(ROUTES)
    assert actual["utterances"]["count"] == freeze["utterances"]["count"], (
        f"utterances.jsonl 有 {actual['utterances']['count']} 条，freeze.json 记的是 "
        f"{freeze['utterances']['count']} 条。\n{_FREEZE_HINT}"
    )


def test_categories_match_freeze(freeze: dict, utterances) -> None:
    """freeze.json 里的类别普查也要对。

    它是一份**给人看的**说明（"这套语料覆盖了哪九类难例"），而给人看的数字最容易腐烂
    ——没人会因为它错了而失败。让它失败。
    """
    counted = dict(sorted(Counter(u.category for u in utterances).items()))
    assert counted == freeze["categories"], (
        f"freeze.json 的类别计数与实际不符。\n实际：{counted}\n"
        f"记录：{dict(sorted(freeze['categories'].items()))}\n{_FREEZE_HINT}"
    )


# 「基线文件记的语料指纹必须与现在的一致」那一条在
# ``tests/test_router_gate.py::test_baseline_row_was_frozen_on_this_corpus``——与金标准
# 同一条分工，理由见 ``test_goldset_integrity.py`` 的对应注释。


# ── 自洽 ────────────────────────────────────────────────────────────────────


def test_ids_are_unique(utterances) -> None:
    """id 唯一。**这条抓过一次真事**：中途插进两条语料而没重排号，两个块共用了一段
    id，而症状是"报告里某两条的成绩互相顶替"——看起来像哪一类退化了。"""
    ids = [u.utterance_id for u in utterances]
    duplicated = sorted({i for i, n in Counter(ids).items() if n > 1})
    assert not duplicated, f"utterances.jsonl 里有重复的 id：{duplicated}"


def test_texts_are_unique(utterances) -> None:
    """同一句话不许出现两次。

    重复的语料不是"多量了一遍"——**它给那一类偷偷加权**：一条被写了两遍的 ``negation``
    在类别分列里算两条，而在准确率里算两个样本。改一个词之后，"这一类的读数是升是降"
    会因此被一句话左右，而报告上看不出为什么。
    """
    duplicated = sorted({t for t, n in Counter(u.text for u in utterances).items() if n > 1})
    assert not duplicated, f"utterances.jsonl 里有重复的输入：{duplicated}"


def test_every_field_is_filled(utterances) -> None:
    for utterance in utterances:
        assert utterance.text.strip(), f"{utterance.utterance_id} 的正文是空的。"
        assert utterance.category.strip(), f"{utterance.utterance_id} 的 category 是空的。"
        assert utterance.utterance_id.strip(), "有一条语料没有 id。"


def test_hard_entries_explain_themselves(utterances) -> None:
    """``hard`` 类的每一条都要有 ``note``。

    这一类是**待办清单**：读它的人要能知道"为什么这条该是 task、以及大概怎么补"。
    没有 note 的 ``hard`` 条目就只是"一条判错的"，下一轮补表时无从下手——而补表正是
    这一类存在的唯一用途。
    """
    missing = [u.utterance_id for u in utterances if u.category == "hard" and not u.note]
    assert not missing, (
        f"hard 类的这几条没有 note：{missing}。这一类是待办清单，"
        f"note 要写出「为什么该是另一个语域」以及「怎么补」。"
    )


def test_registers_match_the_router_enum() -> None:
    """语料的标注词表与代码里的 :class:`~biosnn_memterm.agent.router.Register` 必须一致。

    **两边分头改名字的话，语料会静默地全部判错**，而报告上看起来只是"准确率掉了"
    ——那正是最难查的一种。这条断言把它变成一次导入期就红的事故。
    """
    from_code = tuple(register.value for register in Register)
    assert from_code == REGISTERS, (
        f"bench/routes.py 的 REGISTERS={REGISTERS} 与 agent/router.py 的 "
        f"Register={from_code} 不一致。两边必须同名，否则语料会整份判错而看不出原因。"
    )


def test_expect_must_be_a_known_register(tmp_path) -> None:
    """写错一个字母（``"Task"``、``"任务"``）要在**读盘时**当场炸。

    症状本来会是"那一条永远判错"，而报告上只会看到"准确率低了 0.008"——看起来像模型
    不行。这类"看起来像算法问题"的数据错误，是本项目里最贵的一类 bug。
    """
    path = tmp_path / "utterances.jsonl"
    path.write_text(
        json.dumps({"id": "u-1", "text": "查一下", "expect": "Task", "category": "imperative"})
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="u-1"):
        load_utterances(path)


def test_missing_corpus_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus-dir"):
        load_utterances(tmp_path / "nope.jsonl")


# ── 类别普查的上下限 ────────────────────────────────────────────────────────


def test_every_category_has_a_floor(utterances) -> None:
    """每一类都够几条——**尤其是 ``slash`` 与 ``ambiguous``**。

    见 :data:`_CATEGORY_FLOORS`：这两类的结构断言是**遍历式的**，一类空掉之后那两条
    循环一次都不进，一个违例都产生不出来，而所有测试仍然是绿的。同一件事在别的类别
    上的症状轻一些（只是读数不稳），但同一条守卫管得住。
    """
    counted = Counter(u.category for u in utterances)
    thin = [
        f"{name}：{counted.get(name, 0)} 条，下限 {floor}"
        for name, floor in _CATEGORY_FLOORS.items()
        if counted.get(name, 0) < floor
    ]
    assert not thin, "这几类被削薄到下限以下了：\n  " + "\n  ".join(thin) + f"\n{_FREEZE_HINT}"
    assert sum(counted.values()) >= _TOTAL_FLOOR, (
        f"语料一共只有 {sum(counted.values())} 条，低于 {_TOTAL_FLOOR} 条的下限。"
    )


def test_no_unknown_category(utterances) -> None:
    """不许出现没登记在 :data:`_CATEGORY_FLOORS` 里的类别。

    新加一类当然可以，但要走一次可见的 diff：下限表是**结构化断言挂靠的地方**，
    一个没进表的类别意味着它没有下限、也没有人为它想过"这一类空掉会怎样"。
    """
    unknown = sorted({u.category for u in utterances} - set(_CATEGORY_FLOORS))
    assert not unknown, (
        f"语料里有没登记的类别：{unknown}。请在 tests/test_router_corpus_integrity.py 的 "
        f"_CATEGORY_FLOORS 里给它一条下限，并想清楚「这一类空掉会不会让某条断言空转」。"
    )


def test_hard_is_not_a_dumping_ground(utterances) -> None:
    """``hard`` 类不能占太大。见 :data:`_HARD_MAX_SHARE`。"""
    total = len(utterances)
    hard = sum(1 for u in utterances if u.category == "hard")
    share = hard / total
    assert share <= _HARD_MAX_SHARE, (
        f"hard 类有 {hard}/{total} = {share:.3f}，超过 {_HARD_MAX_SHARE}。这一类的准确率"
        f"恒为 0，所以它不影响门槛松紧，却**掩盖归因**：把判错的条目挪进来，准确率一个"
        f"数位都不动，而「哪一类在退步」这个信息就没了。该补的是规则表。"
    )


# ── 不含私人信息 ────────────────────────────────────────────────────────────


def test_corpus_has_no_personal_identifiers() -> None:
    """语料里不许出现关于作者本人/本人机器/本项目其它仓库的痕迹。

    见 :data:`biosnn_memterm.bench.PERSONAL_DENYLIST`——**不在这个文件里再写一份**：
    两份语料是同一份风险的两个入口，各存一份的话，某天补一个词只会补在其中一边，
    而另一边看起来仍然"有守卫"。

    这条是**建公开仓之前**唯一一个不可逆问题的机械守卫：推上去就收不回来了，而人眼
    扫 120 行中文是扫不干净的。
    """
    hits = personal_identifier_hits(UTTERANCES.read_text(encoding="utf-8"))
    assert not hits, (
        f"utterances.jsonl 里出现了 {hits}。路由语料是虚构的日常输入，"
        f"不该夹带私人的东西——这个仓库要公开。"
    )


# ── 字节层 ──────────────────────────────────────────────────────────────────


def test_file_is_lf_and_ends_with_newline() -> None:
    """没有 ``\\r`` 且以 ``\\n`` 收尾。

    冻结判据是 ``sha256(原始字节)``，而行尾是 Git 最容易被自动改动的东西
    （``core.autocrlf`` 是 Git for Windows 的系统默认）。``.gitattributes`` 里
    ``*.jsonl text eol=lf`` 那条是上游的守卫，这一条是下游的：**即使 .gitattributes
    被改坏**，坏掉的字节也会在这里被挡住，而不是让两个平台上的指纹不一致。
    """
    raw = UTTERANCES.read_bytes()
    assert b"\r" not in raw, "utterances.jsonl 里有 CR——行尾归一那一步漏了它，指纹会漂。"
    assert raw.endswith(b"\n"), "utterances.jsonl 没有以换行收尾，最后一行可能被吃掉。"


def test_reader_is_explicitly_utf8(tmp_path) -> None:
    """读盘必须显式 UTF-8。

    Windows 上 :func:`open` 的默认编码是 cp936，而坏掉的样子**不是**报错——是某些输入
    突然判成了另一个语域，看起来像规则表的问题。这里用一份真 UTF-8 字节的文件钉住它：
    去掉 ``encoding="utf-8"``，这个测试会红。
    """
    path = tmp_path / "utterances.jsonl"
    path.write_bytes(
        json.dumps(
            {
                "id": "u-1",
                "text": "提醒我明天下午三点开会",
                "expect": "task",
                "category": "imperative",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    (loaded,) = load_utterances(path)
    assert loaded.text == "提醒我明天下午三点开会"


def test_corpus_is_loadable_as_a_whole(freeze: dict) -> None:
    """``load_corpus`` 能一次读完，且条数、普查、目录都对。

    单看它是一条自检，但它挡住的是"语料里某一行只有 ``load_utterances`` 读得动"这类
    半坏状态——而 ``tools/bench_router.py`` 走的是 ``load_corpus``。
    """
    corpus = load_corpus()
    assert len(corpus.utterances) == freeze["utterances"]["count"]
    assert corpus.categories == dict(sorted(freeze["categories"].items()))
    assert corpus.routes_dir == ROUTES
