"""在**一簇自己的**基质上回放 —— 睡眠巩固的第二步。

这个文件的重心是**判据的口径**。「回放辨认率」这个数只有与基线的 ``completion_top1``
（top-1 是不是自己）**逐字同义**时才有意义，所以这里钉的不是"效果好不好"，而是
"什么算命中、什么算串扰、什么不算数"：

* 命中看的是**身份**，不是分数。``top-1 是别人`` 就是 ``sleep_confusion``，哪怕它
  的得分低到接近零——基线的 ``confusion_rate`` 正是这么数的
  （``benchmarks/substrate/CORPUS.md``）。
* ``skip`` 不进分母（辨认率的分母是"真回放了的条数"），而 ``skip`` 的成员**也不
  进基质**——被跳过的条目仍然在 W 里投票的话，一个钉住的人格锚会把与它像的记忆
  判成串扰。
* 回放**不改条目**。写 ``reinforce_count`` 是下一层（``MemoryStore.bump_reinforce``）
  的事；纯函数碰数据是这条流水线上最容易被顺手破坏的边界。

:func:`~memcore.consolidation._classify` 是**直接 import 的私有函数**（先例：
``tests/test_recall.py`` 的 ``_fuse``）。理由：它的 ``sleep_replay_miss`` 那一支从
整条流水线上走不到——只要基质非空 ``complete`` 至少回一条，而空基质意味着整簇都
是 ``skip``。那一支是留给"补全什么都没给出来"这个真实结局的，不测就等于不存在。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memcore.consolidation import (
    DEFAULT_DROP,
    W_MODE,
    ReplayOutcome,
    _classify,
    replay_cluster,
)
from memcore.sdr import ASSOC_K, drop_bits, encode_assoc, encode_lexical
from memcore.types import SDR, MemoryItem, MemoryKind, Tier

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBSTRATE_CORPUS = _REPO_ROOT / "benchmarks" / "substrate" / "memories.jsonl"


def _item(memory_id: str, text: str, **changes: Any) -> MemoryItem:
    """一条可以回放的记忆：有 ``lex_sdr``，不是 pinned，不是 COLD。

    要改的字段走 :meth:`~memcore.types.MemoryItem.copy`，不塞进构造参数——那样
    ``tier=Tier.COLD`` 这类覆盖会与这里的默认值撞成"同一个关键字给了两次"。
    """
    item = MemoryItem(
        memory_id=memory_id,
        kind=MemoryKind.FACT,
        tier=Tier.RECENT,
        content=text,
        lex_sdr=encode_lexical(text),
    )
    return item.copy(**changes) if changes else item


#: 三条互不相干的记忆。刻意用**不同领域**的句子：这里要的是"它们彼此不像"这个
#: 前提，而不是一个可调难度的语料——难度由后面那条读语料的用例负责。
_FAR_APART = {
    "m000": "我家的路由器放在书房书架顶层。",
    "m001": "今天午饭吃的是番茄炒蛋和米饭。",
    "m002": "每周三晚上八点有一节游泳课。",
}

_DUPLICATE = "同一句话，写了两遍。"


# ── 三个分支：认出自己 / 认成别人 / 什么都没认出来 ──────────────────────────


def test_a_member_that_completes_to_itself_is_reinforced() -> None:
    """三条互不相干的记忆，各自都能认出自己。

    这是整份文件的基准：**一簇自己的基质上，一条没被跳过、有码、和别人都不像的
    记忆必须认出自己**。如果连这个都做不到，后面那些关于"串扰怎么算"的断言全在
    空转。

    ``probe_jaccard`` 恰好是 0.5：探针是原码丢了一半位之后的样子，而"丢位"是
    确定性的，所以这个数可以当**丢位真的生效了**的自检——它要是 1.0，说明探针
    就是目标本身，那次回放是一次空转的满分。
    """
    members = [_item(memory_id, text) for memory_id, text in _FAR_APART.items()]

    outcomes = replay_cluster(members)

    assert [outcome.memory_id for outcome in outcomes] == list(_FAR_APART)
    for outcome in outcomes:
        assert outcome.decision == "reinforce", outcome
        assert outcome.jaccard == 1.0
        assert outcome.confused_with is None
        assert outcome.detail["probe_jaccard"] == pytest.approx(0.5)


def test_a_duplicate_is_reported_as_confusion_not_as_a_hit() -> None:
    """两条**内容完全相同**的记忆：前者认出自己，后者被前者顶掉，判串扰。

    这是 ``sleep_confusion`` 唯一一条能手工推出期望值的构造。两份码逐位相同，
    :meth:`~memcore.substrate.Substrate.complete` 的得分完全并列，而它按行号稳定
    打破并列（``argsort(kind="stable")``）——所以第二行的 top-1 **必然**是第一行。
    行号由 ``members`` 的顺序决定，而 ``members`` 的顺序由 :func:`partition`
    决定，于是整件事是确定的。

    **这里的 ``jaccard`` 是 1.0，而判定是"串扰"**——两个字段量的不是同一件事：
    ``jaccard`` 量"两条码有多像"，判定量"top-1 是不是我"。重复内容本来就该让前者
    满分、后者判负。别把 ``jaccard == 1.0`` 读成"认对了"。
    """
    members = [
        _item("m000", _DUPLICATE),
        _item("m001", _DUPLICATE),
    ]

    first, second = replay_cluster(members)

    assert (first.decision, first.confused_with) == ("reinforce", None)
    assert (second.decision, second.confused_with) == ("sleep_confusion", "m000")
    assert second.jaccard == 1.0, "两条码逐位相同，Jaccard 就是 1.0——这是对的"


def test_the_decision_follows_identity_not_the_score() -> None:
    """**判据是身份，不是分数。** 分数再低，top-1 是别人就是串扰。

    这是本片最要紧的一条，因为它挡的是一个**很诱人的"改进"**：有人看到 top-1
    只比第二名高一点点，会觉得"这不像认出，更像噪声"，于是给它加一道分数门槛
    ——低于门槛就报 ``sleep_replay_miss``。改完看起来更讲究，实际是把
    「回放辨认率」与基线的 ``completion_top1``／``confusion_rate`` 拆成了两个口径：
    基线**从来不按分数筛**，它数的是"top-1 是不是自己"这个二值事实。

    这里把 ``ranked`` 的得分压到 ``1e-9``：两个方向各一条，低分不能改变任何一边
    的判定。默认的 ``complete(..., top=1)`` 在小簇上给不出这么低的分数，所以这条
    只能直接问 :func:`~memcore.consolidation._classify`——那正是把它做成纯函数的
    用处。
    """
    codes: dict[str, SDR] = {
        memory_id: encode_assoc(encode_lexical(text), k=ASSOC_K, mode=W_MODE)
        for memory_id, text in _FAR_APART.items()
    }
    item = _item("m000", _FAR_APART["m000"])
    probe = drop_bits(codes["m000"], DEFAULT_DROP)
    negligible = 1e-9

    mine = _classify(item, [("m000", negligible)], codes, probe)
    assert mine.decision == "reinforce", "认出自己与分数高低无关"

    theirs = _classify(item, [("m002", negligible)], codes, probe)
    assert theirs.decision == "sleep_confusion", "分数低不是'没认出'，是'认成了别人'"
    assert theirs.confused_with == "m002"


def test_classify_reports_a_miss_when_completion_returns_nothing() -> None:
    """``complete`` 一条都没回时，是一条被看见的 ``miss``，不是一次下标越界。

    **这一支从整条流水线上走不到**（理由见模块 docstring），所以只能直接问
    ``_classify``。留着它是因为"补全什么都没给出来"是一个真实的结局：它该是一条
    事件，而不是让 ``ranked[0]`` 抛 ``IndexError`` 把整轮巩固打断。

    ``jaccard`` 与 ``confused_with`` 都必须是 ``None``——没有第一名，就没有"和
    谁像"这回事。给它们填 0.0 会让"什么都没认出来"与"认出来但完全不像"在下游
    看起来一模一样。
    """
    codes = {"m000": encode_assoc(encode_lexical(_FAR_APART["m000"]), k=ASSOC_K, mode=W_MODE)}
    item = _item("m000", _FAR_APART["m000"])
    probe = drop_bits(codes["m000"], DEFAULT_DROP)

    outcome = _classify(item, [], codes, probe)

    assert outcome.decision == "sleep_replay_miss"
    assert outcome.jaccard is None
    assert outcome.confused_with is None
    assert outcome.detail["probe_jaccard"] == pytest.approx(0.5)


# ── 顺序与边界 ──────────────────────────────────────────────────────────────


def test_the_outcomes_follow_the_input_order() -> None:
    """返回与输入**同序等长**，跳过的条目也占一格。

    顺序是 ``--replay`` 的承重件：重跑时逐条比对靠的就是"同一簇、同一次序、
    同一串决策"。跳过不占格的话，两边会因为某条记忆中途被钉住而整体错位，
    而错位后的比对结果看起来只是"有几条不一样"。
    """
    members = [
        _item("m000", _FAR_APART["m000"]),
        _item("m001", "被钉住的一条。", pinned=True),
        _item("m002", _FAR_APART["m002"]),
    ]

    outcomes = replay_cluster(members)

    assert [outcome.memory_id for outcome in outcomes] == ["m000", "m001", "m002"]
    assert [outcome.decision for outcome in outcomes] == ["reinforce", "skip", "reinforce"]


def test_an_empty_cluster_returns_nothing() -> None:
    """空簇不是错误。图刚建好、或者一簇都没分出来时，这是常态。"""
    assert replay_cluster([]) == ()


def test_an_all_skipped_cluster_returns_all_skips() -> None:
    """整个簇都被跳过：全是 ``skip``，且不因为基质是空的而抛异常。

    ``Substrate.build`` 收到空列表会返回一个**良定义的空基质**（宽度取默认值），
    于是 ``complete`` 恒回空列表。这条把那一段路径走通——不然"整个簇都是人格锚"
    这种输入会在巩固里炸掉，而它本该只是"这一簇没什么可回放的"。
    """
    members = [
        _item("m000", _FAR_APART["m000"], pinned=True),
        _item("m001", _FAR_APART["m001"], tier=Tier.COLD),
    ]

    outcomes = replay_cluster(members)

    assert [outcome.decision for outcome in outcomes] == ["skip", "skip"]


@pytest.mark.parametrize("drop", [0.0, -0.5])
def test_dropping_nothing_is_rejected(drop: float) -> None:
    """``drop <= 0`` 当场拒掉。

    **探针就是目标本身时，补全恒为命中，辨认率 100%**——一个不来自任何测量的满分。
    量具里留着 ``--drop 0`` 是为了看"完美探针的天花板"（那是诊断），生产路径上
    没有这个需要，所以这里宁可抛。
    """
    members = [_item("m000", _FAR_APART["m000"]), _item("m001", _FAR_APART["m001"])]

    with pytest.raises(ValueError, match="100%"):
        replay_cluster(members, drop=drop)


def test_defaults_come_from_the_module_constants() -> None:
    """默认值就是那两个常量本身，而且它们的取值各有出处。

    ``DEFAULT_DROP = 0.5`` 与 ``tools/bench_substrate.py`` 的 ``--drop`` 默认档
    一致——**这不是巧合，是同口径的要求**：两边的探针残缺程度不一样，两个辨认率
    就没法互相印证。``W_MODE = "weight"`` 的依据是 P4a 量出来的那张表
    （``hash`` 档的 W 在每一档规模上都是死的），见
    :data:`~memcore.consolidation.W_MODE` 与
    :func:`~memcore.sdr.encode_assoc` 里的两档对照。
    """
    assert DEFAULT_DROP == 0.5
    assert W_MODE == "weight"


# ── skip：四种原因里的三种，以及"跳过了就不投票" ────────────────────────────


def test_skipped_members_carry_a_specific_reason() -> None:
    """``skip`` 的 ``detail`` 里写的是**具体哪一个原因**，不是"不满足条件"。

    四种原因指向四种不同的处置，糊成一句"跳过了"就没人能判断该改哪一个：钉住的
    要用户自己解钉，COLD 要等它被重新召回，没码的要等编码回填。

    第四种"不在图里"不在这里判——:func:`~memcore.consolidation.partition` 压根
    不会把这类节点放进簇（见 ``tests/test_consolidation_partition.py``）。
    """
    members = [
        _item("m000", _FAR_APART["m000"], pinned=True),
        _item("m001", _FAR_APART["m001"], tier=Tier.COLD),
        MemoryItem(
            memory_id="m002",
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content="还没编码的一条。",
            lex_sdr=None,
        ),
        _item("m003", _FAR_APART["m002"]),
    ]

    outcomes = replay_cluster(members)

    assert [outcome.detail.get("reason") for outcome in outcomes] == [
        "pinned",
        "cold",
        "no_lex_sdr",
        None,
    ]
    for outcome in outcomes[:3]:
        assert outcome.decision == "skip"
        assert outcome.jaccard is None, "没回放就没有相似度可言"
        assert outcome.confused_with is None


def test_a_skipped_member_does_not_get_a_vote_in_the_substrate() -> None:
    """**被跳过的成员不进 W。** 钉住的那条换成它的孪生兄弟，另一条仍然认出自己。

    构造：两条内容相同的记忆，**第一条是钉住的**。如果跳过只体现在"标一个 skip"
    而没有把它们从基质里拿掉，那么 W 里就有两行逐位相同的码，得分并列，稳定打破
    并列取行号最小的那一行——也就是那条**不该参与回放**的钉住记忆。于是第二条被
    判成串扰，而"这一簇里有一条钉住的人格锚"这种输入**在真实库上很常见**：
    人格锚与某条事实高度相似正是它存在的意义。

    反过来说，这也是 ``skip`` 不进分母那条口径的落地：跳过的条目既不被打分，
    也不参与打分。
    """
    members = [
        _item("m000", _DUPLICATE, pinned=True),
        _item("m001", _DUPLICATE),
    ]

    skipped, replayed = replay_cluster(members)

    assert skipped.decision == "skip"
    assert replayed.decision == "reinforce", (
        "钉住的那条不该在 W 里投票——它投了票，这条就会被判成串扰"
    )


def test_replay_does_not_write_to_the_items() -> None:
    """回放是**只读**的：``reinforce_count`` 原样不动。

    "认出自己"这条决策里带着 ``reinforce_before`` / ``reinforce_after``，那是给
    事件留的证据，**不是**已经发生的写入。真写库的是
    ``MemoryStore.bump_reinforce``，一次批量一条 SQL。纯函数顺手把计数改了，会让
    "这一轮到底写了什么"变成一句无法回答的话——而巩固是本仓唯一一处无人值守的
    批量改库。
    """
    item = _item("m000", _FAR_APART["m000"], reinforce_count=7)

    outcome, _ = replay_cluster([item, _item("m001", _FAR_APART["m001"])])

    assert outcome.detail["reinforce_before"] == 7
    assert outcome.detail["reinforce_after"] == 8
    assert item.reinforce_count == 7, "证据字段与写入是两件事，这里只产生证据"


# ── 码档位：用了 weight，且不吃条目上现成的 assoc_sdr ──────────────────────


def _first_corpus_rows(count: int) -> list[dict[str, str]]:
    """读那份**冻结的**基质语料的前 ``count`` 行。

    刻意复用量具语料而不是自己编一批句子：这条用例要量的就是"两档关联码在真实
    文本上分不分得开"，而那份语料（200 条、同一话题下的近义改写）正是为这件事
    造的，冻结在 ``benchmarks/substrate/freeze.json`` 里。自己编一批的话，测的
    是我编得好不好。
    """
    lines = _SUBSTRATE_CORPUS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[:count]]


def test_the_weight_code_tier_is_used_and_a_prefilled_assoc_sdr_is_ignored() -> None:
    """**两件事一起验**：用的是 ``weight`` 档的码，且条目上现成的 ``assoc_sdr`` 不算数。

    这两件事是同一个陷阱的两面。:meth:`~memcore.substrate.Substrate.build` 会
    **优先吃** ``item.assoc_sdr``，只有它是 ``None`` 时才按 ``config.mode`` 现算；
    而生产路径上存的是图 A 用的 ``hash`` 档。于是：

    * 谁把 :func:`~memcore.consolidation.replay_cluster` 里的码从"自己现算"
      改成"传给 ``build`` 让它现算"，只要条目上带着预先算好的 ``hash`` 档码，
      ``W_MODE = "weight"`` 就被**无声忽略**；
    * 谁把 :data:`~memcore.consolidation.W_MODE` 直接改成 ``"hash"``，同样无声。

    两种改法都产出一份**看起来完全正常的辨认率**——只是它会掉到 0.05 上下。
    所以这条用例把同一批 20 条记忆跑两遍：一遍干净，一遍**故意**在每条上预填一份
    ``hash`` 档的 ``assoc_sdr``。正确实现下两者逐条相同；上面两种改法都会让
    "预填"那一遍塌掉。

    门槛取 0.9 而不是实测值：语料若按 ``freeze.json`` 的规矩整份重冻，实测值会
    小幅浮动，而两档之间的差距是数量级的。**写这条时的实测**：干净那遍 0.95
    （19/20），预填 ``hash`` 那遍 0.10（2/20）；把 :data:`W_MODE` 直接改成
    ``"hash"``，干净那遍自己就掉到 0.05（1/20）。三个数都是跑**这一条用例**读出来的
    （第三个来自验红时的那次运行），没有第二份量法。
    """
    rows = _first_corpus_rows(20)
    assert len(rows) == 20, "语料被截短了，这条用例的前提就没了"

    clean = [_item(row["id"], row["content"]) for row in rows]
    # 同一条记忆，但 assoc_sdr 里预填了一份**图 A 那档**（hash）的码。
    primed = [
        item.copy(
            assoc_sdr=encode_assoc(
                # _item 已经把 lex_sdr 编好了，这里只是再取一次同一个对象。
                item.lex_sdr,  # type: ignore[arg-type]
                k=ASSOC_K,
                mode="hash",
            )
        )
        for item in clean
    ]

    def recognition_rate(members: list[MemoryItem]) -> float:
        outcomes = replay_cluster(members)
        replayed = [o for o in outcomes if o.decision != "skip"]
        assert len(replayed) == len(members), "这批条目不该有被跳过的"
        return sum(1 for o in replayed if o.decision == "reinforce") / len(replayed)

    clean_rate = recognition_rate(clean)
    primed_rate = recognition_rate(primed)

    assert clean_rate >= 0.9, (
        f"辨认率只有 {clean_rate:.2f}。这一簇只有 20 条、还是同一话题下的改写，"
        f"用 weight 档时它应当几乎全中；掉到 0.05 上下说明局部 W 建在了 hash 档上。"
    )
    assert primed_rate == clean_rate, (
        "条目上预填的 assoc_sdr 被读了。生产路径上存的是图 A 的 hash 档，"
        "读它等于让 W_MODE='weight' 无声作废。"
    )


def test_outcomes_are_plain_data() -> None:
    """:class:`~memcore.consolidation.ReplayOutcome` 是值对象：可比较、可冻。

    下一层要把它写成 JSONL，所以它必须是一个**装着数字与字符串**的平凡结构，
    而不是一个还攥着基质或数组的东西。这条顺带挡住"顺手把基质的引用塞进 detail"
    这种改动——那会让事件的行体随对象生命周期变形。
    """
    members = [_item("m000", _FAR_APART["m000"]), _item("m001", _FAR_APART["m001"])]

    first = replay_cluster(members)
    second = replay_cluster(members)

    assert first == second
    assert all(isinstance(outcome, ReplayOutcome) for outcome in first)
    assert all(isinstance(outcome.detail, dict) for outcome in first)
