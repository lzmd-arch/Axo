"""一轮巩固的编排 —— 谁参与、改成什么、报出什么。

这个文件守三条边界，每一条错了都不会报错、只会让数字变得"看起来正常"：

1. **纯计算与写入之间那道缝。** :func:`~memcore.consolidation.consolidate` 不改库
   ——``--dry-run`` 与真跑之间只差"调不调 ``bump_reinforce``"这一步。这条缝一旦
   被跨过去（比如编排里顺手把计数改了），``--dry-run`` 就成了假的安全阀。
2. **``bump_reinforce`` 只动一个字段。** 它和 :meth:`~memcore.store.MemoryStore.touch_access`
   长得几乎一样，而正是这份相似让"顺手把 ``last_access_ts`` 也写了"成了一个
   很容易发生的改动——后果是一条从没被召回过的记忆不再衰减。
3. **没数据就说没数据。** 一条都没真回放时辨认率是 ``None`` 而不是 ``0.0``；
   全被跳过的一轮不该看起来像"一条都没认出来"。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from memcore.consolidation import (
    BASE_SEED,
    MIN_CLUSTER,
    REINFORCE,
    SKIP,
    RunReport,
    consolidate,
    run_id_for,
    run_seed_for,
)
from memcore.graph import GraphIndex
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

AT = 1_760_000_000.0

#: 十二条互不相干的记忆。``--limit`` 那几条用例要 ``_pairs(6)``——六个两两团，
#: 也就是十二条——所以这里备足十二条，而不是让 ``_items`` 去循环复用前六条
#: （复用出来的近义句会让"它认不认得出自己"变成一个需要解释的问题）。
_TEXTS = [
    "我家的路由器放在书房书架顶层。",
    "今天午饭吃的是番茄炒蛋和米饭。",
    "每周三晚上八点有一节游泳课。",
    "妈妈对花生过敏，做饭不能放花生油。",
    "医保报销材料交到街道服务中心二楼。",
    "阳台晾衣杆的螺丝松了，得借个内扳手。",
    "图书馆借的三本书下周三到期。",
    "小区门口的快递柜晚上十点断电。",
    "楼下装修从早上八点开始用电钻。",
    "笔记本电池健康度掉到七十八。",
    "国庆回老家的高铁票提前十五天开售。",
    "冰箱冷藏室调到四度，蔬菜能放一周。",
]


def _item(memory_id: str, text: str, **changes: object) -> MemoryItem:
    item = MemoryItem(
        memory_id=memory_id,
        kind=MemoryKind.FACT,
        tier=Tier.RECENT,
        content=text,
        lex_sdr=encode_lexical(text),
    )
    return item.copy(**changes) if changes else item


def _graph(rows: dict[int, list[tuple[int, float]]], size: int, *, top_m: int = 8) -> GraphIndex:
    """手工搭一张关联图。与 ``test_consolidation_partition.py`` 里那份同一个形状。"""
    nbr_idx = np.full((size, top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, top_m), dtype=np.float32)
    for row, neighbours in rows.items():
        for slot, (target, weight) in enumerate(neighbours):
            nbr_idx[row, slot] = target
            nbr_w[row, slot] = weight
    return GraphIndex(
        node_ids=tuple(f"m{index:03d}" for index in range(size)),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="d",
        built_at=0.0,
        top_m=top_m,
        min_weight=0.1,
    )


def _mutual(
    pairs: list[tuple[int, int]], weight: float = 1.0
) -> dict[int, list[tuple[int, float]]]:
    rows: dict[int, list[tuple[int, float]]] = {}
    for a, b in pairs:
        rows.setdefault(a, []).append((b, weight))
        rows.setdefault(b, []).append((a, weight))
    return rows


def _items(count: int = len(_TEXTS), **changes: object) -> list[MemoryItem]:
    return [_item(f"m{index:03d}", _TEXTS[index], **changes) for index in range(count)]


def _one_cluster(size: int) -> GraphIndex:
    """一整个全连的团：标签传播一轮就塌成一个组，切分也不会动它（``size <= 200``）。"""
    return _graph(_mutual([(a, b) for a in range(size) for b in range(a + 1, size)]), size=size)


def _pairs(count: int) -> GraphIndex:
    """``count`` 个互不相连的两两团 —— ``count`` 个独立的簇。"""
    return _graph(_mutual([(2 * index, 2 * index + 1) for index in range(count)]), size=2 * count)


def _replayed_ids(report: RunReport) -> tuple[str, ...]:
    return tuple(outcome.memory_id for outcome in report.outcomes)


# ── 纯计算：一轮跑成什么样 ──────────────────────────────────────────────────


def test_a_round_reinforces_the_memories_it_recognises() -> None:
    """一整个簇、条条有码、互不相干 → 全部认出自己，辨认率 1.0。

    这是整个 S3 的基准：如果连"一簇干净的记忆全部认出自己"都跑不出来，后面关于
    分母、跳过、写库的断言全都建立在一次失败之上。
    """
    items = _items(3)

    report = consolidate(_one_cluster(3), items, at=AT)

    assert len(report.clusters) == 1
    assert report.reinforced == ("m000", "m001", "m002")
    assert report.replayed == 3
    assert report.recognition_rate == 1.0
    assert report.skip_reasons == {}


def test_skip_members_stay_out_of_the_denominator() -> None:
    """被跳过的条目既不算命中也不算未命中——**它们根本不进分母**。

    与基线的 ``completion_summary`` 拿探针数当分母同一姿态：被钉住的、冷掉的、
    还没编码的条目**没有探针**。把它们算进分母等于把"没测"记成"没测中"，而辨认率
    会因为"库里多了几个人格锚"而下降——一个和补全质量毫无关系的下降。
    """
    items = [
        _item("m000", _TEXTS[0]),
        _item("m001", _TEXTS[1]),
        _item("m002", _TEXTS[2], pinned=True),
    ]

    report = consolidate(_one_cluster(3), items, at=AT)

    assert report.replayed == 2, "钉住的那条不该进分母"
    assert report.recognition_rate == 1.0
    assert report.skip_reasons == {"pinned": 1}
    assert _replayed_ids(report) == ("m000", "m001", "m002"), "跳过的仍占一格，顺序不乱"


def test_recognition_rate_is_none_when_nothing_was_replayed() -> None:
    """**整簇都被跳过时辨认率是 ``None``，不是 ``0.0``。**

    这一条守的是本仓那条两面的纪律（「没数据就说没数据」）：``0.0`` 的意思是
    "回放了一轮，一条都没认出来"，``None`` 的意思是"这一轮没有任何回放数据"。
    两者在上游长得一模一样的话，某天会有人去调 ``MAX_CLUSTER`` 修一个
    **根本不存在**的补全质量问题——而这一轮其实只是库里全是人格锚。
    """
    items = _items(2, pinned=True)

    report = consolidate(_one_cluster(2), items, at=AT)

    assert report.replayed == 0
    assert report.reinforced == ()
    assert report.recognition_rate is None, "0.0 会把它说成'一条都没认出来'"
    assert report.skip_reasons == {"pinned": 2}


def test_consolidate_is_pure_it_does_not_touch_the_store(tmp_path: Path) -> None:
    """编排**不写库**。``--dry-run`` 与真跑之间只差"调不调 ``bump_reinforce``"。

    这条是 ``--dry-run`` 这个旗标能不能立住的地基。编排里只要有一处顺手写了库，
    那个旗标就变成了一个**承诺它做不到**的安全阀——而用户会拿它去试真实库。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        items = _items(3)
        for item in items:
            store.add(item)

        report = consolidate(_one_cluster(3), store.load_all(), at=AT)

        assert report.reinforced, "前提：这一轮确实认出了东西"
        for stored in store.load_all():
            assert stored.reinforce_count == 0, "编排碰了库"


# ── 写库：只动一个字段 ──────────────────────────────────────────────────────


def test_bump_reinforce_only_touches_reinforce_count(tmp_path: Path) -> None:
    """**只动 ``reinforce_count``。** ``weight`` 与两个时间戳逐字不动。

    它和 :meth:`~memcore.store.MemoryStore.touch_access` 长得几乎一样，而正是这份
    相似让"顺手把 ``last_access_ts`` 也写了"随时可能发生。后果不响：惰性衰减以
    ``last_access_ts`` 为起点，被巩固"刷新"过的那条会看起来刚刚被召回，于是**不再
    衰减**——一条没人问过的记忆因此长生不老，而库里没有任何一处报错。

    **两个时间戳写成写死的哨兵值，不是"跑一遍再比对前后"。** 早先那版比对
    ``before`` / ``after``，验红时**时红时绿**：Windows 上 ``time.time()`` 的分辨率
    约 15.6 ms，``add`` 与 ``bump`` 落在同一个计时器 tick 里时 ``now_ts()`` 两次
    返回逐位相同的数，"顺手写了 ``last_access_ts``"这个 bug 因此**假绿**。哨兵值
    把这个问题从根上拿掉：库里的值只可能等于当初写进去的那个数。
    """
    access_ts, decay_ts = 1_500_000_000.0, 1_600_000_000.0

    with MemoryStore(tmp_path / "mem.db") as store:
        item = _item(
            "m000",
            _TEXTS[0],
            access_count=3,
            weight=0.42,
            last_access_ts=access_ts,
            last_decay_ts=decay_ts,
        )
        store.add(item)
        before = store.load_all()[0]
        assert before.last_access_ts == access_ts, "前提：哨兵值真的落库了"

        store.bump_reinforce(["m000"])

        after = store.load_all()[0]
        assert after.reinforce_count == before.reinforce_count + 1
        assert after.weight == before.weight
        assert after.access_count == before.access_count
        assert after.last_access_ts == access_ts, "巩固不是'被召回'"
        assert after.last_decay_ts == decay_ts


def test_a_cold_memory_is_skipped_by_the_round_and_never_reinforced(tmp_path: Path) -> None:
    """冷掉的条目在**整轮**里被跳过，它的计数因此不动。

    这是 ``skip`` 里唯一会造成**写库错误**的那一种。钉住的不参与衰减，没编码的
    本来就进不了基质——只有 COLD 是"能跑但不该跑"：它有码、能建进 W、能认出自己，
    于是如果不拦，它会被如实记一次功。后果不是当场可见的：``reinforce_count`` 是
    给"将来拿它定层级阈值"用的独立证据字段，而冷掉的条目因此会带着一摞功劳躺在
    那里，等着某一天被某个阈值捞回上层。

    **顺带一个不那么明显的后果**：这条与它上面那条孪生（内容相同）一起跑，COLD 那条
    若进了 W，就多出一行逐位相同的码，"另一个认出自己"这件的并列会被打破。
    所以这条同时也是"跳过的成员不进基质"在编排层的复述。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        for item in [
            _item("m000", _TEXTS[0]),
            _item("m001", _TEXTS[1], tier=Tier.COLD),
        ]:
            store.add(item)

        report = consolidate(_one_cluster(2), store.load_all(), at=AT)
        store.bump_reinforce(list(report.reinforced))

        assert report.skip_reasons == {"cold": 1}
        assert report.replayed == 1
        counts = {item.memory_id: item.reinforce_count for item in store.load_all()}
        assert counts == {"m000": 1, "m001": 0}, "冷掉的条目被记功了"


def test_bump_reinforce_accumulates_across_rounds(tmp_path: Path) -> None:
    """两次调用是 ``+2``——自增写在 SQL 里，不是"读出来加一再写回"。

    读-改-写在批量场景下要把竞态防一遍，而一次巩固可能加固几百条；SQL 层的自增
    天然没有那个窗口。这条把"累加"这个可观测后果钉住：换成幂等的赋值写法就会红。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        store.add(_item("m000", _TEXTS[0]))

        store.bump_reinforce(["m000"])
        store.bump_reinforce(["m000"])

        assert store.load_all()[0].reinforce_count == 2


def test_bump_reinforce_on_an_empty_list_writes_nothing(tmp_path: Path) -> None:
    """空列表直接返回。冷启动的库上一轮可能一条都没认出来，那不该付出一次空事务。"""
    with MemoryStore(tmp_path / "mem.db") as store:
        store.add(_item("m000", _TEXTS[0]))

        store.bump_reinforce([])

        assert store.load_all()[0].reinforce_count == 0


def test_reinforce_stats_counts_the_entries_that_were_ever_reinforced(tmp_path: Path) -> None:
    """``(被加固过的条数, 加固总次数)``——**两个数不是一回事**。

    "三条被加固过、一共五次"与"五条被加固过、一共五次"对读的人是不同的信息：前者说的是
    **反复认出同一批**，后者说的是**面铺得开**。热度面板两个都写，所以它们必须真的是
    两个数——把条数实现成总数（或反过来）一个字的报错都不会有。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        for index in range(3):
            store.add(_item(f"m{index:03d}", _TEXTS[index % len(_TEXTS)]))
        store.bump_reinforce(["m000", "m001", "m002"])
        store.bump_reinforce(["m000", "m001"])

        assert store.reinforce_stats() == (3, 5)


def test_reinforce_stats_on_an_empty_library_is_two_zeros(tmp_path: Path) -> None:
    """空库是 ``(0, 0)``，**不是"库里有几条记忆"**。

    这一条看着像废话，它挡的是一个具体的写法：聚合 SQL 里漏掉
    ``WHERE reinforce_count > 0`` 的话，有记忆而没跑过巩固的库返回的是
    ``(总条数, 0)``——而热度面板第一件事就是拿第一个数判"有没有东西可写"。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        for index in range(4):
            store.add(_item(f"m{index:03d}", _TEXTS[index % len(_TEXTS)]))

        assert store.reinforce_stats() == (0, 0)


def test_reinforce_stats_does_not_move_with_a_recall(tmp_path: Path) -> None:
    """被召回**不算**被加固。两条轴分开记，这就是它们分开的理由。

    ``touch_access`` 涨的是 ``access_count``。把它也算进巩固的统计里，热度面板会在
    用户每说一句话之后换一个数字——而"巩固到底跑过没有"这个问题的答案，与用户聊了
    多少完全无关。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        store.add(_item("m000", _TEXTS[0]))
        store.touch_access(["m000"], at=AT)

        assert store.reinforce_stats() == (0, 0)


def test_the_reinforced_list_is_what_gets_written(tmp_path: Path) -> None:
    """编排给出的 :attr:`RunReport.reinforced` 就是写进库的那批 id，一条不多一条不少。

    这条把两个模块接起来看："改库改什么"这个决定只有一处出口——报告里那个元组。
    被跳过的、认成别人的、什么都没认出来的，**一个都不该出现在里面**。
    """
    with MemoryStore(tmp_path / "mem.db") as store:
        for item in [
            _item("m000", _TEXTS[0]),
            _item("m001", _TEXTS[1]),
            _item("m002", _TEXTS[2], pinned=True),
        ]:
            store.add(item)

        report = consolidate(_one_cluster(3), store.load_all(), at=AT)
        store.bump_reinforce(list(report.reinforced))

        counts = {item.memory_id: item.reinforce_count for item in store.load_all()}
        assert counts == {"m000": 1, "m001": 1, "m002": 0}


# ── 轮次标识 ────────────────────────────────────────────────────────────────


def test_the_run_id_and_seed_are_derived_from_time_and_base_seed() -> None:
    """同一 (``base_seed``, ``at``) 给同一对 ``run_id``/``run_seed``；换任一个都变。

    **``at`` 用完整的浮点值，不取整秒。** 早先那版取 ``int(at)``，理由是"同一秒内
    跑两次本来也该是同一轮"——这句话是错的，而且错得有声有色：两轮拿到同一个
    ``run_id``，第二轮的事件就被**追加进第一轮那个文件**，``--replay`` 读出一份
    两轮混在一起、哪一轮都对不上的记录，然后报一串"第 k 条不一致，这是 bug"。
    一次取整造出一份永久的假 bug 报告。所以下面这条同一秒内不同 ``at`` **必须**
    是不同的轮次，是这条用例的重心。

    ``run_seed`` 与 ``run_id`` 由同一份材料派生（后缀不同），于是读出一轮的
    ``run_id`` 就能取回它的 ``run_seed``——``--replay`` 的全部可行性在这条上。
    """
    run_id = run_id_for(BASE_SEED, AT)
    seed = run_seed_for(BASE_SEED, AT)

    assert run_id != run_id_for(BASE_SEED, AT + 0.4), "同一秒里跑两轮是两轮，不是一轮"
    assert seed != run_seed_for(BASE_SEED, AT + 0.4)
    assert run_id != run_id_for(BASE_SEED, AT + 1.0)
    assert seed != run_seed_for(BASE_SEED, AT + 1.0)
    assert run_id != run_id_for(BASE_SEED + 1, AT), "--seed 换个值就该是另一轮"
    assert seed != run_seed_for(BASE_SEED + 1, AT)
    assert len(run_id) == 16
    assert 0 <= seed < 2**64


def test_the_round_identity_survives_the_json_round_trip() -> None:
    """``at`` 从事件文件读回来必须派生出**同一个** ``run_id``。

    ``--replay`` 用的时刻来自落盘的那一行（``float(recorded[0]["at"])``），而它
    经过了一次 ``json.dumps`` / ``json.loads``。派生式取 ``at`` 的完整精度，所以
    这条往返是它的**前提**——Python 的浮点在 JSON 里按能往返的最短表示写，读回来
    是同一个 ``double``，于是 ``run_id`` 不因落盘而分家。

    不成立的话，``--replay`` 会对每个 ``run_id`` 都报"重跑出来的 run_id 与文件名
    不同"，而那条错误信息会把人指向 ``--seed``——一个根本不是原因的地方。
    """
    recorded_at = json.loads(json.dumps({"at": AT}))["at"]
    assert recorded_at == AT
    assert run_id_for(BASE_SEED, recorded_at) == run_id_for(BASE_SEED, AT)


def test_the_round_identity_does_not_depend_on_numpy_random_state() -> None:
    """与编码、丢位同一条理由：一轮的标识与抽样不许受全局随机状态影响。

    :mod:`memcore.sdr` 记着 ``np.random.default_rng`` 不保证跨版本给出同一串数。
    用它派生 ``run_seed`` 的话，一次 ``pip install -U numpy`` 就能让"重跑一遍
    逐条比对"漂移——而漂移后的比对结果看起来只是"这一轮有几条不一样"。
    """
    np.random.seed(1)
    first = (run_id_for(BASE_SEED, AT), run_seed_for(BASE_SEED, AT))
    np.random.seed(99999)
    second = (run_id_for(BASE_SEED, AT), run_seed_for(BASE_SEED, AT))
    assert first == second


# ── --limit：这一轮看多少 ───────────────────────────────────────────────────


def test_a_limit_below_the_cluster_floor_is_rejected() -> None:
    """``limit < MIN_CLUSTER`` 当场拒掉。

    那种预算装不下**任何**一个簇，于是这一轮会安静地什么都不做、报告里
    ``recognition_rate`` 是 ``None``——一次看起来像"库里没东西"的空转。
    """
    with pytest.raises(ValueError, match="MIN_CLUSTER"):
        consolidate(_pairs(3), _items(6), at=AT, limit=1)


def test_a_limit_caps_how_much_of_the_library_one_round_touches() -> None:
    """六簇各两条、预算 4 → 只看两簇、共四条，且**整簇进整簇出**。

    切半个簇是不行的：W 是按簇建的，少一条成员就换了一份 W，辨认率随之变成一个
    随 ``--limit`` 变化的数——而它与基线的 ``completion_top1`` 就不再可比。所以
    预算撞上"装不下下一个簇"时是跳过它，不是把它切开。
    """
    report = consolidate(_pairs(6), _items(12), at=AT, limit=4)

    assert len(report.clusters) == 2
    assert len(_replayed_ids(report)) == 4
    assert all(len(cluster) == 2 for cluster in report.clusters)


def test_the_limit_rotates_over_the_library_instead_of_always_taking_the_front() -> None:
    """换了 ``base_seed``，被选中的簇就该变——**否则每天看的永远是开头那几簇**。

    这条不钉"哪个种子选中哪一簇"（那会把一个抽样规则冻成规范），它钉的是"轮换
    这件事真的在发生"：八个种子跑下来，至少出现两种不同的选择。一个恒取前 N 簇的
    实现会让这一条红，而它在功能上**完全说得通**，只是让库的尾部永远巩固不到。
    """
    selections = {
        tuple(
            sorted(
                _replayed_ids(consolidate(_pairs(6), _items(12), at=AT, base_seed=seed, limit=4))
            )
        )
        for seed in range(8)
    }

    assert len(selections) > 1, "八个种子都选中同一批簇——这不是抽样"


def test_a_limited_round_is_reproducible() -> None:
    """同一个 (``base_seed``, ``at``, ``limit``) 跑两次，逐条相同。

    ``--replay`` 的重跑会带上同样的参数，所以这条是那个功能的前提；抽样若用了
    真正的随机数，这里会红——而重跑出来的差异看起来只是"几条例外"。
    """
    first = consolidate(_pairs(6), _items(12), at=AT, limit=4)
    second = consolidate(_pairs(6), _items(12), at=AT, limit=4)

    assert _replayed_ids(first) == _replayed_ids(second)


def test_without_a_limit_the_whole_library_is_taken_in_canonical_order() -> None:
    """``limit=None``（默认）看全部，且按 :func:`partition` 的**规范顺序**。

    默认档必须走规范顺序：``--replay`` 说的"逐条比对"就是拿这个次序当基准，
    而抽样次序是随 ``run_seed`` 变的。
    """
    report = consolidate(_pairs(6), _items(12), at=AT)

    assert _replayed_ids(report) == tuple(f"m{index:03d}" for index in range(12))


# ── 图与库不一致 ────────────────────────────────────────────────────────────


def test_graph_nodes_missing_from_the_store_are_counted_not_swallowed() -> None:
    """图里有、库里没有的节点**要报数**——它们是被彻底删除的记忆。

    一个越来越大的这个数说明图该重建了。静默丢掉它的后果是：库删了一半、图还是
    旧的，巩固照跑、报告上一切正常，只是它巩固的东西里有一半早就不在了。

    同一批里"库里新增、图里没有"的那些**不进这个数**——它们是下一轮建图之后才有
    的东西，:func:`~memcore.consolidation.partition` 压根不会看见它们。
    """
    items = _items(3) + [_item("m009", "建图之后才写进来的一条。")]

    report = consolidate(_one_cluster(3), items, at=AT)

    assert report.missing_from_store == 0
    assert report.missing_from_graph == 1, "库里新增、图里没有的要报出来"
    assert _replayed_ids(report) == ("m000", "m001", "m002"), "m009 不在图里，这一轮不参与"


def test_a_round_whose_graph_has_no_clusters_reports_no_data() -> None:
    """图是空的（还没建过）→ 一轮空转，辨认率 ``None``，不抛异常。

    "图还没建"是冷启动的常态。它该表现为一次**没有数据**的轮次，而不是一次崩溃
    ——托盘上的自动巩固不该因为用户还没点过"重建联想"就弹一个错误出来。
    """
    report = consolidate(_graph({}, size=0), _items(3), at=AT)

    assert report.clusters == ()
    assert report.reinforced == ()
    assert report.recognition_rate is None


def test_the_report_is_a_plain_frozen_value() -> None:
    """报告是值对象：同一输入两次跑出相等的对象，且不可变。

    ``--replay`` 逐条比对与 ``--json`` 输出都建立在"报告是一份平凡数据"上。
    """
    first = consolidate(_one_cluster(3), _items(3), at=AT)
    second = consolidate(_one_cluster(3), _items(3), at=AT)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.at = AT + 1  # type: ignore[misc]


def test_the_decision_names_are_the_ones_the_event_type_reserves() -> None:
    """决策名与 :class:`~memcore.types.ConsolidationEvent` 的取值表逐字一致。

    ``promote`` / ``demote`` **不产生**（阈值无据，见计划书的「不做」），
    所以这里只钉会出现的三个加 ``skip``。名字拼错一个字母不会让任何东西失败——
    只会让某一类决策在报告里永远数不到。
    """
    assert REINFORCE == "reinforce"
    assert SKIP == "skip"
    assert MIN_CLUSTER == 2
