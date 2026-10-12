"""关联图 A。

这个文件盯的是三类**静默错误**——它们全都产出一个看起来正常的图：

* :func:`test_stale_detects_a_content_change_with_the_same_id` —— 只比 id 集合的
  指纹会漏掉最常见的那种陈旧。
* :func:`test_neighbours_pointing_outside_the_table_raise` —— 越界的行号会让
  ``neighbors`` 返回**另一条记忆的 id**。
* :func:`test_build_is_independent_of_input_order` —— 行序若跟着输入顺序走，
  同一批记忆会建出两个不同的图，而指纹相同、``is_stale`` 说它不过期。
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from memcore.errors import SDRCapacityError, StoreCorruptError
from memcore.graph import FORMAT_VERSION, GraphConfig, GraphIndex
from memcore.sdr import encode_lexical
from memcore.types import SDR

WIDTH = 64


def _item(memory_id: str, sdr: SDR) -> SimpleNamespace:
    return SimpleNamespace(memory_id=memory_id, lex_sdr=None, assoc_sdr=sdr)


def _pattern(bits: list[int], *, width: int = WIDTH) -> SDR:
    return SDR(idx=np.array(bits), weights=np.ones(len(bits)), width=width)


def _lex_item(memory_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(memory_id=memory_id, lex_sdr=encode_lexical(text), assoc_sdr=None)


#: 四个节点，相似度是手工算得出来的：
#:     a={0,1,2,3}  b={0,1,2,4}  c={0,1,5,6}  d={7,8,9,10}
#:     J(a,b) = 3/5 = 0.6      J(a,c) = 2/6 ≈ 0.333
#:     J(b,c) = 2/6 ≈ 0.333    J(d,·) = 0
_FIXTURE = [
    _item("01", _pattern([0, 1, 2, 3])),
    _item("02", _pattern([0, 1, 2, 4])),
    _item("03", _pattern([0, 1, 5, 6])),
    _item("04", _pattern([7, 8, 9, 10])),
]


# ── 边的算术 ────────────────────────────────────────────────────────────────


def test_jaccard_weights_are_exact() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.0))
    assert dict(graph.neighbors("01")) == pytest.approx({"02": 0.6, "03": 2 / 6})
    assert dict(graph.neighbors("02")) == pytest.approx({"01": 0.6, "03": 2 / 6})


def test_neighbours_are_ordered_by_weight_descending() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.0))
    weights = [weight for _, weight in graph.neighbors("01")]
    assert weights == sorted(weights, reverse=True)


def test_a_node_is_not_its_own_neighbour() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=4, min_weight=0.0))
    assert "01" not in dict(graph.neighbors("01"))


def test_min_weight_drops_weak_edges() -> None:
    """0.5 的下限会把 J≈0.333 的两条边挡在外面，只留下 J=0.6 的那条。"""
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.5))
    assert dict(graph.neighbors("01")) == pytest.approx({"02": 0.6})


def test_top_m_caps_the_degree() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=1, min_weight=0.0))
    assert len(graph.neighbors("01")) == 1
    assert graph.neighbors("01")[0][0] == "02"  # 最像的那个留下


def test_min_weight_zero_still_refuses_zero_similarity() -> None:
    """下限 0 不等于"什么都连"：Jaccard 恰好为 0 的对一点都不像。"""
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=4, min_weight=0.0))
    assert graph.neighbors("04") == []


def test_degree_is_capped_by_the_population() -> None:
    """``top_m`` 大于节点数时不该崩，也不该造出 -1 之外的假行号。"""
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=99, min_weight=0.0))
    assert graph.nbr_idx.shape == (4, 99)
    # 99 列里绝大多数是空位（-1），它们**不能**被当成 0 号节点放出来——那样
    # "04" 会凭空多出几十个叫 "01" 的邻居。
    assert len(graph.neighbors("01")) == 2
    assert graph.neighbors("04") == []


# ── 扩散 ────────────────────────────────────────────────────────────────────


def _graph_from(
    rows: dict[int, list[tuple[int, float]]], size: int, *, top_m: int = 8
) -> GraphIndex:
    """按行手工搭一张图，空槽照 :meth:`GraphIndex.build` 的形状填 -1 / 0。

    ``rows[3] = [(0, 0.5)]`` 读作"3 号的邻居表第一格是 0，边权 0.5"，也就是**有向边
    3→0**。手工搭而不是 :meth:`build` 出来，是因为这里的重点是**边表本身**（谁有边、
    有没有反向边、hub 的入度），而从 SDR 反推边表会把这几条挡在哈希后面。
    """
    nbr_idx = np.full((size, top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, top_m), dtype=np.float32)
    for row, neighbours in rows.items():
        for slot, (target, weight) in enumerate(neighbours):
            nbr_idx[row, slot] = target
            nbr_w[row, slot] = weight
    return GraphIndex(
        node_ids=tuple(f"m{index}" for index in range(size)),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="d",
        built_at=0.0,
        top_m=top_m,
        min_weight=0.1,
    )


def test_diffuse_takes_the_max_not_the_sum() -> None:
    """入度高的节点**不会**因为"被更多人指向"而把分堆起来。

    这条是 hub 爆炸的护栏，也是这里最容易走偏的一处：``np.add.at`` 写起来更顺手，
    而且在小图上两种写法**看不出区别**——只有在"很多邻居指向同一个节点"时才分叉。
    图上真出现一个 hub 时，sum 会让它凭连接数压过真正的答案。
    """
    # 1、2、3 各有一条边指向 0，边权都是 0.5
    graph = _graph_from({1: [(0, 0.5)], 2: [(0, 0.5)], 3: [(0, 0.5)]}, size=4)
    base = np.array([0.0, 0.8, 0.6, 0.4])

    spread = graph.diffuse(base, decay=1.0)

    # 三条边分别送来 0.40 / 0.30 / 0.20。max 是 0.40，sum 会是 0.90。
    assert spread[0] == pytest.approx(0.40)
    assert spread[0] != pytest.approx(0.90)


def test_diffuse_returns_only_what_the_neighbours_sent() -> None:
    """返回值是"邻居传过来的"，**不含 ``base`` 自己**。

    分开是刻意的：``base`` 就是词法分，而词法通道已经带着它进过一次融合了。混在
    一起等于把同一份证据记两遍——那种错误只会让某个通道看起来更可靠，不会报错。
    """
    graph = _graph_from({0: [(1, 1.0)]}, size=2)

    spread = graph.diffuse(np.array([0.7, 0.0]), decay=1.0)

    assert spread[0] == 0.0  # 没人指回 0，它的种子分不会原样漏进结果
    assert spread[1] == pytest.approx(0.7)


def test_diffuse_walks_exactly_one_hop() -> None:
    """只走一步，不做传递闭包。0 → 1 → 2 不会让 2 拿到 0 的分。"""
    graph = _graph_from({0: [(1, 1.0)], 1: [(2, 1.0)]}, size=3)

    spread = graph.diffuse(np.array([1.0, 0.0, 0.0]), decay=1.0)

    assert spread[1] == pytest.approx(1.0)
    assert spread[2] == 0.0  # 1 的种子是空的，所以这一跳送不出东西


def test_diffuse_walks_both_stored_directions() -> None:
    """两个方向都通——因为"这两个像"在两张邻居表里**各记了一次**。

    遍历全部行因此等价于把 i→j 与 j→i 并起来，不需要额外对称化。
    """
    graph = _graph_from({0: [(1, 0.5)], 1: [(0, 0.25)]}, size=2)

    spread = graph.diffuse(np.array([1.0, 2.0]), decay=1.0)

    assert spread[1] == pytest.approx(0.5)  # 走 0→1
    assert spread[0] == pytest.approx(0.5)  # 走 1→0，边权是另一侧记的 0.25


def test_diffuse_does_not_invent_the_missing_reverse_edge() -> None:
    """但"遍历两个方向"说的是**两边各自记下的边**，不是把每条边补齐成无向边。

    1 号没把 0 记成邻居，就没有 1→0 这条边——哪怕 0 那边记了 1。
    """
    graph = _graph_from({0: [(1, 0.5)]}, size=2)

    spread = graph.diffuse(np.array([0.0, 1.0]), decay=1.0)

    assert spread[0] == 0.0
    assert spread[1] == 0.0


def test_diffuse_scales_linearly_in_decay() -> None:
    """``decay`` 只是一个乘数，不是阈值。"""
    graph = _graph_from({0: [(1, 0.8)]}, size=2)
    base = np.array([1.0, 0.0])

    assert graph.diffuse(base, decay=1.0)[1] == pytest.approx(0.8)
    assert graph.diffuse(base, decay=0.5)[1] == pytest.approx(0.4)
    assert graph.diffuse(base, decay=0.0)[1] == 0.0


def test_diffuse_is_bit_for_bit_the_formula_it_replaced() -> None:
    """与搬走之前 ``bench/substrate.py::graph_outcomes`` 里那段内联写法**逐位相同**。

    "矩阵门槛搬迁前后逐位不变"这句话的根据就是这条。结合顺序或 max/sum 变了的话，
    指标会落在容差里、``--check`` 仍然绿——**绿着漂移**正是要拦的那一种。
    """
    rng = np.random.default_rng(20261011)
    size, top_m = 64, 8
    nbr_idx = rng.integers(0, size, size=(size, top_m))
    nbr_w = rng.random((size, top_m))
    empty = rng.random((size, top_m)) < 0.2
    nbr_idx[empty] = -1
    nbr_w[empty] = 0.0
    graph = replace(_graph_from({}, size=size, top_m=top_m), nbr_idx=nbr_idx, nbr_w=nbr_w)
    base = rng.random(size)
    decay = 0.5

    # 原写法，逐字抄自 graph_outcomes
    rows, slots = np.nonzero(graph.nbr_idx >= 0)
    targets = graph.nbr_idx[rows, slots].astype(np.int64)
    weights = graph.nbr_w[rows, slots].astype(np.float64)
    inline = np.zeros(size, dtype=np.float64)
    np.maximum.at(inline, targets, base[rows] * weights * decay)

    assert np.array_equal(graph.diffuse(base, decay=decay), inline)


def test_edges_lists_exactly_the_non_empty_slots() -> None:
    """``edges()`` 与 :meth:`edge_count` 数的是同一批槽，且三件套互相对得上。"""
    graph = _graph_from({0: [(1, 0.5), (2, 0.25)], 2: [(0, 0.75)]}, size=3)

    rows, targets, weights = graph.edges()

    assert len(rows) == graph.edge_count() == 3
    assert list(rows) == [0, 0, 2]
    assert list(targets) == [1, 2, 0]
    assert list(weights) == pytest.approx([0.5, 0.25, 0.75])


def test_diffuse_on_a_graph_without_edges_is_all_zeros() -> None:
    """空图不是特例：没有边就没有可传的分。"""
    graph = _graph_from({}, size=3)

    spread = graph.diffuse(np.array([1.0, 2.0, 3.0]), decay=0.5)

    assert spread.shape == (3,)
    assert np.array_equal(spread, np.zeros(3))


def test_diffuse_rejects_a_seed_of_the_wrong_shape() -> None:
    """行序配错了**不会抛异常**，只会算出另一个记忆的邻居——所以这里必须自己拦。"""
    graph = _graph_from({0: [(1, 1.0)]}, size=2)

    with pytest.raises(ValueError, match="行序"):
        graph.diffuse(np.array([1.0, 2.0, 3.0]), decay=1.0)


def test_diffuse_rejects_a_negative_decay() -> None:
    """负衰减会让"传得越远越小"反过来——那不是参数，是写错了。"""
    graph = _graph_from({0: [(1, 1.0)]}, size=2)

    with pytest.raises(ValueError, match="负数"):
        graph.diffuse(np.array([1.0, 0.0]), decay=-0.1)


# ── 行序与指纹 ──────────────────────────────────────────────────────────────


def test_build_is_independent_of_input_order() -> None:
    """行序强制按 memory_id 排序，所以输入顺序不影响任何东西。

    这条不成立时，同一批记忆在不同调用点会建出两个不同的图，而 ``source_digest``
    相同——``is_stale`` 会说"不过期"，``neighbors`` 却给出另一套答案。
    """
    forward = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=2, min_weight=0.0))
    backward = GraphIndex.build(
        list(reversed(_FIXTURE)), config=GraphConfig(top_m=2, min_weight=0.0)
    )
    assert forward == backward
    assert forward.node_ids == ("01", "02", "03", "04")


def test_stale_detects_a_content_change_with_the_same_id() -> None:
    """**改了正文但 id 不变**——最常见的陈旧来源，指纹必须能看出来。

    指纹只比 id 集合的话，这里会返回 False，读者就会拿着一个与库对不上的图继续用。
    """
    before = [_lex_item("01", "路由器放在书房书架顶层。")]
    graph = GraphIndex.build(before)
    after = [_lex_item("01", "今天午饭吃的是番茄炒蛋和米饭。")]
    assert graph.is_stale(after)


def test_stale_is_false_for_the_same_population() -> None:
    items = [_lex_item("01", "路由器放在书房书架顶层。"), _lex_item("02", "今天午饭吃的是米饭。")]
    graph = GraphIndex.build(items)
    assert not graph.is_stale(items)
    # 顺序不敏感：图的行序是按 id 排的，指纹也一样。
    assert not graph.is_stale(list(reversed(items)))


def test_stale_notices_an_added_node() -> None:
    items = [_lex_item("01", "路由器放在书房书架顶层。")]
    graph = GraphIndex.build(items)
    assert graph.is_stale([*items, _lex_item("02", "今天午饭吃的是米饭。")])


# ── 悬空与损坏 ──────────────────────────────────────────────────────────────


def test_deleted_nodes_are_filtered_when_live_is_given() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.0))
    live = {"01", "03"}  # 02 被删了
    assert dict(graph.neighbors("01", live=live)) == pytest.approx({"03": 2 / 6})


def test_neighbours_pointing_outside_the_table_raise() -> None:
    """越界的行号必须当场报错。

    它会让 ``neighbors`` 返回**另一条记忆的 id**——一个完全正常-looking 的错误
    答案，比崩掉难查得多。
    """
    with pytest.raises(StoreCorruptError, match="越界"):
        GraphIndex(
            node_ids=("01", "02"),
            nbr_idx=np.array([[5, -1], [-1, -1]], dtype=np.int32),
            nbr_w=np.zeros((2, 2), dtype=np.float32),
            source_digest="x" * 64,
            built_at=0.0,
            top_m=2,
            min_weight=0.1,
        )


def test_mismatched_neighbour_shapes_raise() -> None:
    with pytest.raises(StoreCorruptError):
        GraphIndex(
            node_ids=("01", "02"),
            nbr_idx=np.zeros((2, 3), dtype=np.int32),
            nbr_w=np.zeros((2, 2), dtype=np.float32),
            source_digest="x" * 64,
            built_at=0.0,
            top_m=3,
            min_weight=0.1,
        )


def test_unknown_node_returns_no_neighbours() -> None:
    """图是缓存：一条刚写进去、还没重建的记忆在图里查不到是常态，不该抛。"""
    graph = GraphIndex.build(_FIXTURE)
    assert graph.neighbors("99") == []
    assert graph.index_of("99") == -1


@pytest.mark.parametrize("top_m,min_weight", [(0, 0.1), (-1, 0.1), (2, -0.1), (2, 1.5)])
def test_invalid_config_raises(top_m: int, min_weight: float) -> None:
    with pytest.raises(SDRCapacityError):
        GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=top_m, min_weight=min_weight))


def test_mismatched_widths_raise() -> None:
    items = [
        _item("01", _pattern([0, 1], width=64)),
        _item("02", _pattern([0, 1], width=128)),
    ]
    with pytest.raises(SDRCapacityError):
        GraphIndex.build(items)


# ── 往返 ────────────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path) -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.2), built_at=1234.5)
    path = tmp_path / "graph.npz"
    graph.save(path)
    loaded = GraphIndex.load(path)
    assert loaded == graph
    assert loaded.built_at == 1234.5
    assert loaded.top_m == 3
    assert loaded.min_weight == pytest.approx(0.2)
    assert loaded.neighbors("01") == graph.neighbors("01")


def test_load_rejects_a_missing_key(tmp_path) -> None:
    """少一个键就必须报错，而不是拿一个字段全空的图继续跑。"""
    path = tmp_path / "graph.npz"
    np.savez(path, format_version=np.array(FORMAT_VERSION, dtype=np.int64))
    with pytest.raises(StoreCorruptError, match="缺少这些键"):
        GraphIndex.load(path)


def test_load_rejects_a_version_mismatch(tmp_path) -> None:
    """版本对不上是**丢掉重建**，不是迁移——图是派生缓存。"""
    graph = GraphIndex.build(_FIXTURE)
    path = tmp_path / "graph.npz"
    graph.save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["format_version"] = np.array(FORMAT_VERSION + 1, dtype=np.int64)
    np.savez(path, **payload)
    with pytest.raises(StoreCorruptError, match="格式版本"):
        GraphIndex.load(path)


def test_node_ids_survive_the_round_trip_exactly(tmp_path) -> None:
    """``'<U26'`` 是定长字段：id 超过 26 字符会被**静默截断**，所以要有守卫。"""
    graph = GraphIndex.build(
        [_lex_item("01J" + "A" * 23, "路由器放在书房书架顶层。")],  # 恰好 26 字符
    )
    path = tmp_path / "graph.npz"
    graph.save(path)
    assert GraphIndex.load(path).node_ids == graph.node_ids


# ── 空图与规模 ──────────────────────────────────────────────────────────────


def test_empty_graph_is_well_defined(tmp_path) -> None:
    graph = GraphIndex.build([])
    assert len(graph) == 0
    assert graph.neighbors("01") == []
    assert graph.edge_count() == 0
    path = tmp_path / "graph.npz"
    graph.save(path)
    assert len(GraphIndex.load(path)) == 0


def test_items_without_a_lexical_sdr_are_skipped() -> None:
    items = [
        _lex_item("01", "路由器放在书房书架顶层。"),
        SimpleNamespace(memory_id="02", lex_sdr=None, assoc_sdr=None),
    ]
    assert GraphIndex.build(items).node_ids == ("01",)


def test_edge_count_matches_the_neighbour_lists() -> None:
    graph = GraphIndex.build(_FIXTURE, config=GraphConfig(top_m=3, min_weight=0.0))
    total = sum(len(graph.neighbors(node_id)) for node_id in graph.node_ids)
    assert graph.edge_count() == total


def test_graph_path_lives_beside_the_database() -> None:
    """图必须在 ``memory/`` 里、和 ``mem.db`` 一起整体搬家，而不是散在数据根目录。"""
    from axo.paths import graph_path, memory_db_path

    assert graph_path().name == "graph.npz"
    assert graph_path().parent == memory_db_path().parent
    assert graph_path() != memory_db_path()
