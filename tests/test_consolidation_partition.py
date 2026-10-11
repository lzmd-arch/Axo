"""分簇 —— 睡眠巩固的第一步，也是它唯一一处**图算法**。

这个文件的重心是**确定性**。``tools/sleep_once.py --replay`` 要靠"同一张图两次
跑出同一串事件"才有意义，而事件序列的顺序完全由分簇结果的顺序决定。所以这里
不只验"分出来的簇合不合理"，更要验"**换一次跑法分出来的还是同一批簇、同一个
顺序**"——后者才是可复现性的承重件。

手工搭图而不是 :meth:`~memcore.graph.GraphIndex.build` 出来，理由同
``tests/test_graph.py``：这里的重点是**边表本身**（谁连着谁、边权多大、是不是
双向），从 SDR 反推边表会把这几条挡在哈希后面，而"为什么这两个被分到一起"正是
要验的那件事。
"""

from __future__ import annotations

import numpy as np
import pytest

from memcore.consolidation import MAX_CLUSTER, MIN_CLUSTER, partition
from memcore.graph import GraphIndex


def _graph(rows: dict[int, list[tuple[int, float]]], size: int, *, top_m: int = 8) -> GraphIndex:
    """手工搭一张关联图，空槽照 :meth:`GraphIndex.build` 的形状填 -1 / 0。

    ``rows[3] = [(0, 0.5)]`` 读作"3 号的邻居表第一格是 0，边权 0.5"，也就是
    **有向边 3→0**。

    节点 id 零填充成 ``m000``/``m001``…… 是**为了能按字典序读**：``GraphIndex``
    要求 ``node_ids`` 已排序，而 ``m10`` < ``m2``——不填充的话 10 号会排到 2 号
    前面，手工算期望值时很容易看错，而看错的方向恰好是"测试跟着实现走"。
    """
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
    """把一串无向边写成**双向**的邻接表，省得每个用例手抄两遍。"""
    rows: dict[int, list[tuple[int, float]]] = {}
    for a, b in pairs:
        rows.setdefault(a, []).append((b, weight))
        rows.setdefault(b, []).append((a, weight))
    return rows


def _ring(size: int, weight: float = 1.0) -> dict[int, list[tuple[int, float]]]:
    """一个环：``i`` 连着 ``i+1``（最后一节绕回 0）。这是最干净的"一个大连通分量"。"""
    return _mutual([(index, (index + 1) % size) for index in range(size)], weight)


# ── 分出来的是什么 ──────────────────────────────────────────────────────────


def test_two_cliques_come_out_as_two_clusters() -> None:
    """两块互不相连的三角，就应当是两个簇——而且成员一条不多一条不少。

    这条是所有断言的基准：如果连"两个分得开的团"都分不出来，后面那些"上界、
    下界、顺序"全是空转。
    """
    rows = _mutual([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    graph = _graph(rows, size=6)

    assert partition(graph) == (("m000", "m001", "m002"), ("m003", "m004", "m005"))


def test_an_oversized_component_is_split_into_contiguous_pieces() -> None:
    """12 个节点的一个环、上限 4：切成三块，每块 4 条，且互不重叠。

    **这个环是刻意选的**：它整块是一个连通分量，标签传播**必然**给出一个组
    （每轮每个节点都被邻居拉向同一个标签），所以这条实际测的是区域生长——
    真实库上最短路的 top-m 图就是这个形状，标签传播给不出细簇。

    期望值是**手工推的**：种子取组内加权度最大者（环上人人都是 2，并列取
    ``m000``），然后每次并入"到当前簇边权之和最大"的未分配邻居（并列取
    ``memory_id`` 最小者），满 4 就封簇。
    """
    graph = _graph(_ring(12), size=12)

    assert partition(graph, max_cluster=4) == (
        ("m000", "m001", "m002", "m003"),
        ("m005", "m004", "m006", "m007"),
        ("m009", "m008", "m010", "m011"),
    )


def test_no_cluster_exceeds_the_upper_bound() -> None:
    """上界是硬约束——它背后是 W 的容量悬崖，不是审美。"""
    graph = _graph(_ring(40), size=40)

    sizes = [len(cluster) for cluster in partition(graph, max_cluster=6)]
    assert sizes, "40 个节点不可能一条簇都分不出来"
    assert max(sizes) <= 6


def test_the_upper_bound_is_never_exceeded_by_a_single_dense_blob() -> None:
    """整块全连（每对都有边）也必须被切开——最坏情况下上界照样成立。

    全连图里标签传播一轮就塌成一个组，然后**所有**节点并列同样的加权度，
    于是种子与生长全都要靠"取 ``memory_id`` 最小者"来打破。这条同时是
    确定性的一根探针：并列处的打破规则一动，成员就变了。
    """
    size = 9
    rows = _mutual([(a, b) for a in range(size) for b in range(a + 1, size)])
    graph = _graph(rows, size=size, top_m=size)

    clusters = partition(graph, max_cluster=4)
    assert [len(cluster) for cluster in clusters] == [4, 4]
    assert sorted(member for cluster in clusters for member in cluster) == [
        f"m{index:03d}" for index in range(8)
    ], "第 9 个（m008）落单，按 MIN_CLUSTER 被丢掉"


def test_an_isolated_node_is_dropped_rather_than_clustered_alone() -> None:
    """孤点不进结果。

    **这不是为了整齐。** 单成员簇的 W 只有一个模式，
    :meth:`~memcore.substrate.Substrate.complete` 对任何探针都恒回它自己——
    于是"回放辨认率"被一批必然命中的样本抬高，而那个数字看起来完全正常。
    """
    rows = _mutual([(0, 1)])
    graph = _graph(rows, size=3)  # 2 号谁都不连

    result = partition(graph)
    assert result == (("m000", "m001"),)
    assert all("m002" not in cluster for cluster in result)


def test_the_floor_filters_a_leftover_singleton() -> None:
    """上限 2 的三角切完会剩一个落单的——它按 :data:`MIN_CLUSTER` 被丢掉。

    这条把"上界"与"下界"撞在一起看：切分不会为了让每个节点都有归属而放松
    上界，宁可剩下来不给回应。
    """
    graph = _graph(_mutual([(0, 1), (1, 2), (0, 2)]), size=3)

    assert partition(graph, max_cluster=2) == (("m000", "m001"),)


def test_every_member_appears_at_most_once() -> None:
    """簇之间不许重叠——一条记忆被两个簇各回放一次，就会把 ``reinforce_count``
    加两次，而"加固了几次"这个数正是给人看的证据。"""
    graph = _graph(_ring(25), size=25)

    members = [member for cluster in partition(graph, max_cluster=5) for member in cluster]
    assert len(members) == len(set(members))


# ── 顺序：``--replay`` 的承重件 ────────────────────────────────────────────


def test_clusters_are_ordered_by_their_first_members_id() -> None:
    """簇间按第一个成员的 ``memory_id`` 升序——规范顺序是逐条比对的前提。"""
    graph = _graph(_ring(12), size=12)

    clusters = partition(graph, max_cluster=4)
    firsts = [cluster[0] for cluster in clusters]
    assert firsts == sorted(firsts)


def test_members_are_memory_ids_not_row_indices() -> None:
    """返回的是 ``memory_id``，不是行号。

    行号在这个用例里恰好是 ``0``、``1``、``2``，而 ``memory_id`` 是
    ``"m000"``、``"m001"``、``"m002"``——把行号当成 id 返回的话，事件里的
    ``memory_id`` 会是一个整数，而 :class:`~memcore.types.ConsolidationEvent`
    的字段是 ``str``。这条把那个错误挡在类型检查之外的地方。
    """
    graph = _graph(_mutual([(0, 1)]), size=2)

    (cluster,) = partition(graph)
    assert cluster == ("m000", "m001")
    assert all(isinstance(member, str) for member in cluster)


def test_the_partition_is_deterministic_across_repeated_calls() -> None:
    """同一张图连算 8 次，逐簇逐条相同。

    **这条单独用是不够的**：把并列打破从"取最小"改成"取最后一个"之后它照样全绿
    ——那还是一个确定性的答案，只是换了一个。真正钉住并列规则的是上面那几条
    写死成员的断言。它守的是另一件事：实现里**不许引入**依赖 ``set`` 迭代顺序或
    哈希种子的路径，那种不确定性在小图上时有时无，而在这里会被八次调用放大。
    """
    graph = _graph(_ring(23), size=23)

    results = [partition(graph, max_cluster=5) for _ in range(8)]
    assert all(other == results[0] for other in results[1:])


def test_a_one_direction_edge_still_joins_the_pair() -> None:
    """**边只记一个方向也算连上。**

    ``GraphIndex`` 存的是每个节点**自己**的 top-m，所以 0→1 存在而 1→0 不存在
    是常态。折邻接表时若只按行写一格，得到的是有向图，而被它喂大的标签传播与
    区域生长都会变成方向相关的——症状是分簇取决于哪一端"更主动"，
    而结果看起来仍然像个合理的分簇。

    **但这一条单独测不出那个 bug**：一对节点、边只有一个方向时，标签传播仍会
    把落后的那个拉过来，只写单向照样绿。真正能分辨的是
    :func:`test_the_direction_of_an_edge_does_not_decide_the_seed`——它要的是
    "方向改变**种子**的选择"，那要更长的路径才看得出来。
    """
    # 只有 1→0，没有 0→1。
    graph = _graph({1: [(0, 0.9)]}, size=2)

    assert partition(graph) == (("m000", "m001"),)


def test_the_direction_of_an_edge_does_not_decide_the_seed() -> None:
    """**方向不许影响种子，因而不许影响分簇。**

    构造：四颗卫星**全都指向**中心 0（入边为主），0 自己那一行是空的——真实图上
    这很常见，一条被很多人视为近邻的记忆，自己未必把他们都收进 top-m。

    两个读法给出的答案是**不一样的**，而且都能自圆其说：

    * 折成无向（正确）：中心的无向度是 4，当选种子 → ``(m000, m001, m002, m003)``；
    * 只按行读（错误）：中心的出度是 0，种子落到 1 号，簇只有 ``(m001, m000)``
      两条，剩下三颗各自落单。

    差别有三条成员之多。上一条（一对节点）测不出这个 bug，因为标签传播仍会把
    落后的那一个拉过来——**要看出方向的影响，得有"谁当种子"这件事真的分岔**，
    而那至少要四五个节点加一次切分。
    """
    graph = _graph({index: [(0, 1.0)] for index in range(1, 5)}, size=5)

    assert partition(graph, max_cluster=4) == (("m000", "m001", "m002", "m003"),)


def test_a_graph_of_pure_in_edges_still_shares_one_seed() -> None:
    """上一条的另一半：上限放宽到装得下全部时，整个星形是一簇。

    这条挡的是"折邻接表时把入边整个丢掉"——那样中心会变成孤点，卫星彼此不相邻，
    于是分出来的是一堆单成员簇，全部被 :data:`MIN_CLUSTER` 丢掉，
    ``partition`` 返回空元组。**空元组是一条安静得可怕的失败**：巩固照跑，
    只是什么都没做，而报告上"本轮回放 0 条"看起来完全正常。
    """
    graph = _graph({index: [(0, 1.0)] for index in range(1, 5)}, size=5)

    assert partition(graph) == (("m000", "m001", "m002", "m003", "m004"),)


def test_the_two_directions_take_the_larger_weight() -> None:
    """同一对边的两个方向权重不同时取 ``max``，不是相加。

    取和的后果是把"互相都认为对方很像"抬成两倍——而"像的程度"不该由记账次数
    决定，这是 :meth:`~memcore.graph.GraphIndex.diffuse` 早就定下的口径。
    这里验的是可观测的那一半：权重不影响**成员归属**（0.9 还是 0.1 都连），
    但它必须被读成一个数而不是两个。
    """
    graph = _graph({0: [(1, 0.9)], 1: [(0, 0.1)]}, size=2)

    assert partition(graph) == (("m000", "m001"),)


# ── 边界 ────────────────────────────────────────────────────────────────────


def test_an_empty_graph_yields_no_clusters() -> None:
    """空图不抛异常，返回空元组——"图还没建"是常态，不是错误。"""
    graph = _graph({}, size=0)

    assert partition(graph) == ()


def test_a_graph_with_no_edges_yields_no_clusters() -> None:
    graph = _graph({}, size=5)

    assert partition(graph) == ()


def test_defaults_come_from_the_module_constants() -> None:
    """默认值就是那两个常量本身——它们各有出处，别在签名里抄一份数字。"""
    assert MIN_CLUSTER == 2
    assert MAX_CLUSTER == 200


def test_bad_bounds_are_rejected() -> None:
    """上界小于下界是**没有任何簇能同时满足**的参数，当场拒掉比空跑一轮好。"""
    graph = _graph(_mutual([(0, 1)]), size=2)

    with pytest.raises(ValueError, match="max_cluster"):
        partition(graph, max_cluster=3, min_cluster=5)
    with pytest.raises(ValueError, match="min_cluster"):
        partition(graph, min_cluster=0)
