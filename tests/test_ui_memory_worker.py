"""注入账本的两段渲染之一：召回过程。

:func:`render_recall_detail` 是纯函数，但它住在 ``ui/`` 里（账本是给界面看的），
所以这个文件标 ``gui``——核心腿用 ``--ignore-glob='tests/test_ui_*'`` 排除（**``--ignore``
不认通配符**，写成它一条都排不掉且不报错），与气体仓的分腿纪律一致。

它守的是账本的另一半价值：**"为什么没注入它"与"为什么注入了它"同样重要。**
只显示入选的那几条，用户看到的是"库里好像只有这几条相关"，而实际可能是"其余
二十条分数不够"。两者该做的动作完全不同——前者去补记忆，后者去调阈值。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from axo.ui.memory_worker import render_recall_detail
from memcore.audit import (
    ChannelScores,
    RecallLedger,
    RejectedCandidate,
    RejectReason,
)
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

pytestmark = pytest.mark.gui


def rejected(preview: str, reason: RejectReason, score: float = 0.1) -> RejectedCandidate:
    return RejectedCandidate(
        memory_id=f"id-{preview}", content_preview=preview, reason=reason, score=score
    )


def test_no_ledger_renders_nothing() -> None:
    """召回整个失败了。**空串而不是"没有记忆"**——那句话该由装配账本去说，
    两处都说会让"检索失败"和"库里确实没有"看起来是同一件事。"""
    assert render_recall_detail(None) == ""


def test_the_summary_line_always_comes_first() -> None:
    ledger = RecallLedger(query_preview="路由器", scanned=12, channels=("lexical", "semantic"))
    text = render_recall_detail(ledger)
    assert text.startswith(ledger.summary())
    assert "扫过 12 条" in text


def test_notes_are_shown_even_when_nothing_was_rejected() -> None:
    """降级说明必须出现：没有它，"这一轮没查到"和"库里没有"长得一模一样。"""
    ledger = RecallLedger(
        query_preview="路由器", channels=("lexical",), notes=("未登记，本轮只有词法通道",)
    )
    assert "未登记，本轮只有词法通道" in render_recall_detail(ledger)


def test_rejected_candidates_are_grouped_by_reason() -> None:
    """按原因分组：同一个原因重复十遍没有信息量，而**条数**有。"""
    ledger = RecallLedger(
        query_preview="路由器",
        rejected=(
            rejected("午饭吃的是番茄炒蛋", RejectReason.BELOW_THRESHOLD, 0.04),
            rejected("周三开组会", RejectReason.BELOW_THRESHOLD, 0.08),
            rejected("路由器放在书房，网线在电视柜后面，密码贴在冰箱上", RejectReason.OVER_BUDGET),
        ),
    )
    text = render_recall_detail(ledger)
    assert "2 条：分数不够" in text
    assert "1 条：装不进剩余预算" in text
    # 每一条都要能看见它是什么、以及在哪个数上被淘汰的。
    assert "周三开组会" in text
    assert "得分 0.08" in text


def test_the_bigger_group_is_listed_first() -> None:
    """先读到的应该是**主因**：十条分数不够和十条装不下，指向的动作完全不同。"""
    ledger = RecallLedger(
        query_preview="路由器",
        rejected=(
            rejected("A", RejectReason.OVER_BUDGET),
            rejected("B", RejectReason.BELOW_THRESHOLD),
            rejected("C", RejectReason.BELOW_THRESHOLD),
        ),
    )
    text = render_recall_detail(ledger)
    assert text.index("2 条：分数不够") < text.index("1 条：装不进剩余预算")


def test_a_long_group_is_truncated_with_the_remainder_counted() -> None:
    """截断要**说清楚还剩多少条**。

    悄悄砍掉尾巴的话，账本读起来像是"库里就这八条不相关"，而真相可能是"八百条"。
    """
    ledger = RecallLedger(
        query_preview="路由器",
        rejected=tuple(
            rejected(f"候选 {index}", RejectReason.BELOW_THRESHOLD) for index in range(11)
        ),
    )
    text = render_recall_detail(ledger, max_rejected=3)
    assert "11 条：分数不够" in text
    assert "候选 2" in text
    assert "候选 3" not in text
    assert "…另有 8 条" in text


def test_the_no_encoding_reason_carries_its_own_explanation() -> None:
    """``NO_ENCODING`` 指向的是**数据问题**（向量没回填上），不是排序结果。

    它在账本里的标签必须把这件事说出来——否则它会和"分数不够"混成同一类，而两者
    该做的动作一个是查回填、一个是调阈值。
    """
    ledger = RecallLedger(
        query_preview="路由器",
        rejected=(rejected("没人回填过这条", RejectReason.NO_ENCODING),),
    )
    assert "向量可能没回填上" in render_recall_detail(ledger)


def test_an_injected_memory_is_not_rendered_twice() -> None:
    """入选的那几条由装配账本负责渲染（它带 token 数与权重），这里不重复。

    两处都渲染同一批内容，会让账本看起来像注入了两倍的东西。
    """
    from memcore.audit import RecalledMemory

    ledger = RecallLedger(
        query_preview="路由器",
        injected=(
            RecalledMemory(
                memory_id="id-1",
                content="我家的路由器放在书房书架顶层。",
                kind="fact",
                tier="longterm",
                weight=0.8,
                tokens=18,
                scores=ChannelScores(lexical=0.9, fused=0.9),
            ),
        ),
    )
    assert "我家的路由器放在书房书架顶层。" not in render_recall_detail(ledger)


# ── MemoryProbeWorker：记忆界面的「以它为线索召回一次」 ──────────────────────
#
# 直调 ``run()`` 而不是起线程：这里要断言的是**它算出什么**，线程形状由
# ``test_ui_worker_track.py`` 那五条统一钉住，两件事分开测。


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


class _FakeEmbedding:
    """形状与 :class:`OllamaEmbeddingBackend` 一致的两个成员：``embed`` 与 ``space_id``。"""

    def __init__(self, vector: Any = None, *, boom: bool = False, space_id: str = "space@8"):
        self._vector = vector
        self._boom = boom
        self.space_id = space_id

    def embed(self, texts: Sequence[str]) -> list[Any]:
        if self._boom:
            raise RuntimeError("连接被拒绝")
        return [self._vector if self._vector is not None else np.zeros(8, dtype=np.float32)]


def make_item(text: str, **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 0.7,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


def _probe(store: MemoryStore, content: str, embedding: Any = None) -> RecallLedger:
    from axo.ui.memory_worker import MemoryProbeWorker

    worker = MemoryProbeWorker(store=store, content=content, embedding=embedding)
    captured: list[RecallLedger] = []
    worker.ready.connect(captured.append)
    worker.run()
    assert captured, "探针没有给出账本"
    return captured[0]


def test_probe_without_an_embedding_falls_back_to_lexical(qapp: Any, store: MemoryStore) -> None:
    """**没有嵌入模型不是失败，是一个正常配置。**

    代码里唯一会走这条路的判断是 ``query_vector=None``，而 ``recall`` 对它的
    处理是"只用词法通道"——账本照常产出。
    """
    store.add(make_item("我家的路由器放在书房。"))

    ledger = _probe(store, "路由器放在哪儿")

    assert ledger.scanned == 1
    assert "没有嵌入模型" in " ".join(ledger.notes)


def test_probe_degrades_instead_of_failing_when_embedding_raises(
    qapp: Any, store: MemoryStore
) -> None:
    """嵌入抛异常也是**降级**：不发 ``failed``，账本里写明原因。

    发 failed 的话，用户在"Ollama 那一刻正忙"时点这个按钮只会得到一个错误提示，
    而他本来应该看到词法通道给出的邻居——那正是他要问的东西。
    """
    store.add(make_item("我家的路由器放在书房。"))

    ledger = _probe(store, "路由器放在哪儿", _FakeEmbedding(boom=True))

    assert "嵌入失败" in " ".join(ledger.notes)
    assert "只用语汇匹配" in " ".join(ledger.notes)


def test_probe_does_not_settle(qapp: Any, store: MemoryStore) -> None:
    """**一次观察不能改变被观察的东西。**

    开着 settle 的话，用户每点一次探索按钮，被看到的那几条就会显得更常用、
    衰减得更慢——而真实召回的排序从此带着探索的痕迹，没有任何地方看得出来。
    """
    item = store.add(make_item("我家的路由器放在书房。"))
    before = store.get(item.memory_id)
    assert before is not None

    _probe(store, "我家的路由器放在书房。")

    after = store.get(item.memory_id)
    assert after is not None
    assert after.access_count == before.access_count
    assert after.weight == before.weight


def test_probe_uses_the_exploratory_config(qapp: Any, store: MemoryStore) -> None:
    """阈值降到 0.02、条数放宽到 20——只把当初那 8 条再列一遍的话，
    这个按钮什么也没回答。"""
    from axo.memory.browse import PROBE_CONFIG

    assert PROBE_CONFIG.max_items == 20
    assert PROBE_CONFIG.min_score < 0.10  # 生产是 0.10

    store.add(make_item("我家的路由器放在书房。"))
    ledger = _probe(store, "路由器")

    assert ledger.scanned == 1
    assert len(ledger.injected) >= 1


# ── GraphWorker：建图/载入，以及"什么时候不该建" ─────────────────────────────
#
# 同样直调 ``run()``。**盘上的路径要换掉**：``graph_path()`` 指的是真数据目录，
# 不换的话这个文件会去读写你机器上那份真图——而它是个缓存，症状是"跑完测试之后
# 图变了"，没有任何报错。


@pytest.fixture
def graph_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 ``graph_path()`` 指到 ``tmp_path`` 上，返回那个路径。"""
    import axo.ui.memory_worker as module

    path = tmp_path / "graph.npz"
    monkeypatch.setattr(module, "graph_path", lambda: path)
    return path


def _build(store: MemoryStore, *, force: bool = False):  # noqa: ANN202 — (图, 三态, note)
    from axo.ui.memory_worker import GraphWorker

    worker = GraphWorker(store=store, force=force)
    ready: list[Any] = []
    failed: list[str] = []
    worker.ready.connect(lambda *args: ready.append(args))
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed, f"建图不该失败：{failed}"
    assert len(ready) == 1, f"只该发一次 ready，收到 {len(ready)}"
    return ready[0]


def test_a_small_library_is_built_and_written_to_disk(
    qapp: Any, store: MemoryStore, graph_file: Path
) -> None:
    store.add(make_item("我家的路由器放在书房。"))
    store.add(make_item("路由器是上周换的。"))
    graph, freshness, note = _build(store)

    assert len(graph) == 2
    assert freshness is False, "刚建好的图当然不是过期的，也不是「没比对过」"
    assert "已建好" in note
    assert graph_file.exists(), "图是缓存，但要能省下下次那几秒"


def test_a_large_library_reuses_the_stored_graph_without_rebuilding(
    qapp: Any, store: MemoryStore, graph_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**大库不重建、也不遍历**——这是本版的阈值纪律。

    做法是把阈值压到 2，再预先放一份"只有 1 个节点"的图。若那条捷径失效，worker 会
    照当前 3 条重建，回来的图就有 3 个节点——断言于是红。
    """
    import axo.ui.memory_worker as module

    store.add(make_item("第一条。"))
    store.add(make_item("第二条。"))
    store.add(make_item("第三条。"))

    from memcore.graph import GraphIndex

    stored = GraphIndex.build([make_item("只有这一条。")])
    stored.save(graph_file)
    monkeypatch.setattr(module, "AUTO_GRAPH_MAX", 2)

    graph, freshness, note = _build(store)

    assert len(graph) == 1, "大库那条路必须原样交回盘上那份，不重建"
    assert freshness is None, "没跟库比对过——必须报 None，不许冒充「确认没过期」"
    assert note == module._LARGE_LIB_MISSING_NOTE.format(count=3)
    assert "3 条新记忆不在图里" in note, "成员比对是新加的可见性，别退回只说「可能过期」"


def test_a_large_library_whose_nodes_are_all_present_keeps_the_plain_note(
    qapp: Any, store: MemoryStore, graph_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图与库的成员**一致**时，不冒充"新记忆没入图"——只说"没比对内容"。"""
    import axo.ui.memory_worker as module
    from memcore.graph import GraphIndex

    store.add(make_item("第一条。"))
    store.add(make_item("第二条。"))
    store.add(make_item("第三条。"))
    GraphIndex.build(store.iter_items()).save(graph_file)
    monkeypatch.setattr(module, "AUTO_GRAPH_MAX", 2)

    graph, freshness, note = _build(store)

    assert len(graph) == 3
    assert freshness is None
    assert note == module._LARGE_LIB_NOTE


def test_force_rebuilds_even_a_large_library(
    qapp: Any, store: MemoryStore, graph_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**手动重建必须真的重建**，哪怕库大到自动那条路会直接复用。

    少了这一条，大库上点「重建联想」会把旧图原样还回来——而那个按钮存在的全部理由
    就是大库（自动那条路根本不比对）。按钮看起来坏了，还没有任何报错。
    """
    import axo.ui.memory_worker as module

    store.add(make_item("第一条。"))
    store.add(make_item("第二条。"))
    store.add(make_item("第三条。"))

    from memcore.graph import GraphIndex

    GraphIndex.build([make_item("只有这一条。")]).save(graph_file)
    monkeypatch.setattr(module, "AUTO_GRAPH_MAX", 2)

    graph, _freshness, note = _build(store, force=True)

    assert len(graph) == 3, "force 之下必须按当前的库重建"
    assert "已建好" in note


def test_a_fresh_small_graph_is_reused_rather_than_rebuilt(
    qapp: Any, store: MemoryStore, graph_file: Path
) -> None:
    """库没变就不重建：那几秒（大库是几分钟）省下来，而且**说清楚是复用**。"""
    store.add(make_item("我家的路由器放在书房。"))
    _build(store)

    graph, freshness, note = _build(store)

    assert len(graph) == 1
    assert freshness is False
    assert "不用重建" in note, "复用与重建必须说得出区别，否则用户分不清这次有没有干活"


def test_a_changed_small_library_is_rebuilt(
    qapp: Any, store: MemoryStore, graph_file: Path
) -> None:
    """加了一条 → 指纹变了 → 重建。这一条与上一条合起来才钉住"够用就好"的边界。"""
    store.add(make_item("我家的路由器放在书房。"))
    _build(store)
    store.add(make_item("路由器是上周换的。"))

    graph, freshness, _note = _build(store)

    assert len(graph) == 2
    assert freshness is False


def test_a_corrupt_graph_file_is_rebuilt_not_reported(
    qapp: Any, store: MemoryStore, graph_file: Path
) -> None:
    """盘上那份坏了 = **没有图**，不是失败。

    图是缓存：文件截断、版本对不上、写入时断电，都只说明该重算一遍。让它冒成
    ``failed`` 的话，用户看到的是"建立关联图失败"的红字，而正确答案是重新建一张。
    """
    store.add(make_item("我家的路由器放在书房。"))
    graph_file.write_bytes(b"not an npz at all")

    graph, _freshness, note = _build(store)

    assert len(graph) == 1
    assert "已建好" in note


def test_a_write_failure_still_hands_back_the_graph(
    qapp: Any, store: MemoryStore, graph_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写盘失败**不吞掉刚算好的图**：本次会话里照样能用，下次启动再建一遍。"""
    from memcore.graph import GraphIndex

    store.add(make_item("我家的路由器放在书房。"))

    def boom(self: GraphIndex, path: Path) -> None:
        raise OSError("磁盘满了")

    monkeypatch.setattr(GraphIndex, "save", boom)

    graph, _freshness, note = _build(store)

    assert len(graph) == 1, "算了半天的图不许因为写不进去就丢掉"
    assert "没能写盘" in note and "重启后会再建" in note


def test_the_assembly_worker_never_excludes_the_current_session() -> None:
    """**接线守卫**：装配这一路不再排除本会话写下的记忆。

    纯层全绿不等于接线在——这条守的是"谁调它"。排除一旦接回来，症状是老会话看
    不到自己最近写的那批，而那正是「新老会话记忆可达范围差很远」的真凶之一。
    """
    import importlib

    module = importlib.import_module("axo.ui.memory_worker")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "ids_for_session" not in source, "排除本会话的那条又接回来了"
