"""记忆浏览器窗口：列表、筛选、详情、账本推送。

这个文件标 ``gui``：核心腿用 ``--ignore-glob='tests/test_ui_*'`` 排除（**``--ignore``
不认通配符**，写成它一条都排不掉且不报错），与气体仓的分腿纪律一致。窗口本身只做布局
与信号，所以这里断言的是**接线对不对**——判断句（"这条为什么没被注入"）在
``test_memory_view.py`` 里断言。

三条断言比别处要紧：

* ``test_hitting_the_page_limit_is_said_out_loud`` —— 命中上限不静默截断。
  "列表里没有"被读成"库里没有"会导出完全相反的动作。
* ``test_the_ledger_push_replaces_the_previous_one`` —— **账本靠推送靠赋值，不靠
  捕获引用**。主窗口那个 ``_recall_ledger`` 每轮重新赋值，窗口捕获一次引用之后
  下一轮就指向旧值，而且不报错。
* ``test_a_row_that_vanished_says_so`` —— 选中项被别处删掉时**说出来**，不静默
  退回空白（用户会以为是自己点错了）。

**不造真的 ``MainWindow``**：它一构造就连 Ollama、起多个线程、加载 WebEngine。
主窗口的接线由 ``tools/_smoke_memory.py``（真进程、真退出码）覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biosnn_memterm.memory.browse import MemoryStatus
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.spaces import Space, make_space_id
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

pytestmark = pytest.mark.gui

SPACE = make_space_id("test", "fake-embed", 8)
OTHER_SPACE = make_space_id("test", "other-embed", 8)


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    """一个真库，落在 ``tmp_path`` 上。

    **同时把数据目录也指过去**：热度面板要读 ``memory/consolidation/last.json``
    （见 ``consolidation.last_run``），而那个文件的默认位置是 ``%LOCALAPPDATA%``
    ——不指的话，这几条测试的结果会随"这台机器上跑没跑过巩固"而变。库与
    ``last.json`` 指向两个不同的地方本身就是个说不通的组合。
    """
    monkeypatch.setenv("BIOSNN_MEMTERM_DATA", str(tmp_path))
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


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


def make_window(qapp: Any, store: MemoryStore):  # noqa: ANN202 — 返回值是 Qt 类
    from biosnn_memterm.ui.memory_window import MemoryWindow

    return MemoryWindow(memory=store)


def row_ids(window: Any) -> list[str]:
    from PySide6.QtCore import Qt

    return [
        str(window.memory_list.item(index).data(Qt.ItemDataRole.UserRole))
        for index in range(window.memory_list.count())
    ]


# ── 列表 ────────────────────────────────────────────────────────────────────


def test_the_list_shows_every_entry(qapp: Any, store: MemoryStore) -> None:
    for index in range(5):
        store.add(make_item(f"第 {index} 条。"))

    window = make_window(qapp, store)

    assert window.memory_list.count() == 5
    assert window.memory_list.count() == store.count()


def test_the_list_is_newest_first(qapp: Any, store: MemoryStore) -> None:
    """``summaries()`` 按 ``memory_id DESC``——ULID 单调，所以最新在前。"""
    first = store.add(make_item("先写的。"))
    second = store.add(make_item("后写的。"))

    window = make_window(qapp, store)

    assert row_ids(window) == [second.memory_id, first.memory_id]


def test_a_forgotten_item_is_still_listed(qapp: Any, store: MemoryStore) -> None:
    """**软遗忘的条目必须留在列表里**，否则用户没有任何办法把它恢复回来。"""
    item = store.add(make_item("忘了的。", tier=Tier.COLD))

    window = make_window(qapp, store)

    assert row_ids(window) == [item.memory_id]
    assert "已遗忘" in window.memory_list.item(0).text()


# ── 筛选 ────────────────────────────────────────────────────────────────────


def test_the_search_box_narrows_the_list(qapp: Any, store: MemoryStore) -> None:
    store.add(make_item("我家的路由器放在书房。"))
    store.add(make_item("今天午饭是番茄炒蛋。"))

    window = make_window(qapp, store)
    window.search_edit.setText("路由器")

    assert window.memory_list.count() == 1
    assert "路由器" in window.memory_list.item(0).text()


def test_the_search_box_can_find_an_item_beyond_the_page(qapp: Any, store: MemoryStore) -> None:
    """**搜索走 SQL 全库**，不是"在已载入的那一页里再筛一层"。

    列表一屏只装最近若干条，而"我明明记过这件事"要找的很可能是更早的那条。
    分层的理由就是这个：不这样做，搜索框在库里条目多起来之后会开始"找不到"。
    """
    from biosnn_memterm.ui import memory_window as module

    store.add(make_item("埋在深处的老记忆。"))
    for index in range(4):
        store.add(make_item(f"新的 {index}。"))

    window = make_window(qapp, store)
    # 只载入最近两条：老记忆在那一页之外。
    original = module.DEFAULT_LIST_LIMIT
    module.DEFAULT_LIST_LIMIT = 2
    try:
        window.reload()
        assert window.memory_list.count() == 2

        window.search_edit.setText("老记忆")

        assert window.memory_list.count() == 1
        assert "老记忆" in window.memory_list.item(0).text()
    finally:
        module.DEFAULT_LIST_LIMIT = original


def test_hitting_the_page_limit_is_said_out_loud(qapp: Any, store: MemoryStore) -> None:
    """命中上限**必须说出来**。

    悄悄截断的话，"列表里没有"会被读成"库里没有"——而这两件事该做的下一步完全
    相反（一个是"往下翻/搜一下"，一个是"它真的忘了"）。
    """
    from biosnn_memterm.ui import memory_window as module

    for index in range(5):
        store.add(make_item(f"第 {index} 条。"))

    window = make_window(qapp, store)
    original = module.DEFAULT_LIST_LIMIT
    module.DEFAULT_LIST_LIMIT = 3
    try:
        window.reload()

        text = window.count_label.text()
        assert "2000" not in text  # 说的是**实际生效**的那个数，不是写死的常量
        assert "最近 3 条" in text
        assert "共 5 条" in text
        assert "搜索" in text
    finally:
        module.DEFAULT_LIST_LIMIT = original


def test_the_tier_kind_and_status_filters_narrow_the_list(qapp: Any, store: MemoryStore) -> None:
    anchor = store.add(make_item("人格锚。", kind=MemoryKind.PERSONA_ANCHOR, pinned=True))
    forgotten = store.add(make_item("忘掉的。", tier=Tier.COLD))
    plain = store.add(make_item("普通事实。"))

    window = make_window(qapp, store)

    window.kind_combo.setCurrentIndex(window.kind_combo.findData(MemoryKind.PERSONA_ANCHOR.value))
    assert row_ids(window) == [anchor.memory_id]

    window.kind_combo.setCurrentIndex(0)
    window.status_combo.setCurrentIndex(window.status_combo.findData(MemoryStatus.FORGOTTEN.value))
    assert row_ids(window) == [forgotten.memory_id]

    window.status_combo.setCurrentIndex(window.status_combo.findData(MemoryStatus.PINNED.value))
    assert row_ids(window) == [anchor.memory_id]

    # 「在用」= 不在冷层。钉住的那条也在用——两个维度不互斥。
    window.status_combo.setCurrentIndex(window.status_combo.findData(MemoryStatus.ACTIVE.value))
    assert set(row_ids(window)) == {anchor.memory_id, plain.memory_id}


def test_filtering_keeps_the_selection_when_the_item_survives(
    qapp: Any, store: MemoryStore
) -> None:
    """筛完还留在列表里的那条，要**还是选中的那条**。

    不还原的话，用户筛一下再改一条，改的就是另一条记忆了。
    """
    keep = store.add(make_item("留下的。"))
    store.add(make_item("被筛掉的。"))

    window = make_window(qapp, store)
    # 按 id 选，不按行号：列表是最新在前，而"第 0 行是哪一条"在这条断言里
    # 不是重点——重点是筛完之后**还是它**。
    window.memory_list.setCurrentRow(row_ids(window).index(keep.memory_id))
    assert window._selected_id() == keep.memory_id

    window.search_edit.setText("留下")

    assert window._selected_id() == keep.memory_id


# ── 详情 ────────────────────────────────────────────────────────────────────


def test_selecting_a_row_renders_its_detail(qapp: Any, store: MemoryStore) -> None:
    item = store.add(make_item("我家的路由器放在书房。", source_ref="turn://01JABC/2"))

    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)

    text = window.detail_view.toPlainText()
    assert item.memory_id in text
    assert "我家的路由器放在书房。" in text
    assert "turn://01JABC/2" in text


def test_the_vector_line_is_per_space(qapp: Any, store: MemoryStore) -> None:
    """换过嵌入模型之后库里有多个空间——一条记忆在新空间有向量、旧空间没有是常态。"""
    store.register_space(Space(space_id=SPACE, dim=8, model_ref="旧模型"))
    store.register_space(Space(space_id=OTHER_SPACE, dim=8, model_ref="新模型"))
    item = store.add(make_item("只在新空间里嵌过。"))
    store.set_embedding(item.memory_id, OTHER_SPACE, np.zeros(8, dtype=np.float32))

    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)

    text = window.detail_view.toPlainText()
    assert "无 · 旧模型" in text
    assert "有 · 新模型" in text


def test_the_heat_panel_without_data_carries_no_numbers(qapp: Any, store: MemoryStore) -> None:
    """空库上「SNN 热度」那一块**常驻**，而且不许有任何像数据的数字。

    常驻：它是库一级的摘要，不是某一条记忆的属性——列表空着的时候它照样得说得出话。

    无数字：**没有可写的数就一个数字都不许有**——一个 ``0`` 会让它看起来像算过。
    这一条与下面那条成对，两条各管一段输入。
    """
    import re

    window = make_window(qapp, store)

    text = window.pending_view.toPlainText()
    assert window.memory_list.count() == 0  # 库里空着，面板照常显示
    assert "未启用" in text
    assert re.search(r"\d", text) is None


def test_the_heat_panel_with_data_shows_the_real_numbers(qapp: Any, store: MemoryStore) -> None:
    """库里真有加固过的条目 → 面板要**把数说出来**。

    这是上一条的另一半。只钉"没数字"那一面的话，一个恒返回占位的实现全绿——而那样
    用户跑完一轮巩固之后，热度面板仍然说"巩固还没跑过"，比不说更误导。

    这里**不给 ``last.json``**（那个文件在数据目录里，测试的 ``tmp_path`` 下没有），
    所以时间那一截本来就不该出现——顺带把"拿不到时间就不编"这条在真窗口上也钉一遍。
    """
    item = store.add(make_item("客厅的路由器信号到卧室很弱。"))
    store.bump_reinforce([item.memory_id])

    window = make_window(qapp, store)

    text = window.pending_view.toPlainText()
    assert "1 条被回放认出" in text
    assert "累计 1 次" in text
    assert "还没跑过" not in text, "库里已经加固过了，这句话就是假的"


def test_the_heat_panel_follows_a_finished_round(qapp: Any, store: MemoryStore) -> None:
    """**推一下就重画**，不用等用户手动刷新或重开窗口。

    主窗口跑完一轮巩固会调 :meth:`notify_consolidated`，而这一块画的是库里的
    ``reinforce_count``——不推的话，用户看到的是"巩固完了"（对话区那一行）与一份
    旧的摘要并排摆着，而两个说法不一致时他没有理由相信其中任何一个。
    """
    window = make_window(qapp, store)
    assert "还没跑过" in window.pending_view.toPlainText()

    item = store.add(make_item("客厅的路由器信号到卧室很弱。"))
    store.bump_reinforce([item.memory_id])
    window.notify_consolidated()

    assert "1 条被回放认出" in window.pending_view.toPlainText()


def test_the_association_panel_without_a_graph_carries_no_numbers(
    qapp: Any, store: MemoryStore
) -> None:
    """没 ``set_graph`` 时联想面板也要**说得出话**，而且同样一个数字都不许有。

    它还不能是空白：空白会被读成"这个面板坏了"。而那句话说到底是"图还没建"
    ——不是"没有邻居"（那条记忆可能有一堆邻居，只是图没建）。
    """
    import re

    store.add(make_item("客厅的路由器信号到卧室很弱。"))
    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)

    text = window.assoc_label.text()
    assert "还没有建" in text
    assert window.assoc_view.count() == 0
    assert re.search(r"\d", text) is None


def test_the_association_panel_lists_real_neighbors(qapp: Any, store: MemoryStore) -> None:
    """建了小图、选中一个节点之后，面板里出现邻居行，而且**允许出现数字**。

    与上面两条合起来才是那条边界：**无数据 → 零数字；有数据 → 真数字**。边权与
    记忆权重都是真的，抹成"较强"之类的形容词就等于把唯一能判断强弱的信息换成编的。
    """
    from memcore.graph import GraphIndex

    target = store.add(make_item("路由器放在客厅，卧室信号只剩一格。", weight=0.42))
    focus = store.add(make_item("客厅的路由器信号到卧室很弱。"))
    window = make_window(qapp, store)

    graph = GraphIndex.build([target, focus], built_at=1_700_000_000.0)
    window.set_graph(graph)
    window.memory_list.setCurrentRow(0)  # 最新在前，所以第 0 行是 focus

    assert window.assoc_view.count() >= 1, "两张内容相近的卡之间应当有边。"
    labels = [window.assoc_view.item(i).text() for i in range(window.assoc_view.count())]
    assert any(target.memory_id in label or "路由器" in label for label in labels)
    assert any(ch.isdigit() for label in labels for ch in label), (
        f"邻居行里应当有真的边权/权重数字，实际 {labels}"
    )
    assert "个邻居" in window.assoc_label.text()
    #: 面板里没有选中项时的说法与上面两条都不同——**那是第六种状态**，"还没有建"
    #: 与"没有邻居"都不对（这两句都在假定手上有那么一条）。
    window.memory_list.setCurrentRow(-1)
    assert "选一条记忆" in window.assoc_label.text()


def _neighbor_row_of(window: Any, memory_id: str):  # noqa: ANN202 — 返回 Qt 类
    """在联想面板里找指向这个 id 的那一行。"""
    from PySide6.QtCore import Qt

    for index in range(window.assoc_view.count()):
        item = window.assoc_view.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == memory_id:
            return item
    raise AssertionError(f"联想面板里没有指向 {memory_id} 的行")


def test_clicking_a_neighbor_jumps_to_it(qapp: Any, store: MemoryStore) -> None:
    """点一条邻居 = 在左边的列表里选中它。**跳过去之后详情也跟着换**——因为跳转
    用的是 ``setCurrentRow``，它会发 ``currentItemChanged``，那正是"换了一条"的
    唯一入口。自己写一套"只改详情不改列表"的跳转就会两边不同步。"""
    from memcore.graph import GraphIndex

    target = store.add(make_item("路由器放在客厅，卧室信号只剩一格。"))
    focus = store.add(make_item("客厅的路由器信号到卧室很弱。"))
    window = make_window(qapp, store)
    window.set_graph(GraphIndex.build([target, focus], built_at=1_700_000_000.0))
    window.memory_list.setCurrentRow(0)

    window.assoc_view.itemClicked.emit(_neighbor_row_of(window, target.memory_id))

    assert window._selected_id() == target.memory_id
    assert "卧室信号只剩一格" in window.detail_view.toPlainText()


def test_a_jump_to_a_filtered_out_neighbor_says_so_and_leaves_the_filter_alone(
    qapp: Any, store: MemoryStore
) -> None:
    """邻居在当前筛选之外时**说出来，但不偷偷改筛选**。

    顺手清掉搜索框里的字会让列表"跳"到一个用户没要求的位置：他正看着一批筛过的
    结果，一次点击之后筛选没了——而他不会知道是为什么。（同 ``_restore_selection``
    的诚实口径。）
    """
    from memcore.graph import GraphIndex

    target = store.add(make_item("路由器放在客厅，卧室信号只剩一格。"))
    store.add(make_item("客厅的路由器信号到卧室很弱。"))
    window = make_window(qapp, store)
    window.set_graph(GraphIndex.build(store.load_all(), built_at=1_700_000_000.0))

    # 筛成只剩"路由器"之外的那一条，让 target 从列表里消失。
    window.search_edit.setText("信号到卧室")
    window.memory_list.setCurrentRow(0)
    assert target.memory_id not in row_ids(window)

    window.assoc_view.itemClicked.emit(_neighbor_row_of(window, target.memory_id))

    assert "不在当前列表里" in window.message_label.text()
    assert window.search_edit.text() == "信号到卧室"


def test_a_neighbor_that_was_deleted_says_it_cannot_jump(qapp: Any, store: MemoryStore) -> None:
    """图是旧的：边还指着一条已经删掉的记忆。**点它什么都不该发生，但要说出来。**

    行**没有置灰**是有意的：置灰的控件连 tooltip 都不给看，而这里的理由（"图是
    旧的"）只能点一下才说得出来。
    """
    from memcore.graph import GraphIndex

    target = store.add(make_item("路由器放在客厅，卧室信号只剩一格。"))
    focus = store.add(make_item("客厅的路由器信号到卧室很弱。"))
    window = make_window(qapp, store)
    window.set_graph(GraphIndex.build([target, focus], built_at=1_700_000_000.0))
    window.memory_list.setCurrentRow(0)

    store.delete(target.memory_id)
    window.reload()
    window.memory_list.setCurrentRow(0)

    row = _neighbor_row_of(window, target.memory_id)
    assert "已经不在库里了" in row.text()

    window.assoc_view.itemClicked.emit(row)

    assert "跳不过去" in window.message_label.text()
    assert window._selected_id() == focus.memory_id


# ── 账本推送 ────────────────────────────────────────────────────────────────


def test_the_ledger_push_replaces_the_previous_one(qapp: Any, store: MemoryStore) -> None:
    """**账本靠推送靠赋值。** 主窗口那个 ``_recall_ledger`` 每轮重新赋值，窗口若
    捕获一次引用，下一轮就指向旧值——而且不报错。"""
    from memcore.audit import RecallLedger

    item = store.add(make_item("会被注入的。"))
    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)

    assert "还没有装配过" in window.detail_view.toPlainText()

    window.set_recall_ledger(RecallLedger(query_preview="路由器"))

    assert "不在本轮账本里" in window.detail_view.toPlainText()

    window.set_recall_ledger(None)

    assert "还没有装配过" in window.detail_view.toPlainText()
    assert item.memory_id in window.detail_view.toPlainText()


def test_the_window_knows_which_session_is_current(qapp: Any, store: MemoryStore) -> None:
    """本会话条目是 recall 的第二条早退路径，与钉住**分开**说。"""
    from memcore.audit import RecallLedger

    item = store.add(make_item("本会话写的。", session_id="01JCURRENT"))
    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)
    window.set_recall_ledger(RecallLedger(query_preview="x"))

    before = window.detail_view.toPlainText()
    window.set_current_session("01JCURRENT")
    after = window.detail_view.toPlainText()

    assert "当前这个会话" not in before
    assert "当前这个会话" in after
    assert item.memory_id in after


# ── 竞态 ────────────────────────────────────────────────────────────────────


def test_a_row_that_vanished_says_so(qapp: Any, store: MemoryStore) -> None:
    """选中项在别处被删掉时**说出来**。

    静默退回空白的话，用户看到的是一个正常但空的面板——他会以为是自己点错了，
    而真相是那条记忆已经没了（列表下一次刷新才会变）。
    """
    item = store.add(make_item("马上要被删掉的。"))
    window = make_window(qapp, store)
    window.memory_list.setCurrentRow(0)

    store.delete(item.memory_id)
    window._refresh_detail()

    assert item.memory_id in window.detail_view.toPlainText()
    assert "已经不在了" in window.detail_view.toPlainText()


def test_reloading_after_another_row_was_deleted_does_not_re_enter(
    qapp: Any, store: MemoryStore
) -> None:
    """**回归守卫**：别处删掉一条再刷新列表，不许在 ``clear()`` 里重入 ``clear()``。

    这是一个真实的**进程级崩溃**（Windows ``0xc0000374``／访问违例），修复前
    6/6 复现，会把整条 gui 腿连同 pytest 已收集的输出一起带走——所以它比"断言红"
    严重一档：没有断言能跑，进程就没了。触发要三步齐全：窗口开着且**已选中一行**
    → 库里少了一条**别的**记忆（删被选中那条反而不触发）→ 随后 ``reload()``。

    撞上的机理：``memory_list.clear()`` 的过程中 Qt 会挪动当前行并发
    ``currentItemChanged``（实测挪到了那一条刚被删掉的行上），于是
    ``_refresh_detail`` 查到 ``item is None``，走它那条"另一处刚把它删了"的分支
    再调一次 ``reload``——那是嵌在 ``clear()`` 半途的**第二次 clear()**，Qt 的条目
    析构不可重入，外层 clear 接着在已释放的条目上跑。

    断言放在崩溃点之后，正好也是它的价值所在：**跑得到这两行**就说明没崩。
    """
    doomed = store.add(make_item("这一条马上要从别处删掉。"))
    survivor = store.add(make_item("这一条留在列表里。"))
    window = make_window(qapp, store)
    # 选中**另一条**是触发条件的一部分：删掉被选中的那条不崩（当前行不会挪）。
    pick(window, survivor.memory_id)

    store.delete(doomed.memory_id)
    window.reload()

    assert row_ids(window) == [survivor.memory_id]
    # 选中项按 id 还原，不是"重填后回到第 0 行"——这两者在删过东西之后会分叉。
    assert window._selected_id() == survivor.memory_id


# ── 写动作 ──────────────────────────────────────────────────────────────────
#
# 四个动作都是界面线程同步做的，所以这里可以直接调槽、立刻查库。两个模态对话框
# 抽成了 ``_ask_for_content`` / ``_confirm_delete`` 并在这里被替换掉：弹真对话框
# 要么卡住、要么得靠 QTimer 去点它。


def pick(window: Any, memory_id: str) -> None:
    """按 id 选中一行。列表是最新在前，"第 0 行"不是稳定的地址。"""
    window.memory_list.setCurrentRow(row_ids(window).index(memory_id))


def test_editing_recomputes_the_lexical_encoding_and_drops_the_vector(
    qapp: Any, store: MemoryStore
) -> None:
    """**改正文必须重算词法 SDR，并丢掉旧语义向量。**

    只改 ``content`` 的话召回仍按旧内容匹配它，而 MMR 还会拿旧向量去算相似度
    ——全程不报错。这是整个界面里最容易犯的静默错误，所以两条一起钉。
    """
    store.register_space(Space(space_id=SPACE, dim=8, model_ref="假嵌入"))
    item = store.add(make_item("我家的路由器放在书房。"))
    store.set_embedding(item.memory_id, SPACE, np.zeros(8, dtype=np.float32))
    before_lex = np.array(store.get(item.memory_id).lex_sdr.idx, copy=True)

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._ask_for_content = lambda _item: "我家的路由器换到客厅电视柜上了。"
    window._on_edit()

    after = store.get(item.memory_id)
    assert after.content == "我家的路由器换到客厅电视柜上了。"
    assert not np.array_equal(after.lex_sdr.idx, before_lex)
    # 旧向量没了 —— 该 id 重新回到待补队列，回填管线按新正文重算它。
    assert item.memory_id not in store.embedded_ids(SPACE)
    assert "已改" in window.message_label.text()


def test_an_empty_edit_is_refused_and_the_row_is_untouched(qapp: Any, store: MemoryStore) -> None:
    """空正文：**报错且库不变**。改坏了再报错等于没报错。"""
    item = store.add(make_item("原样。"))
    before = store.get(item.memory_id)

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._ask_for_content = lambda _item: ""
    window._on_edit()

    after = store.get(item.memory_id)
    assert after.content == "原样。"
    assert np.array_equal(after.lex_sdr.idx, before.lex_sdr.idx)
    assert "正文不能为空" in window.message_label.text()


def test_a_punctuation_only_edit_is_refused_too(qapp: Any, store: MemoryStore) -> None:
    """全标点：这是 ``SDRCapacityError``，而用户不该看到回溯。

    ``can_encode`` 与 ``edited_content`` 共用同一处判定，所以提示里说的理由与
    真正拦下它的理由永远是同一条。
    """
    item = store.add(make_item("原样。"))

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._ask_for_content = lambda _item: "！！！？？？"
    window._on_edit()

    assert store.get(item.memory_id).content == "原样。"
    assert "可编码的字符" in window.message_label.text()


def test_cancelling_the_edit_changes_nothing(qapp: Any, store: MemoryStore) -> None:
    item = store.add(make_item("原样。"))

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._ask_for_content = lambda _item: None
    window._on_edit()

    assert store.get(item.memory_id).content == "原样。"
    assert window.message_label.text() == ""


def test_a_failed_write_still_says_why(qapp: Any, store: MemoryStore) -> None:
    """并发下另一处刚把它删了 → ``StoreError``。

    **这句话不能被自己的刷新擦掉。** ``reload()`` 会清空列表，而清空当前项会发
    一条 ``currentItemChanged``，槽里把反馈置空——所以顺序必须是先刷新再说话。
    """
    from memcore.errors import StoreError

    item = store.add(make_item("马上要没的。"))

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._ask_for_content = lambda _item: "改成一个新正文。"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise StoreError("没有这条记忆")

    store.transform = boom  # type: ignore[method-assign]
    window._on_edit()

    assert "改不动" in window.message_label.text()


def test_soft_forgetting_then_restoring(qapp: Any, store: MemoryStore) -> None:
    """软遗忘是**可恢复的**，所以它留在列表里，而恢复要说清落回哪一层。"""
    item = store.add(make_item("先记住它。"))

    window = make_window(qapp, store)
    pick(window, item.memory_id)

    window._on_forget()

    forgotten = store.get(item.memory_id)
    assert forgotten.tier is Tier.COLD
    assert "已遗忘" in window.memory_list.item(0).text()
    assert "不再参与召回" in window.message_label.text()
    # 遗忘之后这两个按钮的角色互换：忘不了（已经忘了），但能恢复。
    assert not window.forget_button.isEnabled()
    assert window.restore_button.isEnabled()

    window._on_restore()

    assert store.get(item.memory_id).tier is Tier.LONGTERM
    assert "已恢复到「长期」" in window.message_label.text()
    assert not window.restore_button.isEnabled()


def test_a_pinned_memory_cannot_be_forgotten(qapp: Any, store: MemoryStore) -> None:
    """**钉住的条目禁软遗忘**，而且置灰必须带理由。

    理由是一条具体的失效模式：``recall`` 在 tier 判定**之前**跳过 pinned，所以
    软遗忘钉住的条目是个无声的空动作——界面显示"已遗忘"，它却仍在每一轮的上下文
    里（人格锚尤其，``persona_texts`` 只看 kind）。要让它消失只有彻底删除。
    """
    item = store.add(make_item("人格锚。", kind=MemoryKind.PERSONA_ANCHOR, pinned=True))

    window = make_window(qapp, store)
    pick(window, item.memory_id)

    assert not window.forget_button.isEnabled()
    assert "彻底删除" in window.forget_button.toolTip()
    assert window.delete_button.isEnabled()  # 删除仍然允许

    # 正常不可达（按钮已灰），但按钮状态与规则分居两处，所以这里直接调一次：
    # 它必须**什么都没做**，并把理由说出来。
    window._on_forget()

    assert store.get(item.memory_id).tier is Tier.LONGTERM
    assert "钉住" in window.message_label.text()


def test_delete_removes_the_row_and_its_vectors(qapp: Any, store: MemoryStore) -> None:
    store.register_space(Space(space_id=SPACE, dim=8, model_ref="假嵌入"))
    item = store.add(make_item("删掉我。"))
    store.set_embedding(item.memory_id, SPACE, np.zeros(8, dtype=np.float32))

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._confirm_delete = lambda _item: True
    window._on_delete()

    assert store.get(item.memory_id) is None
    assert store.embedded_ids(SPACE) == set()
    assert window.memory_list.count() == 0
    assert "不可恢复" in window.message_label.text()


def test_a_cancelled_delete_changes_nothing(qapp: Any, store: MemoryStore) -> None:
    item = store.add(make_item("别删我。"))

    window = make_window(qapp, store)
    pick(window, item.memory_id)
    window._confirm_delete = lambda _item: False
    window._on_delete()

    assert store.get(item.memory_id) is not None
    assert window.memory_list.count() == 1
    assert "已取消" in window.message_label.text()


# ── 线索召回的结果归属 ──────────────────────────────────────────────────────


def test_the_probe_result_is_only_shown_for_its_own_memory(qapp: Any, store: MemoryStore) -> None:
    """**按 id 认领结果，不按"当前选中项"。**

    线索召回在后台跑，几百毫秒里用户完全可能已经点了别的行。把结果贴到那时选中的
    条目上，就是把 A 的邻居显示在 B 的详情里——而且看不出哪里不对。
    """
    from memcore.audit import RecallLedger

    focus = store.add(make_item("探针指的那条。"))
    other = store.add(make_item("另一条。"))

    window = make_window(qapp, store)
    pick(window, focus.memory_id)
    window._probe_ledger = RecallLedger(query_preview="探针指的那条。", scanned=2)
    window._probe_focus_id = focus.memory_id

    pick(window, other.memory_id)
    assert "线索召回" not in window.detail_view.toPlainText()

    pick(window, focus.memory_id)
    assert "线索召回" in window.detail_view.toPlainText()
    assert other.memory_id not in window.detail_view.toPlainText().split("── 线索召回 ──")[1]


# ── 刷新 ────────────────────────────────────────────────────────────────────


def test_the_refresh_button_picks_up_what_was_written_meanwhile(
    qapp: Any, store: MemoryStore
) -> None:
    """**列表不自动跟随**，所以必须有一个明确的刷新入口。

    记忆是每轮实时写的，而这个窗口可以一直开着：一边聊一边看，没有刷新按钮的话
    看到的永远是打开那一刻的快照——而"列表里没有"会被读成"库里没有"，两件事该做
    的下一步完全相反。也没有别的入口能兜住：搜索框清空后再敲一次会重载，但那是
    个没人猜得到的副作用，而**重开窗口不会重载**（``_on_open_memory`` 复用的是
    同一个窗口，``show()`` 不触发载入）。
    """
    earlier = store.add(make_item("打开窗口时就有的。"))
    window = make_window(qapp, store)
    assert row_ids(window) == [earlier.memory_id]

    later = store.add(make_item("窗口开着的时候写进去的。"))
    assert row_ids(window) == [earlier.memory_id]  # 不自动跟随

    window.refresh_button.click()

    # 最新在前——刷新走的是与打开时同一个 `reload()`，而不是一条"只把新条目追加
    # 进去"的旁路：排序、筛选、条数上限三件事会有两套口径，迟早对不上。
    assert row_ids(window) == [later.memory_id, earlier.memory_id]


# ── 人设那两条锚 ────────────────────────────────────────────────────────────
#
# 库里故意有两条 ``PERSONA_ANCHOR``（见 memory/persona.py）。界面上它们的差别只有
# 一处：**内置那条不可改不可删**，用户那条随便改。判据住在核心腿
# （``read_only_reason``），这里验的是界面把它当回事了——置灰、并且说得出为什么。


def test_the_builtin_persona_cannot_be_edited_or_deleted(qapp: Any, store: MemoryStore) -> None:
    """内置那条人设连改和删都不行，**而且两个按钮都要说得出为什么**。

    它是"任何模型读到的身份完全一样"那句话的落点——改掉它就等于改掉那个前提。
    一个灰了却不说理由的按钮，用户会读成"程序坏了"或者"这条记忆有古怪"。
    """
    from biosnn_memterm.memory.persona import ensure_builtin_anchor

    builtin = ensure_builtin_anchor(store)
    assert builtin is not None
    window = make_window(qapp, store)
    pick(window, builtin.memory_id)

    assert not window.edit_button.isEnabled()
    assert not window.delete_button.isEnabled()
    assert window.edit_button.toolTip(), "灰了却不说为什么"
    assert window.edit_button.toolTip() == window.delete_button.toolTip(), (
        "两个按钮被同一条规则挡住，理由也该是同一句"
    )


def test_the_user_persona_is_editable_and_deletable(qapp: Any, store: MemoryStore) -> None:
    """极性：受保护的**只有我们自己写的那个 ref**。

    反过来写（"但凡是人设锚就锁死"）会拒绝用户改自己写下的东西——那比误放开一条
    更糟：用户没有任何办法修正它，除非去动库文件。
    """
    from biosnn_memterm.memory.persona import set_user_persona

    written = set_user_persona(store, "说话简短一点。")
    window = make_window(qapp, store)
    pick(window, written.memory_id)

    assert window.edit_button.isEnabled()
    assert window.delete_button.isEnabled()
    assert window.edit_button.toolTip() == ""


def test_an_anchor_with_no_source_is_treated_as_the_users(qapp: Any, store: MemoryStore) -> None:
    """来源不明的按用户所有理解，**不锁死**。将来多一种角色时它就是那个人的。"""
    item = store.add(
        make_item("说话用文言文。", kind=MemoryKind.PERSONA_ANCHOR, pinned=True, source_ref="")
    )
    window = make_window(qapp, store)
    pick(window, item.memory_id)

    assert window.edit_button.isEnabled()
    assert window.delete_button.isEnabled()


def test_no_selection_says_why_the_buttons_are_grey(qapp: Any, store: MemoryStore) -> None:
    """一条没选中时，改/删/删探的 tooltip 也要有话——它们以前灰着什么都不说。"""
    store.add(make_item("随便一条。"))
    window = make_window(qapp, store)

    window.memory_list.setCurrentRow(-1)

    assert not window.edit_button.isEnabled()
    assert "选一条" in window.edit_button.toolTip()
    assert "选一条" in window.delete_button.toolTip()


def test_the_builtin_persona_is_not_reachable_from_the_delete_button(
    qapp: Any, store: MemoryStore
) -> None:
    """按钮已灰，但**规则与按钮状态分居两处**（判据在核心腿里），所以再调一次
    ``_on_delete``：它必须什么都不删，并把理由说出来。"""
    from biosnn_memterm.memory.persona import ensure_builtin_anchor

    builtin = ensure_builtin_anchor(store)
    assert builtin is not None
    window = make_window(qapp, store)
    pick(window, builtin.memory_id)

    window._on_delete()

    assert store.get(builtin.memory_id) is not None
    assert window.message_label.text(), "拦下了却一声不吭"


def test_the_delete_warning_for_the_user_persona_promises_a_second_ask(
    qapp: Any, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """裁决 4 的落点：**删掉用户那条 = 下次启动重新问一次**，不是"恢复成默认性格"。

    旧文案写的是"下次启动会重新写入一条默认的"，那会让人以为删掉之后人格被重置成
    某个默认性格——而实际上卡片会回来问他。这两句话的差别是"系统替他决定"与
    "他自己决定"，而这一整版功能就是为了把那个决定交回去。
    """
    from PySide6.QtWidgets import QMessageBox

    from biosnn_memterm.memory.persona import set_user_persona
    from biosnn_memterm.ui import memory_window as memory_window_module

    written = set_user_persona(store, "说话简短一点。")
    window = make_window(qapp, store)
    pick(window, written.memory_id)

    seen: list[str] = []

    def fake_question(_parent: Any, _title: str, text: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(memory_window_module.QMessageBox, "question", fake_question)
    window._on_delete()

    assert len(seen) == 2, "彻底删除必须问两次"
    assert "下次启动会重新问你一次" in seen[1]
    assert "内置的那条还在" in seen[1]
    assert "恢复成什么默认的性格" in seen[1]
    assert store.get(written.memory_id) is None
