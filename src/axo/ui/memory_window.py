"""记忆浏览器：一个独立窗口，用来查看、编辑、遗忘、删除记忆。

**为什么是独立窗口**（用户裁决，2026-10-10）：主窗口已经是三格 splitter（会话 /
对话 / 注入账本），再加一格会把对话区挤到不可读。而记忆浏览是"要核对时才打开"
的动作，与逐字稿的阅读节奏不同频。

这个控件**只做两件事**：把 :mod:`axo.memory.browse` 拼好的文本贴到
控件上，以及把用户的点击翻译成对 :class:`~memcore.store.MemoryStore` 的调用。
所有判断句（"这条为什么没被注入"、"缺向量要不要紧"、"钉住意味着什么"）都在纯模块
里，因此可以断言；这里只剩布局与信号。

**界面线程上同步做的三件事**（载入 / 搜索 / 单条编辑或遗忘）：它们都是单表扫描或
单行写加一次轻量 SDR 编码，毫秒级——与 :class:`~axo.memory.writer.
MemoryWriter` 同步写的既有决定一致。唯一含网络调用的动作是「以它为线索召回一次」，
它走 :class:`~axo.ui.memory_worker.MemoryProbeWorker`。

**账本靠推送，不靠传引用。** :meth:`set_recall_ledger` 由主窗口在每轮装配之后调用。
主窗口那个 ``_recall_ledger`` 每轮都被**重新赋值**，窗口若捕获一次引用，下一轮就
指向旧值——而且不报错。这正是本项目最怕的那类静默失效。
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from memcore.audit import RecallLedger
from memcore.errors import StoreError
from memcore.graph import GraphIndex, graph_heat
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, MemorySummary, Tier

from ..llm.ollama import OllamaEmbeddingBackend
from ..memory.browse import (
    MemoryFilter,
    MemoryStatus,
    NeighborRow,
    association_header,
    association_rows,
    filter_summaries,
    kind_label,
    probe_position,
    render_heat_panel,
    render_memory_detail,
    summary_label,
    tier_label,
)
from ..memory.consolidation import last_run
from ..memory.edit import can_encode, edited_content
from ..memory.forget import can_soft_forget, restored, soft_forgotten
from ..memory.persona import BUILTIN_REF, USER_REF, read_only_reason
from .glass import GlassHost, fade_window
from .memory_worker import MemoryProbeWorker
from .thread_tracking import track, wait_for_all
from .window_place import place_window

__all__ = ["DEFAULT_LIST_LIMIT", "MemoryWindow"]

DEFAULT_LIST_LIMIT = 2000
"""一次列多少条。

命中这个上限时界面**必须说出来**（见 :meth:`MemoryWindow._update_count`）：把
"只显示了最近 2000 条"读成"库里只有 2000 条"，是这一层最容易造成的误读，而
用户据此做的动作（"它不记得了"）与真相（"往下翻/搜一下就有了"）完全相反。
"""

_KIND_ORDER: tuple[MemoryKind, ...] = (
    MemoryKind.PERSONA_ANCHOR,
    MemoryKind.FACT,
    MemoryKind.PREFERENCE,
    MemoryKind.DECISION,
    MemoryKind.CONCLUSION,
    MemoryKind.OPEN_QUESTION,
    MemoryKind.TOOL_RESULT,
)
"""下拉框里的类别顺序。人格锚放最前：它是这个界面上最该被找到的那一条。"""

_TIER_ORDER: tuple[Tier, ...] = (
    Tier.WORKING,
    Tier.RECENT,
    Tier.LONGTERM,
    Tier.ARCHIVE,
    Tier.COLD,
)

#: 邻居行上「这条还在不在库里」那一位的角色。与 ``UserRole``（memory_id）分开：
#: 挤进同一个角色里会让 id 与布尔值互相覆盖，而症状是"点谁都不动"。
_PRESENT_ROLE = Qt.ItemDataRole.UserRole + 1

_STATUS_ORDER: tuple[MemoryStatus, ...] = (
    MemoryStatus.ALL,
    MemoryStatus.ACTIVE,
    MemoryStatus.FORGOTTEN,
    MemoryStatus.PINNED,
)


class MemoryWindow(QMainWindow):
    """记忆浏览器窗口。

    Args:
        memory: 记忆库。**注入而不是自己开一个**——沿用 ``MainWindow`` 的注入点
            约定，测试因此可以在 ``tmp_path`` 上造一个真库跑完整的读写路径。
    """

    def __init__(self, *, memory: MemoryStore, accent: str = "", motion: bool = True) -> None:
        super().__init__()
        self._accent = accent or "#6b8afd"
        self._motion = bool(motion)
        self.setWindowTitle("记忆")
        # **与主窗口用同一个函数**，不是各写一遍 `resize(940, 640)`。写死的尺寸在
        # 1366×768 的笔记本上（加上任务栏）就已经顶到屏幕边，在更小的屏上则直接
        # 超出——而这是一个**子窗口**，超出的部分用户连拖都拖不到。
        place_window(self)

        self._memory = memory
        self._spaces = memory.spaces()
        self._embedding: OllamaEmbeddingBackend | None = None
        self._current_session_id = ""
        self._ledger: RecallLedger | None = None
        self._summaries: list[MemorySummary] = []

        #: ``reload()`` 正在跑。**这是防重入的闸门，不是优化。** 见 ``reload`` 的
        #: 文档串：列表清空的过程中 Qt 会挪动当前行并发出选中变化，处理它的
        #: ``_refresh_detail`` 有可能再调回 ``reload``——而那是一次**嵌在
        #: ``clear()`` 半途的第二次 ``clear()``**，会踩到已析构的条目。
        self._reloading = False

        #: 线索召回的结果，以及它属于哪一条。**按 id 记住而不是"当前选中项"**：
        #: 召回在后台跑，几百毫秒里用户完全可能已经点了别的行。把结果贴到那时
        #: 选中的条目上，就是把 A 的邻居显示在 B 的详情里。
        self._probe_ledger: RecallLedger | None = None
        self._probe_focus_id = ""

        #: 关联图（由主窗口注入）。``None`` = 还没建。**窗口自己不建图也不读盘**：
        #: 建图在 5 万条上要几分钟，是后台线程的事，而这个窗口只负责显示。
        self._graph: GraphIndex | None = None
        #: 这份图**过期与否的三态**：``True`` 确定过期、``False`` 刚比对过、
        #: ``None`` **没比对过**（库太大）。**不在这里算**——``is_stale`` 是一次全量
        #: 重编码（5 万条约 63 秒），主窗口在建/载入那一刻算一次随 ``set_graph``
        #: 传进来。窗口若自己去算，界面会在切到某一条时卡住几秒。
        self._graph_stale: bool | None = False

        #: 还在跑的线程的权威清单，收尾时按它来。``thread_tracking.track`` 要求
        #: 接收者有这么一张表（见它的 Args）。
        self._live_threads: list[QThread] = []

        self._build_ui()
        self._wire()
        self._glass = GlassHost(self, accent=self._accent, animated=self._motion)
        self.reload()

    # ── 外部接线 ────────────────────────────────────────────────────────────

    def set_glass(self, accent: str, *, motion: bool) -> None:
        """主题/动效变了：换极光颜色，并决定还要不要动画。"""
        self._accent = accent
        self._motion = bool(motion)
        glass = getattr(self, "_glass", None)
        if glass is not None:
            glass.set_theme(accent, animated=motion)

    def showEvent(self, event: Any) -> None:  # noqa: N802 — Qt 命名
        super().showEvent(event)
        if self._motion:
            fade_window(self)

    def set_embedding(self, embedding: OllamaEmbeddingBackend | None) -> None:
        """接上当前的嵌入后端（``None`` = 这一轮没有语义通道）。

        只用来标"这条记忆在当前空间里有没有向量"，**不在这里跑嵌入**：那个动作
        是探针按钮的事，而它有自己的线程。
        """
        self._embedding = embedding
        self._spaces = self._memory.spaces()
        self._refresh_detail()

    def set_current_session(self, session_id: str) -> None:
        """告诉窗口"用户正在看的是哪个会话"。

        详情里要用它区分两条早退路径之一：属于当前会话的记忆会被召回当作"逐字稿
        里已经有了的内容"排除掉。不知道是哪个会话时传空串——那时窗口会说"不在
        本轮账本里"并**列出可能的原因**，而不是猜一个。
        """
        self._current_session_id = session_id
        self._refresh_detail()

    def set_recall_ledger(self, ledger: RecallLedger | None) -> None:
        """接上本轮账本。``None`` 表示还没装配过（没有会话，或这一轮还没发请求）。"""
        self._ledger = ledger
        self._refresh_detail()

    def set_graph(self, graph: GraphIndex | None, *, stale: bool | None = False) -> None:
        """接上关联图（``None`` = 还没有）。

        照 :meth:`set_embedding` / :meth:`set_recall_ledger` 的形状注入。``stale``
        是**三态**、且由**调用方**算好传进来（理由见 ``_graph_stale`` 那段）：
        ``None`` 表示"没跟库比对过"，顶栏据此说另一句话。

        **这里不做"图旧了就重建"这种体贴。** 重建要在后台线程里跑几分钟，而这个
        方法是界面线程上的一个 setter——顺手重建会让打开记忆窗口这个动作卡住。
        旧了就说出来（顶栏那一句），要不要重建由用户按「设置 → 记忆」里那个按钮。
        """
        self._graph = graph
        self._graph_stale = stale
        self._refresh_association()
        self._refresh_heat()

    # ── 构建界面 ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        outer.addWidget(self._build_filter_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self._build_list_pane())
        splitter.addWidget(self._build_detail_pane())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 560])
        outer.addWidget(splitter, 1)

    def _build_filter_bar(self) -> QWidget:
        bar = QFrame(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.search_edit = QLineEdit(bar)
        self.search_edit.setPlaceholderText("在全部记忆里搜索正文…")
        self.search_edit.setClearButtonEnabled(True)
        row.addWidget(self.search_edit, 1)

        row.addWidget(QLabel("层级", bar))
        self.tier_combo = QComboBox(bar)
        self.tier_combo.addItem("全部", None)
        for tier in _TIER_ORDER:
            self.tier_combo.addItem(tier_label(tier), tier.value)
        row.addWidget(self.tier_combo)

        row.addWidget(QLabel("类别", bar))
        self.kind_combo = QComboBox(bar)
        self.kind_combo.addItem("全部", None)
        for kind in _KIND_ORDER:
            self.kind_combo.addItem(kind_label(kind), kind.value)
        row.addWidget(self.kind_combo)

        row.addWidget(QLabel("状态", bar))
        self.status_combo = QComboBox(bar)
        for status in _STATUS_ORDER:
            self.status_combo.addItem(status.label, status.value)
        row.addWidget(self.status_combo)

        # **列表不自动跟随。** 记忆是每轮实时写的，而这个窗口可以一直开着——
        # 没有这个按钮的话，一边聊一边看，看到的永远是打开那一刻的快照，而
        # 「列表里没有」会被读成「库里没有」。不给自动刷新是因为 reload() 会
        # clear() 列表，那会擦掉用户正在看的详情与刚做过的动作的反馈（与
        # _on_edit 里那条同源）。刷新是一个明确的用户动作，冲突交给他判断。
        self.refresh_button = QPushButton("刷新", bar)
        self.refresh_button.setToolTip("重新载入列表（记忆是每轮实时写的，这里不会自动跟随）")
        row.addWidget(self.refresh_button)
        return bar

    def _build_list_pane(self) -> QWidget:
        pane = QFrame(self)
        pane.setObjectName("memoryListPane")
        pane.setMinimumWidth(280)
        column = QVBoxLayout(pane)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(6)

        self.count_label = QLabel("", pane)
        self.count_label.setObjectName("memoryCount")
        self.count_label.setWordWrap(True)
        column.addWidget(self.count_label)

        self.memory_list = QListWidget(pane)
        self.memory_list.setObjectName("memoryList")
        self.memory_list.setWordWrap(True)
        # 正文里有长 URL / 长英文串时，自动换行仍会留下一条横向滚动条；列表本来就是
        # 纵向浏览的，横向那条只会挤掉一行高度。
        self.memory_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        column.addWidget(self.memory_list, 1)
        return pane

    def _build_action_bar(self) -> QWidget:
        bar = QFrame(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.edit_button = QPushButton("编辑正文", bar)
        self.forget_button = QPushButton("遗忘", bar)
        self.restore_button = QPushButton("恢复", bar)
        self.delete_button = QPushButton("彻底删除", bar)
        # 与主窗口的「删除选中会话」同一支 danger：四个按钮里只有它不可撤销。
        self.delete_button.setProperty("kind", "danger")
        row.addWidget(self.edit_button)
        row.addWidget(self.forget_button)
        row.addWidget(self.restore_button)
        row.addWidget(self.delete_button)
        row.addStretch(1)

        self.probe_button = QPushButton("以它为线索召回一次", bar)
        self.probe_button.setToolTip(
            "用这条记忆的正文当查询跑一次召回，看看它附近还有谁。"
            "用的是探索参数（阈值更低、条数更多），结果里会写明这一点。"
        )
        row.addWidget(self.probe_button)
        return bar

    def _build_detail_pane(self) -> QWidget:
        pane = QFrame(self)
        column = QVBoxLayout(pane)
        column.setContentsMargins(6, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(self._build_action_bar())

        # 动作的反馈（"正文不能为空"、"已丢弃旧向量"）**必须有一个说出口**。
        # 用弹窗会打断连续操作（编辑一条、再编辑下一条时每次都要点确定），
        # 而什么都不说会让失败看起来像"点了没反应"。
        self.message_label = QLabel("", pane)
        self.message_label.setObjectName("memoryMessage")
        self.message_label.setWordWrap(True)
        column.addWidget(self.message_label)

        self.detail_view = QTextBrowser(pane)
        self.detail_view.setObjectName("memoryDetail")
        self.detail_view.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self.detail_view.setPlainText("从左边选一条记忆，这里显示它的全部内容、来源与本轮归因。")
        column.addWidget(self.detail_view, 1)

        # 联想面板：**局部邻居列表**，随选中变化。
        #
        # 为什么是列表而不是图：仓里只有 QPainter 自绘的先例，没有 networkx /
        # matplotlib / QGraphicsView，而一张力导向图要么引进依赖、要么自己写布局。
        # 更要紧的是**它落在纯层**（``browse.association_rows`` / ``association_header``）
        # ——"没有图的时候该说什么"这种判断句只有放在那里才测得动。
        #
        # 顶栏单独一个 QLabel：它要说的话随状态变（没图 / 不在图里 / 图过期），
        # 而列表里装的是**真数据**（边权、记忆权重）。两者混在一个控件里的话，
        # "这一句里的数字是不是编的"就又要靠人眼看。
        self.assoc_label = QLabel("", pane)
        self.assoc_label.setObjectName("memoryAssocNote")
        self.assoc_label.setWordWrap(True)
        column.addWidget(self.assoc_label)

        self.assoc_view = QListWidget(pane)
        self.assoc_view.setObjectName("memoryNeighbors")
        self.assoc_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 固定高度：邻居最多 top_m（8）条，让它跟着内容伸缩会把上面的详情挤得忽高
        # 忽低。滚轮够看剩下的。
        self.assoc_view.setMaximumHeight(160)
        column.addWidget(self.assoc_view, 0)

        # 「SNN 热度」，**常驻显示**——不随选中变化。它是库一级的摘要，不是某一条
        # 记忆的属性。**判断句在纯层**（``render_heat_panel``）：这里只把三个数
        # 递过去，两态文案与"什么时候才敢写数字"都在那一侧，因此可以断言。
        self.pending_view = QTextBrowser(pane)
        self.pending_view.setObjectName("memoryPending")
        self.pending_view.setMaximumHeight(170)
        self._refresh_heat()
        column.addWidget(self.pending_view, 0)
        self._refresh_association()
        return pane

    def _wire(self) -> None:
        self.edit_button.clicked.connect(self._on_edit)
        self.forget_button.clicked.connect(self._on_forget)
        self.restore_button.clicked.connect(self._on_restore)
        self.delete_button.clicked.connect(self._on_delete)
        self.probe_button.clicked.connect(self._on_probe)
        self.refresh_button.clicked.connect(self.reload)
        self.search_edit.textChanged.connect(lambda _text: self.reload())
        self.tier_combo.currentIndexChanged.connect(lambda _index: self.reload())
        self.kind_combo.currentIndexChanged.connect(lambda _index: self.reload())
        self.status_combo.currentIndexChanged.connect(lambda _index: self.reload())
        self.memory_list.currentItemChanged.connect(
            lambda current, _prev: self._on_selected(current)
        )
        # **点一下就跳过去**（不双击）：邻居列表是个索引，用户的意图是"看那一条"。
        # ``currentRowChanged`` 不行——跳转本身就是靠 setCurrentRow 实现的，用它
        # 会把「点邻居」和「列表自己换行」搅在一起。
        self.assoc_view.itemClicked.connect(self._on_neighbor_clicked)

    # ── 载入与筛选 ──────────────────────────────────────────────────────────

    def _current_filter(self) -> MemoryFilter:
        """把三个下拉框读成一个 :class:`MemoryFilter`。

        **必须显式转回枚举。** 下拉框里存的是 ``tier.value`` 这样的**字符串**，
        经 QVariant 回来仍是字符串——而 :func:`~axo.memory.browse.
        filter_summaries` 用的是 ``summary.tier is not criteria.tier``。字符串
        ``"cold"`` 与 ``Tier.COLD`` 相等却不相同，于是"筛一个层级"会得到一个
        **空列表**，看起来像"这一层没有记忆"。（这与 §4.10 里 ``QSettings`` 读
        bool 得到字符串是同一族问题：跨过 Qt 的类型系统，类型就没了。）
        """
        tier = self.tier_combo.currentData()
        kind = self.kind_combo.currentData()
        status = self.status_combo.currentData()
        return MemoryFilter(
            query=self.search_edit.text(),
            tier=Tier(tier) if tier else None,
            kind=MemoryKind(kind) if kind else None,
            status=MemoryStatus(status) if status else MemoryStatus.ALL,
        )

    def reload(self) -> None:
        """重新载入列表。筛选条件变了、或者写动作做完了都调它。

        **查询走 SQL，其余三维在内存里筛。** 不分两套取数逻辑：SQL 这一趟已经把
        ``query``/``tier``/``kind`` 三个条件带上了（那三个是列，能走索引），而
        ``status`` 不是列（它是 ``tier`` 与 ``pinned`` 的组合），所以最后一层一律
        交给 :func:`~axo.memory.browse.filter_summaries`——同一个函数，
        与界面以外的地方口径一致。多筛一遍是幂等的。

        **下面两道闸门都不是优化，是防崩。**

        1. ``_reloading`` 防重入。``memory_list.clear()`` 的过程中 Qt 会挪动当前行
           并发 ``currentItemChanged``——实测：选中第 0 行、而第 1 行刚**从库里**
           被删掉时，当前行会被挪到第 1 行。于是 ``_refresh_detail`` 查到
           ``item is None``，走它那条"另一处刚把它删了"的分支**再调一次
           ``reload``**。那是嵌在 ``clear()`` 半途的第二次 ``clear()``：Qt 的条目
           析构不可重入，外层 clear 接着在已释放的条目上跑，进程以
           ``0xc0000374``（堆损坏／访问违例）直接死掉，连 pytest 都收不回输出。
           闸门让那次重入变成空操作。

           触发条件很窄（**已选中一行 + 库里少了一条别的 + 随后 reload**），但它是
           真实路径：后台的记忆同步线程就能在窗口开着的时候删掉一条。
        2. ``blockSignals`` 圈住清空与重填。半建好的列表不该被任何人观察——期间的
           选中变化是**中间态**（行号与库里的现状对不上），让面板按它重画只会显示
           一条已经不在库里的记忆。信号解除后 ``_update_count`` 会把选中项按 id
           还原，那一次才是真的。
        """
        if self._reloading:
            return
        self._reloading = True
        try:
            flt = self._current_filter()
            # 查询非空时走全库：列表那一页只是"最近 N 条"，而用户一旦开始搜，他要找
            # 的很可能是更早的那条。（浏览是"找那一条"，不是"找相似的那几条"。）
            fetched = self._memory.summaries(
                query=flt.query,
                tier=flt.tier,
                kind=flt.kind,
                limit=DEFAULT_LIST_LIMIT,
            )
            self._summaries = filter_summaries(fetched, flt)

            selected = self._selected_id()
            # 存下旧状态再还原，**不要写死 False**：调用方若本来就关着信号，
            # 这里硬开回去会替它做决定。
            was_blocked = self.memory_list.blockSignals(True)
            try:
                self.memory_list.clear()
                for summary in self._summaries:
                    row = QListWidgetItem(summary_label(summary), self.memory_list)
                    row.setData(Qt.ItemDataRole.UserRole, summary.memory_id)
            finally:
                self.memory_list.blockSignals(was_blocked)
            self._update_count(len(fetched), selected)
            # 热度那一块也在 reload 里重画：它读的是库里的 ``reinforce_count``，
            # 而"重新载入"正是用户要求"按现状再说一遍"的动作。放在最后，与
            # ``_update_count`` 一样属于"列表已经填好之后才说得出口"的那些话。
            self._refresh_heat()
        finally:
            self._reloading = False

    def _update_count(self, fetched: int, previously_selected: str) -> None:
        total = self._memory.count()
        if fetched >= DEFAULT_LIST_LIMIT and not self._current_filter().query:
            # 命中上限必须**说出来**。悄悄截断的话，"列表里没有"会被读成"库里
            # 没有"，而这两件事该做的下一步完全相反。
            self.count_label.setText(
                f"显示最近 {DEFAULT_LIST_LIMIT} 条（库里共 {total} 条）——"
                f"输入搜索以在全部条目中查找。"
            )
        else:
            self.count_label.setText(f"{len(self._summaries)} / {total} 条")
        self._restore_selection(previously_selected)

    def _row_of(self, memory_id: str) -> int | None:
        """这个 id 在列表里是第几行；不在列表里返回 ``None``。

        **按 id 找，不按行号**：写动作（编辑、遗忘）会让 ``memory_id DESC`` 的排序
        基本不变，但一旦真的变了，按行号就是**另一条**记忆——而用户此时正准备对
        "刚才那条"做下一步动作。跳转与还原共用这一条，所以两处的口径不会分叉。
        """
        for index in range(self.memory_list.count()):
            if self.memory_list.item(index).data(Qt.ItemDataRole.UserRole) == memory_id:
                return index
        return None

    def _restore_selection(self, memory_id: str) -> None:
        """尽量把选中项还原到同一个 id 上。"""
        if not memory_id:
            self._refresh_detail()
            return
        row = self._row_of(memory_id)
        if row is not None:
            self.memory_list.setCurrentRow(row)
            return
        # 它被筛掉了（比如刚软遗忘、而筛选条件是「在用」）——如实清空详情，不
        # 让面板继续显示一条已经不在列表里的记忆。
        self._refresh_detail()

    def _selected_id(self) -> str:
        item = self.memory_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _on_selected(self, _current: QListWidgetItem | None) -> None:
        # 换了一条，上一条动作的反馈就过期了。留着它会让"正文不能为空"看起来
        # 是在说刚点开的这一条。
        self._say("")
        self._refresh_detail()
        self._refresh_association()
        self._refresh_heat()

    # ── 详情 ────────────────────────────────────────────────────────────────

    def _current_item(self) -> MemoryItem | None:
        memory_id = self._selected_id()
        return None if not memory_id else self._memory.get(memory_id)

    def _say(self, text: str) -> None:
        """把一句话放到按钮下面那行。**这是动作反馈的唯一出口。**"""
        self.message_label.setText(text)

    def _refresh_detail(self) -> None:
        memory_id = self._selected_id()
        if not memory_id:
            self.detail_view.setPlainText(
                "从左边选一条记忆，这里显示它的全部内容、来源与本轮归因。"
            )
            self._update_actions(None)
            return
        item = self._memory.get(memory_id)
        if item is None:
            # 并发下可能：另一处刚把它删了。**说出来**，不要显示一条不存在的记忆，
            # 也不要静默地退回空白（用户会以为是自己点错了）。
            #
            # **顺序不能反**：``reload()`` 会在末尾调回这里一次（列表被清空，
            # 选中项也就没了），那一趟会把详情写成"请选一条记忆"的白板。先把
            # 列表对齐到库里的现状，再把这句话放到最后写。
            self.reload()
            self.detail_view.setPlainText(
                f"记忆 {memory_id} 已经不在了（可能刚被删除）。列表已刷新。"
            )
            return
        text = render_memory_detail(
            item,
            now=time.time(),
            spaces=self._spaces,
            embedded=self._embedded_by_space(),
            ledger=self._ledger,
            current_session_id=self._current_session_id,
        )
        if self._probe_ledger is not None and self._probe_focus_id == memory_id:
            text = f"{text}\n\n── 线索召回 ──\n{probe_position(self._probe_ledger, memory_id)}"
        self.detail_view.setPlainText(text)
        self._update_actions(item)

    # ── SNN 热度 ────────────────────────────────────────────────────────────

    def notify_consolidated(self) -> None:
        """**巩固刚跑完一轮**（主窗口推过来）。热度那一块要说的数变了。

        为什么要一个专门的方法，而不是让主窗口顺手调 :meth:`_refresh_heat`：那一个
        是私有的，而这是这个窗口对外的第五个注入点（前四个都是 ``set_*``）。名字不叫
        ``set_*`` 是因为它不携带状态——它是一句"重新读一遍"，读的东西（库里的
        ``reinforce_count``、盘上的 ``last.json``）都不由主窗口交过来。
        """
        self._refresh_heat()

    def _refresh_heat(self) -> None:
        """按库里与盘上的现状重画热度那一块。**所有判断句都在纯层。**

        三个数各有各的来源，且**都允许拿不到**：条数与总数走一条聚合 SQL（空库是
        ``(0, 0)``），时间走 ``last.json``（没有、坏了都是 ``None``）。拿不到就说
        "没有"，不补零、不编时间——见 ``render_heat_panel`` 的两态。
        """
        reinforced, total = self._memory.reinforce_stats()
        last = last_run()
        heat = None if self._graph is None or len(self._graph) == 0 else graph_heat(self._graph)
        self.pending_view.setPlainText(
            render_heat_panel(
                heat=heat,
                selected_id=self._selected_id(),
                graph_built_at=None if self._graph is None else self._graph.built_at,
                stale=self._graph_stale,
                reinforced_memories=reinforced,
                total_reinforcements=total,
                last_run_at=None if last is None else last[0],
            )
        )

    # ── 联想面板 ────────────────────────────────────────────────────────────

    def _refresh_association(self) -> None:
        """按当前选中项重画邻居面板。**所有判断句都在纯层。**"""
        memory_id = self._selected_id()
        graph = self._graph
        # 空图（库里一条都没有时 ``build`` 会给一张 0 节点的图）当作"没有图"：
        # 这时说"这条记忆还不在关联图里"是错的——它不是"还不在"，是**没有图**。
        has_graph = graph is not None and len(graph) > 0

        self.assoc_view.clear()
        if not memory_id:
            # **没有选中项是第六种状态**，纯层那五个分支没有一个说得对（它们都在
            # 假定"手上有那么一条"）。这与详情面板那句占位是同一个形状，所以照
            # 它的写法来。
            self.assoc_label.setText("先从左边的列表里选一条记忆，这里显示它的邻居。")
            return

        rows: tuple[NeighborRow, ...] = ()
        if has_graph and graph is not None:
            rows = association_rows(memory_id, graph=graph, lookup=self._memory.get)
            for row in rows:
                entry = QListWidgetItem(row.label, self.assoc_view)
                entry.setData(Qt.ItemDataRole.UserRole, row.memory_id)
                # **可点与否由这一位决定**，不看 label 里写了什么。靠文案判断
                # （"含'已删除'"）会在下次改措辞时静默失效。
                entry.setData(_PRESENT_ROLE, row.present)

        self.assoc_label.setText(
            association_header(
                has_graph=has_graph,
                focus_in_graph=has_graph and graph is not None and graph.index_of(memory_id) >= 0,
                rows=rows,
                built_at=0.0 if graph is None else graph.built_at,
                stale=self._graph_stale,
            )
        )

    def _on_neighbor_clicked(self, item: QListWidgetItem) -> None:
        """点一条邻居：跳到列表里的那一行。**跳不过去就说为什么。**

        照 :meth:`_restore_selection` 的诚实口径：找不到时**不静默改筛选条件**
        （清掉搜索框里的字、把状态下拉框拨回「全部」）——那会悄悄换掉用户正在看的
        那一批，而他不会知道。说不出来就是说不出来。
        """
        memory_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not memory_id:
            return
        if not item.data(_PRESENT_ROLE):
            # 边指向一条已删除的记忆。这一条**只有文案能说清楚**，所以它没有
            # 置灰——置灰的控件连 tooltip 都不给看（同 _update_actions 那条纪律：
            # 置灰必须带理由，而这里的理由只能点一下才说得出来）。
            self._say("这条邻居已经不在库里了（图是旧的），跳不过去。")
            return
        row = self._row_of(memory_id)
        if row is None:
            self._say("那条不在当前列表里（被筛选条件排除，或在最近 N 条之外）。")
            return
        self.memory_list.setCurrentRow(row)

    def _update_actions(self, item: MemoryItem | None) -> None:
        """按选中项决定四个按钮能不能按，并把**不能按的理由**写在 tooltip 上。

        **置灰必须带理由。** 一个点不动的按钮不说明任何事：用户看到"遗忘"是灰的，
        会以为是程序坏了，或者以为这条记忆有什么特殊之处。钉住的条目尤其需要这句
        ——为什么不能遗忘，是这个界面上最不直观的一条规则。

        **内置那条人设连改和删也一起灰掉**（:func:`~axo.memory.persona.read_only_reason`）：
        它是"任何模型读到的身份完全一样"那句话的落点，改掉它就等于改掉前提。
        理由文案来自核心腿，这里只负责显示——判据只有一处。
        """
        has_item = item is not None
        #: 没选中任何一条时的统一说法。以前这三个按钮灰着却什么都不说。
        nothing_chosen = "先从左边的列表里选一条记忆。"

        def reason_for(blocked: str | None) -> str:
            if blocked is not None:
                return blocked
            return "" if has_item else nothing_chosen

        locked = None if item is None else read_only_reason(item)
        self.edit_button.setEnabled(has_item and locked is None)
        self.edit_button.setToolTip(reason_for(locked))
        self.delete_button.setEnabled(has_item and locked is None)
        self.delete_button.setToolTip(reason_for(locked))
        self.probe_button.setEnabled(has_item)
        self.probe_button.setToolTip(reason_for(None))

        forget_reason = None if item is None else can_soft_forget(item)
        self.forget_button.setEnabled(has_item and forget_reason is None)
        self.forget_button.setToolTip("" if forget_reason is None else forget_reason)

        self.restore_button.setEnabled(item is not None and item.tier is Tier.COLD)
        self.restore_button.setToolTip(
            "" if (item is not None and item.tier is Tier.COLD) else "它没有被遗忘，无需恢复。"
        )

    def _embedded_by_space(self) -> dict[str, set[str]]:
        """逐空间问一次「哪些记忆在这个空间里有向量」。

        **不是一次查询**：向量住在 ``embeddings`` 表里、按 ``space_id`` 分区，
        而换过嵌入模型之后库里会有多个空间。一条记忆在新空间里有向量、在旧空间
        里也有（或没有）是常态，界面必须逐空间如实标注。
        """
        return {space.space_id: self._memory.embedded_ids(space.space_id) for space in self._spaces}

    # ── 写动作 ──────────────────────────────────────────────────────────────
    #
    # 四个都是**界面线程同步**做的：单表扫描或单行写 + 一次轻量 SDR 编码，毫秒级。
    # 与 MemoryWriter 同步写的既有决定一致。唯一含网络调用的「线索召回」在下一条
    # 分节里，它有自己的线程。

    def _on_edit(self) -> None:
        """改正文：**重算词法 SDR** + 丢弃旧语义向量。"""
        item = self._current_item()
        if item is None:
            return
        locked = read_only_reason(item)
        if locked is not None:
            # 正常不可达（按钮已置灰），但按钮状态与规则分居两处，这里再问一次
            # ——理由同 _on_forget：规则住在核心腿里，界面只是照着它置灰。
            self._say(locked)
            return
        text = self._ask_for_content(item)
        if text is None or text == item.content:
            return

        # 先在动手之前问一句。把 SDRCapacityError / ValueError 变成人话，而不是
        # 让用户看到一个回溯——判定与真正拦住它的那一处共用 can_encode，所以信息
        # 里说的理由与实际生效的理由永远是同一条。
        reason = can_encode(text)
        if reason is not None:
            self._say(reason)
            return

        try:
            self._memory.transform(item.memory_id, lambda it: edited_content(it, text))
        except StoreError as exc:
            # 并发下可能：另一处刚把它删了。
            #
            # **先刷新再说话，顺序不能反**（与 :meth:`_refresh_detail` 里那条同源）：
            # ``reload()`` 会 ``clear()`` 列表，而清空当前项会发一条
            # ``currentItemChanged``，槽里把上一条动作的反馈抹掉。反过来写的话，
            # "改不动：…" 会在一瞬间被自己擦掉，用户看到的是一个点不动的按钮。
            self.reload()
            self._say(f"改不动：{exc}")
            return

        # **顺序不能反**（见 memory/edit.py 的模块文档）：先丢向量再改正文的话，
        # 万一 transform 抛了异常，向量就白丢了——而它本来是对的。
        self._memory.drop_embedding(item.memory_id)
        self.reload()
        self._say(
            "已改。词法编码按新正文重算了；旧语义向量已丢弃，回填管线会在后台按"
            "新正文重新算它（不必等网络，也不影响词法与召回）。"
        )

    def _on_forget(self) -> None:
        """软遗忘：``tier=COLD`` 加一张写在 ``meta`` 里的回程票。**不是删除。**"""
        item = self._current_item()
        if item is None:
            return
        reason = can_soft_forget(item)
        if reason is not None:
            # 正常不可达（按钮已置灰），但按钮状态与规则分居两处，这里再问一次
            # ——规则住在核心腿里，界面只是照着它置灰。
            self._say(reason)
            return
        at = time.time()
        try:
            self._memory.transform(item.memory_id, lambda it: soft_forgotten(it, at=at))
        except StoreError as exc:
            self.reload()  # 先刷新再说话，理由同 _on_edit
            self._say(f"忘不掉：{exc}")
            return
        self.reload()
        self._say("已遗忘。它不再参与召回，但仍在列表里（筛「已遗忘」可见），随时可以恢复。")

    def _on_restore(self) -> None:
        item = self._current_item()
        if item is None or item.tier is not Tier.COLD:
            return
        try:
            result = self._memory.transform(item.memory_id, lambda it: restored(it)[0])
        except StoreError as exc:
            self.reload()  # 先刷新再说话，理由同 _on_edit
            self._say(f"恢复不了：{exc}")
            return
        self.reload()
        # 「恢复了但恢复到了哪儿」必须说清楚。回程票读不到时它落在工作层——猜一个
        # 层级然后不说，是这里唯一不能做的事。
        if result.tier is Tier.WORKING:
            self._say("已恢复，但没有记录原层级（这条记忆的 meta 里没有回程票），它现在在工作层。")
        else:
            self._say(f"已恢复到「{tier_label(result.tier)}」。")

    def _on_delete(self) -> None:
        """彻底删除。**不可逆，所以问两次。**"""
        item = self._current_item()
        if item is None:
            return
        locked = read_only_reason(item)
        if locked is not None:
            self._say(locked)  # 正常不可达，理由同 _on_edit
            return
        if not self._confirm_delete(item):
            self._say("已取消，什么都没有删。")
            return
        memory_id = item.memory_id
        try:
            self._memory.delete(memory_id)
        except StoreError as exc:
            self._say(f"删不掉：{exc}")
            return
        if self._probe_focus_id == memory_id:
            self._probe_ledger = None
            self._probe_focus_id = ""
        self.reload()
        self._say("已彻底删除。它和它在所有空间里的向量都没了，不可恢复。")

    # ── 需要人回答的两处（可覆盖，测试用） ──────────────────────────────────
    #
    # 两个模态对话框各抽成一个方法，是为了让 gui 测试能 monkeypatch 掉它们。
    # 测试里弹真对话框要么卡住、要么要靠 QTimer 去点它——两种都会让"库里的字节
    # 有没有变"这条断言变得又慢又脆。

    def _ask_for_content(self, item: MemoryItem) -> str | None:
        """问新正文。取消返回 ``None``。"""
        text, accepted = QInputDialog.getMultiLineText(self, "编辑正文", "正文", item.content)
        return text if accepted else None

    def _confirm_delete(self, item: MemoryItem) -> bool:
        """两次确认。第二次要把**后果**说全，不只是重复一遍"确定吗"。"""
        preview = " ".join(item.content.split())
        if len(preview) > 60:
            preview = preview[:60] + "…"
        first = QMessageBox.question(
            self,
            "彻底删除",
            f"要彻底删除这条记忆吗？\n\n{preview}\n\n「遗忘」是可恢复的，这条不是——删除不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if first != QMessageBox.StandardButton.Yes:
            return False

        consequences = [
            "**不可恢复**——它不会出现在「已遗忘」里，也没有回程票。",
            "它在**所有**记忆空间里的语义向量会一并删除（换回旧嵌入模型也不会回来）。",
        ]
        if item.kind is MemoryKind.PERSONA_ANCHOR:
            # 人格锚分两条（见 memory/persona.py），删的后果按是哪一条说：
            #
            # * 用户那条：删掉不是"恢复成默认"，而是**下次启动重新问一次**——那时
            #   库里没有用户那条，对话区就会把卡片再拿出来。
            # * 内置那条：**界面到不了这里**（读只读规则已经把它灰掉了），但规则与
            #   按钮状态分居两处，这里照样把话说全。
            # * 其余（来源不是这两条）：不猜它是什么，只说它会少一段身份约束。
            if item.source_ref == USER_REF:
                consequences.append(
                    "⚠️ 这是**你自己写的那条人设**。删掉之后，下次启动会重新问你一次"
                    "——不是恢复成什么默认的性格，是空着等你重新写。"
                    "（终端内置的那条还在，它不受影响。）"
                )
            elif item.source_ref == BUILTIN_REF:
                consequences.append(
                    "⚠️ 这是**终端内置的那条人设**，它保证任何模型读到的身份都是同一份"
                    "字节，界面上本来不该删得掉它。删了之后，下次启动会重新写入一条"
                    "内容一样的——因为它是程序按定义生成的，不是谁写下的。"
                )
            else:
                consequences.append(
                    "⚠️ 这是一条**人格锚**，进的是每一轮上下文的 L0 层。删掉之后，"
                    "这段身份约束就没了——它不会被重新写出来。"
                )
        second = QMessageBox.question(
            self,
            "再确认一次",
            "\n\n".join(consequences),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return second == QMessageBox.StandardButton.Yes

    # ── 线索召回（唯一一个后台动作） ────────────────────────────────────────

    def _on_probe(self) -> None:
        item = self._current_item()
        if item is None:
            return
        self._probe_ledger = None
        self._probe_focus_id = item.memory_id
        self.probe_button.setEnabled(False)
        self._say("正在做线索召回…")

        thread = QThread(self)
        worker = MemoryProbeWorker(
            store=self._memory, content=item.content, embedding=self._embedding
        )
        worker.ready.connect(self._on_probe_ready)
        worker.failed.connect(self._on_probe_failed)
        worker.ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # 一对一地走 track：worker 的亲和面留在界面线程、run() 靠 DirectConnection
        # 在工作线程里跑。**不要自己写 started.connect**——那三条约定抄第二份时
        # 漏掉的那条不会报错（见 ui/thread_tracking.py 的模块文档）。
        track(self, thread, worker, what="线索召回", slot="probe")
        thread.start()

    def _on_probe_ready(self, ledger: RecallLedger) -> None:
        self._probe_ledger = ledger
        self.probe_button.setEnabled(self._selected_id() != "")
        self._say("")
        self._refresh_detail()

    def _on_probe_failed(self, detail: str) -> None:
        self.probe_button.setEnabled(self._selected_id() != "")
        self._say(detail)

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt 的命名
        """关窗前把线索召回的线程等停。

        **为什么必须等**：一个仍在运行的 ``QThread`` 在析构时会让 Qt 调 ``qFatal``
        （Windows 上退出码 ``0xC0000409``，fast-fail），没有回溯、没有 stderr。
        主窗口的 ``_shutdown`` 等的是同一批东西，理由一字不差——所以用的是同一个
        :func:`~axo.ui.thread_tracking.wait_for_all`。

        传**快照**：循环期间 ``track`` 挂的 release 闭包会把清单换成新列表。
        """
        wait_for_all(list(self._live_threads))
        super().closeEvent(event)
