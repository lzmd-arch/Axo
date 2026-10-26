"""记忆侧的两个后台工作者：把语义通道接上，以及在发请求前装配上下文。

界面线程绝不能被网络等待阻塞，而这一侧有两次网络往返：嵌入查询（每轮）与嵌入
补账（启动时、以及每轮补一小批）。两者都在这里。

::

    MemorySyncWorker     启动时：探测维度 → 建后端 → 登记空间 → 补向量
    AssemblyWorker       每轮：补一小批向量 → 嵌入查询 → 召回 → 装配

**为什么装配单独一个 worker，而不是塞进** :class:`~axo.ui.chat_worker.ChatWorker`。
放进那个类会把"发一次流式请求"变成"嵌入 + 召回 + 装配 + 流式"，而它的 ``stop()``
语义是"停止生成"，不是"停止准备"。拆开之后两段的失败也分得干净：准备失败时用户
看到的是"记忆检索失败"，不是"请求失败"——**这两句话指向的动作完全不同**。

**降级要写在明处。** 没有嵌入模型、Ollama 那一刻正忙、空间还没登记——这些都
不该让用户说不出话，但也不该静默。降级之后召回只剩词法通道，账本上会写明白：
用户需要知道"这一轮没查到记忆"和"记忆里确实没有这条"是两件不同的事。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from memcore.audit import RecallLedger
from memcore.graph import GraphIndex
from memcore.recall import RecallConfig, recall
from memcore.store import MemoryStore
from memcore.types import now_ts

from ..llm.ollama import (
    DEFAULT_OLLAMA_HOST,
    OllamaEmbeddingBackend,
    probe_embedding_dim,
)
from ..memory.backfill import sync_embeddings
from ..memory.browse import PROBE_BUDGET_TOKENS, PROBE_CONFIG
from ..memory.consolidation import newest_memory_id, read_last_run, run_consolidation
from ..memory.persona import persona_texts
from ..paths import graph_path
from ..session.assemble import MemoryInjection, assemble
from ..session.budget import TokenBudget, TokenMeter
from ..session.store import Message

__all__ = [
    "AUTO_GRAPH_MAX",
    "AssemblyWorker",
    "ConsolidationWorker",
    "EmbeddingSetup",
    "GraphWorker",
    "LEXINDEX_WARM_MIN",
    "LexicalIndexWorker",
    "MemoryProbeWorker",
    "MemorySyncWorker",
    "render_recall_detail",
]

#: 每轮对话补多少条向量。
#:
#: 不是 0：刚写下的那条记忆如果没有向量，**下一轮**就召不回它，而"刚说过的话它
#: 就忘了"是最容易被察觉的那类失忆。也不是"补完为止"：那会把一次普通的对话变成
#: 一个长度不可控的任务。8 条在本地嵌入模型上是一次请求、大约一百毫秒，而它足够
#: 消化一轮对话新增的条数（一轮最多一条）。
_PER_TURN_BACKFILL = 8

#: 库里超过这么多条就**不再自动建图**，改用上次存下的那份（如果还在）。
#:
#: 依据 ``memcore/graph.py`` 的实测表：n=2000 的**含编码**成本是 3.0 s，n=20000
#: 就是 66 s，5 万条 322 s。选 2000 是因为它是"一次全量重建仍停在几秒量级"的最大
#: 点，而自动建图只发生在**打开记忆窗口**时（用户主动的动作），3 s 放在后台线程
#: 里可以接受。
#:
#: **中间档（1000 / 3000 / 5000）没有实测**，是按那两档外推的。宁可保守：阈值定
#: 低了，代价是"该自动建的没自动建"，用户点一下「重建联想」就好；定高了，代价是
#: 每次开记忆窗口都可能白等几十秒——而那是**每次**都要付的。
AUTO_GRAPH_MAX = 2000

#: 库里超过这么多条，就在启动后台预建词法倒排。
#:
#: 50k 真实文本放大语料上，索引构建实测 **3.5 s**（机器有重活并行时 22.3 s），
#: 而首轮纯词法召回在它建好
#: 之前必须等这一下。小库构建是毫秒级，不值得为它多起一个线程；2,000 是同一份
#: 语料上「一建就好」的档位（与 :data:`AUTO_GRAPH_MAX` 同一个理由：超过它，
#: 一次全量构建不再是人眼可忽略的量级）。
LEXINDEX_WARM_MIN = 2000

#: 走大库那条路时说的话。**必须说出来**：那条路上图没有跟库比对过（比对是一次全量
#: 重编码，5 万条约 63 秒），所以它可能已经过期，而自动路径不会去发现这件事。
_LARGE_LIB_NOTE = (
    "库太大，没有逐条比对——这份图是上次建好存下的，可能已经过期。"
    "要确定就点「重建联想」（大库要几分钟）。"
)

_LARGE_LIB_MISSING_NOTE = (
    "库太大，没有逐条比对——这份图是上次建好存下的，可能已经过期；"
    "按 id 看有 {count} 条新记忆不在图里，重建后才会参与联想。"
    "要确定就点「重建联想」（大库要几分钟）。"
)


@dataclass(frozen=True)
class EmbeddingSetup:
    """启动时接语义通道的结果。

    ``backend`` 为 ``None`` 表示**这一轮没有语义通道**——不是错误，是一种正常
    配置（没装嵌入模型，或者 Ollama 恰好没起来）。:attr:`note` 说明原因，界面
    据此显示；没有它的话，"语义通道没接上"和"记忆里没有相关的东西"在界面上长得
    一模一样。
    """

    backend: OllamaEmbeddingBackend | None = None
    synced: int = 0
    dim: int = 0
    note: str = ""


class MemorySyncWorker(QObject):
    """把语义通道接上：探测维度 → 建后端 → 登记空间 → 补向量。

    四步在同一处，因为它们是一条链：少了任何一步，后面那步都无事可做，而"空间
    登记了但一直是空的"这种中间态的症状——语义通道总是返回零条——看起来和"库里
    没有相关的记忆"完全一样。

    **只发一次信号，且永远是** ``ready``。补向量失败不该让用户看见一个错误弹窗：
    记忆内容一条没少，少的只是语义通道，而词法通道照常工作。原因写进
    :attr:`EmbeddingSetup.note`，由界面显示在状态栏。
    """

    ready = Signal(object)
    """参数是 :class:`EmbeddingSetup`。"""

    def __init__(
        self,
        *,
        store: MemoryStore,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        allow_remote: bool = False,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._model = model
        self._host = host
        self._allow_remote = allow_remote
        self._limit = limit

    @Slot()
    def run(self) -> None:
        """**异常绝不外泄。** PySide6 会吞掉槽里抛出的异常——那意味着如果这里
        炸了，界面拿不到任何信号，于是"语义通道永远没接上"且没有任何提示。"""
        if not self._model:
            self.ready.emit(EmbeddingSetup(note="没有可用的嵌入模型，记忆只用语汇匹配"))
            return

        try:
            dim = probe_embedding_dim(self._model, self._host, allow_remote=self._allow_remote)
            backend = OllamaEmbeddingBackend(
                model=self._model,
                dim=dim,
                host=self._host,
                allow_remote=self._allow_remote,
            )
        except Exception as exc:  # noqa: BLE001 — 没有嵌入模型是配置问题，不是崩溃
            self.ready.emit(
                EmbeddingSetup(note=f"嵌入模型 {self._model} 不可用（{exc}），记忆只用语汇匹配")
            )
            return

        try:
            synced = sync_embeddings(self._store, backend, limit=self._limit)
        except Exception as exc:  # noqa: BLE001 — 补账失败不该拦着用户说话
            self.ready.emit(
                EmbeddingSetup(
                    backend=backend,
                    dim=dim,
                    note=f"语义通道就绪，但补向量失败（{type(exc).__name__}：{exc}）",
                )
            )
            return

        note = f"语义通道就绪 · {dim} 维"
        note += f"，补了 {synced} 条向量" if synced else "，没有待补的向量"
        self.ready.emit(EmbeddingSetup(backend=backend, synced=synced, dim=dim, note=note))


class AssemblyWorker(QObject):
    """跑一次"补账 → 嵌入查询 → 召回 → 装配"。

    信号只发一次（``ready`` 或 ``failed`` 二者之一），与 ``ChatWorker`` 同一纪律
    ——重复发射会让界面在收到第二个信号时状态错位。
    """

    ready = Signal(object, object)
    """(``Assembly``, ``RecallLedger | None``)。

    降级说明写在 ``assembly.ledger.notes`` 里，不另开一个通道：用户读的账本只有
    一份，而"这一轮少了一条通道"正是账本该说的话。
    """

    failed = Signal(str)

    def __init__(
        self,
        *,
        store: MemoryStore,
        budget: TokenBudget,
        history: Sequence[Message],
        current: str,
        environment: str,
        embedding: OllamaEmbeddingBackend | None = None,
        session_id: str = "",
        recall_config: RecallConfig | None = None,
        graph: GraphIndex | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._budget = budget
        self._history = list(history)
        self._current = current
        self._environment = environment
        self._embedding = embedding
        self._session_id = session_id
        self._config = recall_config
        #: 关联图（**注入而不是自己读盘**，同 ``memcore.recall`` 对图的约定）。
        #: ``None`` = 没有图。它只在 ``config.graph_weight > 0`` 时才有用；给一个
        #: 默认关的 config 配一张真图是**安全的**：那条通道会被短路掉，不算也不看。
        self._graph = graph
        self._meter = TokenMeter()

    @Slot()
    def run(self) -> None:
        notes = self._sync_vectors()
        query_vector = self._query_vector(notes)

        # ── 召回 ────────────────────────────────────────────────────────────
        # 额度给的是 ``usable``（窗口的全部），**不是某一层的份额**——份额已经拆掉
        # 了。装不下由装配器按优先级丢，账本两处各说各的实话：召回说"选出 N 条"，
        # 装配说"发出去 M 条、K 条没装下"。
        #
        # **不排除本会话写下的记忆。** 以前这里按会话 id 把它们排掉，
        # 于是老会话看不到自己最近写的那批——那正是「新老会话之间能够调用到的记忆有
        # 很明显的差距」的两个真凶之一。重复的代价由视图层的临时归并吸收（刚说出口的
        # 碎条目会被并成一条），完整性优先。
        ledger: RecallLedger | None = None
        injected: list[MemoryInjection] = []
        try:
            ledger = recall(
                self._store,
                query_text=self._current,
                budget_tokens=self._budget.usable,
                query_vector=query_vector,
                space_id="" if self._embedding is None else (self._embedding.space_id or ""),
                config=self._config,
                graph=self._graph,
            )
            injected = [
                MemoryInjection(
                    memory_id=item.memory_id,
                    content=item.content,
                    tokens=item.tokens,
                    explain=item.explain,
                    score=item.scores.fused,
                    created_at=item.created_at,
                    span=item.span,
                    session_id=item.session_id,
                    derived_from=item.derived_from,
                )
                for item in ledger.injected
            ]
        except Exception as exc:  # noqa: BLE001 — 召回失败降级成"这一轮没有记忆"
            notes.append(f"记忆检索失败，这一轮不注入记忆（{type(exc).__name__}：{exc}）")

        # ── 装配 ────────────────────────────────────────────────────────────
        try:
            assembly = assemble(
                self._history,
                current=self._current,
                budget=self._budget,
                persona=persona_texts(self._store),
                environment=self._environment,
                recalled=injected,
                meter=self._meter,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"装配上下文失败：{type(exc).__name__}：{exc}")
            return

        self.ready.emit(_with_notes(assembly, notes), ledger)

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _sync_vectors(self) -> list[str]:
        """补一小批向量，返回要说给用户听的话。"""
        if self._embedding is None:
            return []
        try:
            synced = sync_embeddings(self._store, self._embedding, limit=_PER_TURN_BACKFILL)
        except Exception as exc:  # noqa: BLE001 — 补账失败不影响这一轮问答
            return [f"补向量失败（{type(exc).__name__}），这一轮用库里已有的向量"]
        return [f"补了 {synced} 条向量"] if synced else []

    def _query_vector(self, notes: list[str]):
        """把用户这一句编成向量。失败返回 ``None``——那是**降级，不是失败**。"""
        if self._embedding is None:
            notes.append("没有嵌入模型，这一轮只用语汇匹配")
            return None
        try:
            return self._embedding.embed([self._current])[0]
        except Exception as exc:  # noqa: BLE001 — 任何失败都走降级，见模块文档
            notes.append(f"嵌入失败，这一轮只用语汇匹配（{type(exc).__name__}）")
            return None


class MemoryProbeWorker(QObject):
    """「以它为线索召回一次」——记忆界面里**唯一**含网络调用的动作。

    为什么它单独一个 worker 而编辑/遗忘/删除都是同步的：那三件是单行写加一次
    轻量 SDR 编码（毫秒级），而这一件要先把这条记忆的正文编成向量（**一次网络
    往返**）再扫全库。放在界面线程上就是"点一下按钮，窗口僵住半秒"。

    **``settle=False`` 是硬要求，不是优化**（见 :func:`memcore.recall.recall`
    对 ``settle`` 的说明）：这是"看一眼"，不是"想起来"。开着的话，用户每点一次
    探索按钮，被看到的那几条就会显得更常用、衰减得更慢——一次观察改变了被观察
    的东西，而真实召回的排序从此带着探索的痕迹，没有任何地方看得出来。

    **嵌入失败不发 ``failed``**：那是降级，不是失败。``query_vector=None`` 时
    ``recall`` 只走词法通道，照常给出一份账本；原因写进 ``ledger.notes``，由
    界面显示。发 ``failed`` 的话，用户在"没装嵌入模型"时点这个按钮只会得到一个
    错误提示，而他本来应该看到词法通道给出的邻居。
    """

    ready = Signal(object)
    """参数是 :class:`~memcore.audit.RecallLedger`（探索参数下的一份）。"""

    failed = Signal(str)
    """只在**召回本身**炸了的时候发（存储坏了、向量维度不对）。"""

    def __init__(
        self,
        *,
        store: MemoryStore,
        content: str,
        embedding: OllamaEmbeddingBackend | None = None,
        config: RecallConfig | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._content = content
        self._embedding = embedding
        self._config = config

    @Slot()
    def run(self) -> None:
        """**异常绝不外泄。** PySide6 会吞掉槽里抛出的异常——那意味着界面再也
        收不到任何终局信号，按钮会一直停在"召回中"。"""
        notes: list[str] = []
        vector, space_id = self._query_vector(notes)
        try:
            ledger = recall(
                self._store,
                query_text=self._content,
                budget_tokens=PROBE_BUDGET_TOKENS,
                query_vector=vector,
                space_id=space_id,
                config=self._config or PROBE_CONFIG,
                settle=False,
            )
        except Exception as exc:  # noqa: BLE001 — 详见下面的 failed 文案
            self.failed.emit(f"线索召回失败：{type(exc).__name__}：{exc}")
            return
        # 降级说明并进账本，不另开通道：用户读的账本只有一份，而"这一轮少了一条
        # 通道"正是账本该说的话。新备注排在前面——它们说的是"这次召回开始时发生
        # 了什么"，账本自己那几条说的是"扫的时候发生了什么"。
        self.ready.emit(replace(ledger, notes=(*notes, *ledger.notes)))

    def _query_vector(self, notes: list[str]) -> tuple[Any, str]:
        """把这条记忆的正文编成向量。任何失败都返回 ``(None, "")`` 走词法降级。"""
        if self._embedding is None:
            notes.append("没有嵌入模型，这次线索召回只用语汇匹配")
            return None, ""
        try:
            return self._embedding.embed([self._content])[0], (self._embedding.space_id or "")
        except Exception as exc:  # noqa: BLE001 — 降级，不是失败，见类文档
            notes.append(f"嵌入失败，这次线索召回只用语汇匹配（{type(exc).__name__}）")
            return None, ""


class GraphWorker(QObject):
    """建或载入关联图。**一条 worker 服务两条路径**：打开记忆窗口时的自动建、
    设置页的「重建联想」。

    ``ready`` 的参数是 ``(图, 是不是过期, 一句话)``。第二个参数是**三态**
    （``True`` / ``False`` / ``None``）：

    * ``False`` —— 刚建好，或者刚比对过、图是最新的；
    * ``None`` —— **没比对过**（库太大走的就是这条）。这与 ``False`` 不是一回事：
      一个是"确认没过期"，一个是"不知道"。界面据此说不同的话
      （``memory.browse.association_header``）。

    **``force`` 就是那两条路径的分界。** 自动那条（``force=False``）够用就好：
    盘上有图、库不大、且它没过期，就直接用。手动那条（``force=True``）**必须真的
    重建**——用户按那个按钮的全部意思就是"我不信现在这份"，而"不信"最常见的原因
    恰恰是大库（那份图根本没比对过）。少了这个分叉，大库上点「重建联想」会原样
    把旧图还回来，按钮看起来坏了。

    **大库不遍历、不判过期。** ``list(iter_items())`` 在 5 万条上很占内存，而
    ``is_stale`` 本身是一次全量重编码（5 万条约 63 秒）。所以大库把图当"可能过期
    但先用着"，由 :data:`_LARGE_LIB_NOTE` 说出来，用户手动重建兜底。

    **写盘失败不吞掉刚算好的图**：图是缓存，这一份在本次会话里照样能用，下次启动
    再建一次而已。

    **已知代价**：大库手动重建约 4.5 分钟（5 万条：建图 259 s + 编码 63 s），而关窗
    时 ``wait_for_all`` 会等到超时、打印「收尾超时」再放行。**那不是 bug**：后台
    线程不写库，最坏是白算一次。本版**不做取消**，也没有排给哪个阶段。
    """

    ready = Signal(object, object, str)
    """(``GraphIndex``, 过期与否的三态, 给用户看的一句话)。"""

    failed = Signal(str)
    """只在**图本身**建不出来的时候发（SDR 宽度不一致、盘上的文件坏了）。"""

    def __init__(self, *, store: MemoryStore, force: bool = False) -> None:
        super().__init__()
        self._store = store
        #: 自动那条路是 ``False``，设置页那个按钮是 ``True``。**这一趟的全部语义差别
        #: 就在这一个布尔上**（见类文档），所以它不藏在闭包里，就是一个字段。
        self.force = force

    @Slot()
    def run(self) -> None:
        """**异常绝不外泄**（同本模块其它 worker）。发不出去的话，设置页那个按钮
        会永远停在"正在重建"，而且没有任何提示。"""
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — 转成一句人话，见类文档
            self.failed.emit(f"建立关联图失败：{type(exc).__name__}：{exc}")

    def _run(self) -> None:
        path = graph_path()
        # **盘上那份坏了要当"没有图"，不能当"失败"。** 图是缓存：文件截断、版本
        # 对不上、写入时断电——这些都只说明该重算一遍。让 ``load`` 的异常冒出去的话，
        # 用户会得到一个"建立关联图失败"的红字，而正确答案是重新建一张。
        try:
            existing = GraphIndex.load(path) if path.exists() else None
        except Exception:  # noqa: BLE001 — 详见上句
            existing = None

        if existing is not None and not self.force and self._store.count() > AUTO_GRAPH_MAX:
            # **只比成员，不重编码。** 大库不跑 ``is_stale``（5 万条约 63 s），但
            # 一次 ``SELECT memory_id`` 只要几十毫秒，足以把"建图之后又写了 N 条"
            # 说出来——那正是图通道与巩固漏掉新记忆的原因。
            missing = existing.missing_nodes(self._store.memory_ids(encoded_only=True))
            note = (
                _LARGE_LIB_MISSING_NOTE.format(count=len(missing)) if missing else _LARGE_LIB_NOTE
            )
            self.ready.emit(existing, None, note)
            return

        items = list(self._store.iter_items())
        if existing is not None and not self.force and not existing.is_stale(items):
            self.ready.emit(existing, False, f"图是最新的（{len(existing)} 个节点），不用重建。")
            return

        graph = GraphIndex.build(items)
        note = f"已建好：{len(graph)} 个节点、{graph.edge_count()} 条边。"
        try:
            graph.save(path)
        except OSError as exc:
            note += f" 但没能写盘（{type(exc).__name__}），本次会话里能用，重启后会再建。"
        self.ready.emit(graph, False, note)


class LexicalIndexWorker(QObject):
    """启动时在后台预建词法倒排。

    **为什么单独一个 worker。** 50k 真实文本放大语料上，``lexical_index()`` 实测
    3.5 s（机器有重活并行时 22.3 s；见 ``sweep_results/axo_recall_50k_results.md``）。
    它原本发生在第一轮
    纯词法召回的调用栈里：界面不冻（召回在装配线程），但用户说完第一句话要等
    二十多秒才看到回复，而这段时间里没有任何提示。放到启动后台，这段时间与用户
    读界面、选模型重叠掉。

    **失败不发``failed``到界面。** 索引是缓存，不是真相：建失败时第一轮召回会
    再试一次，而那时错误会走它自己的账本/降级路径。这里只把结果报给线程用于
    收尾；没有接收者也完全正常（启动时没人绑 ``ready``）。

    **没有取消。** 与 :class:`GraphWorker` 的大库重建同一条已知代价：50k 索引
    构建期间关窗，``wait_for_all`` 会等 3 秒、打印「收尾超时」再放行。索引构建
    只读库，最坏是白算一次；真实库 166 条是毫秒级，碰不到这个窗口。
    """

    ready = Signal(int)
    """参数是索引里的条目数。"""

    failed = Signal(str)
    """只在建索引本身抛异常时发；发给线程收尾，不弹窗口。"""

    def __init__(self, *, store: MemoryStore) -> None:
        super().__init__()
        self._store = store

    @Slot()
    def run(self) -> None:
        try:
            index = self._store.lexical_index()
        except Exception as exc:  # noqa: BLE001 — 转成一句话，见类文档
            self.failed.emit(f"预建词法索引失败：{type(exc).__name__}：{exc}")
            return
        self.ready.emit(len(index.ids))


class ConsolidationWorker(QObject):
    """跑一轮睡眠巩固。**它只发结果，不碰界面、不碰线程。**

    走的是 :func:`~axo.memory.consolidation.run_consolidation`——**改库的
    次序只有那一份实现**（拍快照 → 逐簇「改库 + 追加事件」→ 写 ``last.json``），
    这里只负责"在后台线程里把它跑完，然后报三件事之一"。

    ``force`` 就是两条路径的分界，与 :class:`GraphWorker` 一字不差：

    * ``False``（定时那条）**够用就好**：库里没新东西就让掉这一轮；
    * ``True``（托盘与设置页的「立即巩固」）**必须真的跑**——用户按那个按钮的全部
      意思就是"我现在就要看一遍"，而"按了却什么都没发生"是最坏的一种反馈。

    **异常绝不外泄**（同本模块其它 worker）：漏出去的话，线程死在半路，界面上那句
    "正在巩固"再也不会变——用户从此以为巩固是坏的，而真正的原因（比如 SDR 宽度
    不一致）只在控制台里。

    **它读的那张图是主窗口手里那个对象，不另开一份。** ``GraphIndex`` 只被整体替换
    （``_on_graph_ready`` 赋一个新的），从不就地修改，所以这里拿到的要么是这一轮的
    完整图、要么是上一轮的完整图，不存在撕开一半的中间态。

    **已知的粗**：那两道门（空库、没有新记忆）问的是 ``newest_memory_id``，也就是
    "最新那条的 id 还是不是上次那条"。id 是 ULID 且单调递增，所以**新增**记忆一定会
    被看见，而**改一条、删一条**不会——那两种情况下这一轮会被让掉，被改的那条今天
    不会被重新认一遍。要发现它们得比对全库的指纹（一次全表扫描 + 哈希），而这道门
    存在的意义恰恰是**别每天都做一次全表扫描**。明天写进来的新记忆会把这一轮带回来，
    那时改动也一并被认。
    """

    ready = Signal(object)
    """跑完了一轮：:class:`~axo.memory.consolidation.ConsolidationRun`。"""

    skipped = Signal()
    """**让掉了这一轮**（自动那条路的空库 / 没有新记忆）。不带参数：让掉的原因不改变
    接收方要做的事（按整个周期重排，什么都不说），说两个理由是没人读的差别。"""

    failed = Signal(str)
    """跑不成，一句人话。"""

    def __init__(self, *, store: MemoryStore, graph: GraphIndex, force: bool = False) -> None:
        super().__init__()
        self._store = store
        self._graph = graph
        #: 见类文档：**这一趟的全部语义差别就在这一个布尔上**，所以它是一个字段，
        #: 不藏在闭包里（同 :class:`GraphWorker`）。
        self.force = force

    @Slot()
    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — 转成一句人话，见类文档
            self.failed.emit(f"睡眠巩固失败：{type(exc).__name__}：{exc}")

    def _run(self) -> None:
        newest = newest_memory_id(self._store)
        if not self.force and self._nothing_to_do(newest):
            self.skipped.emit()
            return
        # ``at`` 取**这一轮真正开始的时刻**：run_id、run_seed 与快照文件名都由它派生，
        # 而 "一轮" 的定义就是这一次调用。
        self.ready.emit(run_consolidation(self._store, self._graph, at=now_ts()))

    def _nothing_to_do(self, newest: str) -> bool:
        """自动那条路的门。**手动那条不走这里**（见 :attr:`force`）。

        两种"没什么可跑的"：**空库**（连报告都没什么好说的），以及**最新那条还是上次
        那条**（``last.json`` 里记着上一轮跑完时的 ``newest_id``）。两者都不写
        ``last.json``——那一份记的是"最近一次**真跑**"，而这一轮没有跑。代价是调度器
        每次唤醒都要重问一次，而问的代价是一次走主键索引的 ``LIMIT 1`` 查询。
        """
        if not newest:
            return True
        last = read_last_run()
        return last is not None and last.get("newest_id") == newest


def _with_notes(assembly, notes: Sequence[str]):  # noqa: ANN001
    """把准备阶段的备注并进装配账本。

    ``AssemblyLedger`` 是 frozen 的（它要被反复读、被比对），所以这里是造一个新的
    而不是往列表里塞。并进去而不是另走一个信号：用户会读的账本只有一份，而"这一轮
    少了一条通道"正是账本该说的话。

    新备注排在前面：它们说的是"这一轮开始时发生了什么"，而装配器自己那几条说的是
    "装的时候发生了什么"——顺序反过来，读者会先看到结论再看到前提。
    """
    if not notes:
        return assembly
    return replace(
        assembly, ledger=replace(assembly.ledger, notes=(*notes, *assembly.ledger.notes))
    )


def render_recall_detail(ledger: RecallLedger | None, *, max_rejected: int = 8) -> str:
    """把召回的过程渲染成给人看的几行。

    账本的价值有一半在这一段：**"为什么没注入它"与"为什么注入了它"同样重要。**
    只显示入选的那几条，用户看到的是"库里好像只有这几条相关"——而实际可能是
    "其余二十条分数不够"。两者该做的动作完全不同（前者去补记忆，后者去调阈值）。

    纯函数，不碰 Qt，所以它能脱离界面测。
    """
    if ledger is None:
        return ""
    lines = [ledger.summary()]
    lines.extend(f"· {note}" for note in ledger.notes)

    if ledger.rejected:
        lines.append("")
        lines.append("未注入：")
        # 按原因分组：同一个原因重复十遍没有信息量，而"十条都是分数不够"和"十条
        # 都是装不下"指向的是两个完全不同的动作。
        grouped: dict[str, list] = {}
        for candidate in ledger.rejected:
            grouped.setdefault(candidate.reason.value, []).append(candidate)
        for _reason, group in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            lines.append(f"  {len(group)} 条：{group[0].reason.label}")
            for candidate in group[:max_rejected]:
                lines.append(f"    · {candidate.content_preview} —— {candidate.explain}")
            if len(group) > max_rejected:
                lines.append(f"    …另有 {len(group) - max_rejected} 条")
    return "\n".join(lines)
