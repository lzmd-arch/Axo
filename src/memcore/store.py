"""记忆库的持久化。

一个 SQLite 文件，WAL 模式。选它而不是"一个 JSON + 一个 npz"，是因为记忆的写入
是**每轮实时发生**的（用户明确要求），而崩溃断电时最不能接受的就是"整库重写成
了半截"。SQLite 给的是逐条事务：一次写入要么完整落盘，要么完全没发生。

三张表，各管一件事::

    spaces      登记过的记忆空间（embedding space），带维度
    memories    记忆条目本身。词法 SDR 以 idx/weights 两个 BLOB 存在行上
    embeddings  (memory_id, space_id) → 向量。**按空间分行存**

``embeddings`` 单独一张表、且主键带 ``space_id``，是为了让"换模型"这件事不需要
搬家：旧模型的向量原样留着，切回去就能立刻用；新模型的向量另起一行。若把向量
存成 ``memories`` 上的一列，换模型就得把全库的向量覆盖掉——那种设计下"换一个
读者"实际上变成了"把书重抄一遍"，抄坏一次就没了。

**持久化的只有词法 SDR 与向量。** ``sem_sdr`` 与 ``assoc_sdr`` 都是**派生**的
（前者由向量编码、后者由词法 SDR 二次稀疏化），现在不存：存一个恒为 NULL 的列
只会让人误以为它已经在用了。

**但它们两个的"派生"性质不同，别按同一条规则处理。**

* ``sem_sdr`` **是**"每个空间一份"：它由某个 embedding 模型产出，换模型就得换一份，
  所以它将来落的话挂在 ``embeddings`` 行上（那行的主键带 ``space_id``）。
* ``assoc_sdr`` **不是**：它只依赖 :attr:`~memcore.types.MemoryItem.lex_sdr`，
  与 embedding 空间**毫无关系**——换模型不需要重算它。所以它永远不会需要按空间
  分行存。P4a 之前这里把两者写成"同样是每个空间一份"，那句话对 ``assoc_sdr``
  是错的，而错的方向恰好是"多存一份派生值"。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import StoreCorruptError, StoreError
from .lexindex import LexicalIndex
from .spaces import Space, SpaceRegistry
from .types import SDR, MemoryItem, MemoryKind, MemorySummary, Taint, Tier, now_ts

__all__ = ["MemoryStore"]

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS spaces (
    space_id   TEXT PRIMARY KEY,
    dim        INTEGER NOT NULL,
    model_ref  TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    tier            TEXT NOT NULL,
    content         TEXT NOT NULL,
    taint           TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_access_ts  REAL NOT NULL,
    last_decay_ts   REAL NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    reinforce_count INTEGER NOT NULL DEFAULT 0,
    weight          REAL NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    space_id        TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    source_ref      TEXT NOT NULL DEFAULT '',
    meta_json       TEXT NOT NULL DEFAULT '{}',
    lex_idx         BLOB,
    lex_w           BLOB,
    lex_width       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_tier    ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_pinned  ON memories(pinned);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    space_id  TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    added_at  REAL NOT NULL,
    PRIMARY KEY (memory_id, space_id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_space ON embeddings(space_id);
"""

#: 权重变化小于这个幅度就不写回。惰性衰减每轮都在算，若每次浮点末位的差异都
#: 触发一次 UPDATE，一次召回就会写几十行——而读者根本分不出 0.5000001 与
#: 0.5000000 的区别。
_DECAY_EPSILON = 1e-4

#: 一次 UPDATE 里 ``meta_json`` 的体积上限（字节）。见 :meth:`MemoryStore.add`。
_MAX_META_BYTES = 256 * 1024

#: 词法索引的 delta 最多容纳多少条新记忆，超过就整库重建一次。
#:
#: 50k 条上整库构建实测约 5 s（静默机器；机器有重活并行时会到 22 s）。按每条
#: 记忆都重建，50k 库的每一轮对话都要付一次；delta 把这个成本摊到每 512 条。
#: 512 条 delta 的 CSR 重建是 512×573 个位，毫秒级；再大就该做一次全量合并。
_LEX_DELTA_MAX = 512


class MemoryStore:
    """记忆库。

    **线程安全**：内部是一个连接 + 一把可重入锁。SQLite 的连接默认不允许跨线程
    使用（``check_same_thread=True``），而本项目的写入发生在界面线程、召回发生在
    工作线程，跨线程是常态。把它关掉并自己加锁，比给每个线程开一个连接简单得多
    ——后者的代价是每个连接一份 WAL 状态与一次 fsync 往返，而本项目的并发度
    （一次写入 + 一次召回）根本用不上。

    锁是**可重入**的：``apply_decay`` 会调 ``get`` 再调 ``update``，两者都要进锁。
    用普通 ``Lock`` 会当场自锁。

    Args:
        path: 库文件路径。``":memory:"`` 是合法的（测试用）。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._ensure_schema()
        self._registry = SpaceRegistry()
        for space in self._load_spaces():
            self._registry.register(space)
        self._revision = 0
        self._lex_index: LexicalIndex | None = None
        self._lex_index_revision = -1
        #: 只串行化"建索引"这件事，**不保护库**。建 50k 索引要 22 s，如果整段
        #: 持有 ``_lock``，界面线程的一次记忆写入会被冻住 22 s；而索引构建在
        #: ``fetchall`` 之后只碰 Python 对象，不需要库锁。
        self._lex_build_lock = threading.Lock()
        self._embedding_revision = 0
        self._normalized_cache: dict[str, tuple[int, tuple[str, ...], np.ndarray]] = {}

    # ── 打开与关闭 ──────────────────────────────────────────────────────────

    def _configure(self) -> None:
        with self._lock:
            # WAL：读者不挡写者、写者不挡读者。本项目正是"界面线程写、工作线程读"
            # 的形状，回滚日志模式下两者会互相阻塞。
            self._connection.execute("PRAGMA journal_mode=WAL")
            # NORMAL：WAL 下这个档位只在检查点丢失最近的提交，不损坏库。
            # FULL 会让每轮记忆写入都多一次 fsync，代价与收益不成比例。
            self._connection.execute("PRAGMA synchronous=NORMAL")
            # 外键约束默认是**关**的——不显式打开，embeddings 的 ON DELETE CASCADE
            # 就是一句装饰。
            self._connection.execute("PRAGMA foreign_keys=ON")
            # 默认的 0 是"只在显式事务里用内存日志"，会导致每次 commit 都建临时文件。
            self._connection.execute("PRAGMA temp_store=MEMORY")

    def _ensure_schema(self) -> None:
        with self._lock:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                self._connection.executescript(_SCHEMA_SQL)
                self._connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                self._connection.commit()
                return
            if version > _SCHEMA_VERSION:
                raise StoreCorruptError(
                    f"库文件 {self.path} 的 schema 版本是 {version}，而这份代码只认到 "
                    f"{_SCHEMA_VERSION}。**不要用旧版本打开它**——旧代码不认识新列，"
                    f"会把它们当成不存在而静默丢数据。请升级，或换一个库文件。"
                )
            if version < _SCHEMA_VERSION:
                self._migrate(version)

    def _migrate(self, from_version: int) -> None:  # noqa: ARG002 — 目前没有升级路径
        raise StoreCorruptError(
            f"库文件 {self.path} 的 schema 版本是 {from_version}，需要升级到 "
            f"{_SCHEMA_VERSION}，但这份代码里没有对应的迁移步骤。"
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def registry(self) -> SpaceRegistry:
        """本库已知的记忆空间。"""
        return self._registry

    # ── 空间 ────────────────────────────────────────────────────────────────

    def register_space(self, space: Space) -> Space:
        """登记（或确认）一个记忆空间。已存在且维度一致时幂等。"""
        with self._lock:
            registered = self._registry.register(space)
            self._connection.execute(
                "INSERT INTO spaces (space_id, dim, model_ref, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(space_id) DO UPDATE SET model_ref=excluded.model_ref",
                (
                    registered.space_id,
                    registered.dim,
                    registered.model_ref,
                    registered.created_at,
                ),
            )
            self._connection.commit()
            return registered

    def _load_spaces(self) -> list[Space]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT space_id, dim, model_ref, created_at FROM spaces"
            ).fetchall()
        return [
            Space(
                space_id=row["space_id"],
                dim=int(row["dim"]),
                model_ref=row["model_ref"],
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    def spaces(self) -> list[Space]:
        return self._registry.all()

    # ── 条目：写 ────────────────────────────────────────────────────────────

    def add(self, item: MemoryItem) -> MemoryItem:
        """插入或整行覆盖一条记忆。

        用 ``ON CONFLICT DO UPDATE`` 而不是 ``INSERT OR REPLACE``：后者在语义上是
        **先 DELETE 再 INSERT**，会连带触发 ``embeddings`` 上的 ``ON DELETE
        CASCADE``——也就是说，重写一条记忆的正文会顺手把它的向量删掉，而这件事
        没有任何报错。改用 UPSERT 后，向量与条目是两条独立的写入路径。

        Raises:
            StoreError: ``meta`` 不是 JSON 可表示的，或者体积过大。
        """
        payload = self._item_to_row(item)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO memories (
                    memory_id, kind, tier, content, taint, created_at, last_access_ts,
                    last_decay_ts, access_count, reinforce_count, weight, pinned,
                    space_id, session_id, source_ref, meta_json,
                    lex_idx, lex_w, lex_width
                ) VALUES (
                    :memory_id, :kind, :tier, :content, :taint, :created_at, :last_access_ts,
                    :last_decay_ts, :access_count, :reinforce_count, :weight, :pinned,
                    :space_id, :session_id, :source_ref, :meta_json,
                    :lex_idx, :lex_w, :lex_width
                )
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind=excluded.kind, tier=excluded.tier, content=excluded.content,
                    taint=excluded.taint, created_at=excluded.created_at,
                    last_access_ts=excluded.last_access_ts, last_decay_ts=excluded.last_decay_ts,
                    access_count=excluded.access_count, reinforce_count=excluded.reinforce_count,
                    weight=excluded.weight, pinned=excluded.pinned, space_id=excluded.space_id,
                    session_id=excluded.session_id, source_ref=excluded.source_ref,
                    meta_json=excluded.meta_json, lex_idx=excluded.lex_idx,
                    lex_w=excluded.lex_w, lex_width=excluded.lex_width
                """,
                payload,
            )
            self._connection.commit()
            self._revision += 1
            self._update_lex_index_after_add(item)
        return item

    def _update_lex_index_after_add(self, item: MemoryItem) -> None:
        """把新增条目接进当前索引；接不了就让索引失效，等下一次全量重建。

        调用方必须已经持有 ``_lock`` 并完成了 ``_revision += 1``。这里只在
        "索引恰好停在上一个 revision"时生效；否则索引本来就该重建，什么都不做。
        """
        index = self._lex_index
        if index is None or self._lex_index_revision != self._revision - 1:
            return
        if index.contains(item.memory_id):
            # UPSERT 覆盖了既有条目：base 里的旧词法位不能留，删掉整份索引。
            # 编辑是低频动作，不值得为它维护 tombstone。
            self._lex_index = None
            self._lex_index_revision = -1
            return
        if item.lex_sdr is None:
            # 新条目没有词法码。索引对既有条目仍然正确；下一轮候选路径会因为
            # ``unencoded_count() > 0`` 退回全量扫描，所以不必重建。
            self._lex_index_revision = self._revision
            return
        if len(index.delta_ids) >= _LEX_DELTA_MAX:
            # delta 满了：停止追加，下一次 lexical_index() 做一次全量合并。
            self._lex_index = None
            self._lex_index_revision = -1
            return
        self._lex_index = index.with_added(item.memory_id, item.lex_sdr.idx, item.lex_sdr.width)
        self._lex_index_revision = self._revision

    def update(self, item: MemoryItem) -> MemoryItem:
        """更新一条已存在的记忆。不存在时抛 :class:`StoreError`。

        与 :meth:`add` 分开，是为了让"我改的是既有条目"与"我写的是新条目"在调用
        处能区分开。用 :meth:`add` 时手滑写错一个 id，会静默地凭空多出一条记忆；
        用这个则会当场报错。
        """
        with self._lock:
            if not self._exists(item.memory_id):
                raise StoreError(f"记忆 {item.memory_id!r} 不存在，不能 update。新建请用 add。")
        return self.add(item)

    def delete(self, memory_id: str) -> bool:
        """删除一条记忆及其全部向量。返回是否真的删掉了东西。"""
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            self._connection.commit()
            if cursor.rowcount > 0:
                self._revision += 1
                # 删除要维护 tombstone 才能增量；删除是低频动作，整库重建更简单。
                self._lex_index = None
                self._lex_index_revision = -1
            return cursor.rowcount > 0

    def transform(self, memory_id: str, change: Callable[[MemoryItem], MemoryItem]) -> MemoryItem:
        """在**同一把锁内**读出、改写、写回一条记忆，返回改写后的条目。

        **为什么不是「先 ``get`` 再 ``update``」**：``update`` 是整行覆盖。两次调用
        之间有一个窗口，而召回路径恰好会在这个窗口里写库——``apply_decay`` 结算
        权重、``touch_access`` 加访问计数，两者都是整行的另一个来源。于是"用户在
        记忆界面里改一条正文"与"这条记忆刚被召回"撞上时，后写的那个会把先写的
        整个盖掉：要么权重被打回旧值，要么用户改的正文凭空回滚。这两种都不报错。

        锁是可重入的，所以 ``get``/``add`` 在锁内再取一次锁没有问题。

        Args:
            memory_id: 要改的条目。
            change: 拿到当前条目，返回改写后的条目。**抛异常则整条不动**——
                这是"编辑成全标点、词法编码失败"这类错误不该留下半个写入的保证。

        Raises:
            StoreError: 条目不存在，或者 ``change`` 改了 ``memory_id``。
        """
        with self._lock:
            item = self.get(memory_id)
            if item is None:
                raise StoreError(
                    f"记忆 {memory_id!r} 不存在，不能 transform。"
                    f"（并发下这可能是「另一处刚把它删了」，界面应当刷新列表。）"
                )
            revised = change(item)
            if revised.memory_id != memory_id:
                raise StoreError(
                    f"transform 的 change 把 memory_id 从 {memory_id!r} 改成了 "
                    f"{revised.memory_id!r}。改 id 等于换一条记忆，而调用方以为自己在"
                    f"改原来那条——写下去的结果是「什么都没改，还多了一条」。"
                )
            self.add(revised)
            return revised

    # ── 条目：读 ────────────────────────────────────────────────────────────

    def _exists(self, memory_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row is not None

    def get(self, memory_id: str) -> MemoryItem | None:
        """按 id 取一条。不存在返回 ``None``。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return None if row is None else self._row_to_item(row)

    def get_many(self, memory_ids: Sequence[str]) -> list[MemoryItem]:
        """按给定顺序批量取条目；不存在的 id 跳过。

        只用于候选子集的召回扫描；一次查询用 ``IN`` 分块，避免逐条往返。
        """
        if not memory_ids:
            return []
        found: dict[str, MemoryItem] = {}
        with self._lock:
            ids = list(memory_ids)
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT * FROM memories WHERE memory_id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    item = self._row_to_item(row)
                    found[item.memory_id] = item
        return [found[memory_id] for memory_id in memory_ids if memory_id in found]

    @property
    def revision(self) -> int:
        """任何会改变词法码的写入都会 +1；倒排索引用它判断缓存是否过期。"""
        return self._revision

    def unencoded_count(self) -> int:
        """没有任何词法码的条目数。候选路径看到它们时会退回全量扫描。"""
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE lex_idx IS NULL"
                ).fetchone()[0]
            )

    def lexical_index(self) -> LexicalIndex:
        """词法倒排索引的缓存。按 ``revision`` 失效；只读 ``lex_*`` 列。

        **构建期间不持有库锁。** 50k 条上 ``LexicalIndex.build`` 实测 22.3 s；
        如果它整段持锁，界面线程的一次 ``add`` / ``delete`` 会被冻住同样久，
        表现是"发一条消息后界面卡死"，而原因看起来在召回、其实在索引构建。
         ``fetchall`` 在锁内（游标不是线程安全的），之后只碰 Python 对象。
         ``_lex_build_lock`` 让同一时刻只有一个构建者，避免预热线程与首轮召回
        各建一份。
        """
        with self._lex_build_lock:
            while True:
                with self._lock:
                    if (
                        self._lex_index is not None
                        and self._lex_index_revision == self._revision
                        and len(self._lex_index.delta_ids) <= _LEX_DELTA_MAX
                    ):
                        return self._lex_index
                    revision = self._revision
                    rows = self._connection.execute(
                        "SELECT memory_id, lex_idx, lex_width FROM memories ORDER BY memory_id"
                    ).fetchall()
                index = LexicalIndex.build(
                    (row["memory_id"], row["lex_idx"], row["lex_width"]) for row in rows
                )
                with self._lock:
                    # 构建期间有写入：这一份已经缺条目，不能缓存，也不能拿它当
                    # "当前"。重来一次；写入通常是一次一条，重来最多多一轮。
                    if self._revision == revision:
                        self._lex_index = index
                        self._lex_index_revision = revision
                        return index

    def count(self) -> int:
        """条目总数。"""
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def iter_items(self, *, batch_size: int = 512) -> Iterator[MemoryItem]:
        """流式遍历全部条目。

        **分批取而不是一次 fetchall**：一条记忆的正文加上 SDR 有几 KB，全库两万条
        就是几十兆的 Python 对象。召回每轮都要扫一遍全库，一次性物化的代价会在
        内存里持续存在。
        """
        with self._lock:
            cursor = self._connection.execute("SELECT * FROM memories ORDER BY memory_id")
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                # 出锁之前把行转成对象——sqlite3.Cursor 不是线程安全的，出了这一层
                # 再迭代它会在另一个线程写库时炸掉。
                items = [self._row_to_item(row) for row in rows]
                yield from items

    def load_all(self) -> list[MemoryItem]:
        """把全库读进内存。基准脚本与跨模型一致性测试用，召回路径不要用它。"""
        return list(self.iter_items())

    def summaries(
        self,
        *,
        query: str = "",
        tier: Tier | None = None,
        kind: MemoryKind | None = None,
        limit: int = 2000,
        offset: int = 0,
    ) -> list[MemorySummary]:
        """列出记忆的**摘要**（不含 SDR 与 meta），最新的在前。

        记忆界面的列表用这个，不用 :meth:`load_all`：后者会把每条记忆的
        ``lex_idx``/``lex_w`` 反序列化出来（约 6.9 KB/条），而列表一个字节都不
        显示它们。这是"两万条记忆的界面"与"两百条记忆的界面"之间唯一的区别。

        Args:
            query: 子串过滤。走 SQL ``LIKE``——**不是**相似度搜索。浏览是"找那一条"，
                模糊匹配留给召回路径（它有自己的编码与阈值）。``%``/``_``/``\\``
                按字面处理，不会被当成通配符。
            tier: 只要这一层。``None`` 表示全部（含 ``COLD``——被软遗忘的条目要
                看得见，否则用户没有任何办法把它恢复回来）。
            kind: 只要这一类。
            limit: 一次最多几条。**调用方要检查是否命中上限**：把"只显示了最近
                2000 条"当成"库里只有 2000 条"，是这一层最容易造成的误读。
            offset: 跳过前几条（分页用）。

        Returns:
            摘要列表，按 ``memory_id`` **降序**——ULID 单调递增，所以这等价于
            "最新在前"，且走主键索引。
        """
        sql = (
            "SELECT memory_id, kind, tier, content, weight, pinned, created_at, "
            "last_access_ts, access_count, session_id, source_ref FROM memories"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if query:
            clauses.append("content LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(query))
        if tier is not None:
            clauses.append("tier = ?")
            params.append(str(tier))
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY memory_id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [
            MemorySummary(
                memory_id=str(row["memory_id"]),
                kind=_enum(MemoryKind, row["kind"], row["memory_id"], "kind"),
                tier=_enum(Tier, row["tier"], row["memory_id"], "tier"),
                content=str(row["content"]),
                weight=float(row["weight"]),
                pinned=bool(row["pinned"]),
                created_at=float(row["created_at"]),
                last_access_ts=float(row["last_access_ts"]),
                access_count=int(row["access_count"]),
                session_id=str(row["session_id"] or ""),
                source_ref=str(row["source_ref"] or ""),
            )
            for row in rows
        ]

    def pinned_items(self) -> list[MemoryItem]:
        """钉住的条目（含人格锚）。装配时它们进 L0，永远不被驱逐。"""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memories WHERE pinned = 1 ORDER BY created_at"
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    # ── 向量 ────────────────────────────────────────────────────────────────

    def set_embedding(self, memory_id: str, space_id: str, vector: np.ndarray) -> None:
        """写入一条记忆在某个空间里的向量。

        维度以**登记表**为准，不是以传进来的数组为准：空间先声明自己是几维，
        向量再来适配它。反过来的话，第一个写错的向量会定义整个空间的维度。
        """
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._lock:
            space = self._registry.require(space_id, flat.size)
            if not self._exists(memory_id):
                raise StoreError(
                    f"记忆 {memory_id!r} 不存在，不能写向量。先 add 条目再补向量——"
                    f"顺序反了的话，向量的所有者会是一行不存在的记录。"
                )
            self._connection.execute(
                "INSERT INTO embeddings (memory_id, space_id, dim, vec, added_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(memory_id, space_id) DO UPDATE SET "
                "dim=excluded.dim, vec=excluded.vec, added_at=excluded.added_at",
                (memory_id, space.space_id, space.dim, flat.tobytes(), now_ts()),
            )
            self._connection.commit()
            self._embedding_revision += 1

    def normalized_embedding_matrix(self, space_id: str) -> tuple[tuple[str, ...], np.ndarray]:
        """某空间的已归一化向量矩阵，按 embedding revision 缓存。

        ``recall`` 每轮都要在这个矩阵上做余弦；50k×768 的每次复制 + 归一化是
        纯浪费。缓存只在 ``set_embedding`` 之后失效。返回的 id 是**元组**：每轮
        再复制 5 万个字符串到 list 里没有意义，调用方只读。
        """
        with self._lock:
            cached = self._normalized_cache.get(space_id)
            if cached is not None and cached[0] == self._embedding_revision:
                return cached[1], cached[2]
            ids, matrix = self.embedding_matrix(space_id)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / np.maximum(norms, 1e-12)
            self._normalized_cache[space_id] = (
                self._embedding_revision,
                tuple(ids),
                normalized,
            )
            return tuple(ids), normalized

    def get_embedding(self, memory_id: str, space_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT dim, vec FROM embeddings WHERE memory_id = ? AND space_id = ?",
                (memory_id, space_id),
            ).fetchone()
            if row is None:
                return None
            dim = int(row["dim"])
            raw = row["vec"]
        # frombuffer 给出的是只读视图；复制一份，免得下游某处就地改写时抛
        # "assignment destination is read-only"——那个报错离现场很远，很难查。
        return np.frombuffer(raw, dtype=np.float32, count=dim).copy()

    def drop_embedding(self, memory_id: str, space_id: str | None = None) -> int:
        """删掉一条记忆的向量，返回删了几行。``space_id=None`` 表示**所有空间**。

        这是「编辑了正文之后」该做的那一步。**存储层不会自动做**——向量住在另一张
        表里，``update`` 是整行覆盖，两者互不知道对方的存在；而正文变了、向量还是
        旧的，是一个**静默**错误：召回会按旧意思命中它、MMR 还会拿旧向量算相似度，
        全程不报错。

        删掉之后该 id 会重新出现在 :meth:`missing_embedding_ids` 里，于是既有的
        补账管线（每轮一小批、启动时全量）会按**新正文**重新嵌入它。这就是"丢弃
        待补"的全部机制——不需要一个新的补账者。

        Returns:
            真正删掉的行数。**0 不是错误**：编辑一条从未嵌入过的记忆（比如嵌入模型
            一直没装）本来就没有向量可删。
        """
        with self._lock:
            if space_id is None:
                cursor = self._connection.execute(
                    "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
                )
            else:
                cursor = self._connection.execute(
                    "DELETE FROM embeddings WHERE memory_id = ? AND space_id = ?",
                    (memory_id, space_id),
                )
            removed = int(cursor.rowcount)
            self._connection.commit()
        return removed

    def embedded_ids(self, space_id: str) -> set[str]:
        """在 ``space_id`` 里**已经有向量**的记忆 id。

        为什么需要它而不是读 :attr:`MemoryItem.has_embedding`：那个属性看的是条目
        的 ``space_id`` 与 ``sem_sdr`` 字段，而这两个字段在库外**恒为空**——写入时
        就留空了（向量是异步补的），补向量的路径只写 ``embeddings`` 表、从不回写
        条目行。所以 ``has_embedding`` 永远返回 False，用它判断"有没有向量"会得到
        一个恒假的答案。

        一次取全集而不是逐条 :meth:`get_embedding`：列表要逐行标"缺语义编码"，
        逐条拉 blob 会把上千个向量读进内存，而这里只需要 id。
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT memory_id FROM embeddings WHERE space_id = ?", (space_id,)
            ).fetchall()
        return {str(row["memory_id"]) for row in rows}

    def items_created_between(self, since: float, until: float) -> list[MemoryItem]:
        """按 ``created_at`` 取一段时间里的条目（升序，半开区间 ``[since, until)``）。

        时间窗召回要用它——「昨天我说了什么」的检索键是时间，不是词面。

        **不做任何语义过滤**：``tier`` / ``taint`` / ``pinned`` 一律不看。「这段时间
        里有什么」是存储层的事实，「哪些该进上下文」是召回层的判断；混在一起之后
        账本就说不清是谁丢的。调用方拿全量再自己筛。

        没有为 ``created_at`` 建索引：它每轮只跑一次、库在千条量级，而加索引要动
        建表语句——本库的纪律是**库结构一字不动**（见 ``docs/memory.md``）。
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memories WHERE created_at >= ? AND created_at < ?"
                " ORDER BY created_at, memory_id",
                (float(since), float(until)),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def ids_for_session(self, session_id: str) -> list[str]:
        """某个会话里写下的全部记忆 id。

        存在的理由只有一个，但它足够硬：**这些记忆的正文此刻正躺在逐字稿里。**
        召回如果再把它们注入一遍，模型会同时看到"你刚才说过的话"和"一条提醒我你
        刚才说过那句话的记忆"——同一件事在一份上下文里出现两次，而第二次还占着
        记忆层的额度。所以要有一个地方能问出"哪些记忆来自这个会话"，让调用方把它们
        从候选里排除掉。

        ``session_id`` 上有索引（``idx_memories_session``），不建索引的话这个查询
        会随库的大小线性增长，而它每轮都要跑一次。
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT memory_id FROM memories WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def missing_embedding_ids(self, space_id: str, *, limit: int | None = None) -> list[str]:
        """在 ``space_id`` 下还没有向量的记忆 id，按创建时间从早到晚。

        「还是空的」这件事必须能被查出来：实时写入时是**先写条目、后补向量**的，
        中间隔着一次 HTTP 调用。补向量这条路只要有一步断了，库里就会永久留下一
        批只有词法通道的记忆——召回质量下降，而没有任何一处报错。
        """
        sql = (
            "SELECT m.memory_id FROM memories m "
            "LEFT JOIN embeddings e ON e.memory_id = m.memory_id AND e.space_id = ? "
            "WHERE e.memory_id IS NULL ORDER BY m.created_at"
        )
        params: tuple[Any, ...] = (space_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (space_id, int(limit))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def embedding_matrix(self, space_id: str) -> tuple[list[str], np.ndarray]:
        """某个空间的全部向量，形状 ``(n, dim)``，以及与之对齐的 id 列表。

        返回的是**一份拷贝**：召回要在这个矩阵上做 matmul，而那是在锁外面跑的
        （一次全库点积几毫秒，不该占着写锁）。n 为 0 时返回 ``(0, dim)`` 的空矩阵
        ——调用方不必为此写分支。
        """
        with self._lock:
            space = self._registry.get(space_id)
            rows = self._connection.execute(
                "SELECT memory_id, dim, vec FROM embeddings WHERE space_id = ? ORDER BY memory_id",
                (space_id,),
            ).fetchall()

        ids: list[str] = []
        vectors = np.zeros((len(rows), space.dim), dtype=np.float32)
        for index, row in enumerate(rows):
            dim = int(row["dim"])
            if dim != space.dim:
                # 只可能来自"空间标识没跟着模型一起换"这种人为错误。
                raise StoreCorruptError(
                    f"记忆 {row['memory_id']!r} 在空间 {space_id!r} 里的向量是 {dim} 维，"
                    f"而空间登记的是 {space.dim} 维。库被写坏了——多半是同一份向量被"
                    f"当成两个不同模型的输出写了进去。"
                )
            ids.append(str(row["memory_id"]))
            vectors[index] = np.frombuffer(row["vec"], dtype=np.float32, count=dim)
        return ids, vectors

    def count_embeddings(self, space_id: str) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE space_id = ?", (space_id,)
                ).fetchone()[0]
            )

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def apply_decay(
        self, item: MemoryItem, at: float, *, half_life_days: float | None = None
    ) -> MemoryItem:
        """结算惰性衰减并（必要时）写回。返回结算后的条目。

        **不需要任何后台任务**：权重的衰减是"读的时候顺便算一下"，公式
        ``w ← w·exp(-ln2·Δt/τ)`` 只依赖 ``last_decay_ts`` 和当前时刻。这直接解决了
        "常驻守护进程"那类设计里最烦人的问题——进程没跑的那段时间记忆该不该衰减？
        在这里，答案自动是"该，因为 Δt 就是两个时间戳之差"。
        """
        if half_life_days is None:
            half_life_days = item.tier.half_life_days
        settled = item.decayed_weight(at, half_life_days=half_life_days)
        if settled >= item.weight - _DECAY_EPSILON:
            # 变化太小，不写回。但仍然把这个更精确的值交给调用方——它只影响本轮
            # 排序，不影响库里的字节。
            return item.copy(weight=settled)
        decayed = item.copy(weight=settled, last_decay_ts=at)
        with self._lock:
            self._connection.execute(
                "UPDATE memories SET weight = ?, last_decay_ts = ? WHERE memory_id = ?",
                (decayed.weight, decayed.last_decay_ts, decayed.memory_id),
            )
            self._connection.commit()
        return decayed

    def touch_access(self, memory_ids: list[str], at: float) -> None:
        """记一次"被召回"。批量更新，一条 SQL。

        逐条 UPDATE 的话，一次注入 8 条记忆就是 8 次 commit——而它们本来是一件事。
        """
        if not memory_ids:
            return
        with self._lock:
            self._connection.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_access_ts = ? "
                "WHERE memory_id = ?",
                [(at, memory_id) for memory_id in memory_ids],
            )
            self._connection.commit()

    def snapshot_to(self, path: Path | str) -> Path:
        """把整个库**在线**复制到 ``path``，返回写成的路径。

        **用 SQLite 的在线备份 API，不是 ``shutil.copy``。** 库跑在 WAL 模式
        （见 :meth:`_configure`）：裸拷 ``mem.db`` 会漏掉 ``-wal`` 里**尚未检查点**
        的提交，而巩固恰恰是在一个刚被写过的库上拍快照——漏掉的正好是最新的那几条，
        也正是最需要保住的那几条。``Connection.backup`` 走的是同一套读事务，写出去
        的是一份自洽的库。

        父目录在这里建（调用方给的是 ``paths.snapshot_dir()`` 拼出来的名字，那个
        函数只解析位置），目标已存在时**覆盖**——同名快照的时间戳精确到秒，撞名
        意味着同一秒里跑了两轮，那种情况下后一份才是该留的。

        在锁内做：备份期间同一个连接上不能有别的写。备份耗时与库大小成正比
        （几百 MB 量级是秒级），而巩固本来就是个后台任务，这里不必再优化。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            destination = sqlite3.connect(str(target))
            try:
                self._connection.backup(destination)
            finally:
                destination.close()
        return target

    def bump_reinforce(self, memory_ids: list[str]) -> None:
        """睡眠巩固里"回放且认出了自己"的那些条目，计数 +1。批量更新，一条 SQL。

        与 :meth:`touch_access` 同形，但**刻意只差三处，每一处都是有意的**：

        * **没有 ``at`` 参数**。``reinforce_count`` 表上没有配套的时间戳列，而这里
          也不该去写 ``last_access_ts``——巩固不是"被外部问到"，那是
          :meth:`touch_access` 的语义。往访问时间上写等于把两条正交的轴混起来：
          惰性衰减以 ``last_access_ts`` 为起点，巩固顺手"刷新"它，会让一条**从来
          没被召回过的**记忆看起来刚刚被用过，衰减因此不再发生。
        * **只动 ``reinforce_count``**，不动 ``weight``。
          :attr:`~memcore.types.MemoryItem.reinforce_count` 的 docstring 明说它与
          ``access_count`` 分开记，是**独立证据字段**，不是权重的输入。
        * **``+ 1`` 写在 SQL 里**，不是读出来加一再写回。一次巩固可能加固几百条，
          读-改-写在批量场景下既慢又要把竞态防一遍——而 SQL 层的自增天然没有竞态。

        空列表直接返回：一轮里一条都没认出来是可能的（冷启动的库上就是这样），
        那不该付出一次空事务的代价。
        """
        if not memory_ids:
            return
        with self._lock:
            self._connection.executemany(
                "UPDATE memories SET reinforce_count = reinforce_count + 1 WHERE memory_id = ?",
                [(memory_id,) for memory_id in memory_ids],
            )
            self._connection.commit()

    def set_tiers(self, rows: Sequence[tuple[str, Tier]]) -> int:
        """批量改层级，返回真的写了几行。

        **只动 ``tier`` 这一列**——与 :meth:`bump_reinforce` 同一条纪律（那个只动
        ``reinforce_count``）。它不改正文、不改权重、不改任何时间戳。

        空列表直接返回（同 ``bump_reinforce`` 那条理由：空事务也要付代价）。
        写不存在的 id 不会报错——记忆可能在这一轮跑的过程中被用户删掉了，而那不该
        让整轮巩固失败。
        """
        if not rows:
            return 0
        with self._lock:
            before = self._connection.total_changes
            self._connection.executemany(
                "UPDATE memories SET tier = ? WHERE memory_id = ?",
                [(tier.value, memory_id) for memory_id, tier in rows],
            )
            self._connection.commit()
            return self._connection.total_changes - before

    def reinforce_stats(self) -> tuple[int, int]:
        """``(被加固过的记忆条数, 加固总次数)``。**空库是 ``(0, 0)``。**

        一条聚合 SQL，不把行捞出来自己数：``COUNT`` 与 ``SUM`` 走同一趟全表扫描，
        比"先载入五万条再 ``sum(item.reinforce_count ...)``"少一次约 6.9 KB/条的
        反序列化——而记忆窗口每开一次都要问这一句。

        ``WHERE reinforce_count > 0`` 那一条**不是优化**。不加的话空库返回的是
        ``(总条数, 0)``，而调用方（热度面板）第一件事就是拿第一个数判"有没有东西
        可写"：**"零条被加固过"必须表现为 ``0``，不能表现为"库里有五万条"**，
        否则那段文案会在一张从没跑过巩固的库上说出关于五万条的话。

        ``COALESCE`` 挡的是 ``SUM`` 在没有匹配行时返回的 ``NULL``——它不是零，
        而是 Python 侧 ``int(None)`` 的一个 ``TypeError``。
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(reinforce_count), 0) "
                "FROM memories WHERE reinforce_count > 0"
            ).fetchone()
        return int(row[0]), int(row[1])

    # ── 行 ↔ 对象 ───────────────────────────────────────────────────────────

    def _item_to_row(self, item: MemoryItem) -> dict[str, Any]:
        try:
            meta_json = json.dumps(item.meta, ensure_ascii=False, sort_keys=False)
        except (TypeError, ValueError) as exc:
            raise StoreError(
                f"记忆 {item.memory_id!r} 的 meta 不是 JSON 可表示的：{exc}。"
                f"meta 按原样存进 SQLite 的 TEXT 列，所以它只能装 JSON 认得的类型"
                f"（对象、数组、字符串、数字、布尔、null）——tuple 会变成 list，"
                f"自定义对象与 numpy 标量则直接拒绝。"
            ) from exc
        encoded = meta_json.encode("utf-8")
        if len(encoded) > _MAX_META_BYTES:
            raise StoreError(
                f"记忆 {item.memory_id!r} 的 meta 有 {len(encoded)} 字节，超过上限 "
                f"{_MAX_META_BYTES}。meta 是给「随条目走的少量附加信息」用的，"
                f"大块正文（例如工具任务的全文）应当另想办法，而不是塞进这里。"
            )

        lex = item.lex_sdr
        return {
            "memory_id": item.memory_id,
            "kind": str(item.kind),
            "tier": str(item.tier),
            "content": item.content,
            "taint": str(item.taint),
            "created_at": float(item.created_at),
            "last_access_ts": float(item.last_access_ts),
            "last_decay_ts": float(item.last_decay_ts),
            "access_count": int(item.access_count),
            "reinforce_count": int(item.reinforce_count),
            "weight": float(item.weight),
            "pinned": 1 if item.pinned else 0,
            "space_id": item.space_id,
            "session_id": item.session_id,
            "source_ref": item.source_ref,
            "meta_json": meta_json,
            "lex_idx": None if lex is None else lex.idx.tobytes(),
            "lex_w": None if lex is None else lex.weights.tobytes(),
            "lex_width": None if lex is None else int(lex.width),
        }

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        lex = _blob_to_sdr(row["lex_idx"], row["lex_w"], row["lex_width"], key="lex_sdr")
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise StoreCorruptError(
                f"记忆 {row['memory_id']!r} 的 meta_json 不是合法 JSON：{exc}。"
                f"这一行是坏数据，继续读下去只会把损坏扩散到召回结果里。"
            ) from exc
        if not isinstance(meta, dict):
            raise StoreCorruptError(
                f"记忆 {row['memory_id']!r} 的 meta 反序列化后是 {type(meta).__name__}，"
                f"不是对象。meta 永远是一个对象——即使它是空的。"
            )

        return MemoryItem(
            memory_id=str(row["memory_id"]),
            kind=_enum(MemoryKind, row["kind"], row["memory_id"], "kind"),
            tier=_enum(Tier, row["tier"], row["memory_id"], "tier"),
            content=str(row["content"]),
            taint=_enum(Taint, row["taint"], row["memory_id"], "taint"),
            created_at=float(row["created_at"]),
            last_access_ts=float(row["last_access_ts"]),
            last_decay_ts=float(row["last_decay_ts"]),
            access_count=int(row["access_count"]),
            reinforce_count=int(row["reinforce_count"]),
            weight=float(row["weight"]),
            pinned=bool(row["pinned"]),
            lex_sdr=lex,
            space_id=str(row["space_id"] or ""),
            session_id=str(row["session_id"] or ""),
            source_ref=str(row["source_ref"] or ""),
            meta=meta,
        )


def _like_pattern(query: str) -> str:
    """把用户输入的搜索串变成 ``LIKE`` 模式，**转义通配符**。

    ``LIKE`` 里 ``%`` 与 ``_`` 是有含义的。不转义的话，搜「50%」会变成"以 50 开头
    的任意串"，搜「a_b」会匹配到「axb」——两种都是"结果比你要的多"，而多出来的
    那些看起来完全正常，没人会去怀疑。
    """
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _enum(enum_type: type, value: Any, memory_id: str, field_name: str) -> Any:
    """把库里的字符串还原成枚举，不认得的取值当场报错。

    **不能静默回退到默认值。** 一个来自未来版本、或者被手改坏了的 tier 值，如果
    被悄悄当成 ``WORKING``，那条记忆就会以一个错误的层级参与召回——既不下沉也
    不归档，而且没有任何线索指向"它的 tier 根本没被读懂"。
    """
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)  # type: ignore[attr-defined]
        raise StoreCorruptError(
            f"记忆 {memory_id!r} 的 {field_name} 是 {value!r}，不在合法取值里（{allowed}）。"
            f"这行数据来自更新的版本，或者被外部改坏了。"
        ) from exc


def _blob_to_sdr(idx_blob: Any, weight_blob: Any, width: Any, *, key: str) -> SDR | None:
    """从三个列还原一个 SDR。三列同为 NULL 表示这条记忆没有这个编码。"""
    if idx_blob is None and weight_blob is None and width is None:
        return None
    if idx_blob is None or weight_blob is None or width is None:
        raise StoreCorruptError(
            f"{key} 的三列没有同时有值（idx={idx_blob is not None}, "
            f"weights={weight_blob is not None}, width={width is not None}）。"
            f"半个 SDR 比没有 SDR 更糟——它会以「有编码」的样子参与相似度计算。"
        )
    return SDR(
        idx=np.frombuffer(bytes(idx_blob), dtype=np.int32),
        weights=np.frombuffer(bytes(weight_blob), dtype=np.float32),
        width=int(width),
    )
