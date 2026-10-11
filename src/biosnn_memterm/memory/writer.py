"""把策略的裁决落到库里 —— 唯一写记忆的地方。

分工：:mod:`~biosnn_memterm.memory.policy` 决定**记什么**（纯函数，没有库也能测），
这里决定**怎么存**（ULID、SDR、溯源指针、库连接）。

**失败要炸，不要吞。** 别的地方吞异常是对的（一次工具调用失败不该中断任务），
但这里不行：这个项目的全部意义就是"记忆不会丢"，而唯一比丢记忆更糟的是**丢了却
不知道**——用户会以为它记住了，直到某天问起来答不出。所以 :meth:`MemoryWriter.record`
让异常往上走，由界面层显示出来。

**这不是一个后台任务。** 用户选的"每轮实时写入"要的就是同步：这一轮说过的话，
在这一轮结束时就该在库里。异步写会让"刚说完就重启"丢掉最后一句——而那恰好是
最容易被用户测试的那一句。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from memcore import MemoryItem, Tier, now_ts
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import SDRCapacityError

from .policy import MemoryDraft, MemoryPolicy, PerItemPolicy, Turn

__all__ = ["MemoryWriter"]


class MemoryWriter:
    """按策略把一轮对话记进库。

    Args:
        store: 记忆库。
        policy: 决定这一轮记什么。换掉它就换了粒度——P3 的 agent 开关就是这么
            实现的（见 :class:`~biosnn_memterm.memory.policy.MemoryPolicy`）。
        clock: 取时间。可注入是为了让"写进去的那条权重是多少"在测试里可复现；
            生产路径用 :func:`memcore.now_ts`。
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: MemoryPolicy | None = None,
        clock=None,
    ) -> None:
        self._store = store
        self.policy: MemoryPolicy = policy or PerItemPolicy()
        self._clock = clock or now_ts

    def record(self, turn: Turn) -> list[MemoryItem]:
        """记下这一轮。返回真正入库的条目（可能是空的）。

        返回列表而不是计数，因为调用方要拿它做两件事：写进注入账本（"这一轮记下了
        什么"），以及回填 embedding。两者都需要 id。
        """
        written: list[MemoryItem] = []
        for draft in self.policy.drafts(turn):
            item = self.record_direct(
                draft, session_id=turn.session_id, source_ref=self.source_ref(turn)
            )
            if item is None:
                continue
            written.append(item)
        return written

    def record_direct(
        self, draft: MemoryDraft, *, session_id: str, source_ref: str
    ) -> MemoryItem | None:
        """按一条**现成的**草稿入库，溯源指针由调用方给。返回入库的条目，编不出
        词法指纹时返回 ``None``。

        P3a 的 ``write_memory`` 工具走这条路：它的内容不来自策略（那是"这一轮对话
        留下什么"的规则），而来自模型的一次显式请求，用户点过头。**但它仍然借
        :meth:`_build`**——ULID、SDR 编码、``SDRCapacityError`` 的兜底都只有一份。
        让工具自己拼一条 :class:`~memcore.MemoryItem` 也能跑，代价是那三件事各多
        一个副本，而"记忆库里有一条编不出指纹的哑弹"正是 ``_build`` 里那段注释
        要防的东西。

        ``source_ref`` 由调用方给而不是在这里算，是因为两种写入的定位成分不同：
        一轮对话是 ``turn://<会话>/<轮次>``，一次工具调用还要带上 ``call_id``
        （同一轮里可以写两次）。**去重查询尚未实现**（P3a 计划 §十一 风险 5），
        所以现在这个指针只用于溯源；形状先写对，那一刻才只是加一个查询。
        """
        item = self._build(draft, session_id=session_id, source_ref=source_ref)
        return None if item is None else self._store.add(item)

    def _build(self, draft: MemoryDraft, *, session_id: str, source_ref: str) -> MemoryItem | None:
        try:
            lex = encode_lexical(draft.content)
        except SDRCapacityError:
            # 编不出 SDR 的记忆在库里是一颗哑弹：它参与不了 MMR 去冗（那一步用的
            # 就是词法 SDR），也几乎召不回来——等于占了一行却永远不出现在上下文里。
            # **跳过它，而不是让整轮写入失败**：这一轮可能还有别的草稿值得记。
            #
            # 策略层已经尽量在前面挡住了这类输入（全标点、纯附和），但那是启发式，
            # 这里是那道必须存在的兜底——而且它是**类型化的**兜底，不是靠猜。
            return None

        return MemoryItem(
            memory_id=new_ulid(),
            kind=draft.kind,
            tier=Tier.WORKING,
            content=draft.content,
            taint=draft.taint,
            created_at=self._clock(),
            weight=draft.weight,
            lex_sdr=lex,
            # ``assoc_sdr`` 留空：**消费者已经存在**（P4a 的关联图 A 与神经元级基质
            # W），但它是 ``lex_sdr`` 的纯函数——``encode_assoc(lex)`` 十几微秒，
            # 存一份派生值只会在库里多一列需要跟着 ``lex_sdr`` 一起维护的副本。
            # 重算比存便宜，也比存可靠：改了正文而漏改这一列，症状是"图里的邻居
            # 还是旧的那批"，没有任何一处会报错。要落库就得同时欠一个 schema 迁移
            # （见 ``memcore/store.py`` 里那段"两者性质不同"）。
            # ``sem_sdr``/``space_id`` 留空：embedding 是异步入库的
            # （``write`` 不调模型），回填器按 ``missing_embedding_ids`` 补。
            session_id=session_id,
            source_ref=source_ref,
            meta=dict(draft.meta),
        )

    @staticmethod
    def source_ref(turn: Turn) -> str:
        """这一轮的溯源指针，形状 ``turn://<会话 id>/<轮次>``。

        **它同时是将来去重的键。** P1 不会产生重复：一轮对话只走一次写入，没有
        "重新生成"这条路。但 P3 的 agent 循环会因为重试而再次落盘同一轮，那时
        "这条已经写过了"就必须判得出来——而判据只能是一个稳定的、从输入算出来的
        字符串。现在把它写对，那一刻就只是加一个查询。

        工具写入（:meth:`record_direct`）用的是另一种形状（``tool://…``）：它不是
        一轮的产物，是模型的一次显式请求。
        """
        return f"turn://{turn.session_id}/{turn.turn_seq}"

    def record_text(
        self,
        user_text: str,
        *,
        reply_text: str = "",
        session_id: str = "",
        turn_seq: int = 0,
        interrupted: bool = False,
        tool_trace: Sequence[Mapping[str, Any]] = (),
    ) -> list[MemoryItem]:
        """便利入口：直接给文本，不必先造一个 :class:`Turn`。

        界面层拿到的就是这几个散的值，让它在每一个调用点手搓 dataclass 只会多出
        "某处漏填了 session_id"这种错——而那会让溯源指针悄悄失效，症状是去重
        永远不命中。

        ``tool_trace`` 是 P3b 加的那一个：:class:`~biosnn_memterm.agent.trace.build_trace`
        的产物，由 ``AgentWorker`` 带出来。**默认空元组**，所以闲聊那条路径一个字都
        不用改（它也确实没改）。装的是哪套策略决定它有没有用——条目模式根本不看它。
        """
        return self.record(
            Turn(
                user_text=user_text,
                reply_text=reply_text,
                session_id=session_id,
                turn_seq=turn_seq,
                interrupted=interrupted,
                tool_trace=tuple(tool_trace),
            )
        )
