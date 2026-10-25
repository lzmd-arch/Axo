"""L3「涌现触发」的接口桩：**只提议，不行动**。

## 它是什么 / 不是什么

计划书与纪要里的 L3 是「SNN daemon 涌现触发（好奇心动作）」。本模块只提供**接口**：

* daemon 在涌现时调用 :meth:`EmergentRetrieval.propose`；
* proposer 返回一个 :class:`RetrievalIntent`（查询、理由、当时的 novelty/activity），
  或者 ``None``；
* daemon 把意图写进 ``bus/events.jsonl``（kind = ``retrieval_intent``），**不抓取、
  不写记忆、不调用任何工具**。

所以它是「机制化主动，涌现待验」的挂点，不是「她已经有好奇心」的证据。

## 默认关闭

``BusDaemonConfig.curiosity_enabled`` 默认 ``False``。打开时必须注入一个 proposer
（否则构造就报错）——没有实现的"开"只会写出一条空事件，比关着更误导。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["DisabledEmergentRetrieval", "EmergentRetrieval", "RetrievalIntent"]


@dataclass(frozen=True)
class RetrievalIntent:
    """一次涌现触发**提议**的检索意图。它还没有被执行。

    Attributes:
        query: 想查什么。**只作提议**，执行与否由未来的执行器决定。
        reason: 一句给日志/界面看的理由。
        novelty: 触发时的 novelty（原样记录，方便回看阈值）。
        activity: 触发时的活动量。
    """

    query: str
    reason: str = ""
    novelty: float = 0.0
    activity: float = 0.0

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("RetrievalIntent.query 不能为空。")


@runtime_checkable
class EmergentRetrieval(Protocol):
    """涌现触发的提议器。**只读**：拿到的只有当前轮的量与最近几条记忆 id。"""

    def propose(
        self, *, novelty: float, activity: float, memories: tuple[str, ...]
    ) -> RetrievalIntent | None:
        """返回一个意图或 ``None``。**实现方不许在这里抓取或写库。**"""
        ...


class DisabledEmergentRetrieval:
    """永远不提议。默认配置下 daemon 用的是"根本不调用"，这个类只给测试/占位用。"""

    def propose(
        self,
        *,
        novelty: float,  # noqa: ARG002 — 协议签名，占位实现刻意不用
        activity: float,  # noqa: ARG002
        memories: tuple[str, ...],  # noqa: ARG002
    ) -> RetrievalIntent | None:
        return None
