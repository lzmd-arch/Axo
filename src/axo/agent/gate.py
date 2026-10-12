"""确认闸门：什么要问、问什么、没答算不算、以及跨线程的那次握手。

文件里是两半，风险完全不同：**上面是纯策略**（能穷举测），**下面是
:class:`ApprovalSlot`**（要动 ``threading``，是本版唯一没有先例的东西，计划书 §三）。
放在一起是因为它们共用 :class:`ApprovalRequest` / :class:`ApprovalDecision`，而拆成
两个文件会让"改协议"变成两处改动；但**测试是分开的**（``test_agent_gate.py`` 里
前一半是表驱动，后一半起真线程），这样前者不会因为后者难测而跟着没人敢动。

``ApprovalSlot`` 里没有一行 Qt：它只依赖 :mod:`threading`，所以能进核心腿。
把它接到信号与卡片上的是 ``ui/agent_worker.py``（第 8 步）。

## "闸门"这个词在本项目里有两个意思

写进代码注释是为了免得后人混名：

* **记忆写入闸门**（:mod:`axo.memory.policy` 的 ``PerItemPolicy.drafts``）
  ——判"这句话值不值得记"。P1 已落地，纯规则、无 UI、每轮都在跑。
* **工具确认闸门**（本模块）——判"这个动作要不要你点头"。有 UI、跨线程、只在
  ``requires_approval`` 的工具上触发。

两者都叫"闸门"，但一个决定**记不记**，一个决定**做不做**。

## 需要确认的判据只有一处

"这个工具要不要点头"声明在 ``Tool.requires_approval`` 上，本模块只**读**它。
不在闸门里再写一张表——两张表迟早会不一致，而表现是某个改状态的工具悄悄变成了
不需要确认。**那是一种没有任何症状的失效**，所以它必须是不可能发生的形状。

## 三种收场，默认都是"不做"

| 情形 | 裁决 |
|---|---|
| 用户点了允许 | :attr:`ApprovalDecision.GRANT` |
| 用户点了拒绝 | :attr:`ApprovalDecision.DENY` |
| 超时 / 取消 / 通道没了（``None``） | :attr:`ApprovalDecision.DENY` |

**fail-closed 是唯一安全的默认。** 改变状态的工具，不确认就是不做——把"没答"
当成"同意"会让用户在一次卡顿或一次超时之后发现记忆里多了一条他从没批准过的
东西，而且没有任何地方记着那件事发生过。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..llm.base import ToolCall
from .handoff import HandoffSlot

__all__ = [
    "APPROVAL_TIMEOUT_S",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalSlot",
    "build_request",
    "resolve_decision",
    "summarize",
]

APPROVAL_TIMEOUT_S = 180.0
"""等用户点头的上限，秒。超时按**拒绝**（见 :func:`resolve_decision`）。

**这是先验值，不是量出来的**——它是个使用手感上的数，不是 token 那种能靠差值法解出
来的东西。取这么长的理由是整轮都被它挡着：用户在读卡片，而这张卡片上有倒计时和一个
「跳过」按钮（=立刻拒绝），所以真正的收场通常由用户点，不由计时器定。计时器只是给
"人走开了"这种情况的兜底。

**它不等于收尾一定会等这么久**：窗口关闭 / 按停止走的是 :meth:`ApprovalSlot.cancel`，
立即唤醒，不等超时。这一条是 ``wait_for_all`` 那个 3 秒预算能成立的前提——没有它，
一个卡在确认上的 worker 会让进程撞上 ``0xC0000409``。
"""

_ARG_PREVIEW_CHARS = 80
"""确认卡片上每个参数值最多显示多少字。

卡片是给人在一秒之内看懂的，不是给模型看的。截断的是**显示**——真正传给工具的
参数一个字节都不动（这一点很重要：显示得短与执行得少是两件事，混了会变成
"界面上看着没问题，实际写进去的是别的东西"）。
"""


class ApprovalDecision(str, Enum):
    """用户对一次工具调用的裁决。

    继承 :class:`str` 的理由与 :class:`~axo.agent.router.Register` 同源：
    它会进逐字稿与账本，能直接序列化就不必写转换表。
    """

    GRANT = "grant"
    DENY = "deny"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ApprovalRequest:
    """一次"要不要做"的问询。

    :attr:`summary` 是**模板从参数生成的**，绝不让模型写。这与
    :mod:`axo.session.assemble` 里"记忆为什么被注入"那条纪律同源：
    让模型解释自己准备做什么，是幻觉的标准配方——而这里的后果比那里重，
    用户是照着这句话点「允许」的。

    Attributes:
        call_id: 这次调用的 id。界面回传裁决时用它比对，避免把一次迟到的点击
            算到另一个调用头上。
        tool_name: 工具名。
        summary: 一句人话，说明**将要发生什么**。
        arguments: 原始参数。界面可以展开看全，但 :attr:`summary` 才是给眼睛的。
        round_index: 这一轮里第几次工具调用（从 0 数）。只用于显示。
    """

    call_id: str
    tool_name: str
    summary: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    round_index: int = 0


def _render_value(value: Any) -> str:
    """把一个参数值渲染成卡片上的一小段字。"""
    if isinstance(value, str):
        text = value
    elif value is None or isinstance(value, bool | int | float):
        text = json.dumps(value, ensure_ascii=False)
    else:
        # 结构化值（列表、对象）走 JSON：给一个"看起来像那么回事"的中文摘要会
        # 藏掉真正的形状，而用户要看的就是形状。
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) > _ARG_PREVIEW_CHARS:
        text = text[:_ARG_PREVIEW_CHARS] + "…"
    return text


def summarize(call: ToolCall, *, tool: object | None = None) -> str:
    """从参数模板化生成一句"将要发生什么"。

    **不是**"调用 write_memory"这种把工具名重复一遍的话——那对用户没有信息量。
    用户需要看到的是内容：要写的是一条什么记忆、要忘的是哪一条、要读的是哪个文件。

    **没有参数的工具是这条规则的例外**：模板什么也生成不出来（只会打印
    ``工具名（无参数）``），而"它到底要做什么"恰恰是用户点「允许」时唯一要看的东西。
    所以这类工具**自己带一句话**（``approval_summary``），这里在参数为空时用它。

    ``tool`` 是那个工具对象，用 ``getattr`` 读、**不进 ``Tool`` 协议**：那个协议是
    ``@runtime_checkable`` 的，往里加成员等于要求十个工具都长这个属性
    （``isinstance`` 只查有没有这个属性），而它只对"要说明范围"的那一个有意义。
    与 ``requires_approval`` 一样，这句话仍然是**工具自己声明的**——不是闸门里的
    第二张表。
    """
    if not call.arguments:
        stated = getattr(tool, "approval_summary", "")
        if isinstance(stated, str) and stated.strip():
            return stated.strip()
        return f"{call.name}（无参数）"
    parts = [f"{key}：{_render_value(value)}" for key, value in call.arguments.items()]
    return "；".join(parts)


def build_request(
    call: ToolCall, *, round_index: int = 0, tool: object | None = None
) -> ApprovalRequest:
    """把一次工具调用变成一次问询。

    ``tool`` 只为了 :func:`summarize` 那句"没有参数时由工具自己说"。调用方
    （``agent/loop.py``）手里有注册表，所以它给得出；不给也能跑（回落旧行为）。
    """
    return ApprovalRequest(
        call_id=call.id,
        tool_name=call.name,
        summary=summarize(call, tool=tool),
        arguments=dict(call.arguments),
        round_index=round_index,
    )


def resolve_decision(answer: ApprovalDecision | None) -> ApprovalDecision:
    """把"没有答案"归到拒绝。

    ``None`` 覆盖三种收场：超时、用户按了停止、以及将来某个还没接上的通道。
    三者对**被执行的动作**来说没有区别——都没有人同意过。合并成一条 fail-closed
    的规则，比给每种情形编一句不同的失败文案更安全：多一种情形就多一次"这次算
    同意吧"的机会。
    """
    return ApprovalDecision.DENY if answer is None else answer


class ApprovalSlot(HandoffSlot["ApprovalRequest", "ApprovalDecision"]):
    """一个待决的**确认**请求。**worker 线程在里面等，界面线程往里面答。**

    机制（为什么是槽不是队列、身份比对、单向取消、线程约定）全在
    :class:`~axo.agent.handoff.HandoffSlot` 那份模块文档里——这里只说确认这一侧特有
    的那件事：**身份就是 ``call_id``**。

    界面回传裁决时必须把 ``call_id`` 一起带回来，对不上就丢弃。拦的是"用户点得慢"这
    种最常见的时序：卡片还在屏幕上，而 worker 已经因为超时/停止走掉了、或已经进了下一
    轮。没有这道比对，那次点击会算到**别的**调用头上——一次张冠李戴的允许。

    （计划书里 :meth:`provide` 的签名只有 ``decision``；比对所需的 ``call_id`` 是它
    自己那张"pending 到达时用户已点过停止"的表要求的。签名照表来。）
    """

    def request(self, req: ApprovalRequest) -> None:
        """登记一个待决问询。**worker 线程调用**。

        身份直接取 :attr:`ApprovalRequest.call_id`：它本来就是"这一次调用"的身份证，
        而且从循环一路带到界面、再带回来，是这条链上唯一一个两边都认得的字符串。
        """
        super().request(req, token=req.call_id)
