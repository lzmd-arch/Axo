"""上下文指示灯：这一轮装配把额度用掉了多少。

主窗口顶栏那一条细进度条与它右边的「上下文 3.2k / 8.9k」，数值全部来自
:class:`~axo.session.assemble.AssemblyLedger`——**与侧栏账本同一份数据**。
这一点是刻意的：灯上写 90%、账本里写 40% 的界面无从排查，而且用户会先信灯。

这里的两个函数只做**判定与排版**，不碰 Qt：状态是 :class:`ContextState`，颜色由
界面去映射（``over``→红、``warn``→黄、``ok``→绿）。分开的好处是阈值逻辑可以
在核心腿里断言，而不是"跑起来看一眼颜色对不对"。
"""

from __future__ import annotations

from enum import Enum

__all__ = ["WARN_RATIO", "ContextState", "context_state", "format_context"]

WARN_RATIO = 0.85
"""进入「黄」的比值。

0.85 而不是 0.95：额度是**估算**额度（``TokenBudget.usable`` 已经按估计器的低估
上界留过一次余量，见 budget.py），所以真实占用比进度条显示的还要紧一些。留
15% 的提前量，是为了让「快满了」这件事在用户还能做点什么的时候出现——等它到
95% 时这一轮已经发出去了，而服务端超限是**静默截断最早的消息**（丢的正是人格锚
所在的位置）。
"""


class ContextState(str, Enum):
    """指示灯的四种状态。"""

    EMPTY = "empty"
    """还没有装配过一轮。灯是空的，不是绿的——「额度用了 0%」与「没有额度」是
    两件事，前者会让人以为一次请求正在顺利进行。"""

    OK = "ok"
    WARN = "warn"
    OVER = "over"
    """超出了 ``usable``。**这不等于请求会失败**：发出去了，只是很可能被服务端
    截掉最早的消息。所以它是红，不是错误弹窗。"""

    def __str__(self) -> str:
        return self.value


def context_state(*, total_tokens: int, usable: int, over_budget: int) -> ContextState:
    """判定当前状态。

    Args:
        total_tokens: 整轮请求的估算代价（``AssemblyLedger.total_tokens``）。
        usable: 这一轮的可支配估算额度（``TokenBudget.usable``）。
        over_budget: 超出 ``usable`` 的量（``AssemblyLedger.over_budget``）。

    判定顺序不能换：**``over_budget`` 优先于比值**。超预算但比值因为某种原因
    落在 0.85 以下（额度很小的时候 ``over_budget`` 只超一点点）时，按比值判会
    显示黄——而它已经超了。

    任一真实装配都满足 ``total_tokens >= CONVERSATION_BASE``（空消息序列才返回 0，
    而装配器不会发出空请求），所以 ``total_tokens <= 0`` 只可能是「还没装配过」。
    """
    if total_tokens <= 0:
        return ContextState.EMPTY
    if over_budget > 0:
        return ContextState.OVER
    if usable <= 0:
        # 额度算不出来（窗口与预留输出太接近）。判成 OVER 而不是 OK：这种配置下
        # 装配器基本发不出东西，「一切正常」是错的。
        return ContextState.OVER
    return ContextState.WARN if total_tokens / usable >= WARN_RATIO else ContextState.OK


def format_context(*, total_tokens: int, usable: int) -> str:
    """右上角那行字，形如 ``上下文 3.2k / 8.9k``。**如实显示，不夹紧到 100%。**

    超预算时它写成 ``上下文 12.0k / 8.9k``——比值 135% 是事实，进度条夹到 100%
    是画法的限制，两者各管各的。把文字也夹了的话，「超了多少」就没地方看了。
    """
    if total_tokens <= 0:
        return "上下文 —"
    return f"上下文 {_compact(total_tokens)} / {_compact(usable)}"


def _compact(tokens: int) -> str:
    """四位数起用 ``k``。保留一位小数——``8.9k`` 与 ``8.4k`` 的差别（约 500 token）
    正好是一条记忆的量级，抹掉它这个指标就没有分辨率了。"""
    if tokens < 1000:
        return str(tokens)
    return f"{tokens / 1000:.1f}k"
