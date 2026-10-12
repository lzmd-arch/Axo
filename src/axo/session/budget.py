"""装配的 token 算术。

这个模块刻意只有**算术**：一段文本、一条消息、一串消息各值多少 token，以及这次
请求还剩多少额度。

**分层配额的表格不在这里**，它在 :mod:`~axo.session.assemble`——配额的
每一项都对应装配器里的一个槽，分开放会出现"表改了、槽没改"这种最难发现的不一致
（预算看起来还在生效，只是分配给了不存在的东西）。

四个常数全是本机（Ollama 0.40.2 + Qwen3-4B）量出来的，量法见
``tools/calibrate_tokens.py --measure-base-price``。**它们不是估计**：同文本三请求
的差值能把它们解出来，而"每条消息开销"这个东西一旦靠猜，装配层能放几轮对话就
变成了运气。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from memcore.tokens import UNDER_COUNT_TOLERANCE, estimate_tokens

from ..llm.base import ChatMessage

__all__ = [
    "CONVERSATION_BASE",
    "DEFAULT_RESERVE_OUTPUT",
    "GENERATION_PROMPT",
    "IMAGE_TOKENS",
    "PER_MESSAGE_OVERHEAD",
    "TokenBudget",
    "TokenMeter",
]

PER_MESSAGE_OVERHEAD = 5
"""每条消息的模板包装。

量法（同文本 ``F``，三个请求的差值）::

    A = [user: F]              B = [user: F, user: F]        C = [user: F+F]
    tokens(F) = C − A = 48            overhead = (B − A) − 48 = 5

换过一批**互不相同**的文本重测，仍是 5——同一文本的重复有可能被当作特殊情况，
而真实历史是内容各异的。顺带量到两条**不用再猜**的事实：

* ``system`` 与 ``user`` 的包装**同价**（``[user F, system F]`` 减 ``[user F, user F]`` 为 0）；
* ``assistant`` 更便宜（交替角色实测 3.33/条）。

所以 5 对每个角色都成立且偏保守。**保守的方向是对的**：包多算一条消息的代价是
少注入一条记忆，包少算的代价是服务端静默截断最早的消息。
"""

GENERATION_PROMPT = 7
"""生成提示（assistant 回合的开场标记）。量法：``A − tokens(F) − 每条包装``。

它每轮对话只出现一次，所以属于 :data:`CONVERSATION_BASE` 而不是每条消息。
"""

CONVERSATION_BASE = PER_MESSAGE_OVERHEAD + GENERATION_PROMPT
"""一个非空对话的固定底价：12。

``A = 60`` 而 ``tokens(F) = 48``，差的 12 就是它（包装 5 + 生成提示 7）。
**注意它不是"每条消息 12"**——那个数是 5，而把两者混起来的后果是预算被高估
一倍多，然后按层反向驱逐会把真正该留的记忆先赶走。

于是整串消息的代价是::

    GENERATION_PROMPT + Σ(内容 + PER_MESSAGE_OVERHEAD)

第一条消息的包装已经含在 :data:`CONVERSATION_BASE` 里，不再另算——
``7 + (5 + c₁) = 12 + c₁`` 与实测的 ``A`` 逐 token 相等。
"""

DEFAULT_RESERVE_OUTPUT = 2048
"""给生成留的额度。

**这个数必须真的传给请求**（``ChatRequest.max_tokens``），否则它只是账面上的
预留：``max_tokens`` 为 ``None`` 时 Ollama 会用同一个 ``num_ctx`` 生成，输出可以
一直涨到把输入挤掉。装配器负责把两者绑在一起。
"""


IMAGE_TOKENS = 1536
"""一条图在上下文里的代价。**先验值——没有标定。**

**它与上面那四个不是一类东西，别混着读。** 那四个是差值法**量**出来的（同一个文本发三次，
差出来的就是包装），而且四条互相印证；这一个本机量不了：**没有能看图的模型**（`qwen3-4b`
的 capabilities 里没有 `vision`）。量法已经有了，只差一个模型：同一个文本发两次，一次带图
一次不带，比对服务端回的 `prompt_eval_count` —— 与 `tools/calibrate_tokens.py` 的差值法
同一套。

取 1536 的理由只有一条：一屏文字缩到长边 1600 之后，按 28×28 分块大致落在这一档；而它
**偏保守**（同 :data:`PER_MESSAGE_OVERHEAD` 那句——包多算的代价是少注入一条记忆，包少算的
代价是服务端静默截断最早的消息）。

**上限是硬的。** 它必须明显小于 :func:`~axo.session.assemble.layer_ceiling` 给 L5 的额度
（本机 ``num_ctx`` 16384 时约 2.8k），否则一张图会把整个工具尾巴挤掉——而"挤掉"在服务端
那侧表现出来是**静默截断最早的消息**，正是装配器与循环两处文档反复点名的最坏形态。
``tests/test_budget.py`` 有一条断言钉着这件事。
"""


@dataclass(frozen=True)
class TokenMeter:
    """一处算清所有 token 代价。

    单点封装的理由与 :func:`memcore.text.is_cjk` 单独成模块相同：账本说 184 token
    而实际发出去 240，根因**总是**两处对同一段文本用了不同的算法。这个类保证只有
    一处。
    """

    per_message_overhead: int = PER_MESSAGE_OVERHEAD

    def text(self, text: str) -> int:
        """一段裸文本的代价（不含任何包装）。"""
        return estimate_tokens(text)

    def message(self, message: ChatMessage) -> int:
        """一条消息的代价，**含**它的模板包装。

        工具调用也要算：回灌给模型的 ``tool_calls`` 是 JSON 形状的正文，它和
        ``content`` 一样占位置。P1 的装配器构造的消息恒为空，这条分支到 P3 才有
        非零值——但**现在就得对**，否则 P3 的第一轮工具调用会发现预算莫名超了。
        """
        total = estimate_tokens(message.content) + self.per_message_overhead
        if message.tool_calls:
            total += self.tool_calls(message.tool_calls)
        if message.images:
            # 按张算，不按字节：一张 1600 宽的截图与一张 800 宽的差在**分块数**上，
            # 而那与 base64 长度不是线性的（PNG 的压缩率取决于画面内容）。用字节数
            # 折算会得到一个"文字多一点的截图反而更便宜"的结论。
            total += IMAGE_TOKENS * len(message.images)
        return total

    def tool_calls(self, calls: Iterable[object]) -> int:
        """工具调用的代价，按 JSON 正文估。

        用估计器算序列化后的 JSON，而不是给一个专门的系数：JSON 的字符构成
        （引号、冒号、逗号、标识符、数字）本来就是估计器标定过的那些档。

        **这个数仍然是推理出来的，不是量出来的——而且现在能说清差在哪一步。**
        Step 0 的探针（``tools/smoke_agent.py``，结论在它自己的模块 docstring 里）
        量的是**工具声明的固定开销**：3 个工具净增 311 token/轮，七个约两倍。那一项
        现在走 :attr:`~axo.session.assemble.AssemblyLedger.tools_tokens`。
        它**没有**单独量这里——声明、``tool_calls``、结果正文是同一轮里一起出现的，
        探针只把声明那一份切了出来（两种不同口径的份额，混在一个数里谁也核对不出来）。

        量法还是同一套差值法（见 :data:`PER_MESSAGE_OVERHEAD`），只是把变量换成这段
        JSON：两次请求只差"同一段文本放在 ``content`` 里还是放在 ``tool_calls`` 里"，
        差值就是两者计价之比——注意 ``assistant`` 的模板包装更便宜，要按那条注记折算。
        **在量出来之前别把它当已知量**：偏差的方向和大小都还没人看过，而它每一轮
        工具调用都要付一次。
        """
        if not calls:
            return 0
        return estimate_tokens(json.dumps(list(calls), ensure_ascii=False, default=str))

    def messages(self, messages: Sequence[ChatMessage]) -> int:
        """整串消息的代价，**含**整轮对话的固定底价。

        空序列返回 0 而不是 :data:`CONVERSATION_BASE`：没有消息就没有请求，
        而"空请求要花 12 token"会让调用方在"这一轮到底发不发"的判断上被一个
        不存在的开销误导。
        """
        if not messages:
            return 0
        return GENERATION_PROMPT + sum(self.message(message) for message in messages)


@dataclass(frozen=True)
class TokenBudget:
    """一次请求的额度。

    三层数，从外到内::

        window        模型的 num_ctx（P0 起是问 /api/show 要的，不是猜的）
        input_limit   window − reserve_output。**超过它服务端会静默截断最早的消息**
        usable        input_limit × (1 − tolerance)。装配器实际可以支配的估算额度

    ``usable`` 为什么除一个系数而不是直接等于 ``input_limit``：估计器承诺的低估
    上界是 :data:`memcore.tokens.UNDER_COUNT_TOLERANCE`，也就是估计值 ``E`` 对应的
    真实值最坏是 ``E/(1−tolerance)``。要让真实值不超过 ``input_limit``，就得让
    ``E ≤ input_limit×(1−tolerance)``。**这不是拍出来的余量，是从承诺反推的**——
    所以它随 :data:`~memcore.tokens.UNDER_COUNT_TOLERANCE` 一起变。

    实测中位比值是 1.04（估计偏高），所以绝大多数请求用不满这个额度；余量只为
    那 5/20 个偏低样本存在。两种错的代价完全不对称：少注入一条记忆无所谓，
    丢掉人格锚不是。
    """

    window: int
    reserve_output: int = DEFAULT_RESERVE_OUTPUT
    tolerance: float = UNDER_COUNT_TOLERANCE

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window 必须是正数，收到 {self.window}。")
        if self.reserve_output < 0:
            raise ValueError(f"reserve_output 不能为负，收到 {self.reserve_output}。")
        if self.window <= self.reserve_output:
            raise ValueError(
                f"window={self.window} 不比 reserve_output={self.reserve_output} 大，"
                f"输入额度是 0 或负数。这个组合下装配器只能发一条空消息。"
            )
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(
                f"tolerance 必须在 [0, 1) 内，收到 {self.tolerance}。取 1 会让 usable 恒为 0。"
            )

    @property
    def input_limit(self) -> int:
        """服务端接受的输入上限。**超过它是静默截断，不是报错。**"""
        return self.window - self.reserve_output

    @property
    def usable(self) -> int:
        """装配器可以支配的**估算**额度。"""
        return int(self.input_limit * (1.0 - self.tolerance))

    def to_dict(self) -> dict[str, int | float]:
        """账本里记一份，好让"这一轮为什么只注入了三条"可核对。"""
        return {
            "window": self.window,
            "reserve_output": self.reserve_output,
            "input_limit": self.input_limit,
            "tolerance": self.tolerance,
            "usable": self.usable,
        }
