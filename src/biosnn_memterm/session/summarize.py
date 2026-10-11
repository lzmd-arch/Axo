"""被挤出上下文的历史，压成一段可注入的文本。

对应分层装配的 L4。当会话长到装不下时，最早的那几轮会被驱逐；驱逐之后模型对
那段历史就完全失忆了——包括**这一轮会话的前提**（用户第一次说了什么）。这个模块
就是那道保险：把被挤出去的部分压成一段短的、确定性的摘要，重新放回上下文。

**它是抽取式的，不是生成式的。** 这个模块不调任何模型，理由有两条，第二条更硬：

1. 每轮都要算一次，调模型会让"发一条消息"变成两次推理；
2. **调模型做摘要，摘要里就会混进模型的幻觉**——而摘要会被当成事实注入，
   接着被写进记忆库。一个"看起来完全合理"的错误摘要是最难发现的一类错，
   它在逐字稿里没有对应物可供核对。

**不做增量状态。** 每一步都从被挤出的那段**整体重算**，而不是维护一个
"上次摘要 + 新腾出来的几条"的滚动状态。增量版看起来更省，但它会在崩溃、
会话回滚、手动删消息之后与实际逐字稿分叉，而分叉的摘要没有任何症状——它只是
慢慢地和逐字稿说的不是一回事。

**确定性**：同样的输入永远得到同样的输出。这样"这一轮为什么注入这段文字"
是可复现的，账本才有意义。
"""

from __future__ import annotations

from collections.abc import Sequence

from .budget import TokenMeter
from .store import Message

__all__ = ["MAX_LINE_CHARS", "first_sentence", "summarize"]

#: 每个发言最多取多少字符。**上限是必须的**：一条很长的助手回复会把整个摘要的
#: 额度吃光，而摘要是按"覆盖了多少轮"计价的，不是按"某一段写得多细"。
MAX_LINE_CHARS = 120

#: 分句标点。**刻意不含英文句点**：``0.40.2``、``recall.py``、``README.md``
#: 里全是句点，按它切会把版本号和路径切成碎片——而碎片拼出来的"摘要"看起来像
#: 一句完整的话，内容却已经错了。中文分号收进来是因为长回复里它常充当分句。
_SENTENCE_ENDERS = "。！？!?；;"

_HEADER = "【更早的对话（{total} 条，已压缩）】"
_FOOTER = "…另有 {skipped} 条未列出"

#: 发言者的显示名。``system`` 不出现——系统行是装配器自己拼的，不是历史。
_SPEAKER = {"user": "用户", "assistant": "助手", "tool": "工具"}


def first_sentence(text: str, *, limit: int = MAX_LINE_CHARS) -> str:
    """取第一句，超出 ``limit`` 就截断。

    先折叠空白再找句读，因为换行在逐字稿里只是排版——它不该让一句话变成两句话的
    代价（也不该被当成句子边界，否则每一行软换行都会截一次）。
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""

    cut = len(collapsed)
    for ender in _SENTENCE_ENDERS:
        found = collapsed.find(ender)
        if found != -1 and found + 1 < cut:
            cut = found + 1
    # 一句太长时不硬切在标点上：截断处加省略号，让"这里被我切了"是看得见的。
    if cut > limit:
        return collapsed[:limit].rstrip() + "…"
    return collapsed[:cut]


def _line(message: Message) -> str:
    """一条消息在摘要里的样子。空内容返回空串（调用方会跳过）。"""
    text = first_sentence(message.content)
    if not text:
        return ""
    return f"{_SPEAKER.get(message.role, message.role)}：{text}"


def summarize(
    messages: Sequence[Message],
    *,
    budget_tokens: int,
    meter: TokenMeter | None = None,
) -> str:
    """把一段历史压成摘要。放不下时返回空串。

    ``budget_tokens`` 是这段**文本**的额度，**不含**它作为一条消息的包装开销——
    包装由装配器另记（它才知道这条会以什么角色发出去）。混在一起会让每次装配都
    差 5 个 token，而 5 个 token 的偏差正是那种"永远对不上账"的账本。

    挑选规则，按优先级：

    1. **第一条永远优先**——它是这一轮会话的前提，丢了它后面全都在答一个没人
       问过的问题；
    2. 然后**从最近往前**扫，装得下的就装。装不下某一条时**跳过它继续往前**，
       而不是就此收手：每行已经被 :data:`MAX_LINE_CHARS` 封顶，大小可比，
       因为一条超长的回复就丢掉它之前的所有内容，代价远大于摘要有几个洞。
    3. 最后接一行"另有 N 条未列出"。**漏掉这件事必须写出来**——不写的话，
       摘要读起来像是完整的，模型就会把"没提到"当成"没发生过"。
    """
    if not messages or budget_tokens <= 0:
        return ""

    meter = meter or TokenMeter()
    lines = [_line(message) for message in messages]
    kept = [(index, text) for index, text in enumerate(lines) if text]
    if not kept:
        return ""

    header = _HEADER.format(total=len(kept))
    # 页脚按**最坏情况**预留：它自己要占几个 token 取决于漏掉几条，而那个数要等
    # 挑完才知道。按总条数预留是个上界（位数不会更多），多留的几位无关紧要。
    ceiling = meter.text(_FOOTER.format(skipped=len(kept)))
    left = budget_tokens - meter.text(header) - ceiling
    if left <= 0:
        # 连表头都放不下。返回空串而不是半句话：装配器会跳过这一层，
        # 而一段被砍掉头的摘要比没有摘要更容易被误读成"历史就这么长"。
        return ""

    chosen: set[int] = set()
    premise_cost = meter.text(kept[0][1])
    if premise_cost <= left:
        chosen.add(kept[0][0])
        left -= premise_cost

    for index, text in reversed(kept[1:]):
        cost = meter.text(text)
        if cost <= left:
            chosen.add(index)
            left -= cost

    if not chosen:
        return ""

    body = [lines[index] for index in sorted(chosen)]
    skipped = len(kept) - len(chosen)
    if skipped:
        body.append(_FOOTER.format(skipped=skipped))
    return "\n".join([header, *body])
