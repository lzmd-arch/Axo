"""什么算一条记忆 —— 纯决策，不碰存储。

**这一层之所以是一个可替换的对象而不是几行 if，是为了粒度的切换。**
用户选的 agent 语义是"一次任务 = 一条记忆"，而两种模式的差别只在这里：策略换一个
实现，写入路径、存储格式、召回、衰减、SDR 编码全都不动（见计划书 §4.8）。把判断
硬编码在 ``_on_reply_finished`` 里，接缝就得挖开 P1 的核心路径。P3b 兑现了这件事：
:class:`TaskPolicy` 是第二个实现，而写手那一侧只多了一个参数。

**策略必须是纯的。** 它拿一个 :class:`Turn` 回来一串 :class:`MemoryDraft`，不读
时钟、不开库、不知道上一个会话是什么。理由有两个，第二个更实际：

1. 纯函数能穷举测——"哪句话会被记下来"是这个项目里最容易被悄悄改坏的一类行为；
2. :class:`TaskPolicy` 能在**不启动 GUI、不连模型**的情况下测（它确实是这样测的），
   那它就不能依赖任何有状态的东西。

面板上的"这条会不会被记住"因此是可以回答的，而不是需要跑一遍才知道。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from memcore import MemoryKind, Taint
from memcore.text import is_word, normalize

__all__ = [
    "ACKNOWLEDGEMENTS",
    "MAX_TASK_TEXT_CHARS",
    "MIN_MEANINGFUL_CHARS",
    "MemoryDraft",
    "MemoryPolicy",
    "PerItemPolicy",
    "TaskPolicy",
    "Turn",
]

MIN_MEANINGFUL_CHARS = 4
"""少于这么多**实义字符**（字、词、数字）的一轮不记。

"好的"、"嗯"、"?" 这一档不是记忆。把它们写进库的代价不只是浪费空间：库会退化成
一份逐字稿的副本，而每一次召回都要为每一句废话算一次余弦——**垃圾记忆不只是占据
位置，它参与竞争**，会把真正该被想起来的那条挤出 MMF 的前 8 名。

门槛取 4 而不是更高，是因为"开会周三"、"明天买牛奶"这些都是要记的。
"""

ACKNOWLEDGEMENTS = frozenset(
    {
        "好的",
        "好",
        "嗯",
        "哦",
        "噢",
        "收到",
        "继续",
        "谢谢",
        "多谢",
        "可以",
        "行",
        "对",
        "是的",
        "没错",
        "明白",
        "知道了",
        "了解",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "thx",
        "hi",
        "hello",
        "hey",
        "你好",
        "在吗",
        "早",
    }
)
"""纯附和。归一化之后**整句相等**才算命中——不做子串匹配。

子串匹配会把"好的，路由器放在书房"判成附和，而那句话正是最该被记住的一类。

**公开而不是私有**（P3a 起）：路由的规则层要用同一张表判"这轮是闲聊"，而两个
地方各存一份的话，某天补一个词（"晓得了"）只会补在其中一边。两处的**用法**不同
（这里整句相等、那里也是整句相等，但那里还排在别的规则之后），**词表是同一份
判断**：同一句话既不该被记忆、也不该被当成任务。
"""

_PREFERENCE_MARKERS = (
    "我喜欢",
    "我习惯",
    "我偏好",
    "我倾向于",
    "我希望你",
    "以后都",
    "以后请",
    "不要再",
    "别再",
    "请始终",
    "请总是",
    "尽量别",
    "麻烦你都",
)
_DECISION_MARKERS = (
    "就用",
    "就用这个",
    "决定用",
    "定下来",
    "确定了",
    "改成",
    "换成",
    "不换了",
    "就这么办",
    "按这个来",
)
"""kind 的判据是一张**标记词表**，不是分类器。

它一定会漏、也一定会误判——所以它刻意只覆盖两类**高价值且写法固定**的记忆
（偏好和决定），其余一律退回 :attr:`MemoryKind.FACT`。用一个 4B 模型来给记忆
分类会把"模型的判断"写进库，而库要独立于模型；用一张小表则错了也看得见、改起来
是一行。

``DECISION``/``CONCLUSION`` 没有对应的标记：P1 判不出来，判错比不判更糟——
一条被标成"决定"的事实会拿到更高的权重，然后长期占据召回前排。
"""

_MURMUR_DISTINCT_CHARS = 2
""""哈哈哈哈"、"嗯嗯嗯嗯"、"好的好的"这一档：**用不同的字数判，不用分词结果判。**

这个规则第一版是查"分词后只有一个不同的 token"，而它**永远不会触发**：库的分词器
同时产出单字 token 和二元组 token（``'哈哈哈哈'`` → ``c:哈`` 与 ``b:c:哈:c:哈``，
两个不同的 token）。踩这个坑的代价不是报错，是那条规则静默失效——而静默失效的
过滤器看起来和"没有这种输入"一模一样。

去掉分词依赖还顺带买到一件事：这条规则不再耦合于分词器的内部格式（``c:``/``b:``
前缀是它的实现细节）。"""

_QUESTION_ENDERS = "？?"

_QUESTION_PARTICLES = ("吗", "么")
"""句末的疑问语气词。

``呢`` **不在表里**，尽管它同样常见。它同时是持续态的句末助词——"他还在图书馆呢"、
"外面下雨呢"都是陈述——而收它的代价是给一整类日常陈述贴上"未决问题"。
"""

_QUESTION_REDUPLICATION = re.compile(r"([一-鿿]{1,2})[不没]\1")
"""正反问（``A 不 A`` / ``A 没 A``）的**形状**，不是一张词表。

这个句式是能产的："去不去"、"该不该"、"可不可以"、"认识不认识" 全是同一个形状，
列举永远补不全，而形状本身够紧——它要求同一个词在 ``不``/``没`` 两侧各出现一次。
汉语里落进这个形状的几乎只有问句，几个高频四字格都**不匹配**：``不知不觉``、
``不多不少``、``不得不``、``不管不顾`` 中间那个 ``不`` 两侧的字并不相同。
"""

_QUESTION_MARKERS = (
    "什么",
    "怎么",
    "怎样",
    "如何",
    "哪",
    "谁",
    "多少",
    "多久",
    "多大",
    "几点",
    "几号",
)
"""疑问词。

**只收词，不收"像不像在问"**：语调判断是分类器的活，而分类器会把模型的判断写进库，
库要独立于模型。``为什么`` 与 ``什么时候`` 不必单列——它们各含 ``什么``。
"""

_QUESTION_FALSE_FRIENDS = (
    # 任指/虚指：同一个疑问词在这里是"任何/某个"的意思，不是在问
    "什么都",
    "什么也",
    "没什么",
    "谁都",
    "谁也",
    "哪都",
    "哪也",
    "哪儿都",
    "哪儿也",
    "哪里都",
    "哪里也",
)
"""上面那张疑问词表的**反例**。

``什么`` 最常见的非疑问用法是任指/虚指——"我今天什么都不想干"、"我没什么"——``哪``
与 ``谁`` 同理（"哪儿都不去"、"谁都不知道"）。不收它们的话，这些每天都在说的陈述会
成片地被贴上"未决问题"，那个标签也就没有信息量了。

**它一定不全。** 这是一个黑名单，而黑名单在汉语里补不完；代价也只是某条记忆的标签
不准，不是内容出错。
"""


@dataclass(frozen=True)
class Turn:
    """一轮对话交给策略的全部材料。

    **P3b 的 :class:`TaskPolicy` 只靠这里面的字段就能工作**（它确实做到了，一个字
    都没往这个 dataclass 上加）：``tool_trace`` 是轨迹层，``reply_text`` 是摘要层
    ——用户读到的就是入库的，``user_text`` 是来历。三样各占一层，谁都不必替谁兜底。
    """

    user_text: str
    reply_text: str = ""
    session_id: str = ""
    turn_seq: int = 0
    interrupted: bool = False
    """被中断的一轮。**照常记**——"用户打断了"本身是有信息量的。

    两种模式都兑现了这半句：:class:`PerItemPolicy` 把它写进 ``meta``，
    :class:`TaskPolicy` 也是。字面上的区别在任务那一侧——一次被打断的任务，正文会
    回落到用户那句（连一个字都没吐出来的时候）。
    """

    tool_trace: tuple[Mapping[str, Any], ...] = ()
    """一次任务**做过什么**，由 :func:`~axo.agent.trace.build_trace` 压出来。

    P1 恒为空——:class:`PerItemPolicy` 不看它。P3b 起由 ``AgentWorker`` 在循环跑完
    （或被打断）之后填进来，:class:`TaskPolicy` 拿它当"这一轮是不是任务"的判据。

    **它是摘录，不是全文。** 每一步带着工具名、参数、成功与否，以及结果的**前若干
    字**——``meta`` 是逐字进 SQLite 的，而 ``memcore`` 对它的体积有硬上限，超了那条
    记忆根本写不进去（见 :mod:`~axo.agent.trace`）。全文在逐字稿里，轨迹
    只负责说明"这个回答是怎么来的"。
    """


@dataclass(frozen=True)
class MemoryDraft:
    """一条**还没进库**的记忆。

    这是策略与写手的分界：策略决定"记什么"，:class:`~axo.memory.writer.MemoryWriter`
    决定"怎么存"（ULID、SDR、时间戳、库连接）。分开之后，"该不该记这条"可以在
    没有数据库的情况下测，而"存得对不对"可以喂假策略来测。
    """

    content: str
    kind: MemoryKind = MemoryKind.FACT
    taint: Taint = Taint.USER
    weight: float = 0.5
    """新记忆的起始权重。

    刻意等于 :class:`memcore.types.MemoryItem` 的默认值——**不在这里加"看起来更
    重要"的加成**。任何"因为这句话更长/含关键词所以起始权重更高"的规则都是一条
    未标定的猜测，而它的后果（长期占据召回前排）比它想解决的问题更难发现。
    权重该由 P2 的 salience 规则表调，那里改得动、也看得见。
    """

    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryPolicy(Protocol):
    """ "这一轮留下什么"的策略。

    实现必须是纯的：同样的一轮永远给同样的草稿，不读时钟、不碰库。理由见模块
    文档——一句话，P3 要换掉它。
    """

    def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
        """这一轮该记下什么。可以不记（返回空序列）。"""
        ...


def _meaningful_chars(text: str) -> int:
    """数**实义字符**：字、词、数字。标点和空白不算。

    "？？？" 有 3 个字符，实义 0 个——按 ``len()`` 它够长，按这个它什么都不算。
    """
    return sum(1 for char in text if is_word(char))


def _looks_like_a_question(text: str) -> bool:
    """这句话是不是一个问句。判据四档，由紧到宽。

    **宁可多判。** 两个方向的代价不对称：漏判（把问句记成 ``FACT``）正是这条规则存在
    的理由——"用户问过路由器放哪"被读成"路由器在某处"，而模型会据此作答，还答得很
    自信；误判（把陈述记成 ``OPEN_QUESTION``）只是给这条记忆换一个标签，它照旧进库、
    照旧参与召回、正文一字不差。所以这里收的是**宽**的那一版。

    只认句末问号的话，**中文不带标点的问句会全部落进 ``FACT``**——"帮我查一下我之前
    记过什么"就是一句。这不是边界情况，口语化的输入本来就不爱打问号。
    """
    stripped = text.rstrip()
    if stripped.endswith(tuple(_QUESTION_ENDERS)):
        return True

    # 先摘掉任指/虚指的写法，再找疑问词——否则"我什么都不想干"会被 "什么" 判成问句。
    probe = stripped
    for phrase in _QUESTION_FALSE_FRIENDS:
        probe = probe.replace(phrase, "")

    if probe.endswith(_QUESTION_PARTICLES):
        return True
    if _QUESTION_REDUPLICATION.search(probe):
        return True
    return any(marker in probe for marker in _QUESTION_MARKERS)


def _kind_of(text: str) -> MemoryKind:
    """标记词表判 kind。命中不了就退回 ``FACT``。

    **偏好/决定两张表排在问句前面，是"具体压过宽泛"。** 那两张表命中的是用户对自己
    的一句陈述（"我以后都想早点睡"），而问句那一档判的是形状、几乎什么都能套上；两者
    同时命中时以更具体的那张为准。"我喜欢什么"这种写法会被判成偏好——它是已知的误判，
    但反过来（问句优先）会让带反问尾巴的真心话丢掉偏好标记，那个方向的代价更大。
    """
    for marker in _PREFERENCE_MARKERS:
        if marker in text:
            return MemoryKind.PREFERENCE
    for marker in _DECISION_MARKERS:
        if marker in text:
            return MemoryKind.DECISION
    if _looks_like_a_question(text):
        # **问题不是事实。** 它没有断言任何东西，记成 FACT 会让模型把"用户问过
        # 路由器放哪"读成"路由器在某处"。OPEN_QUESTION 这个 kind 存在的理由就是它。
        return MemoryKind.OPEN_QUESTION
    return MemoryKind.FACT


@dataclass(frozen=True)
class PerItemPolicy:
    """P1 的默认策略：**用户说的那句话**变成一条记忆。

    这是"每轮实时写入"里的"写入"——实时指的是**不需要等巩固、不需要批处理**，
    不是"每一轮都必然产出一条"。这一轮没有值得记的东西时返回空，是这套规则在
    正常工作，不是它没跑。

    助手那一侧不记：回复是从记忆和这一轮输入推出来的，把它也存一遍等于同一件事
    存两次，而且第二次还带着模型的幻觉。**这个规矩只有一处例外**——
    :class:`TaskPolicy` 拿"一次任务的最终回答"当正文，因为那时它本身就是用户要的
    产物。例外落在那个类里，不影响这里。
    """

    min_chars: int = MIN_MEANINGFUL_CHARS
    question_kind: bool = True
    """把问句标成 :attr:`MemoryKind.OPEN_QUESTION` 而不是 ``FACT``。

    留一个开关是因为这条规则有真实的反对意见：一个**被回答了的**问题并不是"待办
    问题"。P1 判断不了一轮有没有被回答，所以只能保守地按形状标。关掉它，问句就
    一律按 ``FACT`` 记——那样更简单，但会把"用户问过 X"写成"X 成立"。
    """

    def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
        text = turn.user_text.strip()
        if _meaningful_chars(text) < self.min_chars:
            return ()

        folded = normalize(text).strip()
        if folded in ACKNOWLEDGEMENTS:
            return ()

        # 单一重复的嘟囔（"哈哈哈哈"、"嗯嗯嗯嗯"、"好的好的"）。**按不同的字数判**，
        # 不按分词结果——理由见 :data:`_MURMUR_DISTINCT_CHARS`。
        #
        # 门槛放在"不同字数"而不是"重复了几次"上：真正的判据是**信息量**，而"同样
        # 两个字说了四遍"和"同样的字说了两遍"一样没有信息量。
        #
        # 走到这里说明实义字符已经 >= 4，所以不会误伤"买牛奶"这种短句（它先被长度
        # 拦住）；会命中的是"好的好的"、"对对对对"这一档。
        if len({char for char in text if is_word(char)}) <= _MURMUR_DISTINCT_CHARS:
            return ()

        kind = _kind_of(text) if self.question_kind else MemoryKind.FACT
        draft = MemoryDraft(
            content=text,
            kind=kind,
            taint=Taint.USER,
            meta={"turn_seq": turn.turn_seq, "interrupted": turn.interrupted},
        )
        return (draft,)


MAX_TASK_TEXT_CHARS = 2000
"""``TaskPolicy`` 的 ``meta["full_text"]`` 的上限。

它值得存在：开了任务模式之后，用户那句话在记忆库里**没有第二处落脚**（正文是回复）。
截到 2000 字是因为它进的是 ``meta``，而 ``meta`` 有硬体积上限（理由同轨迹，
见 :mod:`~axo.agent.trace`）。超出的部分在逐字稿里，``source_ref`` 指得到。
"""


@dataclass(frozen=True)
class TaskPolicy:
    """**一次任务 = 一条记忆。**

    与 :class:`PerItemPolicy` 的差别只有一处，就是模块文档开头说的那个可替换点：
    **有工具轨迹的那一轮**合成一条——正文是回复，做过的事在 ``meta["tool_trace"]``
    里——其余轮次原样交给 :attr:`fallback`。

    ## 为什么是加法，不是替换

    "一次任务 = 一条记忆"说的是**任务**的粒度。把它做成排他（开关一开，没调工具的那
    些轮次就不再入库）的代价是：用户打开一个叫"任务记忆"的开关，副作用是一部分对话
    从此不被记住——而这一点在界面上完全看不出来。反过来，加法只多一条判据，而且那条
    判据是免费的：**调没调过工具**。

    ``Turn`` 里没有语域，也不该有——它是两条路径的公共形状，塞一个
    :class:`~axo.agent.router.Register` 进来会让"这一轮算不算任务"变成一件
    要跟路由同步的事。工具轨迹本身就是那件事的证据。

    ## 为什么正文是回复

    条目模式记用户那句话，因为用户说的话是他自己的断言。而一次任务的产物是**那个
    回答**——用户过后要问的是"上次让它总结的那份 README 说了什么"，不是"我上次提了
    个什么要求"。``full_text`` 留在 ``meta`` 里，于是来历也还在。

    ## 这个对象不知道开关在哪儿

    开关在设置里（``memory/task_memory``，默认**关**），由 ``MainWindow`` 决定装不装
    它。理由与 :class:`PerItemPolicy` 的 ``question_kind`` 留在构造参数上不同：那一项
    是策略自己的行为开关，而"用哪套粒度"是装配层的选择——策略一旦自己去读设置，它
    就再也测不了"两轮相同的输入该不该得到同样的记忆"。
    """

    fallback: MemoryPolicy = field(default_factory=PerItemPolicy)
    """没调工具的那些轮次交给谁。默认就是条目模式，也就是开关打开之前的行为。"""

    def drafts(self, turn: Turn) -> Sequence[MemoryDraft]:
        trace = tuple(turn.tool_trace)
        if not trace:
            # **没调工具的一轮不是任务。** 交给条目模式，两边逐字段一致——这也是
            # "开关是加法"这句话在代码里的唯一落点。
            return self.fallback.drafts(turn)

        # 中断的任务：一个字都没吐出来时正文回落到用户那句。**照记**——"用户让我
        # 做 X，我被打断了"本身就是这件事的全部内容，丢掉它等于这一轮没发生过。
        content = turn.reply_text.strip() or turn.user_text.strip()
        if not content:
            return ()

        full_text = turn.user_text.strip()
        draft = MemoryDraft(
            content=content,
            kind=MemoryKind.TOOL_RESULT,
            # ``TOOL`` 而不是 ``USER``：这一条的正文是**模型的产物**，只是它记的是
            # 一次工具任务。salience 规则按来源打分，混成 USER 会让它拿到不属于它的权重
            # （见 :class:`memcore.types.Taint`）。
            taint=Taint.TOOL,
            meta={
                "mode": "task",
                "turn_seq": turn.turn_seq,
                "interrupted": turn.interrupted,
                "tool_trace": [dict(step) for step in trace],
                "full_text": full_text[:MAX_TASK_TEXT_CHARS],
                # 原文长度。没有它，"用户只说了 2000 字"与"本来有 5 万字、这里只留了
                # 2000"在读数据时是分不开的。
                "full_text_chars": len(full_text),
            },
        )
        return (draft,)
