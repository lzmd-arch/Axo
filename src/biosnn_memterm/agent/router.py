"""语域路由。

## 语域是什么

**同一个灵魂，两种语域。** 陪伴型助手与干活型助手的分野不是两个产品、不是两种人格，
更不该是界面上一个"模式"开关——用户不该先做一次分类再说话。它是同一套人格在两种
说话方式之间的切换：闲聊时它在，干活时它也在，只是手上多了一件事。

这条要求在仓库外的那份规划稿里（``规划.md``）写成"同一灵魂，两种语域"，原话是
**不要模式切换按钮**。本模块把"这一轮算哪种语域"自动化掉。

## 两个轴，别混名

规划稿把两件正交的事画成了一条链（"闲聊 ⇄ 任务中·可闲聊 ⇄ 等待确认"）。本模块
**拆开**，因为混名之后"用户在确认卡片还没点的时候算哪种语域"会变成一个没法回答的
问题：

- **路由轴**（本模块）：:class:`Register` 两态 ``CHAT ⇄ TASK``，带迟滞。
- **agent 轴**（循环与闸门）：**"等待确认"是 agent 的瞬时状态，不属于路由**。
  一个 task 回合可以正卡在确认卡片上，也可以正在调工具、正在收尾——那些都是
  agent 的状态，路由这一层不区分，也不该区分。

## 路由的三种信号，优先级从高到低

1. **斜杠命令**（:func:`parse_slash`）——用户显式指定，最高优先级，**无视迟滞**。
   这是唯一确定性、可离线断言的一层，也是模型不可靠时**唯一可靠的入口**。
2. **规则层**（:func:`rule_register`）——标记词表，模仿
   :mod:`biosnn_memterm.memory.policy` 的做法：表错了看得见、改起来是一行，而且能
   进 CI 当门槛。它**判不出来时返回 ``None``**，不猜。
3. **模型自分类**（:func:`classify_with_model`，**默认关**）——一次额外 LLM 调用。
   默认关不是保守，是量出来的：2026-10-10 实测 ``qwen3-4b`` 在"两个工具该调哪个"
   上五轮只对一轮（见 ``tools/smoke_agent.py``），它不足以可靠地判自己这一轮属于
   哪个语域。开启与否由 ``benchmarks/routes/`` 的报告定，不由拍脑袋定。

## 迟滞是**不对称**的，理由是代价不对称

规划稿要迟滞是为了治"人格闪跳"。但对称地实现它（往返都要连续 N 次信号）会让
**开箱第一句"帮我查一下 X"拿不到工具**——状态从 CHAT 起步，一次 task 信号不够两次，
于是助手说"我查不了"。那正是本版要交付的那件事，不能死在门口。

所以只有**一个方向**要迟滞：**从 TASK 落回 CHAT**。理由是代价不对称——

* 误进 TASK 的代价只是"这一轮工具可选"，而改状态的动作还有确认闸门兜着；
* 误离 TASK 的代价是**一件正做到一半的事没了工具**，而它恰恰是最常见的那种闪跳：
  用户在任务中间说了句"好的"或"谢谢"，下一次开口就发现助手不干活了。

## 歧义：只出标志，不出文案

:func:`rule_register` 判不出来（表里一条都没命中）时，:func:`route_turn` 给出
``ambiguous=True``，语域退回 ``CHAT``、**并且不改变迟滞状态**（"不知道"不是一个
方向上的信号）。循环据此在**本回合不声明任何工具**，正常聊天，并请模型带一句确认。
规划稿说"确认本身也是陪伴感"——那句话由**模型的措辞**承担，路由这一层是纯函数，
它不产文案，也产不了：它不知道用户上一句说了什么。


## 斜杠命令

**命令本身不进逐字稿正文。** 用户打了 ``/task 帮我查一下``，进逐字稿的是"帮我查一下"
——否则"用户说过 /task"会被当成一句话记进记忆库，而那不是任何人对任何事的陈述。
命令作为**元数据**记进消息（供回看），不进正文。

**仅当轮有效，还是切状态**——两者的区别是有意的，见 :attr:`ParsedTurn.sticky`：
``/task 正文`` 只影响这一轮；裸 ``/task`` 改的是**后续若干轮**的状态。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from memcore.text import is_word, normalize

from ..llm.base import ChatBackend, ChatMessage, ChatRequest
from ..memory.policy import ACKNOWLEDGEMENTS

__all__ = [
    "DEFAULT_HYSTERESIS",
    "HELP_TEXT",
    "MODEL_CLASSIFY_DEFAULT",
    "MODEL_PROMPT",
    "ModelRole",
    "ModelVerdict",
    "ParsedTurn",
    "Register",
    "RegisterState",
    "RouteDecision",
    "RouteSource",
    "RoutedTurn",
    "SlashCommand",
    "classify_with_model",
    "decide",
    "decide_turn",
    "parse_slash",
    "route_turn",
    "rule_register",
]


class Register(str, Enum):
    """语域。继承 :class:`str` 的理由与 :class:`~memcore.types.Tier` 同源：
    它能直接进 JSON、进逐字稿的元数据、进报告，不必写转换表。"""

    CHAT = "chat"
    """陪伴语域：不调工具，好好说话。"""

    TASK = "task"
    """任务语域：可以调工具，改状态的动作要过闸门。"""

    def __str__(self) -> str:
        return self.value


class SlashCommand(str, Enum):
    """已识别的斜杠命令。

    **未知的斜杠不是命令。** ``/home/user/笔记.txt 里写了什么`` 是一句正常的话
    （用户可能就在问一个路径），把它当命令吞掉、或者报一句"未知命令"，都会让
    用户没法正常说话。所以只有表里这几个词被认作命令，其余一律**原样**当作普通
    输入——见 :func:`parse_slash`。
    """

    TASK = "/task"
    CHAT = "/chat"
    HELP = "/help"

    def __str__(self) -> str:
        return self.value


HELP_TEXT = """\
可用命令（只在这一轮生效，除非注明）：

  /task <正文>   这一轮按任务处理，允许调用工具
  /chat <正文>   这一轮按闲聊处理，不调用工具
  /task          不带正文：切入任务语域，并保持到下次切换
  /chat          不带正文：切回闲聊语域，并保持到下次切换
  /help          显示这份说明

不带斜杠时，语域由系统自动判断。
"""


@dataclass(frozen=True)
class ParsedTurn:
    """一次输入解析出来的东西。

    :attr:`text` 是**真正要送给模型的输入**——斜杠命令已经被摘掉。调用方负责
    保证 :attr:`command` 不进逐字稿正文（作为元数据记下来供回看可以）。
    """

    command: SlashCommand | None
    """识别出的命令；``None`` 表示这是一句普通输入。"""

    text: str
    """送给模型的正文。对命令而言是命令之后的部分；对普通输入而言是**原样**。"""

    forced: Register | None = None
    """这一轮被强制的语域。``None`` = 没强制，交给规则层/状态机决定。"""

    sticky: bool = False
    """``True`` 表示这次切换**保持**（裸 ``/task`` / ``/chat``），
    ``False`` 表示只影响这一轮。

    两者的区别不是细节：``/task 帮我查 X`` 是"就这一件事"，而 ``/task`` 是
    "接下来聊的都是正事"。把它们合并成一个"进入任务态"会让前者的语义变成
    "用户从此进了任务模式"——而用户只是想查一件事。
    """

    @property
    def is_help(self) -> bool:
        """``/help``：调用方应当**本地渲染** :data:`HELP_TEXT`，不发请求。

        它不是一轮对话——没有正文、也不该有一次模型调用。做成属性而不是让调用方
        去比 ``command is SlashCommand.HELP``，是为了让"这个分支要单独处理"在一个
        地方说出来。
        """
        return self.command is SlashCommand.HELP

    @property
    def is_bare_switch(self) -> bool:
        """裸 ``/task`` 或 ``/chat``：只切语域，**这一轮没有正文可送**。

        调用方据此判断"要不要真的发一次请求"——正文是空的时候发请求，模型会拿到
        一条空消息，然后答非所问，而这个症状看起来像模型坏了。
        """
        return self.sticky and not self.text


_SWITCHES: dict[str, Register] = {
    SlashCommand.TASK.value: Register.TASK,
    SlashCommand.CHAT.value: Register.CHAT,
}


def parse_slash(raw: str) -> ParsedTurn:
    """解析一句输入里的斜杠命令。

    规则：

    * 只有**开头**的斜杠算命令（``你好 /task`` 里那个不是）。前导空白允许，
      因为输入框里手滑敲个空格太常见了。
    * 命令词大小写不敏感（``/TASK`` 认）。中文输入法下大小写是随机的噪声，
      为一个字母让用户重打一遍没有道理。
    * 只认 :class:`SlashCommand` 里的三个词。**其余斜杠一律原样返回**——见
      :class:`SlashCommand` 的说明。
    * 普通输入**逐字返回，不做任何规整**。这不是偷懒：逐字稿里存的应当就是用户
      打的字，而"顺手 strip 一下"会让某天有人对着逐字稿排查"为什么这段显示不一样"
      时找不到原因。命令的正文则会去掉两侧空白——``/task`` 与正文之间那个空格是
      语法的一部分，不是内容。
    """
    # 只在 lstrip 后的串上做识别，但普通输入返回**原样**。所以这两个变量都要留着。
    probe = raw.lstrip()
    if not probe.startswith("/"):
        return ParsedTurn(command=None, text=raw)

    # 按**第一个空白段**切，不是按单个空格：``split()`` 不带分隔符时把空格、制表符、
    # 换行一视同仁，所以 ``/task\t正文`` 与 ``/task\n正文`` 都认。
    parts = probe.split(maxsplit=1)
    word = parts[0].lower()
    body = parts[1].strip() if len(parts) > 1 else ""

    if word not in _SWITCHES and word != SlashCommand.HELP.value:
        # 认不出来 → 这句话是普通输入。**原样**返回 raw，连前导空白都不动：
        # 不然 "/路径 里有啥" 这种话会悄悄变形。
        return ParsedTurn(command=None, text=raw)

    command = SlashCommand(word)
    if command is SlashCommand.HELP:
        # /help 后面写了东西也当没写——它不是一条能带正文的命令，而"报错说 /help
        # 不接受参数"是让用户为一个无关紧要的笔误再跑一趟。
        return ParsedTurn(command=command, text="")

    return ParsedTurn(
        command=command,
        text=body,
        forced=_SWITCHES[word],
        sticky=not body,
    )


# ── 规则层 ──────────────────────────────────────────────────────────────────

#: 任务标记词。命中的是"要对方去做一件事"，不是"提到了一个动作"。
#:
#: **为什么是表不是分类器**，与 :mod:`biosnn_memterm.memory.policy` 的 kind 判据
#: 同一条纪律：表一定会漏、也一定会误判，但它错了看得见、改起来是一行，而且
#: **确定性**——它能在 CI 里当门槛。用一个 4B 模型来判语域会把"模型的判断"变成
#: 路由的结果，而路由的结果决定要不要把工具交出去。
#:
#: 刻意**不收**的词，每个都有理由：``看看``（"你看看这个笑话"是闲聊）、``几点``
#: （"你几点睡的"是闲聊，只有 ``现在几点`` 收）、``记得``（"记得昨天那个吗"是
#: 闲聊，收的是 ``记住``/``记一下``）。
#:
#: 说是"闲聊"，落到哪一档要看表走完：这三个例子**都走到第 8 条**，也就是
#: ``None``——"判不出来"。它们不是 task，但也不值得硬判成 CHAT：真话是"这几个词
#: 分不出来"，而分不出来有它自己的去处（状态机沿用上一轮，冷启动才转 CHAT）。
#: ``tests/test_router.py::test_the_words_we_chose_not_to_collect_land_on_no_verdict``
#: 把这三条钉住了，免得好心人回头把它们补进表里。
#:
#: 而"要去看已经存下来的东西"那一族由 ``上次`` 承担，它比 ``记得`` 准：语料里
#: 三条该查记忆的问句（``u-0114``/``u-0115``/``u-0120``）都含 ``上次``，而
#: "记得昨天那个吗"不含。
_TASK_MARKERS = (
    # 求助
    "帮我",
    "帮忙",
    "替我",
    # 检索
    "找一下",
    "找找",
    "查找",
    "查一下",
    "查查",
    "查一查",
    "搜一下",
    "搜搜",
    "搜索一下",
    "翻一下",
    # 读
    "读一下",
    "读读",
    "打开",
    "看一下",
    "看一眼",
    "瞧瞧",
    "念一下",
    # 整理与改写
    "列出",
    "列一下",
    "整理一下",
    "总结一下",
    "归纳一下",
    "算一下",
    "算算",
    "翻译一下",
    "改一下",
    "改成",
    "换成",
    # 记忆读写
    "记一下",
    "记下来",
    "记住",
    "写一条",
    "写个记忆",
    "存一条",
    "忘掉",
    "删掉那条",
    "删除那条",
    # 记忆检索——**要去看已经存下来的东西**。与上面那组（往库里写）是两件事，
    # 分开列是因为它们的形状完全不同：写是祈使句，查几乎全是**问句**。
    #
    # 这些词必须在这里，否则问号结尾的句子会被"句末问号 → 闲聊"那条规则吃掉，那一轮
    # **不声明工具**，模型只好回一句"这一轮没有可用的工具"（2026-10-11 实测；同日量
    # 语料，`hard` 类 8 条里 5 条是这一族）。
    #
    # ``上次`` 是主力，而且它**不是动词**：收它是因为它指的是"这件事发生过，而
    # 记录不在你眼前"，也就是"去查"。它比 ``记得`` 准（那句刻意不收的理由见上）。
    # ``我的记忆``/``记忆里`` 收的是用户实际会打的说法（"查看我的记忆"、"看看我的
    # 记忆"、"我的记忆里有什么"）——``查看``/``看看`` 本身不收，所以这里必须由
    # "说的是记忆这件事"来兜。
    "上次",
    "我的记忆",
    "记忆里",
    "记忆库",
    "回想",
    # 下面三个是语料点名的**同义动词缺口**（表里只有 ``记一下``/``念一下``/``忘掉``，
    # 而实际说法是"记录一下我今天的感受"、"把昨天那段话再念一遍"、"别忘了我说的"）。
    # ``别忘`` 整词入表是故意的：否定窗口只看标记**之前**那几个字，而这里的 ``别``
    # 在标记内部，所以它不会被 ``_NEGATIONS`` 反过来判成闲聊——"别忘了我说的"是一句
    # 请求，不是一个否定。
    "记录一下",
    "念一遍",
    "别忘",
    # 提醒与时间
    "提醒我",
    "定个提醒",
    "定一个提醒",
    "叫我一下",
    "现在几点",
    "几点了",
    # ``星期几``/``几号`` 与 ``现在几点`` 同一类：问一个**由外部世界决定、
    # 助手只能去查**的事实。这类问题的形状是问句，但答案不在对话里也不在库里，
    # 所以它是任务而不是闲聊——``get_time`` 就是为它们存在的。
    "星期几",
    "几号",
)

#: 否定词。**它排在整个判定表的最前面**，因为"别帮我查"里明明含着 task 动词。
#: 只收多字词与 ``别``——单字 ``不`` 收不得：``我不太确定，帮我查一下`` 是一句
#: 正常的委托，而它前半段就有一个 ``不``。
_NEGATIONS = ("别", "不要", "不用", "不必", "无需", "不需要", "先别", "别再")

#: 否定词要落在 task 标记**前多远之内**才算否定了它。
#:
#: 取 6 是量出来的边界：``先别急着帮我查`` 里那个 ``别`` 距离 ``帮我`` 有 4 个字，
#: 而 ``我不太确定，帮我查一下`` 里那个 ``不`` 不在窗口里（它也不是否定词）。窗口
#: 再放宽就会开始误伤正常的委托，而误伤的代价是"用户让它干活它不动"。
_NEGATION_WINDOW = 6

#: 闲聊标记词。排在 task 之后——两者同时命中时以 task 为准。
#:
#: 以 task 为准是选的，不是没想过：``帮我查一下这个文件里写了什么？`` 同时命中
#: task 动词与句末问号，而它是一句再清楚不过的委托。反过来（问号优先）会让最常见
#: 的委托句式变成"歧义"，于是**每一次查文件都要先被反问一遍**。
_CHAT_MARKERS = (
    # 情绪与状态
    "累",
    "开心",
    "难过",
    "难受",
    "无聊",
    "郁闷",
    "焦虑",
    "孤独",
    "睡不着",
    "想你",
    "烦",
    # ``心情`` 收，而 ``雨``/``花``/``梦``/``心跳`` **不收**——这条线是 2026-10-12
    # 量出来的，量之前它们看起来是同一类（"语料里判不出来的那几句都含它们"）。
    #
    # 分界线是**这句话关于什么** vs **这句话是不是一件要对方做的事**：``心情`` 说的
    # 是说话人自己的状态，跟上面那十一个词是同一族；``雨``/``花``/``梦`` 是**话题
    # 名词**，任何语域都可能是那个话题——"我养的花开了"是闲聊，而"帮我查一下这种花
    # 怎么养"是委托（这条由 rule 2 先接住，所以收 ``花`` 不冲突，它只是**没有信息
    # 量**）。收话题名词等于把"这句话里出现过某个字"当成语域证据。
    #
    # 两个"看起来该有用其实没命中"的候选也一并记在这里：``做梦`` 没接住
    # ``u-0062``（正文是"我今天做了个奇怪的``梦``"，中间隔着字），``下雨`` 没接住
    # ``u-0068``（正文是"外面的``雨``下得挺大"）。**候选词得先在正文里真出现**——
    # 这两条是照着语料的**意思**挑的词，不是照着**文字**挑的。
    "心情",
    "唉",
    "哎",
    "哈哈",
    "嘿嘿",
    # 寒暄
    "晚安",
    "早安",
    "早上好",
    "晚上好",
    "好久不见",
    "辛苦",
    "谢谢你",
    "多谢",
    "天气",
    # 附和与评价
    "你说得对",
    "有道理",
    "真的吗",
    "是吗",
    "有意思",
    "真好",
    "不错",
    # 话题引子
    "你觉得",
    "你怎么看",
    "什么是",
    "为什么",
    "你是不是",
    "说说你",
    "怎么样",
    "怎么办",
    "聊聊",
    "聊天",
)

_QUESTION_ENDERS = ("？", "?")

#: 句首的礼貌词。它只在**没有 task 信号**时才起作用（见 :func:`rule_register` 的
#: 判定顺序），所以 ``谢谢你，帮我查一下 X`` 会被判成 task——**这是已知的误判**，
#: 语料的 ``hard`` 类别里有它，报告上看得见。不修的理由：把它做成"礼貌词否定后面
#: 一切"会让 ``谢谢，麻烦帮我读一下那份表`` 这类真实委托全部掉进 chat，而那个
#: 方向的代价（用户让它干活它不动）比这个方向（多给了它一个可选的工具，改状态的
#: 动作还有闸门兜着）大。
_COURTESY_PREFIXES = ("谢谢", "多谢", "感谢", "辛苦了", "麻烦你")

#: 文件后缀。**路径形状的判据是"有没有一个像文件名的东西"，不是"有没有斜杠"**：
#: ``3/5 这个方案怎么样`` 里的斜杠不是路径，而按斜杠判就会把它变成一次工具调用。
_FILE_SUFFIXES = (
    "txt",
    "md",
    "json",
    "jsonl",
    "py",
    "csv",
    "log",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "doc",
    "docx",
    "pdf",
    "xls",
    "xlsx",
    "html",
    "htm",
    "sh",
    "bat",
)

# 尾部用负向先行断言而不是 ``\b``：Python 的 ``\w`` **包含中日韩汉字**，所以
# ``笔记.txt里写了什么`` 里 "t" 与 "里" 之间没有词边界，``\b`` 会让这条最自然的
# 中文写法整个失效——而失效的样子是"规则层漏判"，看起来像表收得不够。
_EXTENSION_RE = re.compile(
    r"\.(?:" + "|".join(_FILE_SUFFIXES) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
#: ``D:\notes`` / ``C:/Users`` 这种带盘符的路径。**盘符单独算一个信号**：用户说
#: "D 盘那个文件夹里有什么"时没有扩展名可认，但盘符一样明确。
_DRIVE_RE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")

#: 指一个**位置**的名词。与扩展名互补：``那个文件夹里有什么`` 没有文件名可认，但它
#: 明摆着是在让助手去翻一个目录。
#:
#: 只收这两个词，不收 ``桌面``/``盘``：那两个在日常对话里另有意思（"桌面壁纸不错"、
#: "这个盘该怎么分区"），而收进来的代价是**闲聊被当成任务**。
_LOCATION_WORDS = ("文件夹", "目录")

#: 连续重复的嘟囔（"哈哈哈哈"、"嗯嗯嗯"、"好的好的"）算闲聊。
#:
#: 判据是**不同的字数**而不是重复了几次，与 :mod:`biosnn_memterm.memory.policy`
#: 的 ``_MURMUR_DISTINCT_CHARS`` 同一个做法、同一个理由：真正的判据是信息量，而
#: "同样的两个字说了四遍"和"同样的字说了两遍"一样没有信息量。
#:
#: 它排在 task 之后，所以 ``帮忙``/``算算`` 这些**真正是动词的叠词**不受影响——
#: 那两个各有 2 个不同的字，会先被 task 表接住。
_MURMUR_DISTINCT_CHARS = 2


def _names_a_location(text: str) -> bool:
    """这句话点了一个文件、一条带盘符的路径，或者一个目录。"""
    return bool(
        _EXTENSION_RE.search(text)
        or _DRIVE_RE.search(text)
        or any(word in text for word in _LOCATION_WORDS)
    )


#: 应和词**去掉非词字符之后**的样子，给 :func:`_is_acknowledgement` 用。
#:
#: 派生而不是手写第二份表：改了 ``ACKNOWLEDGEMENTS`` 这里跟着变，不会漂。去字符是
#: 为了两侧同一种处理——正文那侧也要去掉标点，``thank you`` 里的空格与 ``好的，知道了``
#: 里的逗号是同一件事。
_ACKNOWLEDGEMENT_KEYS = frozenset(
    "".join(char for char in word if is_word(char)) for word in ACKNOWLEDGEMENTS
)
_ACK_MAX_CHARS = max(len(word) for word in _ACKNOWLEDGEMENT_KEYS)

#: 句尾的语气助词。**只从尾巴上剥**，剥完仍然要求全句由应和词拼成。
#:
#: 需要它是因为 ``ACKNOWLEDGEMENTS`` 自己对尾缀不一致：表里的 ``知道了`` 带着 ``了``，
#: 而 ``明白`` 不带——于是"明白了"切到最后一个字就断了，而"知道了"接得住。这不是那份
#: 表写错了（它管的是"这一轮值不值得写进记忆"，两处诉求不同），是**这里**该补的一层：
#: 一个语气助词不改变语域，"明白了"和"明白"是同一件事。
_ACK_TRAILING_PARTICLES = "了啦呀啊呢吧嘛哦噢哟"


def _is_acknowledgement(text: str) -> bool:
    """整句是否**全由应和词拼成**——``好的，知道了``、``嗯，明白了``、``行，收到``。

    **不能要求整句精确等于一个应和词**：中文的应和是**拼**出来的，而精确匹配连多一个
    句号都接不住（``normalize`` 只做 NFKC + casefold，不删标点），"好的，知道了"会
    一路走到第 8 条成为"判不出来"（``u-0050``，见 ``benchmarks/routes/CORPUS.md``）。

    判据是**全句切得完**，不是"含有应和词"——``好，我们开始吧`` 切到 ``我`` 就断了，
    照旧"判不出来"。这条边界要紧：放宽成"含应和词"会让任何以"好的"开头的句子都变成
    闲聊，而那正是第 5 条（句首礼貌词）刻意只给 0.7 的地方。

    切分用**可达性**而不是贪心：贪心在"长的那块先吃掉、剩下的切不动"时会误判成否，而
    可达性能把每一条切法都试到。这么点长度（应和词最长 9 字）不值得为性能省这个。
    """
    folded = "".join(char for char in normalize(text) if is_word(char))
    if not folded:
        return False

    candidates = [folded]
    trimmed = folded.rstrip(_ACK_TRAILING_PARTICLES)
    if trimmed and trimmed != folded:
        candidates.append(trimmed)

    def covered(candidate: str) -> bool:
        reachable = [True] + [False] * len(candidate)
        for start in range(len(candidate)):
            if not reachable[start]:
                continue
            for size in range(1, min(_ACK_MAX_CHARS, len(candidate) - start) + 1):
                if candidate[start : start + size] in _ACKNOWLEDGEMENT_KEYS:
                    reachable[start + size] = True
        return reachable[-1]

    return any(covered(candidate) for candidate in candidates)


def rule_register(text: str) -> tuple[Register | None, float]:
    """规则层：标记词表判这一轮属于哪种语域。

    返回 ``(语域, 置信度)``；**判不出来时语域是 ``None``**，交给状态机（以及开启
    时的模型层）。"判不出来"是一种结果，不是一个错误——一个猜错的分类器比一句
    "我不知道"糟得多，因为猜错的后果是把工具交出去或者收回来，而"我不知道"的后果
    只是这一轮多问一句。

    判定是一张**决策表**，从上往下第一条命中的说了算。顺序本身是设计的一部分：

    1. 否定紧邻 task 信号 → ``CHAT``（"别帮我查了"）
    2. task 动词，或点了一个文件/目录 → ``TASK``
    3. 整句**由应和词拼成**（``好的，知道了``）→ ``CHAT``
    4. 重复的嘟囔（按不同的字数判）→ ``CHAT``
    5. 句首礼貌词 → ``CHAT``
    6. 情绪/寒暄/话题引子 → ``CHAT``
    7. 句末问号 → ``CHAT``
    8. 都没有 → ``None``

    **置信度是给人看的，不参与任何判定。** 它是一档粗粒度的手设值，用它做阈值就是
    一个未标定的猜测；而这份报告（``tools/bench_router.py``）正是标定它所需要的
    输入——"低置信的那些句子是不是恰好就是判错的那些"。在量出来之前它不决定任何事。
    """
    stripped = text.strip()
    if not stripped:
        # 空正文不是"闲聊"，是"没有可判的东西"（裸 ``/task`` 走的是斜杠那条路，
        # 到不了这里）。
        return None, 0.0

    markers = [index for marker in _TASK_MARKERS if (index := stripped.find(marker)) >= 0]
    has_location = _names_a_location(stripped)

    if markers or has_location:
        first = min(markers) if markers else len(stripped)
        window = stripped[max(0, first - _NEGATION_WINDOW) : first]
        if any(negation in window for negation in _NEGATIONS):
            return Register.CHAT, 0.75
        return Register.TASK, 0.85 if markers else 0.7

    if _is_acknowledgement(stripped):
        return Register.CHAT, 0.9
    if len({char for char in stripped if is_word(char)}) <= _MURMUR_DISTINCT_CHARS:
        return Register.CHAT, 0.8
    if any(courtesy in stripped[:4] for courtesy in _COURTESY_PREFIXES):
        return Register.CHAT, 0.7
    if any(marker in stripped for marker in _CHAT_MARKERS):
        return Register.CHAT, 0.8
    if stripped.endswith(_QUESTION_ENDERS):
        return Register.CHAT, 0.6
    return None, 0.0


# ── 迟滞状态机 ──────────────────────────────────────────────────────────────

DEFAULT_HYSTERESIS = 2
"""从 TASK 落回 CHAT 需要连续几次反向信号。

**只用一个方向**，理由见模块文档（代价不对称）：进 TASK 立即生效，因为它错了只是
"这一轮工具可选"；出 TASK 要连续两次，因为它错了是一件正做到一半的事突然没了工具。
"""


class RouteSource(str, Enum):
    """这一轮的语域是**谁**定下来的。

    它进报告，所以取值是稳定字符串（同 :class:`Register` 的理由）。它也是
    :meth:`RegisterState.advance` 的分支依据：``SLASH`` 无视迟滞。
    """

    SLASH = "slash"
    """用户显式打了 ``/task`` / ``/chat``。"""

    RULE = "rule"
    """标记词表判出来的。"""

    MODEL = "model"
    """模型自分类判出来的（默认关）。"""

    STATE = "state"
    """规则层换了方向，但迟滞没让它生效——这一轮沿用上一轮的语域。"""

    DEFAULT = "default"
    """一条都没命中，退回 :attr:`Register.CHAT` 并标歧义。"""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RouteDecision:
    """一轮的路由结论。

    :attr:`register` 是**这一轮实际用哪个语域**，不是"当前状态"——被迟滞压住的那
    一轮，它的 ``register`` 是沿用下来的那个，``source`` 是 :attr:`RouteSource.STATE`
    （两者一起才说得清发生了什么）。
    """

    register: Register
    source: RouteSource
    confidence: float = 0.0
    ambiguous: bool = False
    """规则层判不出来。**只出标志，不出文案**——文案由模型在对话里说。

    为真时 :attr:`register` 一定是 ``CHAT``：拿不准的一轮**不给工具**。改状态的
    动作有确认闸门兜着，但"读一个文件""搜一次记忆"没有闸门，而在一句闲聊上翻用户
    的库是不该发生的事。
    """


@dataclass
class RegisterState:
    """语域的迟滞状态。**可变**——它记的就是"上一轮是什么"。

    刻意不是纯函数：迟滞按定义需要跨轮记忆，而把它做成"每轮把历史传进来"只是把
    同一份状态换个地方放。调用方（``MainWindow``）持有**一个**实例，跨轮传下去。
    """

    register: Register = Register.CHAT
    streak: int = 0
    """从 TASK 落回 CHAT 已经连续攒了几次反向信号。"""

    def advance(
        self, decision: RouteDecision, *, threshold: int = DEFAULT_HYSTERESIS
    ) -> RouteDecision:
        """把这一轮的信号喂进状态机，返回**这一轮实际生效**的结论。

        ``threshold`` 是"落回闲聊需要连续几次"，必须 >= 1（1 = 不迟滞）。
        """
        if threshold < 1:
            raise ValueError(f"threshold 至少是 1，收到 {threshold}。")

        if decision.ambiguous:
            # 「不知道」不是一个方向上的信号：既不算反向，也不清计数。清掉的话，
            # 任务中间夹一句含糊的话就会让"马上要落回闲聊"的计数重新开始。
            return decision

        if decision.source is RouteSource.SLASH:
            # 显式覆盖，无视迟滞——用户刚打了 /chat，再按住不放就是不听人话。
            self.register = decision.register
            self.streak = 0
            return decision

        if decision.register is self.register:
            self.streak = 0
            return decision

        if decision.register is Register.TASK:
            self.register = Register.TASK
            self.streak = 0
            return decision

        self.streak += 1
        if self.streak >= threshold:
            self.register = Register.CHAT
            self.streak = 0
            return decision
        return replace(decision, register=self.register, source=RouteSource.STATE)


# ── 组装：一次输入到一轮的结论 ──────────────────────────────────────────────

Classifier = Callable[[str], Register | None]
"""模型自分类的注入点：给一句话，回一个语域或 ``None``（未定）。"""


class ModelRole(str, Enum):
    """模型自分类这一层**参与多少**。

    它不是一个"开关"而是三档，因为量这份量具时需要第三种：报告要能回答"**光靠模型
    够不够**"，而那既不是"规则层 + 模型兜底"，也不是"关掉模型"。

    :attr:`ONLY` 目前**只有** ``tools/bench_router.py`` 在用（见 ``benchmarks/routes/``）。
    把它放在这里而不是让报告自己去拼，是因为三档必须走**同一段判定代码**——报告里
    那个数才代表线上会拿到的数。
    """

    OFF = "off"
    """默认。规则层判不出来就算歧义——不额外花一次调用，也不把判断交给一个
    2026-10-10 实测在"该调哪个工具"上五轮只对一轮的 4B 模型（见 ``tools/smoke_agent.py``）。"""

    FALLBACK = "fallback"
    """规则层判不出来时问一次模型。规则层的**确定性**因此保留：它认得的句子
    不会因为模型的兴致而改判。"""

    ONLY = "only"
    """规则层不参与，每句都问模型。**只用于量具**——线上没有这一档。"""

    def __str__(self) -> str:
        return self.value


MODEL_CLASSIFY_DEFAULT = ModelRole.OFF
"""出厂默认。**由 ``benchmarks/routes/`` 的报告定，不由拍脑袋定**——
在哪份报告把它改成 ``FALLBACK`` 之前，它保持关闭。"""


def decide(
    text: str,
    *,
    classify: Classifier | None = None,
    model_role: ModelRole = MODEL_CLASSIFY_DEFAULT,
) -> RouteDecision:
    """**无状态**的一票：规则层，必要时再问模型。

    与 :func:`route_turn` 分开，是为了让"规则层说了什么"能被单独断言——迟滞是另一个
    问题，把两者混在一个函数里，测量规则层准确率时就会连带量到状态机的历史。

    ``model_role=ONLY`` 时 ``classify`` 必填（否则这一档什么判断都没有）。这条是
    ``ValueError`` 而不是"静默退回规则层"：配错了要当场知道。
    """
    if model_role is ModelRole.ONLY and classify is None:
        raise ValueError("model_role=ONLY 要求给一个 classify——这一档没有别的判据。")

    register, confidence = (None, 0.0) if model_role is ModelRole.ONLY else rule_register(text)
    if register is not None:
        return RouteDecision(register=register, source=RouteSource.RULE, confidence=confidence)

    if classify is not None and model_role is not ModelRole.OFF:
        guess = classify(text)
        if guess is not None:
            # 置信度 0.5 是个诚实的位置：模型层与规则层**各自独立报告准确率**
            # （见 benchmarks/routes/），不共享一个刻度，所以这里不假装能比较。
            return RouteDecision(register=guess, source=RouteSource.MODEL, confidence=0.5)

    return RouteDecision(
        register=Register.CHAT,
        source=RouteSource.DEFAULT,
        confidence=confidence,
        ambiguous=True,
    )


@dataclass(frozen=True)
class RoutedTurn:
    """一次输入解析 + 路由的全部产物。

    调用方要的三件事都在这里，且各自是**一个属性而不是一段 if**：
    :attr:`text`（送给模型的正文）、:attr:`uses_tools`（要不要声明工具）、
    :attr:`wants_clarification`（要不要请模型带一句确认）。
    """

    parsed: ParsedTurn
    decision: RouteDecision

    @property
    def text(self) -> str:
        """送给模型的正文。斜杠命令已被摘掉（见 :func:`parse_slash`）。"""
        return self.parsed.text

    @property
    def register(self) -> Register:
        return self.decision.register

    @property
    def uses_tools(self) -> bool:
        """这一轮该不该把工具声明给模型。

        **歧义时一定为假**——不是因为"task 才给工具"这条，而是因为拿不准的一轮
        什么都不该动。写成两个条件而不是一个，是为了让"歧义也可能有工具"这种改法
        必须动这一行。
        """
        return self.decision.register is Register.TASK and not self.decision.ambiguous

    @property
    def wants_clarification(self) -> bool:
        """要不要请模型带一句确认（规划稿的"歧义反问"）。"""
        return self.decision.ambiguous


def decide_turn(
    raw: str,
    *,
    classify: Classifier | None = None,
    model_role: ModelRole = MODEL_CLASSIFY_DEFAULT,
) -> tuple[ParsedTurn, RouteDecision]:
    """斜杠 → 规则 → （模型）。**不含迟滞。**

    与 :func:`route_turn` 拆开是为了让量具能测到"这一层判了什么"：迟滞是跨轮状态，
    而 ``benchmarks/routes/`` 是一句句独立的输入——把迟滞接进去，读数就会取决于
    语料的排列顺序，而语料顺序是任意的。量具测的是**无状态的这一票**。

    返回 ``(解析, 决策)`` 而不是单个对象，是因为调用方常常只想要其中一个。
    """
    parsed = parse_slash(raw)
    if parsed.forced is not None:
        decision = RouteDecision(register=parsed.forced, source=RouteSource.SLASH, confidence=1.0)
    else:
        decision = decide(parsed.text, classify=classify, model_role=model_role)
    return parsed, decision


def route_turn(
    raw: str,
    state: RegisterState,
    *,
    classify: Classifier | None = None,
    model_role: ModelRole = MODEL_CLASSIFY_DEFAULT,
    threshold: int = DEFAULT_HYSTERESIS,
) -> RoutedTurn:
    """一次输入走完整条路由：斜杠 → 规则 → （模型）→ 迟滞。

    ``state`` **会被就地修改**（见 :class:`RegisterState`），调用方持有一个实例、
    跨轮传同一个。

    **``/task 正文`` 不进状态机。** 迟滞记的是"用户现在处在哪个语域"，而 ``/task``
    后面带着正文时说的只是"这一件事按任务办"（:attr:`ParsedTurn.sticky` 为假）。
    喂进去的话，:meth:`RegisterState.advance` 见到 ``SLASH`` 会**无视迟滞**把状态
    直接改成 ``TASK``——于是"就这一件事"变成"用户从此进了任务模式"，而
    :attr:`ParsedTurn.sticky` 存在的全部意义就是区分这两者。裸 ``/task``
    （sticky 为真）照旧进去，那是用户真的在切状态。

    这一条是 ``MainWindow`` 第一次真的调这个函数（P3a 接线）时才暴露的：
    :func:`decide_turn` 对两种斜杠给出的 ``RouteSource`` 都是 ``SLASH``，
    :meth:`RegisterState.advance` 因此分不出它们。

    对**单次调用**而言这处改动没有可观察差别（``decide_turn`` 给的
    ``decision.register`` 已经是 ``parsed.forced``），差别只出现在第二轮：
    老写法会让下一轮从 ``TASK`` 状态起步。
    """
    parsed, decision = decide_turn(raw, classify=classify, model_role=model_role)
    if parsed.forced is not None and not parsed.sticky:
        return RoutedTurn(parsed=parsed, decision=decision)
    return RoutedTurn(parsed=parsed, decision=state.advance(decision, threshold=threshold))


# ── 模型自分类（默认关） ────────────────────────────────────────────────────

MODEL_PROMPT = """\
判断下面这句话属于哪一种，只回一个词，不要标点、不要解释、不要翻译：

chat  —— 闲聊、寒暄、情绪、感慨，或者一个不需要动手做的事
task  —— 要你去查、去读、去记、去提醒、去整理一件具体的事

用户说：{text}
只回一个词："""
"""自分类的提示词。

"只回一个词"不是提醒，是**解析契约**：:func:`_parse_verdict` 只认 ``chat`` 与
``task`` 两个整词，多一个字就判未定。放宽成"包含 task 就算 task"会让
``这个不需要 task`` 被判成 task——而那种错误没有任何症状。
"""

_VERDICTS = {"chat": Register.CHAT, "task": Register.TASK}

#: 去掉这些再比对。只去**末尾**的标点与引号，不动词本身。
_TRIM = "。.!！?？,，、\"'“”‘’ \t\r\n"


@dataclass(frozen=True)
class ModelVerdict:
    """模型层的一次判定，连同它的原话。

    :attr:`raw` 不是调试残留：报告要能看到"它判不出时到底说了什么"，而那正是
    决定默认值该不该打开这个开关的证据（2026-10-10 的实测说这台机器上的 4B 模型
    判不了，但那是"该调哪个工具"，不是"这句话算哪种语域"——两回事，要重新量）。
    """

    register: Register | None
    """``None`` = 未定。**不猜**：不是那两个词就当作它没回答。"""

    raw: str


def _parse_verdict(reply: str) -> Register | None:
    return _VERDICTS.get(reply.strip().strip(_TRIM).lower())


def classify_with_model(text: str, backend: ChatBackend, *, max_tokens: int = 8) -> ModelVerdict:
    """问一次模型这句话算哪种语域。**默认关**，开启时它是一次额外的 LLM 调用。

    ``backend`` 是注入的——本层不 import ``llm.ollama``（见
    :mod:`biosnn_memterm.agent` 的模块文档），所以这个函数在核心腿里用假后端就能测。

    ``temperature=0.0``：分类不是创作，同一句话两次给出不同答案的模型没法被量。

    ``max_tokens`` 压到 8：这里要的是一个词，而**不传上限的代价在本地模型上很实**——
    它会先自言自语一段再回答。
    """
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=MODEL_PROMPT.format(text=text))],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    reply = "".join(piece.text for piece in backend.stream(request))
    return ModelVerdict(register=_parse_verdict(reply), raw=reply)
