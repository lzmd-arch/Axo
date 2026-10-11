"""路由：斜杠解析、规则层、迟滞、模型层。

斜杠是路由里**唯一不需要测量**的一层：它没有"准不准"，只有"对不对"。规则层与模型
自分类的准确率要靠语料量（``benchmarks/routes/``，门槛在
:mod:`tests.test_router_gate`），而斜杠是确定性的——用户说了什么就是什么。

**所以它先做**（P3a 的第 3 步）：它给手动测试一条"强制走任务语域"的路，让后面
那些需要真 Ollama 的验证（工具循环、确认闸门）不必先等路由做完。2026-10-10 的
实测还给了它第二个理由——模型在"该调哪个工具"上五轮只对一轮，于是显式命令不是
便捷方式，而是**主要的可靠入口**。

斜杠的判据分三组，缺一组这个解析器就可能"看起来对"：

1. **识别**：三个命令都认；大小写、前导空白、各种空白分隔符都认；
2. **不误伤**：普通输入**逐字**返回；以斜杠开头但不是命令的（路径、颜文字）
   也逐字返回——这一组最容易漏，而漏了的表现是"用户打不出某句话"；
3. **语义**：``sticky`` 分得开"这一轮"与"从此以后"；``/help`` 不发请求；
   裸命令没有正文可送。

**规则层那一段是这一层最重要的一份文档。** 它是一张决策表，而"表里哪一条先命中"
就是它的行为定义——所以这一段逐条钉住顺序，不只是钉住结论：``谢谢``（0.9，纯附和）
与 ``谢谢你啦``（0.7，句首礼貌词）差的那 0.2 就是第 3 条与第 5 条谁在前的证据。
把顺序搞反不会让任何一条断言变得"不通过"，只会让某些句子悄悄换一个语域。

**迟滞那一段钉的是不对称。** 进 TASK 立即生效、出 TASK 要连续两次，这不是实现
细节而是取舍（见 ``router.py`` 的模块文档）：对称地做会让开箱第一句"帮我查一下 X"
拿不到工具。所以两个方向各有一条断言。
"""

from __future__ import annotations

import pytest
from fakes import ScriptedBackend, text_chunk

from biosnn_memterm.agent.router import (
    _CHAT_MARKERS,
    DEFAULT_HYSTERESIS,
    HELP_TEXT,
    MODEL_CLASSIFY_DEFAULT,
    ModelRole,
    ParsedTurn,
    Register,
    RegisterState,
    RouteDecision,
    RoutedTurn,
    RouteSource,
    SlashCommand,
    classify_with_model,
    decide,
    decide_turn,
    parse_slash,
    route_turn,
    rule_register,
)

# ── 一、识别 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "command", "forced", "text"),
    [
        (
            "/task 帮我查一下上个月那件事",
            SlashCommand.TASK,
            Register.TASK,
            "帮我查一下上个月那件事",
        ),
        ("/chat 今天有点累", SlashCommand.CHAT, Register.CHAT, "今天有点累"),
        ("/task", SlashCommand.TASK, Register.TASK, ""),
        ("/chat", SlashCommand.CHAT, Register.CHAT, ""),
    ],
)
def test_recognizes_commands(raw: str, command: SlashCommand, forced: Register, text: str) -> None:
    """三个命令的四种用法。``text`` 是**真正送给模型的那句**，命令本身不在里面。"""
    parsed = parse_slash(raw)
    assert parsed.command is command
    assert parsed.forced is forced
    assert parsed.text == text


def test_command_token_does_not_reach_the_model() -> None:
    """命令不得混进正文。

    单独一条，因为它是这段代码存在的实际理由：正文会被写进逐字稿、进而被
    ``MemoryWriter`` 当成一句用户陈述记进记忆库。让 "/task" 进正文等于库里多出
    一条"用户说过 /task"——那不是任何人对任何事的陈述，而且它会一直待在那儿。
    """
    assert "/task" not in parse_slash("/task 帮我查一下").text


@pytest.mark.parametrize(
    "raw",
    ["/TASK 帮我查", "/Task 帮我查", "/tAsK 帮我查", "/CHAT 聊两句"],
)
def test_command_word_is_case_insensitive(raw: str) -> None:
    """大小写不敏感。中文输入法下大小写是随机噪声，为它让用户重打一遍没道理。"""
    parsed = parse_slash(raw)
    assert parsed.command is not None
    assert parsed.forced is not None


@pytest.mark.parametrize("raw", ["  /task 帮我查", "\t/task 帮我查", "\n/task 帮我查"])
def test_leading_whitespace_is_tolerated(raw: str) -> None:
    """前导空白不影响识别——手滑敲个空格太常见。"""
    parsed = parse_slash(raw)
    assert parsed.command is SlashCommand.TASK
    assert parsed.text == "帮我查"


@pytest.mark.parametrize("raw", ["/task\t帮我查", "/task\n帮我查", "/task   帮我查"])
def test_any_whitespace_separates_command_from_body(raw: str) -> None:
    """命令与正文之间的分隔符是"一个空白段"，不是"一个空格"。

    粘贴进来的文本常带制表符或换行；只认空格会让 ``/task\\t正文`` 整个变成一句
    普通输入，而用户明明打了命令——一个不报错的错。
    """
    parsed = parse_slash(raw)
    assert parsed.command is SlashCommand.TASK
    assert parsed.text == "帮我查"


def test_body_keeps_inner_whitespace() -> None:
    """正文两侧去空白，**内部不动**——多行正文要原样送达。"""
    parsed = parse_slash("/task 第一行\n第二行  缩进")
    assert parsed.text == "第一行\n第二行  缩进"


# ── 二、不误伤普通输入 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "你好",
        "今天有点累，随便聊两句吧。",
        "帮我看看 /etc/hosts 里写了什么",  # 斜杠在中间：不是命令
        "  带前导空格的普通输入",
        "换行结尾的输入\n",
        "结尾有空格 ",
    ],
)
def test_ordinary_input_is_returned_verbatim(raw: str) -> None:
    """普通输入**逐字**返回，一个字符都不动。

    为什么连前导/尾随空白都不 strip：逐字稿里存的应当就是用户打的字。顺手规整
    一下不报错、不影响功能，但某天有人对着逐字稿查"为什么这段显示不一样"时会
    找不到原因。**看不见的改动比看得见的错更费时间。**
    """
    parsed = parse_slash(raw)
    assert parsed.command is None
    assert parsed.forced is None
    assert parsed.text == raw


@pytest.mark.parametrize(
    "raw",
    [
        "/home/user/笔记.txt 里写了什么",  # 路径：用户就是在问一个路径
        "/etc/hosts",
        "/unknown 参数",  # 不存在的命令
        "/taskk 打错了一个字母",
        "/",  # 光一个斜杠
        "//双斜杠",
        "/ 空格后是斜杠",
    ],
)
def test_unrecognized_slash_is_ordinary_input(raw: str) -> None:
    """认不出的斜杠**原样**当普通输入，不报错、不吞掉。

    这一组是最容易漏的：把"未知命令"做成一条错误提示，读起来像体贴，实际是让
    用户**没法在句首提一个路径**。而报错文案还会暗示"你打错了命令"，可用户根本
    没想打命令。
    """
    parsed = parse_slash(raw)
    assert parsed.command is None, f"{raw!r} 被当成了命令"
    assert parsed.text == raw, "普通输入被改动了"
    assert parsed.forced is None


def test_slash_in_the_middle_is_not_a_command() -> None:
    """只有**开头**的斜杠算命令。"""
    parsed = parse_slash("你好 /task 帮我查")
    assert parsed.command is None
    assert parsed.text == "你好 /task 帮我查"


def test_empty_input() -> None:
    assert parse_slash("") == ParsedTurn(command=None, text="")


# ── 三、语义：只此一轮 vs 从此以后 ─────────────────────────────────────────


def test_help_is_local_and_carries_no_request() -> None:
    """``/help`` 不出正文、不强制语域——调用方本地渲染 ``HELP_TEXT`` 就完事。

    ``is_help`` 这条断言是给调用方的分支判据：它必须**在发请求之前**被问到，
    否则会为一句"显示帮助"发起一次模型调用，而模型会拿空正文答一段不相干的话。
    """
    parsed = parse_slash("/help")
    assert parsed.command is SlashCommand.HELP
    assert parsed.is_help is True
    assert parsed.text == ""
    assert parsed.forced is None
    assert not parsed.is_bare_switch, "/help 不是语域切换，别把它当切换处理"


def test_help_ignores_trailing_arguments() -> None:
    """``/help 后面写了东西``也当帮助。为一个无关紧要的笔误报错，是让用户白跑一趟。"""
    assert parse_slash("/help 这算什么").is_help is True


@pytest.mark.parametrize(
    ("raw", "sticky"),
    [
        ("/task", True),
        ("/chat", True),
        ("/task   ", True),  # 只有空白 = 没正文
        ("/task 帮我查", False),
        ("/chat 聊两句", False),
    ],
)
def test_sticky_distinguishes_this_turn_from_from_now_on(raw: str, sticky: bool) -> None:
    """``sticky`` 分得开两种语义。

    ``/task 帮我查 X`` 是"就这一件事"，``/task`` 是"接下来聊的都是正事"。
    把它们合并成"进入任务态"会让前者变成"用户从此进了任务模式"——而用户只是想
    查一件事，下次闲聊时会被当成任务处理、开始调工具。**这个错误没有任何症状，
    只会让助手偶尔显得莫名其妙地积极。**
    """
    assert parse_slash(raw).sticky is sticky


@pytest.mark.parametrize("raw", ["/task", "/chat", "/task   ", "\t/chat"])
def test_bare_switch_has_no_text_to_send(raw: str) -> None:
    """裸命令没有正文——调用方据此判断"这一轮不发请求"。"""
    parsed = parse_slash(raw)
    assert parsed.is_bare_switch is True
    assert parsed.text == ""


@pytest.mark.parametrize("raw", ["/task 帮我查", "/chat 聊两句", "普通输入", "/help"])
def test_non_bare_switch_is_not_a_switch(raw: str) -> None:
    """反空转：``is_bare_switch`` 不是"永远真"。

    否则一个把每条输入都当裸切换的调用方会**一句话都不发出去**，而所有断言
    仍然绿——界面看起来只是"模型不回答"。
    """
    assert parse_slash(raw).is_bare_switch is False


# ── 帮助文案与枚举 ─────────────────────────────────────────────────────────


def test_help_text_mentions_every_command() -> None:
    """``HELP_TEXT`` 里必须出现每一个命令。

    加一个命令却忘了写进帮助，用户就无从知道它存在——而这个疏漏不会有任何
    症状。这条把它变成一个会红的断言。
    """
    for command in SlashCommand:
        assert command.value in HELP_TEXT, f"帮助文案里没写 {command.value}"


def test_register_values_are_stable_strings() -> None:
    """两个语域的**字符串值**是接口：它们会进逐字稿元数据与路由报告。

    改一个值等于改一次数据格式（老的报告/逐字稿里会是旧值），所以钉住。
    """
    assert Register.CHAT.value == "chat"
    assert Register.TASK.value == "task"
    assert str(Register.TASK) == "task"


# ── 四、规则层：一张决策表 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "register", "confidence"),
    [
        # 1. 否定紧邻 task 信号 → CHAT。**这一条必须排在最前**：下面这些句子里
        #    明摆着有 task 动词，只是被否定了。
        ("别帮我查了，我自己来吧", Register.CHAT, 0.75),
        ("先别急着帮我查", Register.CHAT, 0.75),
        ("不用提醒我了", Register.CHAT, 0.75),
        # 反例：``不`` 不是否定词（"我不太确定，帮我查一下"是一句正常的委托）。
        # 收进来会让最常见的犹豫式委托全部失效。
        ("我不太确定，帮我查一下", Register.TASK, 0.85),
        # 2. task 动词 → TASK；只有位置信号时置信度低一档（0.7）
        ("帮我找一下去年十月的那份记录", Register.TASK, 0.85),
        ("读一下 docs/plan.md", Register.TASK, 0.85),
        ("桌面上的那个文件夹里有什么", Register.TASK, 0.7),
        ("今天星期几", Register.TASK, 0.85),
        # 3. 整句**由应和词拼成** → CHAT，置信度高于下面第 4 条（嘟囔）。0.9 与 0.8 的
        #    差就是这两条谁在前的证据。
        ("好的", Register.CHAT, 0.9),
        ("收到", Register.CHAT, 0.9),
        ("ok", Register.CHAT, 0.9),
        #    拼出来的也算：``好的，知道了`` 的判据是"全句切得完"，不是整句相等。
        ("好的，知道了", Register.CHAT, 0.9),
        #    ``嗯嗯嗯嗯`` 走的是**这一条**（``嗯`` 本身是应和词，四个"嗯"拼得完）。
        #    2026-10-12 之前它走第 4 条，那时第 3 条要求整句相等——顺序变了，结论
        #    没变，正是下面那条 docstring 说的、只有置信度能抓住的那种退化。
        ("嗯嗯嗯嗯", Register.CHAT, 0.9),
        # 4. 重复的嘟囔，按不同的字数判 → CHAT。两个见证都**不是**应和词拼出来的，
        #    所以它们仍然只能走这一条。
        ("哈哈哈哈哈", Register.CHAT, 0.8),
        ("？？？", Register.CHAT, 0.8),
        # 5. 句首礼貌词 → CHAT。``谢谢你啦`` 里明明有 ``谢谢你``（第 6 条的闲聊标记），
        #    拿到 0.7 说明第 5 条先命中了。
        ("谢谢你啦", Register.CHAT, 0.7),
        ("辛苦了", Register.CHAT, 0.7),
        # 6. 情绪/寒暄/话题引子 → CHAT
        ("今天天气不错", Register.CHAT, 0.8),
        ("我该怎么办？", Register.CHAT, 0.8),
        # 7. 句末问号兜底 → CHAT，最低的置信度
        ("你在想什么？", Register.CHAT, 0.6),
        # 8. 一条都没命中 → **None**。不猜。
        ("把那个东西弄一下", None, 0.0),
        ("就这样吧", None, 0.0),
        ("", None, 0.0),
        ("   ", None, 0.0),
    ],
)
def test_rule_layer_decision_table(text: str, register: Register | None, confidence: float) -> None:
    """规则层的每一档都钉一条。**置信度也断言**——它是顺序的证据。

    置信度自称"给人看的、不参与判定"，但它**恰好是唯一能区分"第几条命中了"的东西**：
    ``哈哈哈哈哈`` 同时命中第 4 条与第 6 条，两边都是 CHAT，只有 0.8 与 0.8……一致；
    而 ``好的`` 命中第 3、4 条，0.9 与 0.8 分得开。所以顺序改坏了、结论没变的那种
    退化，靠结论断言是抓不住的。
    """
    assert rule_register(text) == (register, confidence)


# ── 四之二、"去看已经存下来的东西"那一族 ─────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        # 用户**实际打过的**说法（2026-10-11 的实测现场）。"查看"/"看看"本身不在
        # 表里（见 ``_TASK_MARKERS`` 上面那段"刻意不收"），所以兜住它们的是
        # "说的是记忆这件事"。
        "查看我的记忆",
        "看看我的记忆",
        "我的记忆里有什么？",
        "记忆里关于颜色的那条是什么",
        "记忆库里都有什么",
        # 主力词：``上次``。它不是动词，收它是因为它指的是"这件事发生过，而记录
        # 不在你眼前"——也就是"去查"。
        "我上次说想买什么",
        "我们上次聊到哪儿了？",
        # 语料点名的同义动词缺口：表里原本只有 ``记一下``/``念一下``/``忘掉``。
        "记录一下我今天的感受",
        "把昨天那段话再念一遍",
        "别忘了我说的",
        "回想一下那次",
    ],
)
def test_the_memory_lookup_family_reaches_the_tools(text: str) -> None:
    """查记忆的那一族必须判成 task，**哪怕它是问句**。

    这一族漏掉时很有欺骗性：句子没问题、词表也没坏，是**第 7 条**（句末问号 →
    CHAT）抢在前面把整族吃掉了。而"查记忆"天然是问句（"我上次说想买什么？"），
    所以这一族几乎是精确地踩在第 7 条上。

    后果不是"答得不好"而是"这一轮没有工具"：语域判成 CHAT → 那一轮**不声明任何
    工具**（``agent/loop.py`` 的 ``offered = ()``）→ 模型只能照 ``session/
    environment.py`` 补的那句事实说"这一轮没有工具可用"。用户看到的是模型在
    推卸，实际是路由没给它工具。

    所以这里的断言要点在**问号那两条**上：``我的记忆里有什么？`` 与 ``我们上次聊到
    哪儿了？`` 判成 task 才是对的，而它们正是靠"标记在第 7 条之前命中"才对的。
    """
    assert rule_register(text) == (Register.TASK, 0.85)


def test_别忘_is_not_read_back_as_a_negation() -> None:
    """``别忘`` 整词入表，但它自己不构成否定。

    否定判据看的是标记**之前** ``_NEGATION_WINDOW`` 个字里有没有 ``别``/``不要``。
    ``别忘`` 是**标记本身**，那个 ``别`` 在标记内部、不在窗口里，所以不会被
    ``_NEGATIONS`` 反过来判成闲聊。

    这条不是假想的边界——把 ``别忘`` 拆成先找 ``忘掉`` 那样的写法，窗口就会盖住
    ``别``，于是"别忘了我说的"（一句请求）变成 CHAT，而用户拿不到工具。同一族里
    ``别`` 与标记重叠的写法只此一个，所以单独钉一条。
    """
    assert rule_register("别忘了我说的") == (Register.TASK, 0.85)
    # 反例：真否定仍然要被抓住——``别`` 在窗口里、且没有标记盖住它。
    assert rule_register("别帮我查了，我自己来吧") == (Register.CHAT, 0.75)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("你看看这个笑话", "「看看」不是动词信号——它后面跟的常是笑话，不是文件"),
        ("你几点睡的", "「几点」单独出现时问的是对方，不是钟点"),
        ("记得昨天那个吗", "「记得」是回忆，不是「去查」；收的是「记住」/「记一下」"),
    ],
)
def test_the_words_we_chose_not_to_collect_land_on_no_verdict(text: str, why: str) -> None:
    """刻意不收的那三个词，落点必须是 ``None``。

    ``None`` 是**"判不出来"**，不是 CHAT——这条区别有实际后果：``None`` 会交给
    迟滞状态机（沿用上一轮语域），冷启动时才转 CHAT；而判成 CHAT 会立刻把工具收走。

    钉这一条是为了挡住一种很自然的"改进"：往表里补 ``看看``/``记得`` 能让语料
    分数好看一点（它们确实出现在几条该判 task 的句子里），代价是把"你看看这个
    笑话"、"记得昨天那个吗"一起判成 task。分数的来源是那 120 条，而代价落在
    日常闲聊上——用户看不见后者。
    """
    assert rule_register(text)[0] is None, f"{text} —— {why}"


@pytest.mark.parametrize(
    "text",
    [
        "好的，知道了",
        "嗯，明白了",
        "行，收到",
        "嗯嗯，好的好的",
        "明白了啦",
        "知道了。",
        "thank you",
    ],
)
def test_a_composed_acknowledgement_is_chitchat(text: str) -> None:
    """应和是**拼**出来的，第 3 条不能要求整句精确等于一个应和词。

    判据**不能是** ``folded in ACKNOWLEDGEMENTS``：``normalize`` 只做 NFKC +
    casefold、**不删标点**，所以「好的，知道了」既不是「好的」也不是「知道了」，
    一路走到第 8 条成了"判不出来"。语料上的现场是 ``u-0050``。

    后两条各自钉一个不同的缺口：``明白了啦`` 是**句尾语气助词**（``ACKNOWLEDGEMENTS``
    里是 ``明白`` 而不是 ``明白了``，那份表对尾缀本来就不一致），``知道了。`` 是
    **标点**——两种都跟语域无关。
    """
    assert rule_register(text) == (Register.CHAT, 0.9)


@pytest.mark.parametrize(
    "text",
    [
        "好，我们开始吧",  # 切到「我」就断了，没有全句覆盖
        "我知道了你在想什么",  # 含有「知道了」，但它不是应和
        "那好吧",
    ],
)
def test_acknowledgement_wants_the_whole_sentence(text: str) -> None:
    """判据是**全句切得完**，不是"含有应和词"。

    放宽成"含有"会让任何以「好的」开头的句子都变成闲聊——而第 5 条（句首礼貌词）
    刻意只给 0.7，正是为了不在这件事上把话说满。这三条落回 ``None``（"判不出来"），
    不是 CHAT。
    """
    assert rule_register(text)[0] is None


def test_a_state_word_is_chitchat_but_a_topic_noun_is_not() -> None:
    """收 ``心情``，**不收** ``雨``/``花``/``梦``/``心跳``——分界线是"关于什么"。

    2026-10-12 量过一轮：把 ``梦``/``雨``/``花`` 收进闲聊表能让语料的歧义标志从
    0.1500 掉到 0.1083（chitchat 那七条**一条不剩**）。那几个数好看，但那是**对着
    那七句话过拟合**：话题名词说的是"这句话关于什么"，不是"这句话是要对方做件事
    还是说句话"——任何语域都可能有那个话题。

    同一轮还量出两个"看起来该有用、其实没命中"的候选：``做梦`` 接不住
    ``u-0062``（正文"我今天做了个奇怪的``梦``"，中间隔着字），``下雨`` 接不住
    ``u-0068``（正文"外面的``雨``下得挺大"）。这两个词是照着语料的**意思**挑的，
    不是照着**文字**挑的——所以这条断言连它们一起钉住。
    """
    assert rule_register("我今天心情特别好") == (Register.CHAT, 0.8)
    # 状态词与话题名词混在一句里时，任务标记仍然优先（第 2 条在闲聊表之前）。
    assert rule_register("帮我记一下我的心情") == (Register.TASK, 0.85)

    # 行为那一面：语料里那三条落在 ``None``（"判不出来"，交给迟滞状态机）。
    for text in ("我养的那盆花开了", "外面的雨下得挺大", "我今天做了个奇怪的梦"):
        assert rule_register(text)[0] is None

    # 表格那一面：直接钉住词表。行为断言挡不住"有人换个法子把它们捞进去"，
    # 而这条负例针对的恰恰是**词表**这一步。
    for topic_noun in ("雨", "花", "梦", "心跳", "做梦", "下雨"):
        assert topic_noun not in _CHAT_MARKERS, f"{topic_noun} 是话题名词，不是语域信号"


def test_rule_layer_answers_both_ways_and_says_when_it_cannot() -> None:
    """反空转：这张表**三种回答都给得出**。

    一个"永远 TASK"或"永远 None"的表能过掉上面每一条各自独立的断言吗？不能——但
    它可以过掉"准确率"那一类的聚合断言（一个永远 TASK 的表在 task 占多数的语料上
    看着还行）。所以这里明写一条：三种回答各至少出现一次。
    """
    sample = ["帮我查一下那个文件.txt", "今天有点累", "把那个东西弄一下"]
    answered = {rule_register(text)[0] for text in sample}
    assert answered == {Register.TASK, Register.CHAT, None}


# ── 五、decide：无状态的一票 ───────────────────────────────────────────────


def test_decide_reports_the_rule_layer_source() -> None:
    decision = decide("帮我查一下那份记录")
    assert decision == RouteDecision(
        register=Register.TASK, source=RouteSource.RULE, confidence=0.85
    )


def test_decide_falls_back_to_chat_and_flags_ambiguity() -> None:
    """判不出来 → CHAT + ``ambiguous``。

    **两者必须同时发生。** 只退回 CHAT 而不出标志，就是一次静默的误判（这一轮不给
    工具，用户不知道为什么）；只出标志而不退回一个确定的语域，调用方就没得用。
    """
    decision = decide("把那个东西弄一下")
    assert decision.register is Register.CHAT
    assert decision.source is RouteSource.DEFAULT
    assert decision.ambiguous is True


def test_decide_is_pure() -> None:
    """同一句问两遍逐位相同——它是量具能取到数、门槛能取到 0.02 容差的前提。"""
    assert decide("帮我查一下那份记录") == decide("帮我查一下那份记录")
    assert decide("把那个东西弄一下") == decide("把那个东西弄一下")


# ── 六、decide_turn：斜杠压过规则层 ────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "register", "text"),
    [
        # 规则层对 ``今天有点累`` 会判 CHAT——这里强制 TASK
        ("/task 今天有点累", Register.TASK, "今天有点累"),
        # 反过来：规则层对 ``帮我查一下库里那条记录`` 会判 TASK——这里强制 CHAT。
        # **这条是斜杠层存在的全部意义**：用户说了算。
        ("/chat 帮我查一下库里那条记录", Register.CHAT, "帮我查一下库里那条记录"),
        # 裸切换：只改状态，这一轮没有正文
        ("/task", Register.TASK, ""),
        ("/chat", Register.CHAT, ""),
    ],
)
def test_slash_overrides_the_rule_layer(raw: str, register: Register, text: str) -> None:
    parsed, decision = decide_turn(raw)
    assert decision.register is register
    assert decision.source is RouteSource.SLASH
    assert decision.ambiguous is False, "显式指定不该是歧义——用户已经把话说清楚了"
    assert parsed.text == text


def test_unrecognized_slash_falls_through_to_the_rules() -> None:
    """认不出的斜杠是**普通输入**，于是它照样能被规则层判成 task。

    ``/home/user/笔记.txt 里写了什么`` 里的斜杠不是命令，而那句话本身就点了一个
    文件——两条路合起来的结果是 TASK 且来源是 ``rule``。把认不出的斜杠做成报错，
    用户就没法在句首提一个路径。
    """
    parsed, decision = decide_turn("/home/user/笔记.txt 里写了什么")
    assert parsed.command is None
    assert decision.register is Register.TASK
    assert decision.source is RouteSource.RULE


def test_decide_turn_has_no_hysteresis() -> None:
    """``decide_turn`` 是**无状态的一票**，迟滞不在这一层。

    单独断言是因为量具（``benchmarks/routes/``）整份读数都建立在这条性质上：语料是
    一句句独立输入，把迟滞接进来，读数就会取决于语料的排列顺序——而顺序是任意的。
    """
    # 连问两次反向的话，两次结论必须**一样**（有状态的话第二次会被压住）。
    first = decide_turn("帮我查一下那份记录")[1]
    second = decide_turn("帮我查一下那份记录")[1]
    assert first == second


# ── 七、迟滞：不对称，且"不知道"不是一个方向 ────────────────────────────────


def _chat() -> RouteDecision:
    return RouteDecision(register=Register.CHAT, source=RouteSource.RULE, confidence=0.8)


def _task() -> RouteDecision:
    return RouteDecision(register=Register.TASK, source=RouteSource.RULE, confidence=0.85)


def test_entering_task_is_immediate() -> None:
    """进 TASK **一次就生效**——开箱第一句"帮我查一下 X"必须拿得到工具。

    对称地做迟滞的话，状态从 CHAT 起步，一次 task 信号不够两次，助手会回答"我查
    不了"——而那正是本版要交付的那件事，不能死在门口。
    """
    state = RegisterState()
    effective = state.advance(_task())
    assert effective.register is Register.TASK
    assert effective.source is RouteSource.RULE, "生效了就不该被改写成 STATE"
    assert state.register is Register.TASK


def test_leaving_task_needs_consecutive_reverse_signals() -> None:
    """落回 CHAT 要连续 ``DEFAULT_HYSTERESIS`` 次；中途被压住的那一轮**沿用旧语域**。"""
    state = RegisterState()
    state.advance(_task())

    first = state.advance(_chat())
    assert first.register is Register.TASK, "第一次反向信号不该立刻切"
    assert first.source is RouteSource.STATE, "被压住了就要说清楚是状态压的，不是规则层判的"
    assert state.register is Register.TASK

    second = state.advance(_chat())
    assert second.register is Register.CHAT
    assert second.source is RouteSource.RULE
    assert state.register is Register.CHAT


def test_a_task_signal_resets_the_streak() -> None:
    """中间夹一次 task 信号，计数从头开始。

    没有这一条，用户在任务里说一句"好的"（第一次反向）再说一句正事（task）之后
    只再说一句"好的"就会掉出 TASK——而那个"好的"与前面那次之间隔着一整件正事。
    """
    state = RegisterState()
    state.advance(_task())
    state.advance(_chat())  # 计数 1
    state.advance(_task())  # 清计数
    assert state.streak == 0

    assert state.advance(_chat()).register is Register.TASK
    assert state.register is Register.TASK


def test_same_direction_without_a_streak() -> None:
    """一直在 CHAT 里说闲聊：不会攒出什么东西来（``streak`` 保持 0）。"""
    state = RegisterState()
    for _ in range(3):
        assert state.advance(_chat()).register is Register.CHAT
    assert state.streak == 0


def test_ambiguity_neither_counts_nor_clears() -> None:
    """歧义的一轮**既不算反向信号，也不清计数**。

    清掉的话，任务中间夹一句含糊的话就会让"马上要落回闲聊"的计数重新开始；算作
    反向的话，两句含糊的话就能把用户踢出任务语域——而"不知道"根本不是一个方向。
    """
    state = RegisterState()
    state.advance(_task())
    state.advance(_chat())  # 计数 1
    assert state.streak == 1

    ambiguous = RouteDecision(
        register=Register.CHAT, source=RouteSource.DEFAULT, confidence=0.0, ambiguous=True
    )
    returned = state.advance(ambiguous)
    assert returned is ambiguous, "歧义的结论要原样返回，别在状态机里改它"
    assert state.streak == 1, "歧义不该清计数"

    assert state.advance(_chat()).register is Register.CHAT, "第二句反向的话仍然算数"


def test_slash_ignores_hysteresis() -> None:
    """显式命令**无视迟滞**，两个方向都是，并且把计数清零。

    用户刚打了 ``/chat``，而系统按住不放说"还在任务里"——那是不听人话。同理
    ``/task`` 要立刻进任务态，哪怕上一句刚说"好的"攒了一次反向信号。
    """
    state = RegisterState()
    state.advance(_task())
    state.advance(_chat())  # 计数 1，正卡在迟滞里

    slash_chat = RouteDecision(register=Register.CHAT, source=RouteSource.SLASH, confidence=1.0)
    assert state.advance(slash_chat).register is Register.CHAT
    assert state.register is Register.CHAT
    assert state.streak == 0

    slash_task = RouteDecision(register=Register.TASK, source=RouteSource.SLASH, confidence=1.0)
    assert state.advance(slash_task).register is Register.TASK
    assert state.register is Register.TASK


def test_threshold_is_configurable_and_validated() -> None:
    """``threshold=1`` = 不迟滞；小于 1 是配错了，当场炸。"""
    state = RegisterState()
    state.advance(_task())
    assert state.advance(_chat(), threshold=1).register is Register.CHAT

    with pytest.raises(ValueError, match="至少是 1"):
        RegisterState().advance(_chat(), threshold=0)


def test_default_hysteresis_is_what_the_docs_say() -> None:
    """``DEFAULT_HYSTERESIS`` 是 2。

    钉住它是因为模块文档、:class:`RegisterState` 的注释与语料报告都以"两次"为前提
    写的；改成 1 或 3 之后那些文字会变成**读起来仍然通顺的错误说明**。
    """
    assert DEFAULT_HYSTERESIS == 2


# ── 八、模型层与 ModelRole ─────────────────────────────────────────────────


def test_model_classify_is_off_by_default() -> None:
    """出厂默认**关**。这条是用户拍板的那一项裁决，钉住它。

    它默认关不是保守，是量出来的：2026-10-10 实测这台机器上的 4B 模型在"两个工具
    该调哪个"上五轮只对一轮。改它要由 ``benchmarks/routes/`` 的报告定。
    """
    assert MODEL_CLASSIFY_DEFAULT is ModelRole.OFF


def test_off_never_calls_the_model() -> None:
    """关着的时候**一次都不问**——包括规则层判不出来的那些句子。"""

    def explode(_text: str) -> Register | None:  # pragma: no cover - 被调用即失败
        raise AssertionError("model_role=OFF 时不该调用模型")

    decision = decide("把那个东西弄一下", classify=explode)
    assert decision.source is RouteSource.DEFAULT
    assert decision.ambiguous is True


def test_fallback_only_asks_when_the_rules_are_lost() -> None:
    """``FALLBACK`` 只在规则层判不出来时问模型——**这就是它的全部意义**。

    规则层认得的句子不问，所以它的**确定性**保留下来：同一个句子不会因为模型的兴致
    而这一轮是 task、下一轮是 chat。断言"没问"比断言"问出来的结果对"更要紧。
    """
    asked: list[str] = []

    def classifier(text: str) -> Register | None:
        asked.append(text)
        return Register.TASK

    ruled = decide("帮我查一下那份记录", classify=classifier, model_role=ModelRole.FALLBACK)
    assert ruled.source is RouteSource.RULE, "规则层认得的句子不该改判"
    assert asked == [], "规则层认得的句子根本不该问模型"

    guessed = decide("把那个东西弄一下", classify=classifier, model_role=ModelRole.FALLBACK)
    assert guessed.source is RouteSource.MODEL
    assert guessed.register is Register.TASK
    assert asked == ["把那个东西弄一下"]


def test_fallback_keeps_the_ambiguity_flag_when_the_model_shrugs() -> None:
    """模型也判不出来 → 仍然退回 CHAT + 歧义。**不猜。**"""

    def shrug(_text: str) -> Register | None:
        return None

    decision = decide("把那个东西弄一下", classify=shrug, model_role=ModelRole.FALLBACK)
    assert decision.register is Register.CHAT
    assert decision.source is RouteSource.DEFAULT
    assert decision.ambiguous is True


def test_only_asks_for_every_sentence() -> None:
    """``ONLY`` 是量具那一档：规则层不参与，每句都问。

    它存在的理由见 ``router.py`` 的 :class:`ModelRole`——报告要能回答"**光靠模型
    够不够**"，而那既不是"规则层 + 兜底"，也不是"关掉模型"。
    """
    asked: list[str] = []

    def classifier(text: str) -> Register | None:
        asked.append(text)
        return Register.CHAT

    # 这句话规则层会判 TASK，但 ONLY 档不看规则层。
    decision = decide("帮我查一下那份记录", classify=classifier, model_role=ModelRole.ONLY)
    assert decision.register is Register.CHAT
    assert decision.source is RouteSource.MODEL
    assert decision.confidence == 0.5, "模型层不共享规则层的刻度，不该假装报一个高置信"
    assert asked == ["帮我查一下那份记录"]


def test_only_without_a_classifier_is_a_configuration_error() -> None:
    """``ONLY`` 没给 ``classify`` → ``ValueError``。

    静默退回规则层的话，一份本该是"纯模型"的报告会变成"规则层"的报告，而**数字看起来
    完全正常**——那正是最坏的一种错。
    """
    with pytest.raises(ValueError, match="ONLY"):
        decide("帮我查一下那份记录", model_role=ModelRole.ONLY)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("task", Register.TASK),
        ("TASK", Register.TASK),  # 大小写不敏感
        (" chat\n", Register.CHAT),  # 两侧空白与换行
        ("task。", Register.TASK),  # 末尾标点
        ("“task”", Register.TASK),  # 引号
        ("", None),
        ("   ", None),
        ("这个要说 task", None),  # **不是整词就不算**——见 MODEL_PROMPT 的说明
        ("我不知道", None),
    ],
)
def test_model_verdict_requires_exactly_one_word(reply: str, expected: Register | None) -> None:
    """解析契约：**不是那两个整词就当作它没回答。**

    放宽成"包含 task 就算 task"会让"这个不需要 task"被判成 task——而那种错误没有
    任何症状，只会让某一轮多出几个工具。
    """
    backend = ScriptedBackend(script=[[text_chunk(reply, done=True)]])
    assert classify_with_model("随便一句", backend).register is expected
    assert classify_with_model("随便一句", backend).raw == reply


def test_model_verdict_pins_temperature_and_max_tokens() -> None:
    """``temperature=0`` 与一个很小的 ``max_tokens`` 都要真的传下去。

    前者：同一句话两次给出不同答案的模型没法被量。后者：不传上限时本地模型会先
    自言自语一段再回答，而这里要的只是一个词。
    """
    backend = ScriptedBackend(script=[[text_chunk("task", done=True)]])
    classify_with_model("随便一句", backend, max_tokens=4)
    (request,) = backend.requests
    assert request.temperature == 0.0
    assert request.max_tokens == 4
    assert request.tools == (), "自分类不是工具调用，别把工具声明带上去"
    assert len(request.messages) == 1


# ── 九、RoutedTurn 的两个判据 ──────────────────────────────────────────────


def _routed(raw: str) -> RoutedTurn:
    return route_turn(raw, RegisterState())


def test_uses_tools_and_clarification_are_mutually_exclusive() -> None:
    """三个语域各一条。**歧义时一定不给工具**——拿不准的一轮什么都不该动。

    改状态的动作有确认闸门兜着，但"读一个文件""搜一次记忆"**没有闸门**，而在一句
    闲聊上翻用户的库是不该发生的事。
    """
    task = _routed("/task 帮我查一下")
    chat = _routed("/chat 今天有点累")
    unclear = _routed("把那个东西弄一下")

    assert task.uses_tools is True
    assert task.wants_clarification is False

    assert chat.uses_tools is False
    assert chat.wants_clarification is False

    assert unclear.register is Register.CHAT
    assert unclear.uses_tools is False, "歧义的一轮不给工具"
    assert unclear.wants_clarification is True, "判不出来要请模型带一句确认"


def test_routed_turn_exposes_the_text_without_the_command() -> None:
    """``RoutedTurn.text`` 是**摘掉命令之后**的正文——它是唯一该进逐字稿的那份。"""
    assert _routed("/task 帮我查一下").text == "帮我查一下"
    assert _routed("普通一句").text == "普通一句"


def test_hysteresis_does_not_leak_into_the_plain_reading() -> None:
    """同一个实例上连着两轮闲聊：第二轮仍然是 CHAT（因为第一轮就已经是 CHAT）。

    这条是上面"迟滞不对称"那一段的收口——它顺带说明"起步状态是 CHAT"，于是
    开箱第一句闲聊不会被什么历史压住。
    """
    state = RegisterState()
    first = route_turn("今天有点累", state)
    second = route_turn("今天天气不错", state)
    assert first.register is Register.CHAT
    assert second.register is Register.CHAT
    assert second.decision.source is RouteSource.RULE


# ── 十、route_turn 只推进该推进的那一半 ────────────────────────────────────


def test_a_slash_with_text_does_not_advance_the_state() -> None:
    """``/task 正文`` 是"这一件事"，**不是**"从此进任务模式"。

    ``decide_turn`` 对裸斜杠与带正文斜杠都给 ``RouteSource.SLASH``，所以
    ``RegisterState.advance`` **分不出这两者**——而它见到 ``SLASH`` 会无视迟滞直接
    覆盖。于是只要把这一轮的结论原样喂进去，``/task 正文`` 就会把状态钉成 TASK，
    与 README 写的"命令只影响这一轮"相反。症状是：用户用 ``/task`` 做了一次性的事，
    之后连着两句闲聊都被当成任务（要爬两次迟滞才落回来），而界面上看不出为什么。

    区分它们的只有 ``ParsedTurn.sticky``，所以这条短路只认它。
    """
    state = RegisterState()
    routed = route_turn("/task 帮我查一下", state)

    assert routed.register is Register.TASK, "这一轮要给工具"
    assert state.register is Register.CHAT, "但状态机不该被推进去"
    assert state.streak == 0, "非粘连的命令连计数都不该碰"


def test_a_bare_slash_switch_does_advance_the_state() -> None:
    """与上一条成对。裸 ``/task`` 就是**在切状态**这个动作本身，不推进它才是坏的。

    两条一起才有意义：只写前者的话，把整个 ``advance`` 从 ``route_turn`` 里删掉也
    全绿——那样迟滞就永远不生效了。
    """
    state = RegisterState()
    route_turn("/task", state)

    assert state.register is Register.TASK
