"""装配的 token 算术。

两个常数（每条消息 5、整轮底价 12）是**量出来的**，量法记在
:mod:`axo.session.budget` 的常量文档里。这个文件钉两件事：

1. **公式的形状**——边际代价：往一串消息后面再挂一条，多出来的正好是那一条的
   代价。装配器的贪心遍历、按层反向驱逐都建立在这个形状上（超预算时"去掉最后
   一条能省多少"必须算得准，否则驱逐会在正确的判断上做出错误的取舍）。
2. **余量真的够**——`usable × 1/(1−tolerance) ≤ input_limit`。这是把估计器的
   "不低估 10%" 这个承诺换算成"不会触发服务端截断"的那一步；不成立的话，
   余量就只是一段注释。
"""

from __future__ import annotations

import pytest

from axo.llm.base import ChatMessage, ToolCall
from axo.session.budget import (
    CONVERSATION_BASE,
    DEFAULT_RESERVE_OUTPUT,
    GENERATION_PROMPT,
    IMAGE_TOKENS,
    PER_MESSAGE_OVERHEAD,
    TokenBudget,
    TokenMeter,
)
from memcore.tokens import UNDER_COUNT_TOLERANCE

#: 与 ``tools/calibrate_tokens.py`` 里那个 ``F`` 同一个文本。用它做断言是刻意的：
#: 那三请求差值法量出来的就是它。
F = "记忆终端的装配预算需要一个可靠的每条消息开销常数，" * 3


def user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


@pytest.fixture
def meter() -> TokenMeter:
    return TokenMeter()


# ── 量出来的常数 ────────────────────────────────────────────────────────────


def test_the_measured_constants_are_what_they_are() -> None:
    """12 与 5 的关系是**固定的**，不是两个独立的数。

    它们是同一次测量的两个出口：``A = 60``、``tokens(F) = 48`` 差出 12（底价），
    而 ``B − A − tokens(F)`` 差出 5（每条消息）。把 "每条消息 12" 当成底价用，
    预算会高估一倍多——然后反向驱逐会先赶走真正该留的记忆。
    """
    assert PER_MESSAGE_OVERHEAD == 5
    assert GENERATION_PROMPT == 7
    assert CONVERSATION_BASE == PER_MESSAGE_OVERHEAD + GENERATION_PROMPT == 12


def test_a_single_message_costs_the_base_plus_its_content(meter: TokenMeter) -> None:
    """``A = [user F]`` 实测 60 = 底价 12 + tokens(F) 48 的形状。"""
    message = user(F)
    assert meter.messages([message]) == GENERATION_PROMPT + meter.message(message)
    assert meter.message(message) == meter.text(F) + PER_MESSAGE_OVERHEAD


def test_the_marginal_cost_of_a_message_is_that_message(meter: TokenMeter) -> None:
    """**边际代价的形状。** 挂第 n 条时多出来的，正好是第 n 条自己的代价。

    这一条是装配器贪心与反向驱逐的地基：如果挂第二条比挂第一条便宜（或贵），
    "去掉最后一条能省多少"就会算错，而算错的后果是驱逐该留的、留下该走的。
    """
    first, second, third = user("甲"), user("乙"), user("丙")
    baseline = meter.messages([first])
    assert meter.messages([first, second]) - baseline == meter.message(second)
    assert meter.messages([first, second, third]) - baseline == (
        meter.message(second) + meter.message(third)
    )


def test_an_empty_conversation_costs_nothing(meter: TokenMeter) -> None:
    """没有消息就没有请求。返回底价 12 会让调用方在"这轮发不发"上被误导。"""
    assert meter.messages([]) == 0


def test_the_base_is_paid_once_not_per_message(meter: TokenMeter) -> None:
    """底价只付一次。多挂一条消息只多一条的钱，不再多 12。"""
    one = meter.messages([user(F)])
    two = meter.messages([user(F), user(F)])
    assert two - one == meter.message(user(F))


def test_whitespace_only_content_still_costs_its_wrapper(meter: TokenMeter) -> None:
    """空内容是合法状态（被中断的空回复），它照样占一份包装。"""
    assert meter.text("") == 0
    assert meter.message(user("")) == PER_MESSAGE_OVERHEAD


def test_assistant_wrapper_is_charged_at_the_same_rate(meter: TokenMeter) -> None:
    """实测 assistant 的包装更便宜（交替角色 3.33/条），但一律按 5 收。

    保守的方向是对的，而且两个角色收一样的钱让装配器不必按角色分档——
    分档意味着"能放几轮"取决于消息的**角色配比**，那是个没人能预测的数。
    """
    assert meter.message(ChatMessage(role="user", content=F)) == meter.message(
        ChatMessage(role="assistant", content=F)
    )


# ── 工具调用（P3 的接缝） ───────────────────────────────────────────────────


def test_tool_calls_are_counted_since_they_are_payload(meter: TokenMeter) -> None:
    """``tool_calls`` 是 JSON 形状的正文，和 ``content`` 一样占位置。

    P1 构造的消息恒为空，但**现在就得对**：P3 的第一轮工具调用不该让预算莫名超了。
    """
    plain = ChatMessage(role="assistant", content="")
    calling = ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}),),
    )
    assert meter.message(calling) > meter.message(plain)


def test_no_tool_calls_costs_nothing_extra(meter: TokenMeter) -> None:
    assert meter.tool_calls(()) == 0
    assert meter.message(user("abc")) == meter.text("abc") + PER_MESSAGE_OVERHEAD


# ── 额度 ────────────────────────────────────────────────────────────────────


def test_the_three_layers_of_the_budget() -> None:
    """window → input_limit → usable，一层比一层紧，且方向不能反。"""
    budget = TokenBudget(window=16384, reserve_output=2048)
    assert budget.input_limit == 14336
    # **默认不再打那 10% 折扣**（那是一笔没人看得见的自设额度）。折扣还在，只是要显式要。
    assert budget.usable == 14336
    assert budget.tolerance == 0.0
    assert (
        TokenBudget(window=16384, reserve_output=2048, tolerance=UNDER_COUNT_TOLERANCE).usable
        == 12902
    )
    # tolerance=0 时 usable **等于** input_limit：那道 10% 折扣已经拆掉了
    assert budget.usable == budget.input_limit < budget.window


def test_the_margin_covers_the_estimators_own_promise() -> None:
    """**这个文件的核心断言。**

    估计器承诺"真实值最坏是估计值的 ``1/(1−tolerance)`` 倍"。把额度用满到
    ``usable``，真实值最坏是 ``usable/(1−tolerance)``——它必须仍然塞得进
    ``input_limit``，否则余量就只是一段注释，而超额的表现是服务端静默截断最早
    的消息（人格锚的位置）。
    """
    for window, reserve in ((16384, 2048), (8192, 1024), (32768, 4096), (4096, 512)):
        budget = TokenBudget(window=window, reserve_output=reserve)
        worst_real = budget.usable / (1.0 - budget.tolerance)
        assert worst_real <= budget.input_limit + 1e-9, (
            f"window={window}：把额度用满后真实值最坏 {worst_real:.0f}，"
            f"超过服务端上限 {budget.input_limit}——会被静默截断。"
        )


def test_the_tolerance_is_imported_not_duplicated() -> None:
    """容差与估计器的承诺是**同一个数**。两处各写一份就会漂开。"""
    # 默认值改成 0：以前那 10% 折扣是一笔没人看得见的自设额度（见 TokenBudget.tolerance）
    assert TokenBudget(window=4096).tolerance == 0.0
    assert (
        TokenBudget(window=4096, tolerance=UNDER_COUNT_TOLERANCE).tolerance == UNDER_COUNT_TOLERANCE
    )


def test_the_reserve_is_real_only_if_the_request_gets_it() -> None:
    """``reserve_output`` 必须落到 ``ChatRequest.max_tokens`` 上。

    这里断言的是那个数**存在且非零**——真正的绑定在装配器里（它会构造请求）。
    ``max_tokens`` 为 ``None`` 时 Ollama 用同一个 ``num_ctx`` 生成，输出能一直涨到
    把输入挤掉，那时这个预留就只是账面上的。
    """
    # 默认不预留：只有 Ollama 那条链才需要（它的输出会挤掉输入）
    assert TokenBudget(window=16384).reserve_output == 0
    assert TokenBudget(window=16384, reserve_output=DEFAULT_RESERVE_OUTPUT).reserve_output > 0


def test_a_window_that_leaves_no_input_is_rejected() -> None:
    """这个组合下装配器只能发一条空消息，属于配置错误而不是运行期状况。"""
    with pytest.raises(ValueError, match="不比 reserve_output"):
        TokenBudget(window=2048, reserve_output=2048)
    with pytest.raises(ValueError, match="不比 reserve_output"):
        TokenBudget(window=1024, reserve_output=4096)


def test_bad_parameters_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="window 必须是正数"):
        TokenBudget(window=0)
    with pytest.raises(ValueError, match="不能为负"):
        TokenBudget(window=4096, reserve_output=-1)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TokenBudget(window=4096, tolerance=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TokenBudget(window=4096, tolerance=-0.1)


def test_the_ledger_record_is_complete() -> None:
    """账本里要能回答"这一轮为什么只注入了三条"，所以五层数都要记。"""
    record = TokenBudget(window=16384, reserve_output=2048).to_dict()
    assert set(record) == {"window", "reserve_output", "input_limit", "tolerance", "usable"}
    # tolerance=0 时两者相等（折扣拆掉了）；记录里两个字段都留着，好让"这一轮为什么
    # 只注入了三条"仍然可核对。
    assert record["usable"] == record["input_limit"]


# ── 图：先验常数与它的上限 ──────────────────────────────────────────────────
#
# 这个常数与上面那两个**不是一类东西**：那两个是差值法量出来的，这一个本机量不了
# （没有能看图的模型）。所以这里钉的不是"它准不准"，而是两件能钉的事：**它怎么进
# 公式**，以及**它不会把工具尾巴整个吃掉**。


def test_an_image_adds_a_fixed_cost_per_image(meter: TokenMeter) -> None:
    """一张图 = 一个定值，两张 = 两倍；不带图的那一条**逐 token 不变**。

    按张算而不是按 base64 长度折算：一张 1600 宽的截图与一张 800 宽的差在**分块数**上，
    而那与编码后的字节数不是线性的（PNG 压缩率取决于画面内容）。按字节折算会得出
    "画面复杂一点的截图反而更贵"这种没法解释的结论。
    """
    text = "截了一张屏幕图。"
    plain = ChatMessage(role="tool", content=text, tool_call_id="c1")
    one = ChatMessage(role="tool", content=text, tool_call_id="c1", images=("QUJD",))
    two = ChatMessage(role="tool", content=text, tool_call_id="c1", images=("QUJD", "REVG"))

    assert meter.message(one) == meter.message(plain) + IMAGE_TOKENS
    assert meter.message(two) == meter.message(plain) + 2 * IMAGE_TOKENS


def test_one_image_never_eats_the_whole_tool_tail() -> None:
    """**一张图的计价必须明显小于 L5 的额度。**

    这是这个先验值唯一的硬约束（``IMAGE_TOKENS`` 的 docstring 写了理由）：超了的话，
    工具尾巴会被一张图挤空，而"挤掉"在服务端那一侧是**静默截断最早的消息**——那正是
    装配器与循环两处文档反复点名的最坏形态。

    拿本机那个窗口（``num_ctx`` 16384、给生成留 2048）算，L5 的额度是 **2580**，
    而一张图按 :data:`IMAGE_TOKENS` 记 **1536**——**吃掉约六成**。这条断言要求的是
    "留得下正文"（而不是"刚好小于"）：尾巴里除了图还有工具结果那几句。

    **两张就装不下了**（3072 > 2580），所以"一轮里最多带一张"不是省事，是算出来的；
    真正执行它的是 ``agent/loop.py`` 里那个"图 + 正文超过额度就不带这张图、并在回灌里
    说明"的判断——那一条要等 ``look_at_screen`` 落地才有得测，这里先把数钉住。
    """
    budget = TokenBudget(window=16384, reserve_output=2048)

    # **份额拆掉之后没有"L5 额度"这回事了。** 以前这条断言要求一张图吃得下 L5 的
    # 20% 份额还要给正文留地方；现在图与文字走同一套装箱规则，判据只剩一条：一张图
    # 装得进窗口。装不进的那一刻是真的满了，不是配额用完了。
    assert budget.usable > IMAGE_TOKENS, f"一张图 {IMAGE_TOKENS} ≥ 可用 {budget.usable}"
    assert IMAGE_TOKENS / budget.usable < 2 / 3, (
        f"一张图吃掉 {IMAGE_TOKENS / budget.usable:.0%} 的窗口——正文就没地方放了"
    )


def test_audio_is_metered_by_seconds_not_by_clip() -> None:
    """音频**按秒**扣（先验的每秒常数）：一秒与三十秒差三十倍，按段算会把预算算飞。
    秒数从 WAV 头读；读不出来按回落值（估，不是罚）。"""
    import base64

    from axo.audio import AudioClip
    from axo.session.budget import (
        AUDIO_SECONDS_FALLBACK,
        AUDIO_TOKENS_PER_SECOND,
        TokenMeter,
    )

    meter = TokenMeter()
    plain = meter.message(ChatMessage(role="user", content="你好"))
    one_second = meter.message(
        ChatMessage(role="user", content="你好", audio=(AudioClip(pcm=bytes(32000)).as_base64(),))
    )
    ten_seconds = meter.message(
        ChatMessage(
            role="user",
            content="你好",
            audio=(AudioClip(pcm=bytes(320000)).as_base64(),),
        )
    )

    assert one_second - plain == AUDIO_TOKENS_PER_SECOND
    assert ten_seconds - plain == AUDIO_TOKENS_PER_SECOND * 10

    broken = meter.message(
        ChatMessage(role="user", content="你好", audio=(base64.b64encode(b"not a wav").decode(),))
    )
    assert broken - plain == int(AUDIO_TOKENS_PER_SECOND * AUDIO_SECONDS_FALLBACK), (
        "读不出秒数时该按回落值算，而不是 0（预算 0 = 服务端静默截断最早的消息）"
    )
