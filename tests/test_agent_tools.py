"""工具注册表与四个放行工具。

这个文件的重点**不是**"工具能跑"，而是三条容易写成假绿的纪律：

1. **派发口在任何情况下都返回一个 :class:`ToolOutcome`，绝不往上抛**——
   :func:`test_dispatch_never_raises` 把四种失败各扫一遍。它不检查消息写得
   好不好，只检查"有没有东西逃出去"，因为一旦逃出去，这一轮就整段结束，而模型
   连改一次的机会都没有。
2. **闸门的拒绝只对声明了 ``requires_approval`` 的工具生效**——反空转那条
   （:func:`test_gate_denial_does_not_touch_approval_free_tools`）保证
   "``approved=False`` 就什么都不执行"这种过度实现会被抓住。
3. **``search_memory`` 不改动记忆库**——它是模型的一次试探，与记忆界面里那个
   "以这条为线索召回一次"是同一类动作（见 ``tests/test_recall_settle.py``）。

测试里的库一律是 ``tmp_path`` 下的真 SQLite（不是 ``:memory:``）：``recall`` 会
走真实的扫描路径，而"换成内存库"会让这条断言测的是另一个东西。
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from axo.agent.catalog import (
    MAX_REMINDER_CHARS,
    MAX_WRITE_CHARS,
    EditFileTool,
    FetchUrlTool,
    ForgetMemoryTool,
    GetTimeTool,
    ListFilesTool,
    LookAtScreenTool,
    ReadFileTool,
    ReadMemoryTool,
    RunCommandTool,
    SearchFilesTool,
    SearchMemoryTool,
    SendStickerTool,
    SetReminderTool,
    SystemInfoTool,
    WebSearchTool,
    WriteFileTool,
    WriteMemoryTool,
    default_tools,
)
from axo.agent.reminders import ReminderStore
from axo.agent.tools import (
    Capture,
    Tool,
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    ToolRegistry,
    argument_int,
    argument_str,
)
from axo.llm.base import ToolCall, ToolSpec
from axo.memory.forget import FORGET_KEY
from axo.memory.writer import MemoryWriter
from axo.stickers import stickers_in
from memcore.ids import new_ulid
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Taint, Tier

NOW = 1_760_000_000.0

FIXED_NOW = datetime(2026, 10, 10, 15, 4)
"""改状态的工具用的固定时钟。

与 :data:`NOW` 分开：``NOW`` 是记忆条目的创建时间（喂给 ``MemoryItem`` 的
epoch 秒），这里是"现在几点"（喂给 ``ToolContext.clock`` 的 ``datetime``）。
两者混用会让"回程票上记的是什么时间"读起来对不上。
"""


class _NeverPolicy:
    """一个什么都不记的策略（**记忆写入闸门**）。

    用它来钉"工具写入不过那道闸门"：这条路径上的裁决者是**用户点的那一次头**，
    不是"这句话值不值得记"的规则表。一个把工具写入也送进策略的实现会在这里红
    ——而它在生产里的表现是"用户明确说了记住，助手点头，库里什么都没有"。
    """

    def drafts(self, turn: object) -> list[object]:
        return []


# ── 夹具与替身 ─────────────────────────────────────────────────────────────


def make_item(text: str, **overrides: object) -> MemoryItem:
    fields: dict[str, object] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "weight": 1.0,
        "created_at": NOW - 10 * 86400.0,
        "last_decay_ts": NOW - 10 * 86400.0,
        "last_access_ts": NOW - 10 * 86400.0,
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "mem.db") as opened:
        yield opened


@pytest.fixture
def ctx(store: MemoryStore) -> ToolContext:
    return ToolContext(store=store, session_id="S1", turn_seq=3)


@pytest.fixture
def writer(store: MemoryStore) -> MemoryWriter:
    """真写手 + :class:`_NeverPolicy`：工具写入**必须**不经过策略。"""
    return MemoryWriter(store, policy=_NeverPolicy())


@pytest.fixture
def reminders(tmp_path: Path) -> ReminderStore:
    return ReminderStore(tmp_path / "reminders.jsonl")


@pytest.fixture
def write_ctx(store: MemoryStore, writer: MemoryWriter, reminders: ReminderStore) -> ToolContext:
    """改状态的那三个工具用的上下文：时钟固定、有写手、有提醒簿。"""
    return ToolContext(
        store=store,
        writer=writer,
        reminders=reminders,
        session_id="S1",
        turn_seq=3,
        clock=lambda: FIXED_NOW,
    )


class _Spy:
    """一个记录自己被怎么调用的工具。

    ``rays`` 里的东西会被 ``raise`` 出来——用它来钉"派发口把工具内部异常吃成
    ``ok=False``"这条，而那条断言必须**看到**那个异常真的被抛过（否则一个从不
    执行的实现也能通过）。
    """

    def __init__(
        self,
        *,
        name: str = "spy",
        requires_approval: bool = False,
        raiser: BaseException | None = None,
        outcome: ToolOutcome | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description="测试用替身",
            parameters=parameters or {"type": "object", "properties": {}},
        )
        self.requires_approval = requires_approval
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[ToolContext] = []
        self._raiser = raiser
        self._outcome = outcome

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        self.calls.append(arguments)
        self.contexts.append(ctx)
        if self._raiser is not None:
            raise self._raiser
        return self._outcome or ToolOutcome.success("好")


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=arguments)


# ── 注册表 ─────────────────────────────────────────────────────────────────


def test_eight_tools_are_registered_without_a_read_root() -> None:
    """没配可读目录时 ``read_file`` **不登记**，另外八个都在。

    这不只是安全考虑：工具声明每轮都要重发，把一个注定失败的工具有效地告诉模型
    "你可以读文件"，只会换来它去试一次然后被拒。

    另外八个**不受配置影响**——``write_memory``/``set_reminder``/``list_reminders``/
    ``cancel_reminder`` 要的 ``ctx.writer``/``ctx.reminders``/``ctx.fired``/
    ``ctx.cancellations`` 是运行时装配的。它们"有时在有时不在"会让
    模型的失败经验互相矛盾（同一个名字，上次调得到、这次说不存在），所以那种情况
    由 ``run`` 里的一句话处理，而不是在这里把工具摘掉。
    """
    assert [tool.spec.name for tool in default_tools()] == [
        "get_time",
        "search_memory",
        "read_memory",
        "write_memory",
        "forget_memory",
        "set_reminder",
        "list_reminders",
        "cancel_reminder",
        "system_info",
    ]


def test_read_file_appears_once_a_root_is_configured(tmp_path: Path) -> None:
    names = [tool.spec.name for tool in default_tools(read_root=(tmp_path,))]
    assert names == [
        "get_time",
        "search_memory",
        "read_memory",
        "write_memory",
        "forget_memory",
        "set_reminder",
        "list_reminders",
        "cancel_reminder",
        "read_file",
        "list_files",
        "search_files",
        "system_info",
    ]


def test_look_at_screen_is_registered_only_when_the_model_can_see(tmp_path: Path) -> None:
    """``look_at_screen`` **按模型能力登记**（``default_tools(vision=...)``）。

    与 ``read_file`` 那条是同一条账：模型看不见图时把它登记上去，除了白烧一段 prompt，
    还会**弹一张注定白点的确认卡片**——用户点完头才发现什么都不会发生。所以判据放在
    登记这一处，而不是让工具自己回一句"我看不见"。

    顺序也是钉子：它**追加在最后**（提示缓存靠稳定前缀，插队会让所有请求的前缀失效）。
    """
    without = [tool.spec.name for tool in default_tools()]
    with_vision = [tool.spec.name for tool in default_tools(vision=True)]

    assert "look_at_screen" not in without
    assert with_vision == [*without, "look_at_screen"]


def test_exactly_five_tools_ask_for_confirmation_when_the_model_can_see(tmp_path: Path) -> None:
    """看得见图的模型多一个要点头的工具——**而它不改任何状态**。

    它进来的判据与那四个不同：不是"它改了状态"，而是**整块屏幕的内容会交给模型看一次**
    （一次不可撤回的信息披露，而且屏幕没有路径可判，文件沙箱那条路用不上）。这条断言
    把"多一个要点头的"写死，免得下一个人照着"只拦改状态的"那条规则把它摘掉。
    """
    asking = [
        tool.spec.name
        for tool in default_tools(read_root=(Path(),), vision=True)
        if tool.requires_approval
    ]

    assert asking == [
        "write_memory",
        "forget_memory",
        "set_reminder",
        "cancel_reminder",
        "look_at_screen",
    ]


def test_look_at_screen_hands_the_image_back_with_the_result() -> None:
    """截到的图**随结果回灌**，而且是 base64（后端要的形状）。

    ``content`` 里必须带上尺寸：模型据此知道它看到的不是原始分辨率，而"我看到的是一张
    缩过的图"这件事**它自己看不出来**。``detail`` 是给卡片页脚的，不进模型上下文。
    """
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        capture=lambda: Capture(png=b"PNG", width=1600, height=900),
    )

    outcome = LookAtScreenTool().run({}, ctx)

    assert outcome.ok is True
    assert outcome.images == (base64.b64encode(b"PNG").decode("ascii"),)
    assert "1600×900" in outcome.content
    assert "1600×900" in outcome.detail
    assert outcome.replayable is False, "带图的工具必须标不可回放（图回放不了）"


def test_look_at_screen_says_so_when_there_is_no_capture_channel() -> None:
    """没有截图通道时回一句人话，**不是抛**。

    与只读工具在没有库时那句话同形（``ctx.writer is None`` 那一族）：装配期缺东西是
    正常状态，而模型需要的是一句能转述给用户的话。
    """
    outcome = LookAtScreenTool().run({}, ToolContext(store=MemoryStore(":memory:")))

    assert outcome.ok is False
    assert "没有接上截图通道" in outcome.content
    assert outcome.images == ()


def test_look_at_screen_passes_the_reason_back_verbatim() -> None:
    """截不到的**原因**（锁屏、全黑…）来自界面那一侧，原样转述给模型。

    这一条要紧的是"不要猜屏幕上有什么"那句也跟着回去：一次没截到的调用最容易的坏结局
    是模型开始描述一块它没看见的屏幕。
    """
    ctx = ToolContext(
        store=MemoryStore(":memory:"),
        capture=lambda: Capture(reason="屏幕看起来是黑的（可能是锁屏或独占全屏）"),
    )

    outcome = LookAtScreenTool().run({}, ctx)

    assert outcome.ok is False
    assert "锁屏" in outcome.content
    assert "不要猜" in outcome.content
    assert outcome.replayable is False, "失败也是那一刻的事实"


def test_exactly_four_tools_ask_for_confirmation() -> None:
    """闸门只拦**改变状态**的那四个。

    这条断言的是那个分界本身：多一个"要确认"的工具会让用户白点一次头，少一个就是
    **一个改状态的动作没问过任何人**。后者的症状是没有症状——所以它值得一条测试，
    而这条测试要同时钉住两边（「恰好只有这四个」）。

    ``cancel_reminder`` 进来的理由比「它改状态」具体：**它取消的是一个未来的
    动作**，而误取消的后果用户要等到错过才发现——那正是最该有一个人点头的一类。
    """
    asking = [
        tool.spec.name for tool in default_tools(read_root=(Path(),)) if tool.requires_approval
    ]
    assert asking == ["write_memory", "forget_memory", "set_reminder", "cancel_reminder"]


def test_the_two_snapshot_tools_mark_their_results_as_not_replayable(
    store: MemoryStore, tmp_path: Path
) -> None:
    """``get_time`` / ``read_file`` 的结果是**某一刻的快照**，要标 ``replayable=False``。

    忘标**不会有任何症状**：``ToolOutcome.replayable`` 默认 ``True``，于是那一对被 L5
    逐轮搬进上下文，模型拿着过期的时间戳和旧的文件正文回答"现在几点""文件里写了
    什么"，而它没有任何线索知道那是旧的。默认值只能选一个方向，这里选的是"宁可多
    回放"（理由见 ``Message.no_replay`` 的 docstring：老逐字稿里的工具对本来就都是
    普通结果，默认反过来会把用户已有的工具历史一次抹掉）。

    所以代价是**每加一个快照型工具都要记得标**。这条测试同时钉两个方向：两个已知的
    快照工具必须为 ``False``，而一个"世界没怎么变"的工具必须仍然为 ``True``——只钉
    前一半的话，"把所有工具都标成不回放"也能过，而那等于废掉 L5。

    覆盖边界说清楚：这条盯得住这两个**已经标对**的工具（防止后人把标记删掉），
    盯不住"新加的工具忘了标"——那件事在运行期没有判据（谁也算不出一个工具的结果
    耐不耐放）。它是回归守卫，不是完整性检查。
    """
    (tmp_path / "备忘.txt").write_text("厨房的灯该换了", encoding="utf-8")
    ctx = ToolContext(
        store=store,
        read_root=(tmp_path,),
        clock=lambda: datetime(2026, 10, 10, 14, 32),
    )

    assert GetTimeTool().run({}, ctx).replayable is False
    assert ReadFileTool().run({"path": "备忘.txt"}, ctx).replayable is False
    # 屏幕这一条**更硬**：图本身回放不了（逐字稿只存正文、L5 重建行时不带图），所以
    # 不标 False 的症状不是"旧数据"，而是"模型收到一条自称看过屏幕却没有图的记录"。
    shot_ctx = ToolContext(
        store=store,
        capture=lambda: Capture(png=b"PNG", width=800, height=600),
    )
    assert LookAtScreenTool().run({}, shot_ctx).replayable is False
    # 反空转：检索的结果**照旧**可以回放——查过的记忆下一轮还查得到，不必重查。
    assert SearchMemoryTool().run({"query": "灯"}, ctx).replayable is True


def test_registry_never_silently_drops() -> None:
    """守卫的守卫：声明条数必须等于登记条数。

    一个 `spec` 为空、或者名字为空而被悄悄漏掉的工具，在这里会立刻显形。没有这条
    断言时，"模型调不到某个工具"的症状只能靠人去数——而 `ToolSpec` 的名字非空校验
    只挡住了最浅的一种。
    """
    registry = ToolRegistry(default_tools())
    assert len(registry.specs()) == len(registry)
    assert registry.specs() and len(registry.names) == len(set(registry.names))


def test_registry_order_is_registration_order() -> None:
    """顺序稳定 → 工具声明在请求里的排列稳定 → 后端提示缓存能命中前缀。"""
    registry = ToolRegistry([_Spy(name="b"), _Spy(name="a")])
    assert registry.names == ("b", "a")


def test_duplicate_name_raises_at_assembly() -> None:
    """重名**在装配期就炸**。

    这是本模块唯一的抛出口。运行期抛不行——那时症状是"模型调了一个永远不会被
    执行的工具"，从这一头看不出根因。
    """
    with pytest.raises(ValueError, match="注册了两次"):
        ToolRegistry([_Spy(name="dup"), _Spy(name="dup")])


def test_tools_satisfy_the_protocol() -> None:
    """四个工具都满足 ``Tool`` 协议——协议是 ``runtime_checkable`` 的，
    所以这条能真的跑，而不是只在类型检查器里成立。"""
    for tool in default_tools(read_root=(Path(),)):
        assert isinstance(tool, Tool)


# ── 派发：四种失败都不许抛 ─────────────────────────────────────────────────


def test_dispatch_never_raises(ctx: ToolContext) -> None:
    """**四种失败各扫一遍，一个都不许逃出去。**

    :class:`ToolOutcome` 的存在理由就是这个：失败要回灌给模型让它自己纠正。
    如果某一种逃成了异常，那一轮就整段结束——用户看到"出错了"，模型连改一次的
    机会都没有。所以这条测试只问一件事：**有没有东西逃出去**。
    """
    bad_args = _Spy(name="bad_args", raiser=ToolArgumentError("limit 必须是整数"))
    broken = _Spy(name="broken", raiser=RuntimeError("内部炸了"))
    gated = _Spy(name="gated", requires_approval=True)
    needs_arg = _Spy(
        name="needs_arg",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "查什么"}},
            "required": ["query"],
        },
    )
    registry = ToolRegistry([bad_args, broken, gated, needs_arg])

    attempts = [
        (call("没这个工具"), True),
        (call("needs_arg"), True),  # 缺必需参数
        (call("bad_args"), True),
        (call("broken"), True),
        (call("gated"), False),  # 闸门拒绝
        (call("gated"), True),  # 同意之后仍然要能正常返回
    ]
    for tool_call, approved in attempts:
        outcome = registry.dispatch(tool_call, ctx, approved=approved)
        assert isinstance(outcome, ToolOutcome), f"{tool_call.name} 抛出来了"
        assert outcome.content, f"{tool_call.name} 的失败没有给模型任何可读的话"


def test_unknown_tool_lists_what_exists(ctx: ToolContext) -> None:
    """未知名要**列出可用的**——不然模型只能猜，而猜错的代价是又一整轮。"""
    registry = ToolRegistry(default_tools())
    outcome = registry.dispatch(call("read_files"), ctx)
    assert outcome.ok is False
    assert "search_memory" in outcome.content


def test_missing_required_argument_names_it(ctx: ToolContext) -> None:
    registry = ToolRegistry([SearchMemoryTool()])
    outcome = registry.dispatch(call("search_memory"), ctx)
    assert outcome.ok is False
    assert "query" in outcome.content
    assert "必需的参数" in outcome.content


def test_tool_argument_error_is_reported_as_a_fixable_mistake(ctx: ToolContext) -> None:
    """参数错与工具坏，回灌的语气必须不同。

    前者是"你写错了，这样改"，后者是"这个工具暂时用不了"。混成一句话会让模型
    对着一个坏掉的工具反复重试。
    """
    registry = ToolRegistry([SearchMemoryTool()])
    outcome = registry.dispatch(call("search_memory", query="笔记本", limit="五"), ctx)
    assert outcome.ok is False
    assert "参数不对" in outcome.content
    assert "limit" in outcome.content


def test_tool_exception_keeps_the_stack(ctx: ToolContext) -> None:
    """异常被吃掉，但**栈不能被吃掉**。

    一个只回灌一句"执行失败"的实现在这里会红：那条消息给模型够用了，但下一个
    来看这个问题的人拿不到任何线索，而工具里的异常恰恰是只能靠栈定位的那一类。
    """
    registry = ToolRegistry([_Spy(name="broken", raiser=RuntimeError("内部炸了"))])
    outcome = registry.dispatch(call("broken"), ctx)
    assert outcome.ok is False
    assert "RuntimeError" in outcome.content
    assert "Traceback" in outcome.detail


def test_tool_result_passes_through(ctx: ToolContext) -> None:
    registry = ToolRegistry([_Spy(outcome=ToolOutcome.success("查到了三条"))])
    outcome = registry.dispatch(call("spy"), ctx)
    assert outcome.ok is True
    assert outcome.content == "查到了三条"


# ── 闸门（只到"声明与读取"这一步；握手在 gate.py） ─────────────────────────


def test_needs_approval_reads_the_tool_declaration() -> None:
    registry = ToolRegistry([_Spy(name="a"), _Spy(name="b", requires_approval=True)])
    assert registry.needs_approval("b") is True
    assert registry.needs_approval("a") is False
    assert registry.needs_approval("没这个工具") is False


def test_gate_denial_skips_the_tool(ctx: ToolContext) -> None:
    spy = _Spy(name="gated", requires_approval=True)
    outcome = ToolRegistry([spy]).dispatch(call("gated"), ctx, approved=False)
    assert outcome.ok is False
    assert spy.calls == [], "被拒绝的工具还是被执行了"
    # 回灌必须禁止模型重试或假装做过——Step 0 探针里最主要的失败形态正是"声称做了"。
    assert "不要重试" in outcome.content


def test_gate_denial_does_not_touch_approval_free_tools(ctx: ToolContext) -> None:
    """反空转：``approved=False`` 不是"什么都不执行"。

    闸门只管声明过 ``requires_approval`` 的工具。一个把 ``approved`` 当成全局开关
    的实现（"没同意就都不做"）会让**只读工具也停摆**——而界面上的表现是"助手突然
    什么都查不了"，看不出跟确认卡片有关。
    """
    spy = _Spy(name="free")
    outcome = ToolRegistry([spy]).dispatch(call("free"), ctx, approved=False)
    assert outcome.ok is True
    assert spy.calls == [{}], "不需要确认的工具被 approved=False 挡住了"


def test_approved_gated_tool_runs(ctx: ToolContext) -> None:
    spy = _Spy(name="gated", requires_approval=True)
    outcome = ToolRegistry([spy]).dispatch(call("gated", what="x"), ctx, approved=True)
    assert outcome.ok is True
    assert spy.calls == [{"what": "x"}]


# ── 参数取值：不猜 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["5", 5.0, True, [5], {"n": 5}])
def test_argument_int_refuses_to_guess(bad: object) -> None:
    """``"5"`` 与 ``5`` 的区别不许在这里消失。

    顺手 ``int(value)`` 一下能过，但那天 ``"五"`` 也会被吞掉（或者更糟，被当成 0）。
    ``True`` 尤其：它是 ``int`` 的子类，但那是模型理解错了这个参数，该让它重试。
    """
    with pytest.raises(ToolArgumentError):
        argument_int({"n": bad}, "n")


@pytest.mark.parametrize(("value", "lo", "hi"), [(0, 1, 8), (9, 1, 8), (-1, 1, 8)])
def test_argument_int_bounds_are_enforced(value: int, lo: int, hi: int) -> None:
    with pytest.raises(ToolArgumentError):
        argument_int({"n": value}, "n", minimum=lo, maximum=hi)


def test_argument_int_default_and_absent() -> None:
    assert argument_int({}, "n", default=5) == 5
    assert argument_int({"n": None}, "n", default=5) == 5
    assert argument_int({"n": 3}, "n", default=5) == 3


@pytest.mark.parametrize("bad", [5, ["x"], {"a": 1}, True])
def test_argument_str_refuses_non_strings(bad: object) -> None:
    with pytest.raises(ToolArgumentError):
        argument_str({"s": bad}, "s")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_argument_str_rejects_blank(blank: str) -> None:
    with pytest.raises(ToolArgumentError):
        argument_str({"s": blank}, "s")


def test_argument_str_strips() -> None:
    assert argument_str({"s": "  查一下  "}, "s") == "查一下"


# ── 四个工具 ───────────────────────────────────────────────────────────────


def test_get_time_uses_the_injected_clock(store: MemoryStore) -> None:
    """时间来自 ``ctx.clock``，不是 ``datetime.now()``。

    可注入是这条断言的前提：一个直接调 ``datetime.now()`` 的实现**也能跑**，
    只是它永远测不了——而"日期对不对"这件事在界面上看起来总是对的。
    """
    ctx = ToolContext(store=store, clock=lambda: datetime(2026, 10, 10, 15, 4))
    outcome = ToolRegistry([GetTimeTool()]).dispatch(call("get_time"), ctx)
    assert outcome.ok is True
    assert "2026-10-10" in outcome.content
    assert "15:04" in outcome.content
    assert "星期" in outcome.content


def test_search_memory_finds_and_returns_ids(store: MemoryStore, ctx: ToolContext) -> None:
    item = store.add(make_item("我家的路由器放在书房的书架顶层。"))
    store.add(make_item("周三那家面馆休息。"))

    outcome = ToolRegistry([SearchMemoryTool()]).dispatch(
        call("search_memory", query="路由器放在哪儿"), ctx
    )
    assert outcome.ok is True
    assert item.memory_id in outcome.content, "结果里没有 id，模型就没法用 read_memory 接着看"


def test_search_memory_says_so_when_nothing_matches(store: MemoryStore, ctx: ToolContext) -> None:
    """空结果必须**明说**。

    让模型从"没有输出"里推断"没有找到"，等于给它留出编一条的空间——Step 0 探针里
    最主要的失败形态正是编。
    """
    outcome = ToolRegistry([SearchMemoryTool()]).dispatch(
        call("search_memory", query="量子色动力学里的渐近自由"), ctx
    )
    assert outcome.ok is True
    assert "没有找到" in outcome.content


def test_search_memory_does_not_touch_the_store(store: MemoryStore, ctx: ToolContext) -> None:
    """模型的一次试探不该改写记忆权重。

    与记忆界面里「以这条为线索召回一次」是同一类动作，所以走同一个 ``settle=False``
    ——见 ``tests/test_recall_settle.py`` 的说明：不关掉的话，被看到的条目会显得更
    常用、衰减得更慢，而**真实**召回的排序从此带着这次试探的痕迹。

    **第一版这条测试是空转的，而且绿。** 它用 ``query="路由器"``，那条短查询在词法
    通道上得分低于 ``min_score``，什么都没召回到——于是 ``settle`` 没有任何东西可以
    结算，把代码改成 ``settle=True`` 它照样绿。查出这件事靠的是**故意注入一个错**
    再看测试红不红；而它给这个文件留下的纪律是下面那句断言：

    **一条关于副作用的断言，必须先断言那个副作用有机会发生。**
    """
    item = store.add(make_item("我家的路由器放在书房的书架顶层。"))
    before = store.get(item.memory_id)
    assert before is not None

    outcome = ToolRegistry([SearchMemoryTool()]).dispatch(
        call("search_memory", query="路由器放在哪儿"), ctx
    )

    # 先证"这次召回真的命中了"。少了这一句，上面那句 settle 的断言可以在什么都没
    # 发生的情况下通过——而"什么都没有发生"与"正确地什么都没改"长得一模一样。
    assert item.memory_id in outcome.content, "没召回命中，下面的断言是空转的"

    after = store.get(item.memory_id)
    assert after is not None
    assert after.access_count == before.access_count == 0
    assert after.weight == before.weight
    assert after.last_access_ts == before.last_access_ts


def test_read_memory_returns_the_body(store: MemoryStore, ctx: ToolContext) -> None:
    item = store.add(make_item("备用钥匙在门口鞋柜最上面那格。"))
    outcome = ToolRegistry([ReadMemoryTool()]).dispatch(
        call("read_memory", memory_id=item.memory_id), ctx
    )
    assert outcome.ok is True
    assert "鞋柜最上面那格" in outcome.content


def test_read_memory_hints_at_the_real_mistake(store: MemoryStore, ctx: ToolContext) -> None:
    """id 传错时提示**最常见的那个错因**：把正文里的词当成了 id。

    只说"没有这条"不足以让模型改对——它多半是照着摘要正文抄了一个词。
    """
    outcome = ToolRegistry([ReadMemoryTool()]).dispatch(
        call("read_memory", memory_id="备用钥匙"), ctx
    )
    assert outcome.ok is False
    assert "方括号" in outcome.content


def test_read_file_is_disabled_without_a_root(store: MemoryStore) -> None:
    """硬调一个没登记的 ``read_file`` 也要拿到一句解释，而不是异常。"""
    ctx = ToolContext(store=store)
    outcome = ReadFileTool().run({"path": "备忘.txt"}, ctx)
    assert outcome.ok is False
    assert "没有配置可读目录" in outcome.content


def test_read_file_reads_a_configured_root(store: MemoryStore, tmp_path: Path) -> None:
    (tmp_path / "备忘.txt").write_text("厨房的灯该换了", encoding="utf-8")
    ctx = ToolContext(store=store, read_root=(tmp_path,))
    outcome = ReadFileTool().run({"path": "备忘.txt"}, ctx)
    assert outcome.ok is True
    assert "厨房的灯该换了" in outcome.content


def test_read_file_reports_escape_as_a_failure_not_an_exception(
    store: MemoryStore, tmp_path: Path
) -> None:
    """逃逸是**失败**，不是崩溃。

    ``ReadFileError`` 的消息本来就是写给模型看的，所以它要能原样回灌——模型据此
    知道"该换成允许目录里的路径"，而不是看到一句"出错了"。
    """
    area = tmp_path / "允许"
    area.mkdir()
    (tmp_path / "外面.txt").write_text("x", encoding="utf-8")
    ctx = ToolContext(store=store, read_root=(area,))

    outcome = ReadFileTool().run({"path": "../外面.txt"}, ctx)
    assert outcome.ok is False
    assert "不在允许读取的目录里" in outcome.content


# ── 派发口绑定 call_id ─────────────────────────────────────────────────────


def test_dispatch_binds_call_id(ctx: ToolContext) -> None:
    """``call_id`` 由派发口填，不从参数里来。

    它是 ``tool://<会话>/<轮次>/<call_id>`` 的最后一截，而同一轮里可以调两次
    ``write_memory``——少了它，两条溯源指针会一模一样，回看时认不出哪条是哪次写的。
    """
    spy = _Spy()
    ToolRegistry([spy]).dispatch(ToolCall(id="call_7", name="spy", arguments={}), ctx)
    assert spy.contexts[0].call_id == "call_7"


# ── write_memory ───────────────────────────────────────────────────────────


def test_write_memory_stores_what_the_model_asked_for(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """写进去的东西每一栏都要对得上：内容、taint、kind、meta、溯源指针。

    这不是"字段抄一遍"式的断言。``taint=MODEL`` 与 ``meta["via"]`` 是**唯一**能
    在几个月后回答"这条记忆是谁写的、经过了什么"的东西；漏了它们，库里的每一条
    看起来都像是用户自己说的。
    """
    outcome = ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="备用钥匙在门口鞋柜最上面那格。"), write_ctx
    )
    assert outcome.ok is True

    items = store.load_all()
    assert len(items) == 1
    item = items[0]
    assert item.content == "备用钥匙在门口鞋柜最上面那格。"
    assert item.taint is Taint.MODEL, "工具写入必须标成模型写的"
    assert item.kind is MemoryKind.FACT
    assert item.meta["via"] == "write_memory"
    assert item.meta["call_id"] == "call_1"
    assert item.source_ref == "tool://S1/3/call_1"
    assert item.memory_id in outcome.content, "回灌里没有 id，模型就没法接着 read/forget"


def test_write_memory_does_not_go_through_the_memory_policy(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """**两个闸门不是一回事。**

    记忆写入闸门（``PerItemPolicy.drafts``）判"这句话值不值得记"，是每轮自动跑的
    规则；工具确认闸门判"这个动作要不要你点头"，是用户点的那一下。``write_ctx``
    里的策略什么都不记，而这次写入仍然必须落库——因为裁决者是人，不是规则表。

    一个把工具写入也送进策略的实现会在这里红，而它在生产里的表现是"用户明确说了
    记住、助手点头答应了、库里什么都没有"。
    """
    outcome = ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="周三下午三点和张工在二楼会议室对方案。"), write_ctx
    )
    assert outcome.ok is True, "工具写入被记忆写入闸门拦下了"
    assert len(store.load_all()) == 1


def test_gate_denial_of_write_memory_leaves_the_store_empty(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """用户点「拒绝」之后，库里**一条都不许多**。

    这条是闸门存在的全部意义的直接断言。``dispatch`` 的拒绝分支在通用测试里已经
    扫过一遍，但那一遍用的替身工具、没有副作用——"被拒绝的调用什么都没做"这件事，
    只有在有一个真会写库的工具时才算真的验过。
    """
    outcome = ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="这条不该进库。"), write_ctx, approved=False
    )
    assert outcome.ok is False
    assert store.load_all() == []
    assert "不要重试" in outcome.content


def test_write_memory_says_so_when_the_content_cannot_be_encoded(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """编不出词法指纹时**必须说实话**。

    ``record_direct`` 用 ``None`` 表示"没写进去"，而它的另一个调用方
    （``MemoryWriter.record``）拿 ``None`` 当"跳过这一条"用。工具这边照抄那个态度
    的话，它会回一句"已记住"而库里空空——**这个项目里最坏的一种失效**：用户以为
    记住了，直到某天问起来答不出。
    """
    outcome = ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="？？？"), write_ctx
    )
    assert outcome.ok is False
    assert "没有" in outcome.content
    assert store.load_all() == []


def test_write_memory_without_a_writer_admits_it(store: MemoryStore) -> None:
    """装配期没给 writer（只跑只读工具的运行）：回一句实话，不许假装成功。"""
    outcome = WriteMemoryTool().run({"content": "随便一条。"}, ToolContext(store=store))
    assert outcome.ok is False
    assert "没有" in outcome.content


def test_write_memory_refuses_an_overlong_body(store: MemoryStore, write_ctx: ToolContext) -> None:
    """太长就拒绝，**不截断**。

    截断会把半句话长期存进库，而它以后被当成完整的一句读出来——那个错误没有任何
    症状。拒绝则会让模型分几条重写，而它做得到。
    """
    outcome = ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="很" * (MAX_WRITE_CHARS + 1)), write_ctx
    )
    assert outcome.ok is False
    assert "最多" in outcome.content
    assert store.load_all() == []


# ── forget_memory ──────────────────────────────────────────────────────────


def test_forget_memory_soft_forgets_and_leaves_a_return_ticket(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """软遗忘：层级变 ``COLD``、回程票写进 ``meta``、**条目还在库里**。

    三个断言缺一条都会漏掉一种失效：少了 tier 那条，"忘了"什么都没忘；少了回程票
    那条，用户再也恢复不回来（而界面上写着"可以恢复"）；少了"还在库里"那条，这就
    变成了一次没有二次确认的删除。
    """
    item = store.add(make_item("那台旧笔记本的硬盘密码是 4002。"))
    outcome = ToolRegistry([ForgetMemoryTool()]).dispatch(
        call("forget_memory", memory_id=item.memory_id), write_ctx
    )
    assert outcome.ok is True

    after = store.get(item.memory_id)
    assert after is not None, "软遗忘把条目删掉了——它是可逆的，不许真删"
    assert after.tier is Tier.COLD
    assert after.meta[FORGET_KEY]["previous_tier"] == Tier.LONGTERM.value
    assert after.meta[FORGET_KEY]["at"] == pytest.approx(FIXED_NOW.timestamp())


def test_forget_memory_refuses_a_pinned_item_and_points_at_the_user(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    """钉住的条目忘不掉——而且话要**替模型说完整**。

    ``can_soft_forget`` 的消息是写给界面的（末尾叫人去按「彻底删除」那个按钮）。
    模型没有那个按钮，照着原样回灌的话它会以为自己参数写错了，然后反复重试一个
    结构上不可能成功的动作。
    """
    item = store.add(make_item("叫我小离就好。", pinned=True, kind=MemoryKind.PERSONA_ANCHOR))
    outcome = ToolRegistry([ForgetMemoryTool()]).dispatch(
        call("forget_memory", memory_id=item.memory_id), write_ctx
    )
    assert outcome.ok is False
    assert "你做不到" in outcome.content
    kept = store.get(item.memory_id)
    assert kept is not None and kept.tier is Tier.LONGTERM


def test_forget_memory_on_an_already_forgotten_item_says_so(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    item = store.add(make_item("已经忘过的一条。", tier=Tier.COLD))
    outcome = ToolRegistry([ForgetMemoryTool()]).dispatch(
        call("forget_memory", memory_id=item.memory_id), write_ctx
    )
    assert outcome.ok is False
    assert "已经" in outcome.content


def test_forget_memory_hints_at_the_real_mistake(
    store: MemoryStore, write_ctx: ToolContext
) -> None:
    outcome = ToolRegistry([ForgetMemoryTool()]).dispatch(
        call("forget_memory", memory_id="那台旧笔记本"), write_ctx
    )
    assert outcome.ok is False
    assert "方括号" in outcome.content


# ── set_reminder ───────────────────────────────────────────────────────────


def test_set_reminder_from_a_delay(write_ctx: ToolContext, reminders: ReminderStore) -> None:
    """相对时间：``delay_minutes`` 换算成绝对时刻，登记时刻来自注入的时钟。"""
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="半小时后把汤关火。", delay_minutes=30), write_ctx
    )
    assert outcome.ok is True

    stored = reminders.all()
    assert len(stored) == 1
    assert stored[0].text == "半小时后把汤关火。"
    assert stored[0].due_at == pytest.approx(FIXED_NOW.timestamp() + 30 * 60)
    assert stored[0].created_at == pytest.approx(FIXED_NOW.timestamp())
    assert (stored[0].session_id, stored[0].turn_seq, stored[0].call_id) == ("S1", 3, "call_1")


def test_set_reminder_from_an_absolute_time(
    write_ctx: ToolContext, reminders: ReminderStore
) -> None:
    """绝对时间：模型自己算好日期与时分给过来（先 ``get_time`` 是描述里要求它的）。"""
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="明天早上八点交房租。", due_at="2026-10-11 08:00"), write_ctx
    )
    assert outcome.ok is True
    assert reminders.all()[0].due_at == pytest.approx(datetime(2026, 10, 11, 8, 0).timestamp())


def test_set_reminder_promises_only_what_it_can_do(write_ctx: ToolContext) -> None:
    """回灌里必须说「到点会提醒他」——**因为那件事现在真的会发生**。

    这条测试原先断言的是反面：回灌里必须带「不会自动弹出」。那是 P3a 留下的，
    而那时确实没有东西在读提醒文件。**P3b 把触发实现之后，没有人回来改它**，于是
    模型一直照着一句过期的话对用户说「到点它不会自己弹出来」——用户看到的就是那样
    一句假话（这正是这一版被提起来的原因）。

    现在断言正面事实，**并且同时钉住反面**：那族措辞不许回来。它拦的还是同一件事
    ——别说你做不到的，也别说已经不成立的。
    """
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="明天早上八点交房租。", due_at="2026-10-11 08:00"), write_ctx
    )
    assert "到点会提醒他" in outcome.content
    assert "不会自动弹出" not in outcome.content


def test_set_reminder_refuses_a_past_time(write_ctx: ToolContext, reminders: ReminderStore) -> None:
    """算错了日期就给一次可纠正的拒绝，并把"现在几点"一起回灌。

    收下它的后果是：那条提醒生下来就过期、永远不会被触发，而用户以为约好了。
    """
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="昨天的事。", due_at="2026-10-09 08:00"), write_ctx
    )
    assert outcome.ok is False
    assert "已经过去" in outcome.content
    assert "2026-10-10 15:04" in outcome.content, "没告诉它现在几点，它下次还是算不对"
    assert reminders.all() == []


def test_set_reminder_refuses_a_date_without_a_time(
    write_ctx: ToolContext, reminders: ReminderStore
) -> None:
    """只给日期**不**当成那天的零点。

    那是我们替用户猜的一个时间。提醒是这个项目里唯一"到点"有后果的东西，
    不给它猜——``fromisoformat`` 会静默地这么干，所以这道检查必须在它之前。
    """
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="交房租。", due_at="2026-10-11"), write_ctx
    )
    assert outcome.ok is False
    assert "没有时分" in outcome.content
    assert reminders.all() == []


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"text": "两样都给。", "due_at": "2026-10-11 08:00", "delay_minutes": 30}, "只能给一个"),
        ({"text": "一样都不给。"}, "要给一个时间"),
        ({"text": "约到几十年后。", "delay_minutes": 1_000_000}, "不能大于"),
        ({"text": "零分钟。", "delay_minutes": 0}, "不能小于"),
        ({"text": "字符串的分钟。", "delay_minutes": "30"}, "必须是整数"),
    ],
)
def test_set_reminder_refuses_ambiguous_or_bad_times(
    arguments: dict[str, Any], expected: str, write_ctx: ToolContext, reminders: ReminderStore
) -> None:
    """五种给不对时间的方式，每一种都要拿到一句能照着改的话。

    **这份表不包含量纲错误**（用户说"一小时后"而模型把 3600 秒当成分），因为那一种
    在上限之内、拦不住——``catalog.MAX_DELAY_MINUTES`` 的文档里写明了这一点。
    表里第三行只保证"不会有提醒被静默地约到几十年后"。
    """
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", **arguments), write_ctx
    )
    assert outcome.ok is False
    assert expected in outcome.content
    assert reminders.all() == []


def test_set_reminder_without_a_store_admits_it(store: MemoryStore) -> None:
    outcome = SetReminderTool().run(
        {"text": "随便。", "delay_minutes": 5}, ToolContext(store=store)
    )
    assert outcome.ok is False
    assert "没有" in outcome.content


def test_set_reminder_refuses_an_overlong_text(write_ctx: ToolContext) -> None:
    outcome = ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="事" * (MAX_REMINDER_CHARS + 1), delay_minutes=5), write_ctx
    )
    assert outcome.ok is False
    assert "最多" in outcome.content


def test_write_memory_and_set_reminder_are_separate_files(
    write_ctx: ToolContext, reminders: ReminderStore, store: MemoryStore
) -> None:
    """一条提醒**不进记忆库**，一条记忆**不进提醒簿**。

    两者的消费者完全不同（召回 vs P3b 的托盘），混起来的第一个后果是召回开始返回
    一些"未来的句子"——它们在语义上就是用户提过的一句话，权重、衰减、SDR 全算得
    出来，于是会安安静静地出现在注入账本里。
    """
    ToolRegistry([SetReminderTool()]).dispatch(
        call("set_reminder", text="明天买牛奶。", delay_minutes=600), write_ctx
    )
    ToolRegistry([WriteMemoryTool()]).dispatch(
        call("write_memory", content="牛奶喝完了。"), write_ctx
    )

    assert [r.text for r in reminders.all()] == ["明天买牛奶。"]
    assert [item.content for item in store.load_all()] == ["牛奶喝完了。"]


# ── 表情包（send_sticker）────────────────────────────────────────────────────


def _sticker_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "stickers"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("开心", "无语", "Sad"):
        # 内容不重要——扫描只认后缀。这里写 PNG 的魔数开头，免得哪天多一条
        # "它是不是真图片"的判据时，这些夹具突然变成"坏文件"。
        (folder / f"{name}.png").write_bytes(bytes([137, 80, 78, 71]))
    return folder


def test_the_sticker_tool_is_registered_only_when_there_are_stickers(tmp_path: Path) -> None:
    """目录空着就**不登记**：她不可能成功，而做不到的事不该出现在工具表里
    （同 ``read_file`` 在没有可读根、``look_at_screen`` 在模型看不见图时那条账）。"""
    assert "send_sticker" not in [tool.spec.name for tool in default_tools()]

    names = [tool.spec.name for tool in default_tools(stickers=stickers_in(_sticker_dir(tmp_path)))]

    assert "send_sticker" in names


def test_sending_a_sticker_hands_the_path_to_the_user_channel(
    tmp_path: Path, ctx: ToolContext
) -> None:
    """成功时 **preview** 是那张图的路径——那是给用户看的通道。

    ``images`` 仍然是空的：模型不需要"看见"自己发的那张（那还会白烧一笔图 token）。
    两条通道混起来的后果是表情包被回灌进上下文，而模型开始对着自己的梗图说话。
    """
    tool = SendStickerTool(stickers_in(_sticker_dir(tmp_path)))

    outcome = tool.run({"name": "开心"}, ctx)

    assert outcome.ok
    assert outcome.preview.endswith("开心.png")
    assert outcome.images == ()
    assert "开心" in outcome.content


def test_a_wrong_name_gets_the_whole_list_back(tmp_path: Path, ctx: ToolContext) -> None:
    """失败时**把可用的名单再说一遍**：模型多半是名字记错了，而它手上那份清单
    （在工具描述里）已经过去好几轮了。光说"没有"会让它再猜一次。"""
    tool = SendStickerTool(stickers_in(_sticker_dir(tmp_path)))

    outcome = tool.run({"name": "生气"}, ctx)

    assert not outcome.ok
    assert "开心" in outcome.content and "无语" in outcome.content


def test_a_wrong_case_still_finds_the_sticker(tmp_path: Path, ctx: ToolContext) -> None:
    tool = SendStickerTool(stickers_in(_sticker_dir(tmp_path)))

    outcome = tool.run({"name": "sad"}, ctx)

    assert outcome.ok and outcome.preview.endswith("Sad.png")


def test_a_sticker_without_a_name_is_a_failure_not_a_crash(
    tmp_path: Path, ctx: ToolContext
) -> None:
    """参数缺失/类型不对是**正常失败**（回一句能指导下一步的话），不是异常。"""
    tool = SendStickerTool(stickers_in(_sticker_dir(tmp_path)))

    assert not tool.run({}, ctx).ok
    assert not tool.run({"name": 42}, ctx).ok


def test_the_sticker_tool_is_replayable(tmp_path: Path, ctx: ToolContext) -> None:
    """与 ``get_time`` 那种"某一刻的快照"不同：**"我发过一张开心的表情包"在世界里
    没有变过**，回放它是对的（不回放的话模型会以为自己没发过，然后重复发）。"""
    tool = SendStickerTool(stickers_in(_sticker_dir(tmp_path)))

    assert tool.run({"name": "开心"}, ctx).replayable is True


# ── 只读工具面：目录、搜索、系统信息、联网（A1） ────────────────────────────


class FakeNet:
    available = True

    def __init__(self) -> None:
        self.searched: list[tuple[str, int]] = []
        self.fetched: list[str] = []

    def search(self, query: str, *, count: int = 8) -> list[Any]:
        from axo.net import WebHit

        self.searched.append((query, count))
        return [WebHit(title="结果一", url="https://example.com/1", summary="摘要")]

    def fetch(self, url: str) -> tuple[str, str]:
        self.fetched.append(url)
        return "标题", "正文"


def test_list_files_reports_the_total(tmp_path: Path) -> None:
    (tmp_path / "子目录").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    args = {"path": ".", "limit": 1}
    context = ToolContext(store=MemoryStore(":memory:"), read_root=(tmp_path,))

    outcome = ListFilesTool().run(args, context)

    assert outcome.ok
    assert "共 2 项" in outcome.content
    assert "还有 1 项" in outcome.content


def test_search_files_returns_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "笔记.txt").write_text("第一行\nalpha\n", encoding="utf-8")
    context = ToolContext(store=MemoryStore(":memory:"), read_root=(tmp_path,))

    outcome = SearchFilesTool().run({"pattern": "alpha"}, context)

    assert outcome.ok
    assert "笔记.txt:2" in outcome.content


def test_search_files_says_so_when_nothing_matches(tmp_path: Path) -> None:
    context = ToolContext(store=MemoryStore(":memory:"), read_root=(tmp_path,))

    outcome = SearchFilesTool().run({"pattern": "不存在"}, context)

    assert outcome.ok and "没有找到" in outcome.content


def test_system_info_is_a_few_facts(tmp_path: Path) -> None:
    context = ToolContext(store=MemoryStore(":memory:"), data_root=tmp_path)

    outcome = SystemInfoTool().run({}, context)

    assert outcome.ok
    assert "Python" in outcome.content
    assert "系统" in outcome.content


def test_web_search_formats_sources(tmp_path: Path) -> None:
    net = FakeNet()
    context = ToolContext(store=MemoryStore(":memory:"), net=net)

    outcome = WebSearchTool().run({"query": "最近的新闻", "count": 3}, context)

    assert outcome.ok and "https://example.com/1" in outcome.content
    assert net.searched == [("最近的新闻", 3)]


def test_fetch_url_returns_title_and_body() -> None:
    net = FakeNet()
    context = ToolContext(store=MemoryStore(":memory:"), net=net)

    outcome = FetchUrlTool().run({"url": "https://example.com"}, context)

    assert outcome.ok and "标题" in outcome.content and "正文" in outcome.content


def test_web_tools_fail_cleanly_without_a_channel() -> None:
    context = ToolContext(store=MemoryStore(":memory:"))

    assert WebSearchTool().run({"query": "x"}, context).ok is False
    assert FetchUrlTool().run({"url": "https://example.com"}, context).ok is False


def test_web_tools_appear_only_with_a_channel(tmp_path: Path) -> None:
    """**联网默认不登记**：没有 key/开关关着时，模型看不到这两个名字。"""
    without = [tool.spec.name for tool in default_tools(read_root=(tmp_path,))]
    with_web = [tool.spec.name for tool in default_tools(read_root=(tmp_path,), web=FakeNet())]

    assert "web_search" not in without and "fetch_url" not in without
    assert "web_search" in with_web and "fetch_url" in with_web


# ── 写与执行（A2） ──────────────────────────────────────────────────────────


def _write_ctx(tmp_path: Path, *, data_root: Path | None = None) -> ToolContext:
    return ToolContext(
        store=MemoryStore(":memory:"),
        read_root=(tmp_path,),
        write_root=(tmp_path,),
        data_root=data_root,
        allow_commands=True,
        cancel=None,
    )


def test_write_file_writes_and_is_not_replayable(tmp_path: Path) -> None:
    outcome = WriteFileTool().run({"path": "新.txt", "content": "你好"}, _write_ctx(tmp_path))

    assert outcome.ok and outcome.replayable is False
    assert (tmp_path / "新.txt").read_text(encoding="utf-8") == "你好"


def test_write_file_never_touches_the_data_root(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()

    outcome = WriteFileTool().run(
        {"path": "data/mem.db", "content": "x"}, _write_ctx(tmp_path, data_root=data)
    )

    assert outcome.ok is False and "数据目录" in outcome.content
    assert not (data / "mem.db").exists()


def test_edit_file_replaces_one_match(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("alpha beta", encoding="utf-8")

    outcome = EditFileTool().run(
        {"path": "a.txt", "old_text": "beta", "new_text": "gamma"}, _write_ctx(tmp_path)
    )

    assert outcome.ok
    assert target.read_text(encoding="utf-8") == "alpha gamma"


def test_edit_file_refuses_an_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x x", encoding="utf-8")

    outcome = EditFileTool().run(
        {"path": "a.txt", "old_text": "x", "new_text": "y"}, _write_ctx(tmp_path)
    )

    assert outcome.ok is False and "2 次" in outcome.content
    assert target.read_text(encoding="utf-8") == "x x"


def test_edit_file_refuses_when_the_text_is_absent(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("abc", encoding="utf-8")

    outcome = EditFileTool().run(
        {"path": "a.txt", "old_text": "zzz", "new_text": "y"}, _write_ctx(tmp_path)
    )

    assert outcome.ok is False and "没有找到" in outcome.content


def test_run_command_is_refused_when_the_switch_is_off(tmp_path: Path) -> None:
    context = _write_ctx(tmp_path)
    context = ToolContext(
        store=context.store,
        read_root=context.read_root,
        write_root=context.write_root,
        allow_commands=False,
    )

    outcome = RunCommandTool().run({"command": "echo hi"}, context)

    assert outcome.ok is False and "没有打开" in outcome.content


def test_run_command_reports_output_and_exit(tmp_path: Path) -> None:
    import sys

    command = f'"{sys.executable}" -c "print(123)"'
    outcome = RunCommandTool().run({"command": command, "timeout_s": 20}, _write_ctx(tmp_path))

    assert outcome.ok and "123" in outcome.content
    assert "退出码 0" in outcome.content


def test_write_tools_need_a_write_root_and_commands_need_the_switch(tmp_path: Path) -> None:
    without = [tool.spec.name for tool in default_tools(read_root=(tmp_path,))]
    with_write = [
        tool.spec.name for tool in default_tools(read_root=(tmp_path,), write_root=(tmp_path,))
    ]
    with_commands = [
        tool.spec.name
        for tool in default_tools(
            read_root=(tmp_path,), write_root=(tmp_path,), allow_commands=True
        )
    ]

    assert "write_file" not in without and "edit_file" not in without
    assert "run_command" not in without
    assert "write_file" in with_write and "edit_file" in with_write
    assert "run_command" not in with_write
    assert with_commands[-1] == "run_command"
