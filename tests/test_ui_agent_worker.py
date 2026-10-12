"""``AgentWorker``：把工具循环接在 Qt 信号与真线程上的那一层。

**它自己不含逻辑**——循环在 ``agent/loop.py``、闸门在 ``agent/gate.py``，两者都在
核心腿里被逐事件钉过。这里钉的是三件**只有在 Qt 这一侧才成立**的事：

1. **事件 → 信号的翻译不丢东西**：``statistics`` 是 ``ChatWorker`` 那几个键的
   **超集**（界面复用的正是同一个 ``_on_reply_finished``），``tool_finished`` 的
   ``detail`` 截断到 ``_DETAIL_CHARS``，``messages_appended`` 成对原样转发。
2. **停止必须唤醒一个卡在等确认上的线程。** 这是本版唯一没有先例的形状：
   ``thread.quit()`` 只结束线程的事件循环，而 worker 是被 ``started`` 直连进
   ``run()`` 的（见 ``ui/thread_tracking.py``）——它根本不在事件循环里。少了
   ``stop()`` 里那句 ``slot.cancel()``，收尾 ``wait_for_all`` 等满 3 秒后进程继续
   退出，撞上 Windows fast-fail（``0xC0000409``，无回溯、无 stderr）。
3. **没有人同意过的动作没有被执行。**

第 2 条的最后一段**只能在真进程里验**：那个失败的表现是退出码，进程内的断言看不到
它（历史与教训见 ``ui/thread_tracking.py`` 的模块文档）。所以最后一条用例起一个子
进程，断言它退出码为 0——与 ``test_ui_worker_track.py`` 同一套理由。

标 ``gui``：全部要构造 ``QObject`` 或起 ``QThread``，核心腿用
``--ignore-glob='tests/test_ui_*'`` 排除（``--ignore`` 不认通配符，写成它一条都排不掉
且不报错）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# 裸导入而不是 ``from .fakes``：``tests/`` 没有 ``__init__.py``（见
# ``test_ui_chat_worker.py`` 的说明）。
from fakes import ScriptedBackend, text_chunk, tool_chunk

from axo.agent.catalog import LookAtScreenTool
from axo.agent.tools import Capture, ToolContext, ToolOutcome, ToolRegistry
from axo.llm.base import ChatMessage, ChatRequest, ToolCall, ToolSpec
from axo.session.budget import TokenBudget, TokenMeter
from axo.ui.agent_worker import AgentWorker
from axo.ui.chat_worker import ChatWorker
from axo.ui.thread_tracking import THREAD_WAIT_MS
from memcore.store import MemoryStore

pytestmark = pytest.mark.gui

#: 一个"够大"的额度——正常路径的用例不想被截断或驱逐干扰（与 ``test_agent_loop`` 同值）。
ROOMY = 100_000

#: 等在确认上的那条用例用它：**大**到"超时"绝不可能是它退出的原因——能让它退出的
#: 只有 ``slot.cancel()``。这是那条用例唯一能证明自己的方式。
LONG_TIMEOUT = 60.0


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def store() -> MemoryStore:
    opened = MemoryStore(":memory:")
    yield opened
    opened.close()


class SpyTool:
    """记录被怎么调用的工具（与 ``test_agent_loop.py`` 的同名替身同形）。"""

    def __init__(
        self,
        *,
        name: str = "spy",
        requires_approval: bool = False,
        render: Any = None,
        detail: str = "",
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description="测试用替身",
            parameters={"type": "object", "properties": {}},
        )
        self.requires_approval = requires_approval
        self.calls: list[dict[str, Any]] = []
        self._render = render
        self._detail = detail

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        self.calls.append(arguments)
        text = self._render(arguments) if self._render is not None else "替身工具的结果"
        return ToolOutcome.success(text, detail=self._detail)


def call(name: str = "spy", call_id: str = "call_1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def build(
    backend: Any,
    registry: ToolRegistry,
    store: MemoryStore,
    *,
    approval_timeout: float = LONG_TIMEOUT,
    ceiling: int = ROOMY,
) -> AgentWorker:
    return AgentWorker(
        backend=backend,
        messages=[ChatMessage(role="user", content="帮我看看我的笔记")],
        registry=registry,
        ctx=ToolContext(store=store, session_id="S1", turn_seq=1),
        budget=TokenBudget(window=8192),
        meter=TokenMeter(),
        tools_ceiling=ceiling,
        approval_timeout=approval_timeout,
    )


def run(worker: AgentWorker | ChatWorker, *, on_approval: Any = None) -> SimpleNamespace:
    """跑完 worker 并收齐各路信号。**同步调 ``run()``**，不起线程——信号是直连的。

    它同时伺候 ``AgentWorker`` 与 ``ChatWorker``：三个同形的信号在谁身上都接，另外
    四路只在 :class:`AgentWorker` 上存在（平价那条用例正是靠这一点把两边跑成同一份
    形状的，见 ``test_with_no_tools_...``）。

    ``on_approval`` 是给确认卡片那几条用例的钩子：它在**发问的那一刻**（也就是
    worker 还站在 ``slot.wait`` 门口的时候）被调用，于是"用户点了允许/拒绝/停止"
    这三种时序都能被精确地摆出来，不必起线程去抢时间。
    """
    collected = SimpleNamespace(
        chunks=[],
        finished=[],
        failed=[],
        tools=[],
        rounds=[],
        rows=[],
        reasons=[],
        asks=[],
    )
    worker.chunk.connect(collected.chunks.append)
    worker.finished.connect(lambda text, stats: collected.finished.append((text, stats)))
    worker.failed.connect(collected.failed.append)
    if isinstance(worker, AgentWorker):
        worker.tool_finished.connect(collected.tools.append)
        worker.round_started.connect(
            lambda index, offered: collected.rounds.append((index, offered))
        )
        # ``messages_appended`` 是**两个**参数（行 + "这一对怎么结束的"），所以这里
        # 分两路收。只接第一个参数的老写法会在 Qt 上静默丢第二个——而丢了它，落盘
        # 那一侧就会把一次拒绝记成一次正常结果。
        worker.messages_appended.connect(
            lambda rows, reasons: (
                collected.rows.append(rows),
                collected.reasons.append(reasons),
            )
        )
        worker.approval_needed.connect(
            lambda call_id, name, summary: collected.asks.append((call_id, name, summary))
        )
        if on_approval is not None:
            worker.approval_needed.connect(on_approval)
    worker.run()
    return collected


# ── 与 ``ChatWorker`` 同形（界面复用同一个 ``_on_reply_finished``） ─────────


def test_with_no_tools_the_text_and_the_keys_match_chat_worker(
    qapp: Any, store: MemoryStore
) -> None:
    """``tools=()`` 时两个 worker 产出**同一段文本**，且统计的键是超集。

    这是计划 §二 写死的那条平价测试，也是"三个同形信号"那条约定的证据。它红了
    意味着 ``_on_reply_finished`` 复用不过去了——而那条复用正是这一版不合并两条
    消费路径却仍然安全的前提。

    **两条路径的差别被单独钉出来**：``prompt_tokens`` / ``completion_tokens`` 在
    工具循环这一版**恒为 ``None``**（循环不读 token 统计，见 ``agent/loop.py``）。
    不写这一条的话，"键一样"会被读成"值也一样"，而下一次有人顺手把界面上的
    token 计数接到 ``finished`` 上时，他会拿到一个永远空着的数。
    """
    script = [
        [
            text_chunk("你好"),
            text_chunk("，世界", done=True, prompt_tokens=12, completion_tokens=5),
        ]
    ]
    request = ChatRequest(messages=[ChatMessage(role="user", content="帮我看看我的笔记")])

    chat = run(ChatWorker(ScriptedBackend(script), request))
    agent = run(build(ScriptedBackend(script), ToolRegistry([]), store))

    assert agent.chunks == chat.chunks == ["你好", "，世界"]
    assert agent.failed == chat.failed == []
    chat_text, chat_stats = chat.finished[0]
    agent_text, agent_stats = agent.finished[0]
    assert agent_text == chat_text == "你好，世界"
    assert set(chat_stats) <= set(agent_stats), "ChatWorker 的键在 AgentWorker 上缺了"
    assert agent_stats["interrupted"] is False
    assert agent_stats["prompt_tokens"] is None, "工具循环这一版不上报 token，别把它当真"
    assert agent_stats["completion_tokens"] is None


def test_a_plain_reply_reports_empty_tool_statistics(qapp: Any, store: MemoryStore) -> None:
    """没调工具时那四个任务独有的键**存在且是空值**，不是缺键。

    缺键与"这一轮没调工具"在界面上长得一样（状态栏只是少显示一项），所以要有
    一条断言把它们分开——键名写错一个字母不会有任何东西变红。
    """
    backend = ScriptedBackend([[text_chunk("今天挺好。", done=True)]])
    collected = run(build(backend, ToolRegistry([SpyTool()]), store))

    _text, stats = collected.finished[0]
    assert stats["rounds"] == 0
    assert stats["tool_calls"] == 0
    assert stats["wrapped_up"] is False
    assert stats["unexecuted"] == ()
    assert collected.tools == [] and collected.rows == []
    assert collected.rounds == [(0, 1)], "第 0 轮应当声明着工具"


# ── 工具那一轮 ─────────────────────────────────────────────────────────────


def test_a_tool_call_runs_and_its_result_comes_back_in_the_next_request(
    qapp: Any, store: MemoryStore
) -> None:
    """完整的一圈：执行 → ``tool_finished`` 出 payload → 结果回灌 → 终答。"""
    spy = SpyTool(render=lambda _args: "查到了：备用钥匙在鞋柜里")
    backend = ScriptedBackend(
        script=[
            [text_chunk("我先查一下。"), tool_chunk(call("spy", key="钥匙"))],
            [text_chunk("钥匙在鞋柜里。", done=True)],
        ]
    )

    collected = run(build(backend, ToolRegistry([spy]), store))

    assert spy.calls == [{"key": "钥匙"}]
    assert "备用钥匙在鞋柜里" in backend.last_text(1)
    assert collected.rounds == [(0, 1), (1, 1)]
    assert collected.chunks == ["我先查一下。", "钥匙在鞋柜里。"]

    payload = collected.tools[0]
    assert payload["call_id"] == "call_1" and payload["name"] == "spy"
    assert payload["ok"] is True
    assert payload["denied"] is False
    assert payload["clipped"] is False
    assert payload["content"] == "查到了：备用钥匙在鞋柜里"

    _text, stats = collected.finished[0]
    assert stats["rounds"] == 1 and stats["tool_calls"] == 1
    assert stats["interrupted"] is False


def test_the_pair_is_carried_out_whole_for_the_transcript(qapp: Any, store: MemoryStore) -> None:
    """``messages_appended`` 带的是**整对**，而且 tool 行能配回它的调用。

    ``ui/main_window.py`` 收到它就落盘（``stored_rows``），所以这里断言的两件事
    正是逐字稿回放成立的前提：assistant 行带 ``tool_calls``、tool 行带
    ``tool_call_id``。半个配对写进 JSONL 是模型无法解释的一段对话，而回看时它
    看起来只是"少了几步"。
    """
    backend = ScriptedBackend(
        script=[[tool_chunk(call("spy", call_id="c7"))], [text_chunk("好了", done=True)]]
    )

    collected = run(build(backend, ToolRegistry([SpyTool()]), store))

    assert len(collected.rows) == 1, "一对工具消息应当只发一次"
    pair = collected.rows[0]
    assert [row.role for row in pair] == ["assistant", "tool"]
    assert [tool_call.id for tool_call in pair[0].tool_calls] == ["c7"]
    assert pair[1].tool_call_id == "c7"
    # 第二个参数与行**同一次**送到（见信号文档）。这一对是普通结果，所以是空的。
    assert collected.reasons == [()], "普通结果不该被标成不回放"


def test_a_long_detail_is_clipped_on_the_card(qapp: Any, store: MemoryStore) -> None:
    """``detail`` 截到 ``_DETAIL_CHARS``，而 ``content`` 一个字都不动。

    失败时 ``detail`` 是完整栈，可能几千字，而一张卡片的高度不该由一次异常的回溯
    长度决定。截断的是**显示**：``content`` 与真正回灌进上下文的那份保持一致——
    显示得短与"执行得少"是两件事，混了会变成"界面上看着没问题，实际写进去的是
    别的东西"。
    """
    stack = "Traceback...\n" + "x" * 5000
    spy = SpyTool(render=lambda _args: "短结果", detail=stack)
    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("好了")]])

    collected = run(build(backend, ToolRegistry([spy]), store))

    payload = collected.tools[0]
    assert len(payload["detail"]) == 1000
    assert stack.startswith(payload["detail"][:12])
    assert payload["content"] == "短结果"


def test_a_failure_inside_the_tool_is_reported_not_crashed(qapp: Any, store: MemoryStore) -> None:
    """工具炸了 → 一条 ``ok=False`` 的 payload，**不是**一条 ``failed``。

    这是"派发口吃异常"那个设计在信号这一层的可见证据：抛出去会炸掉整轮，用户看到
    "任务莫名失败了"，而模型连改一次参数的机会都没有。
    """

    class Broken(SpyTool):
        def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
            raise RuntimeError("工具自己坏了")

    backend = ScriptedBackend(script=[[tool_chunk(call("spy"))], [text_chunk("那我换个办法")]])

    collected = run(build(backend, ToolRegistry([Broken()]), store))

    assert collected.failed == []
    payload = collected.tools[0]
    assert payload["ok"] is False and payload["denied"] is False
    assert "RuntimeError" in payload["detail"], "栈没有被吃掉"
    assert "工具自己坏了" in backend.last_text(1), "模型没拿到失败的说明"


# ── 闸门 ───────────────────────────────────────────────────────────────────


def test_the_card_question_carries_the_templated_summary(qapp: Any, store: MemoryStore) -> None:
    """``approval_needed`` 带的是**从参数模板化生成**的摘要，不是模型写的话。

    用户是照着这句话点「允许」的。让模型解释自己准备做什么，是幻觉的标准配方
    （``assemble.py`` 里同一条纪律），而这里的后果更重。

    **没人答**（钩子什么都不做）顺手把 fail-closed 也钉住：超时按拒绝，动作不执行。
    超时值调到几十毫秒——这一条量的是"没答等于不做"，不是"等得起"；"等得起"由
    下面那条停止用例负责，而它证明的必须是 ``cancel`` 而不是超时。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[[tool_chunk(call("write_memory", content="用户喜欢美式"))], [text_chunk("记好了")]]
    )

    collected = run(
        build(backend, ToolRegistry([spy]), store, approval_timeout=0.05),
        on_approval=lambda _call_id, _name, _summary: None,
    )

    assert collected.asks == [("call_1", "write_memory", "content：用户喜欢美式")]
    assert spy.calls == [], "没有人同意，却执行了"


def test_the_screen_tool_gets_the_workers_capture(qapp: Any, store: MemoryStore) -> None:
    """**截图那条接缝是 worker 注进去的**（``replace(ctx, capture=self.grab_screen)``）。

    ``look_at_screen`` 拿到的 ``ctx.capture`` 只有这个 worker 给得出来——它要跨回**界面
    线程**取一张图。没注的话，工具会回一句"这次运行没有接上截图通道"：一句真话，而功能
    整个没接上，界面上看不出来（模型会照那句转述给用户，听起来还挺像回事）。

    这里同步跑（``run`` 不起线程），所以"界面线程截一张图"就地用一张预设的图代替。跨线程
    那一趟由 ``tests/test_agent_handoff.py`` 的真线程测试与
    ``tests/test_ui_capture_chain.py`` 守着——三条各管一段，合起来才是整根线。

    最后那一条断言是这一步的**出口**：图必须真的挂在回灌给模型的那一行上。逐字稿与
    L5 都存不住图（所以工具标了 ``replayable=False``），也就是说**这一行是它唯一存在过
    的地方**。
    """
    backend = ScriptedBackend(
        script=[
            [tool_chunk(call("look_at_screen"))],
            [text_chunk("屏幕上是一份表格")],
        ]
    )
    worker = build(backend, ToolRegistry([LookAtScreenTool()]), store)
    adopted: list[bool] = []

    worker.capture_requested.connect(
        lambda token: adopted.append(
            worker.provide_capture(token, Capture(png=b"PNG", width=800, height=600))
        )
    )

    collected = run(worker, on_approval=lambda call_id, *_: worker.provide_approval(call_id, True))

    assert adopted == [True], "worker 没发截图请求，或者回话没被采纳"
    assert collected.tools and collected.tools[0]["ok"] is True
    assert "屏幕图" in collected.tools[0]["content"]

    tool_row = collected.rows[0][1]
    assert tool_row.role == "tool"
    assert tool_row.images, "图没有挂到回灌给模型的那一行上"
    assert collected.reasons[0] == (("call_1", "stale"),), "带图的结果标了可回放"


def test_allow_runs_the_tool_and_deny_does_not(qapp: Any, store: MemoryStore) -> None:
    """同一个问询，两种点击。**两条一起写**，因为单看任何一条都能被假绿通过：
    一个"永远执行"的实现过第一条，一个"永远不执行"的实现过第二条。"""

    def drive(granted: bool) -> tuple[SpyTool, SimpleNamespace, list[bool]]:
        spy = SpyTool(name="write_memory", requires_approval=True)
        backend = ScriptedBackend(
            script=[[tool_chunk(call("write_memory", content="甲"))], [text_chunk("好")]]
        )
        adopted: list[bool] = []
        worker = build(backend, ToolRegistry([spy]), store)

        collected = run(
            worker,
            on_approval=lambda call_id, _name, _summary: adopted.append(
                worker.provide_approval(call_id, granted)
            ),
        )
        return spy, collected, adopted

    granted_spy, granted_run, granted_adopted = drive(granted=True)
    assert granted_adopted == [True], "同期答的那一下应当被采纳"
    assert granted_spy.calls == [{"content": "甲"}]
    assert granted_run.tools[0]["denied"] is False and granted_run.tools[0]["ok"] is True

    denied_spy, denied_run, denied_adopted = drive(granted=False)
    # **采纳了**，采纳的是一次"拒绝"——``provide`` 的返回值说的是"算不算数"，不是
    # "同不同意"。这两件事混起来，就是"用户以为他同意了、系统认为没人同意"。
    assert denied_adopted == [True]
    assert denied_spy.calls == [], "被拒绝的动作还是执行了"
    assert denied_run.tools[0]["denied"] is True and denied_run.tools[0]["ok"] is False
    assert "没有同意" in denied_run.rows[0][1].content, "拒绝没有回灌给模型"


def test_a_card_that_arrives_after_the_stop_is_dropped(qapp: Any, store: MemoryStore) -> None:
    """停止之后到达的裁决**不被采纳**（``provide`` 返回 ``False``）。

    界面据此把卡片标成"这次点击没有生效"。不区分的话，用户以为他同意了，而闸门
    记的是没人同意过——那是这一版唯一一条静默分歧。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[[tool_chunk(call("write_memory", content="甲"))], [text_chunk("好")]]
    )
    worker = build(backend, ToolRegistry([spy]), store)
    adopted: list[bool] = []

    collected = run(
        worker,
        on_approval=lambda call_id, _n, _s: (
            worker.stop(),
            adopted.append(worker.provide_approval(call_id, True)),
        ),
    )

    assert collected.asks, "夹具没造出「问询已经发出」这个形状"
    assert adopted == [False]
    assert spy.calls == []


def test_stop_during_approval_wakes_the_worker_instead_of_waiting_for_the_timeout(
    qapp: Any, store: MemoryStore
) -> None:
    """**停止必须叫醒它**，不能等满确认超时。

    少了 ``stop()`` 里那句 ``slot.cancel()``，这条会红——而且是**等满
    ``LONG_TIMEOUT`` 之后**才红。收尾路径上等不了那么久（``wait_for_all`` 只给
    3 秒），所以那正是 ``0xC0000409`` 的成因。

    另外两条断言一起钉：**已经流出去的字留着**（用户点停止不等于他要丢掉已经看到的
    那些），而**留着的那段正是用户已经看到的那一段**——停止之后逐字稿里会写进
    "几轮的话拼在一起"的话，回看时看起来像模型精神错乱。

    这里留着的是 ``"我记一下："``（第 0 轮那段），不是 ``""``：退出点落在这一对的
    :class:`MessagesAppended` 上，所以第 1 轮的 ``RoundStarted`` 根本没来过，缓冲
    还没被清掉。已经流出去的字与留在手里的那段由此是同一段，这一条正是"两者不许
    不一致"的证据。

    而 ``rows`` 有一对、``spy`` 一次没跑，两条合起来才是那次修复的全部：**被拒绝的
    这一步仍然要进逐字稿**（模型真的请求过，用户真的没同意），而**没人同意的动作一次
    都没执行过**。
    """
    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[
            [text_chunk("我记一下："), tool_chunk(call("write_memory", content="甲"))],
            [text_chunk("好")],
        ]
    )
    worker = build(backend, ToolRegistry([spy]), store)

    started = time.monotonic()
    collected = run(worker, on_approval=lambda *_args: worker.stop())
    elapsed = time.monotonic() - started

    assert collected.asks, "夹具没造出「worker 站在等确认上」这个形状"
    assert elapsed < 2.0, f"stop() 没能立刻叫醒它（等了 {elapsed:.1f} 秒）"
    assert spy.calls == [], "没人同意过的动作被执行了"
    assert collected.chunks == ["我记一下："]
    text, stats = collected.finished[0]
    assert stats["interrupted"] is True
    assert text == "我记一下：", "留着的那段必须与已经流出去的那段一致"
    assert len(collected.rows) == 1, "被拒绝的那一步没有进逐字稿"


def test_a_stop_later_in_the_turn_keeps_only_that_round(qapp: Any, store: MemoryStore) -> None:
    """停止发生在**第二**轮里：手里留的只有第二轮那段，不是两轮拼起来的话。

    上一条量不到这件事——它停在第 0 轮，而第 0 轮的缓冲从头到尾只有一段，累加与
    不累加在那里长得一模一样。要把"每轮重新开始"钉住，必须让**至少两轮都说出过
    字**，然后停在后面那一轮。

    ``RoundStarted`` 那句 ``parts = []`` 少了的话，症状是**只有用户点停止时才出现**
    的一种逐字稿：模型前面说过的话与最后那句被拼成一段，回看时读起来像它一口气
    说了三段互相矛盾的话，而真正的分界（中间夹着工具结果）在文字里看不出来。

    ``rows`` 只有一对是同一件事的另一面：第二轮那次调用**没有被执行**。停止要么
    在工具跑起来之前生效、要么就等它跑完并留下记录——**不存在"跑了但没记"这一档**，
    这是这个项目唯一不能有的失效（见 ``ui/agent_worker.py`` 的模块文档）。
    """
    spy = SpyTool(name="search_memory")  # 不过闸门，于是第二条路走的是"直接执行"
    backend = ScriptedBackend(
        script=[
            [text_chunk("第一轮的话"), tool_chunk(call("search_memory", call_id="c1"))],
            [text_chunk("第二轮的话"), tool_chunk(call("search_memory", call_id="c2"))],
            [text_chunk("不该走到这里")],
        ]
    )
    worker = build(backend, ToolRegistry([spy]), store)

    def stop_once_round_two_is_out(text: str) -> None:
        if text == "第二轮的话":
            worker.stop()

    worker.chunk.connect(stop_once_round_two_is_out)
    collected = run(worker)

    assert collected.chunks == ["第一轮的话", "第二轮的话"], "夹具没造出「两轮都说过话」"
    assert len(spy.calls) == 1, "第二轮那一步还是执行了"
    assert len(collected.rows) == 1, "没执行的那一步不该有记录"
    text, stats = collected.finished[0]
    assert stats["interrupted"] is True
    assert text == "第二轮的话", f"几轮的话被拼在一起了：{text!r}"


# ── 真线程：起一个 worker，从「界面线程」停它 ──────────────────────────────


def test_the_gui_thread_can_stop_a_worker_blocked_inside_the_slot(
    qapp: Any, store: MemoryStore
) -> None:
    """真线程里的同一条约定：**卡在 ``wait`` 上的 worker 会被 ``stop()`` 叫醒。**

    与上一条的区别是**形状**：上一条的钩子在同一根线程里、在 ``wait`` 门口就把它拦
    下了；这一条走的是真实时序——worker 在另一根线程里真的进入了 ``slot.wait``，
    界面线程在它外面把它停掉。生产里发生的就是后者。

    先断言"它此刻确实还在跑"：一个已经退出的 worker 是能被"停"成功的，而那什么都没
    证明（这条测试就成了永远绿的）。
    """
    from PySide6.QtCore import Qt, QThread

    spy = SpyTool(name="write_memory", requires_approval=True)
    backend = ScriptedBackend(
        script=[[tool_chunk(call("write_memory", content="甲"))], [text_chunk("好")]]
    )
    worker = build(backend, ToolRegistry([spy]), store, approval_timeout=LONG_TIMEOUT)

    asks: list[tuple[str, str, str]] = []
    worker.approval_needed.connect(lambda c, n, s: asks.append((c, n, s)))

    thread = QThread()
    # 与 ``track()`` 同形：亲和面留在界面线程，``run()`` 靠直连在工作线程里跑。
    # 忘了 ``Qt.DirectConnection`` 的话 ``run()`` 会被排队回界面线程，症状看起来像
    # Ollama 慢（见 ``ui/thread_tracking.py``）。
    thread.started.connect(worker.run, Qt.DirectConnection)
    thread.finished.connect(thread.deleteLater)
    worker.finished.connect(thread.quit)

    thread.start()

    deadline = time.monotonic() + 5
    while not asks and time.monotonic() < deadline:
        # pytest 里没有 ``app.exec()``，排队送达的东西要靠这一句派发。
        qapp.processEvents()
        time.sleep(0.001)
    assert asks, "worker 没能进到等确认那一步"

    assert thread.isRunning(), "worker 已经自己退出了——这条测试量不到东西"

    started = time.monotonic()
    worker.stop()
    while thread.isRunning() and time.monotonic() - started < THREAD_WAIT_MS / 1000 + 2:
        qapp.processEvents()
        time.sleep(0.001)
    elapsed = time.monotonic() - started

    assert not thread.isRunning(), f"stop() 唤不醒卡在等确认上的 worker（等了 {elapsed:.1f} 秒）"
    assert elapsed < THREAD_WAIT_MS / 1000, (
        f"它是靠超时结束的（{elapsed:.1f} 秒），而收尾只等 {THREAD_WAIT_MS / 1000} 秒——"
        "那正是 0xC0000409 的成因"
    )
    assert spy.calls == [], "没人同意过的动作被执行了"


# ── 真进程：收尾期间的退出码 ───────────────────────────────────────────────

_CHILD = '''
"""子进程：收尾（``_shutdown`` 的形状）落在一个卡在等确认上的 worker 上。

退出码是这条路径上**唯一**看得见的东西：Windows fast-fail 不带回溯、不带 stderr，
连 Python 的 ``atexit`` 都不跑（见 ``ui/thread_tracking.py`` 的模块文档）。
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, sys.argv[1])

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from axo.agent.tools import ToolContext, ToolOutcome, ToolRegistry
from axo.llm.base import ChatChunk, ChatMessage, ToolCall, ToolSpec
from axo.session.budget import TokenBudget, TokenMeter
from axo.ui.agent_worker import AgentWorker
from axo.ui.thread_tracking import track, wait_for_all
from memcore.store import MemoryStore

_RAN = []


class _Backend:
    """吐一个改状态的工具调用，然后（如果还能继续）就结束了。"""

    model_id = "fake/scripted"
    context_window = 8192

    def count_tokens(self, text):
        return None

    def stream(self, request):
        yield ChatChunk(text="我记一下：", done=False)
        yield ChatChunk(
            text="",
            done=True,
            tool_calls=(ToolCall(id="c1", name="write_memory", arguments={"content": "甲"}),),
        )


class _Gated:
    spec = ToolSpec(
        name="write_memory",
        description="写一条记忆",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string", "description": "正文"}},
            "required": ["content"],
        },
    )
    requires_approval = True

    def run(self, arguments, ctx):
        # 走到这里说明没有人同意过却执行了。记下来由父进程断言——抛异常会被派发口
        # 吃成一条 ok=False 的结果（那是给模型自纠用的），在这里就看不见了。
        _RAN.append(arguments)
        return ToolOutcome.success("写好了")


class _Receiver:
    def __init__(self):
        self._live_threads = []


def main():
    app = QApplication([])
    worker = AgentWorker(
        backend=_Backend(),
        messages=(ChatMessage(role="user", content="记一条：我喜欢美式"),),
        registry=ToolRegistry([_Gated()]),
        ctx=ToolContext(store=MemoryStore(":memory:")),
        budget=TokenBudget(window=8192),
        meter=TokenMeter(),
        tools_ceiling=1000,
        approval_timeout=120.0,
    )
    receiver = _Receiver()
    thread = QThread()
    track(receiver, thread, worker, what="任务线程", slot="agent")

    asked = []
    worker.approval_needed.connect(lambda *args: asked.append(args))
    thread.start()

    deadline = time.monotonic() + 10
    while not asked and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    if not asked:
        print("BLOCKED=never-asked", flush=True)
        return 2

    # 与 MainWindow._shutdown 同形：先 stop，再按在跑的清单收尾。
    worker.stop()
    stragglers = wait_for_all(list(receiver._live_threads))
    print(f"RAN={len(_RAN)} STRAGGLERS={len(stragglers)}", flush=True)
    return 1 if stragglers else 0


raise SystemExit(main())
'''


def test_the_process_exits_cleanly_when_shutdown_happens_during_approval(tmp_path: Path) -> None:
    """收尾落在"卡在等确认上"的 worker 上时，**进程以 0 退出**。

    不这么测就测不到：发不出来的那个症状是退出码，而进程内的断言在崩溃发生之前
    就已经跑完了。三条断言各有各的用处——退出码是那条病史本身；``RAN=0`` 说明
    没有人在没人同意的情况下执行了动作；``STRAGGLERS=0`` 说明收尾真的等到了它，
    而不是"碰巧够快"。
    """
    script = tmp_path / "child_shutdown_during_approval.py"
    script.write_text(_CHILD, encoding="utf-8", newline="\n")
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

    result = subprocess.run(
        [sys.executable, str(script), str(src)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )

    assert result.returncode == 0, (
        f"子进程退出码 {result.returncode}（0xC0000409 = 3221226505）\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "RAN=0" in result.stdout, f"没人同意过的动作被执行了：{result.stdout}"
    assert "STRAGGLERS=0" in result.stdout, f"收尾没等到那个线程：{result.stdout}"
