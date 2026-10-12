"""把纯层的工具循环接到 Qt 信号上。

:func:`~axo.agent.loop.run_agent_loop` 是一个**同步生成器**：它不碰 Qt、
不碰线程，只吐事件。这个类是它唯一的 Qt 侧消费者，干三件事：

1. **事件 → 信号。** 生成器在工作线程里跑，每个事件转成一次 ``emit``，Qt 会把它
   排队送进界面线程。
2. **确认闸门 → 卡片。** ``await_approval`` 那个注入回调在这里实现：登记一次问询、
   发一个信号让界面画卡片、然后在工作线程里等（见 :meth:`_await_approval`）。
3. **逐字稿的行带出来。** :class:`~axo.agent.loop.MessagesAppended`
   原样转出去，由界面写进会话（见那个事件的理由：用户中途停止时，攒到末尾的记录
   会连同这一轮一起消失）。

## 三个信号与 :class:`~.chat_worker.ChatWorker` 同形，另外四个是任务特有的

``chunk`` / ``finished`` / ``failed`` **逐字段**与 ``ChatWorker`` 一致，所以
``_on_reply_finished``、``_on_reply_failed`` 两个槽能直接复用，一个字都不用改。

这不是省事，是**两条消费路径的接缝必须有一个固定的形状**：它们迟早要合并成一条
（见 P3a 计划 §二 的移除准则），而合并的前提是"同一个形状有两种实现"这件事先被
钉住。多出来的四个（``round_started`` / ``tool_finished`` / ``approval_needed`` /
``messages_appended``）只在任务语域里有意义。

## ``statistics`` 里多几个键，不改少任何一个

``finished`` 的第二个参数是**超集**：``ChatWorker`` 的三个键一个不少（缺的
``prompt_tokens``/``completion_tokens`` 是 ``None``——工具循环这一版不上报 token），
再加上 ``rounds`` / ``tool_calls`` / ``wrapped_up`` / ``unexecuted`` / ``tool_trace``。

``unexecuted`` **调用方不许吞掉**：那是"模型还想调工具但没有声明可用了"的那几个名字，
吞掉它就等于回到了 ``ChatWorker`` 被修掉的那个缺陷——静默丢弃。

``tool_trace`` 是这一轮**做过什么**，由
:func:`~axo.agent.trace.build_trace` 压成能进 ``meta`` 的形状，交给
:class:`~axo.memory.policy.TaskPolicy`。**累加点在这里而不是
``LoopDone`` 上**：用户按了停止时生成器被 ``break``，``LoopDone`` 永不产出，而
"被打断的那一轮"恰恰最该留下它做过的那些事。停止**之后**仍然走得到底下那次
``emit``，所以被中断的轨迹天然是完整的。

## 停止为什么必须连槽一起取消

``stop()`` 做两件事：翻 ``_stop_requested``、**并且** ``slot.cancel()``。少了后者，
一个正卡在 :meth:`ApprovalSlot.wait` 里的工作线程没有任何东西能叫醒它，而收尾
``wait_for_all`` 只等 3 秒——超时后进程继续退出，撞上 Windows fast-fail
（``0xC0000409``，无回溯、无 stderr）。**这不是优化，是这条设计成立的前提。**

## 停止只在"没有东西悬在半空"的落点上生效

判据放在**每个事件处理完之后**，而且**跳过一个例外**：``ToolFinished``。

* 放在处理之后：``MessagesAppended`` 是"已经执行过的调用进了逐字稿"那一步，在它
  **之前**退出就等于把那一对丢了。而 :func:`~.run_agent_loop` 恰恰是**先**吐
  ``ToolFinished``、**后**吐 ``MessagesAppended`` 的，两次 ``yield`` 之间夹着一次
  **已经发生过的副作用**（工具真的跑了，``write_memory`` 可能已经落了库）。
* 跳过 ``ToolFinished``：这是同一个窗口的另一半——不加这个例外，循环会在"跑完但还
  没记"的那一刻被中断。

两者合起来才有那条不能破的约定：**已经执行过的调用一定留有记录**。这是这个项目
唯一不能有的失效——一份"做过但没记"的逐字稿，回看时看不出少了什么，而它记的正是
"这次对话真的发生过什么"。

代价是**停止会晚一步**：已经跑起来的那一批里，剩下的几个还会执行完。可以接受——
会改状态的三个工具都要过闸门，而 :meth:`_await_approval` 在已经按过停止时直接返回
``None``（按拒绝），所以那批里剩下的最多是几个只读的查询。**这条推理依赖"改状态的
工具一律过闸门"**，加新工具时要照这条查一遍。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..agent.gate import APPROVAL_TIMEOUT_S, ApprovalDecision, ApprovalRequest, ApprovalSlot
from ..agent.handoff import CaptureSlot
from ..agent.loop import (
    DEFAULT_LIMITS,
    LoopDone,
    LoopLimits,
    MessagesAppended,
    RoundStarted,
    TextDelta,
    ToolFinished,
    run_agent_loop,
)
from ..agent.tools import Capture, ToolContext, ToolRegistry
from ..agent.trace import build_trace
from ..llm.base import ChatBackend, ChatBackendError, ChatMessage
from ..session.budget import TokenBudget, TokenMeter

__all__ = ["AgentWorker"]

_EMPTY_STATISTICS: dict[str, Any] = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "interrupted": False,
    "rounds": 0,
    "tool_calls": 0,
    "wrapped_up": False,
    "unexecuted": (),
    "tool_trace": (),
}
"""``finished`` 的第二个参数缺省长什么样。

写成一个模块常量而不是每次现拼：界面读的是**键名**，而键名写错一个字母不会有任何
东西变红——状态栏上只是少显示一项，看起来像"这一轮没调工具"。
"""

CAPTURE_TIMEOUT_S = 5.0
"""等界面线程回一张截图的上限（秒）。

比确认那个 180 秒短两个数量级，因为这件事**在界面线程上是毫秒级的**（截一块屏 + 缩放 +
编码 PNG）。给 5 秒是留给"界面正忙"的余量，而不是留给"用户在看"的——这条路上没有人
要看什么，只有一根 worker 线程在等。等不到就如实说没截到（见 :meth:`AgentWorker.grab_screen`），
**绝不无限等**：收尾只给 3 秒，任何一次新的等待都必须在这个预算里能结束。
"""

_DETAIL_CHARS = 1000
"""工具卡片上最多显示多少字的 ``detail``。

:attr:`~axo.agent.tools.ToolOutcome.detail` 在失败时是**完整栈**，那可能
几千字。截断而不是不显示：不显示就是把一次异常从界面上抹掉（这个项目里"没有被谁
看到的异常才是静默失败"），而一张卡片的高度不该由一次异常的回溯长度决定。
"""


class AgentWorker(QObject):
    """跑一轮任务语域：可能要调几次工具，然后给出回答。"""

    chunk = Signal(str)
    """一段增量文本。与 :attr:`ChatWorker.chunk` 同义。"""

    finished = Signal(str, dict)
    """生成结束：``(完整文本, 统计)``。

    即使出错或用户点了停止也会发出——界面据此复位状态。``interrupted`` 为真表示
    是用户停的。**统计里比 ``ChatWorker`` 多几个键**（见模块文档），一个都不少。
    """

    failed = Signal(str)
    """出错，参数是可直接显示给用户的句子。"""

    round_started = Signal(int, int)
    """新一轮开始：``(轮次, 这次带上了几个工具声明)``。

    界面用它给上一轮的正文收尾。**不收尾的话，几轮的"边想边说"会连成一条气泡**，
    而它读起来像是模型一口气说了三段互相矛盾的话——真正的分界（中间夹着工具结果）
    在文字里完全看不出来。
    """

    tool_finished = Signal(dict)
    """一次工具调用结束。一个字典而不是五个位置参数，因为它有六项，其中一半是标志。

    键：``call_id`` / ``name`` / ``ok`` / ``denied`` / ``clipped`` / ``content`` /
    ``detail``。前五项的语义与 :class:`~axo.agent.loop.ToolFinished`
    一致；``detail`` 已按 :data:`_DETAIL_CHARS` 截断。
    """

    capture_requested = Signal(str)
    """要一张屏幕截图：参数是这次交接的身份（``CaptureSlot`` 的 token）。

    **界面线程回话时必须把它原样带回来**（:meth:`provide_capture`）：迟到的回答会被
    丢掉，理由见 ``agent/handoff.py`` 的"身份"那一段——那一张**三秒前**的屏幕被当成
    "刚刚看到的"，模型会照着它回答，而谁也看不出差别。
    """

    approval_needed = Signal(str, str, str)
    """有一个改状态的动作在等用户点头：``(call_id, 工具名, 一句话摘要)``。

    **界面必须为它画一张卡片。** 收到这个信号而没有画，用户就永远看不到这次问询，
    而 worker 会一直等到超时然后按拒绝处理——症状是"助手说它没做"。
    """

    messages_appended = Signal(object, object)
    """一对工具消息进了对话：``(tuple[ChatMessage, ...], tuple[tuple[str, str], ...])``，
    **调用方要写进逐字稿**。

    原样转发，不在这里落盘：写会话是界面的事（它才知道当前是哪个会话）。这里
    只保证"已经真的执行过的调用，一定有记录被送出去"。

    **第二个参数是"这一对怎么结束的"**（``(tool_call_id, 原因)``，见
    :attr:`~axo.agent.loop.MessagesAppended.no_replay`）。它必须和第一个
    参数**同一次**送到：分开送的话，交接中间那次落盘就会写出一个"这一对是正常结果"
    的逐字稿，而它其实是被拒绝的那一对——而 L5 会照这个错记法回放它。
    """

    def __init__(
        self,
        *,
        backend: ChatBackend,
        messages: Sequence[ChatMessage],
        registry: ToolRegistry,
        ctx: ToolContext,
        budget: TokenBudget,
        meter: TokenMeter,
        tools_ceiling: int,
        limits: LoopLimits = DEFAULT_LIMITS,
        approval_timeout: float = APPROVAL_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._messages = tuple(messages)
        self._registry = registry
        self._ctx = ctx
        self._budget = budget
        self._meter = meter
        self._tools_ceiling = tools_ceiling
        self._limits = limits
        self._approval_timeout = approval_timeout
        self._stop_requested = False
        #: 一个槽服务一次生成（见 :class:`ApprovalSlot`），所以它在 ``__init__`` 里
        #: 建、在这个 worker 的一生里不动。下一轮是一个新 worker。
        self._slot = ApprovalSlot()
        #: 截图那一次的交接。**与 ``_slot`` 是两个槽**：同一轮里"等用户点头"与"等一张
        #: 图"是两件不相干的事，共用一个槽会让其中一件把另一件的待决状态挤掉。而同一个
        #: 截图槽在一轮里可以用两次（两次工具往返），所以它不能是"一次生成一个"。
        self._capture_slot = CaptureSlot()

    # ── 界面线程调的两个入口 ────────────────────────────────────────────────

    @Slot()
    def stop(self) -> None:
        """请求停止。**界面线程直接调用这个方法**（普通函数调用，不走信号）。

        **两步缺一不可**：翻布尔让循环在下一个事件处退出；``cancel()`` 立刻唤醒
        那个可能正卡在等确认上的线程。只翻布尔的话，用户点停止之后界面看起来
        没反应——要等满确认超时才会收场，而收尾路径只等 3 秒，于是进程以
        ``0xC0000409`` 结束（见模块文档）。
        """
        self._stop_requested = True
        self._slot.cancel()
        # 两个槽都要叫醒：谁正卡在等，谁就是那个会让收尾等满 3 秒的线程。
        self._capture_slot.cancel()

    def provide_approval(self, call_id: str, granted: bool) -> bool:
        """界面线程回传一次裁决。返回**它是否被采纳**。

        **直接函数调用，不走信号**——理由与 :meth:`ChatWorker.stop` 一字不差：收尾
        路径上事件循环已经在退出，排队的东西未必还能被派发。而这个槽是线程安全的
        （``ApprovalSlot`` 自己带锁），所以直连没有代价。

        返回 ``False`` 有三条来路：``call_id`` 对不上（用户点了一张过期的卡片）、
        这张卡片已经答过了、这一轮已经取消。界面据此把卡片标成失效——**不能默认
        它成功**，那会让用户以为自己的点击起了作用。
        """
        decision = ApprovalDecision.GRANT if granted else ApprovalDecision.DENY
        return self._slot.provide(call_id, decision)

    def provide_capture(self, token: str, shot: Capture) -> bool:
        """界面线程回传一张截图。返回**它是否被采纳**。

        与 :meth:`provide_approval` 同一个形状、同一条理由（直连、不走信号），也同一
        条纪律：返回 ``False`` 表示这一张**没人要**（token 对不上、或者这一轮已经
        拆了），界面不该据此以为"它看到了"。
        """
        return self._capture_slot.provide(token, shot)

    def grab_screen(self) -> Capture:
        """截一张屏幕。**这是工具的 ``ctx.capture``，在工作线程里跑。**

        三步：登记一次交接 → 发信号让界面去截 → 有界等待。任一步失败都回报一句人话的
        :class:`Capture` 而不是抛异常——工具那一侧对这几种收场要做的是同一件事（如实
        告诉模型没看成），而异常的差别只会体现在那句话里。
        """
        if self._stop_requested:
            return Capture(reason="这一轮已经停止")
        try:
            token = self._capture_slot.ask()
        except RuntimeError:
            # 槽是单向闩：取消之后再发起只可能是接线错（同 `_await_approval` 那一条）。
            return Capture(reason="这一轮已经停止")
        self.capture_requested.emit(token)
        shot = self._capture_slot.wait(timeout=CAPTURE_TIMEOUT_S)
        if shot is None:
            return Capture(reason="没等到截图：界面那边没有回话")
        return shot

    # ── 工作线程 ────────────────────────────────────────────────────────────

    def _await_approval(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """循环的 ``await_approval`` 回调。**在工作线程里跑。**

        返回 ``None`` 覆盖超时、停止、以及"槽已经取消"三种收场，由循环归成拒绝。
        三种都对，也都不该分开处理：对**被执行的动作**来说它们没有区别——都没有
        人同意过（见 :func:`~axo.agent.gate.resolve_decision`）。
        """
        if self._stop_requested:
            # 已经按过停止，连卡片都不必画：这一轮正在被拆掉。
            return None
        try:
            self._slot.request(request)
        except RuntimeError:
            # 槽是单向闩：取消之后再发起问询只可能是接线错。这里按拒绝收场而不是
            # 让它抛出去——抛出去会炸掉整轮，而用户看到的是"任务莫名失败了"。
            return None
        self.approval_needed.emit(request.call_id, request.tool_name, request.summary)
        return self._slot.wait(timeout=self._approval_timeout)

    @Slot()
    def run(self) -> None:
        """在线程里执行。由 ``QThread.started`` 触发。"""
        parts: list[str] = []
        final_text = ""
        interrupted = False
        statistics = dict(_EMPTY_STATISTICS)
        # 轨迹的两路来源。**攒在这里而不是从信号里回收**：那两个信号是给界面看的，
        # 而界面在停止之后可能已经不再收（卡片刚被清掉）。攒成两份再交给纯函数压，
        # 配对与截断的判断就全在 ``agent/trace.py`` 里，一行 Qt 都不用碰。
        executed: list[ToolFinished] = []
        appended: list[MessagesAppended] = []

        try:
            for event in run_agent_loop(
                backend=self._backend,
                messages=self._messages,
                registry=self._registry,
                # **截图那一次交接由这个 worker 注进去。** 工具拿到的是"给我一张屏幕"
                # 这个动作本身（它不知道 Qt 长什么样），而实现要跨回界面线程——那件事
                # 只有这个 worker 做得到。与 ``dispatch`` 给 ctx 绑 ``call_id`` 同一手法：
                # 谁有能力，谁往那个袋子里放。
                ctx=replace(self._ctx, capture=self.grab_screen),
                await_approval=self._await_approval,
                budget=self._budget,
                meter=self._meter,
                tools_ceiling=self._tools_ceiling,
                limits=self._limits,
            ):
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                    self.chunk.emit(event.text)
                elif isinstance(event, RoundStarted):
                    # 新一轮的正文**重新开始**：``LoopDone.final_text`` 只是最后一轮
                    # 那一段，而中断时手里该留的也只是当前这一段。累加的话，停止之后
                    # 逐字稿里会写进"三轮的话拼在一起"，看起来像模型精神错乱。
                    parts = []
                    self.round_started.emit(event.index, event.tools_offered)
                elif isinstance(event, ToolFinished):
                    executed.append(event)
                    self.tool_finished.emit(
                        {
                            "call_id": event.call_id,
                            "name": event.name,
                            "ok": event.ok,
                            "denied": event.denied,
                            "clipped": event.clipped,
                            "content": event.content,
                            "detail": event.detail[:_DETAIL_CHARS],
                        }
                    )
                elif isinstance(event, MessagesAppended):
                    appended.append(event)
                    self.messages_appended.emit(event.rows, event.no_replay)
                elif isinstance(event, LoopDone):
                    final_text = event.final_text
                    interrupted = False
                    statistics.update(
                        rounds=event.rounds,
                        tool_calls=event.tool_calls,
                        wrapped_up=event.wrapped_up,
                        unexecuted=event.unexecuted,
                    )

                # 停止的判据在**处理完之后**，而且**跳过 ``ToolFinished``**（理由见
                # 模块文档"停止只在没有东西悬在半空的落点上生效"）。放在处理之前的话，
                # 它会在 ``MessagesAppended`` 那一步之前退出，于是"已经跑过、也已经
                # 落了库"的那一对不会进逐字稿——那正是这个项目唯一不能有的失效。
                if self._stop_requested and not isinstance(event, ToolFinished):
                    # 收下已到手的正文再走：用户点了停止不等于他要丢掉已经看到的
                    # 那些字——与 ``ChatWorker`` 的"当前这句说完就停"是同一条。
                    final_text = "".join(parts)
                    interrupted = True
                    break
        except ChatBackendError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — 兜底，绝不把线程炸掉
            # 一个未捕获的异常会静默终止这个线程，界面永远停在"生成中"。
            self.failed.emit(f"任务过程中出现未预期的错误：{type(exc).__name__}: {exc}")
            return

        statistics["interrupted"] = interrupted
        statistics["tool_trace"] = build_trace(executed, appended)
        # 用 return 而不是 try/except/else 收场：``except`` 那两条已经 return 了，
        # 走到这里一定是成功路径。与 ``ChatWorker`` 一样，**要么 finished、
        # 要么 failed，二选一**——两处都发的表现是界面先复位、再报错。
        self.finished.emit(final_text, statistics)
