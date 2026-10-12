"""确认闸门。

前半是纯策略（表驱动，一眼能核对），后半是 :class:`ApprovalSlot`——本版唯一一处
**真起线程**的核心腿测试。分两半写，因为它们的失败方式完全不同：策略写错了是
"该问的没问"，槽写错了是**死锁或张冠李戴的允许**，而后者在界面上看起来只是"点了
没反应"或者"点了别的东西被执行了"。

槽那部分的时间断言都留了很大的余量（等 5 秒的地方只要求 1 秒内返回）。理由不是
宽容，是**这类测试的可信度取决于它测的是"会不会卡住"而不是"快不快"**：一个把
时间卡到毫秒的断言，在有负载的 CI 上会红，然后被人调大，然后没人再看它。
"""

from __future__ import annotations

import threading
import time

import pytest

from axo.agent.catalog import GetTimeTool
from axo.agent.gate import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSlot,
    build_request,
    resolve_decision,
    summarize,
)
from axo.agent.tools import Tool
from axo.llm.base import ToolCall


def req(call_id: str = "call_1") -> ApprovalRequest:
    return ApprovalRequest(
        call_id=call_id,
        tool_name="write_memory",
        summary="内容：备用钥匙在鞋柜最上面那格",
    )


# ── 纯策略 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (ApprovalDecision.GRANT, ApprovalDecision.GRANT),
        (ApprovalDecision.DENY, ApprovalDecision.DENY),
        (None, ApprovalDecision.DENY),
    ],
)
def test_every_answer_maps_to_a_decision(
    answer: ApprovalDecision | None, expected: ApprovalDecision
) -> None:
    """``None``（超时 / 取消 / 通道没了）归成**拒绝**。

    三条放一张表里，是因为它们的区别只在于"没答"是怎么发生的——对被执行的动作来说
    没有区别，都没有人同意过。
    """
    assert resolve_decision(answer) is expected


def test_a_request_carries_the_call_id_the_ui_has_to_send_back() -> None:
    """问询里必须带上 ``call_id``：界面回传裁决时靠它比对。

    没有它，"用户点得慢"和"用户点了别的卡片"就分不开，而分不开的表现是一次张冠李戴
    的允许。
    """
    call = ToolCall(id="call_7", name="forget_memory", arguments={"ids": ["m1"]})
    request = build_request(call, round_index=2)

    assert request.call_id == "call_7"
    assert request.tool_name == "forget_memory"
    assert request.round_index == 2
    assert request.arguments == {"ids": ["m1"]}
    assert "m1" in request.summary


class _StatedTool:
    """一个"没有参数、但要自己说明范围"的工具替身。

    **只有一个可选属性**——这正是要点：那句话不进 ``Tool`` 协议（见
    :func:`test_the_sentence_is_not_part_of_the_tool_protocol`），所以"有这个属性"
    就是全部的形状。
    """

    def __init__(self, text: str = "") -> None:
        if text:
            self.approval_summary = text


def test_a_tool_without_arguments_can_state_its_own_scope() -> None:
    """**没有参数的工具，那句话由它自己说。**

    模板对空参数什么都生成不出来（只有 ``工具名（无参数）``），而"它到底要做什么"恰恰
    是用户点「允许」时唯一要看的东西。这一条是给 ``look_at_screen`` 那种工具留的出口：
    它的范围（整块屏幕）是**固定**的，不该靠参数表达、也表达不出来。
    """
    one = ToolCall(id="c1", name="look_at_screen", arguments={})
    tool = _StatedTool("把整块屏幕的内容交给模型看一次（Axo 自己的窗口会被裁掉）")

    assert summarize(one, tool=tool) == "把整块屏幕的内容交给模型看一次（Axo 自己的窗口会被裁掉）"
    assert build_request(one, tool=tool).summary == summarize(one, tool=tool)


def test_a_tool_without_arguments_and_without_a_sentence_falls_back() -> None:
    """没写那句话的工具仍然是旧形状。**这条是向后兼容的判据**：九个既有工具都没有这个
    属性，而它们一张卡片都不该变。

    空串与纯空白也一律回落——一个"写了但没写东西"的属性不该把卡片变成一行空白。
    """
    one = ToolCall(id="c1", name="get_time", arguments={})

    assert summarize(one) == "get_time（无参数）"
    assert summarize(one, tool=None) == "get_time（无参数）"
    assert summarize(one, tool=_StatedTool()) == "get_time（无参数）"
    assert summarize(one, tool=_StatedTool("   ")) == "get_time（无参数）"


def test_the_statement_never_overrides_the_arguments() -> None:
    """**只在参数为空时用它。** 有参数的工具照旧按参数说——那句话是"模板说不出话"时的
    补充，不是替换。替换的话，``write_memory`` 那张卡片会永远只写一句笼统的话，而
    "要写的是哪一条"正是用户要看的东西（``summarize`` 的 docstring 就是这条纪律）。
    """
    one = ToolCall(id="c1", name="write_memory", arguments={"content": "用户喜欢美式"})
    tool = _StatedTool("我要写一条记忆")

    assert "用户喜欢美式" in summarize(one, tool=tool)
    assert summarize(one, tool=tool) != "我要写一条记忆"


def test_the_sentence_is_not_part_of_the_tool_protocol() -> None:
    """它**不进** ``Tool`` 协议。

    那个协议是 ``@runtime_checkable`` 的，而 ``isinstance`` 对协议成员只查
    ``hasattr``——往里加一个成员，等于要求**每一个**工具都长这个属性，九个既有工具
    会当场不满足协议（``test_tools_satisfy_the_protocol`` 会红）。所以读的一侧用
    ``getattr``，读不到就回落。
    """
    assert isinstance(GetTimeTool(), Tool)
    assert not hasattr(GetTimeTool(), "approval_summary")


def test_a_decision_is_a_plain_string_so_it_can_be_recorded() -> None:
    """裁决能直接进逐字稿与账本，不需要转换表。"""
    assert str(ApprovalDecision.GRANT) == "grant"
    assert ApprovalDecision.DENY == "deny"


# ── 槽：不需要第二根线程的那几件 ────────────────────────────────────────────


def test_a_provided_decision_is_returned_by_the_next_wait() -> None:
    slot = ApprovalSlot()
    slot.request(req())

    assert slot.provide("call_1", ApprovalDecision.GRANT) is True
    assert slot.wait(timeout=1.0) is ApprovalDecision.GRANT


def test_a_deny_travels_through_unchanged() -> None:
    """拒绝必须原样到达——它是这套东西存在的理由。"""
    slot = ApprovalSlot()
    slot.request(req())
    slot.provide("call_1", ApprovalDecision.DENY)

    assert slot.wait(timeout=1.0) is ApprovalDecision.DENY


def test_an_unanswered_request_times_out_into_none_and_therefore_denies() -> None:
    """没人答 → ``None`` → 拒绝。这一条是 fail-closed 的落点。

    也断言它**真的等满了**：一个"立刻返回 None"的实现同样能让上面那句断言通过，
    而它在真机上意味着卡片还没画出来就已经作废了。
    """
    slot = ApprovalSlot()
    slot.request(req())

    start = time.monotonic()
    answer = slot.wait(timeout=0.1)
    elapsed = time.monotonic() - start

    assert answer is None
    assert resolve_decision(answer) is ApprovalDecision.DENY
    assert 0.08 <= elapsed < 5.0, f"没等满期限就返回了（{elapsed:.3f}s）"


def test_a_click_from_a_stale_card_is_refused_and_leaves_the_request_pending() -> None:
    """``call_id`` 对不上就丢弃，**且不影响当前这次待决请求**。

    这是"用户点得慢"那种时序：卡片还在屏幕上，点它的是上一轮那次调用。丢弃之后
    当前这次仍然能被正确答复——只丢弃、不顺手把当前请求也作废，否则一次误点会让
    这一轮直接失败。
    """
    slot = ApprovalSlot()
    slot.request(req("call_now"))

    assert slot.provide("call_old", ApprovalDecision.GRANT) is False
    assert slot.pending is not None and slot.pending.call_id == "call_now"
    assert slot.provide("call_now", ApprovalDecision.GRANT) is True


def test_a_second_click_on_the_same_card_cannot_flip_the_answer() -> None:
    """同一张卡片点两下，**第一次定案**。

    两次点击的值可以不同，所以第二次会被安静地写进裁决——「拒绝」之后再点一下
    「允许」，那个从来没有被批准过的动作就执行了。这是这套握手里唯一一条**能把
    拒绝翻成同意**的路，所以它必须是被堵死的。
    """
    slot = ApprovalSlot()
    slot.request(req())

    assert slot.provide("call_1", ApprovalDecision.DENY) is True
    assert slot.provide("call_1", ApprovalDecision.GRANT) is False
    assert slot.wait(timeout=1.0) is ApprovalDecision.DENY


def test_a_second_request_while_one_is_pending_is_a_loud_error() -> None:
    """同一时刻只支持一个待决请求，越界就抛。

    "两张卡片同时在等"在顺序循环里不会发生，所以它只可能是接线错了——而它安静通过
    的后果是配对错位。

    （**同一个测试里连抛两次**：先是一个已经在等的槽，再是取消之后的槽。两种都是
    接线错误，而它们的修法不同——所以断言里也分别看得见是哪一个。）
    """
    slot = ApprovalSlot()
    slot.request(req("call_1"))
    with pytest.raises(RuntimeError, match="只支持一个"):
        slot.request(req("call_2"))

    stopped = ApprovalSlot()
    stopped.cancel()
    with pytest.raises(RuntimeError, match="已经取消"):
        stopped.request(req("call_3"))


def test_the_next_request_is_allowed_once_the_first_is_answered() -> None:
    """一次生成里可能有多次要确认的调用，所以答完一个必须能接着问下一个。"""
    slot = ApprovalSlot()
    slot.request(req("call_1"))
    slot.provide("call_1", ApprovalDecision.GRANT)
    assert slot.wait(timeout=1.0) is ApprovalDecision.GRANT

    slot.request(req("call_2"))
    # 上一轮那次的裁决不能再被采纳：取走之后槽就清了
    assert slot.provide("call_1", ApprovalDecision.GRANT) is False
    assert slot.provide("call_2", ApprovalDecision.DENY) is True
    assert slot.wait(timeout=1.0) is ApprovalDecision.DENY


def test_cancel_discards_a_decision_that_was_already_provided() -> None:
    """取消连**已经攒下的允许**一起作废。

    时序是：用户点了允许，界面还没来得及送达，这一轮已经被停止拆掉了。这时执行它
    与"没答就算同意"是同一件事——都是让一个没人（在当下）同意过的动作发生。
    """
    slot = ApprovalSlot()
    slot.request(req())
    assert slot.provide("call_1", ApprovalDecision.GRANT) is True

    slot.cancel()

    assert slot.wait(timeout=1.0) is None


def test_wait_after_cancel_returns_at_once() -> None:
    """取消之后 ``wait`` 不阻塞——收尾路径依赖这一点。"""
    slot = ApprovalSlot()
    slot.cancel()

    start = time.monotonic()
    assert slot.wait(timeout=5.0) is None
    assert time.monotonic() - start < 1.0


def test_the_same_decision_is_not_replayed_to_a_second_wait() -> None:
    """取走一次就清空：第二次 ``wait`` 不会拿到同一个裁决。

    拦的是"循环被写成答完再等一次"——那会让同一个改变状态的动作执行两遍。这里的
    收场是白等到超时（用户看到模型没反应），而不是多执行一次。
    """
    slot = ApprovalSlot()
    slot.request(req())
    slot.provide("call_1", ApprovalDecision.GRANT)
    assert slot.wait(timeout=1.0) is ApprovalDecision.GRANT

    assert slot.wait(timeout=0.05) is None


def test_pending_is_visible_until_the_decision_is_taken() -> None:
    """``pending`` 的可见性是界面画卡片与判"这张卡片还有效吗"的依据。"""
    slot = ApprovalSlot()
    assert slot.pending is None

    slot.request(req("call_1"))
    assert slot.pending is not None and slot.pending.call_id == "call_1"

    slot.provide("call_1", ApprovalDecision.GRANT)
    # 答复已经攒下，但 worker 还没取走——卡片仍应显示为待决
    assert slot.pending is not None
    slot.wait(timeout=1.0)
    assert slot.pending is None


# ── 真的起线程 ─────────────────────────────────────────────────────────────


def test_waiting_from_another_thread_is_woken_by_provide() -> None:
    """worker 线程在等，界面线程答复——本版最常见的那条路径。

    ``join`` 带超时并在之后断言线程真的结束了：一个"测试自己挂了但 pytest 仍在跑"
    的死锁，如果只断言结果，会因为断言根本没被执行而变成一条**假绿**。
    """
    slot = ApprovalSlot()
    slot.request(req())
    seen: list[ApprovalDecision | None] = []

    worker = threading.Thread(target=lambda: seen.append(slot.wait(timeout=5.0)), daemon=True)
    worker.start()
    time.sleep(0.05)  # 让它真的进到 wait 里，而不是"先答后等"
    assert slot.provide("call_1", ApprovalDecision.GRANT) is True
    worker.join(timeout=5.0)

    assert not worker.is_alive(), "worker 没有被唤醒——这条路径就是死锁本身"
    assert seen == [ApprovalDecision.GRANT]


def test_cancel_wakes_a_waiting_worker_without_waiting_for_the_timeout() -> None:
    """取消要**立刻**解阻塞。

    这是 ``wait_for_all`` 那 3 秒预算能成立的前提：worker 若卡在等确认上，收尾会
    超时后继续退出，进程撞上 ``0xC0000409``（Windows fast-fail，无回溯、无 stderr）。
    所以这里的 timeout 刻意设成 30 秒——**靠超时返回的话这条测试会挂**，而不是
    慢一点通过。
    """
    slot = ApprovalSlot()
    slot.request(req())
    seen: list[ApprovalDecision | None] = []

    worker = threading.Thread(target=lambda: seen.append(slot.wait(timeout=30.0)), daemon=True)
    start = time.monotonic()
    worker.start()
    time.sleep(0.05)
    slot.cancel()
    worker.join(timeout=5.0)
    elapsed = time.monotonic() - start

    assert not worker.is_alive()
    assert seen == [None]
    assert elapsed < 1.0, f"取消没能立刻解阻塞（{elapsed:.3f}s）"


def test_a_hundred_round_trips_finish_inside_the_budget() -> None:
    """一百轮"登记 → 答复 → 取走"，在真线程上跑完。

    一次往返绿不代表一百次绿：丢唤醒、把事件 clear 错位置、锁的粒度写错，这些毛病
    往往要撞上恰好那个交错的时序才现形。这里还顺带断言**结果逐轮对应**——卡住不
    会，但"答了第 3 轮、被第 4 轮取走"这种错位只有逐轮比对才看得见。
    """
    slot = ApprovalSlot()
    rounds = 100
    seen: list[ApprovalDecision | None] = []

    def work() -> None:
        for index in range(rounds):
            slot.request(req(f"call_{index}"))
            seen.append(slot.wait(timeout=10.0))

    worker = threading.Thread(target=work, daemon=True)
    start = time.monotonic()
    worker.start()
    for index in range(rounds):
        expected = f"call_{index}"
        # **有界**等待，不是 ``while True``：无限忙等会把"槽坏了"变成"整个测试进程挂住"，
        # 而在 CI 上挂住比失败坏得多——它只报超时，不说卡在哪一条、第几轮。这条纪律是
        # 注入验牙时现学的：一次真挂住让我去查了十几分钟的孤儿进程而不是去看断言。
        #
        # **它管的是"待决请求不出现"这一类**（登记丢了、轮次错位）。worker 持着锁不放
        # 那一类它管不着——主线程会卡在 ``slot.pending`` 里面，连这个期限都读不到；
        # 那种由下面的 ``join`` + ``is_alive`` + ``elapsed`` 三条收。
        deadline = time.monotonic() + 10.0
        while True:
            pending = slot.pending
            if pending is not None and pending.call_id == expected:
                break
            if time.monotonic() > deadline:
                seen_id = pending.call_id if pending is not None else "<没有待决请求>"
                pytest.fail(
                    f"第 {index} 轮等不到 {expected!r} 的待决请求"
                    f"（当前 pending={seen_id}，worker 还活着={worker.is_alive()}）"
                )
            time.sleep(0.001)
        assert slot.provide(expected, ApprovalDecision.GRANT) is True
    worker.join(timeout=15.0)
    elapsed = time.monotonic() - start

    assert not worker.is_alive()
    assert seen == [ApprovalDecision.GRANT] * rounds
    assert elapsed < 10.0
