"""跨线程那一次交接：**谁在等、谁在答、答错了算不算**。

这一层有两个用户（确认闸门与截图），所以它单独有测试：``tests/test_agent_gate.py``
里那一组测的是"确认"这条路上的语义（第一次点击定案、拒绝原样到达），这里测的是**机制
本身**——尤其那条只在截图这一侧才真正危险的规则：**迟到的回答必须被丢掉**。

为什么它值得单独一个文件：那一行比对漏掉的症状，两边完全不同，而且都不报错。

* 确认那侧漏了 → 一次**张冠李戴的允许**（用户给 A 点了头，算到了 B 头上）；
* 截图那侧漏了 → 一张**三秒前**的屏幕被当成"刚刚看到的"，模型照着它回答。

两种都是"看起来在工作"。
"""

from __future__ import annotations

import threading
import time

import pytest

from axo.agent.handoff import CaptureSlot, HandoffSlot, new_token
from axo.agent.tools import Capture

SHOT = Capture(png=b"PNG", width=800, height=600)


def test_a_reply_with_the_wrong_token_is_refused() -> None:
    """身份对不上就丢掉。**这是这一层唯一一条不能省的比对。**

    它挡的不是"用户点得快"，而是"界面那一侧做得慢"：一张大截图的编码、或者用户盯着
    卡片犹豫的那几秒，都可能让回答落在 worker 已经放弃之后（甚至已经在问下一件事之后）。
    """
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.request(None, token="A")

    assert slot.provide("B", SHOT) is False
    assert slot.provide("A", SHOT) is True


def test_a_late_reply_never_lands_on_the_next_request() -> None:
    """**迟到的那一个不许被下一次吃掉。** 这条是上一个测试的真实版本。

    第一次请求超时走了、第二次请求已经发出，此时第一次的回答才回来——没有身份的话，
    那一张"三秒前的屏幕"就是第二次看到的屏幕，而模型会照着它回答，谁也看不出差别。
    """
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.request(None, token="first")
    assert slot.wait(timeout=0.01) is None, "第一次应当超时"

    slot.request(None, token="second")

    assert slot.provide("first", SHOT) is False, "迟到的那一个被采纳了"
    assert slot.token == "second"
    assert slot.provide("second", Capture(png=b"NEW")) is True


def test_only_the_first_reply_counts() -> None:
    """已经答过的那一个不再接受第二次回答。

    值可以不同（拒绝之后再点一下允许），所以第二次必须被拒——否则一个从来没有被批准
    过的动作会因为"后来手滑点了另一个按钮"而变成同意。
    """
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.request(None, token="A")

    assert slot.provide("A", SHOT) is True
    assert slot.provide("A", Capture(png=b"OTHER")) is False


def test_the_capture_slot_mints_its_own_token() -> None:
    """截图那一侧的问是"没有参数"的，所以身份由 :meth:`CaptureSlot.ask` 自己发。

    包这一层是为了让调用方既不会忘记带 token、也不会两边各造一个——那正是"回话对不上
    就被丢掉"这条机制的入口。
    """
    slot = CaptureSlot()

    token = slot.ask()

    assert token
    assert slot.token == token
    assert slot.pending is None, "截图没有参数，问的东西就是 None"
    assert slot.provide(token, SHOT) is True
    assert slot.wait(timeout=1.0) == SHOT


def test_tokens_are_not_reused() -> None:
    """两次交接的身份不同。用序号的话，"同一个槽被复用于两次请求"看起来会一模一样，
    而那种复用正是这条机制要拦的那件事。"""
    assert new_token() != new_token()


def test_cancel_is_a_one_way_latch() -> None:
    """取消之后**永久**没有待决请求，再问就抛。

    安静地容忍它的表现是"点了停止之后又弹出一张卡片"——所以这一条是抛，不是返回。
    """
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.cancel()

    with pytest.raises(RuntimeError):
        slot.request(None, token="A")


def test_a_second_pending_request_is_refused() -> None:
    """同一时刻只支持一个待决请求（写在模块文档里的那条）。两个的后果是配对错位。"""
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.request(None, token="A")

    with pytest.raises(RuntimeError):
        slot.request(None, token="B")


def test_wait_is_bounded_and_needs_no_second_thread() -> None:
    """等不到就按超时回来，**不挂住**。

    这一条钉的是"下限"：收尾只给 3 秒，而任何一次新的等待都必须在这个预算里能结束。
    （真线程那一侧由 ``tests/test_agent_gate.py`` 的那条守着，它现在跑的是同一段代码。）
    """
    slot: HandoffSlot[None, Capture] = HandoffSlot()
    slot.request(None, token="A")

    started = time.monotonic()

    assert slot.wait(timeout=0.05) is None
    assert time.monotonic() - started < 2.0


def test_a_reply_crosses_threads() -> None:
    """真起一根线程：worker 侧等，界面侧答。**这条是那条交接的全部意义。**"""
    slot = CaptureSlot()
    got: list[Capture | None] = []

    def worker() -> None:
        token = slot.ask()
        assert token
        got.append(slot.wait(timeout=5.0))

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 5.0
    while slot.token is None and time.monotonic() < deadline:
        time.sleep(0.01)  # 等 worker 把请求登记进来

    assert slot.provide(slot.token or "", SHOT) is True
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert got == [SHOT]
