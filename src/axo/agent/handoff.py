"""跨线程的**一次交接**：worker 线程问、界面线程答。

抽出来是因为它有两个用户了：确认闸门（``gate.ApprovalSlot``）与截图
（``handoff.CaptureSlot``）。第二份方法体正是这个仓点名要防的那件事——
``ui/thread_tracking.py`` 的开头就写着"复制一份那个方法体是这里最容易复发的一件事"，
而下面 :meth:`HandoffSlot.wait` 里那点东西（有界轮询 + 唤醒事件 + 单向取消 + 身份比对）
每一处都是实测换来的。

## 为什么是槽不是队列

**同一时刻最多一个待决请求。** 两边的调用方都是顺序的：确认是"上一个工具的结果回灌完
才可能有下一个"，截图是"一次工具往返只截一张"。写成队列的唯一效果是给这个不存在的状态
留出空间——而它一旦真的出现，后果是**配对错位**（用户点的那一下、或者那一张图，被算到
下一次请求头上）。这条写在这里是为了拦住"顺手改成队列"这个看起来很自然的下一步。

## 身份：两个方向都要带

:meth:`request` 收一个 ``token``，:meth:`provide` 必须拿同一个 token 来对。它挡的是
**迟到**：界面那一侧动作慢（一张 4K 截图的编码可能几十毫秒到几百毫秒），而 worker 那边
已经超时放弃、甚至已经在问下一件事了。没有这道比对，那一张**三秒前**的屏幕会被当成
"刚刚看到的"——而模型会照着它回答。

## 单向取消

:meth:`cancel` 是**闩**，不是复位：翻过去之后这个槽永久没有待决请求，:meth:`request`
直接抛。取消意味着这一轮正在被拆掉，而"拆掉之后又有东西来问"只可能是接线错了——那种错
如果被安静地容忍，表现是"点了停止之后又弹出一张卡片"。**所以一个槽只服务一次生成**：
要开下一轮，就造一个新的。

## 线程约定

``request`` / ``wait`` 在 worker 线程；``provide`` / ``cancel`` 在界面线程
（``cancel`` 也可能来自收尾路径）。状态由一把 :class:`threading.Lock` 护住，另有一个
:class:`threading.Event` 只用来"叫醒"——**它不承载状态**，所以即使某次改动忘了 set，
:meth:`wait` 的有界轮询也会在一个 ``poll`` 之后重新看一遍状态，而不是永远睡下去。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Generic, TypeVar

from .tools import Capture

__all__ = ["CaptureSlot", "HandoffSlot", "new_token"]

Ask = TypeVar("Ask")
Reply = TypeVar("Reply")


def new_token() -> str:
    """一次交接的身份。**由问的那一侧生成**（见模块文档"身份"那一段）。

    用新 id 而不是序号：序号在"同一个槽被复用于两次请求"时看不出差别，而那种复用正是
    这条机制要拦的那件事。
    """
    return uuid.uuid4().hex[:12]


class HandoffSlot(Generic[Ask, Reply]):
    """见模块文档。泛型参数是 ``(问的东西, 答的东西)``。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: tuple[str, Ask] | None = None
        self._reply: Reply | None = None
        self._cancelled = False

    # ── worker 线程 ─────────────────────────────────────────────────────────

    def request(self, ask: Ask, *, token: str) -> None:
        """登记一个待决请求。**worker 线程调用**。

        已经在等一个、或者这一轮已被取消时抛 :class:`RuntimeError`：两种都是接线错误，
        而它们安静通过的表现分别是"两个待决请求"和"停止之后又来一次"。
        """
        with self._lock:
            if self._cancelled:
                raise RuntimeError("这一轮已经取消，不能再发起请求（一个槽只服务一次生成）")
            if self._pending is not None:
                raise RuntimeError(
                    f"已经有一个待决请求 {self._pending[0]!r}；这个槽同一时刻只支持一个"
                )
            self._pending = (token, ask)
            self._reply = None

    def wait(self, *, timeout: float | None = None, poll: float = 0.05) -> Reply | None:
        """等一个回答。**worker 线程调用**。超时与取消都返回 ``None``。

        **取走一次就清空**：同一个槽 ``wait`` 第二次会一直阻塞到超时。这是 fail-closed
        的方向——万一调用方被写成"答完再等一次"，代价是这一轮白等，而不是同一个动作被
        执行两遍。

        **超时也算这一次交接结束了**（把待决清掉，见下面 :meth:`_give_up`）。

        ``poll`` 是有界轮询的步长：即使唤醒事件没被 set（将来的某次改动），最长一个
        ``poll`` 之后也会重新看一遍状态与期限。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._wake.clear()
                if self._reply is not None:
                    reply = self._reply
                    self._reply = None
                    self._pending = None
                    return reply
                if self._cancelled:
                    return None
            if deadline is None:
                remaining = poll
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._give_up()
                    return None
            self._wake.wait(min(poll, remaining))

    def _give_up(self) -> None:
        """等不到了：**把待决清掉**，这一次交接到此为止。

        两件事都由这一句保证，而且它们指向同一个结果：

        * **下一次 :meth:`request` 还能进来**——一轮里可能截两次图（两次工具往返），
          而超时留着一个"永远待决"的请求会让第二次直接抛；
        * **那个迟到的回答会被丢掉**——它回来时已经没有待决请求了（``provide`` 一看
          ``pending is None`` 就返回假），于是"三秒前那张屏幕"不会落到下一次头上。

        第一件事与第二件事本来可以各写一半（超时清 pending、比对只管 token），但那样
        两条路径就有了两个说法；这里让"等不到"成为一个**明确的收场**。
        """
        with self._lock:
            self._pending = None
            self._reply = None

    # ── 界面线程 ────────────────────────────────────────────────────────────

    @property
    def pending(self) -> Ask | None:
        """当前待决的那个"问"，没有则 ``None``。给界面与测试看。"""
        with self._lock:
            return None if self._pending is None else self._pending[1]

    @property
    def token(self) -> str | None:
        """当前待决请求的身份。界面侧回话时原样带回来。"""
        with self._lock:
            return None if self._pending is None else self._pending[0]

    def provide(self, token: str, reply: Reply) -> bool:
        """界面线程回传一个回答。返回**它是否被采纳**。

        ``token`` 必须与当前待决的那一次对得上，否则丢弃并返回 ``False``（见模块文档
        "身份"）。**已经答过的那一个不再接受第二次**：两次回答的值可以不同，于是
        「拒绝」之后再点一下「允许」会把裁决**翻**成同意——而那正是一个从来没有被批准
        过的动作。方向必须是单向的：第一次定案。
        """
        with self._lock:
            if self._cancelled or self._pending is None:
                return False
            if self._pending[0] != token:
                return False
            if self._reply is not None:
                return False  # 已经答过了，第一次定案
            self._reply = reply
            self._wake.set()
            return True

    def cancel(self) -> None:
        """停止 / 收尾：立刻唤醒还卡在 :meth:`wait` 里的 worker。

        **连已经攒下的回答一起清掉**：这一轮正在被拆掉，那它就不该再执行任何动作。
        fail-closed 的方向与超时一致。
        """
        with self._lock:
            self._cancelled = True
            self._pending = None
            self._reply = None
            self._wake.set()


class CaptureSlot(HandoffSlot[None, Capture]):
    """**截图那一次交接**：worker 线程问"现在屏幕上是什么"，界面线程回一张图。

    "问"的东西是 ``None``——它没有参数（不给它选窗口、选区域的能力，见
    ``catalog.LookAtScreenTool``）。身份由 :meth:`ask` 生成，问的那一侧把 token 一路
    带到信号上，界面侧原样带回来。
    """

    def ask(self) -> str:
        """登记一次截图请求，返回它的身份。**worker 线程调用**。

        包一层是为了让调用方不必自己造 token、也不会忘记带——两边用的是同一个字符串，
        而那正是"回话对不上就被丢掉"这条机制的入口。
        """
        token = new_token()
        self.request(None, token=token)
        return token
