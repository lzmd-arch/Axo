"""「一对新起的 ``(线程, worker)``」的接管与收尾——**这套形状是实测换来的疫苗**。

抽成一个模块，是因为**复制一份那个方法体是这里最容易复发的一件事**：两处各写一遍
``started.connect``，漏掉的那个 ``Qt.DirectConnection`` 的症状是"界面在每次网络请求
期间冻住"，看起来像 Ollama 慢；漏掉成员引用的症状是线程 ``isRunning()`` 为真、不报错、
回调永不触发，看起来像网络卡住。

所以规则只有一条：**起后台 worker 的地方一律走 :func:`track`**，不要自己写
``started.connect``。

诊断史（实测，不是推演）：worker 若用 ``moveToThread`` 搬进工作线程，那么**在界面
线程里丢掉它的最后一个 Python 引用**就等于在错误的线程里析构一个 C++ 对象，Qt
直接 ``abort()``；装了 faulthandler 时退出码是 ``3``，没装时是 ``0xC0000409``
（Windows fast-fail），没有回溯、没有 stderr。``_smoke_recall.py`` 的 remember 相位
在这个形状下 2/6 崩，改成"亲和面留在界面线程 + ``started`` 直连"之后 0/8。

:func:`wait_for_all` 同样是从 :meth:`MainWindow._shutdown` 里抽出来的——记忆窗口的
``closeEvent`` 要等的是同一批东西，理由也一字不差。
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from PySide6.QtCore import QObject, Qt, QThread

__all__ = ["THREAD_WAIT_MS", "track", "wait_for_all"]

THREAD_WAIT_MS = 3000
"""收尾时等一个线程停下来的上限（毫秒）。超时也继续退出——宁可能丢一次落盘，
也不要一个关不掉的窗口。"""


def track(receiver: QObject, thread: QThread, worker: QObject, *, what: str, slot: str) -> None:
    """接管一对新起的 ``(线程, worker)``：接上 ``run``、存引用、记账、挂收尾。

    **必须留引用。** ``started`` 连的是 worker 的绑定方法，而 PySide6 的绑定方法
    连接**不持有接收者**：worker 若只是局部变量，方法一返回就被回收，``started``
    便投递到一个已死的对象上。症状极具迷惑性：线程 ``isRunning()`` 为真、不报
    任何错、回调永不触发——看起来像"网络卡住了"，实则是对象没了。``slot`` 是
    ``"model"`` 这样的短名，对应的两个成员名（``_model_thread`` /
    ``_model_worker``）由它拼出来，调用方因此不必自己再存一遍——**存两遍就会
    漏一处**。

    **worker 不做 ``moveToThread``，它的所属线程自始至终是界面线程。** 这是实测
    换来的，不是口味：``run()`` 在工作线程里跑（靠那个 ``DirectConnection``），
    但对象的 Qt 亲和面留在界面线程，于是"界面线程丢掉 worker 的最后一个 Python
    引用"就成了一次**同线程**析构——合法。反过来写（``moveToThread`` 之后再丢
    引用）会让 PySide6 在**错误的线程**里同步析构那个 C++ 对象，Qt 直接
    ``abort()``（细节见模块文档）。

    改法本身**行为等价**：``moveToThread`` 时接收者与发射者同处一线程，
    ``AutoConnection`` 也解析成直连。区别只在于把"直连"从亲和面的副产物变成了
    显式约定，所以调用点不必各写一遍 ``started.connect(worker.run)``——写迟了
    （漏了一个 ``Qt.DirectConnection``）症状是界面在每次网络请求期间冻住。
    ``worker`` 必须有一个 ``run()``。

    Args:
        receiver: 持有这批线程的对象。**必须有一个 ``_live_threads`` 列表**，两个
            成员名也挂在它上面。``MainWindow`` 与 ``MemoryWindow`` 都满足。
        thread: 尚未 ``start()`` 的线程。
        worker: 有 ``run()`` 槽的 ``QObject``，亲和面必须已在 *receiver* 的线程里
            （即"在界面线程里构造出来、没有 ``moveToThread``"）。
        what: 线程名，进 ``setObjectName``，收尾超时时用来说清是谁没停下来。
        slot: 短名，如 ``"model"``、``"probe"``。拼出 ``_<slot>_thread`` 与
            ``_<slot>_worker`` 两个成员。

    ``finished`` 上挂的是**闭包**，不是绑定方法槽。这一点是实测换来的：原先它用
    ``sender()`` 认领是哪个线程，那套在别处能用（见 ``_on_assembly_ready``），但
    在这里漏过一次——一个已经 ``deleteLater`` 掉的 ``QThread`` 留在清单里，下一次
    碰它就是 ``RuntimeError``，而那个异常抛在槽里会被 PySide6 吞掉。闭包把"是
    哪一个"绑死在创建处，不需要任何比对；清单里的比较也一律用 ``is``，绝不用
    ``in``／``==``——那两个会去碰可能已经没了的 C++ 对象。
    """
    thread_attr = f"_{slot}_thread"
    worker_attr = f"_{slot}_worker"
    setattr(receiver, thread_attr, thread)
    setattr(receiver, worker_attr, worker)
    thread.setObjectName(what)
    receiver._live_threads.append(thread)  # noqa: SLF001 — 见 Args：这是本函数的契约

    # ``DirectConnection`` 是这条设计的支点，不是可选的美化：worker 的亲和面在
    # 界面线程，``AutoConnection`` 会因此把 ``run()`` **排队回界面线程**执行。
    thread.started.connect(worker.run, Qt.DirectConnection)

    def release() -> None:
        receiver._live_threads = [  # noqa: SLF001 — 重新赋值而不是原地删，见下
            live
            for live in receiver._live_threads
            if live is not thread  # noqa: SLF001
        ]
        if getattr(receiver, thread_attr) is thread:
            setattr(receiver, thread_attr, None)
            setattr(receiver, worker_attr, None)

    # 顺序有讲究，三条都不能换：
    # 1. ``worker.deleteLater`` 排在最前，让删除先进入界面线程的队列；
    # 2. ``release`` 紧随其后——它那次 ``setattr`` 可能立刻把引用计数打到零，
    #    此时 worker 是一个界面线程对象，就地析构合法，且 ``~QObject`` 会把还在
    #    排队的 DeferredDelete 一并撤掉，两条路都不留悬挂；
    # 3. ``thread.deleteLater`` 垫底——释放 C++ 对象之后，上面那个闭包里的
    #    ``is`` 比较就再也用不上那个线程了。
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(release)
    thread.finished.connect(thread.deleteLater)


def wait_for_all(threads: Iterable[QThread]) -> list[QThread]:
    """让每个还在跑的线程停下来**并且等它停**，返回没能在超时内停下的那些。

    **为什么必须等，而不是交给解释器收尾。** 一个仍在运行的 ``QThread`` 在析构时
    会让 Qt 调 ``qFatal``（Windows 上的表现是退出码 ``0xC0000409``，fast-fail），
    没有回溯、没有 stderr、Python 的 ``atexit`` 都不跑——看起来像 QtWebEngine 崩了，
    实际只是有个线程还活着。冒烟脚本一直带着这个退出码，只是此前没人看 ``$?``。

    **按"在跑的清单"逐个等，不按那些成员变量**：取消时 ``_reset_generation_state``
    会把成员提前清掉，而线程还在跑（装配是这样，故意的）。按成员等就会漏掉它，
    漏掉的那个正是下一次 ``0xC0000409`` 的来源。

    ``threads`` 应当是一份**快照**（调用方传 ``list(self._live_threads)``）：这个
    循环期间 ``release`` 闭包会把清单换成一个新列表。

    超时也继续退出（沿用既有决定：不做线程强杀，杀线程会留下半关闭的连接），但
    **要说出来**——那条路径上没有任何别的线索，留一行字，下一个看到的人不必再花
    一整轮去二分。
    """
    stragglers: list[QThread] = []
    for thread in threads:
        # 循环体整个包在 try 里。清单上是 Python 包装器，而 ``deleteLater`` 删除的
        # 是 C++ 对象——两者能对不上（释放那个闭包是排队送来的，未必先到）。
        # 在这条路径上抛异常的症状是**窗口关不掉**（异常从 ``closeEvent`` 里冲出
        # 去，关闭被中止），而跳过一个已经消失的线程没有任何代价。
        try:
            if not thread.isRunning():
                continue
            # 直连调用而不是靠信号：``quit()`` 是线程安全的，而从信号走要等 GUI 的
            # 事件循环派发——收尾这条路上，事件循环**已经在退出**，排队的东西
            # 未必还能被处理，那正是这个 bug 的成因。
            thread.quit()
            stopped = thread.wait(THREAD_WAIT_MS)
        except RuntimeError as exc:
            print(f"收尾：跳过一个已经消失的线程（{exc}）", file=sys.stderr)
            continue
        if not stopped:
            stragglers.append(thread)
            print(
                f"收尾超时：{thread.objectName() or '后台线程'} 仍在运行，"
                f"进程可能以非零退出码结束。",
                file=sys.stderr,
            )
    return stragglers
