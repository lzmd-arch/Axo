"""端到端冒烟：走真实的 ``_on_submit`` 路径，从提交一路打到会话落盘。

与另外两个冒烟脚本的分工：

* ``smoke_backend.py``  只验后端（Python → Ollama），不含 Qt
* ``_smoke_webview.py`` 只验渲染（Python → JS），不含后端
* 这个脚本验中间那段：**真窗口、真线程、真模型、真落盘**

**网页侧用替身，不用真的 QWebEngineView。** 这不是图省事，是实测出来的：在本机
离屏模式下，第一次 ``runJavaScript``（流式的第一个增量）就会让整个进程被硬杀——
没有异常、没有退出码、``atexit`` 都不执行，日志停在半句话上。把网页留在链路里，
这个脚本就会变成"有时能跑有时不能跑"，而它恰恰是用来回答"到底哪一段坏了"的。

所以两边各自只负责一件事：这个脚本断言 Python 侧的承诺——提交进得去、增量逐片
推得出来、回复落进会话与磁盘、落盘内容与内存一致；网页侧由
``_smoke_webview.py`` 单独断言（它不碰后端）。中间用替身的好处还有一层：替身记
的是**调用序列**，比"页面上出现了气泡"更能定位问题出在哪一步调用上。替身本身在
``tools_support.py`` 里，与 ``_smoke_recall.py`` 共用。

它替不掉"人双击一次"的验收——布局丑不丑、字体对不对，只能靠看。

用临时数据目录，不碰用户的真实会话。

用法::

    uv run --extra app --extra llm python tools/_smoke_e2e.py
    QT_QPA_PLATFORM=offscreen uv run --extra app --extra llm python tools/_smoke_e2e.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# 必须在 import axo 之前设好——paths 模块在首次调用时解析并缓存目录。
_TMP = Path(tempfile.mkdtemp(prefix="memterm-smoke-"))
os.environ["AXO_DATA"] = str(_TMP)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from tools_support import RecordingChatView  # noqa: E402

import axo.ui.main_window as main_window_module  # noqa: E402

# 提示词要能让模型正经生成几十个字，而不是回两个字就停。
#
# 原因是量出来的：``只回答两个字：收到`` 这种答案本体只有两个 token，Ollama 一片就
# 发完了，"增量片数"因此是 1，看起来像流式坏了。换成长一点的回答，同一条链路会
# 稳定给出 20 片以上。**片数取决于网络与模型的发送时机，不该拿它当断言**——它只在
# "至少有个增量被推出来"这一件事上是有意义的（见 report）。
PROMPT = "用三句话介绍一下脉冲神经网络，每句话不超过二十字。"
WAIT_FOR_MODELS_MS = 2500
DEADLINE_S = 180.0
POLL_INTERVAL_MS = 2000


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    # 换掉真视图**必须在构造窗口之前**——真件一被构造出来就会起 QtWebEngine，
    # 而离屏下它一渲染就把进程带走。
    main_window_module.ChatView = RecordingChatView  # type: ignore[misc]
    window = main_window_module.MainWindow()
    fake: RecordingChatView = window.chat_view  # type: ignore[assignment]
    print(f"数据目录：{_TMP}", flush=True)

    beats = 0

    def start() -> None:
        if window._backend is None:
            print("后端未构造，无法发送。先确认 Ollama 在跑且有对话模型。", flush=True)
            app.exit(2)
            return
        print(f"用模型：{window._backend.model_id}", flush=True)
        started_at = time.monotonic()
        window._on_submit(PROMPT)
        QTimer.singleShot(0, lambda: poll(started_at))

    def poll(started_at: float) -> None:
        """等回复落地。

        判据是会话里出现了助手消息——那是 ``_on_reply_finished`` 干的事，也就是
        "整条链路走完"的唯一标志。轮询而不是再挂一个 signal，是因为那个 signal
        已经被 ``_on_reply_finished`` 接走了，重挂只是把同一件事说两遍。
        """
        nonlocal beats
        beats += 1
        if beats == 1 or beats % 5 == 0:
            elapsed = time.monotonic() - started_at
            print(
                f"  …[{elapsed:6.1f}s] 增量={len(fake.streamed)} "
                f"消息={[m.role for m in window._session.messages]} "
                f"streaming={window._streaming}",
                flush=True,
            )
        if any(message.role == "assistant" for message in window._session.messages):
            report()
            return
        if time.monotonic() - started_at > DEADLINE_S:
            print(f"超时：等了 {DEADLINE_S:.0f} 秒还没回复。会话实际内容：", flush=True)
            for message in window._session.messages:
                print(f"    {message.role}: {message.content[:60]!r}", flush=True)
            print(f"    _streaming={window._streaming} 增量={len(fake.streamed)}", flush=True)
            app.exit(3)
            return
        QTimer.singleShot(POLL_INTERVAL_MS, lambda: poll(started_at))

    def report() -> None:
        assistant = next(m for m in window._session.messages if m.role == "assistant")
        print("--- 回复 ---")
        print(assistant.content.strip())
        print("--- 自检 ---")
        print(f"流式增量片数：{len(fake.streamed)}（仅供参考，不参与判定；见 PROMPT 处的说明）")
        print(f"输入 token：{assistant.prompt_tokens}  输出 token：{assistant.completion_tokens}")
        print(f"视图调用序列：{' → '.join(fake.calls) or '（无）'}")

        # 复用窗口自己的 store：它读的就是真目录（此处已被 AXO_DATA
        # 指到临时目录），再新建一个只会多一个可能不一致的路径来源。
        store = window._store
        on_disk = store.load(window._session.session_id)
        print(f"内存消息数：{len(window._session.messages)}  落盘读回：{len(on_disk.messages)}")
        print(f"磁盘上的会话数：{len(store.list_sessions())}")

        problems = []
        # 判定的落点是"字是**推**出来的，不是整段塞进去的"，而不是片数。
        # 片数由模型与网络决定，见 PROMPT 处的说明。
        if not fake.streamed:
            problems.append("没有任何 pushToken，回复不是流式推出来的")
        if len(on_disk.messages) != len(window._session.messages):
            problems.append("落盘消息数与内存不一致")
        if not assistant.content.strip():
            problems.append("回复为空")
        if "appendMessage(assistant)" in fake.calls:
            # 助手的字必须是**推**出来的，不能是整段追加的——否则"流式"是假的。
            problems.append("助手消息走了 appendMessage 而不是流式 pushToken")
        if "endMessage" not in fake.calls:
            problems.append("回复结束后没有 endMessage，网页的光标不会停")

        print("结论：" + ("；".join(problems) if problems else "全部通过"))
        app.exit(1 if problems else 0)

    # 等模型清单回来（走后台线程），再发送。
    QTimer.singleShot(WAIT_FOR_MODELS_MS, start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
