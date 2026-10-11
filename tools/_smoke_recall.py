"""P1 验收：说一条事实 → 重启 → 问它 → 命中，**账本能逐条说明为什么**。

这是计划书 §3.4 那条验收的可执行版本。做成脚本而不是靠人手点，是因为它横跨
**两个进程**：记忆库、向量空间、会话逐字稿都是文件，"重启之后还在"这件事只能由
"第二个进程读到了第一个进程写的东西"来证明——同一进程里重新 ``open`` 一次不算，
那证明的是连接没关干净。

两段分别跑::

    python tools/_smoke_recall.py remember --dir D:\\tmp\\memterm-p1
    python tools/_smoke_recall.py recall   --dir D:\\tmp\\memterm-p1

不给 ``--dir`` 时自己分配一个临时目录**并把两段连起来一次跑完**（中间照样是两个
进程，否则这次的"重启"就是假的）：父进程 fork 出子进程，各跑一段。

网页侧仍用替身（见 ``_smoke_e2e.py`` 的说明：离屏下 QtWebEngine 会被硬杀），
所以这里断言的是 Python 侧的承诺——记忆写进去了、向量补上了、召回命中了、账本
说出了原因。``--dir`` 上的文件（``memory.db``、``sessions/``）**留着不删**：
验收的一半是"能拿眼睛去看"。

用法::

    uv run --extra app --extra llm python tools/_smoke_recall.py
    QT_QPA_PLATFORM=offscreen uv run --extra app --extra llm python tools/_smoke_recall.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ``tools/`` 不是包，所以共用件按脚本目录找。放在 import 之前：``_run_turn`` 里的
# ``from tools_support import ...`` 依赖这一行，而它要能被子进程原样复用。
sys.path.insert(0, str(Path(__file__).resolve().parent))

FACT = "我家的路由器放在书房书架顶层，旁边还有一个备用的交换机。"
QUESTION = "我家的路由器放在哪儿来着？"

WAIT_FOR_MODELS_MS = 2500
DEADLINE_S = 180.0
POLL_INTERVAL_MS = 1000


def _prepare_env(data_dir: str) -> None:
    """数据目录必须在 import biosnn_memterm 之前设好——paths 首次调用时缓存它。"""
    os.environ["BIOSNN_MEMTERM_DATA"] = data_dir
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── 两段共用的一段：起窗口、提交一句话、等回复落地 ─────────────────────────────


def _run_turn(*, wait_for_embedding: bool):
    """构造窗口、提交、等助手消息。返回 ``(window, view, prompt)``。

    返回 window 而不是就地断言：两段各自要检查的东西不同，而"等回复落地"这段
    逻辑必须一模一样——两边各写一遍的话，第二段的等待写松了就会把"召回没命中"
    误报成"回复太慢"。
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from tools_support import RecordingChatView

    import biosnn_memterm.ui.main_window as main_window_module

    app = QApplication.instance() or QApplication(sys.argv)
    main_window_module.ChatView = RecordingChatView  # type: ignore[misc]
    window = main_window_module.MainWindow()
    view = window.chat_view

    state: dict[str, object] = {"done": False}

    def submit_when_ready(started: float) -> None:
        # 语义通道要等一次 ``/api/embed`` 探测 + 一次回填。不等它就提交的话，这一轮
        # 只有词法通道——**那也能命中**，于是脚本会在语义其实坏掉的情况下报"通过"。
        if wait_for_embedding and window._embedding is None and time.monotonic() - started < 60:
            QTimer.singleShot(POLL_INTERVAL_MS, lambda: submit_when_ready(started))
            return
        print(f"    语义通道：{window._embedding_note or '（无说明）'}", flush=True)
        print(f"    库里已有向量：{_vector_count(window)} 条", flush=True)
        window._on_submit(prompt)
        QTimer.singleShot(0, lambda: poll(started))

    def poll(started: float) -> None:
        if any(message.role == "assistant" for message in window._session.messages):
            state["done"] = True
            app.quit()
            return
        if time.monotonic() - started > DEADLINE_S:
            print(f"    超时：等了 {DEADLINE_S:.0f} 秒还没回复", flush=True)
            print(f"    _streaming={window._streaming} 增量={len(view.streamed)}", flush=True)
            app.quit()
            return
        QTimer.singleShot(POLL_INTERVAL_MS, lambda: poll(started))

    prompt = FACT if wait_for_embedding is False else QUESTION
    QTimer.singleShot(WAIT_FOR_MODELS_MS, lambda: submit_when_ready(time.monotonic()))
    app.exec()
    if not state["done"]:
        raise SystemExit("这一轮没有拿到回复，后面的断言没有意义。")
    return window, view, prompt


def _vector_count(window) -> int:
    spaces = window._memory.spaces()
    return sum(window._memory.count_embeddings(space.space_id) for space in spaces)


# ── 第一段：记住 ────────────────────────────────────────────────────────────


def phase_remember() -> int:
    print(f"[1] 说一条事实：{FACT}", flush=True)
    window, _, _ = _run_turn(wait_for_embedding=False)

    memories = [item for item in window._memory.load_all() if not item.pinned]
    print(f"[2] 记忆库里有 {len(memories)} 条非钉住条目：", flush=True)
    for item in memories:
        print(f"      · [{item.kind}] {item.content}", flush=True)
    print(
        f"[3] 向量：{_vector_count(window)} 条  ·  空间：{[s.space_id for s in window._memory.spaces()]}"
    )

    problems = []
    if not any(FACT[:12] in item.content for item in memories):
        problems.append("刚才那句话没进记忆库")
    if _vector_count(window) == 0:
        # 不是致命错误——没有嵌入模型时词法通道照常工作。但这条验收要验的正是
        # 语义通道，所以它必须在这里就是非 0。
        problems.append("一条向量都没有，语义通道没接上")

    print("结论：" + ("；".join(problems) if problems else "记住了"))
    return 1 if problems else 0


# ── 第二段：想起来 ──────────────────────────────────────────────────────────


def phase_recall() -> int:
    print(f"[1] 换一个进程，问同一件事：{QUESTION}", flush=True)
    window, _, _ = _run_turn(wait_for_embedding=True)

    injected = window._assembly.ledger.injected if window._assembly else ()
    print("[2] 这一轮注入了什么（装配账本）：", flush=True)
    print(window._assembly.ledger.render() if window._assembly else "（没有装配结果）", flush=True)
    if window._recall_ledger is not None:
        from biosnn_memterm.ui.memory_worker import render_recall_detail

        print("[3] 召回过程（谁被淘汰、为什么）：", flush=True)
        print(render_recall_detail(window._recall_ledger), flush=True)

    problems = []
    hit = [item for item in injected if "路由器" in item.content]
    if not hit:
        problems.append("那条事实没被召回")
    else:
        explain = hit[0].explain
        print(f"[4] 命中那条的归因：{explain}", flush=True)
        if "语义" not in explain:
            problems.append("命中了但语义通道没有贡献（只有词法通道在工作）")
        # 归因必须与数值一致：这条断言是"解释不是幻觉"的最低限度执法。
        if "被注入" not in explain or "token" not in explain:
            problems.append(f"归因文本不完整：{explain}")

    reply = next(
        message.content for message in window._session.messages if message.role == "assistant"
    )
    print(f"[5] 回复：{reply.strip()[:120]}", flush=True)
    if "书房" not in reply and "书架" not in reply:
        problems.append("回复里没有那条事实的内容——模型没把它用上")

    print("结论：" + ("；".join(problems) if problems else "想起来了，且账本说得出为什么"))
    return 1 if problems else 0


# ── 入口 ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 验收：记住 → 重启 → 想起来")
    parser.add_argument("phase", choices=("remember", "recall", "both"), nargs="?", default="both")
    parser.add_argument("--dir", default="", help="数据目录；两段必须用同一个")
    args = parser.parse_args(argv)

    if args.phase != "both":
        _prepare_env(args.dir or tempfile.mkdtemp(prefix="memterm-p1-"))
        return phase_remember() if args.phase == "remember" else phase_recall()

    data_dir = args.dir or tempfile.mkdtemp(prefix="memterm-p1-")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"数据目录：{data_dir}", flush=True)
    for phase in ("remember", "recall"):
        print(f"\n── {phase} ──────────────────────────────────────────", flush=True)
        # 真的另起一个进程。同进程里重新 open 一次证明不了"重启之后还在"。
        code = subprocess.call(
            [sys.executable, str(Path(__file__).resolve()), phase, "--dir", data_dir]
        )
        if code:
            return code
    print(f"\n全部通过。库留在 {data_dir}，可以拿眼睛去看。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
