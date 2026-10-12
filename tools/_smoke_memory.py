"""记忆界面的冒烟：真窗口、真线程、真退出码，从「点按钮」一路走到「关窗口」。

它覆盖的是 **gui 单元测试不碰的那一段**：``tests/test_ui_memory_window.py`` 不构造
真的 ``MainWindow``（它一连 Ollama、起四个线程、加载 WebEngine），所以主窗口与记忆
窗口之间的接线——按钮、``recall_updated`` 推送、会话与嵌入后端的推送、``_shutdown``
收掉记忆窗口——在这里用真进程验。**首启人设卡片那一步（[7]）也是同一个道理**：卡片
在网页侧，而"答完之后库有没有变"只有真进程量得到。

**不需要 Ollama。** 记忆界面不依赖模型，它读的是本地库；没有模型时那几次推送就是
"后端为 None"，而那恰恰是必须验的一条降级路径（探针只用词法通道）。

网页侧照样用替身（见 ``_smoke_e2e.py`` 的说明：离屏下 QtWebEngine 一渲染就把进程
硬杀），这里换掉的还多一层意义——本脚本要验的东西全在 Qt 控件这一侧，WebEngine
在场只会让"跑不跑得起来"变成一个随机事件。

用法::

    uv run --extra app --extra llm python tools/_smoke_memory.py
    QT_QPA_PLATFORM=offscreen uv run --extra app --extra llm python tools/_smoke_memory.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# 必须在 import axo 之前设好——paths 模块在首次调用时解析并缓存目录。
_TMP = Path(tempfile.mkdtemp(prefix="memterm-memui-"))
os.environ["AXO_DATA"] = str(_TMP)
os.environ["AXO_ISOLATED_STATE"] = tempfile.mkdtemp(prefix="memterm-memui-state-")

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from tools_support import RecordingChatView  # noqa: E402

import axo.ui.main_window as main_window_module  # noqa: E402
from axo.ui.app import configure_isolated_state  # noqa: E402
from memcore.ids import new_ulid  # noqa: E402
from memcore.sdr import encode_lexical  # noqa: E402
from memcore.types import MemoryItem, MemoryKind, Taint, Tier  # noqa: E402

PROBE_DEADLINE_S = 30.0

#: 一条会被编辑、遗忘、恢复、删除，最后留一条给探针用的记忆。
BODY = "我家的路由器放在书房书架顶层，旁边还有一个备用的交换机。"
EDITED = "我家的路由器换到客厅电视柜上了，交换机还在书房。"


def _wait_for(app: QApplication, done: Callable[[], bool], *, seconds: float) -> bool:
    """转事件循环直到 ``done()`` 为真或超时。

    不用 ``app.exec()``：这里只有一个异步动作（线索召回），而它已经有一个现成的
    判据（``_probe_ledger`` 非空）。起一整个事件循环再想办法退出来，比这句多绕。
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if done():
            return True
        time.sleep(0.02)
    return False


def _item(text: str, **overrides: Any) -> MemoryItem:
    fields: dict[str, Any] = {
        "memory_id": new_ulid(),
        "kind": MemoryKind.FACT,
        "tier": Tier.LONGTERM,
        "content": text,
        "taint": Taint.USER,
        "weight": 0.7,
        "session_id": "01SMOKESESSION",
        "source_ref": "turn://01SMOKESESSION/1",
        "lex_sdr": encode_lexical(text),
    }
    fields.update(overrides)
    return MemoryItem(**fields)  # type: ignore[arg-type]


def _select(mw: Any, memory_id: str) -> bool:
    for index in range(mw.memory_list.count()):
        if mw.memory_list.item(index).data(Qt.ItemDataRole.UserRole) == memory_id:
            mw.memory_list.setCurrentRow(index)
            return True
    return False


def main() -> int:
    configure_isolated_state()
    app = QApplication.instance() or QApplication(sys.argv)
    problems: list[str] = []

    # 换掉真视图**必须在构造窗口之前**——真件一被构造出来就会起 QtWebEngine。
    main_window_module.ChatView = RecordingChatView  # type: ignore[misc]
    window = main_window_module.MainWindow()
    print(f"数据目录：{_TMP}", flush=True)

    # ── [0] 启动时的外观推送 ────────────────────────────────────────────────
    #
    # **这一步盯的是一个只在这里看得见的漏法。** ``MainWindow._apply_theme`` 排在
    # ``_build_ui()`` 之前（应用级样式表必须早于控件），所以启动那一次推送必然找
    # 不到 ``chat_view``；补推的那一句一旦丢了，网页侧就永远停在 CSS 默认值上。
    # 而它**只有一种症状会被察觉**：默认预设与默认主色恰好相同，所以主题那一半看
    # 不出来；可见度只有 18 与 60 的差别，多半也认不出。真正刺眼的是"我设的壁纸
    # 重启后不见了"——而设置里那个文件名原样留着（那是刻意的），于是排查会从
    # "图是不是被删了"开始，一路都是错的。gui 单元测试不建 ``MainWindow``，量不到。
    #
    # 判据故意**不看值**，只看三件套有没有发出去：这台机器上 QSettings 是真的注册表
    # （本脚本只换数据目录），值随开发者的设置而变，断言值就等于把脚本绑在某台机器上。
    #
    # 三件各查一次而不是"数够三条"——``setWallpaper`` 是 ``setWallpaperOpacity`` 的
    # 前缀，按前缀数的话"主题推了两次、壁纸一次没推"也能凑够三条。
    pushed = window.chat_view.calls
    missing = [
        name
        for name, present in (
            ("主题", "setTheme" in pushed),
            ("可见度", any(call.startswith("setWallpaperOpacity(") for call in pushed)),
            ("壁纸", "setWallpaper" in pushed),
        )
        if not present
    ]
    if missing:
        problems.append(
            f"启动时外观没推全，缺 {'、'.join(missing)}——"
            f"网页侧会停在 CSS 默认值上，而症状看起来像壁纸没了"
        )
    print(
        f"[0] 启动时的外观推送：推了 "
        f"{[c for c in pushed if c.startswith(('setTheme', 'setWallpaper'))]}",
        flush=True,
    )

    # ── [1] 点「记忆」────────────────────────────────────────────────────────
    window._on_open_memory()
    mw = window._memory_window
    if mw is None:
        print("点了「记忆」但没有窗口——后面全部无法继续。", flush=True)
        return 1
    print(f"[1] 记忆窗口已建：{mw.windowTitle()} · 列表 {mw.memory_list.count()} 条", flush=True)

    # 人格锚是启动时 ensure_builtin_anchor 写进去的，所以它必须在列表里。
    # 这条同时验了两件事：窗口读的是**真库**，而人格锚在界面上是可见的
    # （可见但**不可改**——内置那条的只读规则在 [7] 里验）。
    texts = [mw.memory_list.item(i).text() for i in range(mw.memory_list.count())]
    if not any("人格锚" in text for text in texts):
        problems.append("列表里看不到人格锚——窗口读的可能不是主窗口那个库")
    print(f"    列表内容：{texts}", flush=True)

    # ── [2] 会话与账本的推送 ────────────────────────────────────────────────
    if mw._current_session_id != window._session.session_id:
        problems.append(
            f"窗口不知道当前会话：窗口 {mw._current_session_id!r} "
            f"vs 主窗口 {window._session.session_id!r}"
        )
    # **账本推送用的是信号，不是引用。** 这里直接 emit 一个假账本，验的是"接线通了"
    # ——真账本要发出一次请求才有，而本脚本不依赖模型。
    from memcore.audit import RecallLedger

    window.recall_updated.emit(RecallLedger(query_preview="冒烟"))
    if mw._ledger is None:
        problems.append("recall_updated 没接到：账本推不过去，窗口里的归因永远停在「还没装配过」")
    print(
        f"[2] 会话已推：{mw._current_session_id} · 账本已推：{mw._ledger is not None}", flush=True
    )

    # ── [3] 编辑 ────────────────────────────────────────────────────────────
    item = window._memory.add(_item(BODY))
    mw.reload()
    if not _select(mw, item.memory_id):
        problems.append("写进去的记忆没有出现在列表里")
    before = window._memory.get(item.memory_id)
    mw._ask_for_content = lambda _item: EDITED
    mw._on_edit()
    after = window._memory.get(item.memory_id)
    edited_ok = after.content == EDITED and not np.array_equal(
        after.lex_sdr.idx, before.lex_sdr.idx
    )
    if not edited_ok:
        problems.append("编辑之后正文或词法 SDR 没变")
    print(f"[3] 编辑：{'正文与词法 SDR 都变了' if edited_ok else '没变'}", flush=True)

    # ── [4] 软遗忘 → 恢复 ───────────────────────────────────────────────────
    _select(mw, item.memory_id)
    mw._on_forget()
    forgotten = window._memory.get(item.memory_id).tier is Tier.COLD
    if not forgotten:
        problems.append("软遗忘之后层级不是 COLD")
    mw._on_restore()
    restored_tier = window._memory.get(item.memory_id).tier
    if restored_tier is not Tier.LONGTERM:
        problems.append(f"恢复之后落在了 {restored_tier}，不是原来的长期层")
    print(f"[4] 遗忘 → 恢复：{restored_tier}", flush=True)

    # ── [5] 线索召回（唯一一个真线程）────────────────────────────────────────
    # 留着它别删——探针需要一个还在的条目；下面另写一条来验删除。
    keep = window._memory.add(_item("探针要用的一条：厨房的净水器滤芯三月换过。"))
    mw.reload()
    _select(mw, keep.memory_id)
    mw._on_probe()
    finished = _wait_for(app, lambda: mw._probe_ledger is not None, seconds=PROBE_DEADLINE_S)
    if not finished:
        problems.append(f"线索召回等了 {PROBE_DEADLINE_S:.0f} 秒没有回结果")
    else:
        from axo.memory.browse import probe_position

        detail = probe_position(mw._probe_ledger, keep.memory_id)
        print("[5] 线索召回：", flush=True)
        print("    " + detail.replace("\n", "\n    "), flush=True)
    # **结果到了不等于线程没了。** ``track`` 挂的 release 闭包接的是
    # ``thread.finished``，而那个信号要回到 GUI 事件循环里才被执行——结果先到、
    # 清单后空，中间隔着一次事件循环。真窗口里那个循环一直在转，所以这里也必须
    # 让它转一下再断言，否则量到的是脚本自己的时序，不是产品行为。
    emptied = _wait_for(app, lambda: not mw._live_threads, seconds=5.0)
    print(f"    线程清单（应为空）：{mw._live_threads}", flush=True)
    if not emptied:
        problems.append("线索召回结束后 5 秒，线程还挂在清单里——收尾时会等它，而它不会结束")

    # ── [6] 彻底删除 ────────────────────────────────────────────────────────
    doomed = window._memory.add(_item("马上要被彻底删掉的。"))
    mw.reload()
    _select(mw, doomed.memory_id)
    mw._confirm_delete = lambda _item: True
    mw._on_delete()
    deleted = window._memory.get(doomed.memory_id) is None
    if not deleted:
        problems.append("彻底删除之后条目还在库里")
    print(f"[6] 彻底删除：{'没了' if deleted else '还在'}", flush=True)

    # ── [7] 首启人设卡片 ────────────────────────────────────────────────────
    #
    # 这一段只能在这里验。库是新建的，所以 ``needs_persona`` 为真、``__init__`` 里
    # 应该推过一次 ``asking=True``；而**"忘了在 ``_wire`` 里接 ```persona_answered``"
    # 这种漏法在别处看不出来**——卡片答完了、库却没变，界面上只是一张卡片不肯消失。
    # gui 单元测试不建 ``MainWindow``，所以那条线只有真进程能测。
    from axo.memory.persona import builtin_anchor, user_anchor

    chat = window.chat_view
    if chat.persona_state is None or not chat.persona_state["asking"]:
        problems.append(f"新库启动时没有推 asking=True：{chat.persona_state}")
    print(f"[7] 启动时推的人设状态：{chat.persona_state}", flush=True)

    ANSWER = "说话简短一点，叫我老离。"
    window._on_persona_answered(ANSWER)
    written = user_anchor(window._memory)
    if written is None or written.content != ANSWER:
        problems.append("卡片答完了，人设却没进库——多半是 _wire 里那根线没接")
    if chat.persona_state is None or chat.persona_state["asking"]:
        problems.append("答完了还在问——网页侧的输入框会一直解不开")
    print(
        f"    答完之后：库里 {written.content if written else None!r} · "
        f"asking={chat.persona_state['asking'] if chat.persona_state else None}",
        flush=True,
    )

    # 内置那条**不可改不可删**：它在列表里看得见，两个按钮却是灰的，而且说得清理由。
    builtin = builtin_anchor(window._memory)
    if builtin is None:
        problems.append("库里没有内置那条人设")
    else:
        mw.reload()
        if not _select(mw, builtin.memory_id):
            problems.append("内置那条人设不在列表里")
        elif mw.edit_button.isEnabled() or mw.delete_button.isEnabled():
            problems.append("内置那条人设居然能改/能删——身份约束就不成立了")
        elif not mw.delete_button.toolTip():
            problems.append("内置那条的删除键灰着却不说为什么")
        print(
            f"    内置那条：可改={mw.edit_button.isEnabled()} "
            f"可删={mw.delete_button.isEnabled()} · 理由「{mw.delete_button.toolTip()[:24]}…」",
            flush=True,
        )

    # ── [8] 关窗口 + 收尾 ───────────────────────────────────────────────────
    # 这一步验的是那条最容易在下一次变成 0xC0000409 的路径：关窗要等线程停。
    if not mw.close():
        problems.append("记忆窗口 close() 没被接受")
    if mw._live_threads:
        problems.append("关窗之后线程清单还不空")
    window._shutdown()
    app.processEvents()
    if window._memory_window is not None:
        problems.append("_shutdown 没有收掉记忆窗口")
    print(f"[8] 关窗与收尾：窗口已收 = {window._memory_window is None}", flush=True)

    # **末尾必须把事件队列排空。** 本脚本是这几个冒烟里唯一一个不跑事件循环的：
    # 窗口的 show()/close() 会投递事件，而 ``deleteLater`` 要等事件循环才执行。
    # 带着一队没处理的事件退到解释器收尾，Qt 会走 ``qFatal``——Windows 上的表现
    # 就是 0xC0000409，没有回溯、没有 stderr、``atexit`` 都不跑，看起来像"脚本
    # 跑完了但退出码是坏的"。**实测过**：同样的步骤，末尾不排空必崩、排空就 0。
    #
    # （这一条只关于脚本：真窗口全程有事件循环，所以那条路径不受影响。）
    #
    # **"排空就 0"只在真实平台上成立。** 2026-10-12 实测：退出码 0x0 在原生窗口
    # 平台上是 0，而 ``QT_QPA_PLATFORM=offscreen`` 与 ``minimal`` 下**照样**
    # 0xC0000409——排空也拦不住，崩在解释器收尾。**但结论行照常打印**，所以离屏跑
    # 的时候看那一行、别看退出码（bash 还会把 0xC0000409 报成 127）。
    QTimer.singleShot(200, app.quit)
    app.exec()

    print("\n结论：" + ("；".join(problems) if problems else "记忆界面的接线全部通过"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
