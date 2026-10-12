"""主窗口。

布局::

    ┌──────────────────────────────────────────────────────────┐
    │ 模型 [下拉]  ⟳   ＋ 新会话        ● 状态        注入 ▸  │  ← 顶栏
    ├────────────┬───────────────────────────────┬───────────┤
    │ 会话列表   │  对话区（QWebEngineView）     │ 注入账本  │
    └────────────┴───────────────────────────────┴───────────┘

四段工作，从提交到落盘::

    _on_submit         记录用户这一句 → AssemblyWorker（后台）
    _on_assembly_ready 装配好了 → ChatWorker（后台）
    _on_reply_finished 记下这一轮 → 写进记忆库（界面线程）

三处细节值得说明，都是踩过的坑：

1. **线程生命周期**：worker 与 QThread 都是"用完即弃"，且 ``deleteLater`` 挂在
   ``thread.finished`` 上而不是 ``worker.finished`` 上。挂错地方会在工作线程还在
   收尾时就销毁对象，Windows 上表现为随机崩溃。
2. **停止按钮的语义**：它只是翻一个布尔。后端在客户端断开后会自行停止生成，
   但断开不可能瞬时完成——所以界面上"停止"的反馈是"当前这句说完就停"。**准备阶段
   是另一回事**：那时请求还没发出去，所以停止是即时的（见 :meth:`_on_stop`）。
3. **状态栏不说谎**：后端没起来时明确说"Ollama 未运行"并给出启动命令，而不是
   显示一个含糊的错误码让用户去猜。

**记忆写入是同步的，在界面线程上做。** 这是刻意的：:class:`MemoryWriter` 的文档
写明了理由——"说一句话"不该变成一次模型调用，而 SDR 编码是纯 numpy 的毫秒级运算。
异步写会让"刚说完就重启"丢掉最后一句，而那恰好是用户最容易测试的那一句。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from memcore.errors import MemCoreError
from memcore.graph import GraphIndex
from memcore.recall import RecallConfig
from memcore.store import MemoryStore

from ..agent.catalog import default_tools
from ..agent.gate import APPROVAL_TIMEOUT_S
from ..agent.reminders import CancelledStore, FiredStore, ReminderStore
from ..agent.router import HELP_TEXT, Register, RegisterState, RoutedTurn, route_turn
from ..agent.tools import ToolContext, ToolRegistry
from ..llm.base import ChatBackendError, ChatRequest
from ..llm.ollama import (
    MODEL_ID_PREFIX as OLLAMA_MODEL_ID_PREFIX,
)
from ..llm.ollama import (
    OllamaChatBackend,
    OllamaEmbeddingBackend,
    is_chat_capable,
    pick_embedding_model,
)
from ..llm.openai_compat import (
    MODEL_ID_PREFIX as CLOUD_MODEL_ID_PREFIX,
)
from ..llm.openai_compat import (
    OpenAICompatChatBackend,
    fallback_models,
    pick_cloud_chat_model,
)
from ..memory import (
    BUILTIN_PERSONA_VERSION,
    DEFAULT_PERSONA,
    MemoryWriter,
    PerItemPolicy,
    TaskPolicy,
    builtin_anchor,
    builtin_anchor_version,
    clear_user_persona,
    ensure_builtin_anchor,
    needs_persona,
    set_user_persona,
    user_anchor,
)
from ..memory.browse import format_ts
from ..memory.consolidation import ConsolidationRun, last_run, summary_line
from ..paths import (
    data_dir,
    data_root_source,
    default_data_dir,
    memory_db_path,
    reminders_cancelled_path,
    reminders_fired_path,
    reminders_path,
    sessions_dir,
)
from ..session.assemble import Assembly, Layer, layer_ceiling, stored_rows
from ..session.budget import TokenBudget, TokenMeter
from ..session.environment import build_environment
from ..session.indicator import ContextState, context_state, format_context
from ..session.store import Message, Session, SessionStore, new_session_id
from .agent_worker import AgentWorker
from .app import make_settings
from .chat_view import ChatView
from .chat_worker import ChatWorker, ModelListWorker
from .consolidation_scheduler import MIN_DELAY_MS, ConsolidationScheduler
from .data_move import MoveKind, MovePlan, plan_move
from .data_root import MoveReport, discard_source, stage_move
from .layout import plan_pane_sizes
from .memory_window import MemoryWindow
from .memory_worker import (
    AssemblyWorker,
    ConsolidationWorker,
    EmbeddingSetup,
    GraphWorker,
    MemorySyncWorker,
    render_recall_detail,
)
from .onboarding import OnboardingWizard
from .relaunch import restart_application
from .reminder_scheduler import ReminderScheduler
from .settings_dialog import SettingsDialog
from .settings_store import (
    PROVIDER_OLLAMA,
    BackendSettings,
    MemorySettings,
    SettingsStore,
    ToolSettings,
    wallpaper_file,
)
from .theme import Resolved, Theme, resolve
from .theme_qt import QSettingsBackend, apply_theme, os_color_scheme
from .thread_tracking import track, wait_for_all
from .tray import TrayController, build_icon
from .widgets import SESSION_TIME_ROLE, SESSION_TITLE_ROLE, ElidedLabel, SessionItemDelegate
from .window_place import place_window

__all__ = ["MainWindow"]

#: ``QSettings`` 怎么建**不在这里**：``ui/app.py`` 的 :func:`~axo.ui.app.make_settings`
#: 是唯一入口。两个地方各建一次的话，隔离模式下会变成"启动时读的是 ini 文件、运行时
#: 读写的是注册表"（两参数的构造认不出默认格式，见那个函数的 docstring），症状是
#: "设置改了不生效"。

#: 账本面板展开时的名义宽度**不在这里**：它在
#: :data:`axo.ui.layout.LEDGER_WIDTH`，与夹紧逻辑住在一起。这里再留
#: 一份副本就是两处会各自漂移的常量——而"账本多宽"漂了不会有任何报错，只会让
#: 某个窗口里对话区莫名变窄。

#: 指示灯四种状态的颜色。**只有一种颜色是对的写法**：状态判定在
#: :func:`~axo.session.indicator.context_state` 里（核心腿，可断言），
#: 这里只把枚举映射成色值。判定与配色分居两处，是为了让"什么时候算超了"这件事
#: 不必靠肉眼看颜色来验证。
_CONTEXT_COLORS: dict[ContextState, str] = {
    ContextState.EMPTY: "#4a4a4a",
    ContextState.OK: "#3f9c4f",
    ContextState.WARN: "#d99a2b",
    ContextState.OVER: "#c0392b",
}

_LEDGER_PLACEHOLDER = (
    "还没有发出过请求。\n\n"
    "发出第一条之后，这里显示：\n"
    "· 每一层占了多少 token、上限是多少\n"
    "· 每条被注入的记忆，为什么是它\n"
    "· 哪些候选被淘汰了，以及为什么\n"
    "· 被逐出的历史压成摘要后覆盖了几条"
)

#: 收尾等待的上限**不在这里**：它在 :data:`axo.ui.thread_tracking.
#: THREAD_WAIT_MS`，记忆窗口的 ``closeEvent`` 用的是同一个数。留一份副本在这里
#: 就是两处会各自漂移的那种常量——而"等多久"这件事漂了不会有任何症状，只会让
#: 某个窗口偶发地关得慢一点。

#: 联想通道的**可靠度**（``RecallConfig.graph_weight``）——开关打开时用的那个非零值。
#:
#: **这是个未标定的初值，不是调好的值。** 取扫描表里最小的那一档
#: （``benchmarks/gold/CORPUS.md`` 的「联想腿」）唯一站得住的理由是"损害最小"：
#: 那一档 MRR −0.0037、AUC −0.0172，而 w=1.00 时 AUC 掉 0.0643。四个档位**没有
#: 一档是正的**，所以这里选的不是"最好的"，是"最轻的"。
#:
#: 真正的取值要真实库上的数据：金标准语料是刻意造了近义改写的，而真实库里
#: "邻居"与"词法近重复"的重合度不一定有这么高。**在标定之前不要往上调。**
_GRAPH_CHANNEL_WEIGHT = 0.25


def _last_run_note() -> str:
    """设置页「巩固」那一段的默认那句话：**上一轮是什么时候跑的**。

    两个来源合起来才是完整的一句：``last.json`` 给时刻与条数（跨进程的真源），
    这里给措辞。分列的理由与图状态那一行相同——上面那个窗口对象里没有任何字段
    记得"上次什么时候跑的"，因为那件事**在盘上**、而且可能是上一个进程写下的。

    **读不出来（没有 / 坏了）时说"还没有跑过"**，不编一个时间。那个文件不是证据
    而是给下一次跑用的记录（见 :func:`~axo.memory.consolidation.read_last_run`），
    它丢了只意味着"这一轮该跑"，于是对用户说"还没跑过"既是真话也是他要做的事。

    **"这个数拿得去算吗"那一层守卫在 :func:`~axo.memory.consolidation.last_run`
    里**，不在这里：热度面板要说同一句话，而两处各写一遍的话，某天其中一处忘了挡
    ``true``（``bool`` 是 ``int`` 的子类），症状是"最近一次"显示成 1970 年。这里
    只管措辞。
    """
    last = last_run()
    if last is None:
        return "还没有跑过。"
    at, count = last
    count_text = f"，加固 {count} 条" if count else ""
    return f"最近一次：{format_ts(at)}{count_text}。"


def display_rows(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """把逐字稿的行转成网页要的那份 ``[{role, text, meta}, ...]``。

    **带 ``tool_calls`` 但没有正文的助手行要跳过。** 它是"模型这一轮什么都没说、
    直接去调工具"的那一行；照原样渲染就是一条**空气泡**，而回看的人只会觉得
    "这里怎么空了一块"。有正文的那些（模型边想边说）照常显示——它们与当时屏幕上
    看到的逐字逐句一致。

    **``tool`` 行要显示**，而且要带上 ``tool_name``。它是模型看得见、用户当时也
    看见过的东西（工具结果卡片），回看时少掉它，"助手怎么突然知道了这件事"就成了
    一段没有来由的对话。名字放标题而不是正文：正文是模型要读的那段，标题是给人
    认的。

    工具**调用的声明行**（``tool_calls``）在回看时不再单独渲染——它是实现细节，
    用户要看的是"助手说了什么、工具查到了什么"。真要看完整调用，逐字稿里有。

    **被闸门拒绝的那一行（``no_replay == "denied"``）要换一套说法，不能照原样渲染。**
    它没有执行过，所以它既不是"结果"，正文也不是给人看的话——那是**写给模型的当轮
    指令**（"这一次调用跳过了……不要重试同一次调用……做完你还能做的部分"，见
    ``agent/tools.py`` 的派发口）。照原样渲染等于**把给模型的命令展示给人**，而那段
    语气在人的眼里是坏的：像程序在教训用户。这一位是 P3a 加 ``no_replay`` 时才有的，
    所以它是"这一位终于让界面分得清两种收场"而不是新增的耦合——界面读的是逐字稿里
    已有的字段，不是去问 ``agent/`` 要一张表（那会挨上 ``_PURE_LAYERS`` 那条环）。
    """
    rows: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls and not message.content:
            continue
        meta: dict[str, Any] = {}
        text = message.content
        if message.role == "tool":
            name = message.tool_name or "未知工具"
            if message.no_replay == "denied":
                # 标题用「跳过」而不是「拒绝」：``denied`` 把三种收场合并成一个值
                # （用户点了拒绝 / 超时 / 停止或关窗，见 ``gate.py`` 与 ``loop.py``），
                # 只有"跳过"对三种都成立；写"拒绝"是在替用户认一件他可能没做的事。
                meta["title"] = f"已跳过 · {name}"
                text = "这一步没有执行——它需要你点头，而没有得到允许。"
            else:
                meta["title"] = f"工具结果 · {name}"
        rows.append({"role": message.role, "text": text, "meta": meta})
    return rows


def _details_hint() -> str:
    """出错时"详情在哪"这一句。**按是不是冻结产物分岔。**

    源码检出跑 ``run.bat`` 时父进程留着控制台，回溯打在它上面；而打包出来的 exe
    （见 ``tools/build_exe.py``）**没有控制台**，指过去只会让人去找一个不存在的东西。
    带控制台的版本与它并存，名字是 ``axo-console``。
    """
    if getattr(sys, "frozen", False):
        return "详情要看得跑带控制台的版本（`axo-console.exe`）。"
    return "详情在启动它的那个控制台窗口里（`run.bat` 会保留它）。"


def _install_window_icon(window: QWidget, accent: str) -> None:
    """把画出来的图标装到窗口上：标题栏、任务栏、Alt-Tab 三处看的是它。

    **与 exe 里嵌的那一份是两件事。** ``tools/build_exe.py`` 的 ``--icon`` 决定的是
    资源管理器里那个文件长什么样；窗口自己不设图标，标题栏就空着一格——Windows 不
    会拿 exe 的图标兜底。所以打包过的产物也可能"窗口有、图标没有"。

    画笔与托盘同一支（:func:`~axo.ui.tray.build_icon`），于是图标跟着
    主题的强调色走，换主色时两处一起变。

    应用级与窗口级**都设**：前者让之后新建的设置窗口、记忆窗口也带上（它们是独立
    顶层窗口，不挂在这个窗口上），后者保证主窗口当场生效。
    """
    icon = build_icon(accent)
    QGuiApplication.setWindowIcon(icon)
    window.setWindowIcon(icon)


class MainWindow(QMainWindow):
    """记忆终端主窗口。"""

    recall_updated = Signal(object)
    """每轮召回账本（:class:`~memcore.audit.RecallLedger` 或 ``None``）。

    **记忆窗口靠这个信号拿账本，不靠读 :attr:`_recall_ledger`。** 那个成员每轮被
    **重新赋值**，窗口捕获一次引用之后下一轮就指向旧值——而且不报错。依赖方向也
    反过来：窗口不必持有主窗口的私有可变量，主窗口也不必知道有谁在听。

    ``RecallLedger`` 是 frozen 的，按值共享没有别名风险。
    """

    def __init__(self, *, memory: MemoryStore | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Axo")
        self._place_window()

        self._settings = make_settings()
        self._settings_store = SettingsStore(QSettingsBackend(self._settings))

        # **外观必须在建控件之前定好。** 应用级样式表与调色板一旦设置，之后建的
        # 控件都按它起手画第一次；反过来（先建控件再设）要靠一次全局 repolish
        # 才能追上，而那一步在有一些原生子控件时会漏。所以这一段排在
        # ``_build_ui()`` 之前——见 ui/app.py 的模块文档。
        #
        # 主题与后端设置都从**同一个** ``QSettings`` 读：两类设置共用一份存储，
        # 只是键的空间分开（``appearance/*`` 与 ``backend/*``）。
        self._theme = self._settings_store.load()
        self._resolved: Resolved = self._apply_theme(self._theme)

        #: 当前后端设置**整份**。``provider`` / ``host`` / ``allow_remote`` /
        #: ``cloud_base_url`` / ``api_key`` 从此只有一个家——``_host`` 与
        #: ``_allow_remote`` 是从它上面取的只读属性（见那两处）。拆成一个个字段时，
        #: 「现在是云端吗」要靠反推，而每加一项就多一个漏掉的机会。
        self._backend_settings: BackendSettings = self._settings_store.load_backend()
        #: 上一次模型清单里**服务端报的**上下文窗口，按模型名索引。只为云端那条
        #: 构造路径服务：``OpenAICompatChatBackend`` 的构造会去问一次端点，而那
        #: 发生在界面线程上，不该有网络往返。
        self._model_windows: dict[str, int] = {}
        #: ``read_file`` 允许读的目录。**默认空元组 = 那个工具被禁用**——"没配"不等于
        #: "给个默认目录"，理由见 :mod:`axo.agent.fs`。存在 ``QSettings``
        #: 的 ``tools/*`` 命名空间里，由设置窗口的「工具」页改。
        self._read_roots: tuple[Path, ...] = self._settings_store.load_tools().read_roots
        #: 用不用**任务记忆**（:class:`~axo.memory.policy.TaskPolicy`）。
        #: 默认关，存在 ``QSettings`` 的 ``memory/*`` 里，由设置窗口的「记忆」页改。
        self._task_memory: bool = self._settings_store.load_memory().task_memory
        #: 用不用**联想通道**（第三通道）。默认关，与任务记忆同住在
        #: ``QSettings`` 的 ``memory/*`` 里，由设置窗口的「记忆」页改。
        #:
        #: **它只决定"用不用图"，不决定"建不建图"。** 图是另一个东西、另一个
        #: 时机（见 :meth:`_on_open_memory`）：开关翻成 ON 不会顺手建一张图，因为
        #: 那是几秒到几分钟的计算，而用户翻这个开关的意图是"下一轮召回多看一眼"。
        #: 没有图时打开开关**不是静默失效**——``memcore.recall`` 会往账本里写一句
        #: "联想通道已启用，但没有可用的关联图"，界面在归因里照原样显示。
        self._graph_enabled: bool = self._settings_store.load_memory().graph_channel
        #: 关联图本体。``None`` = 还没有。**只在打开记忆窗口时建/载入**——启动时
        #: 无条件建的话，每个用户每次开程序都要为用不上的功能等几秒。
        self._graph: GraphIndex | None = None
        #: 这份图**过期与否的三态**：``True`` 确定过期、``False`` 刚比对过是最新的、
        #: ``None`` **没比对过**（库太大）。三态而不是布尔：``None`` 与 ``False``
        #: 在界面上要说不同的话（见 ``memory.browse.association_header``）。
        self._graph_freshness: bool | None = None
        #: 图状态的**唯一说法**，设置页那一行照它显示。空串 = 还没人建过图。
        self._graph_note: str = ""
        #: 有没有一趟建图正在跑。**它是"同一时刻只跑一趟"那条约定的载体**：两趟
        #: 会同时写同一个 ``graph.npz``，而哪趟最后落盘决定了盘上是哪张图。
        self._graph_building = False
        #: 睡眠巩固（P4c）的三个状态位。**放在建图这三个旁边是有理由的**：它们是同一
        #: 条推迟链上的两端——没有图就没有分簇，而图是懒建的，于是"到点了但图还没建"
        #: 是这条路径上最常见的分支。
        #:
        #: :attr:`_consolidating` 是**单飞守卫**。两轮同时跑会 `bump_reinforce` 同一批
        #: 条目（计数多涨一次）并抢同一个快照文件名。与 ``_graph_building`` 的区别在
        #: 后果：建图写坏了是"图变旧"，这里写坏了是"计数说谎"。
        self._consolidating = False
        #: 自动那一趟到点时图还没建（或正在建）——把这一轮**挂起**，等图回来再跑。
        #: 不挂起的话，默认开的功能对"从不打开记忆窗口的人"就是一个静默的空转：
        #: 每天醒一次、每天因为没有图而什么都不做。
        self._consolidate_pending = False
        #: 挂起的那一轮是**手动**要的吗。与上面那个布尔分开是因为它们是两条正交的
        #: 轴（要不要跑 / 跑完弹不弹气泡），合并成一个"待办类型"字段会在"自动挂起
        #: 期间用户又点了一次手动"那一刻没有正确答案——而气泡正是手动那条路的全部
        #: 意义（自动跑不弹，否则是每天的噪音）。
        self._consolidate_manual = False
        self._consolidation_enabled: bool = self._settings_store.load_memory().consolidation_enabled
        #: 周期（秒）。**这个数的真源是用户的设置**，调度器那一侧再写一个默认值就是
        #: 第二个说法（见 `ui/consolidation_scheduler.py` 的构造参数说明）。
        self._consolidation_interval_s: int = (
            self._settings_store.load_memory().consolidation_interval_s
        )
        #: **这个进程实际在用哪个数据目录**，以及它听谁的（默认位置 / 你选的 /
        #: 便携指针文件 / 环境变量）。位置本身是 ``ui/app.py`` 在建本窗口**之前**定下
        #: 来的（不能在构造函数里定：下面几行一执行，这四个 store 就把路径焊死了）。
        #: 这里只是把结果记下来给设置页显示——它每次打开都现读，因为真源在 ``paths``
        #: 那边，而搬迁是跨重启才生效的。
        self._data_root: Path = data_dir()
        self._data_source: str = data_root_source()
        self._store = SessionStore(sessions_dir())
        # 记忆库与逐字稿是**两个库**，见 session/store.py 的模块文档：删聊天记录
        # 不该有概率删掉长期记忆。参数可注入是为了让界面能在临时目录上被测试。
        self._memory = memory if memory is not None else MemoryStore(memory_db_path())
        # **装策略，不装开关。** ``writer.policy`` 是公开可写属性，而策略在**每轮写入
        # 时**被读（见 ``MemoryWriter.record``）——所以换粒度只要赋一个新对象，
        # 不必重建写手（重建会连带丢掉 ``_clock`` 之类的注入）。
        self._writer = MemoryWriter(
            self._memory,
            policy=TaskPolicy() if self._task_memory else PerItemPolicy(),
        )
        #: 提醒登记簿。登记归 :mod:`axo.agent.reminders`，**触发**归下面
        #: 那个 :class:`~axo.ui.reminder_scheduler.ReminderScheduler`。
        #: 位置由 ``paths`` 给："应用的数据目录长什么样"在这个程序里只有那一个说法。
        self._reminders = ReminderStore(reminders_path())
        #: 响过的那些。**与登记簿分开一个文件**——``reminders.jsonl`` 的契约是"还没
        #: 被提醒的事情"，往里掺第二种记录会让那句话变成假的（见那个模块的文档）。
        self._reminders_fired = FiredStore(reminders_fired_path())
        #: **第三本账。** 取消过的那一条记在这儿，而不是改写登记簿——那个文件
        #: 只追加（见 `agent/reminders.py` 的 `CancelledStore` 与 `paths`）。
        self._reminders_cancelled = CancelledStore(reminders_cancelled_path())

        self._session: Session = Session(session_id=new_session_id())
        self._backend: OllamaChatBackend | None = None
        self._embedding: OllamaEmbeddingBackend | None = None
        self._embedding_model: str = ""
        self._chat_thread: QThread | None = None
        self._chat_worker: ChatWorker | None = None
        self._agent_thread: QThread | None = None
        self._agent_worker: AgentWorker | None = None
        self._prepare_thread: QThread | None = None
        self._prepare_worker: AssemblyWorker | None = None
        self._model_thread: QThread | None = None
        self._model_worker: ModelListWorker | None = None
        self._setup_thread: QThread | None = None
        self._setup_worker: MemorySyncWorker | None = None
        #: 云端模式下**另起的那一趟**清单（只问本机 Ollama 要嵌入模型）。与
        #: ``_model_thread`` 分开记，因为它们是两个线程：共用一对成员的话，后起的
        #: 那个会把先起的那个的引用挤掉，而收尾清单是按成员名找的——被挤掉的那个
        #: 就变成"没人等的线程"（见 :func:`~.thread_tracking.track`）。
        self._embedmodels_thread: QThread | None = None
        self._embedmodels_worker: ModelListWorker | None = None
        #: **还在跑的线程的权威清单**，收尾时按它来。不复用上面那四个成员：它们
        #: 会被 `_reset_generation_state` 提前清掉（取消时装配线程还在跑，这是
        #: 刻意的），而清掉之后那个线程仍然要在退出前被等——否则它就变成"没人等
        #: 的线程"，正是这个 bug 换一副面孔的样子。
        self._live_threads: list[QThread] = []
        self._shutting_down = False
        #: **真的要退出**（区别于"关窗收进托盘"）。托盘菜单的「退出」与 Ctrl+Q 先
        #: 把它置真再 ``close()``，于是 ``closeEvent`` 走收尾那一支。见
        #: :meth:`_close_should_hide`。
        self._quitting = False
        #: 系统托盘。**不可用时它什么都不做，关窗行为也与托盘出现之前一样**——
        #: 那台机器上拦下关窗等于让应用变得够不着（见 ``ui/tray.py`` 的模块文档）。
        #: 放在 ``_build_ui`` 之前：图标要在窗口第一次显示时就位。
        self._tray = TrayController(accent=self._resolved.accent, parent=self)
        self._tray.show_requested.connect(self._on_tray_show)
        # 「立即巩固」。**与定时那条路共用一个 worker，只差一个 force**——手动按下
        # 就是"我不在乎有没有新东西，现在就跑一遍"，而那正是 GraphWorker 的 force
        # 那一档。两条路分开写两套启动逻辑的话，单飞守卫与推迟分支就会有两份实现，
        # 而它们迟早会在某一边漏掉一条。
        self._tray.consolidate_requested.connect(self._on_manual_consolidation)
        self._tray.quit_requested.connect(self._quit)
        #: 定时触发。**它的前提是上面那个托盘**：关窗即退的进程在"到点了"那一刻已经
        #: 不在了（见 ``ui/reminder_scheduler.py`` 的模块文档）。它算好标题与正文就
        #: 发信号——不认识窗口也不认识托盘，显示在下面那个槽里。
        #:
        #: ``start()`` 不在这里，在 ``__init__`` 的**最后**：启动时错过的那些是
        #: **同步**发出来的，而槽要往 ``chat_view`` 里写。
        self._scheduler = ReminderScheduler(
            reminders=self._reminders, fired=self._reminders_fired, parent=self
        )
        self._scheduler.due.connect(self._on_reminder_fired)
        self._scheduler.missed.connect(self._on_reminder_fired)

        #: 周期巩固的表。**与上面那个并存、不打架**：两个独立 ``QTimer``、两个不同的
        #: 槽、没有共享可变状态（见 ``ui/consolidation_scheduler.py`` 的模块文档）。
        #:
        #: ``due`` **不等于"跑"**：接收方还要查单飞、查图、还可能把这一轮让掉。这个
        #: 类只报"时间到了"——把判据搬进去会让它认识库与图，而它现在连库在哪儿都不
        #: 知道（这正是它能被单测的原因）。
        #:
        #: ``start()`` 与提醒那个不同，在 ``__init__`` 里就能调：它**不同步发信号**
        #: （没有"错过"这个概念），所以 ``chat_view`` 建没建好它都不在乎。
        self._consolidation_scheduler = ConsolidationScheduler(
            interval_s=self._consolidation_interval_s, parent=self
        )
        self._consolidation_scheduler.due.connect(self._on_consolidation_due)

        self._streaming = False
        self._preparing = False
        self._cancel_requested = False
        self._pending_text = ""
        #: 语域的**跨轮**状态。斜杠、规则层、迟滞三条路都汇到这一个对象上——
        #: :class:`~axo.agent.router.RegisterState` 的注释早就写着"调用方
        #: （``MainWindow``）持有**一个**实例，跨轮传下去"，这里就是那一处。
        #:
        #: **三条路都要汇到同一个对象上。** 曾经只有斜杠能改语域，于是规则层在
        #: ``bench/`` 里跑绿却一个生产消费者都没有——症状是"不打 ``/task`` 就永远拿
        #: 不到工具"，而它看起来像模型不听话。
        self._router_state = RegisterState()
        #: **这一轮**实际用的语域。与状态分开，是因为 ``/task 正文`` 只影响一轮
        #: （:attr:`~axo.agent.router.ParsedTurn.sticky` 为假）。合并成一个
        #: 的话，"这一件正事"会变成"用户从此进了任务模式"。
        self._turn_register: Register = Register.CHAT
        #: 这一轮要不要把工具声明给模型。**不写成"看 ``_turn_register`` 就知道"**：
        #: 歧义那一轮的语域也是 ``CHAT``，但原因与真正的闲聊完全不同，而
        #: :attr:`~axo.agent.router.RoutedTurn.uses_tools` 已经把这条区分
        #: 想清楚了。这里存它的结论，不自己再推一遍。
        self._turn_uses_tools = False
        #: 这一轮要不要请模型带一句确认（歧义反问）。它进的是 L1 环境槽，
        #: 见 :func:`~axo.session.environment.build_environment`。
        self._turn_needs_clarification = False
        self._budget: TokenBudget | None = None
        self._assembly: Assembly | None = None
        self._recall_ledger: Any = None
        self._embedding_note = ""
        #: 记忆浏览器窗口。**懒建**：它要等用户点「记忆」才存在（一次都不点的人
        #: 不该为它付构造成本），所以下面每一处推送都要先问它是不是 None。
        self._memory_window: MemoryWindow | None = None
        #: 设置窗口。同样懒建。**非模态是刻意的**：外观改一下就要看一眼主窗口，
        #: 模态会让"拖滑杆 → 看效果"变成"关窗 → 看效果 → 再开窗"。
        self._settings_dialog: Any = None
        #: 新手引导。**每次重开都新建一个**（不像设置窗口那样懒建一次就复用）：
        #: 它那几步是按需要拼出来的（已经写过人设时跳过人设那一步），而顺序表在构造
        #: 时就定死了——复用会让"重开时想改人设"那一次少掉那一步。
        self._onboarding: Any = None

        self._build_ui()
        self._wire()
        self._reload_sessions()
        # 进程级退出也走同一条收尾路径。**这一条不是冗余的**：脚本与将来的"CapsLock
        # 之外的退出方式"都可能让 `app.exec()` 直接返回而窗口从未被 close，
        # 那时 closeEvent 不跑，只有这里收得住那几个线程。
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)
        self._refresh_models()
        self.chat_view.set_status("正在探测本地模型服务…")

        # 人设要先于第一条请求存在：它是 L0 层唯一的内容，而"任何模型读到的完全
        # 一样"这句话的落点就是它。库里**两条**，角色不同（见 memory/persona.py）：
        #
        # 1. **内置那条**：缺了就写、是旧版就升到当前版，已经在当前版上就什么都不做。
        #    判定按 ``source_ref`` 而不是按 ``kind``——按 kind 判的话，"用户自己写过
        #    一条"会让内置那条永远补不进去，症状是身份约束凭空少一半且全程无声。
        #    tests/test_persona.py 用
        #    `test_a_user_anchor_does_not_count_as_the_builtin_one` 钉住了它。
        # 2. **用户那条**：没有就问一次。问的时机在这里，因为"要不要问"是启动时的
        #    事实，中途不会变（本进程内）。
        #
        # 返回值非 ``None`` 因此有两种情形，而它们值得说两句不同的话：**空库新写**
        # 是"你的库在这儿"；**老库升级**改的是模型读到的身份正文，而用户看不见那次
        # 改动——不说就是一次静默变更。所以先读一次版本再动手（读完那个数就作废了）。
        #
        # **这句只说版次，不说这一版改了什么。** 写"这次加了一句 X"在下一版就成了假话，
        # 而它躺在代码里没人会想起来改；改了什么记在 README 的人格锚版本记录里。
        previous_version = builtin_anchor_version(self._memory)
        if ensure_builtin_anchor(self._memory) is not None:
            opening = (
                "已写入内置人格锚。"
                if previous_version is None
                else f"内置人格锚已从第 {previous_version} 版升到第 {BUILTIN_PERSONA_VERSION} 版。"
            )
            self.chat_view.append_message(
                "system",
                f"{opening}它是记忆库里一条钉住的条目，"
                f"任何模型读到的都是同一份字节。库在 `{memory_db_path()}`。",
            )

        #: 本次进程内是否还在等人写人设。**不持久化**：唯一的真源是库里有没有
        #: 用户那条（:func:`~axo.memory.needs_persona`）。
        #:
        #: 于是"删掉之后重新问一次"有两种时机，各由不同的东西负责——删掉之后**本
        #: 进程不再弹**（卡片住在对话流里，而这时候用户多半正在跟它说话），下次启动
        #: 才重问。这就是那条裁决的全部实现，也因此不需要任何 flag 文件。
        self._persona_asking = needs_persona(self._memory)
        self._push_persona_state()

        # **接管提醒——放在最后一步。** ``start()`` 会**同步**发出启动时错过的那些
        # （:meth:`_fire` 里是直接 ``emit``），而 :meth:`_on_reminder_fired` 要往
        # ``chat_view`` 里写；排在 ``_build_ui`` 之前就会炸在构造里。
        #
        # 顺序上还有一条：它排在 ``_shutdown`` 有可能被触发的一切之后，于是"还没有
        # 调度器"这个中间态不存在（``aboutToQuit`` 在上面就接好了）。
        self._scheduler.start()

        # **接管周期巩固。** 排在这里没有提醒那条的时序约束（``start()`` 不同步发
        # 信号），所以它其实可以更早；放在一起是让"这个程序里有两块表"这件事只有
        # 一个落点。
        #
        # **关掉时连表都不排**，而不是"排上了再在槽里退出"：后者每次醒来都要读一遍
        # ``last.json``，而用户关掉它就是为了不再有这回事。停表的另一半在
        # :meth:`_shutdown`。
        if self._consolidation_enabled:
            self._consolidation_scheduler.start()

    # ── 外观 ────────────────────────────────────────────────────────────────

    def _place_window(self) -> None:
        """把窗口放到光标所在那块屏上。逻辑在 :func:`~.window_place.place_window`
        ——记忆窗口用的是同一个函数，两处各写一遍必然会在某一边漏掉那个
        ``devicePixelRatio`` 陷阱。"""
        place_window(self)

    def _apply_theme(self, theme: Theme) -> Resolved:
        """换主题：装到应用上，并推给网页。返回**解析后**的那一份。

        **这是换外观的唯一入口。** 设置窗口的实时预览、系统明暗变化的回调、
        启动时的首次装载，三条路径都走这里——各自 :func:`resolve` 一遍是这类
        代码最容易漂的地方（网页那侧要的是"实际画出来的主色"，浅色下可能已经
        换过一支）。

        ``self.chat_view`` 在启动时还不存在（本方法要排在 ``_build_ui()`` 之前，
        见 ``__init__`` 的注释），所以推送那一半要容得下它缺席。
        """
        self._theme = theme
        app = QApplication.instance()
        if app is None:  # 只可能出现在测试里；没有 app 就没有样式表可设
            resolved = resolve(theme, os_color_scheme())
        else:
            resolved = apply_theme(app, theme, os_color_scheme())
        self._resolved = resolved
        # 图标装在这里而不是 ``__init__`` 里单写一处：换主色（设置窗口的实时预览、
        # 系统明暗翻转）走的是同一个入口，装在这儿才不会有"托盘变了、标题栏还是旧
        # 色"的两处不一致。
        _install_window_icon(self, resolved.accent)
        self._push_appearance()
        return resolved

    def _push_appearance(self) -> None:
        """把外观三件套（主题、壁纸、可见度）推给网页。

        **建完控件之后必须再调一次**——见 :meth:`_build_ui` 末尾那一句。这一半要等
        ``chat_view`` 存在，而 :meth:`_apply_theme` 排在 ``_build_ui()`` 之前
        （应用级样式表必须早于控件，理由见 ``__init__``），所以启动路径上第一次调
        这里时它必然缺席。

        ``chat_view`` 不在时静默返回：本方法在启动路径上会被调用两次，第一次本来
        就找不到它，那不是异常。
        """
        view = getattr(self, "chat_view", None)
        if view is None:
            return
        view.set_theme(self._resolved)
        view.set_wallpaper_opacity(self._theme.wallpaper_visible)
        self._push_wallpaper()

    def _push_wallpaper(self) -> None:
        """把设置里的壁纸文件名变成 ``file://`` 绝对 URL 推给网页。

        **文件不在就当作没有壁纸，并且说一句。** 用户把数据目录搬走、图片被删、
        U 盘被拔都是常态；静默回落会让人以为"壁纸功能坏了"，而直接抛异常则更糟。
        **不清掉设置里的名字**（那是 :func:`~.settings_store.wallpaper_file` 的
        决定）：文件过两天回来了，壁纸自己就恢复。
        """
        view = getattr(self, "chat_view", None)
        if view is None:
            return
        path = wallpaper_file(self._theme.wallpaper)
        if path is None:
            view.set_wallpaper(None)
            if self._theme.wallpaper:
                self.chat_view.set_status(f"壁纸 {self._theme.wallpaper} 找不到了，已按无壁纸显示")
            return
        view.set_wallpaper(QUrl.fromLocalFile(str(path)).toString())

    def _on_os_color_scheme_changed(self, _scheme: Any = None) -> None:
        """系统明暗变了。**这就是裁决 5「两侧一起跟」的落点。**

        **只在 ``mode == "system"`` 时才有实际变化**——用户明确指了明暗时，系统
        说什么都不算（见 :func:`~axo.ui.theme.effective_mode`）。这里
        仍然无脑重算一次，是为了让这条路径只有一个分支：先判模式再决定要不要重算，
        那两处判断迟早会与 ``effective_mode`` 里的那一处漂开。

        **记忆窗口不用单独推。** ``QApplication.setStyleSheet`` 会让 Qt 给所有已存在
        的控件补一次 style change，而记忆窗口是应用的顶层子窗口，自动跟着变。
        """
        self._apply_theme(self._theme)

    # ── 构建界面 ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_context_strip())

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self._build_session_pane())
        self.chat_view = ChatView(splitter)
        splitter.addWidget(self.chat_view)
        splitter.addWidget(self._build_ledger_pane())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        # **三格都不许被拖到塌陷。** 账本本来就可以收，但那个动作归它的按钮管——
        # 让拖拽也能收就有了两个控件管同一个状态，而它们会不一致（按钮显示"已展开"、
        # 面板却是零宽）。收起/展开的判定只留 `ledger_button.isChecked()` 一处。
        splitter.setChildrenCollapsible(False)
        for index in range(3):
            splitter.setCollapsible(index, False)
        # 初始宽度按窗口实际尺寸算，而不是写死 [220, 780, 0]。拉伸因子是 0/1/0，
        # 所以用户一缩放窗口，对话区就吃掉全部增量，这三项只决定**开局**的观感。
        splitter.setSizes(list(plan_pane_sizes(self.width())))
        self._splitter = splitter
        outer.addWidget(splitter, 1)

        # **建完对话区立刻把外观推一次。** ``_apply_theme`` 排在 ``_build_ui()``
        # 之前（应用级样式表必须早于控件），那时 ``chat_view`` 还不存在，所以那一趟
        # 只设了 Qt 那一半；不在这里补一次，网页侧拿到的就永远是 CSS 默认值——主色、
        # 壁纸、可见度全部丢掉。
        #
        # 这个漏法**只有一种症状会被察觉**：默认预设与默认主色恰好相同，所以主题那
        # 一半看不出来；可见度只有 18 与 60 的差别，多半也认不出。真正刺眼的是"我
        # 设的壁纸重启后不见了"。而它格外难查——设置里那个文件名**原样留着**（那是
        # 刻意的，文件回来了壁纸就自己恢复），于是排查会从"图是不是被删了/数据目录
        # 是不是搬了"开始，一路都是错的。它在 `_smoke_memory.py` 的 [0] 里钉住了。
        self._push_appearance()

    def _build_header(self) -> QWidget:
        header = QFrame(self)
        header.setObjectName("header")
        header.setFixedHeight(52)
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        row.addWidget(QLabel("模型", header))

        self.model_combo = QComboBox(header)
        self.model_combo.setMinimumWidth(320)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        row.addWidget(self.model_combo)

        self.refresh_button = QToolButton(header)
        self.refresh_button.setText("⟳")
        self.refresh_button.setToolTip("重新拉取本地模型列表")
        row.addWidget(self.refresh_button)

        self.new_session_button = QPushButton("＋ 新会话", header)
        # `kind` 是样式表的挂载点（见 ui/theme.qss 的 `QPushButton[kind="primary"]`）。
        # 用动态属性而不是逐个 setStyleSheet：应用级 QSS 跟着主题一起换，控件级
        # 样式表不会——写死在控件上的颜色会在切到浅色主题时留在原处。
        self.new_session_button.setProperty("kind", "primary")
        row.addWidget(self.new_session_button)

        # **弹簧负责吸掉富余，状态文字负责在挤的时候让步。** 其余控件都是固定尺寸，
        # 所以能伸缩的只有这两样：窗口宽裕时弹簧吃掉全部余量、标签拿它 `sizeHint`
        # 那么宽（实测 1900px 窗宽给 252px，状态文字全文可见）；窗口变窄时标签一路
        # 压到自己的 `minimumSizeHint`（24px）并加省略号。
        #
        # 这条分工**曾经是坏的**：`ElidedLabel` 那时候用 `Ignored` 策略，而它连
        # `minimumSizeHint` 一起无视，于是标签宽度按 0 参与分配——顶栏那段状态文字
        # 从来没显示过（见 ``ui/widgets.py`` 里 `ElidedLabel` 的类文档）。
        row.addStretch(1)

        # **用 `ElidedLabel` 而不是 `QLabel`。** 顶栏现在有五个按钮加一个模型下拉，
        # 状态文字再长就会被窗口右边**硬裁**掉——而裁掉的那一半没有任何提示。
        # `ElidedLabel` 加省略号，全文仍在 tooltip 与 `text()` 里。
        self.status_label = ElidedLabel("", header)
        self.status_label.setObjectName("status")
        row.addWidget(self.status_label)

        self.stop_button = QPushButton("■ 停止", header)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)

        self.memory_button = QPushButton("记忆", header)
        self.memory_button.setToolTip(
            "打开记忆浏览器：看库里有什么、每一条从哪来、这一轮为什么是它。"
            "可以改正文、遗忘、彻底删除。"
        )
        row.addWidget(self.memory_button)

        # 设置放在「记忆」旁边、账本按钮之前：账本按钮是**面板开关**，贴着它开的那
        # 一格才顺手；设置是弹窗，跟同样是弹窗的记忆归在一起。
        self.settings_button = QPushButton("设置", header)
        self.settings_button.setToolTip("外观（主色 / 明暗 / 壁纸）与后端（服务地址）")
        row.addWidget(self.settings_button)

        self.ledger_button = QPushButton("注入 ▸", header)
        self.ledger_button.setCheckable(True)
        self.ledger_button.setToolTip("查看这一轮往模型里注入了什么，以及为什么是这几条")
        row.addWidget(self.ledger_button)
        return header

    def _build_context_strip(self) -> QWidget:
        """顶栏下面那一条细的上下文指示灯。

        **它在顶栏之外单独一条，而不是塞进顶栏。** 顶栏那 52 像素是固定高的，
        挤进去要么把它撑高（三个面板的高度全变），要么让进度条细到看不见——而
        这条灯的价值恰恰是"不用去看也能感觉到快满了"。

        数值全部来自 :class:`~axo.session.assemble.AssemblyLedger`，
        与侧栏账本**同一份**（见 :meth:`_show_context` 与 :meth:`_render_ledger`
        的调用关系）。灯上写 90%、账本里写 40% 的界面无从排查，而且用户会先信灯。
        """
        strip = QFrame(self)
        strip.setObjectName("contextStrip")
        row = QHBoxLayout(strip)
        row.setContentsMargins(12, 0, 12, 4)
        row.setSpacing(8)

        self.context_bar = QProgressBar(strip)
        self.context_bar.setObjectName("contextBar")
        self.context_bar.setFixedHeight(6)
        self.context_bar.setTextVisible(False)
        self.context_bar.setRange(0, 100)
        self.context_bar.setToolTip(
            "这一轮请求的估算代价占可支配额度的比例。黄 = 快满了；红 = 已超出——"
            "那不等于请求会失败，而是服务端很可能**静默截掉最早的消息**，"
            "也就是人格锚所在的位置。"
        )
        row.addWidget(self.context_bar, 1)

        self.context_label = QLabel("上下文 —", strip)
        self.context_label.setObjectName("contextLabel")
        row.addWidget(self.context_label)

        # 开局就对齐一次：不调的话进度条用的是默认样式，那个样式**看起来也像
        # 一个状态**（Qt 默认是绿色填充），于是"还没有这一轮"会被画成"一切正常"。
        self._show_context(None)
        return strip

    def _show_context(self, assembly: Assembly | None) -> None:
        """把灯对齐到**给定的那一轮**账本；``None`` = 还没有这一轮。

        参数是显式的而不是内部去读 ``self._assembly``：三个复位点（新会话、准备
        失败、还没装配过）要的正是"显示空"，而它们手上并没有一份"空的装配结果"。
        让调用方把"没有这一轮"这件事明写出来，比让这个方法去猜 ``self._assembly``
        是不是还指着上一轮要可靠。

        **比值如实显示，进度条夹到 100%。** ``format_context`` 会写出
        ``上下文 12.0k / 8.9k``（135% 是事实），而进度条只是画法，夹紧它不影响
        那句话要传达的信息。
        """
        ledger = None if assembly is None else assembly.ledger
        total = 0 if ledger is None else ledger.total_tokens
        over = 0 if ledger is None else ledger.over_budget
        usable = self._budget.usable if self._budget is not None else 0

        state = context_state(total_tokens=total, usable=usable, over_budget=over)
        self.context_label.setText(format_context(total_tokens=total, usable=usable))
        if total > 0 and usable > 0:
            self.context_bar.setValue(min(100, round(total * 100 / usable)))
        else:
            self.context_bar.setValue(0)
        self.context_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {_CONTEXT_COLORS[state]}; }}"
        )

    def _build_session_pane(self) -> QWidget:
        pane = QFrame(self)
        pane.setObjectName("sessionPane")
        pane.setMinimumWidth(180)
        column = QVBoxLayout(pane)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)

        title = QLabel("会话", pane)
        title.setObjectName("paneTitle")
        column.addWidget(title)

        self.session_list = QListWidget(pane)
        self.session_list.setObjectName("sessionList")
        # 两行式（标题 + 时间）。**自绘 delegate，不用 setItemWidget**——后者会把
        # 每行变成真实子控件，于是选择点击被吃掉、选中高亮要自己画、
        # `currentItemChanged` 的语义也变得可疑。delegate 只改"怎么画"。
        self.session_list.setItemDelegate(SessionItemDelegate(self.session_list))
        column.addWidget(self.session_list, 1)

        self.delete_session_button = QPushButton("删除选中会话", pane)
        self.delete_session_button.setProperty("kind", "danger")
        column.addWidget(self.delete_session_button)
        return pane

    def _build_ledger_pane(self) -> QWidget:
        """注入账本面板。

        **它默认是收起的**（宽度 0），因为它是"要核对时才看"的东西。展开的宽度是
        固定的：账本是定宽两列的文本，一个会自动缩放的列会让对齐失效——而参差不齐
        的账本比没有账本更难看懂。

        内容有两段，来自两个不同的地方：上半段是装配器的账本（每层占了多少 token、
        哪几层被逐出），下半段是召回的全过程（哪些候选被淘汰了、为什么）。分成两段
        而不是合并，是因为它们回答的是两个问题：**额度花在哪了** 与 **库里还有什么**。
        """
        pane = QFrame(self)
        pane.setObjectName("ledgerPane")
        pane.setMinimumWidth(260)
        column = QVBoxLayout(pane)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)

        title = QLabel("注入账本", pane)
        title.setObjectName("paneTitle")
        column.addWidget(title)

        self.ledger_view = QTextBrowser(pane)
        self.ledger_view.setObjectName("ledgerView")
        self.ledger_view.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        self.ledger_view.setPlainText(_LEDGER_PLACEHOLDER)
        column.addWidget(self.ledger_view, 1)

        self._ledger_pane = pane
        pane.setVisible(False)
        return pane

    def _wire(self) -> None:
        self.chat_view.message_submitted.connect(self._on_submit)
        self.chat_view.persona_answered.connect(self._on_persona_answered)
        self.chat_view.tool_confirm_decided.connect(self._on_tool_confirm)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.refresh_button.clicked.connect(self._refresh_models)
        self.new_session_button.clicked.connect(self._new_session)
        self.stop_button.clicked.connect(self._on_stop)
        self.ledger_button.toggled.connect(self._on_toggle_ledger)
        self.memory_button.clicked.connect(self._on_open_memory)
        self.settings_button.clicked.connect(self._on_open_settings)
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.delete_session_button.clicked.connect(self._on_delete_session)

        # **「两侧一起跟」的那一根线。** 系统明暗变了就重算一次——网页那一侧由
        # :meth:`_apply_theme` 顺带推过去，所以两侧永远是同一次判定的产物。
        # 连的是 app 级的 `styleHints()`（它是每个应用一份的单例），不是窗口的。
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_os_color_scheme_changed)

        quit_action = QAction("退出", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        # **连 ``_quit`` 而不是 ``close``。** 有托盘时 ``close`` 只是收起来，于是
        # Ctrl+Q 这个键会从"退出"变成"藏起来"——一个快捷键悄悄换了含义，而用户
        # 按它正是因为想结束进程。
        quit_action.triggered.connect(self._quit)
        self.addAction(quit_action)

    # ── 会话 ────────────────────────────────────────────────────────────────

    def _reload_sessions(self, select: str | None = None) -> None:
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in self._store.list_sessions():
            item = QListWidgetItem(session.describe())
            item.setData(Qt.ItemDataRole.UserRole, session.session_id)
            # 两个自定义角色是**加法**：`DisplayRole`（上面那行 ``describe()``）原样
            # 不动，因为 `tools/_smoke_window.py` 与若干测试靠 `item.text()` 读它。
            # delegate 只把这两个角色画成两行，不改"这一项是什么"。
            title, stamp = session.describe_parts()
            item.setData(SESSION_TITLE_ROLE, title)
            item.setData(SESSION_TIME_ROLE, stamp)
            item.setToolTip(
                f"{session.session_id}\n建立时模型：{session.model_ref or '（未记录）'}"
            )
            self.session_list.addItem(item)
            if select and session.session_id == select:
                self.session_list.setCurrentItem(item)
        self.session_list.blockSignals(False)

    def _new_session(self) -> None:
        if self._streaming:
            return
        self._session = Session(session_id=new_session_id(), model_ref=self._current_model_id())
        self.chat_view.clear()
        self.session_list.setCurrentItem(None)
        self.chat_view.set_status("新会话")
        self._clear_round_state()
        self._push_context_to_memory_window()

    def _on_session_selected(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if current is None or self._streaming:
            return
        session_id = current.data(Qt.ItemDataRole.UserRole)
        self._session = self._store.load(session_id)
        self.chat_view.load_history(display_rows(self._session.messages))
        self.chat_view.set_status(f"已载入会话 {session_id}")
        # 记忆窗口要按会话区分一条早退路径：属于当前会话的记忆会被召回当作"逐字稿
        # 里已经有的内容"排除。推晚了的话，它给的解释指向**上一个**会话。
        self._push_context_to_memory_window()

    def _on_delete_session(self) -> None:
        item = self.session_list.currentItem()
        if item is None:
            return
        session_id = item.data(Qt.ItemDataRole.UserRole)
        answer = QMessageBox.question(
            self,
            "删除会话",
            f"删除会话 {session_id}？\n\n只删除聊天记录。记忆库里的条目不受影响。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._store.delete(session_id)
        if session_id == self._session.session_id:
            self._new_session()
        self._reload_sessions()

    # ── 模型 ────────────────────────────────────────────────────────────────

    @property
    def _host(self) -> str:
        """本机 Ollama 地址。

        **只读，值取自 :attr:`_backend_settings`。** 从两个字段改成属性，是因为
        后端设置这一份现在有五个字段（provider / host / allow_remote /
        cloud_base_url / api_key），而"现在是哪一个后端"必须由**同一份**回答：
        ``_on_backend_changed`` 拿到的整份与这里留的那份一旦各记各的，症状就是
        "设置里改了、按了应用、行为没变"，或者更坏的——对话走了云端而嵌入还照着
        一个已经改掉的地址找 Ollama。
        """
        return self._backend_settings.host

    @property
    def _allow_remote(self) -> bool:
        """是否允许连非本机地址。**这是 ``llm/ollama.py`` 那道护栏的开关**，
        不勾它时连非本机 host 会当场抛错——所以它必须与 :attr:`_host` 同源。"""
        return self._backend_settings.allow_remote

    def _last_model_key(self) -> str:
        """上次选的模型存在哪个键上。**按后端类型分键。**

        单键不会串（``findData`` 找不到返回 -1，是安全的），但**有损**：切到云端
        会把你原来选的 ``qwen3-4b`` 覆盖成 ``deepseek-chat``，切回本机时下拉框
        停在第一个模型上——一个"我的选择丢了"的抱怨，而配置其实一个字没动。
        """
        return (
            "backend/last_model_cloud" if self._backend_settings.is_cloud else "backend/last_model"
        )

    def _current_model_id(self) -> str:
        """这一轮的模型标识，进会话的 ``model_ref``。**前缀按后端类型分。**

        与 ``<backend>.model_id`` 的拼法一致（``ollama/…`` / ``openai-compat/…``），
        否则翻旧会话时看不出那句回复出自哪一类后端——而这正是这个字段存在的理由。
        """
        name = self.model_combo.currentData()
        if not name:
            return ""
        prefix = (
            CLOUD_MODEL_ID_PREFIX if self._backend_settings.is_cloud else OLLAMA_MODEL_ID_PREFIX
        )
        return f"{prefix}/{name}"

    def _refresh_models(self) -> None:
        if self._model_thread is not None and self._model_thread.isRunning():
            return
        thread = QThread(self)
        # 整个 ``BackendSettings`` 交给 worker，**不是拆开的 host + allow_remote**：
        # 云端要知道 base_url + key（两个字段装不进一个 ``host``），而"这一次清单是
        # 从哪儿拉的"与"当前后端是什么"必须是同一个事实——分开传就有"传了云端地址、
        # provider 还是 ollama"这种半路的组合，症状是去问一个不存在的本机服务。
        #
        # ``allow_remote`` 也仍然一路传到底：llm/ollama.py 的 ``_client`` 对非本机
        # 地址会直接抛错，所以漏传任何一处，那条路径的症状都是"某个功能莫名说连不上，
        # 而别的功能好好的"——设置窗口里勾了框、模型列表能出来、一对话就报错。
        worker = ModelListWorker(self._backend_settings)
        worker.loaded.connect(self._on_models_loaded)
        worker.failed.connect(self._on_models_failed)
        # ★ 这一句决定线程是"跑一次就走"还是"一直空转"。`started` 直接调用 `run()`，
        # 而 QThread 要等 `run()` **返回之后**才进入自己的 `exec()`——没人叫它退出，
        # 事件循环就永不结束，`finished` 永不发射，于是 `deleteLater` 也永不发生。
        #
        # 症状分两层，都很会误导人：运行期看不出任何异常（空转的线程不占 CPU、不报
        # 错），只在这一个地方现形——**解释器收尾时退出码是 `0xC0000409`**
        # （Windows fast-fail），没有回溯、没有 stderr，看起来像 QtWebEngine 的锅。
        # 实测：`_smoke_recall.py both` 报的 127 就是这么来的，而四个脚本一直带着
        # 这个退出码，只是此前没人看 `$?`。
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="模型列表线程", slot="model")
        if self._backend_settings.is_cloud:
            # 云端下**必须另起一趟**问本机 Ollama 要嵌入模型，见那个方法的说明。
            self._refresh_embedding_models()
        thread.start()

    def _refresh_embedding_models(self) -> None:
        """**云端模式下额外跑一趟本机 Ollama**，只为了嵌入模型的清单。

        这是整块接线里最容易漏的一处，因为漏了**不报错、不崩、什么都不显示**：
        :meth:`_on_models_loaded` 以前把同一份 ``models`` 既灌对话下拉框、又交给
        :meth:`_start_memory_setup`。云端那一份是 ``/models`` 来的，里面没有嵌入模型
        （多数厂商根本没有嵌入端点），于是 ``pick_embedding_model`` 恒返回空串，
        ``MemorySyncWorker`` 回一句"没有可用的嵌入模型"——**语义召回静默退化成纯
        词法**，而用户看到的只是"记忆检索老是查不到东西"（见 P3a 那次同形的病史：
        纯层全绿不等于接线在）。

        ``host`` / ``allow_remote`` **照样传下去**：切到云端只是"对话从哪来"变了，
        本机那两个字端一直留在设置里（见 ``BackendPage.current``），所以自建内网
        推理机的人切了云端之后，嵌入不该被指回 ``127.0.0.1``。
        """
        if self._embedmodels_thread is not None and self._embedmodels_thread.isRunning():
            return
        thread = QThread(self)
        worker = ModelListWorker(replace(self._backend_settings, provider=PROVIDER_OLLAMA))
        worker.loaded.connect(self._on_embedding_models_loaded)
        worker.failed.connect(self._on_embedding_models_failed)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="嵌入模型清单线程", slot="embedmodels")
        thread.start()

    @Slot(list)
    def _on_embedding_models_loaded(self, models: list[dict[str, Any]]) -> None:
        """本机那一趟的清单回来了：喂嵌入模型，对话下拉框不碰。"""
        self._start_memory_setup(models)

    @Slot(str)
    def _on_embedding_models_failed(self, detail: str) -> None:
        """本机那一趟没拉到清单（Ollama 没起来、地址不对）。

        **绝不走** :meth:`_set_backend_down`：对话在云端是好的，把整条后端标成
        "不可用"会让人去查云端那把 key，而问题在本机。

        绕开 :meth:`_start_memory_setup` 直接落到同一个收口，是因为那句话里的建议
        必须与前提对得上："装一个嵌入模型再点 ⟳"在 Ollama 没起来时是一句误导——
        用户照着装完，还是连不上。这里把原因一起说出来。
        """
        self._on_embedding_ready(
            EmbeddingSetup(note=f"没有可用的嵌入模型（{detail}），记忆只用语汇匹配")
        )

    @Slot(list)
    def _on_models_loaded(self, models: list[dict[str, Any]]) -> None:
        cloud = self._backend_settings.is_cloud
        previous = self._settings.value(self._last_model_key(), "")
        chat_models = [entry for entry in models if is_chat_capable(entry)]
        # 服务端报的上下文窗口先存下来。``OpenAICompatChatBackend`` 的构造**会去问
        # 一次端点**，而这里是界面线程——一次网络往返就能让窗口冻住几百毫秒到十秒，
        # 形状正是 "启动时窗口出现慢 5 秒" 那种"明明是网络问题却看起来像程序卡死"。
        # 而 ``/models`` 已经把窗口一起给出来了，没有再问一遍的道理。
        self._model_windows = {
            str(entry["name"]): int(entry["context_window"])
            for entry in models
            if isinstance(entry.get("context_window"), int)
        }

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for entry in chat_models:
            # 显示 short_name：`:latest` 是 Ollama 在没写标签时自动补的，摆在下拉框里
            # 只会让人以为模型名里真带这一段。data 仍存服务端的权威名字。
            label = entry.get("short_name") or entry["name"]
            if entry.get("parameter_size"):
                label += f"  · {entry['parameter_size']}"
            if entry.get("quantization"):
                label += f" {entry['quantization']}"
            self.model_combo.addItem(label, entry["name"])
        self.model_combo.blockSignals(False)
        if not cloud:
            # **只在不是云端时从这里挑嵌入模型。** 这一份清单的来源是"当前后端"，
            # 本机时它就是 Ollama 的清单；云端时它里面一个嵌入模型都没有，交给
            # :func:`pick_embedding_model` 只会拿到空串，然后把已经接好的语义通道
            # 拆掉。云端那一侧由 :meth:`_refresh_embedding_models` 单独问本机。
            self._start_memory_setup(models)

        if not models:
            if cloud:
                # 云端返回空清单**不是"没有模型"那么简单**：多数情况是这把 key 的
                # 权限不够、或者余额/配额把它挡在外面，而接口照样回 200。
                self._set_backend_down(
                    "云端服务一个模型都没返回。检查这把 API key 有没有这个权限，再点 ⟳"
                )
            else:
                self._set_backend_down("Ollama 在运行，但一个模型都没有。先装一个模型再点 ⟳")
            return
        if not chat_models:
            # 只装了嵌入模型的情况。此时列表里确实没有可对话的东西，但**不能**把
            # 嵌入模型塞进去凑数——那样错误会推迟到第一次提问时才暴露，且表现为
            # 一个与服务端语义无关的报错。把原因直接说清楚。
            #
            # 云端走不到这一支（云端条目一律打 ``completion``，见 ``_cloud_entry``），
            # 但分支得有个说得通的下文，不然将来真走到这里时是一句空话。
            if cloud:
                self._set_backend_down(
                    f"云端服务返回了 {len(models)} 个模型，但没有可对话的。"
                    f"换一把 API key 或换一个服务地址再点 ⟳"
                )
            else:
                self._set_backend_down(
                    f"Ollama 里有 {len(models)} 个模型，但没有可对话的（全是嵌入模型）。"
                    f"装一个对话模型再点 ⟳"
                )
            return

        index = self.model_combo.findData(previous)
        if index < 0 and cloud:
            # 没记住上次选的（第一次切到云端、或上次那个模型已经不在了）时，云端
            # **不能认第一条**：:func:`list_remote_models` 按名字排序，而一个云端
            # 账户上几十上百个模型是常态（OpenRouter 那种），"第一条"于是接近随机
            # ——它还是用户第一次看到的那个模型。挑我们自己的默认名，挑不到才认第一条。
            index = self.model_combo.findData(pick_cloud_chat_model(models))
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self._on_model_changed()
        hidden = len(models) - len(chat_models)
        suffix = f"（另有 {hidden} 个嵌入模型未列出）" if hidden else ""
        # 文案按后端类型分。**这一处漏掉的症状最难查**：状态栏写着"Ollama 就绪"、
        # 请求却发去了 deepseek，于是用户按那句提示去查本机 Ollama，而它一点问题没有。
        head = "云端服务就绪" if cloud else "Ollama 就绪"
        self._set_backend_up(f"{head} · {len(chat_models)} 个对话模型{suffix}")

    @Slot(str)
    def _on_models_failed(self, detail: str) -> None:
        """这一趟清单没拉到。

        云端下**先把兜底预设摆进下拉框**（本机不摆：本机"没拉到清单"意味着 Ollama
        没起来，摆一份模型名只会让人以为它起来了）。理由是这个位置很常见——地址对了、
        key 也填了，只是这一趟网络不通或者端点临时抽风，而**一个空的对话下拉框会让人
        以为"云端一个模型都没有"**，于是去查一件根本不存在的事。

        预设只是候选，**不冒充"服务端确实有这些"**：状态行与对话区照常写真正的原因，
        用户一看就知道要修什么。
        """
        if self._backend_settings.is_cloud and self.model_combo.count() == 0:
            # **只在它是空的时候才灌。** 每次失败都清空重灌的话，一次偶发的网络抖动
            # 会把用户刚选中的模型连同整份清单一起抹掉。
            self.model_combo.blockSignals(True)
            for entry in fallback_models():
                self.model_combo.addItem(entry.get("short_name") or entry["name"], entry["name"])
            self.model_combo.blockSignals(False)
        self._set_backend_down(detail)

    def _start_memory_setup(self, models: list[dict[str, Any]]) -> None:
        """把语义通道接上：挑嵌入模型 → 后台探测维度 → 补向量。

        **没挑到模型也照样起这个 worker**，让它发一条"没有可用的嵌入模型"回来。少了
        这条路径，用户看到的是"记忆检索老是查不到东西"，而真正的原因（库里根本没有
        向量）在界面上没有任何痕迹——这正是 :class:`EmbeddingSetup` 存在的理由。
        """
        if self._setup_thread is not None and self._setup_thread.isRunning():
            return
        thread = QThread(self)
        worker = MemorySyncWorker(
            store=self._memory,
            model=pick_embedding_model(models),
            host=self._host,
            allow_remote=self._allow_remote,
        )
        worker.ready.connect(self._on_embedding_ready)
        worker.ready.connect(thread.quit)
        self._track(thread, worker, what="记忆同步线程", slot="setup")
        thread.start()

    @Slot(object)
    def _on_embedding_ready(self, setup: EmbeddingSetup) -> None:
        """语义通道接好了（或者接不上）。**两种情况都必须说话。**"""
        self._embedding = setup.backend
        self._embedding_model = setup.backend.model if setup.backend is not None else ""
        self._embedding_note = setup.note
        # 换嵌入模型 = 换一个语义空间，记忆窗口要按**空间**逐条标注"这条有没有
        # 向量"。推晚了它会把新空间按旧后端标注，而那看起来完全正常。
        self._push_context_to_memory_window()
        if setup.note:
            self.chat_view.set_status(setup.note)
        if setup.backend is None:
            # 只有"这一侧彻底没有语义通道"才写进对话区。补向量失败那种情况会在下一轮
            # 由 :class:`AssemblyWorker` 重试并把结果写进注入账本——同一条信息写两遍
            # 会让账本看起来比实际更糟。
            self.chat_view.append_message(
                "system",
                f"**记忆只用语汇匹配**\n\n{setup.note}\n\n"
                f"装一个嵌入模型（例如 `ollama pull nomic-embed-text`）再点 ⟳ 刷新，"
                f"语义召回就会接上；库里的记忆一条都没少。",
            )

    def _on_model_changed(self, _index: int = -1) -> None:
        name = self.model_combo.currentData()
        if not name:
            self._backend = None
            return
        self._settings.setValue(self._last_model_key(), name)
        if self._backend_settings.is_cloud:
            try:
                self._backend = OpenAICompatChatBackend(
                    model=str(name),
                    base_url=self._backend_settings.cloud_base_url,
                    api_key=self._backend_settings.api_key,
                    # 清单里已经带了服务端报的窗口就用它，**不问第二遍**：
                    # ``context_window=None`` 会让构造函数发一次 ``GET /models``，
                    # 而这里在界面线程上。
                    context_window=self._model_windows.get(str(name)),
                )
            except (ChatBackendError, ValueError) as exc:
                # 地址不可信（非 https 且非本机）时构造函数会当场拒。**绝不能让它
                # 抛在槽里**：PySide6 会吞掉异常，于是 ``self._backend`` 停在**上一个**
                # 对象上——下一轮提问发给的是一个用户已经改掉的地址，而界面上没有
                # 任何痕迹。置空并说出来，与其它"后端不可用"走同一条路。
                self._backend = None
                self._set_backend_down(str(exc))
                return
        else:
            self._backend = OllamaChatBackend(
                model=str(name), host=self._host, allow_remote=self._allow_remote
            )
        self.chat_view.set_status(
            f"{self._backend.model_id} · 上下文窗口按 {self._backend.context_window} token 计"
        )

    def _set_backend_up(self, text: str) -> None:
        self.status_label.setText(f"● {text}")
        self.status_label.setProperty("state", "ok")
        self._restyle(self.status_label)
        self.chat_view.set_status(text)
        self._note_backend(text)

    def _set_backend_down(self, detail: str) -> None:
        self.status_label.setText("● 后端不可用")
        self.status_label.setProperty("state", "error")
        self._restyle(self.status_label)
        self.chat_view.set_status(f"后端不可用：{detail}")
        self.chat_view.append_message("system", f"**后端不可用**\n\n{detail}")
        # 这两行回填的是**当前生效的地址**上一次探测的结论——四个调用点全在
        # `_on_models_loaded` / `_on_models_failed` 里，也就是探测的收口处。
        # 所以后端页那一行永远不会停在一句没有下文的"正在重新探测…"上。
        #
        # 它会盖掉「测试连接」写的那一行，这是对的：那一行测的是输入框里**还没应用**
        # 的地址，而这里说的是已经生效的那个。两者不一致时，用户要看的显然是后者。
        self._note_backend(detail)

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        """动态属性变了之后必须让样式表重新求值，否则颜色不会跟着变。"""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ── 对话 ────────────────────────────────────────────────────────────────

    def _route(self, text: str) -> RoutedTurn:
        """这一轮的语域、要不要工具、要不要反问。**纯决策：不碰界面，不发请求。**

        单独一个方法，是为了让它**测得到**。``_on_submit`` 身上挂着 QThread、
        WebEngine 与后端连接，离屏下构造不出来（见 ``test_ui_tools_wire`` 的说明：
        一渲染就把进程硬杀掉），而"规则层到底有没有接上"这件事不该因为构造不出窗口
        就测不到——P3a 恰恰是这么漏掉的：:func:`~axo.agent.router.route_turn`
        有 120 条语料、有实测基线、进 CI 门槛，**却没有一个生产消费者**，
        三个只在 ``bench/`` 里跑绿的文件足以让人以为这条线通了。

        ``self._router_state`` **会被就地修改**（迟滞就是靠它跨轮记事的），所以整个
        窗口只有这一处调路由——多一个调用点就多一个"状态被推着走"的地方。
        """
        return route_turn(text, self._router_state)

    def _on_submit(self, text: str) -> None:
        if self._streaming:
            # 生成中还能提交，说明界面状态与后端脱节了；直接忽略比排队更安全
            return

        # 斜杠命令**在"有没有后端"之前处理**：``/help`` 与裸 ``/task`` 都不需要
        # 发请求，把它们卡在"还没选模型"后面会让用户在 Ollama 没起来时连帮助都
        # 看不到。
        #
        # **路由要在这一行就发生**，而不是等到 ``_on_assembly_ready``：这一轮的语域
        # 决定了 L1 环境槽里写不写"这一轮没有工具可用"，而环境是随装配一起进上下文的。
        # 那时候再判，模型已经拿到一份说错了自己能力的上下文。
        routed = self._route(text)
        parsed = routed.parsed
        if parsed.is_help:
            self.chat_view.append_message("assistant", HELP_TEXT)
            return
        if parsed.is_bare_switch and parsed.forced is not None:
            # 裸切换：**这一轮没有正文可送**，所以只改状态、不发请求。发出去的话
            # 模型会拿到一条空消息然后答非所问，而那个症状看起来像模型坏了。
            #
            # 状态不用在这里赋值：上面 :func:`route_turn` 里的
            # :meth:`RegisterState.advance` 见到 ``SLASH`` 会**无视迟滞**直接覆盖
            # （显式指定优先），所以"用户刚打了 /chat"不会被上一次的 TASK 拖住。
            # 自己再赋一次反而会让两处对"当前语域"各有一个说法。
            self.chat_view.append_message(
                "system", f"已切到{'任务' if parsed.forced is Register.TASK else '闲聊'}语域。"
            )
            return
        # ``/task 正文`` 只影响这一轮（``sticky`` 为假），所以 ``advance`` 拿到的
        # 就是"仅此一轮"的结论，状态机那边不会被改写。
        self._turn_register = routed.register
        self._turn_uses_tools = routed.uses_tools
        self._turn_needs_clarification = routed.wants_clarification
        text = parsed.text
        if not text:
            return

        if self._backend is None:
            self.chat_view.append_message("system", "还没选模型。先启动 Ollama 并点 ⟳ 刷新。")
            return

        # 会话表头的 ``model_ref`` **只在建文件那一刻写一次**（见 ``session/store.py``
        # 的 ``append``），而会话是启动时建的、那时模型列表还没异步拉回来——于是表头
        # 记了个空串，且永久为空。这里补：第一次真的要发请求时模型已经定了。
        # 补晚一步（比如放在 ``_append`` 之后）没用，因为那一行已经落盘了。
        if not self._session.model_ref:
            self._session.model_ref = self._backend.model_id

        # 历史要在**追加本轮输入之前**取：``assemble`` 的 ``history`` 不含当前这一轮。
        # 写成"追加之后再切片"也能work，但那样"history 里到底有没有当前这句"就取决
        # 于切片写没写对，而写错的表现是当前输入出现两次——一个不报错、只是让模型
        # 看到自己把问题重复了一遍的错误。
        history = list(self._session.messages)
        self._append(Message(role="user", content=text))
        self.chat_view.append_message("user", text)

        self._budget = TokenBudget(window=self._backend.context_window)
        self._pending_text = text
        self._streaming = True
        self._preparing = True
        self._cancel_requested = False
        self.chat_view.set_busy(True)
        self.stop_button.setEnabled(True)
        self.chat_view.set_status("正在检索记忆…")

        thread = QThread(self)
        worker = AssemblyWorker(
            store=self._memory,
            budget=self._budget,
            history=history,
            current=text,
            environment=build_environment(
                model_id=self._backend.model_id,
                tools_declared=self._turn_uses_tools,
                needs_clarification=self._turn_needs_clarification,
            ),
            embedding=self._embedding,
            session_id=self._session.session_id,
            # **config 每轮现造**（见 :meth:`_recall_config`）：所以开关一翻，
            # 下一轮就是新的，没有陈旧问题。图对象反过来——它只 load/建一次，
            # 开关只决定用不用它，所以"开着开关但还没有图"是可能的，而那件事由
            # 账本里那句 note 说出来，不静默。
            recall_config=self._recall_config(),
            graph=self._graph,
        )
        worker.ready.connect(self._on_assembly_ready)
        worker.failed.connect(self._on_prepare_failed)
        # 终局信号顺手收掉线程的事件循环。**不接这个，线程会一直活着**：``run()``
        # 是 ``started`` 直接调用的，跑完之后 ``QThread`` 才进入自己的 ``exec()``，
        # 于是那个空转的事件循环不会自己结束，``finished``／``deleteLater`` 也就
        # 永远不会触发——每提一次问就留下一个线程。
        worker.ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="装配线程", slot="prepare")
        thread.start()

    @Slot(object, object)
    def _on_assembly_ready(self, assembly: Assembly, ledger: Any) -> None:
        """装配好了，**这里才发请求**。"""
        if self.sender() is not self._prepare_worker:
            # 上一轮被取消之后仍在跑的装配线程。它送来的是一份**过期**的结果，收下
            # 就等于把上一轮的召回接到这一轮的请求上——而这一轮会有一个新的 worker，
            # ``_cancel_requested`` 那时已经复位成假，光靠那个布尔拦不住。
            return
        self._preparing = False
        if not self._streaming or self._cancel_requested:
            # 用户在准备期间按了停止。准备阶段还没发出去任何东西，所以那次停止是
            # 立即生效的——不需要"说完这句再停"那套。
            return
        if self._backend is None or self._budget is None:
            self._on_prepare_failed("后端或额度在准备期间消失了")
            return

        self._assembly = assembly
        self._recall_ledger = ledger
        self._render_ledger()
        # 记忆窗口靠这个拿账本（它可能还不存在——没人听就等于没发）。**注意发出
        # 去的是这一轮的对象本身，不是 self._recall_ledger 这个可变名字。**
        self.recall_updated.emit(ledger)

        # ``max_tokens`` 必须等于装配器预留的那个数。不传的话 Ollama 会用同一个
        # num_ctx 去生成——输出可以一直涨到把输入挤掉，而**挤掉的是最早的消息，
        # 也就是人格锚的位置**。装配器算的是"输入能占多少"，只有把预留量真的发给
        # 服务端，那个数才不是账面上的。
        #
        # 任务语域走另一条路：请求由工具循环**每一轮现拼**（它要在尾部追加工具
        # 消息、还要在到顶那轮把 ``tools`` 去掉），所以这里只把装配好的消息交给它。
        #
        # 判据是 ``_turn_uses_tools``（路由的结论）而不是 ``_turn_register is TASK``：
        # 歧义那一轮的语域也是 ``CHAT``，但那不是"用户想闲聊"，而是"没判准，什么都
        # 不该动"——两条路的语域相同、结论相同（都不给工具），所以这里用哪个都不改变
        # 行为，但用路由给的结论意味着**将来改那条规则只需要改一处**。
        if self._turn_uses_tools:
            registry = self._tools()
            if len(registry):
                self._start_agent_turn(assembly, registry)
                return
            # 一个工具都没有的任务轮——**说明为什么**再按普通对话走。默默降级的话，
            # 用户看到的是"我让它查一下，它却什么都没查"，而没有任何一处提到
            # 工具集是空的。
            self.chat_view.append_message("system", "这一轮没有可用的工具，按普通对话处理。")

        request = ChatRequest(
            messages=list(assembly.messages), max_tokens=self._budget.reserve_output
        )
        thread = QThread(self)
        worker = ChatWorker(self._backend, request)
        worker.chunk.connect(self.chat_view.push_token)
        worker.finished.connect(self._on_reply_finished)
        worker.failed.connect(self._on_reply_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="生成线程", slot="chat")
        self.chat_view.set_status("正在生成…")
        thread.start()

    def _tools(self) -> ToolRegistry:
        """这一轮可用的工具集。

        **每次重建，不缓存。** 工具集取决于"可读目录"这个设置项，而设置可以在两次
        提问之间改。缓存的话要配一个失效钩子，而漏掉任何一处调用点的症状是"改了设置
        但工具还是老样子"——七个轻量对象的重建成本远比那个 bug 便宜。

        ``read_roots`` 为空时 :func:`~axo.agent.catalog.default_tools`
        **不登记** ``read_file``：看不到它，模型就不会去试（省一段 prompt，也省一次
        必然失败的调用）。三个过闸门的工具在登记上不分条件——要不要用户点头是
        工具自己的 ``requires_approval``，闸门读那一处，这里不重复判断。
        """
        return ToolRegistry(default_tools(read_root=self._read_roots))

    def _start_agent_turn(self, assembly: Assembly, registry: ToolRegistry) -> None:
        """起一轮带工具的请求。"""
        assert self._backend is not None and self._budget is not None
        ctx = ToolContext(
            store=self._memory,
            session_id=self._session.session_id,
            # 轮次序号与 ``_remember`` 用的是**同一个算法**，不是巧合：工具写的记忆
            # 与回合末写的记忆靠 ``source_ref`` 区分，而两者都对不上就无法溯源。
            turn_seq=sum(1 for message in self._session.messages if message.role == "user"),
            writer=self._writer,
            # 提醒簿与可读目录都从主窗口这一份来：工具自己不该知道"应用的数据目录
            # 在哪儿"，那是 :mod:`axo.paths` 与设置的事。
            reminders=self._reminders,
            fired=self._reminders_fired,
            cancellations=self._reminders_cancelled,
            read_root=self._read_roots,
        )
        thread = QThread(self)
        worker = AgentWorker(
            backend=self._backend,
            messages=list(assembly.messages),
            registry=registry,
            ctx=ctx,
            budget=self._budget,
            meter=TokenMeter(),
            tools_ceiling=layer_ceiling(self._budget, Layer.TOOLS),
        )
        worker.chunk.connect(self.chat_view.push_token)
        worker.finished.connect(self._on_reply_finished)
        worker.finished.connect(self._on_tool_round_finished)
        worker.failed.connect(self._on_reply_failed)
        worker.round_started.connect(self._on_tool_round_started)
        worker.tool_finished.connect(self._on_tool_finished)
        worker.approval_needed.connect(self._on_approval_needed)
        worker.messages_appended.connect(self._on_tool_messages)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="任务线程", slot="agent")
        self.chat_view.set_status("正在处理…")
        thread.start()

    @Slot(str)
    def _on_prepare_failed(self, detail: str) -> None:
        """准备阶段失败。

        **不退化成"不带记忆直接发"**。装配失败意味着预算或存储出了问题，而那不是
        一次网络抖动——悄悄降级会让用户在一个坏掉的预算上继续对话，而症状是几轮之后
        模型突然开始忘事。说清楚比蒙混过去好。
        """
        self.chat_view.append_message("system", f"**准备上下文失败**\n\n{detail}")
        self.chat_view.set_status("准备失败")
        # 这一轮没有账本，所以三处一起复位（灯、侧栏、记忆窗口）。只复位灯的话，
        # 侧栏还留着上一轮的数字，而状态栏说"准备失败"——用户会以为这一轮还是
        # 发出去了。
        self._clear_round_state()
        self._reset_generation_state()

    def _clear_round_state(self) -> None:
        """把「这一轮」的三样东西一起清掉：装配结果、召回账本、指示灯。

        **一起清**是重点。上一轮的装配结果与账本是同一件事的三个说法（灯上的比值、
        侧栏的文本、记忆窗口里逐条的归因），留一个旧的就是让两个地方对同一件事给出
        不同答案——而这正是 §11.7 要避免的那类矛盾：没有症状，只有"看起来有点不对"。
        """
        self._assembly = None
        self._recall_ledger = None
        self._show_context(None)
        self.ledger_view.setPlainText(_LEDGER_PLACEHOLDER)
        self.recall_updated.emit(None)

    @Slot(str, dict)
    def _on_reply_finished(self, text: str, statistics: dict[str, Any]) -> None:
        interrupted = bool(statistics.get("interrupted"))
        self.chat_view.end_message()
        self._append(
            Message(
                role="assistant",
                content=text,
                interrupted=interrupted,
                prompt_tokens=statistics.get("prompt_tokens"),
                completion_tokens=statistics.get("completion_tokens"),
            )
        )
        # ``tool_trace`` 只有任务那一侧才有（``ChatWorker`` 的统计里没有这个键），
        # 所以取缺省。**装的是哪套策略决定它有没有用**——条目模式根本不看它。
        self._remember(text, interrupted=interrupted, tool_trace=statistics.get("tool_trace", ()))
        self._reset_generation_state()
        if interrupted:
            self.chat_view.set_status("已停止")
        else:
            prompt_tokens = statistics.get("prompt_tokens")
            completion_tokens = statistics.get("completion_tokens")
            if prompt_tokens or completion_tokens:
                self.chat_view.set_status(
                    f"输入 {prompt_tokens or '?'} token · 输出 {completion_tokens or '?'} token"
                )

    def _remember(
        self, reply_text: str, *, interrupted: bool, tool_trace: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        """把这一轮写进记忆库。

        **失败要说出来**，不能吞：这个项目的全部意义是"记忆不会丢"，而唯一比丢记忆
        更糟的是丢了却不知道——用户会以为它记住了，直到某天问起来答不出。所以这里
        接住异常、显示在对话里，而不是让它顺着 Qt 的信号链消失。

        轮次序号取自**这个会话里已经有几条用户消息**，不是一个进程内的计数器：后者
        在"载入旧会话继续聊"时会从 0 重新开始，于是 ``turn://<会话>/1`` 指向两条
        不同的记忆——而那个指针正是将来去重要用的键。

        ``reply_text`` 与 ``tool_trace`` 是 P3b 兑现的那两样：
        :class:`~axo.memory.policy.TaskPolicy` 拿前者当**正文**（被召回的是
        回答，不是用户那句要求）、拿后者当"这一轮是不是任务"的判据与来历。装的是条目
        模式时两者都被忽略——**那正是"开关是加法"**：打开任务记忆不会让任何一轮突然
        不再被记住。

        ``tool_trace`` 是摘录（见 :mod:`~axo.agent.trace`），空的元组表示
        这一轮没调过工具，不表示"不知道调没调过"。
        """
        if not self._pending_text:
            return
        turn_seq = sum(1 for message in self._session.messages if message.role == "user")
        try:
            written = self._writer.record_text(
                self._pending_text,
                reply_text=reply_text,
                session_id=self._session.session_id,
                turn_seq=turn_seq,
                interrupted=interrupted,
                tool_trace=tool_trace,
            )
        except Exception as exc:  # noqa: BLE001 — 见方法文档：这条异常必须让用户看见
            self.chat_view.append_message(
                "system", f"**记忆写入失败**\n\n{type(exc).__name__}：{exc}"
            )
            return
        if written:
            self.chat_view.set_status(f"已记下 {len(written)} 条记忆")

    @Slot(str)
    def _on_reply_failed(self, detail: str) -> None:
        self.chat_view.append_message("system", f"**请求失败**\n\n{detail}")
        self._reset_generation_state()
        self.chat_view.set_status("请求失败")

    # ── 任务语域特有的四件事 ────────────────────────────────────────────────
    #
    # 下面四个槽只接在 ``AgentWorker`` 上。``_on_reply_finished`` / ``_on_reply_failed``
    # 两个**共用**——那正是 ``AgentWorker`` 的信号与 ``ChatWorker`` 同形的目的。

    @Slot(int, int)
    def _on_tool_round_started(self, index: int, tools_offered: int) -> None:
        """新一轮开始：给上一轮的正文收尾。

        **不收尾的话，几轮的"边想边说"会连成一条气泡**，而它读起来像模型一口气说了
        三段互相矛盾的话——真正的分界（中间夹着工具结果）在文字里完全看不出来。

        第 0 轮不收：那时还没有上一轮。
        """
        if index:
            self.chat_view.end_message()
        self.chat_view.set_status(
            f"第 {index + 1} 轮" + ("" if tools_offered else "（收尾，不再调工具）")
        )

    @Slot(dict)
    def _on_tool_finished(self, payload: dict) -> None:
        """一次工具调用结束：把结果画进对话流。

        先 ``end_message()``：助手那句"我先查一下"与随后的结果是两件事，挤进同一条
        气泡会让回看时分不清哪句是模型说的、哪句是工具查到的。

        ``detail`` 只在成功时当页脚——它那时是账本摘要（这一轮召回为什么是这几条），
        对用户有信息量；失败时它是完整栈，而卡片上那点位置放不下，放上去也只会把
        真正该看的那句失败原因挤走。
        """
        name = payload.get("name") or "工具"
        if payload.get("denied"):
            title = f"已拒绝 · {name}"
        elif payload.get("ok"):
            title = f"工具结果 · {name}"
        else:
            title = f"工具失败 · {name}"
        meta: dict[str, Any] = {"title": title}
        if payload.get("ok") and payload.get("detail"):
            meta["footer"] = payload["detail"]
        if payload.get("clipped"):
            meta["footer"] = (meta.get("footer", "") + " （结果过长，进上下文的已截断）").strip()
        self.chat_view.end_message()
        self.chat_view.append_message("tool", payload.get("content") or "", meta)
        if payload.get("name") == "set_reminder":
            # 新登记的提醒要**立刻**改排。没有这一句，它得等当前这一觉睡满（最长
            # 60 秒）才被看见——而"我刚说了三十秒后提醒我"的人不会等满六十秒。
            # 登记是工具自己落的盘，走到这里时文件里已经有了。
            self._scheduler.reschedule()

    @Slot(str, str)
    def _on_reminder_fired(self, title: str, body: str) -> None:
        """到点了（或启动时收掉了一批错过的）：**两处都要说**。

        * **对话区一条**：用户回看时它在那儿。窗口可能是藏着的，而这条消息本来就
          是"发生过什么"的一部分。
        * **托盘一条气泡**：这是"现在就要知道"唯一的通道。窗口藏在托盘里时状态栏
          没人看得见，只写对话区等于没提醒。

        **不用模态框**：它挡住用户正在做的事，而这条消息要的是"知道一下"，不是
        "停下来处理"。收到这条消息时用户可能正在别的窗口里打字。
        """
        self.chat_view.append_message("system", f"**{title}**\n\n{body}")
        self._tray.show_message(title, body)

    @Slot(str, str, str)
    def _on_approval_needed(self, call_id: str, tool_name: str, summary: str) -> None:
        """有一个改状态的动作在等用户点头：画卡片。

        **状态栏也要说一句**：卡片出现在对话流中间，而用户可能正在往上翻。状态栏是
        唯一一处"现在轮到你"的提示。
        """
        if not self._streaming:
            # 已经复位过的那一轮：它的卡片画出来也没人接（闸门那边这一轮已经拆了）。
            return
        self.chat_view.end_message()
        self.chat_view.append_tool_confirm(
            call_id,
            tool_name,
            summary,
            {"timeoutMs": int(APPROVAL_TIMEOUT_S * 1000)},
        )
        self.chat_view.set_status(f"等待你确认：{tool_name}")

    @Slot(str, bool)
    def _on_tool_confirm(self, call_id: str, granted: bool) -> None:
        """用户点了「允许」/「拒绝」。**直连 worker，不走信号**——理由见 ``stop()``。

        两种结局都要说清楚：``accepted`` 为假表示这次点击**没有被采纳**（卡片过期、
        这一轮已经停止）。它与"用户拒绝了"是两回事，混成一句话就会出现"我以为我
        同意了"那种静默分歧。
        """
        worker = self._agent_worker
        accepted = worker.provide_approval(call_id, granted) if worker is not None else False
        self.chat_view.resolve_tool_confirm(call_id, granted=granted, accepted=accepted)
        if not accepted:
            self.chat_view.set_status("这张确认卡片已经失效了（这一轮已经结束或已在收尾）")

    @Slot(object, object)
    def _on_tool_messages(self, rows: object, no_replay: object) -> None:
        """把已经执行过的工具调用写进逐字稿。

        **落盘在这里，而且是一对一到就落**（不是攒到回合末）：用户中途按停止时，
        那些**已经真的执行过**的调用必须留下记录。一份"做过但没记"的逐字稿比少记
        几步严重得多——它是这个项目唯一不能有的那种失效。

        写的是 ``content``（截断后进对话的那份），与当时真的送进上下文的一致。

        ``no_replay``（``(tool_call_id, 原因)``）**必须和行一起落**，不能丢：它记的是
        "这一对怎么结束的"，而 L5 靠它决定要不要把这一对搬进后续轮次。丢掉的症状是
        逐字稿把一次拒绝记成一次正常结果，然后每轮都把它回放给模型看。
        """
        if not isinstance(rows, tuple):
            # 信号参数没有静态类型可依（``Signal(object, object)``），所以这里判一次
            # 形状。判错就丢掉而不是硬遍历：一段来路不明的行写进逐字稿就再也删不掉了。
            return
        # 原因的形状不对时按"都能回放"处理，而不是把整批丢掉：**行本身是真的**（那些
        # 调用确实执行过），丢了它们比多回放一对严重得多。这一路的默认值偏向留记录。
        reasons = no_replay if isinstance(no_replay, tuple) else ()
        for message in stored_rows(rows, no_replay=reasons):
            self._append(message)

    @Slot(str, dict)
    def _on_tool_round_finished(self, _text: str, statistics: dict[str, Any]) -> None:
        """``AgentWorker`` 的 ``finished`` 上挂的第二件事。

        ``_on_reply_finished`` 已经在同一个信号上跑过了（它按 ``ChatWorker`` 的契约
        复位状态、写助手那条消息），这里只补两件**任务语域独有**的：把
        ``unexecuted`` 说出来、把状态栏换成工具轮次的摘要。

        ``unexecuted`` **不许吞掉**（见 :class:`~axo.agent.loop.LoopDone`）：
        那是"模型还想调工具但这一轮已经没有工具可用了"的那几个名字，吞掉它就等于
        回到了 ``ChatWorker`` 被修掉的那个缺陷——静默丢弃。
        """
        unexecuted = statistics.get("unexecuted") or ()
        if unexecuted:
            self.chat_view.append_message(
                "system",
                "工具调用次数已经用完，模型还想调用："
                + "、".join(str(name) for name in unexecuted)
                + "。\n\n可以再发一句话让它接着做。",
            )
        if statistics.get("interrupted"):
            return  # ``_on_reply_finished`` 已经写了"已停止"，别盖掉它
        rounds = statistics.get("rounds")
        calls = statistics.get("tool_calls")
        if rounds:
            self.chat_view.set_status(f"{rounds} 轮 · 调用工具 {calls} 次")

    def _track(self, thread: QThread, worker: QObject, *, what: str, slot: str) -> None:
        """接管一对新起的 ``(线程, worker)``。

        **实现搬到了 :func:`axo.ui.thread_tracking.track`**，因为 P2 的
        记忆窗口也要起线程（线索召回），而那份方法体里有三条实测换来的约定
        （``Qt.DirectConnection``、亲和面留在界面线程、收尾用闭包而不是
        ``sender()``）。抄第二份的话，漏掉的那条不会报错——它只会让界面在每次网络
        请求期间冻住，或者让进程在某次退出时无声消失。**那里的事故史写得很细，
        改动前先读。**

        这一层留着是为了让四个调用点读起来不变，也为了
        ``tests/test_ui_worker_track.py`` 那五条钉住的行为有一个同名的入口。
        """
        track(self, thread, worker, what=what, slot=slot)

    def _reset_generation_state(self) -> None:
        """回到空闲。

        **三个 worker 与三个线程一起复位。** 只复位生成那一路的话，准备阶段失败之后
        ``_preparing`` 会一直是真，而它的唯一消费者是停止键——症状是"停止键从此
        永远只显示已取消"。任务那一路漏掉的话症状更偏：``_agent_worker`` 还指着上一轮
        的对象，于是一张迟到的确认卡片会点进一个已经结束的闸门（闸门会拒掉它，但
        状态栏给出的解释指向的是错误的原因）。

        这里把引用置空是安全的：worker 的 Qt 亲和面在界面线程（见 :meth:`_track`），
        所以这个 ``None`` 要么只是把引用计数交给 :func:`release` 那个闭包，要么就是
        一次同线程析构。**换掉 ``_track`` 里那条约定之前，先读那里的注释**——
        在旧形状（``moveToThread``）下，正是这一行的同类操作让 Qt ``abort()``。
        """
        self._streaming = False
        self._preparing = False
        self._cancel_requested = False
        self.chat_view.set_busy(False)
        self.stop_button.setEnabled(False)
        self._chat_worker = None
        self._chat_thread = None
        self._agent_worker = None
        self._agent_thread = None
        self._prepare_worker = None
        self._prepare_thread = None
        self._reload_sessions(select=self._session.session_id)

    @Slot()
    def _on_stop(self) -> None:
        """停止。**准备阶段与生成阶段的语义不一样。**

        生成阶段：请求已经发出去了，能让它停的只有"客户端断开"，而断开不可能瞬时
        完成——所以那里是"当前这句说完就停"。

        准备阶段：还什么都没发出去，**停止就是即时的**。没有"这一句"可言，所以这里
        直接把状态复位。后台那个装配线程照常跑完，结果被丢弃（见
        :meth:`_on_assembly_ready` 的发送者校验）——不掐断它不是因为掐不断，
        而是掐断一个正在算 cosine 的 numpy 调用只能靠杀线程，代价远大于让它跑完。
        """
        if self._preparing:
            self._cancel_requested = True
            self.chat_view.set_status("已取消（这一轮还没发出去）")
            self._reset_generation_state()
            return
        if self._agent_worker is not None:
            # **任务那一路必须先走这一支。** 它的 stop() 不只是翻布尔：还要
            # ``slot.cancel()`` 把可能正卡在等确认上的工作线程叫醒。少了那一下，
            # 这个线程要等满确认超时才醒，而收尾只等 3 秒——超时后进程继续退出，
            # 撞上 Windows fast-fail（0xC0000409，无回溯、无 stderr）。
            self._agent_worker.stop()
            self.stop_button.setEnabled(False)
            self.chat_view.set_status("已请求停止，当前这一步做完就停")
            return
        if self._chat_worker is not None:
            # 直连：stop 只是翻一个布尔，不需要跨线程排队
            self._chat_worker.stop()
            self.stop_button.setEnabled(False)
            self.chat_view.set_status("已请求停止，当前这句说完就停")

    def _append(self, message: Message) -> None:
        self._store.append(self._session, message)

    # ── 记忆窗口 ────────────────────────────────────────────────────────────

    @Slot()
    def _on_open_memory(self) -> None:
        """打开记忆浏览器。**独立窗口，不挤进三格 splitter。**"""
        if self._memory_window is None:
            window = MemoryWindow(memory=self._memory)
            # **先补一次**：当前会话、嵌入后端与这一轮的账本都发生在窗口还不存在的
            # 时候，那几次 emit 没有人听。补完之后才接线，否则窗口会拿到的是一份
            # "什么都没有"的快照，而它看起来完全正常。
            window.set_current_session(self._session.session_id)
            window.set_embedding(self._embedding)
            window.set_recall_ledger(self._recall_ledger)
            self.recall_updated.connect(window.set_recall_ledger)
            if self._graph is not None:
                window.set_graph(self._graph, stale=self._graph_freshness)
            self._memory_window = window
        # **建图挂在这里，不挂在启动路径上。** 这是全程序唯一需要邻居面板的地方，
        # 而小库一次全量重建要几秒、大库要几分钟（见 ``AUTO_GRAPH_MAX``）。每次
        # 打开都跑一趟是有意的：图可能在建完之后过期了，而"多久算过期"没有便宜的
        # 判据——`GraphWorker` 自己会决定是重建、复用还是（大库）原样交回来。
        # 已经在跑就不重复起（`_start_graph_build` 返回假），重建进行中开窗也一样。
        self._start_graph_build(force=False)
        self._memory_window.show()
        # 它可能已经在别的窗口后面（或者被最小化过）——只 show() 的话，用户点了
        # 「记忆」什么都没发生，然后以为按钮坏了。
        self._memory_window.raise_()
        self._memory_window.activateWindow()

    # ── 人设 ────────────────────────────────────────────────────────────────
    #
    # 库里两条，角色不同（见 memory/persona.py）：内置那条程序保证、不可改不可删；
    # 用户那条可改可删，没有就在启动时问一次。**写人设只有一个入口**
    # （:meth:`_apply_persona`），三个触发点（卡片、设置页保存、设置页删除）都走它。

    def _persona_texts(self) -> tuple[str, str]:
        """``(内置那段, 用户那段)``。**两段都读库里的实际值**，没写过的那段是空串。

        内置那段不填 :data:`~axo.memory.DEFAULT_PERSONA` 常量：库里的正文
        可能是旧版本写下的（那个字面量改过），而卡片与设置页要展示的正是"现在这个模型
        读到的身份"。展示一处、库里一处，是最不该有的那种分叉。
        """
        anchor = builtin_anchor(self._memory)
        written = user_anchor(self._memory)
        # 左边那个兜底正常不可达：__init__ 里刚 ensure 过，两个分支都保证它在。
        # 兜常量而不是空串，是因为空串会让卡片上那段只读文字变成一片空白——那看起来
        # 像渲染坏了，而不像"这条没了"。
        return (
            anchor.content if anchor is not None else DEFAULT_PERSONA,
            "" if written is None else written.content,
        )

    def _push_persona_state(self) -> None:
        """把"要不要问"推给网页。

        **每次启动都要推一次，包括不用问的那一次**：网页侧的输入框正是由那次推送
        放行的（``asking=False`` 那一趟同时是"可以打字了"的信号）。见
        :meth:`~axo.ui.chat_view.ChatView.set_persona_state`。
        """
        builtin, _ = self._persona_texts()
        self.chat_view.set_persona_state(asking=self._persona_asking, builtin=builtin)

    @Slot(str)
    def _on_persona_answered(self, text: str) -> None:
        """首启卡片收了答案。空串 = 用户选了「这一次先不写」。

        **失败就往对话流里说一句。** 这是卡片那条路，页面上没有第二处能显示结果；
        而静默失败在这里格外贵——用户以为定好了，下次启动卡片又回来，中间出过错
        这件事没有任何痕迹。
        """
        reason = self._apply_persona(text)
        if reason is not None:
            self.chat_view.append_message("system", f"人设没写进去：{reason}")

    def _apply_persona(self, text: str) -> str | None:
        """写或删用户那条人设。返回失败理由，``None`` 表示成功。

        空串 = 删掉那条——「这一次先不写」与「删掉我写的这段」是同一个意思，都走
        这一条路，所以 ``clear_user_persona`` 只会被这一处调用。

        **首尾空白在这里去掉，而不是只靠调用方。** 这张卡片的两个入口（网页侧与
        设置页）各自都会去一道，但去空白是"这段正文是什么"的一部分，属于写路径；
        留在调用方就是三处各有一个说法。去完之后全是空白的、与空串同义——这也正是
        :func:`~axo.memory.edit.can_encode` 的看法。

        **失败不抛也不推 ``asking=False``**：卡片留着、用户打的字也留着（网页侧清空
        输入框是那次推送触发的），他可以改一下再试。返回理由让调用方决定说给谁听
        ——卡片那条路只能说进对话流，设置页那条路说在它自己那一页上。
        """
        text = text.strip()
        try:
            if text:
                set_user_persona(self._memory, text)
            else:
                clear_user_persona(self._memory)
        except (ValueError, MemCoreError) as exc:
            # ValueError 兜住空正文与"全是标点"两种输入错（两者都来自 memory.edit.
            # can_encode，所以这里的理由与记忆窗口编辑时看到的是同一句）；
            # MemCoreError 兜住库写不进去（磁盘满、库损坏）。
            return str(exc)
        self._persona_asking = False
        self._push_persona_state()
        self._refresh_persona_page()
        return None

    def _refresh_persona_page(self) -> None:
        """设置窗口开着的话，把两段正文重新对齐到库里的实际内容。

        写完之后重新填一遍是空操作（用户打的字就是刚存下的那些），**但"保存"键的
        置灰判据是"有没有改动"**——回填之后它才会正确地变灰，那正是"存进去了"唯一
        的可见反馈。
        """
        if self._settings_dialog is not None:
            self._settings_dialog.set_persona(*self._persona_texts())

    @Slot(str)
    def _on_persona_save_requested(self, text: str) -> None:
        """设置页的「保存我写的这段」。反馈写在那一页上——用户正看着它。"""
        reason = self._apply_persona(text)
        if self._settings_dialog is not None:
            self._settings_dialog.set_persona_note(
                f"没写进去：{reason}" if reason else "已写入记忆库，立刻生效——换个模型也一样。"
            )

    @Slot()
    def _on_persona_clear_requested(self) -> None:
        """设置页的「删掉我写的这段」。**不重弹卡片**：裁决 4 说的是下次启动再问。"""
        reason = self._apply_persona("")
        if self._settings_dialog is not None:
            self._settings_dialog.set_persona_note(
                f"没删掉：{reason}"
                if reason
                else "已删掉你写的那段，内置的那段还在。下次启动会再问你一次。"
            )

    # ── 设置窗口 ────────────────────────────────────────────────────────────

    @Slot()
    def _on_open_settings(self) -> None:
        """打开设置窗口。**独立窗口、非模态，且只建一次。**

        非模态是刻意的：主色与壁纸可见度是"必须看着调"的参数，模态会让
        "拖滑杆 → 看效果"变成"关窗 → 看效果 → 再开窗"。

        **复用同一个实例，不每次新建。** 三个理由，前两个各自都足以定案：

        * 连接测试的 ``QThread`` 归对话框所有（见 :meth:`SettingsDialog.closeEvent`）。
          换掉窗口时那个线程还在跑，而它是个**没人等的线程**——正是
          ``_shutdown`` 里那段注释说的 ``0xC0000409``；
        * 外观是**实时生效**的，窗口里显示的必须与主窗口画的是同一份。新建一个
          实例就得在构造参数里补齐当前状态，而那是一次"差点忘了同步某项"的机会；
        * 位置与分类选择会保留。重开时跳回第一个分类是小事，但它每天都在发生。

        **整段包在 ``try`` 里，是因为这个按钮曾经变成过一个死按钮。** 构造器末尾
        就调 ``set_state``，而它一旦抛（真发生过：画壁纸缩略图那行写了个 PySide6
        里不存在的方法），异常就抛在 ``show()`` **之前**——窗口不出现，下一次点击
        走同一条路照样抛，屏幕上一点线索都没有。Qt 只在控制台留下回溯，而这个
        窗口的存在意义恰恰是让不用控制台的人也能用。所以这里把"打不开"变成一句
        看得见的错话，而不是留给用户一个点不动的按钮。

        **打不开就不 show**：半同步的窗口比不出现的窗口更坏——控件显示着旧值，
        用户会以为某项设置是关着的。理由给到控制台，去那儿找回溯。
        """
        try:
            if self._settings_dialog is None:
                dialog = SettingsDialog(
                    self,
                    theme=self._theme,
                    resolved=self._resolved,
                    # **整份传下去**，不是 host + allow_remote 两项。只给那两项的话，
                    # 重开设置窗口时 provider 会回落成 ollama，后端页自己切回本机组
                    # ——用户在云端配置下打开设置，看到的是「当前用本机」，于是他会
                    # 以为切云端根本没生效（而请求其实一直发在云端）。
                    backend=self._backend_settings,
                    tools=ToolSettings(read_roots=self._read_roots),
                    memory=MemorySettings(
                        task_memory=self._task_memory,
                        graph_channel=self._graph_enabled,
                        consolidation_enabled=self._consolidation_enabled,
                        consolidation_interval_s=self._consolidation_interval_s,
                    ),
                )
                dialog.theme_changed.connect(self._on_theme_changed)
                dialog.backend_changed.connect(self._on_backend_changed)
                dialog.tools_changed.connect(self._on_tools_changed)
                dialog.memory_changed.connect(self._on_memory_changed)
                dialog.graph_rebuild_requested.connect(self._on_rebuild_graph)
                # 「立即巩固」直接走**与定时那条路同一个入口**：区分自动/手动的地方
                # 只有一处（`force`），而托盘那一项连的是同一个槽。
                dialog.consolidation_requested.connect(self._on_manual_consolidation)
                dialog.persona_save_requested.connect(self._on_persona_save_requested)
                dialog.persona_clear_requested.connect(self._on_persona_clear_requested)
                dialog.open_data_folder_requested.connect(self._on_open_data_folder)
                dialog.change_data_location_requested.connect(self._on_change_data_location)
                dialog.restore_data_location_requested.connect(self._on_restore_data_location)
                dialog.onboarding_requested.connect(self._on_open_onboarding)
                self._settings_dialog = dialog
            else:
                # 重开时重新对齐：这段时间里系统明暗可能变过（跟随系统时外观已经变了），
                # 后端地址也可能被别的路径改过。**不重新对齐的话，「撤销修改」会回到
                # 一段不再成立的历史**——用户点它，界面回到几十分钟前的样子。
                self._settings_dialog.set_state(
                    self._theme,
                    self._resolved,
                    self._backend_settings,
                    tools=ToolSettings(read_roots=self._read_roots),
                    memory=MemorySettings(
                        task_memory=self._task_memory,
                        graph_channel=self._graph_enabled,
                        consolidation_enabled=self._consolidation_enabled,
                        consolidation_interval_s=self._consolidation_interval_s,
                    ),
                )
            # 图状态那一行**每次打开都重新对齐**（同人设页的道理，也是同一种真源：
            # 图是在这个窗口存在的时候才建的）。不回填的话，用户在设置页看着"还没建"
            # 而记忆窗口里邻居面板已经列出来了——同一件事在两处说两样话。
            self._settings_dialog.set_graph_note(self._graph_note)
            # 巩固那一行同理，只是**真源不同**：它的"上次什么时候跑的"在
            # ``last.json`` 里（跨进程），不在这个对象的某个字段上，所以每次打开
            # 现读一次盘。不回填的话，冷启动后打开设置页会看到一行空白，而这件事
            # 明明有答案。
            self._settings_dialog.set_consolidate_note(_last_run_note())
            # 人设页**每次打开都重新对齐**，而且走的是与 ``set_state`` 不同的方法。
            # 分列的理由是两者的真源不同：外观与后端存在 ``QSettings`` 里，人设存在
            # 记忆库里（卡片、记忆窗口都能改它）。混进同一个 ``set_state`` 会让"这
            # 一页从哪儿读"变含糊，而含糊的代价是设置页显示一段过期的正文。
            self._settings_dialog.set_persona(*self._persona_texts())
            # 数据页同理，**每次打开都现读**——而且它的真源是第三个：外观与后端在
            # QSettings 里、人设在记忆库里，而"现在用哪个目录"在 ``paths`` 里（是
            # 这个**进程**的事实）。搬迁跨重启才生效，所以这里读到的与设置里那一格
            # 未必是同一个值——那正是要显示的东西。
            self._settings_dialog.set_data(self._data_root, self._data_source)
        except Exception as exc:  # noqa: BLE001 —— 见 docstring：要把它变成一句人话
            # 构造失败时上面那句赋值没跑到，于是 `_settings_dialog` 仍是 None：
            # 再点一次会重试，而不是永远拿着一个坏掉的对象。
            self._settings_dialog = None
            QMessageBox.warning(
                self,
                "设置窗口打不开",
                f"设置窗口没能打开：{exc}\n\n"
                f"外观与后端设置都还在，主窗口可以照常用。\n"
                f"{_details_hint()}",
            )
            return
        self._settings_dialog.show()
        # 与记忆窗口同理：它可能已经在主窗口后面。只 show() 的话用户点了「设置」
        # 什么都没发生，然后以为按钮坏了。
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    @Slot(object)
    def _on_theme_changed(self, theme: Theme) -> None:
        """外观变了。**改一次就存一次**，没有"确定"这一步。

        这不是偷懒：窗口里根本没有「确定」键（只有「完成」与「撤销修改」），
        而不存的话，用户调好了主色、直接关掉窗口，下次启动又变回默认——一个
        只有"下次启动"才暴露的丢失。

        回填 ``Resolved`` 而不是让设置窗口自己再 :func:`resolve` 一遍：同一个
        主题算两次，两处迟早会漂，而漂的表现是"说明文字写的主色与屏幕上的不一样"。
        """
        resolved = self._apply_theme(theme)
        self._settings_store.save(theme)
        # 托盘图标按**解析后**的主色重画，与网页那一侧同源：浅色模式下主色可能已经
        # 被调暗过，用原色画出来的图标会和窗口对不上。
        self._tray.set_accent(resolved.accent)
        if self._settings_dialog is not None:
            self._settings_dialog.set_resolved(resolved)

    @Slot(object)
    def _on_backend_changed(self, backend: BackendSettings) -> None:
        """后端设置变了。

        **不做热切换，只换那一份然后重走既有的刷新链路。** ``_refresh_models`` →
        ``_on_models_loaded`` → 恢复 ``backend/last_model*`` → ``_on_model_changed``
        重建对话与嵌入后端——需要做的事它已经全做了。新开一条"切后端"的路径
        就是第二套真源，而它要覆盖的东西和这条一模一样。
        """
        # **整份相等比较，不逐字段列。** 逐字段列的话，每加一项设置就多一个漏掉的
        # 机会，而漏掉的症状是"改了、按了应用、什么都没发生"——用户只会再按一次。
        # 比的是 dataclass 的值（它是冻结的），所以 provider / api_key 变了同样会被
        # 认出来，这正是切云端时必须重探模型的理由。
        if backend == self._backend_settings:
            self._note_backend("后端设置没有变化。")
            return
        if self._streaming:
            # 正在生成的这一轮已经把请求发出去了，换后端影响不了它，却会让
            # 下一轮用上另一个服务端——两轮回复来自两个模型，而界面上看不出来。
            self._note_backend("正在生成，先停止再改后端设置。")
            return

        self._backend_settings = backend
        self._settings_store.save_backend(backend)
        self.model_combo.clear()
        where = backend.cloud_base_url if backend.is_cloud else backend.host
        self._note_backend(f"已改用 {where}，正在重新探测…")
        self._refresh_models()

    def _note_backend(self, text: str) -> None:
        """往设置窗口的后端页写一行结果。窗口没开过就什么都不做。"""
        if self._settings_dialog is not None:
            self._settings_dialog.set_backend_note(text)

    @Slot(object)
    def _on_tools_changed(self, roots: tuple[Path, ...]) -> None:
        """可读目录变了。**改一次就存一次**（同外观那条）。

        存而不同的两个理由，与外观页一字不差：窗口里没有「确定」键（只有「完成」），
        而不存的话，用户加完目录直接关掉窗口，下次启动又没了——一个只有"下次启动"
        才暴露的丢失。

        **不需要热切换。** 工具集本来就每轮重建（见 :meth:`_tools`），所以这里换掉
        这一份，下一轮提问就是新的。而**正在生成的这一轮不受影响**，这也是对的：
        一轮里能读的目录不该在中途变。
        """
        self._read_roots = tuple(roots)
        self._settings_store.save_tools(ToolSettings(read_roots=self._read_roots))

    @Slot(object)
    def _on_memory_changed(self, settings: MemorySettings) -> None:
        """记忆粒度变了。存一份，然后**换掉写手手里那个策略对象**。

        不重建 :class:`~axo.memory.writer.MemoryWriter`：策略是在**每轮
        写入时**被读的（见 ``MemoryWriter.record``），所以赋一个新对象就够了，而且
        "换掉写手"会连带丢掉它的注入（时钟）。这也正是那条接缝当初留成可写属性的
        理由。

        生效时机是**下一轮**：正在生成的这一轮用的是开始时那个策略。这是对的——
        一轮的粒度不该在中途变，那会让同一条记忆的正文来自两套规则。

        **联想开关落的是另一种形状，别照上面这段抄。** 上面那个 ``policy`` 是
        "对象已经存在、要主动换掉"；而召回配置是**每一轮现造的**
        （:meth:`_recall_config`），所以这里只要记下这个布尔，下一轮的配置自然是对的
        ——去 mutate 一个长命的 config 对象反而会引入"它被谁改过"这种看不出来的状态。
        两个字段住在同一次 ``save_memory`` 里，但它们的生效机理不同。

        **巩固那两个字段是第三种形状，而且只有它们要"现在生效"。** 上面两组都是
        "下一轮才看得出区别"（一轮的粒度/通道不该在中途变），而"每天跑一次"这件事
        用户改完是**想马上看到表动**的：把 24 小时改成 1 小时之后要等原来那一觉睡满
        才生效，看起来就是这个开关没反应。所以这里直接把两个值落到调度器上——它
        自己的 ``set_interval_s`` 会当场重排。
        """
        self._task_memory = settings.task_memory
        self._writer.policy = TaskPolicy() if self._task_memory else PerItemPolicy()
        self._graph_enabled = settings.graph_channel
        self._settings_store.save_memory(settings)
        self._consolidation_enabled = settings.consolidation_enabled
        self._consolidation_interval_s = settings.consolidation_interval_s
        self._consolidation_scheduler.set_interval_s(self._consolidation_interval_s)
        if self._consolidation_enabled:
            # ``start()`` 是幂等的（就是再排一次表）。关掉再打开时**要重新排**：
            # 关的那一下把它停了，而"打开"不能只把布尔翻过来。
            self._consolidation_scheduler.start()
        else:
            self._consolidation_scheduler.stop()
            # **正在跑的那一轮不取消**（本版没有取消，与 ``GraphWorker`` 一致）。
            # 它跑完照样写库、照样报结果——比"关掉开关时砍掉一个写了一半的库"
            # 干净得多。
            self._consolidate_pending = False

    def _recall_config(self) -> RecallConfig:
        """这一轮召回用的配置。**每轮现造**，所以它永远不会陈旧。

        这也意味着"改开关"与"改图"是两件事：这个函数只读 ``_graph_enabled``，
        图本身由 :meth:`_on_open_memory` 那一趟负责。开关打开而图还没建时，
        ``recall`` 会照常短路（``graph_weight`` 是给融合用的，图是 ``None``），
        并把"已启用但没有图"写进账本。
        """
        return RecallConfig(
            graph_weight=_GRAPH_CHANNEL_WEIGHT if self._graph_enabled else 0.0,
        )

    # ── 关联图 ──────────────────────────────────────────────────────────────
    #
    # 三条约定，都在这一小段里：
    # 1. **只在打开记忆窗口时自动建**（`_on_open_memory`）。它是这个图上唯一需要
    #    邻居面板的地方；启动时无条件建的话，用不上这个功能的人每次开程序都要等。
    # 2. **小库自动、大库手动**（`AUTO_GRAPH_MAX`）。大库重建一次四分钟起，绝不能
    #    挂在任何一条用户没主动要的路径上。
    # 3. **同一时刻只跑一趟**（`_graph_building`）。两趟会同时写同一个 ``graph.npz``，
    #    而"哪一趟先落盘"决定了盘上最后是哪张图——那种错没有任何报错，只有一张
    #    悄悄变旧的图。

    @Slot()
    def _on_rebuild_graph(self) -> None:
        """设置页「重建联想」按下了。**force**：用户按它就是不信任现在这份。"""
        if not self._start_graph_build(force=True):
            self._settings_dialog.set_graph_note("上一次建图还在跑，等它结束再说。")

    def _start_graph_build(self, *, force: bool) -> bool:
        """起一趟建图。**已经在跑就不起第二趟**，返回有没有起。

        按钮那一侧已经把「按下去到结果回来」这段自己禁用了，所以手动连按挡得住；
        这里挡的是另外两种叠法：**自动那趟还在跑时用户按了重建**，以及**手动重建
        还在跑时用户打开了记忆窗口**。两种都会有两个 worker 同时写 ``graph.npz``。
        """
        if self._graph_building:
            return False
        self._graph_building = True
        if self._settings_dialog is not None and force:
            self._settings_dialog.set_graph_busy(True)

        thread = QThread(self)
        worker = GraphWorker(store=self._memory, force=force)
        worker.ready.connect(self._on_graph_ready)
        worker.failed.connect(self._on_graph_failed)
        # 同装配那一路：不接这两下的话，``run()`` 跑完之后线程自己的事件循环不会
        # 结束，于是每建一次图就留下一个空转线程。
        worker.ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="建图线程", slot="graph")
        thread.start()
        return True

    @Slot(object, object, str)
    def _on_graph_ready(self, graph: GraphIndex, freshness: bool | None, note: str) -> None:
        self._graph_building = False
        self._graph = graph
        self._graph_freshness = freshness
        self._graph_note = note
        if self._settings_dialog is not None:
            # 自动那趟也走这里：按钮本来就是放开的，再放开一次是空操作。而**只有
            # 手动那趟会禁用它**（见 `_start_graph_build`），所以这里不必分辨是谁
            # 回来了——分辨反而会漏掉"自动那趟把用户按下的那次挤掉"这一类。
            self._settings_dialog.set_graph_busy(False)
            self._settings_dialog.set_graph_note(note)
        if self._memory_window is not None:
            self._memory_window.set_graph(graph, stale=freshness)
        # 挂起的那一轮巩固**在这里继续**，不在 ``_on_consolidation_due`` 里：那一枪
        # 当时已经打完了，它唯一能做的就是把这个意图挂起来。而"图来了"是唯一能让
        # 挂起的那一轮继续的事件——所以继续的落点只能在这里。自动与手动都走这条
        # （``_consolidate_manual`` 只说结果要不要弹气泡）。
        if self._consolidate_pending:
            self._start_consolidation(manual=self._consolidate_manual)

    @Slot(str)
    def _on_graph_failed(self, message: str) -> None:
        """建图失败**不动 ``self._graph``**：旧的那份照样能用，只是没有新的。

        清掉它的话，一次失败的「重建联想」会把正在看的邻居面板也弄空——用户看到的
        会是"重建失败"加一个空面板，两件坏事一起。
        """
        self._graph_building = False
        if self._settings_dialog is not None:
            self._settings_dialog.set_graph_busy(False)
            self._settings_dialog.set_graph_note(message)
        # 挂起的那一轮巩固**跟着一起了结**。图建不出来，它就没有分簇的前提——留着
        # ``_consolidate_pending`` 的后果是：手动那一趟会等到下一次"到点"才被人发现
        # 没跑（最长 24 小时），而那期间设置页那一行什么都不说。清掉并说一句，比
        # 让它挂在那里诚实。
        if self._consolidate_pending:
            self._consolidate_pending = False
            manual = self._consolidate_manual
            self._consolidate_manual = False
            if manual:
                self._note_consolidate("建图失败，这一轮巩固没有跑。")

    # ── 睡眠巩固 ────────────────────────────────────────────────────────────
    #
    # 三条约定，都在这一小段里：
    # 1. **同一时刻只跑一趟**（``_consolidating``）。两轮会同时 `bump_reinforce`
    #    同一批条目（计数多涨一次），而事件文件按 run_id 命名——同一个 `at` 秒里起
    #    的第二趟会撞上第一趟的文件名然后抛 ``ValueError``。
    # 2. **没有图就跑不成**（分簇的前提）。那时把这一轮**挂起**，并请一趟建图——
    #    图是懒建的（只在打开记忆窗口时建），所以"从不打开记忆窗口的人"是这条路径
    #    的常客。不挂起的话，默认开的功能对他们就是每天空转一次。
    # 3. **正在生成或装配时推迟一分钟。** 全仓没有真 idle 检测，这是最接近的代理
    #    （见 ``ui/consolidation_scheduler.py`` 的模块文档）——它挡的是"挂在用户正
    #    等回复的那一刻上"，挡不住"用户在别处用这个库"。
    #
    # 自动与手动**不是两条路径**：同一个 worker、同一道单飞守卫、同一段收尾，只差
    # 一个 ``force`` 与"回来说不说气泡"。这与 ``GraphWorker`` 的形状一字不差。

    @Slot()
    def _on_consolidation_due(self) -> None:
        """表响了。**跑不跑是这里决定的，调度器只报时。**

        每一条提前返回之前都要**先把表接上**——那一枪已经打完了，不重排就再也
        不会有下一枪。三条分支的次序不是随意的：先挡"已经在跑"（最便宜、也最该
        挡），再挡"用户正忙"，最后才看图。
        """
        if self._consolidating:
            # 兜底重排（读到的是上一轮那个旧 ``at``，于是多半排成下限一分钟）。那一轮
            # 跑完时会写 ``last.json`` 并重排一次，这里排上的只是"万一它没排"的兜底。
            self._consolidation_scheduler.reschedule()
            return
        if self._streaming or self._preparing:
            # **明确给一分钟，不走 `reschedule()`**：那个算法问的是"距上次多久"，
            # 而此刻的答案明摆着是"早就该跑了"。它也不会写 ``last.json``，所以这一
            # 分钟之后会重新问一遍"该不该跑"，答案仍然是真的——直到用户这一轮结束。
            self._consolidation_scheduler.reschedule(MIN_DELAY_MS)
            return
        if self._graph is None:
            self._consolidate_pending = True
            self._consolidate_manual = False
            self._start_graph_build(force=False)
            return
        self._start_consolidation(manual=False)

    @Slot()
    def _on_manual_consolidation(self) -> None:
        """托盘那项或设置页那个按钮被按下了。

        三条拒绝，每条都要**说一句为什么**：一个按下去没反应的按钮，用户的结论是
        "它坏了"，然后他会再按一次。

        * **已经在跑** → 说一句，**不排队**。巩固一轮可能几分钟，而排一个用户看不见
          的队、他按完就走了；回来时那一轮已经跑过，他不知道那是谁触发的（而它会
          弹一条气泡）。
        * **正在生成** → 说一句，**不推迟**。"推迟"是自动那条路的语汇：手动按下的
          那一趟不该被悄悄排到一个用户没看着的时刻上——那正是他按下它的反面。
        * **没有图** → 不拒绝：挂起这一轮并请一趟建图，建完接着跑。这条路径的常客
          是"从没打开过记忆窗口"的人，而那个按钮不该因为图是懒建的而变成死的。
        """
        if self._consolidating:
            self._note_consolidate("上一轮还在跑，等它结束。（跑完会在对话区报一句。）")
            return
        if self._streaming or self._preparing:
            self._note_consolidate("正在生成，先停止再巩固。")
            return
        if self._graph is None:
            self._consolidate_pending = True
            self._consolidate_manual = True
            started = self._start_graph_build(force=False)
            self._note_consolidate(
                "还没有关联图，先建图，建完就跑这一轮。"
                if started
                else "正在建图，建完就跑这一轮。"
            )
            return
        self._start_consolidation(manual=True)

    def _start_consolidation(self, *, manual: bool) -> bool:
        """起一趟巩固。**已经在跑就不起第二趟**，返回有没有起。

        ``manual`` 做两件事：传给 worker 当 ``force``（跳过"没有新记忆"那道门），
        以及记下来决定结果回来说不说气泡。

        **调用方保证 ``self._graph`` 不是 ``None``。** 不在这里兜一下：兜住的话，
        "没有图却起了巩固"会变成一个静默的空转（worker 拿不到图会抛，异常被它自己
        接住、变成一条 ``failed``），而这条本不该发生的路径上，静默比响一声坏。
        """
        if self._consolidating:
            return False
        self._consolidating = True
        self._consolidate_pending = False
        self._consolidate_manual = manual
        if manual and self._settings_dialog is not None:
            self._settings_dialog.set_consolidate_busy(True)

        thread = QThread(self)
        worker = ConsolidationWorker(store=self._memory, graph=self._graph, force=manual)
        worker.ready.connect(self._on_consolidation_ready)
        worker.skipped.connect(self._on_consolidation_skipped)
        worker.failed.connect(self._on_consolidation_failed)
        # 同建图那一路：不接这三下的话，``run()`` 跑完之后线程自己的事件循环不会
        # 结束，于是每跑一轮就留下一个空转线程。**三个收场都要接**——这个 worker
        # 比 GraphWorker 多一个"让掉"的结局，漏掉它就是一个必然发生的泄漏。
        worker.ready.connect(thread.quit)
        worker.skipped.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._track(thread, worker, what="巩固线程", slot="consolidate")
        thread.start()
        return True

    @Slot(object)
    def _on_consolidation_ready(self, run: ConsolidationRun) -> None:
        """这一轮真跑完了（库已经改过）。

        清单飞、重排、说话，三件事的次序按"最先要做的"排：**先把它从"在跑"里放
        出来**（不然一次异常就能把巩固永久锁死），再排下一枪，最后才是给人看的那些。
        """
        manual = self._consolidate_manual
        self._consolidating = False
        self._consolidate_manual = False
        # **这里必须是 `reschedule()`，不能是"隔一整个周期"。** 这一轮刚把
        # ``last.json`` 写成"现在"，于是它算出来的正是"从现在起一个周期"——而
        # `reschedule_full_interval` 在这里也恰好等价，但那是巧合，不是意图。
        self._consolidation_scheduler.reschedule()
        line = summary_line(run.report)
        # **这一行不进对话区。** 巩固是每天自己跑一次的日常，往对话区写一句"回放了几条、
        # 认出几条"是噪音——对话区只留用户与它说过的话。所以 ``line`` 只去两个地方：
        # 设置页那一行状态，与手动那趟的托盘气泡。
        #
        # **失败那趟是例外，它照样说**（见 ``_on_consolidation_failed``）：成功是日常、
        # 沉默；失败是"本该发生而没发生"，要说。
        #
        # **给开着的记忆窗口推一下。** 热度那一块画的是库里的 ``reinforce_count``，
        # 而这一轮刚刚改过它——不推的话，窗口要等到用户手动「刷新」或重开才说得出
        # 新的数。**这也是自动那趟在人前唯一的落点**：它不弹气泡、也不写对话区，
        # 窗口没开着时这一轮就没有任何看得见的痕迹（README 里如实记了这条）。
        if self._memory_window is not None:
            self._memory_window.notify_consolidated()
        if self._settings_dialog is not None:
            self._settings_dialog.set_consolidate_busy(False)
            self._settings_dialog.set_consolidate_note(line)
        if manual:
            # **只有手动那趟弹气泡。** 自动那趟每天一次，弹出来就是噪音，而它的结果
            # 落在热度面板上（见上）。手动那趟不弹就等于"点了没反应"，而那个动作的
            # 全部意义就是这个回声。
            self._tray.show_message("睡眠巩固", line)

    @Slot()
    def _on_consolidation_skipped(self) -> None:
        """这一轮让掉了（空库，或"最新那条记忆还是上次那条"）。**除了重排，不做声。**

        沉默是刻意的：自动那趟每天至少醒一次，而"今天没有新记忆"是常态、不是事件
        ——每天往对话区写一句"没什么可做的"是纯噪音，气泡更是。

        **必须按"隔一整个周期"重排**（见 ``reschedule_full_interval`` 的 docstring）：
        让掉的一轮没有写 ``last.json``，所以 ``reschedule()`` 读到的还是上一轮那个
        旧 ``at``、算出来必然是负数、被夹成一分钟——于是每分钟醒一次、每分钟再让掉
        一次，直到有东西变。
        """
        self._consolidating = False
        self._consolidate_manual = False
        self._consolidation_scheduler.reschedule_full_interval()

    @Slot(str)
    def _on_consolidation_failed(self, message: str) -> None:
        """这一轮失败了。**报出来，但只在界面上报——不弹气泡。**

        失败与让掉的区别正在于"要不要说"：让掉是正常的日常，失败是一件本该发生而
        没发生的事。所以失败**走对话区**（用户回看读得到，也不打断他），而气泡留给
        "用户刚刚按下的那一趟"——失败的那趟若是手动按的，气泡也弹（见下）。

        **不重试**，隔一个周期再说：失败的原因（图过期、库被占着）多半不会在一分钟
        后消失，而每分钟重试一次只会把一次失败变成一晚上的失败。
        """
        manual = self._consolidate_manual
        self._consolidating = False
        self._consolidate_manual = False
        self._consolidation_scheduler.reschedule_full_interval()
        text = f"睡眠巩固没跑成：{message}"
        self.chat_view.append_message("system", text)
        if self._settings_dialog is not None:
            self._settings_dialog.set_consolidate_busy(False)
            self._settings_dialog.set_consolidate_note(text)
        if manual:
            self._tray.show_message("睡眠巩固", text)

    def _note_consolidate(self, text: str) -> None:
        """把"手动按下去之后发生了什么"说出去。**两个出口，因为两个入口。**

        设置页那个按钮的入口在设置页里，托盘那一项的入口在托盘上——而两者连的是
        同一个槽。分别找"现在哪个窗口开着"来决定说给谁听，会在"两个都开着"和
        "两个都没开"这两种情形上没有答案；两个都写一句，最坏是同一句话说了两遍，
        而用户看到两遍同一句话的代价远小于一次"点了没反应"。

        **气泡在托盘不可用时是空操作**（见 ``ui/tray.py``），所以这里不必先问。
        """
        if self._settings_dialog is not None:
            self._settings_dialog.set_consolidate_busy(False)
            self._settings_dialog.set_consolidate_note(text)
        self._tray.show_message("睡眠巩固", text)

    def _push_context_to_memory_window(self) -> None:
        """把「用户在哪个会话」「当前嵌入后端是哪个」推给记忆窗口。

        这两样**不走** :attr:`recall_updated`：它们与"这一轮召回了什么"变化的时机
        不同（切会话、刷新模型列表都会变），混进同一个信号会让那个信号的语义变得
        含糊——而账本信号一旦含糊，它的消费者就得开始猜。
        """
        if self._memory_window is None:
            return
        self._memory_window.set_current_session(self._session.session_id)
        self._memory_window.set_embedding(self._embedding)

    # ── 账本侧栏 ────────────────────────────────────────────────────────────

    def _on_toggle_ledger(self, checked: bool) -> None:
        self.ledger_button.setText("注入 ▾" if checked else "注入 ▸")
        self._ledger_pane.setVisible(checked)
        # **三格尺寸走与开局同一个函数。** 以前这里是一段手算，于是"开局账本多宽"
        # 与"展开账本多宽"是两条公式——小窗口下 340 会把对话区挤到 240 以下，而
        # 开局那条路径有夹紧、展开这条没有。共用 `plan_pane_sizes` 之后，账本收窄
        # 的下限、对话区不够时先向谁借，只有一处说法。
        total = sum(self._splitter.sizes())
        self._splitter.setSizes(list(plan_pane_sizes(total, ledger_open=checked)))
        if checked:
            self._render_ledger()
            self.ledger_view.verticalScrollBar().setValue(0)

    def _render_ledger(self) -> None:
        """把这一轮的账本画到侧栏。

        两段来自两个不同的地方，**答的是两个不同的问题**：装配器的账本说"额度花在
        哪了、注入了哪几条"，召回账本说"库里还有什么、它们为什么没进来"。合成一份
        文本而不是两个控件，是因为它们要一起被读——分开就要在脑子里对齐两份编号。
        """
        # **指示灯在这里同步，而且在早退之前**：灯与账本必须永远同源，否则会出现
        # "灯说 90%、账本说 40%"这类无从排查的矛盾——而用户会先信灯。
        self._show_context(self._assembly)
        if self._assembly is None:
            return
        parts = [self._assembly.ledger.render()]
        detail = render_recall_detail(self._recall_ledger)
        if detail:
            parts.extend(("", detail))
        self.ledger_view.setPlainText("\n".join(parts))

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """关窗：收尾，然后照常关。

        **在跑的线程都要等**，不只是正在生成的那一个：

        * 装配线程在读记忆库，而库是"一个连接 + 一把锁"——进程退出时正好握在它手里，
          SQLite 的 WAL 就少一次干净的收尾；
        * 补向量那个线程**在写**记忆库，是这几个里唯一会改数据的；
        * **巩固线程也在写记忆库**（``bump_reinforce``、快照、事件文件）。它是这几个
          里**最重**的一个——整库复制是几百 MB 的写，一轮可能跑几分钟，而
          ``wait_for_all`` 只等 3 秒。所以最坏情形是：超时、放行、进程退出。**那不会
          留下半个事务**，三道保险都在（① 快照排在第一次改库之前；② 每一簇是一次
          "改库 + 追加事件"，崩在中途得到的是 ``seq`` 连续的一个前缀；③ WAL 保证每个
          ``UPDATE`` 要么整体落盘要么整体回滚）。第一次看到控制台那句「收尾超时」
          会吓一跳，但它说的是"没等到"，不是"写坏了"。
        * 模型列表线程持有一个到 Ollama 的连接。
        * 生成那一路现在有**两个** worker（``ChatWorker`` 与 ``AgentWorker``，同一时刻
          只有一个在跑），而 P3a 那个可能正**卡在等用户点确认**上——除了
          ``AgentWorker.stop()`` 里的 ``slot.cancel()``，没有任何东西能叫醒它。

        真正常在跑的是补向量与模型列表（生成多半已经结束），所以这里等的通常不是"生成"。

        **分成两支：**

        * :meth:`_close_should_hide` 为真 → ``ignore`` + ``hide``，**进程继续活着**，
          于是托盘还在、提醒还会响。第一次给一句气泡说明它去哪儿了。
        * 为假 → 走收尾那一支拉到底。

        判据抽成 :meth:`_close_should_hide` 是为了**可单测**：这条路径的三种情形
        （有托盘 / 没托盘 / 真的要退出）在离屏环境里没法靠构造一个真窗口来验——
        离屏下构造 ``MainWindow`` 会让进程无声消失（``0xC0000409``，见
        ``ui/agent_worker.py`` 与 ``thread_tracking`` 里的同一段教训）。
        """
        if self._close_should_hide():
            event.ignore()
            self.hide()
            # **说一次就够**：通知区域在很多机器上默认是折叠的，不说的话用户会以为
            # 程序已经退出；而每次关窗都说，它会在一天几十次里变成噪音。
            self._tray.show_message_once(
                "还在后台运行",
                "窗口收到了系统托盘。要真正退出，右键托盘图标选「退出」。",
            )
            return
        self._shutdown()
        super().closeEvent(event)

    def _close_should_hide(self) -> bool:
        """这一次关窗是"收起来"还是"退出"。

        三个条件缺一不可：**有托盘**（否则收起来就够不着了）、**托盘真的在**
        （``active``；不可用时它是一个空壳）、**用户不是真的要退出**
        （:attr:`_quitting`——托盘菜单的「退出」与 Ctrl+Q 会先置它）。
        """
        return self._tray is not None and self._tray.active and not self._quitting

    @Slot()
    def _quit(self) -> None:
        """**真的退出。** 托盘菜单的「退出」与 Ctrl+Q 都连这里。

        先置 :attr:`_quitting` 再 ``close()``：顺序反过来的话，``closeEvent`` 会在
        置位之前读到假，于是"退出"变成了又一次"收起来"——用户按了退出，窗口消失了，
        进程还在，而托盘图标还在那儿。
        """
        self._quitting = True
        self.close()

    @Slot()
    def _on_tray_show(self) -> None:
        """托盘叫回窗口。**三样都要做**，少一样就是"点了没反应"：

        * ``show()``——从隐藏状态回来；
        * ``raise_()``——从别的窗口**底下**钻出来；
        * ``activateWindow()``——拿到键盘焦点。只 ``show`` 的话窗口会出现在最底层，
          用户以为没反应，然后再点一次图标。
        """
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot()
    def _on_open_data_folder(self) -> None:
        """在资源管理器里打开记忆所在的那个文件夹。

        这是"重新下载之后还能找到"那件事的界面一侧：把位置显示出来还不够，还得能一键
        过去——备份、看一眼库里有什么，都得先到那个文件夹。
        """
        self._open_directory(self._data_root)

    def _open_directory(self, path: Path) -> None:
        """**可覆盖的测试缝。** ``QDesktopServices`` 在没有桌面会话的环境里是个空操作，
        而这条链要能在离屏测试里断言"点了一下、开的是那个目录"。"""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @Slot(str)
    def _on_change_data_location(self, chosen: str) -> None:
        self._start_data_move(Path(chosen))

    @Slot()
    def _on_restore_data_location(self) -> None:
        """「恢复默认位置」走**同一条搬迁链**，只是目标换成默认那个。

        不另写一条"只改指针"的短路：那会让它变成唯一一个不判断、不搬迁的入口，而它
        同样可能撞上"目标里已经有别的东西""现在这个已经是默认位置了"。
        """
        self._start_data_move(default_data_dir())

    def _start_data_move(self, destination: Path) -> None:
        """用户挑好了新位置：判断 → 确认 → 写标记 → 重启。**这里不复制。**

        复制发生在新进程里（理由见 :mod:`axo.ui.data_root` 的模块文档）。这一步只做
        两件落盘的事：写一条搬迁标记，然后重启。**位置指针一个字都没改**——所以此刻
        断电、或者重启起不来，程序用的都还是那份完整的源。
        """
        self._set_data_note("")
        plan = plan_move(self._data_root, destination)
        if plan.kind is MoveKind.REFUSE:
            QMessageBox.warning(self, "这个位置不能用", plan.summary)
            return
        if not self._confirm_data_move(plan):
            return

        stage_move(make_settings(), self._data_root, destination)
        if restart_application():
            self._shutdown()
            return
        # **重启失败不是搬迁失败。** 标记已经在注册表里，用户下次自己打开照样会办完
        # ——这条降级路径是"跨重启"那个选择顺带得到的，也是它比"原地搬"稳的地方之一。
        self._set_data_note(
            "已经安排好了：下次打开程序时会自动把记忆搬过去，然后就用新位置。"
            "（刚才没能自动重启，你自己关掉再打开一次也一样。）"
        )

    def _confirm_data_move(self, plan: MovePlan) -> bool:
        """确认框。**可覆盖的测试缝**——弹真对话框的测试跑不了。

        文案里必须有两句实话：**要重启一次**、**旧的那份不会自动删**。少了第一句，
        用户会以为点完就生效；少了第二句，他会以为旧位置已经清掉了。
        """
        if plan.kind is MoveKind.COPY:
            extra = (
                f"\n\n程序要重启一次；重启时会把记忆复制过去"
                f"（约 {plan.total_bytes / 1024**2:.1f} MB），可能要等一会儿。"
            )
        else:
            extra = "\n\n程序要重启一次，重启之后就用那个位置。"
        answer = QMessageBox.question(
            self,
            "换一个放记忆的文件夹",
            f"{plan.summary}{extra}\n\n旧位置那份不会自动删，确认搬好了你自己处置。",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        return answer == QMessageBox.StandardButton.Ok

    def _set_data_note(self, text: str) -> None:
        """搬迁那条链的反馈，落在**数据页**上而不是对话流里——它是设置页上那个动作的
        结果，用户的眼睛在那儿。

        设置窗口没开过时退到对话流：这条路上唯一用得到它的是"重启没成功"，而那时
        总得有个地方说一句，不然用户看到的是"点了没反应"。
        """
        if self._settings_dialog is not None:
            self._settings_dialog.set_data_note(text)
        elif text:
            self.chat_view.append_message("system", text)

    # ── 新手引导 ────────────────────────────────────────────────────────────

    def maybe_start_onboarding(self) -> None:
        """首启弹一次。**由 ``ui/app.py`` 在 ``show()`` 之后调。**

        放在那儿而不是构造函数里：从构造函数里弹窗会让向导出现在主窗口之前。副作用是
        几个冒烟脚本不会看到它——正好，它们的断言都是关于主窗口的。

        判据是一个**存下来的标志**（``app/onboarding_done``），不是从状态推导的。
        理由见 :mod:`axo.ui.onboarding` 的模块文档：向导里有两步不留下任何痕迹，
        "推导"出来的判据会在每次启动都把向导弹回来。

        老用户升级后会弹一次（标志不存在）。这是**必须的**——不弹的话，"记忆放哪"
        这个能力恰好对那些已经有记忆、最需要它的人不可见。
        """
        if self._settings_store.onboarding_done():
            return
        self._open_onboarding(show_persona_step=needs_persona(self._memory))

    @Slot()
    def _on_open_onboarding(self) -> None:
        """从「设置 → 数据 → 重新打开新手引导」进来。**强制带上人设那一步**——
        想重看的人多半就是想改它。"""
        self._open_onboarding(show_persona_step=True)

    def _open_onboarding(self, *, show_persona_step: bool) -> None:
        if self._onboarding is not None:
            self._onboarding.close()
            self._onboarding = None
        try:
            builtin, user = self._persona_texts()
            wizard = OnboardingWizard(
                self,
                root=self._data_root,
                source=self._data_source,
                backend=self._backend_settings,
                builtin_persona=builtin,
                show_persona_step=show_persona_step,
            )
            wizard.set_persona(user)
            # 人设那一步连**已有的槽**：它在成功之后会把 ``_persona_asking`` 置假并
            # 重推状态，于是对话区那张首启卡片自己就不来了——去重是零新逻辑的。
            wizard.persona_answered.connect(self._on_persona_answered)
            wizard.data_location_requested.connect(self._on_change_data_location)
            wizard.restore_data_location_requested.connect(self._on_restore_data_location)
            wizard.backend_changed.connect(self._on_backend_changed)
            wizard.onboarding_finished.connect(self._on_onboarding_finished)
            self._onboarding = wizard
            wizard.show()
            wizard.raise_()
            wizard.activateWindow()
        except Exception as exc:  # noqa: BLE001 —— 变成一句人话，同设置窗口那一段
            self._onboarding = None
            QMessageBox.warning(self, "新手引导打不开", f"{exc}\n\n{_details_hint()}")

    @Slot()
    def _on_onboarding_finished(self) -> None:
        """「完成」与「跳过」都走这儿。叉掉窗口**不发**这个信号，所以那条路不算走过
        ——没做完也没说不做不算决定（见 :mod:`axo.ui.onboarding` 的模块文档）。"""
        self._settings_store.mark_onboarding_done()

    def report_data_move(self, report: MoveReport) -> None:
        """把这次启动在搬迁上做的事说出来。**由 ``ui/app.py`` 在 ``show()`` 之后调。**

        为什么要等到这会儿：搬迁发生在**新进程打开任何东西之前**，那时还没有窗口，
        既没有地方说、抛异常又等于"程序打不开"。

        复制成功的那一路会问一句"旧的那份删不删"——那一问只能在这里问，因为"复制完
        了"这个事实只存在于新进程里。失败与被拒都只说一句、落在对话流里，**不弹框**：
        用户没做错什么，只是那次搬迁没成，而程序照常开着、用的还是原来那个位置。
        """
        summary = " ".join([report.summary, *report.notes])
        if report.ok and report.offers_to_delete_source and report.source is not None:
            self.chat_view.append_message("system", summary)
            self._ask_about_the_old_copy(report.source)
            return
        self.chat_view.append_message("system", summary)

    def _ask_about_the_old_copy(self, source: Path) -> None:
        """问一句旧的那份留不留，并按回答处置。

        **默认是"保留"。** 旧的那份是用户唯一的退路，而删掉不可逆——一个回车按下去
        就没了。所以默认按钮是"保留"，而且真要删还得再确认一次它看起来确实像一份
        数据根（见 :func:`~axo.ui.data_root.discard_source` 那三道闸）。
        """
        if not self._confirm_delete_source(source):
            self.chat_view.append_message("system", f"旧的那份留在 `{source}`，你自己看着办。")
            return
        if discard_source(source, current_root=self._data_root):
            self.chat_view.append_message("system", f"已经把 `{source}` 删掉了。")
        else:
            self.chat_view.append_message(
                "system", f"`{source}` 没能删掉（它看起来不像一份数据根，或者删不动）。"
            )

    def _confirm_delete_source(self, source: Path) -> bool:
        """问一句"旧的那份删不删"。**独立一个方法是为了让测试能盖掉它**——
        弹真对话框的测试跑不了（同 ``settings_dialog`` 里那几个可覆盖的选择框）。"""
        answer = QMessageBox.question(
            self,
            "记忆已经搬到新位置",
            f"要不要把旧位置那份也删掉？\n\n{source}\n\n"
            "留着它不占什么，而且万一新位置出问题，它还能用；"
            "删掉之后就只剩新位置那一份了。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _shutdown(self) -> None:
        """**唯一的退出路径。** 幂等。

        所有能让进程结束的动作都汇到这里：关窗、``aboutToQuit``、将来的托盘退出与
        Ctrl+Q。它只做一件事——让每个线程都**停下来并且等它停**。

        **为什么必须等，而不是交给解释器收尾。** 一个仍在运行的 ``QThread`` 在
        析构时会让 Qt 调 ``qFatal``（Windows 上的表现是退出码 ``0xC0000409``，
        fast-fail），没有回溯、没有 stderr、Python 的 ``atexit`` 都不跑——看起来
        像 QtWebEngine 崩了，实际只是有个线程还活着。四个脚本一直带着这个退出码，
        只是此前没人看 ``$?``。

        ``aboutToQuit`` 那一路是必需的：脚本里 ``app.quit()`` 之后窗口不会被
        ``close``，于是 ``closeEvent`` 不跑，而**收尾仍要发生**。
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        if self._chat_worker is not None:
            self._chat_worker.stop()
        if self._agent_worker is not None:
            # 收尾这条路上，``AgentWorker.stop()`` 的第二件事（``slot.cancel()``）是
            # **必需**的，不是优化：一个正卡在等确认上的工作线程除了它没有任何东西
            # 能叫醒，而下面只等 3 秒。见 ``ui/agent_worker.py`` 的模块文档。
            self._agent_worker.stop()
        # 按**在跑的清单**逐个等，不按那几个成员：取消时 `_reset_generation_state`
        # 会把成员提前清掉，而线程还在跑（装配是这样，故意的）。按成员等就会漏掉
        # 它，漏掉的那个线程正是下一次 0xC0000409 的来源。
        #
        # 传一份**快照**（`list(...)`）：等的过程中 `release` 闭包会把清单换成新列表。
        # 循环体里那三条约定（为什么必须等、为什么直连 quit、超时为什么还要说一句）
        # 全在 `wait_for_all` 的文档里——那里也是记忆窗口复用的同一份实现。
        wait_for_all(list(self._live_threads))
        if self._memory_window is not None:
            # **关掉它**（= 等它的线索召回线程停），不是只 hide()：那个线程在跑
            # 的时候进程退出，就是下一次 0xC0000409。等待逻辑不在这里重复一遍——
            # 窗口自己的 ``closeEvent`` 会做，而且用的是同一个 ``wait_for_all``。
            self._memory_window.close()
            self._memory_window = None
        if self._settings_dialog is not None:
            # **同样要 close 而不是只 hide()**，理由与上面那条一字不差：它自带一个
            # 连接测试线程，而且它是主窗口的子对象——主窗口一析构它就跟着析构，
            # 那时线程若还在跑就是下一次 0xC0000409。等线程与清理壁纸副本都在它
            # 自己的 ``closeEvent`` 里，不在这里重复。
            self._settings_dialog.close()
            self._settings_dialog = None
        if self._onboarding is not None:
            # **同样要 close 而不是只 hide()**：向导里那个后端页自带一个连接测试线程，
            # 而且它是主窗口的子对象——主窗口一析构它就跟着析构，那时线程若还在跑就是
            # 下一次 0xC0000409。等线程在它自己的 closeEvent 里。
            self._onboarding.close()
            self._onboarding = None
        # **停表。** 收尾之后才响的一枪没有任何东西接得住（窗口已经关了、托盘图标
        # 也摘了），而它会在这里多留一个还活着的 QTimer。
        self._scheduler.stop()
        # **停第二块表。** 与上面同一条理由：收尾之后才响的一枪没有东西接得住，而
        # 它（比提醒那条更坏）会去起一个**写库**的线程——那个线程在解释器收尾时
        # 还在跑，就是下一次 ``0xC0000409``。
        self._consolidation_scheduler.stop()
        # **摘掉托盘图标。** 留着的话，Windows 上那个死图标要等鼠标划过才消失——
        # 用户看到的是"我已经退出了，它还在那儿"。
        self._tray.hide()
        # **不关记忆库。** 线程都停稳之后连接是空闲的，SQLite 的 WAL 也不怕一个
        # 没 close 的进程（下次打开时自动恢复）；而 close 掉它会砸掉一类正当用法：
        # 冒烟脚本在 `app.exec()` 返回之后还要读库来打印"记住了什么"。
        # 这里留这句话是为了挡住将来那句好心的 `self._memory.close()`。
        app = QApplication.instance()
        if app is not None:
            app.quit()
