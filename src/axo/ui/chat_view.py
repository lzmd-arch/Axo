"""对话区：一个嵌在 Qt 窗口里的本地网页。

**为什么对话区是网页而不是原生控件**：markdown 流式渲染（表格、代码块、公式）
和 Live2D 都要求一个完整的 web 引擎。用原生 ``QTextBrowser`` 做 markdown 等于
把 CommonMark 重写一遍，而且 Live2D 根本无从谈起。

**为什么不用本地 HTTP 服务器**：``QWebChannel`` 直接把 Python 对象暴露成 JS
对象，不需要端口、不需要令牌、不需要管理服务器生命周期。少一层 IPC 就少一整类
"端口被占/启动竞态/忘记关停"的问题。

JS 与 Python 的契约（两边都是**只增不改**，否则旧前端会静默失联）::

    JS   → Python   window.bridge.send(text)                          一条聊天消息
                    window.bridge.answer_persona(text)               首启人设卡片的答案；"" = 不写
                    window.bridge.tool_confirm(call_id, granted)     确认卡片上的一次点击
                    window.bridge.voice_toggle()                     麦克风按钮：开始/结束听
                    window.bridge.attach_files(uris)                 拖进来的文件：file:// URL 的列表
                    window.bridge.call_toggle()                      开始/结束通话
                    window.bridge.call_interrupt()                   通话中插话：停播并开麦
                    window.bridge.call_mute(muted)                   通话静音开/关
    Python → JS     window.ChatHost.appendMessage(role, text, meta)   role: user|assistant|system|tool|proactive
                                                                      meta: {title?, footer?, image?}
                    window.ChatHost.pushToken(text)                  流式增量
                    window.ChatHost.endMessage()                     一片回复结束
                    window.ChatHost.setBusy(busy)                    生成中
                    window.ChatHost.setStatus(text)                  顶栏状态
                    window.ChatHost.clear()                          清空对话流
                    window.ChatHost.loadHistory(messages)            messages: [{role, text, meta}, ...]
                    window.ChatHost.setWallpaper(path)               file:// 绝对 URL；"" = 无
                    window.ChatHost.setTheme({mode, accent})         mode: dark|light
                    window.ChatHost.setWallpaperOpacity(percent)     0–100
                    window.ChatHost.setPersonaState({asking, builtin}) 首启人设卡片的显隐
                    window.ChatHost.setVoiceState(state)             "" | listening | recognizing
                    window.ChatHost.setCallState(state, meta)        off|listening|capturing|thinking|speaking|error
                    window.ChatHost.setCallLevel(level)              0–100
                    window.ChatHost.setCallPartial(text)             只显示，永不提交
                    window.ChatHost.appendToolConfirm(call_id, name, summary, meta)
                    window.ChatHost.resolveToolConfirm(call_id, granted, accepted)

**``meta`` 目前三个键**：``title``（气泡上方那行小字）、``footer``（下方那行）、
``image``（气泡里的一张图，值是 ``file:`` URL）。加键**不需要动 JS 那一侧的判断**
（多一个键它照旧渲染），但要在上面那张表里写上——理由与下一条一字不差。

**``image`` 只从 ``meta`` 来，不从正文的 Markdown 来。** 图上哪儿是 Python 说了算
（表情包目录是本地扫出来的），而「从文本里认图片」等于把「去哪儿加载」交给了模型：
那是一次可以对外发起的请求。

**这份清单漏过一次 ``setWallpaper``**（加它的时候没同步），于是那个方法在两侧之间
没有任何书面的存在证明。补全它不只是文档洁癖：这张表是"改哪边要动哪边"的唯一索引，
漏一项就意味着下一次改签名时没人会想到去改对面。

**``proactive`` 是一个新角色，而不是给 ``assistant`` 加一个开关。** 她到点主动说的
那一句（见 ``agent/proactive.py``）与"回答用户"是同一张嘴说的两件事，回看时要能一眼
分开：

* 混进 ``.msg-assistant``，回看时读到的是"这是你问来的"——而那一轮**没有 user 行**；
* 用 ``.msg-system``，那是**危险色**（与 ``tool`` 一样的报错配色），而她说得没错什么；
* 两边都不属于，于是给它一支自己的类名（底色仍是助手那一支：她说的就是她说的），
  只在左边加一道强调色。

**JS 那一侧一项都没加。** ``makeBubble`` 拼的是 ``"msg msg-" + role``，``meta.title``
本来就会被渲染成气泡上方那行小字——所以新角色在这个网页里**天然可用**，Python 侧
只要把 role 传对。这也是为什么她的话是**整段出现**而不是逐字长出来的：``pushToken``
建气泡时把角色写死成了 ``assistant``，要让流式也认角色就得给那个方法加一个参数，
而这份契约是只增不改的（见 ``ui/main_window.py`` 的 ``_start_reminder_speech``）。

**命名各自照顾被调用方**：Python → JS 那一侧是 camelCase，因为定义在 JS 里
（``setPersonaState``、``pushToken``）；JS → Python 那一侧是 snake_case，因为定义在
Python 里（``answer_persona``、``send``）。这条不写下来的话，下一个人会在 JS 里顺手
写 ``answerPersona``——而 QWebChannel 找不到槽位时的表现是**静默丢弃**，不是报错。

**``answer_persona`` 单独开一个 slot，不复用 ``send``**：人设的答案不是一条聊天消息，
走 ``send`` 会让它同时进逐字稿和记忆库，下次召回时这句话就成了"用户说过的某句话"。
空串在这里的含义（"这一次先不写"）**是推导出来的、不是约定的**：``memory.edit.can_encode``
本来就拒绝空正文，所以这条路径上空串唯一可能的解释就是"不写"。

**三个外观方法都要能被重复调用，而且要在页面加载完之后重推一次。** 主题与壁纸是
Python 侧的状态，网页是后加载的：只推一次的话，启动时那一次调用落在空页面上，
用户看到的是默认外观而不是自己设的。:meth:`ChatView._on_load_finished` 负责这件事。
:meth:`ChatView.set_persona_state` 走的是同一条路，理由多一层——卡片是否出现决定了
用户能不能打字，露在默认状态上比外观错了严重得多。

**Live2D 的接线口**已经在网页侧留好（``window.Live2DHost``），Python 侧不需要
为此改任何代码——:meth:`ChatView.push_token` 里调用 ``Live2DHost.setMouthOpen``
的那两行就是口型同步的入口，现在指向空实现。

**确认卡片走"两个方法一对"，不是"一个方法加一个回调参数"。** 点一下要经过四层
（JS 按钮 → ``bridge.tool_confirm`` → :class:`~axo.ui.agent_worker.AgentWorker`
→ 闸门），而**裁决可能不被采纳**（卡片过期、这一轮已被停止）。所以点完必须有一次
**回话**：``resolveToolConfirm`` 把卡片改成"已允许/已拒绝/已失效"并把按钮禁掉。
只画卡片、不回话的话，用户点了一个没人接的按钮，而界面看起来完全正常——他以为
那件事做了，实际上闸门把它当没答过（fail-closed 按拒绝）。这是本版唯一一条
"用户以为自己同意了、系统认为没人同意"的路径，所以它必须有一个看得见的收场。

**卡片是对话流里的一个 ``.msg`` 节点**（:meth:`ChatView.append_tool_confirm`），
不是弹窗。模态弹窗会挡住对话本身，而确认要在**上下文里**做：用户是看着助手刚说的
那句话决定要不要点头的。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .theme import Resolved

__all__ = ["ChatView", "WebBridge"]

_CALL_STATES = frozenset({"off", "listening", "capturing", "thinking", "speaking", "error"})

_WEB_ROOT = Path(__file__).resolve().parent / "web"
_INDEX = _WEB_ROOT / "index.html"


class WebBridge(QObject):
    """暴露给网页的 Python 对象。网页侧通过 ``window.bridge`` 访问它。

    **四个 slot，不是一个。** 原本这里写着"只暴露一个"，理由是每多一个方法就多
    一处"Python 与 JS 必须同时改"的耦合点。人设的答案是第二个，加它是因为**它不
    能走 ``send``**：``send`` 发出去的每条消息都会进逐字稿与记忆库，而"叫我小林"
    这句人设混在里面的话，它下次被召回时就是"用户说过的某句话"，而不是身份约束。
    两条路径的区别是数据去向，不是措辞——那正是该分成两个入口的理由。

    确认卡片的点击是第三个，理由同类：它**不是一条聊天消息**，而是一次裁决。
    混进 ``send`` 的话，用户点「允许」会被当成他说了一句"允许"——那句话进记忆库，
    下次召回时它会是一条孤零零的、指向不明的陈述。

    **麦克风是第四个，而且它与前三个不同类**：它不带正文——按一下只是"开始听"
    或"结束听"，而被识别出来的文字走的是 ``MainWindow._on_submit``（与打字**同一条
    路**），不从这里回流。理由与 :meth:`send` 那句一样："用户发了一条消息"在生产
    代码里只该有一个入口。

    四个 slot 都只做一件事：把结果转成信号。网页侧不该在这里被判断，理由同
    "用户输入这条路径尤其应该只有一个入口"。
    """

    submitted = Signal(str)
    """用户发了一条聊天消息（已去空白，且保证非空）。"""

    persona_answered = Signal(str)
    """首启卡片收了答案。空串 = 用户选了「这一次先不写」。"""

    tool_confirm_decided = Signal(str, bool)
    """确认卡片被点了一下：``(call_id, 是否允许)``。

    **不在这一层判"这一次点击还算不算数"**：那要看闸门的当前状态，而这里够不着它。
    这里只保证"一次点击 = 一次信号"，把 ``call_id`` 原样带过去由调用方比对。
    """

    files_dropped = Signal(list)
    """拖进来的文件：``file://`` URL 的列表。

    **这里不转成路径。** URI 的解码（百分号编码、盘符）在 Python 那边做一次就够，
    而这个信号只负责把"网页收到了什么"原样传出来——理由与 :meth:`voice_toggle`
    那句一样：这一层不做判断。

    **只会有 ``file:`` 开头的**：网页那边把别的协议滤掉了（拖一段网页上的图片进来
    时，``dataTransfer`` 里是 http(s) 地址，而那条路是一条我们不该去取的网络请求）。
    """

    call_toggled = Signal()
    """通话按钮被按了一下：开始或挂断。**不带参数**，与 ``voice_toggled`` 同类。"""

    call_interrupted = Signal()
    """通话中点了插话：调用方停播并开麦。"""

    call_muted = Signal(bool)
    """通话静音开关。"""

    voice_toggled = Signal()
    """麦克风按钮被按了一下。:class:`ChatView` 把它原样转发出去。

    **信号与槽同名，但必须在这里各声明一次**：``ChatView`` 连的是
    ``self._bridge.voice_toggled``，而 :meth:`voice_toggle` 里发的是
    ``self.voice_toggled``——那两个 ``self`` 是两个不同的对象。少这一行的症状是
    **应用根本起不来**（构造 ``ChatView`` 时 ``AttributeError``），而它发生在
    真实窗口里、不在任何一条测试里（gui 腿一律不构造 ``ChatView``）。
    钉住它的是 ``tests/test_ui_bridge_contract.py``。
    """

    @Slot(str)
    def send(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.submitted.emit(text)

    @Slot(str)
    def answer_persona(self, text: str) -> None:
        # **不在这里判空**（与 send 相反）：空串在这条路径上是有含义的答案。
        # 去空白照做——全是空格的答案与没写是一回事，而 can_encode 也这么看。
        self.persona_answered.emit((text or "").strip())

    @Slot()
    def call_toggle(self) -> None:
        self.call_toggled.emit()

    @Slot()
    def call_interrupt(self) -> None:
        self.call_interrupted.emit()

    @Slot(bool)
    def call_mute(self, muted: bool) -> None:
        self.call_muted.emit(bool(muted))

    @Slot()
    def voice_toggle(self) -> None:
        """麦克风按钮被按了一下。**不带参数**：它只表达"切换"，不表达"要说什么"。

        识别出来的文字不走这里（见类文档最后一段）——那一条路是
        ``MainWindow._on_submit``，与打字完全同一条。
        """
        self.voice_toggled.emit()

    @Slot(list)
    def attach_files(self, uris: list) -> None:
        """网页拖进来了一批文件。**只发信号，不判断。**"""
        if uris:
            self.files_dropped.emit([str(one) for one in uris])

    @Slot(str, bool)
    def tool_confirm(self, call_id: str, granted: bool) -> None:
        # ``call_id`` 为空的点击直接丢弃：它配不上任何一个待决请求，送上去只会
        # 白跑一趟（闸门会拒），而这里丢掉它至少能在信号上少一次噪声。
        if call_id:
            self.tool_confirm_decided.emit(call_id, bool(granted))


class _SafePage(QWebEnginePage):
    """禁止网页自己导航走或弹新窗口。

    对话内容里出现外链是常态，而一个聊天窗口没有地址栏——若不放行，点一下就
    永久停在别的页面上，回不来。所以拦下所有导航，把外链交给系统浏览器打开。
    """

    def acceptNavigationRequest(self, url: QUrl, nav_type: Any, is_main_frame: bool) -> bool:  # noqa: N802
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return not is_main_frame or url.scheme() == "file"

    def createWindow(self, window_type: Any) -> QWebEnginePage | None:  # noqa: N802, ARG002
        return None  # 不弹新窗口；外链走上面的 acceptNavigationRequest


def _local_file_uris(mime: Any) -> list[str]:
    """拖放数据里的本地文件 URI。**非本地的一律不取。**

    拖一段网页上的图片进来时 ``urls()`` 里是 http(s) 地址，而那条路是一条我们不该
    去取的网络请求（正文已经在网页里，落地成文件是另一回事）。
    """
    uris: list[str] = []
    for url in mime.urls():
        if url.isLocalFile():
            uris.append(url.toString())
    return uris


class _SafeView(QWebEngineView):
    """网页视图，外加**整块区域接收本地文件拖放**。

    网页里那份 ``dragstart``/``drop`` 监听只认得 ``dataTransfer.types`` 里有
    ``"Files"`` 的情形；而从资源管理器拖文件时，QtWebEngine 给的列表并不总是包含
    它（常常只有 ``text/uri-list``）。表现是"拖进去没反应"，或者更糟——浏览器
    自己导航到那个文件，对话区变成一张图。

    所以入口放在 Qt 这一层：视图收到带本地 URL 的拖放就直接接住，不让网页那边再
    猜一遍类型。**非文件的拖放原样交给基类**（拖一段文字进输入框照旧）。
    """

    files_dropped = Signal(list)

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 — Qt 命名
        if _local_file_uris(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        if _local_file_uris(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        uris = _local_file_uris(event.mimeData())
        if uris:
            event.acceptProposedAction()
            self.files_dropped.emit(uris)
            return
        super().dropEvent(event)


class ChatView(QWidget):
    """对话区控件。"""

    message_submitted = Signal(str)
    """用户按了发送。"""

    persona_answered = Signal(str)
    """首启卡片收了答案。空串 = 用户选了「这一次先不写」。"""

    tool_confirm_decided = Signal(str, bool)
    """确认卡片被点了一下：``(call_id, 是否允许)``。由 :class:`WebBridge` 转发。"""

    voice_toggled = Signal()
    """麦克风按钮被按了一下。**主窗口据此开始/结束录音**（见 ``ui/voice.py``）。"""

    call_toggled = Signal()
    """通话按钮：开始或挂断。"""

    call_interrupted = Signal()
    """通话中插话。"""

    call_muted = Signal(bool)
    """通话静音开关。"""

    files_dropped = Signal(list)
    """拖进来一批文件（``file://`` URL）。**与桥那个同名但不是一个对象**——
    理由见 :attr:`WebBridge.voice_toggled` 那条注释：少声明一次，构造就会炸。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = WebBridge(self)
        self._bridge.submitted.connect(self.message_submitted)
        self._bridge.persona_answered.connect(self.persona_answered)
        self._bridge.tool_confirm_decided.connect(self.tool_confirm_decided)
        self._bridge.voice_toggled.connect(self.voice_toggled)
        self._bridge.call_toggled.connect(self.call_toggled)
        self._bridge.call_interrupted.connect(self.call_interrupted)
        self._bridge.call_muted.connect(self.call_muted)
        self._bridge.files_dropped.connect(self.files_dropped)

        #: 外观三件套的**当前值**。存下来是为了在 ``loadFinished`` 之后重推：
        #: 网页是异步加载的，构造期那次推送落在空页面上。不缓存的话就只能等
        #: 调用方在加载完之后再喊一次——而"谁负责重推"一旦含糊，症状是启动后
        #: 外观是默认值、动一下设置才恢复。
        self._theme: dict[str, str] = {}
        self._wallpaper: str = ""
        self._wallpaper_opacity: int = -1  # -1 = 还没设过，别推

        #: 人设卡片的当前状态，**同样是缓存 + 重推**。这里比外观那三项更要紧：
        #: 网页侧的输入框在"正等人设"期间是禁用的，推送丢了就是输入框点不动，
        #: 而页面上没有任何一处会说为什么。``None`` = 还没设过。
        self._persona: dict[str, Any] | None = None

        #: 录音那一路的状态（``""`` / ``"listening"`` / ``"recognizing"``），
        #: **同样缓存 + 重推**。它比外观那三项还要紧一点：页面一刷新，按钮会显示成
        #: 空闲，而麦克风**还开着**——用户看着一个说"没在听"的按钮，它却在听。
        self._voice_state: str = ""

        #: 通话状态三件，**同样缓存 + 重推**。页面刷新之后按钮显示挂断而麦克风还
        #: 开着，是这套缓存要防的那一类事故。
        self._call_state: str = ""
        self._call_meta: dict[str, Any] = {}
        self._call_level: int = -1
        self._call_partial: str = ""

        self._view = _SafeView(self)
        self._page = _SafePage(self._view)
        self._view.setPage(self._page)

        settings = self._page.settings()
        # 本地网页要能读同目录下的 js/css
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        # 反过来关掉：本地内容不该去访问远程资源。壁纸与 Live2D 模型都走本地文件，
        # 这个口子开着只会让"网页偷偷联网"成为可能。
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)

        channel = QWebChannel(self._page)
        channel.registerObject("bridge", self._bridge)
        self._page.setWebChannel(channel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        # 网页侧要用它把相对路径（壁纸、将来的 Live2D 模型）拼成 file:// 绝对
        # 路径。必须等页面加载完再注入——`loadFinished` 每次都触发，所以刷新之后
        # 它也还在。
        self._view.files_dropped.connect(self.files_dropped)
        self._view.loadFinished.connect(self._on_load_finished)
        self._view.load(QUrl.fromLocalFile(str(_INDEX)))

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            return
        base = QUrl.fromLocalFile(str(_WEB_ROOT)).toString()
        self._page.runJavaScript(f"window.BASE = {json.dumps(base)};")
        # 重推外观。**必须排在 BASE 之后**：setWallpaper 会拿 window.BASE 拼
        # 相对路径（本项目的壁纸走绝对 file: URL，用不上它，但顺序上依赖它才是
        # 对的——将来若有相对路径的资源，靠的就是这一句）。
        if self._theme:
            self._call("setTheme", self._theme)
        if self._wallpaper_opacity >= 0:
            self._call("setWallpaperOpacity", self._wallpaper_opacity)
        if self._wallpaper:
            self._call("setWallpaper", self._wallpaper)
        # 人设卡片排最后：asking 时它会抢焦点（focus 到那个输入框），排在
        # setStatus 之类的后面才不会被别的东西把焦点拿走。
        if self._persona is not None:
            self._call("setPersonaState", self._persona)
        # 语音状态与外观同一条理由：页面是后加载的，只推一次会落在空页面上。
        if self._voice_state:
            self._call("setVoiceState", self._voice_state)
        if self._call_state:
            self._call("setCallState", self._call_state, self._call_meta)
        if self._call_level >= 0:
            self._call("setCallLevel", self._call_level)
        if self._call_partial:
            self._call("setCallPartial", self._call_partial)

    # ── 向网页推送 ──────────────────────────────────────────────────────────

    def _call(self, function: str, *args: Any) -> None:
        """在网页里调用 ``window.ChatHost.<function>(...)``。

        **参数一律走 ``JSON.stringify`` 而不是手拼字符串**。手拼的代价在中文与
        引号上立现：``f'"{text}"'`` 遇到换行、反斜杠或引号就把整段 JS 语法搞崩，
        而且是静默崩——页面还在，只是不再响应。JSON 转义是唯一稳妥的做法，且
        ``json.dumps`` 的默认 ``ensure_ascii=True`` 会把非 ASCII 转成 ``\\uXXXX``，
        顺带绕开了脚本编码问题。
        """
        payload = ", ".join(json.dumps(arg, ensure_ascii=True) for arg in args)
        self._page.runJavaScript(f"window.ChatHost && window.ChatHost.{function}({payload});")

    def set_call_state(self, state: str, meta: dict[str, Any] | None = None) -> None:
        """通话状态。认不出的值当 ``off``（理由同 :meth:`set_voice_state`）。"""
        state = state if state in _CALL_STATES else "off"
        self._call_state = state
        self._call_meta = dict(meta or {})
        self._call("setCallState", state, self._call_meta)

    def set_call_level(self, level: int) -> None:
        """输入电平 0–100。**相同值不重复下推**——每次都是一次 ``runJavaScript``。"""
        level = max(0, min(100, int(level)))
        if level == self._call_level:
            return
        self._call_level = level
        self._call("setCallLevel", level)

    def set_call_partial(self, text: str) -> None:
        """实时转写**只显示**。空串清空。"""
        text = text or ""
        if text == self._call_partial:
            return
        self._call_partial = text
        self._call("setCallPartial", text)

    def append_message(self, role: str, text: str, meta: dict[str, Any] | None = None) -> None:
        """追加一条完整消息。"""
        self._call("appendMessage", role, text, meta or {})

    def push_token(self, text: str) -> None:
        """推送一段流式增量。"""
        if text:
            self._call("pushToken", text)

    def end_message(self) -> None:
        """标记当前这片回复结束。"""
        self._call("endMessage")

    def set_busy(self, busy: bool) -> None:
        """生成中/空闲。网页据此决定发送按钮与输入框的状态。"""
        self._call("setBusy", busy)

    def set_status(self, text: str) -> None:
        self._call("setStatus", text)

    def set_voice_state(self, state: str) -> None:
        """录音那一路的状态：``""`` / ``"listening"`` / ``"recognizing"``。

        **认不出的值当空串**：网页侧只认这三个，而一个错字在那里表现为"按钮不亮"——
        与"没在听"长得一模一样。宁可不亮。
        """
        self._voice_state = state if state in ("", "listening", "recognizing") else ""
        self._call("setVoiceState", self._voice_state)

    def clear(self) -> None:
        self._call("clear")

    def load_history(self, messages: Sequence[dict[str, str]]) -> None:
        """整体替换对话流。用于切换会话。"""
        self._call("loadHistory", list(messages))

    def set_wallpaper(self, path: str | None) -> None:
        """设置壁纸。``None`` 表示恢复默认纯色。

        ``path`` 应当是 ``file://`` 绝对 URL（``QUrl.fromLocalFile(...).toString()``），
        不是文件系统路径也不是相对路径——见 :mod:`axo.paths` 里壁纸目录
        在数据目录下、**不在 web 根下**这一事实。
        """
        self._wallpaper = path or ""
        self._call("setWallpaper", self._wallpaper)

    def set_wallpaper_opacity(self, percent: int) -> None:
        """壁纸可见度，0–100。与 :attr:`~axo.ui.theme.Theme.wallpaper_visible`
        是同一个数，**不在这里夹紧**——夹紧是 ``Theme`` 的职责，这里再夹一次会让
        "设置里是 90、画出来是 70"变成两处各有一个说法。"""
        self._wallpaper_opacity = int(percent)
        self._call("setWallpaperOpacity", self._wallpaper_opacity)

    def set_theme(self, resolved: Resolved) -> None:
        """把**解析后**的主题推给网页。

        收 :class:`~axo.ui.theme.Resolved` 而不是 :class:`~axo.ui.theme.Theme`：
        网页要的是"实际画出来的主色"（浅色下可能已经换过一支），而 ``Resolved``
        正是那个问题的答案。传 ``Theme`` 的话每个调用方都得自己 :func:`resolve`
        一遍，两处各算一次是这类代码最容易漂的地方。

        **只发 mode 与 accent 两项。** 网页侧用得到用户主色的地方只有链接、输入框
        焦点边与发送键，其余颜色由 ``[data-theme]`` 那一块按明暗给——那份值在
        ``css/app.css`` 里，与 :mod:`axo.ui.theme` 的 token 同源。
        """
        self._theme = {"mode": resolved.mode, "accent": resolved.accent}
        self._call("setTheme", self._theme)

    def set_persona_state(self, *, asking: bool, builtin: str) -> None:
        """首启人设卡片的显隐。``asking=True`` 时卡片出现、输入框暂时禁用。

        **``builtin`` 是库里那条的正文，不是常量**——调用方要读实际值传进来，
        不要图省事填 :data:`~axo.memory.persona.DEFAULT_PERSONA`。库里
        那条可能是在旧版本写下的（正文字面量改过），而卡片要展示的正是"现在这个
        模型读到的身份"，展示一处、库里一处，是最不该有的那种分叉。

        ``asking=False`` 也要推：网页侧"能不能打字"是**由这次推送解开**的——
        它同时是那个问题的答案。所以调用方不能省掉这一句。
        """
        self._persona = {"asking": bool(asking), "builtin": builtin or ""}
        self._call("setPersonaState", self._persona)

    def append_tool_confirm(
        self,
        call_id: str,
        tool_name: str,
        summary: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """在对话流里追加一张确认卡片。

        ``summary`` 是**模板从参数生成的**（见
        :func:`~axo.agent.gate.summarize`），不是模型写的——用户是照着
        这句话决定要不要点头的，让模型自己解释它准备做什么是幻觉的标准配方。

        ``meta`` 目前只有一项：``{"timeoutMs": 毫秒}``。卡片据此显示倒计时并在到点
        之后把按钮灰掉——**灰掉只是显示**，真正的裁决在闸门那一侧超时后按拒绝（见
        :data:`~axo.agent.gate.APPROVAL_TIMEOUT_S`）。两处都以"拒绝"
        收场，所以它们不一致的最坏后果只是卡片晚了半秒变灰。
        """
        self._call("appendToolConfirm", call_id, tool_name, summary, meta or {})

    def resolve_tool_confirm(self, call_id: str, *, granted: bool, accepted: bool) -> None:
        """卡片有结果了：把按钮禁掉，并写明是哪一种结果。

        ``accepted=False`` 表示这次点击**没有被闸门采纳**（卡片过期、这一轮已被
        停止）。它必须与"拒绝了"分开显示：用户点的是「允许」，而系统记下的是没发生。
        不区分的话，那正是一条"用户以为他同意了、系统认为没人同意"的静默分歧。
        """
        self._call("resolveToolConfirm", call_id, bool(granted), bool(accepted))
