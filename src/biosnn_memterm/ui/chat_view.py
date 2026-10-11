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
    Python → JS     window.ChatHost.appendMessage(role, text, meta)   role: user|assistant|system|tool
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
                    window.ChatHost.appendToolConfirm(call_id, name, summary, meta)
                    window.ChatHost.resolveToolConfirm(call_id, granted, accepted)

**这份清单漏过一次 ``setWallpaper``**（加它的时候没同步），于是那个方法在两侧之间
没有任何书面的存在证明。补全它不只是文档洁癖：这张表是"改哪边要动哪边"的唯一索引，
漏一项就意味着下一次改签名时没人会想到去改对面。

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
（JS 按钮 → ``bridge.tool_confirm`` → :class:`~biosnn_memterm.ui.agent_worker.AgentWorker`
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

_WEB_ROOT = Path(__file__).resolve().parent / "web"
_INDEX = _WEB_ROOT / "index.html"


class WebBridge(QObject):
    """暴露给网页的 Python 对象。网页侧通过 ``window.bridge`` 访问它。

    **三个 slot，不是一个。** 原本这里写着"只暴露一个"，理由是每多一个方法就多
    一处"Python 与 JS 必须同时改"的耦合点。人设的答案是第二个，加它是因为**它不
    能走 ``send``**：``send`` 发出去的每条消息都会进逐字稿与记忆库，而"叫我小林"
    这句人设混在里面的话，它下次被召回时就是"用户说过的某句话"，而不是身份约束。
    两条路径的区别是数据去向，不是措辞——那正是该分成两个入口的理由。

    确认卡片的点击是第三个，理由同类：它**不是一条聊天消息**，而是一次裁决。
    混进 ``send`` 的话，用户点「允许」会被当成他说了一句"允许"——那句话进记忆库，
    下次召回时它会是一条孤零零的、指向不明的陈述。

    三个 slot 都只做一件事：把结果转成信号。网页侧不该在这里被判断，理由同
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


class ChatView(QWidget):
    """对话区控件。"""

    message_submitted = Signal(str)
    """用户按了发送。"""

    persona_answered = Signal(str)
    """首启卡片收了答案。空串 = 用户选了「这一次先不写」。"""

    tool_confirm_decided = Signal(str, bool)
    """确认卡片被点了一下：``(call_id, 是否允许)``。由 :class:`WebBridge` 转发。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = WebBridge(self)
        self._bridge.submitted.connect(self.message_submitted)
        self._bridge.persona_answered.connect(self.persona_answered)
        self._bridge.tool_confirm_decided.connect(self.tool_confirm_decided)

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

        self._view = QWebEngineView(self)
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

    def clear(self) -> None:
        self._call("clear")

    def load_history(self, messages: Sequence[dict[str, str]]) -> None:
        """整体替换对话流。用于切换会话。"""
        self._call("loadHistory", list(messages))

    def set_wallpaper(self, path: str | None) -> None:
        """设置壁纸。``None`` 表示恢复默认纯色。

        ``path`` 应当是 ``file://`` 绝对 URL（``QUrl.fromLocalFile(...).toString()``），
        不是文件系统路径也不是相对路径——见 :mod:`biosnn_memterm.paths` 里壁纸目录
        在数据目录下、**不在 web 根下**这一事实。
        """
        self._wallpaper = path or ""
        self._call("setWallpaper", self._wallpaper)

    def set_wallpaper_opacity(self, percent: int) -> None:
        """壁纸可见度，0–100。与 :attr:`~biosnn_memterm.ui.theme.Theme.wallpaper_visible`
        是同一个数，**不在这里夹紧**——夹紧是 ``Theme`` 的职责，这里再夹一次会让
        "设置里是 90、画出来是 70"变成两处各有一个说法。"""
        self._wallpaper_opacity = int(percent)
        self._call("setWallpaperOpacity", self._wallpaper_opacity)

    def set_theme(self, resolved: Resolved) -> None:
        """把**解析后**的主题推给网页。

        收 :class:`~biosnn_memterm.ui.theme.Resolved` 而不是 :class:`~biosnn_memterm.ui.theme.Theme`：
        网页要的是"实际画出来的主色"（浅色下可能已经换过一支），而 ``Resolved``
        正是那个问题的答案。传 ``Theme`` 的话每个调用方都得自己 :func:`resolve`
        一遍，两处各算一次是这类代码最容易漂的地方。

        **只发 mode 与 accent 两项。** 网页侧用得到用户主色的地方只有链接、输入框
        焦点边与发送键，其余颜色由 ``[data-theme]`` 那一块按明暗给——那份值在
        ``css/app.css`` 里，与 :mod:`biosnn_memterm.ui.theme` 的 token 同源。
        """
        self._theme = {"mode": resolved.mode, "accent": resolved.accent}
        self._call("setTheme", self._theme)

    def set_persona_state(self, *, asking: bool, builtin: str) -> None:
        """首启人设卡片的显隐。``asking=True`` 时卡片出现、输入框暂时禁用。

        **``builtin`` 是库里那条的正文，不是常量**——调用方要读实际值传进来，
        不要图省事填 :data:`~biosnn_memterm.memory.persona.DEFAULT_PERSONA`。库里
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
        :func:`~biosnn_memterm.agent.gate.summarize`），不是模型写的——用户是照着
        这句话决定要不要点头的，让模型自己解释它准备做什么是幻觉的标准配方。

        ``meta`` 目前只有一项：``{"timeoutMs": 毫秒}``。卡片据此显示倒计时并在到点
        之后把按钮灰掉——**灰掉只是显示**，真正的裁决在闸门那一侧超时后按拒绝（见
        :data:`~biosnn_memterm.agent.gate.APPROVAL_TIMEOUT_S`）。两处都以"拒绝"
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
