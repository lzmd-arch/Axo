"""网页侧的冒烟测试：走一遍 Python→JS 的真实调用序列，把渲染结果打出来。

放在 tools/ 而不是 tests/：它需要一个真实的 QtWebEngine 进程，比单元测试重得多，
不适合进每次提交的钩子。用它是为了在改 CSS/JS 之后快速看一眼渲染对不对。

**必须有显示器，不能在离屏下跑。** 离屏下 QtWebEngine 一渲染就把整个进程硬杀掉：
没有异常、没有回溯、没有 stderr、连退出码都没有（``atexit`` 也不跑），看起来就像
"脚本什么都没干"。第一次 ``runJavaScript``、或对窗口调 ``show()``/``resize()`` 都会
触发。所以别的冒烟脚本可以 ``QT_QPA_PLATFORM=offscreen``，这一个不行——凡是新增的
渲染断言都只能落在这里，靠人手动跑着看。

它验五件事，都是**没法在 pytest 里验**的（见上面那段）：

1. 流式 markdown 的渲染、以及 ``<script>`` 与 ``javascript:`` 链接被挡住；
2. **``file://`` 壁纸真能画出来**——壁纸住在数据目录、**不在 web 根下**，所以
   它靠的是 ``LocalContentCanAccessFileUrls`` 那一条设置。**这是本脚本里唯一
   一个"猜错了就得回改设计"的检查**：万一某个 QtWebEngine 版本不放行，
   传输方式要改成带体积上限的 base64（见计划书 §三）。
3. ``setTheme`` / ``setWallpaperOpacity`` 真的改到了 CSS 变量上；
4. 空状态在 ``clear()`` 之后回来、在有消息时让位。
5. **首启人设卡片**：``setPersonaState`` 的显隐、``#empty`` 与它的互斥、以及
   "正等人设时输入框点不动"（``disabled`` 读得回来吗）。第 5 项里还走了一遍
   **JS → Python** 的半程（点按钮 → ``bridge.answer_persona`` → 信号），因为那是
   新的第二个 slot：QWebChannel 找不到槽位时**是静默丢弃**，两侧名字对不上不会
   报错，只会在真机上表现为"点了没反应"。

用法::

    uv run --extra app python tools/_smoke_webview.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from axo.memory.persona import DEFAULT_PERSONA
from axo.ui.app import configure_isolated_state
from axo.ui.chat_view import ChatView
from axo.ui.theme import PRESETS, Theme, resolve

STREAMED = [
    "这是一段",
    "逐字",
    "出来的",
    "回复。\n\n",
    "带 `行内代码` 和列表：\n",
    "- 第一项\n",
    "- 第二项\n\n",
    "```python\nprint('hi')\n```\n\n",
    "> 引用一行\n\n",
    "危险内容：<script>alert(1)</script>\n",
    "[坏链接](javascript:alert(1))\n",
    "[好链接](https://example.com)\n",
]

#: 试的第二支主色。取**最后一个预设**（琥珀）而不是第二支：三支里它离默认色最远，
#: 万一 ``--accent`` 没真的写进去，读回来的默认值一眼就能看出来。
_TRY_PRESET = PRESETS[-1]
_TRY_VISIBLE = 60


def _make_wallpaper() -> Path:
    """造一张壁纸放在**系统临时目录**，不是数据目录。

    刻意放在 web 根与数据目录之外——本脚本要验的就是"跨目录的 ``file://`` 能不能
    加载"。放在 web 根下的话，那个设置项坏了它照样能画出来，于是这条检查空转通过。
    """
    path = Path(tempfile.gettempdir()) / "biosnn-smoke-wallpaper.png"
    image = QImage(64, 40, QImage.Format.Format_RGB32)
    image.fill(QColor("#7a4bd0"))
    if not image.save(str(path), "PNG"):  # pragma: no cover - 磁盘坏了才会有
        raise SystemExit(f"写不出壁纸 {path}")
    return path


def main() -> int:
    # 隔离。这个脚本会起真的 QWebEngineView，不设这两句它会往使用者的真数据目录与
    # 真注册表里写东西（WebEngine 的管线缓存也在其中），而在输出里一个字都看不出来。
    # **都必须在建 QApplication 之前**——理由见 configure_isolated_state 的 docstring。
    os.environ["AXO_DATA"] = tempfile.mkdtemp(prefix="memterm-webview-")
    os.environ["AXO_ISOLATED_STATE"] = tempfile.mkdtemp(prefix="memterm-webview-state-")
    configure_isolated_state()
    app = QApplication.instance() or QApplication(sys.argv)
    view = ChatView()
    view.resize(900, 700)
    wallpaper = _make_wallpaper()

    #: Python 侧收到的答案，按到达顺序。**这就是 JS→Python 那半程的判据**：
    #: 两侧名字对不上（JS 写成 answerPersona、或槽名拼错）时 QWebChannel 只是
    #: 静静丢掉，界面上完全看不出来——只有这个列表会空着。
    answers: list[str] = []
    view.persona_answered.connect(answers.append)

    def ask(script: str, then) -> None:
        """跑一段 JS，把结果交给 ``then``。**每一步都单独一次 ``runJavaScript``**
        ——串成一段长脚本的话，中间哪一步炸了就再也看不到后面那几步的结果。"""
        view._page.runJavaScript(script, then)

    def on_load(ok: bool) -> None:
        if not ok:
            print("页面加载失败")
            app.quit()
            return
        view.append_message("user", "你好，这是**用户**的消息，包含 <b>标签</b>。")
        for piece in STREAMED:
            view.push_token(piece)
        view.end_message()
        view.append_message("system", "**系统消息**：后端不可用。")
        QTimer.singleShot(900, step_html)

    # ── 1. 正文渲染与消毒 ─────────────────────────────────────────────────

    def step_html() -> None:
        ask('document.getElementById("stream").innerHTML', report_html)

    def report_html(html: str) -> None:
        print("--- 渲染出的 HTML ---")
        print(html)
        ask(
            'JSON.stringify({bubbles: document.querySelectorAll(".msg").length,'
            ' scripts: document.querySelectorAll("#stream script").length,'
            " badHref: document.querySelectorAll('#stream a[href^=\"javascript\"]').length,"
            ' emptyHidden: document.getElementById("empty").hidden})',
            report_guard,
        )

    def report_guard(summary: str) -> None:
        print("--- 正文自检 ---")
        print(summary)
        print("（期望 bubbles>0、scripts=0、badHref=0、emptyHidden=true）")
        step_appearance()

    # ── 2. 主题 / 壁纸 / 可见度 ────────────────────────────────────────────

    def step_appearance() -> None:
        view.set_theme(resolve(Theme(preset=_TRY_PRESET.key), "dark"))
        view.set_wallpaper(QUrl.fromLocalFile(str(wallpaper)).toString())
        view.set_wallpaper_opacity(_TRY_VISIBLE)
        # 半秒：背景图是异步解码的，立刻读会读到还没设上的 ``none``——
        # 而那看起来与"file:// 被拦了"一模一样，会白白误导一次。
        QTimer.singleShot(700, dump_style)

    def dump_style() -> None:
        ask(
            "JSON.stringify({"
            " accent: getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(),"
            " theme: document.documentElement.dataset.theme,"
            " hasWallpaper: document.body.classList.contains('has-wallpaper'),"
            " wallpaperImage: getComputedStyle(document.getElementById('wallpaper')).backgroundImage.slice(0, 60),"
            " scrim: getComputedStyle(document.getElementById('scrim')).opacity,"
            " dim: getComputedStyle(document.documentElement).getPropertyValue('--wallpaper-dim').trim()"
            "})",
            report_style,
        )

    def report_style(summary: str) -> None:
        print("--- 外观自检 ---")
        print(summary)
        print(
            f"（期望 accent={_TRY_PRESET.accent}、theme=dark、hasWallpaper=true、"
            f'wallpaperImage 以 url("file: 开头、dim={1 - _TRY_VISIBLE / 100:.2f}）'
        )
        step_empty()

    # ── 3. 空状态回来 ─────────────────────────────────────────────────────

    def step_empty() -> None:
        view.clear()
        QTimer.singleShot(300, dump_empty)

    def dump_empty() -> None:
        ask(
            'JSON.stringify({bubbles: document.querySelectorAll(".msg").length,'
            ' emptyHidden: document.getElementById("empty").hidden,'
            ' emptyText: document.getElementById("empty").textContent.trim().slice(0, 20)})',
            report_empty,
        )

    def report_empty(summary: str) -> None:
        print("--- 空状态自检 ---")
        print(summary)
        print("（期望 bubbles=0、emptyHidden=false、emptyText 非空）")
        step_persona()

    # ── 4. 首启人设卡片 ───────────────────────────────────────────────────
    #
    # 这一段的判据是**成对**的：卡片出现与 #empty 让位是同一件事，输入框禁用与
    # "能打字了"也是同一件事。只读一个方向的话，最坏的那种"两个都显示/两个都没
    # 显示"反而看不出来。

    def step_persona() -> None:
        # 推**真的那段**（库里内置那条的散文），不是随便编一句：卡片上给用户看的
        # 就是这三个换行，所以 white-space 那条 CSS 只有在真文本下才算验过。
        view.set_persona_state(asking=True, builtin=DEFAULT_PERSONA)
        QTimer.singleShot(300, dump_asking)

    _CARD_STATE = (
        "JSON.stringify({"
        ' welcomeHidden: document.getElementById("welcome").hidden,'
        ' emptyHidden: document.getElementById("empty").hidden,'
        ' sendDisabled: document.getElementById("send").disabled,'
        ' inputDisabled: document.getElementById("input").disabled,'
        ' placeholder: document.getElementById("input").placeholder.slice(0, 10),'
        ' builtinHead: document.getElementById("welcome-builtin").textContent.slice(0, 16),'
        ' builtinLines: document.getElementById("welcome-builtin").textContent.split("\\n").length,'
        ' whiteSpace: getComputedStyle(document.getElementById("welcome-builtin")).whiteSpace,'
        ' saveDisabled: document.getElementById("persona-save").disabled,'
        # 「先不写」不是"永不再问"——两处都要说出来，而这是它们唯一能被量到的地方：
        # 第一次手动验收时，「选了它、重启又问一次」被当成 bug 报了上来。
        ' skipTitle: document.getElementById("persona-skip").title.includes("只是这一次"),'
        ' skipNote: document.getElementById("welcome-skip-note").textContent.includes("下次启动")'
        "})"
    )

    def dump_asking() -> None:
        ask(_CARD_STATE, report_asking)

    def report_asking(summary: str) -> None:
        print("--- 人设卡片：正在问 ---")
        print(summary)
        print(
            "（期望 welcomeHidden=false、emptyHidden=true、sendDisabled=true、"
            f"inputDisabled=true、builtinLines={DEFAULT_PERSONA.count(chr(10)) + 1}、"
            "whiteSpace=pre-line、saveDisabled=true、skipTitle=true、skipNote=true）"
        )
        # 走一遍「填字 → 保存」。**先读按钮的初始态**（应为灰：框是空的），再派发
        # 一个 input 事件把它点亮——这一步同时验了"空框不可提交"那条规则。
        ask(
            "var ta = document.getElementById('persona-input');"
            "var btn = document.getElementById('persona-save');"
            "var before = btn.disabled;"
            "ta.value = '说话简短一点，不要客套话。';"
            "ta.dispatchEvent(new Event('input'));"
            "JSON.stringify({disabledBeforeTyping: before, disabledAfterTyping: btn.disabled})",
            report_typing,
        )

    def report_typing(summary: str) -> None:
        print("--- 人设卡片：空框与填字 ---")
        print(summary)
        print("（期望 disabledBeforeTyping=true、disabledAfterTyping=false）")
        ask("document.getElementById('persona-save').click(); true", report_save_click)

    def report_save_click(_ok: object) -> None:
        print("--- 人设卡片：点了「就这样」 ---")
        # 按钮的**文案**要在这里对一下：这一段验的是 JS→Python 那半程，而
        # answer_persona 收不到东西时，界面看起来和"保存成功了"没有区别。
        print("（期望 Python 侧收到那句正文；见下面的回执）")
        ask("document.getElementById('persona-skip').click(); true", report_skip_click)

    def report_skip_click(_ok: object) -> None:
        print("--- 人设卡片：点了「这一次先不写」 ---")
        # 回执**不在这里打印**：click 里的 bridge 调用是异步过 QWebChannel 的，
        # 这一句回调跑的时候它多半还没到 Python。全套答案在末尾一起打。
        print("（期望 Python 侧收到空串；见末尾的回执）")
        view.set_persona_state(asking=False, builtin=DEFAULT_PERSONA)
        QTimer.singleShot(300, dump_idle)

    def dump_idle() -> None:
        ask(_CARD_STATE, finish)

    def finish(summary: str) -> None:
        print("--- 人设卡片：已收起 ---")
        print(summary)
        print(
            "（期望 welcomeHidden=true、emptyHidden=false、sendDisabled=false、"
            "inputDisabled=false）"
        )
        print(f"--- Python 侧收到的答案（按顺序）--- {answers!r}")
        print("（期望 ['说话简短一点，不要客套话。', '']——第二个是空串）")
        app.quit()

    view._view.loadFinished.connect(on_load)
    QTimer.singleShot(30000, lambda: (print("超时"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
