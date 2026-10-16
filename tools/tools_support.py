"""冒烟脚本共用的替身件。

**为什么要有这个文件。** ``_smoke_e2e.py``、``_smoke_recall.py``、``_smoke_memory.py``
都要把真窗口跑起来、又都不能碰 QtWebEngine——原因写在 ``_smoke_e2e.py`` 的模块文档里：
本机离屏模式下第一次 ``runJavaScript`` 会让整个进程被硬杀（无异常、无退出码、
``atexit`` 都不跑）。三个脚本各抄一份替身的话，某天加一个 ``ChatView`` 方法就得
记得改三处，而漏改的症状是"某个冒烟脚本在离屏下偶尔崩"。

**这里只放不碰网络的替身。** 真网络、真模型的断言留给各自的脚本。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

__all__ = ["RecordingChatView"]


class RecordingChatView(QWidget):
    """``ChatView`` 的替身：记下被调了什么，不碰网页。

    接口与真件一致（``MainWindow`` 只用到这几个方法），所以窗口的接线逻辑被原样
    执行——替换掉的只有最后那一步"把字推给 Chromium"。继承 ``QWidget`` 而不是
    ``QObject``，因为真件就是个控件，要能放进 ``QSplitter``。

    记的是**调用序列**而不只是结果：真出问题时，"死在 pushToken 之前"和"死在
    endMessage 之后"是两种完全不同的故障，而只看最终气泡是分不出来的。

    **不许掉队。** 这里是手抄的接口，抄漏一处的症状是：真件加了方法、``MainWindow``
    里接了线，而冒烟脚本在那一行抛 ``AttributeError``——它会安静地坏在那里，直到
    某天有人真的去跑它。所以 ``tests/test_tools_standin.py`` 拿 ``MainWindow`` 里
    出现的每一个 ``chat_view.<名字>`` 逐个核这里有没有定义。
    """

    message_submitted = Signal(str)
    persona_answered = Signal(str)
    tool_confirm_decided = Signal(str, bool)
    voice_toggled = Signal()
    files_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.calls: list[str] = []
        self.streamed: list[str] = []
        #: 最后一次 ``setPersonaState`` 推了什么。**参数留一份原文**：``calls`` 里
        #: 那句只记了 asking，而"卡片上给用户看的那段身份是哪一份"同样要有据可查
        #: （它必须来自库里实际那条，不是 DEFAULT_PERSONA 常量）。
        self.persona_state: dict | None = None

    def push_token(self, text: str) -> None:
        self.streamed.append(text)
        self.calls.append("pushToken")

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append(f"appendMessage({role})")

    def end_message(self) -> None:
        self.calls.append("endMessage")

    def set_busy(self, busy: bool) -> None:
        self.calls.append(f"setBusy({busy})")

    def set_status(self, text: str) -> None:
        self.calls.append("setStatus")

    def set_voice_state(self, state: str) -> None:
        self.calls.append(f"setVoiceState({state})")

    def clear(self) -> None:
        self.calls.append("clear")

    def load_history(self, messages: object) -> None:
        self.calls.append("loadHistory")

    def set_wallpaper(self, path: str | None) -> None:
        self.calls.append("setWallpaper")

    def set_wallpaper_opacity(self, percent: int) -> None:
        self.calls.append(f"setWallpaperOpacity({percent})")

    def set_theme(self, resolved: object) -> None:
        # 参数是 `theme.Resolved`，这里**不 import 它**：这个替身刻意只依赖"长什么
        # 样"（鸭子类型），不依赖具体类型。否则 `ui/theme.py` 里一改名，坏掉的会是
        # 冒烟脚本，而不是那条真正需要被验证的调用链。
        self.calls.append("setTheme")

    def set_persona_state(self, *, asking: bool, builtin: str) -> None:
        # 关键字参数不可省：真件就是这么定义的，而 `MainWindow` 用关键字调它。
        # 这里若收成位置参数，两边签名不一致的问题会等到真件上才暴露。
        self.persona_state = {"asking": bool(asking), "builtin": builtin}
        self.calls.append(f"setPersonaState({asking})")

    def append_tool_confirm(
        self, call_id: str, tool_name: str, summary: str, meta: dict | None = None
    ) -> None:
        self.calls.append(f"appendToolConfirm({tool_name})")

    def resolve_tool_confirm(self, call_id: str, *, granted: bool, accepted: bool) -> None:
        # 关键字参数不可省，理由同 ``set_persona_state``。
        self.calls.append(f"resolveToolConfirm({granted},{accepted})")
