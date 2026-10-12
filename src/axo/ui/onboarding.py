"""新手引导：首启那一次把三件事定下来。

**独立的多步窗口**，不是设置窗口里的一页、也不是对话区里那张卡片。三件事——
记忆存哪、它怎么说话、用哪个模型——在设置窗口里散在三页里，而"第一次打开这个程序"
的人不知道有那三页；对话区那张卡片只管得住其中一件，而且它没有"下一步"这个概念。

## 与首启那张人设卡的关系：去重靠一行 connect

卡片（``ui/web/index.html`` 的 ``#welcome``）与这里的第 2 步问的是同一件事。向导问过
之后卡片自己就不来了——**零新逻辑**：它连的是主窗口已有的
``_on_persona_answered``，而那个槽成功之后会把 ``_persona_asking`` 置假并重推状态，
`_persona_asking` 的契约本来就是"本进程内还等不等人写人设"。

两个方向都要留着：**跳过整个向导 ≠ 回答了人设**，那时卡片照旧出现（与"没写就等于
没做这个决定"同一条立场）。

## 什么时候算"走过了"

只有「完成」与「跳过」会发 :attr:`OnboardingWizard.onboarding_finished`；**叉掉窗口
不算**——一个没做完也没说不做的向导不算决定。代价是叉掉的人下次还会看到，所以底部
必须有一个明确的「跳过」，把"我不想看"变成一个按得下去的答案，而不是靠叉窗口去猜。

这一步与那张卡片的去重规则**刻意不同**（那张卡片是从状态推导、不落标志）。理由很
具体：向导里有两步（位置保持默认、模型保持默认）**不留下任何痕迹**，"推导"出来的
判据会在每次启动都把向导弹回来，永远。而卡片问的那个决定的答案本身就是数据的一部分
（库里有没有那条锚）——那条理由不延伸到"我给用户看过一次说明没有"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .settings_dialog import BackendPage, DataPage, _hint, _section
from .settings_store import BackendSettings
from .thread_tracking import wait_for_all

__all__ = ["OnboardingWizard"]

_STEP_TITLES = {
    "welcome": "欢迎",
    "data": "记忆放在哪",
    "persona": "它怎么说话",
    "backend": "用哪个模型",
    "done": "可以开始了",
}


class OnboardingWizard(QDialog):
    """首启向导。父窗口是主窗口（与 ``SettingsDialog`` 同形，这样 ``_shutdown`` 收得住）。"""

    #: 第 1 步挑了别的文件夹。**只转发**，搬不搬由主窗口那条链决定。
    data_location_requested = Signal(str)
    restore_data_location_requested = Signal()
    #: 第 3 步按下「应用设置」。
    backend_changed = Signal(object)
    #: 第 2 步的答案。**空串 = 「以后再说」**——它与"没走过这一步"不是一回事，见模块文档。
    persona_answered = Signal(str)
    #: 只有「完成」与「跳过」会发。
    onboarding_finished = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        *,
        root: Path,
        source: str,
        backend: BackendSettings,
        builtin_persona: str,
        show_persona_step: bool,
    ) -> None:
        super().__init__(parent)
        self._live_threads: list[object] = []

        self.setWindowTitle("Axo 新手引导")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(640, 520)
        # **显式给这个窗口设图标。** 它一开就在 z 序最上面，而打包自检那一步会去
        # ``EnumWindows`` 顺序里取**第一个**非对话框的顶层窗口来问"窗口上有没有图标"
        # ——取到的正会是它。应用级图标在 ``_apply_theme`` 里已经装好了，这里只是把它
        # 挂到窗口上（Windows 不会拿应用级的兜底，见 `_install_window_icon` 那段）。
        icon = QGuiApplication.windowIcon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        # **顺序表**。第 2 步按需跳过——已经写过人设的人不该被再问一遍。页照样构造，
        # 只是不进这张表，于是"加一页"的改动仍然只有一处。
        self._order: list[str] = ["welcome", "data"]
        if show_persona_step:
            self._order.append("persona")
        self._order += ["backend", "done"]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        self.pages = QStackedWidget(self)
        self._page_index: dict[str, int] = {}
        for name in self._order:
            page = self._build_page(name, root, source, backend, builtin_persona)
            self._page_index[name] = self.pages.addWidget(page)
        outer.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        self.step_label = QLabel()
        self.step_label.setObjectName("hint")
        footer.addWidget(self.step_label)
        footer.addStretch(1)
        self.skip_button = QPushButton("跳过")
        self.skip_button.setToolTip("以后能在这里重开：设置 → 数据 → 重新打开新手引导")
        self.skip_button.clicked.connect(self._on_skip)
        footer.addWidget(self.skip_button)
        self.back_button = QPushButton("上一步")
        self.back_button.clicked.connect(self._on_back)
        footer.addWidget(self.back_button)
        self.next_button = QPushButton("下一步")
        self.next_button.setProperty("kind", "primary")
        self.next_button.clicked.connect(self._on_next)
        footer.addWidget(self.next_button)
        outer.addLayout(footer)

        self._position = 0
        self._sync()

    # ── 每一步长什么样 ──────────────────────────────────────────────────────

    def _build_page(
        self,
        name: str,
        root: Path,
        source: str,
        backend: BackendSettings,
        builtin_persona: str,
    ) -> QWidget:
        if name == "data":
            # **复用设置窗口那一页**，不重写一份。信号形状本来就是"谁宿主谁接"，
            # 在向导里接一遍即可——重写一份的下场是两处行为各自漂。
            self.data = DataPage(self)
            self.data.set_state(root, source)
            self.data.change_location_requested.connect(self.data_location_requested)
            self.data.restore_default_requested.connect(self.restore_data_location_requested)
            return self.data
        if name == "backend":
            self.backend = BackendPage(self)
            self.backend.set_state(backend)
            self.backend.apply_requested.connect(self.backend_changed)
            return self.backend
        if name == "persona":
            return self._build_persona_page(builtin_persona)
        if name == "done":
            return self._build_done_page()
        return self._build_welcome_page()

    def _build_welcome_page(self) -> QWidget:
        page = QWidget(self)
        column = QVBoxLayout(page)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(10)
        column.addWidget(_section("花一分钟，把三件事定下来"))
        column.addWidget(
            _hint(
                "接下来三步：**记忆存在哪儿**、**它该怎么跟你说话**、**用哪个模型**。"
                "每一步都能跳过后改——设置窗口里那三页随时能回到这儿。"
            )
        )
        column.addWidget(
            _hint(
                "有件事值得先说清楚：记忆存在你自己指定的文件夹里，**不归模型**。"
                "换一个模型、重新下载一份程序，它都还在——模型只是读它的那一个。"
            )
        )
        column.addStretch(1)
        return page

    def _build_persona_page(self, builtin_persona: str) -> QWidget:
        page = QWidget(self)
        column = QVBoxLayout(page)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(10)
        column.addWidget(_section("你想让我怎么说话"))
        column.addWidget(_hint("这段是内置的，改不了也删不掉："))
        builtin = QLabel(builtin_persona)
        builtin.setObjectName("builtinPersona")
        builtin.setWordWrap(True)
        builtin.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(builtin)
        column.addWidget(_hint("下面这段是你写的。写进记忆库里，立刻生效，换模型也一样："))
        self.persona_edit = QPlainTextEdit(page)
        self.persona_edit.setFixedHeight(96)
        self.persona_edit.textChanged.connect(self._sync_persona_buttons)
        column.addWidget(self.persona_edit)
        row = QHBoxLayout()
        self.persona_save = QPushButton("就这样")
        self.persona_save.setProperty("kind", "primary")
        self.persona_save.clicked.connect(
            lambda: self._answer_persona(self.persona_edit.toPlainText())
        )
        row.addWidget(self.persona_save)
        self.persona_later = QPushButton("以后再说")
        self.persona_later.clicked.connect(lambda: self._answer_persona(""))
        row.addWidget(self.persona_later)
        row.addStretch(1)
        column.addLayout(row)
        column.addWidget(_hint("「以后再说」只是这一次不写——下次打开还会问一遍（写或删都由你）。"))
        column.addStretch(1)
        self._sync_persona_buttons()
        return page

    def _build_done_page(self) -> QWidget:
        page = QWidget(self)
        column = QVBoxLayout(page)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(10)
        column.addWidget(_section("可以开始了"))
        self.done_label = QLabel()
        self.done_label.setWordWrap(True)
        self.done_label.setObjectName("hint")
        column.addWidget(self.done_label)
        column.addWidget(
            _hint(
                "随时能改：**设置 → 数据**（记忆放哪、重看这个引导）、**设置 → 人设**、"
                "**设置 → 后端 / API**。关掉主窗口是收进托盘，不是退出。"
            )
        )
        column.addStretch(1)
        return page

    # ── 前后走 ──────────────────────────────────────────────────────────────

    def _current_name(self) -> str:
        return self._order[self._position]

    def _sync(self) -> None:
        self.pages.setCurrentIndex(self._page_index[self._current_name()])
        self.step_label.setText(f"第 {self._position + 1} 步 / 共 {len(self._order)} 步")
        self.back_button.setEnabled(self._position > 0)
        last = self._position == len(self._order) - 1
        self.next_button.setText("完成" if last else "下一步")
        if last:
            self._refresh_done_page()
        self._sync_persona_buttons()

    @Slot()
    def _on_next(self) -> None:
        if self._position == len(self._order) - 1:
            self.onboarding_finished.emit()
            self.close()
            return
        self._position += 1
        self._sync()

    @Slot()
    def _on_back(self) -> None:
        if self._position == 0:
            return
        self._position -= 1
        self._sync()

    @Slot()
    def _on_skip(self) -> None:
        """**也算走过了。** 见模块文档：没做完也没说不做不算决定，而"我不想看"必须
        有一个按得下去的答案。"""
        self.onboarding_finished.emit()
        self.close()

    # ── 人设那一步 ──────────────────────────────────────────────────────────

    def _sync_persona_buttons(self) -> None:
        if not hasattr(self, "persona_save"):
            return
        self.persona_save.setEnabled(bool(self.persona_edit.toPlainText().strip()))
        self.persona_later.setEnabled(True)

    def _answer_persona(self, text: str) -> None:
        self.persona_answered.emit(text.strip())

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def set_data_note(self, note: str) -> None:
        """搬迁那条链的反馈落回第 1 步（同 `DataPage.set_note`）。"""
        self.data.set_note(note)

    def set_persona(self, user: str) -> None:
        """预填"你写的那段"。首启时是空串，重开时是现值。

        没构造那一页时（已经写过人设的人不会被再问一遍）安静地什么都不做——调用方
        不必先问一遍顺序表里有没有它。
        """
        if hasattr(self, "persona_edit"):
            self.persona_edit.setPlainText(user)
            self._sync_persona_buttons()

    def set_backend_note(self, note: str) -> None:
        self.backend.set_note(note)

    def _refresh_done_page(self) -> None:
        """最后一步把前几步的选择复述一遍——用户按「完成」之前该看见自己定了什么。"""
        chosen = self.data.path_label.text()
        personality = "写了" if self.persona_edit.toPlainText().strip() else "还没写"
        self.done_label.setText(f"记忆放在：{chosen}\n人设：{personality}")

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """**等它自己的线程停。** 这一页里有后端页那个连接测试线程，不等就是下一次
        ``0xC0000409``（同 ``SettingsDialog.closeEvent``，用的是同一个 ``wait_for_all``）。"""
        wait_for_all(list(self._live_threads))
        super().closeEvent(event)
