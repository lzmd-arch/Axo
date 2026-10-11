"""设置窗口：**一个能长大的设置中心，不是一次性对话框。**

左分类列表 + 右 ``QStackedWidget``，而不是 ``QTabWidget``。选这个形状是因为
"加一个分类 = 加一个 page widget，不动导航"——用户明确说了要有"API 设置等等"，
那就得假设它还会长。

**实时预览，不是确定后才看效果。** 主色与壁纸可见度是"必须看着调"的参数：
拖滑杆时看不到变化，等于让人盲调。所以每次改动都立刻
:attr:`SettingsDialog.theme_changed` 出去，由主窗口 apply 并**回填**
``Resolved``（见 :meth:`SettingsDialog.set_resolved`）——回填而不是让本窗口自己再
:func:`~biosnn_memterm.ui.theme.resolve` 一遍，是因为"实际画出来的主色是什么"
只能有一个说法。

**分类列表里不放空占位页。** 一个点进去什么都没有的分类，比没有这个分类更糟。

**人设那一页不实时生效，这个例外是有理由的。** 上面那条"改一下就立刻出去"对主色与
可见度成立，因为它们只是参数；而人设写下去是**一次落库**，跟着按键走等于每敲一个字
写一次记忆库。所以那一页的按钮自带信号（``persona_save_requested`` /
``persona_clear_requested``），形状与后端页的「应用地址」一致——那一页也是这个道理。

**工具页（可读目录）实时生效，但它多一条：撤得回来。** 加一条目录等于一次**授权**，
所以「撤销修改」必须把它也收回去——那正是快照归 :class:`ToolPage` 自己管的理由。
它也一样只是发信号，落盘归主窗口。

本窗口**不做**任何持久化：它只发信号，写不写、写哪儿归主窗口的
:class:`~biosnn_memterm.ui.settings_store.SettingsStore`。这样"关掉窗口 = 放弃修改"
与"关掉窗口 = 保存"这两种语义都不会溜进来——改动从按下的那一刻就已经生效了。
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..llm import openai_compat
from ..llm.ollama import probe
from ..paths import wallpapers_dir
from .color import parse_hex, readable_foreground
from .settings_store import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_HOST,
    PROVIDER_CLOUD,
    PROVIDER_OLLAMA,
    BackendSettings,
    MemorySettings,
    ToolSettings,
    wallpaper_file,
)
from .theme import (
    CUSTOM_PRESET,
    MODE_DARK,
    MODE_LIGHT,
    MODE_SYSTEM,
    PRESETS,
    WALLPAPER_VISIBLE_MAX,
    Preset,
    Resolved,
    Theme,
    describe_accent,
)
from .thread_tracking import track, wait_for_all

__all__ = ["SettingsDialog"]

#: 壁纸的像素上限（4000 万）。**这不是洁癖**：一张 8000×6000 的图会被 Chromium
#: 解成约 190 MB 的位图，而每次页面重载都要重解一次。
_MAX_PIXELS = 40_000_000
_MAX_PIXELS_TEXT = "4000 万"

#: 文件体积：超过 40 MB 直接拒绝（复制要几秒且界面没有反馈，看起来像卡死）；
#: 超过 20 MB 收下但说一句。
_MAX_BYTES = 40 * 1024 * 1024
_WARN_BYTES = 20 * 1024 * 1024

_WALLPAPER_FILTER = "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"

#: 本模块自己复制进壁纸目录的文件前缀。清理只碰这个前缀——用户自己往那个目录
#: 里放的东西一律不动。
_WALLPAPER_PREFIX = "wallpaper-"

#: 巩固周期在**界面上**的单位是小时，存进设置的是秒（见 ``MemoryPage`` 里那段换算）。
#: 下限 1 小时不是一个"防呆"，而是这个功能的性质：一轮要分簇、建局部权重、逐条回放，
#: 大库上以分钟计；比一小时更密的周期只会让上一轮还没跑完下一轮就到点。上限一周——
#: 再稀疏就等于关掉了，而"关掉"有专门的勾选框，不该藏在数字里。
_MIN_INTERVAL_HOURS = 1
_MAX_INTERVAL_HOURS = 168
_SECONDS_PER_HOUR = 3600

#: 分类列每一行的上下内边距，与 ``theme.qss`` 里 ``#settingsCategories::item`` 的横向
#: 内边距配成一对。**纵向那半边的唯一事实来源在这里**（那把 8 像素从样式表里拿出来了），
#: 所以改这个数要同时看一眼左边那一列还够不够放"后端 / API"。
_CATEGORY_PAD_Y = 8


class _ProbeWorker(QObject):
    """在后台问一次后端在不在。

    **不能直接在界面线程调 ``probe``**：那是一次网络请求，期间界面整个冻住，
    看起来像程序卡了。

    **按后端类型分派**，因为两边的 ``probe`` 回答的不是同一件事：本机问的是
    "Ollama 起没起"（它去探 ``/api/version``），云端问的是"这个地址 + 这把 key
    能不能用"（它去列一次 ``/models``）。不分支的话，云端配置下会去探
    ``127.0.0.1:11434``——一个用户根本没配过的地址。
    """

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, backend: BackendSettings) -> None:
        super().__init__()
        self._backend = backend

    @Slot()
    def run(self) -> None:
        try:
            self._report(self._run_probe())
        except Exception as exc:  # noqa: BLE001 — 任何失败都要变成一行能读的话，绝不冒泡
            self.failed.emit(str(exc))

    def _run_probe(self) -> tuple[str, str] | str | None:
        """``(成功, 话)``、或本机那一路的版本号、或 ``None``（没连上）。"""
        if self._backend.is_cloud:
            return openai_compat.probe(self._backend.cloud_base_url, api_key=self._backend.api_key)
        return probe(self._backend.host, allow_remote=self._backend.allow_remote)

    def _report(self, result: Any) -> None:
        if isinstance(result, tuple):
            ok, detail = result
            # 云端把话说全（含 401 那句原因），所以这里直接转出去，不再套"连上了"。
            (self.done if ok else self.failed).emit(detail)
            return
        self.done.emit(f"连上了 · Ollama {result}" if result else "连上了")


class _ColourSwatch(QPushButton):
    """一个可勾选的预设色块。

    **这里用控件级样式表是对的，别处不是。** 这个颜色是**数据**（预设自己的主色），
    不随主题变——它就是这个预设的定义。控件级样式表"不跟主题走"这个缺点，正好是
    这里要的。别处（按钮、面板）的颜色是装饰，必须走应用级 QSS。
    """

    def __init__(self, preset: Preset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumHeight(36)
        self.set_preset(preset)

    def set_preset(self, preset: Preset) -> None:
        """换掉这个色块代表的颜色。自定义色就是这么显示出来的。"""
        self.preset = preset
        self.setText(preset.label)
        self.setToolTip(f"主色 {preset.accent}；浅色模式下用 {preset.light_accent}")
        # 勾选态用对比色写字（`readable_foreground` 取对比度高的那一侧）——
        # 三个预设都是亮色，写死白字的话琥珀那个会糊成一片。
        # 注意它收的是 **RGB 三元组**不是色号字符串（``derive_roles`` 收字符串是
        # 因为它在内部自己 parse）——传错的表现是画到这一步才炸一个 TypeError。
        on_colour = readable_foreground(parse_hex(preset.accent))
        self.setStyleSheet(
            f"QPushButton {{ border: 2px solid {preset.accent};"
            f" border-radius: 8px; padding: 6px 16px; }}"
            f"QPushButton:checked {{ background: {preset.accent};"
            f" color: {on_colour}; font-weight: bold; }}"
        )


class AppearancePage(QWidget):
    """外观页。发出的每一个 ``changed(Theme)`` 都是**完整且合法**的主题。"""

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = Theme()
        #: 正在**陈述**现状而不是**修改**：见 :meth:`_sync_controls`。
        self._syncing = False
        self._swatches: dict[str, _ColourSwatch] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        outer.addWidget(_section("主色"))
        row = QHBoxLayout()
        row.setSpacing(8)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for preset in PRESETS:
            swatch = _ColourSwatch(preset, self)
            swatch.clicked.connect(self._on_preset_clicked)
            self._group.addButton(swatch)
            self._swatches[preset.key] = swatch
            row.addWidget(swatch)
        self.custom_button = _ColourSwatch(
            Preset(CUSTOM_PRESET, "自定义…", "#8b93a7", "#8b93a7"), self
        )
        self.custom_button.clicked.connect(self._on_custom_clicked)
        self._group.addButton(self.custom_button)
        row.addWidget(self.custom_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.accent_note = _hint("")
        self.accent_note.setObjectName("accentNote")
        outer.addWidget(self.accent_note)

        outer.addWidget(_section("明暗"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        self._mode_group = QButtonGroup(self)
        for value, label in (
            (MODE_SYSTEM, "跟随系统"),
            (MODE_DARK, "强制暗色"),
            (MODE_LIGHT, "强制浅色"),
        ):
            button = QRadioButton(label, self)
            button.setProperty("modeValue", value)
            button.toggled.connect(self._on_mode_toggled)
            self._mode_group.addButton(button)
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)
        outer.addWidget(
            _hint("跟随系统时，Qt 外壳与网页对话区由同一次判定驱动——「两侧一起跟」是刻意的。")
        )

        outer.addWidget(_section("壁纸（只铺对话区）"))
        self.wallpaper_name = QLabel("", self)
        self.wallpaper_name.setObjectName("wallpaperName")
        outer.addWidget(self.wallpaper_name)
        self.wallpaper_preview = QLabel(self)
        self.wallpaper_preview.setFixedHeight(96)
        self.wallpaper_preview.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        outer.addWidget(self.wallpaper_preview)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.pick_button = QPushButton("选择图片…", self)
        self.pick_button.clicked.connect(self._on_pick_wallpaper)
        button_row.addWidget(self.pick_button)
        self.clear_button = QPushButton("清除", self)
        self.clear_button.setProperty("kind", "danger")
        self.clear_button.clicked.connect(self._on_clear_wallpaper)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        outer.addLayout(button_row)

        visible_row = QHBoxLayout()
        visible_row.setSpacing(8)
        visible_row.addWidget(QLabel("可见度", self))
        self.visible_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.visible_slider.setRange(0, WALLPAPER_VISIBLE_MAX)
        self.visible_slider.setSingleStep(1)
        self.visible_slider.setPageStep(5)
        self.visible_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.visible_slider.setTickInterval(10)
        self.visible_slider.valueChanged.connect(self._on_visible_changed)
        visible_row.addWidget(self.visible_slider, 1)
        self.visible_value = QLabel("", self)
        self.visible_value.setMinimumWidth(48)
        visible_row.addWidget(self.visible_value)
        outer.addLayout(visible_row)
        outer.addWidget(
            _hint(
                f"上限 {WALLPAPER_VISIBLE_MAX}%：再高遮罩就只剩三成不透明度，"
                f"而消息是铺满整个对话区滚动的——没有哪一块是可以不看的。"
            )
        )
        outer.addStretch(1)

    # ── 状态进出 ────────────────────────────────────────────────────────────

    @property
    def current_theme(self) -> Theme:
        """现在控件上显示着的主题。清理壁纸副本时要问它。"""
        return self._theme

    def set_state(self, theme: Theme, resolved: Resolved) -> None:
        """按给定主题把控件对齐。**打开窗口时调一次**，也用于「撤销修改」。"""
        self._theme = theme
        self._sync_controls()
        self._refresh_wallpaper(theme)
        self.set_resolved(resolved)

    def _sync_controls(self) -> None:
        """把控件画成 :attr:`_theme` 的样子。**只动控件，不碰文件与说明文字。**

        与 :meth:`set_state` 分开，是因为"取消取色"只需要撤销勾选状态——走
        ``set_state`` 会顺带把壁纸预览重新解码一遍（多一次磁盘读 + 一次缩放）。

        ``_syncing`` 是**必需的**，而且 ``blockSignals`` 顶不了它：下面每一次
        ``setChecked``/``setValue`` 都会经由**子控件**发出 ``toggled``/
        ``valueChanged``，而 ``QWidget.blockSignals`` 只挡这个对象自己的信号，
        挡不住子控件的。没有这个闸门，打开窗口这一下就会往外发一轮信号，
        主窗口照着同一份主题再 apply 一遍——表现上无害，但"陈述现状"与
        "用户改了"从此在接收侧再也分不开。
        """
        self._syncing = True
        try:
            for key, swatch in self._swatches.items():
                swatch.setChecked(key == self._theme.preset)
            self.custom_button.setChecked(self._theme.preset == CUSTOM_PRESET)
            # 自定义那个色块要显示**用户选的色**，否则它永远是同一个灰，
            # 看不出当前是什么。
            self.custom_button.set_preset(
                Preset(CUSTOM_PRESET, "自定义…", self._theme.accent, self._theme.accent)
            )
            for button in self._mode_group.buttons():
                button.setChecked(button.property("modeValue") == self._theme.mode)
            self.visible_slider.setValue(self._theme.wallpaper_visible)
        finally:
            self._syncing = False

    def set_resolved(self, resolved: Resolved) -> None:
        """回填"实际画出来的样子"。由主窗口在 apply 之后调用。

        **这两件事必须分开显示。** 用户选的原色与屏幕上实际用的色，在浅色模式下
        可能不同（预设各带一支手挑的浅色值）。只显示其中一个，用户下次看到对不上
        会以为程序坏了——而那是这一版特意做对的一件事。
        """
        self.accent_note.setText(f"当前：{describe_accent(self._theme, resolved)}")
        self.visible_value.setText(f"{self._theme.wallpaper_visible}%")

    # ── 改一个字段 ──────────────────────────────────────────────────────────

    def _replace(self, **changes: Any) -> None:
        """换掉若干字段，造一个**完整**的 ``Theme`` 发出去。

        只发"变了哪一项"的话，接收方得自己拼当前状态——于是状态有了两个家，而
        两个家迟早会对不上。``Theme`` 是 frozen dataclass，``==`` 按值比，所以
        "没变化就不发"这一句是免费的。
        """
        fields: dict[str, Any] = {
            "preset": self._theme.preset,
            "accent": self._theme.accent,
            "mode": self._theme.mode,
            "wallpaper": self._theme.wallpaper,
            "wallpaper_visible": self._theme.wallpaper_visible,
        }
        fields.update(changes)
        theme = Theme(**fields)
        if theme == self._theme or self._syncing:
            # 相等就不发——`Theme` 是 frozen dataclass，``==`` 按值比，这一句免费。
            # ``_syncing`` 那半见 :meth:`_sync_controls`。
            return
        self._theme = theme
        self.changed.emit(theme)

    # ── 主色 ────────────────────────────────────────────────────────────────

    def _on_preset_clicked(self) -> None:
        for key, swatch in self._swatches.items():
            if swatch.isChecked():
                # 只给 preset：`Theme.__post_init__` 会按预设把 accent 收敛过去。
                # 这里再传一次 accent 就是两套真源，而两套必然在某个路径上分叉。
                self._replace(preset=key)
                return

    def _on_custom_clicked(self) -> None:
        picked = self._ask_for_colour(QColor(self._theme.accent))
        if picked is None or not picked.isValid():
            # 取消：把勾退回去。不退的话色块显示"自定义"、主色却还是上一个，
            # 而用户下一次点它会以为程序记住了他刚才取消掉的那个色。
            #
            # 只同步控件，**不重设整份状态**：真正变的就只有那几个勾。
            self._sync_controls()
            return
        self._replace(preset=CUSTOM_PRESET, accent=picked.name())

    def _ask_for_colour(self, initial: QColor) -> QColor | None:
        """**测试缝。** 覆盖它就不必弹一个真的取色器——照抄
        ``memory_window._ask_for_content`` 的既有模式。"""
        colour = QColorDialog.getColor(initial, self, "选择主色")
        return colour if colour.isValid() else None

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:  # 一组单选里，新的那次 toggled(True) 会跟着一次 toggled(False)
            return
        for button in self._mode_group.buttons():
            if button.isChecked():
                self._replace(mode=button.property("modeValue"))
                return

    # ── 壁纸 ────────────────────────────────────────────────────────────────

    def _on_pick_wallpaper(self) -> None:
        source = self._ask_for_wallpaper()
        if not source:
            return
        installed = self._install_wallpaper(Path(source))
        if installed is None:
            return
        self._replace(wallpaper=installed)
        # 缩略图**立刻**跟上，不等下一次 ``set_state``。等的话，刚换完图的人在这一栏
        # 看不到任何变化，唯一的反馈在网页那边——而它铺满整个对话区，"改没改"要
        # 看一整片底色才认得出。
        #
        # 这里单独调 `_thumbnail` 而不是图省事走 `_refresh_wallpaper`：后者会把
        # 「已复制 …（N MB，加载会慢一点）」那句换成文件名，而刚选完一张大图的
        # 人正是需要看到那句话的人。
        path = wallpaper_file(self._theme.wallpaper)
        pixmap = None if path is None else _thumbnail(path)
        if pixmap is not None:
            self.wallpaper_preview.setPixmap(pixmap)

    def _ask_for_wallpaper(self) -> str:
        """**测试缝。** 返回空串表示用户取消。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择壁纸", "", _WALLPAPER_FILTER)
        return path

    def _install_wallpaper(self, source: Path) -> str | None:
        """把选中的图片复制进壁纸目录，返回文件名；不合格返回 ``None`` 并说明原因。

        **为什么复制而不是记住原路径。** 记住路径的话，用户把图移走、改名、拔掉
        U 盘之后壁纸就没了，症状是"壁纸莫名其妙不见了"；复制进来之后那份副本归
        我们管。文件名受控还有第二个好处：原文件名里的空格、``#``、``%`` 不会跑进 URL。

        **文件名带内容哈希，于是"换一张"永远不覆盖上一张。** 这是「撤销修改」的
        前提：覆盖着写的话，撤销之后那个名字指向的是一个已经不存在的文件，界面会
        说"壁纸找不到了"。没被引用的旧副本在窗口关闭时才清（见 :meth:`prune_wallpapers`）。
        """
        if not source.is_file():
            self._warn(f"找不到这个文件：{source}")
            return None
        reader = QImageReader(str(source))
        if not reader.canRead():
            self._warn("这不是一张能读的图片（或者格式不被支持）。")
            return None
        size = reader.size()
        if size.isValid() and size.width() * size.height() > _MAX_PIXELS:
            self._warn(
                f"这张图是 {size.width()}×{size.height()}，超过 {_MAX_PIXELS_TEXT}像素。"
                f"换一张小一点的——大图每次重载都要重新解码，界面会明显变慢。"
            )
            return None
        raw = source.read_bytes()
        if len(raw) > _MAX_BYTES:
            self._warn(f"这张图 {len(raw) // (1024 * 1024)} MB，超过上限 40 MB。")
            return None

        digest = hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:10]
        target = wallpapers_dir() / f"{_WALLPAPER_PREFIX}{digest}{source.suffix.lower() or '.img'}"
        target.write_bytes(raw)

        note = ""
        if len(raw) > _WARN_BYTES:
            note = f"（{len(raw) // (1024 * 1024)} MB，加载会慢一点）"
        self.wallpaper_name.setText(f"已复制 {source.name} {note}")
        return target.name

    def _on_clear_wallpaper(self) -> None:
        self._replace(wallpaper="")

    def _refresh_wallpaper(self, theme: Theme) -> None:
        """按主题里的文件名刷新预览与那句状态文字。**文件不在不是错误**——
        用户把数据目录搬走、图片被删、U 盘拔了都是常态。"""
        self.wallpaper_preview.clear()
        path = wallpaper_file(theme.wallpaper)
        if path is None:
            if theme.wallpaper:
                self._warn(f"设置里记着 {theme.wallpaper}，但文件不在了。")
            else:
                self._warn("未设置")
            return
        pixmap = _thumbnail(path)
        if pixmap is None:
            self._warn(f"{theme.wallpaper}（预览读不出来，但设置已记下）")
            return
        self.wallpaper_name.setText(theme.wallpaper)
        self.wallpaper_preview.setPixmap(pixmap)

    def _warn(self, text: str) -> None:
        """往预览那一行写字。**不弹框**：选图失败是一种就地可修的输入问题，
        弹一个要点的框只会让"再选一张"多两步。"""
        self.wallpaper_name.setText(text)

    def _on_visible_changed(self, value: int) -> None:
        self.visible_value.setText(f"{value}%")
        self._replace(wallpaper_visible=value)


class BackendPage(QWidget):
    """后端页：**本机 Ollama 与云端 API 二选一。**

    先说本机那一组。**有 host 输入框就必须有 ``allow_remote`` 勾选框**——这不是
    "多给一个选项"：:func:`~biosnn_memterm.llm.ollama._client` 对非本机地址会直接
    抛错，所以不勾那个框的远程 host 是一个**填了就一定报错**的字段。勾选框旁边如实
    写出代价——那句"内容不出设备"的假设从此刻起不再成立。

    云端那一组同理，而且代价更大：API key 会跟着每一个请求发出去，所以那个框旁边
    也要写出**什么会被发走**。措辞里必须点到"记忆"——``AssemblyWorker`` 会把召回的
    记忆注入 prompt，它跟对话一起出去；只说"对话内容"是不完整的（见
    :attr:`cloud_warning`）。

    **两组控件按类型显隐，而不是并排摆着。** 并排的话，"Ollama 地址"与"云端地址"
    两个输入框会同时可填，而它们**同时生效是不可能的**——用户会以为两个都要填。
    """

    apply_requested = Signal(object)
    #: 带的是**整个** :class:`BackendSettings`。以前是 ``(host, allow_remote)``——
    #: 云端测试连接还要知道 base_url 与 key，两个参数装不下第三种后端。
    test_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        outer.addWidget(_section("类型"))
        head = QFormLayout()
        head.setSpacing(8)
        self.provider_combo = QComboBox(self)
        # data 存的是**常量**不是下标：下标会随条目顺序漂，而这两个值要落进设置、
        # 还要被 ``load_backend`` 认出来。
        self.provider_combo.addItem("本机（Ollama）", PROVIDER_OLLAMA)
        self.provider_combo.addItem("云端 API（OpenAI 兼容）", PROVIDER_CLOUD)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        head.addRow("后端", self.provider_combo)
        outer.addLayout(head)

        outer.addWidget(self._build_local_group())
        outer.addWidget(self._build_cloud_group())

        row = QHBoxLayout()
        row.setSpacing(8)
        self.test_button = QPushButton("测试连接", self)
        self.test_button.clicked.connect(self._on_test_clicked)
        row.addWidget(self.test_button)
        # 按钮上写的从"应用地址"改成"应用设置"：切到云端之后要应用的还有 key，
        # 而"地址"这个词会让人以为 key 是自动保存的。
        self.apply_button = QPushButton("应用设置", self)
        self.apply_button.setProperty("kind", "primary")
        self.apply_button.clicked.connect(self._emit)
        row.addWidget(self.apply_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.note = _hint("")
        self.note.setObjectName("backendNote")
        outer.addWidget(self.note)
        outer.addStretch(1)
        self._sync_groups()
        self._sync_warning()

    # ── 两组控件 ────────────────────────────────────────────────────────────

    def _build_local_group(self) -> QWidget:
        """本机组：地址 + 允许远程 + 代价说明。"""
        group = QWidget(self)
        self.local_group = group
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)

        box.addWidget(_section("服务地址"))
        form = QFormLayout()
        form.setSpacing(8)
        self.host_edit = QLineEdit(group)
        self.host_edit.setPlaceholderText(DEFAULT_HOST)
        form.addRow("Ollama 地址", self.host_edit)
        box.addLayout(form)

        self.allow_remote_box = QCheckBox("允许远程主机", group)
        self.allow_remote_box.toggled.connect(self._on_allow_remote_toggled)
        box.addWidget(self.allow_remote_box)
        self.remote_warning = _hint(
            "勾上之后对话内容会离开这台设备——「记忆不出设备」这条假设从此刻起不再成立。"
            "只有在你自建了内网推理机时才该勾它。"
        )
        self.remote_warning.setObjectName("remoteWarning")
        box.addWidget(self.remote_warning)
        return group

    def _build_cloud_group(self) -> QWidget:
        """云端组：地址 + key + 代价说明。"""
        group = QWidget(self)
        self.cloud_group = group
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)

        box.addWidget(_section("云端端点"))
        form = QFormLayout()
        form.setSpacing(8)
        self.cloud_url_edit = QLineEdit(group)
        self.cloud_url_edit.setPlaceholderText(DEFAULT_CLOUD_BASE_URL)
        form.addRow("服务地址", self.cloud_url_edit)

        key_row = QWidget(group)
        key_box = QHBoxLayout(key_row)
        key_box.setContentsMargins(0, 0, 0, 0)
        key_box.setSpacing(8)
        self.api_key_edit = QLineEdit(key_row)
        # 默认遮起来。**这是唯一一条"默认值即安全"的界面约定**：key 会出现在
        # 截图、录屏与共享屏幕里，而"要不要显示"是一个每次都要重新做的决定。
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("sk-…")
        key_box.addWidget(self.api_key_edit, 1)
        self.show_key_box = QCheckBox("显示", key_row)
        self.show_key_box.toggled.connect(self._on_show_key_toggled)
        key_box.addWidget(self.show_key_box)
        form.addRow("API key", key_row)
        box.addLayout(form)

        self.cloud_warning = _hint(
            "云端模式下，这一轮的对话**以及被召回注入的那几条记忆**都会发送到这个地址的"
            "运营方。记忆库本身、嵌入向量与逐字稿仍只在本机；嵌入也仍然用本机 Ollama"
            "（云端厂商多数没有嵌入接口）。"
        )
        self.cloud_warning.setObjectName("remoteWarning")
        box.addWidget(self.cloud_warning)
        return group

    # ── 状态进出 ────────────────────────────────────────────────────────────

    def set_state(self, backend: BackendSettings, note: str = "") -> None:
        # 先把显隐切好再填内容：反过来的话，切类型那一刻会拿旧组合里的值去
        # 触发一次应用（见 ``_on_provider_changed``）。
        index = self.provider_combo.findData(backend.provider)
        self.provider_combo.blockSignals(True)
        self.provider_combo.setCurrentIndex(index if index >= 0 else 0)
        self.provider_combo.blockSignals(False)
        self._sync_groups()

        self.host_edit.setText(backend.host)
        self.allow_remote_box.setChecked(backend.allow_remote)
        self.cloud_url_edit.setText(backend.cloud_base_url)
        self.api_key_edit.setText(backend.api_key)
        self.set_note(note)

    def set_note(self, text: str) -> None:
        """由主窗口回填"应用的结果"（成功 / 正在生成所以拒绝 / 护栏拦下），
        也由连接测试回填（连上了 / 报错原文）。"""
        self.note.setText(text)

    def current(self) -> BackendSettings:
        """控件里现在这一份。

        **两个地址都带出来**，即使当前用不上另一个：``BackendSettings`` 要原样存回
        设置里，漏掉一个就等于"切一次类型把它清空了"——而用户切回来时看到的是一个
        空框（或者更糟，是默认值，于是他会以为自己的配置丢了）。
        """
        return BackendSettings(
            provider=self.provider_combo.currentData() or PROVIDER_OLLAMA,
            host=self.host_edit.text().strip() or DEFAULT_HOST,
            allow_remote=self.allow_remote_box.isChecked(),
            cloud_base_url=self.cloud_url_edit.text().strip() or DEFAULT_CLOUD_BASE_URL,
            api_key=self.api_key_edit.text().strip(),
        )

    # ── 联动 ────────────────────────────────────────────────────────────────

    def _on_provider_changed(self, _index: int) -> None:
        """换类型**当场生效**，与 ``allow_remote`` 勾选框同一条形状。

        不这样的话，"类型"这个下拉框就只是改了一个没人读的字段——用户切到云端、
        界面上两组控件也换了，而实际请求还是发往本机。
        """
        self._sync_groups()
        self._emit()

    def _on_allow_remote_toggled(self, _checked: bool) -> None:
        self._sync_warning()
        self._emit()

    def _on_show_key_toggled(self, shown: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        )

    def _sync_groups(self) -> None:
        cloud = self.current_provider() == PROVIDER_CLOUD
        self.local_group.setVisible(not cloud)
        self.cloud_group.setVisible(cloud)
        self._sync_warning()

    def current_provider(self) -> str:
        return str(self.provider_combo.currentData() or PROVIDER_OLLAMA)

    def _sync_warning(self) -> None:
        """**两组警告各自常亮，只有"当前这一组"是醒目的。**

        不勾/不选时也要能读——用户得先知道代价才能决定要不要切过去，把代价藏起来
        是最糟的做法。而切到云端之后那句话必须更醒目：它说的是一件**已经发生**的事。
        """
        cloud = self.current_provider() == PROVIDER_CLOUD
        self._arm(self.remote_warning, self.allow_remote_box.isChecked() and not cloud)
        self._arm(self.cloud_warning, cloud)

    @staticmethod
    def _arm(label: QLabel, armed: bool) -> None:
        label.setProperty("armed", "yes" if armed else "no")
        label.style().unpolish(label)
        label.style().polish(label)

    def _emit(self) -> None:
        self.apply_requested.emit(self.current())

    def _on_test_clicked(self) -> None:
        self.set_note("正在测试…")
        self.test_requested.emit(self.current())


class PersonaPage(QWidget):
    """人设页：终端内置的那段（只读）+ 你写的那段（可改可删）。

    **这一页与外观页有一条会咬人的区别，所以界面上必须写出来**：点保存就写进记忆库、
    立刻生效，没有"完成"这一步；而底部那句「撤销修改」只回退外观，**不会**撤销人设。
    不写的话，用户会理所当然地以为关窗或点撤销能退回去——而那时库里已经改了。

    **这一页不持有真源。** 与外观/后端不同，人设存在**记忆库**里（首启卡片、记忆
    窗口都能改它），所以本页只能被回填（:meth:`set_persona`），不自己存任何东西。
    ``_saved`` 是"库里现在是什么"的一份影子，只用来判"有没有改动"。
    """

    save_requested = Signal(str)
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        #: 库里那条用户写的人设的正文。空串 = 还没写过。
        self._saved: str = ""

        outer.addWidget(_section("终端内置的那段"))
        # 用 QLabel 而不是只读的文本框：`setReadOnly(True)` 的框看起来与能改的那个
        # 一模一样，用户会先点进去、发现打不了字，才知道它是死的。一个标签在视觉上
        # 就先说清了这件事。代价是不能滚动——这段只有三句，用不上滚动。
        self.builtin_view = QLabel(self)
        self.builtin_view.setWordWrap(True)
        self.builtin_view.setObjectName("builtinPersona")
        # 能选中复制，但选不中"可编辑"。默认的 NoTextInteraction 连选中都不行，
        # 而"这段话我要抄一下"是很自然的念头。
        self.builtin_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.builtin_view)
        outer.addWidget(
            _hint("这段由程序写死，改不了也删不掉。换任何一个模型来读，读到的都是这些字。")
        )

        outer.addWidget(_section("你写的那段"))
        self.user_edit = QPlainTextEdit(self)
        self.user_edit.setPlaceholderText("例如：说话简短直接，不要客套话。叫我小林。")
        self.user_edit.setFixedHeight(96)
        self.user_edit.textChanged.connect(self._sync_buttons)
        outer.addWidget(self.user_edit)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.save_button = QPushButton("保存我写的这段", self)
        self.save_button.setProperty("kind", "primary")
        self.save_button.clicked.connect(self._emit_save)
        row.addWidget(self.save_button)
        self.clear_button = QPushButton("删掉我写的这段", self)
        self.clear_button.clicked.connect(self.clear_requested)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.note = _hint("")
        self.note.setObjectName("personaNote")
        outer.addWidget(self.note)

        # 两句都必须说：第一句是"保存立刻生效"，第二句是"撤销不管这一页"。
        # 它们是这一页与外观页唯一的行为差异，也正是最容易想当然的地方。
        #
        # **这句里不能用 markdown 的 `**`。** QLabel 只认 HTML，`**不会**` 会原样
        # 显示成两个星号——而这个仓里正文有 markdown 渲染的地方只有对话流。
        outer.addWidget(
            _hint(
                "点保存就写进记忆库、立刻生效，不用等「完成」。"
                "底部的「撤销修改」只回退外观，不会撤销这里写下的东西；"
                "要撤掉就用左边的「删掉我写的这段」。"
            )
        )
        outer.addStretch(1)
        self._sync_buttons()

    def set_persona(self, builtin: str, user: str) -> None:
        """由主窗口回填两段正文。**同时重置基线**——"有没有改动"全靠它。"""
        self.builtin_view.setText(builtin)
        self.user_edit.setPlainText(user)
        self._saved = user.strip()
        self.set_note("")
        self._sync_buttons()

    def set_note(self, text: str) -> None:
        """由主窗口回填"上一次保存/删除的结果"。与后端页的 note 同一个用法。"""
        self.note.setText(text)

    def current(self) -> str:
        """框里的正文，已去首尾空白。**空白等于没写**——``memory.edit.can_encode``
        也这么看，所以这里先去一层，用户存进去的就是他看到的。"""
        return self.user_edit.toPlainText().strip()

    def _emit_save(self) -> None:
        self.save_requested.emit(self.current())

    def _sync_buttons(self) -> None:
        """两个按钮的置灰 + **理由**。灰色按钮不写理由在本仓是要被修的毛病（见
        记忆窗口 `_update_actions` 那段注释），这里一开始就带上。"""
        text = self.current()
        empty = text == ""
        unchanged = text == self._saved

        self.save_button.setEnabled(not empty and not unchanged)
        if empty:
            self.save_button.setToolTip("正文不能为空——「先不写」就是右边的「删掉我写的这段」")
        elif unchanged:
            self.save_button.setToolTip("没有改动")
        else:
            self.save_button.setToolTip("")

        self.clear_button.setEnabled(self._saved != "")
        self.clear_button.setToolTip("" if self._saved else "还没有写过人设")


class ToolPage(QWidget):
    """工具页。P3a 只有一项：**可读目录**。

    这一页是"最小权限"那条边界在界面上的**唯一入口**。列在这里的目录，对话里的读
    文件工具会**直接读、不会再问一次**——所以那句提示必须写在列表旁边。用户加一个
    目录时唯一该知道的事就是"加进来等于一直允许"，而把它写进文档等于没写。

    **顺序就是生效顺序。** :func:`~biosnn_memterm.agent.fs.resolve_within` 按列表
    顺序逐个根去试，第一个"字面上落在根里"的会**遮蔽**后面的（那个函数的注释里
    详细写了为什么），所以这里显示的顺序必须就是存下去的顺序。本页不做上移/下移：
    需要争顺序的只有"两个根里都有同名文件"这一种情形，到那时再说。

    **空列表不是错误状态。** 它表示"没有授权任何目录"，于是读文件工具在对话里
    根本不出现（不是出现之后每次都被拒）。所以下面那句话是**陈述**，不是告警。

    **快照归这一页管**（:meth:`revert`）：底部「撤销修改」的提示写着"回到打开这个
    窗口时的样子"，而"哪几个目录允许读"恰恰是这里最该撤得回去的一项。所以它不
    走后端页那种"点了才生效"的形状——加/删当刻就出去，撤销能收回。
    """

    roots_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        #: 「撤销修改」回到的那一份。:meth:`set_state` 会覆盖它。
        self._snapshot: tuple[Path, ...] = ()

        outer.addWidget(_section("可读目录"))

        self.roots_view = QListWidget(self)
        self.roots_view.setObjectName("readRoots")
        self.roots_view.setFixedHeight(150)
        self.roots_view.itemSelectionChanged.connect(self._sync_buttons)
        outer.addWidget(self.roots_view)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.add_button = QPushButton("添加目录…", self)
        self.add_button.clicked.connect(self._on_add_clicked)
        row.addWidget(self.add_button)
        self.remove_button = QPushButton("移除选中", self)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        outer.addLayout(row)

        # 「这一条几乎等于整个盘」。**空着时什么都不显示**（与后端页那句常态都在的
        # 代价说明不同）：它说的是"你刚加的这一条太宽"，不是一件总要读的事。
        self.wide_warning = _hint("")
        self.wide_warning.setObjectName("toolWide")
        outer.addWidget(self.wide_warning)

        outer.addWidget(
            _hint(
                "加进这里的目录，对话里要用到文件时会被直接读取，不会再问一次。"
                "所以挑具体的一两层目录，别把整个盘加进来。"
                "一条都不加时，「读文件」这个工具在对话里根本不会出现。"
            )
        )

        self.note = _hint("")
        self.note.setObjectName("toolNote")
        outer.addWidget(self.note)
        outer.addStretch(1)
        self._sync_buttons()
        self._sync_wide()

    # ── 状态进出 ────────────────────────────────────────────────────────────

    def set_state(self, settings: ToolSettings) -> None:
        """**打开窗口时调，陈述现状。** 顺带重置快照（同外观页的道理：撤销回到的
        是**这一次**打开时的样子）。

        **不发信号**——摆上当前值不是一次改动。上一次那句"这个目录已经在列表里了"
        也在这里擦掉：换一次打开还挂着上一轮的结果，用户会以为刚才那次点击生效了。
        """
        self._snapshot = settings.read_roots
        self._fill(settings.read_roots)
        self.set_note("")

    def set_note(self, text: str) -> None:
        self.note.setText(text)

    def roots(self) -> tuple[Path, ...]:
        """列表里的目录，**按显示顺序**（顺序是有意义的，见类文档）。"""
        return tuple(
            # 值走 ``UserRole`` 而不是读显示的文字：显示那一列将来可能要加"（不存在）"
            # 这类后缀，而生效的值不能跟着变。
            Path(str(self.roots_view.item(row).data(Qt.ItemDataRole.UserRole)))
            for row in range(self.roots_view.count())
        )

    def revert(self) -> None:
        """「撤销修改」的目录那一半。**变了才发信号。**

        没变就不发：不然每按一次「撤销修改」都会往设置里写一遍同样的东西，而
        "这一页本来就没动过"是绝大多数情况。
        """
        if self.roots() == self._snapshot:
            return
        self._fill(self._snapshot)
        self.roots_changed.emit(self.roots())

    # ── 增删 ────────────────────────────────────────────────────────────────

    def _ask_for_directory(self, start: str) -> str:
        """弹目录选择框。**留成可覆盖的一层**，理由与 ``AppearancePage._ask_for_wallpaper``
        一字不差：真弹一个模态框会让这个文件的测试在 CI 上挂住。"""
        return QFileDialog.getExistingDirectory(self, "选择允许读取的目录", start)

    def _on_add_clicked(self) -> None:
        roots = self.roots()
        # 从最后一个已选的目录出发（多半是刚加的那个的邻居）；一条都没有才回家目录。
        # 每次都从家目录开始的话，资料放在 D 盘的用户要点五六次才到得了。
        chosen = self._ask_for_directory(str(roots[-1]) if roots else str(Path.home()))
        if not chosen:
            return  # 用户取消了——这不是一次改动
        path = Path(chosen)
        if path in roots:
            # 静默去重会让"我明明加了"变成一个说不清的状态，所以这里说一句话。
            self.set_note("这个目录已经在列表里了。")
            return
        self._append(path)
        self.set_note("")
        self._after_change()

    def _on_remove_clicked(self) -> None:
        # 从后往前删：先删前面的话，后面那些行的下标就全变了。
        #
        # 行号要问**列表**（``QListWidget.row(item)``），不要问条目——PySide6 的
        # ``QListWidgetItem`` 上没有 ``row()``（实测：``dir()`` 里一个 row 都没有），
        # 而它抛在 Qt 事件循环里：异常被 Qt 吃掉、只往 stderr 打一段回溯，界面上
        # 是"按了没反应"。这一条是被 ``test_removing_takes_only_the_selected_one_out``
        # 抓出来的。
        rows = sorted(
            {self.roots_view.row(item) for item in self.roots_view.selectedItems()}, reverse=True
        )
        if not rows:
            return
        for row in rows:
            self.roots_view.takeItem(row)
        self._after_change()

    def _append(self, path: Path) -> None:
        item = QListWidgetItem(str(path), self.roots_view)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.roots_view.setCurrentItem(item)

    def _fill(self, roots: Sequence[Path]) -> None:
        self.roots_view.clear()
        for root in roots:
            self._append(root)
        self.roots_view.clearSelection()
        self._sync_buttons()
        self._sync_wide()

    def _after_change(self) -> None:
        self._sync_buttons()
        self._sync_wide()
        self.roots_changed.emit(self.roots())

    def _sync_buttons(self) -> None:
        """按钮置灰 + **理由**。灰色按钮不写理由在本仓是要被修的毛病（见记忆窗口
        ``_update_actions`` 那段注释），这里一开始就带上。"""
        has_selection = bool(self.roots_view.selectedItems())
        self.remove_button.setEnabled(has_selection)
        self.remove_button.setToolTip("" if has_selection else "先在列表里选一条")

    def _sync_wide(self) -> None:
        wide = [str(root) for root in self.roots() if _is_wide_root(root)]
        self.wide_warning.setText(
            "这一条几乎等于整个盘：" + "、".join(wide) + "。读文件工具从此能读到那里的一切。"
            if wide
            else ""
        )


class MemoryPage(QWidget):
    """记忆页。三件事：**写入的粒度**（任务记忆）、**召回的通道**（联想）、以及
    **后台的睡眠巩固**。

    这一页管的不是"记不记"（记忆一直在记，那是这个程序的意义），而是**一次任务记成
    几条、记的是哪一句**、**召回时要不要多看一条通道**、以及**已有的记忆要不要定期
    重新认一遍**。所以说明文字是这一页的主体，三个勾选框各只有一行——用户按下它之前
    唯一该知道的事就是"会变成什么样"，而那句话写进文档等于没写。

    **两个开关都默认关，而且这句话要写在界面上。** 关着的时候，写入路径与召回路径
    都与上一个版本逐字段相同（:class:`~biosnn_memterm.memory.policy.PerItemPolicy`、
    ``RecallConfig.graph_weight=0.0``），所以"关"不是少了什么功能，是"和以前一样"
    ——不写出来，用户会以为不勾就是没用上这版的新东西。

    **联想那个开关的代价必须写在它旁边，不能藏在报告里。** 它是本仓唯一一个"打开
    之后可能变差"的开关：实测加这条通道让 hit@1 从 0.8500 掉到 0.8250
    （``benchmarks/substrate/baseline.json``）。所以这一页的读法是有顺序的——它是什么
    → 代价是什么 → 那图从哪来，三句缺一句用户就得靠猜。

    **边改边生效**，与工具页同一形状：勾选框只是一行设置，没有"应用"这一步可等，
    底部有「撤销修改」兜着（:meth:`revert`）。唯一的例外是「重建联想」——那是一次
    几秒到几分钟的计算，按下才跑（:attr:`rebuild_requested`）。
    """

    changed = Signal(object)
    #: 「重建联想」被按下。**不带参数**：建哪张图、存哪儿是主窗口的事（只有它知道
    #: ``graph_path()`` 与当前库多大）。本窗口不做 I/O，同全文件的约定。
    rebuild_requested = Signal()

    #: 「立即巩固」被按下。与上面同形，理由也一样——而且更硬：那一头要起一个后台
    #: 线程、要拍几百 MB 的快照，本窗口连库在哪儿都不该知道。
    consolidate_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        #: 「撤销修改」回到的那一份。:meth:`set_state` 会覆盖它。
        self._snapshot = MemorySettings()
        self._syncing = False

        outer.addWidget(_section("按任务记"))
        self.task_box = QCheckBox("把一次任务合成一条记忆", self)
        self.task_box.setObjectName("taskMemoryBox")
        self.task_box.toggled.connect(self._on_toggled)
        outer.addWidget(self.task_box)

        outer.addWidget(
            _hint(
                "打开后，调过工具的那一轮合成一条记忆：正文是它的回答，"
                "做过的事（读了哪个文件、搜了什么词、哪一步被拒绝）跟着这条一起存。"
                "没有调工具的轮次照旧逐条记——这个开关只多一种粒度，"
                "不会让任何一句话不再被记住。"
            )
        )
        outer.addWidget(_hint("默认关。关着的时候，写入的东西与之前完全一样。"))

        outer.addSpacing(6)
        outer.addWidget(_section("联想通道"))
        self.graph_box = QCheckBox("启用联想通道", self)
        self.graph_box.setObjectName("graphChannelBox")
        self.graph_box.toggled.connect(self._on_toggled)
        outer.addWidget(self.graph_box)

        outer.addWidget(
            _hint("打开后，召回除了看词法与语义，还会把「跟查询内容沾边的记忆的邻居」也拉进来。")
        )
        self.graph_cost = _hint(
            "代价写在前面：门槛 min_score（0.10）是按词法量纲标定的，多一条通道会让"
            "这个阈值不再对应原来的严格程度——命中会变多，误注入也可能变多。本机实测"
            "这条通道是负的（hit@1 0.8500 → 0.8250），所以默认关。开之前请看"
            " benchmarks/gold/CORPUS.md 里「联想腿」那一页的数。"
        )
        self.graph_cost.setObjectName("graphCostWarning")
        outer.addWidget(self.graph_cost)
        outer.addWidget(_hint("这一条只改召回，不改图的建立。图建不建、什么时候重建，看下面这行。"))

        self.rebuild_button = QPushButton("重建联想", self)
        self.rebuild_button.setObjectName("rebuildGraphButton")
        self.rebuild_button.clicked.connect(self._on_rebuild_clicked)
        outer.addWidget(self.rebuild_button)
        self.graph_note = _hint("")
        self.graph_note.setObjectName("graphNote")
        outer.addWidget(self.graph_note)

        outer.addSpacing(6)
        outer.addWidget(_section("睡眠巩固"))
        self.consolidation_box = QCheckBox("每天自动巩固一次", self)
        self.consolidation_box.setObjectName("consolidationBox")
        # 勾选框与间隔是**一件事**：关掉它，那个数字就不再被读。禁用而不是隐藏——
        # 用户填过的值要留着，重新勾上时它还在，而隐藏会让"我上次填的是多少"变成一个
        # 需要记住的问题。
        self.consolidation_box.toggled.connect(self._on_toggled)
        self.consolidation_box.toggled.connect(self._on_consolidation_toggled)
        outer.addWidget(self.consolidation_box)

        outer.addWidget(
            _hint(
                "巩固做的事：把已有的记忆按关联图分簇，每簇用一小块 Hebbian 权重做一次"
                "「挖掉一半、看还认不认得自己」的回放；认出来的那些，巩固次数 +1。"
            )
        )

        interval_row = QHBoxLayout()
        interval_row.setContentsMargins(0, 0, 0, 0)
        interval_row.addWidget(QLabel("每隔", self))
        self.consolidation_hours = QSpinBox(self)
        self.consolidation_hours.setObjectName("consolidationHoursSpin")
        # **界面上是小时，存的是秒**（`memory/consolidation_interval_s`）。这一处换算
        # 是刻意的：86400 那个数字对读它的人毫无意义，而"24 小时"一眼就懂。
        self.consolidation_hours.setRange(_MIN_INTERVAL_HOURS, _MAX_INTERVAL_HOURS)
        self.consolidation_hours.setSuffix(" 小时")
        self.consolidation_hours.valueChanged.connect(self._on_toggled)
        interval_row.addWidget(self.consolidation_hours)
        interval_row.addWidget(QLabel("跑一次", self))
        interval_row.addStretch(1)
        outer.addLayout(interval_row)
        # 开局（``set_state`` 之前）也要自洽：框没勾时那个数字不该像能改的样子。
        self._on_consolidation_toggled(self.consolidation_box.isChecked())

        outer.addWidget(
            _hint(
                "**它会真的改库**（只动「巩固次数」，不改记忆正文、不影响召回排序），"
                "改之前拍一份整库快照，留最近 5 份——所以周期调得越短，磁盘上攒得越快。"
                "默认一天一次，是占位值：真实使用节奏下该多久跑一次还没有实测依据。"
            )
        )
        outer.addWidget(_hint("关掉它只是不自动跑。托盘菜单与下面这个按钮随时可以手动跑一轮。"))

        self.consolidate_button = QPushButton("立即巩固", self)
        self.consolidate_button.setObjectName("consolidateNowButton")
        self.consolidate_button.clicked.connect(self._on_consolidate_clicked)
        outer.addWidget(self.consolidate_button)
        self.consolidate_note = _hint("")
        self.consolidate_note.setObjectName("consolidateNote")
        outer.addWidget(self.consolidate_note)

        outer.addStretch(1)
        self.set_graph_note("")

    # ── 状态进出 ────────────────────────────────────────────────────────────

    def set_state(self, settings: MemorySettings) -> None:
        """**打开窗口时调，陈述现状。** 顺带重置快照（同工具页的道理：撤销回到的
        是**这一次**打开时的样子）。

        ``_syncing`` 是必需的（同外观页那段注释）：``setChecked`` 会经由**子控件**
        发出 ``toggled``，而 ``blockSignals`` 只挡这个对象自己的信号。没有这个闸门，
        打开窗口这一下就会往外发一轮信号，"陈述现状"与"用户改了"在接收侧再也分不开
        ——症状是每次打开设置都往设置里重写一遍同样的值。
        """
        self._snapshot = settings
        self._syncing = True
        try:
            self.task_box.setChecked(settings.task_memory)
            self.graph_box.setChecked(settings.graph_channel)
            self.consolidation_box.setChecked(settings.consolidation_enabled)
            # **小时 ← 秒，四舍五入到整小时**：手改过的注册表里可以是 3600（正好 1 小时）
            # 也可以是 5400（1.5 小时）。后者界面上只能显示成 2 小时——**这是有损的**，
            # 但只要用户不动那个数字，`current()` 就会把它写成 7200。所以这条路径上
            # "打开设置窗口看一眼"会改掉一个手改过的值，而那个值本来就是界面表达不了的。
            # 界面的粒度就是小时，这一点写在下面那条 hint 里。
            self.consolidation_hours.setValue(_hours_of(settings.consolidation_interval_s))
        finally:
            self._syncing = False

    def current(self) -> MemorySettings:
        """**每个控件都要读。** 少读一个的症状不是报错，是"勾了它、关掉设置、再打开
        又变回没勾"——因为写回去的那一份里没有这个字段，而写盘是照这一份写的。
        """
        return MemorySettings(
            task_memory=self.task_box.isChecked(),
            graph_channel=self.graph_box.isChecked(),
            consolidation_enabled=self.consolidation_box.isChecked(),
            consolidation_interval_s=self.consolidation_hours.value() * _SECONDS_PER_HOUR,
        )

    def revert(self) -> None:
        """「撤销修改」的记忆那一半。**变了才发信号**（同工具页）。"""
        if self.current() == self._snapshot:
            return
        self.set_state(self._snapshot)
        self.changed.emit(self.current())

    def _on_toggled(self, _checked: bool) -> None:
        if self._syncing:
            return
        self.changed.emit(self.current())

    # ── 图状态那一行 ────────────────────────────────────────────────────────

    def set_graph_note(self, text: str) -> None:
        """由主窗口回填"图现在是什么状态"。

        ``text`` 为空时写**默认那句**（不是空串）：这一行在窗口刚打开、还没人建过
        图的时候也必须说得通，而空白会被读成"没有这回事"。默认这句与记忆窗口里
        「关联图还没有建」那条顶栏是同一个口径——同一个状态在两处说两样话，用户会
        以为它们是两件事。
        """
        self.graph_note.setText(
            text or "还没建。小库会在打开记忆窗口时自动建；库大了不会，那时点上面这个按钮。"
        )

    def set_rebuild_busy(self, busy: bool) -> None:
        """按下去之后按钮就一直禁用，直到主窗口把结果填回来。

        **成功与失败都只走 :meth:`set_graph_note`，这里不复述结果。** 复述一遍会
        出现两个说法（按钮旁边一个、状态行一个），而它们迟早会在某条分支上分叉。
        所以 ``busy=False`` 只做一件事：把按钮放开。
        """
        self.rebuild_button.setEnabled(not busy)
        if busy:
            self.graph_note.setText("正在重建（大库可能要几分钟）…")

    def _on_rebuild_clicked(self) -> None:
        # 先禁用再 emit：重建是同步进后台线程的，但**结果回来之前用户还能再点一次**
        # ——那会同时跑两个 worker，两次都写同一个文件。禁用这一下是唯一挡住它的东西。
        self.set_rebuild_busy(True)
        self.rebuild_requested.emit()

    # ── 巩固 ────────────────────────────────────────────────────────────────

    def _on_consolidation_toggled(self, checked: bool) -> None:
        """关掉自动那条时把间隔输入框一起灰掉。**不改它的值**（见那一处的注释）。"""
        self.consolidation_hours.setEnabled(checked)

    def set_consolidate_busy(self, busy: bool) -> None:
        """按下去之后按钮就一直禁用，直到主窗口把结果填回来。

        与 :meth:`set_rebuild_busy` 是同一个形状、同一个理由（防连按排出两轮），但
        **多一件事**：这里的按钮要说得清"在跑"——巩固可能几分钟，而"立即巩固"这四个
        字按下去之后如果什么都没变，用户会以为按钮坏了、再按一次。
        """
        self.consolidate_button.setEnabled(not busy)
        if busy:
            self.consolidate_note.setText("正在巩固（大库可能要几分钟）…")

    def set_consolidate_note(self, text: str) -> None:
        """由主窗口回填结果那一行。**成功与失败都走这里**（同图状态那一行的道理：
        复述两处就会出现两个说法，而它们迟早会在某条分支上分叉）。"""
        self.consolidate_note.setText(text)

    def _on_consolidate_clicked(self) -> None:
        self.set_consolidate_busy(True)
        self.consolidate_requested.emit()


class SettingsDialog(QDialog):
    """设置窗口本体。非模态——改一下就要看一眼主窗口，模态会把这件事变成
    "关窗 → 看效果 → 再开窗"。"""

    theme_changed = Signal(object)
    backend_changed = Signal(object)
    tools_changed = Signal(object)
    memory_changed = Signal(object)
    #: 「重建联想」被按下。转出去给主窗口——只有它知道图存哪儿、库有多大。
    graph_rebuild_requested = Signal()
    #: 「立即巩固」被按下。同上：主窗口知道库在哪儿、要不要先等图。
    consolidation_requested = Signal()
    persona_save_requested = Signal(str)
    persona_clear_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        *,
        theme: Theme,
        resolved: Resolved,
        backend: BackendSettings,
        tools: ToolSettings | None = None,
        memory: MemorySettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(False)
        # 对话框给个固定尺寸就够，不跟着屏幕长——它不是工作区，是一张表单。
        self.resize(720, 560)

        #: ``track`` 的契约：它要往这里记账（见 :mod:`biosnn_memterm.ui.thread_tracking`）。
        self._live_threads: list[QThread] = []
        self._probe_thread: QThread | None = None
        self._probe_worker: _ProbeWorker | None = None
        #: 「撤销修改」回到的那一份。`set_state` 会覆盖它。
        self._snapshot: Theme = theme

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.categories = QListWidget(self)
        self.categories.setObjectName("settingsCategories")
        self.categories.setFixedWidth(132)
        # 分类顺序**按碰到的频率排**：人设是"装好之后想起来才去改一次"，比 API 地址
        # 更常碰。顺序在这里是唯一的事实来源——`pages` 的添加顺序、以及
        # `currentRowChanged → setCurrentIndex` 那根线都按它对齐。
        #
        # **行高在这里定，不在样式表里。** 与 `#sessionList`（见 theme.qss 那条
        # 「几何归 delegate，颜色归样式表」）同一个理由：QSS 的 `padding` 与布局各自
        # 理解"这一行多高"，两边各算一次就可能对不上——曾出现过"每一行的盒子是 32 像素、
        # 行距却只有 16 像素"的画面（一条蓝条压住左右邻居的字），而那种错看起来像字体坏了。
        # QSS 那份只留横向内边距（文字离左边框多远），纵向高度由这里的 sizeHint 说了算。
        item_height = self.categories.fontMetrics().height() + 2 * _CATEGORY_PAD_Y
        for label in ("外观", "人设", "后端 / API", "工具", "记忆"):
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, item_height))
            self.categories.addItem(item)
        body.addWidget(self.categories)

        self.pages = QStackedWidget(self)
        self.appearance = AppearancePage(self)
        self.persona = PersonaPage(self)
        self.backend = BackendPage(self)
        self.tools = ToolPage(self)
        self.memory = MemoryPage(self)
        self.pages.addWidget(self.appearance)
        self.pages.addWidget(self.persona)
        self.pages.addWidget(self.backend)
        self.pages.addWidget(self.tools)
        self.pages.addWidget(self.memory)
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)

        footer = QFrame(self)
        footer.setObjectName("settingsFooter")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(12, 8, 12, 8)
        self.revert_button = QPushButton("撤销修改", footer)
        self.revert_button.setToolTip("回到打开这个窗口时的样子")
        self.revert_button.clicked.connect(self._on_revert)
        footer_row.addWidget(self.revert_button)
        footer_row.addStretch(1)
        self.close_button = QPushButton("完成", footer)
        self.close_button.setProperty("kind", "primary")
        self.close_button.clicked.connect(self.close)
        footer_row.addWidget(self.close_button)
        outer.addWidget(footer)

        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.categories.setCurrentRow(0)
        self.appearance.changed.connect(self.theme_changed)
        self.backend.apply_requested.connect(self.backend_changed)
        self.backend.test_requested.connect(self._on_test_requested)
        # 工具页**是**"边改边生效"的，与后端页的「应用地址」不同：加/删一条可读目录
        # 没有"应用"这一步可等——它只是一行设置，而且底部有「撤销修改」兜着。
        self.tools.roots_changed.connect(self.tools_changed)
        # 记忆页也是"边改边生效"：一个勾选框，勾上就是下一次写入生效。
        self.memory.changed.connect(self.memory_changed)
        # 「重建联想」是**动作**不是设置，所以它自己一路信号出去，不混进
        # ``memory_changed``：那一路的接收方做的是"把 MemorySettings 存起来"，
        # 而这件事要起一个后台线程。混在一起的代价是每次勾选都建一遍图。
        self.memory.rebuild_requested.connect(self.graph_rebuild_requested)
        # 「立即巩固」同理，而且它比建图更该走独立信号：那一头要起线程、要拍快照、
        # 要改库。混进 ``memory_changed`` 的代价是**每次勾选或改数字都跑一轮巩固**。
        self.memory.consolidate_requested.connect(self.consolidation_requested)
        # 人设页**没有** `changed` 这种"边改边生效"的信号：外观是拖一下就要看效果的
        # 参数，而写人设是一次显式的动作（它要落进记忆库）。所以这两个信号是按钮发
        # 出来的，与后端页的「应用地址」同一形状。
        self.persona.save_requested.connect(self.persona_save_requested)
        self.persona.clear_requested.connect(self.persona_clear_requested)

        self.set_state(
            theme,
            resolved,
            backend,
            tools=ToolSettings() if tools is None else tools,
            memory=MemorySettings() if memory is None else memory,
        )

    # ── 状态进出 ────────────────────────────────────────────────────────────

    def set_state(
        self,
        theme: Theme,
        resolved: Resolved,
        backend: BackendSettings,
        note: str = "",
        *,
        tools: ToolSettings,
        memory: MemorySettings,
    ) -> None:
        """**打开窗口时调。** 顺带把快照换成这一份——「撤销修改」回到的是**这一次**
        打开时的样子，而不是上一次的。

        ``tools`` 是**只能关键字传**的（它排在 ``note`` 后面，而那个位置按调用点的
        读法属于后端页）。不给它默认值是刻意的：漏传的症状是"工具页显示成空的、
        而设置里其实有目录"，一处看不出来的不一致；漏传直接 ``TypeError`` 更省事。
        """
        self._snapshot = theme
        self.appearance.set_state(theme, resolved)
        self.backend.set_state(backend, note)
        self.tools.set_state(tools)
        self.memory.set_state(memory)

    def set_resolved(self, resolved: Resolved) -> None:
        self.appearance.set_resolved(resolved)

    def set_backend_note(self, note: str) -> None:
        self.backend.set_note(note)

    def set_graph_note(self, note: str) -> None:
        """图状态的唯一出口。**空串会写成默认那句**，见 ``MemoryPage.set_graph_note``。"""
        self.memory.set_graph_note(note)

    def set_graph_busy(self, busy: bool) -> None:
        self.memory.set_rebuild_busy(busy)

    def set_consolidate_busy(self, busy: bool) -> None:
        """同上，转给「巩固」那一页的按钮。

        ``SettingsDialog`` 这一层只做**转发**——它自己不知道巩固跑没跑完，也不知道
        结果是什么。多一层转发的价值在于主窗口手里只有 ``SettingsDialog`` 这一个
        对象，不必伸手进 ``dialog.memory``。
        """
        self.memory.set_consolidate_busy(busy)

    def set_consolidate_note(self, note: str) -> None:
        self.memory.set_consolidate_note(note)

    def set_persona(self, builtin: str, user: str) -> None:
        """**打开窗口时调，写完之后也会再调一次。** 与 :meth:`set_state` 分列两个
        方法，是因为真源不同：外观与后端在 ``QSettings`` 里，人设**在记忆库里**
        （首启卡片、记忆窗口都能改它）。合进 ``set_state`` 会让"这一页从哪儿读"变得
        含糊——而含糊的代价是设置页显示一段已经过期的正文。
        """
        self.persona.set_persona(builtin, user)

    def set_persona_note(self, note: str) -> None:
        self.persona.set_note(note)

    # ── 撤销 ────────────────────────────────────────────────────────────────

    def _on_revert(self) -> None:
        """回到打开时的主题。

        **壁纸也回得来**：换图时按内容哈希命名、不覆盖旧副本，所以快照里那个
        文件名指向的那份还在。旧副本要到 :meth:`closeEvent` 才清。

        **可读目录也一起回去**：按钮上写着"回到打开这个窗口时的样子"，而"哪几个
        目录允许读"是这一页能改的东西里最该撤得回去的一项——它是一条授权。它自己
        管快照（见 :class:`ToolPage`），所以这里只是一句转发。

        **记忆那一页同理**（见 :class:`MemoryPage`）：它改的是入库的粒度，撤不回去
        的话，"撤销修改"这句话在这个窗口里就有两个意思了。
        """
        self.theme_changed.emit(self._snapshot)
        self.tools.revert()
        self.memory.revert()

    # ── 连接测试 ────────────────────────────────────────────────────────────

    def _on_test_requested(self, backend: BackendSettings) -> None:
        """测的是**输入框里还没应用的那一份**，不是当前生效的那一份。

        所以它带的是整个 :class:`BackendSettings`——云端下要测的地址与 key 都可能是
        刚敲进去、还没点「应用设置」的。
        """
        if self._probe_thread is not None and self._probe_thread.isRunning():
            return
        thread = QThread(self)
        worker = _ProbeWorker(backend)
        worker.done.connect(self.backend.set_note)
        worker.failed.connect(self.backend.set_note)
        # ★ 这两句与 `main_window._refresh_models` 里那两句是同一件事：`started`
        # 直连 `run()`，而 QThread 要等 `run()` 返回之后才进自己的 `exec()`——
        # 没人叫它退出，事件循环就永不结束，`finished` 永不发射。细节见
        # `thread_tracking.track` 的文档。
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        track(self, thread, worker, what="连接测试线程", slot="probe")
        thread.start()

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def prune_wallpapers(self) -> None:
        """删掉壁纸目录里**没被引用的** ``wallpaper-*`` 副本。

        只在关窗时做，而且只碰这个前缀——用户自己往那个目录里放的东西一律不动。
        不做的话每换一次图就积一份，换十次之后那目录会变成一堆叫不出名字的文件。

        **保留两份**：快照那份（「撤销修改」要用）与当前那份（正在显示的）。
        """
        keep = {self._snapshot.wallpaper, self.appearance.current_theme.wallpaper}
        for path in wallpapers_dir().glob(f"{_WALLPAPER_PREFIX}*"):
            if path.name in keep:
                continue
            # 删不掉不是错误：可能是别的实例正开着它（Windows 上会拒绝），也可能
            # 只是被占用。为一次后台清理弹一个错误框太重了。
            with contextlib.suppress(OSError):
                path.unlink()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """等连接测试线程停下来，再清掉没用的壁纸副本。

        **等线程**这件事不能省：一个仍在跑的 ``QThread`` 被析构会让 Qt 调
        ``qFatal``（Windows 上退出码 ``0xC0000409``，没有回溯）。理由与
        ``MainWindow._shutdown`` 一字不差。
        """
        wait_for_all(list(self._live_threads))
        self.prune_wallpapers()
        super().closeEvent(event)


def _is_wide_root(path: Path) -> bool:
    """这一条是不是"几乎等于整个盘"。

    两条判据，各自对着一种"看起来很小其实很大"的点击：

    * **驱动器根**——``Path("C:/").parent`` 就是它自己，这既是判据也是理由：根之上
      没有别的东西可以收窄它；
    * **家目录**——它技术上不是根，但里面装着这个人全部的文档、密钥与聊天记录。
      列表里它就一行字，看不出这个分量。

    只加警告，**不拒绝**：一个真要把整个盘交出去的人应当能交出去。``expanduser``
    放在这里，是因为设置里那一格允许被手写成 ``~``（见 ``settings_store._as_path_list``）。
    """
    expanded = path.expanduser()
    return expanded.parent == expanded or expanded == Path.home()


def _thumbnail(path: Path) -> QPixmap | None:
    """按需解出一张 160 宽的缩略图；读不出来返回 ``None``。

    **不走 ``QPixmap(str(path))``**：那会把一张 4000 万像素的图整张解开，而这里
    只要一张预览。先问 ``QImageReader.size()``（只读文件头）再把缩放交给解码器，
    于是内存里从头到尾只有那张小图。
    """
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid():
        reader.setScaledSize(QSize(160, max(1, round(160 * size.height() / size.width()))))
    image = reader.read()
    return None if image.isNull() else QPixmap.fromImage(image)


def _hours_of(seconds: int) -> int:
    """秒 → 界面上的整小时，夹在控件范围里。

    夹的理由与 :func:`~biosnn_memterm.ui.consolidation_scheduler._clamped_interval` 不同：
    那一处是为"手改注册表写进了 0"兜底（让调度器别变成紧循环），这一处只是**别把越界
    的值塞进 ``QSpinBox``**——控件自己也会夹，但那样一来"用户看到的"与"读回去的"就是
    两个数了，而这一页的读法依赖"看到的即存下的"。
    """
    hours = round(seconds / _SECONDS_PER_HOUR)
    return max(_MIN_INTERVAL_HOURS, min(_MAX_INTERVAL_HOURS, hours))


def _section(title: str) -> QLabel:
    label = QLabel(title)
    label.setObjectName("sectionTitle")
    return label


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("hint")
    return label
