"""设置窗口：外观页、人设页、后端页、工具页。

标 ``gui``，但**不碰 WebEngine、不建 ``MainWindow``**。要验的东西必须有一个真
``QApplication`` 才能问：控件之间的互斥、``QFileDialog`` 那两处测试缝、以及
"路径不存在时不崩"。

这个文件守的其实是**四件已经定过案的设计**，每一条都有它想推翻的、看起来更
省事的做法：

1. **每次改动都是一个完整的 ``Theme``，不是"改了哪一项"。** 只发增量的话，接收
   方得自己拼当前状态——于是状态有了两个家。所以断言不看"信号发了"，看
   **发出来的对象里其余字段还在不在**。
2. **换壁纸按内容哈希命名，不覆盖上一份。** 这是「撤销修改」的前提：覆盖着写的话
   撤销之后那个名字指向的是一个已经不存在的文件。``test_installing_a_second_…``
   与 ``test_prune_…`` 两条一起把这条纪律钉住。
3. **取消取色要把勾退回去。** 不退的话色块显示"自定义"、主色却还是上一个，用户
   下一次点它会以为程序记住了他刚才取消掉的那个色。
4. **加一条可读目录等于一次授权，所以「撤销修改」必须把它也收回去。** 那一页因此
   自己管快照（``ToolPage.revert``），而不像后端页那样"点「应用」才生效"。这条
   要是丢了，界面上那句"回到打开这个窗口时的样子"就变成了一句假话——而那时模型
   已经能读那个目录了。
5. **「睡眠巩固」那一格上的三个控件是同一件事的三个说法，而它们各有各的失效方式。**
   勾选框决定"自动跑不跑"、间隔决定"多久一次"、按钮是"现在就跑一次"。关掉勾选框
   **只灰掉间隔、不动按钮**——"不自动跑"与"不许跑"是两件事，混起来用户会以为巩固
   整个被关掉了。另外两处的默认值与前面两格相反（这一格默认开），所以"忘了回填"
   的症状也跟着反过来：不是"开了没勾上"，是"关掉了每次开设置又自己勾上"。

``QColorDialog`` / ``QFileDialog`` 都不弹真的：``AppearancePage`` 上有两个可覆盖的
方法（``_ask_for_colour`` / ``_ask_for_wallpaper``），``ToolPage`` 上是
``_ask_for_directory``，照抄 ``memory_window`` 里 ``_ask_for_content`` 的既有模式。
真弹一个模态取色器会让这个文件在 CI 上挂住。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QLabel

from axo import autostart
from axo.ui import settings_dialog
from axo.ui.settings_dialog import (
    AppearancePage,
    BackendPage,
    ReminderPage,
    SettingsDialog,
    ToolPage,
)
from axo.ui.settings_store import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_HOST,
    PROVIDER_CLOUD,
    PROVIDER_OLLAMA,
    BackendSettings,
    MemorySettings,
    ReminderSettings,
    ToolSettings,
)
from axo.ui.theme import (
    CUSTOM_PRESET,
    PRESETS,
    WALLPAPER_VISIBLE_MAX,
    Theme,
    resolve,
)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    app = QApplication.instance() or QApplication([])
    yield app
    app.setStyleSheet("")


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把数据目录指到临时目录，于是 ``wallpapers_dir()`` 也在那儿。

    设环境变量而不是 monkeypatch ``wallpapers_dir`` 本身：那个函数每次调用都重新
    解析路径（没有缓存），所以环境变量是**唯一**的开关，也正好让"复制到的位置"
    与"清理时扫的位置"是同一个——把函数替换掉的话，两边会各自指向不同目录，
    而这个文件里最要紧的两条断言恰好就是关于这两个位置的。
    """
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    return tmp_path


def _png(path: Path, colour: str = "#123456", size: tuple[int, int] = (8, 8)) -> Path:
    """造一张真图片。**不用 Pillow**：测试依赖表里没有它，而 Qt 本来就在手边。"""
    image = QImage(*size, QImage.Format.Format_RGB32)
    image.fill(QColor(colour))
    assert image.save(str(path), "PNG"), f"写不出 {path}"
    return path


def _watch(page: AppearancePage) -> list[Theme]:
    seen: list[Theme] = []
    page.changed.connect(seen.append)
    return seen


def _page(theme: Theme | None = None) -> tuple[AppearancePage, list[Theme]]:
    page = AppearancePage()
    seen = _watch(page)
    if theme is not None:
        page.set_state(theme, resolve(theme, "dark"))
    return page, seen


# ── 主色 ────────────────────────────────────────────────────────────────────


def test_there_is_a_swatch_for_every_preset(qapp, data_root) -> None:
    page, _ = _page()
    assert set(page._swatches) == {item.key for item in PRESETS}
    assert len(page._swatches) == 4


def test_clicking_a_preset_emits_a_complete_theme(qapp, data_root) -> None:
    """发出来的是**完整**的一份，不是"preset=jade"这样的增量。"""
    page, seen = _page()
    page._swatches["jade"].click()
    assert len(seen) == 1
    assert seen[0].preset == "jade"
    # accent 由 Theme.__post_init__ 从预设收敛过来，接收方不必自己查表
    assert seen[0].accent == PRESETS[1].accent


def test_the_other_fields_survive_a_preset_click(qapp, data_root) -> None:
    """**这条是"发完整对象"的收益所在。**

    只发增量的实现会在这里露馅：它发一个 ``{"preset": "jade"}``，而接收方要是
    老老实实拿它构造 ``Theme``，明亮模式与壁纸设置就一起被重置成默认了——
    一个"换了个颜色，壁纸没了"的 bug，看起来像是壁纸功能有问题。
    """
    start = Theme(preset="amber", mode="light", wallpaper="x.png", wallpaper_visible=42)
    page, seen = _page(start)
    page._swatches["indigo"].click()
    assert seen[0] == Theme(preset="indigo", mode="light", wallpaper="x.png", wallpaper_visible=42)


def test_clicking_the_same_preset_says_nothing(qapp, data_root) -> None:
    """已经选中的那个再点一下**不发信号**。``Theme`` 是 frozen dataclass，
    ``==`` 按值比，所以这一句判断是免费的；不该为一次没有变化的点击去重设整张
    样式表、重推一次网页。"""
    page, seen = _page(Theme(preset="jade"))
    page._swatches["jade"].click()
    assert seen == []


def test_opening_the_window_says_nothing(qapp, data_root) -> None:
    """``set_state`` 是**陈述现状**，不是做修改。

    不挡的话，打开设置窗口这一下就会往外发一轮信号，主窗口照着同一份主题再
    apply 一遍——并在「撤销修改」的快照上盖一个"刚刚才改过"的时间戳。
    """
    page, seen = _page()
    page.set_state(Theme(preset="amber", wallpaper_visible=33), resolve(Theme(), "dark"))
    assert seen == []


def test_a_custom_colour_is_normalised(qapp, data_root, monkeypatch) -> None:
    """取色器给什么形式都行，存下去的必须是唯一形式——否则"同一个颜色"会有多个
    字符串表示，比对与去重都会出错。"""
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_colour", lambda _initial: QColor("#F80"))
    page.custom_button.click()
    assert seen[0].preset == CUSTOM_PRESET
    assert seen[0].accent == "#ff8800"


@pytest.mark.parametrize("answer", [None, QColor()], ids=["none", "invalid"])
def test_cancelling_the_colour_picker_puts_the_tick_back(
    qapp, data_root, monkeypatch, answer
) -> None:
    """取消之后色块不能停在"自定义"上。两种取消都要管：``QColorDialog`` 取消时
    返回一个**非法** ``QColor``（不是 ``None``），而我们的测试缝返回 ``None``。"""
    page, seen = _page(Theme(preset="jade"))
    monkeypatch.setattr(page, "_ask_for_colour", lambda _initial: answer)
    page.custom_button.click()
    assert seen == []
    assert page._swatches["jade"].isChecked()
    assert not page.custom_button.isChecked()


def test_the_custom_swatch_shows_the_chosen_colour(qapp, data_root) -> None:
    """自定义色块要显示**用户选的色**。不设的话它永远是同一个灰，
    "当前主色是什么"在界面上就没有了答案。"""
    page, _ = _page(Theme(preset=CUSTOM_PRESET, accent="#ff8800"))
    assert page.custom_button.preset.accent == "#ff8800"


# ── 明暗与可见度 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["system", "dark", "light"])
def test_each_mode_radio_emits_its_own_value(qapp, data_root, mode) -> None:
    # 起手刻意**不是**要点的那个：点一个已经选中的单选钮不会改变状态，也就不会
    # 发 ``toggled``，于是"system"那一组（默认就是 system）会空转通过。
    page, seen = _page(Theme(mode="light" if mode != "light" else "dark"))
    for button in page._mode_group.buttons():
        if button.property("modeValue") == mode:
            button.click()
            break
    else:  # pragma: no cover - 三个值必须都有对应的单选钮
        pytest.fail(f"没有 {mode} 对应的单选钮")
    assert [item.mode for item in seen] == [mode]


def test_the_slider_emits_the_new_visibility(qapp, data_root) -> None:
    page, seen = _page()
    page.visible_slider.setValue(40)
    assert [item.wallpaper_visible for item in seen] == [40]
    assert page.visible_value.text() == "40%"


def test_the_slider_cannot_go_past_the_deliberate_ceiling(qapp, data_root) -> None:
    """滑杆的上限就是 ``WALLPAPER_VISIBLE_MAX``。**这里断言的是控件**——
    ``Theme`` 自己也会拦，但那时用户已经在拖一个能拖到 100 的滑杆了，
    报错的地方离原因很远。"""
    page, _ = _page()
    assert page.visible_slider.maximum() == WALLPAPER_VISIBLE_MAX


# ── 壁纸 ────────────────────────────────────────────────────────────────────


def test_choosing_a_wallpaper_copies_it_into_the_data_dir(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """**复制，不是记住原路径。** 记住路径的话，用户把图移走、改名、拔掉 U 盘之后
    壁纸就没了，症状是"壁纸莫名其妙不见了"。"""
    source = _png(tmp_path / "我的 照片#1.png")
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(source))
    page.pick_button.click()

    assert len(seen) == 1
    assert seen[0].wallpaper.startswith("wallpaper-")
    copied = settings_dialog.wallpapers_dir() / seen[0].wallpaper
    assert copied.is_file()
    assert copied.read_bytes() == source.read_bytes()


def test_picking_a_wallpaper_puts_the_thumbnail_up_right_away(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """选完图，这一栏里**立刻**要看得出变化。

    它是真能坏的一环：缩略图只在 ``set_state`` 里画，而选完图到下次打开窗口之间
    没有 ``set_state``——所以不补这一下的话，"选完什么也没发生"，唯一的反馈在网页
    那边。
    """
    page, _ = _page()
    assert page.wallpaper_preview.pixmap().isNull()  # 起手是空的
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(_png(tmp_path / "壁纸.png")))
    page.pick_button.click()
    assert not page.wallpaper_preview.pixmap().isNull()


def test_the_thumbnail_is_drawn_again_when_the_window_reopens(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """重开窗口时按设置里那个文件名重新画一遍缩略图。

    **这条守的是一个真发生过的死机。** 画缩略图那行写的是 ``image.toPixmap()``
    ——PySide6 的 ``QImage`` 没有这个方法（要用 ``QPixmap.fromImage``）。它一抛，
    ``set_state`` 就在 ``show()`` 之前中断，于是「设置」按下去什么也不出现，
    再按也一样，而屏幕上没有任何线索。

    以前没被逮住是因为 ``test_prune_keeps_the_snapshot_and_the_current_copy`` 里
    那份"快照副本"是 ``write_bytes(b"x")`` 造的一字节假文件，读出来是空图，走的是
    上面那条 ``_warn`` 分支。所以这里必须真装一张能读的图。
    """
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(_png(tmp_path / "壁纸.png")))
    page.pick_button.click()
    page.set_state(seen[0], resolve(seen[0], "dark"))
    assert not page.wallpaper_preview.pixmap().isNull()
    assert page.wallpaper_name.text() == seen[0].wallpaper


def test_a_non_image_is_refused_without_emitting(qapp, data_root, monkeypatch, tmp_path) -> None:
    """**不弹框，就地写一行字。** 选错文件是一种可以就地重选的输入问题，
    弹一个要点的框只会让"再选一张"多两步。"""
    liar = tmp_path / "其实是文本.png"
    liar.write_text("这不是图片", encoding="utf-8")
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(liar))
    page.pick_button.click()
    assert seen == []
    assert page.wallpaper_name.text()  # 说了一句，不是静默失败


def test_a_missing_file_is_refused_without_emitting(qapp, data_root, monkeypatch, tmp_path) -> None:
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(tmp_path / "没有这个.png"))
    page.pick_button.click()
    assert seen == []
    assert page.wallpaper_name.text()


def test_an_enormous_image_is_refused(qapp, data_root, monkeypatch, tmp_path) -> None:
    """像素上限。**不给它一张 40 万像素的真图**——那是 160 MB 的位图，会让这个
    用例变成整个测试套里最慢的一个。把上限调小，验的是那条判断本身。"""
    monkeypatch.setattr(settings_dialog, "_MAX_PIXELS", 10)
    source = _png(tmp_path / "big.png", size=(8, 8))
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(source))
    page.pick_button.click()
    assert seen == []
    assert "超过" in page.wallpaper_name.text()


def test_an_enormous_file_is_refused_and_a_large_one_is_only_noted(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """体积两档：超过上限**拒**，超过警告线只是**说一句**。后者刻意不拒——
    一张 25 MB 的图能用，只是慢；把它拒掉是替用户做了一个他没要求的选择。"""
    source = _png(tmp_path / "photo.png")
    monkeypatch.setattr(settings_dialog, "_MAX_BYTES", 10)
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(source))
    page.pick_button.click()
    assert seen == []
    assert "超过上限" in page.wallpaper_name.text()

    monkeypatch.setattr(settings_dialog, "_MAX_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(settings_dialog, "_WARN_BYTES", 10)
    page.pick_button.click()
    assert len(seen) == 1  # 收下了
    assert "慢" in page.wallpaper_name.text()


def test_the_same_image_installed_twice_does_not_pile_up(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """按**内容**哈希命名，所以同一张图重复选不会积出两份副本。按时间戳命名的话
    每点一次多一个文件，而它们在界面上长得一模一样、在设置里也分不出来。

    第二次连信号都不发——名字一样 ⇒ 主题一样 ⇒ ``_replace`` 的相等判断直接返回。
    这是对的：用户没有改变任何东西。
    """
    source = _png(tmp_path / "same.png")
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(source))
    page.pick_button.click()
    page.pick_button.click()
    assert len(seen) == 1
    assert len(list(settings_dialog.wallpapers_dir().glob("wallpaper-*"))) == 1
    assert (settings_dialog.wallpapers_dir() / seen[0].wallpaper).is_file()


def test_installing_a_second_wallpaper_keeps_the_first_on_disk(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """**这一条是「撤销修改」的物理前提。**

    覆盖着写的话，撤销之后设置里那个名字指向的是一个已经不在的文件——用户点了
    「撤销修改」，壁纸栏显示"找不到了"。所以换图时旧副本必须还在，直到窗口关闭
    才由 ``prune_wallpapers`` 决定去留。
    """
    first = _png(tmp_path / "a.png", "#111111")
    second = _png(tmp_path / "b.png", "#222222")
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(first))
    page.pick_button.click()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(second))
    page.pick_button.click()

    assert seen[0].wallpaper != seen[1].wallpaper
    for theme in seen:
        assert (settings_dialog.wallpapers_dir() / theme.wallpaper).is_file()


def test_clearing_removes_the_name_but_not_the_file(qapp, data_root, monkeypatch, tmp_path) -> None:
    """「清除」只把设置里的名字去掉。文件留着，因为「撤销修改」还要用它。"""
    source = _png(tmp_path / "c.png")
    page, seen = _page()
    monkeypatch.setattr(page, "_ask_for_wallpaper", lambda: str(source))
    page.pick_button.click()
    page.clear_button.click()
    assert seen[-1].wallpaper == ""
    assert (settings_dialog.wallpapers_dir() / seen[0].wallpaper).is_file()


# ── 撤销与清理 ──────────────────────────────────────────────────────────────


def _dialog(theme: Theme | None = None) -> SettingsDialog:
    theme = theme or Theme()
    return SettingsDialog(
        None,
        theme=theme,
        resolved=resolve(theme, "dark"),
        backend=BackendSettings(),
    )


def test_revert_goes_back_to_the_theme_from_when_the_window_opened(qapp, data_root) -> None:
    opened = Theme(preset="jade")
    dialog = _dialog(opened)
    seen: list[Theme] = []
    dialog.theme_changed.connect(seen.append)

    dialog.appearance._swatches["indigo"].click()
    assert seen, "改了却什么都没发出来，这个用例什么都没有验到"
    seen.clear()

    dialog.revert_button.click()
    assert seen == [opened]


def test_reopening_moves_the_snapshot(qapp, data_root) -> None:
    """重开时 ``set_state`` 会**换掉快照**。

    不换的话「撤销修改」会回到一段不再成立的历史——用户上次打开窗口时是琥珀色、
    后来改成了蓝紫，再打开窗口点撤销，界面会变成几十分钟前的琥珀色。
    """
    dialog = _dialog(Theme(preset="jade"))
    seen: list[Theme] = []
    dialog.theme_changed.connect(seen.append)

    dialog.set_state(
        Theme(preset="amber"),
        resolve(Theme(preset="amber"), "dark"),
        BackendSettings(),
        tools=ToolSettings(),
        memory=MemorySettings(),
        reminders=ReminderSettings(),
    )
    assert seen == [], "set_state 是在陈述现状，不该发信号"

    dialog.revert_button.click()
    assert seen == [Theme(preset="amber")]


def test_prune_keeps_the_snapshot_and_the_current_copy(
    qapp, data_root, monkeypatch, tmp_path
) -> None:
    """清理保留**两份**：快照那份（撤销要用）与当前那份（正在显示）。

    这里走的是真实路径而不是摆几个文件上去——先让快照指向一份手放的副本，再真的
    选一张图（于是当前指向一份新副本），中间那份没被引用的就应当消失。
    """
    wallpapers = settings_dialog.wallpapers_dir()
    snapshot_copy = wallpapers / "wallpaper-aaaaaaaaaa.png"
    stale = wallpapers / "wallpaper-bbbbbbbbbb.png"
    bystander = wallpapers / "我自己放的.png"
    for path in (snapshot_copy, stale, bystander):
        path.write_bytes(b"x")

    dialog = _dialog(Theme(wallpaper=snapshot_copy.name))
    source = _png(tmp_path / "new.png")
    monkeypatch.setattr(dialog.appearance, "_ask_for_wallpaper", lambda: str(source))
    dialog.appearance.pick_button.click()
    current = dialog.appearance.current_theme.wallpaper
    assert current != snapshot_copy.name

    dialog.prune_wallpapers()

    assert snapshot_copy.is_file(), "快照那份被删了，「撤销修改」会指向一个不存在的文件"
    assert (wallpapers / current).is_file(), "正在显示的那份被删了"
    assert not stale.exists(), "没被引用的副本没被清掉，换十次图就积十个文件"
    assert bystander.is_file(), "用户自己放进壁纸目录的文件被误删了"


# ── 后端页 ──────────────────────────────────────────────────────────────────


def test_an_empty_host_falls_back_to_the_default(qapp, data_root) -> None:
    """**空输入框不能变成空 host。** 空字符串会走到 ``_client()`` 里变成一个
    没有 scheme 的 URL，报出来的错与"我没填地址"毫无关系。"""
    page = BackendPage()
    page.host_edit.setText("   ")
    assert page.current().host == DEFAULT_HOST


def test_the_host_is_stripped(qapp, data_root) -> None:
    """从别处复制地址常常带一个尾随空格或换行。"""
    page = BackendPage()
    page.host_edit.setText("  http://192.168.1.9:11434 \n")
    assert page.current().host == "http://192.168.1.9:11434"


def test_arming_remote_changes_the_property(qapp, data_root) -> None:
    """勾上之后那句代价说明要转成告警色——靠一个动态属性，样式表读它。"""
    page = BackendPage()
    assert page.remote_warning.property("armed") == "no"
    page.allow_remote_box.setChecked(True)
    assert page.remote_warning.property("armed") == "yes"


def test_the_warning_is_readable_before_it_is_armed(qapp, data_root) -> None:
    """**代价要在勾之前就能读到。** 用户得先知道会发生什么才能决定要不要勾，
    把说明藏到勾选之后是本末倒置——那时他已经同意了。

    这里断言的是文案不空并且提到了代价，不是"颜色对不对"（那要靠 QSS 求值，
    而全局样式表的求值结果没有可断言的观测口，见 ``test_ui_context_strip.py``
    开头那段同样的取舍）。
    """
    page = BackendPage()
    assert "不出设备" in page.remote_warning.text()
    assert not page.remote_warning.isHidden()


def test_applying_emits_the_typed_in_values(qapp, data_root) -> None:
    page = BackendPage()
    seen: list[BackendSettings] = []
    page.apply_requested.connect(seen.append)
    page.host_edit.setText("http://10.0.0.5:11434")
    page.allow_remote_box.setChecked(True)  # 这一下也会发一次
    page.apply_button.click()
    assert seen[-1] == BackendSettings(host="http://10.0.0.5:11434", allow_remote=True)


def test_testing_asks_about_the_typed_in_host_not_the_applied_one(qapp, data_root) -> None:
    """「测试连接」测的必须是**输入框里**那个地址。

    测已生效的那个是没意义的：那个地址的状态主窗口状态栏上一直写着。用户按这个
    按钮的唯一动机就是"我想在应用之前先看看新地址通不通"。
    """
    page = BackendPage()
    seen: list[BackendSettings] = []
    page.test_requested.connect(seen.append)
    page.host_edit.setText("http://10.0.0.5:11434")
    page.test_button.click()
    assert [item.host for item in seen] == ["http://10.0.0.5:11434"]
    assert seen[0].allow_remote is False
    assert page.note.text()  # 立刻说一句"正在测试"，否则按下去像没反应


def test_a_failed_probe_writes_the_reason_not_a_silent_nothing(qapp, data_root) -> None:
    page = BackendPage()
    page.set_note("连不上 http://10.0.0.5:11434：Connection refused")
    assert "Connection refused" in page.note.text()


# ── 人设页 ──────────────────────────────────────────────────────────────────

# ── 记忆页：任务粒度 ────────────────────────────────────────────────────────


def test_the_memory_box_opens_on_the_stored_value(qapp, data_root) -> None:
    """打开窗口时显示的是**设置里那一份**，不是控件的默认态。

    这一项默认关，所以"忘了回填"与"用户没打开过"在界面上长得一模一样——用户勾上，
    关掉窗口，再打开，看到它还是空的，会以为自己那次没生效。
    """
    dialog = _dialog()
    dialog.set_state(
        Theme(),
        resolve(Theme(), "dark"),
        BackendSettings(),
        tools=ToolSettings(),
        memory=MemorySettings(task_memory=True),
        reminders=ReminderSettings(),
    )
    assert dialog.memory.task_box.isChecked()


def test_the_memory_box_emits_only_when_the_user_moves_it(qapp, data_root) -> None:
    """**陈述现状不发信号，用户动了才发**（同外观页那道闸门）。

    打开窗口这一下就往外发一轮的话，主窗口会照着同一份设置再写一遍库——表现上无害，
    但"打开设置"与"改了设置"从此在接收侧再也分不开。
    """
    dialog = _dialog()
    seen: list[MemorySettings] = []
    dialog.memory_changed.connect(seen.append)

    dialog.set_state(
        Theme(),
        resolve(Theme(), "dark"),
        BackendSettings(),
        tools=ToolSettings(),
        memory=MemorySettings(task_memory=True),
        reminders=ReminderSettings(),
    )
    assert seen == [], "摆上当前值不是一次改动"

    dialog.memory.task_box.setChecked(False)
    assert seen == [MemorySettings(task_memory=False)]


def test_revert_puts_the_memory_switch_back(qapp, data_root) -> None:
    """「撤销修改」也收得回粒度——它改的是**入库的形状**，比一个色号更该撤得回去。

    没变就不发信号：不然每按一次「撤销修改」都会往设置里重写一遍同样的值。
    """
    dialog = _dialog()
    seen: list[MemorySettings] = []
    dialog.memory_changed.connect(seen.append)
    dialog.memory.task_box.setChecked(True)

    dialog.revert_button.click()

    assert dialog.memory.task_box.isChecked() is False, "撤销返回的是打开窗口时那份（关）"
    assert seen == [MemorySettings(task_memory=True), MemorySettings(task_memory=False)]

    dialog.revert_button.click()
    assert len(seen) == 2, "已经回到快照了，再按一次不该再发"


def _memory_dialog(**memory: Any) -> SettingsDialog:
    """开一次带记忆页的窗口。``_dialog()`` 那份不给 ``memory``（走构造器默认值），
    两条路都要有人走，否则"构造器漏传"这一类错没人挡。"""
    dialog = _dialog()
    dialog.set_state(
        Theme(),
        resolve(Theme(), "dark"),
        BackendSettings(),
        tools=ToolSettings(),
        memory=MemorySettings(**memory),
        reminders=ReminderSettings(),
    )
    return dialog


def test_the_graph_box_opens_on_the_stored_value(qapp, data_root) -> None:
    """同任务记忆那一格：显示的是**设置里那一份**。

    这一项也默认关，所以"忘了回填"与"用户没打开过"在界面上长得一模一样。
    """
    assert _memory_dialog(graph_channel=True).memory.graph_box.isChecked()
    assert not _memory_dialog().memory.graph_box.isChecked()


def test_moving_either_memory_box_sends_both_fields(qapp, data_root) -> None:
    """**发出去的是一整份 ``MemorySettings``，两个字段都得在。**

    少带 ``graph_channel`` 的症状不是报错：勾了它、关掉设置、再打开又变回没勾——
    因为写盘是照这一份写的，而那里面根本没有这个字段。少带 ``task_memory`` 则更糟，
    它会把用户之前勾好的任务记忆**静默关掉**。
    """
    dialog = _memory_dialog()
    seen: list[MemorySettings] = []
    dialog.memory_changed.connect(seen.append)

    dialog.memory.graph_box.setChecked(True)
    assert seen == [MemorySettings(task_memory=False, graph_channel=True)]

    dialog.memory.task_box.setChecked(True)
    assert seen[-1] == MemorySettings(task_memory=True, graph_channel=True), (
        "勾另一格时不许把这一格丢掉"
    )


def test_the_two_memory_boxes_do_not_share_state(qapp, data_root) -> None:
    """两个框各管各的。**它们共用同一个槽**（``_on_toggled``），所以"槽读错了框"
    这类错是可能的，而这正是共用一个槽省下来的那点代码要还的债。"""
    dialog = _memory_dialog()
    dialog.memory.graph_box.setChecked(True)

    assert dialog.memory.task_box.isChecked() is False
    assert dialog.memory.current() == MemorySettings(task_memory=False, graph_channel=True)

    dialog.memory.task_box.setChecked(True)
    dialog.memory.graph_box.setChecked(False)
    assert dialog.memory.current() == MemorySettings(task_memory=True, graph_channel=False)


def test_revert_puts_the_graph_switch_back(qapp, data_root) -> None:
    """「撤销修改」也要收得回联想那一格。

    快照是整份 ``MemorySettings``，所以只要 ``current()`` 读全了字段，这里自然是
    对的——但"自然是"正是这一类错最容易藏进去的地方（同上面那条发信号的用例）。
    """
    dialog = _memory_dialog()
    seen: list[MemorySettings] = []
    dialog.memory_changed.connect(seen.append)
    dialog.memory.graph_box.setChecked(True)

    dialog.revert_button.click()

    assert dialog.memory.graph_box.isChecked() is False
    assert seen[-1] == MemorySettings(task_memory=False, graph_channel=False)


def test_the_cost_of_the_graph_switch_is_written_on_the_page(qapp, data_root) -> None:
    """代价**写在勾选框旁边**，不是只写在报告里。

    这是本仓唯一一个"打开之后可能变差"的开关（实测 hit@1 0.8500 → 0.8250）。把数
    留在 ``CORPUS.md`` 里而界面上只写"启用联想通道"，用户能拿到的全部信息就是
    "有个新功能"——那不是同意，是误导。
    """
    dialog = _memory_dialog()
    cost = dialog.memory.graph_cost.text()

    assert "min_score" in cost, "门槛会被移动这件事必须在界面上"
    assert "0.8500" in cost and "0.8250" in cost, "实测是负的，数要照抄"
    assert "CORPUS.md" in cost, "要指得到那一页报告"


def test_rebuilding_the_graph_asks_and_then_waits(qapp, data_root) -> None:
    """「重建联想」按下去：**先禁用自己再发信号**。

    重建要跑几秒到几分钟，而按一次是一次后台计算 + 一次写盘。不禁用的话用户能连按，
    两个 worker 同时写同一个图文件。
    """
    dialog = _memory_dialog()
    seen: list[int] = []
    dialog.graph_rebuild_requested.connect(lambda: seen.append(1))

    dialog.memory.rebuild_button.click()

    assert seen == [1], "按一次发一次，不发或者发两次都是错的"
    assert dialog.memory.rebuild_button.isEnabled() is False
    assert "正在重建" in dialog.memory.graph_note.text()


def test_the_rebuild_button_comes_back_when_the_result_lands(qapp, data_root) -> None:
    """结果回来就把按钮放开，并且**只有一句说法**：状态行。

    按钮旁边不复述结果——复述会出两个说法，而它们迟早会在某条分支上分叉。
    """
    dialog = _memory_dialog()
    dialog.memory.rebuild_button.click()

    dialog.set_graph_busy(False)
    dialog.set_graph_note("已重建：12 个节点、48 条边，建于 12:00:00。")

    assert dialog.memory.rebuild_button.isEnabled() is True
    assert dialog.memory.graph_note.text().startswith("已重建")
    assert "正在重建" not in dialog.memory.graph_note.text()


def test_an_empty_graph_note_still_says_something(qapp, data_root) -> None:
    """状态行为空时写**默认那句**，不留白。

    这一行在"还没人建过图"的时候也必须说得通，而空白会被读成"没有这回事"。默认那
    句与记忆窗口里「关联图还没有建」那条顶栏同口径——同一个状态在两处说两样话，
    用户会以为它们是两件事。
    """
    dialog = _memory_dialog()

    assert dialog.memory.graph_note.text().strip(), "开窗时这一行不该是空的"
    assert "还没建" in dialog.memory.graph_note.text()

    dialog.set_graph_note("")
    assert "还没建" in dialog.memory.graph_note.text()


# ── 睡眠巩固（P4c） ────────────────────────────────────────────────────────


def _hints(page: Any) -> str:
    """那一页上所有 ``hint`` 标签的文字，拼成一段。"""
    return "\n".join(one.text() for one in page.findChildren(QLabel) if one.objectName() == "hint")


def test_the_consolidation_box_opens_on_the_stored_value(qapp, data_root) -> None:
    """这一格**默认开**，与上面两格相反——所以"忘了回填"的症状也是反的。

    任务记忆与联想通道默认关，忘回填的表现是"用户开了、窗口上还是关的"；巩固默认
    开，同一个 bug 的表现变成**用户明明关掉了，每次开设置又自己勾上**——而这一格
    勾着意味着每天有人在你没看着的时候改一次库。两种都错，错法不同，所以这条不能
    与上面那条合并。
    """
    assert _memory_dialog().memory.consolidation_box.isChecked()
    assert not _memory_dialog(consolidation_enabled=False).memory.consolidation_box.isChecked()


def test_the_interval_is_shown_in_hours_not_seconds(qapp, data_root) -> None:
    """存的是秒（``memory/consolidation_interval_s``），显示的是**小时**。

    换算错了不会有任何报错：用户看到"每隔 86400 小时跑一次"，只当是有点怪，而它
    实际是十年——那与"关掉自动跑"在真实使用里没有区别，勾选框却还勾着。
    """
    assert _memory_dialog().memory.consolidation_hours.value() == 24, "86400 秒 = 一天"
    assert _memory_dialog(consolidation_interval_s=3600).memory.consolidation_hours.value() == 1
    assert _memory_dialog(consolidation_interval_s=10800).memory.consolidation_hours.value() == 3


def test_a_hand_edited_interval_rounds_to_the_nearest_hour(qapp, data_root) -> None:
    """5400 秒（1.5 小时）显示成 2 小时——**这一步是有损的，而且被认下来了**。

    界面的粒度就是小时，所以一个手改注册表写进去的 1.5 小时在界面上表达不出来。
    这里钉住的是"表达不出来时取最接近的整小时"，而不是"原值被保住"——写清这一点，
    是为了下一次有人想把粒度改成分钟时，知道这里有一次明说的取舍。
    """
    assert _memory_dialog(consolidation_interval_s=5400).memory.consolidation_hours.value() == 2


def test_everything_this_page_can_produce_is_a_period_the_scheduler_accepts(
    qapp, data_root
) -> None:
    """界面上填得出来的每一个值，调度器都收得下。**注意这是包含关系，不是相等。**

    两个下限是**两个不同的数，而且这是有意的**：调度器那一侧是 60 秒，挡的是"绕过
    界面写进注册表的 0"（见 ``INTERVAL_FLOOR_S`` 的 docstring，它自己写了"这不该是
    设置界面的下限"）；界面这一侧是 1 小时，因为一轮本身以分钟计。所以这里断言的是
    "界面的下限严格大于调度器的下限"——写成相等会让下一个改动以为两边是一回事，而
    把界面的下限降到 60 秒不会让任何一条测试红，只会让用户能填出一个"下一轮追着上
    一轮跑"的周期。

    上限那一侧**必须相等**，而且理由是硬的：``QTimer.start`` 收的是 C 的 ``int``，
    毫秒数超过 ``2**31 - 1`` 会让 PySide6 抛 ``OverflowError``——那会从一个设置项
    一路穿出 Qt 信号，也就是"改一下这个数字，主窗口当场崩掉"。
    """
    from axo.ui.consolidation_scheduler import INTERVAL_CEIL_S, INTERVAL_FLOOR_S

    # **先把窗口接住再取控件**：``_memory_dialog().memory.consolidation_hours`` 这种
    # 一行写法里，那个临时 ``SettingsDialog`` 在取完属性之后就被回收了，PySide6 会
    # 顺手把整棵子控件树删掉——后面读到的是一个已经析构的 C++ 对象。
    dialog = _memory_dialog()
    spin = dialog.memory.consolidation_hours

    assert spin.minimum() * 3600 > INTERVAL_FLOOR_S, "界面比调度器更严，而不是同一个数"
    assert spin.maximum() * 3600 == INTERVAL_CEIL_S
    assert INTERVAL_CEIL_S * 1000 < 2**31 - 1, "排得进 QTimer 的才是合法上限"
    assert "小时" in spin.suffix(), "没有单位的数字会被读成秒"


def test_moving_a_memory_control_sends_both_consolidation_fields(qapp, data_root) -> None:
    """勾选框与间隔发的是**同一整份**，两个字段都得在。

    少带 ``consolidation_enabled`` 的症状是"关掉它、再打开又是勾上的"；少带
    ``consolidation_interval_s`` 更安静——写盘那一份里没有这个字段，于是用户填的
    周期被**默默换回默认的一天**，而没有任何一处报错。
    """
    dialog = _memory_dialog()
    seen: list[MemorySettings] = []
    dialog.memory_changed.connect(seen.append)

    dialog.memory.consolidation_hours.setValue(3)
    assert seen[-1] == MemorySettings(consolidation_interval_s=10800)

    dialog.memory.consolidation_box.setChecked(False)
    assert seen[-1] == MemorySettings(
        consolidation_enabled=False, consolidation_interval_s=10800
    ), "关掉自动那条时不许把周期丢掉"


def test_the_interval_box_is_greyed_without_the_switch_but_keeps_its_value(qapp, data_root) -> None:
    """关掉勾选框 → 间隔**灰掉，但不改值、也不清零**。

    灰而不是藏：用户填过的数字要留着，重新勾上时它还在——"我上次填的是多少"不该
    变成一个需要记住的问题。也不清零：清零看着干净，但那等于用户每开关一次就丢一次
    设置。而"灰着仍然读得出来"是刻意的——读的是控件里的值，不是"它能不能改"。
    """
    dialog = _memory_dialog()
    dialog.memory.consolidation_hours.setValue(6)

    dialog.memory.consolidation_box.setChecked(False)

    assert dialog.memory.consolidation_hours.isEnabled() is False
    assert dialog.memory.consolidation_hours.value() == 6, "灰掉不等于清掉"
    assert dialog.memory.current().consolidation_interval_s == 6 * 3600, "灰着也照样读得出来"

    dialog.memory.consolidation_box.setChecked(True)

    assert dialog.memory.consolidation_hours.isEnabled() is True
    assert dialog.memory.consolidation_hours.value() == 6, "重新勾上时那个数还在"


def test_a_switch_that_opens_off_greys_the_interval_immediately(qapp, data_root) -> None:
    """``set_state`` 填进来的是"关"时，间隔框**一开局就该是灰的**。

    漏了这一步的症状很具体：窗口开出来，框没勾、而数字看着能改，用户去改它——改了
    也没用（那一轮读的是勾选框）。控件长成什么样是一条**独立的**陈述，它得自己自洽，
    不能靠"用户随手按一下之后才会被修正"。
    """
    assert (
        _memory_dialog(consolidation_enabled=False).memory.consolidation_hours.isEnabled() is False
    )


def test_the_consolidate_button_asks_and_then_waits(qapp, data_root) -> None:
    """「立即巩固」按下去：**先禁用自己再发信号**（同「重建联想」）。

    巩固要跑几秒到几分钟，而按一次是一轮真的改库。不禁用的话用户能连按——这里比
    重建多一重代价：连着两轮会各拍一份整库快照，而那是几百 MB 的写。
    """
    dialog = _memory_dialog()
    seen: list[int] = []
    dialog.consolidation_requested.connect(lambda: seen.append(1))

    dialog.memory.consolidate_button.click()

    assert seen == [1], "按一次发一次，发两次就是排出了两轮"
    assert dialog.memory.consolidate_button.isEnabled() is False
    assert "正在巩固" in dialog.memory.consolidate_note.text()


def test_the_consolidate_button_comes_back_when_the_result_lands(qapp, data_root) -> None:
    """结果回来就把按钮放开，并且**只有一句说法**：那一行状态。

    按钮旁边不复述结果——复述会出两个说法（按钮上一个、状态行一个），而它们迟早会
    在某条分支上分叉。所以 ``busy=False`` 只做一件事：把按钮放开。
    """
    dialog = _memory_dialog()
    dialog.memory.consolidate_button.click()

    dialog.set_consolidate_busy(False)
    dialog.set_consolidate_note("巩固：回放 4 条，加固 2 条。")

    assert dialog.memory.consolidate_button.isEnabled() is True
    assert dialog.memory.consolidate_note.text().startswith("巩固：")
    assert "正在巩固" not in dialog.memory.consolidate_note.text()


def test_turning_the_switch_off_does_not_close_the_manual_path(qapp, data_root) -> None:
    """关掉勾选框只是"不自动跑"，手动那条路**照旧开着**。

    手动有两个入口（托盘菜单、这一页的按钮），这条钉的是后者：关掉自动之后按钮还在、
    还按得动。哪天有人把勾选框的槽写成顺手把按钮也禁用掉，用户就只剩一个"巩固整个
    关了"的印象——而他想说的只是"别自己跑"。
    """
    dialog = _memory_dialog(consolidation_enabled=False)

    assert dialog.memory.consolidate_button.isEnabled() is True


def test_the_cost_of_the_automatic_run_is_written_on_the_page(qapp, data_root) -> None:
    """**它会真的改库、会拍快照、而周期是占位值**——三件都写在页面上。

    这是本仓唯一一处"无人值守的批量改库"。把"每天一次"当成实测结论写出来，等于让
    用户以为有人标定过；而它只是个占位（真实使用节奏下该多久一次没有依据）。同一段
    里还得写清快照的代价（整库复制 × 5 份），因为周期调短时被放大的是**磁盘**，
    不是 CPU——而用户看这一页时想的通常是"跑得勤不勤会不会卡"。
    """
    dialog = _memory_dialog()
    text = _hints(dialog.memory)

    assert "真的改库" in text, "无人值守地改库这件事必须在界面上说出来"
    assert "快照" in text and "5 份" in text, "代价是磁盘：整库复制、留 5 份"
    assert "占位" in text, "一天一次没有实测依据，不许写成结论"
    assert "手动" in text, "关掉自动不等于关掉巩固，手动那条路要指出来"


def test_the_categories_line_up_with_the_pages(qapp, data_root) -> None:
    """分类名与页**按同一个顺序对齐**，按下标一一对应。

    它们是两个列表（``categories`` 的条目与 ``pages`` 的 widget），加一个页而忘了
    加标签（或者反过来）时位移是静默的：点「人设」出来的是后端页，而没有任何一处
    报错。所以长度与顺序都要钉住，而不只是"人设这个词出现过"。
    """
    dialog = _dialog()
    labels = [dialog.categories.item(i).text() for i in range(dialog.categories.count())]

    assert labels == ["外观", "人设", "后端 / API", "工具", "记忆", "提醒", "数据"]
    assert dialog.pages.count() == len(labels)
    for row, page in enumerate(
        (
            dialog.appearance,
            dialog.persona,
            dialog.backend,
            dialog.tools,
            dialog.memory,
            dialog.reminders,
            dialog.data,
        )
    ):
        dialog.categories.setCurrentRow(row)
        assert dialog.pages.currentWidget() is page


def test_set_persona_fills_both_boxes(qapp, data_root) -> None:
    """两段正文都来自主窗口回填，而且**内置那段是只读的**。

    用 QLabel 而不是只读文本框：``setReadOnly(True)`` 的框看起来与能改的那个一模
    一样，用户会先点进去、发现打不了字，才知道它是死的。代价是"能选中复制"要自己
    打开——默认的 ``NoTextInteraction`` 连选都选不了，而"这段我要抄一下"很自然。
    """
    dialog = _dialog()

    dialog.set_persona("终端内置的那三句。", "说话简短一点。")

    builtin = dialog.persona.builtin_view
    assert builtin.text() == "终端内置的那三句。"
    assert not hasattr(builtin, "setPlainText"), "内置那段不该是能改的控件"
    assert builtin.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert dialog.persona.user_edit.toPlainText() == "说话简短一点。"


def test_save_is_grey_until_the_text_actually_changes(qapp, data_root) -> None:
    """**置灰的判据是"与库里那份不一样"**，不是"框里有没有字"。

    少了"未改动"这一半的话，用户每开一次设置窗口点一次保存都会白写一次库——而
    ``set_user_persona`` 会因此丢掉一条好端端的向量、再让后台重算一遍。
    """
    dialog = _dialog()
    dialog.set_persona("内置。", "说话简短。")
    page = dialog.persona

    assert not page.save_button.isEnabled(), "刚回填完、没人改过，保存该是灰的"
    assert page.save_button.toolTip()

    page.user_edit.setPlainText("说话简短，另外叫我老离。")
    assert page.save_button.isEnabled()
    assert page.user_edit.toPlainText().startswith("说话简短")


def test_an_empty_box_cannot_be_saved_and_says_what_to_do_instead(qapp, data_root) -> None:
    """空框时「保存」灰着，而且 tooltip 要把人指到「删掉」上。

    这一页上"不写"与"删掉"是同一件事，但**入口只有一个**——空着点保存是"没有动作"，
    不是"删除"。灰色按钮不写理由在本仓是要被修的毛病（见记忆窗口 ``_update_actions``
    那段注释）。
    """
    dialog = _dialog()
    dialog.set_persona("内置。", "")
    page = dialog.persona

    assert not page.save_button.isEnabled()
    assert "删掉" in page.save_button.toolTip()


def test_clear_is_grey_when_nothing_was_ever_written(qapp, data_root) -> None:
    dialog = _dialog()
    dialog.set_persona("内置。", "")

    assert not dialog.persona.clear_button.isEnabled()
    assert dialog.persona.clear_button.toolTip()

    dialog.set_persona("内置。", "我写过一句。")
    assert dialog.persona.clear_button.isEnabled()


def test_the_page_emits_the_stripped_text(qapp, data_root) -> None:
    dialog = _dialog()
    dialog.set_persona("内置。", "")
    seen: list[str] = []
    dialog.persona.save_requested.connect(seen.append)

    dialog.persona.user_edit.setPlainText("  说话简短。 \n")
    dialog.persona.save_button.click()

    assert seen == ["说话简短。"], "框里的首尾空白不该进库"


def test_the_window_forwards_both_persona_signals(qapp, data_root) -> None:
    """页面自己的两个信号要**转出去**。

    这一层转发单独验，是因为 ``SettingsDialog`` 少接一根线时页面照样工作，只有主
    窗口收不到——而那时症状是"点了保存没反应"，看不出断在哪一层。
    """
    dialog = _dialog()
    dialog.set_persona("内置。", "说话简短。")
    saved: list[str] = []
    cleared: list[int] = []
    dialog.persona_save_requested.connect(saved.append)
    dialog.persona_clear_requested.connect(lambda: cleared.append(1))

    dialog.persona.user_edit.setPlainText("改了。")
    dialog.persona.save_button.click()
    dialog.persona.clear_button.click()

    assert saved == ["改了。"]
    assert cleared == [1]


def test_the_persona_note_lands_on_that_page(qapp, data_root) -> None:
    dialog = _dialog()
    dialog.set_persona_note("已写入记忆库。")
    assert dialog.persona.note.text() == "已写入记忆库。"
    # 回填要顺手把上一次的结果擦掉，否则换一次打开还挂着上一轮的"没写进去"。
    dialog.set_persona("内置。", "说话简短。")
    assert dialog.persona.note.text() == ""


def test_the_page_says_revert_does_not_undo_the_persona(qapp, data_root) -> None:
    """**这一页与外观页唯一的行为差异必须写在界面上。**

    外观是实时生效 + 「撤销修改」能退回去；人设是点保存才写、而且撤销**碰不到**它。
    不写的话，用户会理所当然地以为关窗或点撤销能退回去——而那时库里已经改了。
    """
    dialog = _dialog()
    joined = "\n".join(label.text() for label in dialog.persona.findChildren(QLabel))

    assert "撤销" in joined and "不会" in joined
    assert "立刻生效" in joined


# ── 工具页：可读目录 ────────────────────────────────────────────────────────


def _tool_page(roots: tuple[Path, ...] = ()) -> ToolPage:
    page = ToolPage()
    page.set_state(ToolSettings(read_roots=roots))
    return page


def _watch_roots(page: ToolPage) -> list[tuple[Path, ...]]:
    seen: list[tuple[Path, ...]] = []
    page.roots_changed.connect(seen.append)
    return seen


def _dialog_with_roots(roots: tuple[Path, ...]) -> SettingsDialog:
    """带可读目录开一次窗口。``_dialog()`` 那份不给 ``tools``，走构造器的默认值
    （空）——两条路都要有人走，否则"构造器漏传"这一类错就没人挡。"""
    theme = Theme()
    return SettingsDialog(
        None,
        theme=theme,
        resolved=resolve(theme, "dark"),
        backend=BackendSettings(),
        tools=ToolSettings(read_roots=roots),
    )


def test_adding_a_directory_puts_it_in_the_list_and_sends_it_out(
    qapp, monkeypatch, tmp_path
) -> None:
    """加一条目录 = 立刻出去一次完整的新列表。

    传的是**整份**列表而不是"加了哪一条"：与外观页那条纪律同源——只发增量的话，
    接收方得自己拼当前状态，于是状态有了两个家。
    """
    page = _tool_page()
    seen = _watch_roots(page)
    wanted = tmp_path / "笔记"
    monkeypatch.setattr(page, "_ask_for_directory", lambda _start: str(wanted))

    page.add_button.click()

    assert page.roots() == (wanted,)
    assert seen == [(wanted,)]


def test_the_same_directory_twice_is_refused_with_a_reason(qapp, monkeypatch, tmp_path) -> None:
    """重复添加**说一句话**，不静默去重也不加第二条。

    静默的话，"我明明又加了一次"与"它没生效"在界面上分不开。
    """
    wanted = tmp_path / "笔记"
    page = _tool_page((wanted,))
    seen = _watch_roots(page)
    monkeypatch.setattr(page, "_ask_for_directory", lambda _start: str(wanted))

    page.add_button.click()

    assert page.roots() == (wanted,)
    assert seen == [], "列表没变，不该发信号"
    assert page.note.text()


def test_cancelling_the_picker_is_not_a_change(qapp, monkeypatch) -> None:
    """选择框里按取消返回空串——**那不是一次改动**。"""
    page = _tool_page()
    seen = _watch_roots(page)
    monkeypatch.setattr(page, "_ask_for_directory", lambda _start: "")

    page.add_button.click()

    assert page.roots() == ()
    assert seen == []
    assert page.note.text() == ""


def test_the_picker_starts_from_the_last_added_directory(qapp, monkeypatch, tmp_path) -> None:
    """起点是**最后加的那一条**（多半与下一条相邻）；一条都没有才回家目录。

    每次都从家目录开始的话，资料放在 D 盘的人每加一条都要在对话框里走一遍。
    """
    first = tmp_path / "笔记"
    page = _tool_page((first,))
    starts: list[str] = []
    monkeypatch.setattr(page, "_ask_for_directory", lambda start: starts.append(start) or "")

    page.add_button.click()
    assert starts == [str(first)]

    empty = _tool_page()
    monkeypatch.setattr(empty, "_ask_for_directory", lambda start: starts.append(start) or "")
    empty.add_button.click()
    assert starts[-1] == str(Path.home())


def test_removing_takes_only_the_selected_one_out(qapp, tmp_path) -> None:
    first, second = tmp_path / "甲", tmp_path / "乙"
    page = _tool_page((first, second))
    seen = _watch_roots(page)

    page.roots_view.setCurrentRow(0)
    page.remove_button.click()

    assert page.roots() == (second,)
    assert seen == [(second,)]


def test_remove_is_grey_until_a_row_is_selected(qapp, tmp_path) -> None:
    """灰按钮带理由（本仓的既有毛病清单里有一条"灰色按钮不写理由"）。"""
    page = _tool_page((tmp_path / "甲",))
    assert not page.remove_button.isEnabled()
    assert page.remove_button.toolTip()

    page.roots_view.setCurrentRow(0)
    assert page.remove_button.isEnabled()


@pytest.mark.parametrize("wide", [Path("C:/"), Path.home()])
def test_a_wide_directory_says_so(qapp, wide: Path) -> None:
    """**驱动器根与家目录都要提醒一句。**

    这两条的共同点是"看起来很小、其实很大"：列表里它们各占一行字，看不出分量。
    只提醒不拒绝——真要把整个盘交出去的人应当能交出去。
    """
    page = _tool_page((wide,))

    assert str(wide) in page.wide_warning.text()
    assert "整个盘" in page.wide_warning.text()


def test_an_ordinary_directory_says_nothing(qapp) -> None:
    """家目录**下面**的一层是正常写法，不该跟着一起报警——否则那句提醒会变成
    永远挂着的一条，然后就没人读了。"""
    page = _tool_page((Path.home() / "notes",))

    assert page.wide_warning.text() == ""


def test_the_warning_goes_away_when_the_wide_directory_is_removed(
    qapp, monkeypatch, tmp_path
) -> None:
    """提醒跟着列表走，不是一次性弹出来的。"""
    page = _tool_page((Path.home(),))
    assert page.wide_warning.text()

    page.roots_view.setCurrentRow(0)
    page.remove_button.click()

    assert page.wide_warning.text() == ""


def test_set_state_shows_what_is_stored_without_a_word(qapp) -> None:
    """**打开窗口时陈述现状，一句话都不发。**

    发了的话，主窗口会拿它去存一遍——用户只是打开了设置窗口，什么都没改。
    """
    page = _tool_page((Path.home() / "notes",))
    seen = _watch_roots(page)

    page.set_state(ToolSettings(read_roots=(Path.home() / "工作",)))

    assert page.roots() == (Path.home() / "工作",)
    assert seen == []


# ── 「撤销修改」也收得回授权 ────────────────────────────────────────────────


def test_revert_puts_the_directories_back(qapp, data_root, monkeypatch, tmp_path) -> None:
    """**这一条是工具页存在的理由。** 按钮上写着"回到打开这个窗口时的样子"，而
    加一条目录是一次授权——撤销收不回去的话，那句话就是假的，而那时模型已经能
    读那个目录了。"""
    opened = tmp_path / "笔记"
    dialog = _dialog_with_roots((opened,))
    seen = _watch_roots(dialog.tools)
    monkeypatch.setattr(dialog.tools, "_ask_for_directory", lambda _start: str(tmp_path / "别的"))
    dialog.tools.add_button.click()
    assert len(dialog.tools.roots()) == 2, "还没加进去，这个用例什么都没验到"
    seen.clear()

    dialog.revert_button.click()

    assert dialog.tools.roots() == (opened,)
    assert seen == [(opened,)]


def test_revert_says_nothing_when_the_list_was_not_touched(qapp, data_root) -> None:
    """没动过就不发信号：不然每按一次撤销都会往设置里写一遍同样的东西，而"这一页
    本来就没动过"是绝大多数情况。"""
    dialog = _dialog_with_roots((Path.home() / "笔记",))
    seen = _watch_roots(dialog.tools)

    dialog.revert_button.click()

    assert seen == []


def test_reopening_moves_the_directory_snapshot(qapp, data_root, monkeypatch, tmp_path) -> None:
    """重开时快照换成**这一次**打开时的目录（与主题那条同理）。

    不换的话，撤销会把用户带回几十分钟前那一份授权——而那一份里可能有他已经移除
    的目录。
    """
    old, new = tmp_path / "旧的", tmp_path / "新的"
    dialog = _dialog_with_roots((old,))
    theme = Theme()
    dialog.set_state(
        theme,
        resolve(theme, "dark"),
        BackendSettings(),
        tools=ToolSettings(read_roots=(new,)),
        memory=MemorySettings(),
        reminders=ReminderSettings(),
    )
    monkeypatch.setattr(dialog.tools, "_ask_for_directory", lambda _start: str(tmp_path / "临时"))
    dialog.tools.add_button.click()
    seen = _watch_roots(dialog.tools)

    dialog.revert_button.click()

    assert dialog.tools.roots() == (new,)
    assert seen == [(new,)]


# ── 分类列的行几何 ──────────────────────────────────────────────────────────


def test_category_rows_advance_by_the_height_the_code_gave_them(qapp, data_root) -> None:
    """分类列的每一行**按代码给的高度前进**，不按样式表的 padding 前进。

    这里守的是一个真实的坏画面：一条 32 像素的高亮带压在只有 16 像素行距的两行上，
    于是选中项的字与上下两行的字叠在一起（用户报的就是这个，看起来像字体坏了）。
    行高现在由 ``_CATEGORY_PAD_Y`` 加字体度量算出来，样式表只管横向内边距——
    **纵向两边各算一次**正是那个画面可能的来路。

    断言"相邻两行的顶差 == 我们给的高度"，而不是"高度 == 32"：字体随 DPI 变，
    写死一个数就会在高 DPI 的机器上红，而红的理由跟这条想守的东西没关系。
    """
    dialog = _dialog()
    categories = dialog.categories
    dialog.resize(720, 560)
    qapp.processEvents()
    expected = categories.fontMetrics().height() + 2 * settings_dialog._CATEGORY_PAD_Y

    rects = [categories.visualItemRect(categories.item(i)) for i in range(categories.count())]
    tops = [rect.top() for rect in rects]

    assert [rect.height() for rect in rects] == [expected] * len(rects)
    assert tops == sorted(tops), "行必须从上往下排"
    for above, below in zip(tops, tops[1:], strict=False):
        assert below - above == expected, "行距必须等于给出的行高，否则行与行会叠上"


def test_the_row_height_clears_the_tallest_label(qapp, data_root) -> None:
    """行高得容得下最长的那个标签，否则文字会被竖着裁掉。

    与上一条分开：上一条问"布局有没有听代码的"，这一条问"代码给的那个数对不对"。
    """
    dialog = _dialog()
    categories = dialog.categories
    expected = categories.fontMetrics().height() + 2 * settings_dialog._CATEGORY_PAD_Y

    widest = max(
        categories.fontMetrics().horizontalAdvance(categories.item(i).text())
        for i in range(categories.count())
    )

    assert expected > categories.fontMetrics().height()
    assert widest < dialog.categories.width(), "最长的标签要放得下，不然会被省略号截掉"


# ── 后端页：云端 ────────────────────────────────────────────────────────────


def test_switching_to_the_cloud_hides_the_local_group(qapp, data_root) -> None:
    """**两组控件不能同时可填。**

    并排摆着的话，"Ollama 地址"与"云端服务地址"都会亮着，而它们同时生效是不可能的
    ——用户会以为两个都要填，于是把 Ollama 地址也当成云端端点的一部分。

    ``isVisibleTo`` 而不是 ``isVisible``：窗口没 ``show()`` 时后者对**每一个**控件
    都是 ``False``，用它的话这条断言在离屏下恒真。
    """
    page = BackendPage()

    assert page.local_group.isVisibleTo(page)
    assert not page.cloud_group.isVisibleTo(page)

    page.provider_combo.setCurrentIndex(page.provider_combo.findData(PROVIDER_CLOUD))

    assert not page.local_group.isVisibleTo(page)
    assert page.cloud_group.isVisibleTo(page)


def test_switching_the_backend_type_takes_effect_immediately(qapp, data_root) -> None:
    """换类型当场发一次应用。

    不这样的话，"类型"这个下拉框就只是改了一个没人读的字段——用户切到云端、界面
    上两组控件也换了，而实际请求还是发往本机。与 ``allow_remote`` 勾选框同一条形状。
    """
    page = BackendPage()
    seen: list[BackendSettings] = []
    page.apply_requested.connect(seen.append)

    page.provider_combo.setCurrentIndex(page.provider_combo.findData(PROVIDER_CLOUD))

    assert seen and seen[-1].provider == PROVIDER_CLOUD


def test_the_cloud_warning_names_what_actually_leaves_the_machine(qapp, data_root) -> None:
    """**措辞里必须点到"记忆"。**

    ``AssemblyWorker`` 会把召回的几条记忆注入 prompt，它们跟对话一起出去。只说
    "对话内容"是不完整的——那会让人以为记忆库留在本机就等于什么都没出去。
    同时要写清"什么**不**出去"，否则代价看起来比实际大。
    """
    page = BackendPage()

    text = page.cloud_warning.text()
    assert "记忆" in text
    assert "对话" in text
    assert "本机" in text, "要写清哪些东西仍然不出设备"
    assert not page.cloud_warning.isHidden(), "代价要在切过去之前就能读到"


def test_the_cloud_warning_is_armed_once_the_cloud_is_selected(qapp, data_root) -> None:
    """切到云端之后那句话变醒目：它说的是一件**已经发生**的事。

    同时本机那句要退回去——不然两句都红着，看起来像两个警告。
    """
    page = BackendPage()
    page.allow_remote_box.setChecked(True)
    assert page.remote_warning.property("armed") == "yes"

    page.provider_combo.setCurrentIndex(page.provider_combo.findData(PROVIDER_CLOUD))

    assert page.cloud_warning.property("armed") == "yes"
    assert page.remote_warning.property("armed") == "no"


def test_the_key_is_hidden_until_asked_for(qapp, data_root) -> None:
    """**默认遮起来。** key 会出现在截图、录屏与共享屏幕里，而"要不要显示"是一个
    每次都要重新做的决定。"""
    from PySide6.QtWidgets import QLineEdit

    page = BackendPage()
    assert page.api_key_edit.echoMode() == QLineEdit.EchoMode.Password

    page.show_key_box.setChecked(True)
    assert page.api_key_edit.echoMode() == QLineEdit.EchoMode.Normal

    page.show_key_box.setChecked(False)
    assert page.api_key_edit.echoMode() == QLineEdit.EchoMode.Password


def test_switching_away_and_back_keeps_both_sides(qapp, data_root) -> None:
    """**切类型不许清掉另一边的值。**

    ``current()`` 两个地址都带出来，正是为了这一条。漏掉一个的话，用户切到云端再
    切回来，输入框里躺着的要么是空的、要么是默认值——两者都会让他以为配置丢了。
    """
    page = BackendPage()
    page.host_edit.setText("http://192.168.1.9:11434")
    page.allow_remote_box.setChecked(True)
    page.provider_combo.setCurrentIndex(page.provider_combo.findData(PROVIDER_CLOUD))
    page.cloud_url_edit.setText("https://api.deepseek.com/v1")
    page.api_key_edit.setText("sk-abc")

    page.provider_combo.setCurrentIndex(page.provider_combo.findData(PROVIDER_OLLAMA))
    back = page.current()

    assert back.host == "http://192.168.1.9:11434"
    assert back.allow_remote is True
    assert back.cloud_base_url == "https://api.deepseek.com/v1", "切换把 cloud_url 清掉了"
    assert back.api_key == "sk-abc", "切换把 key 清掉了"


def test_an_empty_cloud_url_falls_back_to_the_default(qapp, data_root) -> None:
    """与 host 那条同一个理由：空字符串会变成一个没有 scheme 的 URL，
    报出来的错与"我没填地址"毫无关系。"""
    page = BackendPage()
    page.cloud_url_edit.setText("   ")
    assert page.current().cloud_base_url == DEFAULT_CLOUD_BASE_URL


def test_opening_on_the_cloud_does_not_emit_a_spurious_apply(qapp, data_root) -> None:
    """**打开窗口不是一次修改。**

    ``set_state`` 会把下拉框拨到设置里那一项——拨动会发 ``currentIndexChanged``，
    若不等它，每次打开设置窗口都会白跑一次模型探测（还会把下拉框清空重建）。
    """
    page = BackendPage()
    seen: list[BackendSettings] = []
    page.apply_requested.connect(seen.append)

    page.set_state(BackendSettings(provider=PROVIDER_CLOUD, api_key="sk-x"))

    assert seen == []
    assert page.local_group.isVisibleTo(page) is False


# ── 连接测试：两条后端各探各的 ──────────────────────────────────────────────


def test_the_probe_asks_ollama_for_a_local_backend(qapp, data_root, monkeypatch) -> None:
    asked: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        settings_dialog,
        "probe",
        lambda host, allow_remote=False: asked.append((host, allow_remote)) or "0.40.2",
    )
    got: list[str] = []
    worker = settings_dialog._ProbeWorker(BackendSettings(host="http://127.0.0.1:11434"))
    worker.done.connect(got.append)

    worker.run()

    assert asked == [("http://127.0.0.1:11434", False)]
    assert got == ["连上了 · Ollama 0.40.2"]


def test_the_probe_asks_the_cloud_endpoint_and_the_key(qapp, data_root, monkeypatch) -> None:
    """**不分支的话，云端配置下会去探 ``127.0.0.1:11434``**——一个用户根本没配过的
    地址，于是「测试连接」在一个配得好好的云端后端的窗口里说"连不上"。"""
    asked: list[tuple[str, str]] = []

    def fake_probe(base_url: str, *, api_key: str = "") -> tuple[bool, str]:
        asked.append((base_url, api_key))
        return True, "连上了，列出 3 个模型"

    monkeypatch.setattr(settings_dialog, "openai_compat", SimpleNamespace(probe=fake_probe))
    got: list[str] = []
    worker = settings_dialog._ProbeWorker(
        BackendSettings(
            provider=PROVIDER_CLOUD, cloud_base_url="https://api.deepseek.com/v1", api_key="sk-x"
        )
    )
    worker.done.connect(got.append)

    worker.run()

    assert asked == [("https://api.deepseek.com/v1", "sk-x")]
    assert got == ["连上了，列出 3 个模型"]


def test_a_cloud_probe_failure_shows_its_reason(qapp, data_root, monkeypatch) -> None:
    """401 那句原因要原样显示出来。压成一句"连不上"会让人去查地址，而地址没问题。"""

    def refuse(_base_url: str, *, api_key: str = "") -> tuple[bool, str]:
        return False, "HTTP 401：API key 无效"

    monkeypatch.setattr(settings_dialog, "openai_compat", SimpleNamespace(probe=refuse))
    bad: list[str] = []
    worker = settings_dialog._ProbeWorker(BackendSettings(provider=PROVIDER_CLOUD))
    worker.failed.connect(bad.append)

    worker.run()

    assert bad == ["HTTP 401：API key 无效"]


# ── 数据页：位置显示、灰按钮的理由、以及"它自己什么都不做" ──────────────────


def test_the_data_page_shows_which_folder_is_in_use(qapp, data_root) -> None:
    """路径与"它听谁的"两样都要显示。

    只显示路径是不够的：默认位置与"你在设置里选的"会给出**同一个路径字符串**，
    分不开的话用户没法判断那两个按钮该不该灰。
    """
    page = settings_dialog.DataPage()
    page.set_state(Path(r"C:\some\where\axo"), "stored")

    assert page.path_label.text() == r"C:\some\where\axo"
    assert page.source_label.text() == settings_dialog.data_root_description("stored")
    assert page.change_button.isEnabled() is True
    assert page.restore_button.isEnabled() is True


def test_the_data_page_locks_the_buttons_when_the_environment_decides(qapp, data_root) -> None:
    """环境变量在生效时两个按钮是灰的，**而且旁边写着为什么**。

    灰按钮不带解释，用户看到的就是一个"点了没反应"的界面——而那种症状指不到环境
    变量上去。
    """
    page = settings_dialog.DataPage()
    page.set_state(Path(r"C:\some\where\axo"), "env")

    assert page.change_button.isEnabled() is False
    assert page.restore_button.isEnabled() is False
    assert "AXO_DATA" in page.source_label.text()


def test_changing_the_location_only_asks_it_does_not_move_anything(
    qapp, data_root, monkeypatch
) -> None:
    """挑完目录**只发一个信号**。真正搬不搬是主窗口的事。

    这条钉的是本文件的既有约定（对话框不做 I/O）：它要是自己去复制，就会在"校验
    没通过"或"用户又点了取消"的时候留下半个状态，而那些判断住在主窗口那条链上。
    """
    page = settings_dialog.DataPage()
    page.set_state(Path(r"C:\some\where\axo"), "default")
    monkeypatch.setattr(page, "_ask_for_directory", lambda: r"D:\AxoData")
    asked: list[str] = []
    page.change_location_requested.connect(asked.append)

    page.change_button.click()

    assert asked == [r"D:\AxoData"]


def test_cancelling_the_folder_dialog_says_nothing(qapp, data_root, monkeypatch) -> None:
    """用户按了取消（文件对话框返回空串）→ 什么都不发。"""
    page = settings_dialog.DataPage()
    page.set_state(Path(r"C:\some\where\axo"), "default")
    monkeypatch.setattr(page, "_ask_for_directory", lambda: "")
    asked: list[str] = []
    page.change_location_requested.connect(asked.append)

    page.change_button.click()

    assert asked == []


def test_the_data_page_note_is_writable_from_outside(qapp, data_root) -> None:
    """搬迁那条链的反馈落在这张 note 上（同人设页），不落进对话流。"""
    page = settings_dialog.DataPage()
    page.set_state(Path(r"C:\some\where\axo"), "default")

    page.set_note("已经安排好了")

    assert "已经安排好了" in page.note_label.text()


# ── 提醒页：两个开关 ────────────────────────────────────────────────────────
#
# 这一页只有两个勾选框，而它们各自对着一条"关掉之后提醒照响"的通道。这里量三件
# 别处量不到的事：**开窗时的初值**（三个默认值里的第三个）、**陈述现状不发信号**、
# 以及**撤销回得去**。


def _reminder_page(settings: ReminderSettings | None = None) -> tuple[ReminderPage, list]:
    page = ReminderPage()
    seen: list[ReminderSettings] = []
    page.changed.connect(seen.append)
    if settings is not None:
        page.set_state(settings)
    return page, seen


def test_the_reminder_boxes_start_ticked(qapp: Any) -> None:
    """**默认都开**，而且是"这个控件自己"的初值。

    这一条是三个默认值里的第三个（另两个在 ``ReminderSettings`` 的字段与
    ``load_reminders`` 的 ``, True``）。少写它——比如照抄记忆页写成
    ``QCheckBox("…")`` 之后忘了 ``setChecked``——的症状只有一条路径会露出来：
    新装好的程序里打开设置，那个框是**没勾**的，于是用户以为默认是关的；
    而库里、注册表里、程序的行为里，它一直是开的。
    """
    page, _seen = _reminder_page()

    assert page.current() == ReminderSettings(wake_on_alert=True, speak_on_alert=True)


def test_opening_the_reminder_page_says_nothing(qapp: Any) -> None:
    """``set_state`` 是**陈述现状**，不是做修改（同外观页那条，理由一字不差：
    不挡的话，打开设置这一下就会往设置里重写一遍同样的值）。

    这里比外观页多一层必要：这两个值是**主窗口每次到点现读**的，`set_state` 要是
    漏了 `_syncing`，一次开窗就等于"用户又把开关设了一遍"——平时无害，但它会让
    "改过开关"与"只是打开看了一眼"在日志与行为上再也分不开。
    """
    page, seen = _reminder_page()

    page.set_state(ReminderSettings(wake_on_alert=False, speak_on_alert=True))

    assert seen == []


def test_unticking_one_box_emits_the_whole_setting(qapp: Any) -> None:
    """勾掉一个，发出来的仍然是**完整的一份**（同外观页：发增量的实现会让接收方
    把没提到的那个字段重置回默认值）。"""
    page, seen = _reminder_page(ReminderSettings())

    page.speak_box.setChecked(False)

    assert seen == [ReminderSettings(wake_on_alert=True, speak_on_alert=False)]


def test_revert_puts_the_reminder_boxes_back(qapp: Any) -> None:
    """「撤销修改」回到**打开窗口时**那一份，而且变了才发信号。

    "变了才发"不是为了省事：主窗口收到一次就是一次落盘，而"我什么都没改、只是点了
    撤销"不该在设置文件里留下一次写入。
    """
    page, seen = _reminder_page(ReminderSettings())

    page.wake_box.setChecked(False)
    page.revert()

    assert page.current() == ReminderSettings()
    assert seen[-1] == ReminderSettings()
    # 再按一次：什么都没变，于是什么都不发。
    before = len(seen)
    page.revert()
    assert len(seen) == before


# ── 提醒页：开机自启那一格 ──────────────────────────────────────────────────
#
# 这一格与前两格不同：它**可能在本次运行里根本设不了**（源码检出、或者不是 Windows），
# 而那时界面的责任是"说清为什么"，不是"勾上之后没反应"。


def test_the_autostart_box_is_greyed_out_when_it_cannot_work(qapp: Any, data_root: Path) -> None:
    """设不了的时候：框灰着，命令行那一行**写的正是原因**。

    这一版跑的是源码（不是打包出来的 exe），``sys.executable`` 是 python.exe——写进去的话
    "开机自启"启动的是一个没有脚本可跑的 Python。所以那个框勾不上，而旁边那行把它说清楚。
    **"勾了却没生效"比"勾不上"坏得多。**
    """
    page, _seen = _reminder_page()

    assert page.autostart_box.isEnabled() is False
    assert page.command_label.text() == autostart.unavailable_reason()
    assert page.command_label.text(), "灰着却一个字都不说"


def test_the_autostart_box_shows_the_exact_command(qapp: Any, data_root: Path, monkeypatch) -> None:
    """能设的时候：命令行**逐字摆出来**。

    它是"用户看得见他同意了什么"的落点，也是排查"自启没生效"唯一要看的东西——命令行
    对了，问题就在别处。所以这里比的是那一整串，而不是"含 axo 这个词"。
    """
    exe = Path(r"C:\Program Files\Axoxo.exe")
    monkeypatch.setattr(autostart, "is_supported", lambda: True)
    monkeypatch.setattr(autostart, "current_exe", lambda: exe)
    page, seen = _reminder_page()

    assert page.autostart_box.isEnabled() is True
    assert page.command_label.text() == f"登录时会执行：{autostart.command_for(exe)}"

    page.autostart_box.setChecked(True)
    assert seen[-1].autostart is True


def test_a_greyed_autostart_box_keeps_the_stored_value(qapp: Any, data_root: Path) -> None:
    """设不了的时候，那一格**原样带着快照走**。

    不这样的话：源码检出下打开设置页、随手勾一下「由她说一句」，就会把打包版设好的自启
    **静默写成关**——而那个框本来就是灰的，用户完全看不出这件事发生过。
    """
    page, seen = _reminder_page(ReminderSettings(autostart=True))

    page.speak_box.setChecked(False)

    assert seen == [ReminderSettings(wake_on_alert=True, speak_on_alert=False, autostart=True)]


def test_the_autostart_note_has_its_own_line(qapp: Any, data_root: Path) -> None:
    """主窗口回填结果的那一行（"已设成开机自启" / "自启没能写成：…"）。

    与图状态、巩固那两行同一个形状：**成功也走这里**（空串 = 没事），复述两处就会出现
    两个说法，而它们迟早会在某条分支上分叉。
    """
    page, _seen = _reminder_page()

    page.set_note("自启没能写成：拒绝访问")
    assert page.autostart_note.text() == "自启没能写成：拒绝访问"
    page.set_note("")
    assert page.autostart_note.text() == ""
