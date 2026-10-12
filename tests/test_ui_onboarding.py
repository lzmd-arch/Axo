"""首启向导：步骤顺序、人设那一步的取舍、以及"什么时候才算走过了"。

这个窗口有两处容易写错而且**错了不会报错**的地方：

* **顺序表是按需拼出来的**（已经写过人设的人不该被再问一遍），所以"有哪些步"不是
  一个能一眼看出来的常量——钉住它。
* **什么时候算走过了。** 只有「完成」与「跳过」算，叉掉窗口不算。反过来写的话，
  用户随手关掉、程序就当他看过了；而正着写错（叉掉也算）会让"我不想看"这个意思
  根本没有表达方式。所以这里逐条验。

不构造 `MainWindow`（gui 腿的约定，见 `tests/test_ui_open_settings.py` 的模块文档）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from axo.ui import onboarding as onboarding_module
from axo.ui.onboarding import OnboardingWizard
from axo.ui.settings_dialog import BackendPage, DataPage
from axo.ui.settings_store import BackendSettings

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    """建 app，并**给它装一个应用级图标**。

    那个图标是向导要复制的东西（见 :func:`test_it_has_a_window_icon_...`），而且真
    程序里它确实在启动时就装好了（`_apply_theme` → `_install_window_icon`）。所以
    在这里装一次比在用例中途装了再还原更接近真实——**中途还原那一下会崩**：把一个
    已经发出去的 `QIcon` 换回去，收尾时 Qt 会碰它。
    """
    from PySide6.QtGui import QIcon, QPixmap

    app = QApplication.instance() or QApplication([])
    pixmap = QPixmap(16, 16)
    pixmap.fill()
    app.setWindowIcon(QIcon(pixmap))
    yield app
    app.setStyleSheet("")


#: 这个文件建出来的向导。**退出前一定要收掉**：没有父对象的顶层窗口留到解释器收尾，
#: Qt 会在那儿硬杀进程（退出码 `0xC0000409`，没有回溯）——而 pytest 已经报过
#: "N passed" 了，所以那个退出码看起来像"测试跑完了但命令失败了"。这个坑踩过一次。
_OPEN: list[OnboardingWizard] = []


@pytest.fixture(autouse=True)
def _close_wizards() -> Any:
    yield
    while _OPEN:
        wizard = _OPEN.pop()
        wizard.close()
        wizard.deleteLater()


def _wizard(*, show_persona_step: bool = True, tmp_path: Path | None = None) -> OnboardingWizard:
    wizard = OnboardingWizard(
        None,
        root=tmp_path or Path(r"C:\some\where\axo"),
        source="default",
        backend=BackendSettings(),
        builtin_persona="你是 Axo 的助手。",
        show_persona_step=show_persona_step,
    )
    _OPEN.append(wizard)
    return wizard


def test_the_step_order_is_stable(qapp: Any) -> None:
    """五步，顺序钉死。**每步的名字也在里面**——顺序对了而名字错了同样是一种错位。

    「欢迎」在第 0 步而不是省掉：这个程序里"记忆存在你自己那儿、不归模型"这件事
    是它全部的卖点，而首启是唯一一次用户会读到它的时机。
    """
    wizard = _wizard()

    assert wizard._order == ["welcome", "data", "persona", "backend", "done"]
    assert [onboarding_module._STEP_TITLES[name] for name in wizard._order] == [
        "欢迎",
        "记忆放在哪",
        "它怎么说话",
        "用哪个模型",
        "可以开始了",
    ]


def test_the_persona_step_is_skipped_when_one_already_exists(qapp: Any) -> None:
    """写过人设的人**不该被再问一遍**——页照样构造，只是不进顺序表。

    于是"加一页"的改动仍然只有一处（那张表），而 `set_persona` 那种回填调用也不
    需要先问一遍有没有它。
    """
    wizard = _wizard(show_persona_step=False)

    assert wizard._order == ["welcome", "data", "backend", "done"]
    assert wizard.step_label.text() == "第 1 步 / 共 4 步"


def test_only_finishing_or_skipping_marks_it_done(qapp: Any) -> None:
    """**叉掉窗口不算走过。** 这一条是这个窗口最重要的一个判断。

    没做完也没说不做不算一个决定；而"我不想看"必须有一个按得下去的答案——所以底部
    那个「跳过」是承重件，不是装饰。
    """
    finished: list[int] = []

    finishing = _wizard()
    finishing.onboarding_finished.connect(lambda: finished.append(1))
    while finishing._position < len(finishing._order) - 1:
        finishing.next_button.click()
    finishing.next_button.click()  # 最后一步的按钮是「完成」
    assert finished == [1]

    skipping = _wizard()
    skipping.onboarding_finished.connect(lambda: finished.append(2))
    skipping.skip_button.click()
    assert finished == [1, 2]

    closing = _wizard()
    closing.onboarding_finished.connect(lambda: finished.append(3))
    closing.close()
    assert finished == [1, 2], "叉掉窗口也被当成走过了"


def test_skipping_the_persona_step_still_counts_as_an_answer(qapp: Any) -> None:
    """「以后再说」发的是**空串**，不是"没发"。

    它是那个问题的答案（"这一次先不写"），所以主窗口那条链会顺着它把"本进程别再问
    人设"置上——而卡片因此不会紧接着再弹一次。反过来（什么都不发）的话，用户在向导
    里刚说完"以后再说"，一关掉向导就被同一张卡片再问一遍。
    """
    answers: list[str] = []
    wizard = _wizard()
    wizard.persona_answered.connect(answers.append)

    wizard.persona_later.click()

    assert answers == [""]


def test_the_save_button_is_dead_until_something_is_typed(qapp: Any) -> None:
    """空正文不能提交（同设置窗口人设页那条）。两条路各有各的名字：写点什么用
    「就这样」，什么都不写用「以后再说」。"""
    wizard = _wizard()

    assert wizard.persona_save.isEnabled() is False
    wizard.persona_edit.setPlainText("   ")
    assert wizard.persona_save.isEnabled() is False
    wizard.persona_edit.setPlainText("请你说话短一点")
    assert wizard.persona_save.isEnabled() is True
    wizard.persona_save.click()

    assert wizard.persona_later.isEnabled() is True


def test_it_reuses_the_settings_pages_instead_of_copying_them(qapp: Any) -> None:
    """第 1 步与第 3 步**就是设置窗口那两页**，不是照着抄的副本。

    抄一份的下场是两处行为各自漂：设置页加了一个选项而向导那份没有，用户在向导里
    看到的是一个旧版本的界面——而两处看起来都"能用"。
    """
    wizard = _wizard()

    assert isinstance(wizard.data, DataPage)
    assert isinstance(wizard.backend, BackendPage)


def test_it_forwards_the_pages_signals(qapp: Any) -> None:
    """两页的信号要**转发**出去，主窗口那四根线才接得上。

    本窗口自己不做搬迁、也不存后端设置（同设置窗口的约定）——它只是宿主。
    """
    wizard = _wizard()
    seen: list[tuple[str, object]] = []
    wizard.data_location_requested.connect(lambda path: seen.append(("data", path)))
    wizard.restore_data_location_requested.connect(lambda: seen.append(("restore", None)))
    wizard.backend_changed.connect(lambda settings: seen.append(("backend", settings)))

    wizard.data.change_location_requested.emit(r"D:\AxoData")
    wizard.data.restore_default_requested.emit()
    wizard.backend.apply_requested.emit(BackendSettings())

    assert [kind for kind, _ in seen] == ["data", "restore", "backend"]


def test_it_has_a_window_icon_because_the_packaging_probe_looks_at_it(qapp: Any) -> None:
    """**构造时就显式挂上图标。**

    向导一开就在 z 序最上面，而打包自检那一步取的是 `EnumWindows` 顺序里**第一个**
    非错误框的顶层窗口来问"窗口上有没有图标"——它取到的正会是向导。不挂的话自检会
    报"窗口上却没有图标"，而真正的原因是它看错了窗口。
    """
    from PySide6.QtGui import QGuiApplication

    wizard = _wizard()

    assert not QGuiApplication.windowIcon().isNull(), "夹具没装上应用级图标，这条就成了空转"
    assert not wizard.windowIcon().isNull()


def test_the_last_step_repeats_what_was_chosen(qapp: Any) -> None:
    """按下「完成」之前，用户该看见自己定了什么——否则那一步只是一句"可以开始了"。"""
    wizard = _wizard()
    wizard.set_persona("请你说话短一点")
    while wizard._position < len(wizard._order) - 1:
        wizard.next_button.click()

    assert "请你说话短一点" not in wizard.done_label.text()  # 复述的是"写没写"，不是正文
    assert "写了" in wizard.done_label.text()
    assert str(wizard.data.path_label.text()) in wizard.done_label.text()
