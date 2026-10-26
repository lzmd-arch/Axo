"""语言服务器对话框：能构造、有体检/预览/安装三个入口。"""

from __future__ import annotations

import pytest

from axo.ui.language_dialog import LanguageServersDialog

pytestmark = pytest.mark.gui


def test_dialog_exposes_doctor_plan_and_install(qapp) -> None:
    dialog = LanguageServersDialog()
    try:
        assert dialog.windowTitle() == "语言服务器"
        assert dialog.check_button.text() == "检查现状"
        assert dialog.plan_button.text() == "预览安装计划"
        assert dialog.install_button.text() == "自动安装…"
    finally:
        dialog.close()
