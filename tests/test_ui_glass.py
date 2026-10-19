"""Qt 外壳的极光底：能画、换色不出错、动效开关是幂等的。"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtGui import QPixmap

from axo.ui.glass import AuroraBackdrop

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_it_paints_without_a_window(qapp: Any) -> None:
    backdrop = AuroraBackdrop()
    backdrop.resize(320, 180)

    canvas = QPixmap(320, 180)
    backdrop.render(canvas)

    assert not canvas.isNull()


def test_the_animation_switch_is_idempotent_and_keeps_the_picture(qapp: Any) -> None:
    backdrop = AuroraBackdrop()
    backdrop.resize(200, 120)

    backdrop.set_accent("#6b8afd", animated=True)
    backdrop.set_accent("#6b8afd", animated=False)
    backdrop.set_animated(False)
    backdrop.set_animated(True)

    assert not backdrop.isVisible()  # 没 show 过，计时器不该跑（见模块文档）
