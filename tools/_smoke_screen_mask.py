"""在**真屏幕**上验一次遮罩：开一个品红窗口，截屏、涂掉它，再看那一块成了什么颜色。

计划书里原来把"截图的真实行为"归成**只能手验**的一件事（离屏下 `grabWindow(0)` 给的是一张
800×800 纯黑假图，见 `docs/agent.md` 那条实测）。这一条把它变成一条命令——判据是**自己量
出来的一个像素**，而不是"看着还行"。

用法::

    uv run python tools/_smoke_screen_mask.py     # 会有一个窗口在屏幕上闪一下（约 0.7 秒）

退出码：**0 = 遮罩盖上去了**（窗口中心取回来的是遮罩色）· **1 = 没盖上**（取回来的是品红）·
**2 = 环境不对**（截不到屏、或者没有真显示）。

**为什么值得留。** 2026-10-13：用户照着一张截图问"它是不是真看到了我的窗口"，而模型的描述里
有一句"Axo 的窗口开着，左边只有两条"——**那句话是编的**（图上那一块被涂掉了），但光看模型的
话判断不了。量一次比读十句话有用。
"""

import os
import sys

os.environ.pop("QT_QPA_PLATFORM", None)  # 要真平台，不要离屏
from PySide6.QtCore import QRect, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QWidget

from axo.ui.screen import grab, own_window_rects

app = QApplication([])
w = QWidget()
w.setWindowTitle("遮罩探针")
w.setStyleSheet("background: #ff00ff;")  # 品红：认得出，且不会被误当成遮罩色
w.setGeometry(300, 300, 420, 300)
w.show()


def check():
    global code
    screen = w.screen()
    frame = w.frameGeometry()
    print("屏幕几何（逻辑）：", screen.geometry())
    print("窗口几何（逻辑）：", frame, "| devicePixelRatio:", screen.devicePixelRatio())
    rects = own_window_rects(screen, QApplication.topLevelWidgets())
    print("收到的遮罩矩形：", rects)

    shot = grab(screen, exclude=rects)
    print("截图：", "ok" if shot.ok else shot.reason, shot.width, "×", shot.height)
    if not shot.ok:
        code = 1
        app.quit()
        return

    image = QImage.fromData(shot.png, "PNG")
    ratio = screen.devicePixelRatio()
    centre = QRect(
        round((frame.x() + frame.width() / 2) * ratio),
        round((frame.y() + frame.height() / 2) * ratio),
        1,
        1,
    )
    print("窗口中心在设备像素里的位置：", centre.x(), centre.y())
    sampled = image.pixelColor(centre.x(), centre.y())
    print("那一点的颜色：", sampled.name())
    print("判据：遮罩色是 #3f3f46；品红 #ff00ff 说明**没盖上**")
    code = 0 if sampled.name() == "#3f3f46" else 1
    print("结果：", "✅ 盖住了" if code == 0 else "❌ 没盖住")
    app.quit()


code = 2
QTimer.singleShot(700, check)
app.exec()
sys.exit(code)
