"""窗口组装冒烟测试：起真窗口、连真 Ollama、打一遍各处的状态。

为什么要有它：链路里有三段会在**不同时刻**失败，而它们对外都表现为"打字没反应"。
把它们的中间状态打出来，就不必靠猜：

1. QApplication 能否起来（Qt 平台插件 / 显卡驱动）
2. MainWindow 能否构造（控件连线、QSettings 读取）
3. 后台线程能否拉到模型清单（Ollama 在不在、模型名对不对）
4. 选中的模型能否构造出 :class:`OllamaChatBackend`（``/api/show`` 探询、
   ``num_ctx`` 能不能读到）
5. 网页对话区能否加载（QtWebEngine、CSP、qwebchannel.js 的路径）

它**不发对话请求**——那一段由 ``tools/smoke_backend.py`` 负责，两者互不重叠。

用法::

    uv run --extra app --extra llm python tools/_smoke_window.py

**必须有显示器，不能加 ``QT_QPA_PLATFORM=offscreen``。** 它构造 ``MainWindow``，
里面就有 QWebEngineView，而离屏下 QtWebEngine 一渲染把整个进程硬杀掉——没有异常、
没有回溯、连退出码都没有，看起来就像"脚本什么都没干"。与此相对，``smoke_backend.py``
不碰窗口，离屏跑没问题。

（编号从 1 跳到 3：下面五段失败面里第 2 段"MainWindow 能否构造"没有独立的输出行，
构造失败的话第 1 行就不会出现。）
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from axo.ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow()
    window.resize(1100, 760)
    window.show()
    print("[1] MainWindow 构造完成")

    def dump() -> None:
        combo = window.model_combo
        print(f"[3] 模型下拉框：{combo.count()} 项")
        for index in range(combo.count()):
            print(f"      显示 {combo.itemText(index)!r}  ·  数据 {combo.itemData(index)!r}")

        backend = window._backend
        if backend is None:
            print("[4] 后端未构造 —— 顶部状态：", window.status_label.text())
        else:
            print(
                f"[4] 后端 {backend.model_id} · context_window={backend.context_window} "
                f"（来自 /api/show 的 num_ctx）"
            )

        page_ok = window.chat_view._view.page() is not None
        print(f"[5] 对话区页面对象：{'已创建' if page_ok else '未创建'}")
        print("[*] 顶部状态：", window.status_label.text())

        print("--- 上面四段（[1][3][4][5]）都有输出即全部通过 ---")
        app.quit()

    # 3.5 s 足够：模型清单走一次 HTTP，网页走一次本地加载。
    # 真机冷启动时窗口先出现、下拉框后填上，所以这里必须给等待，不能同步断言。
    QTimer.singleShot(3500, dump)
    QTimer.singleShot(30000, lambda: (print("超时：某一段卡住了"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
