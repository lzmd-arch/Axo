"""界面里的异常往哪儿写。

**打包成窗口程序后 stderr 是空的**：槽里抛出的异常只会让界面停在某一句"正在…"，
而现场什么都不剩（2026-10-31 的"卡在检索记忆"与"对话区没了"都是这样查起来的）。
写盘是唯一能把它带回来的方式。

**三个写入方共用这一处**：进程级 ``sys.excepthook``（:mod:`axo.ui.app`）、槽里的
兜底（``MainWindow`` 那两处 ``try/except``）、以及网页那一侧（渲染进程死掉、JS
抛错，见 :mod:`axo.ui.chat_view`）。以前前两处各抄了一份写盘代码，第三个要用的
时候就成了三份——而三份各自改格式的结果是同一个日志文件里三种分隔符，读的人
得先学会辨认是谁写的。
"""

from __future__ import annotations

from ..paths import data_dir

__all__ = ["ui_error_log_path", "log_ui_error"]

#: 每条记录前面那行分隔。**读日志的人靠它切分**，所以三个写入方必须一致。
_SEPARATOR = "----"


def ui_error_log_path():
    """``logs/ui-errors.log`` 的路径。目录不存在时**不在这里建**。"""
    return data_dir() / "logs" / "ui-errors.log"


def log_ui_error(detail: str) -> None:
    """追加一条。**失败就什么都不做**——写日志这件事本身不许把程序拖垮。"""
    try:
        path = ui_error_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_SEPARATOR + chr(10) + detail)
    except Exception:
        pass
