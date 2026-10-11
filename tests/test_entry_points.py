"""仓库根那些 ``.bat`` 的字节层：**必须是 CRLF**。

这条守卫来自一次实测出来的 bug，不是一条风格偏好。

``.gitattributes`` 里的 ``* text=auto eol=lf`` 把 ``run.bat`` 也写成了 LF，而
cmd.exe 解析 LF-only 的批处理时会**算错行偏移**：它不是报"这个文件坏了"，而是从
某个字节偏移开始把半行当命令执行。实测的逐字输出是

    'rt' is not recognized as an internal or external command
    'equ' is not recognized as an internal or external command
    ) was unexpected at this time

然后整个脚本以 255 退出——**双击 ``run.bat`` 什么都不会发生**，连它自己的错误
提示都不会打印（那段提示也在被错位执行的行里）。纯 ASCII 的 LF 批处理碰巧能跑，
一旦有中文（多字节）就必然错位，所以这个 bug 能在仓库里躺很久，而它坏掉的正是
README 里写的那个"双击启动"。

两个测试分工不同：

* :func:`test_entry_points_exist` 防**空转**——收集不到文件时，下面那个按文件
  参数化的测试会被跳过而不是通过，看着一片绿。
* :func:`test_batch_file_is_crlf` 是本体，逐个文件跑。它查的是**每一个**行尾
  （不只是"含 CRLF"）：混着 LF 的文件同样会让 cmd 错位，而"前 30 行是 CRLF"
  看起来完全正常。

**它当场抓到过一次，而症状不是"文件被写坏了"。** 写这条守卫的同一版里，工作区的
``.gitattributes`` 被临时改回没有 ``*.bat`` 那条规则的样子，而 pre-commit 在提交时
会把未暂存的改动 ``git stash`` 再恢复回来——那一趟 checkout 是按**当时的**
``.gitattributes`` 走的，于是 ``run.bat`` 被安静地写成了 LF。随后 ``*.bat text
eol=crlf`` 加回来，``git status`` 反而是**干净**的：git 入库时按同一套规则归一化，
工作区的 LF 与库里的 LF 相符，没有内容差异可言——**改 ``.gitattributes`` 不会回头
重写已经躺在工作区里的文件**，它只管下一次检出。

所以红了的时候修法是固定的：删掉再检出一次，让属性重新生效::

    rm run.bat && git checkout -- run.bat

（或在任何一次 ``git checkout --force`` 之后顺手确认一遍。）不要手工在编辑器里
转行尾——那取决于编辑器，而这条路只需要 git 自己那一份规则。
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: 收集时就把仓库根的批处理列出来。新增一个 ``.bat`` 会被自动覆盖，不必记得回来
#: 改这个文件——"忘了加进来"是这类守卫最常见的失效方式。
_BATCH_FILES = tuple(sorted(path.name for path in _REPO_ROOT.glob("*.bat")))

#: 必须存在的两个入口。名字在 README 里（启动、跑 CI），所以它们消失时不只是
#: 测试变少，是文档指向了一个不存在的东西。
_REQUIRED = ("run.bat", "ci.bat")


def test_entry_points_exist() -> None:
    """两个入口都在，而且至少收集到了一个批处理。"""
    missing = [name for name in _REQUIRED if name not in _BATCH_FILES]
    assert not missing, (
        f"仓库根找不到 {missing}。它们是本项目的两个入口（双击启动、跑一遍 CI），"
        f"README 里都引用了——找不到说明被删了或改名了，而文档还指着它们。"
    )


@pytest.mark.parametrize("name", _BATCH_FILES)
def test_batch_file_is_crlf(name: str) -> None:
    """没有孤立 ``\\n``，且以 ``\\r\\n`` 收尾。

    判据是"每个 ``\\n`` 前面都紧挨着一个 ``\\r``"，写成"数出来的 ``\\n`` 个数等于
    ``\\r\\n`` 个数"。不用 ``b"\\r\\n" in raw``：那条对**混合**行尾也为真，而混合
    正是 cmd 错位的一种成因。
    """
    raw = (_REPO_ROOT / name).read_bytes()
    lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert lone_lf == 0, (
        f"{name} 里有 {lone_lf} 个孤立 LF，但批处理文件必须是纯 CRLF——cmd.exe 解析 "
        f"LF 批处理时会算错行偏移，把半行当命令执行：症状是双击没反应，只打印一串 "
        f"'xxx is not recognized'。\n"
        f"上游有两道保障（.gitattributes 的 `*.bat text eol=crlf`、pre-commit 里 "
        f"mixed-line-ending 的 exclude），这条是下游的：即使那两处被改坏，坏掉的"
        f"字节也会在这里被挡住，而不是等到有人双击才发现。\n"
        f"修法：`rm {name} && git checkout -- {name}`"
        f"——见本文件 docstring 里「改 .gitattributes 不会回头重写工作区」那段。"
    )
    assert raw.endswith(b"\r\n"), f"{name} 没有以 CRLF 收尾，最后一行可能被吃掉。"
