"""注释里不许再有「修改史」——**这句话自己以前写错过、什么时候改的、谁核过**。

判据只有一条：**关于代码的历史留下，关于注释自己的历史删掉。** "这段代码为什么长这样"
（约束、不变量、量出来的数字、反直觉的取舍）是资产，**日期与实测数值也算资产**；而
"这句注释原先写的是 X、那是假的、没人回头改"是履历，它只有作者自己需要。

**为什么这件事要一条守卫。** 注释里的一句假话能活很久，恰恰因为**它旁边那句有守卫、
它没有**：热度面板上那句"依赖神经元级基质（还没建）"就是这么活下来的——基质在 P4a
就建好了，而旁边那句"依赖 P4c"被测试钉着，于是两句话里只有错的那句一直挂着。散文
没人读第二遍，所以这里挑的是三个**没有正当技术用法**的字面串：

* ``已经变成假话``——“这句话自己过期了”，没有任何技术含义；
* ``没人回头改``——同上，这是履历的措辞；
* ``订正过``——“这一段订正过一次/两次”。

**刻意不禁 ``原先`` / ``后来`` / ``当年`` / ``曾经``。** 它们在技术叙述里有正当用法，
而且常常是唯一准确的词："原先的阈值是 0.02"、"它后来被驱逐"、"曾经写成
``if space is None``，于是那个分支永远不会执行"——最后这句是一句真实的陷阱警告，删掉
它下一个读代码的人就会把那行改回去。禁掉它们会误伤，而误伤的守卫会被绕过。

**范围是 ``src/`` 与 ``tests/`` 的 ``.py``。** README 与 ``docs/`` 是散文，它们**必须**
能讲版本沿革（README 里「人格锚版本记录」那段就是一条操作契约：改常量就要加一行），
不在射程内——同 ``test_repo_hygiene.py`` 里"为什么本文件不检查 README"那条。
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("src", "tests")

#: 三个字面串。**只收"没有正当技术用法"的**，理由见模块文档。
_FORBIDDEN = ("已经变成假话", "没人回头改", "订正过")


def _scanned_files() -> list[Path]:
    """``src/`` 与 ``tests/`` 下的 ``.py``，**不含本文件**。

    排除自己不是偷懒：本文件必须把这三个串写下来才拦得住它们（同
    ``test_repo_hygiene.py`` 里那张被禁词清单的取舍——要机械地拦住一个词，就得把那个
    词写下来）。
    """
    here = Path(__file__).resolve()
    return sorted(
        path
        for directory in _SCANNED_DIRS
        for path in (_REPO_ROOT / directory).rglob("*.py")
        if path.resolve() != here
    )


def test_the_scan_reaches_the_code() -> None:
    """反空转：下面那条守卫的作用域不是一张空表。

    一个守卫最坏的失败方式不是漏报，是**报绿**——目录改名、glob 写错、路径拼成相对
    路径而工作目录不对，都会让它扫到零个文件然后安静地通过。
    """
    files = _scanned_files()
    assert len(files) > 100, f"只扫到 {len(files)} 个文件：{_SCANNED_DIRS} 的路径是不是变了？"
    assert all(path.is_file() for path in files)


def test_no_journal_voice_in_comments_or_docstrings() -> None:
    """``src/`` 与 ``tests/`` 里没有"这段注释自己的经历"。

    **按整份文本查，不按注释节点查**：判据是"这三个串不该出现在这个仓里"，而用
    ``ast``/tokenize 把它们限定到注释与 docstring 会多依赖一个解析器（还得自己判字符串
    字面量），多一处能坏的地方。代价是用户可见文案里的命中也会红——那同样是该红的。
    """
    hits: list[str] = []
    for path in _scanned_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            hits.extend(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{number}: {token}"
                for token in _FORBIDDEN
                if token in line
            )

    assert hits == [], (
        "注释里又出现了「这段注释自己的历史」：\n  "
        + "\n  ".join(hits)
        + "\n\n判据：**关于代码的历史留下，关于注释自己的历史删掉。**\n"
        "留：「这段代码为什么长这样」——约束、不变量、量出来的数字、反直觉的取舍，"
        "**包括日期与实测数值**。\n"
        "删：「这句话以前写错过 / 什么时候改的 / 为什么当时没发现 / 我核过什么」。\n"
        "删完要保证**结论还在**——只删「怎么发现的」。"
    )
