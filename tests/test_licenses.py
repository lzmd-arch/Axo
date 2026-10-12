"""许可全文的完整性：**逐字节 verbatim**，这是随产物发出去的东西。

``licenses/`` 里那两份（GPL-3.0 与 LGPL-3.0）不是"仓库里的参考文档"，它们是
**要随 zip 一起发给别人的副本**。LGPLv3 §4(b) 要求分发者"give all recipients a copy
of this License"——一份被格式化过的文本不是那个 copy。

**为什么要把它钉成断言而不是靠 review。** 这个文件面对的是一个每台机器都会跑、
而且跑的时候**看起来在做一件好事**的改法：``trailing-whitespace`` 钩子。GPL-3.0 正文里
有 39 行、LGPL-3.0 里有 3 行以空白结尾——它们是原文的缩进与排版，不是待清理的脏东西。
钩子把它"修"掉之后，diff 读起来像一次卫生清理，而事实是这份许可不再逐字节等于
FSF 发布的那一份。``.pre-commit-config.yaml`` 已经把 ``licenses/`` 排除掉了；这个文件
是那条排除的**结果**判据（照 :mod:`tests.test_repo_hygiene` 的说法：检查结果而不是意图）。

**为什么真值取自 gnu.org 而不是本地某份。** 只有 FSF 发布的文本才是那个 "copy"。
两份的出处与取用日期记在下面；换版本要连哈希一起换，并在
``THIRD-PARTY-NOTICES.md`` 里同时改掉——那是给别人看的那一份。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axo.bench import file_sha256

_LICENSES = Path(__file__).resolve().parents[1] / "licenses"

#: ``(文件名, sha256, 出处, 取用日期)``。哈希是**原始字节**的，不含任何解码。
#:
#: ``gnu.org/licenses/*.txt`` 自 2007 年发布以来没有改过，所以这几个值是冻结的；
#: 哪天真变了，是"FSF 改了许可文本"这种量级的事，该有人来看一眼，而不是顺手更新断言。
_FROZEN: list[tuple[str, str, str, str]] = [
    (
        "GPL-3.0.txt",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
        "https://www.gnu.org/licenses/gpl-3.0.txt",
        "2026-10-12",
    ),
    (
        "LGPL-3.0.txt",
        "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
        "https://www.gnu.org/licenses/lgpl-3.0.txt",
        "2026-10-12",
    ),
]

#: 正文里必须出现的字串。哈希已经能拦住任何改动，这两条是给**人**看的：断言红了的时候
#: 一眼能看出"这是不是那份许可"，而不是对着一个十六进制串发呆。
_MARKERS = {
    "GPL-3.0.txt": "GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007",
    "LGPL-3.0.txt": "GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007",
}


@pytest.mark.parametrize(("name", "digest", "origin", "fetched"), _FROZEN)
def test_the_shipped_license_text_is_the_verbatim_one(
    name: str, digest: str, origin: str, fetched: str
) -> None:
    """**这个文件的核心断言。** 正文逐字节等于 FSF 发布的那一份。"""
    path = _LICENSES / name
    assert path.is_file(), (
        f"{path} 不在。它要随 zip 发给别人（LGPLv3 §4(b)），从 {origin} 取一份存回去——"
        "建仓时取用日期是 " + fetched
    )
    actual = file_sha256(path)
    assert actual == digest, (
        f"{name} 的正文被改过了。\n  期望 {digest}\n  实际 {actual}\n"
        f"它不是参考文档，是要发给别人的那份许可副本——被格式化过的文本不满足 §4(b)。"
        f"最近一次取用：{origin}（{fetched}）。"
    )


@pytest.mark.parametrize(("name", "marker"), sorted(_MARKERS.items()))
def test_the_text_is_recognisable_as_that_license(name: str, marker: str) -> None:
    """标题与版本行。哈希红了的时候，这一条说明白错的是哪一份、错成什么样子。

    判据是**子串而不是开头**：这两份正文的标题是居中的，行首有 19 个空格。
    """
    text = (_LICENSES / name).read_text(encoding="utf-8")
    assert marker in text, f"{name} 里找不到 {marker.splitlines()[0]!r} 那一行"


def test_no_stray_files_in_the_licenses_dir() -> None:
    """``licenses/`` 只放**全文**，不放别的东西。

    第三方那份短许可（numpy 的 BSD、certifi 的 MPL-2.0……）不 vendoring——打包时
    从 ``.venv/Lib/site-packages/*.dist-info/licenses/`` 现取，那才是实际随包的那一份。
    这条挡的是"顺手把某个第三方许可也丢进来"：那样一来库里就有了两处副本，而其中一处
    没人保证跟得上版本。
    """
    names = sorted(path.name for path in _LICENSES.iterdir() if path.is_file())
    assert names == sorted(name for name, _, _, _ in _FROZEN), (
        f"licenses/ 里多出了文件：{set(names) - {name for name, _, _, _ in _FROZEN}}。"
        "第三方的短许可不存这里，打包时从 dist-info 现取。"
    )
