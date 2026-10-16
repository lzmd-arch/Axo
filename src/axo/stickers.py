"""表情包：**数据根下一个目录，你放什么她就能用什么**。

与音色（``models/voices/``）和语音模型同一个姿态：跟着数据根走、**不创建**、不下载。
判据也一样直白——**一个文件就是一个表情包**，名字就是文件名去掉扩展名。

**为什么不要一份清单文件。** 任何要用户维护的索引，都会在"我明明放进去了"和"她怎么
还是不知道"之间长出一条排查路。目录本身就是清单：丢进去就是加了一个，删掉就是没了。
代价是名字只能取文件名——而那个名字正是用户在资源管理器里看到的那个，所以它反而是最
不需要解释的一种命名。

**只认图片后缀。** 这个目录里出现别的东西（``README.txt``、``.psd``、``Thumbs.db``）
不是表情包。判据只在 :data:`STICKER_SUFFIXES` 一处，扫描与查找共用——两处各写一份的话，
"扫得到却找不到"会变成一种只有用户碰得上的怪事。

**这个模块不认识 Qt，也不发请求**：它只把目录读成一个元组，所以整张表在核心腿里可以
逐条断言（同 :mod:`axo.tts` 的 ``voices_in``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "STICKER_SUFFIXES",
    "Sticker",
    "catalog_line",
    "find_in",
    "find_sticker",
    "stickers_in",
]

#: 认哪些后缀。**比较时统一小写**——用户从手机传过来的照片常常是 ``.JPG``。
STICKER_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


@dataclass(frozen=True)
class Sticker:
    """一个表情包：**名字 + 它落在哪儿**。

    ``name`` 是文件名去掉扩展名。它同时是给模型看的那个标识、与用户在资源管理器里
    看到的那个词——两件事共用一个字符串，是为了让"她发的那张"能被一句话对上。
    """

    name: str
    path: Path

    def as_uri(self) -> str:
        """给网页那侧的 ``file:`` URL。

        **``resolve()`` 不是可选的**：``as_uri`` 要求绝对路径，而数据根在便携模式或
        ``AXO_DATA`` 下本来就是一个相对路径的可能性始终存在（见 ``paths.data_dir``）。
        解析一次比"某台机器上图片空着"好排查得多。
        """
        return self.path.resolve().as_uri()


def stickers_in(root: Path) -> tuple[Sticker, ...]:
    """目录里的全部表情包，**按名字排序**。

    排序是规格的一部分：同一份目录永远给同一个顺序，于是工具描述是稳定的（提示词缓存
    吃得着），测试也不必依赖文件系统的返回次序。目录不存在、或者里面没有认得出的图片，
    都返回空元组——那是**常态**，不是错误（同 ``tts.voices_in`` 对"还没放音色"的姿态）。
    """
    if not root.is_dir():
        return ()
    found = [
        Sticker(name=child.stem, path=child)
        for child in root.iterdir()
        if child.is_file() and child.suffix.lower() in STICKER_SUFFIXES
    ]
    # 按 ``casefold`` 再按原名：``A.png`` 与 ``a.png`` 在小写世界里同名，而它们的先后
    # 必须有个确定的答案，否则同一份目录在两台机器上会给出两种顺序。
    return tuple(sorted(found, key=lambda sticker: (sticker.name.casefold(), sticker.name)))


def find_in(stickers: Sequence[Sticker], name: str) -> Sticker | None:
    """在一份**已经扫好的**目录里按名字找一个。

    工具那一侧走它：目录是装配期扫好传进去的（工具表每轮重建，见
    ``ui/main_window.py`` 的 ``_tools()``），再扫一遍目录等于让"这一轮看到的表情包"
    与"提示词里列的那几个"有可能不是同一批——那种不一致的症状是模型调一个刚刚才
    列出来的名字，然后被告知没有。判据本身与 :func:`find_sticker` 是同一份。
    """
    wanted = name.strip()
    if not wanted:
        return None
    for sticker in stickers:
        if sticker.name == wanted:
            return sticker
    folded = wanted.casefold()
    for sticker in stickers:
        if sticker.name.casefold() == folded:
            return sticker
    return None


def find_sticker(root: Path, name: str) -> Sticker | None:
    """按名字找一个。找不到返回 ``None``。

    **先精确、再忽略大小写。** 模型写名字时大小写写歪是常事（``Sad`` / ``sad``），而
    "差一个字母就用不了"是没必要的一道坎。反过来不成立：如果目录里真有 ``sad`` 与
    ``Sad`` 两个名字不同的表情包，精确那一步会先命中，不会张冠李戴。

    空名字返回 ``None``：那是"没说要发哪张"，不是"找名叫空串的那张"。
    """
    return find_in(stickers_in(root), name)


def catalog_line(stickers: Sequence[Sticker]) -> str:
    """给模型看的那一行：**有什么表情包可用**。

    名字之间用「、」连起来，**不编号**——编号会让模型以为入参是序号，而这个工具要的是
    名字。本函数在目录为空时返回空串，调用方据此决定"不注册这个工具"（与 ``read_file``
    在可读根为空时的姿态一致：没有能力的时候，那个工具就是不在表里）。
    """
    return "、".join(sticker.name for sticker in stickers)
