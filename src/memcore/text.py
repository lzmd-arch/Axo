"""文本的字符分类与切分。

单独一个模块，是因为有两处要用同一套判断，而且它们**必须**用同一套：
:mod:`memcore.sdr` 靠它决定"什么算一个 token"，:mod:`memcore.tokens` 靠它决定
"什么算一个汉字"。两处各写一遍的话，编码器与预算器对同一段文本的理解就会漂移
——表现为"账本说 184 token，实际发出去 240"。
"""

from __future__ import annotations

import unicodedata

__all__ = ["CJK_RANGES", "is_cjk", "is_word", "tokenize"]

#: 视为"一个字"的区段。**按码位判断而不是按正则的 ``\\w``**：``\\w`` 在 Python 里
#: 默认按 Unicode 的字母属性匹配，会把汉字与拉丁字母混进同一个字符类，于是
#: ``"abc汉字"`` 被切成一个 token 而不是两个。
CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 基本区
    (0xF900, 0xFAFF),  # CJK 兼容表意文字
    (0x3040, 0x309F),  # 平假名
    (0x30A0, 0x30FF),  # 片假名
    (0xAC00, 0xD7AF),  # 谚文音节
    (0x20000, 0x2FA1F),  # CJK 扩展 B 及以后
)


def is_cjk(char: str) -> bool:
    """这个字符是否属于「一个字一个音节」的文字。"""
    code = ord(char)
    return any(low <= code <= high for low, high in CJK_RANGES)


def is_word(char: str) -> bool:
    """拉丁/数字/下划线——会被整段当作一个词。"""
    return char.isalnum() or char == "_"


def normalize(text: str) -> str:
    """NFKC 折叠 + casefold。

    用 NFKC 而不是 NFC：全角标点与全角数字在中文输入里很常见，NFKC 把它们折成
    半角，于是「１２３」与 ``123`` 落在同一个 token 上。大小写用 casefold 而非
    ``lower``——前者才是 Unicode 意义上的完整折叠。
    """
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """把文本切成带角色前缀的 token 列表。

    三类 token，前缀保证不同类之间**永不碰撞**（否则 ``"c" + "at"`` 与
    ``"w" + "at"`` 会撞成同一个 token，日语与英语的短词就会互相干扰）::

        c:{字}          单个 CJK 字
        w:{词}          一段连续的拉丁字母/数字
        b:{前}:{后}      相邻两个基本 token 组成的二元组

    二元组是中文这类**没有词边界**的语言里唯一能表达"顺序与搭配"的手段：只做
    单字编码的话，「上海」与「海上」会是同一个 SDR。
    """
    normalized = normalize(text)

    base: list[str] = []
    word: list[str] = []
    for char in normalized:
        if is_cjk(char):
            if word:
                base.append("w:" + "".join(word))
                word.clear()
            base.append("c:" + char)
        elif is_word(char):
            word.append(char)
        else:
            # 标点、空白、emoji 一律是分隔符。**刻意不让它们进入 token**：
            # 标点不承载语义，而让它们参与哈希只会给所有文本加上一份共同的噪声。
            if word:
                base.append("w:" + "".join(word))
                word.clear()
    if word:
        base.append("w:" + "".join(word))

    tokens = list(base)
    # strict=False 是刻意的：这两个序列**本来就差一个**，那正是"相邻"的含义。
    tokens.extend(f"b:{first}:{second}" for first, second in zip(base, base[1:], strict=False))
    return tokens
