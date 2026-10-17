"""把模型吐出来的 Markdown 变成「可以念的句子」。

分句是流式语音的第一道工序：回复还在生成时，已经完整的那几句就该先被合成、先
出声，而不是等整段写完。这个模块只做纯文本变换，**不认识 TTS，也不认识线程**。

## 三条规矩

1. **代码不念。** 围栏代码块整段丢掉；行内代码的反引号去掉、内容保留（里面常常
   是文件名或命令，念出来至少不是噪声）。URL 只念链接文字。
2. **只在真句末切。** 中文的 ``。！？；`` 与英文的 ``!?;`` 和换行。**不认句点**：
   ``3.5``、``v0.6.0``、``api.deepseek.com`` 都会被句点切碎，而 TTS 读到半个
   版本号只会念出一串数字。
3. **长句先找逗号。** 一句话超过 ``max_chars`` 时，优先在最近的 ``，、：`` 处切；
   找不到才硬切。这不是模型预算，是 TTS 一次输入的长度上限。

返回值永远是 ``(完整句子, 还没说完的尾巴)``。流式调用方把尾巴留到下一次
``feed`` 里继续拼；``final=True`` 时尾巴也算一句。
"""

from __future__ import annotations

import re

__all__ = ["clean_for_speech", "split_sentences"]

#: 句末。**没有句点**，理由见模块文档。
_TERMINATORS = "。！？；!?;\n"

#: 长句优先在这里断开。
_SOFT_BREAKS = "，,、：:"

#: 一句话多长就该切（字符）。TTS 组件的单次输入上限远大于它，这个数是为了让
#: 首句尽快完整——太长的话"第一句"本身就要等很久。
DEFAULT_MAX_CHARS = 80

_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARK = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|\*)(.+?)\1")
_WS = re.compile(r"[ \t]+")


def clean_for_speech(text: str) -> str:
    """去掉不该念的 Markdown 记号。**不删正文。**"""
    if not text:
        return ""
    cleaned = _FENCE.sub(" ", text)
    cleaned = _IMAGE.sub(" ", cleaned)  # 图片：alt 也不念，它不是一句话
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _LIST_MARK.sub("", cleaned)
    cleaned = _EMPHASIS.sub(r"\2", cleaned)
    cleaned = _WS.sub(" ", cleaned)
    return cleaned.strip()


def split_sentences(
    text: str, *, final: bool = False, max_chars: int = DEFAULT_MAX_CHARS
) -> tuple[list[str], str]:
    """切出已经完整的句子，返回 ``(句子, 尾巴)``。

    Args:
        text: 已经清洗或尚未清洗的 Markdown。**这里会先清洗**，所以调用方不必
            自己调 :func:`clean_for_speech`。
        final: 是否已到回复结尾。为真时尾巴也作为一句返回（尾巴为空则只有空串）。
        max_chars: 单句上限；超出时优先在逗号处切。
    """
    if max_chars < 1:
        raise ValueError("max_chars 必须为正。")

    cleaned = clean_for_speech(text)
    if not cleaned:
        return [], ""

    sentences: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        piece = current.strip()
        current = ""
        if piece:
            sentences.append(piece)

    index = 0
    while index < len(cleaned):
        char = cleaned[index]
        current += char
        index += 1

        if char in _TERMINATORS:
            flush()
            continue
        if len(current) >= max_chars:
            cut = _last_soft_break(current)
            if cut <= 0:
                cut = len(current)
            sentences.append(current[:cut].strip())
            current = current[cut:]
            # 切完剩下的那一截可能仍然超长（一串没有标点的长数字），交给下一轮
            # 循环继续按 max_chars 切。
    if final:
        flush()
    return sentences, current.strip()


def _last_soft_break(text: str) -> int:
    """最后一个可以断开的位置（断在那个标点**之后**）。没有就返回 0。"""
    for index in range(len(text) - 1, -1, -1):
        if text[index] in _SOFT_BREAKS:
            return index + 1
    return 0
