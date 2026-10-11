"""``read_file`` 的边界：能读哪儿、读多少、读出来是什么。

这个模块存在的唯一理由是**把模型给的一个字符串变成一次安全的读盘**。它是
P3a 里最容易写出"看起来在工作"的一段代码的地方：一个没有边界检查的
``Path(arg).read_text()`` 也能跑、也能通过"能读到文件"的测试。

## 三条边界，各自对应一类具体的错误

1. **目录**——``is_relative_to``，不是 ``str.startswith``。
2. **大小**——超过上限**明说截断**，不静默。
3. **内容**——二进制与坏编码**拒绝**，不做替换。

### 目录：为什么不是 ``startswith``

``str.startswith`` 有一个安静的经典错误：根目录 ``/data`` 会匹配 ``/database``。
按**路径分量**比较就不会——``Path.is_relative_to`` 做的是这件事。

顺带买到的两件事，各由一条测试钉住：

* ``resolve()`` 吃掉 ``..`` 并跟随符号链接，所以**逃逸天然被覆盖**——根目录里
  一个指向外面的符号链接，解析后就不在根里了；
* Windows 上 ``PureWindowsPath`` 的比较**本来就是大小写不敏感的**（实测：
  ``Path("C:/DATA/x").is_relative_to("C:/Data")`` 为真），所以这里**不加**
  ``os.path.normcase``：那是一句不起作用却看起来在承重的代码，而"看起来在承重的
  死代码"比没有更糟——下一个人会以为大小写这件事已经有人管了。

**没配目录 = 工具禁用**，不是"给个默认目录"。默认 cwd 或 home 读起来像体贴，
实际是替用户做了一个他从没做过的授权决定。

### 大小：截断必须留痕

超限时返回正文 **加上一句明确的标记**，而不是静默截断。理由不是"礼貌"：模型
读到半个文件时不会知道自己读到的是半个，它会基于半个文件下结论——而那个结论
在界面上看起来和完整的一样确定。
"""

from __future__ import annotations

import codecs
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "DEFAULT_MAX_LINES",
    "MAX_READ_BYTES",
    "NUL_PROBE_BYTES",
    "ReadFileError",
    "read_text",
    "resolve_within",
]

MAX_READ_BYTES = 64 * 1024
"""一次最多读进来多少字节。

**先验值**，与 ``max_rounds`` 同属待标定的一类——标定它需要知道"一次工具结果
在真实上下文里占多少"，而那要靠 :file:`tools/smoke_agent.py` 量。方向上的取舍是
清楚的：小了只是读得少（模型分页再读一次），大了会把逐字稿挤出上下文。所以先取
一个偏小的数。

取 64 KiB 而不是按 token 算，是因为这里必须先有一道**与模型无关**的闸：读盘
发生在编码之前，用一个还没标定的 token 估计器来限制 I/O 是把两件事绑在一起。
"""

DEFAULT_MAX_LINES = 200
"""默认回灌多少行。与 :data:`MAX_READ_BYTES` 是两道独立的闸——一个 8 KiB 的
单行压缩 JSON 会先撞行数，一个大文件会先撞字节数。"""

NUL_PROBE_BYTES = 4096
"""查 NUL 字节只看开头这么多。

比扩展名黑名单可靠：``.txt`` 里可以是任何东西，而二进制文件（可执行文件、SQLite、
图片、压缩包）的头部几乎必然含 NUL。只看头部是因为要读的本来就是文本——一个前
4 KiB 全是可打印字符的文件，即使更深处是二进制，读出来也是能用的正文。
"""

_SNIP = "\n\n[…]"


class ReadFileError(Exception):
    """读不了。消息是**写给模型看的**——它会被原样回灌。"""


def resolve_within(roots: Sequence[Path], raw: str) -> Path:
    """把模型给的路径解析成一个**确定在**某个允许目录里的绝对路径。

    Raises:
        ReadFileError: 没配目录、路径为空，或者落在所有允许目录之外。
    """
    if not roots:
        # 这个分支不该在正常路径上出现：没配目录时 read_file 根本不出现在
        # registry.specs() 里（见 catalog），模型看不到它就不会调。留着是为了
        # 硬调的情况——那时它也该拿到一句解释，而不是一个 KeyError。
        raise ReadFileError("read_file 没有配置可读目录，这个工具当前不可用。")

    wanted = raw.strip()
    if not wanted:
        raise ReadFileError("path 参数是空的，请给出要读的文件路径。")

    # 不变量是"**解析后的路径必须落在某个允许目录里**"，而不是"输入必须是相对路径"。
    # 两者的区别在绝对路径上：``(anchor / wanted)`` 里一个绝对路径会**替换**掉
    # anchor（``Path("/a") / "/etc/passwd" == "/etc/passwd"``），于是根之外的一律
    # 被下面的判断拒掉——这对安全来说已经够了，因为根**之内**的绝对路径本来就
    # 没越过任何边界。
    fallback: Path | None = None
    for root in roots:
        # ``expanduser`` 放在这里而不是要求调用方先处理：一个存成 ``~/notes``
        # 的配置项如果不展开，会被当成相对当前工作目录的路径——而它"能用"，
        # 只是读的是另一个地方。
        anchor = root.expanduser().resolve()
        candidate = (anchor / wanted).resolve()
        if not candidate.is_relative_to(anchor):
            continue
        if candidate.exists():
            return candidate
        # 记下第一个"字面上落在根里但文件不在"的候选，留作兜底。
        #
        # **这一段是多根能用的前提**：``resolve()`` 是非严格的，一个不存在的路径
        # 也照样会被解析成"落在第一个根里"，于是第一个根会**遮蔽**掉第二个根里
        # 真实存在的那个文件——而症状是"文件不存在"，一个指不到根因的错误。
        if fallback is None:
            fallback = candidate

    if fallback is not None:
        # 让 :func:`read_text` 去说"文件不存在"：那条消息里带着路径，比这里
        # 重说一遍更能指导下一步。
        return fallback

    allowed = "、".join(str(root) for root in roots)
    raise ReadFileError(
        f"路径 {raw!r} 不在允许读取的目录里。允许的目录：{allowed}。"
        "请用相对于这些目录的路径，或者告诉用户去设置里添加目录。"
    )


def read_text(
    path: Path,
    *,
    start_line: int = 1,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = MAX_READ_BYTES,
) -> str:
    """读一个文本文件。

    Args:
        path: **已经过 :func:`resolve_within` 的**路径。
        start_line: 从第几行开始（1 基）。数的是**可读窗口内**的行号——见下。
        max_lines: 最多回灌多少行。
        max_bytes: 字节上限。

    Raises:
        ReadFileError: 不是文件、像二进制、不是 UTF-8，或者 ``start_line``
            超出了可读窗口。

    **可读窗口的边界要说清楚**：只读前 ``max_bytes`` 字节，``start_line`` 在这个
    窗口内数行号。所以一个 10 MiB 的日志，第 5000 行是读不到的——工具给的承诺是
    "让模型看到一页"，不是"让模型读到文件的任意位置"。这条限制是**有意的**：
    要支持任意位置就得先整份读进来，而那正是这个上限要防的事。真需要时，
    让用户用别的办法分片。
    """
    if not path.exists():
        raise ReadFileError(f"文件不存在：{path}")
    if not path.is_file():
        raise ReadFileError(f"{path} 不是一个文件（可能是个目录）。")

    with path.open("rb") as handle:
        head = handle.read(max_bytes + 1)
    total_bytes = path.stat().st_size

    if b"\x00" in head[:NUL_PROBE_BYTES]:
        raise ReadFileError(
            f"{path.name} 看起来是二进制文件（开头就有 NUL 字节），读不出文本内容。"
        )

    cut = len(head) > max_bytes
    if cut:
        head = head[:max_bytes]

    # ``final=cut`` 是这里唯一微妙的一行。截断可能正好切在一个多字节字符中间，
    # 而那不是"文件编码坏了"——是**我们自己切坏的**。增量解码器在 ``final=False``
    # 时会把不完整的尾字节留在缓冲里，于是这个区别自动分开了：
    # 文件本身编码坏 → UnicodeDecodeError；我们切坏了 → 安静丢掉那几个字节。
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        text = decoder.decode(head, final=not cut)
    except UnicodeDecodeError as exc:
        # 不用 ``errors="replace"``：替换符会变成**假内容**，而模型会把它当成
        # 文件里真的有的东西读下去。
        raise ReadFileError(
            f"{path.name} 不是 UTF-8 文本（第 {exc.start} 字节处解不开）。"
            "如果它是别的编码，请用户先转成 UTF-8。"
        ) from exc

    lines = text.splitlines()
    if start_line < 1:
        raise ReadFileError(f"start_line 从 1 开始数，收到 {start_line}。")
    if start_line > len(lines):
        raise ReadFileError(
            f"start_line={start_line} 超出了可读窗口：这个窗口里只有 {len(lines)} 行"
            + ("（文件太大，只读了开头一段）" if cut else "")
            + "。"
        )

    window = lines[start_line - 1 : start_line - 1 + max_lines]
    body = "\n".join(window)

    notes: list[str] = []
    if cut:
        notes.append(f"已截断：文件共 {total_bytes} 字节，这里只读了前 {max_bytes} 字节")
    elif len(window) < len(lines) - (start_line - 1):
        rest = len(lines) - (start_line - 1) - len(window)
        notes.append(f"后面还有 {rest} 行，用 start_line={start_line + len(window)} 继续")
    if notes:
        body += f"{_SNIP}{'；'.join(notes)}…]"
    return body
