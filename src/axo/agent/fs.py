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
import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_MAX_LINES",
    "MAX_LIST_ENTRIES",
    "MAX_READ_BYTES",
    "MAX_SEARCH_BYTES",
    "MAX_SEARCH_FILES",
    "MAX_SEARCH_RESULTS",
    "MAX_WRITE_BYTES",
    "NUL_PROBE_BYTES",
    "ReadFileError",
    "SearchHit",
    "WriteFileError",
    "list_entries",
    "read_text",
    "resolve_for_write",
    "resolve_within",
    "search_text",
    "write_text_atomic",
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


# ── 写：边界、原子写、目录与搜索 ─────────────────────────────────────────────


MAX_WRITE_BYTES = 1024 * 1024
"""一次最多写多少字节。**与读的上限是两件事**：读大了只是占上下文，写大了会
悄悄覆盖掉一个用户没打算换掉的文件，所以这里比读更紧。"""

MAX_LIST_ENTRIES = 200
"""列目录一次最多回灌多少项。总数另报，模型知道自己看到的是不是全部。"""

MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FILES = 200
MAX_SEARCH_BYTES = 4 * 1024 * 1024
"""全文搜索的三道闸：命中数、扫描文件数、读入总字节。谁先到就停，并说明。"""

_SEARCH_LINE_CHARS = 300

_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class WriteFileError(Exception):
    """写不了。消息是**写给模型看的**——它会被原样回灌。"""


@dataclass(frozen=True)
class SearchHit:
    """一条命中：哪个文件、第几行、那一行是什么。"""

    path: Path
    line_no: int
    line: str


def resolve_for_write(
    roots: Sequence[Path],
    raw: str,
    *,
    create_dirs: bool = False,
    deny: Sequence[Path] = (),
) -> Path:
    """把模型给的路径解析成**确定在某个允许目录里**、可以写的绝对路径。

    与 :func:`resolve_within` 共用 ``is_relative_to`` 那条判据，但**不要求文件
    已经存在**——写新文件是常态。

    ``deny`` 是永远不许写的目录（调用方传数据根）：模型不该改自己的记忆库、
    设置与日志。它比"路径不在根里"更严格，因为数据根**可能**就在某个读取根
    底下。

    Raises:
        WriteFileError: 没配根、路径为空、越界、落在 deny 里，或者是一个
            Windows 上根本建不出来的名字。
    """
    if not roots:
        raise WriteFileError("写入工具没有配置可写目录，当前不可用。请用户先到设置里添加一个目录。")

    wanted = raw.strip()
    if not wanted:
        raise WriteFileError("path 参数是空的，请给出要写的文件路径。")

    # **先看名字本身。** 像 ``a:b.txt`` 这种在 Windows 上会被解析成"盘符相对路径"，
    # 交给 ``(anchor / wanted).resolve()`` 之后会跑到根外面，于是用户拿到的错误是
    # "不在允许目录里"——原因对、方向错，而真正的问题是这个名字根本写不出来。
    problem = _windows_name_problem(Path(wanted))
    if problem:
        raise WriteFileError(problem)

    denied = [Path(one).expanduser().resolve() for one in deny]
    for root in roots:
        anchor = Path(root).expanduser().resolve()
        candidate = (anchor / wanted).resolve()
        if not candidate.is_relative_to(anchor):
            continue
        for blocked in denied:
            if candidate == blocked or candidate.is_relative_to(blocked):
                raise WriteFileError(
                    f"路径 {raw!r} 落在不允许写入的数据目录里（{blocked}）。"
                    "那是程序自己的记忆与设置，不能由工具改写。"
                )
        problem = _windows_name_problem(candidate)
        if problem:
            raise WriteFileError(problem)
        if not create_dirs and not candidate.parent.is_dir():
            raise WriteFileError(
                f"{candidate.parent} 不存在。要新建目录，请把 create_dirs 设为 true。"
            )
        return candidate

    allowed = "、".join(str(root) for root in roots)
    raise WriteFileError(
        f"路径 {raw!r} 不在允许写入的目录里。允许的目录：{allowed}。"
        "请用相对于这些目录的路径，或者告诉用户去设置里添加目录。"
    )


def write_text_atomic(
    path: Path,
    content: str,
    *,
    append: bool = False,
    create_dirs: bool = False,
    max_bytes: int = MAX_WRITE_BYTES,
) -> int:
    """写一个 UTF-8 文本文件，返回写入的字节数。

    覆盖模式下走**同目录临时文件 + ``os.replace``**：写到一半断电或进程被杀时，
    原来的文件还在（``os.replace`` 在同一卷上是原子的）。追加模式没法原子，
    但仍然先查大小上限，避免把磁盘写满。

    Raises:
        WriteFileError: 超长、父目录不存在、路径是目录、或者系统拒绝写。
    """
    payload = content.encode("utf-8")
    if len(payload) > max_bytes:
        raise WriteFileError(
            f"要写的内容有 {len(payload)} 字节，超过单次上限 {max_bytes} 字节。"
            "请分几次写，或者先问用户。"
        )

    if path.exists() and path.is_dir():
        raise WriteFileError(f"{path} 是一个目录，不能当文件写。")

    parent = path.parent
    if not parent.is_dir():
        if not create_dirs:
            raise WriteFileError(f"{parent} 不存在。要新建目录，请把 create_dirs 设为 true。")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WriteFileError(f"建目录失败：{exc}") from exc

    if append:
        current = path.stat().st_size if path.exists() else 0
        if current + len(payload) > max_bytes:
            raise WriteFileError(
                f"追加之后文件会超过 {max_bytes} 字节（现在 {current} 字节）。"
                "请换一个文件，或者先问用户。"
            )
        try:
            with path.open("ab") as handle:
                handle.write(payload)
        except OSError as exc:
            raise WriteFileError(f"写文件失败：{exc}") from exc
        return len(payload)

    handle = None
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(dir=str(parent), prefix=".axo-", suffix=".tmp")
        handle = os.fdopen(descriptor, "wb")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        Path(temporary).replace(path)
    except OSError as exc:
        raise WriteFileError(f"写文件失败：{exc}") from exc
    finally:
        if handle is not None:
            handle.close()
        if temporary and Path(temporary).exists():
            with contextlib.suppress(OSError):
                Path(temporary).unlink()
    return len(payload)


def list_entries(path: Path, *, limit: int = MAX_LIST_ENTRIES) -> tuple[list[str], int]:
    """列目录。返回 ``(最多 limit 项, 总项数)``；目录名带结尾斜杠。

    总项数单独给，是为了让模型知道"你看到的是不是全部"——它看不到"还有 37 项"
    的时候，会以为目录里就这些。
    """
    limit = max(1, int(limit))
    if not path.exists():
        raise ReadFileError(f"目录不存在：{path}")
    if path.is_file():
        return [path.name], 1
    if not path.is_dir():
        raise ReadFileError(f"{path} 不是目录，也不是文件。")

    try:
        with os.scandir(path) as iterator:
            entries = list(iterator)
    except OSError as exc:
        raise ReadFileError(f"列目录失败：{exc}") from exc

    def is_directory(entry: os.DirEntry[str]) -> bool:
        try:
            return entry.is_dir()
        except OSError:
            return False

    entries.sort(key=lambda entry: (0 if is_directory(entry) else 1, entry.name.lower()))
    names = [entry.name + ("/" if is_directory(entry) else "") for entry in entries[:limit]]
    return names, len(entries)


def search_text(
    base: Path,
    pattern: str,
    *,
    glob_pattern: str = "*",
    max_results: int = MAX_SEARCH_RESULTS,
    max_files: int = MAX_SEARCH_FILES,
    max_bytes: int = MAX_SEARCH_BYTES,
    case_sensitive: bool = False,
) -> list[SearchHit]:
    """在目录里按正则搜文本。**尽力而为：读不了的文件跳过，不因此失败。**

    三道闸（命中数、文件数、字节数）谁先到就停。返回的列表只包含命中；调用方
    需要知道"有没有提前停"时，自己对照上限与返回值。

    Raises:
        ReadFileError: 基准路径不存在，或者正则写错了。
    """
    if not base.exists():
        raise ReadFileError(f"搜索起点不存在：{base}")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        matcher = re.compile(pattern, flags)
    except re.error as exc:
        raise ReadFileError(f"正则表达式写错了：{exc}") from exc

    base_resolved = base.resolve()
    candidates = [base] if base.is_file() else list(base.rglob(glob_pattern))

    hits: list[SearchHit] = []
    scanned_files = 0
    scanned_bytes = 0
    for candidate in candidates:
        if len(hits) >= max_results or scanned_files >= max_files:
            break
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(base_resolved):
            continue  # 符号链接指向根外
        if not resolved.is_file() or resolved.is_symlink():
            continue
        scanned_files += 1
        try:
            data = resolved.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:NUL_PROBE_BYTES]:
            continue
        if scanned_bytes + len(data) > max_bytes:
            data = data[: max(0, max_bytes - scanned_bytes)]
        scanned_bytes += len(data)
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if matcher.search(line):
                hits.append(
                    SearchHit(path=resolved, line_no=number, line=line.strip()[:_SEARCH_LINE_CHARS])
                )
                if len(hits) >= max_results:
                    break
    return hits


def _windows_name_problem(path: Path) -> str:
    """Windows 上建不出来的名字，**在写之前就说**，而不是让系统抛一个 WinError。"""
    if sys.platform != "win32":
        return ""
    for index, part in enumerate(path.parts):
        if index == 0 and part.endswith(":" + os.sep):
            continue
        if part in (".", ".."):
            continue  # 相对路径分量，交给 is_relative_to 判越界
        if ":" in part:
            return f"路径分量 {part!r} 里含有冒号，Windows 会把它当成数据流语法。"
        if part.endswith((" ", ".")):
            return f"路径分量 {part!r} 以空格或点结尾，Windows 上建不出来。"
        stem = part.split(".", 1)[0].upper()
        if stem in _RESERVED_NAMES:
            return f"{part!r} 是 Windows 的保留设备名，不能当文件名用。"
    return ""
