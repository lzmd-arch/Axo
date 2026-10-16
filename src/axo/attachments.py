"""用户拖进来的文件：**认得出是什么、读得出里面的字**。

三件事，全是纯的：判类型、抽正文、按上限截断。它不认识 Qt，也不发请求——所以整张
判据表在核心腿里可以逐格断言。

## 类型判据**先看文件头，再看后缀**

一个改名成 ``.txt`` 的可执行文件会被当文本读，读出来是一堆乱码——而那些乱码会原样进
上下文（用云端模型时，就是发到了别人的服务器上）。所以分三步：

1. **有魔数的按魔数判**（PNG / JPEG / GIF / WebP / PDF）：后缀写错了也认得出来，
   而"长得像文本的二进制"进不了上下文。
2. **没有魔数的文本类按后缀判**，而且**要求样本看起来真是文本**：前几 KB 里出现 NUL
   字节的一律不当文本。这条判据与 ``agent/fs.py`` 给 ``read_file`` 用的是同一条。
3. 其余一律 ``other``：**只报身份**（叫什么、多大），不读内容。

## 上限是**给上下文**的，不是给磁盘的

一个 200 MB 的日志会把窗口吃光。文本按 :data:`MAX_TEXT_CHARS` 截断，图片按
:data:`MAX_IMAGE_BYTES` 拒收——两个数都只在这里定义，调用方不许各写一份。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IMAGE_SUFFIXES",
    "MAX_IMAGE_BYTES",
    "MAX_TEXT_CHARS",
    "PDF_SUFFIXES",
    "SAMPLE_BYTES",
    "TEXT_SUFFIXES",
    "Attachment",
    "attachment_for",
    "body_of",
    "classify",
]

#: 按后缀认的文本类。**刻意列全**而不是"除了图片都是文本"——反过来写的话，一个
#: ``.exe`` 会被当文本读，而它里面恰好没有 NUL 字节时（有些加壳的东西就是）就读进来了。
TEXT_SUFFIXES = (
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".py",
    ".pyi",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".htm",
    ".css",
    ".sh",
    ".bash",
    ".zsh",
    ".bat",
    ".ps1",
    ".sql",
    ".xml",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".lua",
    ".r",
)

#: 按后缀与魔数一起认的图片。**与 ``stickers.STICKER_SUFFIXES`` 是同一个集合**，
#: 但两处各写一份是有理由的：那边是"她能发的表情包"，这边是"能进视觉通道的图"，
#: 两个集合今天相同、明天未必——而把它们绑成一个常量，改一处会悄悄改掉另一处。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

PDF_SUFFIXES = (".pdf",)

#: 文本进上下文的上限（**字符**，不是字节）。两万字符约一万五千 token——够读完一份
#: 长文档，又不至于把一轮对话的窗口吃光。
MAX_TEXT_CHARS = 20_000

#: 图片的上限（字节）。与 ``ui/screen.py`` 的 ``MAX_BYTES`` 同档：那条路已经证明
#: 这个尺寸能进得了请求。
MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: 判"它是不是文本"时读多少字节。4096 足够撞上 NUL（真正的二进制文件几乎立刻就有）。
SAMPLE_BYTES = 4096

#: 魔数表：(前缀, 类型)。**只看开头几个字节**，够用且不会误判。
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"RIFF", "image"),  # WebP 是 RIFF 容器；下面再看 WEBP 那一段
    (b"%PDF-", "pdf"),
)


@dataclass(frozen=True)
class Attachment:
    """一个拖进来的文件：**它是谁、在哪、多大**。"""

    path: Path
    kind: str
    """``text`` / ``image`` / ``pdf`` / ``other``。"""

    size: int

    @property
    def name(self) -> str:
        """文件名。给模型看的身份就是它——不含路径（路径是本机的事）。"""
        return self.path.name


def attachment_for(path: Path) -> Attachment | None:
    """认一个路径。**目录、不存在的、读不动的都返回 ``None``。**

    不抛：拖进来的东西是什么都不奇怪（一个快捷方式、一个正在被别的程序写着的文件），
    而"拖一个文件进来让程序炸了"是最不该有的一种失败。
    """
    try:
        if not path.is_file():
            return None
        return Attachment(path=path, kind=classify(path), size=path.stat().st_size)
    except OSError:
        return None


def classify(path: Path) -> str:
    """按**魔数优先、后缀其次**判类型。判不出来就是 ``other``。"""
    magic = _read_magic(path)
    if magic is not None:
        for prefix, kind in _MAGIC:
            if magic.startswith(prefix):
                if prefix == b"RIFF":
                    # RIFF 还可能是 wav/avi；只有第 8-12 字节是 WEBP 才算图。
                    return "image" if magic[8:12] == b"WEBP" else "other"
                return kind

    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES and _looks_like_text(path):
        return "text"
    return "other"


def body_of(attachment: Attachment, *, limit: int = MAX_TEXT_CHARS) -> str:
    """这个附件**进上下文的那一段正文**。

    文本类与 PDF 给内容；**图片与其它给一句身份**——图走视觉通道（不进这段正文），
    而其它类型我们读不懂，只能告诉模型"有这么个文件"。**给不出内容时说清楚为什么**，
    不返回空串：一段空白会让模型以为文件是空的。
    """
    if attachment.kind == "text":
        text = _read_text(attachment.path, limit=limit)
        if text is None:
            return f"（{attachment.name} 读不出来：它不像文本文件）"
        return _wrap(attachment.name, text, limit=limit)
    if attachment.kind == "pdf":
        return _pdf_body(attachment, limit=limit)
    if attachment.kind == "image":
        return f"（附了一张图：{attachment.name}，{_human_size(attachment.size)}）"
    return (
        f"（附了一个文件：{attachment.name}，{_human_size(attachment.size)}。这个类型读不出内容）"
    )


def _wrap(name: str, text: str, *, limit: int) -> str:
    """把正文包成一段带出处的块。**截断要说出来**——不然模型以为自己读完了。"""
    clipped = text[:limit]
    head = f"（{name} 的内容）"
    if len(text) > limit:
        head = f"（{name} 的内容，只给了前 {limit} 个字符，后面还有 {len(text) - limit} 个）"
    return f"{head}\n{clipped}"


def _pdf_body(attachment: Attachment, *, limit: int) -> str:
    """PDF 的文字。**没装解析器就说没装**，不假装它没内容。"""
    try:
        from pypdf import PdfReader  # 函数内 import：没装这个 extra 也能 import 本模块
    except ImportError:
        return f"（{attachment.name} 是 PDF，但这个环境里没装 PDF 解析器，读不出内容）"
    try:
        reader = PdfReader(str(attachment.path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 —— 坏 PDF 的花样太多，一律降级成一句人话
        return f"（{attachment.name} 这个 PDF 读不出来：{type(exc).__name__}）"
    text = "\n".join(part for part in pages if part.strip())
    if not text.strip():
        # **扫描件就是这种**：它有内容，只是里面全是图。说清楚，免得模型以为文件是空的。
        return f"（{attachment.name} 是一个 PDF，但里面没有可抽取的文字——多半是扫描件）"
    return _wrap(attachment.name, text, limit=limit)


def _read_magic(path: Path) -> bytes | None:
    try:
        with path.open("rb") as handle:
            return handle.read(12) or None
    except OSError:
        return None


def _looks_like_text(path: Path) -> bool:
    """前几 KB 里有没有 NUL 字节。**与 ``agent/fs.py`` 给 ``read_file`` 同一条判据。**"""
    try:
        with path.open("rb") as handle:
            sample = handle.read(SAMPLE_BYTES)
    except OSError:
        return False
    return b"\x00" not in sample


def _read_text(path: Path, *, limit: int) -> str | None:
    try:
        # 多读一点：截断由 :func:`_wrap` 统一做，好让它能报出"后面还有多少"。
        return path.read_text(encoding="utf-8", errors="replace")[: limit * 2 + 1]
    except OSError:
        return None


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} 字节"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
