"""用户拖进来的文件（``axo.attachments``）。

这个模块的全部价值在于**别把不该进上下文的东西放进去**：一个改名成 ``.txt`` 的
可执行文件、一份两百兆的日志、一个扫描件 PDF。所以断言分两族——判得准、以及
判不准时**说得出为什么**（返回空串是最坏的一种：模型会以为文件是空的）。
"""

from __future__ import annotations

from pathlib import Path

from axo.attachments import (
    IMAGE_SUFFIXES,
    MAX_TEXT_CHARS,
    TEXT_SUFFIXES,
    attachment_for,
    body_of,
    classify,
)

PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ── 判类型 ──────────────────────────────────────────────────────────────────


def test_the_magic_wins_over_the_suffix(tmp_path: Path) -> None:
    """**一个改名成 ``.txt`` 的 PNG 仍然是 PNG。**

    反过来（按后缀信）的后果是：那张图会被当文本读，读出来是一坨乱码，而乱码会原样
    进上下文——用云端模型时就是发到了别人的服务器上。
    """
    path = write(tmp_path, "伪装.txt", PNG_MAGIC + b"rest")

    assert classify(path) == "image"


def test_a_pdf_is_recognised_by_its_magic_too(tmp_path: Path) -> None:
    assert classify(write(tmp_path, "文档.pdf", b"%PDF-1.7\n...")) == "pdf"


def test_a_webp_is_an_image_but_other_riff_files_are_not(tmp_path: Path) -> None:
    """``RIFF`` 是容器：wav 与 avi 也是它。**多看一眼第 8-12 字节**才分得出来。"""
    webp = write(tmp_path, "图.webp", b"RIFF" + bytes(4) + b"WEBP" + b"x")
    wav = write(tmp_path, "声音.wav", b"RIFF" + bytes(4) + b"WAVE" + b"x")

    assert classify(webp) == "image"
    assert classify(wav) == "other"


def test_a_plain_text_file_is_text(tmp_path: Path) -> None:
    for name in ("笔记.md", "脚本.py", "数据.csv"):
        assert classify(write(tmp_path, name, "你好，这是一段文本。".encode())) == "text"


def test_a_binary_with_a_text_suffix_is_not_text(tmp_path: Path) -> None:
    """**前几 KB 里出现 NUL 就不算文本**——与 ``agent/fs.py`` 给 ``read_file``
    同一条判据。少了它，一个没有魔数、后缀又骗人的二进制会被原样读进来。"""
    path = write(tmp_path, "其实是二进制.txt", b"abc\x00def")

    assert classify(path) == "other"


def test_an_unknown_suffix_is_other(tmp_path: Path) -> None:
    """**判不出就是判不出**：``other`` 只报身份，不读内容。"""
    assert classify(write(tmp_path, "安装包.exe", b"MZ\x90\x00")) == "other"
    assert classify(write(tmp_path, "照片.raw", b"II*\x00")) == "other"


def test_a_directory_or_a_missing_file_is_none(tmp_path: Path) -> None:
    """拖进来的东西是什么都不奇怪（快捷方式、正在被写的文件）——**一律返回 None，
    不抛**：拖一个文件进来让程序炸了是最不该有的一种失败。"""
    assert attachment_for(tmp_path) is None
    assert attachment_for(tmp_path / "没有这个文件.txt") is None


def test_attachment_carries_only_the_file_name(tmp_path: Path) -> None:
    """给模型看的身份是**文件名**，不含路径——路径是本机的事。"""
    path = write(tmp_path, "笔记.md", "正文".encode())

    found = attachment_for(path)

    assert found is not None
    assert found.name == "笔记.md"
    assert str(tmp_path) not in found.name
    assert found.size == len("正文".encode())


# ── 抽正文 ──────────────────────────────────────────────────────────────────


def test_a_text_file_gives_its_content_with_a_byline(tmp_path: Path) -> None:
    path = write(tmp_path, "笔记.md", "路由器在书房。".encode())

    body = body_of(attachment_for(path))  # type: ignore[arg-type]

    assert "笔记.md" in body
    assert "路由器在书房。" in body


def test_a_long_file_is_clipped_and_says_so(tmp_path: Path) -> None:
    """截断**必须说出来**：不说的话模型会以为自己读完了整份文件。"""
    path = write(tmp_path, "长日志.log", ("x" * (MAX_TEXT_CHARS + 5000)).encode())

    body = body_of(attachment_for(path))  # type: ignore[arg-type]

    assert body.count("x") == MAX_TEXT_CHARS
    assert "还有 5000 个" in body


def test_an_image_gives_a_line_and_does_not_pretend_to_have_text(tmp_path: Path) -> None:
    """图走视觉通道，这段正文里只留一句身份——**它不该把二进制当文本塞进来**。"""
    path = write(tmp_path, "照片.png", PNG_MAGIC + bytes(2048))

    body = body_of(attachment_for(path))  # type: ignore[arg-type]

    assert "照片.png" in body
    assert "附了一张图" in body


def test_an_unknown_type_says_it_cannot_be_read(tmp_path: Path) -> None:
    """**说得出为什么**，而不是返回空串——空串会让模型以为文件是空的。"""
    path = write(tmp_path, "安装包.exe", b"MZ\x90\x00" + bytes(1024))

    body = body_of(attachment_for(path))  # type: ignore[arg-type]

    assert "安装包.exe" in body
    assert "读不出内容" in body


def test_a_pdf_without_a_parser_says_so_instead_of_lying(tmp_path: Path) -> None:
    """这个环境没装 PDF 解析器时，**如实说没装**——不假装它是空的、也不崩。"""
    path = write(tmp_path, "文档.pdf", b"%PDF-1.7\n" + bytes(64))

    body = body_of(attachment_for(path))  # type: ignore[arg-type]

    assert "文档.pdf" in body
    assert "PDF" in body and ("没装" in body or "读不出来" in body or "扫描件" in body)


def test_the_two_suffix_sets_do_not_overlap() -> None:
    """一个后缀只能属于一族。重叠的话判类型就成了看谁先写——而那种顺序没人会去核。"""
    assert not set(TEXT_SUFFIXES) & set(IMAGE_SUFFIXES)
