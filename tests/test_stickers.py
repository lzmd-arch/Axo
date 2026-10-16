"""表情包目录（``axo.stickers``）。

只有几件事，但每一件错了都会变成一句"她以为有一张其实没有"：只认图片后缀、顺序确定
（工具描述要稳定，测试不能依赖文件系统的返回次序）、名字写歪也要找得到、空目录不是
错误（还没放是常态）。
"""

from __future__ import annotations

from pathlib import Path

from axo.stickers import STICKER_SUFFIXES, Sticker, catalog_line, find_sticker, stickers_in


def touch(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  # 内容不重要，这里只验"认不认得出"
    return path


# ── 扫 ──────────────────────────────────────────────────────────────────────


def test_one_file_is_one_sticker(tmp_path: Path) -> None:
    touch(tmp_path, "开心.png")
    touch(tmp_path, "无语.jpg")

    found = stickers_in(tmp_path)

    assert [sticker.name for sticker in found] == ["开心", "无语"] or [
        sticker.name for sticker in found
    ] == ["无语", "开心"]
    assert all(sticker.path.parent == tmp_path for sticker in found)


def test_only_image_suffixes_count(tmp_path: Path) -> None:
    """``Thumbs.db`` 与一份说明书都不是表情包——认错了它们会出现在她的话头里。"""
    touch(tmp_path, "开心.png")
    touch(tmp_path, "Thumbs.db")
    touch(tmp_path, "说明.txt")
    touch(tmp_path, "未完成.psd")

    assert [sticker.name for sticker in stickers_in(tmp_path)] == ["开心"]


def test_the_suffix_check_ignores_case(tmp_path: Path) -> None:
    """从手机传过来的照片常常是 ``.JPG``。"""
    touch(tmp_path, "照片.JPG")

    assert [sticker.name for sticker in stickers_in(tmp_path)] == ["照片"]


def test_subdirectories_are_not_stickers(tmp_path: Path) -> None:
    """**只扫一层**：子目录是用户的组织方式，不是表情包。"""
    (tmp_path / "旧的").mkdir()
    touch(tmp_path / "旧的", "里面的.png")

    assert stickers_in(tmp_path) == ()


def test_a_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    """还没放表情包是**常态**——同 ``tts.voices_in`` 对"还没放音色"的姿态。"""
    assert stickers_in(tmp_path / "根本没有这个目录") == ()


def test_the_order_is_deterministic(tmp_path: Path) -> None:
    """同一份目录永远给同一个顺序：工具描述要稳定，测试也不该依赖文件系统的次序。

    **按 ``casefold`` 排**，于是 ``alpha`` 排在 ``Beta`` 前面（大写字母在码位上更小，
    不折叠的话中英文混排的顺序会让人一愣）。

    **"同名只差大小写"那一支测不了**：Windows 的文件系统不区分大小写，``A.png`` 与
    ``a.png`` 是同一个文件。那条判据只在大小写敏感的机器上才有第二个样本。
    """
    for name in ("Gamma", "alpha", "Beta"):
        touch(tmp_path, f"{name}.png")

    first = [sticker.name for sticker in stickers_in(tmp_path)]
    second = [sticker.name for sticker in stickers_in(tmp_path)]

    assert first == second
    assert first == ["alpha", "Beta", "Gamma"]


# ── 找 ──────────────────────────────────────────────────────────────────────


def test_find_returns_the_file_it_was_called_with(tmp_path: Path) -> None:
    """精确命中那一步。

    "先精确、再忽略大小写"这个**顺序**在 Windows 上测不出第二个样本（文件系统不区分
    大小写），它只在大小写敏感的机器上才有意义——而那种机器上目录里真的会有两个只差
    大小写的名字，那正是这个顺序要挡的情况。
    """
    touch(tmp_path, "Sad.png")

    found = find_sticker(tmp_path, "Sad")

    assert found is not None and found.name == "Sad"


def test_find_forgives_a_wrong_case(tmp_path: Path) -> None:
    """模型写名字时大小写写歪是常事，而"差一个字母就用不了"是没必要的一道坎。"""
    touch(tmp_path, "Sad.png")

    found = find_sticker(tmp_path, "sad")

    assert found is not None and found.name == "Sad"


def test_find_strips_the_name_it_was_given(tmp_path: Path) -> None:
    """模型有时会把名字连着空白一起递进来。"""
    touch(tmp_path, "开心.png")

    assert find_sticker(tmp_path, "  开心 ") is not None


def test_find_says_no_to_an_empty_name(tmp_path: Path) -> None:
    """空名字是"没说要发哪张"，不是"找名叫空串的那张"。"""
    touch(tmp_path, "开心.png")

    assert find_sticker(tmp_path, "") is None
    assert find_sticker(tmp_path, "   ") is None


def test_find_says_no_to_a_name_that_is_not_there(tmp_path: Path) -> None:
    touch(tmp_path, "开心.png")

    assert find_sticker(tmp_path, "生气") is None


# ── 送给模型的那一行 ────────────────────────────────────────────────────────


def test_the_catalog_line_lists_names_without_numbers(tmp_path: Path) -> None:
    """**不编号**：编号会让模型以为入参是序号，而这个工具要的是名字。"""
    for name in ("开心", "无语", "震惊"):
        touch(tmp_path, f"{name}.png")

    line = catalog_line(stickers_in(tmp_path))

    assert line == "开心、无语、震惊" or line == "无语、震惊、开心"
    assert "1" not in line


def test_an_empty_directory_gives_an_empty_line(tmp_path: Path) -> None:
    """空串是调用方"不注册这个工具"的判据（同 ``read_file`` 在可读根为空时的姿态）。"""
    assert catalog_line(stickers_in(tmp_path)) == ""


def test_the_uri_is_absolute_and_usable_by_the_web_view(tmp_path: Path) -> None:
    """网页那侧用的是 ``file:`` URL，而它**要求绝对路径**。"""
    path = touch(tmp_path, "开心.png")

    uri = Sticker(name="开心", path=path).as_uri()

    assert uri.startswith("file:")
    # **非 ASCII 会被百分号编码**（``开心`` 变成 ``%E5%BC%80%E5%BF%83``）——那正是
    # ``file:`` URL 该有的样子；这一条钉住的是"编码之后仍然指向那个文件"。
    assert uri.endswith(".png")
    assert "%" in uri, "中文文件名没被编码，说明它不是真的 URI"


def test_only_the_four_listed_suffix_families_are_recognised() -> None:
    """判据只有一处：``STICKER_SUFFIXES``。两处各写一份的话，"扫得到却找不到"会变成
    一种只有用户碰得上的怪事。"""
    assert STICKER_SUFFIXES == (".png", ".jpg", ".jpeg", ".gif", ".webp")
