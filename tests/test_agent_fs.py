"""``read_file`` 的边界。

这是 P3a 里唯一一处**模型给的字符串直接变成读盘调用**的地方，所以每一类边界都
有一条指名道姓的测试。它们全都是"负面"测试——正面那条（"能读到文件")一条就够，
而边界漏一个，那个漏掉的口子不会有任何症状：读盘照样成功，只是读到了不该读的
地方。

**测试里不放任何真文件内容**——全部是 ``tmp_path`` 里现造的。这不仅是为了干净，
也是因为仓库卫生那条纪律（``tests/test_repo_hygiene.py``）管着测试目录：往
``tests/fixtures/`` 里放一个"示例文件"就有机会放成别人的真文件。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from axo.agent.fs import (
    DEFAULT_MAX_LINES,
    MAX_READ_BYTES,
    ReadFileError,
    WriteFileError,
    list_entries,
    read_text,
    resolve_for_write,
    resolve_within,
    search_text,
    write_text_atomic,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    area = tmp_path / "notes"
    area.mkdir()
    (area / "备忘.txt").write_text("第一行\n第二行\n第三行", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("这个文件不该被读到", encoding="utf-8")
    return area


# ── 目录边界 ───────────────────────────────────────────────────────────────


def test_reads_a_file_inside_the_root(root: Path) -> None:
    assert resolve_within([root], "备忘.txt") == (root / "备忘.txt").resolve()


def test_nested_relative_path_is_allowed(root: Path) -> None:
    nested = root / "子目录"
    nested.mkdir()
    (nested / "深.txt").write_text("x", encoding="utf-8")
    assert resolve_within([root], "子目录/深.txt").name == "深.txt"


def test_dotdot_escape_is_rejected(root: Path) -> None:
    with pytest.raises(ReadFileError, match="不在允许读取的目录里"):
        resolve_within([root], "../secret.txt")


def test_absolute_path_outside_the_root_is_rejected(root: Path, tmp_path: Path) -> None:
    """根**之外**的绝对路径被拒。

    不变量是"解析后的路径必须落在某个允许目录里"，不是"输入必须是相对路径"——
    所以根**之内**的绝对路径是放行的（它没越过任何边界），而这条测试钉的是真正
    的那道闸。第一版把这两件事混在一起写成"绝对路径一概拒绝"，结果它红了：
    而红的原因是**测试写错了**，不是代码错了。那种红最值得留意，因为"改测试让它
    变绿"和"改代码让它变绿"在这里长得一模一样。
    """
    with pytest.raises(ReadFileError, match="不在允许读取的目录里"):
        resolve_within([root], str(tmp_path / "secret.txt"))


def test_absolute_path_inside_the_root_is_allowed(root: Path) -> None:
    """根之内的绝对路径放行——这不是漏洞，是同一个不变量在另一个方向的取值。"""
    assert resolve_within([root], str(root / "备忘.txt")) == (root / "备忘.txt").resolve()


def test_prefix_trap_database_is_not_data(tmp_path: Path) -> None:
    """``/data`` 不能匹配 ``/database``。

    这是 ``str.startswith`` 那个安静的经典错误。用 ``is_relative_to`` 按路径分量比
    就没有它——而"没有它"必须有条测试，否则某天有人为了"顺手"把判断改写成字符串
    前缀比较时，没有任何东西会响。
    """
    data = tmp_path / "data"
    database = tmp_path / "database"
    data.mkdir()
    database.mkdir()
    (database / "x.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ReadFileError):
        resolve_within([data], "../database/x.txt")


def test_multiple_roots_are_all_checked(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / "在第二个根里.txt").write_text("y", encoding="utf-8")

    found = resolve_within([first, second], "在第二个根里.txt")
    assert found == (second / "在第二个根里.txt").resolve()


def test_no_roots_means_disabled(tmp_path: Path) -> None:
    """没配目录 = 工具不可用，**不是**给一个默认目录。

    "没配"与"用默认值"是两件事。把后者塞进前者，用户就从没做过授权决定却拿到了
    一个可读目录——而且他不会知道。
    """
    with pytest.raises(ReadFileError, match="没有配置可读目录"):
        resolve_within((), "备忘.txt")


@pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
def test_empty_path_is_rejected(root: Path, raw: str) -> None:
    with pytest.raises(ReadFileError, match="path 参数是空的"):
        resolve_within([root], raw)


@pytest.mark.skipif(sys.platform != "win32", reason="大小写不敏感是 Windows 路径的性质")
def test_case_insensitive_on_windows(root: Path) -> None:
    """Windows 上大小写不同仍是同一个根。

    钉的是**行为**，不是机制：``PureWindowsPath`` 的比较本来就是大小写不敏感的，
    所以这里没有写 ``os.path.normcase``（那会是一句不起作用、却看起来在承重的
    代码）。这条测试保证下一台机器上它仍然成立。
    """
    upper = Path(str(root).upper())
    assert resolve_within([upper], "备忘.txt").is_relative_to(root.resolve())


def _link_directory(target: Path, link: Path) -> bool:
    """在 ``link`` 处造一个指向 ``target`` 的目录链接。造不出来返回 ``False``。

    Windows 上 ``Path.symlink_to`` 需要开发者模式或管理员权限（本机实测
    ``WinError 1314 客户端没有所需的特权``），而**目录联接（junction）不需要**。
    多这一条分支是值得的：否则这条测试在本机永远是 skip，而一个永远 skip 的
    安全测试和一个不存在的安全测试没有区别——它俩都不会红。
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if sys.platform != "win32":
        return False
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0


def test_link_escape_is_rejected(root: Path, tmp_path: Path) -> None:
    """根目录里一个指向外面的链接，解析后就不在根里了。

    ``resolve()`` 跟随链接（符号链接与 Windows 的目录联接都跟随），所以这个逃逸
    **天然被覆盖**——但"天然"的东西最容易在重构里被去掉：谁把 ``resolve()`` 删了，
    不会有任何别的测试红。
    """
    outside = tmp_path / "外面"
    outside.mkdir()
    (outside / "secret.txt").write_text("不该被读到", encoding="utf-8")
    if not _link_directory(outside, root / "外链"):
        pytest.skip("这个环境既不允许符号链接也不允许目录联接")

    with pytest.raises(ReadFileError, match="不在允许读取的目录里"):
        resolve_within([root], "外链/secret.txt")


# ── 读内容 ─────────────────────────────────────────────────────────────────


def test_reads_whole_small_file(root: Path) -> None:
    body = read_text(root / "备忘.txt")
    assert body == "第一行\n第二行\n第三行"
    assert "已截断" not in body


def test_byte_limit_leaves_a_marker(root: Path, tmp_path: Path) -> None:
    """超限**明说**截断，并给出文件总大小。

    静默截断比报错更坏：模型读到半个文件时不知道那是半个，它会基于半个文件下结论，
    而那个结论在界面上看起来与完整的一样确定。
    """
    big = tmp_path / "大.txt"
    big.write_text("啊" * 100, encoding="utf-8")  # 300 字节
    body = read_text(big, max_bytes=30)
    assert "已截断" in body
    assert "300 字节" in body


def test_line_limit_names_the_next_start_line(root: Path) -> None:
    """按行截断时告诉模型**下一行从哪儿开始**——不给这个数，它只能从头再读一遍。"""
    many = root / "多行.txt"
    many.write_text("\n".join(f"第{i}行" for i in range(1, 51)), encoding="utf-8")
    body = read_text(many, max_lines=10)
    assert "第10行" in body
    assert "第11行" not in body
    assert "start_line=11" in body
    assert "还有 40 行" in body


def test_start_line_pages_through(root: Path) -> None:
    many = root / "多行.txt"
    many.write_text("\n".join(f"第{i}行" for i in range(1, 51)), encoding="utf-8")
    body = read_text(many, start_line=11, max_lines=10)
    assert body.startswith("第11行")
    assert "第10行" not in body


def test_start_line_beyond_window_is_an_error(root: Path) -> None:
    with pytest.raises(ReadFileError, match="超出了可读窗口"):
        read_text(root / "备忘.txt", start_line=99)


def test_start_line_zero_is_an_error(root: Path) -> None:
    with pytest.raises(ReadFileError, match="从 1 开始数"):
        read_text(root / "备忘.txt", start_line=0)


def test_truncation_does_not_break_a_multibyte_character(tmp_path: Path) -> None:
    """截断正好切在一个多字节字符中间时**不许报编码错**。

    那是我们自己切坏的，不是文件坏了。增量解码器在 ``final=False`` 时把不完整的
    尾字节留在缓冲里，这个区别就自动分开了——而它值得一条测试，因为一个朴素的
    ``head.decode("utf-8")`` 在这里会抛 ``UnicodeDecodeError``，然后一句"这个文件
    不是 UTF-8"会被回灌给模型，而文件明明是好的。
    """
    path = tmp_path / "截断.txt"
    path.write_text("a" * 10 + "中" * 5, encoding="utf-8")  # 10 + 15 字节
    body = read_text(path, max_bytes=11)  # 正好切在第一个"中"的中间
    assert body.startswith("a" * 10)
    assert "不是 UTF-8" not in body


def test_binary_file_is_rejected(tmp_path: Path) -> None:
    """含 NUL 字节 → 拒绝。比扩展名黑名单可靠：``.txt`` 里可以是任何东西。"""
    path = tmp_path / "伪装.txt"
    path.write_bytes(b"PK\x03\x04\x00\x00\x00\x00binary")
    with pytest.raises(ReadFileError, match="二进制"):
        read_text(path)


def test_invalid_utf8_is_rejected_not_replaced(tmp_path: Path) -> None:
    """坏编码 → 报错，**不做替换**。

    ``errors="replace"`` 会把一个 U+FFFD 放进正文，而模型会把它当成文件里真的有的
    字符读下去——假内容和真内容在上下文里长得一模一样。
    """
    path = tmp_path / "乱码.txt"
    path.write_bytes(b"abc\xff\xfedef")
    with pytest.raises(ReadFileError, match="不是 UTF-8"):
        read_text(path)


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ReadFileError, match="文件不存在"):
        read_text(tmp_path / "没有这个.txt")


def test_directory_is_an_error(root: Path) -> None:
    with pytest.raises(ReadFileError, match="不是一个文件"):
        read_text(root)


# ── 常数的自检 ─────────────────────────────────────────────────────────────


def test_limits_are_sane() -> None:
    """上限必须是正数且互相独立。

    一条看起来傻但值得的断言：把 ``MAX_READ_BYTES`` 改成 0（或负数）时，
    ``handle.read(max_bytes + 1)`` 仍然能跑，读出来一个空串——于是工具**静默地
    什么都读不到**，而所有"读到了内容"的测试仍然绿，因为它们用的是默认值。
    """
    assert MAX_READ_BYTES > 0
    assert DEFAULT_MAX_LINES > 0


# ── 写：边界、原子性与上限 ──────────────────────────────────────────────────


def test_write_refuses_when_no_root_is_configured() -> None:
    """没配目录 = 工具不可用，**不猜一个默认目录**（同读的那条纪律）。"""
    with pytest.raises(WriteFileError, match="没有配置可写目录"):
        resolve_for_write([], "a.txt")


def test_write_resolves_inside_the_root(root: Path) -> None:
    target = resolve_for_write([root], "新文件.txt")

    assert target == (root / "新文件.txt").resolve()


def test_write_refuses_an_escape(root: Path) -> None:
    with pytest.raises(WriteFileError, match="不在允许写入的目录里"):
        resolve_for_write([root], "../secret.txt")


def test_write_refuses_the_data_root_even_when_it_is_under_a_root(tmp_path: Path) -> None:
    """数据根可能就在读取根底下；**deny 比"在根里"更严格**。"""
    area = tmp_path / "work"
    data = area / "data"
    (data / "memory").mkdir(parents=True)

    with pytest.raises(WriteFileError, match="数据目录"):
        resolve_for_write([area], "data/memory/mem.db", deny=[data])


def test_write_refuses_a_missing_parent_unless_asked(root: Path) -> None:
    with pytest.raises(WriteFileError, match="不存在"):
        resolve_for_write([root], "新的/子目录/文件.txt")

    target = resolve_for_write([root], "新的/子目录/文件.txt", create_dirs=True)
    assert target.parent.name == "子目录"


def test_write_is_atomic_and_leaves_no_temporary_file(root: Path) -> None:
    target = root / "备忘.txt"

    write_text_atomic(target, "换掉\n")
    write_text_atomic(target, "再换掉\n")

    assert target.read_text(encoding="utf-8") == "再换掉\n"
    assert not list(root.glob(".axo-*.tmp")), "临时文件没清掉"


def test_write_refuses_to_overwrite_a_directory(root: Path) -> None:
    with pytest.raises(WriteFileError, match="目录"):
        write_text_atomic(root, "x")


def test_append_counts_the_existing_size(root: Path) -> None:
    target = root / "日志.txt"
    write_text_atomic(target, "12345")

    with pytest.raises(WriteFileError, match="超过"):
        write_text_atomic(target, "67890", append=True, max_bytes=8)

    write_text_atomic(target, "678", append=True, max_bytes=8)
    assert target.read_text(encoding="utf-8") == "12345678"


def test_write_rejects_oversized_content(root: Path) -> None:
    with pytest.raises(WriteFileError, match="超过单次上限"):
        write_text_atomic(root / "大.txt", "x" * 10, max_bytes=5)


@pytest.mark.skipif(sys.platform != "win32", reason="保留设备名只在 Windows 上有意义")
def test_write_refuses_windows_reserved_names(root: Path) -> None:
    with pytest.raises(WriteFileError, match="保留设备名"):
        resolve_for_write([root], "CON.txt")
    with pytest.raises(WriteFileError, match="冒号"):
        resolve_for_write([root], "a:b.txt")


# ── 列目录 ─────────────────────────────────────────────────────────────────


def test_list_entries_puts_directories_first_and_reports_the_total(tmp_path: Path) -> None:
    area = tmp_path / "清单"
    (area / "子目录").mkdir(parents=True)
    for name in ("b.txt", "a.txt"):
        (area / name).write_text("x", encoding="utf-8")

    names, total = list_entries(area, limit=2)

    assert total == 3
    assert names[0] == "子目录/"
    assert len(names) == 2


def test_list_entries_on_a_file_is_one_entry(tmp_path: Path) -> None:
    file = tmp_path / "一个.txt"
    file.write_text("x", encoding="utf-8")

    assert list_entries(file) == (["一个.txt"], 1)


def test_list_entries_reports_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ReadFileError, match="不存在"):
        list_entries(tmp_path / "没有")


# ── 全文搜索 ───────────────────────────────────────────────────────────────


def test_search_finds_lines_and_is_case_insensitive_by_default(root: Path) -> None:
    (root / "笔记.txt").write_text("第一行\nAlpha 在这里\n第三行", encoding="utf-8")

    hits = search_text(root, "alpha")

    assert len(hits) == 1
    assert hits[0].path.name == "笔记.txt"
    assert hits[0].line_no == 2
    assert "Alpha" in hits[0].line


def test_search_respects_case_when_asked(root: Path) -> None:
    (root / "笔记.txt").write_text("Alpha", encoding="utf-8")

    assert search_text(root, "alpha", case_sensitive=True) == []


def test_search_skips_binary_files(root: Path) -> None:
    (root / "二进制.bin").write_bytes(b"alpha\x00alpha")

    assert search_text(root, "alpha") == []


def test_search_rejects_a_bad_regex(root: Path) -> None:
    with pytest.raises(ReadFileError, match="正则"):
        search_text(root, "(")


def test_search_stops_at_the_result_cap(root: Path) -> None:
    (root / "很多.txt").write_text("命中\n" * 10, encoding="utf-8")

    hits = search_text(root, "命中", max_results=3)

    assert len(hits) == 3


def test_search_does_not_follow_a_symlink_out_of_the_root(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "外面.txt"
    outside.write_text("alpha", encoding="utf-8")
    link = root / "链接.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("这个环境建不了符号链接")

    assert search_text(root, "alpha") == []
