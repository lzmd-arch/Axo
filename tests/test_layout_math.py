"""窗口与分栏的尺寸算术。

这种算术住在 ``_build_ui`` 里就只能靠换分辨率去试，而**换分辨率试不出来的正是
边界**：屏比窗口的最小尺寸还小、账本展开之后对话区只剩 120 像素、用户拖完分隔条
再展开账本。搬到 :mod:`axo.ui.layout` 之后这些都能逐条断言。

**这里钉得最死的一条是"三项之和恒等于 total"。** 那是 splitter 的硬要求：
不给够的话 Qt 会按自己的逻辑重分，于是"我明明设了 340 宽"和屏幕上的东西对不上，
而那种毛病只能靠肉眼发现。
"""

from __future__ import annotations

import pytest

from axo.ui.layout import (
    CHAT_MIN,
    LEDGER_MIN,
    LEDGER_WIDTH,
    MIN_WINDOW_H,
    MIN_WINDOW_W,
    SESSION_MAX,
    SESSION_MIN,
    initial_window_size,
    plan_pane_sizes,
)

#: 本机实测的可用区（``QScreen.availableGeometry()``，已经是逻辑像素）。拿真实
#: 尺寸当一条用例，是为了让"算出来的窗口在这台机器上到底多大"有一个写下来的答案。
_REAL = (2048, 1104)


# ── 初始窗口尺寸 ────────────────────────────────────────────────────────────


def test_real_screen_gives_a_sane_window() -> None:
    width, height = initial_window_size(*_REAL)
    assert (width, height) == (1638, 883)
    assert width < _REAL[0] and height < _REAL[1]


@pytest.mark.parametrize(
    "avail", [(2048, 1104), (1920, 1080), (1366, 768), (2560, 1440), (3840, 2160)]
)
def test_window_never_exceeds_the_cap(avail: tuple[int, int]) -> None:
    """**绝不超过可用区。** 超出的部分恰恰是标题栏和右下角——用户拖不动也够不着。
    写死尺寸的版本在 1366×768 的笔记本上正是这么出问题的。"""
    width, height = initial_window_size(*avail)
    assert 0 < width <= avail[0]
    assert 0 < height <= avail[1]


@pytest.mark.parametrize("avail", [(900, 600), (1024, 640), (800, 480), (640, 480), (200, 200)])
def test_small_screen_falls_back_to_the_minimum(avail: tuple[int, int]) -> None:
    """屏比窗口最小尺寸还小时**以最小尺寸为准**——宁可让窗口超出屏幕一点，
    也不要给一个连顶栏那几个按钮都挤不下的窗口。"""
    width, height = initial_window_size(*avail)
    assert width == MIN_WINDOW_W
    assert height == MIN_WINDOW_H


@pytest.mark.parametrize("avail", [(0, 0), (-1, -1), (0, -1)])
def test_no_screen_information_gives_the_minimum(avail: tuple[int, int]) -> None:
    """拿不到屏幕信息（无头环境、某些远程桌面）时，最小尺寸是唯一一个不需要
    屏幕就能给出的合理答案。"""
    assert initial_window_size(*avail) == (MIN_WINDOW_W, MIN_WINDOW_H)


def test_the_two_axes_are_independent() -> None:
    """**逐轴处理，不是一个开关。** 拿不到宽度不等于拿不到高度——两轴各自兜底，
    这样"只有一个轴是 0"的畸形输入也能给出一个可用的窗口，而不是整份回落到
    最小值、把那个明明知道的轴也丢掉。"""
    assert initial_window_size(0, 1104) == (MIN_WINDOW_W, 883)
    assert initial_window_size(2048, 0) == (1638, MIN_WINDOW_H)


def test_growth_is_monotone_in_screen_size() -> None:
    """屏越大窗口不会反而越小。听起来是废话，但夹紧的两端写反了就会这样。"""
    sizes = [initial_window_size(w, h)[0] for w, h in [(900, 600), (1280, 720), (1920, 1080)]]
    assert sizes == sorted(sizes)


# ``availableGeometry()`` 返回的就是 ``resize()`` 要的逻辑像素，**不要再乘
# ``devicePixelRatio()``**——乘了之后窗口会按缩放比例超出屏幕，而"最大化按钮被顶出
# 屏幕"这种毛病很难联想到是缩放。125% 缩放下 DPR 是 1.25、可用区是 2048×1104（**不是**
# 除以 1.25 之后的数），这一条写在 :mod:`axo.ui.layout` 的模块注释里；
# 它是取屏幕信息那一行代码的约束，不是这个纯函数能断言的——纯函数收到的已经是
# 像素数，它无从知道那个数是怎么来的。


# ── 分栏 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("total", [900, 1000, 1180, 1280, 1638, 1920, 2560, 3840])
@pytest.mark.parametrize("ledger_open", [False, True], ids=["账本收", "账本开"])
def test_panes_always_add_up(total: int, ledger_open: bool) -> None:
    """**splitter 的硬要求。** 不给够的话 Qt 会自己重分，于是设置里的宽度和
    屏幕上的东西对不上。"""
    sizes = plan_pane_sizes(total, ledger_open=ledger_open)
    assert sum(sizes) == total
    assert all(size >= 0 for size in sizes)


def test_ledger_closed_is_zero_width() -> None:
    assert plan_pane_sizes(1638, ledger_open=False)[2] == 0


def test_ledger_open_uses_the_nominal_width_on_a_big_window() -> None:
    """大窗口下账本就是它那个固定宽度——**固定而不是按比例**：账本是定宽两列的
    文本，一个会自动缩放的列会让它的对齐失效。"""
    session, chat, ledger = plan_pane_sizes(1638, ledger_open=True)
    assert ledger == LEDGER_WIDTH
    assert chat == 1638 - session - ledger
    assert chat >= CHAT_MIN


def test_session_column_is_clamped_at_both_ends() -> None:
    """比例在大屏上会给出 400 多像素的空列，在小屏上会挤到显示不出标题。"""
    assert plan_pane_sizes(4000, ledger_open=False)[0] == SESSION_MAX
    assert plan_pane_sizes(900, ledger_open=False)[0] == SESSION_MIN


def test_ledger_is_clamped_on_a_small_window() -> None:
    """小窗口里 340 会把对话区挤没，所以要按比例压。"""
    _, _, ledger = plan_pane_sizes(900, ledger_open=True)
    assert LEDGER_MIN <= ledger <= LEDGER_WIDTH


def test_chat_gets_the_remainder() -> None:
    session, chat, ledger = plan_pane_sizes(1920, ledger_open=False)
    assert chat == 1920 - session
    assert ledger == 0


def test_chat_borrows_from_the_ledger_when_squeezed() -> None:
    """不够时先从账本借——**账本是查资料时才看的，对话区是这个程序存在的理由**。

    820 这个宽度是挑出来的：按名义比例分到这里，对话区只剩 358（差 2 像素），
    于是账本从 262 让到它的下限 260。范围再宽一点就走不到这条分支，这条断言
    会变成在测一条从来没跑过的路。
    """
    total = 820
    session, chat, ledger = plan_pane_sizes(total, ledger_open=True)
    assert chat == CHAT_MIN
    assert ledger == LEDGER_MIN
    assert session == SESSION_MIN
    assert sum((session, chat, ledger)) == total


def test_borrow_order_is_ledger_then_session() -> None:
    """不够时**先动账本、后动会话列**，而且账本让到底才轮到会话列。

    **要靠 ``chat_min`` 才走得到这条分支。** 按名义宽度算，1638 上对话区有 1003
    像素，而两处各自的余量之和是 95（账本 340→260、会话 295→200）——也就是说
    默认参数下，第二次借用永远不会发生（要让会话列有余量得 ``total > 1111``，
    而要让账本贴到底得 ``total ≤ 812``，两个条件互斥）。所以这里把要求提到
    **1100**：差 97，账本先让满 80，会话列再让 17。

    这条断言的意义正在于此——它测的是参数的**优先级**，而优先级只有把额度压到
    极限才看得出来。
    """
    total = 1638
    session, chat, ledger = plan_pane_sizes(total, ledger_open=True, chat_min=1100)
    assert chat == 1100
    assert ledger == LEDGER_MIN  # 账本让到底
    assert SESSION_MIN < session < 295  # 会话列让了，但没到底
    assert sum((session, chat, ledger)) == total


def test_panes_survive_an_absurdly_small_total() -> None:
    """窗口小到连两条下限都放不下时，**让最小值让步而不是让和不等**。"""
    for total in (0, 1, 50, 100, 200, 400):
        sizes = plan_pane_sizes(total, ledger_open=True)
        assert sum(sizes) == total, total
        assert all(size >= 0 for size in sizes)


def test_ledger_width_argument_is_honoured_within_bounds() -> None:
    assert plan_pane_sizes(1638, ledger_open=True, ledger_width=400)[2] == 400
    # 超过比例上限就压回来
    assert plan_pane_sizes(1638, ledger_open=True, ledger_width=1200)[2] < 1200
