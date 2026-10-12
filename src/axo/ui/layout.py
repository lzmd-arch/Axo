"""窗口与分栏的尺寸算术：纯函数，**不 import Qt**。

搬出来的理由和 :mod:`axo.ui.color` 一样，只是换了个方向——这里要测的是
"在 1280×720 的屏上窗口开多大""三格 splitter 在 1000 像素里怎么分"。这种算术住在
``_build_ui`` 里就只能靠换分辨率去试，而**换分辨率试不出来的正是边界**：屏比窗口的
最小尺寸还小、账本展开之后对话区只剩 120 像素、用户拖完分隔条再展开账本。放这里就
能逐条断言。

**两个函数都不碰屏幕。** 可用区由调用方取好传进来（``QScreen.availableGeometry()``），
所以这里可以拿一堆假尺寸来跑。调用方那边只有一句要注意：**``availableGeometry()``
返回的就是逻辑像素**，正是 ``resize()`` 要的单位，**不要再乘 ``devicePixelRatio()``**
——乘了之后窗口会按缩放比例超出屏幕，而"最大化按钮被顶出屏幕"这种毛病很难联想到
是缩放。
"""

from __future__ import annotations

__all__ = [
    "CHAT_MIN",
    "GROW",
    "LEDGER_MIN",
    "LEDGER_WIDTH",
    "MIN_WINDOW_H",
    "MIN_WINDOW_W",
    "SESSION_MAX",
    "SESSION_MIN",
    "initial_window_size",
    "plan_pane_sizes",
]

#: 初始窗口取可用区的这个比例。0.80 而不是"能多大就多大"：留一圈边，
#: 用户第一眼就能看到这是个窗口而不是全屏，也就知道它可以被拖动和缩放。
GROW = 0.80

#: 上限比例。**必须小于 1**，否则窗口会与任务栏/贴边区域严丝合缝，
#: 反而更难拖。
CAP = 0.95

#: 窗口最小尺寸。**屏比这还小时以它为准**——宁可让窗口超出屏幕一点，
#: 也不要给一个连顶栏按钮都挤不下的窗口。
MIN_WINDOW_W = 900
MIN_WINDOW_H = 600

#: 会话列。比例 0.18 是"够放 8 个汉字标题"的宽度；上下夹紧是因为
#: 比例在大屏上会给出 400 多像素的空列，在小屏上会挤到显示不出标题。
SESSION_RATIO = 0.18
SESSION_MIN = 200
SESSION_MAX = 320

#: 账本列的名义宽度。**固定而不是按比例**：账本是定宽两列的文本，
#: 一个会自动缩放的列会让它的对齐失效——而参差不齐的账本比没有账本更难看懂。
LEDGER_WIDTH = 340
LEDGER_MIN = 260
#: 账本最多占这个比例，免得在小窗口里把对话区挤没。
LEDGER_MAX_RATIO = 0.32

#: 对话区的最小宽度。**这是三个面板里唯一不能退让的那个**——它是这个程序
#: 存在的理由，另两个都是查资料用的。所以不够时先从账本借，再从会话列借。
CHAT_MIN = 360


def initial_window_size(
    avail_w: int,
    avail_h: int,
    *,
    grow: float = GROW,
    cap: float = CAP,
    min_w: int = MIN_WINDOW_W,
    min_h: int = MIN_WINDOW_H,
) -> tuple[int, int]:
    """可用区 → 初始窗口尺寸。逐轴 ``夹在 [最小, 可用区*cap]`` 内。

    这样写而不是"写死 1180×760"，是因为写死的尺寸在两种屏上都会出问题：4K 屏上
    是一个缩在角落里的小窗口，1366×768 的笔记本上则**超出屏幕**——而超出的部分
    恰恰是标题栏和右下角，用户拖不动也够不着。

    ``avail_w <= 0``（拿不到屏幕信息，某些远程桌面/无头环境）时返回最小尺寸：
    这是唯一一个不需要屏幕就能给出的合理答案。
    """

    def axis(avail: int, minimum: int) -> int:
        if avail <= 0:
            return minimum
        wanted = round(avail * grow)
        ceiling = max(minimum, int(avail * cap))
        return max(minimum, min(ceiling, wanted))

    return axis(avail_w, min_w), axis(avail_h, min_h)


def plan_pane_sizes(
    total: int,
    *,
    ledger_open: bool = False,
    ledger_width: int = LEDGER_WIDTH,
    chat_min: int = CHAT_MIN,
) -> tuple[int, int, int]:
    """总宽 → 三格 splitter 的尺寸 ``(会话, 对话, 账本)``。**三项之和恒等于 total。**"""
    total = max(0, int(total))

    session = _clamp(round(total * SESSION_RATIO), SESSION_MIN, SESSION_MAX)
    ledger = 0
    if ledger_open:
        # 上界用 max(LEDGER_MIN, ...)：小窗口下 total*0.32 会小于 LEDGER_MIN，
        # 直接夹会得到一个反过来的区间。先保证下限，再让下面的"借"去处理不够。
        ledger = _clamp(
            int(ledger_width), LEDGER_MIN, max(LEDGER_MIN, round(total * LEDGER_MAX_RATIO))
        )

    # 对话区拿余量。不够就借：**先账本、后会话**——账本是查资料时才看的，
    # 会话列则是切换上下文的主路径，但它至少还能靠滚动兜住。
    deficit = chat_min - (total - session - ledger)
    if deficit > 0 and ledger > LEDGER_MIN:
        take = min(deficit, ledger - LEDGER_MIN)
        ledger -= take
        deficit -= take
    if deficit > 0 and session > SESSION_MIN:
        take = min(deficit, session - SESSION_MIN)
        session -= take
        deficit -= take

    # 窗口小到连两条下限都放不下时（正常路径下不会：窗口最小 900），
    # **让最小值让步，而不是让和不等**。和恒等于 total 是 splitter 的硬要求，
    # 不满足的话 Qt 会按自己的逻辑重分，那才是真的不可预测。
    #
    # 账本先夹：它有可能自己就超过 total（``total=0``、账本却按 LEDGER_MIN
    # 定格在 260），不先夹的话下面那两行的差值会是负的，最后三项和比 total 还大。
    ledger = max(0, min(ledger, total))
    session = max(0, min(session, total - ledger))
    chat = max(0, total - session - ledger)
    return session, chat, ledger


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
