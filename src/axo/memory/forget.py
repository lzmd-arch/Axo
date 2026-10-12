"""遗忘与恢复：``tier=COLD`` 加一条写在 ``meta`` 里的回程票。

**遗忘不是删除。** 用户说"忘掉这个"时，他要的是"别再用它回答我"，而不是"我的
数据没了"。这两件事在本项目里被刻意做成两个动作：遗忘改 ``tier``（可逆、可核对），
彻底删除走 :meth:`~memcore.store.MemoryStore.delete`（不可逆、要二次确认）。

**原层级记在 ``meta`` 里，不加新字段。** 恢复需要知道"忘掉之前它属于哪一层"，
否则所有被遗忘的记忆恢复后都会退化成同一层。三个候选——给 ``MemoryItem`` 加一个
``forgotten_from`` 字段、建一张遗忘快照表、写进 ``meta``——选 ``meta`` 的理由是
三条它都占：JSON 可表示、随条目走（删条目它自然消失，不需要清理）、且**不进
``__eq__``**（``meta`` 是自由格式的，比较它没有意义）。加字段要动存储 schema，
而存储层的改动是静默数据损坏的高发区。

``meta`` 的形状是 ``{"forgotten": {"previous_tier": "longterm", "at": 1759...}}``。
:func:`read_forget_record` 对**任何**不合形状的内容都返回 ``None`` 而不是抛异常：
这些字节可能来自以后某个版本的我们，也可能被外部工具改过，而一个读不回来的
回程票的后果只是"恢复后落到默认层"，不该让整条记忆变得打不开。

**``soft_forgotten`` 不检查 ``pinned``**，这是刻意的——理由见它的文档字符串。
"钉住的条目不许软遗忘"这条规则由 :func:`can_soft_forget` 表达，界面问它、按钮
照它置灰。规则在核心腿里，不在控件的 ``setEnabled`` 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memcore import MemoryItem, Tier

__all__ = [
    "FORGET_KEY",
    "ForgetRecord",
    "can_soft_forget",
    "read_forget_record",
    "restored",
    "soft_forgotten",
]

FORGET_KEY = "forgotten"
"""``meta`` 里那个键。写成常量，因为读它和写它必须用同一个字符串。"""


@dataclass(frozen=True)
class ForgetRecord:
    """一条遗忘记录：忘掉之前它是哪一层、什么时候忘的。"""

    previous_tier: Tier
    at: float

    def __post_init__(self) -> None:
        if self.previous_tier is Tier.COLD:
            # 记成"遗忘前的层级是 COLD"等于把回程票写成了单程票：恢复之后它还在
            # 冷层。这个值没有任何一条正常路径写得出来（``soft_forgotten`` 已经
            # 挡住了），所以它出现就说明 meta 是手动改过的或被写坏了。
            raise ValueError(
                "``previous_tier`` 不能是 COLD——那样恢复之后条目仍然在冷层，回程票等于没写。"
            )


def read_forget_record(item: MemoryItem) -> ForgetRecord | None:
    """读这条记忆的遗忘记录。读不到（或读不懂）返回 ``None``。

    **不抛异常**：``meta`` 对 :mod:`memcore` 是不透明的，我们没有资格要求它装
    的一定是我们写进去的东西。读不懂的后果由调用方承担——恢复时落到默认层并在
    界面上说明"没有记录原层级"，比让详情面板整条打不开好得多。
    """
    raw: Any = item.meta.get(FORGET_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        previous_tier = Tier(raw["previous_tier"])
    except (KeyError, ValueError):
        return None
    if previous_tier is Tier.COLD:
        return None
    at = raw.get("at")
    return ForgetRecord(
        previous_tier=previous_tier,
        at=float(at) if isinstance(at, (int, float)) else 0.0,
    )


def can_soft_forget(item: MemoryItem) -> str | None:
    """这条记忆能不能软遗忘。能就返回 ``None``，不能返回一句理由。

    **钉住的条目不能软遗忘**，理由是 :func:`memcore.recall.recall` 的扫描顺序：
    它在 tier 判定**之前**就跳过 pinned 条目（``if item.pinned or ...: continue``）。
    于是软遗忘一条钉住的记忆是个**无声的空动作**——它压根不走召回，所以"忘掉它"
    什么也没改变；更糟的是它会从账本里彻底消失（既不 injected 也不 rejected），
    界面于是显示"已遗忘"而它仍然在每一轮的上下文里。

    人格锚是这条规则最要紧的一例：:func:`~axo.memory.persona.persona_texts`
    只按 ``kind`` 过滤、**不看 tier**，所以软遗忘人格锚连 L0 都拦不住。

    要让它真的消失，只有彻底删除这一条路——所以界面上这个按钮置灰时，提示语要
    指向删除，而不是留一个点了没反应的按钮。
    """
    if item.pinned:
        return (
            "钉住的条目不走常规召回，忘掉它对它没有任何效果——它仍会在每一轮的"
            "上下文里。要让它消失，请用「彻底删除」。"
        )
    if item.tier is Tier.COLD:
        return "它已经被遗忘了。"
    return None


def soft_forgotten(item: MemoryItem, *, at: float) -> MemoryItem:
    """返回一条已软遗忘（``tier=COLD``）的副本，并把原层级记进 ``meta``。

    **重复调用是幂等的**：已经是 ``COLD`` 就原样退回，绝不重写 ``previous_tier``。
    这条不变量值得单独写出来，因为违反它的后果是静默的——第二次遗忘若把
    ``previous_tier`` 记成了 ``COLD``，那么"恢复"会把条目从冷层恢复到冷层，
    界面上显示成功、条目也确实变了（``meta`` 少了一个键），但它再也不会回来。
    而用户点两次遗忘是很自然的事。

    ``pinned`` **不在这里拦**（拦它的规则在 :func:`can_soft_forget`）：这个函数的
    语义是"把 tier 写成 COLD 并留回程票"，它对任何条目都做得到、且做的事都如实。
    加一道检查会让"钉住条目被软遗忘后仍走 L0"这个已知语义变得无法测试，而那条
    语义正是"为什么界面上必须禁掉这个按钮"的证据。
    """
    if item.tier is Tier.COLD:
        return item
    meta = dict(item.meta)
    meta[FORGET_KEY] = {"previous_tier": item.tier.value, "at": at}
    return item.copy(tier=Tier.COLD, meta=meta)


def restored(item: MemoryItem) -> tuple[MemoryItem, bool]:
    """返回 ``(恢复后的副本, 是否读到了原层级)``。

    没有记录（或记录坏了）时落回 :attr:`~memcore.Tier.WORKING`，第二个返回值为
    ``False``——界面据此说明"没有记录原层级，已恢复到工作层"。**猜一个层级然后
    不说**是这里唯一不能做的事。

    无论有没有读到记录，``meta`` 里的那个键都会被删掉：恢复之后它就不是一条
    "被遗忘过"的记忆了，留着会让下一次 :func:`read_forget_record` 读到一个过期
    的回程票（而它指向的是**更早那次**遗忘之前的层级）。
    """
    record = read_forget_record(item)
    meta = dict(item.meta)
    meta.pop(FORGET_KEY, None)
    tier = record.previous_tier if record is not None else Tier.WORKING
    return item.copy(tier=tier, meta=meta), record is not None
