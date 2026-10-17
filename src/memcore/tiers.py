"""层级升降：**记忆不该永远停在 working**。

真库里的数（2026-10-17）：141 条记忆里 **139 条是 ``working``**（半衰期 0.08 天 ≈ 两小时），
其中 **42 条权重已经衰减到 0**。那个五档半衰期（0.08 / 10 / 100 / 1000 天）的设计**一步都
没走过**——写入侧把每条都写成 working，而没有任何东西把它送上去。

这一层补的就是那一步。判据刻意保守，理由在这一族里反复出现：**让它升错的代价是一次性的
（一条不太重要的记忆活得久一点），让它降错的代价是它开始快速淡出**。

* **升**：``reinforce_count >= 2`` —— 睡眠回放两次都认得出自己，说明它不是一次性的。
  一级一级地升，**封顶 LONGTERM**；不自动进 ``ARCHIVE``（那是"长期不用但确实重要"的
  位置，该由人说，不由算法猜）。
* **降**：目前在 ``LONGTERM``、30 天没被访问、而且总共只被访问过不到 2 次——说明它其实
  没人用过。降到 ``RECENT``，**不降到 WORKING**：那一层的两小时半衰期是给"这次会话里
  临时记的"设计的，把一个 30 天没人用的记忆扔进去等于判它死刑——它该慢慢淡，不是立刻消失。
* **不碰**：钉住的、人格锚、已遗忘的（``COLD``）、``ARCHIVE`` 里的、以及**这一轮刚被
  回放加固过的**（它这一轮显然是活的，不该同时被判"没人用"）。

**为什么升降算"记账"而不是"改记忆"**：``tier`` 与 ``weight``/``access_count`` 同一族——
它决定这条记忆的半衰期多长、该不该被召回，不改正文、不删、不合并。用户拍板的口径是
"正文不动、记账照写"，``tier`` 在记账那一侧。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .types import MemoryItem, Tier

__all__ = [
    "DEMOTE_AFTER_DAYS",
    "DEMOTE_BELOW_ACCESS",
    "PROMOTE_AT",
    "TierMove",
    "plan_tier_moves",
]

#: 回放认出自己几次才升一层。**2 次**：一次可能只是"这一轮的注意力刚好落在那儿"，
#: 两次才像"它确实反复被用上"。
PROMOTE_AT = 2

#: 多久没被访问才算"没人用"。30 天：比任何一次短期的使用波动都长。
DEMOTE_AFTER_DAYS = 30.0

#: 少于这个访问次数的才可能被降。它与上一条**是"且"**：只满足一条都不降。
DEMOTE_BELOW_ACCESS = 2

#: 能往上升的层。``LONGTERM`` 封顶——见模块文档。
_PROMOTABLE: tuple[Tier, ...] = (Tier.WORKING, Tier.RECENT, Tier.LONGTERM)

#: 能往下降的层。只有 ``LONGTERM`` 会动（见模块文档那条"不降到 WORKING"）。
_DEMOTABLE: tuple[Tier, ...] = (Tier.LONGTERM,)


@dataclass(frozen=True)
class TierMove:
    """一次层级变更。"""

    memory_id: str
    from_tier: Tier
    to_tier: Tier

    @property
    def decision(self) -> str:
        """ "promote" / "demote"——事件行里那个值。"""
        return "promote" if _rank(self.to_tier) > _rank(self.from_tier) else "demote"

    def describe(self) -> str:
        return f"{self.from_tier.value} → {self.to_tier.value}"


def plan_tier_moves(
    items: Sequence[MemoryItem],
    *,
    now: float,
    reinforced: Iterable[str] = (),
    promote_at: int = PROMOTE_AT,
    demote_after_days: float = DEMOTE_AFTER_DAYS,
    demote_below_access: int = DEMOTE_BELOW_ACCESS,
) -> list[TierMove]:
    """算这一轮该升降哪些。**只算，不写**（同 :func:`axo.memory.consolidation.plan_merges`）。

    Args:
        items: 这一轮开始时的全库快照（``run_consolidation`` 已经读过一份，直接传进来）。
        now: 这一轮的时刻。
        reinforced: 这一轮回放认出自己的 id。它们的 ``reinforce_count`` **按 +1 算**
            （SQL 层的自增还没发生），这样"第二次认出自己就升"在这一轮就能生效。
        promote_at: 升一层要几次加固。
        demote_after_days: 多久没被访问算"没人用"。
        demote_below_access: 访问次数少于几次才算"没人用"。

    Returns:
        按 ``items`` 的顺序；同一轮里一条记忆最多一个移动。
    """
    bumped = set(reinforced)
    horizon = now - demote_after_days * 86400.0
    moves: list[TierMove] = []

    for item in items:
        if item.pinned or item.is_persona_anchor or item.tier is Tier.COLD:
            continue
        if item.tier is Tier.ARCHIVE:
            # ARCHIVE 是"长期不用但确实重要"，只由人搬（那是它存在的理由）。
            continue

        if item.tier in _PROMOTABLE:
            reinforces = item.reinforce_count + (1 if item.memory_id in bumped else 0)
            if reinforces >= promote_at:
                target = _up(item.tier)
                # **到顶了就不产出原地不动的移动**：那会给事件账写一行 from==to 的
                # "升"，而它的 decision 还会算成降（两边秩相等）。到顶之后落到下面
                # "该不该降"的判据里——它可能确实已经没人用了。
                if target is not item.tier:
                    moves.append(TierMove(item.memory_id, item.tier, target))
                    continue

        if item.memory_id in bumped:
            # **这一轮刚被加固过的不判"没人用"**：它此刻显然是活的。
            continue
        if (
            item.tier in _DEMOTABLE
            and item.access_count < demote_below_access
            and item.last_access_ts < horizon
        ):
            moves.append(TierMove(item.memory_id, item.tier, _down(item.tier)))

    return moves


def _rank(tier: Tier) -> int:
    """层级的高低序（``COLD`` 不在这个序里——它不该参与升降）。"""
    return {
        Tier.WORKING: 0,
        Tier.RECENT: 1,
        Tier.LONGTERM: 2,
        Tier.ARCHIVE: 3,
        Tier.COLD: -1,
    }[tier]


def _up(tier: Tier) -> Tier:
    """往上一层，**封顶 LONGTERM**。"""
    return {Tier.WORKING: Tier.RECENT, Tier.RECENT: Tier.LONGTERM, Tier.LONGTERM: Tier.LONGTERM}[
        tier
    ]


def _down(tier: Tier) -> Tier:
    """往下一层，**最低到 RECENT**（见模块文档"不降到 WORKING"）。"""
    return {Tier.LONGTERM: Tier.RECENT, Tier.RECENT: Tier.RECENT}[tier]
