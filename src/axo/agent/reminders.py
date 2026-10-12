"""提醒的登记，以及**谁到点了、谁已经响过**。

``set_reminder`` 工具把一条提醒写进这里的文件。P3a 时这个模块的职责只有一条
——**登记得对、读得回来**——因为"到点了"要有常驻进程在听，而那要等 P3b 的托盘
（否则那是一个没有任何东西在听的时刻）。

P3b 到了，于是这里多了三件**不碰 Qt 的事**（判断全在这一层，Qt 那一侧只负责
"什么时候看一眼"与"弹出来"）：

* :func:`due_now` —— 现在该响的是哪几条；
* :func:`next_due_at` —— 下一次该在什么时候醒（``None`` = 没有将来的提醒）；
* :func:`missed_summary` —— 程序没开着的那些时刻里错过的、聚合成一句话。

以及 :class:`FiredStore`：**已经响过的那些**记在哪儿。它必须存在，否则每次启动
都会把过去所有提醒重弹一遍。

## 为什么它和记忆库分开

提醒是**还没发生的事**，记忆是**发生过的事**。混进 ``memcore`` 的第一个后果是
召回会开始返回一些未来的句子——它们在语义上就是"用户提到过的一句话"，权重、
衰减、SDR 全都算得出来，于是它会安安静静地出现在注入账本里。第二个后果更实在：
"清空聊天记录"与"删掉一条记忆"这类动作不该顺手动到用户还没被提醒的事情。

于是落成一个自己的 JSONL 文件（位置由调用方给，见下）。

## 为什么是追加写、格式宽容

同 :mod:`axo.session.store`：一行一条，只追加。读的时候**跳过读不懂的
行**，而不是让整份文件打不开——这个文件里装着用户还没被提醒的事情，而"因为第 3
行坏了所以第 4 行也不给你看"是最没有必要的一种失败。

## 为什么"已经响过"是**另一个**文件

``reminders.jsonl`` 的契约是"这个文件装的是**还没被提醒的事情**"。往里追加第二种
记录（"这条响过了"）会让那句话变成假的，也会让 :meth:`ReminderStore.all` 从
"读出全部提醒"变成"读出两种东西"——而它现有的每一个调用方都是按前者写的。

所以 :class:`FiredStore` 写一个 sidecar（位置见
:func:`~axo.paths.reminders_fired_path`）。它**只追加**：一次已触发是
一件发生过的事，而"发生过"这件事没有"改"这个动作。重复标记同一个 id 是无害的
（读出来是集合），于是崩溃后重放的代价只是多一行。

## 谁决定文件在哪儿

**调用方**（``ui/main_window.py`` 从 :func:`~axo.paths.reminders_path`
取），不是本模块。理由与 :class:`~memcore.store.MemoryStore` 收一个 ``path`` 一样：
纯层里的东西不该知道"应用的数据目录"长什么样——那是 ``paths.py`` 的事，而测试要能
把它指到 ``tmp_path``。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FiredStore",
    "Reminder",
    "ReminderStore",
    "due_now",
    "missed_summary",
    "new_reminder_id",
    "next_due_at",
]

MISSED_LIMIT = 5
"""错过的那一堆里，一句话最多列几条正文。其余折成"还有 N 条"。"""


def due_now(reminders: Sequence[Reminder], fired: Set[str], now: float) -> tuple[Reminder, ...]:
    """到点了、而且还没响过的那些，**按到期时间先后**。

    ``now`` 由调用方给而不是在这里读时钟：这一层要能在测试里把"现在"摆到任何
    一刻，而"到点了"的全部判断就是这一次比较。同理 ``fired`` 是**传进来的集合**，
    不是在这里读文件——纯函数与 IO 分开，后者在 :class:`FiredStore` 里。

    排序而不是按登记顺序：程序关着了一周之后，用户想知道的是"最早该响的那件事"，
    而那与"哪天登记的"没有关系。
    """
    return tuple(
        sorted(
            (one for one in reminders if one.due_at <= now and one.reminder_id not in fired),
            key=lambda one: (one.due_at, one.reminder_id),
        )
    )


def next_due_at(reminders: Sequence[Reminder], fired: Set[str]) -> float | None:
    """下一次该在什么时候醒。**没有任何将来的提醒时返回 ``None``**。

    返回值可能**是过去**（错过的那些还没被标 fired 时）：调用方据此立刻醒一次把
    它们收掉，而不是傻等一个永远不来的时刻。``None`` 与"过去的某一刻"是两件不同
    的事——前者意味着"没有东西可等"，而这是可以真的把定时器停掉的状态。
    """
    pending = [one.due_at for one in reminders if one.reminder_id not in fired]
    return min(pending) if pending else None


def missed_summary(reminders: Sequence[Reminder], *, limit: int = MISSED_LIMIT) -> str:
    """把错过的那一堆收成**一句话**。

    聚合成一条而不是逐条弹：关了一周再打开会变成一场通知风暴，而风暴的结局是用户
    把通知关掉——那时**所有**提醒都不再有效，包括没过期的。

    也不静默丢掉：错过的事情仍然要说出来，只是不必一次说五遍。
    """
    if not reminders:
        return ""
    listed = "、".join(one.text for one in reminders[:limit])
    if len(reminders) > limit:
        return f"{listed}；还有 {len(reminders) - limit} 条"
    return listed


def new_reminder_id() -> str:
    """提醒 id。形状与 :func:`~memcore.ids.new_ulid` 同类：可排序、可当文件名。

    不 import ``memcore.ids`` 是因为它不是记忆——用同一个 id 空间会让人以为
    "提醒也能被召回"。时间前缀在前，于是文件里天然按登记顺序排。
    """
    return f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class Reminder:
    """一条登记下来的提醒。

    Attributes:
        reminder_id: 稳定 id。将来"取消这条提醒"要用它。
        text: 到点要告诉用户的那句话。
        due_at: 什么时候（epoch 秒，本地时钟）。
        created_at: 什么时候登记的。与 :attr:`due_at` 分开，因为"三天前约的今天的
            事"和"今天约的今天的事"在界面上是两回事。
        session_id: 在哪次会话里登记的。用于溯源，不用于触发。
        turn_seq: 那次会话的第几个回合，同上的定位成分。
        call_id: 是哪一次工具调用登记的。**它是这个数据结构里唯一能把提醒接回
            逐字稿的东西**——没有它，一条提醒就是一句来历不明的话。
    """

    reminder_id: str
    text: str
    due_at: float
    created_at: float
    session_id: str = ""
    turn_seq: int = 0
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """落盘的那个字典。**非空才写**——照 :meth:`~axo.session.store.Message.to_dict`
        的范式，于是以后加字段不会让老行变成一坨空键。"""
        payload: dict[str, Any] = {
            "reminder_id": self.reminder_id,
            "text": self.text,
            "due_at": self.due_at,
            "created_at": self.created_at,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.turn_seq:
            payload["turn_seq"] = self.turn_seq
        if self.call_id:
            payload["call_id"] = self.call_id
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> Reminder | None:
        """读回来。**形状不对一律返回 ``None``**（调用方跳过那一行）。

        参数标成 ``Any`` 而不是 ``dict`` 是如实：这个值来自 ``json.loads``，一行
        ``"就一个字符串"`` 或者 ``[1, 2, 3]`` 都是合法 JSON。类型注解在这里管不住
        文件，所以这道检查必须在**这里**——``all()`` 只管按行读，形状的事一处说了算。

        **不做单位换算、不猜缺的时间**：``due_at`` 缺失或不是数的行直接丢掉。
        给一条没有时间的提醒编一个"大概明天"出来，比丢一行坏数据严重得多——
        它会变成一条用户从没约过的提醒。
        """
        if not isinstance(payload, dict):
            return None
        reminder_id = payload.get("reminder_id")
        text = payload.get("text")
        due_at = payload.get("due_at")
        if not isinstance(reminder_id, str) or not reminder_id:
            return None
        if not isinstance(text, str) or not text.strip():
            # 全空白的正文等于一条**没有内容**的提醒：它会在将来某次"到点了"里
            # 显示成一行空白，而用户没法从那一行看出该做什么。判据用 ``strip()``
            # 是因为 ``"   "`` 与非空字符串一样能过 ``isinstance``。
            # （存下去的仍是原文，不在这里替它修剪——读的一侧只做"收不收"的判断。）
            return None
        if isinstance(due_at, bool) or not isinstance(due_at, (int, float)):
            return None
        created_at = payload.get("created_at")
        return cls(
            reminder_id=reminder_id,
            text=text,
            due_at=float(due_at),
            created_at=float(created_at) if isinstance(created_at, (int, float)) else 0.0,
            session_id=_text(payload.get("session_id")),
            turn_seq=_count(payload.get("turn_seq")),
            call_id=_text(payload.get("call_id")),
        )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class ReminderStore:
    """提醒的落盘。**只追加，不改写。**"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def add(
        self,
        text: str,
        *,
        due_at: float,
        created_at: float,
        session_id: str = "",
        turn_seq: int = 0,
        call_id: str = "",
    ) -> Reminder:
        """登记一条。返回真正落下去的那条（带它自己的 id）。

        **同步写**（同 :class:`~axo.memory.writer.MemoryWriter` 的
        "每轮实时写入"）：用户说完"提醒我明天买牛奶"，这句话在下一句话之前就该
        在文件里。异步写会让"说完立刻关窗口"丢掉它——而这恰好是最容易被试到的
        那一次。

        **失败要炸**，不吞：这条路径上没有"让谁重试"的当事人（模型在闸门那边，
        这里只是一次落盘），所以静默失败等于用户以为约好了而其实没有。
        """
        reminder = Reminder(
            reminder_id=new_reminder_id(),
            text=text,
            due_at=due_at,
            created_at=created_at,
            session_id=session_id,
            turn_seq=turn_seq,
            call_id=call_id,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``newline="\n"``：Windows 上默认会把 ``\n`` 写成 CRLF，而这个文件将来要
        # 被别的工具读（P3b 的托盘、用户自己的脚本）。逐字节可预期的行尾比省一次
        # 参数重要。
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(reminder.to_dict(), ensure_ascii=False) + "\n")
        return reminder

    def all(self) -> list[Reminder]:
        """读出全部提醒，**按登记顺序**。

        文件不存在返回空表而不是抛：还没登记过任何提醒是**常态**，不是错误。
        """
        if not self._path.is_file():
            return []
        found: list[Reminder] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 半行 JSON：断电时最常见的一种残骸
                reminder = Reminder.from_dict(payload)
                if reminder is not None:
                    found.append(reminder)
        return found

    def __len__(self) -> int:
        return len(self.all())


class FiredStore:
    """**已经响过**的提醒 id。只追加，读出来是个集合。

    它存在的原因是 :func:`due_now` 需要知道"这一条已经响过了"——没有它，每次启动
    都会把过去所有提醒重弹一遍，而"设了提醒每次开程序都弹"比没有提醒更烦人。

    **写在前、弹在后**（write-ahead，调用方的纪律）：反过来的话，弹出之后、落盘
    之前进程退出，下次启动会重弹同一条。而反过来的代价（标了却因为弹失败而没弹）
    是用户错过一次提醒——两害相权，重弹更糟：它每次都发生，而弹出失败很少发生。

    与 :class:`ReminderStore` 同一条宽容度：**读的时候跳过读不懂的行**，绝不抛。
    这个文件里装的只是"响过没有"，一行坏了不该让其余全部重弹。
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def fired_ids(self) -> frozenset[str]:
        """响过的那些 id。文件不存在返回空集合（还没响过任何一条是常态）。"""
        if not self._path.is_file():
            return frozenset()
        found: set[str] = set()
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reminder_id = _fired_id(payload)
                if reminder_id:
                    found.add(reminder_id)
        return frozenset(found)

    def mark_fired(self, reminder_id: str, at: float) -> None:
        """记下"这条响过了"。**同步写、失败要炸**——同 :meth:`ReminderStore.add`：
        静默失败的表现是下次启动重弹一条用户已经看过的提醒。

        重复标记同一个 id 无害（读出来是集合），所以崩溃后重放不必先查一遍。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"reminder_id": reminder_id, "at": at}, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def mark_all_fired(self, reminder_ids: Sequence[str], at: float) -> None:
        """一次标一批。**一次打开文件**，不是每条开一次——错过二十条时那二十次
        打开/关闭里任何一次失败都会留下一半已标一半没标的状态。"""
        if not reminder_ids:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            for reminder_id in reminder_ids:
                payload = {"reminder_id": reminder_id, "at": at}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _fired_id(payload: Any) -> str:
    """一行 fired 记录里的 id。形状不对返回空串（调用方跳过那一行）。

    比 :meth:`Reminder.from_dict` 宽容得多，因为它要的信息只有一样：**这是哪一条**。
    时间戳读不出来就丢那一个字段，不必丢掉整行。
    """
    if not isinstance(payload, dict):
        return ""
    reminder_id = payload.get("reminder_id")
    return reminder_id if isinstance(reminder_id, str) else ""
