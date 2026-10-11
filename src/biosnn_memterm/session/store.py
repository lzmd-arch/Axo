"""会话记录的落盘。

**这不是记忆库。** 会话记录是"逐字稿"——它按时间顺序保存每一轮问答的原文，
存在的目的是让界面能恢复、让人能回看。记忆库（:mod:`memcore`）保存的是从逐字稿
里**提炼出来的条目**，带权重、带分层、带关联。

两者刻意分开，因为它们的生命周期完全不同：逐字稿可以有几百兆、可以整段删掉
而不影响助手的人格；记忆条目一条都不能丢错，删一条会改变图上的边。混在一起的
话，"清空聊天记录"这个动作就有概率把长期记忆一起清掉。

格式是 **JSONL**：首行会话元信息，其后每行一条消息。选它而不是一个 JSON 数组，
是为了能追加写——每轮对话只往文件尾追一行，不需要把整个会话读进内存再写回去。
文件很小（一个会话通常几十 KB），所以"原子重写"这条路径仍然保留给元信息变更。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Message", "Session", "SessionStore", "new_session_id"]

_SCHEMA = 3
"""逐字稿的格式版本。

**1 → 2 只是多了三个可选字段**（``tool_calls`` / ``tool_call_id`` / ``tool_name``），
**2 → 3 只多了一个**（``no_replay``）。两次都是一行老数据都没有重写：
:meth:`Message.to_dict` 是"非空才写"的，所以 schema-1/2 的行在 schema-3 的文件里
**逐字节不变**；:meth:`Message.from_dict` 缺键就给默认值，于是同一个读者天然能读三个
版本。**没有迁移脚本，也不需要有**——逐字稿可以几百兆，重写既慢又给"崩溃丢整段"开
一次口子，而 schema-3 需要的只是"新行多一个键"。

反方向（更旧的构建读新版本的文件）理论安全，因为老字段的形状一个都没动；但那条路
**没有自动化护栏**——测不了"一个不存在的旧构建"。写在这里是为了让下一个人知道这是
已知未覆盖的一格，而不是没想到。
"""

#: 会话 id 的合法形状——与 :func:`new_session_id` 的产出**逐字符对应**。
#:
#: 用白名单而不是"黑名单非法字符"。黑名单的漏洞是测试抓出来的：``".."`` 里
#: 没有任何被禁字符，于是它通过了校验，而 ``Path("sessions") / ".."`` 指向的
#: sessions 目录**外面**——:meth:`SessionStore.delete` 会照着删。
#: 白名单让"能不能拼进路径"这件事变成"是不是这个形状"，没有中间地带。
_SESSION_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


def new_session_id() -> str:
    """生成会话 id：可读时间前缀 + 短随机后缀。

    形如 ``20261010-143022-a1b2c3``。刻意不用 UUID：会话 id 会出现在文件名与
    界面上，时间前缀让它可排序、可辨认，一眼能看出"这是上周那次"。
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _optional_str(value: Any) -> str | None:
    """读一个可选的字符串字段。

    ``load()`` 的纪律是**宽容**（半行 JSON 都是跳过而不是报错），这里跟着来：非字符串
    的值归成"没有"，而不是让 ``str(some_dict)`` 变成一个看着像 id 的乱码——那种乱码会
    一路走到装配器，然后在"配对不上"那里变成一个指不到根因的错误。空串同样归成"没有"：
    它作为 ``tool_call_id`` 没有任何意义。
    """
    return value if isinstance(value, str) and value else None


def _read_tool_calls(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """读 ``tool_calls``。形状不对时返回空元组。

    只收字典项，**跳过**列表里的杂项而不是丢掉整条：同 ``load()`` 跳过半行 JSON 的
    纪律——宁可少一个调用，也不要整条消息消失。
    """
    raw = payload.get("tool_calls")
    if not isinstance(raw, list):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, dict))


@dataclass
class Message:
    """一条会话消息。"""

    role: str
    """``user`` / ``assistant`` / ``system`` / ``tool``。"""

    content: str
    at: float = field(default_factory=time.time)
    interrupted: bool = False
    """助手回复是否被用户中途停止。回看时它解释了"为什么这句话说了一半"。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    tool_calls: tuple[dict[str, Any], ...] = ()
    """助手这一轮请求的工具调用，原样存 :class:`~biosnn_memterm.llm.base.ToolCall`
    的字典形状。只在 ``role="assistant"`` 的行上非空。

    **存字典而不是直接存 ``ToolCall``**：``session/`` 不依赖 ``llm/`` 的协议类型，
    而且落盘格式因此不受那个 dataclass 改字段名的影响。转换只在
    :func:`~biosnn_memterm.session.assemble._read_call` 一处做。
    """

    tool_call_id: str | None = None
    """``role="tool"`` 的行答复的是哪一次调用。**它是配对的唯一依据。**

    缺了它的 tool 行是孤儿——模型收到一段它无法解释的对话。装配器把孤儿连同它那一对
    整对丢掉，并记一条账本 note，而不是让整个请求失败。
    """

    tool_name: str | None = None
    """工具名。**仅供回看显示**：协议上 ``tool_call_id`` 已经够配对，但回看时只看到
    一个 32 字符的 id 是没法读的。"""

    no_replay: str | None = None
    """这一对**不在后续轮次回放**的原因。``None`` = 正常回放。

    取值只有两个，各自对应一个具体的失效：

    * ``"denied"``——用户没同意执行（闸门拒绝）。那次拒绝是**当轮**的事，但它是
      一条落盘的 tool 行、会被 L5 逐轮搬进上下文，于是在模型眼里变成一条**仍然生效
      的规矩**：一次拒绝 = 该会话内对该工具的永久封禁。
    * ``"stale"``——结果是**某一刻的快照**（``get_time`` 的"现在"、``read_file`` 的
      文件内容）。回放它等于把过期的时间戳和旧的文件正文当成当前值递给模型。

    **为什么是一个原因字符串而不是一个 bool。** 两种情形指向两个不同的失效，而它们在
    同一份逐字稿里可以同时出现；回看时能读出"这一对为什么没被回放"比"它没被回放"值钱
    得多。更重要的是：一个 bool 会被读成"这一行有没有问题"，而它真正说的是一句
    **关于回放的指令**——字符串的命名把这件事钉在字段名上。

    **为什么记在这里而不是让装配器去猜。** 装配器（``session/``）不能 import
    ``agent/``（``agent/loop.py`` 已经 import ``session/budget``，反向就是环），所以
    "哪个工具的结果不耐放"这条知识**够不到** L5。它必须随数据走。顺带补上的是逐字稿
    一直缺的一位：它记了"调了什么工具"，没记"它怎么结束的"。
    """

    @property
    def title_hint(self) -> str:
        """用这条消息给会话起标题（取第一条用户消息）。"""
        text = " ".join(self.content.split())
        return text[:40] + ("…" if len(text) > 40 else "")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content, "at": self.at}
        if self.interrupted:
            payload["interrupted"] = True
        if self.prompt_tokens is not None:
            payload["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            payload["completion_tokens"] = self.completion_tokens
        # 三个工具字段同样"非空才写"。这不只是省字节：**schema-1 的老行因此逐字节
        # 不变**，往一个老文件里追加新行不需要回头重写它。
        if self.tool_calls:
            payload["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.no_replay is not None:
            payload["no_replay"] = self.no_replay
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Message:
        return cls(
            role=str(payload.get("role", "user")),
            content=str(payload.get("content", "")),
            at=float(payload.get("at") or time.time()),
            interrupted=bool(payload.get("interrupted", False)),
            prompt_tokens=payload.get("prompt_tokens"),
            completion_tokens=payload.get("completion_tokens"),
            tool_calls=_read_tool_calls(payload),
            tool_call_id=_optional_str(payload.get("tool_call_id")),
            tool_name=_optional_str(payload.get("tool_name")),
            no_replay=_optional_str(payload.get("no_replay")),
        )


@dataclass
class Session:
    """一个会话：元信息 + 消息序列。"""

    session_id: str
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model_ref: str = ""
    """创建该会话时用的模型。**仅供显示**——记忆与人格不跟它走，
    换成别的模型继续这个会话是正常用法，不是异常。"""

    messages: list[Message] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.time()

    def describe_parts(self) -> tuple[str, str]:
        """``(标题, 时间戳)``。

        会话列表现在是**两行**画的（标题一行、时间一行），所以这两样得能分开拿。
        但"标题为空时显示什么""时间戳用哪个格式"仍然只有这一处说法——界面那边各写
        一遍 ``strftime`` 的话，列表上行会变成 ``10-10 14:32``、别处却是另一个格式，
        而两边看起来都挺对。
        """
        title = self.title or "（空会话）"
        stamp = time.strftime("%m-%d %H:%M", time.localtime(self.updated_at))
        return title, stamp

    def describe(self) -> str:
        """给下拉框用的一行摘要。"""
        title, stamp = self.describe_parts()
        return f"{stamp}  {title}"


class SessionStore:
    """会话目录的读写。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── 路径 ────────────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> Path:
        """把会话 id 拼成文件路径。

        校验是**形状白名单**（见 :data:`_SESSION_ID_RE`），不是"挡掉非法字符"。
        会话 id 只由本模块生成，但仍然不信它——`:meth:`delete` 会照着这个路径
        删文件，而一个 ``".."`` 就足以让它指向目录树外面。
        """
        if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
            raise ValueError(
                f"非法的会话 id：{session_id!r}。合法形状为 YYYYMMDD-HHMMSS-xxxxxx（十六进制后缀）。"
            )
        return self.root / f"{session_id}.jsonl"

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).is_file()

    # ── 读 ──────────────────────────────────────────────────────────────────

    def load(self, session_id: str) -> Session:
        """读取一个会话。文件不存在时返回空会话，不抛异常。

        返回空会话而非报错，是为了让界面少一处分支：打开一个已被外部删除的
        会话，结果是"看到一个空会话"，而不是一个需要处理的错误。
        """
        path = self._path(session_id)
        session = Session(session_id=session_id)
        if not path.is_file():
            return session

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # 半行 JSON 是崩溃断电的典型残留。跳过它而不是让整个会话
                    # 打不开——用户更可能想看到"丢了一轮"而不是"记录全没了"。
                    continue
                if line_number == 0 and payload.get("schema") is not None:
                    session.title = str(payload.get("title", ""))
                    session.created_at = float(payload.get("created_at") or session.created_at)
                    session.updated_at = float(payload.get("updated_at") or session.updated_at)
                    session.model_ref = str(payload.get("model_ref", ""))
                    continue
                session.messages.append(Message.from_dict(payload))
        return session

    def list_sessions(self, limit: int = 50) -> list[Session]:
        """列出最近的会话（按修改时间倒序）。只读首行，不解析消息。"""
        entries: list[tuple[float, Session]] = []
        for path in self.root.glob("*.jsonl"):
            mtime = _safe_mtime(path)
            session = Session(session_id=path.stem)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = handle.readline()
                payload = json.loads(first)
                session.title = str(payload.get("title", ""))
                session.updated_at = float(payload.get("updated_at") or mtime)
                session.created_at = float(payload.get("created_at") or mtime)
                session.model_ref = str(payload.get("model_ref", ""))
            except (OSError, json.JSONDecodeError, ValueError):
                # 首行坏了也仍然列出来——用户至少能看到它、能删它
                session.updated_at = mtime
            entries.append((mtime, session))
        entries.sort(key=lambda pair: pair[0], reverse=True)
        return [session for _, session in entries[:limit]]

    # ── 写 ──────────────────────────────────────────────────────────────────

    def append(self, session: Session, message: Message) -> None:
        """追加一条消息。

        首行不存在时先写元信息行。**追加而不是重写整个文件**：这样每轮对话的
        磁盘代价恒定，且崩溃最多丢掉最后一行。
        """
        path = self._path(session.session_id)
        session.messages.append(message)
        session.touch()
        if not session.title and message.role == "user":
            session.title = message.title_hint

        needs_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            if needs_header:
                handle.write(json.dumps(self._header(session), ensure_ascii=False) + "\n")
            handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        self._sync_meta(session)

    def _header(self, session: Session) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "model_ref": session.model_ref,
        }

    def _sync_meta(self, session: Session) -> None:
        """把元信息落成一份旁挂文件。

        为什么不就地改写首行：JSONL 的首行长度随标题变化，就地改写意味着要重写
        整个文件。而"标题还是空的"这个状态会让会话列表显示成"（空会话）"，很
        难认。所以元信息另存一个 ``.meta.json``——它是**派生的**，丢了可以重建，
        损坏也不影响正文。
        """
        path = self._path(session.session_id).with_suffix(".meta.json")
        payload = self._header(session)
        temp = path.with_suffix(".meta.json.tmp")
        # newline="\n" 是必须的：Windows 上 write_text 默认把 \n 写成 \r\n，
        # 而这个文件会进版本控制（开发目录下）或跨机器复制，行尾漂移会让 diff
        # 里出现整文件的假变更。
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
        temp.replace(path)  # 原子替换：读者要么看到旧的，要么看到新的

    def delete(self, session_id: str) -> None:
        """删除一个会话（正文与元信息）。"""
        self._path(session_id).unlink(missing_ok=True)
        self._path(session_id).with_suffix(".meta.json").unlink(missing_ok=True)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
