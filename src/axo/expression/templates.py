"""模板库：Phase 2 的"先看模板、再问模型"。

Phase 1 的渲染只有一个来源——本地小模型。模板库把那些**形状固定**的回复（拒绝、
短回答、安慰开场）提前写好：命中就不调模型，省一次调用，也把最容易被模型写走样的
几句话钉死。

## 模板不是"人格"

模板只从 SIR 里取内容（要点、事实），**不新增信息**。所以命中之后仍要过两道检查：
事实一条不许少、问句不许写成陈述句——与模型路径同一套判据。模板坏了和模型坏了
一样，都退回草稿。

## 槽位

``{key_point:0}`` / ``{fact:0}`` 取第几条；``{facts}`` 把全部事实用「、」连起来。
**未知槽位在加载时就拒绝**：一个拼错的槽位若留到运行时，表现是"模板永远不命中"，
而那看起来像"模板库太小"，不会有人去查拼写。
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .sir import SIR, facts_preserved

__all__ = ["BUILTIN_TEMPLATES", "Template", "TemplateBank", "TemplateHit"]

_SLOT = re.compile(r"\{([a-z_]+(?::\d+)?)\}")


@dataclass(frozen=True)
class Template:
    """一条模板。

    Attributes:
        id: 稳定标识，进报告用。
        kind: ``"exact"``（无槽位）或 ``"skeleton"``（有槽位）。
        pattern: 正文，槽位见模块文档。
        intents / tones / lengths: 触发条件；空元组 = 不限。
        facts_min / facts_max: 事实条数的上下限（``None`` = 不限）。
        weight: 同条件多条时的相对权重（整数化后参与确定性选择）。
    """

    id: str
    pattern: str
    kind: str = "skeleton"
    intents: tuple[str, ...] = ()
    tones: tuple[str, ...] = ()
    lengths: tuple[str, ...] = ()
    facts_min: int = 0
    facts_max: int | None = None
    weight: int = 1

    def matches(self, sir: SIR) -> bool:
        if self.intents and sir.intent not in self.intents:
            return False
        if self.tones and sir.tone not in self.tones:
            return False
        if self.lengths and sir.length not in self.lengths:
            return False
        count = len(sir.facts)
        if count < self.facts_min:
            return False
        return self.facts_max is None or count <= self.facts_max

    def render(self, sir: SIR) -> str | None:
        """填槽。**填不出来返回 ``None``**（不猜、不删槽位）。"""
        text = self.pattern
        missing = False

        def replace(match: re.Match[str]) -> str:
            nonlocal missing
            slot = match.group(1)
            if slot == "facts":
                if not sir.facts:
                    missing = True
                    return ""
                return "、".join(sir.facts)
            name, _, index_text = slot.partition(":")
            index = int(index_text) if index_text else 0
            if name == "key_point":
                value = sir.key_points[index] if index < len(sir.key_points) else ""
            elif name == "fact":
                value = sir.facts[index] if index < len(sir.facts) else ""
            else:
                value = ""
            if not value:
                missing = True
            return value

        filled = _SLOT.sub(replace, text)
        if missing or not filled.strip() or _SLOT.search(filled):
            return None
        return filled.strip()


@dataclass(frozen=True)
class TemplateHit:
    """一次命中：模板 id 与填好的正文。"""

    template_id: str
    text: str


_DEFAULT_TEMPLATES: tuple[Template, ...] = (
    Template("refuse-direct", "不知道。", kind="exact", intents=("refuse",), tones=("direct",)),
    Template("refuse-casual", "没找到。", kind="exact", intents=("refuse",), tones=("casual",)),
    Template("refuse-short", "没印象。", kind="exact", intents=("refuse",), lengths=("short",)),
    Template(
        "answer-short", "{key_point:0}。", kind="skeleton", intents=("answer",), lengths=("short",)
    ),
    Template(
        "answer-medium",
        "{key_point:0}。{key_point:1}。",
        kind="skeleton",
        intents=("answer",),
        lengths=("medium",),
    ),
    # **带事实的回答**：要点在前、事实用冒号接在后面。没有这一条时，凡是带数字/URL
    # 的短回答都会整族落到模型路径——模板覆盖率会被这一族拖掉一半。
    Template(
        "answer-short-facts",
        "{key_point:0}：{facts}。",
        intents=("answer",),
        lengths=("short",),
        facts_min=1,
    ),
    Template(
        "answer-medium-facts",
        "{key_point:0}。{key_point:1}：{facts}。",
        intents=("answer",),
        lengths=("medium",),
        facts_min=1,
    ),
    Template("ask-short", "{key_point:0}？", kind="skeleton", intents=("ask",)),
    Template("emotion-casual", "嗯，{key_point:0}。", intents=("emotion",), tones=("casual",)),
    Template("emotion-formal", "好的，{key_point:0}。", intents=("emotion",), tones=("formal",)),
    Template("emotion-direct", "{key_point:0}。", intents=("emotion",), tones=("direct",)),
    Template("emotion-comfort", "嗯，{key_point:0}。", intents=("emotion",), tones=("comfort",)),
    Template("action-casual", "{key_point:0}，好。", intents=("action",), tones=("casual",)),
    Template(
        "action-formal", "好的，{key_point:0}，我去办。", intents=("action",), tones=("formal",)
    ),
    Template("action-direct", "{key_point:0}，我去办。", intents=("action",), tones=("direct",)),
)

#: 随包内置的一组**中性**模板。它们只负责形状，不承载人格措辞；
#: 用户自己的模板放数据目录（见 :meth:`TemplateBank.load`），不进仓库。
BUILTIN_TEMPLATES = _DEFAULT_TEMPLATES


class TemplateBank:
    """一组模板 + 确定性的选择。"""

    def __init__(self, templates: Iterable[Template]) -> None:
        self.templates = tuple(templates)
        for template in self.templates:
            for slot in _SLOT.findall(template.pattern):
                if not self._slot_is_known(slot):
                    raise ValueError(f"模板 {template.id!r} 含未知槽位 {{{slot}}}")
            if template.kind not in ("exact", "skeleton"):
                raise ValueError(f"模板 {template.id!r} 的 kind 必须是 exact / skeleton")

    @staticmethod
    def _slot_is_known(slot: str) -> bool:
        name, _, index = slot.partition(":")
        if name == "facts":
            return not index
        return name in ("key_point", "fact") and (not index or index.isdigit())

    @classmethod
    def builtin(cls) -> TemplateBank:
        return cls(BUILTIN_TEMPLATES)

    @classmethod
    def load(cls, path: Path) -> TemplateBank:
        """从 JSONL 读用户模板。**一条坏模板就整份拒绝**——半份模板库比没有更难查。"""
        templates: list[Template] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number} 不是合法 JSON：{exc}") from exc
            templates.append(
                Template(
                    id=str(row["id"]),
                    pattern=str(row["pattern"]),
                    kind=str(row.get("kind", "skeleton")),
                    intents=tuple(row.get("intents", ())),
                    tones=tuple(row.get("tones", ())),
                    lengths=tuple(row.get("lengths", ())),
                    facts_min=int(row.get("facts_min", 0)),
                    facts_max=None if row.get("facts_max") is None else int(row["facts_max"]),
                    weight=int(row.get("weight", 1)),
                )
            )
        return cls(templates)

    def _ordered(self, sir: SIR) -> list[Template]:
        """满足条件的模板，按草稿哈希定一个起点、再环形展开。

        **不能只选一条。** 一条丢事实的模板先被选中时，正确的那条就被挡住了——
        2026-11-03 实测过：带事实的短回答有专门的模板，但"不带事实"的那条先命中，
        整族落在模型路径上。这里改成"从哈希位置开始，逐条试到能过检查为止"：
        选择仍是确定性的，但不会因为第一条不合适就放弃。
        """
        candidates = [template for template in self.templates if template.matches(sir)]
        if not candidates:
            return []
        start = zlib.crc32(sir.draft.encode("utf-8")) % len(candidates)
        return candidates[start:] + candidates[:start]

    def match(self, sir: SIR) -> Template | None:
        """按 SIR 选第一条候选。**确定性**：同一段草稿永远选同一条。"""
        ordered = self._ordered(sir)
        return ordered[0] if ordered else None

    def render(self, sir: SIR) -> TemplateHit | None:
        """命中并填好、且**事实与问句检查通过**时返回；否则 ``None``。"""
        for template in self._ordered(sir):
            text = template.render(sir)
            if text is None:
                continue
            if facts_preserved(text, sir.facts):
                continue
            if sir.intent == "ask" and ("？" not in text and "?" not in text):
                continue
            return TemplateHit(template.id, text)
        return None
