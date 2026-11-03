"""SIR：把"说什么"与"怎么说"分开的中间表示。

后端模型负责内容与意图，终端侧的表达层负责语气与措辞；两边只通过这里的数据结构
说话。**这一层是纯的**：不 import Qt、不 import httpx、不读设置，所以每一条判定都
能在核心腿里逐条断言。

## 为什么后端不直接输出最终文本

同一段 persona prompt，不同后端会给出不同文体（Qwen3 偏分点、DeepSeek 偏反问、
Llama 偏美式直接）。把"怎么说"挪到终端侧之后，换模型不再改变她说话的节奏——但
**前提是内容不能在这一层被改写**（见 :attr:`SIR.facts`）。

## 两种来源

* **首选**：后端按 :func:`sir_instruction` 的格式吐出 ``<think>`` 块；
* **降级**：后端只吐普通文本（旧模型、云端、或这一轮根本没要求 SIR）——用
  :func:`derive` 从草稿里抽出意图/要点/长度，损失的是"她打算强调什么"，不是内容。

降级是**常态**不是异常：本版不把主对话的提示词改成 SIR 格式（那会与工具调用、
"思考吃额度"这些已有的坑相撞），所以生产路径上实际走的是 :func:`parse` 找不到
标签时的 :func:`derive`。SIR-first 留给"后端确认支持"之后。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "ALLOWED_INTENTS",
    "ALLOWED_LENGTHS",
    "ALLOWED_TONES",
    "SIR",
    "derive",
    "extract_facts",
    "facts_preserved",
    "parse",
    "render_prompt",
    "sir_instruction",
    "style_signature",
]

ALLOWED_INTENTS = ("answer", "ask", "refuse", "emotion", "action")
ALLOWED_TONES = ("casual", "formal", "tease", "comfort", "direct")
ALLOWED_LENGTHS = ("short", "medium", "long")

#: ``<think>`` / ``<content>`` 两块的取法。**不要求闭合标签**：模型漏写闭合比漏写
#: 开头常见得多，而"整段被当成 SIR 丢掉"的后果是这一轮没有回复。
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_CONTENT = re.compile(r"<content\b[^>]*>(.*?)(?:</content\s*>|$)", re.IGNORECASE | re.DOTALL)

_KEY_LINE = re.compile(r"^(意图|语气指示|语气|长度|情感)\s*[:：]\s*(.+)$")
_LIST_HEADER = re.compile(r"^(要点|必须保留的事实|事实)\s*[:：]?\s*$")
_LIST_ITEM = re.compile(r"^\s*[-*·•]\s*(.+)$")

#: 意图的中英文都收：提示词是中文，模型回中文标签是常态。
_INTENT_WORDS = {
    "answer": "answer",
    "回答": "answer",
    "ask": "ask",
    "反问": "ask",
    "提问": "ask",
    "refuse": "refuse",
    "拒绝": "refuse",
    "emotion": "emotion",
    "情绪": "emotion",
    "action": "action",
    "动作": "action",
}
_TONE_WORDS = {
    "casual": "casual",
    "随意": "casual",
    "轻松": "casual",
    "formal": "formal",
    "正式": "formal",
    "tease": "tease",
    "调侃": "tease",
    "comfort": "comfort",
    "安慰": "comfort",
    "direct": "direct",
    "直接": "direct",
}
_LENGTH_WORDS = {
    "short": "short",
    "短": "short",
    "medium": "medium",
    "中": "medium",
    "long": "long",
    "长": "long",
}

_SENTENCE_END = re.compile(r"[。！？!?；;\n]+")
_REFUSAL = ("不知道", "不清楚", "没找到", "没有找到", "没法", "不能", "我不记得", "记不清")
_URL = re.compile(r"https?://[^\s，。；、）)】\]]+")
_WIN_PATH = re.compile(r"[A-Za-z]:\\[^\s，。；、）)】\]]+")
_QUOTED = re.compile(r"[「『“\"]([^」』”\"]{1,40})[」』”\"]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?:\s*%|％)?")
_CODEISH = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*")
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f0ff]")
_CONNECTIVES = ("因为", "所以", "但是", "不过", "而且", "然后", "首先", "其次", "另外")


@dataclass(frozen=True)
class SIR:
    """一条回复的"内容意图"。**风格无关**：它不描述用哪个词。

    Attributes:
        intent: 这一句在做什么（:data:`ALLOWED_INTENTS`）。
        key_points: 要表达的内容，**按优先级排序**（第一条最重要）。表达层可以
            换词、可以合并，但不该整条丢掉——除非 :attr:`facts` 覆盖了它。
        tone: 用户需要什么情绪回应（:data:`ALLOWED_TONES`）。它是**提示**不是指令：
            表达层按人格落地，不逐字照搬。
        length: 长度档位（:data:`ALLOWED_LENGTHS`）。
        facts: **必须原样保留的事实**：数字、日期、URL、路径、引号里的专名。表达层
            不许改写它们；改了就整段退回草稿（见 :func:`facts_preserved`）。
        emotions: 可选的情感标签，用于安慰类回复。
        draft: 后端原文。表达层可以参考它，但不必逐句遵循。
    """

    intent: str = "answer"
    key_points: tuple[str, ...] = ()
    tone: str = "casual"
    length: str = "medium"
    facts: tuple[str, ...] = ()
    emotions: tuple[str, ...] = ()
    draft: str = ""


def _clean(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").strip()


def _one_of(value: str, table: dict[str, str], default: str, allowed: tuple[str, ...]) -> str:
    token = _clean(value).strip(" 。.；;，,").lower()
    if token in table:
        return table[token]
    for word, target in table.items():
        if word in token:
            return target
    return default if default in allowed else allowed[0]


def _split_think(text: str) -> tuple[str, str]:
    """``(think 块, 正文)``。闭合标签缺席时，think 一直算到 ``<content>`` 或文末。"""
    source = text or ""
    opened = _THINK_OPEN.search(source)
    if opened is None:
        content = _CONTENT.search(source)
        return "", (content.group(1).strip() if content else source.strip())
    rest = source[opened.end() :]
    closed = _THINK_CLOSE.search(rest)
    content = _CONTENT.search(rest)
    ends = [match.start() for match in (closed, content) if match is not None]
    end = min(ends) if ends else len(rest)
    think = rest[:end]
    body = content.group(1).strip() if content else ""
    if not body:
        tail = rest[closed.end() :] if closed else rest[end:]
        tail = _CONTENT.sub("", tail)
        body = tail.strip()
    return think, body


def _parse_lists(think: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    points: list[str] = []
    facts: list[str] = []
    bucket: list[str] | None = None
    for raw in think.splitlines():
        line = raw.strip()
        header = _LIST_HEADER.match(line)
        if header:
            bucket = points if header.group(1) == "要点" else facts
            continue
        item = _LIST_ITEM.match(line)
        if item and bucket is not None:
            text = _clean(item.group(1)).strip("。；;，,")
            if text:
                bucket.append(text)
            continue
        if line:
            bucket = None
    return tuple(points[:6]), tuple(facts[:8])


def _parse_scalars(think: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in think.splitlines():
        match = _KEY_LINE.match(raw.strip())
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = _clean(value)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def extract_facts(text: str) -> tuple[str, ...]:
    """从正文里挑出**不许改写**的东西：数字、URL、路径、引号里的专名、带数字的代号。

    **宁少勿多。** 多抽一条的代价是"表达层明明写得好好的，却被判成丢事实、整段退回
    草稿"；少抽一条的代价是那个数被改写。所以只收**形状确定**的几类——中文专名要靠
    后端自己填 ``facts``，正则猜不出来。
    """
    source = text or ""
    found: list[str] = []
    found.extend(match.group(0) for match in _URL.finditer(source))
    found.extend(match.group(0) for match in _WIN_PATH.finditer(source))
    found.extend(match.group(1) for match in _QUOTED.finditer(source))
    found.extend(match.group(0) for match in _NUMBER.finditer(source))
    found.extend(match.group(0) for match in _CODEISH.finditer(source))
    # 单个数字（"3"）不收：它在正文里可能是"第三天"，抽出来只会让表达层不敢用
    # 自己的话。两个字符以上的数字串才有"这是个数"的分量。
    return _dedupe([fact for fact in found if len(fact) >= 2 or not fact.isdigit()])


def derive(draft: str) -> SIR:
    """从一段普通文本抽出 SIR。**降级路径**，不是次品：它是本版生产路径的常态。

    抽不出来的东西（"她想强调哪一句"）就按句序取前几条；意图与长度是给表达层的
    收尾用的，判错了最多是这句话长了一点，不会改内容。

    **``draft`` 原样保留。** 分析用 NFKC 归一化过的那一份（全角数字要能认出来），
    但退回草稿时显示的是用户/后端原本那段字——把全角逗号换成半角、把句末问号削掉，
    是这一层最不该做的"顺手整理"。
    """
    text = (draft or "").strip()
    analysis = _clean(text)
    sentences = [_clean(part).strip("。；;，,") for part in _SENTENCE_END.split(analysis)]
    sentences = [part for part in sentences if part]
    points = tuple(sentences[:6])
    if any(word in analysis for word in _REFUSAL) and len(analysis) < 40:
        intent = "refuse"
    elif analysis.rstrip().endswith(("？", "?")):
        intent = "ask"
    elif _EMOJI.search(analysis) or any(
        word in analysis[:12] for word in ("哈哈", "唉", "呜", "嗯嗯")
    ):
        intent = "emotion"
    else:
        intent = "answer"
    length = "short" if len(analysis) <= 40 else ("medium" if len(analysis) <= 120 else "long")
    return SIR(
        intent=intent,
        key_points=points,
        tone="direct" if length == "short" else "casual",
        length=length,
        facts=extract_facts(analysis),
        draft=text,
    )


def parse(text: str) -> SIR:
    """把后端的一段输出变成 SIR。**没有 ``<think>`` 块时等于 :func:`derive`。**

    缺字段用 :func:`derive` 的结果补齐——模型漏写一行不该让整段回复作废。
    """
    think, body = _split_think(text)
    if not think.strip():
        return derive(body or text)
    fallback = derive(body or think)
    scalars = _parse_scalars(think)
    points, facts = _parse_lists(think)
    intent = _one_of(scalars.get("意图", ""), _INTENT_WORDS, fallback.intent, ALLOWED_INTENTS)
    tone = _one_of(
        scalars.get("语气指示", scalars.get("语气", "")), _TONE_WORDS, fallback.tone, ALLOWED_TONES
    )
    length = _one_of(scalars.get("长度", ""), _LENGTH_WORDS, fallback.length, ALLOWED_LENGTHS)
    emotions = tuple(
        _clean(part) for part in re.split(r"[、,，/\s]+", scalars.get("情感", "")) if _clean(part)
    )
    return SIR(
        intent=intent,
        key_points=points or fallback.key_points,
        tone=tone,
        length=length,
        facts=_dedupe(list(facts) + list(fallback.facts)),
        emotions=emotions,
        draft=(body or fallback.draft).strip(),
    )


def sir_instruction() -> str:
    """给（确认支持 SIR 的）后端模型的一段格式说明。**本版默认不发**。

    发出去的前提是"这一轮不需要工具、也没有别的格式约定"——把主对话的格式改成
    ``<think>`` 会让工具调用与"思考吃额度"两个已有的坑同时变复杂。所以它现在只被
    文档与将来的后端适配层使用。
    """
    return (
        "把回答写成两段：\n"
        "<think>\n意图: answer|ask|refuse|emotion|action\n"
        "要点:\n- 第一条\n- 第二条\n语气指示: casual|formal|tease|comfort|direct\n"
        "长度: short|medium|long\n必须保留的事实:\n- 原样抄写的数字/路径/专名\n</think>\n"
        "<content>原始草稿（可留空）</content>\n"
        "不要在这两段之外输出别的内容；不要加 emoji 与语气词——那些由表达层决定。"
    )


def render_prompt(sir: SIR, persona: str, *, style_note: str = "") -> str:
    """表达层的系统提示。**人格与内容都在里面，但只有内容被逐条列出。**

    ``facts`` 单独成段并要求原样出现，是为了让"不许编"这条在提示词里有位置——
    光靠一句"别改事实"，模型写顺了手照样会改。
    """
    points = "\n".join(f"- {point}" for point in sir.key_points) or "（无）"
    facts = "\n".join(f"- {fact}" for fact in sir.facts) or "（无）"
    tone = {
        "casual": "自然、不端着",
        "formal": "正式、克制",
        "tease": "带一点调侃",
        "comfort": "温和、先接住情绪",
        "direct": "直接，不铺垫",
    }.get(sir.tone, "自然")
    length = {"short": "一两句", "medium": "三到五句", "long": "可以展开"}.get(
        sir.length, "三到五句"
    )
    lines = [
        persona.strip(),
        "",
        "下面是一段内容意图。**只决定怎么说，不要改说什么。**",
        "规则：",
        "- 每个要点都要说到，可以合并、可以换词；不许增加原文没有的信息。",
        "- 「必须保留的事实」必须原样出现，一个字符都不改。",
        "- 不分点、不列 1./2./3.、不写「首先/其次」。",
        "- 不用 emoji、不用颜文字、不多加感叹号。",
        f"- 语气：{tone}。长度：{length}。",
    ]
    if style_note.strip():
        lines.append(f"- {style_note.strip()}")
    lines += ["", f"意图：{sir.intent}", "要点：", points, "必须保留的事实：", facts]
    if sir.draft.strip():
        lines += ["", "草稿（可参考，不必遵循）：", sir.draft.strip()]
    return "\n".join(lines)


_CASE_SENSITIVE_FACT = re.compile(r"^(https?://|/|[A-Za-z]:\\)")


def _normalized(text: str, *, lower: bool = True) -> str:
    folded = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    return folded.lower() if lower else folded


def _fact_is_case_sensitive(fact: str) -> bool:
    """URL 与路径区分大小写；其余（数字、中文专名）不区分。

    Windows 路径其实不分大小写，但同一个 URL 在不同服务器上可能分——**宁可让表达层
    被判成改了事实、退回草稿，也不让一个链接指到别处**。判定方向是保守的那一边。
    """
    return bool(_CASE_SENSITIVE_FACT.match(fact))


def facts_preserved(rendered: str, facts: tuple[str, ...]) -> tuple[str, ...]:
    """返回**丢掉的事实**；空元组表示全都在。

    判据是"归一化后仍是子串"：去空白、NFKC、小写。中文没有词边界，能核对的只有
    "这串字符还在不在"。数字因此会被原样比对（``1,000`` 与 ``1000`` 算不同——这不是
    吹毛求疵：财务与版本号里那点差别正是不能改的地方）。
    """
    folded = _normalized(rendered)
    exact = _normalized(rendered, lower=False)
    missing: list[str] = []
    for fact in facts:
        if _fact_is_case_sensitive(fact):
            if _normalized(fact, lower=False) not in exact:
                missing.append(fact)
        elif _normalized(fact) not in folded:
            missing.append(fact)
    return tuple(missing)


def style_signature(text: str) -> dict[str, float]:
    """一段文本的可量化风格指标。**用于验收对比，不参与生成。**

    评"语气一致"最容易滑进主观判断，而主观判断在三个后端之间比不出东西。这里挑的
    全是确定性指标：句长、感叹号/问号/emoji 密度、连接词使用。它们不完美——写得
    短而冷淡与写得短而干脆在这组数上一样——但**它们能被复现**，而"看着像千岁"不能。
    """
    source = text or ""
    sentences = [part for part in _SENTENCE_END.split(source) if part.strip()]
    length = len(source)
    return {
        "chars": float(length),
        "sentences": float(len(sentences)),
        "avg_sentence": float(length / len(sentences)) if sentences else 0.0,
        "exclamations": float(source.count("！") + source.count("!")),
        "questions": float(source.count("？") + source.count("?")),
        "emoji": float(len(_EMOJI.findall(source))),
        "connectives": float(sum(source.count(word) for word in _CONNECTIVES)),
        "bullets": float(source.count("\n- ") + source.count("\n* ")),
    }
