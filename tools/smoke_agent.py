"""Step 0 探针：**本地模型到底会不会真的吐出工具调用。**

这是 P3a 里唯一"整个方向都压在它上面"的假设。协议层早就写好了
（``llm/base.py`` 的 :class:`ToolSpec`/:class:`ToolCall`，``llm/ollama.py`` 的
``_build_tool`` / ``_parse_tool_calls`` / ``_as_arguments`` / ``_build_message``），
``base.py:190-192`` 甚至写着"Ollama 的 tool_calls 整条一次到齐（实测）"——但
**本仓从未真的发过一次带 tools 的请求**：``grep 'ToolSpec('`` 只有定义处，那几个
解析器零生产调用、零测试覆盖。注释里的"实测"是写下它的那次实测，不是这个仓跑出来的。

所以先把假设验掉，再决定后面那七步值不值得做。它量四样东西：

1. **发不发得出**——模型会不会调、调得对不对（名字、参数形状、有没有 id）；
2. **几轮收敛**——:data:`DEFAULT_MAX_ROUNDS` 的 4 是拍的，这里量出真实分布；
3. **工具占多少 token**——``tools`` 声明 + ``tool_calls`` + 结果回灌的真实开销
   （``session/budget.py:116`` 自认"这个数未标定"，L5 的份额 0.20 同样是拍的）；
4. **回灌后认不认**——模型看到工具结果之后是**用它**，还是继续按自己的印象编。
   这一条比格式正确更重要：格式对但无视结果，工具循环就没有意义。

**它不设确认闸门。** 改状态的工具在这里直接执行，只打印一行"真跑起来这里会问一次"
——闸门与跨线程握手是要单独验的另一件事，本脚本量的是"模型这一侧"。

用法::

    uv run python tools/smoke_agent.py                    # 第一个可用模型，跑全部探针
    uv run python tools/smoke_agent.py qwen3-4b           # 指定模型
    uv run python tools/smoke_agent.py --think            # 打开思考模式（默认关，见下）
    uv run python tools/smoke_agent.py --dump raw.jsonl   # 冻下原始事件当测试夹具

**默认 ``think=False``**，这不是省事的默认值，是量出来的：开着思考时首字延迟从
0.06s 涨到 1.0–2.0s，输出 token 涨到 3–10 倍，而**多步用例的准确率反而更低**
（见下）。4B 模型上，推理把输出预算烧光、连工具都不调了——两次运行撞在
``max_tokens`` 上限上、整轮零工具调用。``--think`` 保留为对照。

────────────────────────────────────────────────────────────────────────────
2026-10-10 首次实测（``qwen3-4b:latest`` / Ollama 0.40.2 / ``think=False``）
────────────────────────────────────────────────────────────────────────────

**一、协议形状：与 ``llm/base.py`` 的注释完全一致，好于预期。**
``tool_calls`` 与 ``done: true`` **在同一片**到达（不分片，故消费端确实不需要
碎片累积状态机）；无 ``type: "function"`` 外壳；``arguments`` 是**对象**；
``function.index`` 在；id 为 32 字符且唯一；带工具调用时 ``content`` 是空串。
"请求 → 执行 → 回灌 → 继续"整圈跑得通，退出码 0。
这四条真实字节冻在 ``tests/fixtures/ollama_tool_calls.json``，
由 ``tests/test_ollama_tools.py`` 逐条钉住。

**二、单工具选择：13/13 全对。** 名字、参数、要不要调，全对；负对照（纯闲聊）
从不误调。**格式与单步选择这一层可以信。**

**三、多步（"先查再写"）：约 1/5。这是本次最重要的发现。**
五轮对照里只有一轮真的调了 ``search_memory`` → ``write_memory``；四轮失败，
而失败的主形态不是报错、不是乱调，是**声称做了却没做**：

    用户：查一下我常去那家面馆周几休息，然后把这条记下来。
    → search_memory（查到了）
    → 模型说：「您常去的那家面馆周一休息。**这条信息已记入长期记忆库。**」
    → write_memory：**一次都没被调用**

次形态是调错工具（该写库时去调了 ``get_time``）。**这一条决定 P3a 的三处设计**：

1. **界面必须显示真实的工具轨迹，绝不转述模型的自述。** 对一个记忆系统来说，
   假的"已记住"是最坏的失败——用户会因此不再自己记着这件事。模型说"已记下"时
   真正发生的事，只有工具调用记录知道。
2. **确认闸门是承重件，不是礼貌。** 改状态的工具必须经用户点头，正是这道闸门让
   模型的不可靠变得无害：没执行就是没执行，卡片不出现本身就是信号。
3. **路由的模型自分类保持关闭**（用户已拍板，此处补上证据）：一个连"两个工具
   该调哪个"都会错的模型，不足以可靠地判自己这一轮属于哪个语域。
   ``/task`` 这类显式命令因此是**主要**的可靠入口，不是便捷方式。

**四、``max_rounds``：观测到的工具轮数分布是 1,1,1,1,2（max=2）。**
``DEFAULT_MAX_ROUNDS = 4`` 作为**上限**是安全的（它是天花板不是目标）。
但注意上一条：多数用例在第 1 轮就"收敛"，不是因为任务做完了，而是因为模型
把该分两步做的事在一轮里说完了。所以收尾轮（到顶后用 ``tools=()`` 再问一次）
比原计划设想的更重要——它不只是"轮数用尽"的兜底，也是多步失败后的常规路径。

**五、工具声明的固定开销：3 个工具 = 净增 311 token/轮**（带 tools 326 vs 不带 15）。
**每一轮都要付**，且随工具数近似线性增长——真正要上的七个工具大约是这个数的两倍。
这是纯开销（不是 L5 的内容预算）；收尾轮用 ``tools=()`` 能把它省掉。

**未测的**：本地只装了 ``qwen3-4b``，**n=1 个模型**。换一个更大的本地模型结论
可能不同，本脚本可重跑。上面的"1/5"是**这一个 4B 模型**上的数，不是本地模型
这个类别的数。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from biosnn_memterm.llm.base import (
    ChatBackendError,
    ChatMessage,
    ChatRequest,
    ToolCall,
    ToolSpec,
)
from biosnn_memterm.llm.ollama import OllamaChatBackend, list_local_models, probe
from memcore.recall import recall
from memcore.sdr import encode_lexical
from memcore.store import MemoryStore
from memcore.types import MemoryItem, MemoryKind, Tier

DEFAULT_MAX_ROUNDS = 4
"""一轮对话里最多允许"模型请求工具 → 执行 → 回灌"几次。**先验值，不是标定值。**
本脚本量出的轮数分布就是标定它的依据。"""


# ── 库：一个一次性的内存库，几条"可检索"的事实 ─────────────────────────────
#
# 用 ``:memory:`` 而不是真库，理由与 bench/harness.py 一字不差：探针会被反复跑，
# 它要是碰了 BIOSNN_MEMTERM_DATA 指向的真库，就会有"跑一次探针把用户记忆的权重
# 动了一遍"这种事故，而且没有任何症状。

_FACTS = (
    "家里的路由器放在书架第二层。",
    "备用钥匙放在门口鞋柜最上面那一格。",
    "上周三去看了牙医，下次复查排在十一月。",
    "那本《哥德尔、艾舍尔、巴赫》借给小林了。",
    "常去的那家面馆周一休息。",
)


def _seed_store() -> MemoryStore:
    """建一个内存库，灌进 :data:`_FACTS`。

    ``lex_sdr`` 在建库时就算好（``encode_lexical`` 是纯 blake2b，不调模型），
    于是 ``search_memory`` 走纯词法通道也能返回东西——探针要验的是工具协议，
    不该被"有没有装嵌入模型"这件事干扰。
    """
    store = MemoryStore(":memory:")
    for index, content in enumerate(_FACTS):
        store.add(
            MemoryItem(
                memory_id=f"probe-{index:03d}",
                kind=MemoryKind.FACT,
                tier=Tier.LONGTERM,
                content=content,
                weight=0.7,
                lex_sdr=encode_lexical(content),
            )
        )
    return store


# ── 三个工具：两个只读，一个改状态 ─────────────────────────────────────────
#
# 形状刻意与 P3a 真正要落的那七个一致（``ToolSpec`` + 一个 run 函数），
# 这样探针验通之后，代码可以直接长出来而不是重写一遍。

_GET_TIME = ToolSpec(
    name="get_time",
    description="查当前日期与时间。任何涉及'现在''今天''几号'的问题都要先调它。",
    parameters={"type": "object", "properties": {}, "required": []},
)

_SEARCH_MEMORY = ToolSpec(
    name="search_memory",
    description="在长期记忆库里检索。当你不确定某件事的细节时先查，不要凭印象回答。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索用的关键词或短句"},
        },
        "required": ["query"],
    },
)

_WRITE_MEMORY = ToolSpec(
    name="write_memory",
    description="把一条值得长期记住的事实写进记忆库。用户明确要求'记下来'时才用。",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "要记住的那句话，自包含、第三人称"},
        },
        "required": ["content"],
    },
)

TOOLS: tuple[ToolSpec, ...] = (_GET_TIME, _SEARCH_MEMORY, _WRITE_MEMORY)

#: 真跑起来要不要过确认闸门。探针不做闸门，但要**打印出这个区别**——否则
#: 读输出的人会以为 P3a 的默认行为就是"模型说要写就写"。
NEEDS_APPROVAL = frozenset({"write_memory"})


def _run_get_time(arguments: dict[str, Any], store: MemoryStore) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S（%A）")


def _run_search_memory(arguments: dict[str, Any], store: MemoryStore) -> str:
    query = str(arguments.get("query") or "").strip()
    if not query:
        return "错误：query 不能为空。"
    ledger = recall(store, query_text=query, budget_tokens=400, settle=False)
    if not ledger.injected:
        return f"没有找到与{query!r}相关的记忆。"
    lines = [f"- {item.content}" for item in ledger.injected]
    return f"检索到 {len(lines)} 条：\n" + "\n".join(lines)


_seeded_writes: list[str] = []


def _run_write_memory(arguments: dict[str, Any], store: MemoryStore) -> str:
    content = str(arguments.get("content") or "").strip()
    if not content:
        return "错误：content 不能为空。"
    # 探针不真写库（写了会污染下一次探针的检索结果），只记下来。
    _seeded_writes.append(content)
    return f"已记下：{content}"


_DISPATCH = {
    "get_time": _run_get_time,
    "search_memory": _run_search_memory,
    "write_memory": _run_write_memory,
}


# ── 探针用例 ───────────────────────────────────────────────────────────────
#
# 四条覆盖四种情形：单工具（时间）、单工具（检索）、多步（先查再写）、
# 以及**负对照**——纯闲聊不该调工具。负对照是必须的：一个"永远调工具"的模型
# 在前三条上看起来完全正常。
#
# **``expect`` 是这条用例的判据，不是提示。** 第一版没有它，只问"调没调工具"，
# 于是下面这条用例——用户明确要求"把这条记下来"、模型只查不写、还在正文里宣称
# "这条信息已记录下来"——被判成了「通过」。**判据不能只问'有没有动静'，要问
# '该做的做完没有'**：一个会在幻觉上报绿的判据比没有判据更坏，因为它把唯一的
# 报警灯涂成了绿色。


@dataclass(frozen=True)
class _Probe:
    label: str
    prompt: str
    expect: tuple[str, ...]
    """这一轮**必须**被调用到的工具名（有序，按应当发生的先后）。
    空元组 = 这一轮一个工具都不该被调用。"""

    claims: tuple[str, ...] = ()
    """出现了这些词就等于在**声称做过某事**。与 ``expect`` 配合使用：正文里声称
    做了、调用记录里却没有，就是幻觉——单独一次实测已经撞见过一次。"""


_PROBES: tuple[_Probe, ...] = (
    _Probe("单工具·时间", "现在几点？只要时间。", ("get_time",)),
    _Probe("单工具·检索", "我的备用钥匙放哪儿了？", ("search_memory",)),
    _Probe(
        "多步·先查再写",
        "查一下我常去那家面馆周几休息，然后把这条记下来。",
        ("search_memory", "write_memory"),
        claims=("已记", "记下", "记录下", "已保存", "存好"),
    ),
    _Probe("负对照·闲聊", "今天有点累，随便聊两句吧。", ()),
)


# ── 跑一轮 ─────────────────────────────────────────────────────────────────


def _stream_once(
    backend: OllamaChatBackend,
    working: list[ChatMessage],
    *,
    tools: tuple[ToolSpec, ...],
    think: bool,
    max_tokens: int,
) -> tuple[str, tuple[ToolCall, ...], int | None, int | None, list[dict[str, Any]], float]:
    """发一次请求，收完一整条流。

    返回 ``(文本, 工具调用, 输入 token, 输出 token, 原始事件, 首字延迟)``。

    **工具调用直接从片里取，不做任何累积拼接。** ``base.py:190-192`` 保证
    Ollama 的 ``tool_calls`` 整条一次到齐（它不像 OpenAI 那样按 index 流式拼
    JSON 字符串碎片）。探针顺带验的就是这条保证——如果哪次真的分片到达，下面的
    ``calls += chunk.tool_calls`` 会拼出重复项而不是报错，届时打印出来的 id
    能看出来。
    """
    request = ChatRequest(
        messages=working,
        tools=tools,
        max_tokens=max_tokens,
        # "think" 是少数几个属于请求体顶层（而不是 options）的参数，见
        # ollama.py 的 _TOP_LEVEL_EXTRA。落进 options 会被静默忽略。
        extra={} if think else {"think": False},
    )
    parts: list[str] = []
    calls: tuple[ToolCall, ...] = ()
    raw_events: list[dict[str, Any]] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    started = time.monotonic()
    first_at: float | None = None

    for chunk in backend.stream(request):
        if chunk.text:
            if first_at is None:
                first_at = time.monotonic() - started
            parts.append(chunk.text)
        if chunk.tool_calls:
            calls += chunk.tool_calls
        if chunk.raw:
            raw_events.append(chunk.raw)
        if chunk.done:
            prompt_tokens = chunk.prompt_tokens
            completion_tokens = chunk.completion_tokens

    return (
        "".join(parts),
        calls,
        prompt_tokens,
        completion_tokens,
        raw_events,
        first_at if first_at is not None else -1.0,
    )


def _run_probe(
    backend: OllamaChatBackend,
    probe: _Probe,
    store: MemoryStore,
    *,
    think: bool,
    max_rounds: int,
    dump: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """跑完一条探针，返回它的证据。"""
    label, prompt = probe.label, probe.prompt
    print(f"\n{'=' * 72}")
    print(f"探针：{label}    期望：{probe.expect or '（不调工具）'}")
    print(f"用户：{prompt}")
    print("-" * 72)

    working: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]
    rounds = 0
    total_prompt = 0
    total_completion = 0
    called: list[str] = []
    final_text = ""
    first_latency = -1.0

    while True:
        try:
            text, calls, ptok, ctok, events, latency = _stream_once(
                backend, working, tools=TOOLS, think=think, max_tokens=512
            )
        except ChatBackendError as exc:
            print(f"  !! 后端出错：{exc}")
            return {
                "label": label,
                "error": str(exc),
                "rounds": rounds,
                "called": called,
                "verdict": f"失败：后端出错 {exc}",
            }

        total_prompt += ptok or 0
        total_completion += ctok or 0
        if first_latency < 0:
            first_latency = latency
        if dump is not None:
            for event in events:
                dump.append({"probe": label, "round": rounds, "event": event})

        if text.strip():
            print(f"  [第 {rounds} 轮模型说] {text.strip()}")

        if not calls:
            final_text = text.strip()
            print(f"  [第 {rounds} 轮无工具调用 → 收敛]")
            break

        rounds += 1
        print(f"  [第 {rounds} 轮工具调用 × {len(calls)}]")
        # 工具调用与它的结果必须成对进 working：assistant 那条带 tool_calls，
        # 紧随其后的 tool 条带同一个 id。ChatMessage.__post_init__ 会挡掉违反
        # 这两条不变量的写法（base.py:132-144），这里不必重复校验。
        working.append(ChatMessage(role="assistant", content=text, tool_calls=calls))

        for call in calls:
            called.append(call.name)
            gate = "  ← 真跑起来这里会问一次（改状态）" if call.name in NEEDS_APPROVAL else ""
            print(f"    · {call.name} id={call.id} args={call.arguments!r}{gate}")
            runner = _DISPATCH.get(call.name)
            if runner is None:
                result = f"错误：没有名为 {call.name} 的工具。"
            else:
                try:
                    result = runner(call.arguments, store)
                except Exception as exc:  # noqa: BLE001 — 探针：把工具内部错误照实抖出来
                    result = f"错误：{type(exc).__name__}: {exc}"
            # 打全文不截行：回灌的正文正是模型下一轮看到的东西，截了就看不出
            # "它到底收到了什么"——而本轮最要紧的那个失败（收到了检索结果却
            # 没写库）恰恰要在"它收到了什么"上判。
            for line in (result or "(空)").splitlines():
                print(f"      → {line}")
            working.append(ChatMessage(role="tool", content=result, tool_call_id=call.id))

        if rounds >= max_rounds:
            print(f"  [已达上限 {max_rounds} 轮，停止]")
            break

    verdict = _verdict(probe, called, rounds, final_text)
    print(f"  {'✓' if verdict.startswith('通过') else '✗'} 结论：{verdict}")
    return {
        "label": label,
        "prompt": prompt,
        "expect": list(probe.expect),
        "rounds": rounds,
        "called": called,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "first_latency_s": round(first_latency, 3),
        "final": final_text,
        "verdict": verdict,
    }


def _verdict(probe: _Probe, called: list[str], rounds: int, final_text: str) -> str:
    """给人看的一句话结论。**判据写在这里，不靠读输出的人自己总结。**

    判的是**该做的做完没有**，不是"有没有动静"：``expect`` 里每个工具都必须真的
    出现过。光看"调了没有"会在下面这条上判绿——用户要求"记下来"、模型查完直接
    宣称"已记录"、调用记录里根本没有 ``write_memory``。
    """
    missing = [name for name in probe.expect if name not in called]
    hallucinated = bool(probe.claims) and any(word in final_text for word in probe.claims)

    if not probe.expect:
        # 负对照：一个工具都不该调。它挡的是"永远调工具"的模型——那种模型在
        # 别的用例上看起来完全正常。
        return "通过（没调工具）" if not called else f"失败：闲聊却调了 {called}"

    if hallucinated and missing:
        # **本探针最重要的一条。** 模型在正文里说"已记下"，而它一次都没调
        # ``write_memory``。这不是格式问题，是"声称做了却没做"——界面必须显示
        # 真实的工具轨迹，绝不能转述模型的自述。
        return f"失败：声称做了但没调 {missing}（正文说'{final_text[:40]}…'）"
    if missing:
        return f"失败：该调 {missing} 却没调（实调 {called or '无'}）"
    if not final_text:
        return "可疑：调了工具但最后没有给出回答"
    return f"通过（{rounds} 轮工具，调了 {called}）"


# ── 工具声明的 token 开销 ──────────────────────────────────────────────────


def _measure_declaration_cost(
    backend: OllamaChatBackend, prompt: str, *, think: bool
) -> tuple[int | None, int | None]:
    """同一个 prompt 各发一次"带 tools"与"不带 tools"，比较输入 token。

    这是 L5 份额与 :meth:`TokenMeter.tool_calls` 的标定来源之一——工具声明是
    **每一轮都要付**的固定开销，它有多大决定了"值不值得默认开着工具"。
    """
    with_tools = ChatRequest(
        messages=[ChatMessage(role="user", content=prompt)],
        tools=TOOLS,
        max_tokens=1,
        extra={} if think else {"think": False},
    )
    without = ChatRequest(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=1,
        extra={} if think else {"think": False},
    )
    values: list[int | None] = []
    for request in (with_tools, without):
        last: int | None = None
        try:
            for chunk in backend.stream(request):
                if chunk.done:
                    last = chunk.prompt_tokens
        except ChatBackendError:
            return None, None
        values.append(last)
    return values[0], values[1]


# ── 入口 ───────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    if "--list" in argv:
        return _list_models()

    version = probe()
    if version is None:
        print("Ollama 没在运行。启动命令：ollama serve")
        return 1
    print(f"Ollama {version}")

    backend = _pick_backend(argv)
    if backend is None:
        return 1

    think = "--think" in argv
    max_rounds = _int_arg(argv, "--rounds", DEFAULT_MAX_ROUNDS)
    dump_path = _str_arg(argv, "--dump")

    print(f"模型 {backend.model_id}")
    print(f"思考模式：{'开' if think else '关'}（--think 切换）")
    print(f"轮数上限：{max_rounds}")
    print(f"工具：{', '.join(spec.name for spec in TOOLS)}")

    store = _seed_store()
    print(f"库：内存，{store.count()} 条")

    dump: list[dict[str, Any]] | None = [] if dump_path else None
    results: list[dict[str, Any]] = []
    # 循环变量别叫 probe：那会遮蔽从 llm.ollama 导入的探活函数 ``probe()``，
    # 让本函数开头的 ``probe()`` 变成 UnboundLocalError（函数体内只要有赋值，
    # 这个名字就全程是局部的）。
    for case in _PROBES:
        results.append(
            _run_probe(
                backend,
                case,
                store,
                think=think,
                max_rounds=max_rounds,
                dump=dump,
            )
        )

    _report(results, backend, think)

    if dump_path and dump:
        path = Path(dump_path)
        # newline="\n"：Windows 上 write_text 默认写 CRLF，而这份夹具要进仓库
        # 并被逐字节比较——行尾漂移会让"冻下来的字节"失去意义。
        path.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in dump) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n原始事件已写入 {path}（{len(dump)} 条）——这是测试夹具的来源。")

    failed = [r for r in results if not str(r.get("verdict", "")).startswith("通过")]
    return 1 if failed else 0


def _report(results: list[dict[str, Any]], backend: OllamaChatBackend, think: bool) -> None:
    print(f"\n{'=' * 72}")
    print("汇总")
    print("-" * 72)
    for entry in results:
        print(
            f"  {entry['label']:<14} 轮={entry['rounds']} "
            f"调={entry.get('called') or '无'} "
            f"in={entry.get('prompt_tokens')} out={entry.get('completion_tokens')} "
            f"首字={entry.get('first_latency_s')}s"
        )
        print(f"    {entry.get('verdict')}")

    rounds = [entry["rounds"] for entry in results if "error" not in entry]
    if rounds:
        print(f"\n  工具轮数分布：{sorted(rounds)}（max={max(rounds)}）")
        print(
            "    ↑ 这是 **max_rounds 的标定来源**。注意它同时说明别的：若每条用例都在\n"
            "      第 1 轮就收敛，说明这套探针**没有真的测到多轮**——'先查再写'这类\n"
            "      用例本该走两轮，只走一轮就是在第一轮就把两件事都'说完了'。"
        )

    with_tools, without = _measure_declaration_cost(backend, "现在几点？", think=think)
    if with_tools is not None and without is not None:
        print(
            f"  工具声明的固定开销：带 tools {with_tools} token vs 不带 {without} "
            f"→ **净增 {with_tools - without} token/轮**（每轮都要付）"
        )
    failed = [
        entry["label"] for entry in results if not str(entry.get("verdict", "")).startswith("通过")
    ]
    print(f"\n  结论：{'全部通过' if not failed else f'{len(failed)} 条未通过 → {failed}'}")


def _pick_backend(argv: list[str]) -> OllamaChatBackend | None:
    models = list_local_models()
    if not models:
        print("Ollama 在运行，但没有模型。")
        return None
    wanted = next((arg for arg in argv[1:] if not arg.startswith("-")), None)
    short = {entry.get("short_name") or entry["name"]: entry for entry in models}
    by_name = {entry["name"]: entry for entry in models}

    if wanted is None:
        # **不能拿 models[0]**：这个列表按名字排序，排头的是 nomic-embed-text
        # ——一个嵌入模型，没有 tools 能力，跑出来只是"模型不调工具"的假阴性。
        entry = next((e for e in models if "tools" in (e.get("capabilities") or ())), models[0])
    elif wanted in short:
        entry = short[wanted]
    elif wanted in by_name:
        entry = by_name[wanted]
    else:
        print(f"没有模型 {wanted!r}。可用：{', '.join(sorted(short))}")
        return None

    caps = tuple(entry.get("capabilities") or ())
    if "tools" not in caps:
        # 不拦，但必须喊出来：这个模型的"不调工具"不是被测出来的结论。
        print(f"!! {entry['name']} 没声明 tools 能力（caps={caps or '无'}）——它会忽略工具声明。")
    return OllamaChatBackend(model=entry["name"])


def _list_models() -> int:
    models = list_local_models()
    if not models:
        print("本地一个模型都没有。")
        return 1
    for entry in models:
        size_gb = entry["size"] / 1024**3
        name = entry.get("short_name") or entry["name"]
        caps = ",".join(entry.get("capabilities") or [])
        print(f"  {name:<24} {entry['parameter_size']:>6} {size_gb:>5.2f} GB  [{caps}]")
    return 0


def _int_arg(argv: list[str], flag: str, default: int) -> int:
    if flag not in argv:
        return default
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return default
    try:
        return int(argv[index + 1])
    except ValueError:
        return default


def _str_arg(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    return argv[index + 1] if index + 1 < len(argv) else None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
