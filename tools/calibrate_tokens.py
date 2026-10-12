"""标定 :func:`memcore.tokens.estimate_tokens` 的四个系数。

:mod:`memcore.tokens` 的模块 docstring 里写着"系数是量出来的，不是猜的"，这个脚本
就是那句话的**出处**。没有它，那四个数字会退化成没人敢动的魔数——而一个不敢动的
系数，在换分词器的那天只会被原样留着。

做法：把一段文本作为**单独一条 user 消息**发给 ``/api/chat``（``num_predict=1``），
读 ``prompt_eval_count``，扣掉整轮对话的固定底价 :data:`BASE_PRICE`。不扣的话，
每条样本都会被加上同一个常数，短样本的比值全是错的。

``BASE_PRICE`` 本身也是量出来的，见 ``--measure-base-price``（两个探针互相印证：
消息包装 5 + 生成提示 7）。它换模型时会变，所以这个脚本每次运行都会把它印出来。

**这个脚本问的不是"平均准不准"，而是"会不会低估"。** 低估让预算算大，而算大的
后果是 Ollama 在服务端静默截断——截掉的是最早的消息，正是人格锚所在的位置。
所以两个汇总表都要看：分档表给的是每个类别的上沿（取上沿才不低估），样本表给的
是最差比值（它决定装配层需要多大的安全余量）。

用法::

    python tools/calibrate_tokens.py                    # 标定 + 复核
    python tools/calibrate_tokens.py --model bge-m3     # 换分词器重标
    python tools/calibrate_tokens.py --measure-base-price

退出码：低估超过 :data:`UNDER_COUNT_TOLERANCE`（文档承诺的容差）为 1，否则为 0。
**不等于程序出错**——它表达的是"当前系数已经配不上这套分词器了"。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memcore.text import is_cjk  # noqa: E402
from memcore.tokens import (  # noqa: E402
    ESTIMATOR_VERSION,
    UNDER_COUNT_TOLERANCE,
    estimate_tokens,
)

DEFAULT_HOST = "http://127.0.0.1:11434"

#: 整轮对话的固定底价：消息包装 5 + 生成提示 7。**不是**每条消息的开销（那是 5）。
#: 量法见 ``--measure-base-price``：空内容发一条，``prompt_eval_count`` 就是它。
BASE_PRICE = 12

#: 分档样本。每类三个，是为了看出**类内差异**——标点那一类内部差 32 倍，正是
#: "一个系数盖不住所有字符"的证据。
GROUPS: dict[str, list[str]] = {
    "汉字": [
        "记忆终端的装配预算需要一个可靠的常数",
        "上海自来水来自海上山西运煤车煤运西山",
        "如果这个常数取得太大就会白白扔掉几条本来放得下的记忆",
    ],
    "数字": [
        "16384 32768 4096 0123456789",
        "0.40.2 6.7.2 1.24 3.10 2026-10-10",
        "123456789012345678901234567890",
    ],
    "拉丁字母（散文）": [
        "the quick brown fox jumps over the lazy dog again and again today",
        "memory terminal stores long term memory on the terminal side",
        "a reliable per message overhead constant for the assembly budget",
    ],
    "拉丁字母（标识符与路径）": [
        "num_ctx prompt_eval_count tool_call_id memory_id space_id lex_sdr",
        # 这一条与下面那条混合样本第二份，原本写的是**带盘符的绝对检出路径**。转公开时
        # 换成了仓库相对路径：样本要的是"一串带反斜杠的路径"这个形状，盘符与检出目录不在
        # 测量的范围里。2026-10-12 复测：这一档上沿仍是 0.329（本条 0.328，旧样本 0.327），
        # 所以 memcore/tokens.py 里那个 0.35 依然盖得住，系数不动。
        "src\\memcore\\recall.py src\\axo\\ui\\main_window.py",
        "src/memcore/recall.py src/axo/llm/base.py tests/test_sdr.py",
    ],
    "标点": [
        "。，。，。，。，。，。，。，。，。，。，",
        "！？；：、（）《》【】…—·「」『』",
        "--------------------------------",
    ],
    # 星平面字符既不是 CJK 也不是 alnum，落进"其他"档按 1.00 算。但 emoji 在
    # BPE 里常常一个字符就吃掉 2~3 个 token，组合序列更多——这一档是用来验证
    # "其他 = 1.00" 到底够不够的。
    "emoji": [
        "😀😀😀😀😀😀😀😀",
        "今天心情不错 👍，明天继续 🚀。",
        "👨👩👧👦 🇨🇳 🏳️🌈",
    ],
}

#: 混合样本：真实记忆与真实提问的样子。（文本, 实测 token）。实测值由
#: ``--refresh`` 重测；默认直接用这里的值，好让脚本在没改系数时也能复核文档里的表。
SAMPLES: list[str] = [
    "记忆终端的装配预算需要一个可靠的每条消息开销常数。",
    "如果这个常数取得太大，就会白白扔掉几条本来放得下的记忆；"
    "取得太小则会在服务端被静默截断，而截掉的是最早的人格锚。",
    "症状是模型忽然忘了自己是谁，排查起来极难定位到预算这一步。",
    "我家的路由器放在书房书架顶层。",
    "今天午饭吃的是番茄炒蛋和米饭。",
    "每周三晚上八点要开组会，会议纪要在共享盘上。",
    "记忆终端用 SQLite 存记忆库，WAL 模式，每条写入都是一个事务。",
    "Axo 是一个把长期记忆放在终端侧的桌面助手，独立于所调用的大模型。",
    "配置项 num_ctx 的值是 16384，不是 32768。",
    "版本 0.40.2 的 Ollama 在 /api/chat 上返回 NDJSON 流。",
    "读取 README.md 并总结前三段。",
    "把 src/memcore/recall.py 里 _fuse 函数的注释改得更清楚一些。",
    "为什么用 blake2b 而不是 numpy 的随机数生成器？",
    "上海自来水来自海上，山西运煤车煤运西山。",
    "这段文本里中英文混排 mixed with English words and 1234 digits.",
    "def recall(store, *, query_text, budget_tokens): return None",
    "短句。",
    "这是一段刻意写得很长的文本，用来测试估计器在长文本上会不会因为重复的二元组而失准，"
    "因为词法 SDR 的去冗与预算都建立在同一套分词之上，如果估计器在长文本上系统性偏低，"
    "那么一条很长的记忆就会以『看起来装得下』的样子挤掉后面好几条短的。",
    "## 结论\n\n---\n\n| 通道 | 权重 |\n|---|---|\n| 词法 | 0.90 |\n",
    "```python\n# ---------------------------------------------------------------\ndef f():\n"
    "    return 1\n```",
]


def call(host: str, model: str, messages: list[dict], num_predict: int = 1) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise SystemExit(f"Ollama 拒绝了这个请求（{exc.code}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"连不上 Ollama（{host}）：{exc.reason}") from exc


def pick_model(host: str, wanted: str | None) -> str:
    if wanted:
        return wanted
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as response:
        names = [entry["name"] for entry in json.load(response).get("models", [])]
    qwen = [name for name in names if name.startswith("qwen3")]
    if not qwen:
        raise SystemExit(f"没找到 qwen3 系列模型，现有：{names or '（空）'}")
    return qwen[0]


def real_tokens(host: str, model: str, text: str) -> int:
    """一段文本的真实 token 数。底价按 :data:`BASE_PRICE` 扣。"""
    body = call(host, model, [{"role": "user", "content": text}])
    return int(body.get("prompt_eval_count") or 0) - BASE_PRICE


def classify(text: str) -> dict[str, int]:
    """按 :func:`memcore.tokens.estimate_tokens` 的分档数一遍字符。"""
    buckets = {"cjk": 0, "digit": 0, "latin": 0, "other": 0, "space": 0}
    for char in text:
        if char.isspace():
            buckets["space"] += 1
        elif char.isdigit():
            buckets["digit"] += 1
        elif is_cjk(char):
            buckets["cjk"] += 1
        elif char.isalnum():
            buckets["latin"] += 1
        else:
            buckets["other"] += 1
    return buckets


def measure_base_price(host: str, model: str) -> None:
    """把固定底价与每条消息开销解出来，让 :data:`BASE_PRICE` 可复核。

    沿用 P0 的三请求差值法（同文本 F）::

        A = [user: F]           1 条消息
        B = [user: F, user: F]  2 条消息，同文本
        C = [user: F+F]         1 条消息，文本翻倍（只有一份包装）

        tokens(F) = C − A        overhead = (B − A) − tokens(F)
    """
    text = "记忆终端的装配预算需要一个可靠的每条消息开销常数，" * 3
    price_a = int(call(host, model, [{"role": "user", "content": text}])["prompt_eval_count"])
    price_b = int(call(host, model, [{"role": "user", "content": text}] * 2)["prompt_eval_count"])
    price_c = int(
        call(host, model, [{"role": "user", "content": text + text}])["prompt_eval_count"]
    )
    price_empty = int(call(host, model, [{"role": "user", "content": ""}])["prompt_eval_count"])

    tokens = price_c - price_a
    overhead = (price_b - price_a) - tokens
    print(f"空内容一条        = {price_empty}   ← 固定底价（包装 + 生成提示）")
    print(f"A [user F]        = {price_a}   （F 长 {len(text)} 字符）")
    print(f"B [user F, user F]= {price_b}")
    print(f"C [user F+F]      = {price_c}")
    print(f"\ntokens(F)         = C−A = {tokens}   （{tokens / len(text):.3f} token/字符）")
    print(f"每条消息模板开销   = B−C = {overhead}")
    print(f"\n脚本用的 BASE_PRICE = {BASE_PRICE}；实测 {price_empty}", end="")
    print("  ✅" if price_empty == BASE_PRICE else "  ⚠️ 不一致，改掉脚本里的常数")


def per_category(host: str, model: str) -> None:
    """分档量。**上沿**那一列才是系数该取的值。"""
    print("── 分档（token/字符）──")
    for label, samples in GROUPS.items():
        print(f"  {label}")
        lows: list[float] = []
        for text in samples:
            real = real_tokens(host, model, text)
            if real <= 0:
                continue
            ratio = real / len(text)
            lows.append(ratio)
            print(f"    {len(text):>3} 字符 → {real:>3} token   {ratio:.3f}   {text[:38]}")
        if lows:
            print(f"    → 上沿 {max(lows):.3f}/字符（取它才不低估）")
    print()


def on_samples(host: str, model: str) -> int:
    """样本表：每条的真实值、估计值、比值，外加汇总。返回退出码。"""
    print("── 混合样本 ──")
    print(f"{'字符':>4} {'估计':>5} {'实测':>5} {'比值':>6}  文本")
    rows: list[tuple[str, int, int, float]] = []
    for text in SAMPLES:
        real = real_tokens(host, model, text)
        if real <= 0:
            continue
        estimated = estimate_tokens(text)
        rows.append((text, estimated, real, estimated / real))
        print(
            f"{len(text):>4} {estimated:>5} {real:>5} {estimated / real:>6.2f}   "
            f"{text[:30].replace(chr(10), '⏎')}"
        )

    ratios = sorted(row[3] for row in rows)
    count = len(ratios)
    under = [value for value in ratios if value < 1.0]
    print(f"\nn = {count}")
    print(f"最小 {ratios[0]:.2f}  中位 {ratios[count // 2]:.2f}  最大 {ratios[-1]:.2f}")
    print(f"低估 {len(under)}/{count}，最差 {min(under):.2f}" if under else "没有低估的样本")

    worst = min(ratio for _, _, _, ratio in rows)
    print(f"\n→ 装配层需要的最小安全余量 = {max(0.0, 1 - worst):.1%}")
    if worst < 1 - UNDER_COUNT_TOLERANCE:
        print(
            f"❌ 最差低估 {1 - worst:.1%} 超过承诺的容差 {UNDER_COUNT_TOLERANCE:.0%}"
            "——系数已经配不上这套分词器了（换模型了？）"
        )
        return 1
    print(f"✅ 在承诺的容差 {UNDER_COUNT_TOLERANCE:.0%} 之内")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=None, help="默认取本机第一个 qwen3")
    parser.add_argument(
        "--measure-base-price", action="store_true", help="只解固定底价与每条消息开销"
    )
    parser.add_argument(
        "--categories", action="store_true", help="只跑分档表（20 次请求，约为样本表的一半）"
    )
    args = parser.parse_args()

    model = pick_model(args.host, args.model)
    print(f"模型 {model}；系数版本 {ESTIMATOR_VERSION}；底价按 {BASE_PRICE} 扣\n")

    if args.measure_base_price:
        measure_base_price(args.host, model)
        return 0

    per_category(args.host, model)
    if args.categories:
        return 0
    return on_samples(args.host, model)


if __name__ == "__main__":
    raise SystemExit(main())
