"""token 估计器。

这个文件里几乎每个数字都是**在本机 Ollama 0.40.2 + Qwen3-4B 上量出来的**，量法见
``tools/calibrate_tokens.py``。把真值内联而不是每次重测，是因为这一条必须能在没有
Ollama 的机器上跑——而它要保护的性质（"估计器不会低估到让预算算大"）恰恰是那种
**只有在 CI 里才拦得住**的退化：改一个系数，本地跑一次对话看不出任何异常，
直到某天模型忽然忘了自己是谁。

为什么要这么在意低估：低估让预算算大，而算大的后果是 Ollama 在服务端静默截断，
截掉的是**最早的消息**——正是人格锚所在的位置。P0 已经因为这个坑返工过一次
（窗口按 32768 猜，实际 ``num_ctx`` 是 16384）。高估的代价只是少注入一条记忆。

两个方向都要钉住，缺一个就不成立：只有"不低估"那一半的话，把所有系数调到 10
就能通过；只有"不高估"那一半的话，把系数归零就能通过。**合起来才把估计器夹在一个
带子里**，而那正是文档承诺给装配层的东西。
"""

from __future__ import annotations

import pytest

from memcore.tokens import UNDER_COUNT_TOLERANCE, estimate_tokens

#: 低估容差。**从被测量的模块里导入，不在这里另写一份**：两处各写一份的话，
#: 重新标定之后就会出现"测试说合格、预算却算大了"的分歧，而预算算大的后果是
#: 服务端静默截断人格锚。装配层用的是同一个常数（见 session/budget.py）。
MAX_UNDER_COUNT = UNDER_COUNT_TOLERANCE

#: 高估上限。实测最差的自然文本是 1.92（一行 Python 签名，标点密集），2.0 是它
#: 之后的第一档整数。
MAX_OVER_COUNT = 2.0

#: 实测 ``(文本, 真实 token)``。真实值的算法是把文本作为单独一条 user 消息发给
#: ``/api/chat``，读 ``prompt_eval_count`` 再扣掉整轮对话的固定底价 12。
#: 改任何一个系数之后，用 ``python tools/calibrate_tokens.py`` 重跑一遍这张表。
MEASURED: list[tuple[str, int]] = [
    ("记忆终端的装配预算需要一个可靠的每条消息开销常数。", 16),
    (
        "如果这个常数取得太大，就会白白扔掉几条本来放得下的记忆；"
        "取得太小则会在服务端被静默截断，而截掉的是最早的人格锚。",
        41,
    ),
    ("症状是模型忽然忘了自己是谁，排查起来极难定位到预算这一步。", 18),
    ("我家的路由器放在书房书架顶层。", 9),
    ("今天午饭吃的是番茄炒蛋和米饭。", 10),
    ("每周三晚上八点要开组会，会议纪要在共享盘上。", 17),
    ("记忆终端用 SQLite 存记忆库，WAL 模式，每条写入都是一个事务。", 23),
    ("Axo 是一个把长期记忆放在终端侧的桌面助手，独立于所调用的大模型。", 22),
    # ↓ 这一条是旧实现最差的样本（0.45）。病根是数字被算进 0.24 的字母桶，
    #   而这条句子里有一半的 token 花在 16384 和 32768 上。
    ("配置项 num_ctx 的值是 16384，不是 32768。", 22),
    ("版本 0.40.2 的 Ollama 在 /api/chat 上返回 NDJSON 流。", 23),
    ("读取 README.md 并总结前三段。", 10),
    ("把 src/memcore/recall.py 里 _fuse 函数的注释改得更清楚一些。", 23),
    ("为什么用 blake2b 而不是 numpy 的随机数生成器？", 17),
    ("上海自来水来自海上，山西运煤车煤运西山。", 14),
    ("这段文本里中英文混排 mixed with English words and 1234 digits.", 19),
    ("def recall(store, *, query_text, budget_tokens): return None", 13),
    ("短句。", 3),
    (
        "这是一段刻意写得很长的文本，用来测试估计器在长文本上会不会因为重复的二元组而失准，"
        "因为词法 SDR 的去冗与预算都建立在同一套分词之上，如果估计器在长文本上系统性偏低，"
        "那么一条很长的记忆就会以『看起来装得下』的样子挤掉后面好几条短的。",
        80,
    ),
    ("## 结论\n\n---\n\n| 通道 | 权重 |\n|---|---|\n| 词法 | 0.90 |\n", 29),
    # emoji 是"其他"那一档最可疑的部分：它们是星平面字符，BPE 常常一个字符就吃掉
    # 两三个 token，组合序列（ZWJ、变体选择符、区域指示符）更多。实测这一档够用。
    ("😀😀😀😀😀😀😀😀", 8),
    ("今天心情不错 👍，明天继续 🚀。", 12),
]

#: 一长串**重复**标点会被 BPE 合成极少的 token，而估计器按字符数算——这是模块
#: docstring 里写明的已知高估（最多 5.3 倍）。单列出来是为了让它们只受"不低估"
#: 那一条约束：把它们混进 :data:`MEASURED` 会逼着上界放宽到没用。
PUNCTUATION_RUNS: list[tuple[str, int]] = [
    ("-" * 32, 1),
    ("```python\n# " + "-" * 63 + "\ndef f():\n    return 1\n```", 15),
]


def test_empty_is_zero() -> None:
    """空串返回 0 而不是 1：空内容在逐字稿里是合法状态（被中断的空回复），
    而给它记 1 token 会让「哪些消息不占预算」这类判断失去意义。"""
    assert estimate_tokens("") == 0


def test_non_empty_text_never_estimates_zero() -> None:
    """非空文本有个 1 token 的地板。全是空白也算——那确实会占掉包装里的一格。"""
    assert estimate_tokens(" ") == 1
    assert estimate_tokens("\n\t ") == 1


def test_whitespace_is_free() -> None:
    """BPE 把词前的空格并进那个词自己的 token，单独再数一遍就是算两次。"""
    assert estimate_tokens("  a b c  ") == estimate_tokens("abc")


def test_digit_is_its_own_class() -> None:
    """**钉的是分档结构，不是系数取值。** 系数可以重标，分档不能并。

    实测一个数字就是 1.000 个 token，而一个拉丁字母是 0.156~0.329。旧实现把两者
    并进同一个 0.24 的桶，于是 ``配置项 num_ctx 的值是 16384`` 只估出实际的 45%。
    """
    digits = estimate_tokens("0123456789")
    letters = estimate_tokens("abcdefghij")
    assert digits == 10, "一个数字一个 token——这是量出来的，不是取的保守值"
    assert digits > letters


def test_never_under_counts_by_more_than_the_documented_tolerance() -> None:
    """**这个文件的核心断言。** 跨全部实测样本，低估不超过 :data:`MAX_UNDER_COUNT`。"""
    offenders = [
        (text, real, estimate_tokens(text) / real)
        for text, real in MEASURED + PUNCTUATION_RUNS
        if estimate_tokens(text) < real * (1 - MAX_UNDER_COUNT)
    ]
    assert not offenders, (
        "估计器低估超限。低估让预算算大，而 Ollama 会静默截断最早的消息"
        f"——那是人格锚的位置。超限的样本：{offenders}"
    )


def test_the_over_count_stays_within_a_factor_of_two() -> None:
    """高估有上界。**没有这一条，把四个系数全调到 10 也能让上面那条通过。**"""
    offenders = [
        (text, real, estimate_tokens(text) / real)
        for text, real in MEASURED
        if estimate_tokens(text) > real * MAX_OVER_COUNT
    ]
    assert not offenders, f"高估超过 {MAX_OVER_COUNT} 倍，预算会被白白吃掉：{offenders}"


def test_the_estimate_never_decreases_when_text_is_appended() -> None:
    """单调性。估计器是线性和取整，这本该显然——但"显然"正是它值得被钉住的原因：
    一旦有人为了修某个长文本的偏差引入非线性（比如按重复二元组打折），装配层的
    贪心遍历就会失去"越靠后的候选越不容易入选"这个隐含假设。"""
    for text, _ in MEASURED:
        for cut in range(1, len(text)):
            assert estimate_tokens(text[:cut]) <= estimate_tokens(text)


@pytest.mark.parametrize(("text", "real"), MEASURED + PUNCTUATION_RUNS)
def test_the_measured_table_is_still_reproducible(text: str, real: int) -> None:
    """每条样本单独报一次，失败时能直接看到是哪一条漂了。

    上面那两条是汇总断言（一个列表推导报全部），这一条是它的人读版本：pytest 会把
    样本的文本前缀印在用例 id 里。
    """
    assert real > 0, "真值表里不该有非正数"
    assert estimate_tokens(text) >= real * (1 - MAX_UNDER_COUNT)
