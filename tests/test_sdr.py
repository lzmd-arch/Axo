"""词法 SDR 编码器。

这个文件里最要紧的一条是 :func:`test_encoding_is_frozen_against_golden_digests`。
其余的断言都在验证"实现符合规格"，而它验证的是**规格本身没有漂移**——这是
"记忆独立于模型"这条承诺在代码里唯一可以自动化的证据。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from memcore.errors import SDRCapacityError
from memcore.sdr import LEXICAL_K, drop_bits, encode_assoc, encode_lexical
from memcore.text import is_cjk, normalize, tokenize
from memcore.types import DEFAULT_SDR_WIDTH, SDR

#: 编码结果的指纹：``blake2b(idx 字节 ‖ weights 字节)``。
#:
#: **这个常量是这份测试的全部价值所在。** 它把"编码只由文本字节决定"从一句
#: 设计意图变成一条会失败的断言：换 numpy 版本、改 _BITS_PER_TOKEN、动 person
#: 前缀、甚至只是把 lexsort 的次键顺序反一下，这里都会当场炸掉。没有它，全库
#: 的 lex_sdr 可以在一次 ``pip install -U numpy`` 之后静默漂移——召回质量下降，
#: 而没有任何一处报错。
#:
#: 改动这些值**不是**修一个测试，是让全库既有的词法 SDR 作废。真要做的话，
#: 那是一次带 ADR 的迁移（旧编码要么重算，要么作为历史字段留着）。
_GOLDEN: dict[str, str] = {
    "记忆独立于模型": "5c9de9a123e638c5a99df7459d6ddf5e",
    "The quick brown fox jumps over the lazy dog.": "34d519ed19dca77631ecd7ef8a8dbf0e",
    "上海": "7a81dc6116cb97551ecc86e8e2f9ab42",
    "海上": "344cd2367611295ff443e6ce8b6c2a61",
}


def _digest(text: str) -> str:
    sdr = encode_lexical(text)
    return hashlib.blake2b(sdr.idx.tobytes() + sdr.weights.tobytes(), digest_size=16).hexdigest()


# ── 冻结 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", sorted(_GOLDEN.items()))
def test_encoding_is_frozen_against_golden_digests(text: str, expected: str) -> None:
    assert _digest(text) == expected, (
        f"{text!r} 的词法 SDR 变了。**这不是一个普通的测试失败**：全库既有的 "
        f"lex_sdr 都是按旧编码写进去的，改了编码意味着它们与新写入的条目不再可比。"
        f"若确实要改，先想清楚旧数据怎么办。"
    )


def test_encoding_does_not_depend_on_numpy_random_state() -> None:
    """numpy 的全局随机状态与编码无关。

    这条测的是"为什么用 blake2b 而不是 np.random.default_rng"那个决定的**可观测
    后果**：任何一个依赖随机状态的实现都会在这里失败。
    """
    np.random.seed(1)
    first = encode_lexical("记忆独立于模型")
    np.random.seed(99999)
    second = encode_lexical("记忆独立于模型")
    assert first == second


def test_encoding_is_stable_across_process_runs() -> None:
    """同一进程内反复编码也必须逐位相同（哈希种子、字典序都不该影响它）。"""
    text = "重复编码同一段文本应当得到同一个 SDR，包括权重。"
    results = [encode_lexical(text) for _ in range(8)]
    for other in results[1:]:
        assert np.array_equal(results[0].idx, other.idx)
        assert np.array_equal(results[0].weights, other.weights)


# ── 形状 ────────────────────────────────────────────────────────────────────


def test_long_text_fills_exactly_k_bits() -> None:
    sdr = encode_lexical("记忆终端的存储用 SQLite 加 WAL 模式，每条写入都是一个事务。")
    assert sdr.k == LEXICAL_K
    assert sdr.width == DEFAULT_SDR_WIDTH


def test_short_text_still_fills_k_bits() -> None:
    """单字也要填满 k 位。

    做法是按轮次反复哈希补位（``_MAX_ROUNDS``）。若这条失败，短文本的 SDR 会
    与长文本的 SDR 在势上不可比——Jaccard 的分母因此系统性偏小。
    """
    assert encode_lexical("猫").k == LEXICAL_K


def test_idx_is_strictly_ascending_and_weights_in_unit_interval() -> None:
    sdr = encode_lexical("上海自来水来自海上")
    assert np.all(np.diff(sdr.idx) > 0)
    assert float(sdr.weights.min()) > 0.0
    assert float(sdr.weights.max()) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", "   ", "！？。，、；：", "!!!???..."])
def test_text_without_encodable_characters_raises(text: str) -> None:
    """全标点或全空白的输入没有 SDR。

    抛异常而不是返回一个空 SDR：空 SDR 会以"编码好了"的样子进入相似度计算，
    然后与所有东西的 Jaccard 都是 0——那是一次看起来完全正常的召回失败。
    """
    with pytest.raises(SDRCapacityError):
        encode_lexical(text)


@pytest.mark.parametrize("width,k", [(0, 1), (10, 0), (10, 11), (10, -1)])
def test_invalid_shape_parameters_raise(width: int, k: int) -> None:
    with pytest.raises(SDRCapacityError):
        encode_lexical("记忆", width=width, k=k)


# ── 区分度 ──────────────────────────────────────────────────────────────────


def test_word_order_changes_the_sdr() -> None:
    """「上海」与「海上」必须是两个不同的 SDR。

    只编码单字的话它们会完全相同——这是中文这类无词边界语言里最容易踩的坑，
    二元组是唯一能表达顺序的手段。
    """
    assert encode_lexical("上海") != encode_lexical("海上")
    assert encode_lexical("上海").jaccard(encode_lexical("海上")) < 0.9


def test_cjk_and_latin_do_not_collide() -> None:
    """字符类前缀保证 ``"c" + "at"`` 不会与 ``"w" + "at"`` 撞成同一个 token。"""
    assert tokenize("cat")[0] == "w:cat"
    assert tokenize("猫")[0] == "c:猫"


def test_related_texts_score_above_unrelated() -> None:
    """同一件事的两种说法，Jaccard 应当高于两件无关的事。

    这条是经验性的，样本只有两对——它固化的是"编码器至少有区分度"这个下界，
    不是召回质量的断言（那由 ``tools/bench_space.py`` 的 40 对标注样本负责）。
    """
    related = encode_lexical("我家的路由器放在书房书架顶层。").jaccard(
        encode_lexical("网络设备搁在书房最上面一排架子上。")
    )
    unrelated = encode_lexical("我家的路由器放在书房书架顶层。").jaccard(
        encode_lexical("今天午饭吃的是番茄炒蛋和米饭。")
    )
    assert related > unrelated


# ── text.py ─────────────────────────────────────────────────────────────────


def test_normalize_folds_fullwidth_and_case() -> None:
    """全角数字与半角数字必须落在同一个 token 上，大小写同理。"""
    assert normalize("ＳＱＬｉｔｅ１２３") == normalize("sqlite123")
    assert tokenize("ＳＱＬｉｔｅ") == tokenize("sqlite")


@pytest.mark.parametrize(
    "char,expected",
    [("汉", True), ("あ", True), ("한", True), ("a", False), ("1", False), ("，", False)],
)
def test_is_cjk_classification(char: str, expected: bool) -> None:
    assert is_cjk(char) is expected


def test_tokenize_includes_bigrams_between_base_tokens() -> None:
    tokens = tokenize("abc汉字")
    assert tokens == ["w:abc", "c:汉", "c:字", "b:w:abc:c:汉", "b:c:汉:c:字"]


def test_tokenize_drops_punctuation() -> None:
    """标点不进 token——它们不承载语义，只会给所有文本加一份共同噪声。"""
    assert tokenize("记忆，终端。") == tokenize("记忆终端")


# ── drop_bits：残缺探针（P4c 从 bench 搬进来的生产算子）─────────────────────


#: 丢位结果的指纹，与上面那份同一个形状。
#:
#: **这份常量的意义与 `_GOLDEN` 不同，别读混。** `_GOLDEN` 冻的是"词法编码不许
#: 漂移"（漂了全库的 `lex_sdr` 就作废）；这一份冻的是**量具与生产共用的那一次
#: 抽样**：巩固的判据叫「回放辨认率」，基线里叫 `completion_top1`，两者可比的前
#: 提是探针的造法只有一份。改这里的值 = 改探针 = 让两边的数字不再可比，
#: 且**没有任何一处会报错**（两个数字都仍然像个合理的百分比）。
_DROP_GOLDEN: dict[tuple[str, float], str] = {
    ("记忆独立于模型", 0.25): "44faddea1b240581867125bf946ad41a",
    ("记忆独立于模型", 0.5): "048d1330ee06eb78840747bc5339faeb",
    ("记忆独立于模型", 0.75): "a98a2dc9101b26c31099d8255343d3c0",
    ("路由器放在书房书架顶层", 0.25): "02b54b1f199eae707495ae93876ccf07",
    ("路由器放在书房书架顶层", 0.5): "97970961692114277171e63e74b2133d",
    ("路由器放在书房书架顶层", 0.75): "1068785fa981059174d60de12f2ffbc2",
    ("上海", 0.25): "b94991921525bd4f865074fb6b249be4",
    ("上海", 0.5): "e130402a40950d618f3ec7377d969008",
    ("上海", 0.75): "a94a74c1368d9139f5594b47383c7968",
}


def _drop_digest(text: str, ratio: float) -> str:
    probe = drop_bits(encode_assoc(encode_lexical(text)), ratio)
    return hashlib.blake2b(
        probe.idx.tobytes() + probe.weights.tobytes(), digest_size=16
    ).hexdigest()


@pytest.mark.parametrize(
    ("text", "ratio", "expected"), sorted((t, r, d) for (t, r), d in _DROP_GOLDEN.items())
)
def test_drop_bits_is_frozen_against_golden_digests(text: str, ratio: float, expected: str) -> None:
    assert _drop_digest(text, ratio) == expected, (
        f"{text!r} 在丢位比例 {ratio} 下的探针变了。探针是量具与生产**共用的**那个"
        f"算子：它一动，巩固的「回放辨认率」与基线的 completion_top1 就不再同口径，"
        f"而两个数字看起来都仍然正常。"
    )


def test_drop_bits_keeps_exactly_the_stride_rule() -> None:
    """保留位就是 ``(序号 × 7919) % 1000 >= 500`` —— 手算得出来，所以写死。

    这条与上面那份摘要重复覆盖了同一件事，但**读法不同**：摘要只说"变了"，
    这条说"变成什么了"。规则是唯一会被误改的地方（换成位索引、换成随机数、
    步长抄错一位），所以它值得一条能一眼看出对错的断言。
    """
    width = 64
    sdr = SDR(
        idx=np.arange(width, dtype=np.int32),
        weights=(np.arange(width, dtype=np.float32) + 1) / width,
        width=4096,
    )
    kept = drop_bits(sdr, 0.5)

    assert kept.idx.tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        13,
        14,
        15,
        16,
        17,
        18,
        25,
        26,
        27,
        28,
        29,
        30,
        38,
        39,
        40,
        41,
        42,
        43,
        50,
        51,
        52,
        53,
        54,
        55,
        62,
        63,
    ]
    # 权重逐位照搬，不做任何重新归一化——探针是原码的一次**取子集**，不是另一份编码。
    assert kept.weights.tolist() == [sdr.weights[i] for i in kept.idx]


def test_drop_bits_preserves_the_source_ordering() -> None:
    """留下的位必须仍是**原码的子序列**：``SDR`` 要求 ``idx`` 严格升序。

    用序号（而不是位索引）做乘法的这个实现天然满足它，但这条不是多余的——
    换成 `np.flatnonzero` 或先排序再切片的写法都可能悄悄破坏子集关系，
    而那会同时改变 Jaccard 的口径。
    """
    lex = encode_lexical("路由器放在书房书架顶层，网络设备搁在最上面一排架子上。")
    assoc = encode_assoc(lex)
    kept = drop_bits(assoc, 0.5)

    assert set(kept.idx.tolist()) <= set(assoc.idx.tolist())
    assert np.all(np.diff(kept.idx) > 0)


def test_drop_bits_at_zero_is_the_identity() -> None:
    """``ratio = 0`` 原样返回（**同一个对象**）——空转的测量不该付出一次拷贝。"""
    assoc = encode_assoc(encode_lexical("记忆独立于模型"))
    assert drop_bits(assoc, 0.0) is assoc
    assert drop_bits(assoc, -1.0) is assoc


def test_drop_bits_never_returns_an_empty_sdr() -> None:
    """丢光就不是"残缺的探针"，是一个空探针——而 ``SDR`` 不允许全零。

    **触发它需要 ``ratio`` 严格大于 1 或码极短**，而不是"比例高"就够了：
    规则是 ``(序号 × 7919) % 1000 >= ratio × 1000``，序号 0 的余数恒为 0，
    所以 k >= 2 的码在 ``ratio <= 1`` 时总有一位能活下来。真正会一位不剩的是
    **k=1**（唯一的余数是 0）——那正是 :func:`encode_assoc` 在极短文本上会产出的
    形状。这时留一位而不是抛异常：调用方要的是"一个更弱的探针"，不是一次崩溃。
    """
    single = SDR(
        idx=np.array([7], dtype=np.int32),
        weights=np.array([1.0], dtype=np.float32),
        width=64,
    )
    assert drop_bits(single, 0.5).idx.tolist() == [7]

    # k=3 且 ratio=1.0：三位全在外（余数 0、919、838 都 < 1000）→ 兜底留第 0 位。
    tiny = SDR(
        idx=np.array([5, 9, 13], dtype=np.int32),
        weights=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        width=64,
    )
    assert drop_bits(tiny, 1.0).idx.tolist() == [5]

    # 而"比例高"本身不触发兜底：k=3、ratio=0.9 时第 1 位（余数 919）本来就在。
    assert drop_bits(tiny, 0.9).idx.tolist() == [9]


def test_drop_bits_does_not_depend_on_numpy_random_state() -> None:
    """与编码同一条理由：它必须只由下标决定，不受全局随机状态影响。

    这条测的是"为什么用固定步长而不是 ``np.random``"那个决定的可观测后果——
    任何一个真的抽签的实现都会在这里失败。
    """
    assoc = encode_assoc(encode_lexical("记忆独立于模型"))
    np.random.seed(1)
    first = drop_bits(assoc, 0.5)
    np.random.seed(99999)
    second = drop_bits(assoc, 0.5)
    assert first == second
