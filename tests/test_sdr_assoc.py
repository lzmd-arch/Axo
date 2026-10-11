"""关联码（SDR 的第二档稀疏度）。

这个文件里最要紧的有两条：

* :func:`test_encoding_is_frozen_against_golden_digests` —— 与
  ``tests/test_sdr.py`` 里那条同源：关联码是**派生**的，但派生规则一旦漂移，
  全库的图 A 边权与 W 的模式都会换一套，而没有任何一处报错。
* :func:`test_hash_subsampling_collapses_the_effective_dimension` —— 它钉住的是
  一个**量出来的负面性质**，不是规格：``mode="hash"`` 会让所有条目的关联码挤在
  少数位上，代价记在 W 的容量上。这条断言存在的意义是别让它悄悄变成"已知问题"。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from memcore.errors import SDRCapacityError
from memcore.sdr import ASSOC_K, LEXICAL_K, encode_assoc, encode_lexical

#: 关联码的指纹：``blake2b(idx 字节 ‖ weights 字节)``。
#:
#: 与 ``test_sdr.py`` 的 ``_GOLDEN`` 同样的价值，同样的纪律：改动这些值不是修
#: 一个测试，是让既有的图 A 与 W 作废。换 ``_PERSON_ASSOC`` 前缀、改 ``ASSOC_K``、
#: 把选择规则的任一环节动一下，这里都会当场炸掉。
_GOLDEN: dict[str, str] = {
    "记忆独立于模型": "fdba939d0bd39986876c0a9a860cc59a",
    "The quick brown fox jumps over the lazy dog.": "3c8f49a00143759a75390a54f8df7b14",
    "上海": "e664d3cab67f0d57c24b01eb1c8a5be0",
    "海上": "e8116e6b30bb7ed42279d25bf15d9bf2",
    "记忆终端的存储用 SQLite 加 WAL 模式，每条写入都是一个事务。": "ec3e8dcc3b74d9ceb7b7242225b4ed1d",
}

_VOCAB = tuple(f"词{i:03d}" for i in range(400))


def _digest(text: str) -> str:
    assoc = encode_assoc(encode_lexical(text))
    return hashlib.blake2b(
        assoc.idx.tobytes() + assoc.weights.tobytes(), digest_size=16
    ).hexdigest()


def _synthetic_corpus(n: int) -> list[str]:
    """确定性的合成语料。

    **不用 ``random``**：本仓对"跨版本给出同一串随机数"这件事没有信任（见
    :mod:`memcore.sdr` 的模块文档），一条用来钉住实现性质的断言更不该建在它上面。
    步长 7 与 13 都与词表长度 400 互素，所以下标铺得开。
    """
    return ["、".join(_VOCAB[(m * 7 + j * 13) % len(_VOCAB)] for j in range(6)) for m in range(n)]


# ── 冻结 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", sorted(_GOLDEN.items()))
def test_encoding_is_frozen_against_golden_digests(text: str, expected: str) -> None:
    assert _digest(text) == expected, (
        f"{text!r} 的关联码变了。**这不是一个普通的测试失败**：图 A 的每一条边权与 "
        f"W 的每一个模式都是从关联码推出来的，改了编码意味着既有的图与基质不再与 "
        f"新写入的条目可比。"
    )


def test_encoding_is_stable_across_repeated_calls() -> None:
    lex = encode_lexical("重复编码同一段文本应当得到同一个关联码，包括权重。")
    results = [encode_assoc(lex) for _ in range(8)]
    for other in results[1:]:
        assert np.array_equal(results[0].idx, other.idx)
        assert np.array_equal(results[0].weights, other.weights)


def test_encoding_does_not_depend_on_numpy_random_state() -> None:
    """与词法编码器同样的理由：任何依赖随机状态的实现都会在这里失败。"""
    lex = encode_lexical("记忆独立于模型")
    np.random.seed(1)
    first = encode_assoc(lex)
    np.random.seed(99999)
    assert first == encode_assoc(lex)


# ── 形状与"视图"性质 ────────────────────────────────────────────────────────


def test_default_k_and_width() -> None:
    assoc = encode_assoc(encode_lexical("记忆终端的存储用 SQLite 加 WAL 模式。"))
    assert assoc.k == ASSOC_K
    assert ASSOC_K < LEXICAL_K, "关联码必须比词法码更稀疏，否则这一档没有存在的理由。"


def test_idx_is_ascending_and_is_a_subset_of_the_lexical_idx() -> None:
    lex = encode_lexical("上海自来水来自海上，路由器放在书房书架顶层。")
    assoc = encode_assoc(lex)
    assert np.all(np.diff(assoc.idx) > 0)
    assert np.isin(assoc.idx, lex.idx).all()


def test_weights_are_carried_over_bit_for_bit() -> None:
    """**关联码是词法码的一个视图，不是另一套编码。**

    这条是 ``assoc.to_dense() == lex.to_dense() 在入选位上的限制`` 的逐位版本。
    它一旦失败，说明关联码引入了词法码里没有的信息——那会让图 A 的边权不再读得
    成"词法相似度"。
    """
    lex = encode_lexical("记忆独立于模型，路由器放在书房书架顶层。")
    assoc = encode_assoc(lex)
    positions = np.searchsorted(lex.idx, assoc.idx)
    assert np.array_equal(assoc.idx, lex.idx[positions])
    assert np.array_equal(assoc.weights, lex.weights[positions])

    dense_lex = lex.to_dense()
    dense_assoc = assoc.to_dense()
    assert np.array_equal(dense_assoc[assoc.idx], dense_lex[assoc.idx])
    assert dense_assoc.sum() == pytest.approx(dense_lex[assoc.idx].sum())


def test_same_text_always_yields_the_same_assoc_code() -> None:
    """确定性选择的可观测后果：同一条文本的关联码完全相同（自反 Jaccard = 1）。

    这条**不是**废话，它与"每个条目独立随机子采样"那种做法正好相反——那种做法下
    同一条文本两次编码会得到两个不同的子集，自反 Jaccard 约 0.06。
    """
    for text in ("记忆独立于模型", "上海", "路由器放在书房书架顶层。"):
        lex = encode_lexical(text)
        assert encode_assoc(lex).jaccard(encode_assoc(lex)) == 1.0


def test_subsets_do_not_cross_text_boundaries() -> None:
    """「上海」与「海上」的关联码必须不同——词法层的区分不能被抽样抹掉。"""
    assert encode_assoc(encode_lexical("上海")) != encode_assoc(encode_lexical("海上"))


# ── 两个选择规则的实测对照 ──────────────────────────────────────────────────


def test_hash_subsampling_collapses_the_effective_dimension() -> None:
    """**``mode="hash"`` 会让关联码挤在少数位上——这是量出来的，不是猜的。**

    ``hash`` 按摘要排名取前 k，而摘要只由位下标决定，于是"摘要小"的那批下标在
    **任何**条目里都容易入选。后果是 80 条合成记忆的关联码只用到几百个位，
    而 ``weight`` 规则（取各自词频最高的位，各条目的高频词不同）铺得开得多。

    代价记在 W 的容量上：Hopfield 的 ``0.138·N`` 里那个 N 是**有效维数**，不是
    4096。这条断言就是那份代价的收据。（哪一侧更值由
    ``tools/bench_substrate.py --assoc`` 裁决，这里只保证这个差异不会被无声抹平。）
    """
    lexes = [encode_lexical(text) for text in _synthetic_corpus(80)]
    by_hash = set().union(*[set(encode_assoc(lex, mode="hash").idx.tolist()) for lex in lexes])
    by_weight = set().union(*[set(encode_assoc(lex, mode="weight").idx.tolist()) for lex in lexes])
    assert len(by_hash) < len(by_weight), (
        f"hash 的并集 {len(by_hash)} 不再小于 weight 的 {len(by_weight)}——"
        f"要么选择规则被改了，要么这个负面性质消失了。两种情况都该先看一眼再改这条断言。"
    )


def test_weight_mode_takes_the_strongest_bits() -> None:
    """``mode="weight"`` 的语义就是"取权重最高的 k 位"，可以逐位核对。"""
    lex = encode_lexical("路由器放在书房书架顶层，网络设备搁在最上面一排架子上。")
    assoc = encode_assoc(lex, k=32, mode="weight")
    strongest = set(np.sort(lex.idx[np.argsort(-lex.weights, kind="stable")[:32]]).tolist())
    assert set(assoc.idx.tolist()) == strongest


def test_hash_mode_is_independent_of_weights() -> None:
    """``hash`` 的选择只看下标——权重全改成 1 不该改变结果。

    这条把"哈希子采样"与"按权重挑"的分界钉死：若实现里不小心掺进了权重，它会在
    这里失败，而 golden digest 未必覆盖得到（权重常常几乎相同）。
    """
    lex = encode_lexical("路由器放在书房书架顶层，网络设备搁在最上面一排架子上。")
    flat = type(lex)(idx=lex.idx.copy(), weights=np.ones_like(lex.weights), width=lex.width)
    assert np.array_equal(encode_assoc(lex).idx, encode_assoc(flat).idx)


# ── 边界 ────────────────────────────────────────────────────────────────────


def test_k_larger_than_lexical_k_is_allowed_and_yields_everything() -> None:
    """k 大于实际可用的位数时**只取实际有的**，不用零权重的位凑数。

    这与 :func:`~memcore.sdr.encode_lexical` 触顶时的处置一致：零权重的活跃位在
    Jaccard 里算"活跃"却对相似度毫无贡献。宁可 k 小于标称值。
    """
    lex = encode_lexical("猫")
    assert encode_assoc(lex, k=LEXICAL_K).k == lex.k


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_raises(k: int) -> None:
    with pytest.raises(SDRCapacityError):
        encode_assoc(encode_lexical("记忆"), k=k)


def test_k_above_width_raises() -> None:
    with pytest.raises(SDRCapacityError):
        encode_assoc(encode_lexical("记忆"), k=99999)


def test_unknown_mode_raises() -> None:
    with pytest.raises(SDRCapacityError, match="选择规则"):
        encode_assoc(encode_lexical("记忆"), mode="random")


def test_narrow_lexical_code_narrows_the_assoc_k() -> None:
    lex = encode_lexical("记忆终端的存储用 SQLite。", k=20)
    assert encode_assoc(lex).k == 20
