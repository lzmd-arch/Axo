"""神经元级基质 W。

这个文件里最要紧的两条：

* :func:`test_respond_matches_a_materialised_w` —— 隐含表示 ``W p = Aᵀ(A p)``
  与真的把 ``AᵀA`` 算出来的那个矩阵**逐位相同**。没有这条，``respond`` 里少乘
  一个权重或者把索引写反了，看起来也只是"补全效果差一点"。
* :func:`test_to_spike_actually_calls_the_bus_bridge` —— 钉住"接线在"。那条
  ``biosnn_bus`` 接口在本仓零调用者已经很久了，而这个仓的教训是**纯层全绿 ≠
  接线在**：删掉 :meth:`Substrate.to_spike` 里那一句转发，别的测试一个都不会红。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from memcore.errors import SDRCapacityError
from memcore.sdr import encode_assoc, encode_lexical
from memcore.substrate import Substrate, SubstrateConfig
from memcore.types import DEFAULT_SDR_WIDTH, SDR


def _item(memory_id: str, lex_sdr: SDR) -> SimpleNamespace:
    """一条只有词法码的条目——关联码由 :meth:`Substrate.build` 现算。"""
    return SimpleNamespace(memory_id=memory_id, lex_sdr=lex_sdr, assoc_sdr=None)


def _assoc_item(memory_id: str, assoc_sdr: SDR) -> SimpleNamespace:
    """一条关联码已经给好的条目。手工核对算术时用它，免得现算的 k 撞上窄宽度。"""
    return SimpleNamespace(memory_id=memory_id, lex_sdr=None, assoc_sdr=assoc_sdr)


def _drop_bits(sdr: SDR, ratio: float) -> SDR:
    """确定性地丢掉 ``ratio`` 比例的活跃位。

    用 ``(i * 7919) % 1000`` 而不是 ``random``：7919 与 1000 互素，所以余数在
    0..999 上铺得开，而结果**只由下标决定**。量具里丢位的规则也该是这个形状。
    """
    keep = ((np.arange(sdr.k) * 7919) % 1000) >= int(ratio * 1000)
    idx = sdr.idx[keep]
    return SDR(idx=idx, weights=sdr.weights[keep], width=sdr.width)


# ── 算术正确性：与物化的 W 对拍 ──────────────────────────────────────────────


def test_respond_matches_a_materialised_w() -> None:
    """``respond`` 必须等于真的把 ``W = AᵀA`` 算出来再做 matvec。

    这条是小尺寸上的**等价性证据**。隐含表示的全部理由就是"不付那 64 MiB"，
    一旦两者不等价，省下来的内存就是拿正确性换的。
    """
    width = 32
    rng_rows = [
        SDR(idx=np.array([0, 5, 9]), weights=np.array([1.0, 0.5, 0.25]), width=width),
        SDR(idx=np.array([5, 7]), weights=np.array([1.0, 0.8]), width=width),
        SDR(idx=np.array([1, 9, 20, 31]), weights=np.array([0.9, 1.0, 0.3, 0.6]), width=width),
    ]
    sub = Substrate.build([_assoc_item(f"m{i}", sdr) for i, sdr in enumerate(rng_rows)])

    # 物化：A 是 (n, width) 的稠密关联码矩阵。
    dense = np.zeros((len(rng_rows), width), dtype=np.float64)
    for row_index, sdr in enumerate(rng_rows):
        dense[row_index, sdr.idx] = sdr.weights
    materialised = dense.T @ dense  # (width, width)

    for probe_idx in ([0], [5], [0, 1, 2, 3], list(range(width))):
        probe = SDR(
            idx=np.array(probe_idx),
            weights=np.full(len(probe_idx), 0.7),
            width=width,
        )
        expected = materialised @ probe.to_dense().astype(np.float64)
        assert np.allclose(sub.respond(probe), expected, atol=1e-5), (
            f"探针 {probe_idx} 上隐含表示与物化 W 不一致——W p = Aᵀ(A p) 这个恒等式被写坏了。"
        )


def test_complete_ranks_the_pattern_that_generated_the_probe_first() -> None:
    """自补全：拿一条记忆的关联码当探针，它自己必须排第一。

    这是最弱的一条基质断言——只要求"还认得自己"，不要求"能从残缺里恢复"。
    后者是经验性的，由 ``tools/bench_substrate.py --drop`` 量。
    """
    texts = [
        "路由器放在书房书架顶层。",
        "今天午饭吃的是番茄炒蛋和米饭。",
        "记忆终端的存储用 SQLite 加 WAL 模式。",
        "下周三下午三点和小李在图书馆碰面。",
    ]
    sub = Substrate.build([_item(f"m{i}", encode_lexical(t)) for i, t in enumerate(texts)])
    for index, text in enumerate(texts):
        probe = encode_assoc(encode_lexical(text))
        ranked = sub.complete(probe, top=len(texts))
        assert ranked[0][0] == f"m{index}", f"{text!r} 的自补全第一名是 {ranked[0][0]}。"


def test_complete_returns_stronger_scores_for_closer_probes() -> None:
    """得分随探针与目标的接近程度单调——这是"排序可用"的最低要求。"""
    text = "路由器放在书房书架顶层，网络设备搁在最上面一排架子上。"
    sub = Substrate.build([_item("m0", encode_lexical(text))])
    full = encode_assoc(encode_lexical(text))
    assert sub.complete(full, top=1)[0][1] > sub.complete(_drop_bits(full, 0.75), top=1)[0][1]


# ── 接线：那个零调用者的总线接口 ────────────────────────────────────────────


def test_to_spike_actually_calls_the_bus_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """**``to_spike`` 必须真的转发到 ``SDR.to_spike_train``。**

    没有这条断言，把这个方法的函数体换成 ``return None`` 之后全套测试依然全绿——
    而它正是 ``biosnn_bus`` 在本仓的第一个（也是唯一一个）调用点。这就是"纯层
    全绿 ≠ 接线在"那条教训的具体形态。
    """
    seen: list[float] = []
    original = SDR.to_spike_train

    def spy(self: SDR, dt: float = 1.0) -> object:
        seen.append(dt)
        return original(self, dt)

    monkeypatch.setattr(SDR, "to_spike_train", spy)
    sub = Substrate.build([_item("m0", encode_lexical("记忆独立于模型。"))])
    pattern = encode_assoc(encode_lexical("记忆独立于模型。"))
    out = sub.to_spike(pattern)

    assert seen == [1.0], "to_spike 没有调用 SDR.to_spike_train——这条线是断的。"
    from biosnn_bus import SpikeTrain

    assert isinstance(out, SpikeTrain)
    assert out.data.shape == (1, DEFAULT_SDR_WIDTH)


# ── 形状与构造 ──────────────────────────────────────────────────────────────


def test_ragged_k_is_padded_with_zero_weight() -> None:
    """变长的关联码装进一个矩形数组，**补齐位的权重是 0**。

    这条刻意让 0 号位在长条目里是**真实活跃位**：短条目的补齐位写的就是 ``idx=0``
    （因为 ``np.zeros`` 填的是 0），如果补齐位的权重不是 0，短条目的模式会变成
    "4 个 0 号位"，得分因此翻四倍。所以下面这个精确的 2.0 同时钉住了两件事——
    补齐语义，以及"内积里补齐位不贡献"。
    """
    short = SDR(idx=np.array([0]), weights=np.array([1.0]), width=64)
    long = SDR(idx=np.array([0, 1, 2, 3]), weights=np.full(4, 1.0), width=64)
    sub = Substrate.build([_assoc_item("short", short), _assoc_item("long", long)])
    assert sub.idx.shape == (2, 4)
    assert sub.weights.shape == (2, 4)
    assert list(sub.idx[0]) == [0, 0, 0, 0]
    assert list(sub.weights[0]) == [1.0, 0.0, 0.0, 0.0]

    probe = SDR(idx=np.array([0]), weights=np.array([1.0]), width=64)
    # W p 在位 0 上是 1·1 + 1·1 = 2，而在位 1..3 上各是 1。
    assert sub.complete(probe, top=2) == [("long", 5.0), ("short", 2.0)]


def test_items_without_a_lexical_sdr_are_skipped() -> None:
    """还没回填编码的条目被跳过，而不是把整个基质建不起来。"""
    items = [
        SimpleNamespace(
            memory_id="has", lex_sdr=encode_lexical("记忆独立于模型。"), assoc_sdr=None
        ),
        SimpleNamespace(memory_id="empty", lex_sdr=None, assoc_sdr=None),
    ]
    sub = Substrate.build(items)
    assert sub.ids == ("has",)
    assert len(sub) == 1


def test_precomputed_assoc_sdr_is_used_as_is() -> None:
    """条目上已经算好的 ``assoc_sdr`` 优先——调用方可能已经付过那次哈希。"""
    lex = encode_lexical("记忆独立于模型。")
    precomputed = SDR(idx=np.array([1, 2, 3]), weights=np.array([1.0, 0.5, 0.25]), width=lex.width)
    sub = Substrate.build([SimpleNamespace(memory_id="m", lex_sdr=lex, assoc_sdr=precomputed)])
    assert list(sub.idx[0]) == [1, 2, 3]


def test_mismatched_widths_raise() -> None:
    """不同宽度的关联码拼不成一个矩阵。

    "悄悄取最小的那个宽度"会让越界的位静默消失——那是一次看不见的丢数据。
    """
    items = [
        _item("a", SDR(idx=np.array([1]), weights=np.array([1.0]), width=64)),
        _item("b", SDR(idx=np.array([1]), weights=np.array([1.0]), width=128)),
    ]
    with pytest.raises(SDRCapacityError):
        Substrate.build(items)


def test_empty_substrate_is_well_defined() -> None:
    """没有条目时基质是空的，但每个方法都还答得上来——不抛、不崩。"""
    sub = Substrate.build([])
    assert len(sub) == 0
    probe = SDR(idx=np.array([1]), weights=np.array([1.0]), width=DEFAULT_SDR_WIDTH)
    assert np.array_equal(sub.respond(probe), np.zeros(DEFAULT_SDR_WIDTH, dtype=np.float32))
    assert sub.complete(probe) == []


def test_probe_width_must_match() -> None:
    sub = Substrate.build([_item("a", SDR(idx=np.array([1]), weights=np.array([1.0]), width=64))])
    with pytest.raises(SDRCapacityError, match="宽度"):
        sub.respond(SDR(idx=np.array([1]), weights=np.array([1.0]), width=128))


@pytest.mark.parametrize("top", [0, -3])
def test_invalid_top_raises(top: int) -> None:
    sub = Substrate.build([_item("a", SDR(idx=np.array([1]), weights=np.array([1.0]), width=64))])
    probe = SDR(idx=np.array([1]), weights=np.array([1.0]), width=64)
    with pytest.raises(ValueError):
        sub.complete(probe, top=top)


def test_complete_returns_at_most_n_items() -> None:
    """``top`` 大于条目数时不该崩，也不该造出不存在 id 的行。"""
    sub = Substrate.build(
        [_item("only", SDR(idx=np.array([1]), weights=np.array([1.0]), width=64))]
    )
    probe = SDR(idx=np.array([1]), weights=np.array([1.0]), width=64)
    assert len(sub.complete(probe, top=99)) == 1


def test_build_is_deterministic() -> None:
    """同样的条目、同样的参数，两次建出来的基质逐位相同。"""
    texts = ["路由器放在书房书架顶层。", "今天午饭吃的是番茄炒蛋和米饭。"]
    items = [_item(f"m{i}", encode_lexical(t)) for i, t in enumerate(texts)]
    assert Substrate.build(items) == Substrate.build(items)


def test_config_changes_the_built_patterns() -> None:
    """``SubstrateConfig.k`` 真的透传到关联码上。"""
    items = [_item("m0", encode_lexical("路由器放在书房书架顶层，网络设备搁在最上面一排架子上。"))]
    assert Substrate.build(items).idx.shape[1] == SubstrateConfig().k
    narrow = Substrate.build(items, config=SubstrateConfig(k=16))
    assert narrow.idx.shape[1] == 16


# ── 骨架库的类型边界 ──────────────────────────────────────────────────────


def _two_row_substrate() -> tuple[Substrate, SDR]:
    """两行的小基质，加一个能从它里面被认出来的探针。"""
    codes = [
        encode_assoc(encode_lexical(text), k=8, mode="weight")
        for text in ("路由器在书房。", "冰箱调到四度。")
    ]
    built = Substrate.build(
        [
            SimpleNamespace(memory_id=f"m{index}", assoc_sdr=code, lex_sdr=None)
            for index, code in enumerate(codes)
        ],
        config=SubstrateConfig(k=8, mode="weight"),
    )
    return built, codes[0]


def test_the_probe_crosses_the_spike_boundary_without_changing_a_value() -> None:
    """探针过一次 ``SpikeTrain`` 再回来，**逐位相同**。

    这是「让骨架库真的在数据路径上」必须付的证明：类型边界不该顺手把数值改了。
    实测 ``rates`` 与 ``to_dense()`` 逐位相等，只有 dtype 从 float32 变成 float64。
    """
    substrate, probe = _two_row_substrate()

    dense = substrate._check_probe(probe)

    assert np.array_equal(dense, probe.to_dense())
    assert dense.dtype == np.float32


def test_complete_really_goes_through_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """**接线守卫**：``complete`` 真的调了 ``to_spike``——不是"有个方法但没人调"。

    纯层全绿不等于接线在，而这一条正是那个形状：``to_spike`` 曾经存在、有文档、
    零调用者。
    """
    substrate, probe = _two_row_substrate()
    seen: list[object] = []
    original = Substrate.to_spike

    def spy(self: Substrate, pattern: SDR) -> object:
        seen.append(pattern)
        return original(self, pattern)

    monkeypatch.setattr(Substrate, "to_spike", spy)

    substrate.complete(probe, top=1)

    assert len(seen) == 1, f"complete 没走类型边界（to_spike 被调了 {len(seen)} 次）"
