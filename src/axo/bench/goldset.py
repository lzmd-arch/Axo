"""金标准语料：读盘、定位、算指纹。

**语料是量具，不是文档。** 这条规矩原本写在 ``tools/bench_space.py`` 里，现在
由 :mod:`tests.test_goldset_integrity` 用 sha256 机械执行：改一个字，指纹就对
不上，测试当场红，并且失败消息直接告诉你该做什么。

所以这个模块的职责里有一条容易被忽略的：**它必须让"读同一份字节"这件事唯一**。
UTF-8 显式写死（Windows 上 :func:`open` 的默认编码是 cp936，会把中文读坏，而
坏掉的样子是"某些查询突然召不回东西"，看起来像算法问题）、``\n`` 换行、
``ensure_ascii=False``。序列化只要有一处随平台漂移，冻结哈希就失效——而失效的
指纹比没有指纹更糟，因为守卫看起来还在。

语料为什么要单独一个目录、而不是像 ``bench_space.py`` 那样写成 Python 常量：
200 + 76 条会造成一个几百行、加载即执行的 ``.py``。JSONL 一行一条，``git diff``
逐条可读，加载不执行任何代码。

**读盘本身不在这个文件里**，在 :mod:`.corpus`——因为路由那份语料
（``benchmarks/routes/``）要用同一套规则读，而两份各写一遍的话，某天有人在一边
加了"坏行报行号"，另一边没加，这件事不会有任何症状。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .corpus import file_sha256, read_jsonl

__all__ = [
    "CREATED_AT",
    "GOLD_WEIGHT",
    "GoldMemory",
    "GoldQuery",
    "GoldSet",
    "corpus_digests",
    "default_gold_dir",
    "load_goldset",
    "load_memories",
    "load_queries",
]

#: 语料里每一条记忆的创建时间，统一是它（2023-11-14T22:13:20Z）。
#:
#: 不固定的后果是**惰性衰减不可复现**：:meth:`memcore.types.MemoryItem.decayed_weight`
#: 以 ``created_at`` 为起点算 ``exp(-Δt/τ)``，而 ``Δt`` 依赖"现在几点"。基准里
#: 的"现在"由 :data:`CREATED_AT` 给出（见 :func:`~axo.bench.harness.run`），
#: 于是衰减因子恒为 1，权重就是语料里写的那个数。
#:
#: 写在代码里而不是每行重复 200 遍：它是一个**全局约定**，不是逐条数据。
CREATED_AT = 1700000000.0

#: 语料里每一条记忆的权重，统一是它。
#:
#: **免混淆变量。** 真实库里权重各不相同，但一组权重参差的语料会让"词法通道排得
#: 准不准"与"这条被访问得多不多"混在一起——而后者是另一个待标定的参数
#: （``_WEIGHT_FLOOR``）。统一成中间值，两个问题就分得开。
GOLD_WEIGHT = 0.7

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GoldMemory:
    """语料里的一条记忆。字段与 :class:`memcore.types.MemoryItem` 一一对应。

    这里刻意**不**复用 ``MemoryItem``：``MemoryItem`` 是可变的（权重更新是高频
    路径），而语料必须是冻结的。借同一张脸会让"改一条语料"看起来像"改一条记忆"。
    """

    memory_id: str
    content: str
    kind: str = "fact"
    tier: str = "longterm"
    taint: str = "user"
    weight: float = GOLD_WEIGHT
    pinned: bool = False
    session_id: str = ""
    created_at: float = CREATED_AT


@dataclass(frozen=True)
class GoldQuery:
    """一条查询与它的 ground truth。

    Attributes:
        expect: **必须**出现的 id。空元组表示"库里确实没有答案"——这是一种合法
            的查询（``no_answer``），而且它不进 hit/MRR 的分母：分母是"有答案的
            查询"，把一条没有答案的查询算作"没命中"是在给自己制造假失败。
        optional: 出现与否都不罚。用于冗余对里"两条都对、但只指定了一条为
            期望"的那种情形——另一条被注入**不是错**，判它错会把"去冗做得好"
            变成一个负分项。
        never: 同主题的干扰项。它进误注入率的分子，是最尖锐的那一类错误：
            看起来相关、其实是另一件事。
        exclude: 传给 ``recall`` 的 ``exclude_ids``。
        budget: 这一轮的 token 预算。默认 400 与 ``session/budget.py`` 的常用档
            接近，``budget`` 类查询会显式调小。
    """

    query_id: str
    query: str
    category: str
    expect: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    budget: int = 400
    note: str = ""


@dataclass(frozen=True)
class GoldSet:
    """一整份金标准集。"""

    memories: tuple[GoldMemory, ...]
    queries: tuple[GoldQuery, ...]
    gold_dir: Path

    @property
    def by_id(self) -> dict[str, GoldMemory]:
        return {memory.memory_id: memory for memory in self.memories}


# ── 读盘 ────────────────────────────────────────────────────────────────────


#: "找不到文件"那句话。**逐份语料各写各的**——把这里的路径写进共用的读盘函数，
#: 另一份语料的错误提示就会指向错误的目录。
_GOLD_HINT = (
    "金标准集在仓库根的 benchmarks/gold/ 下；从源码检出跑的话它就在。"
    "找不到通常是工作目录不对，用 --corpus-dir 指定。"
)


def _read_jsonl(path: Path) -> list[dict]:
    """读这份语料的 JSONL。读盘规则（UTF-8、坏行报行号）在 :mod:`.corpus` 里。"""
    return read_jsonl(path, hint=_GOLD_HINT, what="语料")


def _as_tuple(row: dict, field: str, path: Path) -> tuple[str, ...]:
    value = row.get(field, [])
    if isinstance(value, str):
        raise ValueError(f"{path} 里的 {field} 必须是数组，收到字符串 {value!r}。")
    return tuple(value)


def load_memories(path: Path) -> list[GoldMemory]:
    """读 ``memories.jsonl``。"""
    memories: list[GoldMemory] = []
    for row in _read_jsonl(path):
        memories.append(
            GoldMemory(
                memory_id=row["id"],
                content=row["content"],
                kind=row.get("kind", "fact"),
                tier=row.get("tier", "longterm"),
                taint=row.get("taint", "user"),
                weight=float(row.get("weight", GOLD_WEIGHT)),
                pinned=bool(row.get("pinned", False)),
                session_id=row.get("session_id", ""),
                created_at=float(row.get("created_at", CREATED_AT)),
            )
        )
    return memories


def load_queries(path: Path) -> list[GoldQuery]:
    """读 ``queries.jsonl``。"""
    queries: list[GoldQuery] = []
    for row in _read_jsonl(path):
        queries.append(
            GoldQuery(
                query_id=row["id"],
                query=row["query"],
                category=row["category"],
                expect=_as_tuple(row, "expect", path),
                optional=_as_tuple(row, "optional", path),
                never=_as_tuple(row, "never", path),
                exclude=_as_tuple(row, "exclude", path),
                budget=int(row.get("budget", 400)),
                note=row.get("note", ""),
            )
        )
    return queries


def default_gold_dir() -> Path:
    """定位仓库根的 ``benchmarks/gold/``。

    从本文件往上找，**按"那里有没有 memories.jsonl"判断**，而不是按固定的层数
    （``parents[3]``）。层数会随包结构调整，而"语料在那儿"不会——错误的方式是
    某次重构之后悄悄指向了另一个目录，然后基准拿一份空语料跑出一堆 0。

    装在 wheel 里时找不到（语料只在 sdist 里，见 pyproject 的 sdist include），
    此时报清楚的错，让调用方用 ``--corpus-dir`` 指路。
    """
    for candidate in Path(__file__).resolve().parents:
        gold = candidate / "benchmarks" / "gold"
        if (gold / "memories.jsonl").is_file():
            return gold
    raise FileNotFoundError(
        "在源码树的任何一层都没找到 benchmarks/gold/memories.jsonl。"
        "金标准语料只在源码检出里（sdist 会带上，wheel 不带）——"
        "如果你是从安装好的包跑，请用 --corpus-dir 指向语料目录。"
    )


def load_goldset(gold_dir: Path | None = None) -> GoldSet:
    """把一份金标准集整个读进来。"""
    directory = Path(gold_dir) if gold_dir is not None else default_gold_dir()
    memories = load_memories(directory / "memories.jsonl")
    queries = load_queries(directory / "queries.jsonl")
    return GoldSet(memories=tuple(memories), queries=tuple(queries), gold_dir=directory)


def corpus_digests(gold_dir: Path) -> dict[str, object]:
    """两份 JSONL 的条数与 sha256。**冻结判据就是它。**"""
    directory = Path(gold_dir)
    memories = directory / "memories.jsonl"
    queries = directory / "queries.jsonl"
    return {
        "schema_version": _SCHEMA_VERSION,
        "memories": {
            "count": len(load_memories(memories)),
            "sha256": file_sha256(memories),
        },
        "queries": {
            "count": len(load_queries(queries)),
            "sha256": file_sha256(queries),
        },
    }
