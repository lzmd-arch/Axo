"""路由语料：读盘、定位、算指纹。

**这是路由的量具，不是数据集。** 它回答的问题只有一个："改了规则层的一个词之后，
路由是变准了还是变不准了"。与 :mod:`.goldset` 同一条纪律——改一个字，``freeze.json``
里的 sha256 就对不上，:mod:`tests.test_router_corpus_integrity` 当场红，并且失败
消息直接告诉你该动哪几个文件。

**它量的是"一句话该走哪种语域"，所以每条语料只有三样东西**：一句输入、一个人工
标注的 ``expect``、一个 ``category``。没有特征、没有权重、没有模型输出——分类的
依据全部在 :mod:`biosnn_memterm.agent.router` 里，这里只放**答案**。

**为什么 ``expect`` 存字符串而不是 :class:`~biosnn_memterm.agent.router.Register`**：
与 :class:`.goldset.GoldMemory` 存 ``kind: str`` 同一条理由——语料是**数据格式**，
不是运行时的对象图。借同一张脸会让"改一条语料"看起来像"改一个枚举"。合法值在
读盘处就被挡住（见 :func:`load_utterances`），所以写错一个字母不会溜到运行时。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .corpus import file_sha256, read_jsonl

__all__ = [
    "REGISTERS",
    "RouteCorpus",
    "Utterance",
    "default_routes_dir",
    "load_corpus",
    "load_utterances",
    "routes_digests",
]

_SCHEMA_VERSION = 1

REGISTERS = ("chat", "task")
"""两个合法语域。**与 :class:`~biosnn_memterm.agent.router.Register` 的取值一致**
——有一条断言钉住这个一致性（``tests/test_router_corpus_integrity.py``），因为两边
分头改名字的话，语料会静默地全部判错，而报告上看起来只是"准确率掉了"。
"""

#: "找不到文件"那句话。见 :func:`.goldset._GOLD_HINT` 的同名说明。
_ROUTES_HINT = (
    "路由语料在仓库根的 benchmarks/routes/ 下；从源码检出跑的话它就在。"
    "找不到通常是工作目录不对，用 --corpus-dir 指定。"
)


@dataclass(frozen=True)
class Utterance:
    """一句输入与它的人工标注。

    Attributes:
        expect: ``"chat"`` 或 ``"task"``——**我们希望系统这一轮怎么走**。它记的
            是"应该"，不是"规则层实际会怎么判"；两者不一样的地方正是这个量具的
            读数。
        category: 难例的类别。定门槛时按类别分列，因为"哪一类在退步"比"总分掉了
            半个点"信息量大得多。
        note: 这一条为什么难、或者判对了意味着什么。给人看的，不参与计算——
            但它是语料里唯一能解释"为什么这条该是 task"的地方，所以别省。
    """

    utterance_id: str
    text: str
    expect: str
    category: str
    note: str = ""


@dataclass(frozen=True)
class RouteCorpus:
    """一整份路由语料。"""

    utterances: tuple[Utterance, ...]
    routes_dir: Path

    @property
    def categories(self) -> dict[str, int]:
        """类别普查。**给人看的数字，也是断言的对象**——见 freeze.json。"""
        counted: dict[str, int] = {}
        for utterance in self.utterances:
            counted[utterance.category] = counted.get(utterance.category, 0) + 1
        return dict(sorted(counted.items()))


def load_utterances(path: Path) -> list[Utterance]:
    """读 ``utterances.jsonl``。``expect`` 不是 ``chat``/``task`` 就当场炸。

    在读盘处挡住而不是在比较处：写错一个字母（``"Task"``、``"任务"``）的症状是
    **那一条永远判错**，而报告上只会看到"准确率低了 0.008"。这类"看起来像模型
    不行"的数据错误，是本项目里最贵的一类 bug。
    """
    utterances: list[Utterance] = []
    for row in read_jsonl(path, hint=_ROUTES_HINT, what="清单"):
        expect = row["expect"]
        if expect not in REGISTERS:
            raise ValueError(
                f"{path} 里 {row['id']!r} 的 expect={expect!r} 不是 "
                f"{' 或 '.join(REGISTERS)}。语料的标注词表与代码里的语域枚举必须一致。"
            )
        utterances.append(
            Utterance(
                utterance_id=row["id"],
                text=row["text"],
                expect=expect,
                category=row["category"],
                note=row.get("note", ""),
            )
        )
    return utterances


def default_routes_dir() -> Path:
    """定位仓库根的 ``benchmarks/routes/``。

    与 :func:`.goldset.default_gold_dir` 同一个做法：**按"那里有没有 utterances.jsonl"
    判断**，不按固定的层数（``parents[3]``）。层数会随包结构调整，而"语料在那儿"
    不会——错误的方式是某次重构之后悄悄指向了另一个目录，然后基准拿一份空语料
    跑出一堆 0。
    """
    for candidate in Path(__file__).resolve().parents:
        routes = candidate / "benchmarks" / "routes"
        if (routes / "utterances.jsonl").is_file():
            return routes
    raise FileNotFoundError(
        "在源码树的任何一层都没找到 benchmarks/routes/utterances.jsonl。"
        "路由语料只在源码检出里（sdist 会带上，wheel 不带）——"
        "如果你是从安装好的包跑，请用 --corpus-dir 指向语料目录。"
    )


def load_corpus(routes_dir: Path | None = None) -> RouteCorpus:
    """把一份路由语料整个读进来。"""
    directory = Path(routes_dir) if routes_dir is not None else default_routes_dir()
    return RouteCorpus(
        utterances=tuple(load_utterances(directory / "utterances.jsonl")),
        routes_dir=directory,
    )


def routes_digests(routes_dir: Path) -> dict[str, object]:
    """语料的条数与 sha256。**冻结判据就是它。**"""
    directory = Path(routes_dir)
    path = directory / "utterances.jsonl"
    return {
        "schema_version": _SCHEMA_VERSION,
        "utterances": {
            "count": len(load_utterances(path)),
            "sha256": file_sha256(path),
        },
    }
