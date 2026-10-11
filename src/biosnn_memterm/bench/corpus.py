"""语料读盘的两个通用动作：冻结指纹，与逐行 JSON。

**为什么单列一个模块。** 仓库里有**三份**语料：召回的金标准集
（``benchmarks/gold/``，200 条记忆 + 76 条查询）、路由的量具
（``benchmarks/routes/``，一份带人工标注的输入清单），以及关联基质的量具
（``benchmarks/substrate/``，200 条带簇标签的记忆 + 40 条查询）。它们量的是完全
不相干的东西，但"怎么从盘上读一行行 JSON""怎么算冻结判据"必须是**同一套**：各写
一遍的话，某天有人在一边加了"坏行报出行号"，另一边没加，于是几份语料的严格程度
不同——而这件事不会有任何症状，直到某天一份坏语料静默跑出一堆 0。

**"读同一份字节"这件事也在这里被钉死。** UTF-8 显式写死（Windows 上
:func:`open` 的默认编码是 cp936，会把中文读坏，而坏掉的样子是"某些查询突然召不回
东西"，看起来像算法问题）、指纹取**未经解码与换行转换的原始字节**。只要有一处随
平台漂移，冻结哈希就失效——而失效的指纹比没有指纹更糟，因为守卫看起来还在。

**第三件共用的事：语料里不许出现什么。** 见 :data:`PERSONAL_DENYLIST`——它原本
只写在守金标准集的那个测试文件里，而路由语料是同一份风险的第二个入口。各存一份
的话，新语料会默认继承**零**保护（基质那份就是这么进来的第三份）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

__all__ = [
    "PERSONAL_DENYLIST",
    "file_sha256",
    "personal_identifier_hits",
    "read_jsonl",
]

#: 语料里**不许出现**的东西：关于作者本人、本人机器、或本项目其它仓库的痕迹。
#:
#: 语料是虚构的（金标准集是海洋观测站数据，路由语料是泛化的日常句子），它没有
#: 任何理由含真实身份信息。这条为什么是机械断言而不是一句承诺：建公开仓时
#: "有没有夹带私人信息"是唯一一个**不可逆**的问题——推上去就收不回来了，而人眼
#: 扫几百行中文是扫不干净的。
#:
#: 加词的门槛很低：多拦一个词的成本只是"以后写语料时换个说法"，而漏一个词的代价
#: 是私人信息进了公开历史。**它放在这一层而不是测试里**，是为了让新加的语料没有
#: 机会"忘了继承"这条规则——两次都从同一个常量取。
PERSONAL_DENYLIST: Final = (
    # 身份
    "离中梦蝶",
    "lzmd",
    "269058959",
    "noreply",
    "github",
    # 机器
    "RTX",
    "5060",
    "显存",
    "D 盘",
    "C:\\",
    "D:\\",
    # 本项目的其它仓库与依赖
    "BioSNN",
    "GasSensor",
    "Ollama",
    "PySide6",
    "记忆终端",
)


def personal_identifier_hits(text: str) -> list[str]:
    """这份文本里命中了哪些禁词（空列表 = 干净）。

    **不做大小写折叠**：表里的英文字面量（``Ollama``、``RTX``）按写法就有意区分，
    而``ollama`` 出现在语料里同样是泄漏。要收小写就把它加进表——那是一次看得见的
    改动，比一条隐式的折叠规则好。
    """
    return [word for word in PERSONAL_DENYLIST if word in text]


def file_sha256(path: Path) -> str:
    """文件的 sha256（**原始字节**，不经任何解码与换行转换）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path, *, hint: str = "", what: str = "语料") -> list[dict]:
    """一行一条 JSON。空行跳过，坏行把行号报出来。

    坏行必须**当场炸**而不是跳过：语料少一条的症状是"某个查询突然没有正确答案
    了"，而那看起来像算法问题。

    ``hint`` 是"找不到文件时该去哪儿找"那句话，由各份语料自己给——金标准集在
    ``benchmarks/gold/``、路由语料在 ``benchmarks/routes/``，把其中一个的路径写死
    在这里，另一个的错误提示就会指向错误的目录。``what`` 只用在"一行都没有"那句
    话里（"空语料"与"空清单"对读的人不是同一件事）。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到语料文件 {path}。{hint}")
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {number} 行不是合法 JSON：{exc}") from exc
        if not isinstance(row, dict):
            # 一行一个数组或一个数也是合法 JSON，但它不是一条语料。不挡住的话，
            # 后面那句 row["id"] 会抛 TypeError——报错里看不到行号，而"第几行"
            # 是唯一能把它拉回"语料坏了"的线索。
            raise ValueError(f"{path} 第 {number} 行不是对象，是一个 {type(row).__name__}。")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} 里一行都没有。空{what}会让每一项指标都空转通过。")
    return rows
