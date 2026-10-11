"""embedding 空间基准：这个模型能不能用来做这条记忆通道。

计划书 §6.1 预定的红线::

    mean(cos_同义) > mean(cos_异义) + 0.15   —— 不达标则**拒绝启用该模型**

**这个脚本存在的意义是让"拒绝"可执行。** 一条红线如果只是写在计划书里，等到
实测不达标的那天，最省事的做法是"再调调阈值"；把它写成带退出码的脚本，拒绝就
是一个可以执行的动作，而不是一次需要意志力的克制。

量的是**排序质量**，不是绝对相似度。召回本质上是排序：真正要问的是"相关的那条
能不能排到不相关的前面"，而均值差会被区间重叠掩盖——两个分布可以均值差很大却
大面积交叉。所以除了均值差，还报 AUC（任取一对同义与一对异义，同义分更高的
概率；0.5 = 瞎猜）与"同义对中高于异义最高分的条数"。

顺带把**词法 SDR 通道**用同一组配对量一遍。它是模型无关的对照：如果换一个嵌入
模型带来的提升还不如一个 40 行的哈希编码器，那这笔账就得如实记下来。

用法::

    python tools/bench_space.py                      # 量本机默认模型
    python tools/bench_space.py --model bge-m3:latest
    python tools/bench_space.py --no-lexical         # 只量嵌入通道
    python tools/bench_space.py --pair-prefix        # 走 nomic 推荐的 search_* 前缀

退出码：红线不达标为 1，达标为 0。**不等于程序出错**——它表达的是"按预定标准，
这个模型不予启用"。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: 红线的差值门槛，来自计划书 §6.1。
MARGIN_THRESHOLD = 0.15

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text:latest"

#: 同主题的配对：左边是"用户会说的一句话"，右边是"库里可能存着的那条记忆"。
#: 刻意做成**不同措辞**——同一句话的两次复制会让任何编码器都拿满分，那种样本
#: 除了自我安慰没有别的用处。
#:
#: **语料是冻结的：别跟着阶段代号一起改。** 这些句子是量具，不是文档——改一个字，
#: `--json` 里的数就变了，而已经登记在 `recall.py`、`README.md`、`CORPUS.md` 里的
#: 那些数不会跟着变，两边从此对不上。（2026-10-10 阶段代号重排时就差点把 "P2" 改成
#: "P3"，故留此句。）
#:
#: 冻结的纪律是**同一次提交里一起改**，不是"永远不许改"。2026-10-10 就改过一次，
#: 理由是脱敏：有三对原本写的是真机器的真事实——作者的显卡型号与显存大小、真实盘符、
#: 另一个仓库的名字（此处**不复述具体值**，复述一遍就等于没删）。
#:
#: 这三条的形态**恰好就是记忆条目**：第一人称、具体、可查。仓库还是 private 时它们
#: 只是不体面，转公开的那一刻它们就成了"作者机器与项目清单"的公开记录，而**转公开
#: 不可逆**。所以换成同形状的虚构事实——显卡换一块更老的、盘符换一个别处的、目录名
#: 换成中性的——保留"专名/数字的精确匹配"这个被量的机制，去掉真实指涉。
#:
#: 代价是数字动了：词法 AUC **0.890 → 0.884**，嵌入 AUC **0.733 → 0.760**。四个
#: 文件里的登记点（`recall.py`、`test_recall.py`、`README.md`、`CORPUS.md`）连同这条
#: 注释在同一次提交里一起更新。**改语料而不更新登记点，等于留下一组
#: 指向不存在的尺子的数字。** `tests/test_repo_hygiene.py` 把这条纪律机械化：语料里
#: 再出现真机器/真项目的名字，测试直接红。
RELATED: list[tuple[str, str]] = [
    (
        "我在做一个记忆终端，记忆要独立于所调用的模型。",
        "长期记忆存在本地，换大模型后记忆和人格都不变。",
    ),
    ("把 agent 功能排在新 P2，紧跟 P1。", "工具调用放在第二阶段，紧接着记忆 v0 做。"),
    ("存储用 SQLite 加 WAL 模式。", "数据库选了 SQLite，开启 WAL，写入走事务。"),
    (
        "召回用词法 SDR 做去冗，不用 embedding 余弦。",
        "去重那一步走的是稀疏表示的 Jaccard，换模型后行为一致。",
    ),
    ("我家的路由器放在书房书架顶层。", "网络设备搁在书房最上面一排架子上。"),
    ("今天午饭吃的是番茄炒蛋和米饭。", "中午吃了番茄炒蛋配白饭。"),
    ("token 预算是硬约束，超了要按层反向驱逐。", "上下文装不下时从最外层往里删，人格锚不动。"),
    ("惰性衰减不需要后台任务。", "权重的衰减在读取的时候顺便算，不跑守护进程。"),
    ("我偏好中文回复。", "回答用中文就行。"),
    ("确认闸门用 threading.Event 做跨线程握手。", "等用户点确认那一步用事件对象阻塞工作线程。"),
    ("素材库的索引在 E 盘。", "备份文件都放在 E 盘。"),
    ("那个小工具项目在 toolbox 目录下。", "插件放在仓库外面单独一个目录里。"),
    ("不能对浮点 SDR 调默认阈值 0.5 的二值化。", "稀疏表示的权重值域是 0 到 1，二值化会砍掉大半。"),
    ("PySide6 会吞掉槽函数里抛出的异常。", "Qt 的槽里报错不会冒泡出来，界面会卡在生成中。"),
    ("我把会话记录写成 JSONL，每行一条消息。", "逐字稿是行分隔的 JSON，一行一条。"),
    ("记忆写入是每轮实时发生的。", "每说完一轮就往库里写一条，不等到会话结束。"),
    (
        "模型只通过检索注入读取记忆，永远不拥有它。",
        "大模型看不到整个记忆库，只能看到这一轮注入的那几条。",
    ),
    ("人格锚是 pinned 的普通条目。", "人格那一条就是打了个钉住标记的普通记忆。"),
    ("我有一块 GTX 1060，6GB 显存。", "显卡是 1060，显存只有六个 G。"),
    ("训练数据里中文占比不高。", "这个语料库以英文为主。"),
]

#: 异义配对。**与上面的 RELATED 共用同样的句子**，只是错开搭配——两组用不同的
#: 句子会让"同义 vs 异义"的差值混进"长句 vs 短句"这样的混淆变量。
UNRELATED: list[tuple[str, str]] = [
    ("我在做一个记忆终端，记忆要独立于所调用的模型。", "我家的路由器放在书房书架顶层。"),
    ("存储用 SQLite 加 WAL 模式。", "今天午饭吃的是番茄炒蛋和米饭。"),
    ("把 agent 功能排在新 P2。", "窗外在下雨，楼下的猫躲在车底下。"),
    ("召回用词法 SDR 做去冗。", "我偏好中文回复。"),
    ("token 预算是硬约束。", "素材库的索引在 E 盘。"),
    ("惰性衰减不需要后台任务。", "那个小工具项目在 toolbox 目录下。"),
    ("确认闸门用 threading.Event。", "我有一块 GTX 1060，6GB 显存。"),
    ("PySide6 会吞掉槽函数里的异常。", "训练数据里中文占比不高。"),
    ("会话记录写成 JSONL。", "人格锚是 pinned 的普通条目。"),
    ("记忆写入是每轮实时发生的。", "我家的路由器放在书房书架顶层。"),
    ("模型只通过检索注入读取记忆。", "今天午饭吃的是番茄炒蛋和米饭。"),
    ("我偏好中文回复。", "窗外在下雨，楼下的猫躲在车底下。"),
    ("素材库的索引在 E 盘。", "召回用词法 SDR 做去冗。"),
    ("那个小工具项目在 toolbox 目录。", "token 预算是硬约束。"),
    ("不能对浮点 SDR 调默认阈值 0.5。", "惰性衰减不需要后台任务。"),
    ("我把 GTX 1060 的显存记成 6GB。", "确认闸门用 threading.Event。"),
    ("人格锚不参与常规召回。", "PySide6 会吞掉槽里的异常。"),
    ("中文占比不高的训练数据。", "会话记录写成 JSONL。"),
    ("每轮实时写入记忆。", "记忆写入是每轮实时发生的，但这条在讲天气：今天晴。"),
    ("稀疏表示的权重值域是 (0,1]。", "模型只通过检索注入读取记忆。"),
]


# ── 嵌入通道 ────────────────────────────────────────────────────────────────


def embed(host: str, model: str, texts: list[str]) -> np.ndarray:
    """批量取向量，按行 L2 归一化。"""
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise SystemExit(f"Ollama 拒绝了这个请求（{exc.code}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"连不上 Ollama（{host}）：{exc.reason}") from exc

    vectors = np.asarray(body["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def embed_scores(
    host: str, model: str, pairs: list[tuple[str, str]], *, prefixes: str
) -> np.ndarray:
    """每对配对的余弦。

    ``prefixes`` 为 ``"search"`` 时给两侧加上 nomic 家族文档化的任务前缀。
    Ollama 的 ``/api/embed`` 不会自动加，所以"裸调用"与"按推荐用法调用"是两种
    不同的几何——两种都量，才知道差的是模型还是用法。
    """
    if prefixes == "search":
        left = [f"search_query: {a}" for a, _ in pairs]
        right = [f"search_document: {b}" for _, b in pairs]
    else:
        left = [a for a, _ in pairs]
        right = [b for _, b in pairs]

    left_vectors = embed(host, model, left)
    right_vectors = embed(host, model, right)
    return np.sum(left_vectors * right_vectors, axis=1)


# ── 词法通道（对照） ────────────────────────────────────────────────────────


def lexical_scores(pairs: list[tuple[str, str]]) -> np.ndarray:
    """同一组配对的词法 SDR Jaccard。**不需要模型、不需要网络。**"""
    from memcore.sdr import encode_lexical

    return np.array(
        [encode_lexical(a).jaccard(encode_lexical(b)) for a, b in pairs], dtype=np.float64
    )


# ── 统计 ────────────────────────────────────────────────────────────────────


def auc(related: np.ndarray, unrelated: np.ndarray) -> float:
    """任取一对同义与一对异义，同义分更高的概率。0.5 = 瞎猜。

    用这个而不是只用均值差：两个分布完全可以均值差很大却大面积交叉，而交叉
    正是阈值切不开的原因。
    """
    wins = sum(1 for a in related for b in unrelated if a > b)
    ties = sum(1 for a in related for b in unrelated if a == b)
    return (wins + 0.5 * ties) / (len(related) * len(unrelated))


def report(tag: str, related: np.ndarray, unrelated: np.ndarray) -> dict[str, float]:
    margin = float(related.mean() - unrelated.mean())
    result = {
        "tag": tag,
        "related_mean": float(related.mean()),
        "related_min": float(related.min()),
        "related_max": float(related.max()),
        "unrelated_mean": float(unrelated.mean()),
        "unrelated_min": float(unrelated.min()),
        "unrelated_max": float(unrelated.max()),
        "margin": margin,
        "auc": auc(related, unrelated),
        "above_unrelated_max": int((related > unrelated.max()).sum()),
        "n": len(related),
    }
    # ``related.mean()`` 是 numpy 标量，它参与比较得到的是 ``numpy.bool_``——那是
    # ``json.dumps`` 序列化不了的东西，而这个函数的结果会原样进 ``--json``。
    passed = bool(related.mean() > unrelated.mean() + MARGIN_THRESHOLD)
    result["passed"] = passed

    print(f"\n── {tag} ──")
    print(
        f"  同义 均值 {result['related_mean']:.3f}  区间 [{result['related_min']:.3f}, {result['related_max']:.3f}]"
    )
    print(
        f"  异义 均值 {result['unrelated_mean']:.3f}  区间 [{result['unrelated_min']:.3f}, {result['unrelated_max']:.3f}]"
    )
    print(
        f"  差值 {margin:+.3f}   红线 +{MARGIN_THRESHOLD:.3f}   → {'达标' if passed else '**不达标**'}"
    )
    print(
        f"  AUC {result['auc']:.3f}   同义对中高于异义最高分的 {result['above_unrelated_max']}/{len(related)}"
    )
    if not passed:
        # 把"为什么不能只调阈值"直接写出来，免得下一个读到这份输出的人去调阈值。
        print(
            f"  区间重叠：同义最低 {result['related_min']:.3f} < 异义最高 "
            f"{result['unrelated_max']:.3f}，**任何单一阈值都切不开**"
        )
    return result


def sweep(related: np.ndarray, unrelated: np.ndarray) -> None:
    """扫一遍阈值，看取在哪里能把两组分开。

    这是 :func:`memcore.recall.RecallConfig.min_score` 的标定来源。**注意它量的
    是单对配对**（"用户这句话 → 库里那条记忆"），而真实召回里每条候选都要与
    整个逐字稿窗口竞争，所以真实的最优阈值会比这里低一些——这里的数给出的是
    量纲与上限，不是一个可以直接抄进代码的常数。
    """
    print("\n── 阈值扫描（左列=阈值，中列=20 对同义里通过几对，右列=20 对异义里误过几对） ──")
    best = None
    for threshold in [round(0.04 + 0.02 * i, 2) for i in range(14)]:
        hits = int((related >= threshold).sum())
        false_alarms = int((unrelated >= threshold).sum())
        youden = hits / len(related) - false_alarms / len(unrelated)
        mark = ""
        if best is None or youden > best[1]:
            best = (threshold, youden, hits, false_alarms)
            mark = "  ←"
        print(
            f"  {threshold:.2f}   {hits:2d}/{len(related)}   {false_alarms:2d}/{len(unrelated)}{mark}"
        )
    if best:
        print(
            f"  最优（Youden's J）阈值 {best[0]:.2f}：召回 {best[2]}/{len(related)}，"
            f"误报 {best[3]}/{len(unrelated)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--pair-prefix",
        action="store_true",
        help="给两侧加 nomic 家族的 search_query:/search_document: 前缀",
    )
    parser.add_argument("--no-lexical", action="store_true", help="跳过词法对照")
    parser.add_argument("--sweep", action="store_true", help="扫一遍阈值（标定 min_score）")
    parser.add_argument("--json", action="store_true", help="额外输出一份 JSON")
    args = parser.parse_args()

    print(f"模型 {args.model} @ {args.host}")
    print(f"配对：{len(RELATED)} 对同义 + {len(UNRELATED)} 对异义，中文，短句")

    results: list[dict[str, float]] = []

    prefixes = "search" if args.pair_prefix else "none"
    tag = f"嵌入 {args.model}" + ("（带 search_* 前缀）" if args.pair_prefix else "（裸用法）")
    related = embed_scores(args.host, args.model, RELATED, prefixes=prefixes)
    unrelated = embed_scores(args.host, args.model, UNRELATED, prefixes=prefixes)
    results.append(report(tag, related, unrelated))
    embedding_passed = results[0]["passed"]

    if not args.no_lexical:
        lexical = report(
            "词法 SDR Jaccard（无模型、无 GPU、纯 blake2b）",
            lexical_scores(RELATED),
            lexical_scores(UNRELATED),
        )
        results.append(lexical)
        if lexical["auc"] > results[0]["auc"]:
            print(
                f"\n  ⚠️ 词法通道的 AUC（{lexical['auc']:.3f}）**高于**嵌入通道"
                f"（{results[0]['auc']:.3f}）。在这组中文短句上，一个 40 行的哈希"
                f"编码器比 768 维嵌入排得更准。结论只能这么写：**本组样本、本模型**"
                f"下如此——换一个多语言嵌入模型可能反过来。"
            )

    if args.sweep:
        # 扫融合前的原始词法得分——它是两条通道里唯一在两种配置下量纲都稳的，
        # 也是 min_score 直接比较的那个数。
        sweep(lexical_scores(RELATED), lexical_scores(UNRELATED))

    if args.json:
        print("\n" + json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if embedding_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
