"""仓库卫生：**用户自己的东西永远不进来**，量具语料里没有真机器与真项目。

守的是两类失败，两类都会在"转公开"的那一刻变得不可逆：

1. **运行时的用户数据进了版本控制。** 会话逐字稿、记忆库、壁纸、配置——它们是这个
   程序存在的理由，也正是最不该离开本机的东西。建筑上的第一道防线是
   :mod:`biosnn_memterm.paths`：数据目录默认解析到用户目录，**不在仓库里**。第二道
   是 ``.gitignore``。本文件是第三道，而且与前两道不同——**它检查的是结果而不是
   意图**：``git ls-files`` 里有什么，推上去就有什么，``.gitignore`` 写对没写对
   与它无关。

2. **量具语料里混进真实细节。** ``tools/bench_space.py`` 的 40 句是全仓唯一"形态上
   就是记忆条目"的东西——第一人称、具体、可查。作者的显卡型号、真实盘符、另一个
   仓库的名字（2026-10-10 已换成同形状的虚构事实）里没有一条是"秘密"，但它们连起来
   就是一份作者机器与项目清单，而**转公开不可逆**。

**为什么这两条必须是断言而不是注释。** 建仓时"有没有夹带私人信息"是唯一一个推上去
就收不回来的问题，而人眼扫 200 行中文是扫不干净的——这一点
``tests/test_goldset_integrity.py`` 的 ``_PERSONAL_DENYLIST`` 已经写过一遍，本文件是
同一件武器对着另一个靶子。

**为什么本文件不检查 README。** README 是散文，它**必须**能写 "Ollama"、"PySide6"、
"记忆终端" 这些词（它们是本项目的技术栈与名字，不是私人信息）。可机械化的判据是
"这串字符不该出现"，而 README 里几乎每个词都该出现。README 的脱敏只能靠人改一次，
改完靠 review 保持——它不在本文件的射程内，别以为这里绿了 README 就干净。

**一处知情的取舍：denylist 自己写着被禁的词。** 要机械地拦住一个词，就得把那个词
写下来——所以本文件（以及 ``test_goldset_integrity.py``）里仍然能找到那些字符串，
``git ls-files | grep -i rtx`` 也仍然会有命中。这不是漏网：语料里的**正文**已经干净，
剩下的只是"一张被禁词的清单"，它读起来是"作者大概有某块显卡、用某个盘符"，而不是
任何一句关于作者的陈述。

之所以接受它：**没有一个通用的规则能替代它**。禁"任何显卡型号"会连虚构的替代品一起
禁掉，而如果替代品不能是具体型号，"两句共享一个稀有专名"这个机制就没了——那机制正是
这组配对量的东西。别的具体名词（相机、打印机……）也一样是真产品，换成哪一个都得有一张
点名清单。**这条取舍是选出来的，不是没想到**；真要更严，唯一的方向是接受语料失去
区分度。
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from biosnn_memterm import paths

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_SPACE = _REPO_ROOT / "tools" / "bench_space.py"
_GOLD = _REPO_ROOT / "benchmarks" / "gold"

# ── 运行时数据：哪些路径算"用户的东西" ─────────────────────────────────────

#: 按后缀识别。``.db-wal`` / ``.db-shm`` 是 SQLite 的 WAL 伴生文件——**它们和主库
#: 一样含正文**，只挡 ``*.db`` 是最常见的那种漏。
_RUNTIME_SUFFIXES = frozenset(
    {
        ".db",
        ".db-wal",
        ".db-shm",
        ".db-journal",
        ".sqlite",
        ".sqlite3",
        ".npz",  # P4 的关联基质图
        ".npy",
        ".gguf",  # README 教用户自己 create 的模型权重
    }
)

#: 按目录名识别。复数，为了跟源码目录 ``src/biosnn_memterm/session/`` 区分开。
_RUNTIME_DIRS = frozenset({"data", "sessions", "wallpapers"})

#: 哪些目录里的 ``.jsonl`` 是**量具语料**；其余 ``.jsonl`` 一律当逐字稿。
#:
#: **一份语料登记一次，不是"benchmarks/ 下面的都放行"。** 那条更松的写法会让
#: ``benchmarks/transcripts/2026-10-10.jsonl`` 这种误放进来的逐字稿被静默放行——
#: 而"放行"在这里的含义是"推上去就收不回来了"。逐个登记的另一半好处是：加一份
#: 新语料时**必须动这个文件**，于是"这里还有一份语料"这件事在 diff 里看得见。
_JSONL_ALLOWED_UNDER = frozenset(
    {("benchmarks", "gold"), ("benchmarks", "routes"), ("benchmarks", "substrate")}
)

#: 走工作区时的排除项（没有 ``.git`` 时用）。有 ``.git`` 时 ``git ls-files``
#: 本来就不会列出它们。
_WALK_SKIP = frozenset(
    {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
)


def _tracked_paths() -> list[str]:
    """**下一次推送会带上去的**那些路径。

    有 ``.git`` 就用 ``git ls-files``：它答的正是这个问题，而且只答这个问题——
    工作区里多几个未跟踪的库文件是正常的（那正是 ``.gitignore`` 的作用），不该红。

    没有 ``.git``（比如从 sdist 解包出来跑测试）就退化成走一遍工作区：那里没有
    任何东西被忽略，"存在"就等于"会被带走"，判定只会更严，不会更松。
    """
    if (_REPO_ROOT / ".git").exists():
        raw = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        return [p for p in raw.decode("utf-8").split("\0") if p]

    found: list[str] = []
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(_REPO_ROOT)
        if _WALK_SKIP.intersection(relative.parts[:-1]):
            continue
        found.append(relative.as_posix())
    return found


def _is_runtime_artifact(raw: str) -> bool:
    """这条路径属于"用户的东西"吗。

    四个判据是**或**的关系，写成一个表达式而不是四个 ``elif``：它们是四条独立的
    线索（名字、后缀、语料例外、所在目录），不是一条判断的四个分支。
    """
    path = PurePosixPath(raw)
    return (
        path.name == "config.toml"  # config.example.toml 名字不同，不中
        or path.suffix.lower() in _RUNTIME_SUFFIXES
        or (path.name.endswith(".jsonl") and tuple(path.parts[:2]) not in _JSONL_ALLOWED_UNDER)
        or bool(_RUNTIME_DIRS.intersection(path.parts[:-1]))  # 只认目录，不认文件名
    )


def _runtime_violations(paths_: list[str]) -> list[str]:
    """哪些路径属于"用户的东西"。返回原样的路径串，好直接贴进失败消息。"""
    return [raw for raw in paths_ if _is_runtime_artifact(raw)]


# ── 一、运行时数据 ─────────────────────────────────────────────────────────


def test_default_data_dir_is_outside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认解析出来的数据目录不在仓库里。

    ``paths.py`` 的模块 docstring 把这条说成"只有一个规则，但值得单独一个模块"，
    这里把它变成可执行的：住在仓库里有三个后果，最隐蔽的那个是把私人对话连同
    embedding 一起提交进版本控制。

    **量的是默认值，所以先把环境变量摘掉**——``BIOSNN_MEMTERM_DATA`` 是给开发用的
    逃生口，它指到哪儿都行（``.gitignore`` 与上面的断言会兜住文件层面），但"默认"
    必须是仓库外面。
    """
    monkeypatch.delenv("BIOSNN_MEMTERM_DATA", raising=False)
    data = paths.data_dir().resolve()
    repo = _REPO_ROOT.resolve()
    assert repo not in data.parents, (
        f"默认数据目录解析到了仓库里面：{data}。\n"
        "记忆库不能住在仓库目录里（src/biosnn_memterm/paths.py 的 docstring 讲了"
        "三个后果）。默认值应当来自 %LOCALAPPDATA% / XDG_DATA_HOME。"
    )


def test_runtime_data_is_not_tracked() -> None:
    """索引里没有运行时的用户数据。

    这条查的是 ``git ls-files``——**它不读 .gitignore**。两者不重复：``.gitignore``
    是"以后别再放进来"，这条是"现在里面确实没有"。删掉 ``.gitignore`` 里的一行不会
    让这条变红（规则没了，但文件也还没进来），可一旦真有人 ``git add -f`` 了一份
    逐字稿，它就红——而那正是需要红的时刻。
    """
    violations = _runtime_violations(_tracked_paths())
    assert violations == [], (
        "这些文件已经在版本控制里，推上去就等于把它们发出去：\n  "
        + "\n  ".join(sorted(violations))
        + "\n\n修法：git rm --cached <路径>，确认 .gitignore 覆盖它，然后**重写历史**"
        "（git filter-repo）——已经 push 过的话，删掉文件不等于删掉内容。\n"
        "如果其中某个是**测试夹具**而不是用户数据（比如将来的 tests/data/*.jsonl），"
        "那它属于例外：在 .gitignore 里给它补一行 !放行，并在本文件的 "
        "_JSONL_ALLOWED_UNDER 或后缀表里说明为什么。"
    )


# ── 二、量具语料 ───────────────────────────────────────────────────────────


def _is_git_ignored(relative: str) -> bool:
    """``git check-ignore`` 对这条路径的答案：忽略规则会挡住它吗。

    ``git check-ignore`` 只看规则、不看文件在不在索引里，所以它答的正是这里要问的
    那个问题（``git ls-files`` 答不了——索引里的东西当然没被忽略）。
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=_REPO_ROOT,
        capture_output=True,
    )
    # 0 = 被忽略、1 = 没被忽略。别的退出码是真的出错了——那时宁可炸，也别把它当成
    # "没被忽略"从而安静地放行。
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore 对 {relative} 的退出码是 {result.returncode}："
            f"{result.stderr.decode('utf-8', 'replace')}"
        )
    return result.returncode == 0


def test_registered_corpus_directories_survive_the_ignore_rules() -> None:
    """登记过的语料目录里的 ``.jsonl``，**不许被 ``.gitignore`` 挡住**。

    这条补的是上面 :func:`test_runtime_data_is_not_tracked` 的盲区：那条查
    ``git ls-files``，而索引里的东西当然没被忽略——"一份语料被忽略规则抓走"它答
    不了。路由语料落地时正是这样坏的：``.gitignore`` 没加放行，那份
    ``utterances.jsonl`` 只在工作区里，**本地每条腿都绿**，而门槛与指纹测试要等
    **别人检出之后**才红（README 记着这次事故）。

    所以这里问的是另一个问题——不是"索引里有没有"，而是"``_JSONL_ALLOWED_UNDER``
    里的每个目录，打开来里面的 ``.jsonl`` 会不会被推上去"。它今天就能红：把
    ``.gitignore`` 里对应的那行 ``!…/*.jsonl`` 删掉即可。

    **只查"被忽略"，不查"已入索引"**：后者取决于此刻有没有 ``git add`` 过，会让
    测试的结果随暂存区漂移。没被忽略却还没入索引的文件在 ``git status`` 里是看得见
    的 ``??``——这条要守的是**看不见**的那一种。
    """
    if not (_REPO_ROOT / ".git").exists():
        pytest.skip("没有 .git 就无从判断哪些路径会被推上去")

    problems: list[str] = []
    for parts in sorted(_JSONL_ALLOWED_UNDER):
        directory = _REPO_ROOT.joinpath(*parts)
        files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
        if not files:
            problems.append(f"{'/'.join(parts)}/ 里一份 .jsonl 都没有——登记表过期了？")
            continue
        for path in files:
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if _is_git_ignored(relative):
                problems.append(f"{relative} 被 .gitignore 挡住了")

    assert not problems, (
        "登记在 _JSONL_ALLOWED_UNDER 里的语料有推不上去的：\n  "
        + "\n  ".join(problems)
        + "\n\n修法：在 .gitignore 里照 `!benchmarks/gold/*.jsonl` 那两行给它补一行"
        "放行。症状特别隐蔽——本地每条腿都绿，红的是 CI 与别人的检出。"
    )


#: ``tools/bench_space.py`` 的语料里**不许出现**的东西。
#:
#: 与 ``test_goldset_integrity.py`` 的 ``_PERSONAL_DENYLIST`` 是**两份不同的表**，
#: 不是抄漏了：那份守的是虚构语料（连 "显存"、"PySide6"、"记忆终端" 都不许出现，
#: 因为一个海洋观测站的故事里没有这些词），这份守的是本项目的量具——它天然要谈
#: PySide6、SQLite、SDR 这些本项目的技术，**只有"具体到某台机器/某个仓库"的才禁**。
_MACHINE_AND_NEIGHBOUR_DENYLIST = (
    # 作者的机器：显卡型号与显存档位、真实盘符
    "RTX",
    "5060",
    "C 盘",
    "D 盘",
    "C:\\",
    "D:\\",
    # 相邻仓库的名字。语料是"记忆条目"的形状，点到别的仓库就等于公开一张项目清单
    "GasSensor",
    "BioSNN",
    # 身份
    "离中梦蝶",
    "lzmd",
)


def _bench_space_sentences() -> list[str]:
    """用 ast 取出 ``RELATED`` / ``UNRELATED`` 里的句子。

    **不 import 那个模块**：它在导入时就会往 ``sys.path`` 里塞 src/，而这里只需要
    两个字面量。ast 还顺带保证了"取到的确实是这两张表"——将来有人把语料改成从文件
    读，这里的 ``assert`` 会红，提醒把守卫挪到新位置，而不是安静地守卫一个空列表。
    """
    tree = ast.parse(_BENCH_SPACE.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        if target in {"RELATED", "UNRELATED"} and value is not None:
            found[target] = [s for pair in ast.literal_eval(value) for s in pair]

    assert set(found) == {"RELATED", "UNRELATED"}, (
        f"在 {_BENCH_SPACE.name} 里没找到 RELATED/UNRELATED 两个字面量表（找到的是 "
        f"{sorted(found)}）。语料搬家了？把这个守卫挪过去，别让它守着一个空列表。"
    )
    return [sentence for group in found.values() for sentence in group]


def test_bench_space_corpus_has_no_machine_details() -> None:
    """量具语料里没有真实机器 / 相邻仓库 / 作者身份。

    改语料是允许的（冻结的纪律是"同一次提交里一起改数字"，不是"永远不许改"），
    但**不能改回真实细节**——所以这条不是"别动语料"，是"动完还得是虚构的"。
    """
    corpus = "\n".join(_bench_space_sentences())
    hits = [token for token in _MACHINE_AND_NEIGHBOUR_DENYLIST if token in corpus]
    assert hits == [], (
        f"tools/bench_space.py 的语料里出现了不该有的词：{hits}。\n"
        "这些句子的形态就是记忆条目，写进真实细节等于把作者机器/项目清单记进仓库。"
        "换成同形状的虚构事实即可——**要保留的机制是'两句共享一个稀有专名或数字'**"
        "（那是被量的东西），不是那个专名本身。"
    )


def test_bench_space_corpus_is_still_loadable() -> None:
    """反空转：上面那条守卫的作用域不是一张空表。

    否则把语料删空就能让 ``test_bench_space_corpus_has_no_machine_details`` 变绿
    ——一个守卫最坏的失败方式不是漏报，是**报绿**。
    """
    sentences = _bench_space_sentences()
    assert len(sentences) == 80, f"语料应当是 20 对 + 20 对（80 句），实际 {len(sentences)} 句。"
    assert all(s.strip() for s in sentences), "语料里有空串。"
