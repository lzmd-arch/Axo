"""数据目录的解析。

只有一个规则，但值得单独一个模块：**记忆库不能住在仓库目录里**。

住在仓库里有三个后果，每一个都够糟：``git clean -xdf`` 抹掉全部记忆；用户
换个地方 clone 就"失忆"；以及最隐蔽的——把库提交进版本控制，私人对话内容
连同 embedding 一起进了远端。

因此顺序是（**四档，从上往下第一个说了算**）：

1. 环境变量 ``AXO_DATA``——开发逃生口，也是全部测试与打包自检的隔离开关；
2. 程序目录旁的便携指针文件 :data:`_PORTABLE_NAME`（**只读**，见
   :func:`portable_data_root`）；
3. :func:`set_data_root` 钉进来的那个（界面里选的，存在注册表里，由 Qt 层告知）；
4. 系统用户数据目录。

**这一层不许 import Qt。** 第 3 档的值住在注册表里，读它要用 ``QSettings``，而
``ui/settings_store.py`` 已经 import 了本模块——反过来 import 就是环。所以本模块
只负责"被告诉一个根"，由 ``ui/data_root.py`` 在启动时读注册表再告知。

仓库里的 ``data/`` 只在开发时通过环境变量显式指过去，且已在 .gitignore 里。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "VOICE_MODEL_NAME",
    "consolidation_dir",
    "consolidation_events_path",
    "consolidation_last_path",
    "data_dir",
    "data_root_source",
    "default_data_dir",
    "env_data_root",
    "graph_path",
    "memory_db_path",
    "portable_data_root",
    "portable_pointer_path",
    "reminders_cancelled_path",
    "reminders_fired_path",
    "reminders_path",
    "voice_model_dir",
    "sessions_dir",
    "set_data_root",
    "snapshot_dir",
    "wallpapers_dir",
]

_ENV_OVERRIDE = "AXO_DATA"
_APP_DIR_NAME = "axo"

_PORTABLE_NAME = "axo-data.txt"
"""程序目录旁那个"这一份程序用哪个数据目录"的指针文件。

**只读，从不创建、从不修改。** 那三条限制是刻意的，各自挡住一件事：程序目录可能
根本写不了（装在 ``Program Files`` 里）；"程序不往自己旁边写文件"是发行说明里
写过的承诺，而打包的发布闸门会扫产物树；开发态那个"程序目录"是仓库根，那里有
一整族禁止写入的守卫（见 ``tests/test_repo_hygiene.py``）。

内容是一行绝对路径，``#`` 开头当注释。想便携就自己放一份。
"""

_override: Path | None = None
"""进程内的一次性覆盖（第 3 档）。**只在启动时、建任何 store 之前设一次。**

这一层每个函数都是**当场现算**的（没有缓存，这是有意的），所以中途改它会让已经
建好的东西留在旧根、之后新建的落在新根——那种"一半一半"比"改不了"难查得多。
界面里改位置走的是"写标记 + 重启"，见 ``ui/data_root.py``。
"""


def env_data_root() -> Path | None:
    """环境变量 ``AXO_DATA`` 指的那个目录；没设就是 ``None``。

    单列出来是给界面用的：设了它的时候界面里那两个改位置的按钮不生效（它优先级
    最高），而一个点了没反应的按钮必须有一句话解释。
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if not raw:
        return None
    return Path(raw).expanduser()


def default_data_dir() -> Path:
    """不含任何覆盖的默认根目录（第 4 档）。

    Windows: ``%LOCALAPPDATA%``（不进漫游配置——记忆库可能有几百 MB，跟着域账号
    漫游会把登录拖垮）；其它平台按 XDG 惯例。

    单列出来是为了让界面能说"恢复默认位置"而**不必猜**默认值是什么。
    """
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / _APP_DIR_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / _APP_DIR_NAME


def portable_pointer_path() -> Path | None:
    """第 2 档那个指针文件**应该**在哪（不保证它存在）。

    冻结态是 ``axo.exe`` 所在目录；源码态是仓库根。两个都不是"当前工作目录"——
    从桌面快捷方式起、或者 ``cd`` 到别处再起，工作目录都不一样，把指针挂在那上面
    等于让它在不同起法下指向不同文件。
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        # paths.py 在 src/axo/ 下：parents[0]=src/axo、[1]=src、[2]=仓库根。
        base = Path(__file__).resolve().parents[2]
    return base / _PORTABLE_NAME


def portable_data_root() -> Path | None:
    """读便携指针文件；没有、读不出、或内容不是绝对路径，一律 ``None``。

    **坏内容当没有，不抛。** 这个文件是用户手写的（见 :data:`_PORTABLE_NAME`），
    而"用户在程序旁边放了一个格式不对的文件"不该让程序起不来。取第一条有效行；
    那一条不是绝对路径就整份作废——相对路径的含义取决于工作目录，对 GUI 而言那是
    一次**静默的位置变更**，比"没设"危险得多。
    """
    pointer = portable_pointer_path()
    try:
        text = pointer.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = Path(stripped).expanduser()
        return candidate if candidate.is_absolute() else None
    return None


def set_data_root(root: Path | str | None) -> None:
    """把数据根钉在 ``root``；``None`` = 取消覆盖、回到原规则。

    **不要用它去盖过环境变量**——盖不过。上面那四档的次序是定死的，
    ``AXO_DATA`` 与便携文件都在这一档之上，所以设了它们的时候这个函数等于没设。
    界面靠 :func:`data_root_source` 把这件事说清楚。
    """
    global _override
    _override = Path(root).expanduser() if root is not None else None


def data_root_source() -> str:
    """:func:`data_dir` 现在听谁的：``"env"`` / ``"portable"`` / ``"stored"`` / ``"default"``。

    与 :func:`data_dir` 用同一套判据、同一个次序，**分两个函数写是为了让界面能说
    出理由**——"位置是 `D:\\AxoData`"和"位置是 `D:\\AxoData`，因为你在设置里选过"
    是两句不同的话，而只有后者能解释"为什么我点了没反应"。
    """
    if env_data_root() is not None:
        return "env"
    if portable_data_root() is not None:
        return "portable"
    if _override is not None:
        return "stored"
    return "default"


def data_dir() -> Path:
    """记忆终端的数据根目录，不存在则创建。"""
    root = env_data_root()
    if root is None:
        root = portable_data_root()
    if root is None:
        root = _override
    if root is None:
        root = default_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sessions_dir() -> Path:
    """会话记录目录。"""
    path = data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def wallpapers_dir() -> Path:
    """用户壁纸目录。"""
    path = data_dir() / "wallpapers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_db_path() -> Path:
    """记忆库文件。

    **文件本身不在这里创建**，只保证父目录存在：建库是
    :class:`memcore.store.MemoryStore` 的事（它要设 WAL、建表、写 schema 版本），
    在这里先建一个空文件会让"文件存在但表不存在"变成一个要单独处理的中间态。

    放在 ``memory/`` 子目录而不是直接对着 ``data_dir()``，是因为库不只一个文件——
    它们必须跟着库整体备份、整体搬家：

    * :func:`graph_path` 的 ``graph.npz``：**P4a 已经落在这里**，与 ``mem.db`` 并列；
    * :func:`snapshot_dir` 的巩固前快照（保留最近 5 份）：**P4c 落**，由
      ``memory/consolidation.py`` 那个入口拍、也由它轮转。

    三件都放在这个子目录里，是因为它们必须跟着库整体备份、整体搬家——散在数据根
    目录里迟早会被漏掉一个。
    """
    path = data_dir() / "memory" / "mem.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def graph_path() -> Path:
    """关联图缓存（``memcore.graph.GraphIndex`` 的落盘处）。

    **它是缓存，不是真相**：``mem.db`` 里有什么，图就该长什么样；对不上的正确
    做法是丢掉重算，而不是迁移（``GraphIndex.load`` 的版本校验就按这个来）。
    所以它和 :func:`memory_db_path` 一样**不在这里创建空文件**——"文件在但内容是
    空的"与"还没有建过图"是两种状态，前者会让 ``load`` 报一个误导人的损坏错误。

    ``memory/`` 目录由 :func:`memory_db_path` 保证存在；单独调这个函数时父目录
    可能还没有，所以这里也建一次（幂等）。
    """
    path = data_dir() / "memory" / "graph.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def consolidation_dir() -> Path:
    """睡眠巩固的目录（事件与"上次跑过没有"的登记都在这儿）。

    与库并列放在 ``memory/`` 下，理由同 :func:`memory_db_path`：它们必须跟着库
    整体搬家。**尤其是事件**——事件说的是"哪一轮改了哪些条目"，把它留在原地而把
    库搬走，那份记录就再也对不上任何东西了。

    目录**在这里建**（幂等），文件不建：同 :func:`memory_db_path` 的姿态，
    "文件在但内容空"与"还没有"是两种状态。
    """
    path = data_dir() / "memory" / "consolidation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def consolidation_events_path(run_id: str) -> Path:
    """某一轮巩固的事件文件 ``<run_id>.jsonl``。

    **一轮一个文件**，不是全轮共用一个。这样 ``--replay <run_id>`` 只要一个 id
    就能定位证据，而"读一份文件、按 run_id 过滤"会在文件长大之后变得像在翻一本
    合订本。代价是文件个数随时间增长——它们是审计证据，**不轮转**（与快照刻意
    不对称：快照大、事件小，见 :func:`snapshot_dir`）。

    ``run_id`` 是**外部输入**（``--replay`` 直接吃命令行给的那个字符串），而它
    要拼进文件名，所以拼之前必须验形状——调用方负责（见
    ``tools/sleep_once.py`` 的 ``_checked_run_id``）：``../`` 这类值在这里会
    安静地跑出这个目录。
    """
    return consolidation_dir() / f"{run_id}.jsonl"


def consolidation_last_path() -> Path:
    """上一次巩固的登记：``{"run_id", "at", "count", "newest_id"}``。

    调度器靠它回答"24 小时到了没有"，热度面板靠它说"最近一次巩固是什么时候"。
    **不另开一套 state 机制**：一个进程活不过重启的字段表，不值得为它单开一个
    模块，而它天然属于巩固目录——与它登记的那件事住在一起。

    文件不在这里创建：还没跑过是常态，先建一个空文件只会让"文件在但内容空"与
    "文件还没有"变成两种要分开处理的状态（同 :func:`reminders_path`）。
    """
    return consolidation_dir() / "last.json"


def snapshot_dir() -> Path:
    """巩固前的整库快照目录。

    形态是**整库复制**（``mem-<UTC yyyymmddTHHMMSS>.db``），不是增量——巩固会
    批量改 ``reinforce_count``，出事时需要的正是"改之前那一份完整的库"。

    **保留最近 5 份，轮转由 ``memory/consolidation.py`` 那个入口做，不在这里**：
    本模块只解析位置、不做 I/O（除了建目录），而"删掉最旧的那份"是一件有后果的
    事，它该跟"拍快照"写在同一个地方，读代码的人一眼能看到"留几份"这个数。
    """
    path = data_dir() / "memory" / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reminders_path() -> Path:
    """提醒登记簿（``agent/reminders.py`` 的 ``ReminderStore`` 落盘处）。

    与记忆库分家、**直接躺在数据根目录**：它只有一行行的文本，没有 WAL、没有旁挂的
    快照与图文件，凑不出 :func:`memory_db_path` 那种"必须整体搬家"的一堆东西，
    所以不需要自己的子目录。反过来，把它塞进 ``memory/`` 才是错的那个——"清空记忆"
    与"重建索引"这类操作会以目录为单位下手，而用户还没被提醒的事情不该跟着走。

    **文件本身不在这里创建**（同 :func:`memory_db_path`）：还没约过提醒是常态，先建
    一个空文件只会让"文件在但内容空"与"文件还没有"变成两种要分开处理的状态。真正
    落盘的是 :meth:`~axo.agent.reminders.ReminderStore.add`，它自己建父目录。
    """
    return data_dir() / "reminders.jsonl"


def reminders_fired_path() -> Path:
    """**已经响过**的提醒记在这儿（:class:`~axo.agent.reminders.FiredStore`）。

    **另开一个文件，不往登记簿里追加第二种记录。** ``reminders.jsonl`` 的契约是
    "这个文件装的是**还没被提醒的事情**"（见 :func:`reminders_path` 与
    ``agent/reminders.py`` 的模块文档）；往里掺 fired 行会让那句话变成假的，也会让
    :meth:`~axo.agent.reminders.ReminderStore.all` 从"读出全部提醒"变成
    "读出两种东西"——而它的每一个调用方都按前者写的。

    ``.jsonl`` 同样是一行一条：这个文件会**一直长大**（每次响过追加一行），而
    "不想丢一行就重写整个文件"正是 JSONL 想避免的那件事。
    """
    return data_dir() / "reminders.fired.jsonl"


def reminders_cancelled_path() -> Path:
    """**被取消掉**的提醒 id（``agent.reminders.CancelledStore``）。

    **第三本账，不是把取消写进登记簿。** ``reminders.jsonl`` 的契约是"还没被提醒的
    事情"，往里掺第三种记录会让那句话变成假的；而重写老行更不行——那个文件是只追加
    的（没有 WAL、没有备份，重写一次就是一次全损）。

    于是三本账各说一件事：``reminders.jsonl`` = 登记了这件事，
    ``reminders.fired.jsonl`` = 这一次响过了，这一本 = 这条撤了。

    > **再加第四本之前，先把这三本合成一本事件账。** 三本是"三件事各自发生过"，
    > 四本就是我们其实需要一本次序账了。
    """
    return data_dir() / "reminders.cancelled.jsonl"


#: 语音识别模型的名字（也是它解压后的目录名）。换模型时**这一行与 `asr` 里那句人话
#: 一起改**——它们是同一个事实的两处说法。
VOICE_MODEL_NAME = "vosk-model-small-cn-0.22"


def voice_model_dir() -> Path:
    """语音识别模型放在哪儿。

    **在数据根底下**：于是它跟着"用户把数据搬到别的盘"一起走——代价是搬迁时多搬 42 MB，
    好处是"这个程序的东西都在一个文件夹里"这条不破。把这个选择写在这里，是为了让它成为
    一件**定下来的事**，而不是某天有人发现搬迁变慢时的一个谜。

    与 :func:`memory_db_path` 一样**不创建**：还没下模型是常态，先建一个空目录只会让
    "目录在、模型不在"与"什么都没有"变成两种要分开处理的状态。
    """
    return data_dir() / "models" / VOICE_MODEL_NAME
