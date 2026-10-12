"""数据目录的解析。

只有一个规则，但值得单独一个模块：**记忆库不能住在仓库目录里**。

住在仓库里有三个后果，每一个都够糟：``git clean -xdf`` 抹掉全部记忆；用户
换个地方 clone 就"失忆"；以及最隐蔽的——把库提交进版本控制，私人对话内容
连同 embedding 一起进了远端。

因此顺序是：环境变量 ``AXO_DATA`` → 系统用户数据目录。仓库里的
``data/`` 只在开发时通过环境变量显式指过去，且已在 .gitignore 里。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "consolidation_dir",
    "consolidation_events_path",
    "consolidation_last_path",
    "data_dir",
    "graph_path",
    "memory_db_path",
    "reminders_fired_path",
    "reminders_path",
    "sessions_dir",
    "snapshot_dir",
    "wallpapers_dir",
]

_ENV_OVERRIDE = "AXO_DATA"
_APP_DIR_NAME = "axo"


def data_dir() -> Path:
    """记忆终端的数据根目录，不存在则创建。"""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        root = Path(override).expanduser()
    else:
        # Windows: %LOCALAPPDATA%（不进漫游配置——记忆库可能有几百 MB，跟着
        # 域账号漫游会把登录拖垮）；其它平台按 XDG 惯例。
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            root = Path(local_appdata) / _APP_DIR_NAME
        else:
            root = (
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
                / _APP_DIR_NAME
            )
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
