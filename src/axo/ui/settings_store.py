"""设置的读写：**不 import Qt。**

``QSettings`` 换成这个薄封装，图的是可测：界面里最难写的测试就是"设了再读回来
是不是同一个值"，而 ``QSettings`` 在测试里会写到真实注册表、还会在用例之间互相
串。这里只要一个 ``value``/``setValue`` 两方法的鸭子类型（``SettingsBackend``），
测试传个 dict 就够了，本机 ``QSettings`` 由 :mod:`.theme_qt` 适配。

**三条纪律，都是被踩过的坑：**

1. **读到的值一律当作不可信输入。** 注册表可以被手改、被上一个版本写坏、被别的
   程序占用。所以 :meth:`SettingsStore.load` 逐字段兜底、**绝不抛**——`Theme` 的
   构造函数是会抛的（非法色号当场炸），而"设置文件坏了导致程序打不开"是最糟的
   一种失败：用户没有任何办法从界面里修好它。
2. **逐字段兜底，不是整体兜底。** 一个坏色号不该把壁纸选择也一起丢掉。所以是每个
   字段各自回落，而不是"try: 构造 Theme except: 返回默认"。
3. **类型会丢。** ``QSettings`` 在 Windows 上走注册表，读回来的 bool 常常是字符串
   ``"true"``/``"false"``（见 :func:`_as_bool`）。这条 ``memory_window.py`` 里已经
   记过一次了，这里再钉一遍——存浮点数就会在某台机器上变成 ``"0.18"``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from ..paths import wallpapers_dir
from ..secrets import protect, unprotect
from .theme import (
    CUSTOM_PRESET,
    DEFAULT_THEME,
    PRESETS,
    Theme,
    clamp_mode,
    clamp_wallpaper_visible,
    preset_by_key,
)

__all__ = [
    "BackendSettings",
    "MemorySettings",
    "ReminderSettings",
    "SETTINGS_APP",
    "SETTINGS_ORG",
    "SettingsBackend",
    "SettingsStore",
    "ToolSettings",
    "wallpaper_file",
]

_DEFAULT_PRESET = PRESETS[0]

#: 键前缀。与 ``main_window`` 直接用的 ``backend/host`` 同一套命名——
#: 两类设置共用一个 ``QSettings``，只是**键的空间分开**：``appearance/*`` 归这里，
#: ``backend/*`` 归 :class:`BackendSettings` 与 ``main_window`` 的 ``last_model``，
#: ``tools/*`` 归 :class:`ToolSettings`，``memory/*`` 归 :class:`MemorySettings`，
#: ``reminders/*`` 归 :class:`ReminderSettings`，
#: ``app/*`` 归"这台机器"（见 :data:`_K_DATA_ROOT`）。
#: 不加前缀混着放的话，"哪些键属于外观"就只能靠 grep 回答。
_NS = "appearance"
_K_PRESET = f"{_NS}/preset"
_K_ACCENT = f"{_NS}/accent"
_K_MODE = f"{_NS}/mode"
_K_WALLPAPER = f"{_NS}/wallpaper"
_K_VISIBLE = f"{_NS}/wallpaper_visible"

_K_BACKEND_HOST = "backend/host"
_K_ALLOW_REMOTE = "backend/allow_remote"
_K_PROVIDER = "backend/provider"
_K_CLOUD_BASE_URL = "backend/cloud_base_url"
#: **盘上存的是密文**（Windows DPAPI 的 base64），不是 key 本身。见
#: :mod:`axo.secrets` 与下面 :meth:`SettingsStore.load_backend`。
_K_API_KEY = "backend/api_key"

_K_READ_ROOTS = "tools/read_roots"

_K_TASK_MEMORY = "memory/task_memory"
_K_GRAPH_CHANNEL = "memory/graph_channel"
_K_CONSOLIDATION_ENABLED = "memory/consolidation_enabled"
_K_CONSOLIDATION_INTERVAL_S = "memory/consolidation_interval_s"

#: 键名与字段名**逐字对应**（``memory/task_memory`` 那一套）：读的人不必在脑子里做一次
#: 映射，而测试那边的"两个键各跑一遍同一张表"也能靠 ``key.split("/")[-1]`` 拿到字段。
_K_WAKE_ON_ALERT = "reminders/wake_on_alert"
_K_SPEAK_ON_ALERT = "reminders/speak_on_alert"
_K_AUTOSTART = "reminders/autostart"

#: **这台机器**级别的两个键。与上面四族分开，因为它们不是"某一块功能怎么跑"，
#: 而是"程序自己住哪、用户看过说明没有"。
#:
#: ``app/data_root`` 是**记忆与数据放在哪个目录**（绝对路径字符串；空串 = 没设过，
#: 用默认）。它是"删掉程序重新解压一份新的，记忆还在"的全部机制——注册表在
#: ``HKCU`` 下，与程序目录毫无关系。
_K_DATA_ROOT = "app/data_root"
_K_ONBOARDING_DONE = "app/onboarding_done"

#: ``QSettings`` 的组织名与应用名。
#:
#: **从这里公开出去**（原来是 ``ui/main_window.py`` 里的私有常量）：现在有两个
#: 地方要建 ``QSettings``——``ui/app.py`` 在**建主窗口之前**读一次位置指针，而
#: ``MainWindow`` 建那个真正长期使用的。两处各写一份字符串的话，改了其中一处就会
#: 变成"启动时读的是 A，运行时读写的是 B"，症状是"设置改了不生效"。
SETTINGS_ORG = "Axo"
SETTINGS_APP = "axo"

#: 与 :data:`~axo.llm.ollama.DEFAULT_OLLAMA_HOST` 同值。
#: **刻意在这里再写一遍而不是 import 过来**：本模块是纯数据层，不该把
#: ``llm`` 包（连带 httpx）拉进 import 图；而这条值一变，那边的默认值也就没意义了。
#: ``test_settings_store.py`` 断言两者一致。
DEFAULT_HOST = "http://127.0.0.1:11434"

#: 与 :data:`~axo.llm.openai_compat.DEFAULT_OPENAI_BASE_URL` 同值，
#: 理由与 :data:`DEFAULT_HOST` 完全一样（纯层不 import 那个包，所以写两遍，
#: 由 ``test_settings_store.py`` 断言一致）。
DEFAULT_CLOUD_BASE_URL = "https://api.deepseek.com/v1"

#: 后端类型。**是"这一类后端"而不是"哪一个厂牌"**：``cloud`` 底下换 DeepSeek、
#: OpenRouter 还是自建的 vLLM，都还是 ``cloud``。
PROVIDER_OLLAMA = "ollama"
PROVIDER_CLOUD = "cloud"
_PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_CLOUD)


class SettingsBackend(Protocol):
    """``QSettings`` 用得到的那两个方法。**故意只要两个。**"""

    def value(self, key: str, default: object = ...) -> object: ...

    def setValue(self, key: str, value: object) -> None: ...  # noqa: N802 (Qt 命名)


@dataclass(frozen=True)
class BackendSettings:
    """后端设置。

    **两类后端共用这一份，而不是两个 dataclass。** 换后端是一个开关，不是两套配置
    并存——两个 dataclass 会让"当前用的是哪一类"这个问题有两个答案（哪一个非空？）。

    ``host`` 是空串时用 :data:`DEFAULT_HOST`。

    Attributes:
        provider: :data:`PROVIDER_OLLAMA` 或 :data:`PROVIDER_CLOUD`。读到别的值一律
            回落本机 Ollama：那意味着设置是在更新的版本里写的，而"以为在用云端、
            实际在连本机"比"以为在用云端、实际报了错"危险得多。
        host: Ollama 服务地址。**只在 provider 是 ollama 时有意义。**
        allow_remote: 是否允许连非本机地址。

            **这是 :func:`~axo.llm.ollama._client` 那道护栏的开关**：不勾它，
            非本机 host 会直接抛 ``ChatBackendError``。所以设置窗口里放了 host 输入框就
            必须同时放这个勾选框——否则那是个"填了就一定报错"的字段。
        cloud_base_url: 云端 OpenAI 兼容端点的根，如 ``https://api.deepseek.com/v1``。
            **不与 ``host`` 复用同一个键**：``host`` 被"只连本机"那道护栏盯着、界面上
            也写着"Ollama 地址"，复用会让"切回 Ollama 时输入框里躺着 DeepSeek 的 URL"
            变成常态。
        api_key: **明文。** 盘上的是 DPAPI 密文（见 :mod:`axo.secrets`），
            加解密只在 :class:`SettingsStore` 里发生一次——两处解密点迟早会漂。
    """

    provider: str = PROVIDER_OLLAMA
    host: str = DEFAULT_HOST
    allow_remote: bool = False
    cloud_base_url: str = DEFAULT_CLOUD_BASE_URL
    api_key: str = ""

    @property
    def is_cloud(self) -> bool:
        """界面与主窗口问"现在是云端吗"的唯一入口——别在各处比字符串。"""
        return self.provider == PROVIDER_CLOUD


@dataclass(frozen=True)
class ToolSettings:
    """工具设置。P3a 只有一项。

    Attributes:
        read_roots: ``read_file`` 允许读的目录。**空元组 = 那个工具被禁用**——
            这不是"给个默认目录"，而是"用户没授权过任何地方"。默认 cwd 或 home
            读起来像体贴，实际是替用户做了一个他从没做过的授权决定
            （见 :mod:`~axo.agent.fs`）。

            存的是**绝对路径字符串**（用户是拿文件对话框挑的），读回来时才变成
            ``Path``。
    """

    read_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class MemorySettings:
    """记忆设置。

    Attributes:
        task_memory: 用不用 :class:`~axo.memory.policy.TaskPolicy`
            ——调过工具的那一轮合成一条，正文是回答。

            **默认关。** 与工具设置里"模型自分类默认关"同一姿态：这是新的入库
            *语义*，不是新的开关灯，先不替用户做主。关着的时候写入路径与 P1 逐字段
            相同（:class:`~axo.memory.policy.PerItemPolicy`），所以这个
            字段取假不是"少一个功能"，是"和上一个版本一样"。

            **开关是加法**：打开之后没调工具的轮次照旧逐条记——判据只有"这一轮有没有
            工具轨迹"这一条（见 :class:`TaskPolicy`）。
        graph_channel: 召回用不用联想通道（沿关联图扩散一步）。

            **默认关，而且这一版建议就让它关着。** 与上面那一项不同，这一条不是
            "先不替用户做主"，是**量出来的**：金标准语料上一档扫描里四项指标没有
            一项变好，AUC 从 0.9093 掉到 0.8450（``benchmarks/gold/CORPUS.md`` 的
            「联想腿」那页有两套语料上互相印证的负结果）。开关留着是因为"在真实库上
            是不是也这样"还没有答案——那要用户自己的数据。

            **它只改召回，不改图的建立。** 图建不建、什么时候重建，是另一个问题
            （见「设置 → 记忆」里的「重建联想」那一行）。
        consolidation_enabled: 跑不跑**睡眠巩固**（后台定时把已有的记忆重新认一遍，
            认出来的那些 ``reinforce_count + 1``，改库之前拍一份整库快照）。

            **默认开，这是本仓唯一一个默认开的"会自己改库"的开关。** 与上面两项
            同一姿态本来该是"先不替用户做主"，但这一项不同：巩固的全部产出是一个
            只增不减的计数（``reinforce_count``，记忆详情里显示"巩固 N 次"），它
            **不改变任何一条记忆的内容、不参与召回打分**——所以"开着"最坏是白花
            一点 cpu，而"关着"意味着这个功能在用户从没听说过它的情况下永远不运行。
            真正的风险（无人值守改库）由快照与逐簇事务兜着，见
            ``memory/consolidation.py`` 的模块文档。

            关掉它**不关门动那条路**：托盘与设置页的「立即巩固」照跑——那一下是
            用户当场要的，与"要不要每天自动跑一次"是两件事。
        consolidation_interval_s: 两次自动巩固之间隔多久（秒）。**86400（一天）
            是占位值，没有实测依据**——周期该多长取决于真实使用节奏下一轮要跑多久、
            快照攒多快，那要用户自己的库跑上一阵才有数。所以设置页把它摆在明面上，
            而不是写死在这里。下限 60 秒（``consolidation_scheduler.MIN_DELAY_MS``
            同值），再往下没有意义：一轮本身可能跑几分钟。
    """

    task_memory: bool = False
    graph_channel: bool = False
    consolidation_enabled: bool = True
    consolidation_interval_s: int = 86_400


@dataclass(frozen=True)
class ReminderSettings:
    """提醒设置：**到点那一刻要不要打扰他**，以及**程序没开着的时候怎么办**。

    前两项管的是"提醒响的时候发生什么"，**不改变提醒本身**：关掉任意一个，那条提醒
    照旧被标成响过、托盘照旧弹气泡、对话区照旧留一条——变的只是"是谁说的、有没有把
    他叫过来"。第三项管的是另一种时候：**程序根本没开着**。

    Attributes:
        wake_on_alert: 到点时把窗口叫出来。分寸在 ``main_window._wake_up`` 里：**窗口
            正开着时只闪任务栏、绝不抢焦点**；最小化时恢复；藏在托盘里才真的叫出来。

            **默认开**，与记忆那两组"先不替用户做主"的姿态不同：到点了不出声，这个
            功能就等于不存在。而"叫出来"最坏的结果是把窗口从别的窗口底下翻上来一次，
            用户一眼就能看出是程序干的（它带着那条提醒）。
        speak_on_alert: 到点时由模型生成一句话说出来。

            **默认开，但这是三个"会自己动"的开关里唯一一个该有逃生口的**：它要一次
            本机模型推理——实测 3–15 秒（见 ``main_window.SPEECH_MAX_TOKENS``），模型
            慢、或者说话声吵的时候，用户会想关掉它。**关掉之后提醒照响**：托盘气泡与
            对话区那条系统消息一个字都不变（见 ``main_window._on_reminder_fired`` 里
            那条降级路径）——那正是这个功能可以放心默认开着的理由。
        autostart: 开机时自动运行一次无头入口（``--remind``），把机器关着的那段时间里
            到点的提醒补上。

            **默认关，而且是这三个里唯一一个默认关的。** 它不是"程序内部的一个行为"，
            是**在用户的机器上装一个东西**：往 ``HKCU\\...\\Run`` 写一个值。这种事只能由
            用户显式要求——与工具页那条"可读目录空着 = 没授权过任何地方"同一个姿态：
            不替他做一个他从没做过的授权决定。

            它同时是**唯一一项在源码检出下做不到的设置**（``sys.executable`` 是
            python.exe，写进去等于让登录时开一个没有脚本的 Python），所以设置页那个
            勾选框会按 ``autostart.unavailable_reason()`` 灰掉，而不是勾上之后没反应。
    """

    wake_on_alert: bool = True
    speak_on_alert: bool = True
    autostart: bool = False


class SettingsStore:
    """把 :class:`~axo.ui.theme.Theme` 与四组设置
    （:class:`BackendSettings` / :class:`ToolSettings` / :class:`MemorySettings` /
    :class:`ReminderSettings`）在 ``SettingsBackend`` 上读写成扁平键。

    每一组一对 ``load_*`` / ``save_*``，键前缀见上面那段。**不合成一个大
    ``load()``**：设置窗口按页保存，而"改外观时把工具设置也重写一遍"会让一次失败
    的写入波及它不该碰的键。
    """

    def __init__(self, backend: SettingsBackend) -> None:
        self._backend = backend

    # ── 外观 ────────────────────────────────────────────────────────────────

    def load(self) -> Theme:
        """读出主题。**任何损坏的字段各自回落，绝不抛。**"""
        preset = self._text(_K_PRESET).strip() or _DEFAULT_PRESET.key
        if preset != CUSTOM_PRESET and preset_by_key(preset) is None:
            # 未知预设：可能是降级回旧版本之后读到了新版本的键。回落默认，
            # 而不是抛——用户没有任何办法从界面里改掉一个读不出来的值。
            preset = _DEFAULT_PRESET.key

        accent = self._text(_K_ACCENT).strip()
        try:
            # 借 Theme 自己的校验来判合法：与其在这里重写一遍十六进制解析，
            # 不如让它抛一次。**选了预设时这一项其实用不上**（预设说了算），
            # 但自定义色就是靠这一句兜住的。空串也会在这里被拦下
            # （``parse_hex("")`` 抛），所以不需要另判一次空。
            Theme(preset=CUSTOM_PRESET, accent=accent)
        except ValueError:
            accent = _DEFAULT_PRESET.accent

        return Theme(
            preset=preset,
            accent=accent,
            mode=clamp_mode(self._text(_K_MODE).strip()),
            wallpaper=self._text(_K_WALLPAPER).strip(),
            wallpaper_visible=clamp_wallpaper_visible(
                self._backend.value(_K_VISIBLE, DEFAULT_THEME.wallpaper_visible)
            ),
        )

    def save(self, theme: Theme) -> None:
        """写回。**可见度写整数百分比**，不写浮点——见模块注释第 3 条。"""
        self._backend.setValue(_K_PRESET, theme.preset)
        self._backend.setValue(_K_ACCENT, theme.accent)
        self._backend.setValue(_K_MODE, theme.mode)
        self._backend.setValue(_K_WALLPAPER, theme.wallpaper)
        self._backend.setValue(_K_VISIBLE, int(theme.wallpaper_visible))

    # ── 后端 ────────────────────────────────────────────────────────────────

    def load_backend(self) -> BackendSettings:
        """读出后端设置。

        ``mode`` 那条的兜底交给 :meth:`Theme` 的校验，这里只做三件事：空串换成默认
        host/地址、把可能读回成字符串的 bool 收成真 bool、**把密文解回明文 key**。

        **解密失败一律得到空串，不抛**（换机器、换 Windows 账户、密文被手改都是这一路）。
        那是刻意的降级：正确行为是"当作没配"，让用户重填一个——而 ``load_backend`` 在
        启动路径上，这里抛一次程序就打不开了，用户连重填的界面都进不去。
        """
        provider = self._text(_K_PROVIDER).strip()
        return BackendSettings(
            provider=provider if provider in _PROVIDERS else PROVIDER_OLLAMA,
            host=self._text(_K_BACKEND_HOST).strip() or DEFAULT_HOST,
            allow_remote=_as_bool(self._backend.value(_K_ALLOW_REMOTE, False)),
            cloud_base_url=self._text(_K_CLOUD_BASE_URL).strip() or DEFAULT_CLOUD_BASE_URL,
            api_key=unprotect(self._text(_K_API_KEY)),
        )

    def save_backend(self, settings: BackendSettings) -> None:
        self._backend.setValue(_K_PROVIDER, settings.provider)
        self._backend.setValue(_K_BACKEND_HOST, settings.host)
        # **存真 bool**，读的时候仍然按 :func:`_as_bool` 收一遍：Qt 在 Windows 上
        # 存得住 bool，但别的平台/别的版本不一定，读取侧不该依赖写入侧的类型。
        self._backend.setValue(_K_ALLOW_REMOTE, bool(settings.allow_remote))
        self._backend.setValue(_K_CLOUD_BASE_URL, settings.cloud_base_url)
        # **key 加密后再落盘。** 空串原样存空串（``protect`` 自己会短路），于是
        # "没配 key"与"配了又清掉"在盘上是同一个状态——这正是我们想要的。
        self._backend.setValue(_K_API_KEY, protect(settings.api_key))

    # ── 工具 ────────────────────────────────────────────────────────────────

    def load_tools(self) -> ToolSettings:
        """读出工具设置。

        **只收绝对路径。** 相对路径的含义取决于**进程的工作目录**，而 GUI 程序的
        工作目录不是用户以为的那个（双击图标启动时它是快捷方式的"起始位置"）。
        一个存成 ``notes`` 的可读目录会在某次启动后指向别的地方——而那是一次
        **静默的授权变更**。所以这类条目直接丢掉，与 :func:`wallpaper_file` 把
        ``../secret.jpg`` 削成 basename 是同一种防御：设置可以被手改，读的一侧
        不能假设它一定是自己写的那种形状。

        （正常路径上不会出现相对路径：唯一的写入口是设置窗口的目录选择框，它给的
        永远是绝对路径。）

        "绝对"判的是**任一平台的口味**，不是当前这台机器的口味——理由见
        :func:`_is_absolute_anywhere`：条目是字符串，它可能是在另一台机器上写的。
        """
        return ToolSettings(read_roots=_as_path_list(self._backend.value(_K_READ_ROOTS, [])))

    def save_tools(self, settings: ToolSettings) -> None:
        """写回。**写字符串列表**——``QSettings`` 存得住列表，但读回来的形状要看
        平台（见 :func:`_as_path_list`），所以读取侧不依赖写入侧。"""
        self._backend.setValue(_K_READ_ROOTS, [str(root) for root in settings.read_roots])

    # ── 记忆 ────────────────────────────────────────────────────────────────

    def load_memory(self) -> MemorySettings:
        """读出记忆设置。bool 走 :func:`_as_bool`——注册表里读回来的 ``"false"``
        是真值，一个勾选框会自己勾上，而症状是"入库的粒度悄悄换了"。

        **巩固那个开关的默认值是 ``True``，而这里必须写成 ``, True``。** 三处默认
        值（dataclass 的字段默认、这里、设置页的勾选框）少写一处，症状是"新装好的
        程序里那个框没勾上"——而巩固从此一次都不跑，界面上看不出来。
        """
        return MemorySettings(
            task_memory=_as_bool(self._backend.value(_K_TASK_MEMORY, False)),
            graph_channel=_as_bool(self._backend.value(_K_GRAPH_CHANNEL, False)),
            consolidation_enabled=_as_bool(
                self._backend.value(_K_CONSOLIDATION_ENABLED, True), True
            ),
            consolidation_interval_s=_as_int(
                self._backend.value(_K_CONSOLIDATION_INTERVAL_S, 86_400), 86_400
            ),
        )

    def save_memory(self, settings: MemorySettings) -> None:
        """写回。存真 bool，读取侧仍然按 :func:`_as_bool` 收一遍（同后端那一节）。"""
        self._backend.setValue(_K_TASK_MEMORY, bool(settings.task_memory))
        self._backend.setValue(_K_GRAPH_CHANNEL, bool(settings.graph_channel))
        self._backend.setValue(_K_CONSOLIDATION_ENABLED, bool(settings.consolidation_enabled))
        self._backend.setValue(_K_CONSOLIDATION_INTERVAL_S, int(settings.consolidation_interval_s))

    # ── 提醒 ────────────────────────────────────────────────────────────────

    def load_reminders(self) -> ReminderSettings:
        """读出提醒设置。

        **两个默认值都是 ``True``，而这里必须写成 ``, True``。** 三处默认值（dataclass
        的字段默认、这里、设置页那两个勾选框的初值）少写一处，症状是"新装好的程序里
        那个框没勾上"——而"到点由她开口"从此一次都不跑，界面上一点异常都看不出来
        （同 :meth:`load_memory` 里那句，那一条是被真实踩过才写下来的）。
        """
        return ReminderSettings(
            wake_on_alert=_as_bool(self._backend.value(_K_WAKE_ON_ALERT, True), True),
            speak_on_alert=_as_bool(self._backend.value(_K_SPEAK_ON_ALERT, True), True),
            # **这一位的默认是 ``False``**，所以这里那个 ``, False`` 正是"三处默认值"
            # 里的第二处——写成 ``, True`` 的话，新装好的程序会在**用户没要求过**的
            # 情况下往注册表里写自启（见 :attr:`ReminderSettings.autostart`）。
            autostart=_as_bool(self._backend.value(_K_AUTOSTART, False), False),
        )

    def save_reminders(self, settings: ReminderSettings) -> None:
        """写回。存真 bool，读取侧仍然按 :func:`_as_bool` 收一遍（同记忆那一节）。"""
        self._backend.setValue(_K_WAKE_ON_ALERT, bool(settings.wake_on_alert))
        self._backend.setValue(_K_SPEAK_ON_ALERT, bool(settings.speak_on_alert))
        self._backend.setValue(_K_AUTOSTART, bool(settings.autostart))

    # ── 这台机器（app/*）────────────────────────────────────────────────────

    def stored_data_root(self) -> str:
        """用户在界面里选的记忆/数据目录；没设过或值不可用就是空串。

        **只收绝对路径**，与 :meth:`load_tools` 同一条理由、同一套判据
        （:func:`_is_absolute_anywhere`）：这个值决定的是"程序去哪儿读写用户的
        全部记忆"，而相对路径的含义取决于进程的工作目录——双击图标启动时它是
        快捷方式的"起始位置"。一次静默的位置变更比"没设"危险得多。

        **坏值当没设过，不抛。** 注册表可以被手改，而这里抛一次程序就打不开了，
        用户连改回来的界面都进不去（同 :meth:`load_backend` 对解密失败的处置）。
        """
        raw = self._text(_K_DATA_ROOT).strip()
        if not raw:
            return ""
        # ``expanduser`` 在判绝对**之前**，且用返回字符串的那个版本——理由与
        # :func:`_as_path_list` 里那段一字不差（``~/Axo` 是用户能写、且有意义的一种）。
        expanded = os.path.expanduser(raw)
        return expanded if _is_absolute_anywhere(expanded) else ""

    def save_data_root(self, root: str | Path | None) -> None:
        """写下位置。``None`` 与空串都存成空串——"恢复默认位置"就是把这一格清掉，
        而不是把默认路径抄一份进来：抄进来的话，将来默认值变了，那一格会继续指着
        旧地方。"""
        self._backend.setValue(_K_DATA_ROOT, "" if root is None else str(root).strip())

    def onboarding_done(self) -> bool:
        """新手引导走完过没有（**「完成」与「跳过」都算**，叉掉窗口不算）。"""
        return _as_bool(self._backend.value(_K_ONBOARDING_DONE, False))

    def mark_onboarding_done(self, done: bool = True) -> None:
        self._backend.setValue(_K_ONBOARDING_DONE, bool(done))

    def _text(self, key: str) -> str:
        """取一个字符串键。非字符串一律当空——注册表里可能存着别的东西。"""
        value = self._backend.value(key, "")
        return value if isinstance(value, str) else ""


def _as_bool(value: object, default: bool = False) -> bool:
    """把读回来的东西收成 bool。

    ``QSettings`` 读一个写进去的 ``True``，在 Windows 注册表上可能是字符串
    ``"true"``。直接 ``bool(value)`` 的话 ``bool("false")`` 是 **True**——一个
    勾选框会自己勾上，而 :func:`_client` 的护栏就悄悄开了。这是本模块里最值得
    单独写一个函数的一件事。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: object, default: int) -> int:
    """把读回来的东西收成 int。**收不了就用 ``default``，绝不抛。**

    与 :func:`_as_bool` 同一处病因：``QSettings`` 在 Windows 上读一个写进去的整数，
    拿回来的可能是字符串 ``"86400"``；手改过注册表的人还能写进 ``"一天"``。让它抛的
    话症状是**主窗口构造失败**（这个值在启动路径上被读），而正确答案是拿默认值接着
    跑起来、由设置页把坏值显示出来。

    ``bool`` 要**先**挡掉：它是 ``int`` 的子类，``int(True)`` 是 1——一个被写坏的
    布尔会让周期变成 1 秒，而那个值会被调度器夹到下限，症状从"打不开"变成"偷偷跑得
    很密"，更难查。
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else default  # NaN 不能转
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _is_absolute_anywhere(text: str) -> bool:
    """这段**文字**在任一平台的口味下算绝对路径吗。

    **不能用 ``Path.is_absolute()``**：它的判据是当前平台的口味，而这里判的是一个
    会跟着配置文件走的**字符串**。用户在文件对话框里挑的目录写进 ``QSettings`` 之后，
    形状是**写下它的那台机器**的形状——Windows 上写的 ``C:/notes`` 在 POSIX 口味下
    只是一个叫 ``C:`` 的普通目录，于是被判成相对路径而丢掉；反过来 ``/home/notes``
    在 Windows 口味下没有盘符（``PureWindowsPath("/home/notes").is_absolute()`` 是假）。
    症状是**可读目录安静地少了一条**，而原因（判据是平台口味的）在界面上完全看不出来。

    这条失效是 CI 的 Linux 腿抓出来的：2026-10-10 五个断言同时红，全是这一个原因。

    **参数是字符串，不是 ``Path``，这是故意的。** 中途写过一版收 ``Path`` 再用
    ``str(path)`` 判，那在 Windows 上是错的：``str(Path("/home/notes"))`` 会被规范化成
    ``\\home\\notes``——两种口味下都不是绝对路径，于是 POSIX 形状的条目在本机照样被丢
    掉，而 Linux 上看起来是对的。**归一化会把"写下时的形状"抹掉**，所以判必须判原文。

    代价是两种口味都收下：Linux 上的一份 ``C:/notes`` 会留在列表里。它不会越过沙箱
    ——真正决定"能不能读"的是 :func:`~axo.agent.fs.resolve_within`，它比的
    是解析后的真实路径，一个不存在的目录读不到任何东西。
    """
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def _as_path_list(value: object) -> tuple[Path, ...]:
    """把读回来的一格收成一个有序去重的绝对路径元组。**不抛。**

    这里要挡的是两种"读回来的形状和写的不是一回事"：

    1. **单元素列表可能读成字符串。** ``QSettings`` 上写一个只有一项的列表，某些
       后端读回来就是一个裸 ``str``。照着"列表"去迭代它，得到的是**逐个字符**——
       于是可读目录会变成一堆单字母目录，而那个症状（"读不了文件"）指不到这里。
    2. **元素可能不是字符串。** 手改过的注册表里可以是数字、可以是嵌套列表。

    非绝对路径的条目丢掉，理由见 :meth:`SettingsStore.load_tools`。
    """
    if isinstance(value, (str, Path)):
        raw: list[object] = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        return ()

    found: list[Path] = []
    seen: set[str] = set()
    for entry in raw:
        if isinstance(entry, (str, Path)):
            # ``expanduser`` 必须在判绝对**之前**，而且用的是 ``os.path`` 那个**返回
            # 字符串**的版本：``~/notes`` 相对的是家目录（一个稳定的东西，手改注册表
            # 的人写它是有意义的），展开之后它就是绝对路径；被丢掉的是真正取决于工作
            # 目录的那种（``notes``）。走 ``Path.expanduser()`` 会把分隔符一起归一化，
            # 于是"写下时的形状"没了——判绝对正需要那个形状，见
            # :func:`_is_absolute_anywhere`。
            expanded = os.path.expanduser(str(entry).strip())
        else:
            continue
        if not expanded:
            continue
        if not _is_absolute_anywhere(expanded):
            continue
        path = Path(expanded)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        found.append(path)
    return tuple(found)


def wallpaper_file(name: str) -> Path | None:
    """``Theme.wallpaper`` 里那个文件名对应的**实际存在的**路径；没有则 ``None``。

    存的是文件名（不是完整路径），理由见 :attr:`~axo.ui.theme.Theme.wallpaper`。
    这个函数是唯一把它变成路径的地方——路径不存在时返回 ``None`` 而不是抛，
    因为那是**常态**：用户把数据目录搬走了、图片被删了、U 盘拔了。

    **注意 load() 不会因为这个函数而清掉设置里的名字。** 名字留在设置里是有意的：
    文件过两天又回来了（重新插上盘、把目录搬回来）时，壁纸就自己恢复了。要不要
    说一句"壁纸没找到"是调用方的事——它才知道该往哪儿说。
    """
    if not name:
        return None
    # 只取 basename：设置里的名字若被手改成 "../../something"，这里也不放它出去。
    # 文件是设置窗口自己复制进去的，永远落在壁纸目录的第一层。
    candidate = wallpapers_dir() / Path(name).name
    return candidate if candidate.is_file() else None
