"""设置的读写。

**这里测的几乎全是"读到坏数据怎么办"。** 因为那是这一层唯一会出错的地方：正常
路径就是几行赋值，而注册表是可以被手改、被上一个版本写坏、被别的程序占用的。
"设置文件坏了导致程序打不开"是最糟的一种失败——用户没有任何办法从界面里修好它，
因为界面正是打不开的那个东西。

所以纪律是：**逐字段兜底、绝不抛**。逐字段而不是整体，是因为一个坏色号不该把
壁纸选择也一起丢掉。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from axo import paths
from axo.llm.ollama import DEFAULT_OLLAMA_HOST
from axo.ui.settings_store import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_HOST,
    PROVIDER_CLOUD,
    PROVIDER_OLLAMA,
    BackendSettings,
    MemorySettings,
    SettingsStore,
    ToolSettings,
    wallpaper_file,
)
from axo.ui.theme import (
    CUSTOM_PRESET,
    DEFAULT_THEME,
    PRESETS,
    WALLPAPER_VISIBLE_DEFAULT,
    WALLPAPER_VISIBLE_MAX,
    Theme,
)


class FakeBackend:
    """``SettingsBackend`` 的替身。一个 dict 就够了——这正是当初不直接依赖
    ``QSettings`` 的理由：真 ``QSettings`` 会写到真实注册表，用例之间互相串。"""

    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self.data: dict[str, object] = dict(initial or {})
        self.writes: list[tuple[str, object]] = []

    def value(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 (Qt 命名)
        self.data[key] = value
        self.writes.append((key, value))


# ── 往返 ────────────────────────────────────────────────────────────────────


def test_empty_backend_gives_the_default_theme() -> None:
    assert SettingsStore(FakeBackend()).load() == DEFAULT_THEME


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_roundtrip_each_preset(preset) -> None:
    """存进去的是**用户选的那一支**（暗色那支），不是浅色解析后的那一支。

    这条区分很要紧：设置里存的永远是"用户选了什么"，而"实际画出来的是什么"由
    :func:`~axo.ui.theme.resolve` 在渲染时决定。两者混起来之后，用户在
    浅色模式下点一下"确定"，主色就被永久换掉了——而且再也换不回来。
    """
    store = SettingsStore(FakeBackend())
    original = Theme(preset=preset.key, mode="light")
    store.save(original)
    loaded = store.load()
    assert loaded == original
    assert loaded.accent == preset.accent


def test_roundtrip_full_theme() -> None:
    backend = FakeBackend()
    store = SettingsStore(backend)
    original = Theme(
        preset=CUSTOM_PRESET,
        accent="#12ab34",
        mode="light",
        wallpaper="cats.jpg",
        wallpaper_visible=42,
    )
    store.save(original)
    assert store.load() == original


def test_visible_is_stored_as_an_integer() -> None:
    """**存整数百分比，不存浮点。** ``QSettings`` 在 Windows 上走注册表，读回来的
    类型会丢——存 float 就会在某台机器上变成字符串 ``"0.18"``。"""
    backend = FakeBackend()
    SettingsStore(backend).save(Theme(wallpaper_visible=42))
    written = dict(backend.writes)
    assert written["appearance/wallpaper_visible"] == 42
    assert isinstance(written["appearance/wallpaper_visible"], int)


# ── 损坏的值 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["不是颜色", "#12345", "", "rgb(1,2,3)", "#ff8800ff"])
def test_broken_accent_falls_back(bad: str) -> None:
    store = SettingsStore(
        FakeBackend({"appearance/preset": CUSTOM_PRESET, "appearance/accent": bad})
    )
    assert store.load().accent == DEFAULT_THEME.accent


def test_broken_accent_does_not_take_the_other_fields_with_it() -> None:
    """**逐字段兜底的意义在这里。** 一个坏色号不该把用户挑的壁纸也一起丢掉。"""
    store = SettingsStore(
        FakeBackend(
            {
                "appearance/preset": CUSTOM_PRESET,
                "appearance/accent": "坏了",
                "appearance/mode": "light",
                "appearance/wallpaper": "cats.jpg",
                "appearance/wallpaper_visible": 55,
            }
        )
    )
    theme = store.load()
    assert theme.accent == DEFAULT_THEME.accent
    assert theme.mode == "light"
    assert theme.wallpaper == "cats.jpg"
    assert theme.wallpaper_visible == 55


@pytest.mark.parametrize("bad", ["没有这个预设", "Indigo", "", 7, None])
def test_unknown_preset_falls_back(bad: object) -> None:
    """未知预设可能是"降级回旧版本之后读到了新版本的键"。回落默认，
    而不是抛——用户没有任何办法从界面里改掉一个读不出来的值。"""
    store = SettingsStore(FakeBackend({"appearance/preset": bad}))
    assert store.load().preset == DEFAULT_THEME.preset


@pytest.mark.parametrize("bad", ["purple", "", 3, None])
def test_unknown_mode_falls_back_to_system(bad: object) -> None:
    store = SettingsStore(FakeBackend({"appearance/mode": bad}))
    assert store.load().mode == DEFAULT_THEME.mode


@pytest.mark.parametrize(
    "stored,expected",
    [
        (-5, 0),
        (WALLPAPER_VISIBLE_MAX + 1, WALLPAPER_VISIBLE_MAX),
        ("不是数字", WALLPAPER_VISIBLE_DEFAULT),
        (None, WALLPAPER_VISIBLE_DEFAULT),
        (33, 33),
    ],
)
def test_out_of_range_visible_is_clamped(stored: object, expected: int) -> None:
    store = SettingsStore(FakeBackend({"appearance/wallpaper_visible": stored}))
    assert store.load().wallpaper_visible == expected


@pytest.mark.parametrize("stored", [7, None, ["a"], {"k": 1}])
def test_non_string_wallpaper_becomes_empty(stored: object) -> None:
    assert SettingsStore(FakeBackend({"appearance/wallpaper": stored})).load().wallpaper == ""


def test_load_never_raises_on_a_fully_corrupt_store() -> None:
    """把每个键都填上垃圾，``load()`` 仍要给出一份可用的主题。

    壁纸那一格给的是 **7（非字符串）** 而不是一个像文件名的字符串——
    ``"??"`` 是合法字符串，会被原样留着（要不要判它存不存在是使用侧的事，
    见 :func:`wallpaper_file`），那样这个"全部回落默认"的断言就不成立了。
    """
    store = SettingsStore(
        FakeBackend(
            {
                "appearance/preset": "??",
                "appearance/accent": "??",
                "appearance/mode": "??",
                "appearance/wallpaper": 7,
                "appearance/wallpaper_visible": "??",
            }
        )
    )
    assert store.load() == DEFAULT_THEME


# ── 后端设置 ────────────────────────────────────────────────────────────────


def test_default_host_matches_the_ollama_module() -> None:
    """本模块刻意不 import ``llm``（那是纯数据层，不该把 httpx 拉进 import 图），
    所以默认值写了两遍。这一条就是那句"写两遍"的执法者。"""
    assert DEFAULT_HOST == DEFAULT_OLLAMA_HOST


def test_default_cloud_url_matches_the_openai_compat_module() -> None:
    """与上一条同一个理由、同一个形状：默认地址写了两遍，这一条盯着它们别漂。

    漂了的表现是"界面预填的地址与实际请求的地址不是同一个"——用户按预填的填完，
    点测试连接是通的（界面拿的那份），一说话就 404（后端拿的那份）。
    """
    from axo.llm.openai_compat import DEFAULT_OPENAI_BASE_URL

    assert DEFAULT_CLOUD_BASE_URL == DEFAULT_OPENAI_BASE_URL


def test_empty_host_falls_back_to_default() -> None:
    store = SettingsStore(FakeBackend({"backend/host": "   "}))
    assert store.load_backend().host == DEFAULT_HOST


def test_backend_roundtrip() -> None:
    store = SettingsStore(FakeBackend())
    store.save_backend(BackendSettings(host="http://192.168.1.9:11434", allow_remote=True))
    loaded = store.load_backend()
    assert loaded.host == "http://192.168.1.9:11434"
    assert loaded.allow_remote is True


def test_allow_remote_defaults_to_false() -> None:
    """**默认必须是关的。** 这是 ``_client`` 那道护栏的开关：不勾它，非本机地址会
    直接抛错，而不是把内容发出去。"""
    assert SettingsStore(FakeBackend()).load_backend().allow_remote is False


@pytest.mark.parametrize(
    "stored,expected",
    [
        (True, True),
        (False, False),
        # Windows 注册表把写进去的 bool 读回来常常是字符串。**这是本文件里最要紧
        # 的一组用例**：直接 bool(value) 的话 bool("false") 是 True，一个勾选框会
        # 自己勾上，护栏就悄悄开了。
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("随便什么", False),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_allow_remote_survives_the_type_trap(stored: object, expected: bool) -> None:
    store = SettingsStore(FakeBackend({"backend/allow_remote": stored}))
    assert store.load_backend().allow_remote is expected


# ── 壁纸文件 ────────────────────────────────────────────────────────────────


@pytest.fixture
def wallpapers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把数据目录指到 tmp_path，免得往真实用户目录里写东西。"""
    monkeypatch.setenv("AXO_DATA", str(tmp_path))
    return paths.wallpapers_dir()


def test_empty_name_is_no_file() -> None:
    assert wallpaper_file("") is None


def test_missing_file_is_none(wallpapers: Path) -> None:
    """**这不是异常情况，是常态**：用户把数据目录搬走了、图片被删了、U 盘拔了。
    所以返回 None 而不是抛。"""
    assert wallpaper_file("没有这个.jpg") is None


def test_existing_file_resolves(wallpapers: Path) -> None:
    target = wallpapers / "cats.jpg"
    target.write_bytes(b"x")
    assert wallpaper_file("cats.jpg") == target


def test_traversal_is_neutralised(wallpapers: Path, tmp_path: Path) -> None:
    """设置里的名字若被手改成 ``../secret.jpg``，也不该把它指到壁纸目录之外。

    文件是设置窗口自己复制进去的，永远落在第一层；而**渲染层只读我们自己给的
    路径**，这是一层纵深防御。
    """
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"x")
    assert wallpaper_file("../secret.jpg") is None
    assert wallpaper_file("..\\secret.jpg") is None


def test_load_keeps_a_name_whose_file_is_gone(wallpapers: Path) -> None:
    """**``load()`` 不会因为文件不在就把名字清掉。** 名字留着是有意的：文件过两天
    又回来了（重新插上盘、把目录搬回来）时，壁纸就自己恢复了。要不要说一句
    "壁纸没找到"是调用方的事——它才知道该往哪儿说。
    """
    store = SettingsStore(FakeBackend({"appearance/wallpaper": "没有这个.jpg"}))
    assert store.load().wallpaper == "没有这个.jpg"
    assert wallpaper_file(store.load().wallpaper) is None


# ── 工具设置：可读目录 ──────────────────────────────────────────────────────


def test_no_read_roots_by_default() -> None:
    """**没配 = 工具被禁用**，不是"给个默认目录"。

    默认 cwd 或 home 读起来像体贴，实际是替用户做了一个他从没做过的授权决定。
    """
    assert SettingsStore(FakeBackend()).load_tools().read_roots == ()


def test_read_roots_roundtrip() -> None:
    store = SettingsStore(FakeBackend())
    original = ToolSettings(read_roots=(Path("C:/notes"), Path("D:/工作/资料")))
    store.save_tools(original)
    assert store.load_tools() == original


def test_read_roots_are_stored_as_strings() -> None:
    """写进去的是字符串，不是 ``Path``——``QSettings`` 存不下 ``Path``，它会写成一个
    读不回来的东西。"""
    backend = FakeBackend()
    SettingsStore(backend).save_tools(ToolSettings(read_roots=(Path("C:/notes"),)))
    written = dict(backend.writes)["tools/read_roots"]
    assert written == ["C:\\notes"] or written == ["C:/notes"]


def test_a_single_root_read_back_as_a_bare_string_still_works() -> None:
    """**单元素列表可能读成字符串**（``QSettings`` 各后端的经典差异）。

    照着"列表"去迭代一个裸字符串，得到的是**逐个字符**——可读目录会变成一堆单字母
    目录，而症状是"读不了文件"，指不到这里。
    """
    store = SettingsStore(FakeBackend({"tools/read_roots": "C:/notes"}))
    assert store.load_tools().read_roots == (Path("C:/notes"),)


@pytest.mark.parametrize("stored", [7, None, {"a": 1}, [["C:/notes"]]])
def test_broken_read_roots_fall_back_to_nothing(stored: object) -> None:
    """读不懂就是"没有可读目录"（= 工具禁用），**不抛**。

    fail-closed 的方向在这里和别处一致：读不出来的授权不算授权。
    """
    store = SettingsStore(FakeBackend({"tools/read_roots": stored}))
    assert store.load_tools().read_roots == ()


@pytest.mark.parametrize("stored", ["notes", "./notes", "../notes", "   ", ""])
def test_relative_read_roots_are_dropped(stored: str) -> None:
    """**相对路径丢掉**，不留给它一条靠工作目录解析的路。

    相对路径的含义取决于进程的工作目录，而 GUI 程序的工作目录不是用户以为的那个
    （双击图标启动时它是快捷方式的"起始位置"）。一个存成 ``notes`` 的可读目录会在
    某次启动后指向别的地方——那是一次**静默的授权变更**。

    与 :func:`wallpaper_file` 把 ``../secret.jpg`` 削成 basename 是同一种防御：
    设置可以被手改，读的一侧不能假设它一定是自己写的那种形状。
    """
    store = SettingsStore(FakeBackend({"tools/read_roots": [stored]}))
    assert store.load_tools().read_roots == ()


@pytest.mark.parametrize("stored", ["C:/notes", "D:/工作/资料", "/home/notes"])
def test_a_root_written_on_another_platform_is_not_dropped(stored: str) -> None:
    """两个平台的形状都收下——判据**不是**当前这台机器的口味。

    "绝对"在这里判的是一个会跟着配置文件走的**字符串**：``C:/notes`` 在 POSIX 口味
    下只是一个叫 ``C:`` 的普通目录（``Path.is_absolute()`` 为假），``/home/notes`` 在
    Windows 口味下没有盘符（同样为假）。用 ``Path.is_absolute()`` 的话，这两种形状
    各自在**另一半**平台上被安静地丢掉——可读目录少一条，而界面上完全看不出来。

    三个参数各会在一种平台上红，所以这条不依赖"跑它的那台机器是什么系统"：Windows
    上红的是 ``/home/notes``，Linux 上是前两个。（这个失效是 CI 的 Linux 腿先抓出来
    的——本机的判据恰好对，五个断言同时红才让人看见。）
    """
    store = SettingsStore(FakeBackend({"tools/read_roots": [stored]}))
    assert store.load_tools().read_roots == (Path(stored),)


def test_a_home_relative_root_is_expanded_not_dropped() -> None:
    """``~/notes`` **收下**——它相对的是家目录，不是工作目录。

    这条与上一条一起把规则说清楚了：被拒的是"取决于 cwd"，不是"看起来不像绝对
    路径"。展开必须发生在判绝对之前，否则这条会跟着一起被丢掉。
    """
    roots = SettingsStore(FakeBackend({"tools/read_roots": ["~/notes"]})).load_tools().read_roots
    assert len(roots) == 1
    assert roots[0].is_absolute()
    assert roots[0] == Path.home() / "notes"


def test_duplicate_read_roots_are_collapsed() -> None:
    """同一个目录加两次只留一条——列表里出现两遍是看得见的杂乱。"""
    store = SettingsStore(FakeBackend({"tools/read_roots": ["C:/notes", "C:/notes"]}))
    assert store.load_tools().read_roots == (Path("C:/notes"),)


def test_read_root_order_is_kept() -> None:
    """顺序要留着：``resolve_within`` 逐个根去试，而**第一个字面上匹配的根会遮蔽
    后面的**（见 ``agent/fs.py`` 里那段关于多根的说明）。所以"用户看到的顺序"必须
    就是"实际生效的顺序"。"""
    store = SettingsStore(FakeBackend({"tools/read_roots": ["D:/b", "C:/a"]}))
    assert store.load_tools().read_roots == (Path("D:/b"), Path("C:/a"))


def test_load_tools_never_raises() -> None:
    """同 ``load()``：设置坏了不该让程序打不开——界面正是打不开的那个东西。"""
    store = SettingsStore(FakeBackend({"tools/read_roots": [None, 3, "C:/ok"]}))
    assert store.load_tools().read_roots == (Path("C:/ok"),)


# ── 记忆设置：两个开关 ──────────────────────────────────────────────────────


def test_task_memory_is_off_by_default() -> None:
    """**默认必须是关的。**

    这是入库的**语义**，不是一盏灯：打开之后同一轮对话落进库的东西就换了形状。
    与工具设置里"模型自分类默认关"同一姿态——新的入库语义先不替用户做主。
    """
    assert SettingsStore(FakeBackend()).load_memory().task_memory is False


def test_task_memory_roundtrip() -> None:
    store = SettingsStore(FakeBackend())
    store.save_memory(MemorySettings(task_memory=True))
    assert store.load_memory().task_memory is True


def test_graph_channel_is_off_by_default() -> None:
    """联想的默认值**只能是关**，理由与上面那条不同。

    任务粒度那句话说的是"先不替用户做主"；而联想这一条是**量出来的**：金标准语料上
    AUC 从 0.9093 掉到 0.8450，四项指标没有一项变好。所以这个默认值不是姿态，是结论
    ——一个没量过的默认值在这里是会被"顺手打开试试"的。
    """
    assert SettingsStore(FakeBackend()).load_memory().graph_channel is False


def test_graph_channel_roundtrip() -> None:
    store = SettingsStore(FakeBackend())
    store.save_memory(MemorySettings(graph_channel=True))
    assert store.load_memory().graph_channel is True


@pytest.mark.parametrize(
    "task_memory,graph_channel",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_the_two_memory_switches_do_not_share_a_key(task_memory: bool, graph_channel: bool) -> None:
    """两个开关读写**各自的键**，四种组合都不许串。

    **只测"存 True 读回 True"是抓不到串键的**：读侧若把 ``_K_GRAPH_CHANNEL`` 写成
    ``_K_TASK_MEMORY``（复制粘贴一行、名字只差一半，正是最容易漏的那种），打开联想
    读回来的就是任务粒度那一格——两个默认值都是假，勾上都是真，于是"各自往返"四条
    全绿，而真实症状是**两个勾选框永远同进同出**。所以这里比的是四种组合的**整体
    形状**，而且必须让两格取值不同才构成判别。
    """
    store = SettingsStore(FakeBackend())
    store.save_memory(MemorySettings(task_memory=task_memory, graph_channel=graph_channel))
    loaded = store.load_memory()
    assert (loaded.task_memory, loaded.graph_channel) == (task_memory, graph_channel)


@pytest.mark.parametrize("key", ["memory/task_memory", "memory/graph_channel"])
@pytest.mark.parametrize(
    "stored,expected",
    [
        (True, True),
        (False, False),
        # 同一个坑，只是这次的后果不同：勾选框自己勾上，于是用户以为写进去的还是
        # 老样子，而实际粒度已经换了——**换了却没有一处看得出来**。
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("随便什么", False),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_the_memory_switches_survive_the_type_trap(
    key: str, stored: object, expected: bool
) -> None:
    """两个键各跑一遍同一张表。

    **参数化到两个键是刻意的**：这张表量的是 :func:`_as_bool`，而两个字段读的是同一个
    函数。分写成两个测试的话，将来谁给其中一个换成"自己写的收口"——比如图那格写成
    ``value is True``——只会红一半，而另一半（更常用的那个）照样绿。
    """
    store = SettingsStore(FakeBackend({key: stored}))
    loaded = store.load_memory()
    assert getattr(loaded, key.split("/")[-1]) is expected


def test_the_memory_keys_live_in_their_own_namespace() -> None:
    """**键的空间分开**，与前面三组一样。

    塞进 ``tools/*`` 的话，"哪些键属于工具"就只能靠 grep 回答——而这几项管的都不是
    哪个工具能用、也不是沙箱授权：一个管记忆落库的粒度，一个管召回用不用联想，后两个
    管睡眠巩固跑不跑、隔多久跑。

    写出去的键**逐个列出来**（不是"含哪些"）：多写一格就是多占一格命名空间，而
    ``save_memory`` 写的是整组设置——将来加字段时这条会红，提醒改的人想一下新键该
    落哪儿。
    """
    backend = FakeBackend()
    SettingsStore(backend).save_memory(MemorySettings(task_memory=True, graph_channel=True))
    assert [key for key, _ in backend.writes] == [
        "memory/task_memory",
        "memory/graph_channel",
        "memory/consolidation_enabled",
        "memory/consolidation_interval_s",
    ]


# ── 云端后端与密钥 ──────────────────────────────────────────────────────────
#
# 这一组里**只有"读回来还是原来那把 key"需要 Windows**：非 Windows 上 ``protect``
# 走的是"存不下"那一支（返回空串，绝不静默存明文），于是往返本来就该得到空串。
# 其余几条在两边都有意义，而且 Linux 上验的正是那条纪律——盘上永远不会出现明文。
_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="DPAPI 是 Windows 的系统调用")


def test_a_key_never_lands_on_disk_in_the_clear() -> None:
    """**这一条是这个功能的全部理由。**

    ``BackendSettings.api_key`` 是明文（内存里必须能用），但写进设置的那一格必须是
    密文。断言"不相等"不够——base64 一遍也不相等，而 base64 不是加密。所以还断言
    盘上那一格里**找不到原串**。
    """
    fake = FakeBackend()
    store = SettingsStore(fake)
    key = "sk-0123456789abcdef0123456789abcdef"

    store.save_backend(BackendSettings(provider=PROVIDER_CLOUD, api_key=key))

    stored = fake.data["backend/api_key"]
    assert isinstance(stored, str)
    assert stored != key
    assert key not in stored
    assert "sk-0123" not in stored


@_WINDOWS_ONLY
def test_a_key_comes_back_as_the_same_plaintext() -> None:
    """加密那一侧再怎么像密文，用例要的是**读回来还是那把 key**——否则设置窗口一开
    就显示一串乱码，用户会以为坏了。

    **这条只在 Windows 上有意义**（见上面那个 marker 的说明）：别处 ``protect``
    存不下东西，往返本来就该得到空串——那时该断言的是上一条。"""
    store = SettingsStore(FakeBackend())
    store.save_backend(BackendSettings(api_key="sk-abcdefabcdef"))

    assert store.load_backend().api_key == "sk-abcdefabcdef"


def test_an_empty_key_round_trips_as_empty() -> None:
    """没配 key 与清了 key 在盘上要是同一个状态，不然"清掉它"会留下一串能解的密文。"""
    fake = FakeBackend()
    store = SettingsStore(fake)
    store.save_backend(BackendSettings(api_key=""))

    assert fake.data["backend/api_key"] == ""
    assert store.load_backend().api_key == ""


def test_an_undecryptable_key_degrades_to_empty_and_does_not_raise() -> None:
    """**换机器、换 Windows 账户、注册表被手改都是这一路。**

    ``load_backend`` 在启动路径上：这里抛一次，程序就打不开了，而用户连重填 key 的
    界面都进不去。所以降级必须是"当作没配"。
    """
    store = SettingsStore(FakeBackend({"backend/api_key": "bm90LWEtY2lwaGVydGV4dA=="}))

    assert store.load_backend().api_key == ""


def test_an_unknown_provider_falls_back_to_local() -> None:
    """设置里写着一个我们不认识的后端类型（更新的版本写的、或者被手改）→ 回落本机。

    **方向要紧**：回落本机是"什么都没发出去"；若回落云端，用户以为在用本机模型，
    而对话内容其实已经出了这台机器。
    """
    for stored in ("deepseek", "CLOUD", "", "  ", "openai"):
        assert SettingsStore(FakeBackend({"backend/provider": stored})).load_backend().provider == (
            PROVIDER_OLLAMA
        )


def test_the_provider_round_trips() -> None:
    store = SettingsStore(FakeBackend())
    store.save_backend(BackendSettings(provider=PROVIDER_CLOUD, cloud_base_url="https://x/v1"))

    loaded = store.load_backend()
    assert loaded.provider == PROVIDER_CLOUD
    assert loaded.is_cloud is True
    assert loaded.cloud_base_url == "https://x/v1"
    assert loaded.api_key == ""


def test_an_empty_cloud_url_falls_back_to_default() -> None:
    store = SettingsStore(FakeBackend({"backend/cloud_base_url": "   "}))
    assert store.load_backend().cloud_base_url == DEFAULT_CLOUD_BASE_URL


def test_local_and_cloud_addresses_do_not_share_a_key() -> None:
    """**两格各管各的。**

    复用 ``host`` 的话，切到云端会把 Ollama 地址覆盖成 ``https://api.deepseek.com/v1``，
    切回来时输入框里躺着别人的 URL——而那个字段旁边写着"Ollama 地址"。
    """
    store = SettingsStore(FakeBackend())
    store.save_backend(
        BackendSettings(host="http://127.0.0.1:11434", cloud_base_url="https://api.deepseek.com/v1")
    )
    loaded = store.load_backend()

    assert loaded.host == "http://127.0.0.1:11434"
    assert loaded.cloud_base_url == "https://api.deepseek.com/v1"
