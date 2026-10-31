"""打包声明与盘上的事实对齐：**资产搬了家，构建工具还指着原地。**

这一步挡的是最可能烂掉的那件事。``tools/build_exe.py`` 里那张 ``DATAS`` 表把
``ui/theme.qss`` 与 ``ui/web/`` 送进镜像，而镜像路径是从源码路径推出来的——两边最
容易分叉的方式不是写错，是**搬文件**：某个 js 挪进子目录、网页换个目录名，构建工具
不会跟着动。

那种错**最终会被发现**：PyInstaller 找不到源文件时直接报错。但它报在**你要构建的
时候**——也就是发版前，而那时离改文件已经过了很久。这里让它报在**你改文件的那一
次**。

**只查声明，不查产物。** 产物对不对是构建脚本自检的事（它拿得到盘上的 ``dist/``），
而这条腿在核心 pytest 里跑——**不需要 PyInstaller，也不需要 Qt**。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

import pytest

NL_CHAR = chr(10)
QUOT = chr(34)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_TOOL = _REPO_ROOT / "tools" / "build_exe.py"

_README_JARGON = (
    "P6a",
    "ADR",
    "腿",
    "守卫",
    "语料",
    "判据",
    "闸门",
    "自检",
    "冻结",
    "onedir",
    "PYZ",
    "dist-info",
    "检出",
    "本仓",
    "回归",
)
"""``README.txt`` 里不许出现的词。它随 zip 发给**使用者**，而使用者不在这个仓库里。

``P6a``/``ADR`` 是本仓自己的编号；``腿``/``守卫``/``语料``/``判据``/``闸门``/``自检``
在这里各有精确含义，对外全是噪音；``冻结``/``onedir``/``PYZ``/``dist-info`` 是实现细节。
"""


def _declared(name: str) -> object:
    """读出 ``tools/build_exe.py`` 里模块级常量 ``name`` 的**字面量**值。

    按 AST 读而不是 import：import 那个文件会把它整个模块体跑一遍（它现在只有几个
    常量，将来未必），而这里要的只是几个字面量。AST 读法**顺带把"必须是字面量"钉
    住了**——写成 ``str(Path(...))`` 之类就再也读不出来，而那种写法同时会让这条守卫
    静默失效，所以读不出来时报错比猜一个值好。
    """
    tree = ast.parse(_BUILD_TOOL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{_BUILD_TOOL.name} 里没有模块级常量 {name}")


def test_the_entry_point_is_a_real_file() -> None:
    """入口文件在。本仓没有 ``[project.scripts]``，所以这一条就是"构建工具指着的东西
    存在"的全部保证。"""
    entry = _declared("ENTRY")
    assert isinstance(entry, str)
    assert (_REPO_ROOT / entry).is_file(), f"入口不在：{entry}"


def test_the_frozen_entry_has_no_relative_imports() -> None:
    """被冻结的那个入口**自己不许用相对 import**。

    这是实测踩出来的坑，不是假想的：PyInstaller 收的是一份**脚本**，它把入口文件当
    ``__main__`` 跑，于是 ``__package__`` 是空的——``from .main_window import ...``
    在**运行时**炸掉。而无控制台的产物撞上未捕获异常只弹一个错误框，进程还活着，
    所以"活体探测"当时给的是假绿，真正的症状是"双击了，什么都没发生"。

    判据是这道守卫能机械判的那一半：入口文件里**没有一个 ``level > 0`` 的 import**。
    另一半（被它 import 的模块自己能不能按名字找到）由构建时那条自检负责。
    """
    entry = _declared("ENTRY")
    assert isinstance(entry, str)
    tree = ast.parse((_REPO_ROOT / entry).read_text(encoding="utf-8"))

    relative = [
        f"第 {node.lineno} 行：{'.' * node.level}{node.module or ''}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative == [], (
        f"{entry} 里有相对 import：\n  " + "\n  ".join(relative) + "\n"
        "冻结时这个文件被当 __main__ 跑，__package__ 是空的，这些 import 会在运行时炸。"
    )


def test_every_declared_asset_exists() -> None:
    """``DATAS`` 里每一条源路径都在盘上。"""
    datas = _declared("DATAS")
    assert isinstance(datas, tuple)
    assert datas, "DATAS 是空的——那等于一个资产都不带，产物会白屏"

    missing = [source for source, _ in datas if not (_REPO_ROOT / source).exists()]
    assert missing == [], f"这些资产声明了但盘上没有：{missing}"


def test_the_mirror_lands_inside_the_package() -> None:
    """目标路径必须**在包内、且与源码同形**。

    这一条守的是那个让应用代码一个字都不用改的假设：PyInstaller 给包内模块设的
    ``__file__`` 指向 mirror 里的路径，于是 ``Path(__file__).with_name("theme.qss")``
    只有在 qss 落在 ``axo/ui/`` 下时才读得到。目标写歪了 PyInstaller
    **不报错**——它照放，而那边读到一个不存在的路径，症状是白屏或少一套样式。
    """
    for source, target in _declared("DATAS"):
        expected = PurePosixPath(source).relative_to("src")
        where = expected.parent if (_REPO_ROOT / source).is_file() else expected
        assert target == str(where), f"{source} 该放进 {where}，声明的是 {target}"


def test_no_spec_file_lands_in_the_repository() -> None:
    """仓库根不该有 ``.spec``。

    构建参数写在 ``tools/build_exe.py`` 里（``--specpath build``），理由是 ``.spec``
    是 ``.py`` 之外的散件：ruff 扫不到、测试读不到，等于一块不进任何腿的代码。
    """
    stray = sorted(path.name for path in _REPO_ROOT.glob("*.spec"))
    assert stray == [], f"仓库根出现了 .spec：{stray}——构建参数该写在 tools/build_exe.py 里"


def test_the_frozen_directories_stay_out_of_version_control() -> None:
    """``dist/`` 与 ``build/`` 在 ``.gitignore`` 里。

    产物几百 MB，而且是**从这里的源码生成的**——进了版本控制就是每次构建都往历史里
    塞一份二进制，而那些字节再也删不掉。
    """
    lines = {
        line.strip()
        for line in (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "dist/" in lines
    assert "build/" in lines


def test_only_debug_variants_are_pruned() -> None:
    """``_DEBUG_RESOURCES`` 里**只许出现调试变体**，而且只许落在 Qt 的资源目录里。

    那条"构建后删掉"的名单，正当性全部来自一句话：删的是**调试变体**、不是运行件——
    每个被删的文件都有一个同名的非调试孪生留在镜像里。这句话很容易在某次编辑里被悄悄
    破坏：多写一行把 ``qtwebengine_resources.pak`` 也删了，构建照样成功、自检照样绿，
    而屏幕上是白屏。所以判据要机械：文件名里必须带 ``.debug.``。
    """
    patterns = _declared("_DEBUG_RESOURCES")
    assert isinstance(patterns, tuple)
    assert patterns, "_DEBUG_RESOURCES 空着的话，那次删除就是空转"
    for pattern in patterns:
        path = PurePosixPath(pattern)
        assert ".debug." in path.name, f"{pattern} 不像调试变体——运行件不许删"
        assert path.parts[:2] == ("PySide6", "resources"), (
            f"{pattern} 落在 Qt 的资源目录之外，那里没有「孪生在旁边」这条保证"
        )


def test_no_absolute_path_is_hardcoded() -> None:
    """构建工具里**一个字面量绝对路径都不许有**，桌面那条尤其。

    桌面可能被 OneDrive 接管（那时它在 ``%USERPROFILE%\\OneDrive\\桌面``），也可能被组
    策略挪到别处，所以 ``build_exe.py`` 是**问系统要**的（``SHGetKnownFolderPath``）。
    这个判断很容易在某次"顺手修一下"里退化成
    ``Path(r"C:\\Users\\<名字>\\Desktop")``——那样在这台机器上照样能用，在别的机器上则
    **静默落到一个存在的、但不是桌面的目录**：快捷方式"建成功了、人却看不见"，正是这一
    步要治的那个症状。

    判据取最宽的那条：``X:\\`` 形状的字面量一个都不许有。那条路只有一条正当走法（从
    ``__file__`` 或环境变量算出来），所以宽判据在这里不误伤。
    """
    littered = sorted(
        {
            node.value
            for node in ast.walk(ast.parse(_BUILD_TOOL.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.match(r"^[A-Za-z]:[\\/]", node.value)
        }
    )
    assert littered == [], f"{_BUILD_TOOL.name} 里有写死的绝对路径：{littered}"


def test_the_shortcut_is_declared_as_a_literal_name() -> None:
    """桌面快捷方式的名字是模块级的字面量常量、且不是空的。

    ``.lnk`` 的文件名由它拼出来（``<名字>.lnk``），所以空着不会报错——只会得到一个
    ``.lnk``，桌面上看不出哪里不对。名字还必须与界面上的显示名一致（那是给人点的图标），
    而"一致"这一半机器判不了，由人读。
    """
    name = _declared("SHORTCUT_NAME")
    assert isinstance(name, str)
    assert name.strip(), "SHORTCUT_NAME 空着会建出一个叫 .lnk 的文件"


def test_only_gpl_only_qt_families_are_pruned() -> None:
    """``_GPL_ONLY_MODULES`` 里只许是 Qt6 的模块 DLL，伴随清单只许是 ``PySide6/`` 下的东西。

    那条剔除清单的全部正当性来自一句很具体的话：**这几族只在 GPLv3 下提供**，而本项目
    按 LGPL-3.0 那一支用 Qt。往里加一个别的文件名（"反正用不到，顺手把 Qt6Sql.dll 也
    删了"）就把这张表换成了另一种东西——一份"用不到的东西"清单。那种清单没有许可上的
    理由，只有体积上的，而它随时可能删掉一个真的在用的模块；症状是用户机器上的白屏或
    崩溃，构建这边一切正常。

    伴随清单非空这一条同样要紧：它记的是**实测**出来的一批加载器（四个 QML 插件目录
    与一个输入法插件，它们的导入表里写着那 8 个名字）。空着的话
    ``_check_gpl_pruned`` 会报"还有文件提到它"，而那正是它该报的。
    """
    modules = _declared("_GPL_ONLY_MODULES")
    assert isinstance(modules, tuple)
    assert modules, "空着的话那次剔除就是空转"
    for module in modules:
        assert module.startswith("Qt6") and module.endswith(".dll"), (
            f"{module} 不像 Qt6 的模块 DLL——这张表只收「只在 GPLv3 下提供」的那几族"
        )

    companions = _declared("_GPL_ONLY_COMPANIONS")
    assert isinstance(companions, tuple)
    assert companions, "伴随清单空着，剔除会因为「还有文件提到它」而红"
    for companion in companions:
        assert companion.startswith("PySide6/"), f"{companion} 落在 PySide6/ 之外"
        assert not companion.endswith(modules), f"{companion} 与 _GPL_ONLY_MODULES 重叠"


def test_the_release_readme_speaks_to_its_reader() -> None:
    """zip 里那份 ``README.txt`` 是写给**拿到它的人**的，不许出现仓库内部的行话。

    这些词在仓库里各有精确的含义（``腿`` 指流水线上的一条 job、``判据`` 指一条机器可
    验的断言、``语料`` 指召回基准的那批文本），对读者全是噪音：他不知道那是什么，只会
    觉得这份说明不是写给他的。而它**是**唯一一份他会读的说明。

    允许列表在这里没用——"哪些词算行话"是一份会长的名单，所以判据取黑名单那一侧，
    并且由人来补：这条红了，要么改文案，要么把这个词加进来。
    """
    text = _declared("_RELEASE_README")
    assert isinstance(text, str)
    found = [word for word in _README_JARGON if word in text]
    assert found == [], f"发给读者的说明里出现了仓库内部的行话：{found}"


def test_the_release_readme_answers_the_questions_a_reader_has() -> None:
    """读者一定会问的几件事，说明书里得有答案。

    **这不是"文案完整性"的美化检查。** 这几条各自对应一次真实的卡住：解压之后只拖出
    exe（旁边那个文件夹是程序的一部分）、被 SmartScreen 拦下以为中了毒、装完发现不能
    聊天（还缺一个模型）、想卸载却不知道记忆存在哪。少一条，那个人就会在某个环节停下
    来，而他能问的人只有这份文本。
    """
    text = _declared("_RELEASE_README")
    assert isinstance(text, str)
    for needle in (
        "{version}",  # 版本号是打包时替换的，没有占位符就永远是空的
        "Ollama",  # 不装它就聊不了天——最常卡住的一步
        "SmartScreen",
        "%LOCALAPPDATA%",  # 记忆在哪、怎么卸载
        "LICENSE",
        "LGPL-3.0",  # 唯一的 copyleft 依赖，读者有权知道
    ):
        assert needle in text, f"发给读者的说明里没提 {needle}"


def _probe_names() -> set[str]:
    """``_probe`` 函数体里出现过的标识符（含 ``shutil.rmtree`` 那个 ``rmtree``）。

    按 AST 读而不是 grep 源码：grep 会把注释与 docstring 里的同一个词也算上，而
    这条要问的是"代码里到底调没调它"——`_probe` 的 docstring 里恰好大段讨论这件事，
    所以两种读法给出的答案会不一样。
    """
    tree = ast.parse(_BUILD_TOOL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_probe":
            found = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            found |= {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
            return found
    raise AssertionError(f"{_BUILD_TOOL.name} 里没有 _probe 这个函数")


def test_the_probe_isolates_the_settings_and_the_qt_cache() -> None:
    """**这条挡的是"以为隔离好了，其实没有"。**

    活体探测原来只给子进程设了 ``AXO_DATA``，而那只重定向 ``axo.paths``（记忆与
    数据）。程序还会往两个与它无关的地方写：``QSettings`` 落真注册表
    （``backend/last_model``）、WebEngine 的管线缓存落真 ``%LOCALAPPDATA%``。

    这两条**在构建日志里一个字都不显示**，所以"这次构建碰过我的东西没有"只能靠
    机器判据回答。它也是新手引导的前置：引导那个"走过了"的标记写在同一个注册表里，
    探测一旦把它写成 1，这台机器上引导再也不会出现。
    """
    names = _probe_names()

    assert "_PROBE_STATE_ENV" in names, "_probe 没有给子进程设隔离界面设置的那个变量"
    assert "_PROBE_STATE_PREFIX" in names, "_probe 没有为隔离出来的设置目录建临时目录"
    assert "rmtree" in names, "探测起完没清理，临时目录会攒在系统临时目录里"


def test_the_two_copies_of_the_remind_flag_agree() -> None:
    """``--remind`` 这个名字也写了两份，而**后果比环境变量那条更持久**：

    ``axo.autostart`` 会把它写进 ``HKCU\...\Run``，从此固定在那台机器上。两边改岔了，
    已经设过自启的机器会在下次登录时把一个**不认识的参数**交给程序——而那个症状
    （登录时什么都不发生）离这次改动已经过去几周。
    """
    from axo.ui.app import REMIND_FLAG

    assert _declared("_REMIND_FLAG") == REMIND_FLAG


def test_the_remind_check_runs_the_real_product_with_an_isolated_data_dir() -> None:
    """``--remind`` 那条自检必须：起真产物、喂一条到点的提醒、**隔离数据目录**。

    与 ``test_the_probe_isolates_the_settings_and_the_qt_cache`` 同一条纪律，
    而且这一条更硬一点：那一次会**写** fired 账。不隔离的话，每次构建都会往使用者的
    真提醒簿里多写一行"构建自检：这条不该真的打扰你"——而构建日志里一个字都不显示。
    """
    tree = ast.parse(_BUILD_TOOL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_check_remind":
            found = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            found |= {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
            # **字符串字面量也算**：`environment["AXO_DATA"] = ...` 里那个键是一个常量，
            # 而"有没有隔离数据目录"这件事正好只在那一行上看得出来。
            found |= {
                item.value
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            break
    else:
        raise AssertionError(f"{_BUILD_TOOL.name} 里没有 _check_remind 这个函数")

    assert "_REMIND_FLAG" in found, "自检没有真的把那个开关交给产物"
    assert "AXO_DATA" in found, "自检没有隔离数据目录"
    assert "_PROBE_STATE_ENV" in found, "自检没有隔离界面设置（那次会读数据根的设置）"
    assert "rmtree" in found, "自检起完没清理，临时目录会攒在系统临时目录里"


def test_the_two_copies_of_the_environment_variable_name_agree() -> None:
    """那个环境变量名写了两份，而**写歪了的症状是"设了一个没人认的变量"**——
    自检以为隔离好了，实际照样在改使用者的注册表。

    为什么是两份而不是 import 过来：``build_exe.py`` 是**构建工具**，它在 ``src``
    还没进 ``sys.path`` 的时候就要能被 AST 读出来（``_declared`` 的 docstring 说了
    为什么按 AST 读）。所以这里照 ``settings_store.DEFAULT_HOST`` 那条"刻意写两遍、
    用测试钉死一致"的姿态办。
    """
    from axo.ui.app import _ISOLATED_STATE_ENV

    assert _declared("_PROBE_STATE_ENV") == _ISOLATED_STATE_ENV


def _build_tool():
    """把 ``tools/build_exe.py`` 当模块加载（它 import 期只算几个常量与函数定义）。

    这个文件别处一律按 AST 读**字面量**；这一条要问的是两个**函数的行为**，那就只能
    真的把模块装上跑。它 import 期不碰 PyInstaller（那个在 ``_require_pyinstaller``
    里现 import），所以这条在核心腿里照样跑得动。
    """
    import importlib.util

    # **3.10 上直接跳过。** 这个模块顶层 ``import tomllib``（3.11 才有），而它本身只在
    # 打包用的那台机器上跑（打包脚本一直是 3.12）；核心腿要跑 3.10，所以这一条只在
    # 3.11+ 上真跑——CI 的 3.12 那一条会跑到它。
    pytest.importorskip("tomllib", reason="build_exe.py 自己要 tomllib（3.11+）")

    spec = importlib.util.spec_from_file_location("_build_exe_under_test", _BUILD_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_license_files_inside_the_zip_have_distinct_names() -> None:
    """收进 zip 的许可文件必须**各占一个名字**，而且路径要能认出它出自哪个 dist-info。

    2026-10-12 实测踩到：``setuptools`` 的 RECORD 里有 12 份许可，其中 11 份是它
    **vendored** 的那些包自己的
    （``setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE``…）。当时那个函数
    返回的是"第一个 ``.dist-info/licenses/`` 之后"那一段，于是这 11 份全成了 ``LICENSE``
    ——zip 里同一个名字被写十二遍，``zipfile`` 只留最后一份，**另外十一份静默消失**，
    只在 stderr 上留一句 UserWarning。

    这是最坏的一种失效：包照打、清单照在、读者以为许可齐了，而里面少了几份义务。所以
    这条按结果钉两件事——名字两两不同，且每一份的路径里都还看得见 ``.dist-info/``
    （后者正是那个修法的形状：返回**完整相对路径**，不切那一段）。
    """
    build_tool = _build_tool()
    arcs: list[str] = []
    for distribution, _version in build_tool._declared_frozen_set():
        for tail, _path in build_tool._distribution_license_files(distribution):
            assert ".dist-info/" in tail, (
                f"{distribution} 收进来的 {tail!r} 看不出出自哪个 dist-info——"
                "那正是「vendored 的几份会被切成同一个名字」的那个形状"
            )
            arcs.append(f"licenses/{build_tool._canonical(distribution)}/{tail}")

    assert arcs, "一份许可都没收到——这条判据在空转，先去查 _declared_frozen_set"
    duplicated = sorted({arc for arc in arcs if arcs.count(arc) > 1})
    assert duplicated == [], f"zip 里会重名的许可条目：{duplicated}。重名的会被后来的覆盖掉。"


def test_torch_never_enters_this_repository() -> None:
    """**PyTorch 不进本仓**（pyproject 里那条纪律）——语音那一族把它推给了独立组件。

    克隆 TTS 与本地音频模型都在 torch 栈上，而它们以「自带依赖的独立程序」形态存在
    （与 Ollama 同姿态）：Axo 只带调用端。这条断言钉的是那条纪律**没被顺手破掉**——
    破掉的症状是产物暴涨几个 GB、许可表换血，而"加一个依赖"在某个 PR 里看着毫无代价。
    """
    # **剥掉 # 注释再找**：那条纪律的正文本身就写着「PyTorch 不进本仓」——
    # 在注释里搜到它自己不算违规（第一版就这么误红的）。
    body = NL_CHAR.join(
        line
        for line in (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").split(NL_CHAR)
        if not line.strip().startswith("#")
    ).lower()
    for banned in ("torch", "f5-tts", "gpt-sovits"):
        assert f"{QUOT}{banned}" not in body and f"{banned}>=" not in body, (
            f"pyproject.toml 的依赖里出现了 {banned}——PyTorch 不进本仓（见那条纪律）；"
            "语音引擎是独立组件，Axo 只带调用端"
        )


def test_package_version_matches_pyproject() -> None:
    """``axo.__version__`` 必须与 ``pyproject.toml`` 一致。

    两处漂移时，``app.setApplicationVersion`` 与开屏读的是前者、安装包与仓库
    标签读的是后者：程序会自称旧版本，看起来像"升级没成功"，而没有任何一步会
    报错。发版脚本的 ``bump()`` 同时改两处，这条测试守的是"有人手改了其中一处"。
    """
    declared = re.search(
        r'^version = "([^"]+)"',
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.M,
    )
    assert declared is not None
    tree = ast.parse((_REPO_ROOT / "src" / "axo" / "__init__.py").read_text(encoding="utf-8"))
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    assert values == [declared.group(1)]


def test_mcp_servers_ship_with_the_package() -> None:
    """桌面包必须带两个 MCP 入口，否则写档位在装好的机器上空转。

    这条检查的是"构建工具声明了什么"与"入口脚本真的存在且是绝对 import"：
    入口脚本被 PyInstaller 当 ``__main__`` 跑，相对 import 在那种情况下解析不了，
    症状是产物能构建、双击没反应。
    """
    from tools import build_exe

    assert set(build_exe.MCP_ENTRIES) == {"axo-mcp", "axo-mcp-write"}
    root = Path(build_exe.__file__).resolve().parents[1]
    for entry in build_exe.MCP_ENTRIES.values():
        source = (root / entry).read_text(encoding="utf-8")
        assert "from axo." in source, entry
        assert "from ." not in source, entry
