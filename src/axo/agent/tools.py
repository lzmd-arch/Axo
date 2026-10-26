"""工具的机制：一个工具长什么样、怎么注册、怎么派发。

**这个模块只有机制，没有内容。** 具体工具在 :mod:`~axo.agent.catalog`
里——分开是因为两者的改动节奏完全不同：注册表与派发纪律一旦定了就该稳定，而工具
的措辞、参数、返回格式会反复调。混在一起时，每调一次工具描述都要重读一遍派发逻辑。

## 为什么失败一律返回 ``ok=False``，而不是抛

派发口有四种失败：**未知工具名**、**参数不合法**、**工具自己炸了**、**闸门拒绝**。
四种全部变成 :class:`ToolOutcome`，一句给模型看的话，不往上抛。

理由只有一个，但它足够硬：这些失败**必须回灌给模型让它自己纠正**。模型把
``read_file`` 的路径写错一个字，正确的反应是让它看到"这个路径不在允许的目录里，
允许的目录是 X"然后重试；如果抛出去，这一轮就整段结束，用户看到"出错了"，而模型
连改一次的机会都没有。**工具循环的价值有一半在这里。**

这与另一处的纪律**相反**，两处都要记牢：

* **回合末的记忆写入**（``ui/main_window.py`` 的 ``_remember``）：写失败必须炸给用户看。
  那条路径上没有"让谁重试"这回事，静默失败等于用户的记忆没落库而界面说一切正常。
* **工具派发**（这里）：失败是一次对话机会，不是一次事故。

同一个项目里两种相反的纪律，判据是**这次失败之后还有没有一个能自己做决定的
当事人**：模型在，就回灌；模型不在（收尾写库），就炸。

## 异常被吃掉，但栈不能被吃掉

``except Exception`` 在这个项目里通常是坏味道（静默失败）。这里的豁免条件是
**它没有丢掉任何信息**：异常类型、消息与完整栈都进 :attr:`ToolOutcome.detail`，
而 ``detail`` 会进工具轨迹、进界面、进逐字稿。用户点开那张卡片能看到栈，
模型拿到的是那句话。**没有被谁看到的异常才是静默失败。**
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from memcore.store import MemoryStore

from ..llm.base import ToolCall, ToolSpec
from .reminders import CancelledStore, FiredStore, ReminderStore

__all__ = [
    "Tool",
    "ToolArgumentError",
    "ToolContext",
    "ToolOutcome",
    "ToolRegistry",
    "argument_int",
    "argument_str",
    "argument_text",
]


class ToolArgumentError(ValueError):
    """模型给的参数不合法。

    单独一个类型，是为了让派发口能把"模型该改参数"与"工具自己坏了"分开——前者
    回灌的是"参数不对，这样写才对"，后者回灌的是"这个工具暂时用不了"。两者对模型
    的下一步指示完全不同，混成一句话会让它对着一个坏掉的工具反复重试。

    工具作者抛这个；:meth:`ToolRegistry.dispatch` 接。
    """


@dataclass(frozen=True)
class Capture:
    """一次截图的收成。**成功与失败都是它**，工具那边不处理异常。

    ``png`` 非空 = 拿到了图；否则 ``reason`` 是一句人话（屏幕是黑的、没有截图通道、
    没等到界面线程回话…）。把"为什么没截到"做成数据而不是异常，是因为这几种收场在
    工具那一侧要做的是同一件事——**如实告诉模型这一条没看成**，而异常的差别只该体现
    在 ``reason`` 那句话里。

    ``width`` / ``height`` 是**交出去那张图**的尺寸（裁过、缩过之后的），只用于回灌
    与卡片页脚：模型据此知道它看到的不是原始分辨率。
    """

    png: bytes | None = None
    reason: str = ""
    width: int = 0
    height: int = 0

    @property
    def ok(self) -> bool:
        return self.png is not None


@dataclass(frozen=True)
class ToolContext:
    """一次工具调用能碰到的东西。

    **刻意是一个显式的袋子，而不是让工具自己去 import 单例**：于是"这个工具能
    读写什么"在它的签名上一眼看得见，测试里也能把每一样换成替身——``clock`` 换成
    固定时刻，``read_root`` 换成临时目录。

    Attributes:
        store: 记忆库。只读工具（``search_memory``/``read_memory``）用它。
        writer: 写入器。**只有过闸门的工具用得到**（``write_memory``），所以可以是
            ``None``——一次只放行了只读工具的运行里，它就该是 ``None``。这里的类型
            写宽是因为 ``agent/`` 不该为了一个注入口去 import 记忆写入侧的实现。
        fired: 响过的那本账。**只读**——没有任何工具写它（写它的是调度器与无头
            入口）。``list_reminders`` 靠它算出重复提醒的「下一次是几号」。
        cancellations: 取消过的那本账。``cancel_reminder`` 写它，``list_reminders``
            与 ``due_now`` 读它。
        reminders: 提醒登记簿（``write_memory`` 之外的另一个写口，``set_reminder``
            用）。同样可以是 ``None``。
        session_id: 当前会话。工具写入的 ``source_ref`` 要带它，否则溯源断了。
        turn_seq: 当前是第几个用户回合。同上的定位成分。
        call_id: **这一次调用**的 id。由 :meth:`ToolRegistry.dispatch` 绑定，工具
            不要自己填——它是 ``tool://<会话>/<轮次>/<call_id>`` 这个溯源指针的
            最后一截，而"同一轮里写了两次"只有它能分辨。留空只在直接调 ``run()``
            的测试里出现。
        clock: 取当前时间。**可注入**，所以 ``get_time`` 能被断言而不是只能被观察。
            改状态的工具也从这里取时间（``forget_memory`` 的回程票、``set_reminder``
            的登记时刻），于是"写进去的时间戳是多少"在测试里是可控的。
        read_root: ``read_file`` 的沙箱根。**默认空元组 = 该工具被禁用**——见
            :mod:`~axo.agent.fs` 与 ``catalog`` 里的说明。
        capture: 截一张屏幕。**注入的**，与 ``writer``/``reminders`` 同一姿态：那一头
            要跨线程回界面线程取一张图（见 ``ui/agent_worker.py``），而这一层不该知道
            Qt 长什么样。为 ``None`` 时 ``look_at_screen`` 回一句"这次没有截图通道"——
            与只读工具在没有库时的那句话同形。
    """

    store: MemoryStore
    writer: Any | None = None
    reminders: ReminderStore | None = None
    fired: FiredStore | None = None
    cancellations: CancelledStore | None = None
    session_id: str = ""
    turn_seq: int = 0
    call_id: str = ""
    clock: Callable[[], datetime] = datetime.now
    read_root: tuple[Path, ...] = ()
    write_root: tuple[Path, ...] = ()
    """允许**写**的目录。默认空 = 写入工具不登记（同 ``read_root``）。"""
    data_root: Path | None = None
    """程序自己的数据目录。**永远不许写**——模型不该改自己的记忆库、设置与日志。"""
    checkpoints: Any = None
    """文件检查点存储（``CheckpointStore``）。写工具在获批后先 capture；None = 不回滚。"""
    jobs: Any = None
    """后台作业管理器（``CommandJobs``）。None = 后台命令工具不可用。"""
    test_session: Any = None
    """同一会话的测试历史（``TestSession``）。None = ``run_tests`` 不做跨次对比。

    **跨用户回合复用**：上一轮的失败集合与文件快照留在这里，下一轮第一次跑测试就能
    说出“与上次相比”。换会话时由主窗口重建；放在 ToolContext 上而不是模块级单例，
    是因为两个窗口/两轮并发运行时不能互相污染。
    """
    repo_map_cache: Any = None
    """仓库地图缓存（``RepoMapCache``）。None = 每次重扫。

    键里带工作区快照摘要：文件没变就命中，改一个字节就失效。快照截断时不缓存。
    """
    approval_selection: Callable[[str], tuple[str, ...] | None] | None = None
    """界面在确认卡片上按文件勾选的结果：``call_id -> 选中的路径``。

    ``None`` = 没限定（整包应用）；空元组 = 用户一个文件都没选。只有需要“逐文件”
    的工具（目前是 ``apply_patch``）读它；其它工具不看。
    """
    lsp_manager: Any = None
    """语言服务器管理器（``LspManager``）。None = ``lsp`` 工具不可用。

    服务器进程是长生命周期的，所以由主窗口持有、``ToolContext`` 只拿引用；找不到
    对应语言的服务器时工具会明确让模型回退到 ``find_symbol``/``repo_map``。
    """
    allow_commands: bool = False
    """允许执行命令。默认假；打开后每次执行仍然过闸门。"""
    cancel: Any = None
    """取消信号（``threading.Event``）。**只给会阻塞的工具**（命令执行）用。"""
    net: Any = None
    """:mod:`axo.net` 的实例（联网搜索/抓取）。``None`` = 不登记那两个工具。"""
    web_cache: Any = None
    """原文缓存 + 台账（:class:`axo.web_cache.FetchCache`）。``None`` = 只回灌、不落盘。

    **工具只往里追加，不读记忆库、也不写记忆库**——入库是 daemon 巩固路径的事。
    """
    weather: Any = None
    """天气客户端（:class:`axo.weather.OpenMeteoWeather`）。``None`` = 不登记天气工具。"""
    open_external: Callable[[str, str], str] | None = None
    """打开外部程序/网址：``(目标, "path"|"url") -> 空串或一句失败原因``。"""
    clipboard_read: Callable[[], str] | None = None
    """读剪贴板。"""
    clipboard_write: Callable[[str], str] | None = None
    """写剪贴板：返回空串或一句失败原因。"""
    notify: Callable[[str, str], None] | None = None
    """发一条桌面通知。"""
    capture: Callable[[], Capture] | None = None
    music: Any | None = None
    """本地音乐服务（``MusicService`` 的鸭子类型）。``None`` = 不登记音乐工具。"""


@dataclass(frozen=True)
class ToolOutcome:
    """一次工具调用的结果。

    :attr:`content` 是**回灌给模型的那段正文**，不是给用户看的。写成给模型看的
    语气（"这个路径不在允许的目录里，允许的目录是 …"），因为它的下一个读者确实
    是模型，而模型要据此决定下一步做什么。

    Attributes:
        ok: 成功与否。**失败也是正常返回**，见模块文档。
        content: 回灌正文。失败时是一句能指导下一步的说明。
        detail: 给人看的补充（异常栈、账本附注）。**不进模型上下文**——栈进上下文
            只会白烧 token 并让模型开始道歉。
        images: 随结果一起**回灌给模型的图**（base64 的 PNG，不带 ``data:`` 前缀）。
            现在只有 ``look_at_screen`` 用。它**只活在这一次工具往返里**：落逐字稿的
            那条路只存 ``content``、L5 回放也重建不出图（``assemble._tool_pair`` 造行
            时不带 ``images``），所以带图的工具**必须**同时标 ``replayable=False``——
            否则下一轮模型会收到一条"你看过屏幕"却没有图的记录，然后开始编。
            ``tests/test_agent_tools.py`` 有一条断言替这件事站岗。
        replayable: **这次结果值不值得在后续轮次里回放**。``False`` 的会连同它那一
            对落进逐字稿时标 :attr:`~axo.session.store.Message.no_replay`，
            由 L5 跳过。两种情形要标 ``False``：

            * **当轮的一次性事件**——闸门拒绝。它是"这一次调用被跳过了"，不是一条
              关于这个工具的规矩；回放它会让模型以为那个禁令仍然生效。
            * **某一刻的快照**——``get_time`` 的"现在"、``read_file`` 的文件正文。
              它们的语义是**当前值**，而回放递过去的是**当时值**，中间隔了几轮。

            反过来，``search_memory`` 这类"世界没怎么变、结果还能当真"的默认
            ``True``：不回放它，模型就得把已经查过的东西再查一遍。
    """

    ok: bool
    content: str
    detail: str = ""
    replayable: bool = True
    images: tuple[str, ...] = ()
    plan: tuple[Any, ...] = ()
    """``update_plan`` 交出的步骤清单。**只进界面与 run log，不进逐字稿。**"""
    preview: str = ""
    """给**用户**看一眼的那张图的本地路径（绝对路径）。空串表示没有。

    **与 :attr:`images` 不是一回事，别混。** 那个是回灌给模型的图（base64、进上下文）；
    这个是界面上要显示的那一张。表情包正是后者：她发了一张图给用户看，而模型不需要
    "看见"自己发的那张——它知道"我发过一张"就够了。

    它是**绝对路径**；落进逐字稿时由 ``ui/main_window.py`` 换成相对数据根的路径
    （数据根会搬家，而逐字稿不该因此失效）。
    """

    @classmethod
    def success(
        cls,
        content: str,
        *,
        detail: str = "",
        replayable: bool = True,
        images: tuple[str, ...] = (),
        preview: str = "",
        plan: tuple[Any, ...] = (),
    ) -> ToolOutcome:
        return cls(
            ok=True,
            content=content,
            detail=detail,
            replayable=replayable,
            images=images,
            preview=preview,
            plan=plan,
        )

    @classmethod
    def failure(
        cls,
        reason: str,
        *,
        detail: str = "",
        replayable: bool = True,
        images: tuple[str, ...] = (),
    ) -> ToolOutcome:
        """失败。``reason`` 会原样回灌，所以它要写成对模型有用的句子。"""
        return cls(ok=False, content=reason, detail=detail, replayable=replayable, images=images)


@runtime_checkable
class Tool(Protocol):
    """一个工具。

    Attributes:
        spec: 工具声明。直接进 ``ChatRequest.tools``，所以 ``description`` 是写给
            模型看的提示词，不是注释。
        requires_approval: **要不要用户点头**。声明在工具上、由闸门读——单一事实
            来源。若写在闸门的表里，同一个工具"要不要确认"就有两处说法，而它们
            迟早会不一致，表现是某个改状态的工具悄悄变成了不需要确认。
    """

    spec: ToolSpec
    requires_approval: bool

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        """执行。

        **必须是有界的**（计划书 §十一 风险 3）：工具跑在 worker 线程里，而收尾
        ``wait_for_all`` 只等 3 秒。一个会阻塞十几秒的工具会让进程以
        ``0xC0000409`` 退出——无回溯、无 stderr，只有退出码。所以 P3a 的工具里
        **没有网络调用**。

        取消**不是**工具的责任：``ctx`` 里没有取消标志，因为一个"检查取消标志"的
        约定，在真正需要它的那天一定会被某个工具忘掉。取消由派发口在调用前后处理。
        """
        ...


class ToolRegistry:
    """一组工具。

    注册表只有两条职责，但两条都值得一个类：**把声明凑成一份能直接喂给后端的
    列表**，以及**把一次 :class:`ToolCall` 派发到正确的工具并保证它一定返回**。

    重名会**直接抛**（见 :meth:`__init__`）——这是本模块唯一的抛出口，理由是它
    发生在装配期而不是运行期：启动时炸掉看得见，而"同名工具后者被静默丢掉"的表现
    是模型某天开始调一个永远不会被执行的工具。
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        ordered = tuple(tools)
        seen: set[str] = set()
        for tool in ordered:
            name = tool.spec.name
            if name in seen:
                raise ValueError(
                    f"工具名 {name!r} 注册了两次。后者会被静默丢掉，而模型调到那个"
                    "名字时的表现是「工具不存在」——一个看不出根因的症状。"
                )
            seen.add(name)
        self._tools = ordered

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(self.names) or '空'})"

    @property
    def names(self) -> tuple[str, ...]:
        """登记的工具名，**按注册顺序**。顺序稳定是有意义的：它决定工具声明在请求
        里的排列，而稳定前缀能让后端的提示缓存命中。"""
        return tuple(tool.spec.name for tool in self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """工具声明，直接作为 ``ChatRequest.tools`` 传下去。"""
        return tuple(tool.spec for tool in self._tools)

    def get(self, name: str) -> Tool | None:
        return next((tool for tool in self._tools if tool.spec.name == name), None)

    def needs_approval(self, name: str) -> bool:
        """闸门读这个。未知工具名返回 ``False``——它根本不会被执行，派发口会以
        "未知工具"拒掉它，那条路径上有自己的说明。"""
        tool = self.get(name)
        return bool(tool and tool.requires_approval)

    def dispatch(
        self,
        call: ToolCall,
        ctx: ToolContext,
        *,
        approved: bool = True,
    ) -> ToolOutcome:
        """执行一次工具调用。**任何情况下都返回一个 :class:`ToolOutcome`。**

        Args:
            call: 模型发出的这次调用。
            ctx: 工具能碰到的东西。
            approved: 闸门的裁决。**只对 ``requires_approval`` 的工具生效**——
                不需要确认的工具拿到 ``False`` 也照常执行，因为压根没有人问过它。

        绑定 :attr:`ToolContext.call_id` 是这一步的职责，不是工具作者的：溯源指针
        （``tool://<会话>/<轮次>/<call_id>``）的最后一截只有派发口拿得到。让工具自己
        从参数里找它是错的（模型不会传它），让每个工具从别处取也会各有各的取法——
        而取错的症状是"记忆里的溯源指向另一次调用"，回看时看不出来。
        """
        if getattr(call, "argument_error", ""):
            # **参数拼不出来不是终局。** 直接抛会让整轮死掉，而模型完全有机会
            # 重发一次；这里仍然按协议返回一条 tool 结果（配对不能断）。
            return ToolOutcome.failure(
                f"这次调用 {call.name} 的参数不是合法 JSON 对象（{call.argument_error}）。"
                "请重新发出这次调用，参数要是一个 JSON 对象。",
                detail=call.argument_error,
            )

        tool = self.get(call.name)
        if tool is None:
            return ToolOutcome.failure(
                f"没有名为 {call.name!r} 的工具。可用的工具：{'、'.join(self.names) or '（无）'}。",
                detail=f"模型请求了未注册的工具 {call.name!r}",
            )

        if tool.requires_approval and not approved:
            # 文案与 ``replayable=False`` 是**同一处修复的两半**，改一处不够：
            #
            # 拒绝说的是"**这一次**调用被跳过了"，不是"这个工具被禁了"。但它落的是一
            # 条普通的 tool 行，而 L5 会把工具对逐轮搬回上下文——于是一句当轮的话在
            # 模型眼里变成一条仍在生效的规矩，一次拒绝就成了该会话内对该工具的永久
            # 封禁。``replayable=False`` 让这一对根本不进后续轮次的上下文，是主修；
            # 文案改成"这一轮里"是补刀：路径上只要还有一处把工具行搬进上下文（未来的
            # 新层、界面里的回看渲染）都不至于把它读成规矩。
            return ToolOutcome.failure(
                f"用户没有同意执行 {call.name}，这一次调用跳过了。"
                "这一轮里不要重试同一次调用，也不要假装它已经完成——"
                "在回答里说明这一步被跳过，并做完你还能做的部分。",
                detail="闸门拒绝",
                replayable=False,
            )

        missing = _missing_required(tool.spec, call.arguments)
        if missing:
            return ToolOutcome.failure(
                f"{call.name} 缺少必需的参数：{'、'.join(missing)}。"
                f"请补齐后重试。参数说明：{_describe(tool.spec)}",
                detail=f"缺少必需参数 {missing}",
            )

        try:
            return tool.run(call.arguments, replace(ctx, call_id=call.id))
        except ToolArgumentError as exc:
            # 模型自己给错了参数：回灌的是"怎么改"，不是"出了什么事"。
            return ToolOutcome.failure(f"{call.name} 的参数不对：{exc}", detail="参数不合法")
        except Exception as exc:
            # 见模块文档：这里吃异常的前提是**信息没丢**——类型、消息、完整栈都在
            # detail 里，而 detail 会进工具轨迹与界面。
            return ToolOutcome.failure(
                f"{call.name} 执行失败（{type(exc).__name__}）：{exc}",
                detail=traceback.format_exc(),
            )


def _missing_required(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """按 JSON Schema 的 ``required`` 查缺参。

    **只查 required 这一条**，不做完整的 JSON Schema 校验。理由：一份手写的
    JSON Schema 校验器是这个项目里最容易"看起来在工作"的东西——它能跑、能过测试，
    但覆盖的永远是它自己实现的那几个关键字。缺参是唯一一种**必然导致工具内部
    KeyError** 的错误（而 KeyError 也会被派发口接住，只是回灌的话不如这句清楚）；
    类型不对、值越界都由工具自己抛 :class:`ToolArgumentError` 说，因为那里它知道
    这个参数到底该是什么。
    """
    required = spec.parameters.get("required") or ()
    return [key for key in required if key not in arguments]


def _describe(spec: ToolSpec) -> str:
    """把参数表压成一行给人（和模型）看的话。"""
    properties = spec.parameters.get("properties") or {}
    if not properties:
        return "这个工具不需要参数。"
    return "；".join(
        f"{name}（{body.get('description', '')}）" for name, body in properties.items()
    )


def argument_text(arguments: dict[str, Any], key: str, *, required: bool = True) -> str:
    """取一段**要原样使用**的文本（文件内容、old_text、补丁）。

    与 :func:`argument_str` 的区别只有一个：**不 strip**。代码的第一行缩进、末尾换行
    都是内容的一部分；strip 之后再写盘/匹配，症状是"模型说要改，文件却没变"或
    "写进去的代码少了缩进"，而且逐字稿里看不出哪里错了。
    """
    value = arguments.get(key)
    if value is None:
        if required:
            raise ToolArgumentError(f"{key} 不能为空。")
        return ""
    if not isinstance(value, str):
        raise ToolArgumentError(
            f"{key} 必须是字符串，收到的是 {type(value).__name__}（{value!r}）。"
        )
    if required and not value.strip():
        raise ToolArgumentError(f"{key} 不能为空。")
    return value


def argument_int(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """从参数里取一个整数，**不猜**。

    模型给的参数形状是自由的：``{"limit": 5}``、``{"limit": "5"}``、
    ``{"limit": "五"}`` 都可能。这里只接受真正的整数，其余一律抛
    :class:`ToolArgumentError` 让模型重试。

    为什么不做 ``int(value)`` 那种"顺手转一下"：转成功之后，"5" 与 5 的区别就
    永远从这条路径上消失了——而它消失的那天，``"五"`` 也一样会被吞掉（或者更糟，
    被当成 0）。**一个能被纠正的错误，比一个被悄悄吸收的错误便宜得多。**
    """
    if key not in arguments or arguments[key] is None:
        return default
    value = arguments[key]
    # ``bool`` 是 ``int`` 的子类，但把 ``True`` 当 1 用是另一回事——模型给个布尔值
    # 说明它理解错了这个参数，该让它重试而不是替它圆过去。
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolArgumentError(f"{key} 必须是整数，收到的是 {type(value).__name__}（{value!r}）。")
    if minimum is not None and value < minimum:
        raise ToolArgumentError(f"{key} 不能小于 {minimum}，收到 {value}。")
    if maximum is not None and value > maximum:
        raise ToolArgumentError(f"{key} 不能大于 {maximum}，收到 {value}。")
    return value


def argument_str(arguments: dict[str, Any], key: str, *, required: bool = True) -> str:
    """从参数里取一个非空字符串。同 :func:`argument_int`：**不猜**。"""
    value = arguments.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolArgumentError(f"{key} 不能为空。")
        return ""
    if not isinstance(value, str):
        raise ToolArgumentError(
            f"{key} 必须是字符串，收到的是 {type(value).__name__}（{value!r}）。"
        )
    return value.strip()
