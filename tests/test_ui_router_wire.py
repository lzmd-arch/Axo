"""语域路由那根线：用户的一句话 → ``MainWindow._router_state`` → 这一轮给不给工具。

**这个文件守的是 P3a 漏掉的那一格。** 规则层（一张手写词表）、不对称迟滞、120 条
语料、``bench_router.py`` 的 CI 门槛——全都做完了、全都绿，而在界面上**没有一个
消费者**：``main_window`` 只 import 了 ``HELP_TEXT`` / ``Register`` / ``parse_slash``，
语域只由裸斜杠决定。症状是"不打 ``/task`` 就永远拿不到工具"，而它看起来像模型
不听话（2026-10-10 的三轮人工验收就是这么被误读的：模型说"已通过 write_memory
工具保存"，而那一轮请求里根本没有 ``tools``）。

所以这里钉三件事，缺一件这条线就还是断的：

1. **判得对**——不打斜杠的一句话，规则层给出的语域与"要不要工具"要对；
2. **记得住**——迟滞跨轮，状态是**同一个**对象，不是每轮新起一个；
3. **真的被用上**——``_on_submit`` 消费路由的结论，而不是把结论算出来丢掉
   （这正是原来那个 bug 的形状：``route_turn`` 一直都在，没人调它）。

**为什么不建 ``MainWindow``。** 与 ``test_ui_tools_wire`` / ``test_ui_agent_slots``
同一条理由：离屏下 QtWebEngine 一渲染就把整个进程硬杀掉。``_route`` 只碰 ``self``
上的一个属性，所以拿**未绑定方法**配一个人造的 ``self`` 来调；哪天它长出新的依赖，
这个文件会以 ``AttributeError`` 的方式说出来，而不是悄悄测了个别的东西。

第 3 条（"真的被用上"）测不到行为——它要一个能跑的窗口——所以改用 ast 读源码：
断言 ``_on_submit`` 里**有**对 ``self._turn_uses_tools`` 的赋值，且整个类里**没有**
``self._register`` 这个旧旁路。形制与 ``test_repo_hygiene`` 的源码检查一致。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from axo.agent.router import Register, RegisterState, RouteSource
from axo.ui.main_window import MainWindow

pytestmark = pytest.mark.gui

_SOURCE = Path(__file__).resolve().parent.parent / "src/axo/ui/main_window.py"


def _router() -> SimpleNamespace:
    """一个只带路由状态的假 ``self``。"""
    return SimpleNamespace(_router_state=RegisterState())


def _route(window: SimpleNamespace, text: str):
    return MainWindow._route(window, text)  # type: ignore[arg-type]


# ── 1. 判得对：不打斜杠的句子 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "帮我查一下我之前记过什么",
        "写一条记忆：我最喜欢的颜色是靛蓝",
        "记住：我最喜欢靛蓝",
        "提醒我明天下午三点开会",
    ],
)
def test_a_plain_task_sentence_gets_tools_without_any_slash(text: str) -> None:
    """**这一条就是坏掉的那一格。** 用户没打 ``/task``，而这句话明摆着是要干活。

    四条都是 2026-10-10 人工验收里实际敲过（或本该被这样判）的句子；前三条当时
    全被判成闲聊，因为判语域的那一步在界面上根本不存在。
    """
    routed = _route(_router(), text)
    assert routed.register is Register.TASK
    assert routed.uses_tools is True


@pytest.mark.parametrize("text", ["今天天气不错", "早上好呀", "我有点累了", "谢谢你"])
def test_chitchat_does_not_get_tools(text: str) -> None:
    """反向的一条。判据放宽到"总是给工具"的话，这条会红——而那不是修好了，是换了
    一种坏法：每一轮都声明七个工具，模型在一句寒暄上翻用户的库。
    """
    routed = _route(_router(), text)
    assert routed.register is Register.CHAT
    assert routed.uses_tools is False


def test_an_unclear_sentence_gets_no_tools_and_asks_back() -> None:
    """判不出来时**什么都不动**，并请模型带一句确认（歧义反问）。

    ``uses_tools`` 为假不是"当成闲聊"，而是"拿不准的一轮不该动任何东西"——改状态的
    动作还有确认闸门兜着，但"读一个文件""搜一次记忆"没有闸门。
    """
    routed = _route(_router(), "思考哥德巴赫猜想")
    assert routed.uses_tools is False
    assert routed.wants_clarification is True


# ── 2. 记得住：迟滞跨轮 ─────────────────────────────────────────────────────


def test_the_state_survives_between_calls() -> None:
    """**状态是同一个对象**，所以迟滞跨轮成立：从任务落回闲聊要连续两次反向信号。

    这条同时钉住"每轮新起一个 ``RegisterState``"那种写法——那样写单看每一轮都对，
    迟滞却永远不生效，症状是"用户在任务中间插一句'好的'，下一轮工具就没了"，
    也就是这条迟滞本来要治的那个人格闪跳。
    """
    window = _router()
    assert _route(window, "帮我查一下笔记").register is Register.TASK

    first = _route(window, "今天天气不错")
    assert first.register is Register.TASK, "一次反向不够，不应该落回闲聊"
    assert first.decision.source is RouteSource.STATE, "被迟滞压住的那一轮要说明是状态说了算"

    assert _route(window, "哈哈真有意思").register is Register.CHAT


def test_a_bare_slash_switch_overrides_the_hysteresis_at_once() -> None:
    """裸 ``/chat`` **当场生效**，不会被上一次的 ``TASK`` 拖住——用户在明确说话。"""
    window = _router()
    _route(window, "帮我查一下笔记")

    assert _route(window, "/chat").register is Register.CHAT
    assert _route(window, "嗯").register is Register.CHAT


# ── 3. 只影响这一轮的那个区别 ───────────────────────────────────────────────


def test_a_slash_with_text_does_not_change_the_state() -> None:
    """``/task 正文`` 是"这一件事"，不是"从此进任务模式"。

    这一轮必须判成 ``TASK``；而**状态机不能被推进去**。喂进去的话
    ``RegisterState.advance`` 见到 ``SLASH`` 会无视迟滞直接覆盖，于是接下来几轮
    都得靠迟滞往回爬——``ParsedTurn.sticky`` 存在的全部意义就是区分这两者。
    """
    window = _router()
    routed = _route(window, "/task 帮我查一下笔记")

    assert routed.register is Register.TASK
    assert routed.uses_tools is True
    assert window._router_state.register is Register.CHAT, "只该影响一轮，状态不该被改写"


def test_a_bare_slash_switch_does_change_the_state() -> None:
    """与上一条成对：裸 ``/task`` 是**真的在切状态**，它必须进状态机。"""
    window = _router()
    _route(window, "/task")

    assert window._router_state.register is Register.TASK


# ── 4. 真的被用上：源码级守卫 ───────────────────────────────────────────────


def _main_window_class() -> ast.ClassDef:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
            return node
    raise AssertionError("main_window.py 里找不到 MainWindow")


def test_the_old_register_bypass_is_gone() -> None:
    """``self._register`` 这个字段必须**不存在**。

    它是那个 bug 的载体：一个只有裸斜杠能改的字段，规则层的结论进不来。留在那里
    没人用也无妨——但只要它还在，下次有人想让某个分支改语域时，最省事的写法就是
    去改它，于是同一条线又断一次。删掉比留一条注释管用。
    """
    attrs = {
        node.attr for node in ast.walk(_main_window_class()) if isinstance(node, ast.Attribute)
    }
    assert "_register" not in attrs
    assert "_router_state" in attrs, "路由状态都不在，说明这一整块又被摘掉了"


def test_on_submit_actually_consumes_the_routing_verdict() -> None:
    """``_on_submit`` 要把路由的三样结论都**存下来**。

    这条防的是原来那个 bug 的**精确形状**：结论一直在（``route_turn`` 有测试、
    有语料、进门槛），而调用方没接。只测 ``_route`` 是抓不到这个的——它照样绿。

    换成行为测试需要一个能跑的窗口，而离屏下建不出来（见模块文档），所以退一步
    读源码。它比行为断言弱：能证明"赋值发生了"，不能证明"赋值之后用对了"。
    后面那一半由 ``test_ui_agent_slots``（工具那一侧）与人工验收兜着。
    """
    submit = next(
        (
            node
            for node in _main_window_class().body
            if isinstance(node, ast.FunctionDef) and node.name == "_on_submit"
        ),
        None,
    )
    assert submit is not None, "找不 _on_submit"

    assigned = {
        node.targets[0].attr
        for node in ast.walk(submit)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
    }
    for field in ("_turn_register", "_turn_uses_tools", "_turn_needs_clarification"):
        assert field in assigned, f"_on_submit 没有把 {field} 存下来"
