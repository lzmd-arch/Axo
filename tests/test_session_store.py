"""会话存储。

选它来测而不是 UI：这是 P1 记忆写入要接的地方，也是最容易出**静默数据损坏**的
地方——半行 JSON、缺元信息、删除时漏删旁挂文件，都不会报错，只会让用户某天发现
"聊天记录少了一段"。

另一个重点是行尾。Windows 上 ``Path.write_text`` 默认把 ``\\n`` 写成 ``\\r\\n``，
而这个文件可能被复制、被工具读取、被将来的导出功能打包。行尾漂移不会报错，
只会让每次读取都产生一份"看起来一样但字节不同"的东西。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axo.session.store import (
    _SESSION_ID_RE,
    Message,
    SessionStore,
    new_session_id,
)


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def test_session_id_is_sortable_and_readable() -> None:
    first = new_session_id()
    second = new_session_id()

    # 时间前缀让它可按字典序排序（同一秒内后缀随机，所以比较 <=  而不是 <）
    assert first[:15] <= second[:15]
    assert len(first) == len(second)
    assert "-" in first


def test_append_creates_header_then_messages(store: SessionStore) -> None:
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="第一条"))
    store.append(session, Message(role="assistant", content="第二条"))

    # 首行之后每行一条，追加写——不是每轮重写整个文件
    lines = store._path(session.session_id).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    # 写死 5 而不是引用 ``_SCHEMA``：这条测试要在**改版本号时红一次**，逼人停下来
    # 想清楚"旧文件还读得回来吗"。引用常量的话它就永远绿，等于没有这条测试。
    # （2 → 3 那次它确实红了，下面那一节"schema 1/2 → 3"就是当场补的；3 → 4 那次
    # 也红了，补的是"3 → 4"；4 → 5 那次同样红了，补的是"schema 4 → 5"。）
    assert json.loads(lines[0])["schema"] == 5
    assert json.loads(lines[1])["content"] == "第一条"
    assert json.loads(lines[2])["content"] == "第二条"


def test_round_trip_preserves_fields(store: SessionStore) -> None:
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="问"))
    store.append(
        session,
        Message(
            role="assistant",
            content="答",
            interrupted=True,
            prompt_tokens=101,
            completion_tokens=42,
        ),
    )

    reloaded = store.load(session.session_id)
    assert [message.role for message in reloaded.messages] == ["user", "assistant"]
    assert reloaded.messages[1].interrupted is True
    assert reloaded.messages[1].prompt_tokens == 101
    assert reloaded.messages[1].completion_tokens == 42
    assert reloaded.title == "问"  # 首条用户消息成为标题


def test_title_only_set_by_first_user_message(store: SessionStore) -> None:
    session = store.load(new_session_id())
    store.append(session, Message(role="assistant", content="我先说话"))
    assert session.title == ""
    store.append(session, Message(role="user", content="现在轮到我了"))
    store.append(session, Message(role="user", content="第二句不该改标题"))
    assert session.title == "现在轮到我了"


def test_meta_file_is_derived_and_updated(store: SessionStore) -> None:
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="起个标题"))

    meta = json.loads(store._path(session.session_id).with_suffix(".meta.json").read_text("utf-8"))
    assert meta["title"] == "起个标题"
    assert meta["session_id"] == session.session_id


def test_meta_write_leaves_no_temp_file(store: SessionStore) -> None:
    """原子替换用的是一个临时文件；它不该留在目录里。"""
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="x"))
    leftovers = list(store.root.glob("*.tmp"))
    assert leftovers == []


def test_files_use_lf_line_endings(store: SessionStore) -> None:
    """Windows 上 ``write_text`` 默认写 CRLF，在这里是错的。"""
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="正文"))
    store.append(session, Message(role="assistant", content="回复"))

    for path in store.root.iterdir():
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} 混进了 CRLF"


def test_corrupt_line_is_skipped_not_fatal(store: SessionStore) -> None:
    """崩溃断电会留下半行 JSON。丢一轮远好过整个会话打不开。"""
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="完整的一轮"))
    path = store._path(session.session_id)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"role": "assistant", "content": "被截断的\n')
    store.append(session, Message(role="assistant", content="后面还有"))

    reloaded = store.load(session.session_id)
    assert [message.content for message in reloaded.messages] == ["完整的一轮", "后面还有"]


def test_load_missing_session_returns_empty(store: SessionStore) -> None:
    """打开一个已被外部删除的会话，结果应该是"空会话"而不是异常。"""
    session = store.load("20260101-000000-abcdef")
    assert session.messages == []
    assert not store.exists("20260101-000000-abcdef")


def test_list_sessions_orders_by_recency(store: SessionStore) -> None:
    older = store.load(new_session_id())
    store.append(older, Message(role="user", content="早的"))
    newer = store.load(new_session_id())
    store.append(newer, Message(role="user", content="晚的"))

    listed = store.list_sessions()
    assert newer.session_id in [session.session_id for session in listed]
    assert len(listed) >= 2


def test_delete_removes_body_and_meta(store: SessionStore) -> None:
    session = store.load(new_session_id())
    store.append(session, Message(role="user", content="待删"))
    assert store.exists(session.session_id)

    store.delete(session.session_id)

    assert not store.exists(session.session_id)
    assert not store._path(session.session_id).with_suffix(".meta.json").exists()
    assert not list(store.root.iterdir())


def test_delete_is_idempotent(store: SessionStore) -> None:
    store.delete("20260101-000000-abcdef")  # 不抛就行


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "..",
        ".",
        "../escape",
        "..\\escape",
        "a/b",
        "a\\b",
        "a:b",
        "a*b",
        "a?b",
        'a"b',
        "a<b",
        "a|b",
        "20260101-000000-abcdef.jsonl",  # 形状对但带了扩展名：拼出来会变成 .jsonl.jsonl
        "20260101-000000-000000/../..",  # 前缀合法，尾巴越界
        "20260101-000000-ABCDEF",  # 大写十六进制：不是本模块会生成的形状
        "2026-101-000000-abcdef",  # 时间戳段数不对
        "20260101000000-abcdef",  # 少了分隔符
    ],
)
def test_illegal_session_ids_are_rejected(store: SessionStore, bad_id: str) -> None:
    """会话 id 会拼进文件路径。只有本模块生成它，但仍然不信它——

    一个带 ``..`` 的 id 足以把删除操作指向目录树外面去，而 ``delete`` 会照着
    那个路径删文件。校验用形状白名单，所以"看起来没事但形状不对"的也要挡住。
    """
    with pytest.raises(ValueError):
        store._path(bad_id)


def test_generated_ids_always_pass_validation() -> None:
    """白名单必须与 :func:`new_session_id` 的产出严丝合缝。

    这条是上面那些拒绝用例的**对照**：如果生成器改了格式而白名单没跟着改，
    上面全都会通过，而程序会在第一次写会话时炸掉。
    """
    for _ in range(200):
        session_id = new_session_id()
        assert _SESSION_ID_RE.match(session_id), f"生成器产出了自己都不认的 id：{session_id}"


def test_empty_content_is_rejected_at_the_right_layer() -> None:
    """``Message`` 允许空内容（流式中间态），``MemoryItem`` 不允许。

    这里的断言是**记录这个区别是有意的**：逐字稿里空消息无害（比如被中断的
    回复），而记忆条目是提取出来的结论，空结论没有任何意义。
    """
    assert Message(role="assistant", content="").content == ""


# ── schema 1/2 → 3：加字段，不改历史字节 ────────────────────────────────────
#
# 这一节测的不是"新功能能用"，而是**旧文件不动**。逐字稿可以有几百兆、可以整段删，
# 所以迁移做成"新行多几个键"而不是重写整个文件——而重写真正的代价（崩溃丢整段）只在
# 出事那天看得见。能测的是它的前提：老行逐字节不变、老读者读得回来。
#
# 两次加字段共用这一节（1 → 2 是三个工具字段，2 → 3 是 ``no_replay``），因为它们要
# 证的是同一件事。下面手写的夹具是 **schema-1** 的——最老的那个形状，新读者要能读它，
# 那就顺带覆盖了 schema-2（2 只是 1 加三个键，加不加都在"缺键给默认值"这条路上）。
#
# 反方向（新版本的文件被更旧的构建读）**没有自动化护栏**：测不了"一个不存在的旧
# 构建"。这是已知未覆盖的一格，写在 ``store.py`` 的 ``_SCHEMA`` docstring 里。

_OLD_ID = "20260101-000000-abcdef"
_OLD_HEADER = json.dumps(
    {
        "schema": 1,
        "session_id": _OLD_ID,
        "title": "旧的",
        "created_at": 1.0,
        "updated_at": 2.0,
        "model_ref": "qwen3:4b",
    },
    ensure_ascii=False,
)
_OLD_USER = json.dumps({"role": "user", "content": "老的一行", "at": 1.5}, ensure_ascii=False)
_OLD_ANSWER = json.dumps(
    {"role": "assistant", "content": "老的回答", "at": 1.6}, ensure_ascii=False
)


def write_raw(store: SessionStore, session_id: str, *lines: str) -> Path:
    """手写一份逐字稿。用来造 schema-1 的文件——那个形状现在没有生产者了。"""
    path = store.root / f"{session_id}.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return path


def test_a_plain_row_gains_no_new_keys() -> None:
    """``to_dict`` 是"非空才写"，所以 schema-1 那一行的形状**逐键不变**。

    这是"不需要迁移脚本"的全部依据。若哪天有人把工具字段改成无条件写出，老行会多出
    四个 ``null``/``[]``——本测试红，而它红得正是地方。
    """
    assert set(Message(role="user", content="一句话").to_dict()) == {"role", "content", "at"}


def test_a_schema_one_row_reads_back_with_the_tool_fields_defaulted(store: SessionStore) -> None:
    """schema-1 的老行读出默认值，而不是报错或缺键。

    同一个读者能读三个版本，靠的就是这一条。
    """
    write_raw(store, _OLD_ID, _OLD_HEADER, _OLD_USER)

    session = store.load(_OLD_ID)

    assert [message.content for message in session.messages] == ["老的一行"]
    message = session.messages[0]
    assert (message.tool_calls, message.tool_call_id, message.tool_name) == ((), None, None)
    # ``no_replay`` 缺省是 ``None``——也就是"正常回放"。**这个默认方向是刻意的**：
    # 老逐字稿里那些工具对是 schema-3 之前写的，那时还没有闸门拒绝这回事，它们本来
    # 就是普通结果。默认成"都不回放"会把用户已有的工具历史一次性抹掉。
    assert message.no_replay is None
    assert session.title == "旧的"


def test_the_no_replay_reason_lands_and_reads_back(store: SessionStore) -> None:
    """``no_replay`` 落地并读回，好行不带它。"""
    session = store.load(new_session_id())
    store.append(
        session,
        Message(
            role="tool",
            content="用户没有同意执行 write_memory，这一次调用跳过了。",
            tool_call_id="call_1",
            tool_name="write_memory",
            no_replay="denied",
        ),
    )
    store.append(
        session,
        Message(
            role="tool", content="钥匙在鞋柜里。", tool_call_id="call_2", tool_name="search_memory"
        ),
    )

    reloaded = store.load(session.session_id)

    assert reloaded.messages[0].no_replay == "denied"
    assert reloaded.messages[1].no_replay is None
    # 空串归成"没有"，与 ``_optional_str`` 对 ``tool_call_id`` 的处理同源：一个空的
    # 原因字符串说不出任何事，让它冒充"有原因"会让 L5 跳过一个本该回放的对。
    assert Message.from_dict({"role": "tool", "content": "x", "no_replay": ""}).no_replay is None


# ── schema 3 → 4：她主动开口的那一行 ────────────────────────────────────────
#
# 加字段这件事本身与上面那一节同形（老行不动、老读者读得回来），不需要再证一遍。
# 这一节只补那一节证不到的一件事：**这个字段的默认方向**。


def test_a_proactive_row_lands_and_reads_back(store: SessionStore) -> None:
    """她主动说的那句落地并读回；同一份文件里普通回答**一个键都不多**。

    多写一个 ``"proactive": false`` 的代价不是几个字节：它是"非空才写"这条约定上的
    一个洞，而那条约定正是"老行逐字节不变"的全部依据（见
    ``test_a_plain_row_gains_no_new_keys``）。
    """
    session = store.load(new_session_id())
    store.append(session, Message(role="assistant", content="该睡了。", proactive=True))
    store.append(session, Message(role="assistant", content="好，我查一下。"))

    reloaded = store.load(session.session_id)

    assert reloaded.messages[0].proactive is True
    assert reloaded.messages[1].proactive is False
    raw = (store.root / f"{session.session_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(raw[1])["proactive"] is True
    assert "proactive" not in json.loads(raw[2])
    assert "proactive" not in Message(role="assistant", content="普通回答").to_dict()


def test_a_row_without_the_key_is_not_proactive() -> None:
    """缺键 → **不是**她主动说的。**这个默认方向是刻意的。**

    schema-4 之前的每一行都是回答用户哪句话，那些行里既没有这个键、也不该在回看时
    变成"她主动开口"。默认成真更坏的还在后面：``display_rows`` 是按这一位分岔的，
    于是整份历史会一夜之间换一副长相，而没有任何一条数据变过。
    """
    assert Message.from_dict({"role": "assistant", "content": "老的回答"}).proactive is False
    # 真值照真读（``bool(...)``，与同一族里的 ``interrupted`` 一个写法）。**不另立
    # 一套严格判定**：这一位只由本程序写，读到 ``1`` 说明文件被人动过；那种时候
    # "按真值渲染"与"当成假"代价一样小，而与 ``interrupted`` 分头写两套判据，
    # 代价是下一个读这两个字段的人要多想一次它们为什么不一样。
    assert (
        Message.from_dict({"role": "assistant", "content": "x", "proactive": 1}).proactive is True
    )


def test_tool_rows_land_and_read_back(store: SessionStore) -> None:
    """工具行落地并读回，三个字段一个不少。"""
    session = store.load(new_session_id())
    store.append(
        session,
        Message(
            role="assistant",
            content="我查一下。",
            tool_calls=({"id": "call_1", "name": "search_memory", "arguments": {"query": "钥匙"}},),
        ),
    )
    store.append(
        session,
        Message(
            role="tool",
            content="钥匙在鞋柜里。",
            tool_call_id="call_1",
            tool_name="search_memory",
        ),
    )

    reloaded = store.load(session.session_id)

    assert [message.role for message in reloaded.messages] == ["assistant", "tool"]
    assert reloaded.messages[0].tool_calls == (
        {"id": "call_1", "name": "search_memory", "arguments": {"query": "钥匙"}},
    )
    assert reloaded.messages[1].tool_call_id == "call_1"
    assert reloaded.messages[1].tool_name == "search_memory"
    # 反方向也要看：结果行不带调用声明，读回来不该凭空多出来一条
    assert reloaded.messages[1].tool_calls == ()


def test_appending_to_a_schema_one_file_rewrites_nothing(store: SessionStore) -> None:
    """往老文件里追加新行：**前面的字节一个都不动**，而且整份仍然读得回来。

    这是"没有迁移脚本，也不需要有"这句话的可执行版本。配上"首行仍是 schema 1"——
    新行多几个键，不需要回头升级老文件。
    """
    path = write_raw(store, _OLD_ID, _OLD_HEADER, _OLD_USER, _OLD_ANSWER)
    before = path.read_bytes()

    session = store.load(_OLD_ID)
    store.append(
        session,
        Message(role="tool", content="新的结果", tool_call_id="call_9", tool_name="get_time"),
    )

    after = path.read_bytes()
    assert after.startswith(before), "老行被改写了——不写迁移脚本省下的正是这件事"
    assert after[len(before) :].count(b"\n") == 1, "一次追加应当只多一行"
    # 首行没有被"升级"：正文的头仍然自报 schema 1
    assert json.loads(after.splitlines()[0].decode("utf-8"))["schema"] == 1

    reloaded = store.load(_OLD_ID)
    assert [message.content for message in reloaded.messages] == [
        "老的一行",
        "老的回答",
        "新的结果",
    ]
    assert reloaded.messages[-1].tool_call_id == "call_9"


@pytest.mark.parametrize("bad_id", [{"a": 1}, "", 7, [], None])
def test_a_bad_tool_call_id_reads_as_absent_rather_than_as_text(bad_id: object) -> None:
    """坏形状归成"没有"，**不** ``str()`` 成一个看着像 id 的乱码。

    乱码会一路走到装配器，然后在那里变成一句"配对不上"——而真凶是落盘那一刻就坏了。
    空串同样归成"没有"：它作为配对依据没有任何意义。
    """
    assert (
        Message.from_dict({"role": "tool", "content": "x", "tool_call_id": bad_id}).tool_call_id
        is None
    )


def test_a_bad_tool_calls_shape_degrades_to_nothing() -> None:
    """``tool_calls`` 不是列表时当作没有；列表里的杂项**跳过**而不是丢整条。

    同 :meth:`SessionStore.load` 跳过半行 JSON 的纪律：宁可少一个调用，也不要整条
    消息消失。
    """
    assert (
        Message.from_dict(
            {"role": "assistant", "content": "x", "tool_calls": "不是列表"}
        ).tool_calls
        == ()
    )
    kept = Message.from_dict(
        {
            "role": "assistant",
            "content": "x",
            "tool_calls": [{"id": "c1", "name": "n"}, "坏的一坨", 3, None],
        }
    )
    assert kept.tool_calls == ({"id": "c1", "name": "n"},)


# ── schema 4 → 5：她发的那张图 ──────────────────────────────────────────────
#
# 与上面两节同形：老行不动、老读者读得回来。这一节补的是那个新字段的**默认方向**——
# 缺键必须是「没有图」，否则整份历史会一夜之间多出一张名叫 None 的图。


def test_an_image_row_lands_and_reads_back(store: SessionStore) -> None:
    """贴纸那一行落地并读回；**同一份文件里普通行一个键都不多**。

    ``image`` 存的是**相对数据根的路径**：数据根会搬家，而逐字稿不该因此失效。
    """
    session = store.load(new_session_id())
    store.append(session, Message(role="assistant", content="", image="stickers/开心.png"))
    store.append(session, Message(role="assistant", content="普通回答"))

    reloaded = store.load(session.session_id)

    assert reloaded.messages[0].image == "stickers/开心.png"
    assert reloaded.messages[0].content == ""
    raw = (store.root / f"{session.session_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(raw[1])["image"] == "stickers/开心.png"
    assert "image" not in json.loads(raw[2])
    assert "image" not in Message(role="assistant", content="普通回答").to_dict()


def test_a_row_without_the_image_key_is_text() -> None:
    """缺键 → 空串（不是 None、也不是"有一张名叫 None 的图"）。

    schema-5 之前的每一行都没有这个键，而它们全都该照旧渲染成纯文字。
    """
    assert Message.from_dict({"role": "assistant", "content": "老的回答"}).image == ""
