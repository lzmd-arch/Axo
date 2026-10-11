"""巩固事件：一行一条、自足、可逐条比对。

这个文件的重心是**契约**，不是巩固算法。``ConsolidationEvent`` 的 docstring 从
P4c 之前就写着「把该轮所有事件按序落库」「``--replay <run_id>`` 能用同一个
``run_seed`` 重跑一遍并逐条比对」——这个文件就是来兑现那两句话的，所以它盯的是
三件事：

1. **行的形状**（字段名、字段序、``seq`` 的起点）——它是落盘格式的规格，改了它
   已经写出去的历史事件就读不回来了，所以拿字面量钉死，不拿"跑一遍看看"当期望。
2. **一行自足**（``run_id`` / ``run_seed`` / ``seq`` 都在行里，没有文件头）——
   文件被截断时剩下的行仍然可用，这是审计证据该有的性质。
3. **比对的口径**——比什么、不比什么，每一处"不比"都得有理由。写得含糊的比对会
   在第一次重跑时给人一堆假警报，然后所有人开始忽略它。

比对的两侧是**刻意不对称**的：记录那一侧是**手写的字面量**，重跑那一侧才由被测
代码生成。两侧都现算的话，" ``seq`` 从 1 起"这类改动会同时改掉两侧，比对照样全绿
——那正是这一片的验红要红的那个 bug:验红时把 ``seq`` 改成从 1 起，红的就是这个
文件里拿字面量当期望的那几条。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcore.consolidation import (
    EPOCH,
    EVENT_FIELDS,
    ReplayOutcome,
    RunReport,
    append_events,
    cluster_event_rows,
    compare_events,
    event_rows,
    events_of,
    read_events,
)
from memcore.errors import MemCoreError

_AT = 1_760_000_000.0
_RUN_ID = "0123456789abcdef"
_RUN_SEED = 987654321


def _outcome(memory_id: str, decision: str, **changes: object) -> ReplayOutcome:
    base = ReplayOutcome(memory_id=memory_id, decision=decision, jaccard=1.0)
    return ReplayOutcome(**{**vars(base), **changes})  # type: ignore[arg-type]


def _report(*clusters: tuple[ReplayOutcome, ...], **changes: object) -> RunReport:
    """手搭一份报告。

    不走 :func:`~memcore.consolidation.consolidate`：这个文件测的是**序列化与
    比对**，把基质与分簇拉进来会让"某条断言为什么红"变成需要先读三层才能回答的
    问题。真的跑一轮由 ``tests/test_consolidation_run.py`` 覆盖。
    """
    fields: dict[str, object] = {
        "run_id": _RUN_ID,
        "run_seed": _RUN_SEED,
        "at": _AT,
        "clusters": clusters,
    }
    return RunReport(**{**fields, **changes})  # type: ignore[arg-type]


# ── 行的形状 ────────────────────────────────────────────────────────────────


def test_a_row_has_exactly_the_declared_fields_in_the_declared_order() -> None:
    """字段名与**顺序**就是规格。多一个少一个都算改了格式。

    顺序也是规格的一部分：前三个回答"哪一轮、第几条"，中间是决策，``detail``
    压最后——它是唯一会长大的字段，放前面会把决策挤到屏幕外。用 :data:`EVENT_FIELDS`
    而不是再抄一遍字面量，是为了让"规格"只有一处；下面那条用例把手写的行与它对齐。
    """
    report = _report((_outcome("m000", "reinforce"),))

    (row,) = event_rows(report)

    assert tuple(row) == EVENT_FIELDS
    assert len(EVENT_FIELDS) == len(set(EVENT_FIELDS)), "字段重名了"


def test_the_events_are_the_outcomes_in_cluster_order() -> None:
    """事件 = 决策，**按簇序**，一条不多一条不少。

    ``ConsolidationEvent`` 在 P4c 之前是全仓零生产者的空壳；这条钉住它被真的
    填上了值，而不是另起一套字段名相近的东西。
    """
    report = _report(
        (_outcome("m000", "reinforce"), _outcome("m001", "skip", jaccard=None)),
        (_outcome("m002", "sleep_confusion", jaccard=0.31, confused_with="m003"),),
    )

    events = events_of(report)

    assert [event.memory_id for event in events] == ["m000", "m001", "m002"]
    assert [event.decision for event in events] == ["reinforce", "skip", "sleep_confusion"]
    assert events[2].confused_with == "m003"
    assert all(event.run_id == report.run_id for event in events)
    assert all(event.at == report.at for event in events)
    assert all(event.epoch == EPOCH == 0 for event in events), "v1 单轮，epoch 恒 0"


def test_seq_starts_at_zero_and_runs_across_clusters() -> None:
    """``seq`` 从 **0** 起，跨簇连续——它是**下标**不是计数。

    **这条是本片的验红目标**：把 ``_rows_of`` 的 ``seq_from + offset`` 改成
    ``seq_from + offset + 1``（"第 1 条"的读法），红的就是这里。两种起点各自都能
    自洽，所以必须在某一处定死；定死在这里，是因为已经落盘的历史文件按 0 起排。
    """
    report = _report(
        (_outcome("m000", "reinforce"), _outcome("m001", "reinforce")),
        (_outcome("m002", "reinforce"), _outcome("m003", "reinforce"), _outcome("m004", "skip")),
    )

    assert [row["seq"] for row in event_rows(report)] == [0, 1, 2, 3, 4]


def test_a_single_cluster_slice_carries_the_round_wide_seq() -> None:
    """按簇取行时 ``seq`` 接着上一簇往下数，不从 0 重来。

    这条守的是崩溃时的形状：真跑时每簇"改库 + 追加事件"在同一次循环里完成，
    中途挂掉留下的是 ``seq`` 连续的**前缀**。每簇各自从 0 起的话，前缀里的
    ``seq`` 会是 ``0,1,0,1,0``——看起来像三份互相独立的记录，而比对要的是
    一整轮的顺序。
    """
    report = _report(
        (_outcome("m000", "reinforce"), _outcome("m001", "reinforce")),
        (_outcome("m002", "reinforce"), _outcome("m003", "skip")),
    )

    first = cluster_event_rows(report, 0)
    second = cluster_event_rows(report, 1)

    assert [row["seq"] for row in first] == [0, 1]
    assert [row["seq"] for row in second] == [2, 3]
    assert first + second == event_rows(report), "分簇写出去的行必须与整轮的一致"


def test_a_row_is_plain_json_ready_data() -> None:
    """行里不许有非 JSON 的东西（元组、SDR、枚举）。

    这是 ``--json`` 输出与 JSONL 落盘共用的前提；一个元组混在 ``detail`` 里，
    写出去会变成数组、读回来是列表，于是"重跑逐条相同"这件事在**没改任何逻辑**的
    情况下变成假的。``sort_keys=False`` 也要保持——字段序是规格。
    """
    report = _report((_outcome("m000", "skip", jaccard=None, detail={"reason": "cold"}),))

    (row,) = event_rows(report)

    text = json.dumps(row, ensure_ascii=False)
    assert json.loads(text) == row
    assert row["detail"] == {"reason": "cold"}
    assert row["jaccard"] is None, "不适用时是 null，不是 0.0"
    assert row["confused_with"] is None


# ── 落盘与读回 ──────────────────────────────────────────────────────────────


def test_a_written_round_reads_back_identically(tmp_path: Path) -> None:
    """写出去再读回来，逐行等于原样。

    这条是"文件就是证据"的地基：读回来的东西只要与写出去的不同，后面所有比对
    都在比一个被序列化改过的东西。
    """
    report = _report(
        (_outcome("m000", "reinforce"), _outcome("m001", "skip", detail={"reason": "cold"})),
        (_outcome("m002", "sleep_confusion", jaccard=0.42, confused_with="m003"),),
    )
    path = tmp_path / "runs" / f"{report.run_id}.jsonl"

    written = append_events(event_rows(report), path)

    assert written == 3
    assert read_events(path) == event_rows(report)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3, "一行一条，不换行"


def test_appending_twice_accumulates_instead_of_overwriting(tmp_path: Path) -> None:
    """第二次追加**接在后面**，不覆盖。

    分簇写出去就是这个形状：一轮里调用 n 次，每次一簇。用 ``"w"`` 打开的话，
    最后一簇会把前面所有簇的事件抹掉——而库里那些记忆已经被加固过了，
    事件却没了，这正是审计最不能接受的那种不一致。
    """
    report = _report(
        (_outcome("m000", "reinforce"),),
        (_outcome("m001", "reinforce"),),
    )
    path = tmp_path / f"{report.run_id}.jsonl"

    append_events(cluster_event_rows(report, 0), path)
    append_events(cluster_event_rows(report, 1), path)

    assert read_events(path) == event_rows(report)


def test_a_round_with_no_events_writes_no_file(tmp_path: Path) -> None:
    """空轮**不落空文件**。文件存在 = 那一轮确实跑出了事件。

    空文件会在列表里看起来像"跑过"，而 ``--replay`` 读它得到零条、与零条比、
    报"复现成功"——一场什么都没发生的胜利。与 ``paths.reminders_path`` 同姿态：
    路径函数不预建文件。
    """
    path = tmp_path / "empty.jsonl"

    assert append_events((), path) == 0
    assert not path.exists()


def test_a_missing_file_says_so_instead_of_looking_like_an_empty_round(tmp_path: Path) -> None:
    """没有文件是**错误**，不是空轮。两者的处置完全不同：一个是 run_id 写错了，
    一个是那一轮真的什么都没发生。"""
    with pytest.raises(MemCoreError, match="没有这一轮的事件文件"):
        read_events(tmp_path / "nope.jsonl")


def test_a_corrupt_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    """坏行当场报错，并指出**第几行**。

    跳过坏行会让 ``--replay`` 拿一份短了的记录去比对，然后报"序列不一致"——把
    责任推给巩固的确定性，而真正的问题是这个文件。行号是唯一能让人去修它的线索。
    """
    path = tmp_path / "run.jsonl"
    path.write_text('{"seq": 0}\n{不是 JSON\n{"seq": 2}\n', encoding="utf-8", newline="\n")

    with pytest.raises(MemCoreError, match="第 2 行"):
        read_events(path)


def test_a_line_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    """合法 JSON、但不是对象——同样拒掉。事件行永远是一个对象。"""
    path = tmp_path / "run.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8", newline="\n")

    with pytest.raises(MemCoreError, match="不是对象"):
        read_events(path)


def test_blank_lines_are_skipped_not_treated_as_corrupt(tmp_path: Path) -> None:
    """尾部空行（编辑器加的那个）不算坏数据——否则人人都会撞上这个错。"""
    path = tmp_path / "run.jsonl"
    path.write_text('{"seq": 0, "memory_id": "m000"}\n\n', encoding="utf-8", newline="\n")

    assert len(read_events(path)) == 1


# ── 比对的口径 ──────────────────────────────────────────────────────────────


#: **记录那一侧**，手写的。字段与 :data:`EVENT_FIELDS` 对齐（由
#: :func:`test_the_recorded_fixture_matches_the_declared_schema` 钉住），值取自一份
#: 真实跑过的轮次。写死而不是现算，是为了让"重跑"那一侧真的有可能对不上。
_RECORDED: tuple[dict[str, object], ...] = (
    {
        "run_id": _RUN_ID,
        "run_seed": _RUN_SEED,
        "seq": 0,
        "memory_id": "m000",
        "decision": "reinforce",
        "at": _AT,
        "epoch": 0,
        "jaccard": 1.0,
        "confused_with": None,
        "detail": {"probe_jaccard": 0.5},
    },
    {
        "run_id": _RUN_ID,
        "run_seed": _RUN_SEED,
        "seq": 1,
        "memory_id": "m001",
        "decision": "sleep_confusion",
        "at": _AT,
        "epoch": 0,
        "jaccard": 0.42,
        "confused_with": "m002",
        "detail": {"probe_jaccard": 0.49},
    },
    {
        "run_id": _RUN_ID,
        "run_seed": _RUN_SEED,
        "seq": 2,
        "memory_id": "m002",
        "decision": "skip",
        "at": _AT,
        "epoch": 0,
        "jaccard": None,
        "confused_with": None,
        "detail": {"reason": "cold"},
    },
)


def _replayed() -> tuple[dict[str, object], ...]:
    """重跑那一侧——**由被测代码生成**，形状与 ``_RECORDED`` 同源。"""
    return event_rows(
        _report(
            (_outcome("m000", "reinforce", detail={"probe_jaccard": 0.5}),),
            (
                _outcome(
                    "m001",
                    "sleep_confusion",
                    jaccard=0.42,
                    confused_with="m002",
                    detail={"probe_jaccard": 0.49},
                ),
                _outcome("m002", "skip", jaccard=None, detail={"reason": "cold"}),
            ),
        )
    )


def test_the_recorded_fixture_matches_the_declared_schema() -> None:
    """手写的那份记录与 :data:`EVENT_FIELDS` 对齐——否则下面的比对测的是别的东西。

    没有这条，``_RECORDED`` 里少一个字段的话，``compare_events`` 会因为
    ``None != 值`` 而红，看起来像"复现失败"，实际是夹具打错了字。
    """
    for row in _RECORDED:
        assert tuple(row) == EVENT_FIELDS


def test_a_replay_of_the_same_round_compares_equal() -> None:
    """逐条相同则一条问题都不报。**空元组 = 复现成功。**

    这是 ``--replay`` 的全部意义所在：一轮巩固若不能重跑出同一串事件，
    ``ConsolidationEvent`` 的 docstring 里那两句话就是假的。
    """
    assert compare_events(_RECORDED, _replayed()) == ()


def test_a_flipped_decision_is_named_with_its_index() -> None:
    """决策变了 → 报出**第几条、哪个字段、两侧各是什么**。

    返回值是字符串而不是 ``bool``，就是为了这一条：一轮不可复现是要去查的，
    而"第 1 条：decision 记录是 sleep_confusion、重跑是 reinforce"直接指向现场。
    """
    broken = list(_replayed())
    broken[1] = {**broken[1], "decision": "reinforce"}

    (problem,) = compare_events(_RECORDED, broken)

    assert "第 1 条" in problem
    assert "decision" in problem
    assert "sleep_confusion" in problem and "reinforce" in problem


def test_a_shifted_sequence_is_reported_as_an_identity_mismatch() -> None:
    """``seq`` 整体错位（或整体加一）→ 红。

    这正是本片验红时要红的那个 bug：把 ``seq`` 从 0 起改成 1 起，
    :func:`_rows_of` 生成的那一侧全体加一，而记录那一侧还是 0 起。
    """
    shifted = [{**row, "seq": int(row["seq"]) + 1} for row in _replayed()]  # type: ignore[arg-type]

    problems = compare_events(_RECORDED, shifted)

    assert len(problems) == 3, "三条的 seq 全对不上，就该报三条"
    assert all("seq" in problem for problem in problems)


def test_jaccard_is_compared_with_a_tolerance() -> None:
    """``jaccard`` 按容差比：抖动在 1e-6 以内放过，超了报出来。

    浮点写出去再读回来本该逐位相同，但这一列是**连续量**，将来某版若改成
    float32 存就会差一点点——那时不该出现一屏假警报。而 0.42 与 0.44 的差别
    是"串扰有多远"的真实差别，必须报。
    """
    nudged = [
        {**row, "jaccard": 1.0000001 if row["jaccard"] == 1.0 else row["jaccard"]}
        for row in _replayed()
    ]
    assert compare_events(_RECORDED, nudged) == ()

    shifted = list(_replayed())
    shifted[1] = {**shifted[1], "jaccard": 0.44}
    (problem,) = compare_events(_RECORDED, shifted)
    assert "jaccard" in problem


def test_the_timestamp_and_the_run_identity_are_not_compared() -> None:
    """``at`` / ``run_id`` / ``run_seed`` / ``detail`` **不比**，各有理由。

    重跑发生在另一个时刻：``at`` 天然不同，``run_id`` 与 ``run_seed`` 又都由
    ``at`` 派生——拿它们比对等于要求"重跑必须发生在同一秒"，那这个功能就没法用了。
    真正要防的"抽样顺序变了"会在 ``memory_id`` 那一列上显形。``detail`` 里的
    ``reinforce_before`` 是**库的当前状态**，重跑时库已经加过了。

    这条把四处"不比"一次钉住：将来谁想顺手把它们加进比对，会先看到这条用例
    和它上面这些理由。
    """
    other = [
        {
            **row,
            "at": 1_800_000_000.0,
            "run_id": "ffffffffffffffff",
            "run_seed": 1,
            "detail": {"完全不同": "的证据"},
        }
        for row in _replayed()
    ]

    assert compare_events(_RECORDED, other) == ()


def test_a_shorter_or_longer_replay_is_reported_row_by_row() -> None:
    """长度对不上也逐条报：缺的说"没跑出来"，多的说"多出来了"。

    只说一句"条数不同"的话，看不出是**尾部少了一簇**（可能是崩溃留下的前缀，
    那是有意义的）还是中间错位。
    """
    shorter = _replayed()[:1]
    longer = _replayed() + (_replayed()[0],)

    missing = compare_events(_RECORDED, shorter)
    extra = compare_events(_RECORDED, longer)

    assert missing == ("第 1 条：记录里有，重跑没跑出来", "第 2 条：记录里有，重跑没跑出来")
    assert extra == ("第 3 条：记录里没有，重跑多出来了",)
