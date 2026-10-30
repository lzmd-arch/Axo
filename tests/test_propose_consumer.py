"""提议消费者（P3）：提议只进队列，门控说了算，审计对得上 MCP 侧。

判据取自计划书 §4.4 的 A1–A6，能在无界面、无模型的环境里跑的那几条在这里：
A1 端到端、A2 拒绝路径、A4 审计、A6 未授权时写不进去。A3（界面撤销）在 P4。
"""

from __future__ import annotations

import json

import pytest

from axo.propose_consumer import (
    GateConfig,
    ProposalQueue,
    ProposeConsumer,
    build_consumer,
    salience,
)
from memcore.store import MemoryStore


def _consumer(tmp_path, **kwargs) -> ProposeConsumer:
    store = MemoryStore(":memory:")
    return ProposeConsumer(
        store,
        tmp_path / "queue.sqlite3",
        audit_path=tmp_path / "propose_audit.jsonl",
        **kwargs,
    )


def _events(path) -> list[str]:
    return [json.loads(line)["event"] for line in path.read_text("utf-8").splitlines()]


def test_a1_proposal_lands_after_the_gate(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("the trigger fired 3 times across 20 turns", proposal_id="p1", source="user")
    result = consumer.decide_pending()
    assert result["counts"]["accepted"] == 1
    assert result["entries"] == 1
    assert consumer.store.count() == 1
    assert any("trigger" in item.content for item in consumer.store.iter_items())


def test_a2_duplicate_is_rejected_and_the_store_is_untouched(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("the trigger fired 3 times across 20 turns", proposal_id="p1", source="user")
    consumer.decide_pending()
    assert consumer.store.count() == 1

    consumer.ingest("the trigger fired 3 times across 20 turns", proposal_id="p2", source="user")
    decided = consumer.decide_pending()
    assert decided["counts"]["rejected"] == 1
    assert decided["entries"] == 1
    assert "duplicate" in consumer.queue.get(2)["reason"]


def test_a4_audit_events_match_the_mcp_side(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest(
        "a memorable line about the consolidation gate", proposal_id="p1", source="user"
    )
    consumer.decide_pending()
    events = _events(tmp_path / "propose_audit.jsonl")
    assert events[:3] == ["proposal.received", "gate.accepted", "memory.applied"]


def test_a6_ingest_alone_never_writes(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("this text must not reach the store yet", proposal_id="p1", source="user")
    assert consumer.store.count() == 0
    assert consumer.status()["counts"]["pending"] == 1


def test_read_only_store_is_refused(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    store.close()
    readonly = MemoryStore(tmp_path / "mem.db", read_only=True)
    with pytest.raises(ValueError, match="可写库"):
        ProposeConsumer(readonly, tmp_path / "queue.sqlite3")


def test_short_llm_text_is_held_not_accepted(tmp_path):
    # llm + 太短：0.15 + 0 + 0.25 = 0.40，落在 0.30–0.60 的待审带
    consumer = _consumer(tmp_path)
    consumer.ingest("ok", proposal_id="p1", source="llm")
    consumer.decide_pending()
    assert consumer.queue.get(1)["status"] == "pending"
    assert consumer.store.count() == 0


def test_below_the_floor_is_rejected(tmp_path):
    # 同样的短句，但已经很像库里已有内容：0.15 + 0 + 0.05 = 0.20 < 0.30
    assert salience("ok", source="llm", max_similarity=0.8) < 0.30


def test_quota_holds_instead_of_dropping(tmp_path):
    consumer = _consumer(tmp_path, config=GateConfig(quota_hourly=0))
    consumer.ingest(
        "the gate accepted 20 proposals in the last hour", proposal_id="p1", source="user"
    )
    consumer.decide_pending()
    row = consumer.queue.get(1)
    assert row["status"] == "pending"
    assert "quota" in row["reason"]


def test_ingest_is_idempotent(tmp_path):
    consumer = _consumer(tmp_path)
    first = consumer.ingest(
        "a first proposal that enters the queue", proposal_id="same", source="user"
    )
    second = consumer.ingest(
        "a first proposal that enters the queue", proposal_id="same", source="user"
    )
    assert first["seq"] == second["seq"]
    assert second["replayed"] is True
    assert consumer.queue.counts()["pending"] == 1


def test_salience_weights_are_the_frozen_ones():
    # 与 biosnn 侧同一组常量：用户 > 客户端 > 模型推断
    body = "the scheduler fired once per conversation in 20 trials"
    assert salience(body, source="user", max_similarity=0.0) > salience(
        body, source="llm", max_similarity=0.0
    )


def test_queue_schema_is_the_shared_protocol(tmp_path):
    queue = ProposalQueue(tmp_path / "q.sqlite3")
    seq, created = queue.enqueue("x", {"text": "a", "tags": [], "source": "client"})
    assert created and queue.get(seq)["payload"]["text"] == "a"
    queue.decide(seq, "accepted", "fine")
    assert queue.counts()["accepted"] == 1


def test_a3_revert_takes_the_store_back(tmp_path):
    consumer = _consumer(tmp_path)
    before = consumer.store.count()
    consumer.ingest(
        "a proposal that will be accepted and then undone", proposal_id="p1", source="user"
    )
    consumer.decide_pending()
    assert consumer.store.count() == before + 1

    seq = consumer.queue.list()[0]["seq"]
    payload = consumer.revert(seq, "A3 rollback")
    assert payload["reverted"] is True
    assert consumer.store.count() == before
    assert consumer.queue.get(seq)["status"] == "reverted"


def test_revert_twice_is_refused(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("an entry that is reverted exactly once", proposal_id="p1", source="user")
    consumer.decide_pending()
    seq = consumer.queue.list()[0]["seq"]
    consumer.revert(seq, "first")
    with pytest.raises(ValueError, match="已经撤销过"):
        consumer.revert(seq, "second")


def test_revert_of_a_pending_proposal_is_refused(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("ok", proposal_id="p1", source="llm")
    consumer.decide_pending()
    seq = consumer.queue.list()[0]["seq"]
    with pytest.raises(ValueError, match="不是 accepted"):
        consumer.revert(seq, "nope")


def test_build_consumer_is_none_when_the_switch_is_off(tmp_path):
    # 「关着」必须是"不存在"，不是"存在但不写"——后者任何一条忘了检查开关的路径都能写
    assert (
        build_consumer(MemoryStore(":memory:"), enabled=False, queue_path=tmp_path / "q.sqlite3")
        is None
    )


def test_build_consumer_opens_the_queue_when_the_switch_is_on(tmp_path):
    store = MemoryStore(":memory:")
    consumer = build_consumer(store, enabled=True, queue_path=tmp_path / "q.sqlite3")
    assert consumer is not None
    consumer.ingest("a proposal from the switch-on path", proposal_id="p1", source="user")
    assert consumer.store.count() == 0
