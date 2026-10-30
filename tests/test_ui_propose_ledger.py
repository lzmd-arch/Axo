"""提议台账窗口（P4b）：行数据是纯函数，窗口只验"建得起来、撤销接得上"。

Qt 的坑在这里是具体的：顶层窗口不关会拖住进程，所以每个用例都 ``close()`` 并在
fixture 里统一收尾——与 ``test_ui_consolidation_chain`` 同一套写法。
"""

from __future__ import annotations

from typing import Any

import pytest

from axo.propose_consumer import ProposeConsumer
from axo.ui.propose_ledger import ProposeLedgerWindow, ledger_rows
from memcore.store import MemoryStore


def _consumer(tmp_path) -> ProposeConsumer:
    return ProposeConsumer(
        MemoryStore(":memory:"),
        tmp_path / "queue.sqlite3",
        audit_path=tmp_path / "audit.jsonl",
    )


def test_ledger_rows_summarise_the_queue(tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("the trigger fired 3 times across 20 turns", proposal_id="p1", source="user")
    consumer.ingest("ok", proposal_id="p2", source="llm")
    consumer.decide_pending()

    rows = ledger_rows(consumer)
    assert len(rows) == 2
    assert rows[0][1] == "llm"  # 倒序：最新的一条先出现
    assert {row[3] for row in rows} == {"已接受", "待审"}


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.gui
def test_window_shows_the_rows_and_undoes_a_proposal(qapp, tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("the trigger fired 3 times across 20 turns", proposal_id="p1", source="user")
    consumer.decide_pending()
    assert consumer.store.count() == 1

    window = ProposeLedgerWindow(consumer)
    try:
        assert window.table.rowCount() == 1
        window.table.selectRow(0)
        assert window.undo_button.isEnabled() is True

        window._on_undo_clicked()
        assert consumer.store.count() == 0
        assert consumer.queue.get(1)["status"] == "reverted"
        assert window.table.rowCount() == 1
        assert ledger_rows(consumer)[0][3] == "已撤销"
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.gui
def test_undo_is_disabled_for_a_pending_row(qapp, tmp_path):
    consumer = _consumer(tmp_path)
    consumer.ingest("ok", proposal_id="p1", source="llm")
    consumer.decide_pending()

    window = ProposeLedgerWindow(consumer)
    try:
        window.table.selectRow(0)
        assert window.undo_button.isEnabled() is False
    finally:
        window.close()
        window.deleteLater()
