"""提议台账窗口：看外部客户端提了什么、门控怎么判的、要不要撤销（P4b）。

窗口本身只做展示与一次点击；判定与落库仍只在 :mod:`axo.propose_consumer` 里。
表格的每一行对应队列里的一条提议，撤销就是调 ``consumer.revert(seq, reason)``——
**界面不是第二条写路径**，它只是把那条已存在的撤销调用摆到人面前。

行数据由 :func:`ledger_rows` 负责（纯函数、无 Qt），所以"表格里显示什么"能在没有
事件循环的测试里直接验；窗口部分的测试只验"建得起来、行数对、关得掉"。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..propose_consumer import ProposeConsumer

__all__ = ["ProposeLedgerWindow", "ledger_rows", "STATUS_LABELS"]

STATUS_LABELS = {
    "pending": "待审",
    "accepted": "已接受",
    "rejected": "已拒绝",
    "reverted": "已撤销",
}

#: 表格里给正文留的宽度字符数——太长的提议不该把一张表撑成横向滚动条。
_TEXT_LIMIT = 60


def ledger_rows(consumer: ProposeConsumer, *, limit: int = 50) -> list[list[str]]:
    """把队列行摊成表格单元：时间、来源、正文、判定、理由。"""
    rows: list[list[str]] = []
    for row in consumer.queue.list(limit=limit):
        payload = row["payload"]
        text = str(payload.get("text", ""))
        if len(text) > _TEXT_LIMIT:
            text = text[: _TEXT_LIMIT - 1] + "…"
        rows.append(
            [
                str(row["created_at"]).replace("T", " ").replace("Z", ""),
                str(payload.get("source", "?")),
                text,
                STATUS_LABELS.get(row["status"], row["status"]),
                str(row["reason"]),
            ]
        )
    return rows


class ProposeLedgerWindow(QDialog):
    """只读的提议台账 + 撤销。撤销要求先选中一行，且那一行必须是「已接受」。"""

    def __init__(self, consumer: ProposeConsumer, parent: Any = None) -> None:
        super().__init__(parent)
        self._consumer = consumer
        self.setWindowTitle("提议台账")
        self.resize(880, 460)

        layout = QVBoxLayout(self)
        self.summary = QLabel("", self)
        self.summary.setObjectName("proposeSummary")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, 5, self)
        self.table.setObjectName("proposeTable")
        self.table.setHorizontalHeaderLabels(["时间", "来源", "内容", "判定", "理由"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.approve_button = QPushButton("批准选中的提议", self)
        self.approve_button.setObjectName("proposeApproveButton")
        self.approve_button.setEnabled(False)
        self.approve_button.setToolTip("把一条「待审」的提议按你的判断放行，写进记忆")
        self.approve_button.clicked.connect(self._on_approve_clicked)
        buttons.addWidget(self.approve_button)
        self.undo_button = QPushButton("撤销选中的提议", self)
        self.undo_button.setObjectName("proposeUndoButton")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        buttons.addWidget(self.undo_button)
        buttons.addStretch(1)
        self.close_button = QPushButton("关闭", self)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh()

    # ------------------------------------------------------------ 展示
    def refresh(self) -> None:
        rows = ledger_rows(self._consumer)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, value)
                self.table.setItem(index, column, item)
        status = self._consumer.status()
        counts = status["counts"]
        self.summary.setText(
            "待审 {p} · 已接受 {a} · 已拒绝 {r} · 已撤销 {v} · 库里共 {n} 条".format(
                p=counts["pending"],
                a=counts["accepted"],
                r=counts["rejected"],
                v=counts["reverted"],
                n=status["entries"],
            )
        )
        self._on_selection_changed()

    # ------------------------------------------------------------ 撤销
    def selected_seq(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        position = rows[0].row()
        # 队列按 seq 倒序列出，所以第 0 行是最新的一条
        proposals = self._consumer.queue.list(limit=50)
        if position >= len(proposals):
            return None
        return int(proposals[position]["seq"])

    def _on_selection_changed(self) -> None:
        seq = self.selected_seq()
        status = self._consumer.queue.get(seq)["status"] if seq is not None else None
        self.undo_button.setEnabled(status == "accepted")
        # 「待审」必须有一个出口：只显示不处置的话，pending 永远变不成记忆
        self.approve_button.setEnabled(status == "pending")

    def _on_approve_clicked(self) -> None:
        seq = self.selected_seq()
        if seq is None:
            return
        try:
            payload = self._consumer.approve(seq, "approved from the proposal ledger")
        except ValueError as error:
            QMessageBox.warning(self, "批准失败", str(error))
            return
        self.refresh()
        self.summary.setText(self.summary.text() + f" · 已批准第 {payload['seq']} 条")

    def _on_undo_clicked(self) -> None:
        seq = self.selected_seq()
        if seq is None:
            return
        try:
            payload = self._consumer.revert(seq, "revoked from the proposal ledger")
        except ValueError as error:
            QMessageBox.warning(self, "撤销失败", str(error))
            return
        self.refresh()
        self.summary.setText(self.summary.text() + f" · 已撤销第 {payload['seq']} 条")
