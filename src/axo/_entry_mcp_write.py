"""冻结入口：``axo-mcp-write``（写档位：提议 → 门控 → 落库）。

绝对 import 的理由同 ``_entry_mcp_read.py``。设置没打开时 ``main_write`` 会打印提示并
以 2 退出——那是**预期行为**，不是崩溃。
"""

from axo.propose_server import main_write

raise SystemExit(main_write())
