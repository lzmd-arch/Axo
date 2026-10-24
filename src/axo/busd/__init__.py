"""Axo 的常驻脉冲总线 daemon —— **独立进程**，不是 UI 里的一个线程。

架构（用户 2026-10-24 拍板）：

    Axo 主程序（Qt）  ← 只读事件文件，不 import 本包的任何运行时依赖
          ↑  IPC（本地文件事件，单向）
    SNN daemon（本包，``python -m axo.busd``）
      ├─ 实例化 ``biosnn_bus.SpikeBus``（真总线，真 ``step()``）
      ├─ 新记忆 → :class:`~axo.busd.plugin.MemorySpikePlugin` 编码 → 融合 → 发放状态落盘
      ├─ 巩固 / "做梦"：调用 ``axo.memory.consolidation.run_consolidation``（同一份实现）
      └─ 涌现阈值 → 主动事件（问候建议）→ 写进 ``bus/events.jsonl``

**为什么是独立进程**：总线要一直醒着，而 GUI 会关、会重启、会在退出时等线程。
把总线放进 UI 进程，等于把它绑在"窗口开着"这件事上；而 ``memcore.substrate``
那段话已经说清楚：只把 SDR 转成 ``SpikeTrain`` 不算接上总线。真的接上 = 有一个
进程持续 ``step()``，并且**它的状态是别人读得到的**（:mod:`axo.busd.state`）。

**UI 只依赖** :mod:`axo.busd.events`（纯 stdlib 的 JSONL 读写），不 import
numpy / biosnn_bus / 本包的 daemon —— 这条是"只接收，不依赖"的可断言形式。
"""
