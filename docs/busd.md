# 常驻脉冲总线 daemon（`axo.busd`）

> 状态：已落地（2026-10-24）。入口 `python -m axo.busd`。
> 依赖：`biosnn-bus>=0.4.0,<0.5`（2026-10-25 从 0.3.0 升级；
> `SpikeBus.step()` 返回 `float32`，见 ADR-0012 的修订记录）。

## 架构

```
Axo 主程序（Qt）  ← 只读 bus/events.jsonl，不 import daemon
      ↑ 文件事件（单向）
SNN daemon（独立进程）
  ├─ SpikeBus（真 biosnn-bus，真 step()）
  ├─ 新记忆 → 编码 → 融合 → bus/provenance.jsonl（含输入 sha256 与种子）
  ├─ 巩固（"做梦"）：调用 axo.memory.consolidation.run_consolidation（同一份实现）
  └─ 涌现判定 → greeting_suggestion → bus/events.jsonl
```

## 一轮做什么

1. 按 `memory_id` 游标取一批新记忆（只读，不改库）；
2. 每条取关联码（`encode_assoc`，k=64）→ `(T, 4096)`；
3. `SpikeBus.step({"memory": rows})` → `(T, bus_dim)`（ADR-0012：第二维恒为 `bus_dim`）；
4. 写 `bus/provenance.jsonl`：输入哈希、参数、输出形状、发放率、耗时；
5. 游标前进。

**游标只看新条目**：改一条、删一条不会被看见（与巩固那道门同一条取舍）。

## 涌现判定

`novelty = 1 - cos(本次发放向量, 上次发放向量)`，同时要求活动量
`mean(|rates|) ≥ floor`。第一次跑没有基线，`novelty` 记 0 且不触发。
**两个阈值是初值，没有标定过**（`emergence_novelty=0.35`、`emergence_floor=0.0`）。

## L3 涌现触发（接口桩，默认关闭）

`axo.busd.curiosity` 定义 `EmergentRetrieval.propose(...) -> RetrievalIntent | None`。
**默认关闭**（`BusDaemonConfig.curiosity_enabled=False`）；打开时必须注入一个 proposer，
否则构造就报错。daemon 只把意图写进 `bus/events.jsonl`（kind=`retrieval_intent`）：
**不抓取、不写记忆、不调用任何工具**。

边界与下一步：

- 这是「机制化主动，涌现待验」的挂点，**不是**"她已经有好奇心"的证据；
- 将来的执行器（agent 的工具面）读这个事件 → Search/Fetch → 原文进 `fetch-cache`、
  摘要经 daemon 的巩固路径入库；写入仍只过一个门；
- 当前默认关，所以正常运行时不会出现 `retrieval_intent` 事件。

## 文件

| 路径 | 内容 |
| --- | --- |
| `bus/state.json` | 游标、计数、上次输出形状/活动量/新颖度、上次错误 |
| `bus/provenance.jsonl` | 每轮一行，可复现（种子 + 输入哈希） |
| `bus/events.jsonl` | 给 UI 的事件（`greeting_suggestion` / `retrieval_intent` / `dream_done` / 启停） |
| `bus/last_rates.npy` | 上一次的 `(bus_dim,)` 发放向量，算 novelty 用 |

## 启停

- GUI 启动时按设置 `bus/enabled`（默认开）拉起子进程，退出时显式收掉；
- 打包版没有可用的解释器，**不会**拉起 daemon（状态栏会说明），但事件轮询仍在；
- Windows 上 `terminate()` 是硬杀，不会写 `daemon_stop`；状态每轮落盘，没有数据风险。

## 边界

- daemon **不学**任何东西：没有 STDP、没有反向传播、不把浮点电流说成 0/1 脉冲；
- 它不决定"要不要开口"：建议只是候选话头，开口与否仍归 UI 侧的闸门；
- 巩固与 UI 调度器共用 `consolidation/last.json` 同一本账，两边都开着不会同一晚跑两轮。
