# 编码任务基准

固定 8 个任务，判据不是"模型说通过"，而是**跑测试 + 保护文件哈希**：

| 任务 | 语言 | 内容 | 保护 |
| :--- | :--- | :--- | :--- |
| `bugfix_add` | Python | 修 `calc.py` 的加法 | `test_calc.py` 不得改 |
| `feature_slugify` | Python | 实现 `slugify` | `test_strings.py` 不得改 |
| `bugfix_cli` | Python | 修端口范围 | `test_cli.py` 不得改 |
| `multi_file_tax` | Python | 让 `pricing.py` 使用 `config.py` 的税率 | `test_pricing.py` 不得改；另有隐藏税率测试 |
| `multi_file_limiter` | Python | 让 `limiter.py` 使用 `config.py` 的限流窗口/次数 | 公开测试只有基础用例；隐藏测试覆盖次数与过期 |
| `parse_duration` | Python | 从零实现 `1h30m` 解析 | 公开测试一例；隐藏测试覆盖复合/秒/小时/非法输入 |
| `js_calc` | JavaScript | 修 `calc.js` 的加法（Node 内置 test runner） | `calc.test.js`、`package.json` 不得改；另有隐藏测试 |
| `multi_file_locate` | Python | 在 4 个模块里找到真正负责归一化的地方并修首尾空白 | `tests/test_service.py` 不得改；另有隐藏空白测试 |

Rust 的 `repo_map`/`run_tests` 支持已经落地，但**没有进基准**：本机 rustc 是 MSVC
工具链，而机器上没有 Visual Studio C++ build tools，`cargo test` 无法链接。这是环境
限制，不是 Rust 解析/测试探测的缺口。

每个任务在临时 git 仓库中从 initial commit 开始；写/命令自动批准（沙箱）；`run_tests`
或 `run_command` 跑测试。测试文件哈希变了一律记失败——不许通过改测试"通过"。

## 运行

```bash
python tools/coding_eval.py --model qwen3-4b
python tools/coding_eval.py --model qwen3-4b --task bugfix_add --keep

# 直接用 Axo 设置里已配置的云端 DeepSeek key（不打印）
python tools/coding_eval.py --backend axo-settings --model deepseek-chat

# 或从环境变量读
DEEPSEEK_API_KEY=... python tools/coding_eval.py --backend openai --model deepseek-chat --base-url https://api.deepseek.com/v1
```

结果：`benchmarks/coding/results-deepseek.json`（云端）与 `results.json`（本机 fallback），含 tools 序列、轮数、耗时、token、测试输出。

## 2026-10-26 云端 DeepSeek 基线（实际 agent 模型）

```text
backend=axo-settings  model=deepseek-chat
bugfix_add       passed=True   tools=8  rounds=6  elapsed=8.63s   prompt 35,709 / completion 534
feature_slugify  passed=True   tools=8  rounds=6  elapsed=7.47s   prompt 35,183 / completion 363
bugfix_cli       passed=True   tools=8  rounds=6  elapsed=8.41s   prompt 35,926 / completion 493
multi_file_tax   passed=True   tools=8  rounds=6  elapsed=6.70s   prompt 34,964 / completion 355
multi_file_limiter passed=True tools=10 rounds=7 elapsed=8.81s  prompt 41,324 / completion 439
parse_duration   passed=True  tools=14 rounds=12 elapsed=15.65s  prompt 68,203 / completion 1,321
passed 6/6；总 token：prompt 251,309 / completion 3,505
```

结果：`benchmarks/coding/results-deepseek.json`。云端 6/6 说明这套 harness 与工具面可用；任务仍偏小，
不能据此宣称达到 Codex。

> **口径说明**：这份基线测于 `repo_map` / `command_list` 与 `js_calc` /
> `multi_file_locate` 两个任务加入之前。工具声明与任务集变大后，单轮 prompt token
> 与总题数都会变；本表数字不要当成当前代码的成本。通过率未重测（结论仍以
> `results-deepseek.json` 里那次运行的 `tools` 序列为准）。

## 本机 fallback 基线（qwen3-4b，Ollama，think=false）

```text
bugfix_add       passed=True   tools=6 rounds=6 elapsed=36.67s
feature_slugify  passed=False  tools=8 rounds=8 elapsed=31.21s
bugfix_cli       passed=True   tools=6 rounds=6 elapsed=11.20s
passed 2/3
```

- `feature_slugify`：模型最后声称"测试通过"，但实际输出仍是
  `assert 'Hello-World' == 'hello-world'` 失败；同时 `protected_ok=false`（保护文件哈希
  变了）。这是**模型自述不可信**的又一个证据，基准不按 final_text 判。
- **实际 agent 使用云端 DeepSeek；这里的 qwen3-4b 只是本机 fallback，不代表云端模型。** 用 `--backend axo-settings` 跑同一任务/判据。
- 成本按**墙钟时间/工具数/轮数/token**记录；云端 token 见上。没有把价格换算成金额（价格随厂商/模型变化，不在这里硬编码）。
