# 编码任务基准

固定 9 个任务，判据不是"模型说通过"，而是**跑测试 + 保护文件哈希**：

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
| `rust_calc` | Rust | 修 `src/lib.rs` 的加法（`cargo test`） | `tests/public.rs`、`Cargo.toml` 不得改；另有隐藏负数测试 |

Rust 已进基准：本机 Build Tools 18 装好但不在 PATH，`_minimal_env()` 通过
`vswhere` + `vcvarsall.bat` 只补 `PATH`/`LIB`/`INCLUDE` 等编译变量；`cargo test`
可正常链接。环境变量仍是白名单，不含 API key。

每个任务在临时 git 仓库中从 initial commit 开始；写/命令自动批准（沙箱）；`run_tests`
或 `run_command` 跑测试。测试文件哈希变了一律记失败——不许通过改测试"通过"。

## 运行

```bash
python tools/coding_eval.py --model qwen3-4b
python tools/coding_eval.py --model qwen3-4b --task bugfix_add --keep

# 直接用 Axo 设置里已配置的云端 DeepSeek key（Windows DPAPI；不打印）
python tools/coding_eval.py --backend axo-settings --model deepseek-chat

# 或从环境变量读
DEEPSEEK_API_KEY=... python tools/coding_eval.py --backend openai --model deepseek-chat --base-url https://api.deepseek.com/v1
```

结果：`benchmarks/coding/results-deepseek-8tasks.json`（8 题云端）与 `results.json`（本机 fallback），含 tools 序列、轮数、耗时、token、测试输出。旧的 6 题云端结果留在 `results-deepseek.json`，不要和新基线混用。

## 2026-10-26 云端 DeepSeek 基线（实际 agent 模型，8 题）

```text
backend=axo-settings  model=deepseek-chat
bugfix_add       passed=True   tools=7  rounds=5  elapsed=7.17s   prompt 35,320 / completion 298
feature_slugify  passed=True   tools=7  rounds=5  elapsed=6.36s   prompt 35,824 / completion 397
bugfix_cli       passed=True   tools=9  rounds=7  elapsed=7.65s   prompt 48,291 / completion 378
multi_file_tax   passed=True   tools=8  rounds=5  elapsed=6.50s   prompt 35,980 / completion 376
multi_file_limiter passed=True tools=9 rounds=6  elapsed=7.70s   prompt 42,873 / completion 411
parse_duration   passed=True  tools=13 rounds=11 elapsed=15.58s  prompt 78,208 / completion 1,193
js_calc          passed=True   tools=8  rounds=5  elapsed=6.28s   prompt 35,994 / completion 331
multi_file_locate passed=True  tools=7  rounds=4  elapsed=5.69s   prompt 29,961 / completion 326
passed 8/8；总 token：prompt 342,451 / completion 3,710
```

结果：`benchmarks/coding/results-deepseek-8tasks.json`。8/8 说明当前工具面、8 个任务与云端
DeepSeek 的组合可用；任务仍偏小，不能据此宣称达到 Codex。

## 真实仓库任务（单个冒烟，不是完整基准）

```bash
python tools/coding_eval.py --backend axo-settings --model deepseek-chat \
  --real-repo D:\Projects\BioSNN-MemoryTerminal --task real_axo_snapshot \
  --out benchmarks/coding/results-deepseek-real.json
```

任务把 Axo 的 **HEAD 快照**复制到临时目录，在 `src/axo/agent/snapshot.py` 注入一个
`count >= max_files` → `count > max_files` 的 off-by-one，让公开测试
`tests/test_test_session.py` 失败；隐藏测试再检查 `max_files` 是硬上限。保护
`tests/test_test_session.py` 与 `tests/test_repo_map.py`，改测试记失败。

```text
source_commit=c671ae1  model=deepseek-chat
real_axo_snapshot passed=True tools=14 rounds=12 elapsed=28.59s
prompt 195,229 / completion 1,379
tools: repo_overview → run_tests → search_files → read_file ×2 →
       edit_file → run_tests×4 → run_command×2 → git_diff
```

结果：`benchmarks/coding/results-deepseek-real.json`。用例只有 1 个、且是恶意注入的
显式 bug，**不能代表真实仓库的整体能力**；它证明的是“在真实目录结构里定位—读—
改—跑测试—看 diff”这条链能走通。

## Rust 单题（真实 MSVC 工具链）

```text
backend=axo-settings  model=deepseek-chat
rust_calc passed=True tools=6 rounds=4 elapsed=10.12s
prompt 29,759 / completion 256
tools: repo_overview → repo_map → read_file×2 → edit_file → run_tests
```

结果：`benchmarks/coding/results-deepseek-rust.json`。Rust LSP 也用真实 `rust-analyzer`
验过：definition 跳到 `src/lib.rs:1:8`，references 返回调用点，pull diagnostics 返回空诊断。

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
