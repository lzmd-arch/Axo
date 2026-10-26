# 编码代理（Codex 类）现状与路线

目标不是把 Axo 变成另一个 Codex 产品，而是让它的 **agent 工具面**具备 Codex 类编码
代理的基础能力：读仓库、改多文件、跑测试、看差异、在确认闸门下自治完成一步。

## 1. 本轮已落地（2026-10-26）

| 项 | 变化 |
| :--- | :--- |
| **写工具接线** | `MainWindow._tools()` 之前漏传 `write_root`/`allow_commands`：纯层测试全绿，真实界面里 `write_file`/`edit_file`/`run_command` 从未登记。已补，并由 `tests/test_ui_tools_wire.py` 守卫。 |
| **文本原样** | 新增 `argument_text`：`write_file` 的 content、`edit_file` 的 old/new、`apply_patch` 的 patch、clipboard 不再被 `strip()`，避免代码缩进/末尾换行被吃掉。 |
| **git 工具** | 只读 `git_status` / `git_diff` / `git_log` / `repo_overview`（含 AGENTS/README 开头、最近提交、顶层目录）。 |
| **读取行号** | `read_file` 默认给每行加行号（可 `line_numbers=false` 关闭），方便写 unified diff。 |
| **多文件补丁** | `apply_patch`：unified diff，先 `git apply --check` 再应用；需确认；拒绝绝对路径与 `..`。 |
| **路由** | “实现/修复/重构/调试/跑测试/改代码/优化/仓库”等编码请求进入 TASK；裸“测试”不收，避免误伤提醒语料。 |
| **命令时限** | 单条命令上限 120 → **600 秒**，支持长测试；取消/超时仍杀整棵进程树。 |
| **测试循环** | `run_tests`：自动探测 pytest/npm/pnpm/yarn/cargo/go/mvn/gradle/dotnet/rspec/phpunit/swift/mix/dart/ctest/make，跑完把失败行摘成摘要，提示“读断言、最小修改、别只重跑”；需确认。 |
| **补丁预览/回滚** | `preview_patch` 只检查/展示 diff；`write_file`/`edit_file`/`apply_patch` 应用前写文件检查点，`rollback` 恢复最近一次或指定 id。 |
| **长任务** | `start_command` 后台启动立即返回 job_id，`command_status` 读输出尾部/退出码，`command_stop` 杀进程树；不阻塞对话。 |
| **编码评测** | `tools/coding_eval.py`：9 个固定任务（单文件修复/实现/CLI、多文件配置/定位、JavaScript、Rust、隐藏测试），临时 git 仓库、保护测试文件、自动批准沙箱；记录通过率/工具数/轮数/耗时/token。 |
| **测试** | 新增 git/patch/检查点/后台/路由/接线/空白保留测试；全量 Axo 测试见 `pytest` 日志。 |

### 1.1 第二批（同日）

| 项 | 变化 |
| :--- | :--- |
| **仓库地图** | 新增只读 `repo_map`：目录 + 文件大小 + 常见语言符号与行号（Python 用 AST；JS/TS/Vue/Svelte、Go、Rust、Java、C/C++、C#、Ruby、PHP、Swift、Kotlin、Scala、Lua、R、Shell/PowerShell、Perl、Dart、Elixir、Haskell、Clojure 用声明行匹配）。跳过 `__pycache__`/`.venv`/`node_modules` 等噪声目录；文件数、符号数、总字符都有上限，截断时写明“还有多少没列”。 |
| **测试会话记忆** | 新增 `axo.agent.test_session.TestSession`：记录第几次测试、失败集合的新增/消失/仍失败，以及工作区文件快照（路径+大小+mtime，跳过缓存目录）。失败完全相同且文件没变时明确提示“原地重跑不会改变结果”；快照截断时不断言“没有变化”。**跨用户回合复用**，换会话时重建。 |
| **后台增量日志** | `command_status` 新增 `cursor` 字节偏移：从 0 读全量，或拿上次的 `next_cursor` 只读新增；`has_more` 表示还没追完；UTF-8 字符不会被切半。新增 `command_list` 列当前作业。`CommandJobs.prune()` 清旧作业的内存记录（日志保留）。 |
| **接线守卫** | `tests/test_ui_tools_wire.py` 增加 `repo_map`/`command_list` 的工具表断言，并用 AST 检查 `_start_agent_turn` 的 `ToolContext(...)` 调用确实带 `test_session` / `repo_map_cache`。 |

### 1.2 第三批（同日）

| 项 | 变化 |
| :--- | :--- |
| **跨轮测试记忆** | `TestSession` 从“每轮新建”改成“每会话一份”：`MainWindow._test_session` 跨用户回合保留，`_workspace_context()` 把 `brief()` 一句“上次测试”注入 L1；历史落 `<session_id>.tests.json`，重启后恢复；新会话/切会话时重建。`brief()` 有 500 字符上限。 |
| **上下文压缩** | 工具尾巴超预算驱逐整对时，`loop._summarize_pair` 把工具名、参数、成败、结果开头压成一条 `system` 摘要，放在工作尾巴前；最多 30 行 / 2400 字符。记录仍由 `MessagesAppended` 完整保留——摘要是提示，不是审计。 |
| **地图缓存** | `RepoMapCache`（LRU 8）键里带工作区快照摘要：文件没变命中，改一个字节失效；快照截断时不缓存。`ToolContext.repo_map_cache` 由主窗口注入。 |
| **符号检索** | 新增只读 `find_symbol`：在跨语言符号索引里按名字搜索，返回 `文件:行: 声明`；大小写不敏感、结果有上限。它是静态索引，不是 LSP。 |
| **LSP 桥** | 新增可选 `lsp` 工具：definition / references / symbols / diagnostics，stdio JSON-RPC、超时/进程收尾、按文件根缓存客户端；无服务器时明确回退。服务器不随包分发，用 `AXO_LSP_<语言>`、`AXO_LSP_BIN_DIRS`、`AXO_JDTLS_HOME` 配置。**每次调用过闸门**（服务器可能执行构建脚本）。本机真机验过 Python/TS/JS/PHP/Rust/Go/C/C++/Java/C#/Shell/Lua/Svelte/Perl/Dart；`tools/language_doctor.py` 列出 26 个常见后缀的符号索引/LSP 服务器状态。 |
| **多语言测试探测** | `detect_test_command` 覆盖 pytest/tox/nox、npm/pnpm/yarn/bun、deno、cargo、go、maven/gradle、dotnet、rspec/rake、phpunit、swift、mix、dart/flutter、ctest、sbt、stack/cabal、clojure/lein、zig、bazel、meson、R testthat、prove、Pester、make；`_test_files` 按多语言文件名/目录约定列前 8 个；失败行标记覆盖 Go/Rust/Jest/JUnit/GoogleTest/RSpec/PHPUnit/.NET/ExUnit/XCTest 等。 |
| **逐文件补丁** | `apply_patch` / `preview_patch` 支持 `files` 过滤；确认卡片按文件出勾选框，`ToolContext.approval_selection` 把用户勾选结果交给工具；空选择或未知文件拒绝且不改盘。卡片上的 diff 按 `+`/`-`/`@@` 逐行着色。 |
| **后台日志 UI** | 主窗口 700 ms 定时轮询 `CommandJobs.status(cursor=...)`，把增量日志推进网页的作业日志块（每作业最多留 200 KB），不经过模型、不占工具轮次；作业结束后在标题写退出码。 |
| **多语言评测** | 新增 `js_calc`（Node 内置 test runner）、`multi_file_locate`（4 模块定位）与 `rust_calc`（cargo test），共 9 题；Windows 上 `_minimal_env()` 用 vswhere + vcvarsall 补 MSVC 编译变量，Rust 已能真实链接。 |

## 2. 工作区自动上下文（本轮新增）

`axo.agent.workspace.build_workspace_context()` 在**任务语域**生成一段只读摘要，
由 `MainWindow._workspace_context()` 放进 L1。它只写事实：仓库根、分支与未提交改动数、
最近 3 条提交、顶层目录、AGENTS.md 前 40 行。不读 README（太长，需要时用 `read_file`），
上限 3000 字符；失败时返回空串，不打断对话。

同一段还会按项目类型探测一条测试命令（覆盖 Python/Node/Deno/Rust/Go/JVM/.NET/Ruby/PHP/Swift/Elixir/Dart/CMake/Bazel/Meson 等，见 1.2），
并列出前 8 个测试文件；只写事实、不执行。

### 2.1 云端 DeepSeek 基线（实际 agent 模型）

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

结果：`benchmarks/coding/results-deepseek-8tasks.json`。8/8 只说明这 8 个任务、
工具面与云端 DeepSeek 的组合可用；当前任务集共 9 题（Rust 单题见 2.3），任务仍偏小，
不能据此宣称达到 Codex。旧的 6 题结果留在 `benchmarks/coding/results-deepseek.json`，
不要和新基线混用。

### 2.2 真实仓库冒烟（单个任务）

```text
source_commit=c671ae1  model=deepseek-chat
real_axo_snapshot passed=True tools=14 rounds=12 elapsed=28.59s
prompt 195,229 / completion 1,379
tools: repo_overview → run_tests → search_files → read_file ×2 →
       edit_file → run_tests×4 → run_command×2 → git_diff
```

结果：`benchmarks/coding/results-deepseek-real.json`。任务复制 Axo HEAD，在
`snapshot.py` 注入 off-by-one；保护测试文件，隐藏测试检查 `max_files` 硬上限。
只有 1 个显式 bug，不能代表真实仓库整体能力；它证明“真实目录里定位—读—改—跑测试—
看 diff”这条链能走通。

### 2.3 Rust 单题（真实 MSVC 工具链）

```text
rust_calc passed=True tools=6 rounds=4 elapsed=10.12s
prompt 29,759 / completion 256
tools: repo_overview → repo_map → read_file×2 → edit_file → run_tests
```

结果：`benchmarks/coding/results-deepseek-rust.json`。Windows 上 `_minimal_env()` 用
`vswhere` + `vcvarsall.bat` 只补 MSVC 编译变量，`cargo test` 能真实链接。Rust LSP 也用
真实 `rust-analyzer` 验过 definition / references / pull diagnostics。

### 2.4 本机 fallback 基线（qwen3-4b，think=false）

```text
bugfix_add       passed=True   6 tools / 6 rounds / 36.67s
feature_slugify  passed=False  8 tools / 8 rounds / 31.21s
bugfix_cli       passed=True   6 tools / 6 rounds / 11.20s
passed 2/3
```

`feature_slugify` 的模型 final_text 声称“测试通过”，实际测试仍失败，且保护文件哈希变了；
基准按测试与保护文件判，不按自述判。**实际 agent 使用云端 DeepSeek；4B 只是本机 fallback。** 云端结果见上；任务与判据见 `benchmarks/coding/README.md`。

## 3. 与 Codex 的差距

| 维度 | Axo 现在 | Codex 类目标 |
| :--- | :--- | :--- |
| 仓库上下文 | **已自动注入 L1**（任务语域，最多 3000 字符）：工作区、分支/改动、最近提交、顶层目录、AGENTS 前 40 行；`repo_overview` 看更多，`repo_map` 给目录 + 多语言符号索引（带快照缓存），`find_symbol` 按名字定位，`lsp` 提供语义定义/引用/诊断（需自装服务器） | 会话开始自动带仓库规则/结构 |
| 编辑 | `preview_patch` + `apply_patch`（支持逐文件勾选/过滤）+ `rollback`（文件检查点）；确认卡片 diff 逐行着色 | patch 预览、应用/拒绝、回滚 |
| 执行 | 同步命令最长 600 s；后台 `start_command` + `command_status(cursor)` 增量日志 + `command_list`；UI 700 ms 推增量 | 长任务、后台任务、实时推送 |
| 测试循环 | **`run_tests` 多语言自动探测 + 失败摘要 + `TestSession` 跨轮对比**（新增/消失/原地重跑） | 自动识别测试命令、失败→修复循环 |
| 计划 | `update_plan` 有，但不强制 | 计划与执行步骤绑定、可恢复 |
| 上下文 | token 预算 + 工具结果裁剪 + **被驱逐整对的确定性摘要**；`repo_map` 有界符号索引 | 仓库地图、符号索引、压缩/恢复 |
| 评测 | **`tools/coding_eval.py` 固定 9 任务**（Python + JavaScript + Rust）；8 题基线 **8/8**（342,451 / 3,710），Rust 单题 1/1（29,759 / 256）；真实仓库冒烟 1/1（195,229 / 1,379） | 固定任务集 + 真实仓库任务 + 通过率/成本 |
| 模型 | DeepSeek/本地 4B 可选 | 强模型 + 可靠工具调用 |

## 4. 下一阶段（建议顺序）

1. **工作区模式**：L1 已带工作区/AGENTS 摘要与测试命令，`repo_map` 已给多语言符号索引与快照缓存；下一步补关键清单（README/CI 配置/依赖文件）的自动摘要。
2. **测试循环**：已落地失败摘要、跨轮 `TestSession`、同轮对比与 `<session_id>.tests.json` 落盘；下一步把“每轮改了什么、失败集合怎么变”也写进逐字稿供回看。
3. **补丁审查**：已落地 `preview_patch`、逐文件勾选/过滤与 diff 逐行着色；下一步补 hunk 级选择与“应用后立刻显示逐文件结果”。
4. **LSP**：已接可选 JSON-RPC 桥（definition/references/symbols/diagnostics），并有假服务器协议测试；本机已真机验证 Python/TS/JS/PHP/Rust/Go/C/C++/Java/C#/Shell/Lua/Svelte。运行 `python tools/language_doctor.py` 看每个常见后缀缺哪个服务器；下一步补 Ruby/Dart/Kotlin/Swift/R 等的运行时或安装文档。
5. **长任务**：已落地后台启动/增量游标/列表/内存清理/UI 实时日志；下一步做独立作业面板与日志保留策略。
6. **编码评测**：9 个任务（含 JavaScript/Rust）、云端 DeepSeek 8 题 **8/8**、Rust 单题 1/1 与真实仓库冒烟 1/1 已落地；下一步跑 9 题全量基线、加更多真实仓库任务（不止一个显式 bug）。

## 5. 常见语言支持现状

共 26 个常见后缀进入同一张矩阵：**符号索引**（`repo_map`）、**LSP 语言映射/默认服务器**
（`lsp`）与**测试探测**（`detect_test_command`）必须同时存在；`tests/test_language_matrix.py`
逐个后缀守这条。未知后缀仍列文件，只是不摘符号。

本机真机验过 LSP 的：Python、JavaScript/TypeScript、PHP、Rust、Go、C/C++、Java、
C#、Shell、Lua、Svelte、Perl、Dart。Rust 还跑了完整 `cargo test` 基准。

没装真实服务器/运行时的：Ruby（RubyGems 缺 MSYS2）、Kotlin（Kotlin LS 不兼容 Java 25）、
Swift、R、PowerShell、Vue 之外的其余生态等。用：

```powershell
python tools/language_doctor.py
```

看每个后缀缺哪个服务器；`AXO_LSP_<语言>` 指定命令，`AXO_LSP_BIN_DIRS` 放便携
工具链目录，`AXO_JDTLS_HOME` 指向便携 jdtls。

## 6. 边界

- 不把 Codex 的模型能力当成工具面能补齐的东西；强模型仍是上限。
- 所有写入/命令仍需用户确认；不引入鼠标/键盘注入或无限自治。
- 本路线不启动声库 M5；两者独立。
