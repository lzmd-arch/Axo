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
| **编码评测** | `tools/coding_eval.py`：8 个固定任务（单文件修复/实现/CLI、多文件配置/定位、JavaScript、隐藏测试），临时 git 仓库、保护测试文件、自动批准沙箱；记录通过率/工具数/轮数/耗时/token。 |
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
| **跨轮测试记忆** | `TestSession` 从“每轮新建”改成“每会话一份”：`MainWindow._test_session` 跨用户回合保留，`_workspace_context()` 把 `brief()` 一句“上次测试”注入 L1；新会话/切会话时重置。`brief()` 有 500 字符上限。 |
| **上下文压缩** | 工具尾巴超预算驱逐整对时，`loop._summarize_pair` 把工具名、参数、成败、结果开头压成一条 `system` 摘要，放在工作尾巴前；最多 30 行 / 2400 字符。记录仍由 `MessagesAppended` 完整保留——摘要是提示，不是审计。 |
| **地图缓存** | `RepoMapCache`（LRU 8）键里带工作区快照摘要：文件没变命中，改一个字节失效；快照截断时不缓存。`ToolContext.repo_map_cache` 由主窗口注入。 |
| **多语言测试探测** | `detect_test_command` 覆盖 pytest/tox/nox、npm/pnpm/yarn/bun、deno、cargo、go、maven/gradle、dotnet、rspec/rake、phpunit、swift、mix、dart/flutter、ctest、sbt、stack/cabal、clojure/lein、zig、bazel、meson、R testthat、prove、Pester、make；`_test_files` 按多语言文件名/目录约定列前 8 个；失败行标记覆盖 Go/Rust/Jest/JUnit/GoogleTest/RSpec/PHPUnit/.NET/ExUnit/XCTest 等。 |
| **逐文件补丁** | `apply_patch` / `preview_patch` 支持 `files` 过滤；确认卡片按文件出勾选框，`ToolContext.approval_selection` 把用户勾选结果交给工具；空选择或未知文件拒绝且不改盘。卡片上的 diff 按 `+`/`-`/`@@` 逐行着色。 |
| **后台日志 UI** | 主窗口 700 ms 定时轮询 `CommandJobs.status(cursor=...)`，把增量日志推进网页的作业日志块（每作业最多留 200 KB），不经过模型、不占工具轮次；作业结束后在标题写退出码。 |
| **多语言评测** | 新增 `js_calc`（Node 内置 test runner）与 `multi_file_locate`（4 模块定位）两个任务，共 8 题；Rust 工具支持已落地，但本机缺 MSVC C++ build tools，`cargo test` 无法链接，暂不进基准。 |

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
bugfix_add        passed=True   8 tools / 6 rounds / 8.63s   35,709+534 tokens
feature_slugify   passed=True   8 tools / 6 rounds / 7.47s   35,183+363 tokens
bugfix_cli        passed=True   8 tools / 6 rounds / 8.41s   35,926+493 tokens
multi_file_tax    passed=True   8 tools / 6 rounds / 6.70s   34,964+355 tokens
multi_file_limiter passed=True 10 tools / 7 rounds / 8.81s   41,324+439 tokens
parse_duration    passed=True  14 tools / 12 rounds / 15.65s 68,203+1,321 tokens
passed 6/6；prompt 251,309 / completion 3,505
```

结果：`benchmarks/coding/results-deepseek.json`。6/6 只说明旧任务集可用，
任务仍偏小，不能据此宣称达到 Codex。这份 token 基线测于 `repo_map` / `command_list`
以及 `js_calc` / `multi_file_locate` 两个新任务加入之前；当前是 8 题，通过率与成本
都未重测。

### 2.2 本机 fallback 基线（qwen3-4b，think=false）

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
| 仓库上下文 | **已自动注入 L1**（任务语域，最多 3000 字符）：工作区、分支/改动、最近提交、顶层目录、AGENTS 前 40 行；`repo_overview` 看更多，`repo_map` 给目录 + 多语言符号索引（带快照缓存） | 会话开始自动带仓库规则/结构 |
| 编辑 | `preview_patch` + `apply_patch`（支持逐文件勾选/过滤）+ `rollback`（文件检查点）；确认卡片 diff 逐行着色 | patch 预览、应用/拒绝、回滚 |
| 执行 | 同步命令最长 600 s；后台 `start_command` + `command_status(cursor)` 增量日志 + `command_list`；UI 700 ms 推增量 | 长任务、后台任务、实时推送 |
| 测试循环 | **`run_tests` 多语言自动探测 + 失败摘要 + `TestSession` 跨轮对比**（新增/消失/原地重跑） | 自动识别测试命令、失败→修复循环 |
| 计划 | `update_plan` 有，但不强制 | 计划与执行步骤绑定、可恢复 |
| 上下文 | token 预算 + 工具结果裁剪 + **被驱逐整对的确定性摘要**；`repo_map` 有界符号索引 | 仓库地图、符号索引、压缩/恢复 |
| 评测 | **`tools/coding_eval.py` 固定 8 任务**（Python + JavaScript）；云端 DeepSeek 旧基线 6/6，8 题未重测 | 固定任务集 + 通过率/成本 |
| 模型 | DeepSeek/本地 4B 可选 | 强模型 + 可靠工具调用 |

## 4. 下一阶段（建议顺序）

1. **工作区模式**：L1 已带工作区/AGENTS 摘要与测试命令，`repo_map` 已给多语言符号索引与快照缓存；下一步补关键清单（README/CI 配置/依赖文件）的自动摘要。
2. **测试循环**：已落地失败摘要、跨轮 `TestSession` 与同轮对比；下一步把“第几轮改了什么、失败集合怎么变”持久化到逐字稿/会话文件，支持重启后恢复。
3. **补丁审查**：已落地 `preview_patch`、逐文件勾选/过滤与 diff 逐行着色；下一步补 hunk 级选择与“应用后立刻显示逐文件结果”。
4. **长任务**：已落地后台启动/增量游标/列表/内存清理/UI 实时日志；下一步做独立作业面板与日志保留策略。
5. **编码评测**：已落地 8 个任务（含 JavaScript）与按任务指定的测试命令；下一步跑新的云端基线、加大真实仓库任务，并在有 MSVC build tools 的机器上补 Rust 任务。

## 5. 边界

- 不把 Codex 的模型能力当成工具面能补齐的东西；强模型仍是上限。
- 所有写入/命令仍需用户确认；不引入鼠标/键盘注入或无限自治。
- 本路线不启动声库 M5；两者独立。
