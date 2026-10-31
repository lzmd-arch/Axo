# 凭据与 API key 安全

目标：API key 不进入远端仓库、不进入 agent 可读的上下文、不进入工具子进程、不进入评测结果。

## 1. 存储

- Axo 的云端 key 由 `SettingsStore` 用 Windows DPAPI 加密后写入 `QSettings`；明文只在
  `load_backend()` 的一次解密中出现，随后交给后端实例，不落文件。
- 仓库 `.gitignore` 明确挡住 `.env`、`.env.*`、`*.key`、`*.pem`、`secrets.*`、`config.toml`。
- `tests/test_repo_hygiene.py` 用 `git check-ignore` 复核这些规则。

## 2. 运行 agent 时

- 模型只能看到工具声明与工具结果；key 不在系统提示、记忆、逐字稿或工具结果里。
- `run_command` / `run_tests` / 后台命令都用 `axo.agent.shell._minimal_env()` 的**环境变量
  白名单**启动子进程，不继承 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`。
- 写文件/编辑/补丁/回滚只碰用户授权的可写根；`data_root` 永远拒绝。
- `repo_map` / `find_symbol` 只回灌仓库根、相对路径、文件大小和符号声明；不把文件正文发给模型。
  `.env` 这类文件最多出现在文件名列表里，不会被读取。
- 评测的测试子进程 `_run_tests()` 也显式改用 `_minimal_env()`，不会把 key 传给 agent
  生成的代码。Windows 上 `_minimal_env()` 额外补 MSVC 的 `PATH`/`LIB`/`INCLUDE` 等
  编译变量（用 vswhere + vcvarsall 解析），同样不含 key。
- `lsp` 只启动用户已安装/用 `AXO_LSP_<语言>` 显式指定的语言服务器；子进程环境
  同样是白名单（额外保留 `VIRTUAL_ENV`/`JAVA_HOME` 等构建变量，不含 key）。
  `AXO_LSP_BIN_DIRS`/`AXO_JDTLS_HOME` 是用户自己写的工具目录，不是模型可改的。
  `axo-lsp-install` 默认只打印计划，`--yes` 才调用 npm/rustup/go/dotnet/winget；
  它不提权、不自动同意许可证，也不把安装输出里的内容交给模型。
  **每次 LSP 调用都过确认闸门**：服务器可能执行构建脚本/过程宏，不能当纯只读。
  服务器本身仍以当前用户权限运行，不要配置不可信服务器。

## 3. 评测结果

- `tools/coding_eval.py` 只把 backend/model/task/tools 名/成败/轮数/耗时/token 写进结果；
  **不写 key，也不写工具正文**（工具正文可能带临时绝对路径）。
- 已对现有 `benchmarks/coding/results*.json` 做清洗：去除工具正文、替换临时绝对路径；
  文件中没有 `sk-`、`api_key`、用户名或临时目录。
- 云端 key 通过 `--backend axo-settings` 从 Axo 设置读取；进程内使用，不打印。
  也可用 `--backend openai --api-key-env NAME` 从环境变量读。

## 4. 已知边界

- 插件与主程序同进程，仍然受信任插件模型约束；不要安装不可信插件。**插件默认关**：只有 `plugins/enabled.json` 里列出的目录才会加载；每次扫描写 `plugin_audit.jsonl`（名字、`plugin.py` 的 sha256、状态、工具名、失败原因），"上次跑没跑过这段第三方代码"有账可查。
- 本机用户权限下的程序理论上可以读自己的 DPAPI 密文；本设计保证的是“不会因仓库/日志/
  agent 上下文而公开”，不是对抗本机恶意程序。
- 评测对固定任务自动批准写/命令；不要把评测入口用于运行不可信的外部任务。
- 如果曾经把 key 贴进文件、日志或聊天，立即在服务商侧轮换；删除历史不能撤回已泄露的 key。

## 记忆库的写保护（MCP 只读）

**2026-10-30 用户要求：Axo 侧只允许读，不允许对记忆库做任何更改。** 这条不是靠调用方自觉，
而是靠 SQLite 的引擎层：

- MCP 适配器（`axo-mcp` / `axo.mcp_server`）用 `MemoryStore(db, read_only=True)` 打开库，
  连接走 `file:<path>?mode=ro` URI，**任何写入都会被 SQLite 拒绝**
  （`attempt to write a readonly database`）；不提供任何打开可写的开关；
- 库文件不存在时直接报错退出（`只读模式要求记忆库已存在`），**不会顺手建一个空库**；
- 只读连接不改 `journal_mode`（那是写库头）、不建表、不做迁移；
- `memory.search` 走 `memcore.recall(..., settle=False)`，不落权重、不写访问计数；
- 工具面只有三个只读项（`state.query` / `memory.search` / `trigger.status`），
  `tests/test_mcp_server.py` 用**写动词黑名单**守住工具名，用**库文件逐字节哈希**
  守住"跑完一整轮会话库没变"；
- WAL 模式下只读连接会生成 SQLite 自己的运行时索引（`mem.db-shm`、空的 `mem.db-wal`）——
  那是读路径的机制，不是数据；不用 `immutable=1` 就是因为它会忽略 WAL，在应用同时写入时
  读出旧数据。

测试：`tests/test_store_read_only.py`（引擎层拒写、缺文件报错、`:memory:` 拒绝、
不建 schema）与 `tests/test_mcp_server.py` 的写保护几条。

## 可写档位的边界（提议消费者，2026-10-31 落地）

只读档位与可写档位共用一套代码，但写权限不是"打开一个开关"这么简单：

1. **`axo-mcp` 一行没改**：它仍然用 `mode=ro` 打开库，三个工具里没有任何写动词；
2. **新增的 `axo.propose_consumer` 是唯一会写记忆的组件**，而它只消费**提议队列**，
   不提供"按 id 直接改/删"的入口；
3. 消费者拿到只读库会直接抛错（`需要一个可写库`），不会悄悄降级成"什么都写不进去"；
4. 每条提议、判定、落库都写进 `propose_audit.jsonl`，事件名与 BioSNN 的 MCP 写版本一致；
5. 提议在落库前过门控：重复（相似度 ≥ 0.90）与显著度 < 0.30 直接拒，0.30–0.60 或超配额待审；
6. **入口是 `axo-mcp-write`，不是 `axo-mcp`**：设置关着时它直接退出（exit 2）并提示去哪里打开；
   打开之后，设置页的勾选框、记忆窗口的提议台账、撤销按钮才真正有东西可操作。
