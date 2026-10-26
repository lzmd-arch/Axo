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

- 插件与主程序同进程，仍然受信任插件模型约束；不要安装不可信插件。
- 本机用户权限下的程序理论上可以读自己的 DPAPI 密文；本设计保证的是“不会因仓库/日志/
  agent 上下文而公开”，不是对抗本机恶意程序。
- 评测对固定任务自动批准写/命令；不要把评测入口用于运行不可信的外部任务。
- 如果曾经把 key 贴进文件、日志或聊天，立即在服务商侧轮换；删除历史不能撤回已泄露的 key。
