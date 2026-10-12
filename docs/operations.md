# 运维与排障

> 从 README 拆出的开发文档。**内容与拆分前逐字相同**，只改了标题层级与跨文件引用。

[← 返回 README](../README.md)

## 如果 `ollama pull` 报证书错误

有些网络里 `registry.ollama.ai` 这一个**主机名**会被中间设备拦掉，报的是
`x509: certificate signed by unknown authority`。核对方法：

```bash
echo | openssl s_client -connect registry.ollama.ai:443 2>&1 | grep -iE "s:|i:"
```

若签发出的是一张自签名证书（例如 `O=redirect-cnzz`），而 `ollama.ai` 本身的证书
链是正常的，那就是针对这个域名的拦截，不是 Ollama 的问题。

**不要用关掉证书校验的办法绕**——那等于让一台中间设备随便签发证书。
可行的替代路径是从 `hf-mirror.com` 取 GGUF 再导入（`ollama create` 支持本地
GGUF，完全不需要联网）：

```bash
# 1. 取 GGUF（hf-mirror 通了就行；huggingface.co 的直连在这种网络里也常被拦）
curl -L -C - -o /tmp/Qwen3-4B-Q4_K_M.gguf \
  https://hf-mirror.com/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf

# 2. 校验（大小与 sha256 可从 hf-mirror 的 /api/models/<repo>?blobs=true 拿到）
#    sha256: 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5

# 3. 导入（Modelfile 内容见下面的「模型定义」）
ollama create qwen3-4b -f Modelfile.qwen3-4b
```

**模型存哪，得先定。** Ollama 默认把权重放 `%USERPROFILE%\.ollama\models`，一个
4B 模型 2.5 GB 起，几条就把系统盘吃掉。换盘的办法是环境变量 `OLLAMA_MODELS`：

```powershell
# ⚠ 这是**进程级**的，重启机器就没了。要持久请写进系统环境变量。
# ⚠ 而且**服务端也读它**——只给 CLI 设、不给服务设，CLI 写到一个盘而服务端去
#   系统盘找，症状是"导入说成功了，但模型列表里没有"。
$env:OLLAMA_MODELS = "E:\ollama-models"      # 盘符换成你自己的
# 设完之后要重启 ollama（服务端在启动时读一次，运行中改环境变量它看不见）
```

**模型定义**（`Modelfile.qwen3-4b`，采样参数取自 Qwen 官方 Model Card）：

```
FROM ./Qwen3-4B-Q4_K_M.gguf
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_ctx 16384
PARAMETER num_predict 2048
```

`num_ctx` 是 16384 而不是 32768：4B-Q4 的权重占约 2.5 GB 显存，8 GB 这一档的卡上
16K 的 KV cache 还放得下，32K 就要开始往内存换页、速度明显掉。
系统提示**刻意留空**——人格锚是终端侧的东西，走上下文装配进 system 消息，
不写死在模型定义里；写死就等于把性格绑回了模型。

## 跑测试与 CI

```bash
ci.bat                                # 本地跑一遍（七条腿，Windows）
uv run python tools/ci_local.py       # 同上，跨平台入口；--fast 跳过三条门槛腿
uv run pytest -p no:pytest-qt --ignore-glob='tests/test_ui_*'   # 只要核心腿
uv run pytest -m gui                 # 只要 gui 腿（**不要**加 -p no:pytest-qt）
```

`ci.bat` 与 `.github/workflows/ci.yml` 的六个 job 一一对应：`lint`（ruff check +
format）、`test`（核心腿，Python 3.10 与 3.12）、`goldset`（金标准 `--check`，报告传
artifact）、`router`（路由 `--check`，同样传报告）、`substrate`（基质 `--check`，同样传
报告）、`package`（构建 wheel → 干净 venv 装一遍 → 断言里面**没有** PySide6）。命令只有
一份，写在 `tools/ci_local.py` 里，`.bat` 只是它的一层壳——两边各抄一份的话，CI 改了个
参数而本地没跟上，症状是"本地全绿、推上去红"。

**本地比 CI 多一条 gui 腿**（`ci_local.py` 的第四条，标着「本地独有」）：CI 的 `test`
job 排除 `tests/test_ui_*`，而那两百多条里有两条守卫守的是本仓踩过的坑（接线、`0xC0000409`）。
在这条腿存在之前，它们跑与不跑取决于有没有人记得手敲 `pytest -m gui`。它**进不了**
`--fast`——`--fast` 的用途恰是"只改了界面"。它只证明**本机**全过，不证明 CI 机器上会过。

> **三条门槛腿的退出码有含义**（`bench_recall.py` / `bench_router.py` /
> `bench_substrate.py` 共用一套）：**1** = 跌破基线容差、**2** = 用法或环境不对、
> **3** = 结构不变量被违反。**3 是 bug**，不是"数字不好看"——别顺手把容差调宽。
> 前两条带 `--no-semantic` / `--no-model`：让"这台机器装没装模型"决定提交能不能过是
> 不可接受的；基质那条**不需要开关**，它的全部输入是 blake2b 与 numpy。
>
> **巩固那两条命令行只用 0 / 2 / 3，没有 1。** `bench_consolidation.py` 只报告（0）或
> 报用法/环境不对（2）——它不进门槛，见 [memory.md](memory.md) 的「睡眠巩固（P4c）」那一节；`sleep_once.py` 多一个
> **3**（重放对不上）。**1 在本仓的含义是"跌破基线容差"**，而巩固没有基线可跌，所以它不
> 产生这个码，将来有了默认路径再说。
>
> 「找不到语料文件 / 基线文件 / `--config` 指的文件」也走 **2**。这条是 2026-10-10 补的：
> 在那之前两个脚本都让它冒到顶层，退的是 **1**——于是"工作目录不对"这类**环境**问题在 CI
> 上伪装成了一条门槛失败，而 1 的含义是"行为确实变了，去看看是不是有意改的"。这句谎话
> 会让人去查一个根本没跑起来的基准。

> **`-p no:pytest-qt` 是核心腿的一部分，不是可选的优化。** 排除 `test_ui_*` 只是不
> **选**那些文件；pytest-qt 的 `pytest_configure` 在**收集之前**就把 QtGui 导入了，而
> 它在环境里（dev 组带 PySide6）。无头 Linux 上这条腿因此一个测试都跑不到：
>
> ```
> INTERNALERROR> ImportError: libEGL.so.1: cannot open shared object file
> ```
>
> 退出码 3，报错里没有一个字提到 Qt 或 gui 测试——**CI 第一次跑就是死在这里**，而本机
> （有显示器的 Windows）完全复现不出来。给 runner 装 `libEGL` 也能"修好"，但那是为了
> 让一个**用不到的插件**能 import 而给 CI 装图形栈，与"核心腿不该需要 Qt"正相反。
> 三个地方都带着它：`.github/workflows/ci.yml`、`tools/ci_local.py`、pre-commit 的
> `memcore-layering`。gui 腿那一条不要加——加了是 `fixture 'qtbot' not found`。
>
> 代价是核心腿末尾挂一行 `1575 passed, 1 warning`，那条警告是
> `Unknown config option: qt_api`（插件没加载，所以它注册的 ini 键没人认领）。无害，
> **别顺手加 `--strict-config` 把它变成错误**——那会让几条腿一起红，理由写在
> `pyproject.toml` 那段注释里。

> **`ci.bat` 与 `run.bat` 必须是 CRLF。** 这不是风格问题：cmd.exe 解析 LF-only 的批处理
> 时会算错行偏移，把半行当命令执行（`'rt' is not recognized`、`) was unexpected at this
> time`），然后以 255 退出。纯 ASCII 的 LF 批处理碰巧能跑，有中文就必然错位——所以
> `run.bat` 曾经是坏的。守它的是 `.gitattributes` 的 `*.bat text eol=crlf`、
> pre-commit 里 `mixed-line-ending` 的 `exclude`，以及
> `tests/test_entry_points.py`（**别在编辑器里把它"修"成 LF**）。
>
> 顺带一条：**改 `.gitattributes` 不会回头重写工作区里已有的文件**，它只管下一次检出的
> 那一刻。属性改对之后 `git status` 反而是干净的（git 入库时按同一套规则归一化，工作区
> 的 LF 与库里的 LF 相符），而那个文件仍然是坏的。修法是让 git 重写一遍：
> `rm run.bat && git checkout -- run.bat`。

核心腿 **1575** 条、gui 腿 **368** 条，合计 1943。新增测试会让这三个数字变化；数条数用
`--collect-only -q` 加一段求和（摘要行被 `addopts="-q"` 压掉了）：

```bash
uv run pytest --collect-only -q | awk -F': ' '/^tests\/.*: [0-9]+$/ {s+=$2} END {print s}'
```

两条腿分的是**"能不能不碰 Qt 就验完"**，不是"重要不重要"：

| 腿 | 测什么 | 为什么在这儿 |
|---|---|---|
| **核心腿** | 记忆与召回、装配与预算、逐字稿 schema、**工具注册表与沙箱**、**agent 循环与闸门**、**路由的每一条判定**、三份基准 | 这些模块**一行 Qt 都没有**，所以能跑得快、跑得稳、在没有图形栈的机器上也能跑 |
| **gui 腿** | 确认卡片出现在对话流里、点了允许/拒绝的去向、**等待确认期间点停止**、**收尾落在卡着等确认的 worker 上时进程退出码是 0**、`tools=()` 时 `AgentWorker` 与 `ChatWorker` 产出同一段文本、**语域路由真的接进了对话回合**（`test_ui_router_wire.py`）、**顶栏状态文字真的拿得到宽度**（`test_ui_elided_label.py`，见下）、**被拒绝的工具行回看时不把写给模型的文案渲染给用户**（`test_ui_tool_rows_display.py`）、**关窗的两支与托盘的菜单**（`test_ui_close_to_tray.py` / `test_ui_tray.py`）、**提醒到点真的会响**（`test_ui_reminder_scheduler.py`）、**联想开关与图真的从设置页走到了召回配置**（`test_ui_graph_chain.py`） | 只验**接线**。gui 测试一律不构造真的 `MainWindow`。**CI 排除了它**（无头 Linux 上不装 Qt），但 `tools/ci_local.py` 会跑它——本地跑一次就带上，不必手敲，`--fast` 也跳不过 |

**`tools` 非空才走 `AgentWorker`，为空仍走 `ChatWorker`**，所以两条消费路径短期并存。
缓解是一条平价测试（上面表里最后一行），而**移除准则写在那条测试自己的 docstring 里**：
平价测试一过就把 `tools=()` 也并进 agent 循环、删掉 `ChatWorker`。本版**不合并**——合并不
带来新能力，却要碰已经稳定的生成路径。

**反空转四条必须一起钉。** 只断言"最终有文本"的话，一个**从不执行工具的循环**也能过。
所以 `tests/test_agent_loop.py` 里同一个测试同时钉四件事：spy 工具被调用**恰好 1 次**
（`== 1`，不是 `>= 1`）、第 2 次请求的 `messages` 里**含那条工具结果的正文**、第 1 次请求的
`tools` **非空且条数 == 注册表条数**、`backend.stream` 被调用 **≥ 2 次**。再配一条"守卫的
守卫"（`test_registry_never_silently_drops` 断言登记的工具数等于 `specs()` 的条数），防某个
工具因 `spec` 空或名字空被静默漏掉。

三份语料各有自己的守门测试，形制一致：`test_goldset_integrity.py` /
`test_router_corpus_integrity.py` / `test_substrate_corpus_integrity.py` 管**语料本身**
（指纹、自洽、无个人信息、类别不能空转），`test_recall_gate.py` / `test_router_gate.py` /
`test_substrate_gate.py` 管**门槛**（结构不变量、可复现、与基线比）。基质那份多两条这一层
特有的：簇名必须在**记忆侧**存在（否则同簇对永远为空，AUC 会以"分母为零"的方式假绿），
以及每条查询的 `expect` **不许写满整簇**（写满就等于把答案抄给了排名）。

**顶栏那句状态文字曾经从来没显示过，而三条老守卫全是绿的。** `ElidedLabel` 用
`QSizePolicy.Ignored`，而 `IgnoreFlag` 让布局把 `sizeHint` 与 `minimumSizeHint` **一起**
无视，于是标签宽度参与分配时是 **0**——`text()` 有内容、tooltip 有内容、`paintEvent` 拿 0
去算省略号得到空串（"● Ollama 就绪 · 4 个对话模型"实测宽度 0、画出 `''`）。老测试全是
**拿孤零零一个标签**量的，而那个标签自己的 `sizeHint`/`minimumSizeHint` 一直都对，所以
问不到"布局实际给了多少"。修法是策略改成 `Preferred`（实测 1900px 宽时给 252px 全显、
1100px 时省略、窗口最小宽度只多 12 像素），守卫改成**拿真顶栏量**：
`test_the_status_label_actually_gets_room_in_the_real_header` 建一个普通 `QWidget` 当
`self` 调 `MainWindow._build_header`，断言布局给它的宽度大于 0。**这条守卫拿旧策略验过红**
（`assert 0 > 0`），没验红的守卫等于没写。

## 出问题时先跑哪个脚本

界面出问题的第一个问题永远是"哪一段坏了"。下面几个脚本一人管一段，都不需要
点界面，所以可以逐个排除：

```bash
uv run --extra app --extra llm python tools/_smoke_window.py    # 窗口+接线+模型清单（必须有显示器）
uv run --extra app --extra llm python tools/smoke_backend.py    # Python → Ollama
uv run --extra app --extra llm python tools/_smoke_e2e.py       # 提交 → 流式 → 落盘
uv run --extra app --extra llm python tools/_smoke_recall.py    # 记住 → 重启 → 想起来（P1 验收）
uv run --extra app --extra llm python tools/_smoke_memory.py    # 记忆界面八步（P2 + 人设验收）
uv run --extra app --extra llm python tools/smoke_agent.py      # 模型会不会真的吐工具调用（P3a 探针）
uv run --extra app python tools/_smoke_webview.py               # Python → 网页渲染（必须有显示器）
uv run python tools/sleep_once.py --dry-run                     # 巩固这条路算不算得通（不写库、不拍快照）
```

`smoke_backend.py` 不显式传模型名时**挑第一个能对话的**（`pick_chat_model`），而不是取列表
队首。理由是个具体的坑：`/api/tags` 的返回值**按名字排序**，队首正是 `nomic-embed-text`
——一个不能对话的嵌入模型。拿它去发 `/api/chat`，报回来的错与这个脚本要验的东西毫无关系，
而根因是"选择列表本身就不该从这里取"。全是嵌入模型时它直说并退 1，**不拿嵌入模型凑数**。

`smoke_agent.py` 是 P3a 的 **Step 0 探针**，也是本版唯一"整个方向都压在它上面"的那条假设
的验证：本地模型到底会不会吐出格式良好的 `tool_calls`。它**不设确认闸门**（改状态的工具
直接执行，只打印一行"真跑起来这里会问一次"）——它量的是模型的这一侧，闸门与跨线程握手是
另一件事。**结论写在它自己的模块 docstring 里，是 P3a 三处设计的依据**，也是
`max_rounds`、工具声明 token 开销的标定来源：那个 4B 模型在单工具选择上 13/13 全对、负
对照从不误调，但在"先查再写"的多步任务上**五轮只对一轮**，而失败的主形态是**声称做了却
没做**。这正是"界面显示真实轨迹、绝不转述自述"与"模型自分类默认关"两条的来处。`--dump`
能把真实字节冻成测试夹具（已经冻了一份，见 `tests/fixtures/ollama_tool_calls.json`）。

> **它现在退出码是 1，而且是预期的**——`return 1 if failed` 里的那条"多步·先查再写"至今
> 没通过，那正是它要留下的记录。**别把 1 读成"脚本坏了"**，也别为了让它是 0 而把那条用例
> 删掉：那是把这个探针变成一次自我祝贺。

`_smoke_recall.py` 的 `both` 模式**真的另起两个进程**——同进程里重新 open 一次库
证明不了"重启之后还在"。它的退出码就是验收结论，别只看最后那行字。

`_smoke_memory.py` 走的是「点按钮 → 建窗口 → 编辑 → 遗忘 → 恢复 → 探针 → 删除 →
首启人设卡片 → 关窗收尾」八步，**不需要 Ollama**（记忆界面不依赖模型，没有模型恰好是
必须验的降级路径）。它覆盖的正是 gui 单元测试碰不到的那一段：那些测试一律不构造真的
`MainWindow`（它一连 Ollama、起好几个线程、加载 WebEngine），所以主窗口与记忆窗口之间
的接线只能在这个脚本里验。第七步（人设）还有一层意义——**"`_wire` 里忘了接
`persona_answered`"这种漏法在别处看不出来**：卡片答完了、库却没变，界面上只表现为
一张不肯消失的卡片。它跑起来要几秒，最后一行是结论。

**退出码要在子进程里量。** 异常退出的那一段发生在解释器收尾，同进程里没有别的办法
观测；所以上面几个脚本每个都是真进程，`$LASTEXITCODE`（PowerShell）/ `$?`（bash）
是真的。别把它们串成一条 `&&` 链——前一个非零会静默跳过后面几个，看起来像"全过"。

**哪些能在离屏下跑，取决于它建不建真的网页。** `_smoke_e2e.py` 与 `_smoke_memory.py`
把对话区换成了替身（正是为了绕开下面这条），所以**加 `QT_QPA_PLATFORM=offscreen` 没问
题**，无显示器的环境里也能跑。相对的，`_smoke_window.py` 与 `_smoke_webview.py`
**必须有显示器**：前者构造真的 `MainWindow`（里面就有 QWebEngineView），后者要读渲染
结果。**别在离屏下跑真正的渲染路径**——本机实测，离屏模式下第一次 `runJavaScript` 会
让整个进程被硬杀，连异常都没有、连退出码都没有，看起来就像"脚本什么都没干"。网页侧
的承诺因此由 `_smoke_webview.py` 单独负责，而 Python 侧由那两个用替身的脚本负责。
加 `AXO_SOFTWARE_RENDER=1` 可去掉离屏 GPU 的刷屏报错。

`_smoke_webview.py` 现在依次打四段：正文渲染（含 `<script>` 与 `javascript:` 链接是否
被挡住）、外观（读回 `--accent`、`data-theme`、`has-wallpaper`、`#wallpaper` 的
`backgroundImage`、遮罩不透明度）、空状态（`clear()` 之后 `#empty` 是否回来）、人设卡片
（`setPersonaState` 的显隐与 `#empty` 互斥、正等人设时 `#input`/`#send` 的 `disabled`
读得回来、以及点两个按钮走一遍 **JS → Python** 那条半程）。第二段是**外观那版**唯一一处
"猜错了就得回改设计"的检查：**壁纸住在数据目录、不在 web 根下**，能不能画出来取决于
`LocalContentCanAccessFileUrls` 在本机 QtWebEngine 上放不放行——万一不放，传输方式要
改成带体积上限的 base64。它刻意造一张放在**系统临时目录**的图来验这件事；图放在 web
根下的话，那个设置项坏了它照样画得出来，这条检查就空转了。

第四段里的 JS → Python 半程也是**只有这里能验**：QWebChannel 找不到槽位时是**静默
丢弃**，两侧名字对不上（比如 JS 里写 `answerPersona`）不报任何错，只表现为"点了没
反应"。所以脚本末尾把 Python 收到的答案按顺序打出来，对照下一条期望行看。

## 排障

| 症状 | 原因与处理 |
|---|---|
| 顶栏红字「后端不可用」 | Ollama 没起来。跑 `ollama serve`，然后点顶栏的 ⟳ |
| 下拉框是空的但状态是绿的 | Ollama 在跑但一个模型都没有。`ollama pull <模型名>` |
| 云端后端报 `401`，或说密钥无效 | key 没填、填错，或者**这份设置是从别的机器 / 别的 Windows 账户搬过来的**——DPAPI 按「当前用户 + 这台机器」加密，那边解不开，于是按"没配过"处理。重新填一次就好，这是加密方案的代价 |
| 云端后端报 `429` | 对方在限流（免费档常见），或者余额用完了。与本地那份设置无关——key 与地址都对也会这样 |
| 切到云端后模型下拉是空的 | `/models` 拉不到（key 不对是 401，地址写错是 404，网络不通是超时）。框里那两个 `deepseek-*` 是**兜底预设**、不是从服务端读来的，所以能先选一个用；按状态栏那句话修好之后点顶栏 ⟳ 换成真的清单 |
| 云端下能聊天，但记忆说"只用语汇匹配" | **本机 Ollama 没在跑，或者里面没有嵌入模型。** 云端模式下嵌入仍然走本机（多数云端厂商没有嵌入端点）。`ollama serve` + `ollama pull nomic-embed-text`，重开一次或点 ⟳。注意这条**不影响对话**——两件事在云端模式下是分开的 |
| 窗口空白、没有任何内容 | 显卡驱动让 QtWebEngine 的 GPU 进程崩了。设环境变量 `AXO_SOFTWARE_RENDER=1` 再启动 |
| 中文显示成方块 | 系统缺中文字体。本项目的 CSS 首选 Microsoft YaHei UI |
| 首次提问等半分钟才出字 | 正常，模型正在从硬盘加载进显存（4B-Q4 冷启约 35 s）。加载完就常驻，第二次提问首字约 2 s |
| 想不起来 | 看右侧账本，别猜。它分得清三种情况：**语义通道没接上**（去装嵌入模型）、**分数不够**（去调阈值）、**库里确实没有**——三者的处理完全不同 |
| 记忆窗口里看不到刚说的那条 | 列表在打开时载入一次，**不自动跟随**。点「刷新」 |
| 记忆窗口里那条显示「没有语义向量」 | 正常，多半是刚编辑过。编辑是纯本地的，向量交回填管线下一轮补上（见 [README](../README.md) 的「记忆界面」） |
| 「遗忘」按钮点不动 | 它是**钉住**的条目。钉住的条目不走召回，软遗忘对它是个空动作，所以置灰；要让它消失请看 tooltip 里说的彻底删除 |
| 记忆窗口里「本轮归因」说不在账本里 | 两种可能，界面给不出唯一结论：这条**钉住了**（recall 在扫描前就跳过），或它属于**本会话**（被 `exclude_ids` 排除）。都不是错误 |
| 一启动就看见人设卡片 | 库里还没有你写的那条人设——全新的库，或者你把那条删了。这是**设计如此**，见 [README](../README.md) 的「首次运行：定一次人设」 |
| 输入框点不动、发送键是灰的 | 同一件事的另一半：卡片正等着回答。答完（写了或点「这一次先不写」）就解锁 |
| 选了「先不写」，下次启动又问一次 | **有意的**，见 [README](../README.md) 的「首次运行：定一次人设」最后一段：不写就等于没做这个决定。永久不问就写下任意一句 |
| 记忆窗口里人设那条改不动、也删不掉 | 那是**内置**那条，tooltip 里写着理由。要改说话方式去「设置 → 人设」改你自己写的那条；内置那条改不了也删不掉，是"任何模型读到的身份完全一样"的代价 |
| 退出码非零（`0xC0000409` 或 `3`） | 见下面的「退出码怎么读」。别按 QtWebEngine 查，它通常是无辜的 |
| 换过一个地址后某项功能说连不上、别的却好好的 | 十有八九是 `allow_remote` 没传到那一个构造点。非本机地址**必须**在五处都带上它：模型列表、记忆同步、对话后端、嵌入后端（由前者构造）、连接测试 |
| 勾了「允许远程主机」还是连不上 | 勾选框只解除**本程序**的护栏，不负责网络。先用「测试连接」看返回的是 `Connection refused`（服务没起）、超时（防火墙）还是 404（地址写成了 `/v1`） |
| 壁纸没出来 | 看图是不是被删了——文件不在时设置里的名字**刻意留着**（数据目录搬回来壁纸就自己恢复）。换过盘符/搬过数据目录是常见原因；壁纸按文件名从 `wallpapers/` 取，不记绝对路径 |
| 换壁纸后旧的那张还在磁盘上 | 正常，直到设置窗口关闭才清。按内容哈希命名是**「撤销修改」的前提**：覆盖着写的话，撤销之后那个名字指向一个已经不存在的文件 |
| 浅色模式下主色与我选的不一样 | 预设各带一支手挑的浅色值，自定义色会按对比度下限现算。设置窗口里写着「你选的是 `#xxxxxx`」——**刻意不悄悄换掉**，也不改用户存的色 |
| 控件看起来不像 Windows 原生的 | 外观那版起 `QApplication.setStyle("Fusion")`。这是**刻意的**：Windows 原生样式会忽略相当一部分 QSS 规则，出现"颜色改了但控件还是系统灰"。想回到原生观感就删掉 `ui/app.py` 里那一行（主题会跟着退化成半套） |
| 顶栏状态文字变成 `…` | 那是**省略**不是截断：把鼠标停上去有全文。加了「设置」之后顶栏有 5 个按钮 + 模型下拉，状态文字必须可压缩 |
| 双击 `run.bat` 没反应，或只打印一串 `'xxx' is not recognized` | 行尾被改成了 LF。cmd.exe 解析 LF 批处理时会错位，把半行当命令执行，然后以 255 退出——**看起来像"双击了但什么都没发生"**。见上面「跑测试与 CI」里那段：`.bat` 必须是 CRLF，`tests/test_entry_points.py` 就是拦这个的 |
| `bench_recall` 说连不上 Ollama | **正常状态，不是降级。** 语义通道是可选的，没有它就只有词法通道（`recall.py:168-171`）。CI 与 pre-commit 一律用 `--no-semantic` |
| 金标准门槛红了 | 先分流，别急着改容差：**语料 sha256 不符** = 有人改了题（那要在同一次提交里更新 `freeze.json` 并重跑 `--freeze`）；**指标跌破容差** = 召回行为真的变了；**结构断言红**（退出码 3）= pinned / cold / excluded 被注入，**那是 bug，不是标定问题**。`--check` 会把前两条线索都打出来 |
| 模型说「已记住」但记忆库里没有那条 | **先看工具轨迹，别信那句话。** 本机实测的主形态正是"声称做了却没做"（`write_memory` 一次都没被调用）。工具行是唯一的事实来源 |
| 该用工具的时候它只聊天 | 这一轮被判成了聊语域。用 `/task 正文` 强制一轮试试：能调说明是路由判偏（那要去补词表），不能调才是工具这一侧的问题 |
| 确认卡片迟迟不出现、整轮像卡住 | **设计如此**：整轮被这张卡片挡着（默认 180 s）。卡片上有倒计时和一个「跳过」= 立刻拒绝。人走开了由计时器兜底，按**拒绝**收场 |
| 点了「允许」却没落库 | 卡片的 `call_id` 与当前待决请求比对不上（比如期间点过停止）时会被丢弃并**标记为已失效**。看卡片上的状态，别重复点 |
| `read_file` 说「未配置可读目录」 | 去「设置 → 工具」加一条。**空 = 那个工具被禁用**，默认既不是当前目录也不是家目录（最小权限） |
| 路由门槛红了 | 与金标准同一条纪律，先分流（语料 sha256 / 指标跌破 / 退出码 3 = bug）。补一句这一层特有的：规则层是**一张手写词表**，改一个词能同时弄坏另一条——那正是这个门槛要抓的。`hard` 那 8 条判错的清单在 `benchmarks/routes/CORPUS.md` |
| 基质门槛红了 | 同样三档分流。这一层特有的：**它守的东西最容易被间接触坏**——量具的输入是 `encode_assoc` 的 `ASSOC_K` 与 `person` 前缀、以及建图的 `top_m` / `min_weight`，动其中任何一个都会让 `neighbor_auc` 或 `completion_top1` 变，而动手的人往往以为自己在改别处（所以它也在 pre-commit 里）。另外别拿 `--scale` 那一列去比 `baseline.json`：前者是**合成语料**、用来把容量悬崖看出来，后者是 `benchmarks/substrate/` 实测档冻下来的基线——**两把尺子** |
| 关掉窗口但任务管理器里进程还在 | **设计如此**：窗口收进了系统托盘（右键图标 →「退出」，或 Ctrl+Q）。第一次收起来时会弹一条气泡说明 |
| 通知区域里找不到托盘图标 | Windows 默认把新图标折进「隐藏的图标」里（点任务栏那个 `^`）。托盘**真的不可用**时（极少数精简版系统）程序会退回"关窗即退出"，不会出现"关不掉也找不到"的状态 |
| 设了提醒却没响 | 三个地方依次看：**程序当时开着吗**（关窗收进托盘仍然算开着，真退出就不算了，错过的会在下次启动聚合成一句）、**托盘气泡有没有被系统静音**（对话区里仍然有一条，那是唯一可靠的落点）、**`%LOCALAPPDATA%\axo\reminders.jsonl` 里有没有那条**。第三处排除掉之后剩下的都在前面两处 |
| 重启之后同一条提醒又弹了一遍 | 理论上不会：响过的 id 记在 `reminders.fired.jsonl`（先落盘再弹）。真出现了先看那个文件里有没有那一行——**没有**说明当时落盘失败了，那是个 bug，把 `reminders.jsonl` 与 fired 那个文件一起留着 |
| 核心腿在没图形栈的机器上报 `libEGL.so.1` 且退出码 3、测试一条都没跑 | pytest-qt 的 `pytest_configure` 在收集前就导入了 QtGui，而**排除 `test_ui_*` 拦不住它**。给命令加 `-p no:pytest-qt`（见上面「跑测试与 CI」）。别去装 `libEGL`——那是为了让用不到的插件能 import 而给机器装图形栈 |
| `git status` 干净，可 `.bat` 里的行尾就是 LF | 改 `.gitattributes` **不会**回头重写工作区已有的文件，而工作区 LF 与库里 LF 相符，所以没有任何差异可报。`rm run.bat && git checkout -- run.bat` 让属性重新生效；确认办法是 `tests/test_entry_points.py` |
| 巩固跑过了，「SNN 热度」还是一个数字都没有 | 两种可能，界面分得开：**根本没跑过**（说"巩固还没跑过"）与**跑过但一条都没被回放认出**（明说这句）。后者是正常的——新的记忆还没进关联图，或者都没通过探针。看不出来时去 `%LOCALAPPDATA%\axo\memory\consolidation\last.json` 看有没有那个文件 |
| 到点了却没跑巩固 | 三处依次看：设置「记忆」里勾选框还在不在；**盘上有没有图**（`memory/graph.npz`，图是懒建的，从没打开过记忆窗口就没有，那时这一轮会先请一趟建图、建完再跑）；`last.json` 里的 `newest_id` 是不是等于库里最新那条——**库里没有新记忆时自动那条会让掉一轮**（巩固的对象是已有的记忆，重跑只是把同一批再认一遍）。这三道门**手动那条都不走**：托盘的「立即巩固」随时能强制跑一趟 |
| 退出时打印「收尾超时：… 仍在运行」，而当时正在巩固 | 不是 bug：收尾只等 3 秒，巩固可能是几分钟。**有快照在**、每簇是一次「改库 + 追加事件」，所以最坏是这一轮只跑了一部分，而**库与事件是同一个前缀**、`seq` 连得上。下一次到点会重跑 |
| 巩固跑到一半发现选错了参数，想停下 | 本版**没有取消**（与建图同一姿态，也没有排给哪个阶段）。要退就托盘退出——见上一行的收尾行为；库里的改动由快照兜底，`memory/snapshots/` 里那几份整库副本可以直接拿来还原 |
| 冻结出来的 exe 双击了没反应 | 先跑带控制台的那一版（`python tools/build_exe.py --console`），回溯会打在控制台上。**无控制台的版本撞上未捕获异常只弹一个错误框、进程还活着**——所以"任务管理器里进程在"不能当成"起来过" |
| 构建报某模块没被收进去 | 去 `build/axo/axo/warn-axo.txt` 看（PyInstaller 会在 workpath 下**再套一层同名目录**，别按 `build/axo/warn-*.txt` 找）。里面绝大多数行是无害的，要看的只是 `tools/build_exe.py` 里 `_FORBIDDEN_MISSING` 那六个名字 |
| 冻结产物的对话区白屏 | 两种可能，构建时自检分得开：**web 资产没进镜像**（第 2 条）与 **QtWebEngine 的运行件没带上**（第 3 条，`QtWebEngineProcess.exe` / `qtwebengine_locales` / `plugins`）。重跑一次构建看它报哪条，别去改应用代码 |
| 首次双击弹 SmartScreen | 正常：exe 没有代码签名。点「更多信息 → 仍要运行」是一次性的 |
| 冻结产物没有自定义图标 | 构建时画图标那一步失败了，只打一句话、不中断构建（图标是外观，不该让整次构建挂掉）。重跑看那句报错 |
| 窗口标题栏、任务栏上没图标 | 与资源管理器里那个**文件**图标是两回事：前者是程序启动时自己装的（`ui/main_window.py` 的 `_install_window_icon`，跟着主题主色走），后者才是嵌在 exe 里的那一份。所以这一条只会出现在旧产物上——构建自检第 5 条会拦下它，重跑一次构建即可 |
| 构建完了，可 exe 找不到 | 它在 `dist/axo/`，而 `dist/` 在 `.gitignore` 里——编辑器里是灰的、远端上不存在，所以"看不见"是常态。构建成功时桌面会多一个快捷方式；**没多就是那次自检没过**（快捷方式只在全过之后才放），回构建输出里看它报的是哪一条 |
| 桌面那个快捷方式点了没反应，或指向旧位置 | 它里面是**绝对路径**（`.lnk` 的固有属性），仓库搬过家它就指着旧的。重新构建一次会覆盖成新的（`--no-shortcut` 那次不会重新放）。桌面被 OneDrive 接管过也不影响——路径是问系统要的 |

## 退出码怎么读

本项目的异常退出**几乎只有一种：`abort()`**。它在没装 faulthandler 时表现为
`0xC0000409`（Windows fast-fail：没有回溯、没有 stderr、Python 的 `atexit` 都不跑），
装了 faulthandler 时是 `3`——**这是同一个病的两种长相**，不是一个 Qt 问题加一个
Python 问题。已知成因只有两个，靠 stderr 或栈顶分开：

| 线索 | 成因 |
|---|---|
| stderr 有「收尾超时：… 仍在运行」 | 一个**还在跑的线程**被析构，`~QThread` 触发 `qFatal`。收尾路径没走到（`_shutdown()` 是唯一入口，别绕过它） |
| faulthandler 的栈顶停在 `main_window.py` 的 `release` | **在界面线程里丢掉一个住在工作线程里的 QObject 的最后一个引用** |

**P3a 给第一条加了一个新的触发方式：worker 卡在"等用户点确认"上。** `wait_for_all`
收尾时做的是 `thread.quit()` + `thread.wait(3000)`，而 **`quit()` 只结束线程的事件循环**，
worker 却是被 `started` 以 `Qt.DirectConnection` 直连进 `run()` 的——它根本不在事件循环里，
`quit()` 唤不醒它。于是它若正卡在确认上，收尾会 3 秒超时后继续退出，进程撞上
`0xC0000409`。所以 `AgentWorker.stop()` 必须做**两件**事：翻停止标志 + **`slot.cancel()`
立即唤醒还卡在 `wait` 里的 worker**。

> 这一条**不是优化，是这条设计成立的前提**。同理，`_shutdown()` 与「停止」都必须也调
> `_agent_worker.stop()`——`_reset_generation_state` 现在复位的是"三个 worker + 三个线程"，
> 它的 docstring 早就警告过"只复位一路"的症状。回归测试在 gui 腿：
> `test_ui_agent_slots.py`（等待确认期间点停止 → worker 在预算时间内退出）与
> `test_ui_agent_worker.py`。

第二条的修法是结构性的，写在 `MainWindow._track` 的文档里：worker 的 Qt 亲和面
**留在界面线程**，`run()` 靠 `Qt.DirectConnection` 在工作线程里直连执行。于是"丢最后
一个引用"永远是同线程析构。两个方向都要守住——亲和面搬回工作线程（`moveToThread`）
会招回这个 abort，`started` 少写一个 `DirectConnection` 则会让 `run()` 排队回界面线程
执行，症状是**界面在每次请求期间冻住**（看起来像 Ollama 慢）。回归测试：
`tests/test_ui_worker_track.py`（标 `gui`，核心腿用 `--ignore-glob='tests/test_ui_*'` 排除）。

> **`--ignore` 不认通配符。** 它收的是**路径**。写成 `--ignore=tests/test_ui_*` 或
> `--ignore=tests/test_ui_` 时它一条都排不掉，而且**不报错**——命令照常跑、照常
> 全绿，只是"核心腿"里混进了 gui 测试。要通配就用 `--ignore-glob`（它要引号，
> 否则由 shell 先展开）；按标记排则是 `-m "not gui"`。两条现在都给出 1575 条，
> gui 腿是 368 条（数条数的办法见上面「跑测试与 CI」）。

**还有一个长得一样但性质不同的退出码**，只在冒烟脚本里出现：脚本自己在收尾前
**没把事件队列排空**。窗口的 `show()`/`close()` 会投递事件，`deleteLater` 要等事件
循环才执行；带着一队未处理事件退到解释器收尾，Qt 同样走 `qFatal`，症状一模一样
（`0xC0000409`、无回溯、无 stderr）。**这不是产品的第三种成因**——真窗口全程有事件
循环，不受影响——但排查时很容易把它算到产品头上（`tools/_smoke_memory.py` 就曾因此
被误判）。那个脚本末尾排了一个 `QTimer.singleShot(200, app.quit)` + `app.exec()`，
原因写在它自己的注释里，改动那个脚本时别把这一句删掉。

## 打包（P6a）：冻结成免安装的 Windows 产物

```bash
uv sync --all-extras                  # PyInstaller 在 pack 那一组里，见下
python tools/build_exe.py             # 构建 + 自检 + 在桌面放一个快捷方式
python tools/build_exe.py --dry-run   # 只打印要跑的命令，不动盘
python tools/build_exe.py --console   # 带控制台的排障版，与上面那个并存
python tools/build_exe.py --no-shortcut   # 只构建，不碰桌面
```

产物是**一个目录**（onedir，`dist/axo/` 里是 exe 加一个 `_internal/`），不是
单文件。单文件每次启动都要把几百 MB 解到临时目录，而这个产物的几百 MB 里主要是
QtWebEngine——解压那一下正好压在启动路径上（现在实测**起一次 1.9 秒**，含窗口出现）。
机器上**不需要装 Python**，`_internal` 里带着解释器与全部依赖。

**产物藏在哪、怎么找到它。** `dist/` 是构建输出、在 `.gitignore` 里，所以它在编辑器里是灰的、
在远端上根本不存在——"构建完了却找不到 exe"是常态，不是谁弄错了。所以**自检全过之后会在桌面
放一个快捷方式**（文件名取界面上的显示名，即 `Axo.lnk`，目标与工作目录都指着
产物），双击即启动；`--no-shortcut` 关掉它。三条要知道的：

* **只在自检全过之后才放。** 指向一个自检没过的 exe，比桌面上什么都没有更糟。
* **桌面是问系统要的**，不是拼 `%USERPROFILE%\Desktop`：桌面可能被 OneDrive 接管（那时它在
  `%USERPROFILE%\OneDrive\桌面`），也可能被组策略挪走。拼字符串会静默落到一个存在的、但不是
  桌面的目录里，于是"建成功了、人却看不见"——那正是这一步要治的症状。走的是
  `SHGetKnownFolderPath`，它会把这些重定向一并算进去。
* **快捷方式里是绝对路径**（`.lnk` 的固有属性）。仓库搬了家它就指向旧位置——重新构建一次会
  覆盖成新的。建不出来时**不中断构建**（与画图标同一姿态），但会打印目标路径让人自己建一个。

**不写 `.spec`。** 构建参数在 `tools/build_exe.py` 里拼出来（`--specpath build`），理由是
`.spec` 属于 `.py` 之外的散件：ruff 扫不到、测试读不到，等于一块不进任何腿的代码。参数与
自检逻辑住同一个文件，两边都能被 AST 读到（`tests/test_packaging.py`）。

**图标是构建时现画的，仓里没有图片文件。** 调 `ui/tray.py` 那个托盘图标本身，按
256/64/48/32/24/16 六档各画一张、手工拼成 `.ico`——**每一档现画**，因为把 64px 那张缩到
16px 会把中心那个点缩没，而 16px 恰恰是任务栏最常用的那一档。画不出来时**不中断构建**，
只说一句（图标是外观，不该让整次构建挂掉）。

**自检是这一步的正文。** 冻结坏掉的方式几乎都不抛异常——资源没带进去、QtWebEngine 的
平台件落在镜像外、某个 import 漏了——它们表现为"双击之后什么都没发生"。所以构建完立刻查
盘上的这六条：

1. 产物在（`dist/axo/axo.exe`）。
2. **镜像路径假设成立**：`_internal/axo/ui/` 下的 qss 与三处网页资产都在。
   这一条成立，应用代码才一个字都不用改（那两处读取是 `Path(__file__).with_name(...)`）。
3. **QtWebEngine 的运行件在**：`QtWebEngineProcess.exe`、`translations/qtwebengine_locales/`、
   `plugins/`，以及 `resources/` 下四份运行件（它们是第 6 条删掉的那些的孪生）。
4. **缺失清单干净**：读 PyInstaller 自己写的 `warn-<name>.txt`，只比**整名**——numpy 会留下
   一大坨 `numpy._core.atleast_2d` 这种**属性式**记录（实测 175 行），按前缀比会让这份名单
   天天在响。
5. **真起一次，看有没有开出窗口、窗口上有没有图标**。判据不是"进程还在不在"：无控制台版本
   撞上未捕获异常会弹一个模态错误框、进程活着，那是**假绿**（实测踩过）。所以要求它名下有一个
   可见顶层窗口、且不是对话框、也不是控制台，**且那个窗口上挂着图标**（问 `WM_GETICON` 那三
   个槽，全 0 就是没有）。图标单列一条是因为它是唯一"窗口开出来了、看着像好了、其实是半成品"
   的那一档：exe 里嵌的那份只影响资源管理器里那个文件的外观，窗口自己不设，标题栏左上角就空
   着一格——而三处里只有一处会被顺手注意到。
6. **调试变体真的删了**（见下一条）。

**体积：521.2 MB，其中删掉了 77.2 MB。** 首次构建 598.4 MB，量出来的构成是 QtWebEngine
自己（`Qt6WebEngineCore.dll` 194.1 MB）、它的资源与语言包（`resources/` 101.4 MB +
`translations/` 53.0 MB）、软件 OpenGL 兜底（`opengl32sw.dll` 19.7 MB）、numpy 的
BLAS（`numpy.libs/` 20.2 MB）。里面**能删且删得干净**的只有一组：`resources/` 里的
**调试变体**——`qtwebengine_devtools_resources.debug.pak` 一个文件 72.3 MB，加上另外四份
`.debug.pak` / `.debug.bin`，共 77.2 MB。它们是给 Qt WebEngine 的**调试版**用的（这里链的
是发布版 DLL），而且**每一个都有同名的非调试孪生一起进了镜像**，所以删的是"同一份东西的
另一版"。自检第 3 条钉住孪生那一侧、第 6 条钉住"真的删掉了"，`tests/test_packaging.py`
里那条守卫则钉住"名单里只许出现调试变体"。删完当场实测：主窗口照常开出来（窗口类
`Qt6120QWindowIcon`、标题就是产品名），它下面那个 `QtWebEngineProcess.exe --type=renderer`
子进程在 15 秒里一直活着——**后者才是"白屏"的判据**，因为资源或语言包缺失的典型表现正是
那个子进程起不来，而主窗口照样好好地开在那儿。

**其余几块一律不动，这是有意的。** 它们都不是"没被用到的 Python 模块"，而是 Qt 自己的运行
件（语言包、软件渲染兜底、QML 插件树）；砍它们省下的是几十 MB，换来的是**白屏**——而这里
唯一的机器判据（窗口开没开出来）**根本不走那些代码路径**，所以那种坏法机器验不出来，只能
等你双击之后看屏幕。`--exclude-module` 一组都没加，理由同此。真要再审，触发条件是 Live2D
接入（那时 QtQuick 那一侧才真的进画面，也才有办法验）。

**四条如实的限制：**

* **CI 验不了这一环。** 六条腿全在 `ubuntu-latest`，而 **Windows exe 只能在 Windows 上
  构建**；名字叫「构建 wheel」的那条腿构建的是 **wheel 不是 exe**，也不碰 PyInstaller。
  所以这一环唯一的守卫是构建脚本自己那六条自检。
* **exe 没有代码签名**，首次运行会被 SmartScreen 拦一次（「更多信息 → 仍要运行」）。
* **仍然需要本机装着 Ollama 与一个嵌入模型。** 冻结改的是"跑起来需不需要装 Python"，
  不是"记忆从哪来"——云端后端那条路也一样仍要本机嵌入模型。
* **`benchmarks/` 不进产物**，界面也不 import `bench`。所以"wheel 要不要连语料一起带"
  那条**仍然开着**：P6a 冻结的是 exe，不是 wheel（见 `pyproject.toml` 与三份 `CORPUS.md`）。

**只靠眼睛验的两件事**（没有一条腿能替）：双击**桌面上那个快捷方式**（或
`dist/axo/axo.exe`），确认窗口、**对话区（那一区是网页，最可能因为
资产没带对而白屏）**、托盘图标、记忆窗口都正常；以及那个图标在三处的观感——它是程序画的圆角
方块加一个点，放大到 256px 好不好看、缩到任务栏那一档还认不认得出来，只有看了才知道。
（**"有没有"不在此列**：自检第 5 条会拦下没图标的产物。）

**冻结的入口是 `src/axo/__main__.py`**，不是 `ui/app.py`：PyInstaller 收的是
**一份脚本**，它把入口文件当 `__main__` 跑、`__package__` 是空的，而 `ui/app.py` 里那几处
相对 import 在那种情况下解析不了。`run.bat` 走的仍然是 `-m axo.ui.app`，源码
检出那条路没动。

### 发给别人：`--release`

上面那一条产出的是**你自己机器上能跑的东西**。要给别人一份"下载、解压、双击"就行的
压缩包，多跑一个开关：

```bash
python tools/build_exe.py --release                 # 全跑，最后打一个 zip
python tools/build_exe.py --release --skip-build    # 产物已经在 dist/ 里，只跑闸门
```

`--release` 不是第二个工具，是同一条流水线后面接了几步：构建 → 自检六条 → **剔 GPL-only
模块** → **隔离的首启探测** → **残留闸门** → **与第三方清单对账** → 打 zip。理由是这两个
工具迟早会漂——本仓刚吃过一次"替身跟丢"的亏。**任何一步红就不出 zip**：一个"打出来了但
内容不对"的压缩包比没有更糟，因为它会被发出去。

四条只有发布这一步才跑的判据：

1. **剔掉 GPL-only 的那几族 Qt 模块。** PyInstaller 的 PySide6 钩子把整个 wheel 的 DLL
   照单全收（实测 154 个），其中 Charts / DataVisualization / Graphs / VirtualKeyboard
   历来只在 GPLv3 下提供。删之前**先证明删得干净**：全树按字节扫一遍，除了要删的那些
   文件自己，没有任何别的文件提到那 8 个名字才动手；提到了就**一个都不删**并报红——半剔
   的树比没剔的更坏（既没消掉许可风险，又留下一堆指向空处的插件）。实测那 8 个名字的
   引用者只有 4 个 QML 插件目录与 1 个输入法插件，都在 `_GPL_ONLY_COMPANIONS` 里。
2. **首启探测。** 给子进程一个临时的 `AXO_DATA`，起完之后**断言那个目录是首启
   形状**：库建出来了、库里**只有内置人格锚一条**、没有逐字稿、没有壁纸。这一句就是
   "别人下载到的是最初始化的版本"的机器判据。（顺带修掉一个旧毛病，见下。）
3. **残留闸门。** 产物树里不许有 `*.db` / `*.log` / `*.jsonl`，不许有 `sessions/`、
   `wallpapers/`、`benchmarks/`、`tests/`、`tools/`；内容上，全树不许出现**检出路径**，
   而**用户目录那一族**只在仓自己的文件里（顶层 exe 与 `_internal/axo/`）搜。
   这条区分是实测逼出来的：产物 3157 个文件里有 270 个（Qt 的 DLL、CPython 的 `.pyd`）嵌着
   **它们构建机**的用户目录路径——Qt 的构建机用户名是 `qt`、CPython 的落在一个临时构建
   目录里，都不是我们的痕迹。全树搜会天天误报，然后被无视。这里**不把那些路径写进本文**
   也是同一个理由：判据里的针是从 `Path.home()` 算出来的，README 里再抄一遍反而让
   "全仓搜不到用户目录路径"这句话不成立了。
4. **与 `THIRD-PARTY-NOTICES.md` 对账。** 读产物的 `PYZ-*.toc`（**产物里到底冻结了哪些
   包，答案在这里，不在 `_internal/` 的目录列表里**——纯 Python 字节码全在 exe 体内那个
   PYZ 归档里，按 `_internal/` 找 httpx 会得出"没带进去"的错误结论），用
   `importlib.metadata.packages_distributions()` 反查发行包，再与那份清单里机器可读的
   那张表逐条比。**多一个、少一个、版本对不上，都红**——一份没人核对的手写清单，唯一的
   结局是慢慢与现实脱节，而那种失效看起来一切正常。

**隔离数据目录这一条修掉的是个真毛病。** 原来的活体探测只弹掉了 `QT_QPA_PLATFORM`，
没有隔离数据目录——于是子进程按 `paths.data_dir()` 落到 `%LOCALAPPDATA%\axo`，
**每次构建都拿你自己的真库起一遍应用**，往真库里写内置锚、真库里本来就有的东西也一并被
读到，而构建日志里一个字都不显示。现在子进程拿到一个临时目录，"这次构建碰过我的库没有"
这个问题才有答案，而且答案是"没有"。

产物是 `dist/axo-<版本>-win64.zip`（版本从 `pyproject.toml` 读，不写第二份常量），
顶层一个目录，里面是产物树 + `README.txt`（**写给拿到它的人**，不出现本仓行话——由
`tests/test_packaging.py` 机械地查）+ `LICENSE` + `THIRD-PARTY-NOTICES.txt` + `licenses/`
（GPL/LGPL 全文，逐字节钉在 `tests/test_licenses.py` 里；其余第三方的短许可打包时从装好的
`dist-info` 现取，按发行包分目录）。**归档内文件名一律 ASCII**：中文文件名在 zip 里要靠一个
通用的 UTF-8 标志位，老解压器会乱码，而正文照旧中文——乱码只发生在文件名上，恰好是最容易
被当成"文件坏了"的那一半。

**验红怎么验。** `--skip-build` 存在的理由不是省时间，是让闸门可以被验红：往 `dist/` 里塞
一个空的 `memory.db`、再往仓自己的资产里塞一行含检出路径的文本，跑 `--release --skip-build`，
必须报出这两条并且**不出 zip**；塞一个假的 `Qt6Charts.dll` 并让另一个文件提到它，剔除那一步
必须**拒绝删并报红**。两条都实测过。

**解压到仓外再起一次**也实测过：在空目录里解开，以隔离的 `AXO_DATA` 起来等 15 秒
再结束，产物目录**一个文件都没多、没少、没变**，而数据目录里是 `memory/mem.db` 一条
`persona://default`、外加空的 `sessions/` 与 `wallpapers/`。
