# 第三方组件与许可

这个文件有两个读者：**拿到 zip 的使用者**（它随产物一起发出去，名字是
`THIRD-PARTY-NOTICES.txt`），以及**来审这个仓库的人**。两边要的是同一件事：产物里到底
带了谁的东西、按什么许可、义务怎么履行。

本项目自身（`biosnn_memterm`、`memcore`、`tools`、`tests`）按 **Apache-2.0** 发布，
全文见仓根的 [`LICENSE`](LICENSE)。下面列的都是**别人写的东西**。

---

## 一、随产物分发的东西

下面这张表是**机器可读的**：`tools/build_exe.py --release` 会解析
`frozen-set:begin/end` 之间的行，与构建产物的 `PYZ-00.toc` 逐包、逐版本比对，对不上就
**不出 zip**。理由见本文件最后一节——一份没人核对的清单会悄悄漂掉。

**改这张表就是改判据**：加一个依赖、升一个版本，都要在同一行里改，否则构建红。

<!-- frozen-set:begin -->
| 发行包 | 版本 | 许可 |
|---|---|---|
| PySide6 | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| PySide6_Addons | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| PySide6_Essentials | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| PySide6_Pdf | 6.12.0.140 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| PySide6_WebEngine | 6.12.0.140 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| shiboken6 | 6.12.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| PyYAML | 6.0.3 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| anyio | 4.15.1 | MIT |
| biosnn-bus | 0.1.0 | Apache-2.0 |
| certifi | 2026.7.22 | MPL-2.0 |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| pyinstaller | 6.22.3 | GPL-2.0-or-later WITH Bootloader-exception |
<!-- frozen-set:end -->

几处需要说明的：

- **PySide6 那六行是同一棵安装树。** PyPI 上 `PySide6` 分成 5 个 wheel（本体 +
  Essentials + Addons + WebEngine + Pdf），磁盘上落进同一个 `PySide6/` 目录，
  `shiboken6` 是它的绑定层。版本号里 `.140` 的那两个是 Qt 补丁号，不是另外的版本线。
- **`biosnn-bus` 只有 PyPI 元数据。** 它的 dist-info 里没有许可正文，sdist 声明的是
  Apache-2.0（作者与本源出同一个人）。所以它的许可全文取自 Apache 官网，不取自 wheel。
- **`pytest` 进产物的是 `_pytest` 这一个包。** 自检路径上有一段替身/探针代码 import 了
  它（断言重写器），于是整个 `_pytest` 被连带冻结；pytest 本体（命令行入口、插件）没有进。
  列在这里是因为"被冻结了"就是"要声明"，与它为什么进来无关。
- **`pyinstaller` 进产物的是 bootloader**（那几 MB 的 exe 头部），不是它的 Python 包。
  它按 GPL-2.0-or-later 发布，**带一个明文例外**：允许用它构建并分发非自由程序，
  被构建的程序本身不受 GPL 约束。例外原文在它的 `COPYING.txt` 里。

**没有随产物分发、但你会用到的：**

- **Ollama**（本机推理运行时）与**嵌入模型**（`nomic-embed-text`、`bge-m3` 一类）
  由使用者自己安装与 `ollama pull`，不在这份产物里，因此不在上表。它们的许可随它们
  各自的发行方式。
- **Microsoft VC 运行库**（`VCRUNTIME140.dll`、`api-ms-win-*` 那几十个）。PyInstaller
  会把它们复制到 `_internal/`。它们按 Visual Studio 的可再分发条款随附，不改变本项目
  的许可。

---

## 二、LGPLv3 的三件义务，以及怎么履行的

PySide6 与 shiboken6 是同一份许可的三选一：`LGPL-3.0-only OR GPL-2.0-only OR
GPL-3.0-only`。**本项目选 LGPL-3.0 那一支**，所以下面三件事必须做到：

1. **附全文（§4(b)）。** zip 里的 `licenses/` 目录下有 `GPL-3.0.txt` 与
   `LGPL-3.0.txt`，取自 <https://www.gnu.org/licenses/>，**逐字节未改**——它们由
   `tests/test_licenses.py` 用 sha256 钉住，因为"钩子顺手清理过行尾空白"也是一种改动，
   而改动过的文本不再是 §4(b) 要的那份副本。LGPLv3 本身是对 GPLv3 的一组附加许可，
   所以两份都要给。
   其余第三方的**短**许可（numpy 的 BSD、certifi 的 MPL-2.0、httpx 的 BSD……）
   不在这里重复存放，而是打包时从 `.venv/Lib/site-packages/*.dist-info/licenses/`
   现取——那才是实际随包的那一份，不会与 wheel 里的版本脱节。

2. **可替换（§4(d)(1)）。** 产物是 **onedir** 布局：Qt 的 DLL 是 `_internal/PySide6/`
   下的独立文件，不是链接进 exe 的。使用者可以自己替换其中任何一个（换成自己构建的
   Qt、或另一个兼容版本）而**不需要重新链接本程序**，也不需要本项目的源码。
   这是选 onedir 而不是 onefile 的一个直接后果——onefile 会把它们压进一个自解压归档，
   替换就只剩"重新打包"一条路。

3. **源码可取得（§6）。** Qt 与 PySide6 的对应源码：
   - Qt：<https://download.qt.io/archive/qt/> 与 <https://code.qt.io/>（6.12.0 那一支）
   - PySide6 / shiboken6：<https://pypi.org/project/PySide6/> 上有 sdist，
     上游在 <https://code.qt.io/cgit/pyside/pyside-setup.git/>

**本项目没有修改 Qt / PySide6 / shiboken6 的任何部分。** 全部按原样分发。

**产物里没有 GPL-only 的模块。** PyInstaller 的 PySide6 钩子会把整个 wheel 的 DLL
照单全收，其中 Charts / DataVisualization / VirtualKeyboard / Graphs 这几族历来只在
GPLv3 下提供（Qt 把它与 LGPL 版分开授权）。`tools/build_exe.py` 在打包时把它们剔掉，
剔除前逐个验证"产物里没有任何别的二进制提到过它的名字"，提到就**拒绝删并报错**。
所以随产物去的 Qt 只有 LGPLv3（与更宽松）的部分。

---

## 三、这张表为什么是判据而不是文档

一份手写的依赖清单，唯一的结局是**慢慢与现实脱节**：某天有人加了一个依赖，
清单上没写，而 zip 里带着它出去了。这类失效之所以危险，是因为它看起来一切正常——
`THIRD-PARTY-NOTICES.md` 还在，读上去很完整。

所以 `--release` 会做这件事：构建完，读产物的 `PYZ-00.toc`（那里是**实际被冻结的**
每一个 Python 模块），用 `importlib.metadata.packages_distributions()` 反查发行包，
再与上面 `frozen-set` 之间的表逐条比对——**多一个、少一个、版本对不上，都红**。
两份东西不一样的时候，不能自动决定该改哪一边；工具只负责让差异可见。
