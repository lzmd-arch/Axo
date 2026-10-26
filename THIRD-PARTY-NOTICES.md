# 第三方组件与许可

这个文件有两个读者：**拿到 zip 的使用者**（它随产物一起发出去，名字是
`THIRD-PARTY-NOTICES.txt`），以及**来审这个仓库的人**。两边要的是同一件事：产物里到底
带了谁的东西、按什么许可、义务怎么履行。

本项目自身（`axo`、`memcore`、`tools`、`tests`）按 **Apache-2.0** 发布，
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
| biosnn-bus | 0.4.0 | Apache-2.0 |
| certifi | 2026.10.19 | MPL-2.0 |
| cffi | 2.1.1 | MIT |
| charset-normalizer | 3.5.2 | MIT |
| colorama | 0.4.6 | BSD-3-Clause |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
| qrcode | 8.2 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pypdf | 6.21.2 | BSD-3-Clause |
| pytest | 9.1.1 | MIT |
| requests | 2.34.2 | Apache-2.0 |
| setuptools | 84.2.1 | MIT |
| srt | 3.5.3 | MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| tinytag | 2.3.2 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| vosk | 0.3.44 | Apache-2.0 |
| pyinstaller | 6.22.3 | GPL-2.0-or-later WITH Bootloader-exception |
<!-- frozen-set:end -->

几处需要说明的：

- **PySide6 那六行是同一棵安装树。** PyPI 上 `PySide6` 分成 5 个 wheel（本体 +
  Essentials + Addons + WebEngine + Pdf），磁盘上落进同一个 `PySide6/` 目录，
  `shiboken6` 是它的绑定层。版本号里 `.140` 的那两个是 Qt 补丁号，不是另外的版本线。
- **`biosnn-bus` 与 `vosk` 都只有 PyPI 元数据。** 两家的 dist-info 里都没有许可正文：
  `biosnn-bus` 的 sdist 声明 Apache-2.0（作者与本源出同一个人），而 `vosk` 的 METADATA
  里 `License` 字段写的是 `UNKNOWN`、只有一条 classifier 说 Apache-2.0。所以这两份许可
  全文都取自 Apache 官网（<https://www.apache.org/licenses/LICENSE-2.0>），不取自 wheel。
  构建那边由 `_EXPECTED_WITHOUT_LICENSES` 盯着——**"少了一份许可"永远是一个要解释的状态**，
  哪天冒出一个新的第三家，构建当场红。
- **`vosk` 拉进来八行。** 它自己的 `__init__.py` 在模块顶层就 `import srt` /
  `import requests` / `from tqdm import tqdm`（那是它自带的命令行转写工具用的），于是
  `srt`、`requests`（连带 `urllib3` / `charset-normalizer` / `idna` / `certifi`）、
  `tqdm`（连带 Windows 下的 `colorama`）全都跟着进产物。**本项目只用 `vosk.Model` 与
  `vosk.KaldiRecognizer` 两个类**，但"实际用没用到"不是这张表的判据，"被冻结了"才是
  （同下面 `pytest` 那一条）。上面那几行的许可见表；`vosk` 自己的 C 库与 GCC 运行库
  见第二节末尾。
- **`cffi` 随产物发的是扩展模块，不是 Python 包。** `cryptography` 声明依赖它，产物
  `_internal/` 根上有 `_cffi_backend.cp312-win_amd64.pyd`；`cffi/` 本身没有冻进 PYZ，
  所以构建对账时它走 `_FROZEN_EXEMPT`。`pycparser` 只在 cffi 编译期用得到，**一个字
  节都没进产物**，因此不在表里。
- **`setuptools` 是打包器带进来的，不是应用依赖。** PyInstaller 给这个产物加了一个
  `pyi_rth_setuptools` 运行期钩子，它把 `setuptools` 整个收进 PYZ（含它 vendored 的
  `packaging` / `wheel` / `jaraco` 等）。应用代码一个字都没用到它。
- **`pytest` 进产物的是 `_pytest` 这一个包。** 自检路径上有一段替身/探针代码 import 了
  它（断言重写器），于是整个 `_pytest` 被连带冻结；pytest 本体（命令行入口、插件）没有进。
  列在这里是因为"被冻结了"就是"要声明"，与它为什么进来无关。
- **`vosk` 那个 wheel 里还带着三个 GCC 的 C++ 运行库**（`libstdc++-6.dll` 25.4 MB、
  `libgcc_s_seh-1.dll`、`libwinpthread-1.dll`）。`libvosk.dll` 是拿 MinGW 编的，链接到
  它们；它们按 GPL-3.0-with-GCC-exception 发布，那个例外明文允许"用它编译出来的程序
  按任意许可分发"。它们**不是靠 import 链找到的**：`vosk/__init__.py` 里那句是
  ``_ffi.dlopen(os.path.dirname(__file__) + "/libvosk.dll")``，一个运行时拼出来的路径，
  静态分析看不见——所以 `tools/build_exe.py` 显式加了 `--collect-binaries vosk`，
  自检里也有一条专门盯着这四个文件在不在（第 3 条）。
- **`vosk` 拉进来一大串传递依赖，它们不是本项目直接用的。** `vosk/__init__.py` 在
  **模块顶层**就 `import srt` / `import requests` / `from tqdm import tqdm`（那是它自己
  那个命令行转写工具用的），所以只要 `axo/asr.py` import 了 vosk，这几家就一起进产物。
  本项目**只用到 `vosk.Model` / `vosk.KaldiRecognizer` 两个类**——但"实际用没用到"不是
  这张表的判据，"被冻结了"才是（同上面 `pytest` 那一条）。`vosk` 本身按 Apache-2.0，
  它那几家的许可见上表。
- **`vosk-model-small-cn-0.22`（中文小模型，约 42 MB）在 macOS 包里随附**，按
  Apache-2.0 分发，来源是 <https://alphacephei.com/vosk/models>。它不在 PyPI 表里，
  因为没有 Python dist-info；Windows zip 不带它，使用者自己放到数据根下的 `models/`。
  两份产物都只使用它做离线语音识别，模型的许可不改变本项目自身许可。
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
   替换就只剩"重新打包"一条路。macOS 的 `.app` 同一条布局：Qt 的 framework 在
   `Contents/Frameworks/` 下，是可按文件替换的；`licenses/` 与
   `THIRD-PARTY-NOTICES.txt` 随 zip/dmg 一起发。

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

**还有一份 FFmpeg，它也是 copyleft，而且是量出来的。** Qt 6 的多媒体后端默认走 FFmpeg，
PySide6 的 wheel 把那份运行库直接放在 `PySide6/` 下：`avcodec-63.dll`（13.5 MB）、
`avformat-63.dll`、`avutil-61.dll`、`swresample-7.dll`、`swscale-10.dll`，外加
`plugins/multimedia/ffmpegmediaplugin.dll`。语音输入只用到麦克风，这几份是 Qt 那条链
拉进来的（见下面"为什么不干脆剔掉"）。**它们是 LGPL 版，不是 GPL 版**——这不是猜的：
这些 DLL 里嵌着它自己那份构建配置，从 `avcodec-63.dll` 的字节里读得出来一整串：

    --prefix=/c/FFmpeg-n9.0.1/build/msvc/installed --disable-programs --disable-doc
    --disable-debug --enable-network --disable-lzma --enable-pic --disable-vulkan
    --disable-v4l2-m2m --disable-decoder=truemotion1 --disable-avdevice --disable-avfilter
    --enable-zlib --toolchain=msvc --enable-shared

里面**没有 `--enable-gpl`，也没有 `--enable-nonfree`**——那两条是 FFmpeg 变成 GPL 的仅
有的开关。版本也在同一串里：**FFmpeg 9.0.1**。FFmpeg 自己声明的是"LGPL 2.1 或更高"，
所以本项目按 **LGPLv3** 那一支使用它，而 `licenses/LGPL-3.0.txt` 与 `GPL-3.0.txt` 就是
它的许可文本（LGPLv3 是对 GPLv3 的一组附加许可）。源码：9.0.1 的源码包在
<https://ffmpeg.org/releases/>，配置就是上面那一串。上面 §4(d)(1) 那条"可替换"同样由
onedir 满足——它们是 `_internal/PySide6/` 下的独立 DLL。

**为什么不干脆把这几份剔掉。** 剔掉省 18.7 MB、少一份 copyleft，账面上更干净。但 Qt 6
的默认多媒体后端就是 FFmpeg，剔掉之后接手的是 `windowsmediaplugin.dll`（Media
Foundation），而**能验到的只有"它枚举得出设备"**（`QT_MEDIA_BACKEND=windows` 下与默认
后端给出同一份清单：两台输入设备、三台输出设备），**验不到它录不录得到音**——而"录得到
音"正是这个功能唯一的用途。本仓对"删掉一个东西"的判据一向是"能证明它是多余的"（那是
调试变体那 77.2 MB 的删法），这里证明不了，所以留着，并把它的来源与许可写清楚。

最后还有一份 GCC 的 C++ 运行库（`_internal/vosk/libstdc++-6.dll` 等三个）。它按
GPL-3.0-with-GCC-exception 发布，那个例外**明文允许**用它编译出来的程序按任意许可分发。

---

## 三、这张表为什么是判据而不是文档

一份手写的依赖清单，唯一的结局是**慢慢与现实脱节**：某天有人加了一个依赖，
清单上没写，而 zip 里带着它出去了。这类失效之所以危险，是因为它看起来一切正常——
`THIRD-PARTY-NOTICES.md` 还在，读上去很完整。

所以 `--release` 会做这件事：构建完，读产物的 `PYZ-00.toc`（那里是**实际被冻结的**
每一个 Python 模块），用 `importlib.metadata.packages_distributions()` 反查发行包，
再与上面 `frozen-set` 之间的表逐条比对——**多一个、少一个、版本对不上，都红**。
两份东西不一样的时候，不能自动决定该改哪一边；工具只负责让差异可见。
