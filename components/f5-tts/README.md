# Axo 的 F5-TTS 组件

这是 **可选的第二条 TTS 路线**。Axo 0.13.0 起分发版默认用随包的 ZipVoice
（CPU、零安装、在设置里默认开启「优先使用随包千岁语音」）；这个组件给愿意用
GPU / F5 音质的人。Axo 在设置里关掉那个开关后，才会连 `http://127.0.0.1:8800`
的这个服务。

组件与主程序**分开安装**：torch 不进 Axo 的仓库/安装包，这里也不带模型权重。

## 安装

需要一个 Python 3.10/3.11。**Windows + NVIDIA**（F5 的 CUDA 路线）：

```bat
cd components\f5-tts
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

**macOS / 无 CUDA**：

```bash
cd components/f5-tts
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchaudio
uv pip install --python .venv/bin/python -r requirements.txt
```

顺序不能反：先装 CUDA 版 torch，再装 `f5-tts`；反了会拿到 CPU 版 torch，合成会慢
一个数量级。第一次运行会从 HuggingFace 下载 F5-TTS 权重（约 1.4 GB）。

## 启动

音色根就是 Axo 数据根下的 `models/voices/`；Windows 默认是
`%LOCALAPPDATA%\axo\models\voices`，macOS 是
`~/Library/Application Support/axo/models/voices`。

```bat
run_windows.bat
```

```bash
./run_macos.command
```

或手动：

```bash
.venv/bin/python server.py --voices "$HOME/Library/Application Support/axo/models/voices" --port 8800 --device auto
```

`--device` 可写 `cuda` / `mps` / `cpu` / `auto`。服务默认只监听 `127.0.0.1`，
不要在没有鉴权的情况下把它暴露到局域网。

## 音色

```text
<voices-root>/<名字>/ref.wav    10–30 秒参考音
<voices-root>/<名字>/ref.txt    参考音的逐字稿
```

`ref.txt` 必须与参考音逐字对应；对不上时音色/咬字会明显变差。用哪个音色由 Axo
设置里的音色下拉框传给请求的 `voice` 字段；空串 = 第一个可用音色。

## 契约

```http
POST /v1/audio/speech
{"input": "要念的中文", "voice": "名字", "response_format": "wav"}
→ 200 audio/wav
```

`GET /v1/audio/speech` 返回 `{"voices": [...]}`，只给人看/排障。Axo 的调用端在
`src/axo/llm/speech.py`；引擎换掉时改这个 `server.py`，主程序一行不用改。

## 许可

F5-TTS 代码按 MIT；**预训练权重按其模型卡是 CC-BY-NC-4.0**（训练数据 Emilia 的
限制），只允许非商业再分发。组件不带权重，首次运行由 `f5-tts` 从 HuggingFace
下载。参考音的使用权归提供者；不要把别人的私人录音放进公开仓库。
