"""语音识别那一层：库/模型在不在、喂片怎么进、三种失败各说哪句话。

**这里的 vosk 是一个假件**（塞进 ``sys.modules``，同 ``tests/test_autostart.py`` 处理
``winreg`` 的手法）：真跑一遍要 42 MB 的模型，而 CI 上既没有模型、也不该为了几条断言去下
一个。假件同时让两件真机上看不出来的事**可断言**：模型是**第一次喂片时**才加载的（那句
"加载与说话并行"的承诺），以及"喂进来的字节数是采样宽度的整数倍"（错位一个字节的症状是
识别出莫名其妙的字，而不是报错）。

**真模型的行为不在这里量**：识得准不准、中文标点怎么给，那只能靠人按着麦克风说一句。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from axo import asr
from axo.asr import SAMPLE_RATE, Recognizer, Transcription, recognize


class FakeVosk:
    """``vosk`` 的替身：记下每次会话收到的音频，并按脚本回一句话。

    ``Model`` 的构造**被记下来**——"第一次喂片时才加载"这条承诺只能这样验（它省的是
    用户开口前那几百毫秒，而不是几条 CPU 指令）。
    """

    def __init__(self, *, text: str = "该睡了", raw: object | None = None, boom: str = "") -> None:
        self.text = text
        self.raw = raw
        self.boom = boom
        self.loaded = 0
        self.sessions: list[FakeVosk._Session] = []

        fake = self

        class _Model:
            def __init__(self, path: str) -> None:
                if fake.boom:
                    raise RuntimeError(fake.boom)
                fake.loaded += 1
                self.path = path

        class _Session:
            def __init__(self, _model: object, rate: int) -> None:
                self.rate = rate
                self.audio = b""
                fake.sessions.append(self)

            def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802 (vosk 命名)
                self.audio += data
                return True

            def FinalResult(self) -> str:  # noqa: N802
                if fake.raw is not None:
                    return fake.raw if isinstance(fake.raw, str) else json.dumps(fake.raw)
                return json.dumps({"text": fake.text})

        self.Model = _Model
        self.KaldiRecognizer = _Session


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """一个"模型已经放好了"的目录（内容无所谓，假件不看它）。"""
    path = tmp_path / "vosk-model-small-cn-0.22"
    path.mkdir()
    return path


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeVosk | None) -> None:
    """把假 vosk 塞进去；``None`` = **库没装**（把它从 ``sys.modules`` 里挖掉）。"""
    if fake is None:
        monkeypatch.setitem(sys.modules, "vosk", None)  # import 会当场失败
    else:
        monkeypatch.setitem(sys.modules, "vosk", fake)


# ── 没库 / 没模型：各一句人话 ────────────────────────────────────────────────


def test_a_missing_library_is_a_sentence_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """库没装 → 一句告诉人去哪儿装的话。**不是 ImportError。**

    这条路径跑在后台线程里，异常抛出去没有人接得住，而用户看到的是"按了没反应"。
    """
    _install(monkeypatch, None)

    result = recognize(b"\x00\x00" * 100, model_dir=Path(r"C:\nope"))

    assert result.ok is False
    assert "没有装语音识别库" in result.reason
    assert "uv sync --extra voice" in result.reason


def test_a_missing_model_says_where_to_put_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """模型没放 → 说清**放到哪个目录**（比"没有模型"多一句，而那句正是用户要的）。"""
    _install(monkeypatch, FakeVosk())

    result = recognize(b"\x00\x00" * 100, model_dir=tmp_path / "models" / "没有这个")

    assert result.ok is False
    assert "还没有语音模型" in result.reason
    assert str(tmp_path) in result.reason
    assert ".zip" in result.reason


def test_a_broken_model_is_also_a_sentence(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """模型加载失败（半个目录、版本不匹配、磁盘坏了）同样归成一句人话。"""
    _install(monkeypatch, FakeVosk(boom="模型文件不完整"))

    result = recognize(b"\x00\x00" * 100, model_dir=model_dir)

    assert result.ok is False
    assert "加载不了" in result.reason and "模型文件不完整" in result.reason


# ── 走通了 ──────────────────────────────────────────────────────────────────


def test_it_returns_what_the_engine_heard(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    _install(monkeypatch, FakeVosk(text="帮我记一下牛奶喝完了"))

    result = recognize(b"\x01\x02" * 400, model_dir=model_dir)

    assert result.ok is True
    assert result.text == "帮我记一下牛奶喝完了"
    assert result.reason == ""


def test_the_model_is_loaded_on_the_first_chunk_not_at_construction(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """**加载发生在第一片音频进来时**，不是构造时。

    这一条是那句"加载与说话并行"的承诺：用户在按完按钮之后立刻开口，而模型那一两秒正好
    盖在他说话的时间里。改成构造时加载的话，症状是"每句话的开头两个字丢掉"——而它看起来
    像识别得不准。
    """
    fake = FakeVosk()
    _install(monkeypatch, fake)

    recognizer = Recognizer(model_dir=model_dir)
    assert fake.loaded == 0, "还没喂任何音频就把模型加载了"

    recognizer.feed(b"\x01\x02" * 10)
    assert fake.loaded == 1


def test_the_sample_rate_reaches_the_engine(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """采样率要真的传下去，而且默认就是 16000。

    与 Qt 那边**必须同一个数**：不一致的症状不是报错，而是识别出一串莫名其妙的字。
    """
    fake = FakeVosk()
    _install(monkeypatch, fake)

    Recognizer(model_dir=model_dir).feed(b"\x01\x02" * 10)

    assert fake.sessions and fake.sessions[0].rate == SAMPLE_RATE == 16000


def test_a_half_sample_is_carried_over_to_the_next_chunk(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """**奇数个字节要留到下一片**，不是丢掉也不是硬喂。

    麦克风那一片一片的读法不保证按采样对齐，而错位一个字节之后每一片都错位——症状同样是
    "识别出一串莫名其妙的字"。
    """
    fake = FakeVosk()
    _install(monkeypatch, fake)

    recognizer = Recognizer(model_dir=model_dir)
    recognizer.feed(b"\x01\x02\x03")  # 3 字节 = 一个半采样
    recognizer.feed(b"\x04")  # 半个 + 半个 = 一个整的
    recognizer.finish()

    assert fake.sessions[0].audio == b"\x01\x02\x03\x04", "剩下的那半个字节没被接上"


def test_all_the_chunks_reach_the_engine_in_order(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """喂进去的音频要**一片不少、顺序不变**地到引擎那侧。"""
    fake = FakeVosk()
    _install(monkeypatch, fake)

    recognizer = Recognizer(model_dir=model_dir)
    for index in range(5):
        recognizer.feed(bytes([index]) * 100)
    recognizer.finish()

    assert fake.sessions[0].audio == b"".join(bytes([i]) * 100 for i in range(5))


# ── 没听见 / 没听清 ─────────────────────────────────────────────────────────


def test_silence_is_told_apart_from_a_missing_microphone(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """这两句是**分开的**：一个字节都没进来 vs 进来了但没听出字。

    它们对用户是两件事（前者要去看麦克风，后者只要再说一遍），而混成一句会让第一种情形
    永远查不出来。
    """
    _install(monkeypatch, FakeVosk(text=""))

    nothing = recognize(b"", model_dir=model_dir)
    heard_nothing = recognize(b"\x00\x00" * 100, model_dir=model_dir)

    assert "没有听到声音" in nothing.reason
    assert "没有听清" in heard_nothing.reason


def test_a_broken_json_result_reads_as_not_heard(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """引擎回的 JSON 读不出来时当"没听清"，而不是把那段原文当识别结果。

    把 ``{"text": ...}`` 的原文当句子交出去的话，用户会在输入框里看到一段 JSON。
    """
    _install(monkeypatch, FakeVosk(raw="{不是 JSON"))

    result = recognize(b"\x00\x00" * 100, model_dir=model_dir)

    assert result.ok is False
    assert "没有听清" in result.reason


def test_whitespace_only_result_is_not_a_sentence(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """全是空白的结果与空结果同一收场。"""
    _install(monkeypatch, FakeVosk(text="   "))

    assert recognize(b"\x00\x00" * 100, model_dir=model_dir).ok is False


# ── 形状 ────────────────────────────────────────────────────────────────────


def test_transcription_is_the_same_shape_as_the_other_receipts() -> None:
    """``ok`` 的判据是"有文本、且没有原因"——与 ``agent.tools.Capture`` 那套一模一样。

    两处形状一致不是巧合：调用方（界面）对这两种收成的处理是同一种——把 ``reason``
    说出来，或者拿 ``text`` 去做下一步。
    """
    assert Transcription(text="你好").ok is True
    assert Transcription(reason="没听清").ok is False
    assert Transcription().ok is False


def test_the_module_imports_without_vosk(monkeypatch: pytest.MonkeyPatch) -> None:
    """**没装 vosk 也能 import 这个模块。**

    CI 上跑核心腿的机器不一定装了它（``voice`` 是个 extra），而"库在不在"这件事必须
    在**调用时**变成一句人话，不是一次 import 期的 ImportError。
    """
    _install(monkeypatch, None)

    assert asr.SAMPLE_RATE == 16000
    assert callable(asr.recognize)
    assert Recognizer(model_dir=Path("nope")).problem, "库没装，却什么原因都不给"
