"""云端后端在 ``MainWindow`` 上的接线：清单从哪儿来、嵌入从哪儿挑、话是怎么说的。

**不构造真的 ``MainWindow``**（它一连 Ollama、起几个线程、加载 WebEngine）。这里把
``MainWindow`` 的那几个方法**赋成类属性**给一个只带必要属性的替身——
``test_ui_agent_slots.py``、``test_ui_chat_persona.py`` 用的是同一套办法：替身缺了
哪个属性会当场 ``AttributeError``，而不是悄悄测了别的东西。

**为什么值得单独一个文件。** 这一批改动的每一处失效方式都是**沉默**的：

* **漏掉那趟专喂嵌入的 Ollama 清单** → ``pick_embedding_model`` 恒返回空串，
  ``MemorySyncWorker`` 回一句"没有可用的嵌入模型"，**语义召回静默退化成纯词法**。
  不报错、不崩、界面上只表现为"记忆检索老是查不到东西"——P3a 那次（纯层全绿、
  接线不在）是同一种病史。
* **状态文案漏了 provider** → 状态栏写着"Ollama 就绪"、请求却发去了 deepseek。
  用户按那句话去查本机 Ollama，而它一点问题没有。
* **``last_model`` 不分键** → 切一次云端，本机原来选的模型名被覆盖掉。
* **云端地址不可信时构造函数抛在槽里** → PySide6 吞掉异常，``_backend`` 停在**上一个**
  对象上：下一轮提问发给一个用户已经改掉的地址，界面上没有任何痕迹。
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import pytest

from biosnn_memterm.llm.base import ChatBackendError
from biosnn_memterm.ui import main_window as wire
from biosnn_memterm.ui.main_window import MainWindow
from biosnn_memterm.ui.settings_store import (
    PROVIDER_CLOUD,
    PROVIDER_OLLAMA,
    BackendSettings,
)

pytestmark = pytest.mark.gui

_CLOUD = BackendSettings(
    provider=PROVIDER_CLOUD,
    host="http://192.168.1.9:11434",
    allow_remote=True,
    cloud_base_url="https://api.deepseek.com/v1",
    api_key="sk-x",
)
_LOCAL = BackendSettings(provider=PROVIDER_OLLAMA, host="http://127.0.0.1:11434")

_CHAT_ENTRY: dict[str, Any] = {"name": "deepseek-chat", "capabilities": ("completion",)}
_EMBED_ENTRY: dict[str, Any] = {"name": "nomic-embed-text", "capabilities": ("embedding",)}


@pytest.fixture(scope="module")
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ── 替身 ────────────────────────────────────────────────────────────────────


class FakeSignal:
    """一个只记槽、能被手动 ``emit`` 的信号。

    真信号要一个 ``QObject`` 才算活着，而这里要测的是**"谁接在那根线上"**——
    手动 emit 正是把那根线拉直来看。
    """

    def __init__(self) -> None:
        self.slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        self.slots.append(slot)

    def emit(self, *args: Any) -> None:
        for slot in self.slots:
            slot(*args)


class FakeWorker:
    """``ModelListWorker`` 的替身：记下**收到的那份后端设置**。"""

    def __init__(self, backend: BackendSettings) -> None:
        self.backend = backend
        self.loaded = FakeSignal()
        self.failed = FakeSignal()


class FakeThread:
    """``QThread`` 的替身。真线程在这里只会让测试变成一次竞速。"""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.started = 0
        self.quit_calls = 0

    def start(self) -> None:
        self.started += 1

    def quit(self, *_args: Any) -> None:  # noqa: N802 — 与 Qt 同名
        # Qt 允许把带参信号接到少参（甚至零参）的槽上，多出来的部分会被丢掉——
        # ``worker.loaded`` 就是一路接到 ``thread.quit`` 的，所以这里必须收得下。
        self.quit_calls += 1

    def isRunning(self) -> bool:  # noqa: N802 — 与 Qt 同名
        return False


class FakeSettings:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})
        self.written: list[tuple[str, Any]] = []

    def value(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802 — 与 Qt 同名
        self.values[key] = value
        self.written.append((key, value))


class FakeCombo:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.current = -1

    def blockSignals(self, _on: bool) -> None:  # noqa: N802 — 与 Qt 同名
        pass

    def clear(self) -> None:
        self.items = []
        self.current = -1

    def addItem(self, label: str, data: str) -> None:  # noqa: N802 — 与 Qt 同名
        self.items.append((label, data))
        if self.current < 0:
            self.current = 0

    def findData(self, data: Any) -> int:  # noqa: N802 — 与 Qt 同名
        for index, (_label, value) in enumerate(self.items):
            if value == data:
                return index
        return -1

    def count(self) -> int:
        return len(self.items)

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 — 与 Qt 同名
        self.current = index

    def currentData(self) -> str | None:  # noqa: N802 — 与 Qt 同名
        if 0 <= self.current < len(self.items):
            return self.items[self.current][1]
        return None

    def set_current(self, name: str) -> None:
        self.items = [(name, name)]
        self.current = 0


class Recorder:
    """对话区替身：只记调用。"""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def set_status(self, text: str) -> None:
        self.calls.append(("setStatus", text))

    def append_message(self, role: str, text: str, meta: dict | None = None) -> None:
        self.calls.append(("appendMessage", role, text, meta))

    @property
    def texts(self) -> list[str]:
        """**只数真正写进对话区的那几句。** ``set_status`` 是状态行，不是一条消息——
        把它一起数进来的话，"只写了一条"这类断言会被状态行凑够数。"""
        return [call[2] for call in self.calls if call[0] == "appendMessage"]


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[BackendSettings] = []

    def save_backend(self, backend: BackendSettings) -> None:
        self.saved.append(backend)


class Stub:
    """``MainWindow`` 上这一批方法，以及它们碰得到的东西。"""

    _host = MainWindow._host
    _allow_remote = MainWindow._allow_remote
    _current_model_id = MainWindow._current_model_id
    _last_model_key = MainWindow._last_model_key
    _refresh_embedding_models = MainWindow._refresh_embedding_models
    _on_embedding_models_loaded = MainWindow._on_embedding_models_loaded
    _on_embedding_models_failed = MainWindow._on_embedding_models_failed
    _on_models_loaded = MainWindow._on_models_loaded
    _on_model_changed = MainWindow._on_model_changed
    _on_backend_changed = MainWindow._on_backend_changed
    _on_embedding_ready = MainWindow._on_embedding_ready

    def __init__(self, backend: BackendSettings = _LOCAL) -> None:
        self._backend_settings = backend
        self._backend = None
        self._model_windows: dict[str, int] = {}
        self._model_thread = None
        self._embedmodels_thread = None
        self._streaming = False
        self.model_combo = FakeCombo()
        self._settings = FakeSettings()
        self._settings_store = FakeStore()
        self.chat_view = Recorder()
        self._embedding = None
        self._embedding_model = ""
        self._embedding_note = ""
        # 记账：`_track` 在真窗口里接线程，这里只要知道"起了几对、叫什么名字"。
        self.tracked: list[tuple[str, str]] = []
        self.notes: list[str] = []
        self.up: list[str] = []
        self.down: list[str] = []
        self.setup_calls: list[list[dict[str, Any]]] = []
        self.pushed = 0
        self.refreshes = 0

    def _track(self, _thread: Any, _worker: Any, *, what: str, slot: str) -> None:
        self.tracked.append((what, slot))

    def _start_memory_setup(self, models: list[dict[str, Any]]) -> None:
        self.setup_calls.append(models)

    def _on_models_failed(self, detail: str) -> None:
        """对话那一趟失败：**走真件**（它自己会落到 ``_set_backend_down``）。

        不转发给真件的话，这里测的就只是"信号接上了"，而"云端下摆不摆兜底预设"
        那一半逻辑整个跑不到——一条恒绿的假守卫。
        """
        MainWindow._on_models_failed(self, detail)

    def _push_context_to_memory_window(self) -> None:
        self.pushed += 1

    def _set_backend_up(self, text: str) -> None:
        self.up.append(text)

    def _set_backend_down(self, detail: str) -> None:
        self.down.append(detail)

    def _note_backend(self, text: str) -> None:
        self.notes.append(text)

    def _refresh_models(self) -> None:  # type: ignore[override]
        self.refreshes += 1
        MainWindow._refresh_models(self)


# ── 清单：几条线、各问谁 ────────────────────────────────────────────────────


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[FakeWorker]:
    """把 ``ModelListWorker`` / ``QThread`` 换成替身，返回"造出来的 worker"清单。"""
    workers: list[FakeWorker] = []

    def make_worker(backend: BackendSettings) -> FakeWorker:
        worker = FakeWorker(backend)
        workers.append(worker)
        return worker

    monkeypatch.setattr(wire, "ModelListWorker", make_worker)
    monkeypatch.setattr(wire, "QThread", FakeThread)
    return workers


def test_a_local_backend_runs_one_list_and_feeds_the_embedding_picker(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """本机时**只有一趟**，而且那一份清单同时喂对话下拉框与嵌入模型。

    先钉住没改的那一半：改动全在云端那一侧，本机这条路上多跑一趟就是白费一次
    网络往返，还会让两个 worker 争同一份 ``_embedding``。
    """
    window = Stub(_LOCAL)

    MainWindow._refresh_models(window)

    assert [worker.backend.provider for worker in made] == [PROVIDER_OLLAMA]

    made[0].loaded.emit([_CHAT_ENTRY, _EMBED_ENTRY])

    assert window.setup_calls == [[_CHAT_ENTRY, _EMBED_ENTRY]]
    assert window.up == ["Ollama 就绪 · 1 个对话模型（另有 1 个嵌入模型未列出）"]


def test_a_cloud_backend_still_asks_ollama_for_the_embedding_models(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """**这一条是整块接线的守卫。**

    云端那一份 ``/models`` 里没有嵌入模型（多数厂商根本没有嵌入端点），所以必须
    **另起一趟**问本机 Ollama。漏掉它的后果不是一条错误，是语义召回静默退化成纯
    词法——用户只会觉得"记忆检索老是查不到东西"。
    """
    window = Stub(_CLOUD)

    MainWindow._refresh_models(window)

    assert [worker.backend.provider for worker in made] == [PROVIDER_CLOUD, PROVIDER_OLLAMA]
    # 本机那一侧的两个端**照样传下去**：切云端只换了"对话从哪来"，自建内网推理机
    # 的人不该因为切了云端就把嵌入指回 127.0.0.1。
    assert made[1].backend.host == "http://192.168.1.9:11434"
    assert made[1].backend.allow_remote is True
    # 两趟是两个槽位。共用一个的话，后起的会把先起的引用挤掉，被挤掉的那个线程
    # 就变成"没人等的线程"。
    assert [slot for _what, slot in window.tracked] == ["model", "embedmodels"]


def test_the_embedding_list_only_ever_feeds_the_memory_setup(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """那一趟的产物**只喂嵌入**，不灌对话下拉框。

    灌进去的话，选择框里会出现 ``nomic-embed-text`` 这种不能对话的东西，用户选中
    它之后错误要到第一次提问才暴露，且报的是与服务端语义无关的错。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[1].loaded.emit([_EMBED_ENTRY])

    assert window.setup_calls == [[_EMBED_ENTRY]]
    assert window.model_combo.items == []


def test_the_cloud_chat_list_never_picks_an_embedding_model(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """反过来也一样：云端那一份清单**不许**交给嵌入挑选。

    交给它的话 ``pick_embedding_model`` 恒返回空串，于是 ``MemorySyncWorker`` 会回
    一句"没有可用的嵌入模型"——把刚刚由另一趟接好的语义通道**拆掉**。两条线交叉
    的症状比"少一条线"更难查：它看起来像 Ollama 那边时好时坏。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[0].loaded.emit([_CHAT_ENTRY])

    assert window.setup_calls == []
    assert window.up == ["云端服务就绪 · 1 个对话模型"]


def test_a_cloud_list_that_is_empty_names_the_likely_cause(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """云端回空清单**不是"没有模型"**：多半是这把 key 的权限或余额挡着，而接口
    照样回 200。照抄 Ollama 那句"先装一个模型"会把人支去一个云端根本没有的动作。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[0].loaded.emit([])

    assert len(window.down) == 1
    assert "API key" in window.down[0]
    assert "装一个模型" not in window.down[0]


def test_a_failed_cloud_list_still_offers_something_to_pick(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """云端拉不到清单时**先摆一份预设**，用户离线也能选一个。

    一个空的下拉框会让人以为"云端一个模型都没有"，于是去查一件根本不存在的事；
    而这个位置很常见——地址对了、key 也填了，只是这一趟没通。预设只是候选：
    真正的原因照常写在状态行里，不冒充"服务端确实有这些"。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[0].failed.emit("连不上 https://api.deepseek.com/v1（超时）")

    assert [data for _label, data in window.model_combo.items] == [
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    assert window.down == ["连不上 https://api.deepseek.com/v1（超时）"]


def test_a_failed_local_list_offers_nothing(qapp: Any, made: list[FakeWorker]) -> None:
    """本机拉不到清单 = Ollama 没起来，摆一份模型名只会让人以为它起来了。"""
    window = Stub(_LOCAL)
    MainWindow._refresh_models(window)

    made[0].failed.emit("连不上 http://127.0.0.1:11434")

    assert window.model_combo.items == []


def test_a_transient_cloud_failure_does_not_wipe_a_working_list(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """清单已经拉到了，之后某一次重探失败：**不许把清单清空重灌预设**。

    那会把用户正选着的模型一起抹掉，而失败可能只是一次网络抖动——症状是"用着用着
    模型自己变成了 deepseek-chat"。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)
    made[0].loaded.emit([{"name": "my-finetune", "capabilities": ("completion",)}])

    made[0].failed.emit("超时")

    assert [data for _label, data in window.model_combo.items] == ["my-finetune"]


def test_the_local_list_that_is_empty_keeps_its_own_wording(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """本机那一句不为云端改口径。两条话术各自指对方向，是这一整块的用处所在。"""
    window = Stub(_LOCAL)
    MainWindow._refresh_models(window)

    made[0].loaded.emit([])

    assert window.down == ["Ollama 在运行，但一个模型都没有。先装一个模型再点 ⟳"]


def test_a_failed_embedding_list_names_the_reason_and_spares_the_chat_backend(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """本机那一趟拉不到清单时：**说原因，但不把整条后端标成不可用**。

    走 ``_set_backend_down`` 的话，对话在云端明明是好的，用户会去查那把 key 和那个
    地址。而那句"装一个嵌入模型再点 ⟳"在 Ollama 没起来时是一句误导——照着装完还是
    连不上，所以原因必须一起说出来。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[1].failed.emit("连不上 http://192.168.1.9:11434（服务没起来）")

    assert window.down == []
    assert window._embedding is None
    assert "连不上 http://192.168.1.9:11434" in window._embedding_note
    # 落到与"没装嵌入模型"**同一个收口**：对话区那句仍然是一句，不是两句。
    assert len([text for text in window.chat_view.texts if "记忆只用语汇匹配" in str(text)]) == 1


# ── 会话里记的是哪一类后端 ──────────────────────────────────────────────────


def test_the_session_records_which_kind_of_backend_answered(qapp: Any) -> None:
    """``model_ref`` 的前缀按后端类型分，且与两个后端模块自己的 ``model_id`` 同源。

    前缀写错/漏掉的话，翻旧会话时看不出那句话出自哪一类后端——而这个字段存在的
    唯一理由就是这个。
    """
    local = Stub(_LOCAL)
    local.model_combo.set_current("qwen3-4b")
    cloud = Stub(_CLOUD)
    cloud.model_combo.set_current("deepseek-chat")

    assert MainWindow._current_model_id(local) == "ollama/qwen3-4b"
    assert MainWindow._current_model_id(cloud) == "openai-compat/deepseek-chat"
    # 没选模型时是空串，**不是** "ollama/None"。
    assert MainWindow._current_model_id(Stub(_LOCAL)) == ""


def test_the_two_backends_remember_their_own_last_model(qapp: Any) -> None:
    """``last_model`` 分键。单键不会串，但**有损**：切一次云端，本机原来选的那个
    模型名就被 ``deepseek-chat`` 覆盖掉了，切回来时下拉框停在第一个模型上。"""
    cloud = Stub(_CLOUD)
    cloud.model_combo.set_current("deepseek-chat")
    local = Stub(_LOCAL)
    local.model_combo.set_current("qwen3-4b")

    assert MainWindow._last_model_key(cloud) == "backend/last_model_cloud"
    assert MainWindow._last_model_key(local) == "backend/last_model"


def test_a_cloud_list_with_no_memory_picks_our_default_not_the_first(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """**云端没记住上次选的时，不能认清单第一条。**

    :func:`~biosnn_memterm.llm.openai_compat.list_remote_models` 按名字排序，而一个
    云端账户上几十上百个模型是常态——"第一条"因此接近随机，而它正是用户切过去之后
    看到的第一个模型。本机那条路照旧认第一条（Ollama 自己给的顺序有意义）。
    """
    window = Stub(_CLOUD)

    MainWindow._refresh_models(window)
    made[0].loaded.emit(
        [
            {"name": "aaa-first-alphabetically", "capabilities": ("completion",)},
            {"name": "deepseek-chat", "capabilities": ("completion",)},
            {"name": "zzz-last", "capabilities": ("completion",)},
        ]
    )

    assert window.model_combo.currentData() == "deepseek-chat"


# ── 构造对话后端 ────────────────────────────────────────────────────────────


class FakeCloudBackend:
    """``OpenAICompatChatBackend`` 的替身：记下构造参数，不发请求。"""

    made: list[dict[str, Any]] = []
    boom: str = ""

    def __init__(self, **fields: Any) -> None:
        if FakeCloudBackend.boom:
            raise ChatBackendError(FakeCloudBackend.boom)
        FakeCloudBackend.made.append(fields)
        self.model = fields["model"]
        self.context_window = fields.get("context_window") or 4096

    @property
    def model_id(self) -> str:
        return f"openai-compat/{self.model}"


@pytest.fixture
def cloud_backend(monkeypatch: pytest.MonkeyPatch) -> type[FakeCloudBackend]:
    FakeCloudBackend.made = []
    FakeCloudBackend.boom = ""
    monkeypatch.setattr(wire, "OpenAICompatChatBackend", FakeCloudBackend)
    monkeypatch.setattr(wire, "OllamaChatBackend", lambda **_kw: None)
    return FakeCloudBackend


def test_the_cloud_backend_is_built_with_the_window_the_server_reported(
    qapp: Any, cloud_backend: type[FakeCloudBackend], made: list[FakeWorker]
) -> None:
    """清单里已经带了窗口就用它，**不问第二遍**。

    ``context_window=None`` 会让构造函数去发一次 ``GET /models``，而构造发生在界面
    线程上——一次网络往返就够让窗口冻住几百毫秒到十秒，症状正是"明明是网络问题却
    看起来像程序卡死"。
    """
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)
    entry = {"name": "deepseek-chat", "capabilities": ("completion",), "context_window": 131072}

    made[0].loaded.emit([entry])

    assert cloud_backend.made[0]["context_window"] == 131072
    assert cloud_backend.made[0]["api_key"] == "sk-x"
    assert cloud_backend.made[0]["base_url"] == "https://api.deepseek.com/v1"


def test_a_window_the_server_did_not_report_falls_back_to_asking(
    qapp: Any, cloud_backend: type[FakeCloudBackend], made: list[FakeWorker]
) -> None:
    """服务端没报窗口时传 ``None``——那正是"去问一次"的表达，不是 0。"""
    window = Stub(_CLOUD)
    MainWindow._refresh_models(window)

    made[0].loaded.emit([_CHAT_ENTRY])

    assert cloud_backend.made[0]["context_window"] is None


def test_an_untrusted_cloud_address_never_leaves_the_old_backend_in_place(
    qapp: Any, cloud_backend: type[FakeCloudBackend], made: list[FakeWorker]
) -> None:
    """地址不可信时构造函数会拒（非 https 且非本机）。**异常不能抛在槽里**：
    PySide6 会吞掉它，于是 ``_backend`` 停在**上一个**对象上——下一轮提问发给一个
    用户已经改掉的地址，而界面上没有任何痕迹。
    """
    window = Stub(_CLOUD)
    window._backend = "上一轮的后端"
    cloud_backend.boom = "云端地址必须是 https（或本机 http）：key 会随每次请求发出去"
    MainWindow._refresh_models(window)

    made[0].loaded.emit([_CHAT_ENTRY])

    assert window._backend is None, "留着上一个后端 = 静默地把请求发去旧地址"
    assert window.down and "https" in window.down[0]


# ── 应用新设置 ──────────────────────────────────────────────────────────────


def test_changing_only_the_api_key_still_re_probes(qapp: Any, made: list[FakeWorker]) -> None:
    """只换 key 也算换后端。

    判据以前是 ``host`` 与 ``allow_remote`` 两个字段——**只换 key 会被判成"没变化"**，
    于是新 key 存了、界面说"已应用"，而列表还是拿旧 key 拉的（或者干脆停在上一份
    清单上）。整份 dataclass 相等比较就不会漏，加字段也不会。
    """
    window = Stub(_CLOUD)

    MainWindow._on_backend_changed(window, replace(_CLOUD, api_key="sk-new"))

    assert window.refreshes == 1
    assert window._backend_settings.api_key == "sk-new"
    assert window._settings_store.saved[-1].api_key == "sk-new"
    assert window.notes == ["已改用 https://api.deepseek.com/v1，正在重新探测…"]


def test_applying_without_changing_anything_does_not_re_probe(
    qapp: Any, made: list[FakeWorker]
) -> None:
    """按了「应用设置」但一个字没改：不重探。

    重探是白费一次网络往返，还会把模型下拉框清空重建——一个用户没要求的中断。
    """
    window = Stub(_CLOUD)

    MainWindow._on_backend_changed(window, _CLOUD)

    assert window.refreshes == 0
    assert window._settings_store.saved == []
    assert window.notes == ["后端设置没有变化。"]


def test_switching_the_provider_alone_is_a_change(qapp: Any, made: list[FakeWorker]) -> None:
    """地址一个字没改、只把类型从本机换成云端，也必须重探——不然清单还是本机那份。"""
    window = Stub(_LOCAL)
    same_addresses = replace(_LOCAL, provider=PROVIDER_CLOUD, api_key="sk-x")

    MainWindow._on_backend_changed(window, same_addresses)

    assert window.refreshes == 1
    assert window._backend_settings.is_cloud


def test_a_change_mid_generation_is_refused_not_queued(qapp: Any, made: list[FakeWorker]) -> None:
    """生成中改后端：说清楚，不动。

    换掉之后下一轮会用另一个服务端，而两轮回复来自两个模型——界面上看不出来。
    """
    window = Stub(_CLOUD)
    window._streaming = True

    MainWindow._on_backend_changed(window, replace(_CLOUD, api_key="sk-new"))

    assert window.refreshes == 0
    assert window._backend_settings.api_key == "sk-x", "拒绝了就不该已经改掉"
    assert "正在生成" in window.notes[0]
