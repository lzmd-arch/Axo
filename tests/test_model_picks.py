""" "用哪个模型做哪件事"的三条判据。

它们看着像几行 if，但**判错的方式是安静的**：

* 把嵌入模型选进对话下拉框——错误推迟到第一次提问才暴露，且报的是一个与服务端
  语义无关的错（这条已经发生过一次，见 README 排障表）；
* 把嵌入模型挑去跑对话冒烟脚本——同一个坑的第二个入口，而且更隐蔽：``/api/tags``
  **按名字排序**，``nomic-embed-text`` 就排在最前，于是"取第一个"看着完全合理；
* 把语义通道接到一个非默认的嵌入模型上——``space_id`` 里带模型名，换一个就是
  另一个空间，库里已有的向量一条都用不上，而界面上什么都没变。

所以三条判据各钉一组反例。
"""

from __future__ import annotations

from biosnn_memterm.llm.ollama import is_chat_capable, pick_chat_model, pick_embedding_model


def entry(name: str, *capabilities: str, short: str = "") -> dict[str, object]:
    return {
        "name": name,
        "short_name": short or name.split(":")[0],
        "capabilities": capabilities,
    }


# ── 谁能对话 ────────────────────────────────────────────────────────────────


def test_a_completion_model_can_chat() -> None:
    assert is_chat_capable(entry("qwen3-4b:latest", "tools", "thinking", "completion"))


def test_an_embedding_only_model_cannot_chat() -> None:
    """本机实测：``nomic-embed-text`` 只报 ``['embedding']``。"""
    assert not is_chat_capable(entry("nomic-embed-text:latest", "embedding"))


def test_a_model_that_reports_nothing_is_treated_as_usable() -> None:
    """问不到能力时**当作可用**。那是"不知道"，不是"不行"。

    当成"不行"会让老版本 Ollama 上的模型全部从下拉框里消失——而那个症状看起来
    像是真的没装模型。
    """
    assert is_chat_capable(entry("mystery:7b"))
    assert is_chat_capable({"name": "mystery:7b", "capabilities": []})


# ── 谁做嵌入 ────────────────────────────────────────────────────────────────


def test_the_preferred_embedding_model_wins_even_when_it_is_not_first() -> None:
    """``/api/tags`` 是按名字排序的，所以"第一个"随时可能变——不能靠顺序。"""
    models = [entry("bge-m3:latest", "embedding"), entry("nomic-embed-text:latest", "embedding")]
    assert pick_embedding_model(models) == "nomic-embed-text:latest"


def test_the_preferred_model_is_matched_without_its_default_tag() -> None:
    """标签是 Ollama 在没写标签时补的，库里存的写法不由我们决定。"""
    models = [entry("nomic-embed-text:latest", "embedding", short="nomic-embed-text")]
    assert pick_embedding_model(models) == "nomic-embed-text:latest"


def test_a_chat_model_is_never_picked_as_the_embedder() -> None:
    """只有对话模型时返回空串，**不拿一个对话模型凑数**。

    凑数的后果不是报错，而是库里多出一个没有向量的空间——语义通道永远返回零条，
    看起来和"库里没有相关的记忆"一模一样。
    """
    assert pick_embedding_model([entry("qwen3-4b:latest", "tools", "completion")]) == ""


def test_an_empty_list_gives_an_empty_name() -> None:
    assert pick_embedding_model([]) == ""


def test_a_fallback_embedder_is_used_when_the_preferred_one_is_absent() -> None:
    """装的是别的嵌入模型时照用——**但只能有一个候选**时才允许有歧义。

    两个都不是认定的那个时取第一个：这是在"用某个模型"与"完全不接语义通道"之间
    的取舍，前者至少让记忆能被语义召回，而后者会让刚写下的记忆连一次都召不回。
    """
    models = [entry("bge-m3:latest", "embedding"), entry("nomic-embed-text-v2:latest", "embedding")]
    assert pick_embedding_model(models) == "bge-m3:latest"


# ── 谁被挑来对话 ────────────────────────────────────────────────────────────


def test_an_embedding_model_is_never_picked_as_the_chat_model() -> None:
    """**这一个坑是"按顺序取第一个"的默认写法自己造成的。**

    ``list_local_models`` 按名字排序，``nomic-embed-text`` 排在最前——于是取队首
    拿到的是一个不能对话的模型。老写法 ``models[0]["name"]`` 就是这么错的。
    """
    models = [
        entry("nomic-embed-text:latest", "embedding"),
        entry("qwen3-4b:latest", "tools", "thinking", "completion"),
    ]
    assert pick_chat_model(models) == "qwen3-4b:latest"


def test_only_embedding_models_gives_an_empty_name() -> None:
    """全是嵌入模型时返回空串，**不拿嵌入模型凑数**。

    与 :func:`pick_embedding_model` 那条反例同源：凑数的后果不是当场报错，而是
    推迟到真正用起来才暴露，且那时报的错与服务端语义无关。
    """
    assert pick_chat_model([entry("nomic-embed-text:latest", "embedding")]) == ""


def test_no_models_at_all_gives_an_empty_name() -> None:
    assert pick_chat_model([]) == ""


def test_a_model_reporting_nothing_is_still_picked_for_chat() -> None:
    """问不到能力的**当作可用**——与 :func:`is_chat_capable` 同一条取舍。

    当成"不行"的后果在界面上已经写过一次：老版本 Ollama 上的模型会全部消失，
    而那个症状看起来像是真的没装模型。
    """
    assert pick_chat_model([entry("mystery:7b")]) == "mystery:7b"
