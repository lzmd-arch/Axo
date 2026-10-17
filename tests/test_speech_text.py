"""把模型输出切成可以念的句子。"""

from __future__ import annotations

from axo.speech_text import clean_for_speech, split_sentences


def test_sentences_split_on_chinese_punctuation() -> None:
    sentences, rest = split_sentences("今天下雨了。你带伞了吗？我没带。", final=False)

    assert sentences == ["今天下雨了。", "你带伞了吗？", "我没带。"]
    assert rest == ""


def test_the_tail_stays_until_the_next_chunk_or_the_end() -> None:
    sentences, rest = split_sentences("第一句。第二", final=False)
    assert sentences == ["第一句。"]
    assert rest == "第二"

    sentences, rest = split_sentences("第二句。", final=True)
    assert sentences == ["第二句。"]
    assert rest == ""


def test_final_flushes_a_tail_without_punctuation() -> None:
    sentences, rest = split_sentences("就这样吧", final=True)

    assert sentences == ["就这样吧"]
    assert rest == ""


def test_periods_inside_versions_and_urls_do_not_split() -> None:
    """**不认句点**：``3.5``、``v0.6.0``、域名被切碎之后 TTS 只会念一串数字。"""
    sentences, _rest = split_sentences("版本是 v0.6.0，地址是 api.deepseek.com。", final=True)

    assert sentences == ["版本是 v0.6.0，地址是 api.deepseek.com。"]


def test_a_long_sentence_breaks_at_a_comma_first() -> None:
    body = "，".join(["这是一段很长的内容"] * 10) + "。"

    sentences, _rest = split_sentences(body, final=True, max_chars=30)

    assert len(sentences) > 1
    assert all(len(one) <= 30 for one in sentences)
    assert all(one.endswith(("，", "。")) for one in sentences)


def test_a_long_run_without_punctuation_is_hard_cut() -> None:
    sentences, _rest = split_sentences("啊" * 100, final=True, max_chars=30)

    assert len(sentences) == 4
    assert all(len(one) <= 30 for one in sentences)


def test_code_fences_are_not_spoken() -> None:
    text = "先这样：\n```python\nprint('hi')\n```\n然后再说。"

    sentences, _rest = split_sentences(text, final=True)

    assert all("print" not in one for one in sentences)
    assert all("```" not in one for one in sentences)
    assert any("先这样" in one for one in sentences)
    assert any("然后再说" in one for one in sentences)


def test_links_become_their_label_and_images_disappear() -> None:
    cleaned = clean_for_speech("看[这个页面](https://example.com)和 ![图](a.png) 就好。")

    assert "https://" not in cleaned
    assert "a.png" not in cleaned
    assert "这个页面" in cleaned


def test_inline_code_keeps_its_text() -> None:
    cleaned = clean_for_speech("打开 `config.toml` 看一下。")

    assert "config.toml" in cleaned
    assert "`" not in cleaned


def test_markdown_markers_are_stripped() -> None:
    text = "# 标题\n- 第一项\n1. 第二项\n**重点**在这里。"

    cleaned = clean_for_speech(text)

    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "重点" in cleaned


def test_emphasis_does_not_eat_snake_case_identifiers() -> None:
    """下划线不作为强调记号——否则 ``read_file`` 里的下划线会被当成开合标记。"""
    assert clean_for_speech("调用 read_file 和 write_file。") == "调用 read_file 和 write_file。"


def test_empty_input_yields_nothing() -> None:
    assert split_sentences("", final=True) == ([], "")
    assert split_sentences("   ", final=False) == ([], "")
