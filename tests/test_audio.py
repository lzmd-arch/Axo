"""``axo.audio``：PCM 与 WAV 的来回、时长、认不出来时的收场。

这个模块三家共用（采集、预算、播放），错一处的症状是**噪音或预算算飞**而不是报错，
所以往返、坏输入、边角三个方向都要钉住。stdlib only——这条腿不碰 Qt 也不碰网络。
"""

from __future__ import annotations

from axo.audio import AudioClip, decode_wav, wav_seconds


def _clip(seconds: float = 1.0, rate: int = 16000) -> AudioClip:
    """一秒 16 kHz 单声道 16 位的静音。"""
    return AudioClip(pcm=b"\x00\x00" * int(rate * seconds), sample_rate=rate)


def test_a_round_trip_keeps_every_field() -> None:
    """PCM → WAV → PCM：**采样率/声道/位宽/字节逐一相同**。

    少一个字段的检查就会放过"头写对了、数据被当成立体声切"这一类：单声道变双声道
    的症状是播放速度翻倍、音高变调，而它看起来像引擎坏了。
    """
    original = _clip(seconds=0.5, rate=22050)

    back = decode_wav(original.as_wav())

    assert back is not None
    assert back.pcm == original.pcm
    assert back.sample_rate == 22050
    assert back.channels == 1
    assert back.sample_width == 2


def test_duration_is_computed_from_pcm() -> None:
    """时长按**裸 PCM** 算（这个对象手里就是数据，不该再去读自己的头）。"""
    assert _clip(seconds=2.0).duration_s == 2.0
    assert _clip(seconds=0.5).duration_s == 0.5


def test_base64_is_the_wav_without_a_data_prefix() -> None:
    """base64 出去的是 **WAV**、**不带 ``data:`` 前缀**——同 ``ChatMessage.images``
    的约定（前缀由后端序列化时补）。两处不一致的症状是引擎认不出格式。"""
    import base64

    clip = _clip()
    encoded = clip.as_base64()

    assert not encoded.startswith("data:")
    assert decode_wav(base64.b64decode(encoded)) is not None


def test_garbage_in_returns_none_not_an_exception() -> None:
    """认不出来 → ``None``，**不抛**。

    这条路径上"认不出"是常态（引擎回了个 mp3、回了半段），各调用方要的收场一致：
    当没有这段音频。抛的话它会在后台线程里炸出来，而那里没有人接得住。
    """
    assert decode_wav(b"") is None
    assert decode_wav(b"RIFF....WAVEjunkjunkjunk") is None
    assert decode_wav(b"not a wav at all") is None
    assert wav_seconds(b"not a wav at all") is None


def test_wav_seconds_matches_the_duration() -> None:
    """``wav_seconds`` 与 ``duration_s`` 是同一个数的两个入口（预算读的是字节）。"""
    clip = _clip(seconds=3.0)
    assert wav_seconds(clip.as_wav()) == 3.0


def test_an_empty_clip_says_so() -> None:
    assert AudioClip(pcm=b"").empty is True
    assert _clip().empty is False
