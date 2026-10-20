"""音乐工具：搜歌、歌单、歌词、播放与播放控制。

**这一组不过确认闸门**（用户拍板）：播放是可逆的本地动作，随时能暂停；而语音点歌
每次弹一张卡片，等于把“放首歌”变成“先点一下”。搜索/歌词/状态是只读的，照旧并行。

失败一律是一句人话（同目录里其余工具的纪律）：没有通道、没配歌库、没找到、播放器
报错，都走 :meth:`ToolOutcome.failure`，不抛。
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..llm.base import ToolSpec
from .tools import ToolContext, ToolOutcome, argument_int, argument_str

__all__ = [
    "MusicControlTool",
    "MusicLyricsTool",
    "MusicPlayTool",
    "MusicPlaylistsTool",
    "MusicSearchTool",
    "MusicStatusTool",
]

_LIMIT_TEXT = 4000


def _clip(text: str, limit: int = _LIMIT_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _channel(ctx: ToolContext) -> Any | None:
    music = getattr(ctx, "music", None)
    return music if music is not None and getattr(music, "available", False) else None


def _no_channel() -> ToolOutcome:
    return ToolOutcome.failure(
        "这一轮没有接上音乐库。告诉用户去「设置 → 音乐」选一个文件夹并扫描。"
    )


class MusicSearchTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_search",
        description=(
            "在本地音乐库里按歌名/艺术家/专辑搜索。只读，不要用它播放；"
            "要播放就把结果里的 id 交给 music_play。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索词；空串列出前几首。"},
                "limit": {"type": "integer", "description": "最多返回几首（1–30，默认 10）。"},
            },
            "required": [],
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        query = argument_str(arguments, "query", required=False)
        limit = max(1, min(30, argument_int(arguments, "limit", default=10)))
        found = music.search(query, limit=limit)
        if not found:
            return ToolOutcome.failure(f"音乐库里没有找到与「{query}」相关的曲目。")
        lines = [
            f"{index}. [{track.track_id}] {track.title}"
            + (f" — {track.artist}" if track.artist else "")
            + (f"（{track.album}）" if track.album else "")
            for index, track in enumerate(found, start=1)
        ]
        return ToolOutcome.success(_clip("\n".join(lines)))


class MusicPlaylistsTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_playlists",
        description="列出音乐库里的歌单（.m3u/.m3u8）与曲目数。只读。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        playlists = music.playlists()
        if not playlists:
            return ToolOutcome.success("音乐库里没有歌单文件（.m3u/.m3u8）。")
        lines = [f"{one.name}（{len(one.track_ids)} 首）" for one in playlists]
        return ToolOutcome.success(_clip("\n".join(lines)))


class MusicLyricsTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_lyrics",
        description=(
            "取一首歌的歌词；不给 query 就是当前播放的这首。有同步歌词时带时间戳。"
            "只读。没有歌词就如实说没有，不要编。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "歌名/艺术家；留空取当前曲目。"},
            },
            "required": [],
        },
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        query = argument_str(arguments, "query", required=False)
        track, result = music.lyrics_for(query)
        if track is None:
            return ToolOutcome.failure("没有找到这首曲目。先用 music_search 确认歌名。")
        lyrics = result.lyrics
        if not lyrics.ok:
            return ToolOutcome.failure(result.reason or f"《{track.title}》没有歌词。")
        source = {"lrc": "同目录 .lrc", "embedded": "文件内嵌", "lrclib": "LRCLIB"}.get(
            lyrics.source, lyrics.source
        )
        lines = [f"《{track.title}》—— {track.artist}（来源：{source}）"]
        if lyrics.synced:
            from ..music.library import format_duration

            lines.extend(f"[{format_duration(line.at_s)}] {line.text}" for line in lyrics.lines)
        else:
            lines.extend(line.text for line in lyrics.lines)
        return ToolOutcome.success(_clip("\n".join(lines)))


class MusicStatusTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_status",
        description="当前在放什么、进度、音量、随机/循环与队列位置。只读快照。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    requires_approval: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = True

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        state = music.state()
        if state.track is None:
            return ToolOutcome.success("现在没有在播放的曲目。", replayable=False)
        from ..music.library import format_duration

        text = (
            f"{'正在播放' if state.playing else '已暂停'}：{state.track.display}\n"
            f"{format_duration(state.position_s)} / {format_duration(state.duration_s)}"
            f"　音量 {state.volume}%　"
            f"随机 {'开' if state.shuffle else '关'}　循环 {state.repeat.value}"
            f"　队列 {state.queue_index + 1}/{state.queue_length}"
        )
        return ToolOutcome.success(text, replayable=False)


class MusicPlayTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_play",
        description=(
            "播放音乐。给 query 就搜第一首，给 track_id 就精确播，给 playlist 就播整张歌单。"
            "三个至少给一个。这是本地播放，不需要确认。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "歌名或艺术家。"},
                "track_id": {"type": "string", "description": "music_search 结果方括号里的 id。"},
                "playlist": {"type": "string", "description": "歌单名（music_playlists 列出的）。"},
            },
            "required": [],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        query = argument_str(arguments, "query", required=False)
        track_id = argument_str(arguments, "track_id", required=False)
        playlist = argument_str(arguments, "playlist", required=False)
        if not (query or track_id or playlist):
            return ToolOutcome.failure("要给 query、track_id 或 playlist 里的一个。")
        try:
            message = music.play(query=query, track_id=track_id, playlist=playlist)
        except Exception as exc:  # noqa: BLE001 — 服务把失败说成人话，这里只兜底
            return ToolOutcome.failure(str(exc) or "播放失败。")
        return ToolOutcome.success(message or "开始播放。", replayable=False)


class MusicControlTool:
    spec: ClassVar[ToolSpec] = ToolSpec(
        name="music_control",
        description=(
            "控制播放：pause / resume / playpause / next / prev / stop / shuffle / repeat / "
            "volume / seek。不要用它选新歌——选歌用 music_play。不需要确认。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "pause",
                        "resume",
                        "playpause",
                        "next",
                        "prev",
                        "stop",
                        "shuffle",
                        "repeat",
                        "volume",
                        "seek",
                    ],
                },
                "volume": {"type": "integer", "description": "0–100，配合 action=volume。"},
                "position_s": {"type": "number", "description": "秒，配合 action=seek。"},
                "repeat": {
                    "type": "string",
                    "enum": ["off", "all", "one"],
                    "description": "配合 action=repeat；不给就循环切换。",
                },
            },
            "required": ["action"],
        },
    )
    requires_approval: ClassVar[bool] = False

    def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        music = _channel(ctx)
        if music is None:
            return _no_channel()
        action = argument_str(arguments, "action")
        value: Any = None
        if action == "volume":
            value = argument_int(arguments, "volume", default=80)
        elif action == "seek":
            raw = arguments.get("position_s")
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                return ToolOutcome.failure("seek 要一个秒数（position_s）。")
            value = float(raw)
        elif action == "repeat":
            value = argument_str(arguments, "repeat", required=False) or None
        try:
            message = music.control(action, value)
        except Exception as exc:  # noqa: BLE001
            return ToolOutcome.failure(str(exc) or "控制失败。")
        return ToolOutcome.success(message or "已执行。", replayable=False)
