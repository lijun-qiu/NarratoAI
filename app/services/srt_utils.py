#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""SRT parsing, slicing, remapping and merging utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.models import const
from app.utils import utils

_TIME_LINE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)

# 字幕轨道标记（写入 SRT 时使用私有区字符，烧录前解码）
SUBTITLE_TRACK_ORIGINAL = "original"
SUBTITLE_TRACK_NARRATION = "narration"
SUBTITLE_TRACK_PICTURE = "picture_narration"
_SUBTITLE_TRACK_MARKER = "\uE000"


def encode_subtitle_track_label(label: str, text: str) -> str:
    """Encode subtitle track type into SRT text for later color routing."""
    track = (label or "").strip()
    cleaned = clean_subtitle_dialogue_text(text)
    if track in (SUBTITLE_TRACK_ORIGINAL, "原声", SUBTITLE_TRACK_PICTURE):
        if track == "原声":
            track = SUBTITLE_TRACK_ORIGINAL
        return f"{_SUBTITLE_TRACK_MARKER}{track}{_SUBTITLE_TRACK_MARKER}{cleaned}"
    return cleaned


def decode_subtitle_track_label(text: str) -> tuple[str, str]:
    """Return (track_label, visible_text)."""
    raw = (text or "").strip()
    if raw.startswith(_SUBTITLE_TRACK_MARKER):
        end = raw.find(_SUBTITLE_TRACK_MARKER, len(_SUBTITLE_TRACK_MARKER))
        if end > 0:
            track = raw[len(_SUBTITLE_TRACK_MARKER) : end]
            visible = raw[end + len(_SUBTITLE_TRACK_MARKER) :]
            return track, clean_subtitle_dialogue_text(visible)
    return "", clean_subtitle_dialogue_text(raw)


# 去掉说话人/角色前缀，只保留对白正文
_SPEAKER_PREFIX_PATTERNS = (
    re.compile(r"^说话人\s*\d+\s*[:：]\s*"),
    re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9·]{1,12}说\s*[:：]\s*"),
    re.compile(r"^【解说】\s*"),
)


def clean_subtitle_dialogue_text(text: str) -> str:
    """Remove speaker/role prefixes; keep spoken content only."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for pattern in _SPEAKER_PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned, count=1).strip()
    return cleaned


def strip_subtitle_punctuation(text: str) -> str:
    """Remove punctuation marks for on-screen subtitle display."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for punct in sorted(const.PUNCTUATIONS, key=len, reverse=True):
        cleaned = cleaned.replace(punct, "")
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class SrtEntry:
    start_ms: int
    end_ms: int
    text: str
    label: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def _time_str_to_ms(value: str) -> int:
    return int(round(utils.time_to_seconds(value.replace(".", ",")) * 1000))


def _ms_to_srt_time(ms: int) -> str:
    return utils.seconds_to_time(ms / 1000.0).replace(".", ",")


def parse_srt(content: str) -> list[SrtEntry]:
    if not content or not str(content).strip():
        return []

    entries: list[SrtEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line_idx = 1 if lines[0].isdigit() else 0
        if time_line_idx >= len(lines):
            continue
        match = _TIME_LINE_RE.search(lines[time_line_idx])
        if not match:
            continue

        start_ms = _time_str_to_ms(match.group(1))
        end_ms = _time_str_to_ms(match.group(2))
        text_lines = lines[time_line_idx + 1 :]
        raw_text = "\n".join(text_lines).strip()
        track_label, text = decode_subtitle_track_label(raw_text)
        if not text:
            continue
        entries.append(SrtEntry(start_ms=start_ms, end_ms=end_ms, text=text, label=track_label))
    return entries


def parse_srt_file(path: str) -> list[SrtEntry]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fp:
        return parse_srt(fp.read())


def entries_to_srt(entries: Iterable[SrtEntry]) -> str:
    lines: list[str] = []
    output_index = 0
    for entry in entries:
        text = encode_subtitle_track_label(entry.label, entry.text)
        if not text:
            continue
        output_index += 1
        lines.append(str(output_index))
        lines.append(
            f"{_ms_to_srt_time(entry.start_ms)} --> {_ms_to_srt_time(entry.end_ms)}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def write_srt_file(entries: Iterable[SrtEntry], output_path: str) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(entries_to_srt(entries))
    return output_path


_SCRIPT_TIMESTAMP_RANGE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}-\d{2}:\d{2}:\d{2},\d{3}$"
)


def format_timestamp_ms(ms: int) -> str:
    """脚本 timestamp 单端：HH:MM:SS,mmm（与 check_script 校验一致）。"""
    return utils.format_time(max(0, int(ms)) / 1000.0)


def normalize_script_timestamp_range(timestamp: str) -> str:
    """
    将 LLM/后处理产生的各种时间戳统一为 HH:MM:SS,mmm-HH:MM:SS,mmm。
    支持 `.` 毫秒、单段、SRT 箭头等；无法解析时返回 1 秒占位区间。
    """
    text = (timestamp or "").strip()
    if _SCRIPT_TIMESTAMP_RANGE_RE.match(text):
        return text

    normalized = (
        text.replace("-->", "-")
        .replace("—", "-")
        .replace("–", "-")
    )
    normalized = re.sub(r"\s*-\s*", "-", normalized)

    if "-" not in normalized:
        start_ms = _time_str_to_ms(normalized)
        end_ms = start_ms + 1000
        return f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(end_ms)}"

    start_text, end_text = normalized.split("-", 1)
    start_ms = _time_str_to_ms(start_text.strip())
    end_ms = _time_str_to_ms(end_text.strip())
    if end_ms <= start_ms:
        end_ms = start_ms + 1000
    return f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(end_ms)}"


def parse_timestamp_range(timestamp: str) -> tuple[int, int]:
    text = (timestamp or "").strip()
    if "-" not in text:
        return 0, 0
    start_text, end_text = text.split("-", 1)
    return _time_str_to_ms(start_text.strip()), _time_str_to_ms(end_text.strip())


def parse_edited_time_range_seconds(edited_time_range: str) -> tuple[float, float]:
    text = (edited_time_range or "").strip()
    if "-" not in text:
        return 0.0, 0.0
    start_text, end_text = text.split("-", 1)
    return utils.time_to_seconds(start_text.strip()), utils.time_to_seconds(end_text.strip())


def extract_entries_in_range(
    entries: list[SrtEntry],
    range_start_ms: int,
    range_end_ms: int,
) -> list[SrtEntry]:
    if range_end_ms <= range_start_ms:
        return []

    clipped: list[SrtEntry] = []
    for entry in entries:
        if entry.end_ms <= range_start_ms or entry.start_ms >= range_end_ms:
            continue
        clip_start = max(entry.start_ms, range_start_ms)
        clip_end = min(entry.end_ms, range_end_ms)
        if clip_end <= clip_start:
            continue
        clipped.append(
            SrtEntry(
                start_ms=clip_start - range_start_ms,
                end_ms=clip_end - range_start_ms,
                text=entry.text,
                label=entry.label,
            )
        )
    return clipped


def offset_entries(entries: list[SrtEntry], offset_ms: int) -> list[SrtEntry]:
    if offset_ms <= 0:
        return [SrtEntry(e.start_ms, e.end_ms, e.text, e.label) for e in entries]
    return [
        SrtEntry(
            start_ms=e.start_ms + offset_ms,
            end_ms=e.end_ms + offset_ms,
            text=e.text,
            label=e.label,
        )
        for e in entries
    ]


def split_text_to_entries(
    text: str,
    duration_ms: int,
    *,
    max_chars: int = 18,
    label: str = "",
) -> list[SrtEntry]:
    cleaned = (text or "").strip()
    if not cleaned or duration_ms <= 0:
        return []

    chunks = utils.split_string_by_punctuations(cleaned)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not chunks:
        chunks = [cleaned]

    merged: list[str] = []
    buffer = ""
    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue
        if len(buffer) + len(chunk) <= max_chars:
            buffer += chunk
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)

    total_chars = max(1, sum(len(chunk) for chunk in merged))
    cursor = 0
    entries: list[SrtEntry] = []
    for index, chunk in enumerate(merged):
        if index == len(merged) - 1:
            end_ms = duration_ms
        else:
            end_ms = cursor + int(duration_ms * (len(chunk) / total_chars))
            end_ms = max(cursor + 200, end_ms)
        entries.append(
            SrtEntry(start_ms=cursor, end_ms=min(duration_ms, end_ms), text=chunk, label=label)
        )
        cursor = end_ms
    return entries


def clip_entries_to_duration(entries: list[SrtEntry], max_duration_ms: int) -> list[SrtEntry]:
    if not entries or max_duration_ms <= 0:
        return []
    clipped: list[SrtEntry] = []
    for entry in entries:
        if entry.start_ms >= max_duration_ms:
            continue
        end_ms = min(entry.end_ms, max_duration_ms)
        if end_ms <= entry.start_ms:
            continue
        clipped.append(
            SrtEntry(
                start_ms=entry.start_ms,
                end_ms=end_ms,
                text=entry.text,
                label=entry.label,
            )
        )
    return clipped


def extend_entries_to_speech_end(
    entries: list[SrtEntry],
    speech_duration_ms: int,
    *,
    inter_gap_ms: int = 50,
) -> list[SrtEntry]:
    """将每条字幕延续到下一条开始或语音结束，避免话未说完字幕先消失。"""
    if not entries or speech_duration_ms <= 0:
        return entries

    sorted_entries = sorted(entries, key=lambda e: (e.start_ms, e.end_ms))
    extended: list[SrtEntry] = []
    for i, entry in enumerate(sorted_entries):
        start_ms = max(0, entry.start_ms)
        if i + 1 < len(sorted_entries):
            next_start = sorted_entries[i + 1].start_ms
            end_ms = max(entry.end_ms, next_start - inter_gap_ms)
        else:
            end_ms = max(entry.end_ms, speech_duration_ms)

        end_ms = min(end_ms, speech_duration_ms)
        if end_ms <= start_ms:
            continue
        extended.append(
            SrtEntry(start_ms=start_ms, end_ms=end_ms, text=entry.text, label=entry.label)
        )
    return extended


def align_entries_to_speech_duration(
    entries: list[SrtEntry],
    speech_duration_ms: int,
    video_duration_ms: int = 0,
) -> list[SrtEntry]:
    """对齐字幕与语音时长。

    - 字幕偏短：整体拉伸放慢（解决出现太快、过早消失）
    - 字幕偏长：截断到边界，绝不压缩时间轴（压缩会导致字幕抢跑）
    """
    if not entries or speech_duration_ms <= 0:
        return entries

    cap_ms = speech_duration_ms
    if video_duration_ms > 0:
        cap_ms = min(speech_duration_ms, video_duration_ms)

    max_end = max(entry.end_ms for entry in entries)
    if max_end <= 0:
        return entries

    # 字幕总时长 < 语音：等比拉伸，让出现/切换与语速同步
    if max_end < cap_ms - 80:
        ratio = cap_ms / max_end
        stretched: list[SrtEntry] = []
        for entry in entries:
            start_ms = int(round(entry.start_ms * ratio))
            end_ms = min(int(round(entry.end_ms * ratio)), cap_ms)
            if end_ms <= start_ms:
                continue
            stretched.append(
                SrtEntry(start_ms=start_ms, end_ms=end_ms, text=entry.text, label=entry.label)
            )
        return extend_entries_to_speech_end(stretched, cap_ms)

    # 字幕总时长 > 边界：截断，不压缩（压缩会让字幕跑得比说话快）
    if max_end > cap_ms + 80:
        clipped = clip_entries_to_duration(entries, cap_ms)
        return extend_entries_to_speech_end(clipped, cap_ms)

    return extend_entries_to_speech_end(entries, cap_ms)


def fit_entries_to_video_timeline(entries: list[SrtEntry], video_duration_ms: int) -> list[SrtEntry]:
    """将片段内字幕时间轴对齐到裁剪后视频的真实时长。"""
    return align_entries_to_speech_duration(
        entries,
        speech_duration_ms=video_duration_ms,
        video_duration_ms=video_duration_ms,
    )


def merge_and_sort_entries(groups: Iterable[Iterable[SrtEntry]]) -> list[SrtEntry]:
    merged: list[SrtEntry] = []
    for group in groups:
        merged.extend(group)
    merged.sort(key=lambda item: (item.start_ms, item.end_ms))
    return merged


def resolve_overlaps(entries: list[SrtEntry], min_gap_ms: int = 50) -> list[SrtEntry]:
    if not entries:
        return []

    resolved: list[SrtEntry] = []
    for entry in entries:
        if not resolved:
            resolved.append(entry)
            continue
        prev = resolved[-1]
        if entry.start_ms < prev.end_ms + min_gap_ms:
            entry = SrtEntry(
                start_ms=max(entry.start_ms, prev.end_ms + min_gap_ms),
                end_ms=max(entry.end_ms, prev.end_ms + min_gap_ms + 200),
                text=entry.text,
                label=entry.label,
            )
        if entry.end_ms <= entry.start_ms:
            continue
        resolved.append(entry)
    return resolved


def dialogue_match_key(text: str) -> str:
    """对白匹配用：去标点空白，便于 original_line 与字幕条目对齐。"""
    cleaned = clean_subtitle_dialogue_text(str(text or ""))
    cleaned = cleaned.strip("「」\"'“”‘’")
    return re.sub(r"[\s\W_]+", "", cleaned)


def _merge_contiguous_entries(
    entries: list[SrtEntry],
    center_index: int,
    *,
    merge_gap_ms: int,
    max_span_ms: int,
) -> tuple[int, int]:
    lo = hi = center_index
    while lo > 0 and entries[lo].start_ms - entries[lo - 1].end_ms <= merge_gap_ms:
        if entries[hi].end_ms - entries[lo - 1].start_ms > max_span_ms:
            break
        lo -= 1
    while hi < len(entries) - 1 and entries[hi + 1].start_ms - entries[hi].end_ms <= merge_gap_ms:
        if entries[hi + 1].end_ms - entries[lo].start_ms > max_span_ms:
            break
        hi += 1
    return entries[lo].start_ms, entries[hi].end_ms


def find_subtitle_span_for_line(
    entries: list[SrtEntry],
    line_text: str,
    *,
    near_start_ms: int,
    near_end_ms: int | None = None,
    max_span_ms: int = 22_000,
    merge_gap_ms: int = 400,
    search_window_ms: int = 8_000,
) -> tuple[int, int] | None:
    """
    为 OST=1 原声查找字幕完整起止：优先匹配 original_line 所在条目，
    并向前后合并间隔很短的字幕条，避免半句被截断。
    """
    if not entries:
        return None

    anchor_end = near_end_ms if near_end_ms is not None else near_start_ms
    needle = dialogue_match_key(line_text)
    best_index: int | None = None
    best_score = -1

    for index, entry in enumerate(entries):
        if entry.end_ms < near_start_ms - search_window_ms:
            continue
        if entry.start_ms > anchor_end + search_window_ms:
            break

        hay = dialogue_match_key(entry.text)
        matched = False
        if needle and hay:
            if needle in hay or hay in needle:
                matched = True
            elif len(needle) >= 4 and len(hay) >= 4 and needle[:4] == hay[:4]:
                matched = True
        elif not needle and entry.start_ms <= anchor_end and entry.end_ms >= near_start_ms:
            matched = True

        if not matched:
            continue

        distance = abs(entry.start_ms - near_start_ms)
        overlap = min(entry.end_ms, anchor_end) - max(entry.start_ms, near_start_ms)
        score = overlap if overlap > 0 else -distance
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        overlapping = [
            index
            for index, entry in enumerate(entries)
            if entry.end_ms >= near_start_ms and entry.start_ms <= anchor_end + 500
        ]
        if not overlapping:
            return None
        best_index = overlapping[0]

    return _merge_contiguous_entries(
        entries,
        best_index,
        merge_gap_ms=merge_gap_ms,
        max_span_ms=max_span_ms,
    )


def find_subtitle_span_global(
    entries: list[SrtEntry],
    line_text: str,
    *,
    max_span_ms: int = 22_000,
    merge_gap_ms: int = 400,
) -> tuple[int, int] | None:
    """在全片字幕中查找台词所在条目并合并相邻短间隔字幕条。"""
    if not entries:
        return None

    needle = dialogue_match_key(line_text)
    best_index: int | None = None
    best_score = -1

    for index, entry in enumerate(entries):
        hay = dialogue_match_key(entry.text)
        matched = False
        if needle and hay:
            if needle in hay or hay in needle:
                matched = True
            elif len(needle) >= 4 and len(hay) >= 4 and needle[:4] == hay[:4]:
                matched = True
        if not matched:
            continue
        score = len(hay)
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        return None

    return _merge_contiguous_entries(
        entries,
        best_index,
        merge_gap_ms=merge_gap_ms,
        max_span_ms=max_span_ms,
    )
