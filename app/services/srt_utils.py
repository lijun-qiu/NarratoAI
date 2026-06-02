#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""SRT parsing, slicing, remapping and merging utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.utils import utils

_TIME_LINE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)

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
        text = clean_subtitle_dialogue_text("\n".join(text_lines).strip())
        if not text:
            continue
        entries.append(SrtEntry(start_ms=start_ms, end_ms=end_ms, text=text))
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
        text = clean_subtitle_dialogue_text(entry.text)
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
