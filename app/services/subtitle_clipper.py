#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Extract clip-local SRT subtitles from a source subtitle file."""

from __future__ import annotations

import os
import re
from typing import Optional

import pysrt
from loguru import logger


def time_str_to_seconds(time_str: str) -> float:
    """Parse SRT-style timestamp to seconds."""
    text = (time_str or "").strip()
    if not text:
        return 0.0

    milliseconds = 0.0
    if "," in text:
        time_part, ms_part = text.split(",", 1)
        milliseconds = int(ms_part) / 1000.0
    else:
        time_part = text

    parts = [int(part) for part in time_part.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
    return hours * 3600 + minutes * 60 + seconds + milliseconds


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _load_subtitles(source_path: str) -> pysrt.SubRipFile:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pysrt.open(source_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"无法读取字幕文件: {source_path} ({last_error})")


def _is_play_original_placeholder(text: str) -> bool:
    normalized = (text or "").strip()
    return bool(re.match(r"^播放原片\d*$", normalized)) or bool(
        re.match(r"^播放原生[_a-f0-9]*$", normalized)
    )


def _write_srt_entries(entries: list[tuple[float, float, str]], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    blocks: list[str] = []
    for index, (start_sec, end_sec, text) in enumerate(entries, start=1):
        if end_sec <= start_sec:
            continue
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{seconds_to_srt_time(start_sec)} --> {seconds_to_srt_time(end_sec)}",
                    cleaned,
                ]
            )
        )
    if not blocks:
        return ""
    content = "\n\n".join(blocks) + "\n"
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(content)
    return output_path


def clip_subtitles_from_source(
    source_path: str,
    timestamp_range: str,
    output_path: str,
) -> Optional[str]:
    """Extract overlapping subtitles and remap them to clip-local timestamps."""
    if not source_path or not os.path.exists(source_path):
        return None

    try:
        start_sec, end_sec = [time_str_to_seconds(part) for part in timestamp_range.split("-", 1)]
    except ValueError:
        logger.warning(f"无效的时间范围: {timestamp_range}")
        return None

    if end_sec <= start_sec:
        return None

    try:
        subtitles = _load_subtitles(source_path)
    except ValueError as exc:
        logger.warning(str(exc))
        return None

    entries: list[tuple[float, float, str]] = []
    for sub in subtitles:
        sub_start = sub.start.ordinal / 1000.0
        sub_end = sub.end.ordinal / 1000.0
        if sub_end <= start_sec or sub_start >= end_sec:
            continue

        clip_start = max(sub_start, start_sec) - start_sec
        clip_end = min(sub_end, end_sec) - start_sec
        text = sub.text.replace("\n", " ").strip()
        if text:
            entries.append((clip_start, clip_end, text))

    if not entries:
        return None

    written = _write_srt_entries(entries, output_path)
    return written or None


def create_fallback_native_subtitle(
    narration: str,
    duration: float,
    output_path: str,
) -> Optional[str]:
    """Use narration text as a single native subtitle block when source SRT has no overlap."""
    text = (narration or "").strip()
    if not text or _is_play_original_placeholder(text):
        return None
    if duration <= 0:
        duration = max(len(text) * 0.25, 1.0)
    written = _write_srt_entries([(0.0, duration, text)], output_path)
    return written or None


def enrich_native_subtitles(
    script_list: list[dict],
    source_subtitle_path: str,
    task_id: str,
) -> list[dict]:
    """Attach clip-local native subtitle files for OST=1 segments."""
    if not source_subtitle_path or not os.path.exists(source_subtitle_path):
        logger.warning(f"未找到原片字幕文件，无法生成原声字幕: {source_subtitle_path}")
        return script_list

    from app.utils import utils

    output_dir = utils.task_dir(task_id)

    for item in script_list:
        if item.get("OST") != 1:
            continue
        if item.get("subtitle") and os.path.exists(item.get("subtitle", "")):
            continue

        segment_id = item.get("_id", "unknown")
        timestamp_range = item.get("timestamp") or item.get("sourceTimeRange") or ""
        output_path = os.path.join(output_dir, f"native_subtitle_{segment_id}.srt")

        subtitle_path = None
        if timestamp_range:
            subtitle_path = clip_subtitles_from_source(
                source_subtitle_path,
                timestamp_range,
                output_path,
            )

        if not subtitle_path:
            subtitle_path = create_fallback_native_subtitle(
                item.get("narration", ""),
                float(item.get("duration") or 0),
                output_path,
            )

        if subtitle_path:
            item["subtitle"] = subtitle_path
            logger.info(f"片段 {segment_id} 已生成原声字幕: {subtitle_path}")
        else:
            logger.warning(f"片段 {segment_id} 未能生成原声字幕")

    return script_list
