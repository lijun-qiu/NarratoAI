#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""根据脚本片段估算成片时长（与剪辑阶段 OST 裁剪规则一致）。"""

from __future__ import annotations

import re
from typing import Any

from app.services.update_script import calculate_duration

_PLAYBACK_PREFIX = "播放原片"


def format_duration_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{sec:02d}秒"
    if minutes:
        return f"{minutes}分{sec:02d}秒"
    return f"{sec}秒"


def estimate_narration_duration_seconds(narration: str, voice_rate: float = 1.0) -> float:
    text = (narration or "").strip()
    if not text or text.startswith(_PLAYBACK_PREFIX):
        return 0.0

    rate = voice_rate if voice_rate and voice_rate > 0 else 1.0
    english_words = len(re.findall(r"\b\w+\b", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fa5]", text))

    if english_words > chinese_chars:
        estimated = max(1.0, english_words * 0.35)
    else:
        estimated = max(1.0, chinese_chars * 0.3)

    return round(estimated / rate, 2)


def estimate_segment_duration(item: dict[str, Any], voice_rate: float = 1.0) -> float:
    ost = int(item.get("OST", 0) or 0)
    timestamp = str(item.get("timestamp", "") or "")

    if ost == 1:
        return max(0.0, calculate_duration(timestamp))

    existing_duration = item.get("duration")
    if isinstance(existing_duration, (int, float)) and existing_duration > 0:
        return round(float(existing_duration), 2)

    return estimate_narration_duration_seconds(str(item.get("narration", "") or ""), voice_rate)


def estimate_script_duration(
    script_items: list[dict[str, Any]],
    *,
    voice_rate: float = 1.0,
) -> dict[str, Any]:
    ost_counts = {0: 0, 1: 0, 2: 0}
    narration_seconds = 0.0
    original_seconds = 0.0
    segment_durations: list[float] = []

    for item in script_items:
        if not isinstance(item, dict):
            continue

        ost = int(item.get("OST", 0) or 0)
        if ost not in ost_counts:
            ost_counts[ost] = 0
        ost_counts[ost] += 1

        segment_seconds = estimate_segment_duration(item, voice_rate)
        segment_durations.append(segment_seconds)

        if ost == 1:
            original_seconds += segment_seconds
        else:
            narration_seconds += segment_seconds

    total_seconds = round(sum(segment_durations), 2)
    segment_count = len(segment_durations)

    return {
        "total_seconds": total_seconds,
        "segment_count": segment_count,
        "ost_counts": ost_counts,
        "narration_seconds": round(narration_seconds, 2),
        "original_seconds": round(original_seconds, 2),
        "formatted_total": format_duration_seconds(total_seconds),
        "voice_rate": voice_rate if voice_rate and voice_rate > 0 else 1.0,
    }
