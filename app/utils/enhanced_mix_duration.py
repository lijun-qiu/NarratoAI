#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Compute enhanced-mix script duration targets from uploaded video length."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.subtitle_clipper import time_str_to_seconds
from app.utils.utils import format_time


def _is_ost1_item(item: dict) -> bool:
    try:
        if int(item.get("OST") or 0) == 1:
            return True
    except (TypeError, ValueError):
        pass
    narration = str(item.get("narration") or "").strip()
    return bool(re.match(r"^播放原片\d*$", narration))


def estimate_script_playback_seconds(script_items: list[dict]) -> float:
    """Estimate merged output length (matches clip_video narration logic)."""
    total = 0.0
    for item in script_items:
        ts = item.get("timestamp") or ""
        if "-" not in str(ts):
            continue
        try:
            start_str, end_str = str(ts).split("-", 1)
            start_sec = time_str_to_seconds(start_str.strip())
            end_sec = time_str_to_seconds(end_str.strip())
        except (ValueError, IndexError):
            continue
        span = max(0.0, end_sec - start_sec)
        if _is_ost1_item(item):
            total += span
        else:
            narration = str(item.get("narration") or "").strip()
            tts_est = len(narration) / 4.0 if narration else 0.0
            total += max(tts_est, span)
    return total


# 成片播放时长 = 原片时长 × 比例（按上传视频动态计算）
MIN_OUTPUT_RATIO = 0.42
TARGET_OUTPUT_RATIO = 0.50
MAX_OUTPUT_RATIO = 0.52

# 建议片段数：按目标成片时长反推，控制片段不要过多
TARGET_SECONDS_PER_SEGMENT = 18.0
MAX_PLAYBACK_SECONDS_PER_SEGMENT = 12.0


@dataclass(frozen=True)
class EnhancedMixDurationPlan:
    video_duration_sec: float
    video_duration: str
    min_duration: str
    target_duration: str
    max_duration: str
    min_segment_count: int
    max_segment_count: int
    narration_span_min: int
    narration_span_max: int
    ost_span_min: int
    ost_span_max: int
    plan_summary: str

    def to_prompt_parameters(self) -> dict[str, str]:
        return {
            "video_duration": self.video_duration,
            "video_duration_sec": f"{self.video_duration_sec:.1f}",
            "min_duration": self.min_duration,
            "target_duration": self.target_duration,
            "max_duration": self.max_duration,
            "min_segment_count": str(self.min_segment_count),
            "max_segment_count": str(self.max_segment_count),
            "narration_span_range": f"{self.narration_span_min}-{self.narration_span_max}",
            "ost_span_range": f"{self.ost_span_min}-{self.ost_span_max}",
            "duration_plan_summary": self.plan_summary,
        }


def build_enhanced_mix_duration_plan(video_duration_sec: float) -> EnhancedMixDurationPlan:
    """
    Derive enhanced-mix rules from uploaded video length.

    Example: 6:00 source → target ~3:00, hard cap ~3:07, suggest 10-12 segments (not 20+).
    """
    seconds = max(float(video_duration_sec), 1.0)

    min_sec = seconds * MIN_OUTPUT_RATIO
    target_sec = seconds * TARGET_OUTPUT_RATIO
    max_sec = seconds * MAX_OUTPUT_RATIO

    min_items = max(8, int(round(target_sec / TARGET_SECONDS_PER_SEGMENT)))
    max_items = min_items + 2
    cap_by_playback = int(max_sec / MAX_PLAYBACK_SECONDS_PER_SEGMENT)
    cap_by_source = max(8, int(round(seconds / 25.0)))
    max_items = min(max_items, cap_by_playback, cap_by_source)

    if min_items > max_items:
        min_items = max(8, max_items - 1)

    if seconds < 120:
        narr_min, narr_max = 5, 12
        ost_min, ost_max = 3, 8
    elif seconds < 600:
        narr_min, narr_max = 6, 14
        ost_min, ost_max = 3, 10
    else:
        narr_min, narr_max = 8, 16
        ost_min, ost_max = 4, 12

    video_label = format_time(seconds)
    min_label = format_time(min_sec)
    target_label = format_time(target_sec)
    max_label = format_time(max_sec)

    summary = (
        f"原片 {video_label} → 目标成片 {target_label}（约 50%），"
        f"严禁超过 {max_label}；建议仅 {min_items}-{max_items} 个片段（勿过多）"
    )

    return EnhancedMixDurationPlan(
        video_duration_sec=seconds,
        video_duration=video_label,
        min_duration=min_label,
        target_duration=target_label,
        max_duration=max_label,
        min_segment_count=min_items,
        max_segment_count=max_items,
        narration_span_min=narr_min,
        narration_span_max=narr_max,
        ost_span_min=ost_min,
        ost_span_max=ost_max,
        plan_summary=summary,
    )


def cap_enhanced_mix_script_playback(
    script_items: list[dict],
    video_duration_sec: float,
) -> tuple[list[dict], str]:
    """
    Trim segment count / playback estimate so the script stays within MAX_OUTPUT_RATIO.
    """
    if not script_items or video_duration_sec <= 0:
        return script_items, ""

    plan = build_enhanced_mix_duration_plan(video_duration_sec)
    max_playback = video_duration_sec * MAX_OUTPUT_RATIO
    items = list(script_items)
    notes: list[str] = []

    if len(items) > plan.max_segment_count:
        items = items[: plan.max_segment_count]
        notes.append(f"片段数已裁至 {plan.max_segment_count} 个")

    floor_count = max(8, plan.min_segment_count - 2)
    while len(items) > floor_count and estimate_script_playback_seconds(items) > max_playback:
        items.pop()
        if "已移除尾部片段以控制成片时长" not in notes:
            notes.append("已移除尾部片段以控制成片时长")

    if estimate_script_playback_seconds(items) > max_playback:
        notes.append(
            f"预计仍超过 {format_time(max_playback)}，请重新生成或手动删减/缩短单段跨度"
        )

    return items, "；".join(notes)
