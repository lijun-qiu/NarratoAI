#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：自适应场景分段（切镜 + 1–10 秒随机采样）。"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
from typing import Any

from app.services.documentary.video_episode_constants import (
    SCENE_DETECT_THRESHOLD,
    SEGMENT_MAX_SECONDS,
    SEGMENT_MIN_SECONDS,
    SEGMENT_SPLIT_POLICY,
)

_SCENE_CUT_RE = re.compile(r"pts_time:([\d.]+)")


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def schedule_seed(*, video_path: str, start_offset_seconds: float) -> int:
    raw = f"{os.path.abspath(video_path)}|{start_offset_seconds:.3f}|{SEGMENT_SPLIT_POLICY}"
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)


def detect_scene_cut_seconds(
    video_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    threshold: float = SCENE_DETECT_THRESHOLD,
) -> list[float]:
    """检测指定区间内切镜时间点（相对片头绝对秒）。"""
    if not video_path or not os.path.isfile(video_path):
        return []
    if duration_seconds is not None and duration_seconds <= 0:
        return []

    cmd = ["ffmpeg", "-hide_banner"]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", video_path])
    if duration_seconds is not None and duration_seconds > 0:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend(
        [
            "-vf",
            f"select='gt(scene\\,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=600,
        )
    except (subprocess.SubprocessError, OSError, UnicodeError):
        return []

    cuts: list[float] = []
    for match in _SCENE_CUT_RE.finditer(result.stderr or ""):
        try:
            relative = float(match.group(1))
        except ValueError:
            continue
        absolute = start_seconds + relative
        cuts.append(round(absolute, 3))
    return sorted(set(cuts))


def build_adaptive_segment_schedule(
    duration_seconds: float,
    *,
    start_offset_seconds: float = 0.0,
    video_path: str = "",
    scene_cuts: list[float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """
    自适应分段：
    - 同场景内按 1–10 秒随机步长采样；
    - 遇切镜点立即结束当前格并新开一格；
    - 末段不足 min 时保留余量。
    """
    duration = max(0.0, float(duration_seconds))
    start_base = max(0.0, float(start_offset_seconds))
    end_limit = start_base + duration
    if duration <= 0.01:
        return []

    if scene_cuts is None and video_path:
        scene_cuts = detect_scene_cut_seconds(
            video_path,
            start_seconds=start_base,
            duration_seconds=duration,
        )
    cuts_in_range = sorted(
        {
            cut
            for cut in (scene_cuts or [])
            if start_base + 0.05 < cut < end_limit - 0.05
        }
    )

    if seed is None and video_path:
        seed = schedule_seed(video_path=video_path, start_offset_seconds=start_base)
    rng = random.Random(seed if seed is not None else 0)

    ranges: list[str] = []
    cursor = start_base
    cut_index = 0
    while cursor < end_limit - 0.01:
        while cut_index < len(cuts_in_range) and cuts_in_range[cut_index] <= cursor + 0.05:
            cut_index += 1
        next_cut = cuts_in_range[cut_index] if cut_index < len(cuts_in_range) else end_limit

        step = rng.randint(SEGMENT_MIN_SECONDS, SEGMENT_MAX_SECONDS)
        random_boundary = min(cursor + step, end_limit)

        if next_cut < random_boundary:
            seg_end = next_cut
        else:
            seg_end = random_boundary

        remaining = end_limit - cursor
        is_tail = remaining <= float(SEGMENT_MAX_SECONDS) + 0.05
        if not is_tail and seg_end - cursor < SEGMENT_MIN_SECONDS:
            seg_end = min(cursor + SEGMENT_MIN_SECONDS, end_limit, next_cut)

        if seg_end <= cursor + 0.01:
            seg_end = min(cursor + 1.0, end_limit)
        if seg_end <= cursor + 0.01:
            break

        start_label = _format_timestamp(cursor)
        end_label = _format_timestamp(seg_end)
        if start_label == end_label:
            if seg_end >= end_limit - 0.01:
                break
            seg_end = min(cursor + 1.0, end_limit)
            end_label = _format_timestamp(seg_end)
            if start_label == end_label:
                break

        ranges.append(f"{start_label}-{end_label}")
        cursor = seg_end

    return ranges


def build_segment_schedule(
    duration_seconds: float,
    *,
    start_offset_seconds: float = 0.0,
    video_path: str = "",
    scene_cuts: list[float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """按当前策略生成分段时间窗列表。"""
    return build_adaptive_segment_schedule(
        duration_seconds,
        start_offset_seconds=start_offset_seconds,
        video_path=video_path,
        scene_cuts=scene_cuts,
        seed=seed,
    )


def segment_policy_summary(*, payload: dict[str, Any] | None = None) -> str:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    if policy == "adaptive_scene":
        return (
            f"自适应场景格（同场景 {SEGMENT_MIN_SECONDS}-{SEGMENT_MAX_SECONDS} 秒采样，切镜即切分）"
        )
    legacy = (payload or {}).get("segment_interval_seconds")
    if legacy:
        return f"固定 {legacy} 秒格"
    return f"自适应场景格（{SEGMENT_MIN_SECONDS}-{SEGMENT_MAX_SECONDS} 秒）"


def average_segment_seconds(*, payload: dict[str, Any] | None = None) -> float:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    if policy == "adaptive_scene":
        return (SEGMENT_MIN_SECONDS + SEGMENT_MAX_SECONDS) / 2.0
    legacy = (payload or {}).get("segment_interval_seconds")
    if legacy:
        try:
            return float(legacy)
        except (TypeError, ValueError):
            pass
    return (SEGMENT_MIN_SECONDS + SEGMENT_MAX_SECONDS) / 2.0


def is_adaptive_segment_policy(payload: dict[str, Any] | None = None) -> bool:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    return policy == "adaptive_scene"
