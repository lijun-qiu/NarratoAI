#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：分镜切段时间窗（切镜点切分，无随机采样）。"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
from typing import Any

from loguru import logger

from app.services.documentary.video_episode_constants import (
    SCENE_CANDIDATE_THRESHOLD,
    SCENE_CUT_MODE,
    SCENE_DETECT_THRESHOLD,
    SCENE_ENVIRONMENT_DIFF_THRESHOLD,
    SCENE_FRAME_SAMPLE_AFTER_SECONDS,
    SCENE_FRAME_SAMPLE_BEFORE_SECONDS,
    SCENE_MAX_SECONDS,
    SCENE_MIN_MERGE_SECONDS,
    SCENE_MIN_SEGMENT_SECONDS,
    SEGMENT_MAX_SECONDS,
    SEGMENT_MIN_SECONDS,
    SEGMENT_SPLIT_POLICY,
    VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS,
    VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS,
)

_SCENE_CUT_RE = re.compile(r"pts_time:([\d.]+)")
_FRAME_SIGNATURE_GRID = 4


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def schedule_seed(*, video_path: str, start_offset_seconds: float) -> int:
    raw = f"{os.path.abspath(video_path)}|{start_offset_seconds:.3f}|{SEGMENT_SPLIT_POLICY}"
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)


def _threshold_to_content_detector(threshold: float) -> float:
    """将旧版 FFmpeg scenecut 阈值 (0–1) 映射为 ContentDetector 灵敏度。"""
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        value = SCENE_CANDIDATE_THRESHOLD
    return max(1.0, min(100.0, value * 100.0))


def _run_ffmpeg_scene_detect_fallback(
    video_path: str,
    *,
    start_seconds: float,
    duration_seconds: float | None,
    threshold: float,
) -> list[float]:
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


def _run_pyscenedetect_scene_detect(
    video_path: str,
    *,
    start_seconds: float,
    duration_seconds: float | None,
    threshold: float,
) -> list[float]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
        from scenedetect.frame_timecode import FrameTimecode
    except ImportError:
        logger.warning("PySceneDetect 未安装，回退 FFmpeg scenecut")
        return _run_ffmpeg_scene_detect_fallback(
            video_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            threshold=threshold,
        )

    try:
        video = open_video(video_path)
        fps = float(video.frame_rate or 30.0)
        if start_seconds > 0:
            video.seek(FrameTimecode(start_seconds, fps=fps))

        end_time = None
        if duration_seconds is not None and duration_seconds > 0:
            end_time = FrameTimecode(start_seconds + duration_seconds, fps=fps)

        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=_threshold_to_content_detector(threshold))
        )
        scene_manager.detect_scenes(video, end_time=end_time)
        scene_list = scene_manager.get_scene_list()
        cuts = [
            round(scene_start.get_seconds(), 3)
            for scene_start, _scene_end in scene_list[1:]
        ]
        return sorted(set(cuts))
    except Exception as exc:
        logger.warning(f"PySceneDetect 切点检测失败，回退 FFmpeg scenecut: {exc}")
        return _run_ffmpeg_scene_detect_fallback(
            video_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            threshold=threshold,
        )


def detect_edit_cut_seconds(
    video_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    threshold: float = SCENE_CANDIDATE_THRESHOLD,
) -> list[float]:
    """检测硬切/剪辑切点（含对白反打），用于后续场景过滤。"""
    if not video_path or not os.path.isfile(video_path):
        return []
    if duration_seconds is not None and duration_seconds <= 0:
        return []
    return _run_pyscenedetect_scene_detect(
        video_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        threshold=threshold,
    )


def _extract_frame_signature(video_path: str, timestamp: float) -> list[float] | None:
    """4x4 网格平均 RGB，用于比较切点前后环境是否明显变化。"""
    grid = _FRAME_SIGNATURE_GRID
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        f"scale={grid}:{grid}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    expected = grid * grid * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    values = [float(byte) / 255.0 for byte in result.stdout[:expected]]
    return values


def _signature_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def filter_environment_change_cuts(
    video_path: str,
    candidate_cuts: list[float],
    *,
    diff_threshold: float = SCENE_ENVIRONMENT_DIFF_THRESHOLD,
    sample_before_seconds: float = SCENE_FRAME_SAMPLE_BEFORE_SECONDS,
    sample_after_seconds: float = SCENE_FRAME_SAMPLE_AFTER_SECONDS,
    video_duration_seconds: float | None = None,
) -> list[float]:
    """
    仅保留场景/环境明显变化的硬切。

    比较「切点前参考帧」与「切点后稳定帧」，而非切点紧邻两帧：
    同场景对白反打前后画面仍接近，换景（走廊→审讯室等）则差异持续偏大。
    """
    if not candidate_cuts:
        return []

    kept: list[float] = []
    for cut_time in candidate_cuts:
        reference_time = max(0.0, cut_time - sample_before_seconds)
        stable_time = cut_time + sample_after_seconds
        if video_duration_seconds is not None:
            stable_time = min(video_duration_seconds, stable_time)
        if stable_time <= reference_time + 0.05:
            continue
        reference_sig = _extract_frame_signature(video_path, reference_time)
        stable_sig = _extract_frame_signature(video_path, stable_time)
        if not reference_sig or not stable_sig:
            continue
        if _signature_distance(reference_sig, stable_sig) >= diff_threshold:
            kept.append(cut_time)
    return kept


def detect_scene_cut_seconds(
    video_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    threshold: float = SCENE_DETECT_THRESHOLD,
    scene_cut_mode: str = SCENE_CUT_MODE,
    candidate_threshold: float = SCENE_CANDIDATE_THRESHOLD,
    environment_diff_threshold: float = SCENE_ENVIRONMENT_DIFF_THRESHOLD,
    sample_before_seconds: float = SCENE_FRAME_SAMPLE_BEFORE_SECONDS,
    sample_after_seconds: float = SCENE_FRAME_SAMPLE_AFTER_SECONDS,
) -> list[float]:
    """检测用于切段的时间点：默认仅场景/环境明显变化，不含对白反打。"""
    mode = (scene_cut_mode or SCENE_CUT_MODE).strip()
    if mode == "edit_cut":
        return detect_edit_cut_seconds(
            video_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            threshold=threshold,
        )

    edit_cuts = detect_edit_cut_seconds(
        video_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        threshold=candidate_threshold,
    )
    return filter_environment_change_cuts(
        video_path,
        edit_cuts,
        diff_threshold=environment_diff_threshold,
        sample_before_seconds=sample_before_seconds,
        sample_after_seconds=sample_after_seconds,
        video_duration_seconds=(
            None if duration_seconds is None else start_seconds + duration_seconds
        ),
    )


def _coalesce_segments_to_min_duration(
    segments: list[tuple[float, float]],
    min_segment_seconds: float,
) -> list[tuple[float, float]]:
    """将仍短于最小时长的相邻段继续合并。"""
    if min_segment_seconds <= 0 or len(segments) <= 1:
        return segments

    merged: list[tuple[float, float]] = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        prev_duration = prev_end - prev_start
        curr_duration = end - start
        if prev_duration < min_segment_seconds or curr_duration < min_segment_seconds:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_segment_seconds:
        tail_start, tail_end = merged.pop()
        prev_start, _prev_end = merged[-1]
        merged[-1] = (prev_start, tail_end)
    return merged


def split_interval_into_subwindows(
    start: float,
    end: float,
    *,
    min_seconds: float = VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS,
    max_seconds: float = VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS,
) -> list[tuple[float, float]]:
    """将一段时间区间切为 5–10 秒分析窗（最后一段可短于 min_seconds）。"""
    min_sec = max(1.0, float(min_seconds))
    max_sec = max(min_sec, float(max_seconds))
    start = float(start)
    end = float(end)
    if end <= start + 0.01:
        return [(start, end)]

    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end - 0.01:
        remaining = end - cursor
        if remaining <= max_sec:
            windows.append((cursor, end))
            break
        chunk = min(max_sec, max(min_sec, remaining))
        if remaining - chunk < min_sec and remaining > max_sec:
            chunk = max(min_sec, remaining - min_sec)
        seg_end = min(cursor + chunk, end)
        if seg_end <= cursor + 0.01:
            break
        windows.append((cursor, seg_end))
        cursor = seg_end
    return windows


def build_subsegment_schedule_for_range(
    time_range: str,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> list[str]:
    """单个切镜段 time_range → 5–10 秒子窗口列表（绝对时间）。"""
    from app.services.documentary.video_episode_constants import (
        VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS,
        VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS,
    )

    cleaned = (time_range or "").strip()
    if not cleaned or "-" not in cleaned:
        return [cleaned] if cleaned else []
    parts = re.split(r"[-—]", cleaned, maxsplit=1)
    start_label = parts[0].strip()
    end_label = (parts[1] if len(parts) > 1 else parts[0]).strip()

    def _parse_hms(label: str) -> float:
        bits = label.replace(",", ".").split(":")
        try:
            if len(bits) == 3:
                return int(bits[0]) * 3600 + int(bits[1]) * 60 + float(bits[2])
            if len(bits) == 2:
                return int(bits[0]) * 60 + float(bits[1])
            return float(bits[0])
        except (TypeError, ValueError):
            return 0.0

    start = _parse_hms(start_label)
    end = _parse_hms(end_label)
    if end < start:
        end = start
    min_sec = float(min_seconds or VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS)
    max_sec = float(max_seconds or VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS)
    windows = split_interval_into_subwindows(start, end, min_seconds=min_sec, max_seconds=max_sec)
    return [f"{_format_timestamp(s)}-{_format_timestamp(e)}" for s, e in windows]


def build_episodic_subsegment_schedule(
    upload_ranges: list[str],
    *,
    segment_split_policy: str | None = None,
) -> list[str]:
    """上传段/切镜段 time_range 列表 → 全片 5–10 秒 episodic_segments 时间窗。"""
    policy = (segment_split_policy or SEGMENT_SPLIT_POLICY).strip()
    if policy == "adaptive_scene":
        return [str(item).strip() for item in upload_ranges if str(item).strip()]
    schedule: list[str] = []
    for time_range in upload_ranges:
        cleaned = str(time_range or "").strip()
        if not cleaned:
            continue
        sub = build_subsegment_schedule_for_range(cleaned)
        schedule.extend(sub if sub else [cleaned])
    return schedule


def build_time_chunk_segment_schedule(
    duration_seconds: float,
    *,
    start_offset_seconds: float = 0.0,
    chunk_seconds: float = 900.0,
) -> list[str]:
    """按固定时长切段上传（不做切镜检测）。"""
    duration = max(0.0, float(duration_seconds))
    start_base = max(0.0, float(start_offset_seconds))
    end_limit = start_base + duration
    step = max(60.0, float(chunk_seconds))
    if duration <= 0.01:
        return []

    ranges: list[str] = []
    cursor = start_base
    while cursor < end_limit - 0.01:
        seg_end = min(cursor + step, end_limit)
        if seg_end <= cursor + 0.01:
            break
        ranges.append(f"{_format_timestamp(cursor)}-{_format_timestamp(seg_end)}")
        cursor = seg_end
    return ranges


def build_scene_cut_segment_schedule(
    duration_seconds: float,
    *,
    start_offset_seconds: float = 0.0,
    video_path: str = "",
    scene_cuts: list[float] | None = None,
    min_merge_seconds: float = SCENE_MIN_MERGE_SECONDS,
    min_segment_seconds: float = SCENE_MIN_SEGMENT_SECONDS,
    max_scene_seconds: float = SCENE_MAX_SECONDS,
    scene_detect_threshold: float = SCENE_DETECT_THRESHOLD,
) -> list[str]:
    """
    按切镜点切段：
    - 每个切镜边界单独成段；
    - 过短镜头合并到相邻段；
    - 无切镜的长镜头按 max_scene_seconds 上限再切。
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
            threshold=scene_detect_threshold,
        )
    cuts_in_range = sorted(
        {
            cut
            for cut in (scene_cuts or [])
            if start_base + 0.05 < cut < end_limit - 0.05
        }
    )

    raw_segments: list[tuple[float, float]] = []
    cursor = start_base
    for cut in cuts_in_range:
        if cut > cursor + 0.01:
            raw_segments.append((cursor, cut))
        cursor = cut
    if cursor < end_limit - 0.01:
        raw_segments.append((cursor, end_limit))
    if not raw_segments:
        raw_segments = [(start_base, end_limit)]

    merged: list[tuple[float, float]] = []
    for start, end in raw_segments:
        seg_duration = end - start
        if merged and seg_duration < min_merge_seconds:
            merged[-1] = (merged[-1][0], end)
        elif merged and (merged[-1][1] - merged[-1][0]) < min_merge_seconds:
            prev_start, _prev_end = merged.pop()
            merged.append((prev_start, end))
        else:
            merged.append((start, end))

    merged = _coalesce_segments_to_min_duration(merged, min_segment_seconds)

    ranges: list[str] = []
    for start, end in merged:
        cursor = start
        while cursor < end - 0.01:
            seg_end = min(cursor + max_scene_seconds, end)
            if seg_end <= cursor + 0.01:
                break
            ranges.append(f"{_format_timestamp(cursor)}-{_format_timestamp(seg_end)}")
            cursor = seg_end
    return ranges


def build_adaptive_segment_schedule(
    duration_seconds: float,
    *,
    start_offset_seconds: float = 0.0,
    video_path: str = "",
    scene_cuts: list[float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """旧策略：切镜 + 同场景 1–10 秒随机采样（仅兼容历史 artifact）。"""
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
    segment_split_policy: str | None = None,
    min_merge_seconds: float = SCENE_MIN_MERGE_SECONDS,
    min_segment_seconds: float = SCENE_MIN_SEGMENT_SECONDS,
    max_scene_seconds: float = SCENE_MAX_SECONDS,
    scene_detect_threshold: float = SCENE_DETECT_THRESHOLD,
    chunk_seconds: float = 900.0,
) -> list[str]:
    """按策略生成分段时间窗列表。"""
    policy = (segment_split_policy or SEGMENT_SPLIT_POLICY).strip()
    if policy == "time_chunk":
        return build_time_chunk_segment_schedule(
            duration_seconds,
            start_offset_seconds=start_offset_seconds,
            chunk_seconds=chunk_seconds,
        )
    if policy == "adaptive_scene":
        return build_adaptive_segment_schedule(
            duration_seconds,
            start_offset_seconds=start_offset_seconds,
            video_path=video_path,
            scene_cuts=scene_cuts,
            seed=seed,
        )
    return build_scene_cut_segment_schedule(
        duration_seconds,
        start_offset_seconds=start_offset_seconds,
        video_path=video_path,
        scene_cuts=scene_cuts,
        min_merge_seconds=min_merge_seconds,
        min_segment_seconds=min_segment_seconds,
        max_scene_seconds=max_scene_seconds,
        scene_detect_threshold=scene_detect_threshold,
    )


def segment_policy_summary(*, payload: dict[str, Any] | None = None) -> str:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    if policy == "time_chunk":
        try:
            chunk_sec = float((payload or {}).get("upload_chunk_seconds") or 900.0)
        except (TypeError, ValueError):
            chunk_sec = 900.0
        minutes = chunk_sec / 60.0
        label = f"{minutes:g} 分钟" if minutes >= 1 else f"{chunk_sec:g} 秒"
        return (
            f"按时间切段（每 {label} 上传）+ 段内 "
            f"{VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS:g}–"
            f"{VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS:g}s 分析窗"
        )
    if policy == "scene_cut":
        return (
            f"切镜切段 + 镜内 {VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS:g}–"
            f"{VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS:g}s 分析窗"
        )
    if policy == "adaptive_scene":
        return (
            f"自适应场景格（同场景 {SEGMENT_MIN_SECONDS}-{SEGMENT_MAX_SECONDS} 秒采样，切镜即切分）"
        )
    legacy = (payload or {}).get("segment_interval_seconds")
    if legacy:
        return f"固定 {legacy} 秒格"
    return "分镜切段"


def average_segment_seconds(*, payload: dict[str, Any] | None = None) -> float:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    if policy == "time_chunk":
        try:
            return float((payload or {}).get("upload_chunk_seconds") or 900.0)
        except (TypeError, ValueError):
            return 900.0
    if policy == "scene_cut":
        return 8.0
    if policy == "adaptive_scene":
        return (SEGMENT_MIN_SECONDS + SEGMENT_MAX_SECONDS) / 2.0
    legacy = (payload or {}).get("segment_interval_seconds")
    if legacy:
        try:
            return float(legacy)
        except (TypeError, ValueError):
            pass
    return 8.0


def is_adaptive_segment_policy(payload: dict[str, Any] | None = None) -> bool:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    return policy == "adaptive_scene"


def is_time_chunk_segment_policy(payload: dict[str, Any] | None = None) -> bool:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    return policy == "time_chunk"


def is_scene_cut_segment_policy(payload: dict[str, Any] | None = None) -> bool:
    policy = str((payload or {}).get("segment_split_policy") or SEGMENT_SPLIT_POLICY)
    return policy == "scene_cut"
