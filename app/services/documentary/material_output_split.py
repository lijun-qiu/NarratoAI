#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""素材预处理输出切割：将抽帧分析 JSON / 字幕 SRT 按时间均分为多份。"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Callable

from loguru import logger

from app.services.documentary.documentary_subtitle_enrichment import parse_timestamp_range_ms
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.srt_utils import SrtEntry, write_srt_file
from app.utils import utils

MAX_SPLIT_PARTS = 10
MIN_SPLIT_PARTS = 1


def clamp_split_parts(part_count: int | float | None) -> int:
    try:
        value = int(part_count or 1)
    except (TypeError, ValueError):
        value = 1
    return max(MIN_SPLIT_PARTS, min(MAX_SPLIT_PARTS, value))


def ms_to_timestamp(ms: int) -> str:
    return utils.seconds_to_time(max(0, ms) / 1000.0).replace(".", ",")


def compute_equal_split_ranges(total_ms: int, part_count: int) -> list[dict[str, Any]]:
    """按总时长均分，返回每份的起止毫秒与时间范围字符串。"""
    part_count = clamp_split_parts(part_count)
    total_ms = max(1, int(total_ms or 0))

    if part_count <= 1:
        return [
            {
                "part_index": 1,
                "start_ms": 0,
                "end_ms": total_ms,
                "time_range": f"{ms_to_timestamp(0)}-{ms_to_timestamp(total_ms)}",
            }
        ]

    ranges: list[dict[str, Any]] = []
    part_duration = total_ms // part_count
    for index in range(part_count):
        start_ms = index * part_duration
        end_ms = total_ms if index == part_count - 1 else (index + 1) * part_duration
        ranges.append(
            {
                "part_index": index + 1,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "time_range": f"{ms_to_timestamp(start_ms)}-{ms_to_timestamp(end_ms)}",
            }
        )
    return ranges


def resolve_part_index(time_ms: int, windows: list[dict[str, Any]]) -> int:
    """按时间点归属唯一 part；最后一份包含右边界。"""
    if not windows:
        return 1
    point = max(0, int(time_ms))
    for window in windows:
        start_ms = int(window["start_ms"])
        end_ms = int(window["end_ms"])
        is_last = int(window["part_index"]) == int(windows[-1]["part_index"])
        if is_last:
            if start_ms <= point <= end_ms:
                return int(window["part_index"])
        elif start_ms <= point < end_ms:
            return int(window["part_index"])
    return int(windows[-1]["part_index"])


def group_items_by_part(
    items: list[Any],
    windows: list[dict[str, Any]],
    *,
    start_ms_getter: Callable[[Any], int],
) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {int(window["part_index"]): [] for window in windows}
    for item in items:
        part_index = resolve_part_index(start_ms_getter(item), windows)
        grouped.setdefault(part_index, []).append(item)
    return grouped


def _timestamp_bounds_ms(timestamp: str) -> tuple[int, int]:
    text = (timestamp or "").strip()
    if not text:
        return 0, 0
    if "-" in text:
        return parse_timestamp_range_ms(text)
    point = parse_timestamp_range_ms(text)[0]
    return point, point


def _time_range_bounds_ms(time_range: str) -> tuple[int, int]:
    return _timestamp_bounds_ms(time_range)


def _strip_split_suffix(stem: str) -> str:
    return re.sub(r"_part\d+$", "", stem)


def split_output_path(base_path: str, part_index: int, part_count: int) -> str:
    directory = os.path.dirname(base_path) or "."
    stem, ext = os.path.splitext(os.path.basename(base_path))
    stem = _strip_split_suffix(stem)
    if part_count <= 1:
        return os.path.join(directory, f"{stem}{ext}")
    width = max(2, len(str(part_count)))
    return os.path.join(directory, f"{stem}_part{part_index:0{width}d}{ext}")


def infer_artifact_duration_ms(artifact: dict[str, Any]) -> int:
    duration_ms = 0

    video_path = str(artifact.get("video_path") or "").strip()
    if video_path and os.path.isfile(video_path):
        try:
            from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

            seconds = DocumentaryFrameExtractionService._get_video_duration_sec(video_path)
            if seconds > 0:
                return int(seconds * 1000)
        except Exception as exc:
            logger.warning(f"读取视频时长失败，改用分析数据推断: {exc}")

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        _, end_ms = _time_range_bounds_ms(str(batch.get("time_range") or ""))
        duration_ms = max(duration_ms, end_ms)

    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        _, end_ms = _timestamp_bounds_ms(str(segment.get("timestamp") or ""))
        duration_ms = max(duration_ms, end_ms)

    for observation in artifact.get("frame_observations") or []:
        if not isinstance(observation, dict):
            continue
        point_ms = _timestamp_bounds_ms(str(observation.get("timestamp") or ""))[0]
        duration_ms = max(duration_ms, point_ms)

    return max(duration_ms, 1)


def split_frame_analysis_artifact(
    artifact: dict[str, Any],
    part_count: int,
) -> list[dict[str, Any]]:
    """将完整抽帧分析 artifact 按时间均分为多份子 artifact。"""
    part_count = clamp_split_parts(part_count)
    total_ms = infer_artifact_duration_ms(artifact)
    windows = compute_equal_split_ranges(total_ms, part_count)

    source_batches = [batch for batch in (artifact.get("batches") or []) if isinstance(batch, dict)]
    source_segments = [segment for segment in (artifact.get("scene_segments") or []) if isinstance(segment, dict)]
    source_observations = [
        observation for observation in (artifact.get("frame_observations") or []) if isinstance(observation, dict)
    ]
    source_summaries = [
        summary for summary in (artifact.get("overall_activity_summaries") or []) if isinstance(summary, dict)
    ]

    batch_groups = group_items_by_part(
        source_batches,
        windows,
        start_ms_getter=lambda batch: _time_range_bounds_ms(str(batch.get("time_range") or ""))[0],
    )
    segment_groups = group_items_by_part(
        source_segments,
        windows,
        start_ms_getter=lambda segment: _timestamp_bounds_ms(str(segment.get("timestamp") or ""))[0],
    )
    observation_groups = group_items_by_part(
        source_observations,
        windows,
        start_ms_getter=lambda observation: _timestamp_bounds_ms(str(observation.get("timestamp") or ""))[0],
    )
    summary_groups = group_items_by_part(
        source_summaries,
        windows,
        start_ms_getter=lambda summary: _time_range_bounds_ms(str(summary.get("time_range") or ""))[0],
    )

    parts: list[dict[str, Any]] = []
    for window in windows:
        part_index = int(window["part_index"])
        part_payload: dict[str, Any] = {
            "artifact_version": artifact.get("artifact_version"),
            "generated_at": artifact.get("generated_at"),
            "video_path": artifact.get("video_path"),
            "frame_interval_seconds": artifact.get("frame_interval_seconds"),
            "vision_batch_size": artifact.get("vision_batch_size"),
            "vision_llm_provider": artifact.get("vision_llm_provider"),
            "vision_model_name": artifact.get("vision_model_name"),
            "vision_max_concurrency": artifact.get("vision_max_concurrency"),
            "split_part_index": part_index,
            "split_part_count": part_count,
            "split_time_range": window["time_range"],
            "scene_segments": [deepcopy(item) for item in segment_groups.get(part_index, [])],
            "batches": [deepcopy(item) for item in batch_groups.get(part_index, [])],
            "frame_observations": [deepcopy(item) for item in observation_groups.get(part_index, [])],
            "overall_activity_summaries": [deepcopy(item) for item in summary_groups.get(part_index, [])],
        }
        if artifact.get("source_artifact_path"):
            part_payload["source_artifact_path"] = artifact.get("source_artifact_path")
        parts.append(part_payload)

    return parts


def save_split_frame_analysis_artifacts(
    source_path: str,
    part_count: int,
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """保留原完整 JSON，另存切割后的多份子文件。"""
    part_count = clamp_split_parts(part_count)
    if part_count <= 1:
        return {"part_paths": [], "part_count": 1}

    payload = artifact or load_analysis_artifact(source_path)
    parts = split_frame_analysis_artifact(payload, part_count)
    part_paths: list[str] = []

    for part in parts:
        index = int(part.get("split_part_index") or len(part_paths) + 1)
        target_path = split_output_path(source_path, index, part_count)
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        part["source_artifact_path"] = os.path.abspath(source_path)
        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(part, fp, ensure_ascii=False, indent=2)
        part_paths.append(target_path)
        logger.info(f"抽帧分析切割输出: {target_path}")

    return {
        "source_path": source_path,
        "part_count": part_count,
        "part_paths": part_paths,
    }


def split_srt_entries(
    entries: list[SrtEntry],
    part_count: int,
    *,
    total_ms: int | None = None,
) -> tuple[list[dict[str, Any]], list[list[SrtEntry]]]:
    """按时间窗口切割字幕条目（保留全局时间轴）。"""
    part_count = clamp_split_parts(part_count)
    if not entries:
        windows = compute_equal_split_ranges(total_ms or 1, part_count)
        return windows, [[] for _ in windows]

    inferred_total = total_ms if total_ms is not None else max(entry.end_ms for entry in entries)
    windows = compute_equal_split_ranges(inferred_total, part_count)
    entry_groups = group_items_by_part(
        entries,
        windows,
        start_ms_getter=lambda entry: int(entry.start_ms),
    )
    grouped = [entry_groups.get(int(window["part_index"]), []) for window in windows]

    return windows, grouped


def save_split_srt_files(
    entries: list[SrtEntry],
    base_path: str,
    part_count: int,
    *,
    total_ms: int | None = None,
) -> dict[str, Any]:
    """保留原完整 SRT，另存切割后的多份子文件。"""
    part_count = clamp_split_parts(part_count)
    if part_count <= 1:
        return {"part_paths": [], "part_count": 1}

    windows, grouped = split_srt_entries(entries, part_count, total_ms=total_ms)
    part_paths: list[str] = []

    for window, part_entries in zip(windows, grouped):
        index = int(window["part_index"])
        target_path = split_output_path(base_path, index, part_count)
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        write_srt_file(part_entries, target_path)
        part_paths.append(target_path)
        logger.info(f"字幕切割输出: {target_path}（{len(part_entries)} 条）")

    return {
        "source_path": base_path,
        "part_count": part_count,
        "part_paths": part_paths,
    }
