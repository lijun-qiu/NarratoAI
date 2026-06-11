#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""视频与抽帧分析 JSON 的默认配对（同名 stem + _frame_analysis.json）。"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

from app.utils import utils

FRAME_ANALYSIS_ARTIFACT_VERSION = "documentary-frame-analysis-v4"
_LEGACY_ARTIFACT_VERSIONS = frozenset(
    {
        "documentary-frame-analysis-v1",
        "documentary-frame-analysis-v2",
        "documentary-frame-analysis-v3",
    }
)
_LEGACY_ARTIFACT_ERROR = (
    "旧版抽帧 JSON 已不再支持，请重新执行抽帧分析以生成 "
    f"{FRAME_ANALYSIS_ARTIFACT_VERSION} 格式"
)


def analysis_artifact_dir() -> str:
    return os.path.join(utils.storage_dir(), "temp", "analysis")


def sanitize_video_stem(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path or ""))[0]
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip()
    return sanitized or "video"


def default_analysis_path_for_video(video_path: str) -> str:
    stem = sanitize_video_stem(video_path)
    return os.path.join(analysis_artifact_dir(), f"{stem}_frame_analysis.json")


def default_test_analysis_path_for_video(
    video_path: str,
    *,
    max_duration_seconds: float = 5.0,
    start_time_seconds: float = 0.0,
) -> str:
    stem = sanitize_video_stem(video_path)
    duration_tag = int(max(1, round(float(max_duration_seconds))))
    start_tag = int(max(0, round(float(start_time_seconds))))
    if start_tag > 0:
        filename = f"{stem}_frame_analysis_test_from{start_tag}s_{duration_tag}s.json"
    else:
        filename = f"{stem}_frame_analysis_test_{duration_tag}s.json"
    return os.path.join(analysis_artifact_dir(), filename)


def normalize_video_path(video_path: str) -> str:
    if not video_path:
        return ""
    return os.path.normcase(os.path.abspath(video_path))


def is_compact_or_minimal_artifact(payload: dict[str, Any]) -> bool:
    version = str(payload.get("artifact_version") or "")
    return version.endswith("-compact") or version.endswith("-minimal-scene")


def is_legacy_analysis_artifact(payload: Any) -> bool:
    """检测旧版抽帧 JSON（frame_paths / overall_activity_summaries 等）。"""
    if not isinstance(payload, dict):
        return False
    version = str(payload.get("artifact_version") or "")
    if version in _LEGACY_ARTIFACT_VERSIONS:
        return True
    if payload.get("overall_activity_summaries"):
        return True
    for batch in payload.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        if batch.get("frame_paths"):
            return True
        if batch.get("observations"):
            return True
    for observation in payload.get("frame_observations") or []:
        if isinstance(observation, dict) and observation.get("frame_path"):
            return True
    for batch in payload.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for observation in batch.get("frame_observations") or []:
            if isinstance(observation, dict) and observation.get("frame_path"):
                return True
    return False


def is_valid_analysis_artifact(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if is_legacy_analysis_artifact(payload):
        return False
    if payload.get("scene_segments"):
        return True
    if payload.get("frame_timeline"):
        return True
    if payload.get("batches"):
        return True
    if payload.get("frame_observations"):
        return True
    return False


def load_analysis_artifact(analysis_json_path: str) -> dict[str, Any]:
    with open(analysis_json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if is_legacy_analysis_artifact(payload):
        raise ValueError(_LEGACY_ARTIFACT_ERROR)
    if not is_valid_analysis_artifact(payload):
        raise ValueError(f"无效的抽帧分析文件: {analysis_json_path}")
    return payload


def find_paired_frame_analysis_path(video_path: str) -> str:
    """查找与视频配对的已有抽帧分析（优先默认路径）。"""
    if not video_path or not os.path.isfile(video_path):
        return ""

    default_path = default_analysis_path_for_video(video_path)
    if os.path.isfile(default_path):
        try:
            load_analysis_artifact(default_path)
            return default_path
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    normalized_video = normalize_video_path(video_path)
    if not normalized_video or not os.path.isdir(analysis_artifact_dir()):
        return ""

    for file_path in sorted(
        glob.glob(os.path.join(analysis_artifact_dir(), "*_frame_analysis.json")),
        key=os.path.getmtime,
        reverse=True,
    ):
        try:
            artifact = load_analysis_artifact(file_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        artifact_video = normalize_video_path(str(artifact.get("video_path") or ""))
        if artifact_video and artifact_video == normalized_video:
            return file_path
    return ""


def resolve_reusable_analysis_path(
    video_path: str,
    *,
    explicit_path: str | None = None,
    reuse: bool = True,
) -> str | None:
    if explicit_path and os.path.isfile(explicit_path):
        try:
            load_analysis_artifact(explicit_path)
            return explicit_path
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if not reuse:
        return None

    candidates: list[str] = []

    default_path = default_analysis_path_for_video(video_path)
    if default_path not in candidates and os.path.isfile(default_path):
        candidates.append(default_path)

    normalized_video = normalize_video_path(video_path)
    if normalized_video and os.path.isdir(analysis_artifact_dir()):
        for file_path in sorted(
            glob.glob(os.path.join(analysis_artifact_dir(), "*_frame_analysis.json")),
            key=os.path.getmtime,
            reverse=True,
        ):
            if file_path in candidates:
                continue
            try:
                artifact = load_analysis_artifact(file_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            artifact_video = normalize_video_path(str(artifact.get("video_path") or ""))
            if artifact_video and artifact_video == normalized_video:
                candidates.append(file_path)

    for candidate in candidates:
        try:
            load_analysis_artifact(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return candidate
    return None
