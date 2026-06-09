#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧分析 JSON 精简导出：去重、去调试字段，显著减小体积。"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from app.utils import utils

from app.services.documentary.documentary_subtitle_enrichment import (
    clean_subtitle_punctuation,
    partition_subtitle_entries_across_segments,
    resolve_segment_subtitle_text,
    resolve_segment_time_range,
)
from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.documentary.frame_extraction_rules import (
    SCENE_SEGMENT_CORE_FIELDS,
    SCENE_SEGMENT_EDITOR_FIELDS,
    SCENE_SEGMENT_FIELD_COMMENTS,
)
from app.services.documentary.frame_timeline_sampling import (
    dedupe_scene_environment_across_segments,
    normalize_scene_segments,
    resolve_frame_max_segment_duration_ms,
)
from app.services.documentary.frame_analysis_pairing import (
    analysis_artifact_dir,
    is_valid_analysis_artifact,
    load_analysis_artifact,
    sanitize_video_stem,
)

COMPACT_ARTIFACT_VERSION = "documentary-frame-analysis-v3-compact"
MINIMAL_SCENE_ARTIFACT_VERSION = "documentary-frame-analysis-v3-minimal-scene"

_KEYFRAME_FILENAME_RE = re.compile(r"^keyframe_\d{6}_\d{9}\.jpg$", re.IGNORECASE)
_SCENE_SEGMENT_SUBTITLE_FIELDS = (
    "subtitle",
    "subtitles",
    "subtitle_start",
    "subtitle_end",
    "subtitle_text_source",
    "subtitle_entries",
)

_SCENE_SEGMENT_FIELDS = (
    *SCENE_SEGMENT_CORE_FIELDS,
    *SCENE_SEGMENT_EDITOR_FIELDS,
    "characters",
    *_SCENE_SEGMENT_SUBTITLE_FIELDS,
)

MINIMAL_SCENE_SEGMENT_FIELDS = (
    *SCENE_SEGMENT_CORE_FIELDS,
    *SCENE_SEGMENT_EDITOR_FIELDS,
    "subtitle",
)

_OBSERVATION_FIELDS = (
    "timestamp",
    "characters",
    "observation",
    "subtitle",
    "subtitle_start",
    "subtitle_end",
    "subtitle_text_source",
    "burned_in_subtitle",
    "has_burned_in_subtitle",
)

_METADATA_FIELDS = (
    "video_path",
    "keyframe_cache_key",
    "frame_interval_seconds",
    "vision_batch_size",
    "vision_llm_provider",
    "vision_model_name",
    "vision_max_concurrency",
    "generated_at",
)

_SCENE_SEGMENT_FIELD_COMMENTS: dict[str, str] = dict(SCENE_SEGMENT_FIELD_COMMENTS)
_SCENE_SEGMENT_FIELD_COMMENTS.update(
    {
        "time_range": "字幕对位剪辑范围：subtitle_entries 首条 start 至末条 end；无条目时同 timestamp",
        "subtitle": "该时段内合并字幕对白（多句以；连接，已去重复标点）",
        "subtitle_entries": "字幕逐条列表（剪辑时间片段，每项含 start / end / text）",
        "subtitle_start": "对位字幕起始时间",
        "subtitle_end": "对位字幕结束时间",
        "subtitle_text_source": "字幕来源（如 srt / burned_in）",
    }
)

MINIMAL_SCENE_FIELD_COMMENTS: dict[str, str] = {
    key: _SCENE_SEGMENT_FIELD_COMMENTS[key]
    for key in MINIMAL_SCENE_SEGMENT_FIELDS
}

COMPACT_FIELD_COMMENTS: dict[str, str] = {
    **_SCENE_SEGMENT_FIELD_COMMENTS,
    "artifact_version": "精简版 artifact 版本号",
    "compacted_at": "精简导出时间（ISO8601）",
    "source_artifact_path": "来源完整抽帧 JSON 路径",
    "keyframe_cache_key": "关键帧缓存目录名（配合 frame_files 还原图片路径）",
    "video_segment_overview": "全片片段概览：段数、时间跨度、各段摘要",
    "scene_segments": "全片场景片段列表（脚本生成主用）",
    "frame_observations": "逐帧观察（字幕校准时可选）",
    "batches": "批次索引与时间范围（含 frame_files，无 raw_response）",
    "batch_index": "批次序号，从 0 起",
    "time_range": "scene_segments：subtitle_entries 首尾对位；batches：批次覆盖的时间范围",
    "status": "批次分析状态（success / failed）",
    "burned_in_subtitle": "画面底部硬字幕 OCR 原文（frame_observations 内）",
    "has_burned_in_subtitle": "该帧是否检测到硬字幕",
}


def _prepend_field_comments(payload: dict[str, Any], comments: dict[str, str]) -> dict[str, Any]:
    """JSON 不支持注释，用 field_comments 作为首字段说明各键含义。"""
    return {"field_comments": comments, **payload}


def is_compact_analysis_artifact(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    version = str(payload.get("artifact_version") or "")
    return version.endswith("-compact") or version.endswith("-minimal-scene")


def resolve_scene_segment_observation(segment: dict[str, Any]) -> str:
    """scene_segments 的 observation；缺失时由 action/emotion/key_visual 拼成。"""
    direct = str(segment.get("observation") or "").strip()
    if direct:
        return direct
    parts: list[str] = []
    for key in ("action", "emotion", "key_visual"):
        value = str(segment.get(key) or "").strip()
        if value:
            parts.append(value)
    return "；".join(parts)


def slim_scene_segment_core(segment: dict[str, Any]) -> dict[str, str]:
    """抽帧 scene_segments 核心六字段。"""
    return {
        "timestamp": str(segment.get("timestamp") or "").strip(),
        "scene": str(segment.get("scene") or "").strip(),
        "observation": resolve_scene_segment_observation(segment),
        "action": str(segment.get("action") or "").strip(),
        "emotion": str(segment.get("emotion") or "").strip(),
        "key_visual": str(segment.get("key_visual") or "").strip(),
    }


def keyframe_basename(path: str) -> str:
    """关键帧路径 → 文件名（keyframe_HHMMSS_mmmmmm.jpg）。"""
    name = os.path.basename(str(path or "").replace("\\", "/"))
    if _KEYFRAME_FILENAME_RE.fullmatch(name):
        return name
    return name if name else ""


def resolve_keyframe_cache_dir(artifact: dict[str, Any]) -> str:
    cache_key = str(artifact.get("keyframe_cache_key") or "").strip()
    if not cache_key:
        return ""
    return os.path.join(utils.temp_dir(), "keyframes", cache_key)


def resolve_frame_file_path(artifact: dict[str, Any], filename: str) -> str:
    """由 artifact 缓存目录 + 文件名还原绝对路径。"""
    basename = keyframe_basename(filename)
    if not basename:
        return ""
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    cache_dir = resolve_keyframe_cache_dir(artifact)
    if not cache_dir:
        return ""
    candidate = os.path.join(cache_dir, basename)
    return candidate if os.path.isfile(candidate) else ""


def resolve_batch_frame_files(
    artifact: dict[str, Any],
    batch: dict[str, Any],
) -> list[str]:
    """从 frame_files + keyframe_cache_key 还原批次关键帧绝对路径列表。"""
    files = [str(name) for name in (batch.get("frame_files") or []) if str(name).strip()]
    if not files:
        return []
    resolved = [resolve_frame_file_path(artifact, name) for name in files]
    return [path for path in resolved if path and os.path.isfile(path)]


def compact_frame_storage_in_artifact(artifact: dict[str, Any]) -> None:
    """规范化 frame_files 短文件名，剥离已废弃的 frame_paths / frame_path 字段。"""
    if not isinstance(artifact, dict):
        return

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        files = [str(name) for name in (batch.get("frame_files") or []) if str(name).strip()]
        if files:
            batch["frame_files"] = [keyframe_basename(name) for name in files if keyframe_basename(name)]
        batch.pop("frame_paths", None)

    for observation in artifact.get("frame_observations") or []:
        if isinstance(observation, dict):
            observation.pop("frame_path", None)

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for observation in batch.get("frame_observations") or []:
            if isinstance(observation, dict):
                observation.pop("frame_path", None)
        batch.pop("frame_paths", None)


def build_video_segment_overview(artifact: dict[str, Any]) -> dict[str, Any]:
    """抽帧完成后：汇总全片片段数量与各段剧情摘要，便于快速理解视频结构。"""
    segments = artifact.get("scene_segments")
    if not isinstance(segments, list):
        segments = []
    segments = [segment for segment in segments if isinstance(segment, dict)]
    if not segments:
        return {}

    ordered = sorted(segments, key=lambda item: str(item.get("timestamp") or ""))
    overview_items: list[dict[str, Any]] = []
    outline_parts: list[str] = []

    for index, segment in enumerate(ordered, start=1):
        timestamp = str(segment.get("timestamp") or "").strip()
        scene = str(segment.get("scene") or "").strip() or "未标场景"
        summary = resolve_scene_segment_observation(segment)
        if not summary:
            summary = str(segment.get("action") or "").strip()
        summary = summary[:160]
        overview_items.append(
            {
                "index": index,
                "timestamp": timestamp,
                "scene": scene,
                "summary": summary,
            }
        )
        time_label = timestamp.split("-", 1)[0].strip() if timestamp else ""
        snippet = summary[:72] + ("…" if len(summary) > 72 else "")
        outline_parts.append(f"#{index} {time_label} {scene}：{snippet}")

    first_range = str(ordered[0].get("timestamp") or "").strip()
    last_range = str(ordered[-1].get("timestamp") or "").strip()
    if first_range and last_range:
        start_label = first_range.split("-", 1)[0].strip()
        end_label = last_range.split("-", 1)[-1].strip()
        time_span = f"{start_label}-{end_label}" if start_label and end_label else first_range
    else:
        time_span = first_range or last_range

    return {
        "segment_count": len(overview_items),
        "time_span": time_span,
        "segments": overview_items,
        "narrative_outline": (
            f"全片共 {len(overview_items)} 个片段："
            + " → ".join(outline_parts)
        ),
    }


def slim_scene_segment_for_artifact(segment: dict[str, Any]) -> dict[str, Any]:
    """写入 artifact 的 scene_segment：核心六字段 + 剪辑师扩展字段 + 可选 subtitle。"""
    slim = slim_scene_segment_core(segment)
    for key in SCENE_SEGMENT_EDITOR_FIELDS:
        value = str(segment.get(key) or "").strip()
        if value:
            slim[key] = value
    for key in ("subtitle",):
        if key not in segment:
            continue
        value = segment.get(key)
        if value in (None, "", []):
            continue
        if key == "subtitle" and isinstance(value, str):
            value = resolve_segment_subtitle_text({"subtitle": value})
        if value:
            slim[key] = value
    if "batch_index" in segment:
        slim["batch_index"] = int(segment.get("batch_index", 0))
    return slim


def _slim_subtitle_entries(entries: Any) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        return []
    slim_entries: list[dict[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        entry = {
            key: str(item.get(key) or "").strip()
            for key in ("start", "end", "text")
            if str(item.get(key) or "").strip()
        }
        if entry.get("start") and entry.get("end") and entry.get("text"):
            slim_entries.append(entry)
    return slim_entries


def _attach_segment_time_range_field(segment: dict[str, Any]) -> None:
    """写入 time_range：subtitle_entries 首条 start 至末条 end（一小段完整对白）。"""
    clip_range = resolve_segment_time_range(segment)
    if clip_range:
        segment["time_range"] = clip_range


def slim_scene_segment_minimal(segment: dict[str, Any]) -> dict[str, Any]:
    """极精简导出：核心六字段 + 可选 subtitle 文本。"""
    core = slim_scene_segment_core(segment)
    subtitle = resolve_segment_subtitle_text(segment)

    slim: dict[str, Any] = {}
    for key in MINIMAL_SCENE_SEGMENT_FIELDS:
        if key == "subtitle":
            if subtitle:
                slim["subtitle"] = subtitle
            continue
        value = str(core.get(key) or "").strip()
        if value:
            slim[key] = value
    return slim


def _slim_scene_segment(segment: dict[str, Any], *, keep_batch_meta: bool) -> dict[str, Any]:
    slim = {key: segment[key] for key in _SCENE_SEGMENT_FIELDS if key in segment}
    if "subtitle" in slim and isinstance(slim["subtitle"], str):
        slim["subtitle"] = resolve_segment_subtitle_text(segment)
    _attach_segment_time_range_field(slim)
    if keep_batch_meta and "batch_index" in segment:
        slim["batch_index"] = segment["batch_index"]
    return slim


def _slim_frame_observation(observation: dict[str, Any], *, keep_batch_meta: bool) -> dict[str, Any]:
    slim = {key: observation[key] for key in _OBSERVATION_FIELDS if key in observation}
    if keep_batch_meta and "batch_index" in observation:
        slim["batch_index"] = observation["batch_index"]
    return slim


def _collect_top_level_segments(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    segments = artifact.get("scene_segments")
    if isinstance(segments, list) and segments:
        collected = [segment for segment in segments if isinstance(segment, dict)]
    else:
        collected = []
        for batch in artifact.get("batches") or []:
            if not isinstance(batch, dict):
                continue
            for segment in batch.get("scene_segments") or []:
                if isinstance(segment, dict):
                    payload = dict(segment)
                    payload.setdefault("batch_index", batch.get("batch_index"))
                    payload.pop("time_range", None)
                    collected.append(payload)
    collected = normalize_scene_segments(
        collected,
        max_duration_ms=resolve_frame_max_segment_duration_ms(get_documentary_settings()),
        settings=get_documentary_settings(),
    )
    partition_subtitle_entries_across_segments(collected)
    return collected


def _collect_top_level_observations(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    observations = artifact.get("frame_observations")
    if isinstance(observations, list) and observations:
        return [observation for observation in observations if isinstance(observation, dict)]

    collected: list[dict[str, Any]] = []
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for observation in batch.get("frame_observations") or []:
            if isinstance(observation, dict):
                payload = dict(observation)
                payload.setdefault("batch_index", batch.get("batch_index"))
                payload.setdefault("time_range", batch.get("time_range"))
                collected.append(payload)
    return collected


def rebuild_batches_from_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """从精简或完整 artifact 重建可用于字幕校准的 batches。"""
    existing = artifact.get("batches")

    segments_by_batch: dict[int, list[dict[str, Any]]] = {}
    observations_by_batch: dict[int, list[dict[str, Any]]] = {}
    summaries_by_batch: dict[int, str] = {}
    time_range_by_batch: dict[int, str] = {}
    status_by_batch: dict[int, str] = {}

    if isinstance(existing, list):
        for batch in existing:
            if not isinstance(batch, dict):
                continue
            batch_index = int(batch.get("batch_index", 0))
            time_range = str(batch.get("time_range") or "")
            if time_range:
                time_range_by_batch[batch_index] = time_range
            status_by_batch[batch_index] = str(batch.get("status") or "success")
            summary = str(
                batch.get("overall_activity_summary")
                or batch.get("summary")
                or batch.get("fallback_summary")
                or ""
            ).strip()
            if summary and batch_index not in summaries_by_batch:
                summaries_by_batch[batch_index] = summary

    for segment in _collect_top_level_segments(artifact):
        batch_index = int(segment.get("batch_index", 0))
        if batch_index not in time_range_by_batch:
            clip_range = resolve_segment_time_range(segment)
            if clip_range:
                time_range_by_batch[batch_index] = clip_range
        segments_by_batch.setdefault(batch_index, []).append(
            {
                key: value
                for key, value in segment.items()
                if key not in {"batch_index"}
            }
        )

    for observation in _collect_top_level_observations(artifact):
        batch_index = int(observation.get("batch_index", 0))
        if batch_index not in time_range_by_batch:
            time_range_by_batch[batch_index] = str(observation.get("time_range") or "")
        observations_by_batch.setdefault(batch_index, []).append(
            {
                key: value
                for key, value in observation.items()
                if key not in {"batch_index", "time_range", "frame_path"}
            }
        )

    batch_indices = sorted(
        set(time_range_by_batch.keys())
        | set(segments_by_batch.keys())
        | set(observations_by_batch.keys())
        | set(summaries_by_batch.keys())
        | set(status_by_batch.keys())
    )
    rebuilt: list[dict[str, Any]] = []
    for batch_index in batch_indices:
        rebuilt.append(
            {
                "batch_index": batch_index,
                "time_range": time_range_by_batch.get(batch_index, ""),
                "status": status_by_batch.get(batch_index, "success"),
                "scene_segments": segments_by_batch.get(batch_index, []),
                "frame_observations": observations_by_batch.get(batch_index, []),
                "overall_activity_summary": summaries_by_batch.get(batch_index, ""),
            }
        )
    return rebuilt


def compact_analysis_artifact(
    artifact: dict[str, Any],
    *,
    include_frame_observations: bool = True,
    include_summaries: bool = True,
    include_batch_index: bool = True,
    keep_batch_meta: bool = True,
    source_path: str = "",
) -> dict[str, Any]:
    """
    将完整抽帧分析 JSON 整理为精简版。

    移除：raw_response、批次内重复 scene_segments/frame_observations、fallback_summary 等。
    """
    if not is_valid_analysis_artifact(artifact):
        raise ValueError("无效的抽帧分析 artifact")

    segments = [
        _slim_scene_segment(segment, keep_batch_meta=keep_batch_meta)
        for segment in _collect_top_level_segments(artifact)
    ]

    compact: dict[str, Any] = {
        "artifact_version": COMPACT_ARTIFACT_VERSION,
        "compacted_at": datetime.now().isoformat(),
        "scene_segments": segments,
    }

    for key in _METADATA_FIELDS:
        if key in artifact and artifact.get(key) not in (None, ""):
            compact[key] = artifact[key]

    if source_path:
        compact["source_artifact_path"] = os.path.abspath(source_path)

    overview = artifact.get("video_segment_overview")
    if include_summaries and isinstance(overview, dict) and overview:
        compact["video_segment_overview"] = overview

    if include_frame_observations:
        compact["frame_observations"] = [
            _slim_frame_observation(observation, keep_batch_meta=keep_batch_meta)
            for observation in _collect_top_level_observations(artifact)
        ]

    if include_batch_index:
        compact_batches: list[dict[str, Any]] = []
        for batch in (artifact.get("batches") or []):
            if not isinstance(batch, dict):
                continue
            slim_batch: dict[str, Any] = {
                "batch_index": int(batch.get("batch_index", 0)),
                "time_range": str(batch.get("time_range") or ""),
                "status": str(batch.get("status") or "success"),
            }
            if keep_batch_meta:
                frame_files = [
                    str(name)
                    for name in (batch.get("frame_files") or [])
                    if str(name).strip()
                ]
                if frame_files:
                    slim_batch["frame_files"] = frame_files
            compact_batches.append(slim_batch)
        compact["batches"] = compact_batches
        if not compact["batches"]:
            compact["batches"] = [
                {
                    "batch_index": batch["batch_index"],
                    "time_range": batch["time_range"],
                    "status": batch["status"],
                }
                for batch in rebuild_batches_from_artifact(compact)
            ]

    return _prepend_field_comments(compact, COMPACT_FIELD_COMMENTS)


def minimal_scene_segments_artifact(
    artifact: dict[str, Any],
    *,
    source_path: str = "",
) -> dict[str, Any]:
    """
    极精简：仅 scene_segments，每条含 timestamp / scene / observation /
    action / emotion / key_visual / subtitle。不含 batches、逐帧路径等。
    """
    if not is_valid_analysis_artifact(artifact):
        raise ValueError("无效的抽帧分析 artifact")

    segments = [
        slim_scene_segment_minimal(segment)
        for segment in _collect_top_level_segments(artifact)
        if isinstance(segment, dict)
    ]
    compact: dict[str, Any] = {
        "artifact_version": MINIMAL_SCENE_ARTIFACT_VERSION,
        "scene_segments": segments,
    }
    if source_path:
        compact["source_artifact_path"] = os.path.abspath(source_path)
    for key in _METADATA_FIELDS:
        if key in artifact and artifact.get(key) not in (None, ""):
            compact[key] = artifact[key]
    return _prepend_field_comments(compact, MINIMAL_SCENE_FIELD_COMMENTS)


def compress_scene_segment_storage(segment: dict[str, Any]) -> None:
    """去掉 legacy subtitle_entries，仅保留 subtitle 文本。"""
    if not isinstance(segment, dict):
        return
    segment.pop("subtitle_entries", None)
    segment.pop("time_range", None)
    subtitle = resolve_segment_subtitle_text(segment)
    if subtitle:
        segment["subtitle"] = subtitle
    else:
        segment.pop("subtitle", None)


def normalize_analysis_artifact_storage(
    artifact: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """整理 scene_segments 存储（去 legacy subtitle_entries、环境去重），不删逐帧 observation。"""
    from app.services.documentary.documentary_settings import get_documentary_settings

    cfg = settings or get_documentary_settings()
    dedupe_env = bool(cfg.get("dedupe_scene_environment", True))

    segments = artifact.get("scene_segments")
    if isinstance(segments, list) and segments:
        ordered = sorted(
            [segment for segment in segments if isinstance(segment, dict)],
            key=lambda item: str(item.get("timestamp") or ""),
        )
        if dedupe_env:
            dedupe_scene_environment_across_segments(ordered)
        for segment in ordered:
            compress_scene_segment_storage(segment)
        artifact["scene_segments"] = ordered

        segments_by_batch: dict[int, list[dict[str, Any]]] = {}
        for segment in ordered:
            if "batch_index" not in segment:
                continue
            batch_index = int(segment.get("batch_index", 0))
            segments_by_batch.setdefault(batch_index, []).append(segment)
        for batch in artifact.get("batches") or []:
            if not isinstance(batch, dict):
                continue
            batch_index = int(batch.get("batch_index", 0))
            if str(batch.get("status") or "").lower() != "success":
                batch["scene_segments"] = []
                batch["frame_observations"] = []
                batch.pop("raw_response", None)
                batch.pop("fallback_summary", None)
                continue
            batch["scene_segments"] = segments_by_batch.get(batch_index, [])

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        if str(batch.get("status") or "").lower() != "success":
            batch.pop("raw_response", None)
            batch.pop("scene_segments", None)
            batch.pop("frame_observations", None)
            batch.pop("fallback_summary", None)

    compact_frame_storage_in_artifact(artifact)
    overview = build_video_segment_overview(artifact)
    if overview:
        artifact["video_segment_overview"] = overview

    return artifact


def strip_frame_analysis_debug_payload(
    artifact: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """剥离调试字段与逐帧 observation 正文（仅保留硬字幕等 slim 字段）。"""
    from app.services.documentary.documentary_settings import get_documentary_settings

    cfg = settings or get_documentary_settings()
    if not cfg.get("strip_frame_analysis_debug_fields", True):
        return artifact

    has_scene_segments = bool(
        isinstance(artifact.get("scene_segments"), list) and artifact.get("scene_segments")
    )

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        status = str(batch.get("status") or "").lower()
        if status == "success":
            batch.pop("raw_response", None)
            batch.pop("fallback_summary", None)
            batch.pop("error_message", None)
            # frame_files + keyframe_cache_key 保留，供 OCR / 重跑批次还原路径
        else:
            batch.pop("raw_response", None)
            batch.pop("scene_segments", None)
            batch.pop("frame_observations", None)
            batch.pop("fallback_summary", None)
        batch.pop("subtitle", None)
        batch.pop("subtitle_entries", None)
        batch.pop("subtitle_excerpt", None)
        if has_scene_segments:
            batch.pop("frame_observations", None)
            batch.pop("scene_segments", None)

    if has_scene_segments:
        slim_observations: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for observation in artifact.get("frame_observations") or []:
            if not isinstance(observation, dict) or id(observation) in seen_ids:
                continue
            seen_ids.add(id(observation))
            slim = _slim_frame_observation(observation, keep_batch_meta=True)
            slim.pop("observation", None)
            if slim:
                slim_observations.append(slim)
        artifact["frame_observations"] = slim_observations

    return artifact


def compress_analysis_artifact(
    artifact: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    strip_debug: bool | None = None,
) -> dict[str, Any]:
    """
    压缩抽帧 JSON 体积（原地修改）：
    - 同场景环境去重、去掉 legacy subtitle_entries（始终执行）
    - 可选：去掉 batch 调试字段与逐帧 observation 正文（strip_debug）
    """
    from app.services.documentary.documentary_settings import get_documentary_settings

    cfg = settings or get_documentary_settings()
    normalize_analysis_artifact_storage(artifact, settings=cfg)
    if strip_debug is None:
        strip_debug = bool(cfg.get("strip_frame_analysis_debug_fields", True))
    if strip_debug:
        strip_frame_analysis_debug_payload(artifact, settings=cfg)
    return artifact


def default_compact_output_path(source_path: str) -> str:
    directory = os.path.dirname(source_path) or analysis_artifact_dir()
    stem = os.path.splitext(os.path.basename(source_path))[0]
    if stem.endswith("_frame_analysis"):
        stem = stem[: -len("_frame_analysis")]
    elif stem.endswith("_frame_analysis_compact"):
        stem = stem[: -len("_frame_analysis_compact")]
    return os.path.join(directory, f"{stem}_frame_analysis_compact.json")


def default_minimal_scene_output_path(source_path: str) -> str:
    directory = os.path.dirname(source_path) or analysis_artifact_dir()
    stem = os.path.splitext(os.path.basename(source_path))[0]
    for suffix in (
        "_frame_analysis_minimal",
        "_frame_analysis_compact",
        "_frame_analysis",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return os.path.join(directory, f"{stem}_frame_analysis_minimal.json")


def estimate_compact_sizes(
    artifact: dict[str, Any],
    *,
    source_bytes: int | None = None,
) -> dict[str, int]:
    """返回各精简档位 JSON 字节大小，供 UI 展示。"""
    original = source_bytes if source_bytes is not None else len(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
    )
    presets = {
        "minimal": compact_analysis_artifact(
            artifact,
            include_frame_observations=False,
            include_summaries=False,
            include_batch_index=False,
            keep_batch_meta=False,
        ),
        "script": compact_analysis_artifact(
            artifact,
            include_frame_observations=False,
            include_summaries=True,
            include_batch_index=True,
            keep_batch_meta=True,
        ),
        "calibration": compact_analysis_artifact(
            artifact,
            include_frame_observations=True,
            include_summaries=True,
            include_batch_index=True,
            keep_batch_meta=True,
        ),
        "minimal_scene": minimal_scene_segments_artifact(artifact),
    }
    sizes = {"original": original}
    for name, payload in presets.items():
        sizes[name] = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return sizes


def save_minimal_scene_analysis_artifact(
    source_path: str,
    *,
    output_path: str = "",
    indent: int = 2,
) -> dict[str, Any]:
    """仅导出 scene_segments（timestamp / scene / observation）。"""
    artifact = load_analysis_artifact(source_path)
    compact = minimal_scene_segments_artifact(artifact, source_path=source_path)
    target_path = (output_path or default_minimal_scene_output_path(source_path)).strip()
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as fp:
        json.dump(compact, fp, ensure_ascii=False, indent=indent)
    original_bytes = os.path.getsize(source_path)
    compact_bytes = os.path.getsize(target_path)
    return {
        "output_path": target_path,
        "original_bytes": original_bytes,
        "compact_bytes": compact_bytes,
        "reduction_percent": round(100 * (1 - compact_bytes / original_bytes), 1) if original_bytes else 0,
        "artifact": compact,
    }


def save_compact_analysis_artifact(
    source_path: str,
    *,
    output_path: str = "",
    indent: int = 2,
    **compact_options: Any,
) -> dict[str, Any]:
    """读取源 JSON，写出精简版，返回路径与体积统计。"""
    artifact = load_analysis_artifact(source_path)
    if compact_options.get("minimal_scene_only"):
        return save_minimal_scene_analysis_artifact(
            source_path,
            output_path=output_path,
            indent=indent,
        )
    options = {key: value for key, value in compact_options.items() if key != "minimal_scene_only"}
    compact = compact_analysis_artifact(artifact, source_path=source_path, **options)

    target_path = (output_path or default_compact_output_path(source_path)).strip()
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as fp:
        json.dump(compact, fp, ensure_ascii=False, indent=indent)

    original_bytes = os.path.getsize(source_path)
    compact_bytes = os.path.getsize(target_path)
    return {
        "output_path": target_path,
        "original_bytes": original_bytes,
        "compact_bytes": compact_bytes,
        "reduction_percent": round(100 * (1 - compact_bytes / original_bytes), 1) if original_bytes else 0,
        "artifact": compact,
    }
