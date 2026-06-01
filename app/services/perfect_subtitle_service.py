#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Perfect dual-track subtitle builder for film/TV narration.

- OST=1: extract original dialogue from source SRT (fallback: ASR on clip)
- OST=0/2: ASR on TTS audio for word-accurate timing (fallback: existing TTS srt / proportional split)
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Optional

from loguru import logger

from app.config import config
from app.services.asr_subtitle_provider import transcribe_media_to_entries
from app.services.srt_utils import (
    SrtEntry,
    extract_entries_in_range,
    merge_and_sort_entries,
    offset_entries,
    parse_edited_time_range_seconds,
    parse_srt_file,
    parse_timestamp_range,
    resolve_overlaps,
    split_text_to_entries,
    write_srt_file,
)
from app.utils import utils


PERFECT_SUBTITLE_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "provider": "auto",
    "ost1_from_source_srt": True,
    "ost1_asr_fallback": True,
    "ost0_use_asr": True,
    "prefer_existing_tts_subtitle": False,
    "export_separate_tracks": True,
    "narration_label": "",
    "original_label": "",
    "max_chars": 18,
    "max_duration": 4.0,
}


def get_perfect_subtitle_settings() -> dict[str, Any]:
    settings = deepcopy(PERFECT_SUBTITLE_DEFAULTS)
    section = config.perfect_subtitle if hasattr(config, "perfect_subtitle") else {}
    if isinstance(section, dict):
        for key in PERFECT_SUBTITLE_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]
    return settings


def is_perfect_subtitle_enabled() -> bool:
    return bool(get_perfect_subtitle_settings().get("enabled", True))


def _edited_offset_ms(segment: dict[str, Any]) -> int:
    start_sec, _ = parse_edited_time_range_seconds(segment.get("editedTimeRange", ""))
    return int(round(start_sec * 1000))


def _segment_duration_ms(segment: dict[str, Any]) -> int:
    duration = segment.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return int(round(float(duration) * 1000))

    start_ms, end_ms = parse_timestamp_range(segment.get("timestamp", ""))
    if end_ms > start_ms:
        return end_ms - start_ms
    return 0


def _load_segment_subtitle_entries(path: str) -> list[SrtEntry]:
    if not path or not os.path.exists(path):
        return []
    return parse_srt_file(path)


def _build_ost1_entries(
    segment: dict[str, Any],
    source_entries: list[SrtEntry],
    *,
    task_id: str,
    settings: dict[str, Any],
) -> list[SrtEntry]:
    segment_id = segment.get("_id", "unknown")
    clip_start_ms, clip_end_ms = parse_timestamp_range(segment.get("timestamp", ""))
    if clip_end_ms <= clip_start_ms:
        clip_start_ms, clip_end_ms = parse_timestamp_range(segment.get("sourceTimeRange", ""))

    label = str(settings.get("original_label") or "")
    local_entries: list[SrtEntry] = []

    if settings.get("ost1_from_source_srt", True) and source_entries:
        local_entries = extract_entries_in_range(source_entries, clip_start_ms, clip_end_ms)
        for entry in local_entries:
            entry.label = label

    if not local_entries and settings.get("ost1_asr_fallback", True):
        media_path = segment.get("video") or segment.get("audio")
        if media_path and os.path.exists(media_path):
            try:
                local_entries = transcribe_media_to_entries(
                    media_path,
                    task_dir=utils.task_dir(task_id),
                    segment_id=segment_id,
                    provider=settings.get("provider", "auto"),
                    max_chars=int(settings.get("max_chars", 18)),
                    max_duration=float(settings.get("max_duration", 4.0)),
                )
                for entry in local_entries:
                    entry.label = label
            except Exception as exc:
                logger.warning(f"OST=1 片段 #{segment_id} ASR 回退失败: {exc}")

    return offset_entries(local_entries, _edited_offset_ms(segment))


def _build_narration_entries(
    segment: dict[str, Any],
    *,
    task_id: str,
    settings: dict[str, Any],
) -> list[SrtEntry]:
    segment_id = segment.get("_id", "unknown")
    label = str(settings.get("narration_label") or "")
    duration_ms = _segment_duration_ms(segment)
    local_entries: list[SrtEntry] = []

    existing_subtitle = segment.get("subtitle") or ""
    if settings.get("prefer_existing_tts_subtitle") and existing_subtitle:
        local_entries = _load_segment_subtitle_entries(existing_subtitle)

    if not local_entries and settings.get("ost0_use_asr", True):
        audio_path = segment.get("audio") or ""
        if audio_path and os.path.exists(audio_path):
            try:
                local_entries = transcribe_media_to_entries(
                    audio_path,
                    task_dir=utils.task_dir(task_id),
                    segment_id=f"narr_{segment_id}",
                    provider=settings.get("provider", "auto"),
                    max_chars=int(settings.get("max_chars", 18)),
                    max_duration=float(settings.get("max_duration", 4.0)),
                )
            except Exception as exc:
                logger.warning(f"OST=0 片段 #{segment_id} ASR 转写失败，尝试回退: {exc}")

    if not local_entries and existing_subtitle:
        local_entries = _load_segment_subtitle_entries(existing_subtitle)

    if not local_entries:
        narration = str(segment.get("narration") or "").strip()
        if narration and duration_ms > 0:
            local_entries = split_text_to_entries(
                narration,
                duration_ms,
                max_chars=int(settings.get("max_chars", 18)),
                label=label,
            )

    for entry in local_entries:
        if label and not entry.label:
            entry.label = label
    return offset_entries(local_entries, _edited_offset_ms(segment))


def build_perfect_subtitles(
    script_list: list[dict[str, Any]],
    *,
    task_id: str,
    source_subtitle_path: Optional[str] = None,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Optional[str]]:
    cfg = settings or get_perfect_subtitle_settings()
    task_dir = utils.task_dir(task_id)

    source_entries = parse_srt_file(source_subtitle_path) if source_subtitle_path else []
    if source_subtitle_path and not source_entries:
        logger.warning(f"源字幕无法解析或为空: {source_subtitle_path}")

    narration_groups: list[list[SrtEntry]] = []
    original_groups: list[list[SrtEntry]] = []

    sorted_segments = sorted(
        script_list,
        key=lambda item: (
            _edited_offset_ms(item),
            int(item.get("_id", 0) or 0),
        ),
    )

    for segment in sorted_segments:
        ost = int(segment.get("OST", 0) or 0)
        if ost == 1:
            original_groups.append(_build_ost1_entries(segment, source_entries, task_id=task_id, settings=cfg))
        elif ost in (0, 2):
            narration_groups.append(_build_narration_entries(segment, task_id=task_id, settings=cfg))

    narration_entries = resolve_overlaps(merge_and_sort_entries(narration_groups))
    original_entries = resolve_overlaps(merge_and_sort_entries(original_groups))
    merged_entries = resolve_overlaps(
        merge_and_sort_entries([narration_entries, original_entries])
    )

    outputs = {
        "merged": None,
        "narration": None,
        "original": None,
    }

    if merged_entries:
        outputs["merged"] = write_srt_file(merged_entries, os.path.join(task_dir, "perfect_subtitle_merged.srt"))
        logger.info(f"完美字幕已生成: {outputs['merged']} ({len(merged_entries)} 条)")

    if cfg.get("export_separate_tracks", True):
        if narration_entries:
            outputs["narration"] = write_srt_file(
                narration_entries, os.path.join(task_dir, "perfect_subtitle_narration.srt")
            )
        if original_entries:
            outputs["original"] = write_srt_file(
                original_entries, os.path.join(task_dir, "perfect_subtitle_original.srt")
            )

    return outputs


def build_merged_subtitle_path(
    script_list: list[dict[str, Any]],
    *,
    task_id: str,
    source_subtitle_path: Optional[str] = None,
) -> Optional[str]:
    if not is_perfect_subtitle_enabled():
        return None

    try:
        result = build_perfect_subtitles(
            script_list,
            task_id=task_id,
            source_subtitle_path=source_subtitle_path,
        )
        return result.get("merged")
    except Exception as exc:
        logger.error(f"完美字幕生成失败，将回退到旧合并逻辑: {exc}")
        return None
