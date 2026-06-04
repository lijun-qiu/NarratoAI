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
from app.services.video_output_settings import (
    get_video_output_settings,
    is_picture_narration_enabled,
)
from app.services.srt_utils import (
    SrtEntry,
    clip_entries_to_duration,
    align_entries_to_speech_duration,
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
from app.services.update_script import probe_media_duration


PERFECT_SUBTITLE_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "provider": "auto",
    "ost1_from_source_srt": True,
    "ost1_asr_fallback": True,
    "ost0_use_asr": True,
    "prefer_existing_tts_subtitle": True,
    "export_separate_tracks": True,
    "defer_asr_until_final": False,
    "narration_label": "",
    "original_label": "",
    "max_chars": 18,
    "max_duration": 6.0,
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


def is_deferred_subtitle_enabled() -> bool:
    """先合成成片（含水印/旁白），最后再 API 转写并烧录主字幕。"""
    settings = get_perfect_subtitle_settings()
    if not settings.get("enabled", True):
        return False
    return bool(settings.get("defer_asr_until_final", False))


def _edited_offset_ms(segment: dict[str, Any]) -> int:
    start_sec, _ = parse_edited_time_range_seconds(segment.get("editedTimeRange", ""))
    return int(round(start_sec * 1000))


def _segment_clip_range_ms(segment: dict[str, Any]) -> tuple[int, int]:
    """片段在源片上的裁剪范围；优先 sourceTimeRange（与 ffmpeg 输出文件名一致）。"""
    for key in ("sourceTimeRange", "timestamp"):
        start_ms, end_ms = parse_timestamp_range(segment.get(key, ""))
        if end_ms > start_ms:
            return start_ms, end_ms
    return 0, 0


def _segment_video_duration_ms(segment: dict[str, Any]) -> int:
    video_path = segment.get("video") or ""
    if video_path:
        video_sec = probe_media_duration(video_path)
        if video_sec > 0:
            return int(round(video_sec * 1000))

    duration = segment.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return int(round(float(duration) * 1000))

    start_ms, end_ms = parse_timestamp_range(segment.get("timestamp", ""))
    if end_ms > start_ms:
        return end_ms - start_ms
    return 0


def _segment_duration_ms(segment: dict[str, Any]) -> int:
    return _segment_video_duration_ms(segment)


def _segment_audio_duration_ms(segment: dict[str, Any]) -> int:
    audio_path = segment.get("audio") or ""
    if audio_path and os.path.exists(audio_path):
        audio_sec = probe_media_duration(audio_path)
        if audio_sec > 0:
            return int(round(audio_sec * 1000))
    return 0


def _align_segment_subtitle_entries(
    entries: list[SrtEntry],
    segment: dict[str, Any],
) -> list[SrtEntry]:
    video_ms = _segment_video_duration_ms(segment)
    audio_ms = _segment_audio_duration_ms(segment)
    if not entries:
        return entries

    # 解说段以 TTS 音频时长为基准，避免被较短的视频时长压缩导致字幕抢跑
    if audio_ms > 0:
        speech_ms = audio_ms
        video_cap = video_ms if video_ms > 0 else audio_ms
    elif video_ms > 0:
        speech_ms = video_ms
        video_cap = video_ms
    else:
        return entries

    return align_entries_to_speech_duration(
        entries,
        speech_duration_ms=speech_ms,
        video_duration_ms=video_cap,
    )


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
    clip_start_ms, clip_end_ms = _segment_clip_range_ms(segment)

    label = str(settings.get("original_label") or "")
    local_entries: list[SrtEntry] = []

    if settings.get("ost1_from_source_srt", True) and source_entries:
        local_entries = extract_entries_in_range(source_entries, clip_start_ms, clip_end_ms)
        for entry in local_entries:
            entry.label = "original"

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
                    entry.label = "original"
            except Exception as exc:
                logger.warning(f"OST=1 片段 #{segment_id} ASR 回退失败: {exc}")

    video_ms = _segment_video_duration_ms(segment)
    if video_ms > 0 and local_entries:
        local_entries = _align_segment_subtitle_entries(local_entries, segment)

    display_prefix = label.strip()
    if display_prefix:
        for entry in local_entries:
            if display_prefix not in entry.text:
                entry.text = f"{display_prefix}{entry.text}"
    return offset_entries(local_entries, _edited_offset_ms(segment))


def _build_narration_entries(
    segment: dict[str, Any],
    *,
    task_id: str,
    settings: dict[str, Any],
) -> list[SrtEntry]:
    segment_id = segment.get("_id", "unknown")
    display_label = str(settings.get("narration_label") or "").strip()
    duration_ms = _segment_duration_ms(segment)
    local_entries: list[SrtEntry] = []

    existing_subtitle = segment.get("subtitle") or ""
    if settings.get("prefer_existing_tts_subtitle", True) and existing_subtitle:
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
                label="narration",
            )

    video_ms = _segment_video_duration_ms(segment)
    if video_ms > 0 and local_entries:
        local_entries = clip_entries_to_duration(local_entries, video_ms)
        local_entries = _align_segment_subtitle_entries(local_entries, segment)

    for entry in local_entries:
        entry.label = entry.label or "narration"
        if display_label and display_label not in entry.text:
            entry.text = f"{display_label}{entry.text}"
    return offset_entries(local_entries, _edited_offset_ms(segment))


def _build_ost1_picture_narration_entries(
    segment: dict[str, Any],
    *,
    video_output: dict[str, Any],
) -> list[SrtEntry]:
    """原声段旁白字幕：画面/动作/情绪描述，取自脚本 picture 字段。"""
    picture = str(segment.get("picture") or "").strip()
    if not picture:
        return []

    duration_ms = _segment_duration_ms(segment)
    if duration_ms <= 0:
        return []

    display_sec = float(video_output.get("picture_narration_duration", 2.0))
    display_ms = int(max(0.5, display_sec) * 1000)
    display_ms = min(display_ms, duration_ms)

    max_chars = int(video_output.get("picture_narration_max_chars", 16))
    local_entries = split_text_to_entries(
        picture,
        display_ms,
        max_chars=max_chars,
        label="picture_narration",
    )
    return offset_entries(local_entries, _edited_offset_ms(segment))


def build_picture_narration_subtitle_path(
    script_list: list[dict[str, Any]],
    *,
    task_id: str,
    video_output: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """为 OST=1 原声段生成左侧旁白描述字幕轨。"""
    cfg = video_output or get_video_output_settings()
    if not is_picture_narration_enabled(cfg):
        return None

    sorted_segments = sorted(
        script_list,
        key=lambda item: (
            _edited_offset_ms(item),
            int(item.get("_id", 0) or 0),
        ),
    )

    picture_groups: list[list[SrtEntry]] = []
    for segment in sorted_segments:
        if int(segment.get("OST", 0) or 0) != 1:
            continue
        entries = _build_ost1_picture_narration_entries(segment, video_output=cfg)
        if entries:
            picture_groups.append(entries)

    if not picture_groups:
        return None

    merged = resolve_overlaps(merge_and_sort_entries(picture_groups))
    if not merged:
        return None

    task_dir = utils.task_dir(task_id)
    output_path = write_srt_file(merged, os.path.join(task_dir, "picture_narration.srt"))
    logger.info(f"原声段旁白字幕已生成: {output_path} ({len(merged)} 条)")
    return output_path


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
