#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频网格快扫：压缩整片上传，按 5–30 秒固定格输出简化 JSON。"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.config.llm_gateway_router import describe_llm_route, resolve_llm_credentials
from app.services.documentary.frame_analysis_pairing import analysis_artifact_dir, sanitize_video_stem
from app.services.documentary.video_episode_analysis import (
    VideoEpisodeAnalysisService,
    _clean_json_output,
    _format_timestamp,
    _parse_time_range_bounds,
    _prepare_chunk_reference_context,
    _probe_duration_seconds,
)
from app.services.documentary.video_episode_constants import UPLOAD_TRANSCODE_PROFILES
from app.services.documentary.plot_reference import (
    build_plot_reference_prompt_section,
    normalize_plot_reference,
)
from app.services.prompts.documentary.video_whole_grid_analysis import (
    build_grid_batch_prompt_addon,
    build_grid_batch_time_anchor_block,
    build_grid_schedule_prompt_block,
    build_previous_grid_batch_tail_context,
    build_whole_grid_analysis_prompt,
)

WHOLE_GRID_PLOT_REFERENCE_MAX_CHARS = 1200
_CONTINUATION_MARKERS = ("画面延续上段", "画面无明显变化", "本窗未解析", "本窗模型未返回")
from app.utils import utils

WHOLE_GRID_ARTIFACT_VERSION = "documentary-video-whole-grid-v1"
WHOLE_GRID_MIN_INTERVAL = 5
WHOLE_GRID_MAX_INTERVAL = 30
WHOLE_GRID_DEFAULT_INTERVAL = 20
WHOLE_GRID_DEFAULT_BATCH_COUNT = 2
WHOLE_GRID_ONE_SHOT_MAX_INTERVAL = 30
_MAX_GRID_SEGMENTS_PER_API_CALL = 40
_DEFAULT_MAX_OUTPUT_TOKENS = 32000
_WHOLE_GRID_TIMEOUT = 900.0
_WHOLE_GRID_ONE_SHOT_TIMEOUT = 1800.0
_MAX_API_RETRIES = 3

WHOLE_GRID_TRANSCODE_PROFILE_CHAIN = (
    "grid_whole",
    "grid_whole_compact",
    "grid_whole_tiny",
    "grid_whole_ultra",
)

VIDEO_WHOLE_GRID_DEFAULTS: dict[str, Any] = {
    "max_upload_mb": 24.0,
    "grid_interval_seconds": WHOLE_GRID_DEFAULT_INTERVAL,
    # 分批段数：>1 时按时间轴均分为 N 段（默认 2 段）；0 = 按模型每批格数自动切分
    "batch_count": WHOLE_GRID_DEFAULT_BATCH_COUNT,
    # 0 = 按视觉模型自动折中（推荐）；>0 则强制覆盖
    "max_segments_per_api_call": 0,
    "max_output_tokens": 0,
    "auto_batch_by_model": True,
    "force_one_shot": False,
    "upload_transcode_profile": "grid_whole",
}


@dataclass(frozen=True)
class GridBatchPlan:
    """网格输出分批策略（按模型能力折中）。"""

    model_id: str
    profile_label: str
    max_segments_per_api_call: int
    max_output_tokens: int
    tokens_per_segment: int
    min_output_tokens: int = 6000


def _normalize_grid_model_id(model_name: str) -> str:
    text = (model_name or "").strip().lower()
    if "/" in text:
        text = text.split("/", 1)[-1].strip()
    return text


def _model_grid_batch_profile(model_name: str) -> GridBatchPlan:
    model_id = _normalize_grid_model_id(model_name)
    if "gemini-3-flash-preview" in model_id or model_id == "gemini-3-flash":
        return GridBatchPlan(
            model_id=model_id,
            profile_label="gemini-3-flash-preview",
            max_segments_per_api_call=72,
            max_output_tokens=64000,
            tokens_per_segment=150,
            min_output_tokens=8000,
        )
    if any(
        token in model_id
        for token in (
            "gemini-3.1-pro",
            "gemini-2.5-pro",
            "gemini-3-pro",
            "gpt-4o",
            "qwen-vl-max",
            "qwen2.5-vl-72b",
        )
    ):
        return GridBatchPlan(
            model_id=model_id,
            profile_label="strong-vision",
            max_segments_per_api_call=56,
            max_output_tokens=48000,
            tokens_per_segment=165,
            min_output_tokens=7000,
        )
    if any(
        token in model_id
        for token in (
            "gemini-3.1-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "qwen-vl-plus",
            "qwen2.5-vl-32b",
        )
    ):
        return GridBatchPlan(
            model_id=model_id,
            profile_label="balanced-flash",
            # 整片视频时间定位较难，单批略小更稳
            max_segments_per_api_call=32,
            max_output_tokens=32000,
            tokens_per_segment=170,
            min_output_tokens=6500,
        )
    return GridBatchPlan(
        model_id=model_id or "default",
        profile_label="conservative",
        max_segments_per_api_call=_MAX_GRID_SEGMENTS_PER_API_CALL,
        max_output_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        tokens_per_segment=180,
        min_output_tokens=6000,
    )


def resolve_grid_batch_plan(
    model_name: str,
    settings: dict[str, Any] | None = None,
) -> GridBatchPlan:
    """结合 config 与模型能力，确定每批格数与输出 token 上限。"""
    cfg = settings or get_video_whole_grid_settings()
    base = _model_grid_batch_profile(model_name)
    if not bool(cfg.get("auto_batch_by_model", True)):
        segments = int(cfg.get("max_segments_per_api_call") or base.max_segments_per_api_call)
        output_tokens = int(cfg.get("max_output_tokens") or base.max_output_tokens)
        return GridBatchPlan(
            model_id=base.model_id,
            profile_label=f"{base.profile_label}+manual",
            max_segments_per_api_call=max(8, segments),
            max_output_tokens=max(6000, output_tokens),
            tokens_per_segment=base.tokens_per_segment,
            min_output_tokens=base.min_output_tokens,
        )

    segments_override = int(cfg.get("max_segments_per_api_call") or 0)
    output_override = int(cfg.get("max_output_tokens") or 0)
    return GridBatchPlan(
        model_id=base.model_id,
        profile_label=base.profile_label,
        max_segments_per_api_call=(
            max(8, segments_override) if segments_override > 0 else base.max_segments_per_api_call
        ),
        max_output_tokens=(
            max(6000, output_override) if output_override > 0 else base.max_output_tokens
        ),
        tokens_per_segment=base.tokens_per_segment,
        min_output_tokens=base.min_output_tokens,
    )


def estimate_grid_batch_max_tokens(
    segment_count: int,
    batch_plan: GridBatchPlan | None = None,
    *,
    one_shot: bool = False,
) -> int:
    plan = batch_plan or GridBatchPlan(
        model_id="default",
        profile_label="conservative",
        max_segments_per_api_call=_MAX_GRID_SEGMENTS_PER_API_CALL,
        max_output_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        tokens_per_segment=180,
    )
    if one_shot:
        return plan.max_output_tokens
    count = max(1, int(segment_count))
    estimated = plan.min_output_tokens + count * plan.tokens_per_segment
    return min(plan.max_output_tokens, max(plan.min_output_tokens, estimated))


def _is_output_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "finish_reason=length" in text or "finish_reason': 'length" in text


def resolve_one_shot_grid_interval(
    video_duration_seconds: float,
    *,
    requested_interval: int,
) -> tuple[int, int, bool]:
    """单次生成：按用户格距建表，不因预估格数拦截或自动加粗。"""
    requested = max(
        WHOLE_GRID_MIN_INTERVAL,
        min(WHOLE_GRID_MAX_INTERVAL, int(requested_interval)),
    )
    count = len(
        build_grid_schedule(
            video_duration_seconds,
            interval_seconds=requested,
            max_interval_seconds=WHOLE_GRID_MAX_INTERVAL,
        )
    )
    return requested, count, False


def estimate_grid_run_plan(
    video_duration_seconds: float,
    *,
    grid_interval_seconds: int,
    model_name: str,
    settings: dict[str, Any] | None = None,
    force_one_shot: bool | None = None,
) -> dict[str, Any]:
    cfg = settings or get_video_whole_grid_settings()
    plan = resolve_grid_batch_plan(model_name, cfg)
    requested = max(
        WHOLE_GRID_MIN_INTERVAL,
        min(WHOLE_GRID_MAX_INTERVAL, int(grid_interval_seconds)),
    )
    if force_one_shot is None:
        force_one_shot = bool(cfg.get("force_one_shot", False))
    if force_one_shot:
        effective, count, adjusted = resolve_one_shot_grid_interval(
            video_duration_seconds,
            requested_interval=requested,
        )
        return {
            "one_shot": True,
            "grid_interval_requested": requested,
            "grid_interval_effective": effective,
            "grid_interval_auto_adjusted": adjusted,
            "grid_segment_count": count,
            "api_call_count": 1,
            "batch_plan": plan,
        }

    schedule = build_grid_schedule(video_duration_seconds, interval_seconds=requested)
    batches = resolve_schedule_batches(
        schedule,
        settings=cfg,
        batch_plan=plan,
        force_one_shot=False,
    )
    return {
        "one_shot": False,
        "grid_interval_requested": requested,
        "grid_interval_effective": requested,
        "grid_interval_auto_adjusted": False,
        "grid_segment_count": len(schedule),
        "api_call_count": len(batches),
        "batch_plan": plan,
    }


def estimate_grid_api_batch_count(
    video_duration_seconds: float,
    *,
    grid_interval_seconds: int,
    model_name: str,
    settings: dict[str, Any] | None = None,
) -> tuple[int, int, GridBatchPlan]:
    run_plan = estimate_grid_run_plan(
        video_duration_seconds,
        grid_interval_seconds=grid_interval_seconds,
        model_name=model_name,
        settings=settings,
    )
    plan = run_plan["batch_plan"]
    return (
        int(run_plan["grid_segment_count"]),
        int(run_plan["api_call_count"]),
        plan,
    )

def _get_grid_transcode_profile(name: str) -> dict[str, Any]:
    profile = UPLOAD_TRANSCODE_PROFILES.get(name)
    if not profile:
        raise ValueError(f"未知网格上传转码档位: {name}")
    return dict(profile)


def get_video_whole_grid_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = deepcopy(VIDEO_WHOLE_GRID_DEFAULTS)
    try:
        from app.config.config import _cfg

        section = _cfg.get("video_whole_grid_analysis", {})
    except Exception:
        section = {}
    if isinstance(section, dict):
        for key in VIDEO_WHOLE_GRID_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]
    if overrides:
        for key, value in overrides.items():
            if key in VIDEO_WHOLE_GRID_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def default_video_whole_grid_analysis_path(video_path: str) -> str:
    stem = sanitize_video_stem(video_path)
    return os.path.join(analysis_artifact_dir(), f"{stem}_video_whole_grid_analysis.json")


def build_grid_schedule(
    duration_seconds: float,
    *,
    interval_seconds: int,
    start_offset_seconds: float = 0.0,
    max_interval_seconds: int | None = None,
) -> list[str]:
    cap = max_interval_seconds or WHOLE_GRID_MAX_INTERVAL
    interval = max(WHOLE_GRID_MIN_INTERVAL, min(cap, int(interval_seconds)))
    total = max(0.0, float(duration_seconds))
    offset = max(0.0, float(start_offset_seconds))
    end_limit = offset + total
    schedule: list[str] = []
    cursor = offset
    while cursor < end_limit - 0.01:
        window_end = min(cursor + interval, end_limit)
        if window_end - cursor < 0.5:
            break
        schedule.append(f"{_format_timestamp(cursor)}-{_format_timestamp(window_end)}")
        cursor = window_end
    return schedule


def split_schedule_for_batches(
    schedule: list[str],
    *,
    max_per_batch: int,
) -> list[list[str]]:
    limit = max(1, int(max_per_batch))
    if len(schedule) <= limit:
        return [schedule]
    batches: list[list[str]] = []
    cursor = 0
    while cursor < len(schedule):
        batches.append(schedule[cursor : cursor + limit])
        cursor += limit
    return batches


def split_schedule_into_batches(
    schedule: list[str],
    *,
    batch_count: int,
) -> list[list[str]]:
    """按时间轴均分为固定段数（如前段/后段各一半）。"""
    if not schedule:
        return []
    count = max(1, int(batch_count))
    if count == 1:
        return [schedule]
    base_size = len(schedule) // count
    remainder = len(schedule) % count
    batches: list[list[str]] = []
    cursor = 0
    for index in range(count):
        size = base_size + (1 if index < remainder else 0)
        if size <= 0:
            continue
        batches.append(schedule[cursor : cursor + size])
        cursor += size
    return batches or [schedule]


def resolve_schedule_batches(
    schedule: list[str],
    *,
    settings: dict[str, Any],
    batch_plan: GridBatchPlan,
    force_one_shot: bool,
) -> list[list[str]]:
    if force_one_shot:
        return [schedule]
    batch_count = int(settings.get("batch_count") or WHOLE_GRID_DEFAULT_BATCH_COUNT)
    if batch_count > 1:
        return split_schedule_into_batches(schedule, batch_count=batch_count)
    return split_schedule_for_batches(
        schedule,
        max_per_batch=batch_plan.max_segments_per_api_call,
    )


def parse_whole_grid_payload(raw_text: str) -> dict[str, Any]:
    cleaned = _clean_json_output(raw_text)
    if not cleaned:
        raise ValueError("模型返回为空")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(f"无法解析 JSON: {exc}") from exc
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("模型返回不是 JSON 对象")
    return normalize_whole_grid_payload(payload)


def normalize_whole_grid_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "overall_summary": str(payload.get("overall_summary") or "").strip(),
        "key_conflict": str(payload.get("key_conflict") or "").strip(),
        "grid_segments": [],
    }
    segments = payload.get("grid_segments")
    if not isinstance(segments, list):
        segments = payload.get("episodic_segments")
    if isinstance(segments, list):
        for index, item in enumerate(segments, start=1):
            if not isinstance(item, dict):
                continue
            description = str(
                item.get("description")
                or item.get("key_events")
                or item.get("narration")
                or ""
            ).strip()
            if not description:
                continue
            chars = item.get("characters")
            if chars is None:
                chars = item.get("involved_characters")
            char_list = (
                [str(name).strip() for name in chars if str(name).strip()]
                if isinstance(chars, list)
                else []
            )
            try:
                segment_id = int(item.get("segment_id") or index)
            except (TypeError, ValueError):
                segment_id = index
            normalized["grid_segments"].append(
                {
                    "segment_id": segment_id,
                    "time_range": str(item.get("time_range") or "").strip().replace("—", "-"),
                    "description": description,
                    "characters": char_list,
                    "dialogue": str(item.get("dialogue") or item.get("quote") or "").strip(),
                }
            )
    return normalized


def enforce_grid_schedule(
    segments: list[dict[str, Any]],
    expected_ranges: list[str],
) -> list[dict[str, Any]]:
    by_range: dict[str, dict[str, Any]] = {}
    for item in segments:
        key = str(item.get("time_range") or "").strip().replace("—", "-")
        if key:
            by_range[key] = item
    enforced: list[dict[str, Any]] = []
    for index, time_range in enumerate(expected_ranges, start=1):
        matched = by_range.get(time_range)
        if matched:
            description = str(matched.get("description") or "").strip() or "画面无明显变化"
            chars = list(matched.get("characters") or [])
            dialogue = str(matched.get("dialogue") or "").strip()
        else:
            description = "（本窗模型未返回，待补全）"
            chars = []
            dialogue = ""
        enforced.append(
            {
                "segment_id": index,
                "time_range": time_range,
                "description": description,
                "characters": chars,
                "dialogue": dialogue,
            }
        )
    return enforced


def detect_grid_timeline_warnings(
    segments: list[dict[str, Any]],
    *,
    min_duplicate_gap_seconds: int = 120,
) -> list[str]:
    """检测时间轴错位：同一句台词/描述出现在相距较远的窗口。"""
    warnings: list[str] = []
    dialogue_hits: dict[str, list[int]] = {}
    filler_streak = 0
    max_filler_streak = 0

    for item in segments:
        if not isinstance(item, dict):
            continue
        start, _end = _parse_time_range_bounds(str(item.get("time_range") or ""))
        dialogue = str(item.get("dialogue") or "").strip()
        if len(dialogue) >= 8:
            dialogue_hits.setdefault(dialogue, []).append(start)
        desc = str(item.get("description") or "").strip()
        if any(marker in desc for marker in _CONTINUATION_MARKERS) or desc.startswith("（本窗"):
            filler_streak += 1
            max_filler_streak = max(max_filler_streak, filler_streak)
        else:
            filler_streak = 0

    for dialogue, starts in dialogue_hits.items():
        if len(starts) < 2:
            continue
        ordered = sorted(starts)
        for index in range(1, len(ordered)):
            gap = ordered[index] - ordered[index - 1]
            if gap >= min_duplicate_gap_seconds:
                quote = dialogue if len(dialogue) <= 24 else f"{dialogue[:24]}…"
                warnings.append(
                    f"台词重复疑似时间轴错位：「{quote}」在 "
                    f"{_format_timestamp(ordered[index - 1])} 与 "
                    f"{_format_timestamp(ordered[index])} 再次出现"
                )
                break

    if max_filler_streak >= 8:
        warnings.append(
            f"连续 {max_filler_streak} 格为占位/延续描述，可能整段批次未对齐时间窗"
        )
    return warnings


def merge_whole_grid_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    summary_parts: list[str] = []
    conflict_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    for partial in partials:
        summary = str(partial.get("overall_summary") or "").strip()
        if summary:
            summary_parts.append(summary)
        conflict = str(partial.get("key_conflict") or "").strip()
        if conflict:
            conflict_parts.append(conflict)
        for item in partial.get("grid_segments") or []:
            if isinstance(item, dict) and item.get("time_range"):
                segments.append(item)
    segments.sort(key=lambda item: _parse_time_range_bounds(str(item.get("time_range") or ""))[0])
    for index, item in enumerate(segments, start=1):
        item["segment_id"] = index
    return {
        "overall_summary": " ".join(summary_parts)[:800] if summary_parts else "",
        "key_conflict": "；".join(conflict_parts)[:500] if conflict_parts else "",
        "grid_segments": segments,
    }


def load_video_whole_grid_artifact(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"整片网格分析 JSON 不存在: {path}")
    with open(path, encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"整片网格分析 JSON 格式无效: {path}")
    if payload.get("field_comments"):
        payload = {key: value for key, value in payload.items() if key != "field_comments"}
    return payload


class VideoWholeGridAnalysisService:
    """整片压缩上传 + 固定时间网格分析。"""

    @staticmethod
    def _resolve_model_settings(
        *,
        vision_model_name: str | None = None,
        vision_api_key: str | None = None,
        vision_base_url: str | None = None,
    ) -> tuple[str, str, str]:
        return VideoEpisodeAnalysisService._resolve_model_settings(
            vision_model_name=vision_model_name,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
        )

    @staticmethod
    def _transcode_whole_video(
        video_path: str,
        *,
        output_path: str,
        profile_name: str,
        start_seconds: float = 0.0,
        duration_seconds: float | None = None,
    ) -> None:
        import subprocess

        profile = _get_grid_transcode_profile(profile_name)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        width = int(profile.get("width") or 480)
        fps = int(profile.get("fps") or 10)
        crf = str(profile.get("crf") or 32)
        preset = str(profile.get("preset") or "veryfast")
        no_audio = bool(profile.get("no_audio")) or str(profile.get("audio_bitrate") or "") == "0"
        cmd = ["ffmpeg", "-y"]
        if start_seconds > 0:
            cmd.extend(["-ss", str(start_seconds)])
        cmd.extend(["-i", video_path])
        if duration_seconds and duration_seconds > 0:
            cmd.extend(["-t", str(duration_seconds)])
        cmd.extend(
            [
                "-vf",
                f"scale={width}:-2,fps={fps}",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                crf,
            ]
        )
        if no_audio:
            cmd.append("-an")
        else:
            cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    str(profile.get("audio_bitrate") or "32k"),
                ]
            )
        cmd.extend(["-movflags", "+faststart", output_path])
        subprocess.run(cmd, check=True, capture_output=True)

    @classmethod
    def _ensure_single_upload_video(
        cls,
        video_path: str,
        *,
        work_dir: str,
        max_upload_mb: float,
        preferred_profile: str,
    ) -> tuple[str, str, float, str | None]:
        """压缩整片为单个上传文件；始终返回整片路径（不按时间切段）。"""
        os.makedirs(work_dir, exist_ok=True)
        chain = [preferred_profile] + [
            name for name in WHOLE_GRID_TRANSCODE_PROFILE_CHAIN if name != preferred_profile
        ]
        seen: set[str] = set()
        ordered_profiles: list[str] = []
        for name in chain:
            if name not in seen and name in UPLOAD_TRANSCODE_PROFILES:
                ordered_profiles.append(name)
                seen.add(name)

        best_path = ""
        best_profile = ordered_profiles[-1]
        best_size_mb = 0.0
        for profile_name in ordered_profiles:
            output_path = os.path.join(work_dir, f"whole_grid_{profile_name}.mp4")
            if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
                cls._transcode_whole_video(
                    video_path,
                    output_path=output_path,
                    profile_name=profile_name,
                )
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"网格快扫整片压缩 {profile_name}: {size_mb:.2f}MB")
            best_path = output_path
            best_profile = profile_name
            best_size_mb = size_mb
            if size_mb <= max_upload_mb:
                return output_path, profile_name, size_mb, None

        warning = (
            f"整片压缩后仍 {best_size_mb:.1f}MB > 限制 {max_upload_mb:.1f}MB，"
            "将仍按「整片一次上传」发送（网关可能拒收，可调大 max_upload_mb）"
        )
        logger.warning(warning)
        return best_path, best_profile, best_size_mb, warning

    async def _analyze_upload_once(
        self,
        *,
        provider: Any,
        upload_path: str,
        prompt: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        part_offset_seconds: float,
        part_duration_seconds: float,
        reference_image_paths: list[str] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        timeout = float(timeout_override or _WHOLE_GRID_TIMEOUT)
        for attempt in range(1, _MAX_API_RETRIES + 1):
            try:
                raw = await provider.analyze_video(
                    upload_path,
                    prompt,
                    api_key=api_key,
                    api_base=base_url,
                    timeout_override=timeout,
                    max_tokens=max_tokens,
                    reference_image_paths=reference_image_paths,
                    scene_time_range=(
                        f"{_format_timestamp(part_offset_seconds)}-"
                        f"{_format_timestamp(part_offset_seconds + part_duration_seconds)}"
                        if part_duration_seconds > 0
                        else ""
                    ),
                )
                return parse_whole_grid_payload(raw)
            except Exception as exc:
                last_error = exc
                if _is_output_length_error(exc):
                    raise
                logger.warning(f"网格快扫 API 失败 ({attempt}/{_MAX_API_RETRIES}): {exc}")
        raise last_error or RuntimeError("网格快扫 API 调用失败")

    async def analyze_episode(
        self,
        *,
        video_path: str,
        drama_title: str = "",
        drama_id: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        vision_model_name: str | None = None,
        vision_api_key: str | None = None,
        vision_base_url: str | None = None,
        grid_interval_seconds: int | None = None,
        force_one_shot: bool | None = None,
        max_upload_mb: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        output_path: str | None = None,
        plot_reference: str = "",
    ) -> dict[str, Any]:
        progress = progress_callback or (lambda _p, _m: None)
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        settings = get_video_whole_grid_settings()
        if max_upload_mb is None:
            max_upload_mb = float(settings.get("max_upload_mb", 24.0))
        requested_interval = (
            int(grid_interval_seconds)
            if grid_interval_seconds is not None
            else int(settings.get("grid_interval_seconds", WHOLE_GRID_DEFAULT_INTERVAL))
        )
        requested_interval = max(
            WHOLE_GRID_MIN_INTERVAL,
            min(WHOLE_GRID_MAX_INTERVAL, requested_interval),
        )
        if force_one_shot is None:
            force_one_shot = bool(settings.get("force_one_shot", False))
        preferred_profile = str(settings.get("upload_transcode_profile") or "grid_whole").strip()

        model_name, api_key, base_url = self._resolve_model_settings(
            vision_model_name=vision_model_name,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
        )
        batch_plan = resolve_grid_batch_plan(model_name, settings)

        video_duration_seconds = _probe_duration_seconds(video_path)
        grid_interval_effective = requested_interval
        grid_interval_auto_adjusted = False
        if force_one_shot:
            grid_interval_effective, _segment_count, grid_interval_auto_adjusted = (
                resolve_one_shot_grid_interval(
                    video_duration_seconds,
                    requested_interval=requested_interval,
                )
            )
            max_per_batch = _segment_count
            logger.info(
                f"整片网格快扫(单次生成) {model_name} → {describe_llm_route(model_name, role='vision')} · "
                f"{batch_plan.profile_label} · 格距 {requested_interval}s→{grid_interval_effective}s · "
                f"约 {_segment_count} 格 · API×1"
            )
        else:
            batch_count = int(settings.get("batch_count") or WHOLE_GRID_DEFAULT_BATCH_COUNT)
            max_per_batch = batch_plan.max_segments_per_api_call
            batch_mode = (
                f"{batch_count} 段均分"
                if batch_count > 1
                else f"{max_per_batch} 格/批"
            )
            logger.info(
                f"整片网格快扫(分批) {model_name} → {describe_llm_route(model_name, role='vision')} · "
                f"{batch_plan.profile_label} · {batch_mode}"
            )

        full_schedule = build_grid_schedule(
            video_duration_seconds,
            interval_seconds=grid_interval_effective,
            max_interval_seconds=(
                WHOLE_GRID_ONE_SHOT_MAX_INTERVAL if force_one_shot else WHOLE_GRID_MAX_INTERVAL
            ),
        )
        if not full_schedule:
            raise ValueError("无法生成时间网格")

        save_path = output_path or default_video_whole_grid_analysis_path(video_path)
        work_dir = os.path.join(
            utils.storage_dir(),
            "temp",
            "video_whole_grid_upload",
            sanitize_video_stem(video_path),
        )
        ref_count = len(character_references or [])
        interval_label = (
            f"{requested_interval}s→{grid_interval_effective}s（自动加粗）"
            if grid_interval_auto_adjusted
            else f"{grid_interval_effective}s"
        )
        mode_label = "单次生成" if force_one_shot else "分批生成"
        progress(
            5,
            f"{mode_label} · 网格 {interval_label} · 共 {len(full_schedule)} 格 · 模型 {model_name}"
            + (f" · 头像参照 {ref_count} 人" if ref_count else ""),
        )

        progress(10, "正在压缩整片视频（单次上传模式）...")
        upload_path, used_profile, master_size_mb, upload_warning = self._ensure_single_upload_video(
            video_path,
            work_dir=work_dir,
            max_upload_mb=max_upload_mb,
            preferred_profile=preferred_profile,
        )
        size_hint = f"{master_size_mb:.1f}MB"
        if upload_warning:
            progress(18, f"压缩完成 · {used_profile} · {size_hint} · 整片一次上传（超限警告）")
        else:
            progress(18, f"压缩完成 · {used_profile} · {size_hint} · 整片一次上传")
        if force_one_shot:
            schedule_batches = [full_schedule]
            progress(
                22,
                f"整片上传 {size_hint} · **API 1 次** 生成 {len(full_schedule)} 格"
                f"（max_tokens≤{batch_plan.max_output_tokens}）",
            )
        else:
            schedule_batches = resolve_schedule_batches(
                full_schedule,
                settings=settings,
                batch_plan=batch_plan,
                force_one_shot=False,
            )
            batch_count = int(settings.get("batch_count") or WHOLE_GRID_DEFAULT_BATCH_COUNT)
            batch_hint = (
                f"均分 {batch_count} 段"
                if batch_count > 1
                else f"每批≤{max_per_batch} 格"
            )
            progress(
                22,
                f"整片上传 {size_hint} · 输出 {len(schedule_batches)} 批（{batch_hint}）",
            )

        from app.services.llm.openai_compatible_provider import OpenAICompatibleVisionProvider

        provider = OpenAICompatibleVisionProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )

        character_names = [
            str(item.get("name") or "").strip()
            for item in (character_references or [])
            if isinstance(item, dict) and item.get("name")
        ]
        resolved_plot = normalize_plot_reference(
            plot_reference or "",
            max_chars=WHOLE_GRID_PLOT_REFERENCE_MAX_CHARS,
        )
        plot_prompt_section = build_plot_reference_prompt_section(
            resolved_plot,
            max_chars=WHOLE_GRID_PLOT_REFERENCE_MAX_CHARS,
        )
        partials: list[dict[str, Any]] = []
        require_summary = True
        one_shot_mode = bool(force_one_shot)
        one_shot_fallback = False
        batch_queue: list[list[str]] = list(schedule_batches)
        total_api_calls = len(batch_queue)
        done_calls = 0
        previous_batch_segments: list[dict[str, Any]] = []
        batch_index = 0

        while batch_index < len(batch_queue):
            batch_schedule = batch_queue[batch_index]
            reference_paths, naming_block = _prepare_chunk_reference_context(
                chunk_index=batch_index,
                drama_title=drama_title,
                character_references=character_references,
                relationship_diagram_path=relationship_diagram_path,
            )
            schedule_block = build_grid_schedule_prompt_block(
                batch_schedule,
                grid_interval_seconds=grid_interval_effective,
            )
            prompt = build_whole_grid_analysis_prompt(
                drama_title=drama_title,
                video_duration_seconds=video_duration_seconds,
                grid_interval_seconds=grid_interval_effective,
                segment_schedule_block=schedule_block,
                character_names=character_names if not naming_block else None,
                require_summary=require_summary,
                plot_reference_section=plot_prompt_section if batch_index == 0 else "",
            )
            if naming_block.strip():
                prompt = f"{naming_block.strip()}\n\n{prompt}"
            if one_shot_mode:
                prompt += (
                    "\n\n## 单次生成说明\n"
                    f"- **一次**输出全部 {len(batch_schedule)} 条 `grid_segments`，"
                    "禁止拆成多段或省略时间窗。\n"
                    "- 每条必须对应该 `time_range` 内的真实画面与对白。\n"
                )
            else:
                prompt += build_grid_batch_time_anchor_block(batch_schedule)
                tail_section = build_previous_grid_batch_tail_context(previous_batch_segments)
                if tail_section.strip():
                    prompt += f"\n\n{tail_section.strip()}"
                prompt += build_grid_batch_prompt_addon(
                    batch_index=batch_index,
                    batch_count=total_api_calls,
                    batch_size=len(batch_schedule),
                )
            prompt += (
                "\n\n## 上传说明\n"
                "本次上传为 **完整视频**（非切片）；"
                + (
                    "请按全部时间窗一次性输出 `grid_segments`。\n"
                    if one_shot_mode
                    else "请根据全片内容，**仅**输出本批列出的时间窗对应的 `grid_segments`。\n"
                )
                + "- `characters` 须对照上方定妆照/拼图与视频可见面孔逐脸确认后再写规范姓名。\n"
            )

            done_calls += 1
            call_progress = 30 + int(60 * done_calls / max(total_api_calls, 1))
            ref_label = (
                f" · 参照图 {len(reference_paths)} 张"
                if reference_paths
                else ""
            )
            api_label = (
                f"API 1/1 · {len(batch_schedule)} 格{ref_label}"
                if one_shot_mode
                else f"API {done_calls}/{total_api_calls} · 本批 {len(batch_schedule)} 格{ref_label}"
            )
            progress(call_progress, f"整片上传 · {api_label}")
            try:
                parsed = await self._analyze_upload_once(
                    provider=provider,
                    upload_path=upload_path,
                    prompt=prompt,
                    api_key=api_key,
                    base_url=base_url,
                    max_tokens=estimate_grid_batch_max_tokens(
                        len(batch_schedule),
                        batch_plan,
                        one_shot=one_shot_mode,
                    ),
                    part_offset_seconds=0.0,
                    part_duration_seconds=video_duration_seconds,
                    reference_image_paths=reference_paths,
                    timeout_override=(
                        _WHOLE_GRID_ONE_SHOT_TIMEOUT if one_shot_mode else _WHOLE_GRID_TIMEOUT
                    ),
                )
            except Exception as exc:
                if (
                    one_shot_mode
                    and _is_output_length_error(exc)
                    and len(full_schedule) > 1
                ):
                    fallback_batch_count = int(
                        settings.get("batch_count") or WHOLE_GRID_DEFAULT_BATCH_COUNT
                    )
                    if fallback_batch_count > 1:
                        fallback_hint = f"均分 {fallback_batch_count} 段"
                        batch_queue = split_schedule_into_batches(
                            full_schedule,
                            batch_count=fallback_batch_count,
                        )
                    else:
                        fallback_batch_size = batch_plan.max_segments_per_api_call
                        fallback_hint = f"每批≤{fallback_batch_size} 格"
                        batch_queue = split_schedule_for_batches(
                            full_schedule,
                            max_per_batch=fallback_batch_size,
                        )
                    logger.warning(
                        f"单次生成输出超限（{len(full_schedule)} 格），"
                        f"自动切换分批（{fallback_hint}）: {exc}"
                    )
                    progress(
                        24,
                        f"单次输出超限 · 自动切换分批 · 共 {len(full_schedule)} 格 → {fallback_hint}",
                    )
                    one_shot_mode = False
                    one_shot_fallback = True
                    total_api_calls = len(batch_queue)
                    batch_index = 0
                    partials.clear()
                    require_summary = True
                    previous_batch_segments = []
                    done_calls = 0
                    continue
                raise
            enforced = enforce_grid_schedule(
                parsed.get("grid_segments") or [],
                batch_schedule,
            )
            parsed["grid_segments"] = enforced
            partials.append(parsed)
            previous_batch_segments = enforced
            if require_summary:
                require_summary = False
            batch_index += 1

        if not partials:
            raise ValueError("网格快扫未获得任何有效结果")

        progress(92, "正在合并网格结果...")
        merged = merge_whole_grid_partials(partials)
        merged["grid_segments"] = enforce_grid_schedule(
            merged.get("grid_segments") or [],
            full_schedule,
        )
        coverage_warnings = detect_grid_timeline_warnings(merged.get("grid_segments") or [])
        if len(normalize_plot_reference(plot_reference or "")) > WHOLE_GRID_PLOT_REFERENCE_MAX_CHARS:
            coverage_warnings.insert(
                0,
                f"剧情参考过长（>{WHOLE_GRID_PLOT_REFERENCE_MAX_CHARS} 字），已截断注入；"
                "完整关系网不宜整段粘贴，建议只写本集前情 3–5 句",
            )

        artifact = {
            "artifact_version": WHOLE_GRID_ARTIFACT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "video_path": os.path.abspath(video_path),
            "video_duration_seconds": round(video_duration_seconds, 3),
            "drama_title": drama_title,
            "drama_id": (drama_id or drama_title).strip(),
            "plot_reference": resolved_plot,
            "plot_reference_truncated": len((plot_reference or "").strip()) > len(resolved_plot),
            "coverage_warnings": coverage_warnings,
            "character_references": [
                {
                    "name": str(item.get("name") or "").strip(),
                    "path": str(item.get("path") or "").strip(),
                }
                for item in (character_references or [])
                if isinstance(item, dict) and item.get("name")
            ],
            "relationship_diagram_path": relationship_diagram_path or "",
            "vision_model_name": model_name,
            "analysis_mode": "whole_video_grid",
            "analysis_status": "complete",
            "one_shot": bool(force_one_shot) and not one_shot_fallback,
            "one_shot_requested": bool(force_one_shot),
            "one_shot_fallback": one_shot_fallback,
            "batch_count": int(settings.get("batch_count") or WHOLE_GRID_DEFAULT_BATCH_COUNT),
            "grid_interval_seconds": grid_interval_effective,
            "grid_interval_requested": requested_interval,
            "grid_interval_auto_adjusted": grid_interval_auto_adjusted,
            "grid_segment_count": len(merged.get("grid_segments") or []),
            "upload_profile": used_profile,
            "upload_size_mb": round(master_size_mb, 2),
            "upload_part_count": 1,
            "api_call_count": done_calls,
            "single_upload": True,
            "upload_warning": upload_warning or "",
            "batch_profile": batch_plan.profile_label,
            "max_segments_per_api_call": batch_plan.max_segments_per_api_call,
            "max_output_tokens": batch_plan.max_output_tokens,
            **merged,
        }

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(artifact, fp, ensure_ascii=False, indent=2)

        progress(
            100,
            f"完成 · {artifact['grid_segment_count']} 格 · API {done_calls} 次 · 已保存",
        )
        logger.info(f"整片网格快扫已保存: {save_path}")
        artifact["output_path"] = save_path
        return artifact
