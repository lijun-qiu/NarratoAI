#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频单集剧情分析：直接传 mp4 给视觉模型，输出结构化 JSON。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.config.llm_gateway_router import describe_llm_route, resolve_llm_credentials
from app.services.documentary.frame_analysis_pairing import analysis_artifact_dir, sanitize_video_stem
from app.services.documentary.frame_reference_images import (
    prepare_reference_prefix_images,
    resolve_reference_collage_mode,
)
from app.services.drama_character_registry import resolve_media_path
from app.services.prompts.documentary.video_episode_analysis import (
    build_reference_carryover_naming_block,
    build_video_episode_analysis_prompt,
    build_video_episode_chunk_prompt,
    build_video_episode_vision_reference_prompt_section,
)
from app.utils import utils

VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION = "documentary-video-episode-analysis-v6"
SEGMENT_INTERVAL_SECONDS = 10
_DEFAULT_MAX_UPLOAD_MB = 12.0
_DEFAULT_CHUNK_SECONDS = 300.0
_MIN_CHUNK_SECONDS = 60.0
_VIDEO_ANALYSIS_TIMEOUT = 900.0
_MAX_CHUNK_RETRIES = 3

VIDEO_EPISODE_FIELD_COMMENTS: dict[str, str] = {
    "_readme": "JSON 不支持 // 注释；本 field_comments 对象置于文件最前，说明各字段含义，不参与业务逻辑。",
    "artifact_version": "本 JSON 结构版本号",
    "generated_at": "生成时间（ISO8601）",
    "video_path": "源视频绝对路径",
    "video_duration_seconds": "源视频总时长（秒）",
    "drama_title": "剧名/单集所属作品",
    "drama_id": "剧目 ID（与抽帧分析人物库一致）",
    "character_references": "分析时参照的人物头像列表（name + path）",
    "relationship_diagram_path": "人物关系图路径（若有）",
    "vision_model_name": "视觉模型名称",
    "analysis_mode": "分析模式（direct_video=整片直传）",
    "analysis_status": "分析状态：complete=全部段完成；incomplete=部分段失败可补全",
    "chunk_count": "上传分段总数（长片按约5分钟/段切分）",
    "completed_chunk_count": "已成功完成的分段数",
    "failed_chunk_indices": "失败分段索引列表（0 起），可点击补全重试",
    "segment_interval_seconds": "情节片段固定时间窗长度（秒），当前为 10",
    "segment_split_policy": "切分策略标识（fixed_10s=固定10秒一格）",
    "episodic_segment_count": "episodic_segments 条数",
    "coverage_warnings": "时间窗/片段约束校验告警（非空表示模型输出曾偏离固定格子）",
    "overall_summary": "本集/本段核心剧情概括（约200字内）",
    "key_conflict": "本集/本段最核心的矛盾冲突（一句话）",
    "episodic_segments": "固定10秒情节片段列表（全片时间轴绝对时间）",
    "episodic_segments.segment_id": "片段序号，从 1 起",
    "episodic_segments.title": "片段标题（4-6字）",
    "episodic_segments.time_range": "片内绝对时间窗，格式 HH:MM:SS-HH:MM:SS，须与固定格子一致",
    "episodic_segments.key_events": "该10秒内关键事件（一句话）",
    "episodic_segments.narration": "纪录片旁白（第三人称，20-50字，可用于后期配音）",
    "episodic_segments.environment_description": "场景环境（地点、光线、氛围、布景等，15-40字）",
    "episodic_segments.involved_characters": "该片段涉及人物（规范姓名或「剧中未明确交代」）",
    "important_dialogues": "重要台词列表",
    "important_dialogues.speaker": "说话人（须遵守人物命名规则）",
    "important_dialogues.timestamp": "台词出现的片内绝对时间戳（HH:MM:SS）",
    "important_dialogues.quote": "视频中实际听到的原话",
    "important_dialogues.significance": "该台词的重要性/揭示的信息",
    "cliffhangers_or_foreshadowing": "悬念或伏笔列表",
    "cliffhangers_or_foreshadowing.description": "悬念/伏笔描述",
    "cliffhangers_or_foreshadowing.possible_interpretation": "可能对后续剧情的影响解读",
}


def _prepend_field_comments(payload: dict[str, Any], comments: dict[str, str]) -> dict[str, Any]:
    """JSON 不支持注释，用 field_comments 作为首字段说明各键含义。"""
    return {"field_comments": comments, **payload}


def default_video_episode_analysis_path(video_path: str) -> str:
    stem = sanitize_video_stem(video_path)
    return os.path.join(analysis_artifact_dir(), f"{stem}_video_episode_analysis.json")


def default_checkpoint_path(output_path: str) -> str:
    return f"{output_path}.checkpoint.json"


def load_video_episode_checkpoint(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"无法读取整片视频分析检查点: {path} ({exc})")
        return None
    return payload if isinstance(payload, dict) else None


def save_video_episode_checkpoint(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _chunks_meta(chunks: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "offset_seconds": round(float(chunk.get("offset_seconds") or 0), 3),
            "duration_seconds": round(float(chunk.get("duration_seconds") or 0), 3),
        }
        for chunk in chunks
    ]


def is_checkpoint_compatible(
    checkpoint: dict[str, Any],
    *,
    video_path: str,
    video_duration_seconds: float,
    chunks: list[dict[str, Any]],
) -> bool:
    if not checkpoint:
        return False
    if checkpoint.get("artifact_version") != VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION:
        return False
    if os.path.abspath(str(checkpoint.get("video_path") or "")) != os.path.abspath(video_path):
        return False
    if abs(float(checkpoint.get("video_duration_seconds") or 0) - video_duration_seconds) > 1.0:
        return False
    if int(checkpoint.get("total_chunks") or 0) != len(chunks):
        return False
    saved_meta = checkpoint.get("chunks_meta") or []
    current_meta = _chunks_meta(chunks)
    if len(saved_meta) != len(current_meta):
        return False
    for saved, current in zip(saved_meta, current_meta):
        if not isinstance(saved, dict):
            return False
        if abs(float(saved.get("offset_seconds") or 0) - current["offset_seconds"]) > 0.5:
            return False
        if abs(float(saved.get("duration_seconds") or 0) - current["duration_seconds"]) > 0.5:
            return False
    return True


def summarize_checkpoint_progress(
    checkpoint: dict[str, Any] | None,
    total_chunks: int,
) -> dict[str, int]:
    chunk_results = (checkpoint or {}).get("chunk_results") or {}
    completed = failed = pending = 0
    for index in range(total_chunks):
        entry = chunk_results.get(str(index)) or chunk_results.get(index)
        if not isinstance(entry, dict):
            pending += 1
            continue
        status = str(entry.get("status") or "").strip()
        if status == "completed" and entry.get("partial"):
            completed += 1
        elif status == "failed":
            failed += 1
        else:
            pending += 1
    return {
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "total": total_chunks,
    }


def validate_chunk_partial(
    partial: dict[str, Any],
    *,
    chunk_offset_seconds: float,
    chunk_duration_seconds: float,
    chunk_index: int,
    total_chunks: int,
) -> list[str]:
    """校验单段模型输出是否满足固定时间窗约束，不符合则触发重试。"""
    issues: list[str] = []
    expected_schedule = build_fixed_segment_schedule(
        chunk_duration_seconds,
        start_offset_seconds=chunk_offset_seconds,
    )
    expected_count = len(expected_schedule)
    segments = partial.get("episodic_segments") or []
    filled = sum(
        1
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("key_events") or "").strip()
    )
    narration_filled = sum(
        1
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("narration") or "").strip()
    )
    environment_filled = sum(
        1
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("environment_description") or "").strip()
    )
    min_required = max(1, int(expected_count * 0.75))
    if filled < min_required:
        issues.append(f"有效片段 {filled}/{expected_count} 不足（至少 {min_required}）")
    if narration_filled < min_required:
        issues.append(f"旁白 narration {narration_filled}/{expected_count} 不足（至少 {min_required}）")
    if environment_filled < min_required:
        issues.append(
            f"环境描述 environment_description {environment_filled}/{expected_count} 不足（至少 {min_required}）"
        )

    if chunk_index == 0 and not str(partial.get("overall_summary") or "").strip():
        issues.append("首段缺少 overall_summary")

    enforced = enforce_episodic_segment_schedule(segments, expected_schedule)
    chunk_warnings = validate_episodic_segments(
        enforced,
        video_duration_seconds=chunk_offset_seconds + chunk_duration_seconds,
        expected_time_ranges=expected_schedule,
    )
    for warning in chunk_warnings:
        if any(
            keyword in warning
            for keyword in ("数量", "空隙", "少于", "不一致", "缺失", "缺少")
        ):
            issues.append(warning)
        elif "应为" in warning and "时长" in warning:
            issues.append(warning)

    if total_chunks == 1 and not issues and not filled:
        issues.append("未返回任何情节片段")

    return issues


def _clean_json_output(output: str) -> str:
    text = (output or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _parse_timestamp_seconds(value: str) -> int:
    cleaned = str(value or "").strip()
    if not cleaned:
        return 0
    cleaned = cleaned.split("-", 1)[0].strip()
    parts = cleaned.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return int(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]))
        if len(parts) == 2:
            return int(float(parts[0]) * 60 + float(parts[1]))
        return int(float(parts[0]))
    except (TypeError, ValueError):
        return 0


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_video_episode_analysis_payload(raw_text: str) -> dict[str, Any]:
    """解析模型返回的 JSON。"""
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
    return normalize_video_episode_analysis_payload(payload)


def _parse_time_range_start_seconds(value: str) -> int:
    start, _end = _parse_time_range_bounds(value)
    return start


def _parse_time_range_bounds(value: str) -> tuple[int, int]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return 0, 0
    parts = re.split(r"[-—]", cleaned, maxsplit=1)
    start = _parse_timestamp_seconds(parts[0].strip())
    end = _parse_timestamp_seconds(parts[1].strip() if len(parts) > 1 else parts[0].strip())
    if end < start:
        end = start
    return start, end


def _prepare_chunk_reference_context(
    *,
    chunk_index: int,
    drama_title: str,
    character_references: list[dict[str, str]] | None,
    relationship_diagram_path: str,
) -> tuple[list[str], str]:
    """首段附带头像/关系图；后续段仅 prompt 沿用规则。"""
    refs = [
        {"name": str(item.get("name") or "").strip(), "path": str(item.get("path") or "").strip()}
        for item in (character_references or [])
        if isinstance(item, dict) and item.get("name") and item.get("path")
    ]
    rel_path = resolve_media_path(relationship_diagram_path)
    if not refs and not rel_path:
        return [], ""

    settings = {
        "frame_reference_use_collage": True,
        "frame_reference_token_saver": True,
        "frame_reference_max_edge": 384,
        "frame_reference_individual_max_heads": 6,
        "default_video_theme": drama_title,
    }
    head_paths = [item["path"] for item in refs if os.path.isfile(item["path"])]
    use_collage = resolve_reference_collage_mode(settings, head_count=len(head_paths))

    if chunk_index == 0:
        prefix_paths, _carryover = prepare_reference_prefix_images(
            batch_index=0,
            relationship_diagram_path=rel_path,
            character_references=refs,
            settings=settings,
        )
        naming_block = build_video_episode_vision_reference_prompt_section(
            drama_label=drama_title,
            character_references=refs,
            relationship_diagram_attached=bool(rel_path),
            reference_image_count=len(prefix_paths),
            character_collage=use_collage and len(head_paths) >= 2,
        )
        return prefix_paths, naming_block

    naming_block = build_reference_carryover_naming_block(
        drama_label=drama_title,
        character_references=refs,
        relationship_diagram_attached=bool(rel_path),
    )
    return [], naming_block


def build_fixed_segment_schedule(
    duration_seconds: float,
    *,
    interval_seconds: int = SEGMENT_INTERVAL_SECONDS,
    start_offset_seconds: float = 0.0,
) -> list[str]:
    """生成固定长度时间窗（末段不足 interval 时保留余量）。"""
    duration = max(0.0, float(duration_seconds))
    start_base = max(0.0, float(start_offset_seconds))
    end_limit = start_base + duration
    ranges: list[str] = []
    cursor = start_base
    while cursor < end_limit - 0.01:
        seg_end = min(cursor + interval_seconds, end_limit)
        ranges.append(f"{_format_timestamp(cursor)}-{_format_timestamp(seg_end)}")
        cursor = seg_end
    return ranges


def build_segment_schedule_prompt_block(fixed_time_ranges: list[str]) -> str:
    if not fixed_time_ranges:
        return ""
    lines = [
        "## 固定时间窗口（硬性要求）",
        (
            f"你必须输出 **恰好 {len(fixed_time_ranges)} 条** `episodic_segments`，"
            "`time_range` 必须与下列窗口 **完全一致**（字符级一致，仅用 `-` 连接）："
        ),
    ]
    for index, time_range in enumerate(fixed_time_ranges, start=1):
        lines.append(f"{index}. `{time_range}`")
    lines.extend(
        [
            f"- 除最后一条外，每条窗口长度 **必须恰好 {SEGMENT_INTERVAL_SECONDS} 秒**；禁止合并、禁止改短或拉长。",
            "- 每条都必须填写 4-6 字 `title`、该窗口内 `key_events`、`narration`（纪录片旁白）、"
            "`environment_description`（场景环境）、`involved_characters`。",
            f"- 某 {SEGMENT_INTERVAL_SECONDS} 秒若无明显新事件，`key_events` 写「画面/对话延续上段」，"
            "`narration` / `environment_description` 可写「延续上段」，**不得跳过任何窗口**。",
        ]
    )
    return "\n".join(lines)


def enforce_episodic_segment_schedule(
    segments: list[dict[str, Any]],
    fixed_time_ranges: list[str],
) -> list[dict[str, Any]]:
    """将模型输出对齐到固定时间窗。"""
    if not fixed_time_ranges:
        return segments

    window_to_segment: list[int | None] = []
    for time_range in fixed_time_ranges:
        exp_start, exp_end = _parse_time_range_bounds(time_range)
        matched_index: int | None = None
        for seg_index, segment in enumerate(segments):
            seg_start, seg_end = _parse_time_range_bounds(segment.get("time_range", ""))
            if seg_start < exp_end and seg_end > exp_start:
                matched_index = seg_index
                break
        window_to_segment.append(matched_index)

    segment_first_window: dict[int, int] = {}
    for window_index, seg_index in enumerate(window_to_segment):
        if seg_index is None or seg_index in segment_first_window:
            continue
        segment_first_window[seg_index] = window_index

    enforced: list[dict[str, Any]] = []
    for index, time_range in enumerate(fixed_time_ranges, start=1):
        window_index = index - 1
        seg_index = window_to_segment[window_index]
        if seg_index is not None:
            segment = segments[seg_index]
            title = str(segment.get("title") or "").strip() or f"片段{index:02d}"
            characters = [
                str(name).strip()
                for name in (segment.get("involved_characters") or [])
                if str(name).strip()
            ]
            base_events = str(segment.get("key_events") or "").strip()
            base_narration = str(segment.get("narration") or "").strip()
            base_environment = str(segment.get("environment_description") or "").strip()
            if window_index == segment_first_window.get(seg_index):
                key_events = base_events or "画面/对话延续上段"
                narration = base_narration or key_events
                environment_description = base_environment or "环境延续上段"
            else:
                key_events = "（承接上段）"
                narration = "（承接上段）"
                environment_description = (
                    enforced[-1].get("environment_description") or "环境延续上段"
                    if enforced
                    else "环境延续上段"
                )
        else:
            title = f"片段{index:02d}"
            key_events = "画面/对话延续上段"
            narration = "画面延续上段"
            environment_description = (
                enforced[-1].get("environment_description") or "剧中未明确交代"
                if enforced
                else "剧中未明确交代"
            )
            characters = []
            if enforced:
                characters = list(enforced[-1].get("involved_characters") or [])

        enforced.append(
            {
                "segment_id": index,
                "title": title,
                "time_range": time_range,
                "key_events": key_events,
                "narration": narration,
                "environment_description": environment_description,
                "involved_characters": characters,
            }
        )

    return enforced


def validate_episodic_segments(
    segments: list[dict[str, Any]],
    *,
    video_duration_seconds: float = 0.0,
    expected_time_ranges: list[str] | None = None,
) -> list[str]:
    """检查片段是否符合固定时间窗，返回警告列表。"""
    warnings: list[str] = []
    video_end = max(0, int(video_duration_seconds))

    if expected_time_ranges:
        if len(segments) != len(expected_time_ranges):
            warnings.append(
                f"片段数量 {len(segments)} 与固定窗口 {len(expected_time_ranges)} 不一致"
            )
        for segment, expected in zip(segments, expected_time_ranges):
            actual = str(segment.get("time_range") or "").strip().replace("—", "-")
            if actual != expected:
                warnings.append(f"片段 time_range 应为 {expected}，实际为 {actual}")

    for index, segment in enumerate(segments):
        time_range = str(segment.get("time_range") or "").strip()
        if not time_range:
            warnings.append(f"片段 #{segment.get('segment_id')} 缺少 time_range")
            continue
        start, end = _parse_time_range_bounds(time_range)
        duration = end - start
        if duration <= 0:
            warnings.append(f"片段 {time_range} 时间范围无效")
            continue
        is_tail = video_end > 0 and end >= video_end - 1
        if not is_tail and duration != SEGMENT_INTERVAL_SECONDS:
            warnings.append(
                f"片段 {time_range} 时长 {duration}s，应为 {SEGMENT_INTERVAL_SECONDS}s"
            )
        elif is_tail and duration > SEGMENT_INTERVAL_SECONDS:
            warnings.append(
                f"末段 {time_range} 时长 {duration}s 超过 {SEGMENT_INTERVAL_SECONDS}s"
            )

    for index in range(1, len(segments)):
        prev_end = _parse_time_range_bounds(segments[index - 1].get("time_range", ""))[1]
        curr_start = _parse_time_range_bounds(segments[index].get("time_range", ""))[0]
        if curr_start > prev_end + 1:
            warnings.append(
                f"片段 {segments[index - 1].get('time_range')} 与 "
                f"{segments[index].get('time_range')} 之间存在时间空隙"
            )

    return warnings


def normalize_video_episode_analysis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """补齐字段并规范列表结构。"""
    normalized: dict[str, Any] = {
        "overall_summary": str(payload.get("overall_summary") or "").strip(),
        "key_conflict": str(payload.get("key_conflict") or "").strip(),
        "episodic_segments": [],
        "important_dialogues": [],
        "cliffhangers_or_foreshadowing": [],
    }

    segments = payload.get("episodic_segments")
    if isinstance(segments, list):
        for index, item in enumerate(segments, start=1):
            if not isinstance(item, dict):
                continue
            key_events = str(item.get("key_events") or "").strip()
            if not key_events:
                continue
            chars = item.get("involved_characters")
            char_list = (
                [str(name).strip() for name in chars if str(name).strip()]
                if isinstance(chars, list)
                else []
            )
            segment_id = item.get("segment_id")
            try:
                segment_id = int(segment_id)
            except (TypeError, ValueError):
                segment_id = index
            time_range = str(item.get("time_range") or "").strip().replace("—", "-")
            normalized["episodic_segments"].append(
                {
                    "segment_id": segment_id,
                    "title": str(item.get("title") or "").strip(),
                    "time_range": time_range,
                    "key_events": key_events,
                    "narration": str(item.get("narration") or "").strip() or key_events,
                    "environment_description": str(
                        item.get("environment_description") or ""
                    ).strip()
                    or "剧中未明确交代",
                    "involved_characters": char_list,
                }
            )

    dialogues = payload.get("important_dialogues")
    if isinstance(dialogues, list):
        for item in dialogues:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or item.get("text") or "").strip()
            if not quote:
                continue
            normalized["important_dialogues"].append(
                {
                    "speaker": str(item.get("speaker") or "").strip(),
                    "timestamp": str(item.get("timestamp") or "").strip(),
                    "quote": quote,
                    "significance": str(item.get("significance") or "").strip(),
                }
            )

    cliffhangers = payload.get("cliffhangers_or_foreshadowing")
    if isinstance(cliffhangers, list):
        for item in cliffhangers:
            if isinstance(item, dict):
                description = str(item.get("description") or "").strip()
                if not description:
                    continue
                normalized["cliffhangers_or_foreshadowing"].append(
                    {
                        "description": description,
                        "possible_interpretation": str(
                            item.get("possible_interpretation") or ""
                        ).strip(),
                    }
                )
            elif isinstance(item, str) and item.strip():
                normalized["cliffhangers_or_foreshadowing"].append(
                    {
                        "description": item.strip(),
                        "possible_interpretation": "",
                    }
                )

    normalized["episodic_segments"].sort(
        key=lambda item: _parse_time_range_start_seconds(item.get("time_range", ""))
    )
    for index, segment in enumerate(normalized["episodic_segments"], start=1):
        segment["segment_id"] = index
    normalized["important_dialogues"].sort(
        key=lambda item: _parse_timestamp_seconds(item.get("timestamp", ""))
    )
    return normalized


def merge_video_episode_partial_analyses(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """合并分段分析结果。"""
    if not partials:
        return normalize_video_episode_analysis_payload({})
    if len(partials) == 1:
        return normalize_video_episode_analysis_payload(partials[0])

    summaries = [str(item.get("overall_summary") or "").strip() for item in partials if item]
    merged_summary = " ".join(summary for summary in summaries if summary)[:400]

    key_conflicts = [str(item.get("key_conflict") or "").strip() for item in partials if item]
    merged_conflict = "；".join(conflict for conflict in key_conflicts if conflict)

    segments: list[dict[str, Any]] = []
    dialogues: list[dict[str, Any]] = []
    cliffhangers: list[dict[str, Any]] = []
    seen_cliff_descriptions: set[str] = set()

    for partial in partials:
        normalized = normalize_video_episode_analysis_payload(partial)
        segments.extend(normalized.get("episodic_segments") or [])
        dialogues.extend(normalized.get("important_dialogues") or [])
        for cliff in normalized.get("cliffhangers_or_foreshadowing") or []:
            description = str(cliff.get("description") or "").strip()
            if not description or description in seen_cliff_descriptions:
                continue
            seen_cliff_descriptions.add(description)
            cliffhangers.append(dict(cliff))

    return normalize_video_episode_analysis_payload(
        {
            "overall_summary": merged_summary,
            "key_conflict": merged_conflict,
            "episodic_segments": segments,
            "important_dialogues": dialogues,
            "cliffhangers_or_foreshadowing": cliffhangers,
        }
    )


def load_video_episode_analysis_artifact(path: str) -> dict[str, Any]:
    """读取整片视频分析 JSON（忽略 field_comments）。"""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"整片视频分析 JSON 不存在: {path}")
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"整片视频分析 JSON 格式无效: {path}")
    analysis = normalize_video_episode_analysis_payload(raw)
    for key in (
        "artifact_version",
        "video_path",
        "video_duration_seconds",
        "drama_title",
        "drama_id",
        "analysis_status",
        "segment_interval_seconds",
        "episodic_segment_count",
        "coverage_warnings",
    ):
        if key in raw:
            analysis[key] = raw.get(key)
    return analysis


def _sample_items_uniformly(items: list[Any], max_count: int) -> list[Any]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    if max_count == 1:
        return [items[0]]
    step = (len(items) - 1) / (max_count - 1)
    indices = sorted({int(round(index * step)) for index in range(max_count)})
    return [items[index] for index in indices]


def _format_episodic_segment_markdown(segment: dict[str, Any]) -> str:
    characters = "、".join(segment.get("involved_characters") or []) or "剧中未明确交代"
    return (
        f"- `{segment.get('time_range', '')}` **{segment.get('title', '')}** · "
        f"{segment.get('key_events', '')}\n"
        f"  - 旁白：{segment.get('narration', '')}\n"
        f"  - 环境：{segment.get('environment_description', '')}\n"
        f"  - 人物：{characters}"
    )


def build_video_episode_analysis_markdown(
    payload: dict[str, Any],
    *,
    max_chars: int = 30000,
    max_segments: int = 180,
) -> str:
    """将整片视频分析转为供蓝图/脚本参考的 Markdown。"""
    normalized = normalize_video_episode_analysis_payload(payload)
    lines: list[str] = ["# 整片视频分析摘要"]
    summary = str(normalized.get("overall_summary") or "").strip()
    if summary:
        lines.extend(["", "## 剧情概括", summary])
    conflict = str(normalized.get("key_conflict") or "").strip()
    if conflict:
        lines.extend(["", "## 核心冲突", conflict])

    segments = normalized.get("episodic_segments") or []
    sampled = _sample_items_uniformly(segments, max_segments)
    if segments:
        lines.extend(
            [
                "",
                "## 固定时间窗情节片段",
                (
                    f"共 {len(segments)} 条（每 {SEGMENT_INTERVAL_SECONDS} 秒一格）"
                    + (f"，以下均匀采样 {len(sampled)} 条" if len(sampled) < len(segments) else "")
                ),
            ]
        )
        lines.extend(_format_episodic_segment_markdown(segment) for segment in sampled)

    dialogues = normalized.get("important_dialogues") or []
    if dialogues:
        lines.extend(["", "## 重要台词"])
        for item in dialogues[:40]:
            lines.append(
                f"- `{item.get('timestamp', '')}` **{item.get('speaker', '')}**："
                f"「{item.get('quote', '')}」"
                + (f"（{item.get('significance', '')}）" if item.get("significance") else "")
            )

    cliffhangers = normalized.get("cliffhangers_or_foreshadowing") or []
    if cliffhangers:
        lines.extend(["", "## 悬念/伏笔"])
        for item in cliffhangers[:20]:
            lines.append(f"- {item.get('description', '')}")

    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n…（整片视频分析摘要已截断）"


def summarize_video_episode_markdown(markdown: str, max_chars: int) -> str:
    text = (markdown or "").strip()
    if not text:
        return "（无整片视频分析）"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n…（整片视频分析摘要已截断）"


def video_episode_summary_usable(summary: str) -> bool:
    text = (summary or "").strip()
    return bool(text) and text not in {"（无整片视频分析）", "（无）"}


def build_video_episode_time_bounds_section(payload: dict[str, Any]) -> str:
    """构思蓝图用：整片视频分析时间边界与首尾片段锚点。"""
    duration = float(payload.get("video_duration_seconds") or 0)
    segments = payload.get("episodic_segments") or []
    if duration <= 0 and not segments:
        return ""
    duration_label = _format_timestamp(duration) if duration > 0 else "未知"
    lines = [
        "## 整片视频分析时间边界与场景锚点",
        f"- 源视频总时长：**{duration_label}**（{duration:.1f}s）",
        f"- 情节片段粒度：每 **{SEGMENT_INTERVAL_SECONDS} 秒** 一格，共 **{len(segments)}** 条",
    ]
    if segments:
        lines.append(f"- 首段：`{segments[0].get('time_range', '')}` · {segments[0].get('key_events', '')}")
        lines.append(
            f"- 末段：`{segments[-1].get('time_range', '')}` · {segments[-1].get('key_events', '')}"
        )
    lines.append("- 蓝图中的时间段/画面要点须落在此分析覆盖范围内，禁止编造超出上限的时间戳")
    return "\n".join(lines)


def _probe_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        return max(0.0, float((result.stdout or "").strip()))
    except ValueError:
        return 0.0


def _transcode_for_upload(
    video_path: str,
    *,
    output_path: str,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    high_fidelity: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", video_path])
    if duration_seconds and duration_seconds > 0:
        cmd.extend(["-t", str(duration_seconds)])
    if high_fidelity:
        video_filter = "scale=640:-2,fps=15"
        crf = "28"
        audio_bitrate = "48k"
    else:
        video_filter = "scale=480:-2,fps=12"
        crf = "32"
        audio_bitrate = "32k"
    cmd.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            crf,
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True)


def build_time_segment_plan(
    duration_seconds: float,
    *,
    chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
) -> list[tuple[float, float]]:
    """按时长切分上传计划：(start_seconds, segment_duration)。"""
    duration = max(0.0, float(duration_seconds))
    if duration <= 0:
        return []
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.5:
        segment_duration = min(chunk_seconds, duration - start)
        segments.append((start, segment_duration))
        start += segment_duration
    return segments


def _transcode_chunk_within_size_limit(
    video_path: str,
    *,
    output_path: str,
    start_seconds: float,
    segment_duration: float,
    max_upload_mb: float,
    high_fidelity: bool = False,
) -> float:
    """转码单段视频，若仍超限则继续缩短时长。"""
    current_duration = segment_duration
    while current_duration >= _MIN_CHUNK_SECONDS:
        _transcode_for_upload(
            video_path,
            output_path=output_path,
            start_seconds=start_seconds,
            duration_seconds=current_duration,
            high_fidelity=high_fidelity,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if size_mb <= max_upload_mb:
            return current_duration
        os.remove(output_path)
        current_duration = max(_MIN_CHUNK_SECONDS, current_duration / 2)
    raise ValueError(
        f"无法将视频片段压缩到 {max_upload_mb:.1f}MB 以内 "
        f"(start={_format_timestamp(start_seconds)})"
    )


def _plan_video_chunks(
    video_path: str,
    *,
    max_upload_mb: float = _DEFAULT_MAX_UPLOAD_MB,
    chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
    work_dir: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    progress = progress_callback or (lambda _p, _m: None)
    progress(6, "正在读取视频时长...")
    duration = _probe_duration_seconds(video_path)
    if duration <= 0:
        raise ValueError(f"无法读取视频时长: {video_path}")

    duration_label = _format_timestamp(duration)
    high_fidelity = duration <= 300

    if duration <= chunk_seconds:
        progress(8, f"正在压缩视频（时长 {duration_label}，单段上传）...")
        full_preview = os.path.join(work_dir, "upload_full.mp4")
        actual_duration = _transcode_chunk_within_size_limit(
            video_path,
            output_path=full_preview,
            start_seconds=0.0,
            segment_duration=duration,
            max_upload_mb=max_upload_mb,
            high_fidelity=high_fidelity,
        )
        size_mb = os.path.getsize(full_preview) / (1024 * 1024)
        logger.info(
            f"整片视频分析：单段上传 {actual_duration:.0f}s / {size_mb:.2f}MB"
        )
        progress(14, f"压缩完成 · {size_mb:.1f}MB · 单段上传")
        return [
            {
                "path": full_preview,
                "offset_seconds": 0.0,
                "duration_seconds": actual_duration,
            }
        ]

    planned_segments = build_time_segment_plan(duration, chunk_seconds=chunk_seconds)
    total_planned = max(1, len(planned_segments))
    chunks: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration - 0.5:
        planned_duration = min(chunk_seconds, duration - start)
        chunk_path = os.path.join(work_dir, f"upload_chunk_{index:02d}.mp4")
        compress_progress = 8 + int(6 * index / total_planned)
        progress(
            compress_progress,
            f"正在压缩第 {index + 1}/{total_planned} 段 "
            f"（{_format_timestamp(start)} 起，约 {planned_duration:.0f}s）...",
        )
        actual_duration = _transcode_chunk_within_size_limit(
            video_path,
            output_path=chunk_path,
            start_seconds=start,
            segment_duration=planned_duration,
            max_upload_mb=max_upload_mb,
            high_fidelity=high_fidelity,
        )
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        logger.info(
            f"整片视频分析：第 {index + 1} 段 "
            f"{_format_timestamp(start)}+{actual_duration:.0f}s / {size_mb:.2f}MB"
        )
        chunks.append(
            {
                "path": chunk_path,
                "offset_seconds": start,
                "duration_seconds": actual_duration,
            }
        )
        start += actual_duration
        index += 1
    progress(14, f"压缩完成 · 共 {len(chunks)} 段待上传分析")
    return chunks


class VideoEpisodeAnalysisService:
    """整片视频单集分析服务。"""

    @staticmethod
    def count_incomplete_chunks(
        checkpoint: dict[str, Any] | None,
        total_chunks: int,
    ) -> int:
        summary = summarize_checkpoint_progress(checkpoint, total_chunks)
        return summary["failed"] + summary["pending"]

    @staticmethod
    def _resolve_model_settings(
        *,
        vision_model_name: str | None = None,
        vision_api_key: str | None = None,
        vision_base_url: str | None = None,
    ) -> tuple[str, str, str]:
        model_name = vision_model_name or config.app.get("vision_openai_model_name") or ""
        if not model_name:
            raise ValueError("未配置视觉模型 vision_openai_model_name")
        api_key, base_url = resolve_llm_credentials(model_name, role="vision")
        if vision_api_key:
            api_key = vision_api_key
        if vision_base_url:
            base_url = vision_base_url
        if not api_key:
            raise ValueError(f"未配置模型 {model_name} 的 API Key")
        return model_name, api_key, base_url or ""

    async def _analyze_chunk_with_retries(
        self,
        *,
        provider: Any,
        chunk: dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        video_duration_seconds: float,
        drama_title: str,
        character_references: list[dict[str, str]] | None,
        relationship_diagram_path: str,
        api_key: str,
        base_url: str,
        progress: Callable[[float, str], None],
        analyze_progress: int,
    ) -> tuple[dict[str, Any], int]:
        chunk_duration_seconds = float(chunk.get("duration_seconds") or 0)
        chunk_offset_seconds = float(chunk.get("offset_seconds") or 0)
        chunk_schedule = build_fixed_segment_schedule(
            chunk_duration_seconds,
            start_offset_seconds=chunk_offset_seconds,
        )
        schedule_block = build_segment_schedule_prompt_block(chunk_schedule)
        reference_paths, naming_block = _prepare_chunk_reference_context(
            chunk_index=chunk_index,
            drama_title=drama_title,
            character_references=character_references,
            relationship_diagram_path=relationship_diagram_path,
        )
        if total_chunks == 1:
            prompt = build_video_episode_analysis_prompt(
                drama_title=drama_title,
                video_duration_seconds=video_duration_seconds,
                segment_schedule_block=schedule_block,
                character_naming_block=naming_block,
            )
        else:
            prompt = build_video_episode_chunk_prompt(
                drama_title=drama_title,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                offset_seconds=chunk_offset_seconds,
                chunk_duration_seconds=chunk_duration_seconds,
                segment_schedule_block=schedule_block,
                character_naming_block=naming_block,
            )
        chunk_max_tokens = max(
            8000,
            int(2000 + 120 * (chunk_duration_seconds / SEGMENT_INTERVAL_SECONDS)),
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_CHUNK_RETRIES + 1):
            try:
                if reference_paths and chunk_index == 0:
                    progress(
                        analyze_progress,
                        f"第 {chunk_index + 1}/{total_chunks} 段 · "
                        f"附带 {len(reference_paths)} 张头像参照 · 调用模型中"
                        + (f"（重试 {attempt}/{_MAX_CHUNK_RETRIES}）" if attempt > 1 else "")
                        + "...",
                    )
                raw = await provider.analyze_video(
                    chunk["path"],
                    prompt,
                    api_key=api_key,
                    api_base=base_url,
                    reference_image_paths=reference_paths,
                    timeout_override=_VIDEO_ANALYSIS_TIMEOUT,
                    max_tokens=min(chunk_max_tokens, 64000),
                )
                parsed_partial = parse_video_episode_analysis_payload(raw)
                issues = validate_chunk_partial(
                    parsed_partial,
                    chunk_offset_seconds=chunk_offset_seconds,
                    chunk_duration_seconds=chunk_duration_seconds,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
                if issues:
                    raise ValueError("片段约束不符合: " + " | ".join(issues[:3]))
                return parsed_partial, attempt
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_CHUNK_RETRIES:
                    progress(
                        analyze_progress,
                        f"第 {chunk_index + 1}/{total_chunks} 段不符合约束或调用失败，"
                        f"重试 {attempt}/{_MAX_CHUNK_RETRIES}（{exc}）...",
                    )
                    await asyncio.sleep(min(2 * attempt, 6))
        raise ValueError(str(last_error) if last_error else "分段分析失败")

    async def analyze_episode(
        self,
        *,
        video_path: str,
        drama_title: str = "罚罪2",
        drama_id: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        vision_model_name: str | None = None,
        vision_api_key: str | None = None,
        vision_base_url: str | None = None,
        max_upload_mb: float = _DEFAULT_MAX_UPLOAD_MB,
        progress_callback: Callable[[float, str], None] | None = None,
        output_path: str | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        progress = progress_callback or (lambda _p, _m: None)
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        model_name, api_key, base_url = self._resolve_model_settings(
            vision_model_name=vision_model_name,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
        )
        logger.info(f"整片视频分析 {model_name} → {describe_llm_route(model_name, role='vision')}")

        progress(2, f"正在初始化 · 模型 {model_name}")

        work_dir = os.path.join(utils.storage_dir(), "temp", "video_episode_upload", sanitize_video_stem(video_path))
        os.makedirs(work_dir, exist_ok=True)

        video_duration_seconds = _probe_duration_seconds(video_path)
        duration_label = _format_timestamp(video_duration_seconds)
        progress(4, f"视频时长 {duration_label}，准备压缩上传...")

        chunks = _plan_video_chunks(
            video_path,
            max_upload_mb=max_upload_mb,
            work_dir=work_dir,
            progress_callback=progress,
        )
        total_chunks = len(chunks)
        logger.info(
            f"整片视频分析：时长 {duration_label}，"
            f"共 {total_chunks} 段上传（max_upload_mb={max_upload_mb}）"
        )
        ref_count = len(character_references or [])
        if ref_count:
            progress(
                15,
                f"上传准备完成 · 共 {total_chunks} 段 · 已加载 {ref_count} 张人物头像参照",
            )
        else:
            progress(15, f"上传准备完成 · 共 {total_chunks} 段 · 开始调用视觉模型")

        from app.services.llm.openai_compatible_provider import OpenAICompatibleVisionProvider

        provider = OpenAICompatibleVisionProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )

        save_path = output_path or default_video_episode_analysis_path(video_path)
        checkpoint_path = default_checkpoint_path(save_path)
        chunk_results: dict[str, dict[str, Any]] = {}
        checkpoint: dict[str, Any] | None = None
        if resume:
            checkpoint = load_video_episode_checkpoint(checkpoint_path)
            if checkpoint and is_checkpoint_compatible(
                checkpoint,
                video_path=video_path,
                video_duration_seconds=video_duration_seconds,
                chunks=chunks,
            ):
                chunk_results = {
                    str(key): value
                    for key, value in (checkpoint.get("chunk_results") or {}).items()
                    if isinstance(value, dict)
                }
                summary = summarize_checkpoint_progress(checkpoint, total_chunks)
                if summary["completed"] or summary["failed"]:
                    progress(
                        15,
                        f"续跑补全 · 已完成 {summary['completed']}/{total_chunks} 段"
                        + (f"，失败 {summary['failed']} 段" if summary["failed"] else "")
                        + (f"，待处理 {summary['pending']} 段" if summary["pending"] else ""),
                    )
            elif checkpoint:
                logger.warning("整片视频分析检查点与当前视频/分段计划不兼容，将从头分析")
                checkpoint = None

        def _persist_checkpoint() -> None:
            save_video_episode_checkpoint(
                checkpoint_path,
                {
                    "artifact_version": VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION,
                    "video_path": os.path.abspath(video_path),
                    "video_duration_seconds": round(video_duration_seconds, 3),
                    "total_chunks": total_chunks,
                    "chunks_meta": _chunks_meta(chunks),
                    "chunk_results": chunk_results,
                    "updated_at": datetime.now().isoformat(),
                },
            )

        for index, chunk in enumerate(chunks):
            chunk_key = str(index)
            existing = chunk_results.get(chunk_key)
            if (
                isinstance(existing, dict)
                and existing.get("status") == "completed"
                and existing.get("partial")
            ):
                segment_count = len((existing.get("partial") or {}).get("episodic_segments") or [])
                done_progress = 16 + int(68 * (index + 1) / max(total_chunks, 1))
                progress(
                    done_progress,
                    f"第 {index + 1}/{total_chunks} 段已缓存 · 跳过 · 本段 {segment_count} 条情节片段",
                )
                continue

            chunk_duration_seconds = float(chunk.get("duration_seconds") or 0)
            chunk_offset_seconds = float(chunk.get("offset_seconds") or 0)
            chunk_start_label = _format_timestamp(chunk_offset_seconds)
            chunk_end_label = _format_timestamp(chunk_offset_seconds + chunk_duration_seconds)
            analyze_progress = 16 + int(68 * index / max(total_chunks, 1))
            progress(
                analyze_progress,
                f"第 {index + 1}/{total_chunks} 段 · {chunk_start_label}-{chunk_end_label} · "
                f"正在上传并调用模型（约 {chunk_duration_seconds:.0f}s）...",
            )
            try:
                parsed_partial, retry_count = await self._analyze_chunk_with_retries(
                    provider=provider,
                    chunk=chunk,
                    chunk_index=index,
                    total_chunks=total_chunks,
                    video_duration_seconds=video_duration_seconds,
                    drama_title=drama_title,
                    character_references=character_references,
                    relationship_diagram_path=relationship_diagram_path,
                    api_key=api_key,
                    base_url=base_url,
                    progress=progress,
                    analyze_progress=analyze_progress,
                )
                chunk_results[chunk_key] = {
                    "status": "completed",
                    "partial": parsed_partial,
                    "retry_count": retry_count,
                }
            except Exception as exc:
                logger.warning(f"整片视频分析第 {index + 1} 段失败: {exc}")
                chunk_results[chunk_key] = {
                    "status": "failed",
                    "error": str(exc),
                    "retry_count": _MAX_CHUNK_RETRIES,
                }
                _persist_checkpoint()
                done_progress = 16 + int(68 * (index + 1) / max(total_chunks, 1))
                progress(
                    done_progress,
                    f"第 {index + 1}/{total_chunks} 段失败（已保存进度，可稍后补全）: {exc}",
                )
                continue

            _persist_checkpoint()
            segment_count = len(parsed_partial.get("episodic_segments") or [])
            done_progress = 16 + int(68 * (index + 1) / max(total_chunks, 1))
            retry_note = (
                f" · 重试 {retry_count} 次后通过"
                if retry_count > 1
                else ""
            )
            progress(
                done_progress,
                f"第 {index + 1}/{total_chunks} 段完成 · 本段 {segment_count} 条情节片段{retry_note}",
            )

        partials: list[dict[str, Any]] = []
        failed_chunk_indices: list[int] = []
        for index in range(total_chunks):
            entry = chunk_results.get(str(index))
            if (
                isinstance(entry, dict)
                and entry.get("status") == "completed"
                and entry.get("partial")
            ):
                partials.append(entry["partial"])
            else:
                failed_chunk_indices.append(index)

        if not partials:
            raise ValueError(
                "所有上传分段均未成功完成分析；已保留检查点，请稍后点击「补全未完成分析」重试"
            )

        completed_chunks = len(partials)
        progress(
            86,
            f"正在合并 {completed_chunks}/{total_chunks} 段分析结果"
            + (f"（{len(failed_chunk_indices)} 段待补全）" if failed_chunk_indices else "")
            + "...",
        )
        analysis = merge_video_episode_partial_analyses(partials)
        merged_segment_count = len(analysis.get("episodic_segments") or [])
        progress(90, f"合并完成 · 共 {merged_segment_count} 条情节片段")
        full_schedule = build_fixed_segment_schedule(video_duration_seconds)
        analysis["episodic_segments"] = enforce_episodic_segment_schedule(
            analysis.get("episodic_segments") or [],
            full_schedule,
        )
        merged_segment_count = len(analysis.get("episodic_segments") or [])
        progress(92, f"正在对齐固定 {SEGMENT_INTERVAL_SECONDS} 秒时间窗（共 {merged_segment_count} 条）...")
        coverage_warnings = validate_episodic_segments(
            analysis.get("episodic_segments") or [],
            video_duration_seconds=video_duration_seconds,
            expected_time_ranges=full_schedule,
        )
        if coverage_warnings:
            logger.warning(
                "整片视频分析片段约束告警: " + " | ".join(coverage_warnings[:5])
            )
            progress(
                94,
                f"校验完成 · {len(coverage_warnings)} 条告警（已写入 coverage_warnings）",
            )
        else:
            progress(94, "校验完成 · 片段时长符合约束")

        analysis_status = "complete" if not failed_chunk_indices else "incomplete"
        artifact = {
            "artifact_version": VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "video_path": os.path.abspath(video_path),
            "video_duration_seconds": round(video_duration_seconds, 3),
            "drama_title": drama_title,
            "drama_id": (drama_id or drama_title).strip(),
            "character_references": [
                {"name": str(item.get("name") or "").strip(), "path": str(item.get("path") or "").strip()}
                for item in (character_references or [])
                if isinstance(item, dict) and item.get("name")
            ],
            "relationship_diagram_path": relationship_diagram_path or "",
            "vision_model_name": model_name,
            "analysis_mode": "direct_video",
            "analysis_status": analysis_status,
            "chunk_count": total_chunks,
            "completed_chunk_count": completed_chunks,
            "failed_chunk_indices": failed_chunk_indices,
            "segment_interval_seconds": SEGMENT_INTERVAL_SECONDS,
            "segment_split_policy": "fixed_10s",
            "episodic_segment_count": len(analysis.get("episodic_segments") or []),
            "coverage_warnings": coverage_warnings,
            **analysis,
        }

        progress(96, f"正在保存 JSON 到 {os.path.basename(save_path)}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        saved_artifact = _prepend_field_comments(artifact, VIDEO_EPISODE_FIELD_COMMENTS)
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(saved_artifact, fp, ensure_ascii=False, indent=2)

        if failed_chunk_indices:
            _persist_checkpoint()
            progress(
                100,
                f"部分完成 · {completed_chunks}/{total_chunks} 段 · "
                f"{merged_segment_count} 条片段 · 失败段 {failed_chunk_indices}（可补全）",
            )
            logger.warning(
                f"整片视频分析部分完成: {save_path} · 失败段 {failed_chunk_indices}"
            )
        else:
            if os.path.isfile(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except OSError as exc:
                    logger.warning(f"无法删除检查点 {checkpoint_path}: {exc}")
            progress(
                100,
                f"分析完成 · {merged_segment_count} 条片段 · "
                f"{len(analysis.get('important_dialogues') or [])} 条台词",
            )
            logger.info(f"整片视频分析已保存: {save_path}")
        saved_artifact["output_path"] = save_path
        saved_artifact["checkpoint_path"] = checkpoint_path if failed_chunk_indices else ""
        return saved_artifact
