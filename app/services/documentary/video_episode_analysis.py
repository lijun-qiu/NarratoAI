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
    split_character_references_into_collage_sheets,
    collage_max_heads_per_sheet,
)
from app.services.drama_character_registry import resolve_media_path
from app.services.documentary.video_episode_constants import (
    SCENE_CANDIDATE_THRESHOLD,
    SCENE_DETECT_THRESHOLD,
    SCENE_ENVIRONMENT_DIFF_THRESHOLD,
    SCENE_FRAME_SAMPLE_AFTER_SECONDS,
    SCENE_FRAME_SAMPLE_BEFORE_SECONDS,
    SCENE_MAX_SECONDS,
    SCENE_MIN_MERGE_SECONDS,
    SCENE_MIN_SEGMENT_SECONDS,
    SEGMENT_INTERVAL_SECONDS,
    SEGMENT_MAX_SECONDS,
    SEGMENT_MIN_SECONDS,
    SEGMENT_SPLIT_POLICY,
    get_upload_transcode_profile,
    get_video_episode_scene_settings,
    get_video_episode_upload_settings,
    resolve_upload_transcode_profile_name,
)
from app.services.documentary.video_episode_segment_schedule import (
    average_segment_seconds,
    build_segment_schedule,
    detect_edit_cut_seconds,
    detect_scene_cut_seconds,
    segment_policy_summary,
)
from app.services.documentary.plot_reference import build_plot_reference_prompt_section
from app.services.prompts.documentary.video_episode_analysis import (
    build_reference_carryover_naming_block,
    build_video_episode_analysis_prompt,
    build_video_episode_chunk_prompt,
    build_video_episode_vision_reference_prompt_section,
)
from app.utils import utils

VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION = "documentary-video-episode-analysis-v12"
_MIN_SCENE_SECONDS = 0.5
_VIDEO_ANALYSIS_TIMEOUT = 900.0
_MAX_CHUNK_RETRIES = 3
# 单次 API 最多输出的情节窗条数，超出则对同一上传段分批调用
_MAX_SEGMENTS_PER_API_CALL = 32

VIDEO_EPISODE_FIELD_COMMENTS: dict[str, str] = {
    "_readme": "JSON 不支持 // 注释；本 field_comments 对象置于文件最前，说明各字段含义，不参与业务逻辑。",
    "artifact_version": "本 JSON 结构版本号",
    "generated_at": "生成时间（ISO8601）",
    "video_path": "源视频绝对路径",
    "video_duration_seconds": "源视频总时长（秒）",
    "drama_title": "剧名/单集所属作品",
    "drama_id": "剧目 ID（与抽帧分析人物库一致）",
    "plot_reference": "用户提供的剧情参考说明（分析理解辅助，非画面依据）",
    "character_references": "分析时参照的人物头像列表（name + path）",
    "relationship_diagram_path": "人物关系图路径（若有）",
    "vision_model_name": "视觉模型名称",
    "analysis_mode": "分析模式（direct_video=整片直传）",
    "analysis_status": "分析状态：complete=全部段完成；incomplete=部分段失败可补全",
    "chunk_count": "上传分镜段总数（每段对应一个切镜片段）",
    "completed_chunk_count": "已成功完成的分镜段数",
    "failed_chunk_indices": "失败分镜段索引列表（0 起），可点击补全重试",
    "segment_min_seconds": f"上传段最短时长（秒），当前 {SCENE_MIN_SEGMENT_SECONDS}",
    "segment_max_seconds": f"无切镜长镜头上限（秒），当前 {SCENE_MAX_SECONDS}",
    "segment_split_policy": f"切分策略（{SEGMENT_SPLIT_POLICY}=按场景环境变化切段）",
    "episodic_segment_count": "episodic_segments 条数",
    "coverage_warnings": "时间窗/片段约束校验告警（非空表示模型输出曾偏离预计算时间窗）",
    "overall_summary": "本集/本段核心剧情概括（约200字内）",
    "key_conflict": "本集/本段最核心的矛盾冲突（一句话）",
    "episodic_segments": "分镜情节片段列表（全片时间轴绝对时间，一切镜一段）",
    "episodic_segments.segment_id": "片段序号，从 1 起",
    "episodic_segments.title": "片段标题（4-6字）",
    "episodic_segments.time_range": "片内绝对时间窗，格式 HH:MM:SS-HH:MM:SS，须与预计算窗口一致",
    "episodic_segments.key_events": "该时间窗内关键事件（切镜格须写「场景切换至…」）",
    "episodic_segments.narration": "纪录片旁白（第三人称，15-35字，可用于后期配音）",
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


def is_checkpoint_source_compatible(
    checkpoint: dict[str, Any],
    *,
    video_path: str,
    video_duration_seconds: float,
) -> bool:
    """压缩前检查：视频路径/时长/版本是否与检查点一致。"""
    if not checkpoint:
        return False
    if checkpoint.get("artifact_version") != VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION:
        return False
    if os.path.abspath(str(checkpoint.get("video_path") or "")) != os.path.abspath(video_path):
        return False
    if abs(float(checkpoint.get("video_duration_seconds") or 0) - video_duration_seconds) > 1.0:
        return False
    return True


def describe_checkpoint_incompatibility(
    checkpoint: dict[str, Any],
    *,
    video_path: str,
    video_duration_seconds: float,
    chunks: list[dict[str, Any]] | None = None,
) -> str:
    if not checkpoint:
        return "无检查点"
    if checkpoint.get("artifact_version") != VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION:
        return "分析版本已更新，旧检查点失效"
    if os.path.abspath(str(checkpoint.get("video_path") or "")) != os.path.abspath(video_path):
        return "视频文件与检查点不一致"
    if abs(float(checkpoint.get("video_duration_seconds") or 0) - video_duration_seconds) > 1.0:
        return "视频时长与检查点不一致"
    if chunks is not None:
        if int(checkpoint.get("total_chunks") or 0) != len(chunks):
            return f"分段数量变化（检查点 {checkpoint.get('total_chunks')} ≠ 当前 {len(chunks)}）"
        saved_meta = checkpoint.get("chunks_meta") or []
        current_meta = _chunks_meta(chunks)
        if len(saved_meta) != len(current_meta):
            return "分段边界数量不一致"
        for saved, current in zip(saved_meta, current_meta):
            if not isinstance(saved, dict):
                return "检查点分段元数据损坏"
            if abs(float(saved.get("offset_seconds") or 0) - current["offset_seconds"]) > 0.5:
                return "分段起始时间与检查点不一致"
            if abs(float(saved.get("duration_seconds") or 0) - current["duration_seconds"]) > 0.5:
                return "分段时长与检查点不一致"
    return "未知原因"


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


def checkpoint_needs_resume(checkpoint: dict[str, Any] | None) -> bool:
    """是否仍有压缩或模型分析未完成的工作。"""
    if not checkpoint:
        return False
    total = int(checkpoint.get("total_chunks") or 0)
    meta = checkpoint.get("chunks_meta") or []
    if total > 0 and len(meta) < total:
        return True
    chunk_total = total or max(len(meta), 1)
    summary = summarize_checkpoint_progress(checkpoint, chunk_total)
    return summary["failed"] + summary["pending"] > 0


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


def estimate_video_episode_chunk_max_tokens(segment_count: int) -> int:
    """按情节窗条数估算输出 token。"""
    count = max(1, int(segment_count))
    if count == 1:
        return 12000
    return min(64000, max(12000, 3500 + count * 320))


def split_schedule_for_api_batches(
    schedule: list[str],
    *,
    max_per_batch: int = _MAX_SEGMENTS_PER_API_CALL,
) -> list[list[str]]:
    if len(schedule) <= max(1, max_per_batch):
        return [schedule]
    batches: list[list[str]] = []
    cursor = 0
    while cursor < len(schedule):
        batches.append(schedule[cursor : cursor + max_per_batch])
        cursor += max_per_batch
    return batches


def build_schedule_batch_prompt_addon(
    *,
    batch_index: int,
    batch_count: int,
    batch_size: int,
) -> str:
    if batch_count <= 1:
        return ""
    lines = [
        "## 本批输出范围（分批 · 同一视频仅分析下列时间窗）",
        f"- 本批为第 **{batch_index + 1}/{batch_count}** 批，共 **{batch_size}** 条 `time_range`",
        "- **仅**输出本批窗口对应的 `episodic_segments`，条数须与列表一致",
    ]
    if batch_index > 0:
        lines.extend(
            [
                "- 本批 **`overall_summary` / `key_conflict` 填空字符串 `\"\"`**",
                "- 本批 `important_dialogues` / `cliffhangers_or_foreshadowing` 输出 `[]`",
            ]
        )
    return "\n".join(lines)


def validate_chunk_partial(
    partial: dict[str, Any],
    *,
    chunk_offset_seconds: float,
    chunk_duration_seconds: float,
    chunk_index: int,
    total_chunks: int,
    expected_schedule: list[str] | None = None,
    require_overall_summary: bool = True,
) -> list[str]:
    """校验单段模型输出是否满足时间窗约束，不符合则触发重试。"""
    issues: list[str] = []
    if expected_schedule is None:
        issues.append("缺少预计算时间窗")
        return issues
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

    if require_overall_summary and chunk_index == 0 and not str(partial.get("overall_summary") or "").strip():
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
    """每段上传均附带头像/关系图（拼图每张最多 4 人），便于逐格识脸。"""
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
        "frame_reference_token_saver": False,
        "frame_reference_attach_mode": "every_batch",
        "frame_reference_max_edge": 384,
        "frame_reference_individual_max_heads": 4,
        "frame_reference_collage_max_heads": 4,
        "default_video_theme": drama_title,
    }
    head_paths = [item["path"] for item in refs if os.path.isfile(item["path"])]
    use_collage = resolve_reference_collage_mode(settings, head_count=len(head_paths))
    max_per_sheet = collage_max_heads_per_sheet(settings)
    collage_sheets = split_character_references_into_collage_sheets(refs, max_per_sheet=max_per_sheet)

    prefix_paths, _carryover = prepare_reference_prefix_images(
        batch_index=chunk_index,
        relationship_diagram_path=rel_path,
        character_references=refs,
        settings=settings,
    )
    if prefix_paths:
        naming_block = build_video_episode_vision_reference_prompt_section(
            drama_label=drama_title,
            character_references=refs,
            relationship_diagram_attached=bool(rel_path),
            reference_image_count=len(prefix_paths),
            character_collage=use_collage and len(head_paths) >= 2,
            collage_sheets=collage_sheets if use_collage else None,
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
    interval_seconds: int | None = None,
    start_offset_seconds: float = 0.0,
    video_path: str = "",
) -> list[str]:
    """兼容旧名：现统一走自适应场景分段。"""
    del interval_seconds
    return build_segment_schedule(
        duration_seconds,
        start_offset_seconds=start_offset_seconds,
        video_path=video_path,
    )


def build_segment_schedule_prompt_block(time_ranges: list[str]) -> str:
    if not time_ranges:
        return ""
    policy_label = segment_policy_summary()
    if SEGMENT_SPLIT_POLICY == "scene_cut" and len(time_ranges) == 1:
        time_range = time_ranges[0]
        return "\n".join(
            [
                "## 本分镜时间窗（硬性要求 · 已预计算）",
                f"上传视频即 **一个完整分镜镜头**；你必须输出 **恰好 1 条** `episodic_segments`。",
                f"`time_range` 必须为 **`{time_range}`**（全片绝对时间，字符级一致）。",
                "- 分析本镜画面、对白与人物；`key_events` 写该镜关键事件。",
                "- 若本镜为硬切/换场开头，可在 `key_events` 写「场景切换至…」。",
                "- 须填写 `title`（4-6字）、`narration`、`environment_description`、`involved_characters`。",
            ]
        )
    lines = [
        f"## 分镜时间窗口（硬性要求 · {policy_label}）",
        (
            f"你必须输出 **恰好 {len(time_ranges)} 条** `episodic_segments`，"
            "`time_range` 必须与下列窗口 **完全一致**（字符级一致，仅用 `-` 连接）："
        ),
    ]
    if SEGMENT_SPLIT_POLICY == "scene_cut":
        lines.append(
            "- 每个窗口对应 **一个切镜片段**；**禁止**合并、改短或拉长预计算窗口。"
        )
    else:
        lines.append(
            f"- 同场景内每格 **{SEGMENT_MIN_SECONDS}–{SEGMENT_MAX_SECONDS} 秒**；"
            "遇**切镜/换场**边界已单独切格，该格 `key_events` 须写「场景切换至…」"
        )
    for index, time_range in enumerate(time_ranges, start=1):
        lines.append(f"{index}. `{time_range}`")
    lines.extend(
        [
            "- 每条都必须填写 4-6 字 `title`、该窗口内 `key_events`、`narration`（纪录片旁白）、"
            "`environment_description`（场景环境）、`involved_characters`。",
            "- **不得跳过任何窗口**。",
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
    segment_split_policy: str = SEGMENT_SPLIT_POLICY,
) -> list[str]:
    """检查片段是否符合时间窗策略，返回警告列表。"""
    warnings: list[str] = []
    video_end = max(0, int(video_duration_seconds))

    if expected_time_ranges:
        if len(segments) != len(expected_time_ranges):
            warnings.append(
                f"片段数量 {len(segments)} 与预计算窗口 {len(expected_time_ranges)} 不一致"
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
        if expected_time_ranges and index < len(expected_time_ranges):
            expected_range = expected_time_ranges[index]
            if str(segment.get("time_range") or "").strip().replace("—", "-") == expected_range:
                continue
        if segment_split_policy == "adaptive_scene":
            if not is_tail and duration > SEGMENT_MAX_SECONDS + 0.5:
                warnings.append(
                    f"片段 {time_range} 时长 {duration}s，超过同场景上限 {SEGMENT_MAX_SECONDS}s"
                )
            elif not is_tail and duration + 0.05 < SEGMENT_MIN_SECONDS:
                warnings.append(
                    f"片段 {time_range} 时长 {duration}s，短于同场景下限 {SEGMENT_MIN_SECONDS}s"
                    "（切镜格除外）"
                )
        elif segment_split_policy == "scene_cut":
            if not is_tail and duration > SCENE_MAX_SECONDS + 0.5:
                warnings.append(
                    f"片段 {time_range} 时长 {duration}s，超过分镜上限 {SCENE_MAX_SECONDS}s"
                )
        else:
            legacy_interval = SEGMENT_INTERVAL_SECONDS
            if not is_tail and duration != legacy_interval:
                warnings.append(
                    f"片段 {time_range} 时长 {duration}s，应为 {legacy_interval}s"
                )
            elif is_tail and duration > legacy_interval:
                warnings.append(
                    f"片段 {time_range} 时长 {duration}s 超过 {legacy_interval}s"
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


_CONTINUATION_ENV_MARKERS = ("延续上段", "承接上段", "同上", "环境延续", "画面延续")
_SCENE_CHANGE_MARKERS = (
    "场景切换",
    "硬切",
    "转场",
    "切至",
    "切换至",
    "镜头切",
    "另起",
    "回忆",
    "闪回",
    "画面切",
    "转镜",
    "切到",
    "来到",
    "转至",
    "切换",
    "翌日",
    "次日",
)


def _is_environment_continuation(text: str) -> bool:
    blob = (text or "").strip()
    if not blob:
        return True
    return any(marker in blob for marker in _CONTINUATION_ENV_MARKERS)


def _has_scene_change_marker(*texts: str) -> bool:
    blob = " ".join(str(item or "").strip() for item in texts if item)
    return any(marker in blob for marker in _SCENE_CHANGE_MARKERS)


def _character_sets_fully_changed(
    previous: list[str],
    current: list[str],
) -> bool:
    prev = {name for name in previous if name}
    curr = {name for name in current if name}
    if not prev or not curr:
        return False
    return not (prev & curr)


def _resolve_concrete_environment(
    segments: list[dict[str, Any]],
    index: int,
) -> str:
    """向上查找最近一条非「延续上段」的环境描述。"""
    for pos in range(index, -1, -1):
        env = str(segments[pos].get("environment_description") or "").strip()
        if env and not _is_environment_continuation(env):
            return env
    return ""


def _environments_suggest_same_scene(previous_env: str, current_env: str) -> bool:
    prev = (previous_env or "").strip()
    curr = (current_env or "").strip()
    if _is_environment_continuation(curr):
        return True
    if _is_environment_continuation(prev):
        return not _has_scene_change_marker(curr)
    if not prev or not curr:
        return False
    if prev == curr:
        return True
    indoor_markers = ("室内", "房间", "屋内", "餐桌", "窗前")
    if any(marker in prev for marker in indoor_markers) and any(
        marker in curr for marker in indoor_markers
    ):
        if not _has_scene_change_marker(curr):
            return True
    return False


def build_upload_chunk_boundary_seconds(
    chunks_meta: list[dict[str, Any]] | None,
) -> list[int]:
    """上传分段之间的衔接时刻（秒，不含 0 与片尾）。"""
    if not chunks_meta:
        return []
    boundaries: list[int] = []
    for index, meta in enumerate(chunks_meta):
        if index >= len(chunks_meta) - 1 or not isinstance(meta, dict):
            continue
        offset = float(meta.get("offset_seconds") or 0)
        duration = float(meta.get("duration_seconds") or 0)
        if duration > 0:
            boundaries.append(int(round(offset + duration)))
    return boundaries


_PER_GRID_CONTINUITY_NOTE_RE = re.compile(
    r"本格原标注\s*([^。]+)。"
)
_PER_GRID_KEY_EVENTS_PREFIX_RE = re.compile(
    r"^（(?:承接|延续)上段[^）]*）(?:；)?"
)


def revert_per_grid_continuity_overrides(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """撤销旧版「逐格强制承接」对人物/事件的改写。"""
    reverted: list[dict[str, Any]] = []
    for segment in segments:
        current = dict(segment)
        note = str(current.get("continuity_note") or "").strip()
        if note.startswith("待核对：") and "本格原标注" in note:
            match = _PER_GRID_CONTINUITY_NOTE_RE.search(note)
            if match:
                restored = [
                    name.strip()
                    for name in match.group(1).split("、")
                    if name.strip()
                ]
                if restored:
                    current["involved_characters"] = restored
            key_events = str(current.get("key_events") or "").strip()
            cleaned = _PER_GRID_KEY_EVENTS_PREFIX_RE.sub("", key_events).strip()
            if cleaned:
                current["key_events"] = cleaned
            current.pop("continuity_note", None)
        reverted.append(current)
    return reverted


def _find_segment_index_at_or_before(
    segments: list[dict[str, Any]],
    boundary_sec: int,
) -> int | None:
    """边界时刻之前（含）结束的最后一格。"""
    best_index: int | None = None
    best_end = -1
    for index, segment in enumerate(segments):
        _start, end = _parse_time_range_bounds(str(segment.get("time_range") or ""))
        if end <= boundary_sec and end > best_end:
            best_end = end
            best_index = index
    return best_index


def _find_segment_index_at_or_after(
    segments: list[dict[str, Any]],
    boundary_sec: int,
) -> int | None:
    """边界时刻起（含）的第一格。"""
    for index, segment in enumerate(segments):
        start, _end = _parse_time_range_bounds(str(segment.get("time_range") or ""))
        if start >= boundary_sec:
            return index
    return None


def infer_chunks_meta_from_artifact(
    *,
    video_duration_seconds: float,
    chunk_count: int,
    chunk_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """旧 JSON 无 chunks_meta 时，按上传策略估算分段边界。"""
    if chunk_seconds is None:
        chunk_seconds = float(
            get_video_episode_upload_settings().get("chunk_seconds", 300.0)
        )
    if chunk_count <= 1 or video_duration_seconds <= 0:
        return []
    meta: list[dict[str, Any]] = []
    start = 0.0
    while start < video_duration_seconds - 0.5 and len(meta) < chunk_count:
        duration = min(chunk_seconds, video_duration_seconds - start)
        meta.append(
            {
                "offset_seconds": round(start, 3),
                "duration_seconds": round(duration, 3),
            }
        )
        start += duration
    return meta


def _text_mentions_any_character(text: str, names: list[str]) -> bool:
    blob = str(text or "")
    return any(name and name in blob for name in names)


def _should_fix_boundary_characters(
    *,
    prev_chars: list[str],
    curr_chars: list[str],
    key_events: str,
    narration: str,
    title: str,
    environment: str,
) -> bool:
    """同场景连续且文本未明确引入新人物时，可安全校正人物名单。"""
    if not prev_chars:
        return False
    if not curr_chars:
        return True
    if not _character_sets_fully_changed(prev_chars, curr_chars):
        return False
    if _has_scene_change_marker(key_events, narration, title, environment):
        return False
    text_blob = " ".join([key_events, narration, title, environment])
    curr_only = [name for name in curr_chars if name not in prev_chars]
    if curr_only and _text_mentions_any_character(text_blob, curr_only):
        return False
    return True


def _apply_boundary_character_fix(
    segment: dict[str, Any],
    *,
    prev_chars: list[str],
    original_chars: list[str],
    boundary_label: str,
) -> dict[str, Any]:
    current = dict(segment)
    current["involved_characters"] = list(prev_chars)
    if original_chars and original_chars != prev_chars:
        current["continuity_note"] = (
            f"上传分段边界 {boundary_label} 已校正人物：原标注 "
            f"{'、'.join(original_chars)} → 与上一段末 {'、'.join(prev_chars)} 一致"
            f"（同场景连续，分段衔接误换人）"
        )
    return current


def repair_upload_chunk_boundary_continuity(
    segments: list[dict[str, Any]],
    *,
    chunk_boundary_seconds: list[int] | None = None,
    chunks_meta: list[dict[str, Any]] | None = None,
    max_grids_after_boundary: int = 3,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    仅在约 300s 上传分段边界检查衔接：
    - 不修改段内普通 5 秒格
    - 边界后首几格若同场景连续、未切镜、且文本未引入新人物 → 人物校正为上一段末
    - 无法确认时仅 continuity_note + coverage_warning，不改写
    """
    if not segments:
        return [], []

    boundaries = list(chunk_boundary_seconds or [])
    if not boundaries and chunks_meta:
        boundaries = build_upload_chunk_boundary_seconds(chunks_meta)
    if not boundaries:
        return list(segments), []

    result = revert_per_grid_continuity_overrides(segments)
    warnings: list[str] = []

    for boundary_sec in boundaries:
        before_index = _find_segment_index_at_or_before(result, boundary_sec)
        after_index = _find_segment_index_at_or_after(result, boundary_sec)
        if before_index is None or after_index is None or before_index >= after_index:
            continue

        previous = result[before_index]
        current = dict(result[after_index])
        prev_chars = [
            str(name).strip()
            for name in (previous.get("involved_characters") or [])
            if str(name).strip()
        ]
        curr_chars = [
            str(name).strip()
            for name in (current.get("involved_characters") or [])
            if str(name).strip()
        ]
        prev_env = str(previous.get("environment_description") or "")
        curr_env = str(current.get("environment_description") or "")
        resolved_prev_env = _resolve_concrete_environment(result, before_index) or prev_env
        key_events = str(current.get("key_events") or "")
        narration = str(current.get("narration") or "")
        title = str(current.get("title") or "")

        same_scene = _environments_suggest_same_scene(resolved_prev_env, curr_env)
        if _is_environment_continuation(curr_env) and not _has_scene_change_marker(
            key_events, narration, title, curr_env
        ):
            same_scene = True

        if not same_scene:
            continue

        boundary_label = _format_timestamp(boundary_sec)
        prev_range = previous.get("time_range") or ""

        for offset in range(max(1, max_grids_after_boundary)):
            grid_index = after_index + offset
            if grid_index >= len(result):
                break
            current = dict(result[grid_index])
            curr_chars = [
                str(name).strip()
                for name in (current.get("involved_characters") or [])
                if str(name).strip()
            ]
            key_events = str(current.get("key_events") or "")
            narration = str(current.get("narration") or "")
            title = str(current.get("title") or "")
            curr_env = str(current.get("environment_description") or "")

            grid_same_scene = _environments_suggest_same_scene(resolved_prev_env, curr_env)
            if _is_environment_continuation(curr_env) and not _has_scene_change_marker(
                key_events, narration, title, curr_env
            ):
                grid_same_scene = True
            if offset > 0 and not grid_same_scene:
                break

            needs_fix = (
                prev_chars
                and (
                    not curr_chars
                    or _character_sets_fully_changed(prev_chars, curr_chars)
                )
                and _should_fix_boundary_characters(
                    prev_chars=prev_chars,
                    curr_chars=curr_chars,
                    key_events=key_events,
                    narration=narration,
                    title=title,
                    environment=curr_env,
                )
            )
            if not needs_fix:
                if offset == 0:
                    # 首格无法自动校正：文本已引入新人物或已切镜，仅告警
                    if (
                        curr_chars
                        and prev_chars
                        and _character_sets_fully_changed(prev_chars, curr_chars)
                        and not _has_scene_change_marker(key_events, narration, title, curr_env)
                    ):
                        curr_range = current.get("time_range") or ""
                        warnings.append(
                            f"上传分段边界 {boundary_label}: 上一段末 `{prev_range}` 人物 "
                            f"{'、'.join(prev_chars)} → 下一段 `{curr_range}` 人物 "
                            f"{'、'.join(curr_chars)}，环境似连续且文本提及新人物，请人工核对"
                        )
                        current["continuity_note"] = (
                            f"上传分段边界 {boundary_label}：上一段末 {'、'.join(prev_chars)}，"
                            f"本格 {'、'.join(curr_chars)}； narration/事件已提及新人物或需切镜说明，未自动校正"
                        )
                        result[grid_index] = current
                break

            original_chars = list(curr_chars)
            curr_range = current.get("time_range") or ""
            fixed = _apply_boundary_character_fix(
                current,
                prev_chars=prev_chars,
                original_chars=original_chars,
                boundary_label=boundary_label,
            )
            result[grid_index] = fixed
            warnings.append(
                f"上传分段边界 {boundary_label}: `{curr_range}` 人物 "
                f"{'、'.join(original_chars) if original_chars else '（空）'} → "
                f"已校正为 {'、'.join(prev_chars)}（同场景连续）"
            )

    return result, warnings


def repair_episodic_segment_continuity(
    segments: list[dict[str, Any]],
    *,
    chunks_meta: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """兼容旧调用：仅做上传分段边界检查，不再逐格强制承接。"""
    return repair_upload_chunk_boundary_continuity(
        segments,
        chunks_meta=chunks_meta,
    )


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
            entry: dict[str, Any] = {
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
            note = str(item.get("continuity_note") or "").strip()
            if note:
                entry["continuity_note"] = note
            normalized["episodic_segments"].append(entry)

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
    segments = list(analysis.get("episodic_segments") or [])
    if segments:
        chunks_meta = raw.get("chunks_meta") if isinstance(raw.get("chunks_meta"), list) else None
        if not chunks_meta and int(raw.get("chunk_count") or 0) > 1:
            chunks_meta = infer_chunks_meta_from_artifact(
                video_duration_seconds=float(raw.get("video_duration_seconds") or 0),
                chunk_count=int(raw.get("chunk_count") or 0),
            )
        repaired, continuity_warnings = repair_upload_chunk_boundary_continuity(
            segments,
            chunks_meta=chunks_meta,
        )
        analysis["episodic_segments"] = repaired
        if continuity_warnings:
            existing = list(analysis.get("coverage_warnings") or raw.get("coverage_warnings") or [])
            for item in continuity_warnings:
                if item not in existing:
                    existing.append(item)
            analysis["coverage_warnings"] = existing
    for key in (
        "artifact_version",
        "video_path",
        "video_duration_seconds",
        "drama_title",
        "drama_id",
        "analysis_status",
        "segment_min_seconds",
        "segment_max_seconds",
        "segment_split_policy",
        "scene_cut_count",
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
    note = str(segment.get("continuity_note") or "").strip()
    note_line = f"\n  - 连续性：{note}" if note else ""
    return (
        f"- `{segment.get('time_range', '')}` **{segment.get('title', '')}** · "
        f"{segment.get('key_events', '')}\n"
        f"  - 旁白：{segment.get('narration', '')}\n"
        f"  - 环境：{segment.get('environment_description', '')}\n"
        f"  - 人物：{characters}{note_line}"
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
                "## 自适应场景情节片段",
                (
                    f"共 {len(segments)} 条（{segment_policy_summary(payload=payload)}）"
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


def build_video_episode_script_reference_section(
    payload: dict[str, Any],
    *,
    max_chars: int = 12000,
    max_segments: int = 96,
) -> str:
    """脚本 JSON 生成用：整片视频分析 + 固定时间格索引（对齐蓝图视频格）。"""
    markdown = build_video_episode_analysis_markdown(
        payload,
        max_chars=max(4000, max_chars - 3000),
        max_segments=max_segments,
    )
    schedule = build_video_episode_schedule_index(payload, max_rows=min(48, max_segments))
    parts = [markdown]
    if schedule:
        parts.extend(["", schedule])
    text = "\n".join(parts).strip()
    return summarize_video_episode_markdown(text, max_chars)


def summarize_video_episode_markdown(markdown: str, max_chars: int) -> str:
    text = (markdown or "").strip()
    if not text:
        return "（无整片视频分析）"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n…（整片视频分析摘要已截断）"


_UNNAMED_CHARACTER_LABEL = "剧中未明确交代"


def extract_video_episode_character_lexicon(payload: dict[str, Any]) -> dict[str, Any]:
    """从整片视频分析 JSON 汇总规范人物名与重要台词摘录。"""
    normalized = normalize_video_episode_analysis_payload(payload)
    names: set[str] = set()
    snippets: list[str] = []

    for ref in payload.get("character_references") or []:
        if not isinstance(ref, dict):
            continue
        name = str(ref.get("name") or "").strip()
        if name and name != _UNNAMED_CHARACTER_LABEL:
            names.add(name)

    for segment in normalized.get("episodic_segments") or []:
        if not isinstance(segment, dict):
            continue
        for raw_name in segment.get("involved_characters") or []:
            name = str(raw_name or "").strip()
            if name and name != _UNNAMED_CHARACTER_LABEL:
                names.add(name)

    for item in normalized.get("important_dialogues") or []:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        quote = str(item.get("quote") or "").strip()
        ts = str(item.get("timestamp") or "").strip()
        if speaker and speaker != _UNNAMED_CHARACTER_LABEL:
            names.add(speaker)
        if quote:
            label = speaker or "?"
            snippets.append(f"[{ts or '?'}] {label}：「{quote}」")

    return {"names": names, "subtitle_snippets": snippets}


def build_video_episode_character_lexicon_markdown(
    payload: dict[str, Any],
    *,
    max_chars: int = 4000,
) -> tuple[str, dict[str, Any]]:
    """生成供构思蓝图参照的整片视频分析人物索引 Markdown。"""
    empty: dict[str, Any] = {"names": set(), "subtitle_snippets": []}
    if not isinstance(payload, dict) or not payload:
        return "", empty

    lexicon = extract_video_episode_character_lexicon(payload)
    names = sorted(str(name) for name in lexicon.get("names") or set())
    snippets = list(lexicon.get("subtitle_snippets") or [])

    if len(snippets) > 1:
        target = max(8, min(len(snippets), max(1200, max_chars - 800) // 80))
        if len(snippets) <= target:
            sampled = snippets
        else:
            step = (len(snippets) - 1) / max(target - 1, 1)
            picked: list[int] = []
            for index in range(target):
                idx = min(len(snippets) - 1, int(round(index * step)))
                if idx not in picked:
                    picked.append(idx)
            sampled = [snippets[i] for i in picked]
    else:
        sampled = snippets

    lines = [
        "## 整片视频分析人物索引（只读 · 规范姓名以此为准）",
        "- **出现人物（据视频分析 involved_characters / speaker）**："
        + ("、".join(names) if names else "（未识别到具名人物）"),
        "- **人名须与上方索引及剧集对照表一致**；ASR 谐音/简称须归并为同一人",
        "- **重要台词摘录（important_dialogues）**：",
    ]
    for snippet in sampled:
        lines.append(f"  - {snippet}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n…（索引已截断）"
    return text, lexicon


def video_episode_summary_usable(summary: str) -> bool:
    text = (summary or "").strip()
    return bool(text) and text not in {"（无整片视频分析）", "（无）"}


def normalize_video_grid_range(value: str) -> str:
    return str(value or "").strip().replace("—", "-").replace(",", "")


def collect_video_episode_time_bounds(payload: dict[str, Any]) -> dict[str, Any]:
    """从整片视频分析汇总时间边界与固定时间格列表。"""
    segments = payload.get("episodic_segments") or []
    segment_ranges: list[str] = []
    segment_bounds_ms: list[tuple[int, int]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        time_range = normalize_video_grid_range(str(segment.get("time_range") or ""))
        if not time_range or "-" not in time_range:
            continue
        start_sec, end_sec = _parse_time_range_bounds(time_range)
        segment_ranges.append(time_range)
        segment_bounds_ms.append((start_sec * 1000, end_sec * 1000))

    duration = float(payload.get("video_duration_seconds") or 0)
    if segment_bounds_ms:
        min_ms = segment_bounds_ms[0][0]
        max_ms = segment_bounds_ms[-1][1]
    elif duration > 0:
        min_ms = 0
        max_ms = int(duration * 1000)
    else:
        min_ms = 0
        max_ms = 0

    return {
        "min_ms": min_ms,
        "max_ms": max_ms,
        "segment_ranges": segment_ranges,
        "segment_bounds_ms": segment_bounds_ms,
        "segment_count": len(segment_ranges),
    }


def build_video_episode_schedule_index(
    payload: dict[str, Any],
    *,
    max_rows: int = 96,
) -> str:
    """构思蓝图用：固定时间格索引表（须逐格引用）。"""
    segments = payload.get("episodic_segments") or []
    if not segments:
        return ""
    sampled = _sample_items_uniformly(segments, max_rows)
    lines = [
        "### 视频分析时间格索引（自适应场景格；上传分段边界见 coverage_warnings）",
        "| 视频格 time_range | 标题 | 人物 | 关键事件 |",
        "|---|---|---|---|",
    ]
    for segment in sampled:
        if not isinstance(segment, dict):
            continue
        time_range = normalize_video_grid_range(str(segment.get("time_range") or ""))
        title = str(segment.get("title") or "").strip()[:12]
        events = str(segment.get("key_events") or "").strip()[:32]
        chars = "、".join(segment.get("involved_characters") or [])[:20] or "—"
        lines.append(f"| `{time_range}` | {title} | {chars} | {events} |")
    if len(segments) > len(sampled):
        lines.append(
            f"\n> 全片共 **{len(segments)}** 格，上表均匀采样 **{len(sampled)}** 格；"
            "蓝图按**完整情节段**取材，主视频格须来自索引表，一场戏可跨多格，禁止自造窗口。"
        )
    return "\n".join(lines)


def is_video_grid_span_allowed(span: str, segment_ranges: list[str]) -> bool:
    """判断视频格区间是否为索引表中连续格子的合法并集（一场戏可跨多格）。"""
    normalized = normalize_video_grid_range(span)
    if not normalized or "-" not in normalized:
        return False
    allowed = {
        normalize_video_grid_range(item)
        for item in segment_ranges
        if str(item or "").strip()
    }
    if normalized in allowed:
        return True
    intervals: list[tuple[int, int]] = []
    for item in segment_ranges:
        start_sec, end_sec = _parse_time_range_bounds(normalize_video_grid_range(item))
        if end_sec > start_sec:
            intervals.append((start_sec, end_sec))
    if not intervals:
        return False
    start_sec, end_sec = _parse_time_range_bounds(normalized)
    if end_sec <= start_sec:
        return False
    pos = start_sec
    while pos < end_sec - 0.01:
        candidates = [end for start, end in intervals if start <= pos + 0.01 and end > pos + 0.01]
        if not candidates:
            return False
        pos = max(candidates)
    return pos >= end_sec - 0.01


def build_plot_blueprint_narrative_granularity_rules(
    *,
    ost1_duration_min: int = 8,
    ost1_duration_max: int = 18,
    payload: dict[str, Any] | None = None,
) -> str:
    """构思蓝图：完整情节段粒度规则（禁止按单格/单句对白碎切）。"""
    grid_label = segment_policy_summary(payload=payload)
    return (
        "## 情节段粒度（硬性 · 禁止碎切）\n"
        "- **每条 = 一段完整内容**：一场戏、一个冲突回合或一条清晰叙事线（通常 **30 秒–3 分钟** 原片跨度）\n"
        f"- **禁止**按每个 {grid_label} 各写一行；**禁止**按每条 SRT 字幕各写一行\n"
        "- **原片时间线**仅 **10–14 条**完整情节段，不是 30+ 条碎点\n"
        f"- **OST=1 字幕窗**：合并相邻对白为 **{ost1_duration_min}–{ost1_duration_max} 秒**连续区间；"
        f"单句不足 {ost1_duration_min} 秒须并入同场前后对白\n"
        "- **视频格**可跨多格：如 `00:09:40-00:10:10` 覆盖审讯室整场，起止须落在索引表连续格子上\n"
        "- **OST=0 串场**：用 1 段解说讲完该情节段因果，不要拆成多个碎点\n"
        "- **成片叙事顺序**每条 `_id` 也应是一段完整叙事单元（原声块或解说块），不是单句台词"
    )


def build_plot_blueprint_video_time_rules(
    *,
    ost1_duration_min: int = 8,
    ost1_duration_max: int = 18,
    payload: dict[str, Any] | None = None,
) -> str:
    grid_label = segment_policy_summary(payload=payload)
    return (
        "## 双时间轴对齐规则（硬性 · 精准控制）\n"
        f"- **视频格**（画面/剧情）：起止须对齐索引表连续 {grid_label}；**一场戏可跨多格**\n"
        "- **字幕窗**（对白/OST=1）：只能使用 SRT 索引中的 `HH:MM:SS,mmm-HH:MM:SS,mmm`（毫秒级）\n"
        "- **原片时间线**每条 = **一段完整情节**；格式："
        "`视频格 \\`00:01:20-00:01:50\\` · 【本段讲什么】事件摘要 · 字幕窗 \\`…\\`（无对白写「无对白」）`\n"
        f"- **OST=1** 字幕窗 **{ost1_duration_min}–{ost1_duration_max} 秒**；禁止 2–6 秒单句碎段\n"
        "- **OST=0** 取画时间须落在对应视频格跨度内；铺垫下一段 OST=1 时起点 = 字幕窗开始 − 约 10 秒\n"
        "- 禁止编造索引表与 SRT 中不存在的时间；冲突时：**台词/OST 以字幕为准，画面以视频格为准**"
    )


def build_video_episode_time_bounds_section(payload: dict[str, Any]) -> str:
    """构思蓝图用：整片视频分析时间边界、固定格索引与对齐规则。"""
    bounds = collect_video_episode_time_bounds(payload)
    duration = float(payload.get("video_duration_seconds") or 0)
    segments = payload.get("episodic_segments") or []
    if duration <= 0 and not segments:
        return ""
    duration_label = _format_timestamp(duration) if duration > 0 else "未知"
    policy_label = segment_policy_summary(payload=payload)
    lines = [
        "## 整片视频分析时间边界",
        f"- 源视频总时长：**{duration_label}**（{duration:.1f}s）",
        f"- **{policy_label}**，共 **{bounds['segment_count']}** 条",
    ]
    if segments:
        lines.append(
            f"- 首格：`{segments[0].get('time_range', '')}` · {segments[0].get('key_events', '')}"
        )
        lines.append(
            f"- 末格：`{segments[-1].get('time_range', '')}` · {segments[-1].get('key_events', '')}"
        )
    schedule = build_video_episode_schedule_index(payload)
    from app.services.short_drama_settings import get_short_drama_settings

    sd_cfg = get_short_drama_settings()
    ost1_min = int(sd_cfg.get("ost1_duration_min", 8) or 8)
    ost1_max = int(sd_cfg.get("ost1_duration_max", 18) or 18)
    granularity = build_plot_blueprint_narrative_granularity_rules(
        ost1_duration_min=ost1_min,
        ost1_duration_max=ost1_max,
        payload=payload,
    )
    rules = build_plot_blueprint_video_time_rules(
        ost1_duration_min=ost1_min,
        ost1_duration_max=ost1_max,
        payload=payload,
    )
    parts = ["\n".join(lines)]
    if schedule:
        parts.extend(["", schedule])
    parts.extend(["", granularity, "", rules])
    return "\n".join(parts)


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
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
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
    profile: str = "standard",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cfg = get_upload_transcode_profile(profile)
    width = int(cfg.get("width") or 640)
    fps = int(cfg.get("fps") or 15)
    crf = str(cfg.get("crf") or 28)
    audio_bitrate = str(cfg.get("audio_bitrate") or "48k")
    preset = str(cfg.get("preset") or "fast")
    video_filter = f"scale={width}:-2,fps={fps}"
    cmd = ["ffmpeg", "-y"]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", video_path])
    if duration_seconds and duration_seconds > 0:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            preset,
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


def _upload_master_path(work_dir: str, profile: str) -> str:
    return os.path.join(work_dir, f"upload_master_{profile}.mp4")


def _ensure_upload_master_video(
    video_path: str,
    *,
    work_dir: str,
    profile: str,
    progress_callback: Callable[[float, str], None] | None = None,
    resume: bool = False,
    checkpoint: dict[str, Any] | None = None,
) -> str:
    """将原片整段压缩为上传母版（默认 720p），分镜截取基于该文件。"""
    progress = progress_callback or (lambda _p, _m: None)
    master_path = _upload_master_path(work_dir, profile)
    prof = get_upload_transcode_profile(profile)
    width = int(prof.get("width") or 720)
    if (
        resume
        and checkpoint
        and str(checkpoint.get("upload_transcode_profile") or "") == profile
        and os.path.isfile(master_path)
        and os.path.getsize(master_path) > 1024
    ):
        size_mb = os.path.getsize(master_path) / (1024 * 1024)
        progress(12, f"复用已缓存 {width}p 压缩片 · {size_mb:.1f}MB")
        return master_path

    progress(10, f"正在将原片压缩为 {width}p（CRF {prof.get('crf')}）...")
    _transcode_for_upload(
        video_path,
        output_path=master_path,
        profile=profile,
    )
    size_mb = os.path.getsize(master_path) / (1024 * 1024)
    logger.info(f"整片视频分析：上传母版 {width}p · {size_mb:.2f}MB")
    progress(13, f"压缩完成 · {width}p · {size_mb:.1f}MB")
    return master_path


def _extract_scene_clip(
    video_path: str,
    *,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
    max_upload_mb: float,
    fallback_profile: str = "compact",
) -> None:
    """从已压缩母版按时间窗截取分镜；母版为 720p 时优先流复制，超限再降档转码。"""
    if duration_seconds < _MIN_SCENE_SECONDS:
        raise ValueError(
            f"分镜过短（{duration_seconds:.2f}s < {_MIN_SCENE_SECONDS}s）: "
            f"{_format_timestamp(start_seconds)}"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        video_path,
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb <= max_upload_mb:
        return
    try:
        os.remove(output_path)
    except OSError:
        pass
    _transcode_for_upload(
        video_path,
        output_path=output_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        profile=fallback_profile,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > max_upload_mb:
        logger.warning(
            f"分镜段仍超限 {size_mb:.1f}MB > {max_upload_mb:.1f}MB "
            f"（{_format_timestamp(start_seconds)}），仍将上传"
        )


def _plan_scene_cut_chunks(
    video_path: str,
    *,
    scene_ranges: list[str],
    max_upload_mb: float | None = None,
    work_dir: str,
    progress_callback: Callable[[float, str], None] | None = None,
    checkpoint: dict[str, Any] | None = None,
    resume: bool = False,
    on_chunk_compressed: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> list[dict[str, Any]]:
    """按预计算分镜时间窗逐段截取，每段单独上传分析。"""
    upload_cfg = get_video_episode_upload_settings()
    if max_upload_mb is None:
        max_upload_mb = float(upload_cfg.get("max_upload_mb", 24.0))

    progress = progress_callback or (lambda _p, _m: None)
    if not scene_ranges:
        raise ValueError("未检测到任何分镜时间窗")

    saved_meta = (checkpoint or {}).get("chunks_meta") or [] if resume else []
    estimated_total = max(
        int(checkpoint.get("total_chunks") or 0) if checkpoint else 0,
        len(scene_ranges),
        1,
    )
    chunks: list[dict[str, Any]] = []
    index = 0

    while index < len(saved_meta):
        meta = saved_meta[index]
        if not isinstance(meta, dict) or index >= len(scene_ranges):
            break
        chunk_path = os.path.join(work_dir, f"scene_{index:04d}.mp4")
        if not os.path.isfile(chunk_path) or os.path.getsize(chunk_path) <= 1024:
            break
        offset_seconds = float(meta.get("offset_seconds") or 0)
        duration_seconds = float(meta.get("duration_seconds") or 0)
        chunks.append(
            {
                "path": chunk_path,
                "offset_seconds": offset_seconds,
                "duration_seconds": duration_seconds,
                "time_range": scene_ranges[index],
            }
        )
        index += 1
        cut_progress = 8 + int(6 * index / max(estimated_total, index + 1))
        progress(
            cut_progress,
            f"第 {index}/{estimated_total} 镜已截取 · 复用缓存 "
            f"（{scene_ranges[index - 1]}）",
        )

    while index < len(scene_ranges):
        time_range = scene_ranges[index]
        start_seconds, end_seconds = _parse_time_range_bounds(time_range)
        duration_seconds = max(0.0, end_seconds - start_seconds)
        chunk_path = os.path.join(work_dir, f"scene_{index:04d}.mp4")
        total_display = max(estimated_total, len(scene_ranges))
        cut_progress = 8 + int(6 * index / total_display)
        progress(
            cut_progress,
            f"正在截取第 {index + 1}/{total_display} 镜 · `{time_range}` ...",
        )
        _extract_scene_clip(
            video_path,
            output_path=chunk_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            max_upload_mb=max_upload_mb,
        )
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        logger.info(
            f"整片视频分析：第 {index + 1} 镜 {time_range} / {size_mb:.2f}MB"
        )
        chunks.append(
            {
                "path": chunk_path,
                "offset_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "time_range": time_range,
            }
        )
        if on_chunk_compressed:
            on_chunk_compressed(chunks, index)
        index += 1

    progress(14, f"分镜截取完成 · 共 {len(chunks)} 镜待上传分析")
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
        previous_chunk_partial: dict[str, Any] | None = None,
        source_video_path: str = "",
        scene_cuts: list[float] | None = None,
        plot_reference: str = "",
    ) -> tuple[dict[str, Any], int]:
        chunk_duration_seconds = float(chunk.get("duration_seconds") or 0)
        chunk_offset_seconds = float(chunk.get("offset_seconds") or 0)
        preset_range = str(chunk.get("time_range") or "").strip()
        if preset_range:
            chunk_schedule = [preset_range]
        elif SEGMENT_SPLIT_POLICY == "scene_cut":
            chunk_schedule = [
                f"{_format_timestamp(chunk_offset_seconds)}-"
                f"{_format_timestamp(chunk_offset_seconds + chunk_duration_seconds)}"
            ]
        else:
            chunk_schedule = build_segment_schedule(
                chunk_duration_seconds,
                start_offset_seconds=chunk_offset_seconds,
                video_path=source_video_path,
                scene_cuts=scene_cuts,
            )
        schedule_batches = split_schedule_for_api_batches(chunk_schedule)
        reference_paths, naming_block = _prepare_chunk_reference_context(
            chunk_index=chunk_index,
            drama_title=drama_title,
            character_references=character_references,
            relationship_diagram_path=relationship_diagram_path,
        )

        merged_partial: dict[str, Any] | None = None
        max_attempts = 0

        for batch_index, batch_schedule in enumerate(schedule_batches):
            schedule_block = build_segment_schedule_prompt_block(batch_schedule)
            batch_addon = build_schedule_batch_prompt_addon(
                batch_index=batch_index,
                batch_count=len(schedule_batches),
                batch_size=len(batch_schedule),
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
                    previous_chunk_partial=previous_chunk_partial,
                )
                if SEGMENT_SPLIT_POLICY == "scene_cut":
                    prompt = (
                        f"{prompt}\n\n"
                        "## 分镜说明\n"
                        "本请求上传的视频文件 **仅包含一个分镜镜头**；"
                        "请只分析该镜画面，输出 1 条 `episodic_segments`。"
                    )
            if batch_addon:
                prompt = f"{prompt}\n\n{batch_addon}"
            plot_section = build_plot_reference_prompt_section(plot_reference)
            if plot_section.strip():
                prompt = f"{prompt}\n\n{plot_section.strip()}"

            chunk_max_tokens = estimate_video_episode_chunk_max_tokens(len(batch_schedule))
            require_summary = batch_index == 0 and chunk_index == 0

            last_error: Exception | None = None
            for attempt in range(1, _MAX_CHUNK_RETRIES + 1):
                try:
                    batch_label = (
                        f" · 输出批 {batch_index + 1}/{len(schedule_batches)}"
                        f"（{len(batch_schedule)} 窗 · max_tokens≈{chunk_max_tokens}）"
                        if len(schedule_batches) > 1
                        else f" · {len(batch_schedule)} 窗 · max_tokens≈{chunk_max_tokens}"
                    )
                    scene_time_range = str(chunk.get("time_range") or "").strip()
                    range_hint = f" · `{scene_time_range}`" if scene_time_range else ""
                    progress(
                        analyze_progress,
                        f"第 {chunk_index + 1}/{total_chunks} 镜{range_hint} · "
                        f"附带 {len(reference_paths)} 张参照图 · 调用模型中{batch_label}"
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
                        max_tokens=chunk_max_tokens,
                        scene_index=chunk_index + 1,
                        scene_total=total_chunks,
                        scene_time_range=scene_time_range,
                    )
                    parsed_partial = parse_video_episode_analysis_payload(raw)
                    issues = validate_chunk_partial(
                        parsed_partial,
                        chunk_offset_seconds=chunk_offset_seconds,
                        chunk_duration_seconds=chunk_duration_seconds,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        expected_schedule=batch_schedule,
                        require_overall_summary=require_summary,
                    )
                    if issues:
                        raise ValueError("片段约束不符合: " + " | ".join(issues[:3]))
                    merged_partial = (
                        merge_video_episode_partial_analyses([merged_partial, parsed_partial])
                        if merged_partial
                        else parsed_partial
                    )
                    max_attempts = max(max_attempts, attempt)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < _MAX_CHUNK_RETRIES:
                        progress(
                            analyze_progress,
                            f"第 {chunk_index + 1}/{total_chunks} 段"
                            + (
                                f" 批 {batch_index + 1}/{len(schedule_batches)}"
                                if len(schedule_batches) > 1
                                else ""
                            )
                            + f" 不符合约束或调用失败，重试 {attempt}/{_MAX_CHUNK_RETRIES}（{exc}）...",
                        )
                        await asyncio.sleep(min(2 * attempt, 6))
            else:
                raise ValueError(str(last_error) if last_error else "分段分析失败")

        if not merged_partial:
            raise ValueError(str(last_error) if last_error else "分段分析失败")

        final_issues = validate_chunk_partial(
            merged_partial,
            chunk_offset_seconds=chunk_offset_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            expected_schedule=chunk_schedule,
            require_overall_summary=(chunk_index == 0),
        )
        if final_issues:
            raise ValueError("片段约束不符合: " + " | ".join(final_issues[:3]))
        return merged_partial, max_attempts

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
        max_upload_mb: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        output_path: str | None = None,
        resume: bool = True,
        plot_reference: str = "",
    ) -> dict[str, Any]:
        progress = progress_callback or (lambda _p, _m: None)
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        resolved_plot_reference = (plot_reference or "").strip()

        model_name, api_key, base_url = self._resolve_model_settings(
            vision_model_name=vision_model_name,
            vision_api_key=vision_api_key,
            vision_base_url=vision_base_url,
        )
        logger.info(f"整片视频分析 {model_name} → {describe_llm_route(model_name, role='vision')}")

        progress(2, f"正在初始化 · 模型 {model_name}")

        save_path = output_path or default_video_episode_analysis_path(video_path)
        checkpoint_path = default_checkpoint_path(save_path)

        work_dir = os.path.join(utils.storage_dir(), "temp", "video_episode_upload", sanitize_video_stem(video_path))
        os.makedirs(work_dir, exist_ok=True)

        video_duration_seconds = _probe_duration_seconds(video_path)
        duration_label = _format_timestamp(video_duration_seconds)
        progress(4, f"视频时长 {duration_label}，准备按分镜切段...")

        progress(6, "正在检测场景切换并生成分段时间窗...")
        scene_settings = get_video_episode_scene_settings()
        scene_cut_mode = str(scene_settings.get("scene_cut_mode") or "environment_change").strip()
        edit_cuts = detect_edit_cut_seconds(
            video_path,
            duration_seconds=video_duration_seconds,
            threshold=float(
                scene_settings.get("scene_candidate_threshold")
                or scene_settings.get("scene_detect_threshold")
                or SCENE_DETECT_THRESHOLD
            ),
        )
        scene_cuts = detect_scene_cut_seconds(
            video_path,
            duration_seconds=video_duration_seconds,
            threshold=float(scene_settings.get("scene_detect_threshold") or SCENE_DETECT_THRESHOLD),
            scene_cut_mode=scene_cut_mode,
            candidate_threshold=float(
                scene_settings.get("scene_candidate_threshold") or SCENE_CANDIDATE_THRESHOLD
            ),
            environment_diff_threshold=float(
                scene_settings.get("scene_environment_diff_threshold")
                or SCENE_ENVIRONMENT_DIFF_THRESHOLD
            ),
            sample_before_seconds=float(
                scene_settings.get("scene_frame_sample_before_seconds")
                or SCENE_FRAME_SAMPLE_BEFORE_SECONDS
            ),
            sample_after_seconds=float(
                scene_settings.get("scene_frame_sample_after_seconds")
                or SCENE_FRAME_SAMPLE_AFTER_SECONDS
            ),
        )
        full_schedule = build_segment_schedule(
            video_duration_seconds,
            video_path=video_path,
            scene_cuts=scene_cuts,
            min_merge_seconds=float(scene_settings.get("scene_min_merge_seconds") or SCENE_MIN_MERGE_SECONDS),
            min_segment_seconds=float(
                scene_settings.get("scene_min_segment_seconds") or SCENE_MIN_SEGMENT_SECONDS
            ),
            max_scene_seconds=float(scene_settings.get("scene_max_seconds") or SCENE_MAX_SECONDS),
            scene_detect_threshold=float(
                scene_settings.get("scene_candidate_threshold")
                or scene_settings.get("scene_detect_threshold")
                or SCENE_DETECT_THRESHOLD
            ),
        )
        if scene_cut_mode == "environment_change":
            progress(
                8,
                f"硬切 {len(edit_cuts)} 处 · 场景切换 {len(scene_cuts)} 处 · "
                f"共 {len(full_schedule)} 段（{segment_policy_summary()}）",
            )
        else:
            progress(
                8,
                f"切镜 {len(scene_cuts)} 处 · 共 {len(full_schedule)} 段"
                f"（{segment_policy_summary()}）",
            )

        upload_cfg = get_video_episode_upload_settings()
        if max_upload_mb is None:
            max_upload_mb = float(upload_cfg.get("max_upload_mb", 24.0))
        upload_profile = resolve_upload_transcode_profile_name(upload_cfg)
        upload_prof = get_upload_transcode_profile(upload_profile)
        logger.info(
            f"整片视频分镜上传：{upload_prof.get('width')}p · max_upload_mb={max_upload_mb} · "
            f"分镜数={len(full_schedule)} · policy={SEGMENT_SPLIT_POLICY}"
        )

        checkpoint: dict[str, Any] | None = None
        chunk_results: dict[str, dict[str, Any]] = {}
        if resume:
            checkpoint = load_video_episode_checkpoint(checkpoint_path)
            if checkpoint and is_checkpoint_source_compatible(
                checkpoint,
                video_path=video_path,
                video_duration_seconds=video_duration_seconds,
            ):
                chunk_results = {
                    str(key): value
                    for key, value in (checkpoint.get("chunk_results") or {}).items()
                    if isinstance(value, dict)
                }
                meta_len = len(checkpoint.get("chunks_meta") or [])
                summary = summarize_checkpoint_progress(
                    checkpoint,
                    int(checkpoint.get("total_chunks") or 0) or max(meta_len, 1),
                )
                if meta_len or summary["completed"] or summary["failed"]:
                    progress(
                        5,
                        f"续跑补全 · 已截取 {meta_len} 镜"
                        + (
                            f" · 已分析 {summary['completed']}/{summary['total']} 段"
                            if summary["completed"] or summary["failed"]
                            else ""
                        )
                        + (f" · 失败 {summary['failed']} 段" if summary["failed"] else ""),
                    )
            elif checkpoint:
                reason = describe_checkpoint_incompatibility(
                    checkpoint,
                    video_path=video_path,
                    video_duration_seconds=video_duration_seconds,
                )
                logger.warning(f"整片视频分析检查点不可用（{reason}），将从头分析")
                progress(5, f"检查点不可用（{reason}），将从头分析")
                checkpoint = None
                chunk_results = {}
        else:
            checkpoint = None
            chunk_results = {}
            if os.path.isfile(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except OSError as exc:
                    logger.warning(f"无法删除旧检查点 {checkpoint_path}: {exc}")

        def _persist_checkpoint(*, total_chunks: int | None = None) -> None:
            save_video_episode_checkpoint(
                checkpoint_path,
                {
                    "artifact_version": VIDEO_EPISODE_ANALYSIS_ARTIFACT_VERSION,
                    "video_path": os.path.abspath(video_path),
                    "video_duration_seconds": round(video_duration_seconds, 3),
                    "upload_transcode_profile": upload_profile,
                    "total_chunks": total_chunks if total_chunks is not None else len(chunks),
                    "chunks_meta": _chunks_meta(chunks),
                    "chunk_results": chunk_results,
                    "updated_at": datetime.now().isoformat(),
                },
            )

        upload_source_path = _ensure_upload_master_video(
            video_path,
            work_dir=work_dir,
            profile=upload_profile,
            progress_callback=progress,
            resume=bool(resume and checkpoint),
            checkpoint=checkpoint,
        )

        chunks: list[dict[str, Any]] = []

        def _on_chunk_compressed(compressed_chunks: list[dict[str, Any]], _index: int) -> None:
            nonlocal chunks
            chunks = list(compressed_chunks)
            _persist_checkpoint(total_chunks=max(int((checkpoint or {}).get("total_chunks") or 0), len(chunks)))

        chunks = _plan_scene_cut_chunks(
            upload_source_path,
            scene_ranges=full_schedule,
            max_upload_mb=max_upload_mb,
            work_dir=work_dir,
            progress_callback=progress,
            checkpoint=checkpoint,
            resume=bool(resume and checkpoint),
            on_chunk_compressed=_on_chunk_compressed,
        )
        total_chunks = len(chunks)
        _persist_checkpoint(total_chunks=total_chunks)
        logger.info(
            f"整片视频分析：时长 {duration_label}，"
            f"共 {total_chunks} 镜上传（max_upload_mb={max_upload_mb}）"
        )
        ref_count = len(character_references or [])
        if ref_count:
            progress(
                15,
                f"上传准备完成 · 共 {total_chunks} 段 · 已加载 {ref_count} 张人物头像参照",
            )
        else:
            progress(15, f"分镜准备完成 · 共 {total_chunks} 镜 · 开始调用视觉模型")

        from app.services.llm.openai_compatible_provider import OpenAICompatibleVisionProvider

        provider = OpenAICompatibleVisionProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )

        if resume and checkpoint and not is_checkpoint_compatible(
            checkpoint,
            video_path=video_path,
            video_duration_seconds=video_duration_seconds,
            chunks=chunks,
        ):
            reason = describe_checkpoint_incompatibility(
                checkpoint,
                video_path=video_path,
                video_duration_seconds=video_duration_seconds,
                chunks=chunks,
            )
            logger.warning(f"分镜截取完成后检查点边界不一致（{reason}），模型结果将不续用")
            progress(15, f"分镜边界变化（{reason}），已截取片段仍可用，模型分析从头校验")
            chunk_results = {}

        summary = summarize_checkpoint_progress(checkpoint, total_chunks) if checkpoint else {
            "completed": 0,
            "failed": 0,
            "pending": total_chunks,
            "total": total_chunks,
        }
        if summary["completed"] or summary["failed"]:
            progress(
                15,
                f"续跑补全 · 已完成 {summary['completed']}/{total_chunks} 段"
                + (f"，失败 {summary['failed']} 段" if summary["failed"] else "")
                + (f"，待处理 {summary['pending']} 段" if summary["pending"] else ""),
            )

        last_successful_partial: dict[str, Any] | None = None
        for index, chunk in enumerate(chunks):
            chunk_key = str(index)
            existing = chunk_results.get(chunk_key)
            if (
                isinstance(existing, dict)
                and existing.get("status") == "completed"
                and existing.get("partial")
            ):
                last_successful_partial = existing["partial"]
                segment_count = len((existing.get("partial") or {}).get("episodic_segments") or [])
                done_progress = 16 + int(68 * (index + 1) / max(total_chunks, 1))
                progress(
                    done_progress,
                    f"第 {index + 1}/{total_chunks} 镜已缓存 · 跳过 · 本镜 {segment_count} 条情节片段",
                )
                continue

            chunk_duration_seconds = float(chunk.get("duration_seconds") or 0)
            chunk_offset_seconds = float(chunk.get("offset_seconds") or 0)
            chunk_start_label = _format_timestamp(chunk_offset_seconds)
            chunk_end_label = _format_timestamp(chunk_offset_seconds + chunk_duration_seconds)
            analyze_progress = 16 + int(68 * index / max(total_chunks, 1))
            chunk_time_range = str(chunk.get("time_range") or "").strip()
            range_label = chunk_time_range or f"{chunk_start_label}-{chunk_end_label}"
            progress(
                analyze_progress,
                f"第 {index + 1}/{total_chunks} 镜 · `{range_label}` · "
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
                    previous_chunk_partial=last_successful_partial,
                    source_video_path=video_path,
                    scene_cuts=scene_cuts,
                    plot_reference=resolved_plot_reference,
                )
                chunk_results[chunk_key] = {
                    "status": "completed",
                    "partial": parsed_partial,
                    "retry_count": retry_count,
                }
                last_successful_partial = parsed_partial
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
                    f"第 {index + 1}/{total_chunks} 镜失败（已保存进度，可稍后补全）: {exc}",
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
                f"第 {index + 1}/{total_chunks} 镜完成 · 本镜 {segment_count} 条情节片段{retry_note}",
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
            failed_label = "、".join(str(i + 1) for i in failed_chunk_indices) or "全部"
            raise ValueError(
                "所有上传分段均未成功完成分析（失败段: "
                f"{failed_label}）。常见原因：网关返回空响应、输出 JSON 过长被截断。"
                "已保留检查点，请稍后点击「补全未完成分析」重试，或更换视觉模型/网关。"
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
        analysis["episodic_segments"] = enforce_episodic_segment_schedule(
            analysis.get("episodic_segments") or [],
            full_schedule,
        )
        repaired_segments, continuity_warnings = repair_upload_chunk_boundary_continuity(
            analysis.get("episodic_segments") or [],
            chunks_meta=_chunks_meta(chunks),
        )
        analysis["episodic_segments"] = repaired_segments
        merged_segment_count = len(analysis.get("episodic_segments") or [])
        progress(
            92,
            f"正在对齐分镜时间窗（共 {merged_segment_count} 条）...",
        )
        coverage_warnings = validate_episodic_segments(
            analysis.get("episodic_segments") or [],
            video_duration_seconds=video_duration_seconds,
            expected_time_ranges=full_schedule,
            segment_split_policy=SEGMENT_SPLIT_POLICY,
        )
        if continuity_warnings:
            logger.warning(
                "整片视频分析连续性告警: " + " | ".join(continuity_warnings[:5])
            )
            for item in continuity_warnings:
                if item not in coverage_warnings:
                    coverage_warnings.append(item)
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
            "plot_reference": resolved_plot_reference,
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
            "chunks_meta": _chunks_meta(chunks),
            "segment_min_seconds": SCENE_MIN_SEGMENT_SECONDS,
            "segment_max_seconds": SCENE_MAX_SECONDS,
            "segment_split_policy": SEGMENT_SPLIT_POLICY,
            "scene_cut_count": len(scene_cuts),
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
