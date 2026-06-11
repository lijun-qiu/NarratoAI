#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧每秒视觉索引（frame_timeline）：供视频分析推断说话人与场景参照。"""

from __future__ import annotations

import re
from typing import Any

from app.services.documentary.frame_timeline_sampling import (
    _timestamp_to_ms,
)

FRAME_TIMELINE_CORE_FIELDS = (
    "timestamp",
    "title",
    "scene",
    "characters",
    "burned_in_subtitle",
    "visual_cue",
)

_CHARACTER_SPLIT_RE = re.compile(r"[、,，/]")
_GENDER_SUFFIX_RE = re.compile(r"[\(（][男女][\)）]$")


def _strip_character_gender_suffix(name: str) -> str:
    return _GENDER_SUFFIX_RE.sub("", (name or "").strip()).strip()


def _normalize_characters(raw: Any) -> list[str]:
    names: list[str] = []
    if isinstance(raw, str):
        names = [part.strip() for part in _CHARACTER_SPLIT_RE.split(raw) if part.strip()]
    elif isinstance(raw, list):
        names = [str(name).strip() for name in raw if str(name).strip()]
    cleaned = [_strip_character_gender_suffix(name) for name in names]
    return sorted({name for name in cleaned if name})


def normalize_frame_timeline_entry(
    entry: dict[str, Any],
    *,
    default_timestamp: str = "",
) -> dict[str, Any]:
    """规范单条 frame_timeline 记录（仅保留核心字段）。"""
    if not isinstance(entry, dict):
        entry = {}
    timestamp = str(entry.get("timestamp") or default_timestamp or "").strip()
    burned = str(
        entry.get("burned_in_subtitle")
        or entry.get("subtitle_text")
        or entry.get("on_screen_subtitle")
        or ""
    ).strip()
    return {
        "timestamp": timestamp,
        "title": str(entry.get("title") or "").strip(),
        "scene": str(entry.get("scene") or entry.get("observation") or "").strip(),
        "characters": _normalize_characters(entry.get("characters")),
        "burned_in_subtitle": burned,
        "visual_cue": str(entry.get("visual_cue") or "").strip(),
    }


def timeline_entry_to_frame_observation(entry: dict[str, Any]) -> dict[str, Any]:
    """将 frame_timeline 条目转为兼容旧链路的 frame_observations 结构。"""
    scene = str(entry.get("scene") or "").strip()
    visual = str(entry.get("visual_cue") or "").strip()
    observation = scene
    if visual and visual not in scene:
        observation = f"{scene}，{visual}" if scene else visual
    burned = str(entry.get("burned_in_subtitle") or "").strip()
    payload: dict[str, Any] = {
        "timestamp": str(entry.get("timestamp") or "").strip(),
        "observation": observation,
        "burned_in_subtitle": burned,
        "has_burned_in_subtitle": bool(burned),
    }
    characters = entry.get("characters")
    if isinstance(characters, list) and characters:
        payload["characters"] = list(characters)
    return payload


def frame_observation_to_timeline_entry(observation: dict[str, Any]) -> dict[str, Any]:
    """从 frame_observations 反推 frame_timeline 条目。"""
    if not isinstance(observation, dict):
        return normalize_frame_timeline_entry({})
    scene = str(observation.get("scene") or observation.get("observation") or "").strip()
    scene = re.sub(r"^\[(?:远景|全景|中景|近景|特写|大特写)\]\s*", "", scene)
    scene = re.sub(r"，?烧录字幕:.*$", "", scene).strip()
    visual = str(observation.get("visual_cue") or "").strip()
    if not visual:
        obs_text = str(observation.get("observation") or "")
        if "，" in obs_text and obs_text != scene:
            tail = obs_text.split("，", 1)[-1].strip()
            if tail and tail != scene:
                visual = tail
    return normalize_frame_timeline_entry(
        {
            "timestamp": observation.get("timestamp"),
            "title": observation.get("title") or observation.get("segment_title") or "",
            "scene": scene,
            "characters": observation.get("characters"),
            "burned_in_subtitle": observation.get("burned_in_subtitle"),
            "visual_cue": visual,
        },
        default_timestamp=str(observation.get("timestamp") or ""),
    )


def _iter_timeline_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ts: set[str] = set()

    def add_entry(item: dict[str, Any]) -> None:
        normalized = normalize_frame_timeline_entry(item)
        ts = normalized.get("timestamp") or ""
        if ts and ts in seen_ts:
            return
        if ts:
            seen_ts.add(ts)
        merged.append(normalized)

    for batch in data.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for item in batch.get("frame_timeline") or []:
            if isinstance(item, dict):
                add_entry(item)
    for item in data.get("frame_timeline") or []:
        if isinstance(item, dict):
            add_entry(item)
    if merged:
        merged.sort(key=lambda row: _timestamp_to_ms(str(row.get("timestamp") or "")))
        return merged

    for batch in data.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for item in batch.get("frame_observations") or []:
            if isinstance(item, dict):
                add_entry(frame_observation_to_timeline_entry(item))
    for item in data.get("frame_observations") or []:
        if isinstance(item, dict):
            add_entry(frame_observation_to_timeline_entry(item))

    merged.sort(key=lambda row: _timestamp_to_ms(str(row.get("timestamp") or "")))
    return merged


def extract_frame_timeline_from_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(artifact, dict):
        return []
    return _iter_timeline_dicts(artifact)


def build_timeline_batch_summary(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    parts: list[str] = []
    last_title = ""
    for entry in entries:
        ts = str(entry.get("timestamp") or "").strip()
        title = str(entry.get("title") or entry.get("scene") or "").strip()
        if not title or title == last_title:
            continue
        label = ts.split(",")[0] if ts else "?"
        parts.append(f"{label} {title}")
        last_title = title
    return " → ".join(parts[:12])


def build_frame_timeline_chunk_prompt_block(
    artifact: dict[str, Any],
    *,
    start_offset_seconds: float = 0,
    end_offset_seconds: float | None = None,
    pad_seconds: float = 2.0,
    max_entries: int = 150,
    max_chars: int = 14000,
) -> str:
    """为视频分析 chunk 生成只读 frame_timeline 参照块（Markdown 表格）。"""
    timeline = extract_frame_timeline_from_artifact(artifact)
    if not timeline:
        return ""

    pad_ms = int(max(0.0, pad_seconds) * 1000)
    start_ms = max(0, int(start_offset_seconds * 1000) - pad_ms)
    end_ms = (
        int(end_offset_seconds * 1000) + pad_ms
        if end_offset_seconds is not None
        else None
    )

    filtered: list[dict[str, Any]] = []
    for entry in timeline:
        ts_ms = _timestamp_to_ms(str(entry.get("timestamp") or ""))
        if ts_ms < start_ms:
            continue
        if end_ms is not None and ts_ms > end_ms:
            continue
        filtered.append(entry)

    if not filtered:
        return ""

    if len(filtered) > max_entries:
        step = (len(filtered) - 1) / max(max_entries - 1, 1)
        picked: list[dict[str, Any]] = []
        for index in range(max_entries):
            idx = min(len(filtered) - 1, int(round(index * step)))
            if not picked or picked[-1] is not filtered[idx]:
                picked.append(filtered[idx])
        filtered = picked

    lines = [
        "## 抽帧视觉索引 frame_timeline（只读 · 说话人推理第一依据）",
        "- 本表由**抽帧+头像对比**生成；**不表示**硬字幕已归属说话人",
        "- 推断 `important_dialogues.speaker` 时：对齐台词时间 → 查本表 `characters` + `visual_cue`",
        "- 嘴型/手势发言 vs 聆听/反应镜须分开；过肩镜头字幕常属背对镜头者",
        "",
        "| 时间 | 标题 | 场景 | 人物 | 硬字幕 | 视觉线索 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in filtered:
        ts = str(entry.get("timestamp") or "").strip()
        short_ts = ts.split(",")[0] if ts else "?"
        title = str(entry.get("title") or "—").strip() or "—"
        scene = str(entry.get("scene") or "—").strip() or "—"
        chars = "、".join(entry.get("characters") or []) or "—"
        subtitle = str(entry.get("burned_in_subtitle") or "—").strip() or "—"
        visual = str(entry.get("visual_cue") or "—").strip() or "—"
        for col in (title, scene, subtitle, visual):
            if len(col) > 48:
                pass
        title = title[:36] + "…" if len(title) > 36 else title
        scene = scene[:40] + "…" if len(scene) > 40 else scene
        subtitle = subtitle[:40] + "…" if len(subtitle) > 40 else subtitle
        visual = visual[:32] + "…" if len(visual) > 32 else visual
        lines.append(f"| `{short_ts}` | {title} | {scene} | {chars} | {subtitle} | {visual} |")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 24].rstrip() + "\n…（timeline 已截断）"
    return text


def attach_frame_timeline_to_artifact(artifact: dict[str, Any]) -> None:
    """落盘前合并各批次 frame_timeline，并写入顶层索引。"""
    if not isinstance(artifact, dict):
        return

    timeline: list[dict[str, Any]] = []
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_timeline: list[dict[str, Any]] = []
        for item in batch.get("frame_timeline") or []:
            if isinstance(item, dict):
                normalized = normalize_frame_timeline_entry(item)
                batch_timeline.append(normalized)
                timeline.append(normalized)
        if not batch_timeline:
            for obs in batch.get("frame_observations") or []:
                if isinstance(obs, dict):
                    normalized = frame_observation_to_timeline_entry(obs)
                    batch_timeline.append(normalized)
                    timeline.append(normalized)
        if batch_timeline:
            batch["frame_timeline"] = batch_timeline

    if not timeline:
        timeline = extract_frame_timeline_from_artifact(artifact)

    timeline.sort(key=lambda row: _timestamp_to_ms(str(row.get("timestamp") or "")))
    artifact["frame_timeline"] = timeline
    if timeline and not artifact.get("output_mode"):
        if str(artifact.get("frame_output_mode") or "") == "slim_timeline":
            artifact["output_mode"] = "slim_timeline"


def _slim_timeline_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """仅保留 frame_timeline 核心字段（无 batch 元数据）。"""
    normalized = normalize_frame_timeline_entry(entry)
    return {key: normalized[key] for key in FRAME_TIMELINE_CORE_FIELDS}


def filter_timeline_characters_to_references(artifact: dict[str, Any]) -> None:
    """Slim：剔除不在已选参照图名单内的规范姓名（防模型凭字幕/剧情臆测）。"""
    if not isinstance(artifact, dict):
        return
    ref_names = {
        str(item.get("name") or "").strip()
        for item in (artifact.get("character_references") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    if not ref_names:
        return
    for entry in artifact.get("frame_timeline") or []:
        if not isinstance(entry, dict):
            continue
        chars = entry.get("characters")
        if not isinstance(chars, list) or not chars:
            continue
        kept: list[str] = []
        for name in chars:
            canonical = _strip_character_gender_suffix(str(name).strip())
            if canonical in ref_names:
                kept.append(canonical)
        entry["characters"] = sorted(set(kept))


def populate_timeline_characters_from_text(artifact: dict[str, Any]) -> None:
    """已弃用：slim 模式 characters 仅来自模型面孔匹配，不从 scene/visual_cue 文本回填。"""
    return


def finalize_slim_timeline_artifact(
    artifact: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Slim 落盘：合并各批次为单一 frame_timeline，去掉 scene_segments / frame_observations 等冗余副本。
    """
    from app.services.documentary.documentary_settings import get_documentary_settings

    if not isinstance(artifact, dict):
        return artifact

    cfg = settings or get_documentary_settings()
    attach_frame_timeline_to_artifact(artifact)
    filter_timeline_characters_to_references(artifact)

    timeline: list[dict[str, Any]] = []
    seen_ts: set[str] = set()
    for entry in artifact.get("frame_timeline") or []:
        if not isinstance(entry, dict):
            continue
        slim = _slim_timeline_entry(entry)
        ts = str(slim.get("timestamp") or "").strip()
        if ts and ts in seen_ts:
            continue
        if ts:
            seen_ts.add(ts)
        timeline.append(slim)

    timeline.sort(key=lambda row: _timestamp_to_ms(str(row.get("timestamp") or "")))
    artifact["frame_timeline"] = timeline
    artifact["output_mode"] = "slim_timeline"
    artifact["frame_output_mode"] = "slim_timeline"

    slim_batches: list[dict[str, Any]] = []
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        payload: dict[str, Any] = {
            "batch_index": batch.get("batch_index"),
            "status": batch.get("status"),
            "time_range": batch.get("time_range"),
        }
        error_message = str(batch.get("error_message") or "").strip()
        if error_message:
            payload["error_message"] = error_message
        vision_model = str(batch.get("vision_model_used") or "").strip()
        if vision_model:
            payload["vision_model_used"] = vision_model
        if str(batch.get("status") or "").lower() != "success":
            frame_files = batch.get("frame_files")
            if isinstance(frame_files, list) and frame_files:
                payload["frame_files"] = list(frame_files)
        slim_batches.append(payload)
    artifact["batches"] = slim_batches

    for key in (
        "scene_segments",
        "frame_observations",
        "video_segment_overview",
    ):
        artifact.pop(key, None)

    if not cfg.get("compress_frame_analysis_on_save", False):
        for batch in artifact.get("batches") or []:
            if isinstance(batch, dict):
                batch.pop("raw_response", None)

    return artifact
