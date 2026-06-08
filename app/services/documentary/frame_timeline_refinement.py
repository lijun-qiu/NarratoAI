#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧结果时间轴精炼：逐帧观察 → 可读的 scene_segments / 批次摘要。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.frame_timeline_sampling import (
    _SCENE_LABEL_FROM_TEXT_RE,
    infer_scene_label_from_segment,
)

_FRAME_OBS_PREFIX_RE = re.compile(
    r"^\[(?:远景|全景|中景|近景|特写|大特写)\]\s*([^，,、]+)"
)
_EXTERIOR_LOCATION_HINTS = ("停车场", "废弃", "室外", "广场", "街道", "码头", "天台", "楼顶", "甩尾", "侧滑", "行驶")
_ROOF_HINTS = ("车顶", "车体上方", "趴伏", "抓边", "趴车")
_INTERIOR_MISLABEL = "车内"


def infer_location_from_frame_observation(text: str) -> str:
    """从 frame_observations.observation 提取地点短语。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = _FRAME_OBS_PREFIX_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    match = _SCENE_LABEL_FROM_TEXT_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    lead = re.match(r"^([\u4e00-\u9fff]{2,12})[，,]", cleaned)
    if lead:
        return lead.group(1).strip()
    return ""


def normalize_location_bucket(label: str) -> str:
    """将地点归到分段用的桶（同桶连续帧合并为一条 segment）。"""
    text = (label or "").strip()
    if not text:
        return "未知"
    if any(token in text for token in _ROOF_HINTS):
        return "车顶"
    if "车内" in text:
        return "车内"
    if "车外" in text or "车旁" in text:
        return "车外"
    if "停车场" in text or "废弃" in text:
        return "停车场"
    if "审讯" in text:
        return "审讯室"
    if "天台" in text or "楼顶" in text:
        return "楼顶天台"
    return text


def count_distinct_location_buckets(frame_observations: list[dict[str, Any]]) -> int:
    buckets: set[str] = set()
    for frame in frame_observations:
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "").strip()
        if not obs:
            continue
        bucket = normalize_location_bucket(infer_location_from_frame_observation(obs))
        if bucket != "未知":
            buckets.add(bucket)
    return len(buckets)


def correct_vehicle_interior_mislabels_in_frames(frame_observations: list[dict[str, Any]]) -> int:
    """
    追逐/停车场段落中，前后均为室外停车场景、中间特写误标「车内」时改为「车顶」。
    仅改 observation 文本，不猜人名。
    """
    if len(frame_observations) < 3:
        return 0

    buckets = [
        normalize_location_bucket(infer_location_from_frame_observation(str(item.get("observation") or "")))
        for item in frame_observations
    ]
    exterior = {"停车场", "车外", "车顶"}

    def _nearest_exterior_bucket(start: int, step: int) -> str:
        index = start
        while 0 <= index < len(buckets):
            if buckets[index] in exterior:
                return buckets[index]
            if buckets[index] not in {"", "未知", "车内"}:
                return buckets[index]
            index += step
        return ""

    corrected = 0
    for index, frame in enumerate(frame_observations):
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "")
        if _INTERIOR_MISLABEL not in obs:
            continue
        prev_exterior = _nearest_exterior_bucket(index - 1, -1)
        next_exterior = _nearest_exterior_bucket(index + 1, 1)
        if prev_exterior in exterior and next_exterior in exterior:
            frame["observation"] = obs.replace(_INTERIOR_MISLABEL, "车顶")
            buckets[index] = "车顶"
            corrected += 1
    return corrected


def _timestamp_range_from_frames(frames: list[dict[str, Any]]) -> str:
    if not frames:
        return ""
    start = str(frames[0].get("timestamp") or "").strip()
    end = str(frames[-1].get("timestamp") or "").strip()
    if start and end:
        return f"{start}-{end}" if start != end else start
    return start or end


def _compact_observations(observations: list[str]) -> str:
    cleaned = [item.strip() for item in observations if item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]}；{cleaned[1]}"
    return f"{cleaned[0]}；…；{cleaned[-1]}"


def _action_from_observations(observations: list[str]) -> str:
    actions: list[str] = []
    for obs in observations:
        text = obs.strip()
        if not text:
            continue
        body = re.sub(r"^\[[^\]]+\]\s*", "", text)
        actions.append(body)
    return " → ".join(actions[:4]) + ("…" if len(actions) > 4 else "")


def build_scene_segments_from_frame_observations(
    frame_observations: list[dict[str, Any]],
    *,
    batch_index: int = 0,
    time_range: str = "",
) -> list[dict[str, Any]]:
    """按连续相同地点桶，将逐帧观察合并为多条 scene_segments。"""
    frames = [item for item in frame_observations if isinstance(item, dict)]
    if len(frames) < 2:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bucket = ""

    for frame in frames:
        obs = str(frame.get("observation") or "").strip()
        bucket = normalize_location_bucket(infer_location_from_frame_observation(obs)) if obs else "未知"
        if current and bucket != current_bucket:
            groups.append(current)
            current = []
        current_bucket = bucket
        current.append(frame)
    if current:
        groups.append(current)

    if len(groups) <= 1:
        return []

    segments: list[dict[str, Any]] = []
    for group in groups:
        observations = [str(item.get("observation") or "").strip() for item in group]
        observations = [item for item in observations if item]
        if not observations:
            continue
        first_obs = observations[0]
        scene = infer_scene_label_from_segment({"observation": first_obs, "action": first_obs})
        if not scene:
            scene = normalize_location_bucket(infer_location_from_frame_observation(first_obs))
        timestamp = _timestamp_range_from_frames(group)
        if not timestamp and time_range:
            timestamp = time_range
        segments.append(
            {
                "batch_index": batch_index,
                "timestamp": timestamp,
                "scene": scene or "未知",
                "observation": _compact_observations(observations),
                "action": _action_from_observations(observations),
                "emotion": "紧张" if any(token in first_obs for token in ("追逐", "奔跑", "甩尾", "持枪")) else "",
                "key_visual": first_obs,
                "edit_role": "动作" if any(token in first_obs for token in ("奔跑", "甩尾", "持枪", "趴")) else "定场",
                "importance": "高" if any(token in " ".join(observations) for token in ("持枪", "甩尾", "追捕")) else "中",
            }
        )
    return segments


def build_batch_timeline_summary(frame_observations: list[dict[str, Any]]) -> str:
    """按时间顺序生成批次事件链，便于仅读 JSON 即可还原「发生了什么」。"""
    parts: list[str] = []
    for frame in frame_observations:
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "").strip()
        if not obs:
            continue
        ts = str(frame.get("timestamp") or "").strip()
        short_ts = ts.split(",")[0][-8:] if ts else ""
        body = re.sub(r"^\[[^\]]+\]\s*", "", obs)
        label = f"{short_ts} {body}" if short_ts else body
        if parts and parts[-1] == label:
            continue
        parts.append(label)
    if not parts:
        return ""
    if len(parts) == 1:
        return f"本批次：{parts[0]}"
    return "本批次：" + " → ".join(parts)


def should_rebuild_segments_from_frames(
    scene_segments: list[dict[str, Any]],
    frame_observations: list[dict[str, Any]],
) -> bool:
    frames = [item for item in frame_observations if isinstance(item, dict)]
    if len(frames) < 3:
        return False
    distinct = count_distinct_location_buckets(frames)
    if distinct < 2:
        return False
    if len(scene_segments) <= 1:
        return True
    return False


def refine_batch_from_frame_observations(
    scene_segments: list[dict[str, Any]],
    frame_observations: list[dict[str, Any]],
    *,
    batch_index: int = 0,
    time_range: str = "",
    overall_summary: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """精炼单批次：修正空间误判、按逐帧重建 segments、生成时间轴摘要。"""
    frames = [dict(item) for item in frame_observations if isinstance(item, dict)]
    mislabel_fixes = correct_vehicle_interior_mislabels_in_frames(frames)
    segments = list(scene_segments)

    if should_rebuild_segments_from_frames(segments, frames):
        rebuilt = build_scene_segments_from_frame_observations(
            frames,
            batch_index=batch_index,
            time_range=time_range,
        )
        if len(rebuilt) >= 2:
            segments = rebuilt
            logger.info(
                f"抽帧 batch #{batch_index}：已由 {len(frames)} 条逐帧观察重建为 {len(rebuilt)} 条 scene_segments"
            )

    timeline_summary = build_batch_timeline_summary(frames)
    if mislabel_fixes:
        logger.info(f"抽帧 batch #{batch_index}：已修正 {mislabel_fixes} 条「车内→车顶」空间误判")

    if timeline_summary and (
        not (overall_summary or "").strip()
        or count_distinct_location_buckets(frames) >= 2
        or len(segments) >= 2
    ):
        summary = timeline_summary
    else:
        summary = (overall_summary or "").strip() or timeline_summary

    return segments, frames, summary
