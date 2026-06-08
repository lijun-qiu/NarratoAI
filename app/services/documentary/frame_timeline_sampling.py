#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧 JSON 转 Markdown 与时间轴均匀采样（无重型依赖，避免循环导入）。"""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Callable

from loguru import logger

_NAMED_CHARACTER_RE = re.compile(r"([\u4e00-\u9fffA-Za-z·]{2,8})\(([男女])\)")


def _timestamp_to_ms(timestamp: str) -> int:
    text = (timestamp or "").strip()
    try:
        if "," in text:
            time_part, ms_part = text.split(",", 1)
            milliseconds = int(ms_part)
        else:
            time_part = text
            milliseconds = 0
        parts = [int(part) for part in time_part.split(":") if part]
        while len(parts) < 3:
            parts.insert(0, 0)
        hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
        return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds
    except Exception:
        return 0


def parse_timestamp_range_ms(time_range: str) -> tuple[int, int]:
    text = (time_range or "").strip()
    if "-" not in text:
        ms = _timestamp_to_ms(text)
        return ms, ms
    start_text, end_text = text.split("-", 1)
    start_ms = _timestamp_to_ms(start_text.strip())
    end_ms = _timestamp_to_ms(end_text.strip())
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    return start_ms, end_ms


def _segment_subtitle_display(segment: dict) -> str:
    """展示用字幕文本（优先 subtitle 字段，兼容旧 JSON 的 subtitle_entries）。"""
    text = str(segment.get("subtitle") or "").strip()
    if text:
        return text
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list):
        parts = [
            str(item.get("text") or "").strip()
            for item in entries
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        return "；".join(parts)
    return ""


def format_scene_segment(
    segment: dict,
    index: int,
    *,
    env_context: dict[str, str] | None = None,
) -> str:
    display = resolve_segment_display_fields(segment, env_context)
    if env_context is not None:
        update_segment_environment_context(env_context, segment)

    timestamp = segment.get("timestamp", "")
    scene = display.get("scene", "")
    characters = segment.get("characters") or []
    if isinstance(characters, list):
        characters_text = "、".join(str(name) for name in characters if str(name).strip())
    else:
        characters_text = str(characters)
    lines = [f"## 场景 {index}", f"- 时间：{timestamp}"]
    if scene:
        lines.append(f"- 场景：{scene}")
    if characters_text:
        lines.append(f"- 人物：{characters_text}")
    observation = str(display.get("observation") or segment.get("observation") or "").strip()
    if observation:
        lines.append(f"- 观察：{observation}")
    for label, key in (
        ("动作", "action"),
        ("情绪", "emotion"),
        ("关键视觉", "key_visual"),
        ("音效/原声", "audio_cue"),
        ("重要度", "importance"),
    ):
        if key in {"emotion", "key_visual"}:
            value = str(display.get(key) or "").strip()
        else:
            value = str(segment.get(key) or "").strip()
        if value:
            lines.append(f"- {label}：{value}")
    subtitle_text = _segment_subtitle_display(segment)
    if subtitle_text:
        lines.append(f"- 字幕：{subtitle_text}")
    return "\n".join(lines) + "\n\n"


def collect_scene_segments_from_analysis(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        return []

    top_level = data.get("scene_segments")
    if isinstance(top_level, list) and top_level:
        return normalize_scene_segments(
            [segment for segment in top_level if isinstance(segment, dict)]
        )

    segments: list[dict] = []
    batches = data.get("batches")
    if not isinstance(batches, list):
        return segments

    def batch_sort_key(batch: dict) -> tuple[int, int]:
        time_range = str(batch.get("time_range") or "")
        start = time_range.split("-", 1)[0].strip()
        start_ms, _ = parse_timestamp_range_ms(start)
        return start_ms, int(batch.get("batch_index", 0) or 0)

    for batch in sorted(
        (item for item in batches if isinstance(item, dict)),
        key=batch_sort_key,
    ):
        batch_segments = batch.get("scene_segments") or []
        if batch_segments:
            segments.extend(seg for seg in batch_segments if isinstance(seg, dict))
            continue
        time_range = str(batch.get("time_range") or "").strip()
        if not time_range:
            continue
        summary = (
            batch.get("overall_activity_summary")
            or batch.get("summary")
            or batch.get("fallback_summary")
            or ""
        )
        segments.append(
            {
                "timestamp": time_range,
                "observation": str(summary).strip(),
            }
        )
    return normalize_scene_segments(segments)


def _segment_time_bounds(segment: dict) -> tuple[int, int]:
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list) and entries:
        starts: list[str] = []
        ends: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            end = str(item.get("end") or "").strip()
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        if starts and ends:
            return parse_timestamp_range_ms(f"{starts[0]}-{ends[-1]}")
    time_text = str(segment.get("timestamp") or "").strip()
    return parse_timestamp_range_ms(time_text)


def _segment_timestamp_bounds(segment: dict) -> tuple[int, int]:
    """仅按 segment.timestamp 解析时间范围（去重分组用）。"""
    return parse_timestamp_range_ms(str(segment.get("timestamp") or "").strip())


def _segment_richness_score(segment: dict) -> int:
    score = 0
    for key in ("scene", "observation", "action", "emotion", "key_visual", "subtitle"):
        score += len(str(segment.get(key) or "").strip())
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list):
        score += min(len(entries), 5) * 20
    return score


def _segment_subtitle_alignment_score(segment: dict) -> int:
    """字幕中心落在 timestamp 窗口内时得分更高。"""
    entries = segment.get("subtitle_entries")
    if not isinstance(entries, list) or not entries:
        return 0
    starts_ms: list[int] = []
    ends_ms: list[int] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        start = str(item.get("start") or "").strip()
        end = str(item.get("end") or "").strip()
        if start:
            starts_ms.append(_timestamp_to_ms(start))
        if end:
            ends_ms.append(_timestamp_to_ms(end))
    if not starts_ms or not ends_ms:
        return 0
    center = (min(starts_ms) + max(ends_ms)) // 2
    seg_start, seg_end = _segment_timestamp_bounds(segment)
    if seg_start <= center <= seg_end:
        return 5000
    distance = seg_start - center if center < seg_start else center - seg_end
    return max(0, 5000 - distance)


def _segment_dedup_score(segment: dict) -> int:
    return _segment_richness_score(segment) + _segment_subtitle_alignment_score(segment)


def _segment_overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _unique_nonempty_parts(*texts: str) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for text in texts:
        value = str(text or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        parts.append(value)
    return "；".join(parts)


def _normalize_scene_part(part: str) -> str:
    text = str(part or "").strip()
    while text.startswith("从"):
        text = text[1:].strip()
    return text


def _parse_scene_chain(scene: str) -> list[str]:
    """将单场景名或「从A切换至B…」解析为有序场景列表。"""
    text = str(scene or "").strip()
    if not text:
        return []
    if "切换至" not in text:
        normalized = _normalize_scene_part(text)
        return [normalized] if normalized else []
    if text.startswith("从"):
        text = text[1:]
    parts: list[str] = []
    for part in text.split("切换至"):
        normalized = _normalize_scene_part(part)
        if normalized:
            parts.append(normalized)
    return parts


def _merge_scene_chains(*chains: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for chain in chains:
        for scene in chain:
            if scene not in seen:
                seen.add(scene)
                merged.append(scene)
    return merged


def _format_scene_chain(scenes: list[str]) -> str:
    if not scenes:
        return ""
    if len(scenes) == 1:
        return scenes[0]
    return "从" + "切换至".join(scenes)


def _format_observation_transition(scenes: list[str]) -> str:
    if len(scenes) < 2:
        return ""
    return "画面由" + "切换至".join(scenes)


def _is_scene_transition_observation(text: str) -> bool:
    value = str(text or "").strip()
    return value.startswith("画面由") and "切换至" in value


def _strip_transition_observations(text: str) -> str:
    if not text:
        return ""
    parts = [
        part.strip()
        for part in text.split("；")
        if part.strip() and not _is_scene_transition_observation(part)
    ]
    return "；".join(parts)


def _scene_transition_label(earlier_scene: str, later_scene: str) -> str:
    scenes = _merge_scene_chains(
        _parse_scene_chain(earlier_scene),
        _parse_scene_chain(later_scene),
    )
    return _format_scene_chain(scenes)


def _merge_timestamp_range(first: str, second: str) -> str:
    first_text = str(first or "").strip()
    second_text = str(second or "").strip()
    if not first_text:
        return second_text
    if not second_text:
        return first_text
    first_start, first_end = _segment_timestamp_bounds({"timestamp": first_text})
    second_start, second_end = _segment_timestamp_bounds({"timestamp": second_text})
    start_ms = min(first_start, second_start)
    end_ms = max(first_end, second_end)
    return f"{_ms_to_timestamp_label(start_ms)}-{_ms_to_timestamp_label(end_ms)}"


def _merge_subtitle_entries_lists(
    first_entries: list | None,
    second_entries: list | None,
) -> list[dict]:
    by_start: dict[str, dict] = {}
    for entries in (first_entries, second_entries):
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            if not start:
                continue
            by_start[start] = item
    return sorted(
        by_start.values(),
        key=lambda row: _timestamp_to_ms(str(row.get("start") or "")),
    )


def _join_subtitle_entry_texts(entries: list[dict]) -> str:
    texts: list[str] = []
    for item in entries:
        text = str(item.get("text") or "").strip()
        if text:
            texts.append(text)
    return "；".join(texts)


def _merge_characters(first: Any, second: Any) -> list[str] | Any:
    merged: list[str] = []
    seen: set[str] = set()
    for value in (first, second):
        if isinstance(value, list):
            items = value
        elif value in (None, ""):
            items = []
        else:
            items = [value]
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


DEFAULT_MAX_SEGMENT_DURATION_MS = 30_000


def _split_field_parts(text: str, *, strip_transitions: bool = False) -> list[str]:
    parts = [part.strip() for part in str(text or "").split("；") if part.strip()]
    if strip_transitions:
        parts = [part for part in parts if not _is_scene_transition_observation(part)]
    return parts


def _entry_time_bounds(entry: dict) -> tuple[int, int]:
    start = _timestamp_to_ms(str(entry.get("start") or ""))
    end = _timestamp_to_ms(str(entry.get("end") or ""))
    if end < start:
        start, end = end, start
    return start, end


def _entries_in_range(entries: list, start_ms: int, end_ms: int) -> list[dict]:
    matched: list[dict] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        entry_start, entry_end = _entry_time_bounds(item)
        center = (entry_start + entry_end) // 2
        if start_ms <= center <= end_ms:
            matched.append(item)
    return matched


def _single_scene_label(scene: str) -> str:
    chain = _parse_scene_chain(scene)
    if chain:
        return chain[0]
    return str(scene or "").strip()


def _segments_share_same_scene(first: dict, second: dict) -> bool:
    scene_first = _single_scene_label(str(first.get("scene") or ""))
    scene_second = _single_scene_label(str(second.get("scene") or ""))
    if not scene_first or not scene_second:
        return True
    return scene_first == scene_second


def _combine_same_scene_segments(keeper: dict, other: dict) -> dict:
    """同批同场景：合并时间范围与文本字段。"""
    if _segment_dedup_score(other) > _segment_dedup_score(keeper):
        keeper, other = other, keeper
    result = dict(keeper)
    result["timestamp"] = _merge_timestamp_range(
        str(keeper.get("timestamp") or ""),
        str(other.get("timestamp") or ""),
    )
    scene_label = _single_scene_label(str(result.get("scene") or ""))
    if not scene_label:
        scene_label = _single_scene_label(str(other.get("scene") or ""))
    if scene_label:
        result["scene"] = scene_label
    for key in ("observation", "action", "emotion", "key_visual"):
        merged = _unique_nonempty_parts(
            str(keeper.get(key) or ""),
            str(other.get(key) or ""),
        )
        if merged:
            result[key] = merged
    merged_entries = _merge_subtitle_entries_lists(
        keeper.get("subtitle_entries"),
        other.get("subtitle_entries"),
    )
    if merged_entries:
        result["subtitle_entries"] = merged_entries
        merged_subtitle = _join_subtitle_entry_texts(merged_entries)
        if merged_subtitle:
            result["subtitle"] = merged_subtitle
    elif str(other.get("subtitle") or "").strip() and not str(result.get("subtitle") or "").strip():
        result["subtitle"] = other.get("subtitle")
    result["characters"] = _merge_characters(keeper.get("characters"), other.get("characters"))
    result.pop("time_range", None)
    return result


def _merge_segment_group(group: list[dict]) -> dict:
    if not group:
        return {}
    merged = dict(group[0])
    for segment in group[1:]:
        merged = _combine_same_scene_segments(merged, segment)
    return merged


def merge_same_scene_within_batch(segments: list[dict]) -> list[dict]:
    """同 batch_index + 同 scene 标签合并为一条 segment。"""
    cleaned = [segment for segment in segments if isinstance(segment, dict)]
    if len(cleaned) <= 1:
        return cleaned

    by_batch: dict[int, list[dict]] = {}
    for segment in cleaned:
        batch_index = int(segment.get("batch_index", 0))
        by_batch.setdefault(batch_index, []).append(segment)

    merged_all: list[dict] = []
    for batch_index in sorted(by_batch.keys()):
        by_scene: dict[str, list[dict]] = {}
        for segment in by_batch[batch_index]:
            label = _single_scene_label(str(segment.get("scene") or "")) or "__empty__"
            by_scene.setdefault(label, []).append(segment)
        for group in by_scene.values():
            merged_all.append(_merge_segment_group(group))
    return sorted(merged_all, key=lambda item: _segment_timestamp_bounds(item)[0])


def prune_cross_scene_overlaps(
    segments: list[dict],
    *,
    overlap_ratio: float = 0.5,
) -> list[dict]:
    """跨场景时间重叠：保留信息更全的一条，剔除脑补冲突段。"""
    cleaned = [segment for segment in segments if isinstance(segment, dict)]
    if len(cleaned) <= 1:
        return cleaned

    ordered = sorted(
        cleaned,
        key=lambda item: (
            _segment_timestamp_bounds(item)[0],
            -_segment_dedup_score(item),
        ),
    )
    removed: set[int] = set()
    total = len(ordered)
    for i in range(total):
        if i in removed:
            continue
        bounds_i = _segment_timestamp_bounds(ordered[i])
        scene_i = _single_scene_label(str(ordered[i].get("scene") or ""))
        for j in range(i + 1, total):
            if j in removed:
                continue
            bounds_j = _segment_timestamp_bounds(ordered[j])
            if _segment_overlap_ratio(bounds_i, bounds_j) < overlap_ratio:
                continue
            scene_j = _single_scene_label(str(ordered[j].get("scene") or ""))
            if scene_i and scene_j and scene_i == scene_j:
                continue
            if _segment_dedup_score(ordered[i]) >= _segment_dedup_score(ordered[j]):
                removed.add(j)
            else:
                removed.add(i)
                break
    return [ordered[index] for index in range(total) if index not in removed]


def _combine_duplicate_segments(keeper: dict, other: dict) -> dict:
    """同一时间段重复 segment：保留信息更全的一条，仅合并字幕条目。"""
    if _segment_dedup_score(other) > _segment_dedup_score(keeper):
        keeper, other = other, keeper
    result = dict(keeper)
    merged_entries = _merge_subtitle_entries_lists(
        keeper.get("subtitle_entries"),
        other.get("subtitle_entries"),
    )
    if merged_entries:
        result["subtitle_entries"] = merged_entries
        merged_subtitle = _join_subtitle_entry_texts(merged_entries)
        if merged_subtitle:
            result["subtitle"] = merged_subtitle
    result.pop("time_range", None)
    return result


def _split_bloated_segment(
    segment: dict,
    *,
    max_duration_ms: int = DEFAULT_MAX_SEGMENT_DURATION_MS,
) -> list[dict]:
    scene_chain = _parse_scene_chain(str(segment.get("scene") or ""))
    observation_parts = _split_field_parts(
        str(segment.get("observation") or ""),
        strip_transitions=True,
    )
    action_parts = _split_field_parts(str(segment.get("action") or ""))
    emotion_parts = _split_field_parts(str(segment.get("emotion") or ""))
    visual_parts = _split_field_parts(str(segment.get("key_visual") or ""))

    start_ms, end_ms = _segment_timestamp_bounds(segment)
    duration_ms = max(end_ms - start_ms, 0)

    split_count = 1
    if len(scene_chain) > 1:
        split_count = len(scene_chain)
    elif duration_ms > max_duration_ms:
        part_counts = [
            len(observation_parts),
            len(action_parts),
            len(emotion_parts),
            len(visual_parts),
        ]
        max_parts = max(part_counts) if part_counts else 1
        if max_parts > 1:
            split_count = max_parts
        else:
            split_count = max(2, (duration_ms + max_duration_ms - 1) // max_duration_ms)

    if split_count <= 1:
        cleaned = dict(segment)
        if scene_chain:
            cleaned["scene"] = scene_chain[0]
        if observation_parts:
            cleaned["observation"] = observation_parts[0] if len(observation_parts) == 1 else "；".join(observation_parts)
        cleaned.pop("time_range", None)
        return [cleaned]

    entries = segment.get("subtitle_entries")
    entry_list = entries if isinstance(entries, list) else []
    chunk_ms = max(duration_ms // split_count, 1)

    results: list[dict] = []
    for index in range(split_count):
        seg_start = start_ms + index * chunk_ms
        seg_end = end_ms if index == split_count - 1 else min(start_ms + (index + 1) * chunk_ms, end_ms)
        sub_entries = _entries_in_range(entry_list, seg_start, seg_end)

        payload: dict[str, Any] = {}
        for key in ("batch_index", "characters", "audio_cue", "importance"):
            if key in segment:
                payload[key] = segment[key]

        payload["timestamp"] = (
            f"{_ms_to_timestamp_label(seg_start)}-{_ms_to_timestamp_label(seg_end)}"
        )
        payload["scene"] = scene_chain[index] if index < len(scene_chain) else scene_chain[-1]

        for field, parts in (
            ("observation", observation_parts),
            ("action", action_parts),
            ("emotion", emotion_parts),
            ("key_visual", visual_parts),
        ):
            if not parts:
                continue
            if index < len(parts):
                payload[field] = parts[index]
            elif len(parts) == 1:
                payload[field] = parts[0]

        if sub_entries:
            payload["subtitle_entries"] = sub_entries
            merged_subtitle = _join_subtitle_entry_texts(sub_entries)
            if merged_subtitle:
                payload["subtitle"] = merged_subtitle
        elif index == 0 and str(segment.get("subtitle") or "").strip():
            payload["subtitle"] = str(segment.get("subtitle") or "").strip()

        results.append(payload)
    return results


def split_scene_segments(
    segments: list[dict],
    *,
    max_duration_ms: int = DEFAULT_MAX_SEGMENT_DURATION_MS,
) -> list[dict]:
    """将含多场景链或过长时段的 segment 拆成单场景、语义完整的片段。"""
    split: list[dict] = []
    for segment in segments:
        if isinstance(segment, dict):
            split.extend(
                _split_bloated_segment(segment, max_duration_ms=max_duration_ms)
            )
    return sorted(split, key=lambda item: _segment_timestamp_bounds(item)[0])


def dedupe_scene_segments(
    segments: list[dict],
    *,
    overlap_merge_ratio: float = 0.75,
) -> list[dict]:
    """
    去重 scene_segments：同时间段保留信息更全的一条，仅合并字幕条目。

    不同场景的重叠片段不再拼接为「切换至」链，避免 observation/scene 膨胀。
    """
    cleaned = [segment for segment in segments if isinstance(segment, dict)]
    if len(cleaned) <= 1:
        return cleaned

    by_timestamp: dict[tuple[int, int], dict] = {}
    for segment in cleaned:
        key = _segment_timestamp_bounds(segment)
        existing = by_timestamp.get(key)
        if existing is None:
            by_timestamp[key] = dict(segment)
        else:
            by_timestamp[key] = _combine_duplicate_segments(existing, segment)

    ordered = sorted(by_timestamp.values(), key=lambda item: _segment_timestamp_bounds(item)[0])

    merged: list[dict] = []
    for segment in ordered:
        bounds = _segment_timestamp_bounds(segment)
        if not merged:
            merged.append(segment)
            continue
        prev = merged[-1]
        prev_bounds = _segment_timestamp_bounds(prev)
        overlap = _segment_overlap_ratio(prev_bounds, bounds)
        if overlap >= overlap_merge_ratio and _segments_share_same_scene(prev, segment):
            merged[-1] = _combine_duplicate_segments(prev, segment)
            continue
        merged.append(segment)

    by_start: dict[int, dict] = {}
    for segment in merged:
        start_ms, _ = _segment_timestamp_bounds(segment)
        existing = by_start.get(start_ms)
        if existing is None:
            by_start[start_ms] = segment
        else:
            by_start[start_ms] = _combine_duplicate_segments(existing, segment)

    return sorted(by_start.values(), key=lambda item: _segment_timestamp_bounds(item)[0])


def normalize_scene_segments(
    segments: list[dict],
    *,
    max_duration_ms: int = DEFAULT_MAX_SEGMENT_DURATION_MS,
    strict_scene_rules: bool = True,
    cross_scene_overlap_prune_ratio: float = 0.5,
) -> list[dict]:
    """去重后拆分：每个 segment 对应单一连续场景；可选硬性合并/剔除重叠冲突。"""
    normalized = dedupe_scene_segments(segments)
    if strict_scene_rules:
        normalized = merge_same_scene_within_batch(normalized)
        normalized = prune_cross_scene_overlaps(
            normalized,
            overlap_ratio=cross_scene_overlap_prune_ratio,
        )
    split = split_scene_segments(
        normalized,
        max_duration_ms=max_duration_ms,
    )
    if strict_scene_rules:
        split = merge_same_scene_within_batch(split)
        split = prune_cross_scene_overlaps(
            split,
            overlap_ratio=cross_scene_overlap_prune_ratio,
        )
    return split


def update_segment_environment_context(
    env_context: dict[str, str],
    segment: dict[str, Any],
) -> None:
    """按时间顺序更新场景环境上下文（供展示时继承）。"""
    raw_scene = str(segment.get("scene") or "").strip()
    if raw_scene:
        new_label = _single_scene_label(raw_scene)
        old_label = _single_scene_label(env_context.get("scene", "")) if env_context.get("scene") else ""
        if old_label and new_label != old_label:
            env_context.pop("key_visual", None)
            env_context.pop("emotion", None)
        env_context["scene"] = raw_scene
    for key in ("key_visual", "emotion"):
        value = str(segment.get(key) or "").strip()
        if value:
            env_context[key] = value


def resolve_segment_display_fields(
    segment: dict[str, Any],
    env_context: dict[str, str] | None = None,
) -> dict[str, str]:
    """合并存储字段与 inherited 环境，供 Markdown / picture 展示。"""
    ctx = env_context or {}
    scene = str(segment.get("scene") or "").strip() or str(ctx.get("scene") or "").strip()
    key_visual = str(segment.get("key_visual") or "").strip() or str(ctx.get("key_visual") or "").strip()
    emotion = str(segment.get("emotion") or "").strip() or str(ctx.get("emotion") or "").strip()
    observation = str(segment.get("observation") or "").strip()
    if not observation:
        parts = [value for value in (segment.get("action"), emotion, key_visual) if str(value or "").strip()]
        observation = "；".join(str(part).strip() for part in parts)
    return {
        "scene": scene,
        "key_visual": key_visual,
        "emotion": emotion,
        "observation": observation,
    }


def dedupe_scene_environment_across_segments(segments: list[dict[str, Any]]) -> None:
    """同一场景连续片段：环境字段与 observation 已出现过的分句不重复写入。"""
    effective_scene_label = ""
    effective_key_visual = ""
    effective_emotion = ""
    env_clauses_seen: set[str] = set()

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        raw_scene = str(segment.get("scene") or "").strip()
        if raw_scene:
            scene_label = _single_scene_label(raw_scene)
            if scene_label == effective_scene_label and effective_scene_label:
                segment.pop("scene", None)
            else:
                effective_scene_label = scene_label
                effective_key_visual = ""
                effective_emotion = ""
                env_clauses_seen = {raw_scene}
                for part in _split_field_parts(raw_scene):
                    env_clauses_seen.add(part)

        for field, effective_value in (
            ("key_visual", effective_key_visual),
            ("emotion", effective_emotion),
        ):
            value = str(segment.get(field) or "").strip()
            if not value:
                continue
            if value == effective_value and effective_value:
                segment.pop(field, None)
                continue
            if field == "key_visual":
                effective_key_visual = value
            else:
                effective_emotion = value
            for part in _split_field_parts(value):
                env_clauses_seen.add(part)

        observation = str(segment.get("observation") or "").strip()
        if observation:
            parts = _split_field_parts(observation, strip_transitions=True)
            filtered = [part for part in parts if part not in env_clauses_seen]
            if len(filtered) < len(parts):
                if filtered:
                    segment["observation"] = "；".join(filtered)
                else:
                    segment.pop("observation", None)
            for part in filtered:
                env_clauses_seen.add(part)


def _ms_to_timestamp_label(ms: int) -> str:
    from app.utils import utils

    return utils.seconds_to_time(ms / 1000.0).replace(".", ",")


def _uniform_sample_segment_indices(
    segments: list[dict],
    max_chars: int,
    *,
    format_fn: Callable[[dict, int], str],
) -> list[int]:
    count = len(segments)
    if count == 0:
        return []
    if count == 1:
        return [0]

    formatted = [format_fn(segment, index + 1) for index, segment in enumerate(segments)]
    total_chars = sum(len(text) for text in formatted)
    if total_chars <= max_chars:
        return list(range(count))

    bounds = [_segment_time_bounds(segment) for segment in segments]
    min_ms = min(start for start, _ in bounds)
    max_ms = max(end for _, end in bounds)
    duration_ms = max(max_ms - min_ms, 1)
    midpoints = [(start + end) / 2 for start, end in bounds]

    header_reserve = 140
    budget = max(800, max_chars - header_reserve)
    avg_len = max(120, total_chars / count)
    target_count = max(8, min(count, int(budget / avg_len)))

    chosen: set[int] = {0, count - 1}
    for bucket in range(target_count):
        target_ms = min_ms + (bucket + 0.5) * duration_ms / target_count
        best_index = min(
            range(count),
            key=lambda idx: (
                abs(midpoints[idx] - target_ms),
                -len(formatted[idx]),
                idx,
            ),
        )
        chosen.add(best_index)

    selected = sorted(chosen)

    def selected_size(indices: list[int]) -> int:
        return sum(len(formatted[idx]) for idx in indices)

    while len(selected) > 2 and selected_size(selected) > budget:
        removable = selected[1:-1]
        if not removable:
            break
        mid_pos = len(selected) // 2
        selected.pop(mid_pos)

    return selected


def frame_analysis_to_timeline_sampled_markdown(
    json_file_path: str,
    max_chars: int,
) -> str:
    if not os.path.exists(json_file_path):
        return f"错误: 文件 {json_file_path} 不存在"

    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        return f"处理JSON文件时出错: {traceback.format_exc() if logger.level('DEBUG') else exc}"

    segments = collect_scene_segments_from_analysis(data)
    if not segments:
        return "（无抽帧描述）"

    segments.sort(key=lambda segment: _segment_time_bounds(segment)[0])
    env_context: dict[str, str] = {}
    selected_indices = _uniform_sample_segment_indices(
        segments,
        max_chars,
        format_fn=lambda segment, output_index: format_scene_segment(
            segment,
            output_index,
            env_context=env_context,
        ),
    )

    if len(selected_indices) >= len(segments):
        env_context = {}
        markdown = "".join(
            format_scene_segment(segments[idx], output_index + 1, env_context=env_context)
            for output_index, idx in enumerate(selected_indices)
        )
        logger.info(
            f"抽帧时间轴采样：全量 {len(segments)} 段（{len(markdown)} 字），未截断"
        )
        return markdown

    first_ms, _ = _segment_time_bounds(segments[selected_indices[0]])
    _, last_ms = _segment_time_bounds(segments[selected_indices[-1]])
    header = (
        f"（抽帧摘要：全片时间轴均匀采样 {len(selected_indices)}/{len(segments)} 段，"
        f"覆盖 {_ms_to_timestamp_label(first_ms)}–{_ms_to_timestamp_label(last_ms)}）\n\n"
    )
    env_context = {}
    body = "".join(
        format_scene_segment(segments[idx], output_index + 1, env_context=env_context)
        for output_index, idx in enumerate(selected_indices)
    )
    markdown = header + body
    if len(markdown) > max_chars:
        markdown = markdown[: max_chars - 24].rstrip() + "\n…（抽帧摘要已截断）"
    logger.info(
        f"抽帧时间轴采样：{len(selected_indices)}/{len(segments)} 段，"
        f"输出 {len(markdown)} 字（预算 {max_chars}）"
    )
    return markdown


def _segment_label(segment: dict) -> str:
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list) and entries:
        starts: list[str] = []
        ends: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            end = str(item.get("end") or "").strip()
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        if starts and ends:
            return f"{starts[0]}-{ends[-1]}"
    return str(segment.get("timestamp") or "").strip()


def extract_frame_subtitle_lexicon(data: dict) -> dict:
    """从抽帧 JSON 汇总字幕对白与 action 中的人物名/称呼。"""
    names: set[str] = set()
    snippets: list[str] = []

    for segment in collect_scene_segments_from_analysis(data):
        label = _segment_label(segment)
        subtitle = str(segment.get("subtitle") or "").strip()
        action = str(segment.get("action") or "").strip()
        observation = str(segment.get("observation") or "").strip()

        for match in _NAMED_CHARACTER_RE.finditer(action + observation):
            names.add(match.group(1))

        if subtitle:
            snippets.append(f"[{label}] {subtitle}")
        entries = segment.get("subtitle_entries")
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                start = str(item.get("start") or "").strip()
                if text:
                    prefix = start or label
                    snippets.append(f"[{prefix}] {text}")

    for batch in data.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        obs_list = batch.get("frame_observations") or batch.get("observations") or []
        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            ts = str(obs.get("timestamp") or "").strip()
            burned = str(obs.get("burned_in_subtitle") or "").strip()
            attached = str(obs.get("subtitle") or "").strip()
            text = burned if obs.get("has_burned_in_subtitle") and burned else attached
            if text:
                snippets.append(f"[{ts or '?'}] {text}")

    for obs in data.get("frame_observations") or []:
        if not isinstance(obs, dict):
            continue
        ts = str(obs.get("timestamp") or "").strip()
        burned = str(obs.get("burned_in_subtitle") or "").strip()
        attached = str(obs.get("subtitle") or "").strip()
        text = burned if obs.get("has_burned_in_subtitle") and burned else attached
        if text:
            snippets.append(f"[{ts or '?'}] {text}")

    return {
        "names": names,
        "subtitle_snippets": snippets,
    }


def build_frame_subtitle_lexicon_markdown(
    json_file_path: str,
    *,
    max_chars: int = 4000,
) -> tuple[str, dict]:
    """生成供 LLM 参照的抽帧字幕人物索引 Markdown。"""
    empty: dict = {"names": set(), "subtitle_snippets": []}
    if not json_file_path or not os.path.isfile(json_file_path):
        return "", empty

    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning(f"读取抽帧字幕索引失败: {exc}")
        return "", empty

    lexicon = extract_frame_subtitle_lexicon(data)
    names = sorted(str(name) for name in lexicon.get("names") or set())
    snippets = list(lexicon.get("subtitle_snippets") or [])

    if len(snippets) > 1:
        target = max(8, min(len(snippets), max(1200, max_chars - 800) // 80))
        if len(snippets) <= target:
            sampled = snippets
        else:
            step = (len(snippets) - 1) / max(target - 1, 1)
            picked: list[int] = []
            for i in range(target):
                idx = min(len(snippets) - 1, int(round(i * step)))
                if idx not in picked:
                    picked.append(idx)
            sampled = [snippets[i] for i in picked]
    else:
        sampled = snippets

    lines = [
        "## 抽帧字幕人物索引（只读 · 对白与称呼以此为准）",
        "- **出现人物（据抽帧 action 标注）**："
        + ("、".join(names) if names else "（未识别到具名人物）"),
        "- **对白/称呼须与下方摘录一致**：含全名、小名、昵称、关系称呼（老叶、小跃、师傅等）",
    ]
    lines.append("- **对白摘录（按时间均匀采样）**：")
    for snippet in sampled:
        lines.append(f"  - {snippet}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n…（索引已截断）"
    return text, lexicon

