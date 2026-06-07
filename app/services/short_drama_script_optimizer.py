#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧解说脚本后处理：time_range 对位、OST 比例、原声段 picture 旁白格式。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.documentary.documentary_script_optimizer import (
    _align_items_to_frame_time_ranges,
    _strip_internal_clip_flags,
)
from app.services.short_drama_settings import (
    compute_short_drama_ost_bounds,
    get_short_drama_settings,
)
from app.services.srt_utils import (
    parse_timestamp_range,
    repair_or_drop_invalid_timestamp_items,
)

AUTO_NARRATION_MARKER = "__AUTO_NARRATION__"


def wrap_picture_in_double_quotes(text: str) -> str:
    """原声段 picture 旁白用英文双引号包裹。"""
    value = (text or "").strip()
    if not value:
        return value
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value
    return f'"{value}"'


def strip_picture_quotes(text: str) -> str:
    value = (text or "").strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].strip()
    return value


def count_short_drama_segments(items: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    ost1 = sum(1 for item in items if int(item.get("OST", 0) or 0) == 1)
    ost0 = sum(1 for item in items if int(item.get("OST", 0) or 0) == 0)
    return ost1, ost0, len(items)


def validate_short_drama_script_counts(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = get_short_drama_settings(settings)
    ost1_count, ost0_count, total = count_short_drama_segments(items)
    bounds = compute_short_drama_ost_bounds(total, cfg)

    issues: List[str] = []
    if ost0_count < bounds["ost0_min"]:
        issues.append(
            f"解说 OST=0 仅 {ost0_count} 段，要求至少 {bounds['ost0_min']} 段"
            f"（约占 {cfg.get('narration_percent', 30)}%）"
        )
    if bounds["ost0_max"] > 0 and ost0_count > bounds["ost0_max"]:
        issues.append(f"解说 OST=0 共 {ost0_count} 段，建议不超过 {bounds['ost0_max']} 段")
    if ost1_count > bounds["ost1_max"]:
        issues.append(
            f"原声 OST=1 共 {ost1_count} 段，不得超过 {bounds['ost1_max']} 段"
            f"（约占 {cfg.get('original_audio_percent', 70)}%）"
        )

    ok = not issues
    return {
        "ok": ok,
        "ost1_count": ost1_count,
        "ost0_count": ost0_count,
        "total": total,
        **bounds,
        "message": "段数比例符合要求" if ok else "；".join(issues),
    }


def _playback_sort_key(item: Dict[str, Any]) -> int:
    return int(item.get("_id") or 0)


def _segment_duration_ms(timestamp: str) -> int:
    try:
        start_ms, end_ms = parse_timestamp_range(timestamp)
        return max(0, end_ms - start_ms)
    except Exception:
        return 0


def _convert_ost1_to_ost0_bridge(item: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(item)
    scene = strip_picture_quotes(str(row.get("picture") or ""))
    row["OST"] = 0
    if scene:
        row["narration"] = f"而这时，{scene.rstrip('。！？')}。"
    else:
        row["narration"] = AUTO_NARRATION_MARKER
    row.pop("original_line", None)
    if str(row.get("narration") or "").startswith("播放原片"):
        row["narration"] = AUTO_NARRATION_MARKER if not scene else row["narration"]
    return row


def break_consecutive_ost1(
    items: List[Dict[str, Any]],
    *,
    max_run: int = 2,
    protect_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """连续原声超过 max_run 时，将多余段转为 OST=0 串场解说。"""
    if max_run < 1:
        return items
    protected = protect_ids or {1}
    result: List[Dict[str, Any]] = []
    run = 0
    for item in items:
        row = dict(item)
        item_id = int(row.get("_id") or 0)
        if int(row.get("OST", 0) or 0) == 1:
            run += 1
            if run > max_run and item_id not in protected:
                converted = _convert_ost1_to_ost0_bridge(row)
                logger.info(
                    f"片段 #{item_id} 连续原声第 {run} 段，已转为 OST=0 串场解说"
                )
                result.append(converted)
                run = 0
                continue
        else:
            run = 0
        result.append(row)
    return result


def _protected_ost0_indices(items: List[Dict[str, Any]]) -> set[int]:
    protected: set[int] = set()
    ost0_indices = [
        idx for idx, item in enumerate(items) if int(item.get("OST", 0) or 0) == 0
    ]
    if ost0_indices:
        protected.add(ost0_indices[0])
        if len(ost0_indices) > 1:
            protected.add(ost0_indices[-1])
    return protected


def convert_ost1_to_ost0_for_ratio(
    items: List[Dict[str, Any]],
    slots_needed: int,
    *,
    protect_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """按比例不足时，均匀选取 OST=1 转为 OST=0 串场（保留开篇爆燃原声）。"""
    if slots_needed <= 0:
        return items

    protected = set(protect_ids or {1})
    protected |= _protected_ost0_indices(items)

    candidates = [
        (idx, item)
        for idx, item in enumerate(items)
        if int(item.get("OST", 0) or 0) == 1
        and int(item.get("_id") or 0) not in protected
    ]
    if not candidates:
        return items

    step = max(1, len(candidates) // max(slots_needed, 1))
    pick_indices = {candidates[i][0] for i in range(0, len(candidates), step)}
    if len(pick_indices) < slots_needed:
        for idx, _ in candidates:
            pick_indices.add(idx)
            if len(pick_indices) >= slots_needed:
                break

    result: List[Dict[str, Any]] = []
    converted = 0
    for idx, item in enumerate(items):
        if idx in pick_indices and converted < slots_needed:
            if int(item.get("OST", 0) or 0) == 1:
                result.append(_convert_ost1_to_ost0_bridge(item))
                converted += 1
                logger.info(f"片段 #{item.get('_id')} 原声转解说，平衡 3:7 比例")
                continue
        result.append(dict(item))
    return result


def trim_excess_ost1_segments(
    items: List[Dict[str, Any]],
    ost1_max: int,
    *,
    protect_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """原声段超过上限时，移除最短的原声段。"""
    if ost1_max <= 0:
        return items
    protected = set(protect_ids or {1})
    ordered = sorted([dict(item) for item in items], key=_playback_sort_key)
    removed = 0
    while sum(1 for item in ordered if int(item.get("OST", 0) or 0) == 1) > ost1_max:
        candidates = [
            (idx, item)
            for idx, item in enumerate(ordered)
            if int(item.get("OST", 0) or 0) == 1
            and int(item.get("_id") or 0) not in protected
        ]
        if not candidates:
            break
        drop_idx = min(
            candidates,
            key=lambda pair: _segment_duration_ms(str(pair[1].get("timestamp") or "")),
        )[0]
        dropped = ordered.pop(drop_idx)
        removed += 1
        logger.info(f"原声段超限，移除 OST=1 #{dropped.get('_id')}")

    if removed:
        for index, item in enumerate(ordered, 1):
            item["_id"] = index
    return ordered


def enforce_short_drama_ost_ratio(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """后处理：打断连续原声、补解说段、裁剪过多原声。"""
    if not items:
        return items

    cfg = get_short_drama_settings(settings)
    max_run = int(cfg.get("max_consecutive_ost1", 4) or 4)
    result = break_consecutive_ost1(items, max_run=max_run, protect_ids={1})

    validation = validate_short_drama_script_counts(result, cfg)
    ost0_need = max(0, validation["ost0_min"] - validation["ost0_count"])
    if ost0_need > 0:
        result = convert_ost1_to_ost0_for_ratio(result, ost0_need, protect_ids={1})

    validation = validate_short_drama_script_counts(result, cfg)
    if validation["ost1_count"] > validation["ost1_max"]:
        result = trim_excess_ost1_segments(
            result,
            validation["ost1_max"],
            protect_ids={1},
        )

    validation = validate_short_drama_script_counts(result, cfg)
    if not validation["ok"]:
        logger.warning(f"短剧解说 OST 比例仍未达标: {validation['message']}")
    else:
        logger.info(
            f"短剧解说 OST 比例 OK：解说 {validation['ost0_count']} / "
            f"原声 {validation['ost1_count']}（目标解说≥{validation['ost0_min']}）"
        )
    return result


def format_ost1_picture_narrations(
    items: List[Dict[str, Any]],
    *,
    wrap_quotes: bool = True,
) -> List[Dict[str, Any]]:
    """为 OST=1 原声段规范化 picture 旁白字段。"""
    for item in items:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        picture = str(item.get("picture") or "").strip()
        if not picture:
            logger.warning(f"片段 #{item.get('_id')} OST=1 缺少 picture 旁白描述")
            continue
        if wrap_quotes:
            item["picture"] = wrap_picture_in_double_quotes(picture)
    return items


def align_short_drama_items_to_frame_time_ranges(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    ost1_hard_max: float = 0,
    skip_opening_item_id: int = 1,
) -> List[Dict[str, Any]]:
    """将 timestamp 对齐到抽帧 JSON 的 time_range（字幕对位剪辑范围）。"""
    if not (frame_analysis_path or "").strip():
        return items
    aligned = _align_items_to_frame_time_ranges(
        items,
        subtitle_content=subtitle_content,
        frame_analysis_path=frame_analysis_path,
        ost1_hard_max=ost1_hard_max,
        skip_opening_item_id=skip_opening_item_id,
    )
    return _strip_internal_clip_flags(aligned)


def optimize_short_drama_script_items(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """短剧解说脚本生成后的统一后处理。"""
    if not items:
        return items

    cfg = settings or get_short_drama_settings()
    ost1_max = float(cfg.get("ost1_duration_max", 18) or 18)
    min_ost1_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)

    if (frame_analysis_path or "").strip():
        items = align_short_drama_items_to_frame_time_ranges(
            items,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=ost1_max,
        )
        logger.info("短剧解说：已按抽帧 time_range（字幕对位）校正片段边界")

    items = enforce_short_drama_ost_ratio(items, cfg)

    items = format_ost1_picture_narrations(
        items,
        wrap_quotes=bool(cfg.get("picture_wrap_double_quotes", True)),
    )

    items = repair_or_drop_invalid_timestamp_items(
        items,
        subtitle_content=subtitle_content,
        ost1_min_duration_ms=min_ost1_ms,
    )
    return items
