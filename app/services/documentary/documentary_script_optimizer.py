#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说脚本后处理：OST 归一化与原声高光段校验。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.srt_utils import parse_timestamp_range


def _segment_duration_sec(timestamp: str) -> float:
    start_ms, end_ms = parse_timestamp_range(timestamp or "")
    if end_ms <= start_ms:
        return 0.0
    return (end_ms - start_ms) / 1000.0


def finalize_documentary_script_items(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """归一化 LLM 输出的逐帧解说脚本 OST 字段。"""
    if not items:
        return []

    cfg = get_documentary_settings(settings)
    default_ost = int(cfg.get("default_narration_ost", 2))
    if default_ost not in (0, 2):
        default_ost = 2

    highlights_enabled = bool(cfg.get("enable_original_audio_highlights", True))
    ost1_min = float(cfg.get("ost1_duration_min", 4))
    ost1_max = float(cfg.get("ost1_duration_max", 12))
    max_ost1 = int(cfg.get("max_ost1_segments", 6))

    result: List[Dict[str, Any]] = []
    ost1_counter = 0

    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue

        item = dict(raw)
        item["_id"] = int(item.get("_id") or index + 1)

        if not highlights_enabled:
            item["OST"] = 2
            result.append(item)
            continue

        try:
            ost = int(item.get("OST", default_ost))
        except (TypeError, ValueError):
            ost = default_ost

        if ost not in (0, 1, 2):
            ost = default_ost

        if ost == 1:
            duration = _segment_duration_sec(str(item.get("timestamp") or ""))
            if ost1_counter >= max_ost1:
                logger.warning(
                    f"片段 #{item['_id']} 标记 OST=1 但已达上限 {max_ost1}，回退为 OST={default_ost}"
                )
                ost = default_ost
            elif duration > 0 and duration < ost1_min:
                logger.warning(
                    f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 短于 {ost1_min}s，回退为 OST={default_ost}"
                )
                ost = default_ost
            elif duration > ost1_max:
                logger.info(
                    f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 超过建议上限 {ost1_max}s，保留"
                )

        if ost == 1:
            ost1_counter += 1
            item["narration"] = f"播放原片{ost1_counter}"
            item["OST"] = 1
        else:
            item["OST"] = ost

        result.append(item)

    if highlights_enabled and ost1_counter:
        logger.info(f"逐帧解说脚本含 {ost1_counter} 段 OST=1 原声高光")

    return result
