#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧脚本通用时间戳对位：内容用到哪段，timestamp 就对哪段（字幕/蓝图）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.short_drama_blueprint_script import parse_scene_segment_blueprint
from app.services.short_drama_settings import get_short_drama_settings
from app.services.srt_utils import (
    dialogue_match_key,
    find_subtitle_span_global,
    format_timestamp_ms,
    parse_srt,
    parse_timestamp_range,
)

# picture 与 original_line 语义互斥（通用规则，非单集硬编码）
_PICTURE_LINE_CONFLICTS: Tuple[Tuple[str, ...], Tuple[str, ...], ...] = (
    (("楼顶", "纵身", "跃下", "天台边缘", "跳下"), ("配枪", "交出来", "闯多大的祸", "停职")),
    (("灵堂", "遗照", "葬礼"), ("二十八起", "金鼎集团")),
    (("祠堂", "祭祖", "舞狮"), ("审讯", "配枪")),
)

_TIMESTAMP_ALIGN_TOLERANCE_MS = 8_000


def strip_picture_quotes(text: str) -> str:
    value = (text or "").strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].strip()
    return value


def strip_dialogue_quotes(text: str) -> str:
    return str(text or "").strip().strip("「」").strip()


def detect_picture_line_incoherence(picture: str, line: str) -> bool:
    """picture 与 original_line 是否明显不属于同一场景。"""
    pic = strip_picture_quotes(picture)
    dlg = strip_dialogue_quotes(line)
    if not pic or not dlg:
        return False
    for pic_keys, line_keys in _PICTURE_LINE_CONFLICTS:
        pic_hit = any(key in pic for key in pic_keys)
        line_hit = any(key in dlg for key in line_keys)
        if pic_hit and line_hit:
            return True
    return False


def _subtitle_span_for_line(
    entries: list,
    line: str,
    *,
    max_span_ms: int,
) -> Optional[Tuple[int, int]]:
    cleaned = strip_dialogue_quotes(line)
    if not cleaned or not entries:
        return None
    span = find_subtitle_span_global(
        entries,
        cleaned,
        max_span_ms=max_span_ms,
    )
    if not span or span[1] <= span[0]:
        return None
    start_ms, end_ms = span
    if end_ms - start_ms > max_span_ms:
        end_ms = start_ms + max_span_ms
    return start_ms, end_ms


def timestamp_aligns_with_subtitle_span(
    timestamp: str,
    span: Tuple[int, int],
    *,
    tolerance_ms: int = _TIMESTAMP_ALIGN_TOLERANCE_MS,
) -> bool:
    try:
        start_ms, end_ms = parse_timestamp_range(timestamp)
    except Exception:
        return False
    span_start, span_end = span
    if abs(start_ms - span_start) > tolerance_ms:
        return False
    if end_ms < span_start or start_ms > span_end + tolerance_ms:
        return False
    return True


def _apply_span_to_item(
    item: Dict[str, Any],
    span: Tuple[int, int],
    *,
    item_id: Any = "?",
    reason: str = "",
) -> bool:
    start_ms, end_ms = span
    new_ts = f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(end_ms)}"
    old_ts = str(item.get("timestamp") or "")
    if new_ts == old_ts:
        return False
    item["timestamp"] = new_ts
    logger.info(
        f"片段 #{item_id} 已按{reason}对位 timestamp: {old_ts!r} → {new_ts!r}"
    )
    return True


def _score_picture_scene_match(picture: str, scene: Dict[str, Any]) -> float:
    pic = strip_picture_quotes(picture)
    if not pic:
        return 0.0
    blob = " ".join(
        [
            str(scene.get("place") or ""),
            str(scene.get("picture_hint") or ""),
            str(scene.get("block") or ""),
            str(scene.get("time_range") or ""),
        ]
    )
    score = 0.0
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", pic):
        if token in blob:
            score += float(len(token))
    return score


def _find_best_blueprint_scene_for_picture(
    picture: str,
    scenes: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for scene in scenes:
        score = _score_picture_scene_match(picture, scene)
        if score > best_score:
            best_score = score
            best = scene
    if best is None or best_score <= 0:
        return None
    return best


def _scene_timestamp_span(
    scene: Dict[str, Any],
    *,
    max_span_ms: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    ranges = scene.get("timestamp_ranges") or []
    if not ranges:
        return None
    try:
        start_ms, end_ms = parse_timestamp_range(
            f"{ranges[0][0]}-{ranges[0][1]}"
        )
    except Exception:
        return None
    if end_ms <= start_ms:
        return None
    if max_span_ms and end_ms - start_ms > max_span_ms:
        end_ms = start_ms + max_span_ms
    return start_ms, end_ms


def realign_ost1_from_blueprint_scene(
    item: Dict[str, Any],
    *,
    scenes: List[Dict[str, Any]],
    entries: list,
    ost1_max_ms: int,
) -> bool:
    """picture 与台词错位时，从蓝图场景找回 quote + timestamp。"""
    picture = str(item.get("picture") or "")
    scene = _find_best_blueprint_scene_for_picture(picture, scenes)
    if not scene:
        return False

    quotes = scene.get("quotes") or []
    for quote in quotes:
        span = _subtitle_span_for_line(entries, quote, max_span_ms=ost1_max_ms)
        if not span:
            continue
        item["original_line"] = f"「{strip_dialogue_quotes(quote)}」"
        _apply_span_to_item(
            item,
            span,
            item_id=item.get("_id"),
            reason="蓝图场景台词",
        )
        return True

    span = _scene_timestamp_span(scene, max_span_ms=ost1_max_ms)
    if span:
        _apply_span_to_item(
            item,
            span,
            item_id=item.get("_id"),
            reason="蓝图场景时间窗",
        )
        return True
    return False


def align_ost1_item_to_subtitle_line(
    item: Dict[str, Any],
    entries: list,
    *,
    ost1_max_ms: int,
) -> bool:
    line = strip_dialogue_quotes(str(item.get("original_line") or ""))
    if not line:
        return False
    span = _subtitle_span_for_line(entries, line, max_span_ms=ost1_max_ms)
    if not span:
        return False
    return _apply_span_to_item(
        item,
        span,
        item_id=item.get("_id"),
        reason="字幕 original_line",
    )


def align_ost0_item_to_blueprint_scene(
    item: Dict[str, Any],
    scenes: List[Dict[str, Any]],
    *,
    ost0_min_ms: int,
) -> bool:
    """OST=0：将 timestamp 落入最匹配蓝图场景的时间窗。"""
    if not scenes:
        return False

    narration = str(item.get("narration") or "")
    picture = strip_picture_quotes(str(item.get("picture") or ""))
    try:
        cur_start, cur_end = parse_timestamp_range(str(item.get("timestamp") or ""))
    except Exception:
        cur_start, cur_end = 0, 0

    best_scene: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for scene in scenes:
        score = _score_picture_scene_match(picture, scene)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", narration[:80]):
            blob = str(scene.get("block") or "")
            if token in blob:
                score += len(token) * 0.5
        for start_text, end_text in scene.get("timestamp_ranges") or []:
            try:
                seg_start, seg_end = parse_timestamp_range(
                    f"{start_text}-{end_text}"
                )
            except Exception:
                continue
            if seg_start <= cur_start <= seg_end:
                score += 20.0
            distance = min(abs(cur_start - seg_start), abs(cur_start - seg_end))
            score += max(0.0, 10.0 - distance / 60_000.0)
        if score > best_score:
            best_score = score
            best_scene = scene

    if not best_scene:
        return False

    span = _scene_timestamp_span(best_scene)
    if not span:
        return False

    seg_start, seg_end = span
    target_end = max(cur_end, seg_start + ost0_min_ms)
    target_end = min(target_end, seg_end)
    if target_end <= seg_start:
        target_end = min(seg_end, seg_start + ost0_min_ms)

    new_ts = f"{format_timestamp_ms(seg_start)}-{format_timestamp_ms(target_end)}"
    if new_ts == str(item.get("timestamp") or ""):
        return False
    item["timestamp"] = new_ts
    logger.info(
        f"片段 #{item.get('_id')} 已对齐蓝图场景时间窗: {new_ts}"
    )
    return True


def align_script_items_to_source_material(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    plot_blueprint: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """通用对位：OST=1 按 original_line 字幕；冲突时按 picture→蓝图；OST=0 按蓝图场景窗。"""
    if not items:
        return items

    entries = parse_srt(subtitle_content or "")
    if not entries and not (plot_blueprint or "").strip():
        return items

    cfg = get_short_drama_settings(settings)
    ost1_max_ms = int(float(cfg.get("ost1_duration_max", 5) or 5) * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)
    scene_info = parse_scene_segment_blueprint(plot_blueprint)
    scenes: List[Dict[str, Any]] = list(scene_info.get("scenes") or [])

    ordered = sorted([dict(item) for item in items], key=lambda row: int(row.get("_id") or 0))
    fixed = 0

    for item in ordered:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        picture = str(item.get("picture") or "")
        line = strip_dialogue_quotes(str(item.get("original_line") or ""))

        if line and picture and detect_picture_line_incoherence(picture, line):
            if realign_ost1_from_blueprint_scene(
                item,
                scenes=scenes,
                entries=entries,
                ost1_max_ms=ost1_max_ms,
            ):
                fixed += 1
                continue

        if line and entries:
            span = _subtitle_span_for_line(entries, line, max_span_ms=ost1_max_ms)
            if span and not timestamp_aligns_with_subtitle_span(
                str(item.get("timestamp") or ""),
                span,
            ):
                if align_ost1_item_to_subtitle_line(
                    item, entries, ost1_max_ms=ost1_max_ms
                ):
                    fixed += 1
                    continue

        if line and picture and detect_picture_line_incoherence(picture, line):
            if realign_ost1_from_blueprint_scene(
                item,
                scenes=scenes,
                entries=entries,
                ost1_max_ms=ost1_max_ms,
            ):
                fixed += 1

    for item in ordered:
        if int(item.get("OST", 0) or 0) != 0:
            continue
        if item.get("_opening_climax_replay"):
            continue
        if align_ost0_item_to_blueprint_scene(
            item, scenes, ost0_min_ms=ost0_min_ms
        ):
            fixed += 1

    if fixed:
        logger.info(f"短剧解说：通用素材对位已修正 {fixed} 段 timestamp/台词")
    return ordered


def collect_content_timestamp_issues(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    plot_blueprint: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """通用校验：picture / original_line / timestamp 须与字幕或蓝图一致。"""
    issues: List[str] = []
    entries = parse_srt(subtitle_content or "")
    cfg = get_short_drama_settings(settings)
    ost1_max_ms = int(float(cfg.get("ost1_duration_max", 5) or 5) * 1000)

    for item in items:
        item_id = item.get("_id", "?")
        picture = str(item.get("picture") or "")
        line = strip_dialogue_quotes(str(item.get("original_line") or ""))
        ts = str(item.get("timestamp") or "")

        if int(item.get("OST", 0) or 0) != 1:
            continue

        if line and picture and detect_picture_line_incoherence(picture, line):
            issues.append(
                f"片段 #{item_id} picture 与 original_line 语义冲突（须同场景）"
            )
            continue

        if line and entries:
            span = _subtitle_span_for_line(entries, line, max_span_ms=ost1_max_ms)
            if span and not timestamp_aligns_with_subtitle_span(ts, span):
                issues.append(
                    f"片段 #{item_id} timestamp 与 original_line 字幕时间不对位"
                )

    return issues
