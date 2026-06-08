#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说脚本后处理：OST 归一化与原声高光段校验。"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.documentary.documentary_settings import (
    FAZU2_FORBIDDEN_NARRATION_PHRASES,
    FAZU2_WRONG_CHARACTER_NAMES,
    compute_compact_segment_bounds,
    compute_max_ost1_segments,
    compute_ost1_segment_bounds,
    get_documentary_compact_settings,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_fazu2_opening_climax_hint,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    parse_timestamp_range_ms,
    resolve_segment_subtitle_text,
    resolve_segment_time_range,
)
from app.services.documentary.frame_analysis_compact import _collect_top_level_segments
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.documentary.opening_climax_resolver import (
    apply_opening_climax_fix,
    apply_opening_climax_chronological_replay,
)
from app.services.srt_utils import (
    SrtEntry,
    dialogue_match_key,
    find_subtitle_span_for_line,
    format_timestamp_ms,
    normalize_script_timestamp_range,
    parse_srt,
    parse_timestamp_range,
    repair_or_drop_invalid_timestamp_items,
)


def _segment_duration_sec(timestamp: str) -> float:
    start_ms, end_ms = parse_timestamp_range(timestamp or "")
    if end_ms <= start_ms:
        return 0.0
    return (end_ms - start_ms) / 1000.0


def _item_sort_key(item: Dict[str, Any]) -> tuple[int, int]:
    timestamp = str(item.get("timestamp") or "")
    start = timestamp.split("-", 1)[0].strip()
    try:
        start_ms, _ = parse_timestamp_range(start)
    except Exception:
        start_ms = 0
    return start_ms, int(item.get("_id") or 0)


def _is_preservable_ost1_dialogue(narration: str) -> bool:
    text = (narration or "").strip()
    return bool(text) and not text.startswith("播放原片")


def _normalize_fazu2_ost1_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """V2：OST=1 统一为 narration=播放原片 + original_line。"""
    narration = str(item.get("narration") or "").strip()
    original_line = str(item.get("original_line") or "").strip()

    bold_match = re.search(r"\*\*「([^」]+)」\*\*", narration)
    if bold_match:
        extracted = f"「{bold_match.group(1)}」"
        if not original_line:
            original_line = extracted
        narration = "播放原片"
    elif narration.startswith("「") and narration.endswith("」") and len(narration) <= 40:
        if not original_line:
            original_line = narration
        narration = "播放原片"
    elif narration.startswith("播放原片"):
        narration = "播放原片"

    if original_line and not original_line.startswith("「"):
        original_line = f"「{original_line.strip('「」')}」"

    item["narration"] = narration or "播放原片"
    if original_line:
        item["original_line"] = original_line
    return item


def _ms_to_timestamp(ms: int) -> str:
    return format_timestamp_ms(ms)


def _narration_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", (text or "").strip()))


def _min_narration_duration_sec(text: str) -> float:
    chars = _narration_char_count(text)
    if chars <= 0:
        return 0.0
    return (chars / 10.0) * 1.5


_GENERIC_OST0_PICTURE_RE = re.compile(
    r"^(解说引入正叙|解说过渡|解说道别|开场画面|过渡画面|过渡)$"
)


def _is_generic_ost0_picture(picture: str) -> bool:
    text = str(picture or "").strip().strip('"').strip("'")
    if not text:
        return True
    return bool(_GENERIC_OST0_PICTURE_RE.match(text))


def _segment_picture_hint(segment: Dict[str, Any]) -> str:
    for key in ("observation", "action", "scene"):
        text = str(segment.get(key) or "").strip()
        if not text:
            continue
        clause = text.split("；")[0].split(";")[0].strip()
        if len(clause) >= 4:
            return clause[:80]
    return ""


def _find_frame_segment_near_time(
    segments: List[Dict[str, Any]],
    target_ms: int,
) -> Dict[str, Any] | None:
    if not segments:
        return None
    best_segment: Dict[str, Any] | None = None
    best_distance = float("inf")
    for segment in segments:
        clip_range = resolve_segment_time_range(segment)
        if not clip_range or "-" not in clip_range:
            clip_range = str(segment.get("timestamp") or "")
        if not clip_range or "-" not in clip_range:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range_ms(clip_range)
        except Exception:
            continue
        if start_ms <= target_ms <= end_ms:
            return segment
        distance = min(abs(target_ms - start_ms), abs(target_ms - end_ms))
        if distance < best_distance:
            best_distance = distance
            best_segment = segment
    return best_segment


def _align_fazu2_ost0_to_adjacent_ost1(
    items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    frame_analysis_path: str = "",
) -> List[Dict[str, Any]]:
    """
    按 _id 播放顺序调整 OST=0 取画时间：
    - 下一段为 OST=1（铺垫引出原声）：起点 = 下一段原声开始 − ost0_lead_before_ost1_sec
    - 上一段为 OST=1 且非「引出下一段原声」：沿用上一段原声同场 timestamp
    """
    if not items:
        return items

    lead_sec = float(cfg.get("ost0_lead_before_ost1_sec", 10) or 10)
    lead_ms = max(1000, int(lead_sec * 1000))
    gap_ms = 50
    segments = _load_frame_segments(frame_analysis_path)
    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    adjusted = 0

    for index, item in enumerate(ordered):
        if int(item.get("OST", 0)) != 0:
            continue

        next_item = ordered[index + 1] if index + 1 < len(ordered) else None
        prev_item = ordered[index - 1] if index > 0 else None
        narration = str(item.get("narration") or "").strip()
        min_dur_ms = max(3000, int(_min_narration_duration_sec(narration) * 1000))

        target_start_ms: int | None = None
        target_end_ms: int | None = None
        mode = ""

        if next_item is not None and int(next_item.get("OST", 0)) == 1:
            try:
                next_start_ms, next_end_ms = parse_timestamp_range(
                    str(next_item.get("timestamp") or "")
                )
            except Exception:
                next_start_ms = next_end_ms = 0
            if next_start_ms > 0:
                target_end_ms = max(next_start_ms - gap_ms, 0)
                target_start_ms = max(0, target_end_ms - max(min_dur_ms, lead_ms))
                if target_end_ms <= target_start_ms:
                    target_start_ms = max(0, next_start_ms - lead_ms)
                    target_end_ms = target_start_ms + max(min_dur_ms, lead_ms)
                    if target_end_ms > next_start_ms - gap_ms:
                        target_end_ms = max(target_start_ms + 1000, next_start_ms - gap_ms)
                mode = "lead_in"

        elif prev_item is not None and int(prev_item.get("OST", 0)) == 1:
            try:
                prev_start_ms, prev_end_ms = parse_timestamp_range(
                    str(prev_item.get("timestamp") or "")
                )
            except Exception:
                prev_start_ms = prev_end_ms = 0
            if prev_end_ms > prev_start_ms:
                target_start_ms = prev_start_ms
                target_end_ms = max(prev_end_ms, prev_start_ms + min_dur_ms)
                mode = "commentary"

        if target_start_ms is None or target_end_ms is None or target_end_ms <= target_start_ms:
            continue

        try:
            old_start_ms, old_end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        except Exception:
            old_start_ms, old_end_ms = 0, 0

        if (
            abs(old_start_ms - target_start_ms) < 500
            and abs(old_end_ms - target_end_ms) < 500
        ):
            continue

        item.pop("_clip_aligned", None)
        item["timestamp"] = (
            f"{_ms_to_timestamp(target_start_ms)}-{_ms_to_timestamp(target_end_ms)}"
        )
        adjusted += 1

        if segments:
            segment = _find_frame_segment_near_time(segments, target_start_ms)
            if segment and _is_generic_ost0_picture(str(item.get("picture") or "")):
                hint = _segment_picture_hint(segment)
                if hint:
                    item["picture"] = hint

        logger.info(
            f"片段 #{item.get('_id')} OST=0 取画已对齐（{mode}）："
            f"{(old_end_ms - old_start_ms) / 1000.0:.1f}s @ "
            f"{_ms_to_timestamp(old_start_ms)} → "
            f"{(target_end_ms - target_start_ms) / 1000.0:.1f}s @ "
            f"{_ms_to_timestamp(target_start_ms)}"
        )

    if adjusted:
        logger.info(f"罚罪2 OST=0 取画时间对齐：已调整 {adjusted} 段")
    return ordered


def _is_quote_only_narration(text: str) -> bool:
    """narration 是否仅为一句原声台词（应标 OST=1）。"""
    narration = (text or "").strip()
    if not narration or narration.startswith("播放原片"):
        return False
    bold_quote = re.match(r"^\*\*「[^」]+」\*\*\.?$", narration)
    if bold_quote:
        return True
    if len(narration) > 28:
        return False
    if "：" in narration or ":" in narration:
        if len(narration) > 18:
            return False
    if narration.startswith("「") and narration.endswith("」"):
        return True
    if narration.startswith("『") and narration.endswith("』"):
        return True
    if narration.startswith("『") and narration.endswith("』"):
        return True
    if narration.startswith('"') and narration.count('"') >= 2 and len(narration) < 40:
        return True
    compact = re.sub(r"[\s「」『』""\"'（）()。！？!?]", "", narration)
    return len(compact) <= 24 and ("「" in text or "」" in text)


def _is_mechanical_grid_timestamp(timestamp: str) -> bool:
    """检测 00:06:00,000-00:06:12,000 等整分等间隔编造时间戳。"""
    ts = (timestamp or "").strip()
    if not ts:
        return False
    if re.search(r":00:00,000|:06:00,000|:12:00,000|:18:00,000|:24:00,000|:30:00,000|:36:00,000", ts):
        return True
    try:
        start_ms, end_ms = parse_timestamp_range(ts)
    except Exception:
        return False
    duration_ms = end_ms - start_ms
    if duration_ms in (12000, 15000) and start_ms % 360000 == 0:
        return True
    return False


def _normalize_fazu2_script_items(
    items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    max_ost1 = compute_max_ost1_segments(len(items), cfg)
    ost1_count = sum(1 for item in items if int(item.get("OST", 0)) == 1)
    mechanical_hits = 0

    for item in items:
        ts = str(item.get("timestamp") or "")
        if _is_mechanical_grid_timestamp(ts):
            mechanical_hits += 1

        ost = int(item.get("OST", 0))
        narration = str(item.get("narration") or "").strip()
        if ost != 0 or not _is_quote_only_narration(narration):
            continue
        if ost1_count >= max_ost1:
            logger.warning(
                f"片段 #{item.get('_id')} 为裸台词但 OST=1 已满 {max_ost1}，"
                "请合并进前后解说段或删减原声"
            )
            continue
        item["OST"] = 1
        ost1_count += 1
        logger.info(
            f"片段 #{item.get('_id')} narration 仅为台词，已改为 OST=1"
        )

    if mechanical_hits:
        logger.warning(
            f"检测到 {mechanical_hits} 段疑似编造等间隔时间戳"
            "（如 00:06:00-00:06:12），请重新生成并改用字幕/抽帧真实时间"
        )
    return items


_GENERIC_CHARACTER_RE = re.compile(
    r"警员\s*\d+|警察\s*\d+|说话人\s*\d+|男子\s*\d+|女子\s*[A-Z\d]|黑衣人\s*\d+"
)


def _warn_fazu2_forbidden_phrases(item: Dict[str, Any]) -> None:
    narration = str(item.get("narration") or "")
    picture = str(item.get("picture") or "")
    original_line = str(item.get("original_line") or "")
    if int(item.get("OST", 0)) != 0:
        text = f"{narration}\n{original_line}"
    else:
        text = f"{narration}\n{picture}"
    hits = [phrase for phrase in FAZU2_FORBIDDEN_NARRATION_PHRASES if phrase in text]
    if hits:
        logger.warning(
            f"片段 #{item.get('_id')} 含禁用表述 {hits}，建议改用具体人名后重新生成"
        )
    wrong_names = [wrong for wrong, _ in FAZU2_WRONG_CHARACTER_NAMES if wrong in text]
    if wrong_names:
        logger.warning(
            f"片段 #{item.get('_id')} 含错误人名 {wrong_names}，请按字幕规范修正"
        )
    if any(name in text for name in ("胡小跃", "小跃")) and any(
        term in text for term in ("女警", "女警察", "她")
    ):
        logger.warning(
            f"片段 #{item.get('_id')} 胡小跃/小跃段落疑似性别与画面对不上，"
            "请对照抽帧确认人称与职级"
        )
    generic = _GENERIC_CHARACTER_RE.findall(text)
    if generic:
        logger.warning(
            f"片段 #{item.get('_id')} 使用编号式人物称呼 {generic}，"
            "应改为字幕/剧情中的具体人名"
        )


def _adjust_fazu2_ost0_timestamps(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    ordered = sorted(items, key=_item_sort_key)
    next_starts: List[int] = []
    for item in ordered:
        start_ms, _ = parse_timestamp_range(str(item.get("timestamp") or ""))
        next_starts.append(start_ms)
    for index, item in enumerate(ordered):
        if int(item.get("OST", 0)) != 0:
            continue
        if item.get("_clip_aligned"):
            continue
        narration = str(item.get("narration") or "").strip()
        min_sec = _min_narration_duration_sec(narration)
        if min_sec <= 0:
            continue
        start_ms, end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        duration_sec = max(0.0, (end_ms - start_ms) / 1000.0)
        if duration_sec >= min_sec:
            continue
        needed_end_ms = start_ms + int(min_sec * 1000)
        if index + 1 < len(next_starts):
            needed_end_ms = min(needed_end_ms, next_starts[index + 1] - 50)
        if needed_end_ms <= start_ms:
            continue
        item["timestamp"] = (
            f"{_ms_to_timestamp(start_ms)}-{_ms_to_timestamp(needed_end_ms)}"
        )
        logger.info(
            f"片段 #{item.get('_id')} 解说时长 {duration_sec:.1f}s 短于要求 {min_sec:.1f}s，"
            f"已延长至 {(needed_end_ms - start_ms) / 1000.0:.1f}s"
        )
    return ordered


def _load_frame_segments(frame_analysis_path: str) -> List[Dict[str, Any]]:
    path = (frame_analysis_path or "").strip()
    if not path or not os.path.isfile(path):
        return []
    try:
        artifact = load_analysis_artifact(path)
        return _collect_top_level_segments(artifact)
    except Exception as exc:
        logger.warning(f"读取抽帧 JSON 用于 subtitle_entries 对位失败: {exc}")
        return []


def _range_overlap_ms(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _item_match_text(item: Dict[str, Any]) -> str:
    ost = int(item.get("OST", 0))
    if ost == 1:
        return str(item.get("original_line") or item.get("narration") or "")
    return " ".join(
        str(item.get(key) or "")
        for key in ("narration", "picture", "original_line")
    )


def _score_frame_segment_for_item(
    item: Dict[str, Any],
    segment: Dict[str, Any],
    *,
    item_start_ms: int,
    item_end_ms: int,
) -> float:
    clip_range = resolve_segment_time_range(segment)
    if not clip_range or "-" not in clip_range:
        return -1.0
    seg_start, seg_end = parse_timestamp_range_ms(clip_range)
    overlap = _range_overlap_ms(item_start_ms, item_end_ms, seg_start, seg_end)
    if overlap <= 0:
        fallback = str(segment.get("timestamp") or "").strip()
        if fallback and "-" in fallback:
            ts_start, ts_end = parse_timestamp_range_ms(fallback)
            overlap = _range_overlap_ms(item_start_ms, item_end_ms, ts_start, ts_end)
    if overlap <= 0:
        return -1.0

    score = overlap / 1000.0
    line_key = dialogue_match_key(_item_match_text(item))
    subtitle_text = resolve_segment_subtitle_text(segment)
    if line_key and subtitle_text:
        sub_key = dialogue_match_key(subtitle_text)
        if line_key in sub_key or sub_key in line_key:
            score += 8.0
        elif len(line_key) >= 4 and len(sub_key) >= 4 and line_key[:4] == sub_key[:4]:
            score += 4.0

    if int(item.get("OST", 0)) == 0:
        blob = " ".join(
            str(segment.get(key) or "")
            for key in ("scene", "observation", "action", "key_visual")
        )
        picture = str(item.get("picture") or "").strip()
        if picture:
            for token in picture.replace("，", " ").replace("、", " ").split():
                token = token.strip()
                if len(token) >= 2 and token in blob:
                    score += 1.5
    return score


def _segment_subtitle_entries_to_srt(segment: Dict[str, Any]) -> List[SrtEntry]:
    entries = segment.get("subtitle_entries")
    if not isinstance(entries, list):
        return []
    result: List[SrtEntry] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        start = str(item.get("start") or "").strip()
        end = str(item.get("end") or "").strip()
        text = str(item.get("text") or "").strip()
        if not (start and end and text):
            continue
        try:
            start_ms, end_ms = parse_timestamp_range_ms(f"{start}-{end}")
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        result.append(SrtEntry(start_ms=start_ms, end_ms=end_ms, text=text))
    return result


def _resolve_segment_clip_range(
    segment: Dict[str, Any],
    item: Dict[str, Any],
    *,
    item_start_ms: int,
    item_end_ms: int,
    max_ost1_span_ms: int | None = None,
) -> str:
    """从 scene_segment.subtitle_entries 解析剪辑区间；OST=1 优先匹配台词对应条目。"""
    base = resolve_segment_time_range(segment)
    if int(item.get("OST", 0)) != 1:
        return base

    srt_entries = _segment_subtitle_entries_to_srt(segment)
    if not srt_entries:
        return base

    line_text = str(item.get("original_line") or item.get("narration") or "")
    span = find_subtitle_span_for_line(
        srt_entries,
        line_text,
        near_start_ms=item_start_ms,
        near_end_ms=item_end_ms,
        max_span_ms=max_ost1_span_ms or 22_000,
    )
    if not span:
        return base
    cue_start, cue_end = span
    return f"{format_timestamp_ms(cue_start)}-{format_timestamp_ms(cue_end)}"


def _find_frame_clip_range_for_item(
    item: Dict[str, Any],
    segments: List[Dict[str, Any]],
    *,
    item_start_ms: int,
    item_end_ms: int,
    max_ost1_span_ms: int | None = None,
) -> str:
    if not segments:
        return ""
    best_segment: Dict[str, Any] | None = None
    best_score = -1.0
    for segment in segments:
        score = _score_frame_segment_for_item(
            item,
            segment,
            item_start_ms=item_start_ms,
            item_end_ms=item_end_ms,
        )
        if score > best_score:
            best_score = score
            best_segment = segment
    if best_segment is None or best_score < 0:
        return ""
    return _resolve_segment_clip_range(
        best_segment,
        item,
        item_start_ms=item_start_ms,
        item_end_ms=item_end_ms,
        max_ost1_span_ms=max_ost1_span_ms,
    )


def _apply_clip_range_to_item(
    item: Dict[str, Any],
    clip_range: str,
    *,
    previous_end_ms: int | None = None,
    ost1_hard_max_ms: int | None = None,
) -> bool:
    if not clip_range or "-" not in clip_range:
        return False
    clip_start, clip_end = parse_timestamp_range_ms(clip_range)
    if clip_end <= clip_start:
        return False
    if ost1_hard_max_ms and clip_end - clip_start > ost1_hard_max_ms:
        clip_end = clip_start + ost1_hard_max_ms

    clip_start_final = clip_start
    if previous_end_ms is not None and clip_start < previous_end_ms:
        if previous_end_ms < clip_end:
            clip_start_final = previous_end_ms
        else:
            logger.debug(
                f"片段 #{item.get('_id')} 对位区间与前段结束重叠且无法推移，"
                f"保留原 clip_start"
            )

    if clip_end <= clip_start_final:
        return False

    clip_start = clip_start_final

    old_start, old_end = parse_timestamp_range(str(item.get("timestamp") or ""))
    if clip_start == old_start and clip_end == old_end:
        item["_clip_aligned"] = "subtitle_entries"
        return True

    item["timestamp"] = (
        f"{format_timestamp_ms(clip_start)}-{format_timestamp_ms(clip_end)}"
    )
    item["_clip_aligned"] = "subtitle_entries"
    logger.info(
        f"片段 #{item.get('_id')} 已对齐抽帧 subtitle_entries："
        f"{(old_end - old_start) / 1000.0:.1f}s → {(clip_end - clip_start) / 1000.0:.1f}s"
    )
    return True


def _align_items_to_frame_time_ranges(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    ost1_hard_max: float = 0,
    skip_opening_item_id: int = 1,
) -> List[Dict[str, Any]]:
    """将所有片段 timestamp 对齐到抽帧 JSON 的 subtitle_entries（字幕对位剪辑范围）。"""
    segments = _load_frame_segments(frame_analysis_path)
    if not segments:
        return items

    entries = parse_srt(subtitle_content or "")
    max_ost1_span_ms = int(max(3.0, ost1_hard_max or 22) * 1000) if ost1_hard_max else None
    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    previous_end_ms: int | None = None

    for item in ordered:
        item_id = int(item.get("_id") or 0)
        if item_id == skip_opening_item_id and int(item.get("OST", 0)) == 1:
            try:
                _, previous_end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
            except Exception:
                previous_end_ms = None
            continue

        try:
            start_ms, end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        except Exception:
            continue

        ost1_cap = max_ost1_span_ms if int(item.get("OST", 0)) == 1 else None
        clip_range = _find_frame_clip_range_for_item(
            item,
            segments,
            item_start_ms=start_ms,
            item_end_ms=end_ms,
            max_ost1_span_ms=ost1_cap,
        )
        applied = _apply_clip_range_to_item(
            item,
            clip_range,
            previous_end_ms=previous_end_ms,
            ost1_hard_max_ms=ost1_cap,
        )
        if not applied and previous_end_ms is not None and clip_range:
            applied = _apply_clip_range_to_item(
                item,
                clip_range,
                previous_end_ms=None,
                ost1_hard_max_ms=ost1_cap,
            )
            if applied:
                logger.info(
                    f"片段 #{item.get('_id')} 对位：前段结束约束导致区间无效，"
                    f"已改用抽帧 subtitle_entries 原区间"
                )
        if applied:
            try:
                _, previous_end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
            except Exception:
                pass
            continue

        if int(item.get("OST", 0)) != 1 or not entries:
            try:
                _, previous_end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
            except Exception:
                pass
            continue

        line_text = str(item.get("original_line") or item.get("narration") or "")
        span = find_subtitle_span_for_line(
            entries,
            line_text,
            near_start_ms=start_ms,
            near_end_ms=end_ms,
            max_span_ms=max_ost1_span_ms or 22_000,
        )
        if not span:
            continue
        cue_start, cue_end = span
        if cue_end <= cue_start:
            continue
        fallback_range = f"{format_timestamp_ms(cue_start)}-{format_timestamp_ms(cue_end)}"
        applied = _apply_clip_range_to_item(
            item,
            fallback_range,
            previous_end_ms=previous_end_ms,
            ost1_hard_max_ms=ost1_cap,
        )
        if not applied and previous_end_ms is not None:
            applied = _apply_clip_range_to_item(
                item,
                fallback_range,
                previous_end_ms=None,
                ost1_hard_max_ms=ost1_cap,
            )
        if applied:
            item["_clip_aligned"] = "subtitle_fallback"
            logger.info(f"片段 #{item.get('_id')} 原声未命中抽帧段，回退字幕整句对位")
        try:
            _, previous_end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        except Exception:
            pass

    return repair_or_drop_invalid_timestamp_items(
        ordered,
        subtitle_content=subtitle_content,
        ost1_min_duration_ms=max(800, int(max(3.0, ost1_hard_max or 8) * 500)),
    )


def _strip_internal_clip_flags(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in items:
        item.pop("_clip_aligned", None)
        item.pop("_opening_climax_replay", None)
    return items


def _align_ost1_to_complete_subtitle_cues(
    items: List[Dict[str, Any]],
    subtitle_content: str,
    ost1_hard_max: float,
    *,
    frame_analysis_path: str = "",
) -> List[Dict[str, Any]]:
    """兼容旧调用：统一走全片 subtitle_entries 对位。"""
    return _align_items_to_frame_time_ranges(
        items,
        subtitle_content=subtitle_content,
        frame_analysis_path=frame_analysis_path,
        ost1_hard_max=ost1_hard_max,
    )


def _adjust_fazu2_ost1_timestamps(
    items: List[Dict[str, Any]],
    ost1_min: float,
    ost1_max: float = 0,
) -> List[Dict[str, Any]]:
    """金句原声段过短时延长 time戳；在允许范围内尽量接近 ost1_max。"""
    if not items or ost1_min <= 0:
        return items
    ordered = sorted(items, key=_item_sort_key)
    next_starts: List[int] = []
    for item in ordered:
        start_ms, _ = parse_timestamp_range(str(item.get("timestamp") or ""))
        next_starts.append(start_ms)
    min_ms = int(ost1_min * 1000)
    prefer_ms = int((ost1_max or ost1_min) * 1000)
    for index, item in enumerate(ordered):
        if int(item.get("OST", 0)) != 1:
            continue
        if item.get("_clip_aligned"):
            continue
        start_ms, end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        current_ms = end_ms - start_ms
        target_end_ms = start_ms + max(min_ms, prefer_ms)
        if index + 1 < len(next_starts):
            target_end_ms = min(target_end_ms, next_starts[index + 1] - 50)
        if target_end_ms <= end_ms:
            continue
        if target_end_ms <= start_ms:
            continue
        item["timestamp"] = (
            f"{_ms_to_timestamp(start_ms)}-{_ms_to_timestamp(target_end_ms)}"
        )
        logger.info(
            f"片段 #{item.get('_id')} 原声 OST=1 时长 {current_ms / 1000.0:.1f}s，"
            f"已延长至 {(target_end_ms - start_ms) / 1000.0:.1f}s"
        )
    return ordered


def _enforce_narration_after_ost1_by_id(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按 _id 播放顺序：解说不打断原声，夹在两个 OST=1 之间的解说移到后续原声播完之后。"""
    result = [
        dict(item)
        for item in sorted(items, key=lambda item: int(item.get("_id") or 0))
    ]
    moved_count = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 2:
            if (
                int(result[i].get("OST", 0)) == 1
                and int(result[i + 1].get("OST", 0)) == 0
                and int(result[i + 2].get("OST", 0)) == 1
            ):
                ost0 = result.pop(i + 1)
                j = i + 1
                while j < len(result) and int(result[j].get("OST", 0)) == 1:
                    j += 1
                result.insert(j, ost0)
                moved_count += 1
                changed = True
                logger.info(
                    f"解说片段 #{ost0.get('_id')} 从原声段之间移至后续原声播完之后"
                )
            else:
                i += 1

    for index, item in enumerate(result, 1):
        item["_id"] = index

    if moved_count:
        logger.info(f"原声/解说顺序修正：移动 {moved_count} 段解说到原声结束之后")
    return result


def _enforce_fazu2_playback_gaps(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 _id 播放顺序：后一段开始不得早于前一段结束（解说须等原声播完）。"""
    if len(items) < 2:
        return items
    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    for index in range(1, len(ordered)):
        prev = ordered[index - 1]
        curr = ordered[index]
        try:
            _, prev_end = parse_timestamp_range(str(prev.get("timestamp") or ""))
            curr_start, curr_end = parse_timestamp_range(str(curr.get("timestamp") or ""))
        except Exception:
            continue
        if curr_start >= prev_end:
            continue
        if curr.get("_clip_aligned"):
            continue
        gap_ms = 50
        new_start = prev_end + gap_ms
        duration_ms = max(500, curr_end - curr_start)
        new_end = new_start + duration_ms
        curr["timestamp"] = (
            f"{_ms_to_timestamp(new_start)}-{_ms_to_timestamp(new_end)}"
        )
        logger.info(
            f"片段 #{curr.get('_id')} 开始早于前段结束，已顺延至原声整句/片段播完后"
        )
    return ordered


def _warn_fazu2_ost_interleave(items: List[Dict[str, Any]]) -> None:
    """检查解说是否夹在两个原声之间（后处理会重排）。"""
    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    for index in range(1, len(ordered) - 1):
        if (
            int(ordered[index - 1].get("OST", 0)) == 1
            and int(ordered[index].get("OST", 0)) == 0
            and int(ordered[index + 1].get("OST", 0)) == 1
        ):
            logger.warning(
                f"片段 #{ordered[index].get('_id')} 解说夹在两段原声之间，"
                "将移至原声播完之后"
            )


def _break_consecutive_ost1(
    items: List[Dict[str, Any]],
    default_ost: int,
) -> List[Dict[str, Any]]:
    if len(items) < 2:
        return items
    fixed: List[Dict[str, Any]] = []
    for item in items:
        if (
            fixed
            and int(fixed[-1].get("OST", 0)) == 1
            and int(item.get("OST", 0)) == 1
        ):
            prev = dict(fixed[-1])
            prev["OST"] = default_ost
            logger.warning(
                f"片段 #{prev.get('_id')} 与 #{item.get('_id')} 连续 OST=1，"
                f"已将前段改为 OST={default_ost}"
            )
            fixed[-1] = prev
        fixed.append(item)
    return fixed


def _evenly_sample_items(items: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    indices = sorted({min(len(items) - 1, max(0, round(index * step))) for index in range(count)})
    while len(indices) < count:
        for candidate in range(len(items)):
            if candidate not in indices:
                indices.append(candidate)
                break
        else:
            break
    indices = sorted(indices)[:count]
    return [items[index] for index in indices]


def trim_narration_items_to_max(
    items: List[Dict[str, Any]],
    max_count: int,
) -> List[Dict[str, Any]]:
    """按时间线均匀裁剪至 max_count 段，优先保留 OST=1。"""
    if not items or max_count <= 0 or len(items) <= max_count:
        return items

    ordered = sorted(items, key=_item_sort_key)
    ost1_items = [item for item in ordered if int(item.get("OST", 0)) == 1]
    other_items = [item for item in ordered if int(item.get("OST", 0)) != 1]

    if len(ost1_items) >= max_count:
        picked = _evenly_sample_items(ost1_items, max_count)
    else:
        picked = list(ost1_items)
        picked.extend(_evenly_sample_items(other_items, max_count - len(picked)))
        picked.sort(key=_item_sort_key)

    for index, item in enumerate(picked, 1):
        item["_id"] = index

    logger.info(f"解说脚本从 {len(items)} 段裁剪至 {len(picked)} 段（上限 {max_count}）")
    return picked


def trim_compact_script_items(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """精剪模式：超出上限时按时间线均匀裁剪，优先保留 OST=1。"""
    if not items:
        return []
    cfg = get_documentary_compact_settings(settings)
    max_total = int(cfg.get("max_total_segments", 0) or 0)
    if max_total <= 0:
        return items
    return trim_narration_items_to_max(items, max_total)


def _format_compact_hook_template(template: str, work_name: str) -> str:
    name = (work_name or "").strip() or "本期"
    text = (template or "").strip()
    return text.replace("{work_name}", name).replace("某某某", name)


def _opening_hook_already_applied(existing: str, opening: str) -> bool:
    text = (existing or "").strip()
    hook = (opening or "").strip().rstrip("。！!")
    if not text or not hook:
        return False
    if hook in text:
        return True
    prefix_len = min(16, len(hook))
    if prefix_len >= 4 and text.startswith(hook[:prefix_len]):
        return True
    return False


def _closing_hook_already_applied(existing: str, closing: str) -> bool:
    text = (existing or "").strip()
    hook = (closing or "").strip().rstrip("。！!")
    if not text or not hook:
        return False
    if hook in text:
        return True
    tail = hook[-min(12, len(hook)) :]
    return bool(tail) and len(tail) >= 4 and text.endswith(tail)


def apply_compact_opening_closing_hooks(
    items: List[Dict[str, Any]],
    work_name: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """逐帧精剪 V2：为第 2 段补「宝子们」开场，为末段补道别。"""
    if not items:
        return items
    cfg = get_documentary_compact_settings(settings)
    if not is_fazu2_compact_settings(cfg):
        return items
    if not cfg.get("enable_opening_closing_hook", True):
        return items

    transition_tpl = str(
        cfg.get("transition_hook_template") or "故事，得从头讲起。"
    ).strip()
    closing = _format_compact_hook_template(
        str(cfg.get("closing_hook_template") or "宝子们，我们下期再见！"),
        work_name,
    )
    if not transition_tpl and not closing:
        return items

    updated = [dict(item) for item in items]
    ost0_indices = [
        i for i, item in enumerate(updated) if int(item.get("OST", 0)) == 0
    ]
    if not ost0_indices:
        return items

    first_idx = ost0_indices[0]
    last_idx = len(updated) - 1

    if transition_tpl and first_idx == 1:
        existing = str(updated[first_idx].get("narration") or "").strip()
        if not existing.startswith("宝子们"):
            body = existing
            if body.startswith(transition_tpl.rstrip("。")):
                updated[first_idx]["narration"] = f"宝子们，{body}"
            else:
                updated[first_idx]["narration"] = f"宝子们，{transition_tpl}{body}"
            logger.info(
                f"已应用精剪正叙开场: {updated[first_idx]['narration'][:50]}..."
            )

    if closing:
        existing = str(updated[last_idx].get("narration") or "").strip()
        if not _closing_hook_already_applied(existing, closing):
            if existing:
                updated[last_idx]["narration"] = f"{existing.rstrip('。')}。{closing}"
            else:
                updated[last_idx]["narration"] = closing
            logger.info(f"已应用精剪结尾白: {closing}")

    return updated


def finalize_documentary_script_items(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    work_name: str = "",
    subtitle_content: str = "",
    *,
    subtitle_frame_analysis: str = "",
    frame_analysis_path: str = "",
) -> List[Dict[str, Any]]:
    """归一化 LLM 输出的逐帧解说脚本 OST 字段。"""
    if not items:
        return []

    if settings and is_compact_documentary_settings(settings):
        cfg = get_documentary_compact_settings(settings)
    else:
        cfg = get_documentary_settings(settings)

    if (subtitle_content or "").strip() and is_compact_documentary_settings(cfg):
        items = apply_opening_climax_fix(
            items,
            subtitle_content=subtitle_content,
            subtitle_frame_analysis=subtitle_frame_analysis,
            append_custom_prompt=str(cfg.get("append_custom_prompt") or ""),
            opening_hint=resolve_fazu2_opening_climax_hint(cfg),
            frame_analysis_path=frame_analysis_path,
            settings=cfg,
            enabled=bool(cfg.get("enable_original_audio_highlights", True)),
        )
    default_ost = int(cfg.get("default_narration_ost", 2))
    if default_ost not in (0, 2):
        default_ost = 2

    highlights_enabled = bool(cfg.get("enable_original_audio_highlights", True))
    ost1_min = float(cfg.get("ost1_duration_min", 3))
    ost1_max = float(cfg.get("ost1_duration_max", 12))
    ost1_hard_max = float(cfg.get("ost1_duration_hard_max", ost1_max) or ost1_max)
    fazu2_mode = is_fazu2_compact_settings(cfg)
    max_ost1 = compute_max_ost1_segments(len(items), cfg)

    result: List[Dict[str, Any]] = []
    ost1_counter = 0

    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue

        item = dict(raw)
        item["_id"] = int(item.get("_id") or index + 1)

        if not highlights_enabled:
            item["OST"] = default_ost
            result.append(item)
            continue

        try:
            ost = int(item.get("OST", default_ost))
        except (TypeError, ValueError):
            ost = default_ost

        if ost not in (0, 1, 2):
            ost = default_ost

        if default_ost == 0 and ost == 2:
            logger.warning(
                f"片段 #{item['_id']} OST=2 在精剪模式下不可用，转为 OST=0"
            )
            ost = 0

        if ost == 1:
            duration = _segment_duration_sec(str(item.get("timestamp") or ""))
            if ost1_counter >= max_ost1:
                logger.warning(
                    f"片段 #{item['_id']} 标记 OST=1 但已达上限 {max_ost1}，回退为 OST={default_ost}"
                )
                ost = default_ost
            elif duration > 0 and duration < ost1_min:
                if not fazu2_mode:
                    logger.warning(
                        f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 短于 {ost1_min}s，回退为 OST={default_ost}"
                    )
                    ost = default_ost
                else:
                    logger.info(
                        f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 短于 {ost1_min}s，"
                        f"保留原声并在后处理延长 time戳"
                    )
            elif duration > ost1_hard_max:
                logger.warning(
                    f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 超过 {ost1_hard_max}s，"
                    f"回退为 OST={default_ost}"
                )
                ost = default_ost
            elif duration > ost1_max:
                logger.info(
                    f"片段 #{item['_id']} OST=1 时长 {duration:.1f}s 超过建议 {ost1_max}s，保留"
                )

        if ost == 1:
            ost1_counter += 1
            existing_narration = str(item.get("narration") or "").strip()
            if fazu2_mode:
                item = _normalize_fazu2_ost1_fields(item)
                if not str(item.get("original_line") or "").strip():
                    logger.warning(
                        f"片段 #{item['_id']} OST=1 缺少 original_line，请按 V2 模板补充"
                    )
            elif _is_preservable_ost1_dialogue(existing_narration):
                item["narration"] = existing_narration
            else:
                item["narration"] = f"播放原片{ost1_counter}"
            item["OST"] = 1
        else:
            item["OST"] = ost

        result.append(item)

    if highlights_enabled and ost1_counter:
        logger.info(
            f"逐帧解说脚本含 {ost1_counter} 段 OST=1 原声高光"
            f"（上限 {max_ost1}，约每 {cfg.get('ost1_every_n_segments', 10)} 段 1 原声）"
        )

    if fazu2_mode:
        result = _normalize_fazu2_script_items(result, cfg)
        _warn_fazu2_ost_interleave(result)
        for item in result:
            _warn_fazu2_forbidden_phrases(item)
        result = _enforce_narration_after_ost1_by_id(result)
        result = apply_opening_climax_chronological_replay(
            result,
            settings=cfg,
            enabled=bool(cfg.get("enable_opening_climax_chronological_replay", True)),
        )
        result = _enforce_narration_after_ost1_by_id(result)

    if is_compact_documentary_settings(cfg) and (
        (frame_analysis_path or "").strip() or (subtitle_content or "").strip()
    ):
        result = _align_items_to_frame_time_ranges(
            result,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=ost1_hard_max,
        )

    if fazu2_mode:
        result = _adjust_fazu2_ost1_timestamps(result, ost1_min, ost1_max)
        result = _align_fazu2_ost0_to_adjacent_ost1(
            result,
            cfg,
            frame_analysis_path=frame_analysis_path,
        )
        result = _adjust_fazu2_ost0_timestamps(result)
        result = _enforce_fazu2_playback_gaps(result)
        ost0_count = sum(1 for item in result if int(item.get("OST", 0)) == 0)
        ost0_min = int(cfg.get("ost0_segment_min", 30) or 30)
        if ost0_count < ost0_min:
            logger.warning(
                f"罚罪2模式解说段 {ost0_count} 段，低于建议最少 {ost0_min} 段"
            )
        ost1_count = sum(1 for item in result if int(item.get("OST", 0)) == 1)
        min_ost1, max_ost1 = compute_ost1_segment_bounds(len(result), cfg)
        if ost1_count < min_ost1:
            logger.warning(
                f"罚罪2模式原声 OST=1 仅 {ost1_count} 段，低于要求 {min_ost1}–{max_ost1} 段"
            )
        elif ost1_count > max_ost1:
            logger.warning(
                f"罚罪2模式原声 OST=1 共 {ost1_count} 段，超过上限 {max_ost1} 段"
            )

    if is_compact_documentary_settings(cfg):
        min_segments, target_segments, max_segments = compute_compact_segment_bounds(
            cfg, source_duration_sec=None
        )
        item_count = len(result)
        if item_count < min_segments or item_count > max_segments:
            logger.warning(
                f"精剪脚本段数 {item_count} 不在 {min_segments}–{max_segments} 段范围内"
                f"（目标约 {target_segments} 段）"
            )

    if fazu2_mode:
        theme_name = (work_name or "").strip() or str(cfg.get("fazu2_core_theme") or "").strip()
        result = apply_compact_opening_closing_hooks(
            result, work_name=theme_name, settings=cfg
        )

    for item in result:
        raw_ts = str(item.get("timestamp") or "").strip()
        fixed_ts = normalize_script_timestamp_range(raw_ts)
        if fixed_ts != raw_ts:
            logger.info(
                f"片段 #{item.get('_id')} timestamp 已规范: {raw_ts!r} -> {fixed_ts!r}"
            )
        item["timestamp"] = fixed_ts

    return _strip_internal_clip_flags(result)
