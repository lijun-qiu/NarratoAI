#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说脚本后处理：扩展过短的 OST=1 原声片段时间戳，使成片时长接近目标比例。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.update_script import calculate_duration
from app.services.film_tv_settings import (
    TV_CONTENT_SERIES,
    format_tv_line_template,
    get_film_tv_settings,
)
from app.services.srt_utils import extract_entries_in_range, parse_srt

AUTO_NARRATION_MARKER = "__AUTO_NARRATION__"

_SRT_BLOCK_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
    re.MULTILINE,
)


def _time_str_to_seconds(time_str: str) -> float:
    time_str = time_str.strip()
    if "," in time_str:
        main, ms = time_str.split(",", 1)
        milliseconds = int(ms)
    else:
        main = time_str
        milliseconds = 0
    hours, minutes, seconds = map(int, main.split(":"))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def _seconds_to_time_str(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def parse_srt_cues(subtitle_content: str) -> List[Tuple[float, float]]:
    """从 SRT 字幕文本解析时间轴片段（秒）。"""
    if not subtitle_content:
        return []

    cues: List[Tuple[float, float]] = []
    for match in _SRT_BLOCK_RE.finditer(subtitle_content):
        start = _time_str_to_seconds(match.group(1))
        end = _time_str_to_seconds(match.group(2))
        if end > start:
            cues.append((start, end))

    cues.sort(key=lambda item: item[0])
    return cues


def parse_timestamp_range(timestamp: str) -> Tuple[float, float]:
    start_str, end_str = timestamp.split("-", 1)
    return _time_str_to_seconds(start_str), _time_str_to_seconds(end_str)


def format_timestamp_range(start_sec: float, end_sec: float) -> str:
    return f"{_seconds_to_time_str(start_sec)}-{_seconds_to_time_str(end_sec)}"


def estimate_narration_duration(text: str) -> float:
    """估算解说 TTS 时长（秒），中文约 0.35 秒/字。"""
    chars = len(re.sub(r"\s+", "", text or ""))
    return max(3.0, chars * 0.35)


def estimate_output_duration(items: List[Dict[str, Any]]) -> float:
    _, _, total = estimate_duration_breakdown(items)
    return total


def estimate_duration_breakdown(items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """返回 (解说秒数, 原声秒数, 总秒数)。"""
    ost0_sec = 0.0
    ost1_sec = 0.0
    for item in items:
        if item.get("OST") == 1:
            ost1_sec += calculate_duration(item["timestamp"])
        else:
            ost0_sec += estimate_narration_duration(item.get("narration", ""))
    return ost0_sec, ost1_sec, ost0_sec + ost1_sec


def validate_film_tv_duration_ratio(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    *,
    tolerance: float = 8.0,
) -> Dict[str, Any]:
    """校验成片时长中解说/原声占比是否接近配置目标。"""
    settings = get_film_tv_settings(settings)
    ost0_sec, ost1_sec, total = estimate_duration_breakdown(items)
    if total <= 0:
        return {
            "ok": False,
            "narration_sec": ost0_sec,
            "original_sec": ost1_sec,
            "total_sec": total,
            "narration_pct": 0.0,
            "original_pct": 0.0,
            "narration_target": int(settings["narration_percent"]),
            "original_target": int(settings["original_audio_percent"]),
            "message": "无法估算成片时长占比",
        }

    narration_pct = ost0_sec / total * 100.0
    original_pct = ost1_sec / total * 100.0
    target_narr = int(settings["narration_percent"])
    target_orig = int(settings["original_audio_percent"])
    issues: List[str] = []

    if target_narr >= target_orig:
        if narration_pct < target_narr - tolerance:
            issues.append(
                f"解说时长占比 {narration_pct:.0f}%，低于目标 {target_narr}%（容差 {tolerance:.0f}%）"
            )
        if original_pct > target_orig + tolerance:
            issues.append(
                f"原声时长占比 {original_pct:.0f}%，高于目标 {target_orig}%（容差 {tolerance:.0f}%）"
            )
    else:
        if abs(narration_pct - target_narr) > tolerance:
            issues.append(f"解说时长占比 {narration_pct:.0f}%，偏离目标 {target_narr}%")
        if abs(original_pct - target_orig) > tolerance:
            issues.append(f"原声时长占比 {original_pct:.0f}%，偏离目标 {target_orig}%")

    ok = not issues
    return {
        "ok": ok,
        "narration_sec": ost0_sec,
        "original_sec": ost1_sec,
        "total_sec": total,
        "narration_pct": narration_pct,
        "original_pct": original_pct,
        "narration_target": target_narr,
        "original_target": target_orig,
        "message": "时长占比符合配置" if ok else "；".join(issues),
    }


def validate_film_tv_script(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    subtitle_content: str = "",
) -> Dict[str, Any]:
    """段数 + 成片时长占比 + 结构（穿插/时间轴）综合校验。"""
    count_val = validate_film_tv_script_counts(items, settings)
    ratio_val = validate_film_tv_duration_ratio(items, settings)
    struct_val = validate_film_tv_structure(items, subtitle_content, settings)
    issues = []
    if not count_val["ok"]:
        issues.append(count_val["message"])
    if not ratio_val["ok"]:
        issues.append(ratio_val["message"])
    if not struct_val["ok"]:
        issues.append(struct_val["message"])
    return {
        **count_val,
        **{f"ratio_{k}": v for k, v in ratio_val.items() if k not in count_val},
        **{f"struct_{k}": v for k, v in struct_val.items() if k not in count_val and k != "ok"},
        "narration_pct": ratio_val["narration_pct"],
        "original_pct": ratio_val["original_pct"],
        "narration_target": ratio_val["narration_target"],
        "original_target": ratio_val["original_target"],
        "max_consecutive_ost0": struct_val["max_consecutive_ost0"],
        "max_consecutive_ost1": struct_val["max_consecutive_ost1"],
        "unanchored_ost0_count": struct_val["unanchored_ost0_count"],
        "ok": count_val["ok"] and ratio_val["ok"] and struct_val["ok"],
        "message": "；".join(issues) if issues else "段数、时长占比与结构均符合配置",
    }


def max_consecutive_ost_run(items: List[Dict[str, Any]], ost_value: int) -> int:
    """按播放顺序统计连续同类型片段的最大长度。"""
    max_run = 0
    current = 0
    for item in sorted(items, key=_timestamp_sort_key):
        if item.get("OST") == ost_value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _ost0_overlaps_subtitle_cues(
    start_sec: float,
    end_sec: float,
    cues: List[Tuple[float, float]],
    tolerance: float = 2.0,
) -> bool:
    for cue_start, cue_end in cues:
        if cue_end >= start_sec - tolerance and cue_start <= end_sec + tolerance:
            return True
    return False


def validate_film_tv_structure(
    items: List[Dict[str, Any]],
    subtitle_content: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验原声/解说穿插与解说段时间轴是否贴字幕。"""
    settings = get_film_tv_settings(settings)
    max_ost0 = int(settings.get("max_consecutive_ost0", 3))
    max_ost1 = int(settings.get("max_consecutive_ost1", 3))
    cues = parse_srt_cues(subtitle_content)

    max_run_0 = max_consecutive_ost_run(items, 0)
    max_run_1 = max_consecutive_ost_run(items, 1)
    issues: List[str] = []
    if max_run_0 > max_ost0:
        issues.append(f"连续解说 OST=0 达 {max_run_0} 段，上限 {max_ost0}")
    if max_run_1 > max_ost1:
        issues.append(f"连续原声 OST=1 达 {max_run_1} 段，上限 {max_ost1}")

    unanchored = 0
    if cues:
        for item in items:
            if item.get("OST") != 0:
                continue
            try:
                start_sec, end_sec = parse_timestamp_range(item["timestamp"])
            except (ValueError, AttributeError, KeyError):
                unanchored += 1
                continue
            if not _ost0_overlaps_subtitle_cues(start_sec, end_sec, cues):
                unanchored += 1
    elif any(item.get("OST") == 0 for item in items):
        issues.append("缺少字幕素材，无法校验解说时间轴")

    if unanchored:
        issues.append(f"有 {unanchored} 段解说时间戳未对齐字幕")

    ok = not issues
    return {
        "ok": ok,
        "max_consecutive_ost0": max_run_0,
        "max_consecutive_ost1": max_run_1,
        "max_consecutive_ost0_limit": max_ost0,
        "max_consecutive_ost1_limit": max_ost1,
        "unanchored_ost0_count": unanchored,
        "message": "结构符合要求" if ok else "；".join(issues),
    }


def build_film_tv_script_summary(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    subtitle_content: str = "",
) -> Dict[str, Any]:
    """构建页面展示用的成片脚本统计。"""
    settings = get_film_tv_settings(settings)
    validation = validate_film_tv_script(items, settings, subtitle_content)
    ost0_sec, ost1_sec, total_sec = estimate_duration_breakdown(items)
    ost1_count, ost0_count, total_count = count_film_tv_segments(items)
    ost0_items = [item for item in items if item.get("OST") == 0]
    ost1_items = [item for item in items if item.get("OST") == 1]
    return {
        "ost0_count": ost0_count,
        "ost1_count": ost1_count,
        "total_count": total_count,
        "ost0_sec": ost0_sec,
        "ost1_sec": ost1_sec,
        "total_sec": total_sec,
        "narration_pct": validation["narration_pct"],
        "original_pct": validation["original_pct"],
        "narration_target": validation["narration_target"],
        "original_target": validation["original_target"],
        "max_consecutive_ost0": validation["max_consecutive_ost0"],
        "max_consecutive_ost1": validation["max_consecutive_ost1"],
        "max_consecutive_ost0_limit": validation.get("max_consecutive_ost0_limit", 3),
        "max_consecutive_ost1_limit": validation.get("max_consecutive_ost1_limit", 3),
        "unanchored_ost0_count": validation.get("unanchored_ost0_count", 0),
        "validation_ok": validation["ok"],
        "validation_message": validation["message"],
        "ost0_ids": [int(item.get("_id", 0)) for item in sorted(ost0_items, key=_timestamp_sort_key)],
        "ost1_ids": [int(item.get("_id", 0)) for item in sorted(ost1_items, key=_timestamp_sort_key)],
    }


def _allocate_ost0_window_from_cues(
    cues: List[Tuple[float, float]],
    cursor: float,
    *,
    min_span: float = 8.0,
    max_span: float = 22.0,
) -> Optional[Tuple[float, float]]:
    merged = _merge_cues(cues, gap=0.4)
    for start, end in merged:
        if end <= cursor + 0.2:
            continue
        clip_start = max(start, cursor + 0.1)
        clip_end = min(end, clip_start + max_span)
        if clip_end - clip_start >= min_span:
            return clip_start, clip_end
        if end - clip_start >= min_span * 0.75:
            return clip_start, min(end, clip_start + min_span)
    return None


def fix_ost0_timestamps_from_subtitles(
    items: List[Dict[str, Any]],
    subtitle_content: str,
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """将未贴字幕的解说段时间戳重映射到字幕时间轴。"""
    cues = parse_srt_cues(subtitle_content)
    if not cues:
        return items

    settings = get_film_tv_settings(settings)
    result = [dict(item) for item in items]
    if source_duration_sec is None or source_duration_sec <= 0:
        source_duration_sec = max(end for _, end in cues) + 1.0

    ordered = sorted(result, key=_timestamp_sort_key)
    cursor = 0.0
    fixed = 0
    for item in ordered:
        if item.get("OST") == 1:
            try:
                _, end = parse_timestamp_range(item["timestamp"])
                cursor = max(cursor, end)
            except (ValueError, AttributeError, KeyError):
                pass
            continue
        try:
            start_sec, end_sec = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError, KeyError):
            start_sec, end_sec = cursor, cursor + 12.0
        needs_fix = not _ost0_overlaps_subtitle_cues(start_sec, end_sec, cues)
        if not needs_fix:
            cursor = max(cursor, end_sec)
            continue
        window = _allocate_ost0_window_from_cues(cues, cursor)
        if window is None:
            window = _allocate_ost0_window_from_cues(cues, max(0.0, cursor - 30.0))
        if window is None:
            continue
        item["timestamp"] = format_timestamp_range(window[0], window[1])
        cursor = window[1]
        fixed += 1

    if fixed:
        logger.info(f"已将 {fixed} 段解说时间戳重映射到字幕时间轴")
    return finalize_film_tv_playback_order(ordered, settings)


def fix_consecutive_ost0_blocks(
    items: List[Dict[str, Any]],
    subtitle_content: str,
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """连续解说超过上限时，在块中插入原声段以打断。"""
    settings = get_film_tv_settings(settings)
    max_run = int(settings.get("max_consecutive_ost0", 3))
    cues = parse_srt_cues(subtitle_content)
    if not cues:
        return items

    if source_duration_sec is None or source_duration_sec <= 0:
        source_duration_sec = max(end for _, end in cues) + 1.0

    result = finalize_film_tv_playback_order([dict(item) for item in items], settings)
    ost1_min = int(settings["ost1_duration_min"])
    ost1_max = int(settings["ost1_duration_max"])
    next_id = max((int(i.get("_id", 0) or 0) for i in result), default=0) + 1
    inserted = 0

    changed = True
    while changed:
        changed = False
        run = 0
        idx = 0
        while idx < len(result):
            if result[idx].get("OST") == 0:
                run += 1
                if run > max_run:
                    run_start_idx = idx - run + 1
                    block_start = 0.0
                    if run_start_idx >= 0:
                        try:
                            block_start = parse_timestamp_range(
                                result[run_start_idx]["timestamp"]
                            )[0]
                        except (ValueError, AttributeError, KeyError):
                            block_start = 0.0
                    occupied_ost1 = _merge_time_ranges(
                        [
                            parse_timestamp_range(item["timestamp"])
                            for item in result
                            if item.get("OST") == 1
                        ]
                    )
                    inserted_clip = False
                    for cue_start, cue_end in cues:
                        if cue_start + ost1_min > cue_end:
                            continue
                        if cue_start < block_start - 5.0:
                            continue
                        clip_start = cue_start
                        clip_end = min(cue_end, clip_start + ost1_max)
                        if clip_end - clip_start < ost1_min:
                            continue
                        candidate = (clip_start, clip_end)
                        if any(
                            _range_overlaps(candidate, occ)
                            for occ in occupied_ost1
                        ):
                            continue
                        result.insert(
                            idx,
                            {
                                "_id": next_id,
                                "timestamp": format_timestamp_range(clip_start, clip_end),
                                "picture": "剧情高光",
                                "narration": f"播放原片{next_id}",
                                "OST": 1,
                            },
                        )
                        next_id += 1
                        inserted += 1
                        changed = True
                        inserted_clip = True
                        run = 0
                        idx += 1
                        break
                    if inserted_clip:
                        continue
                    merge_idx = idx - 1
                    if merge_idx >= 0 and result[merge_idx].get("OST") == 0:
                        prev_narr = str(result[merge_idx].get("narration") or "")
                        cur_narr = str(result[idx].get("narration") or "")
                        merged = (prev_narr + cur_narr)[: int(settings["narration_chars_max"]) + 20]
                        result[merge_idx]["narration"] = merged
                        result.pop(idx)
                        changed = True
                        run -= 1
                        continue
                    run = max_run
            else:
                run = 0
            idx += 1

    if inserted:
        logger.info(f"已在连续解说块中插入 {inserted} 段原声以符合穿插规则")

    for _ in range(40):
        if max_consecutive_ost_run(result, 0) <= max_run:
            break
        before_len = len(result)
        inner_changed = True
        while inner_changed:
            inner_changed = False
            run = 0
            idx = 0
            while idx < len(result):
                if result[idx].get("OST") == 0:
                    run += 1
                    if run > max_run:
                        merge_idx = idx - 1
                        if merge_idx >= 0 and result[merge_idx].get("OST") == 0:
                            prev_narr = str(result[merge_idx].get("narration") or "")
                            cur_narr = str(result[idx].get("narration") or "")
                            cap = int(settings.get("narration_chars_max", 78)) + 20
                            result[merge_idx]["narration"] = (prev_narr + cur_narr)[:cap]
                            result.pop(idx)
                            inner_changed = True
                            run -= 1
                            continue
                else:
                    run = 0
                idx += 1
        if len(result) == before_len and not inner_changed:
            break

    return finalize_film_tv_playback_order(result, settings)


def fix_film_tv_script_structure(
    items: List[Dict[str, Any]],
    subtitle_content: str = "",
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """校正解说时间轴与连续解说块。"""
    settings = get_film_tv_settings(settings)
    result = fix_ost0_timestamps_from_subtitles(
        items, subtitle_content, source_duration_sec, settings
    )
    result = fix_consecutive_ost0_blocks(
        result, subtitle_content, source_duration_sec, settings
    )
    result = finalize_film_tv_playback_order(result, settings)
    if max_consecutive_ost_run(result, 0) > int(settings.get("max_consecutive_ost0", 3)):
        result = fix_consecutive_ost0_blocks(
            result, subtitle_content, source_duration_sec, settings
        )
        result = finalize_film_tv_playback_order(result, settings)
    return result


def _merge_cues(cues: List[Tuple[float, float]], gap: float = 0.3) -> List[Tuple[float, float]]:
    if not cues:
        return []
    merged = [cues[0]]
    for start, end in cues[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_cues_near(cues: List[Tuple[float, float]], center: float, window: float = 1.0) -> List[Tuple[float, float]]:
    return [(s, e) for s, e in cues if e >= center - window and s <= center + window]


def _occupied_ranges(items: List[Dict[str, Any]], exclude_index: int) -> List[Tuple[float, float]]:
    """仅 OST=1 片段占用原片时间轴；OST=0 结束时间不参与裁剪，不应阻挡扩展。"""
    ranges: List[Tuple[float, float]] = []
    for idx, item in enumerate(items):
        if idx == exclude_index or item.get("OST") != 1:
            continue
        try:
            ranges.append(parse_timestamp_range(item["timestamp"]))
        except (ValueError, AttributeError):
            continue
    return sorted(ranges, key=lambda r: r[0])


def _range_overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _can_use_range(candidate: Tuple[float, float], occupied: List[Tuple[float, float]]) -> bool:
    return not any(_range_overlaps(candidate, other) for other in occupied)


def expand_ost1_segment(
    start_sec: float,
    end_sec: float,
    cues: List[Tuple[float, float]],
    occupied: List[Tuple[float, float]],
    min_duration: float,
    max_duration: float,
    source_duration: float,
) -> Tuple[float, float]:
    """将 OST=1 片段时间戳扩展到合理长度，且不与其他片段重叠。"""
    duration = end_sec - start_sec
    if duration >= min_duration:
        return start_sec, end_sec

    center = (start_sec + end_sec) / 2.0
    nearby = _find_cues_near(cues, center, window=2.0)
    if nearby:
        start_sec = min(start_sec, min(s for s, _ in nearby))
        end_sec = max(end_sec, max(e for _, e in nearby))

    # 对称扩展直到达到最小时长；若一侧被占用则只向另一侧扩展
    while end_sec - start_sec < min_duration:
        grow = min_duration - (end_sec - start_sec) + 0.05
        candidates = []
        if start_sec > 0:
            candidates.append((max(0.0, start_sec - grow), end_sec))
        candidates.append((start_sec, min(source_duration, end_sec + grow)))
        if start_sec > 0 and end_sec < source_duration:
            half = grow / 2.0
            candidates.append((max(0.0, start_sec - half), min(source_duration, end_sec + half)))

        best = None
        best_len = end_sec - start_sec
        for cand_start, cand_end in candidates:
            candidate = (cand_start, cand_end)
            if not _can_use_range(candidate, occupied):
                continue
            cand_len = cand_end - cand_start
            if cand_len > best_len:
                best = candidate
                best_len = cand_len

        if best is None:
            break
        start_sec, end_sec = best
        if start_sec <= 0.0 and end_sec >= source_duration:
            break

    # 继续向两侧吸收相邻字幕，直到 max_duration
    merged_cues = _merge_cues(cues)
    cue_idx = None
    for i, (cs, ce) in enumerate(merged_cues):
        if cs <= center <= ce or (cs <= end_sec and ce >= start_sec):
            cue_idx = i
            break

    if cue_idx is not None:
        lo, hi = cue_idx, cue_idx
        while hi - lo + 1 <= 20:
            cand_start = merged_cues[lo][0]
            cand_end = merged_cues[hi][1]
            if cand_end - cand_start > max_duration:
                break
            candidate = (cand_start, cand_end)
            if _can_use_range(candidate, occupied):
                start_sec, end_sec = cand_start, cand_end
            if end_sec - start_sec >= max_duration:
                break
            expanded = False
            if lo > 0:
                test = (merged_cues[lo - 1][0], end_sec)
                if test[1] - test[0] <= max_duration and _can_use_range(test, occupied):
                    lo -= 1
                    start_sec = test[0]
                    expanded = True
            if hi < len(merged_cues) - 1:
                test = (start_sec, merged_cues[hi + 1][1])
                if test[1] - test[0] <= max_duration and _can_use_range(test, occupied):
                    hi += 1
                    end_sec = test[1]
                    expanded = True
            if not expanded:
                break

    if end_sec - start_sec > max_duration:
        half = (start_sec + end_sec) / 2.0
        start_sec = half - max_duration / 2.0
        end_sec = half + max_duration / 2.0
        start_sec = max(0.0, start_sec)
        end_sec = min(source_duration, end_sec)

    return start_sec, end_sec


def enforce_narration_after_ost1(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    确保 OST=0 解说不打断 OST=1 原声：将夹在两个原声段之间的解说移到后续原声段播完之后。
    """
    result = [dict(item) for item in sorted(items, key=_timestamp_sort_key)]
    moved_count = 0

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 2:
            if (
                result[i].get("OST") == 1
                and result[i + 1].get("OST") == 0
                and result[i + 2].get("OST") == 1
            ):
                ost0 = result.pop(i + 1)
                j = i + 1
                while j < len(result) and result[j].get("OST") == 1:
                    j += 1
                result.insert(j, ost0)
                moved_count += 1
                changed = True
                logger.info(
                    f"解说片段 #{ost0.get('_id')} 从原声段之间移至后续原声播完之后"
                )
            else:
                i += 1

    for idx, item in enumerate(result, 1):
        item["_id"] = idx

    if moved_count:
        logger.info(f"原声/解说顺序修正：移动 {moved_count} 段解说到原声结束之后")

    return result


def normalize_ost_types(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """影视解说模式禁止使用 OST=2，转换为 OST=1（保留原声）。"""
    for item in items:
        if item.get("OST") == 2:
            item["OST"] = 1
            logger.warning(f"片段 #{item.get('_id')} OST=2 已转为 OST=1（原声播放期间禁止解说叠加）")
    return items


def count_film_tv_segments(items: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """统计 OST=1 / OST=0 / 总段数。"""
    ost1 = sum(1 for item in items if item.get("OST") == 1)
    ost0 = sum(1 for item in items if item.get("OST") == 0)
    return ost1, ost0, len(items)


def validate_film_tv_script_counts(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    校验影视解说脚本是否满足配置中的最少段数要求。

    Returns:
        ok, ost1_count, ost0_count, total, ost1_min, ost0_min, total_min, message
    """
    settings = get_film_tv_settings(settings)
    ost1_count, ost0_count, total = count_film_tv_segments(items)
    ost1_min = int(settings["ost1_segment_min"])
    ost0_min = int(settings["ost0_segment_min"])
    total_min = ost1_min + ost0_min

    issues: List[str] = []
    if ost1_count < ost1_min:
        issues.append(f"原声 OST=1 仅 {ost1_count} 段，要求至少 {ost1_min} 段")
    if ost0_count < ost0_min:
        issues.append(f"解说 OST=0 仅 {ost0_count} 段，要求至少 {ost0_min} 段")
    if total < total_min:
        issues.append(f"总段数 {total}，要求至少 {total_min} 段（{ost1_min}+{ost0_min}）")

    ok = not issues
    message = "段数符合配置要求" if ok else "；".join(issues)
    return {
        "ok": ok,
        "ost1_count": ost1_count,
        "ost0_count": ost0_count,
        "total": total,
        "ost1_min": ost1_min,
        "ost0_min": ost0_min,
        "total_min": total_min,
        "message": message,
    }


def _collect_item_ranges(items: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for item in items:
        try:
            ranges.append(parse_timestamp_range(item["timestamp"]))
        except (ValueError, AttributeError, KeyError):
            continue
    return ranges


def _merge_time_ranges(ranges: List[Tuple[float, float]], gap: float = 0.5) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in sorted(ranges, key=lambda r: r[0])[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_timeline_gaps(
    occupied: List[Tuple[float, float]],
    source_duration: float,
    min_gap: float,
) -> List[Tuple[float, float]]:
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in _merge_time_ranges(occupied):
        if start - cursor >= min_gap:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if source_duration - cursor >= min_gap:
        gaps.append((cursor, source_duration))
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    return gaps


def _picture_hint_from_subtitle(
    srt_entries: list,
    start_sec: float,
    end_sec: float,
) -> str:
    from app.services.picture_narration_builder import (
        build_picture_narration_from_subtitle_context,
    )

    return build_picture_narration_from_subtitle_context(
        srt_entries,
        start_sec,
        end_sec,
    )


def _timestamp_sort_key(item: Dict[str, Any]) -> Tuple[float, int]:
    try:
        start, _ = parse_timestamp_range(item["timestamp"])
        return start, int(item.get("_id", 0) or 0)
    except (ValueError, AttributeError, KeyError):
        return 0.0, int(item.get("_id", 0) or 0)


def _renumber_items_by_time(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(items, key=_timestamp_sort_key)
    for idx, item in enumerate(ordered, 1):
        item["_id"] = idx
    return ordered


def is_ost_grouped_by_type(items: List[Dict[str, Any]]) -> bool:
    """检测脚本是否按 OST 类型分组（先全部原声再全部解说，或反之）。"""
    if len(items) < 2:
        return False
    has0 = has1 = False
    seen_ost0 = False
    for item in items:
        ost = item.get("OST")
        if ost == 0:
            has0 = True
            seen_ost0 = True
        elif ost == 1:
            has1 = True
            if seen_ost0:
                return False
    if not (has0 and has1):
        return False
    first_ost = items[0].get("OST")
    last_ost = items[-1].get("OST")
    return first_ost != last_ost


def finalize_film_tv_playback_order(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """按原片时间轴重排 _id 播放顺序，使 OST=0 与 OST=1 穿插排列。"""
    if not items:
        return items
    settings = get_film_tv_settings(settings)
    ordered = _renumber_items_by_time([dict(item) for item in items])
    if settings.get("enforce_narration_after_ost1", True):
        ordered = enforce_narration_after_ost1(ordered)
    return ordered


def _estimate_narrative_end_sec(
    items: List[Dict[str, Any]],
    source_duration_sec: float,
) -> float:
    """估算剧情有效结束时间，避免在片尾空白处补解说段。"""
    ost1_ends: List[float] = []
    ends: List[float] = []
    for item in items:
        if str(item.get("picture") or "").strip() in ("剧情过渡", "本集收尾"):
            continue
        try:
            start, end = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError, KeyError):
            continue
        ends.append(end)
        if item.get("OST") == 1:
            ost1_ends.append(end)
    if ost1_ends:
        return min(source_duration_sec, max(ost1_ends) + 25.0)
    if not ends:
        return source_duration_sec * 0.92
    ends.sort()
    pivot = ends[int(len(ends) * 0.88)]
    return min(source_duration_sec, pivot + 20.0)


def _find_ost0_indices_by_time(items: List[Dict[str, Any]]) -> List[int]:
    indexed: List[Tuple[int, float]] = []
    for idx, item in enumerate(items):
        if item.get("OST") != 0:
            continue
        try:
            start, _ = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError, KeyError):
            start = float(idx)
        indexed.append((idx, start))
    indexed.sort(key=lambda pair: pair[1])
    return [idx for idx, _ in indexed]


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _strip_opening_from_picture(picture: str, opening_line: str) -> str:
    pic = (picture or "").strip()
    opening = (opening_line or "").strip()
    if opening and pic.startswith(opening):
        pic = pic[len(opening):].lstrip("，。！! ")
    return pic or "开场画面"


def _clamp_ost0_timestamp(item: Dict[str, Any], *, max_span: float = 22.0) -> None:
    if item.get("OST") != 0:
        return
    try:
        start, end = parse_timestamp_range(item["timestamp"])
    except (ValueError, AttributeError, KeyError):
        return
    if end - start > max_span:
        item["timestamp"] = format_timestamp_range(start, start + max_span)


def _remove_late_filler_narrations(
    items: List[Dict[str, Any]],
    closing_line: str,
    narrative_end_sec: float,
) -> List[Dict[str, Any]]:
    """删除片尾空白处的多余解说段，并去掉非最后一段上的收尾话术。"""
    closing_norm = _normalize_text_for_match(closing_line)
    ost0_indices = _find_ost0_indices_by_time(items)
    if not ost0_indices:
        return items

    closing_idx = -1
    candidates: List[Tuple[float, int]] = []
    for idx in ost0_indices:
        narr = _normalize_text_for_match(str(items[idx].get("narration") or ""))
        if not (closing_norm and closing_norm in narr):
            continue
        try:
            start, _ = parse_timestamp_range(items[idx]["timestamp"])
        except (ValueError, AttributeError, KeyError):
            continue
        if start <= narrative_end_sec + 1.0:
            candidates.append((start, idx))
    if candidates:
        closing_idx = max(candidates, key=lambda pair: pair[0])[1]
    else:
        eligible = []
        for idx in ost0_indices:
            try:
                start, _ = parse_timestamp_range(items[idx]["timestamp"])
            except (ValueError, AttributeError, KeyError):
                continue
            if start <= narrative_end_sec + 1.0:
                eligible.append((start, idx))
        closing_idx = max(eligible, key=lambda pair: pair[0])[1] if eligible else ost0_indices[-1]

    try:
        closing_end = parse_timestamp_range(items[closing_idx]["timestamp"])[1]
    except (ValueError, AttributeError, KeyError):
        closing_end = narrative_end_sec

    keep: List[Dict[str, Any]] = []
    removed = 0
    for idx, item in enumerate(items):
        try:
            start, end = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError, KeyError):
            keep.append(item)
            continue

        if item.get("OST") == 0 and idx != closing_idx:
            narr = str(item.get("narration") or "")
            if closing_line and closing_line in narr:
                item = dict(item)
                item["narration"] = narr.replace(closing_line, "").strip()

        if (
            item.get("OST") == 0
            and idx != closing_idx
            and start > closing_end + 0.5
        ):
            removed += 1
            continue
        if (
            item.get("OST") == 0
            and idx != closing_idx
            and str(item.get("picture") or "").strip() == "剧情过渡"
            and start > narrative_end_sec - 60.0
        ):
            removed += 1
            continue
        keep.append(item)

    if removed:
        logger.info(f"已移除 {removed} 段片尾多余解说（避免破坏收尾规则）")
    return keep


def _trim_excess_ost1_segments(
    items: List[Dict[str, Any]],
    max_count: int,
    min_keep: int,
) -> List[Dict[str, Any]]:
    """原声段数超过上限时，优先移除最短的原声段。"""
    result = [dict(item) for item in items]
    while True:
        ost1_items: List[Tuple[int, float]] = []
        for idx, item in enumerate(result):
            if item.get("OST") != 1:
                continue
            try:
                start, end = parse_timestamp_range(item["timestamp"])
                ost1_items.append((idx, end - start))
            except (ValueError, AttributeError, KeyError):
                ost1_items.append((idx, 0.0))
        if len(ost1_items) <= max_count or len(ost1_items) <= min_keep:
            break
        drop_idx = min(ost1_items, key=lambda pair: pair[1])[0]
        result.pop(drop_idx)
    return result


def _shrink_longest_ost1_segment(
    items: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> bool:
    """缩短一段原声时长以降低原声占比。"""
    min_duration = float(settings["ost1_duration_min"])
    candidates: List[Tuple[int, float, float, float]] = []
    for idx, item in enumerate(items):
        if item.get("OST") != 1:
            continue
        try:
            start, end = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError, KeyError):
            continue
        duration = end - start
        if duration > min_duration + 0.4:
            candidates.append((idx, start, end, duration))
    if not candidates:
        return False
    idx, start, end, duration = max(candidates, key=lambda row: row[3])
    new_end = max(start + min_duration, end - 1.0)
    if new_end >= end - 0.05:
        return False
    items[idx]["timestamp"] = format_timestamp_range(start, new_end)
    return True


def _remove_shortest_ost1_segment(
    items: List[Dict[str, Any]],
    *,
    min_keep: int,
) -> bool:
    ost1_items: List[Tuple[int, float]] = []
    for idx, item in enumerate(items):
        if item.get("OST") != 1:
            continue
        try:
            start, end = parse_timestamp_range(item["timestamp"])
            ost1_items.append((idx, end - start))
        except (ValueError, AttributeError, KeyError):
            ost1_items.append((idx, 0.0))
    if len(ost1_items) <= min_keep:
        return False
    drop_idx = min(ost1_items, key=lambda pair: pair[1])[0]
    items.pop(drop_idx)
    return True


def _pad_narration_text_for_ratio(
    items: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> bool:
    """为偏短的解说段补字，提高解说时长占比。"""
    chars_min = int(settings["narration_chars_min"])
    chars_max = int(settings["narration_chars_max"])
    for item in items:
        if item.get("OST") != 0:
            continue
        text = str(item.get("narration") or "").strip()
        if not text or text.startswith("播放原片"):
            continue
        plain_len = len(re.sub(r"\s+", "", text))
        if plain_len >= chars_min:
            continue
        padded = text + "。这一段冲突正在升级，局势也越发扑朔迷离。"
        if len(re.sub(r"\s+", "", padded)) > chars_max:
            padded = padded[:chars_max]
        item["narration"] = padded
        return True
    return False


def rebalance_film_tv_duration_ratio(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    按配置的解说/原声时长占比校正脚本：
    限制原声段数、缩短过长原声、补全偏短解说。
    """
    settings = get_film_tv_settings(settings)
    result = [dict(item) for item in items]
    target_narr = int(settings["narration_percent"]) / 100.0
    target_orig = int(settings["original_audio_percent"]) / 100.0
    ost1_max = int(settings["ost1_segment_max"])
    ost1_min = int(settings["ost1_segment_min"])

    result = _trim_excess_ost1_segments(result, ost1_max, ost1_min)

    for _ in range(80):
        _, ost1_sec, total = estimate_duration_breakdown(result)
        if total <= 0:
            break
        orig_ratio = ost1_sec / total
        narr_ratio = 1.0 - orig_ratio
        if narr_ratio >= target_narr - 0.05 and orig_ratio <= target_orig + 0.05:
            break
        if orig_ratio > target_orig + 0.05:
            if _shrink_longest_ost1_segment(result, settings):
                continue
            if _remove_shortest_ost1_segment(result, min_keep=ost1_min):
                continue
            break
        if narr_ratio < target_narr - 0.05:
            if _pad_narration_text_for_ratio(result, settings):
                continue
            break
        break

    result = finalize_film_tv_playback_order(result, settings)
    ratio = validate_film_tv_duration_ratio(result, settings)
    logger.info(
        f"时长占比校正: 解说 {ratio['narration_pct']:.0f}% / 原声 {ratio['original_pct']:.0f}% "
        f"(目标 {ratio['narration_target']}/{ratio['original_target']})"
    )
    return result


def supplement_film_tv_segment_counts(
    items: List[Dict[str, Any]],
    subtitle_content: str = "",
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    当 LLM 输出段数不足时，依据字幕时间轴在空白区间自动补 OST=1 / OST=0 段。
    补入的 OST=0 解说文案标记为 AUTO_NARRATION_MARKER，需后续 LLM 填充。
    """
    settings = get_film_tv_settings(settings)
    result = [dict(item) for item in items]
    validation = validate_film_tv_script_counts(result, settings)
    ratio = validate_film_tv_duration_ratio(result, settings)
    if validation["ok"] and ratio["ok"]:
        struct = validate_film_tv_structure(result, subtitle_content, settings)
        if struct["ok"]:
            return result, validate_film_tv_script(result, settings, subtitle_content)

    narration_heavy = int(settings["narration_percent"]) >= int(settings["original_audio_percent"])
    ost1_max_count = int(settings["ost1_segment_max"])
    ost1_min_count = int(settings["ost1_segment_min"])
    result = _trim_excess_ost1_segments(result, ost1_max_count, ost1_min_count)
    validation = validate_film_tv_script_counts(result, settings)

    cues = parse_srt_cues(subtitle_content)
    srt_entries = parse_srt(subtitle_content) if subtitle_content else []

    if source_duration_sec is None or source_duration_sec <= 0:
        if cues:
            source_duration_sec = max(end for _, end in cues) + 1.0
        else:
            ends = [end for _, end in _collect_item_ranges(result)]
            source_duration_sec = (max(ends) + 1.0) if ends else 600.0

    narrative_end_sec = _estimate_narrative_end_sec(result, source_duration_sec)

    ost1_min = int(settings["ost1_duration_min"])
    ost1_max = int(settings["ost1_duration_max"])
    next_id = max((int(i.get("_id", 0) or 0) for i in result), default=0) + 1

    def _supplement_ost0_segments() -> int:
        nonlocal next_id
        validation_local = validate_film_tv_script_counts(result, settings)
        ost0_need = max(0, validation_local["ost0_min"] - validation_local["ost0_count"])
        if ost0_need <= 0:
            return 0
        occupied = _merge_time_ranges(_collect_item_ranges(result))
        gaps = _find_timeline_gaps(occupied, source_duration_sec, min_gap=10.0)
        added = 0
        for gap_start, gap_end in gaps:
            if added >= ost0_need:
                break
            if gap_start > narrative_end_sec:
                continue
            available = min(gap_end, narrative_end_sec) - gap_start
            if available < 10.0:
                continue
            seg_len = min(15.0, max(12.0, available * 0.8))
            seg_start = gap_start + (available - seg_len) / 2.0
            seg_end = seg_start + seg_len
            result.append(
                {
                    "_id": next_id,
                    "timestamp": format_timestamp_range(seg_start, seg_end),
                    "picture": _picture_hint_from_subtitle(srt_entries, seg_start, seg_end),
                    "narration": AUTO_NARRATION_MARKER,
                    "OST": 0,
                }
            )
            next_id += 1
            added += 1
        if added:
            logger.info(f"自动补入 {added} 段 OST=0 解说（待填充文案，来自时间轴空白）")

        validation_local = validate_film_tv_script_counts(result, settings)
        ost0_need = max(0, validation_local["ost0_min"] - validation_local["ost0_count"])
        if ost0_need > 0:
            ost0_min_target = int(settings["ost0_segment_min"])
            bin_size = source_duration_sec / max(ost0_min_target, 1)
            ost0_starts = {
                parse_timestamp_range(item["timestamp"])[0]
                for item in result
                if item.get("OST") == 0
            }
            for bin_idx in range(ost0_min_target):
                if ost0_need <= 0:
                    break
                bin_start = bin_idx * bin_size
                bin_end = min(source_duration_sec, (bin_idx + 1) * bin_size)
                if bin_start > narrative_end_sec:
                    continue
                if any(bin_start <= start < bin_end for start in ost0_starts):
                    continue
                seg_len = min(15.0, max(12.0, bin_end - bin_start - 0.5))
                if seg_len < 10.0 or bin_end - bin_start < 10.0:
                    continue
                seg_start = bin_start + max(0.0, (bin_end - bin_start - seg_len) / 2.0)
                seg_end = seg_start + seg_len
                result.append(
                    {
                        "_id": next_id,
                        "timestamp": format_timestamp_range(seg_start, seg_end),
                        "picture": _picture_hint_from_subtitle(srt_entries, seg_start, seg_end),
                        "narration": AUTO_NARRATION_MARKER,
                        "OST": 0,
                    }
                )
                next_id += 1
                added += 1
                ost0_need -= 1
                ost0_starts.add(seg_start)
            if added:
                logger.info(f"自动补入 {added} 段 OST=0 解说（均匀分桶，待填充文案）")
        return added

    def _supplement_ost1_segments() -> int:
        nonlocal next_id
        validation_local = validate_film_tv_script_counts(result, settings)
        ost1_count = validation_local["ost1_count"]
        if ost1_count >= ost1_max_count:
            return 0
        ost1_need = max(0, validation_local["ost1_min"] - ost1_count)
        if ost1_need <= 0:
            return 0
        if narration_heavy:
            ratio_local = validate_film_tv_duration_ratio(result, settings)
            if ratio_local["original_pct"] > ratio_local["original_target"] + 5:
                logger.info("解说占比模式：原声时长已偏高，跳过自动补原声段")
                return 0
        if not cues:
            return 0
        occupied = _merge_time_ranges(_collect_item_ranges(result))
        gaps = _find_timeline_gaps(occupied, source_duration_sec, min_gap=float(ost1_min))
        added = 0
        for gap_start, gap_end in gaps:
            if added >= ost1_need or ost1_count + added >= ost1_max_count:
                break
            gap_cues = [(s, e) for s, e in cues if e > gap_start and s < gap_end]
            if not gap_cues:
                continue
            clip_start = max(gap_start, min(s for s, _ in gap_cues))
            clip_end = min(gap_end, max(e for _, e in gap_cues))
            duration = clip_end - clip_start
            if duration < ost1_min:
                continue
            if duration > ost1_max:
                clip_end = clip_start + ost1_max
            result.append(
                {
                    "_id": next_id,
                    "timestamp": format_timestamp_range(clip_start, clip_end),
                    "picture": _picture_hint_from_subtitle(srt_entries, clip_start, clip_end),
                    "narration": f"播放原片{next_id}",
                    "OST": 1,
                }
            )
            next_id += 1
            added += 1
        if added:
            logger.info(f"自动补入 {added} 段 OST=1 原声")
        return added

    if narration_heavy:
        _supplement_ost0_segments()
        _supplement_ost1_segments()
    else:
        _supplement_ost1_segments()
        _supplement_ost0_segments()

    result = _renumber_items_by_time(result)
    result = fix_film_tv_script_structure(
        result, subtitle_content, source_duration_sec, settings
    )
    result = rebalance_film_tv_duration_ratio(result, settings)
    validation = validate_film_tv_script(result, settings, subtitle_content)
    if not validation["ok"]:
        logger.warning(f"自动补段后仍未达标: {validation['message']}")
    else:
        logger.info("自动补段后段数与时长占比已达标")
    return result, validation


def fill_auto_narration_placeholders(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """无 LLM 时用 picture 生成兜底解说，去掉 AUTO 标记。"""
    settings = get_film_tv_settings(settings)
    chars_min = int(settings["narration_chars_min"])
    chars_max = int(settings["narration_chars_max"])
    result = [dict(item) for item in items]
    for item in result:
        if item.get("narration") != AUTO_NARRATION_MARKER:
            continue
        picture = str(item.get("picture") or "剧情").strip()
        text = f"此时，{picture}。随着调查深入，更多线索浮出水面。"
        if len(text) < chars_min:
            text += "真相往往藏在细节之中。"
        if len(text) > chars_max:
            text = text[:chars_max]
        item["narration"] = text
    return result


def _truncate_chars(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", "", text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _ensure_bookend_narration_segments(
    items: List[Dict[str, Any]],
    source_duration_sec: float,
) -> List[Dict[str, Any]]:
    """保证按播放顺序的第一段与最后一段为 OST=0 解说。"""
    if not items:
        return items

    result = [dict(item) for item in items]
    source_duration_sec = max(source_duration_sec, 30.0)

    if result[0].get("OST") != 0:
        try:
            first_start, _ = parse_timestamp_range(result[0]["timestamp"])
        except (ValueError, AttributeError, KeyError):
            first_start = 1.0
        seg_end = max(8.0, min(first_start - 0.2, source_duration_sec * 0.08))
        seg_start = max(0.0, seg_end - 14.0)
        result.insert(
            0,
            {
                "timestamp": format_timestamp_range(seg_start, seg_end),
                "picture": "本集开场",
                "narration": AUTO_NARRATION_MARKER,
                "OST": 0,
            },
        )

    if result[-1].get("OST") != 0:
        try:
            _, last_end = parse_timestamp_range(result[-1]["timestamp"])
        except (ValueError, AttributeError, KeyError):
            last_end = source_duration_sec * 0.9
        seg_start = min(last_end + 0.2, source_duration_sec - 14.0)
        seg_end = max(seg_start + 8.0, source_duration_sec - 0.5)
        result.append(
            {
                "timestamp": format_timestamp_range(seg_start, seg_end),
                "picture": "本集收尾",
                "narration": AUTO_NARRATION_MARKER,
                "OST": 0,
            },
        )

    for idx, item in enumerate(result, 1):
        item["_id"] = idx
    return result


def _merge_opening_narration(
    opening_line: str,
    recap_text: str,
    body: str,
    *,
    max_chars: int,
) -> str:
    body = (body or "").strip()
    if body == AUTO_NARRATION_MARKER:
        body = ""
    if body.startswith(opening_line):
        merged = body
    elif recap_text and body:
        merged = f"{opening_line}{recap_text}{body}"
    elif recap_text:
        merged = f"{opening_line}{recap_text}"
    elif body:
        merged = f"{opening_line}{body}"
    else:
        merged = opening_line
    return _truncate_chars(merged, max_chars)


def _merge_closing_narration(body: str, closing_line: str, *, max_chars: int) -> str:
    body = (body or "").strip()
    if body == AUTO_NARRATION_MARKER:
        body = ""
    closing = closing_line.strip()
    if not closing:
        return body
    plain = re.sub(r"\s+", "", body)
    closing_plain = re.sub(r"\s+", "", closing)
    if plain.endswith(closing_plain):
        return _truncate_chars(body, max_chars)
    if closing_plain and closing_plain in plain:
        body = body.replace(closing_line, "").strip()
        body = body.replace(closing_plain, "").strip()
    if body:
        merged = f"{body}{closing}"
    else:
        merged = closing
    return _truncate_chars(merged, max_chars)


def apply_tv_series_bookends(
    items: List[Dict[str, Any]],
    *,
    film_name: str,
    settings: Optional[Dict[str, Any]] = None,
    subtitle_content: str = "",
    source_duration_sec: Optional[float] = None,
    prev_episode_recap: str = "",
) -> List[Dict[str, Any]]:
    """
    电视剧模式：首尾必须为解说，并注入可配置的开场/收尾话术；
    非首集可在开场语后接上集回顾。
    """
    cfg = get_film_tv_settings(settings)
    if cfg.get("content_type") != TV_CONTENT_SERIES:
        return items
    if not items:
        return items

    cues = parse_srt_cues(subtitle_content)
    if source_duration_sec is None or source_duration_sec <= 0:
        if cues:
            source_duration_sec = max(end for _, end in cues) + 1.0
        else:
            ends = []
            for item in items:
                try:
                    _, end = parse_timestamp_range(item["timestamp"])
                    ends.append(end)
                except (ValueError, AttributeError, KeyError):
                    pass
            source_duration_sec = (max(ends) + 1.0) if ends else 600.0

    episode = max(1, int(cfg.get("episode_number") or 1))
    opening_line = format_tv_line_template(
        cfg.get("tv_opening_line_template") or "",
        film_name,
        episode,
    )
    closing_line = format_tv_line_template(
        cfg.get("tv_closing_line_template") or "",
        film_name,
        episode,
    )
    opening_max = int(cfg.get("opening_chars_max") or 110)
    closing_max = int(cfg.get("narration_chars_max") or 78) + 30
    narrative_end_sec = _estimate_narrative_end_sec(items, source_duration_sec)

    ordered = finalize_film_tv_playback_order([dict(item) for item in items], cfg)
    ordered = _remove_late_filler_narrations(ordered, closing_line, narrative_end_sec)
    ordered = _ensure_bookend_narration_segments(ordered, source_duration_sec)
    ordered = finalize_film_tv_playback_order(ordered, cfg)

    ost0_indices = _find_ost0_indices_by_time(ordered)
    if not ost0_indices:
        return ordered

    first_idx = ost0_indices[0]
    last_idx = ost0_indices[-1]

    recap_text = ""
    if (
        episode > 1
        and cfg.get("tv_recap_prev_episode", True)
        and (prev_episode_recap or "").strip()
    ):
        recap_text = prev_episode_recap.strip()
        if recap_text and recap_text[-1] not in "。！？.!?":
            recap_text += "。"

    first = ordered[first_idx]
    first["picture"] = _strip_opening_from_picture(str(first.get("picture") or ""), opening_line)
    _clamp_ost0_timestamp(first)
    first["narration"] = _merge_opening_narration(
        opening_line,
        recap_text,
        str(first.get("narration") or ""),
        max_chars=opening_max,
    )

    for idx in ost0_indices[1:-1]:
        narr = str(ordered[idx].get("narration") or "")
        if closing_line and closing_line in narr:
            ordered[idx]["narration"] = narr.replace(closing_line, "").strip()

    last = ordered[last_idx]
    last["narration"] = _merge_closing_narration(
        str(last.get("narration") or ""),
        closing_line,
        max_chars=closing_max,
    )

    ordered = _remove_late_filler_narrations(ordered, closing_line, narrative_end_sec)
    ordered = finalize_film_tv_playback_order(ordered, cfg)

    final_ost0 = _find_ost0_indices_by_time(ordered)
    closing_id = ordered[final_ost0[-1]].get("_id") if final_ost0 else "?"
    logger.info(
        f"电视剧分集话术已应用：第 {episode} 集，"
        f"开场{'含上集回顾' if recap_text else '无回顾'}，"
        f"收尾段 _id={closing_id}"
    )
    return ordered


def optimize_film_tv_script(
    items: List[Dict[str, Any]],
    subtitle_content: str = "",
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    优化影视解说脚本，扩展过短的 OST=1 原声片段。

    Returns:
        优化后的脚本 items 列表（新副本）
    """
    if not items:
        return items

    settings = get_film_tv_settings(settings)
    min_duration = settings["ost1_duration_min"]
    max_duration = settings["ost1_duration_max"]
    target_orig_ratio = int(settings["original_audio_percent"]) / 100.0
    narration_heavy = int(settings["narration_percent"]) >= int(settings["original_audio_percent"])
    cues = parse_srt_cues(subtitle_content)

    if source_duration_sec is None:
        if cues:
            source_duration_sec = max(end for _, end in cues) + 1.0
        else:
            max_end = 0.0
            for item in items:
                try:
                    _, end = parse_timestamp_range(item["timestamp"])
                    max_end = max(max_end, end)
                except (ValueError, AttributeError):
                    pass
            source_duration_sec = max_end + 1.0

    optimized = [dict(item) for item in items]
    expanded_count = 0

    for idx, item in enumerate(optimized):
        if item.get("OST") != 1:
            continue

        try:
            start_sec, end_sec = parse_timestamp_range(item["timestamp"])
        except (ValueError, AttributeError):
            continue

        old_duration = end_sec - start_sec
        if old_duration >= min_duration:
            continue

        if narration_heavy:
            _, ost1_sec, total = estimate_duration_breakdown(optimized)
            if total > 0 and ost1_sec / total >= max(0.0, target_orig_ratio - 0.02):
                continue

        occupied = _occupied_ranges(optimized, exclude_index=idx)
        new_start, new_end = expand_ost1_segment(
            start_sec,
            end_sec,
            cues,
            occupied,
            min_duration=min_duration,
            max_duration=max_duration,
            source_duration=source_duration_sec,
        )
        new_duration = new_end - new_start
        if new_duration > old_duration + 0.1:
            item["timestamp"] = format_timestamp_range(new_start, new_end)
            expanded_count += 1
            logger.info(
                f"OST=1 片段 #{item.get('_id')} 时长 {old_duration:.1f}s → {new_duration:.1f}s"
            )

    before = estimate_output_duration(items)
    after = estimate_output_duration(optimized)
    target = source_duration_sec * settings["target_duration_percent"] / 100.0

    logger.info(
        f"影视脚本时长估算: {before:.0f}s → {after:.0f}s "
        f"(原片 {source_duration_sec:.0f}s, 目标约 {target:.0f}s, 扩展 {expanded_count} 段 OST=1)"
    )

    if after < target * 0.7:
        logger.warning(
            f"优化后成片时长仍偏短 ({after:.0f}s < 目标 {target:.0f}s)，"
            f"建议重新生成脚本或增加 OST=1 段数（目标 {settings['ost1_segment_min']}-{settings['ost1_segment_max']} 段）"
        )

    optimized = normalize_ost_types(optimized)
    optimized = finalize_film_tv_playback_order(optimized, settings)
    optimized = fix_film_tv_script_structure(
        optimized, subtitle_content, source_duration_sec, settings
    )
    optimized = rebalance_film_tv_duration_ratio(optimized, settings)

    validation = validate_film_tv_script(optimized, settings, subtitle_content)
    if not validation["ok"]:
        logger.warning(f"影视脚本未达标: {validation['message']}")

    return optimized
