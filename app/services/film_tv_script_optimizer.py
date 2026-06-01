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
from app.services.film_tv_settings import get_film_tv_settings
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
    total = 0.0
    for item in items:
        if item.get("OST") == 1:
            total += calculate_duration(item["timestamp"])
        else:
            total += estimate_narration_duration(item.get("narration", ""))
    return total


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
    result = [dict(item) for item in sorted(items, key=lambda x: x.get("_id", 0))]
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
    if not srt_entries:
        return "剧情过渡"
    clipped = extract_entries_in_range(
        srt_entries,
        int(round(start_sec * 1000)),
        int(round(end_sec * 1000)),
    )
    if not clipped:
        return "剧情过渡"
    text = clipped[0].text.strip().replace("\n", " ")
    if len(text) > 24:
        return text[:24]
    return text or "剧情过渡"


def _renumber_items_by_time(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]) -> Tuple[float, int]:
        try:
            start, _ = parse_timestamp_range(item["timestamp"])
            return start, int(item.get("_id", 0) or 0)
        except (ValueError, AttributeError, KeyError):
            return 0.0, int(item.get("_id", 0) or 0)

    ordered = sorted(items, key=sort_key)
    for idx, item in enumerate(ordered, 1):
        item["_id"] = idx
    return ordered


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
    if validation["ok"]:
        return result, validation

    cues = parse_srt_cues(subtitle_content)
    srt_entries = parse_srt(subtitle_content) if subtitle_content else []

    if source_duration_sec is None or source_duration_sec <= 0:
        if cues:
            source_duration_sec = max(end for _, end in cues) + 1.0
        else:
            ends = [end for _, end in _collect_item_ranges(result)]
            source_duration_sec = (max(ends) + 1.0) if ends else 600.0

    ost1_min = int(settings["ost1_duration_min"])
    ost1_max = int(settings["ost1_duration_max"])
    next_id = max((int(i.get("_id", 0) or 0) for i in result), default=0) + 1

    ost1_need = max(0, validation["ost1_min"] - validation["ost1_count"])
    if ost1_need > 0 and cues:
        occupied = _merge_time_ranges(_collect_item_ranges(result))
        gaps = _find_timeline_gaps(occupied, source_duration_sec, min_gap=float(ost1_min))
        added = 0
        for gap_start, gap_end in gaps:
            if added >= ost1_need:
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

    validation = validate_film_tv_script_counts(result, settings)
    ost0_need = max(0, validation["ost0_min"] - validation["ost0_count"])
    if ost0_need > 0:
        occupied = _merge_time_ranges(_collect_item_ranges(result))
        gaps = _find_timeline_gaps(occupied, source_duration_sec, min_gap=10.0)
        added = 0
        for gap_start, gap_end in gaps:
            if added >= ost0_need:
                break
            available = gap_end - gap_start
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

    validation = validate_film_tv_script_counts(result, settings)
    ost0_need = max(0, validation["ost0_min"] - validation["ost0_count"])
    if ost0_need > 0:
        occupied = _merge_time_ranges(_collect_item_ranges(result))
        ost0_min_target = int(settings["ost0_segment_min"])
        bin_size = source_duration_sec / max(ost0_min_target, 1)
        ost0_starts = {
            parse_timestamp_range(item["timestamp"])[0]
            for item in result
            if item.get("OST") == 0
        }
        added = 0
        for bin_idx in range(ost0_min_target):
            if ost0_need <= 0:
                break
            bin_start = bin_idx * bin_size
            bin_end = min(source_duration_sec, (bin_idx + 1) * bin_size)
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
            occupied = _merge_time_ranges(_collect_item_ranges(result))
        if added:
            logger.info(f"自动补入 {added} 段 OST=0 解说（均匀分桶，待填充文案）")

    result = _renumber_items_by_time(result)
    validation = validate_film_tv_script_counts(result, settings)
    if not validation["ok"]:
        logger.warning(f"自动补段后仍未达标: {validation['message']}")
    else:
        logger.info("自动补段后段数已达标")
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
    if settings.get("enforce_narration_after_ost1", True):
        optimized = enforce_narration_after_ost1(optimized)

    validation = validate_film_tv_script_counts(optimized, settings)
    if not validation["ok"]:
        logger.warning(f"影视脚本段数未达标: {validation['message']}")

    return optimized
