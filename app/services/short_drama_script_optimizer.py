#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧解说脚本后处理：subtitle_entries 对位、OST 比例、原声段 picture 旁白格式。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.documentary.documentary_script_optimizer import (
    _align_items_to_frame_time_ranges,
    _strip_internal_clip_flags,
)
from app.services.short_drama_settings import (
    format_ost1_max_segments_rule,
    get_short_drama_settings,
    resolve_ost1_max_segments,
    summarize_short_drama_playback,
)
from app.services.documentary.documentary_script_optimizer import (
    _align_fazu2_ost0_to_adjacent_ost1,
    _enforce_narration_after_ost1_by_id,
)
from app.services.short_drama_blueprint_script import parse_scene_segment_blueprint
from app.services.short_drama_timestamp_alignment import (
    align_script_items_to_source_material,
    collect_content_timestamp_issues,
    detect_picture_line_incoherence,
)
from app.services.srt_utils import (
    find_subtitle_span_for_line,
    find_subtitle_span_global,
    format_timestamp_ms,
    is_valid_script_timestamp_range,
    parse_srt,
    parse_timestamp_range,
    repair_or_drop_invalid_timestamp_items,
    script_timestamp_duration_ms,
)


AUTO_NARRATION_MARKER = "__AUTO_NARRATION__"

_TRANSITION_POOL = (
    "随后",
    "另一边",
    "与此同时",
    "更让人揪心的是",
    "紧接着",
    "镜头一转",
    "谁也没想到",
    "偏偏在这时",
    "值得注意的是",
    "接下来",
)

_ERSHI_RE = re.compile(r"而这时[，,]?")
_SHOT_PREFIX_RE = re.compile(
    r"^(?:特写|中景|全景|航拍|低机位|高机位|快速剪辑)[：:]\s*"
)


def _normalize_picture_for_compare(text: str) -> str:
    value = strip_picture_quotes(str(text or "")).strip()
    value = _SHOT_PREFIX_RE.sub("", value)
    return re.sub(r"\s+", "", value)


def is_picture_echo_narration(narration: str, picture: str) -> bool:
    """narration 是否只是在复述 picture（如「随后，特写：…」）。"""
    narr = str(narration or "").strip().rstrip("。！？")
    pic = strip_picture_quotes(str(picture or "")).strip().rstrip("。！？")
    if not narr or not pic:
        return False

    narr_norm = _normalize_picture_for_compare(narr)
    pic_norm = _normalize_picture_for_compare(pic)
    if narr_norm and pic_norm and narr_norm == pic_norm:
        return True

    for prefix in _TRANSITION_POOL:
        marker = f"{prefix}，"
        if not narr.startswith(marker):
            continue
        body = narr[len(marker) :].strip()
        body_norm = _normalize_picture_for_compare(body)
        if body_norm and pic_norm:
            if body_norm == pic_norm:
                return True
            if len(body_norm) <= len(pic_norm) + 2 and pic_norm in body_norm:
                return True
    return False


def remove_picture_echo_narrations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将「随后，特写：…」等复述 picture 的空洞 narration 改为概括句。"""
    fixed = 0
    alt_idx = 0
    for item in items:
        if int(item.get("OST", 0) or 0) != 0:
            continue
        narration = str(item.get("narration") or "").strip()
        picture = str(item.get("picture") or "").strip()
        if not is_picture_echo_narration(narration, picture):
            continue
        scene = strip_picture_quotes(picture).rstrip("。！？")
        line = str(item.get("original_line") or "").strip().strip("「」")
        core = _SHOT_PREFIX_RE.sub("", scene).strip() or scene
        prefix = _TRANSITION_POOL[alt_idx % len(_TRANSITION_POOL)]
        alt_idx += 1
        if line:
            item["narration"] = f"{prefix}，{core}。原话大意：{line}。"
        else:
            item["narration"] = (
                f"{prefix}，{core}背后的压力正在升级，冲突进入新的节点。"
            )
        fixed += 1
        logger.info(f"片段 #{item.get('_id')} 已去除 picture 复述型 narration")
    if fixed:
        logger.info(f"短剧解说：已修复 {fixed} 段 picture 复述型 narration")
    return items


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


def validate_short_drama_script_duration(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验成片总时长 8–13 分钟、解说/原声时长比约 3:7。"""
    cfg = get_short_drama_settings(settings)
    summary = summarize_short_drama_playback(items)
    total_sec = float(summary.get("total_sec") or 0)
    min_sec = int(cfg.get("target_output_minutes_min", 8) or 8) * 60
    max_sec = int(cfg.get("target_output_minutes_max", 13) or 13) * 60
    narr_target = float(cfg.get("narration_percent", 30) or 30) / 100.0
    tolerance = float(cfg.get("narration_ratio_tolerance", 0.10) or 0.10)

    issues: List[str] = []
    if total_sec <= 0:
        issues.append("无法估算成片总时长（各段 timestamp/narration 可能无效）")
    elif total_sec < min_sec:
        issues.append(
            f"成片总时长约 {total_sec / 60:.1f} 分钟，低于目标 {min_sec / 60:.0f} 分钟"
        )
    elif total_sec > max_sec:
        issues.append(
            f"成片总时长约 {total_sec / 60:.1f} 分钟，超过目标 {max_sec / 60:.0f} 分钟"
        )

    if total_sec > 0:
        narr_ratio = float(summary.get("narration_sec") or 0) / total_sec
        low = narr_target - tolerance
        high = narr_target + tolerance
        if narr_ratio < low:
            issues.append(
                f"解说时长占比约 {summary.get('narration_pct')}%，"
                f"低于目标约 {int(narr_target * 100)}%（容差 ±{int(tolerance * 100)}%）"
            )
        elif narr_ratio > high:
            issues.append(
                f"解说时长占比约 {summary.get('narration_pct')}%，"
                f"高于目标约 {int(narr_target * 100)}%（容差 ±{int(tolerance * 100)}%）"
            )

    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "message": "成片时长与占比符合要求" if ok else "；".join(issues),
        **summary,
    }


def format_short_drama_duration_retry_hint(
    validation: Dict[str, Any],
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = get_short_drama_settings(settings)
    min_min = int(cfg.get("target_output_minutes_min", 8) or 8)
    max_min = int(cfg.get("target_output_minutes_max", 13) or 13)
    narr_pct = int(cfg.get("narration_percent", 30) or 30)
    orig_pct = int(cfg.get("original_audio_percent", 70) or 70)
    lines = [
        "【成片时长修正】上次 JSON 时长/占比未达标，输出无效：",
        validation.get("message") or "成片时长或解说/原声占比不符",
        "必须遵守：",
        f"- 按 `_id` 播放顺序累加，**成片总时长 {min_min}–{max_min} 分钟**",
        f"- **解说 vs 原声成片时长约 {narr_pct}:{orig_pct}**（解说为主）",
        f"- OST=1 {format_ost1_max_segments_rule(cfg)}，每段 ≤{cfg.get('ost1_duration_max', 5)} 秒",
        f"- OST=0 每段 ≥{cfg.get('narration_chars_min', 40)} 字；禁止长原声",
    ]
    return "\n".join(lines)


def validate_short_drama_script_counts(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """段数统计（仅供 UI 展示，不再作为生成硬性门槛）。"""
    cfg = get_short_drama_settings(settings)
    ost1_count, ost0_count, total = count_short_drama_segments(items)
    summary = summarize_short_drama_playback(items)
    return {
        "ok": True,
        "ost1_count": ost1_count,
        "ost0_count": ost0_count,
        "total": total,
        "message": (
            f"共 {total} 段；成片约 {summary.get('total_sec', 0) / 60:.1f} 分钟，"
            f"解说 {summary.get('narration_pct', 0)}% / 原声 {summary.get('original_pct', 0)}%"
        ),
        **summary,
    }


def score_short_drama_script_quality(
    dur_validation: Dict[str, Any],
    ts_validation: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> float:
    """估算脚本与目标的接近程度，分数越低越接近达标。"""
    cfg = get_short_drama_settings(settings)
    min_sec = int(cfg.get("target_output_minutes_min", 8) or 8) * 60
    max_sec = int(cfg.get("target_output_minutes_max", 13) or 13) * 60
    target_mid = (min_sec + max_sec) / 2.0
    narr_target = float(cfg.get("narration_percent", 30) or 30) / 100.0

    total_sec = float(dur_validation.get("total_sec") or 0)
    score = 0.0

    if total_sec <= 0:
        score += 100.0
    elif total_sec < min_sec:
        score += (min_sec - total_sec) / 60.0 * 3.0
    elif total_sec > max_sec:
        score += (total_sec - max_sec) / 60.0 * 3.0
    else:
        score += abs(total_sec - target_mid) / 60.0 * 0.3

    if total_sec > 0:
        narr_ratio = float(dur_validation.get("narration_sec") or 0) / total_sec
        score += abs(narr_ratio - narr_target) * 15.0

    for issue in ts_validation.get("issues") or []:
        text = str(issue)
        if "零时长" in text or "起止相同" in text:
            score += 8.0
        elif "缺少有效 timestamp" in text:
            score += 6.0
        else:
            score += 1.0

    for issue in dur_validation.get("issues") or []:
        score += 1.5

    if dur_validation.get("ok") and ts_validation.get("ok"):
        score -= 1.0

    return round(score, 3)


def pick_best_short_drama_script_candidate(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从多次生成结果中选取最接近达标的一条。"""
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("score") or 0))


def validate_short_drama_script_timestamps(
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    *,
    subtitle_content: str = "",
    plot_blueprint: str = "",
) -> Dict[str, Any]:
    """校验 LLM 输出：禁止零时长 timestamp，OST=1 须满足最小时长。"""
    cfg = get_short_drama_settings(settings)
    ost1_min_ms = int(float(cfg.get("ost1_duration_min", 2) or 2) * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)
    narr_chars_min = int(cfg.get("narration_chars_min", 40) or 40)
    issues: List[str] = []

    for item in items:
        item_id = item.get("_id", "?")
        ts = str(item.get("timestamp") or "").strip()
        ost = int(item.get("OST", 0) or 0)
        min_ms = ost1_min_ms if ost == 1 else ost0_min_ms

        if not ts or "-" not in ts:
            issues.append(f"片段 #{item_id} 缺少有效 timestamp")
            continue

        duration_ms = script_timestamp_duration_ms(ts)
        if duration_ms <= 0:
            issues.append(
                f"片段 #{item_id} timestamp 起止相同或无效（零时长不可裁剪）: {ts}"
            )
            continue

        if ost == 1 and not is_valid_script_timestamp_range(ts, min_duration_ms=min_ms):
            issues.append(
                f"片段 #{item_id} OST=1 时长 "
                f"({duration_ms / 1000.0:.2f}s，允许 {min_ms / 1000.0:.1f}–"
                f"{int(float(cfg.get('ost1_duration_max', 5) or 5) * 1000) / 1000.0:.1f}s): {ts}"
            )
        ost1_max_ms = int(float(cfg.get("ost1_duration_max", 5) or 5) * 1000)
        if ost == 1 and duration_ms > ost1_max_ms + 200:
            issues.append(
                f"片段 #{item_id} OST=1 超过 {ost1_max_ms / 1000.0:.1f}s 上限: {ts}"
            )

        if ost == 0:
            narration = str(item.get("narration") or "")
            char_count = len(re.sub(r"\s+", "", narration))
            if char_count < narr_chars_min:
                issues.append(
                    f"片段 #{item_id} 解说过短（{char_count} 字 < {narr_chars_min} 字），"
                    "须完整表达一条脉络"
                )
            est_sec = _estimate_narration_duration_sec(narration)
            if est_sec < ost0_min_ms / 1000.0:
                issues.append(
                    f"片段 #{item_id} 解说估算时长过短 "
                    f"({est_sec:.1f}s < {ost0_min_ms / 1000.0:.1f}s)"
                )

    overlap_issues = _find_timestamp_overlap_issues(items)
    issues.extend(overlap_issues)
    issues.extend(
        collect_content_timestamp_issues(
            items,
            subtitle_content=subtitle_content,
            plot_blueprint=plot_blueprint,
            settings=cfg,
        )
    )

    ershi_count = sum(
        len(_ERSHI_RE.findall(str(item.get("narration") or "")))
        for item in items
        if int(item.get("OST", 0) or 0) == 0
    )
    max_ershi = int(cfg.get("max_ershi_per_script", 2) or 2)
    if ershi_count > max_ershi:
        issues.append(
            f"「而这时」出现 {ershi_count} 次，超过上限 {max_ershi} 次，请换用多样转折词"
        )

    ost1_count = sum(1 for item in items if int(item.get("OST", 0) or 0) == 1)
    ost1_max_segments = resolve_ost1_max_segments(cfg)
    if ost1_max_segments > 0 and ost1_count > ost1_max_segments:
        issues.append(
            f"OST=1 共 {ost1_count} 段，超过上限 {ost1_max_segments} 段（应解说为主）"
        )

    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "message": "时间戳有效" if ok else "；".join(issues[:6])
            + (f" 等共 {len(issues)} 处" if len(issues) > 6 else ""),
    }


def _estimate_narration_duration_sec(text: str) -> float:
    chars = len(re.sub(r"\s+", "", text or ""))
    return max(3.0, chars * 0.35)


def format_short_drama_timestamp_retry_hint(
    validation: Dict[str, Any],
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 LLM 重试用的时间戳修正说明。"""
    cfg = get_short_drama_settings(settings)
    ost1_min = int(cfg.get("ost1_duration_min", 8) or 8)
    ost1_max = int(cfg.get("ost1_duration_max", 18) or 18)
    ost0_min = int(cfg.get("ost0_duration_min", 5) or 5)
    narr_min = int(cfg.get("narration_chars_min", 20) or 20)
    lines = [
        "【timestamp 硬性修正】上次 JSON 存在无效时间轴，输出无效：",
        validation.get("message") or "存在零时长或过短 timestamp",
        "必须遵守：",
        f"- 每条 timestamp 格式 HH:MM:SS,mmm-HH:MM:SS,mmm，**结束时间必须严格大于开始时间**（禁止零时长）",
        f"- OST=1 {format_ost1_max_segments_rule(cfg)}，每段 ≤{cfg.get('ost1_duration_max', 5)} 秒",
        f"- OST=0 每段 ≥{cfg.get('narration_chars_min', 40)} 字",
        "- 禁止 timestamp 重叠；正叙段尽量首尾相接",
        "- OST=1 的 original_line 须在 SRT 中可定位；picture 与台词须同场景",
    ]
    return "\n".join(lines)


def repair_short_drama_script_timestamps(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    plot_blueprint: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """仅修复时间轴：素材对位 + subtitle_entries + 扩展，不改 narration 主干文案。"""
    if not items:
        return items

    cfg = settings or get_short_drama_settings()
    ost1_min_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)
    ost1_max_ms = int(float(cfg.get("ost1_duration_max", 18) or 18) * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)

    if (subtitle_content or "").strip() or (plot_blueprint or "").strip():
        items = align_script_items_to_source_material(
            items,
            subtitle_content=subtitle_content,
            plot_blueprint=plot_blueprint,
            settings=cfg,
        )

    if (frame_analysis_path or "").strip():
        items = align_short_drama_items_to_frame_time_ranges(
            items,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=float(cfg.get("ost1_duration_max", 18) or 18),
        )

    items = repair_or_drop_invalid_timestamp_items(
        items,
        subtitle_content=subtitle_content,
        min_duration_ms=max(500, ost0_min_ms),
        ost1_min_duration_ms=ost1_min_ms,
        ost1_max_duration_ms=ost1_max_ms,
        drop_unrecoverable=False,
    )
    items = cap_ost1_segment_durations(
        items, max_sec=float(cfg.get("ost1_duration_max", 5) or 5)
    )
    items = resolve_source_timestamp_overlaps(items)
    items = diversify_ershi_transitions(
        items, max_ershi=int(cfg.get("max_ershi_per_script", 2) or 2)
    )
    return items


def _playback_sort_key(item: Dict[str, Any]) -> int:
    return int(item.get("_id") or 0)


def _segment_duration_ms(timestamp: str) -> int:
    try:
        start_ms, end_ms = parse_timestamp_range(timestamp)
        return max(0, end_ms - start_ms)
    except Exception:
        return 0


def _find_timestamp_overlap_issues(items: List[Dict[str, Any]]) -> List[str]:
    """检测原片时间轴上的 timestamp 重叠。"""
    ranges: list[tuple[int, int, int]] = []
    for item in items:
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(ts)
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        ranges.append((int(item.get("_id") or 0), start_ms, end_ms))

    issues: List[str] = []
    ranges.sort(key=lambda row: (row[1], row[0]))
    for index in range(len(ranges) - 1):
        id_a, start_a, end_a = ranges[index]
        id_b, start_b, end_b = ranges[index + 1]
        if start_b < end_a:
            issues.append(
                f"片段 #{id_a} 与 #{id_b} 在原片时间轴重叠"
                f"（{format_timestamp_ms(start_b)} < {format_timestamp_ms(end_a)}）"
            )
    return issues


def diversify_ershi_transitions(
    items: List[Dict[str, Any]],
    *,
    max_ershi: int = 2,
) -> List[Dict[str, Any]]:
    """将超出上限的「而这时」替换为多样转折词。"""
    if max_ershi < 0:
        return items
    seen = 0
    alt_idx = 0
    for item in items:
        if int(item.get("OST", 0) or 0) != 0:
            continue
        narration = str(item.get("narration") or "")
        if not _ERSHI_RE.search(narration):
            continue
        seen += 1
        if seen <= max_ershi:
            continue
        replacement = _TRANSITION_POOL[alt_idx % len(_TRANSITION_POOL)]
        alt_idx += 1
        item["narration"] = _ERSHI_RE.sub(f"{replacement}，", narration, count=1)
        logger.info(f"片段 #{item.get('_id')} 已将「而这时」替换为「{replacement}」")
    return items


def cap_ost1_segment_durations(
    items: List[Dict[str, Any]],
    *,
    max_sec: float = 5.0,
) -> List[Dict[str, Any]]:
    """原声段时长硬上限（默认 5 秒）。"""
    max_ms = max(1000, int(float(max_sec) * 1000))
    capped = 0
    for item in items:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(ts)
        except Exception:
            continue
        if end_ms - start_ms <= max_ms:
            continue
        item["timestamp"] = (
            f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(start_ms + max_ms)}"
        )
        capped += 1
    if capped:
        logger.info(f"短剧解说：已截断 {capped} 段 OST=1 至 ≤{max_sec}s")
    return items


def _convert_ost1_item_to_narration(
    item: Dict[str, Any],
    *,
    alt_idx: int = 0,
) -> Dict[str, Any]:
    scene = strip_picture_quotes(str(item.get("picture") or ""))
    line = str(item.get("original_line") or "").strip().strip("「」")
    prefix = _TRANSITION_POOL[alt_idx % len(_TRANSITION_POOL)]
    if line and scene:
        summary = f"{prefix}，{scene.rstrip('。！？')}。原话大意：{line}。"
    elif scene:
        summary = f"{prefix}，{scene.rstrip('。！？')}。"
    elif line:
        summary = f"{prefix}，原片台词大意：{line}。"
    else:
        summary = AUTO_NARRATION_MARKER
    converted = dict(item)
    converted["OST"] = 0
    converted["narration"] = summary
    converted.pop("original_line", None)
    return converted


def enforce_opening_head_ost1_limit(
    items: List[Dict[str, Any]],
    *,
    head_count: int = 3,
    max_ost1: int = 1,
    transition_index: int = 0,
) -> List[Dict[str, Any]]:
    """播放顺序开头若干段内仅保留一段 OST=1（避免开篇连放两段原声）。"""
    if max_ost1 <= 0 or head_count <= 0:
        return items
    ordered = sorted([dict(item) for item in items], key=_playback_sort_key)
    head_indices = [
        idx
        for idx, item in enumerate(ordered)
        if int(item.get("_id") or 0) <= head_count
        and int(item.get("OST", 0) or 0) == 1
        and not item.get("_opening_climax_replay")
    ]
    if len(head_indices) <= max_ost1:
        return ordered

    head_indices.sort(key=lambda idx: (int(ordered[idx].get("_id") or 0), idx))
    drop_indices = set(head_indices[max_ost1:])
    alt_idx = transition_index
    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(ordered):
        if idx not in drop_indices:
            result.append(item)
            continue
        result.append(_convert_ost1_item_to_narration(item, alt_idx=alt_idx))
        alt_idx += 1
        logger.info(
            f"片段 #{item.get('_id')} 开篇区原声重复，已转为 OST=0 解说概括"
        )

    for index, item in enumerate(result, 1):
        item["_id"] = index
    return result


def enforce_scene_ost1_after_narration(
    items: List[Dict[str, Any]],
    *,
    transition_index: int = 0,
) -> List[Dict[str, Any]]:
    """除开篇 #1 外，场景爆燃 OST=1 须紧跟在 OST=0 解说之后。"""
    ordered = sorted([dict(item) for item in items], key=_playback_sort_key)
    alt_idx = transition_index
    converted = 0
    for index, item in enumerate(ordered):
        if int(item.get("OST", 0) or 0) != 1:
            continue
        if int(item.get("_id") or 0) == 1 or item.get("_opening_climax_replay"):
            continue
        if index == 0:
            continue
        prev = ordered[index - 1]
        if int(prev.get("OST", 0) or 0) == 0:
            continue
        ordered[index] = _convert_ost1_item_to_narration(item, alt_idx=alt_idx)
        alt_idx += 1
        converted += 1
        logger.info(
            f"片段 #{item.get('_id')} 原声前缺少解说铺垫，已转为 OST=0 概括"
        )
    if converted:
        logger.info(f"短剧解说：已修正 {converted} 段未配对的原声为解说")
    return ordered


def convert_excess_ost1_to_narration(
    items: List[Dict[str, Any]],
    *,
    ost1_max: int = 10,
    transition_index: int = 0,
) -> List[Dict[str, Any]]:
    """超出原声段数上限时，将多余 OST=1 转为解说概括。"""
    if ost1_max <= 0:
        return items
    ordered = sorted([dict(item) for item in items], key=_playback_sort_key)
    ost1_indices = [
        idx
        for idx, item in enumerate(ordered)
        if int(item.get("OST", 0) or 0) == 1
    ]
    if len(ost1_indices) <= ost1_max:
        return ordered

    # 优先保留播放顺序靠前的 OST=1（开篇爆燃 _id=1）
    ost1_indices.sort(
        key=lambda idx: (
            int(ordered[idx].get("_id") or 0),
            idx,
        )
    )
    drop_indices = set(ost1_indices[ost1_max:])
    alt_idx = transition_index
    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(ordered):
        if idx not in drop_indices:
            result.append(item)
            continue
        result.append(_convert_ost1_item_to_narration(item, alt_idx=alt_idx))
        alt_idx += 1
        logger.info(f"片段 #{item.get('_id')} 原声超限，已转为 OST=0 解说概括")

    for index, item in enumerate(result, 1):
        item["_id"] = index
    return result


def resolve_source_timestamp_overlaps(
    items: List[Dict[str, Any]],
    *,
    gap_ms: int = 50,
    min_duration_ms: int = 500,
) -> List[Dict[str, Any]]:
    """消除原片时间轴上的 timestamp 重叠（按 _id 优先级保留前段）。"""
    ordered = sorted([dict(item) for item in items], key=_playback_sort_key)
    occupied_end = -1
    fixed = 0

    for item in ordered:
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(ts)
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        if occupied_end >= 0 and start_ms < occupied_end + gap_ms:
            shift = occupied_end + gap_ms - start_ms
            start_ms += shift
            end_ms += shift
            if end_ms - start_ms < min_duration_ms:
                end_ms = start_ms + min_duration_ms
            item["timestamp"] = (
                f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(end_ms)}"
            )
            fixed += 1
        try:
            _, end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
            occupied_end = max(occupied_end, end_ms)
        except Exception:
            pass

    if fixed:
        logger.info(f"短剧解说：已消除 {fixed} 处 timestamp 重叠")
    return ordered


def _convert_ost1_to_ost0_bridge(
    item: Dict[str, Any],
    *,
    transition_index: int = 0,
) -> Dict[str, Any]:
    row = dict(item)
    scene = strip_picture_quotes(str(row.get("picture") or ""))
    row["OST"] = 0
    prefix = _TRANSITION_POOL[transition_index % len(_TRANSITION_POOL)]
    if scene:
        row["narration"] = f"{prefix}，{scene.rstrip('。！？')}。"
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
    alt_idx = 0
    for item in items:
        row = dict(item)
        item_id = int(row.get("_id") or 0)
        if int(row.get("OST", 0) or 0) == 1:
            run += 1
            if run > max_run and item_id not in protected:
                converted = _convert_ost1_to_ost0_bridge(row, transition_index=alt_idx)
                alt_idx += 1
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
    """后处理：打断过长连续原声，并记录成片时长占比。"""
    if not items:
        return items

    cfg = get_short_drama_settings(settings)
    max_run = int(cfg.get("max_consecutive_ost1", 4) or 4)
    result = break_consecutive_ost1(items, max_run=max_run, protect_ids={1})

    validation = validate_short_drama_script_duration(result, cfg)
    if not validation["ok"]:
        logger.warning(f"短剧解说成片时长/占比未达标: {validation['message']}")
    else:
        logger.info(
            f"短剧解说成片 OK：总时长 {validation['total_sec'] / 60:.1f} 分钟，"
            f"解说 {validation['narration_pct']}% / 原声 {validation['original_pct']}%"
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
    """将 timestamp 对齐到抽帧 JSON 的 subtitle_entries（字幕对位剪辑范围）。"""
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


def trim_picture_narration_text(text: str, max_chars: int) -> str:
    """旁白烧录字数上限（标点计入）。"""
    value = strip_picture_quotes(str(text or "").strip())
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    trimmed = value[:max_chars].rstrip("，。！？、；： ")
    return trimmed or value[:max_chars]


def realign_ost1_timestamps_to_subtitles(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """OST=1：按 SRT 合并对白块，确保高燃原声不被半句截断。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return items

    cfg = get_short_drama_settings(settings)
    ost1_min_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)
    ost1_max_ms = int(float(cfg.get("ost1_duration_max", 18) or 18) * 1000)
    adjusted = 0

    for item in items:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(ts)
        except Exception:
            continue

        line = str(item.get("original_line") or item.get("narration") or "")
        line = re.sub(r"^播放原片\d*$", "", line).strip().strip("「」")

        span = None
        if line:
            span = find_subtitle_span_global(
                entries,
                line,
                max_span_ms=ost1_max_ms,
            )
        if not span:
            span = find_subtitle_span_for_line(
                entries,
                line,
                near_start_ms=start_ms,
                near_end_ms=max(end_ms, start_ms + 500),
                max_span_ms=ost1_max_ms,
            )
        if not span:
            span = find_subtitle_span_for_line(
                entries,
                "",
                near_start_ms=start_ms,
                near_end_ms=max(end_ms, start_ms + 500),
                max_span_ms=ost1_max_ms,
            )
        if not span or span[1] <= span[0]:
            continue

        cue_start, cue_end = span
        if cue_end - cue_start > ost1_max_ms:
            cue_end = cue_start + ost1_max_ms
        if cue_end - cue_start < min(ost1_min_ms, ost1_max_ms):
            cue_end = min(cue_start + ost1_max_ms, entries[-1].end_ms)

        new_ts = f"{format_timestamp_ms(cue_start)}-{format_timestamp_ms(cue_end)}"
        if new_ts != ts and is_valid_script_timestamp_range(
            new_ts, min_duration_ms=min(500, ost1_min_ms)
        ):
            item["timestamp"] = new_ts
            adjusted += 1

    if adjusted:
        logger.info(f"短剧解说：已按 SRT 对位/扩展 {adjusted} 段 OST=1 原声裁切范围")
    return items


def align_ost0_timestamps_for_narration_and_lead_in(
    items: List[Dict[str, Any]],
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """OST=0：timestamp 跨度匹配解说 TTS，并在引出下一段原声前留出铺垫画面。"""
    if not items:
        return items

    cfg = get_short_drama_settings(settings)
    lead_ms = max(
        1000,
        int(float(cfg.get("ost0_lead_before_ost1_sec", 8) or 8) * 1000),
    )
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)
    ordered = sorted(items, key=_playback_sort_key)
    adjusted = 0

    for index, item in enumerate(ordered):
        if int(item.get("OST", 0) or 0) != 0:
            continue
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(ts)
        except Exception:
            continue

        narration = str(item.get("narration") or "").strip()
        min_dur_ms = max(
            ost0_min_ms,
            int(_estimate_narration_duration_sec(narration) * 1000),
        )

        next_item = ordered[index + 1] if index + 1 < len(ordered) else None
        if next_item is not None and int(next_item.get("OST", 0) or 0) == 1:
            try:
                next_start_ms, _ = parse_timestamp_range(
                    str(next_item.get("timestamp") or "")
                )
                target_start = max(0, next_start_ms - lead_ms)
                if target_start < start_ms:
                    start_ms = target_start
            except Exception:
                pass

        target_end = max(end_ms, start_ms + min_dur_ms)
        new_ts = f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(target_end)}"
        if new_ts != ts:
            item["timestamp"] = new_ts
            adjusted += 1

    if adjusted:
        logger.info(f"短剧解说：已调整 {adjusted} 段 OST=0 取画以贴合解说/铺垫原声")
    return ordered


def fill_ost1_picture_from_blueprint(
    items: List[Dict[str, Any]],
    *,
    plot_blueprint: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """OST=1 picture 为空时，从蓝图场景「画面要点」补全旁白字幕素材。"""
    scene_info = parse_scene_segment_blueprint(plot_blueprint)
    scenes = scene_info.get("scenes") or []
    if not scenes:
        return items

    cfg = get_short_drama_settings(settings)
    max_chars = int(cfg.get("picture_narration_max_chars", 16) or 16)
    filled = 0

    for item in items:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        picture = strip_picture_quotes(str(item.get("picture") or ""))
        if picture:
            continue
        ts = str(item.get("timestamp") or "").strip()
        if "-" not in ts:
            continue
        try:
            start_ms, _ = parse_timestamp_range(ts)
        except Exception:
            continue

        best_hint = ""
        best_distance = float("inf")
        for scene in scenes:
            hint = str(scene.get("picture_hint") or "").strip()
            if not hint:
                continue
            for start_text, end_text in scene.get("timestamp_ranges") or []:
                try:
                    seg_start, seg_end = parse_timestamp_range(
                        f"{start_text}-{end_text}"
                    )
                except Exception:
                    continue
                if seg_start <= start_ms <= seg_end:
                    best_hint = hint
                    best_distance = 0.0
                    break
                distance = min(abs(start_ms - seg_start), abs(start_ms - seg_end))
                if distance < best_distance:
                    best_distance = distance
                    best_hint = hint
            if best_distance == 0.0:
                break

        if best_hint:
            item["picture"] = trim_picture_narration_text(best_hint, max_chars)
            filled += 1

    if filled:
        logger.info(f"短剧解说：已从蓝图补全 {filled} 段 OST=1 picture 旁白")
    return items


def trim_ost1_picture_narrations(
    items: List[Dict[str, Any]],
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    cfg = get_short_drama_settings(settings)
    max_chars = int(cfg.get("picture_narration_max_chars", 16) or 16)
    for item in items:
        if int(item.get("OST", 0) or 0) != 1:
            continue
        picture = strip_picture_quotes(str(item.get("picture") or ""))
        if picture:
            item["picture"] = trim_picture_narration_text(picture, max_chars)
    return items


def optimize_short_drama_script_items(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    plot_blueprint: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """短剧解说脚本生成后的统一后处理。"""
    if not items:
        return items

    cfg = settings or get_short_drama_settings()
    ost1_max = float(cfg.get("ost1_duration_max", 5) or 5)
    min_ost1_ms = int(float(cfg.get("ost1_duration_min", 2) or 2) * 1000)
    max_ost1_ms = int(ost1_max * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)
    ost1_max_segments = resolve_ost1_max_segments(cfg)
    opening_head_max = int(cfg.get("opening_head_max_ost1", 1) or 1)
    opening_head_count = int(cfg.get("opening_head_segment_count", 3) or 3)
    max_ershi = int(cfg.get("max_ershi_per_script", 2) or 2)

    items = _enforce_narration_after_ost1_by_id(items)
    items = enforce_opening_head_ost1_limit(
        items,
        head_count=opening_head_count,
        max_ost1=opening_head_max,
    )

    if (subtitle_content or "").strip() or (plot_blueprint or "").strip():
        items = align_script_items_to_source_material(
            items,
            subtitle_content=subtitle_content,
            plot_blueprint=plot_blueprint,
            settings=cfg,
        )

    if (subtitle_content or "").strip():
        items = realign_ost1_timestamps_to_subtitles(
            items,
            subtitle_content=subtitle_content,
            settings=cfg,
        )

    items = cap_ost1_segment_durations(items, max_sec=ost1_max)
    items = enforce_scene_ost1_after_narration(items)
    items = convert_excess_ost1_to_narration(items, ost1_max=ost1_max_segments)
    items = resolve_source_timestamp_overlaps(items)
    items = remove_picture_echo_narrations(items)
    items = diversify_ershi_transitions(items, max_ershi=max_ershi)

    if (frame_analysis_path or "").strip():
        items = align_short_drama_items_to_frame_time_ranges(
            items,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=ost1_max,
        )
        logger.info("短剧解说：已按抽帧 subtitle_entries（字幕对位）校正片段边界")

    if (subtitle_content or "").strip() or (plot_blueprint or "").strip():
        items = align_script_items_to_source_material(
            items,
            subtitle_content=subtitle_content,
            plot_blueprint=plot_blueprint,
            settings=cfg,
        )

    items = _align_fazu2_ost0_to_adjacent_ost1(
        items,
        cfg,
        frame_analysis_path=frame_analysis_path or "",
    )
    items = align_ost0_timestamps_for_narration_and_lead_in(items, settings=cfg)

    items = fill_ost1_picture_from_blueprint(
        items,
        plot_blueprint=plot_blueprint,
        settings=cfg,
    )
    items = trim_ost1_picture_narrations(items, settings=cfg)
    items = enforce_short_drama_ost_ratio(items, cfg)

    items = format_ost1_picture_narrations(
        items,
        wrap_quotes=bool(cfg.get("picture_wrap_double_quotes", True)),
    )

    items = repair_or_drop_invalid_timestamp_items(
        items,
        subtitle_content=subtitle_content,
        min_duration_ms=max(500, ost0_min_ms),
        ost1_min_duration_ms=min_ost1_ms,
        ost1_max_duration_ms=max_ost1_ms,
        drop_unrecoverable=False,
    )
    return items
