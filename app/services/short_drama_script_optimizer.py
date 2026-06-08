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
    get_short_drama_settings,
    summarize_short_drama_playback,
)
from app.services.srt_utils import (
    is_valid_script_timestamp_range,
    parse_timestamp_range,
    repair_or_drop_invalid_timestamp_items,
    script_timestamp_duration_ms,
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
        f"- **解说 vs 原声成片时长约 {narr_pct}:{orig_pct}**（不限制段数，按时长占比）",
        f"- 每段须完整表达一条脉络：解说 OST=0 每段 ≥{cfg.get('narration_chars_min', 20)} 字；"
        f"原声 OST=1 每段 {cfg.get('ost1_duration_min', 8)}–{cfg.get('ost1_duration_max', 18)} 秒",
        "- 禁止过短碎段；宁可少切几段，也不要堆大量 2–3 秒片段",
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
) -> Dict[str, Any]:
    """校验 LLM 输出：禁止零时长 timestamp，OST=1 须满足最小时长。"""
    cfg = get_short_drama_settings(settings)
    ost1_min_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)
    narr_chars_min = int(cfg.get("narration_chars_min", 20) or 20)
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
                f"片段 #{item_id} timestamp 过短 "
                f"({duration_ms / 1000.0:.2f}s < {min_ms / 1000.0:.1f}s): {ts}"
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
        f"- OST=1 每段时长 {ost1_min}–{ost1_max} 秒，须从字幕复制完整对白区间",
        f"- OST=0 每段 ≥{narr_min} 字、估算时长 ≥{ost0_min} 秒，须完整表达一条脉络",
        "- `_id` 为成片播放顺序；原片 timestamp 可倒叙/闪回，不要求按 _id 单调递增",
    ]
    return "\n".join(lines)


def repair_short_drama_script_timestamps(
    items: List[Dict[str, Any]],
    *,
    subtitle_content: str = "",
    frame_analysis_path: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """仅修复时间轴：subtitle_entries 对位 + 短 timestamp 扩展，不改 narration/picture 文案。"""
    if not items:
        return items

    cfg = settings or get_short_drama_settings()
    ost1_min_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)
    ost1_max_ms = int(float(cfg.get("ost1_duration_max", 18) or 18) * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)

    if (frame_analysis_path or "").strip():
        items = align_short_drama_items_to_frame_time_ranges(
            items,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=float(cfg.get("ost1_duration_max", 18) or 18),
        )

    return repair_or_drop_invalid_timestamp_items(
        items,
        subtitle_content=subtitle_content,
        min_duration_ms=max(500, ost0_min_ms),
        ost1_min_duration_ms=ost1_min_ms,
        ost1_max_duration_ms=ost1_max_ms,
        drop_unrecoverable=False,
    )


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
    max_ost1_ms = int(ost1_max * 1000)
    ost0_min_ms = int(float(cfg.get("ost0_duration_min", 5) or 5) * 1000)

    if (frame_analysis_path or "").strip():
        items = align_short_drama_items_to_frame_time_ranges(
            items,
            subtitle_content=subtitle_content,
            frame_analysis_path=frame_analysis_path,
            ost1_hard_max=ost1_max,
        )
        logger.info("短剧解说：已按抽帧 subtitle_entries（字幕对位）校正片段边界")

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
