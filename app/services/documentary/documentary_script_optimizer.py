#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说脚本后处理：OST 归一化与原声高光段校验。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.documentary.documentary_settings import (
    FAZU2_FORBIDDEN_NARRATION_PHRASES,
    compute_compact_segment_bounds,
    compute_max_ost1_segments,
    compute_ost1_segment_bounds,
    get_documentary_compact_settings,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
)
from app.services.srt_utils import parse_timestamp_range
from app.utils import utils


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


def _ms_to_timestamp(ms: int) -> str:
    return utils.seconds_to_time(max(0, ms) / 1000.0).replace(".", ",")


def _narration_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", (text or "").strip()))


def _min_narration_duration_sec(text: str) -> float:
    chars = _narration_char_count(text)
    if chars <= 0:
        return 0.0
    return (chars / 10.0) * 1.5


def _is_quote_only_narration(text: str) -> bool:
    """narration 是否仅为一句原声台词（应标 OST=1）。"""
    narration = (text or "").strip()
    if not narration or narration.startswith("播放原片"):
        return False
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
    if int(item.get("OST", 0)) != 0:
        text = narration
    else:
        text = f"{narration}\n{picture}"
    hits = [phrase for phrase in FAZU2_FORBIDDEN_NARRATION_PHRASES if phrase in text]
    if hits:
        logger.warning(
            f"片段 #{item.get('_id')} 含禁用表述 {hits}，建议改用具体人名后重新生成"
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


def _adjust_fazu2_ost1_timestamps(
    items: List[Dict[str, Any]],
    ost1_min: float,
) -> List[Dict[str, Any]]:
    """金句原声段过短时延长 time戳，避免被误判后回退为 OST=0。"""
    if not items or ost1_min <= 0:
        return items
    ordered = sorted(items, key=_item_sort_key)
    next_starts: List[int] = []
    for item in ordered:
        start_ms, _ = parse_timestamp_range(str(item.get("timestamp") or ""))
        next_starts.append(start_ms)
    min_ms = int(ost1_min * 1000)
    for index, item in enumerate(ordered):
        if int(item.get("OST", 0)) != 1:
            continue
        start_ms, end_ms = parse_timestamp_range(str(item.get("timestamp") or ""))
        current_ms = end_ms - start_ms
        if current_ms >= min_ms:
            continue
        needed_end_ms = start_ms + min_ms
        if index + 1 < len(next_starts):
            needed_end_ms = min(needed_end_ms, next_starts[index + 1] - 50)
        if needed_end_ms <= start_ms:
            continue
        item["timestamp"] = (
            f"{_ms_to_timestamp(start_ms)}-{_ms_to_timestamp(needed_end_ms)}"
        )
        logger.info(
            f"片段 #{item.get('_id')} 原声 OST=1 时长 {current_ms / 1000.0:.1f}s 短于 {ost1_min:.1f}s，"
            f"已延长至 {(needed_end_ms - start_ms) / 1000.0:.1f}s"
        )
    return ordered


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
    """逐帧精剪：为首尾 OST=0 解说段写入宝子们开场/结尾。"""
    if not items:
        return items
    cfg = get_documentary_compact_settings(settings)
    if not is_fazu2_compact_settings(cfg):
        return items
    if not cfg.get("enable_opening_closing_hook", True):
        return items

    opening = _format_compact_hook_template(
        str(cfg.get("opening_hook_template") or "宝子们，我们开始《{work_name}》啦！"),
        work_name,
    )
    closing = _format_compact_hook_template(
        str(cfg.get("closing_hook_template") or "宝子们，我们下期再见！"),
        work_name,
    )
    if not opening and not closing:
        return items

    updated = [dict(item) for item in items]
    ost0_indices = [
        i for i, item in enumerate(updated) if int(item.get("OST", 0)) == 0
    ]
    if not ost0_indices:
        return items

    first_idx = ost0_indices[0]
    last_idx = ost0_indices[-1]

    if opening:
        existing = str(updated[first_idx].get("narration") or "").strip()
        hook = opening.rstrip("。！!")
        if not _opening_hook_already_applied(existing, hook):
            if existing:
                updated[first_idx]["narration"] = f"{hook}。{existing.lstrip('。')}"
            else:
                updated[first_idx]["narration"] = hook
            logger.info(f"已应用精剪开场白: {updated[first_idx]['narration'][:50]}...")

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
) -> List[Dict[str, Any]]:
    """归一化 LLM 输出的逐帧解说脚本 OST 字段。"""
    if not items:
        return []

    if settings and is_compact_documentary_settings(settings):
        cfg = get_documentary_compact_settings(settings)
    else:
        cfg = get_documentary_settings(settings)
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
                if _is_preservable_ost1_dialogue(existing_narration):
                    item["narration"] = existing_narration
                elif existing_narration:
                    item["narration"] = existing_narration
                else:
                    logger.warning(
                        f"片段 #{item['_id']} OST=1 缺少台词原文，请按模板填写 narration"
                    )
                    item["narration"] = f"播放原片{ost1_counter}"
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
        result = _break_consecutive_ost1(result, default_ost)
        for item in result:
            _warn_fazu2_forbidden_phrases(item)
        result = _adjust_fazu2_ost1_timestamps(result, ost1_min)
        result = _adjust_fazu2_ost0_timestamps(result)
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

    return result
