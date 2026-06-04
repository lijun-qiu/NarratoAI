#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧精剪：检测时间线空洞并用小批量 LLM 补段，保证每 N 秒至少 1 个 item。"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from app.services.documentary.documentary_narration_chunker import split_frame_markdown_sections
from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    is_compact_documentary_settings,
)
from app.services.srt_utils import parse_timestamp_range
from app.utils import utils


def _ms_to_timestamp(ms: int) -> str:
    return utils.seconds_to_time(max(0, ms) / 1000.0).replace(".", ",")


def _slot_time_range(slot_start_ms: int, interval_ms: int, source_end_ms: int) -> str:
    end_ms = min(slot_start_ms + interval_ms, source_end_ms)
    if end_ms <= slot_start_ms:
        end_ms = slot_start_ms + 1000
    return f"{_ms_to_timestamp(slot_start_ms)}-{_ms_to_timestamp(end_ms)}"


def _parse_sections(markdown: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for section in split_frame_markdown_sections(markdown):
        time_range = ""
        match = re.search(r"- 时间范围：(.+)\n", section)
        if match:
            time_range = match.group(1).strip()
        summary = ""
        summary_match = re.search(r"- 片段描述：(.+?)(?:\n|$)", section, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip().replace("\n", " ")
        start_ms, end_ms = 0, 0
        if time_range and "-" in time_range:
            start_text, end_text = time_range.split("-", 1)
            try:
                start_ms, _ = parse_timestamp_range(start_text.strip())
                _, end_ms = parse_timestamp_range(end_text.strip())
            except Exception:
                pass
        sections.append(
            {
                "time_range": time_range,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "summary": summary[:500],
                "raw": section,
            }
        )
    return sections


def _item_covers_slot(item: Dict[str, Any], slot_start_ms: int, slot_end_ms: int) -> bool:
    timestamp = str(item.get("timestamp") or "")
    if not timestamp:
        return False
    try:
        start_ms, end_ms = parse_timestamp_range(timestamp)
    except Exception:
        return False
    return start_ms < slot_end_ms and end_ms > slot_start_ms


def find_uncovered_timeline_slots(
    items: List[Dict[str, Any]],
    *,
    source_duration_sec: float,
    interval_sec: int = 30,
) -> List[Dict[str, Any]]:
    """返回未覆盖的 30s 时间格（含建议 timestamp 与画面摘要）。"""
    if source_duration_sec <= 0:
        return []

    interval_ms = max(1000, interval_sec * 1000)
    source_end_ms = int(source_duration_sec * 1000)
    slot_count = max(1, int(math.ceil(source_duration_sec / interval_sec)))

    uncovered: List[Dict[str, Any]] = []
    for index in range(slot_count):
        slot_start_ms = index * interval_ms
        if slot_start_ms >= source_end_ms:
            break
        slot_end_ms = min(slot_start_ms + interval_ms, source_end_ms)
        if any(_item_covers_slot(item, slot_start_ms, slot_end_ms) for item in items):
            continue
        uncovered.append(
            {
                "slot_index": index,
                "start_ms": slot_start_ms,
                "end_ms": slot_end_ms,
                "timestamp": _slot_time_range(slot_start_ms, interval_ms, source_end_ms),
            }
        )
    return uncovered


def _pick_section_for_slot(sections: List[Dict[str, Any]], slot_start_ms: int) -> Dict[str, Any]:
    if not sections:
        return {"summary": "", "time_range": ""}
    best = sections[0]
    best_dist = 10**18
    for section in sections:
        start_ms = int(section.get("start_ms") or 0)
        end_ms = int(section.get("end_ms") or start_ms)
        if start_ms <= slot_start_ms <= end_ms:
            return section
        mid = (start_ms + end_ms) // 2 if end_ms > start_ms else start_ms
        dist = abs(mid - slot_start_ms)
        if dist < best_dist:
            best_dist = dist
            best = section
    return best


def attach_section_hints(
    slots: List[Dict[str, Any]],
    markdown: str,
) -> List[Dict[str, Any]]:
    sections = _parse_sections(markdown)
    enriched: List[Dict[str, Any]] = []
    for slot in slots:
        section = _pick_section_for_slot(sections, int(slot["start_ms"]))
        enriched.append(
            {
                **slot,
                "picture_hint": (section.get("summary") or "画面片段")[:24],
                "frame_summary": section.get("summary") or "",
                "frame_time_range": section.get("time_range") or "",
            }
        )
    return enriched


def _build_fill_prompt(
    slots: List[Dict[str, Any]],
    *,
    chars_min: int,
    chars_max: int,
) -> str:
    lines = [
        "## 补段任务（必须遵守）",
        f"- 为下列 **{len(slots)}** 个原片时间窗口**各生成 1 个** JSON item，不得合并、不得遗漏",
        f"- 每段 `narration` **{chars_min}–{chars_max} 字**，深度拉片口吻，禁止剧情梗概与流水账",
        "- `OST` 一律 **0**（纯解说）；`timestamp` 必须使用下列给定值，严禁改写",
        "- 只输出 JSON：`{{\"items\":[...]}}`，不要解释",
        "",
        "### 待补窗口",
    ]
    for index, slot in enumerate(slots, 1):
        lines.append(
            f"{index}. timestamp=`{slot['timestamp']}` | 画面参考：{slot.get('frame_summary') or slot.get('picture_hint')}"
        )
    lines.append("")
    lines.append("### 输出示例（items 数量必须等于待补窗口数量）")
    lines.append(
        '{"items":[{"_id":1,"timestamp":"00:00:00,000-00:00:30,000",'
        '"picture":"低机位特写","narration":"镜头把人物的犹豫压在沉默里。",'
        '"OST":0}]}'
    )
    return "\n".join(lines)


def _parse_fill_items(raw: str) -> List[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            return []

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _fallback_items_from_slots(slots: List[Dict[str, Any]], *, chars_max: int) -> List[Dict[str, Any]]:
    """LLM 补段失败时，用抽帧摘要生成短解说占位，保证时间线覆盖。"""
    result: List[Dict[str, Any]] = []
    for slot in slots:
        hint = (slot.get("frame_summary") or slot.get("picture_hint") or "画面").strip()
        if len(hint) > chars_max - 8:
            hint = hint[: max(4, chars_max - 8)] + "…"
        narration = f"注意这组镜头：{hint}"[:chars_max]
        result.append(
            {
                "timestamp": slot["timestamp"],
                "picture": (slot.get("picture_hint") or "画面")[:12],
                "narration": narration,
                "OST": 0,
            }
        )
    return result


def _merge_and_sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]) -> int:
        timestamp = str(item.get("timestamp") or "")
        start = timestamp.split("-", 1)[0].strip()
        try:
            start_ms, _ = parse_timestamp_range(start)
            return start_ms
        except Exception:
            return 0

    ordered = sorted(items, key=sort_key)
    for index, item in enumerate(ordered, 1):
        item["_id"] = index
    return ordered


def fill_timeline_coverage_gaps(
    items: List[Dict[str, Any]],
    *,
    frame_markdown: str,
    source_duration_sec: float,
    settings: Optional[Dict[str, Any]] = None,
    generate_fn: Callable[[str], str],
    max_slots_per_call: int = 8,
) -> List[Dict[str, Any]]:
    """
    对未覆盖的 timeline 格子补段；优先小批量 LLM，失败则用抽帧摘要占位。
    """
    cfg = settings or get_documentary_compact_settings()
    if not is_compact_documentary_settings(cfg):
        return items
    if not cfg.get("enable_full_timeline_coverage", True):
        return items
    if source_duration_sec <= 0:
        return items

    interval_sec = max(1, int(cfg.get("coverage_interval_sec", 30)))
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))
    merged = list(items or [])

    for round_index in range(3):
        uncovered = find_uncovered_timeline_slots(
            merged,
            source_duration_sec=source_duration_sec,
            interval_sec=interval_sec,
        )
        if not uncovered:
            break
        enriched = attach_section_hints(uncovered, frame_markdown)
        logger.info(
            f"时间线补段第 {round_index + 1} 轮：待补 {len(enriched)} 个窗口"
            f"（原片每 {interval_sec}s 至少 1 段）"
        )
        batch_size = max(1, int(max_slots_per_call))
        for offset in range(0, len(enriched), batch_size):
            batch = enriched[offset : offset + batch_size]
            prompt = _build_fill_prompt(batch, chars_min=chars_min, chars_max=chars_max)
            parsed_items: List[Dict[str, Any]] = []
            try:
                raw = generate_fn(prompt)
                parsed_items = _parse_fill_items(raw)
            except Exception as exc:
                logger.warning(f"补段 LLM 调用失败，使用抽帧占位: {exc}")

            by_timestamp: Dict[str, Dict[str, Any]] = {}
            for item in parsed_items:
                ts = str(item.get("timestamp") or "").strip()
                if ts:
                    by_timestamp[ts] = item

            for slot in batch:
                ts = slot["timestamp"]
                if ts in by_timestamp:
                    item = dict(by_timestamp[ts])
                else:
                    item = _fallback_items_from_slots([slot], chars_max=chars_max)[0]
                item["timestamp"] = ts
                item["OST"] = 0
                if not item.get("picture"):
                    item["picture"] = (slot.get("picture_hint") or "画面")[:12]
                merged.append(item)

        merged = _merge_and_sort_items(merged)

    remaining = find_uncovered_timeline_slots(
        merged,
        source_duration_sec=source_duration_sec,
        interval_sec=interval_sec,
    )
    if remaining:
        logger.warning(f"补段后仍有 {len(remaining)} 个时间窗未覆盖，写入占位段")
        merged.extend(
            _fallback_items_from_slots(
                attach_section_hints(remaining, frame_markdown),
                chars_max=chars_max,
            )
        )
        merged = _merge_and_sort_items(merged)

    return merged
