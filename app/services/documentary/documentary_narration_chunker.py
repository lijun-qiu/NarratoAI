#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说：长视频解说输入分块，避免超出文本模型上下文。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.documentary.documentary_settings import (
    build_fazu2_generation_anti_patterns,
    build_fazu2_narration_copy_hard_requirements,
    build_fazu2_script_output_reference,
    compute_compact_segment_bounds,
    format_ost1_segment_hint,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_fazu2_core_theme,
)
from app.services.srt_utils import parse_timestamp_range


@dataclass(frozen=True)
class NarrationChunkPlan:
    markdown: str
    time_range: str
    segment_min: int
    segment_max: int
    chunk_index: int
    chunk_total: int


def estimate_text_tokens(text: str) -> int:
    """粗估 token 数（中文偏多时约 1 字 ≈ 1.2 token）。"""
    return max(1, int(len(text or "") * 1.2))


def split_frame_markdown_sections(markdown: str) -> List[str]:
    parts = re.split(r"(?=^## 片段 \d+\n)", markdown or "", flags=re.MULTILINE)
    return [part.strip() for part in parts if part.strip()]


def _section_time_range(section: str) -> str:
    match = re.search(r"- 时间范围：(.+)\n", section)
    return (match.group(1).strip() if match else "").strip()


def _time_range_to_ms(time_range: str) -> tuple[int, int]:
    text = (time_range or "").strip()
    if not text or "-" not in text:
        return 0, 0
    start_text, end_text = text.split("-", 1)
    try:
        start_ms, _ = parse_timestamp_range(start_text.strip())
        _, end_ms = parse_timestamp_range(end_text.strip())
        return start_ms, end_ms
    except Exception:
        return 0, 0


def merge_chunk_time_range(sections: List[str]) -> str:
    if not sections:
        return ""
    starts: list[int] = []
    ends: list[int] = []
    for section in sections:
        time_range = _section_time_range(section)
        start_ms, end_ms = _time_range_to_ms(time_range)
        if end_ms >= start_ms:
            starts.append(start_ms)
            ends.append(end_ms)
    if not starts:
        first = _section_time_range(sections[0])
        last = _section_time_range(sections[-1])
        if first and last and first != last:
            return f"{first.split('-', 1)[0].strip()}-{last.split('-', 1)[-1].strip()}"
        return first or last
    from app.utils import utils

    def _ms_to_timestamp(ms: int) -> str:
        return utils.seconds_to_time(ms / 1000.0).replace(".", ",")

    return f"{_ms_to_timestamp(min(starts))}-{_ms_to_timestamp(max(ends))}"


def split_sections_evenly(sections: List[str], chunk_count: int) -> List[str]:
    """按片段数量均分为 N 份（用于全片覆盖时段数过多、输入不长的情况）。"""
    if not sections:
        return []
    n = max(1, min(int(chunk_count), len(sections)))
    if n <= 1:
        return ["\n\n".join(sections)]
    size = math.ceil(len(sections) / n)
    return ["\n\n".join(sections[i : i + size]) for i in range(0, len(sections), size)]


def group_sections_into_chunks(
    sections: List[str],
    *,
    max_chars_per_chunk: int,
    max_sections_per_chunk: int,
) -> List[str]:
    if not sections:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for section in sections:
        section_len = len(section)
        should_flush = bool(current) and (
            current_len + section_len + 2 > max_chars_per_chunk
            or len(current) >= max_sections_per_chunk
        )
        if should_flush:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(section)
        current_len += section_len + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunk_coverage_override(
    *,
    settings: Optional[Dict[str, Any]] = None,
    time_range: str,
    segment_min: int,
    segment_max: int,
    chunk_index: int,
    chunk_total: int,
    core_theme: str = "",
) -> str:
    cfg = settings or get_documentary_settings()
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))

    if is_fazu2_compact_settings(cfg):
        total_min, _, total_max = compute_compact_segment_bounds(cfg)
        theme = resolve_fazu2_core_theme(core_theme, cfg)
        rules_block = build_fazu2_narration_copy_hard_requirements(theme, settings=cfg)
        anti_block = build_fazu2_generation_anti_patterns()
        seg_target = max(2, (segment_min + segment_max) // 2)
        return (
            f"{rules_block}\n"
            f"{anti_block}\n"
            f"## 本分块精剪（故事讲述型）\n"
            f"- 全片第 **{chunk_index}/{chunk_total}** 块；本段剧情在 **{time_range}** 附近\n"
            f"- 本块约 **{seg_target}** 个情节点；**严禁超过 {segment_max} 段**（全片 **{total_min}–{total_max} 段**）\n"
            f"- **讲故事**：对白写入 OST=0 解说；仅金句用 OST=1（≤6 段/全片）\n"
            f"- `timestamp` 从字幕**原样复制**；禁止整分等间隔编造\n"
            f"- 解说 OST=0：每段 **{chars_min}–{chars_max} 字**；**必须写人名**，禁止警员1/说话人1\n"
            f"{format_ost1_segment_hint(cfg, estimated_items=segment_max)}"
            f"- 本块 `_id` 从 1 起编；合并后按剧情重排；**禁止 OST=2**\n"
        )

    if is_compact_documentary_settings(cfg):
        interval = max(1, int(cfg.get("coverage_interval_sec", 30)))
        seg_target = max(2, (segment_min + segment_max) // 2)
        coverage_line = (
            f"- 本块原片时间轴 **每 {interval} 秒至少 1 段**，不要大段跳过\n"
            if cfg.get("enable_full_timeline_coverage", True)
            else "- 只选本块华彩镜头，允许跳剪\n"
        )
        return (
            f"## 本分块精剪（必须遵守）\n"
            f"- 这是全片第 **{chunk_index}/{chunk_total}** 块，**仅处理原片 {time_range}**\n"
            f"{coverage_line}"
            f"- 本块 items **{segment_min}–{segment_max} 段**（目标约 **{seg_target}** 段，**严禁超过 {segment_max}**）\n"
            f"- 解说 **OST=0**，每段 **{chars_min}–{chars_max} 字**\n"
            f"{format_ost1_segment_hint(cfg, estimated_items=segment_max)}"
            f"- `_id` 从 1 起编；时间戳落在 {time_range} 内；**禁止 OST=2**\n"
        )

    interval = int(cfg.get("coverage_interval_sec", 30))
    return (
        f"## 本分块覆盖（必须遵守）\n"
        f"- 这是全片第 **{chunk_index}/{chunk_total}** 块，**仅处理原片 {time_range}**\n"
        f"- 在此范围内原片时间轴上**每 {interval} 秒至少 1 段**解说\n"
        f"- 本块 items 建议 **{segment_min}–{segment_max} 段**（合并后会重排 `_id`）\n"
        f"- 解说每段 **{chars_min}–{chars_max} 字**；时间戳严禁重叠"
    )


def allocate_chunk_segment_targets(
    *,
    total_sections: int,
    section_counts: List[int],
    total_min: int,
    total_target: int,
    max_items_per_chunk: int = 20,
) -> List[tuple[int, int]]:
    if not section_counts:
        return []

    per_chunk_cap = max(8, max_items_per_chunk)
    weights = [max(1, count) for count in section_counts]
    weight_sum = sum(weights)
    allocations: List[tuple[int, int]] = []

    for count in section_counts:
        share = count / weight_sum
        seg_min = max(2, int(round(total_min * share)))
        seg_target = max(seg_min, int(round(total_target * share)))
        seg_max = max(seg_target + 1, int(round(seg_target * 1.25)))
        seg_max = min(seg_max, per_chunk_cap)
        seg_min = min(seg_min, seg_max)
        allocations.append((seg_min, seg_max))

    return allocations


def _chunk_fits_limits(
    markdown: str,
    *,
    max_chars_per_chunk: int,
    max_sections_per_chunk: int,
) -> bool:
    if len(markdown) > max_chars_per_chunk:
        return False
    section_count = len(split_frame_markdown_sections(markdown))
    return section_count <= max_sections_per_chunk


def _resolve_chunk_markdowns(
    sections: List[str],
    *,
    max_chars_per_chunk: int,
    max_sections_per_chunk: int,
    min_chunk_count: int,
) -> List[str]:
    """以段数容量为下限；仅当单块超出字符/片段上限时才增加块数。"""
    if not sections:
        return []

    min_count = max(1, min(min_chunk_count, len(sections)))
    for chunk_count in range(min_count, len(sections) + 1):
        chunk_markdowns = split_sections_evenly(sections, chunk_count)
        if all(
            _chunk_fits_limits(
                chunk_md,
                max_chars_per_chunk=max_chars_per_chunk,
                max_sections_per_chunk=max_sections_per_chunk,
            )
            for chunk_md in chunk_markdowns
        ):
            return chunk_markdowns

    return group_sections_into_chunks(
        sections,
        max_chars_per_chunk=max_chars_per_chunk,
        max_sections_per_chunk=max_sections_per_chunk,
    )


def plan_narration_chunks(
    markdown: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
    total_segment_min: int = 0,
    total_segment_target: int = 0,
) -> List[NarrationChunkPlan]:
    cfg = settings or get_documentary_settings()
    max_chars = int(cfg.get("narration_chunk_max_chars", 50000))
    max_sections = int(cfg.get("narration_chunk_max_sections", 12))

    sections = split_frame_markdown_sections(markdown)
    if not sections:
        return [
            NarrationChunkPlan(
                markdown=markdown,
                time_range="",
                segment_min=max(1, total_segment_min),
                segment_max=max(2, total_segment_target or total_segment_min or 5),
                chunk_index=1,
                chunk_total=1,
            )
        ]

    max_items_per_call = max(
        8, int(cfg.get("narration_chunk_max_items_per_call", 15) or 15)
    )
    segment_chunk_count = 1
    if total_segment_target > 0:
        segment_chunk_count = max(
            1, math.ceil(total_segment_target / max_items_per_call)
        )

    chunk_markdowns = _resolve_chunk_markdowns(
        sections,
        max_chars_per_chunk=max_chars,
        max_sections_per_chunk=max_sections,
        min_chunk_count=segment_chunk_count,
    )

    if len(chunk_markdowns) <= 1:
        return [
            NarrationChunkPlan(
                markdown=markdown,
                time_range=merge_chunk_time_range(sections),
                segment_min=max(1, total_segment_min),
                segment_max=max(2, total_segment_target or total_segment_min or 5),
                chunk_index=1,
                chunk_total=1,
            )
        ]

    chunk_section_counts = [
        len(split_frame_markdown_sections(chunk_md)) for chunk_md in chunk_markdowns
    ]
    segment_targets = allocate_chunk_segment_targets(
        total_sections=len(sections),
        section_counts=chunk_section_counts,
        total_min=max(total_segment_min, len(chunk_markdowns) * 2),
        total_target=max(total_segment_target, total_segment_min),
        max_items_per_chunk=max_items_per_call,
    )

    plans: List[NarrationChunkPlan] = []
    for index, chunk_md in enumerate(chunk_markdowns, 1):
        chunk_sections = split_frame_markdown_sections(chunk_md)
        seg_min, seg_max = segment_targets[index - 1]
        plans.append(
            NarrationChunkPlan(
                markdown=chunk_md,
                time_range=merge_chunk_time_range(chunk_sections),
                segment_min=seg_min,
                segment_max=seg_max,
                chunk_index=index,
                chunk_total=len(chunk_markdowns),
            )
        )
    return plans


def should_chunk_narration_input(
    narration_input: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> bool:
    cfg = settings or get_documentary_settings()
    max_chars = int(cfg.get("narration_input_max_chars", 90000))
    max_tokens = int(cfg.get("narration_input_max_tokens", 100000))
    text = narration_input or ""
    return len(text) > max_chars or estimate_text_tokens(text) > max_tokens


def should_force_narration_chunking(
    narration_input: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
    total_segment_min: int = 0,
) -> bool:
    """输入过长，或全片覆盖要求段数超过单次模型可稳定输出时，强制分块。"""
    cfg = settings or get_documentary_settings()
    if should_chunk_narration_input(narration_input, settings=cfg):
        return True
    max_items = max(8, int(cfg.get("narration_chunk_max_items_per_call", 15) or 15))
    return total_segment_min > max_items


def reduce_markdown_to_summaries(markdown: str) -> str:
    """仅保留各批次时间范围与片段描述，用于极端超长时的兜底。"""
    sections = split_frame_markdown_sections(markdown)
    if not sections:
        return markdown
    reduced: List[str] = []
    for index, section in enumerate(sections, 1):
        time_range = _section_time_range(section)
        summary_match = re.search(r"- 片段描述：(.+?)(?:\n|$)", section)
        summary = summary_match.group(1).strip() if summary_match else ""
        reduced.append(
            f"## 片段 {index}\n"
            f"- 时间范围：{time_range}\n"
            f"- 片段描述：{summary or '（无摘要）'}\n"
        )
    return "\n\n".join(reduced)


def merge_narration_items(
    chunks: List[List[Dict[str, Any]]],
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for chunk_items in chunks:
        merged.extend(chunk_items)

    cfg = settings or get_documentary_settings()
    if not is_fazu2_compact_settings(cfg):
        def sort_key(item: Dict[str, Any]) -> tuple[int, int]:
            timestamp = str(item.get("timestamp") or "")
            start = timestamp.split("-", 1)[0].strip()
            try:
                start_ms, _ = parse_timestamp_range(start)
            except Exception:
                start_ms = 0
            return start_ms, int(item.get("_id") or 0)

        merged.sort(key=sort_key)

    for index, item in enumerate(merged, 1):
        item["_id"] = index
    return merged
