#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""开篇爆燃段（第 1 段 OST=1）解析与校正：从字幕/对照分析/抽帧中定位，并修正 LLM 误选。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from app.services.documentary.documentary_settings import (
    is_compact_documentary_settings,
    resolve_append_custom_prompt,
    resolve_fazu2_opening_climax_hint,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    parse_timestamp_range_ms,
    resolve_segment_subtitle_text,
    resolve_segment_time_range,
)
from app.services.documentary.frame_analysis_compact import _collect_top_level_segments
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.srt_utils import (
    dialogue_match_key,
    find_subtitle_span_for_line,
    find_subtitle_span_global,
    format_timestamp_ms,
    parse_srt,
    parse_timestamp_range,
)

_QUOTE_RE = re.compile(r"「([^」]+)」")
_TIMESTAMP_RANGE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*[-–—~至到]\s*(\d{2}:\d{2}:\d{2},\d{3})"
)
_SINGLE_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}")

_HIGH_ENERGY_KEYWORDS = (
    "枪", "枪战", "枪声", "搏斗", "扭打", "打斗", "追逐", "爆炸", "尖叫",
    "生死", "对决", "激烈", "冲突", "刺杀", "血", "掏枪", "砸", "怒吼",
    "崩溃", "牺牲", "跳楼", "纵身", "跃下", "坠落", "名场面", "对峙",
    "抓捕", "审讯", "咆哮", "绝望",
)
_LOW_ENERGY_EARLY_MARKERS = (
    "我说句没觉悟", "你都到厅级", "干嘛非要", "狗贩子", "想清楚了",
    "胡小跃是我的徒弟", "我知道", "死于自杀", "拨云见日", "徒弟被陷害",
    "悲愤揭露", "被逼上绝路", "厅级",
)
_OPENING_HINT_LATE_JUMP_MARKERS = ("跳楼", "纵身", "跃下", "坠落", "牺牲")
_OPENING_SCENE_JUMP_KEYWORDS = (
    "跳楼", "纵身", "跃下", "坠落", "边缘", "站起", "纵身跃", "夜色",
)
_OPENING_SCENE_ROOFTOP_KEYWORDS = ("楼顶", "天台", "屋顶")
_OPENING_JUMP_PICTURE_MARKERS = ("跳楼", "纵身", "跃下", "坠落", "边缘纵身", "站起纵身")
_OPENING_WRONG_PICTURE_MARKERS = ("打电话", "手持手机", "手机通话", "街头手持手机", "警服在夜间街头")
_MIN_OPENING_QUOTE_CHARS = 4
_OPENING_SECTION_RE = re.compile(
    r"##\s*开头高潮方案.*?(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_EPISODE_SUMMARY_RE = re.compile(
    r"##\s*本集主线摘要.*?(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_OST1_SECTION_RE = re.compile(
    r"##\s*OST\s*=\s*1.*?(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_OPENING_USAGE_MARKERS = (
    "开头高潮", "开头爆燃", "开篇", "开头", "第1段", "第 1 段", "爆燃", "高潮前置",
)
_REASON_LINE_RE = re.compile(
    r"爆燃理由[：:]\s*(.+)",
    re.IGNORECASE,
)


@dataclass
class OpeningClimaxMatch:
    timestamp: str
    original_line: str = ""
    picture: str = ""
    source: str = ""
    quote: str = ""


def _extract_quoted_strings(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTE_RE.finditer(text or "") if match.group(1).strip()]


def _extract_timestamp_ranges(text: str) -> list[tuple[str, str]]:
    ranges: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _TIMESTAMP_RANGE_RE.finditer(text or ""):
        start, end = match.group(1), match.group(2)
        key = f"{start}-{end}"
        if key not in seen:
            seen.add(key)
            ranges.append((start, end))
    for token in _SINGLE_TIMESTAMP_RE.findall(text or ""):
        if not any(token in f"{start}-{end}" for start, end in ranges):
            ranges.append((token, token))
    return ranges


def _timestamp_to_ms(timestamp: str) -> int:
    start_ms, _ = parse_timestamp_range_ms(timestamp)
    return start_ms


def _format_range(start_ms: int, end_ms: int) -> str:
    return f"{format_timestamp_ms(start_ms)}-{format_timestamp_ms(end_ms)}"


def _hint_expects_late_jump(hint_text: str) -> bool:
    text = (hint_text or "").strip()
    return any(marker in text for marker in _OPENING_HINT_LATE_JUMP_MARKERS)


def _opening_span_limits(settings: Optional[dict[str, Any]] = None) -> tuple[int, int]:
    cfg = settings or {}
    min_ms = int(float(cfg.get("ost1_duration_min", 8) or 8) * 1000)
    hard_max = cfg.get("ost1_duration_hard_max", cfg.get("ost1_duration_max", 18))
    max_ms = int(float(hard_max or 18) * 1000)
    min_ms = max(min_ms, 8_000)
    max_ms = max(max_ms, min_ms + 1_000, 10_000)
    return min_ms, min_ms if max_ms < min_ms else max_ms


def _expand_quote_variants(quotes: list[str]) -> list[str]:
    variants: list[str] = []
    for quote in quotes:
        text = (quote or "").strip()
        if not text:
            continue
        for candidate in (text, text.rstrip("。！？!?"), f"{text.rstrip('。！？!?')}。"):
            if candidate and candidate not in variants:
                variants.append(candidate)
        if "天就快亮了" in text:
            for alt in ("天就快亮了", "天就快了", "天快亮了", "天就快亮了。"):
                if alt not in variants:
                    variants.append(alt)
    return variants


def _quote_keys_match(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False
    if len(needle) < _MIN_OPENING_QUOTE_CHARS:
        return needle == hay
    if hay in needle and len(hay) < max(_MIN_OPENING_QUOTE_CHARS, len(needle) // 2):
        return False
    if needle in hay or hay in needle:
        return True
    if len(needle) >= 4 and len(hay) >= 4 and needle[:4] == hay[:4]:
        return True
    return False


def _expand_opening_span(
    entries: list,
    start_ms: int,
    end_ms: int,
    *,
    min_span_ms: int,
    max_span_ms: int,
    merge_gap_ms: int = 4_000,
) -> tuple[int, int]:
    if not entries:
        return start_ms, end_ms

    center_index: int | None = None
    best_distance: int | None = None
    for index, entry in enumerate(entries):
        if entry.end_ms < start_ms - 500:
            continue
        if entry.start_ms > end_ms + 500:
            break
        if entry.start_ms <= end_ms and entry.end_ms >= start_ms:
            distance = abs(entry.end_ms - end_ms)
            if center_index is None or best_distance is None or distance < best_distance:
                center_index = index
                best_distance = distance
    if center_index is None:
        if end_ms - start_ms >= min_span_ms:
            return start_ms, end_ms
        return max(0, end_ms - min_span_ms), end_ms

    lo = hi = center_index
    while lo > 0 and entries[lo].start_ms - entries[lo - 1].end_ms <= merge_gap_ms:
        if entries[hi].end_ms - entries[lo - 1].start_ms > max_span_ms:
            break
        lo -= 1
    while hi < len(entries) - 1 and entries[hi + 1].start_ms - entries[hi].end_ms <= merge_gap_ms:
        if entries[hi + 1].end_ms - entries[lo].start_ms > max_span_ms:
            break
        hi += 1

    expanded_end = max(entries[hi].end_ms, end_ms)
    expanded_start = entries[lo].start_ms
    if expanded_end - expanded_start < min_span_ms:
        expanded_start = max(0, expanded_end - min_span_ms)
    if expanded_end - expanded_start > max_span_ms:
        expanded_start = expanded_end - max_span_ms
    return expanded_start, expanded_end


def _segment_blob(segment: dict[str, Any]) -> str:
    return " ".join(
        str(segment.get(key) or "")
        for key in ("scene", "observation", "action", "emotion", "key_visual", "subtitle")
    )


def _segment_has_jump_cues(segment: dict[str, Any]) -> bool:
    blob = _segment_blob(segment)
    return any(keyword in blob for keyword in _OPENING_SCENE_JUMP_KEYWORDS)


def _segment_is_rooftop(segment: dict[str, Any]) -> bool:
    blob = _segment_blob(segment)
    return any(keyword in blob for keyword in _OPENING_SCENE_ROOFTOP_KEYWORDS)


def _picture_has_jump_cues(picture: str) -> bool:
    text = (picture or "").strip()
    return any(marker in text for marker in _OPENING_JUMP_PICTURE_MARKERS)


def _picture_has_phone_cues(picture: str) -> bool:
    text = (picture or "").strip()
    return any(marker in text for marker in _OPENING_WRONG_PICTURE_MARKERS)


def _prefer_opening_picture(
    *,
    blueprint_picture: str = "",
    segment_picture: str = "",
    hint_expects_late_jump: bool = False,
) -> str:
    blueprint = (blueprint_picture or "").strip()
    segment = (segment_picture or "").strip()
    if hint_expects_late_jump:
        if _picture_has_jump_cues(blueprint):
            return blueprint
        if _picture_has_jump_cues(segment):
            return segment
        if blueprint and not _picture_has_phone_cues(blueprint):
            return blueprint
        if segment and not _picture_has_phone_cues(segment):
            return segment
        return blueprint or segment or "夜色楼顶，胡小跃站在边缘，纵身跃下"
    return blueprint or segment


def parse_opening_climax_from_analysis(markdown: str) -> dict[str, Any]:
    """从字幕×抽帧策划蓝图提取开篇爆燃线索（优先本集主线，而非机械搜字幕）。"""
    return parse_episode_blueprint(markdown)


def parse_episode_blueprint(markdown: str) -> dict[str, Any]:
    """解析策划蓝图：本集主线、开头高潮方案、OST=1 开头项。"""
    text = (markdown or "").strip()
    if not text:
        return {}

    opening_match = _OPENING_SECTION_RE.search(text)
    opening_section = opening_match.group(0) if opening_match else ""

    summary_match = _EPISODE_SUMMARY_RE.search(text)
    episode_summary = summary_match.group(0) if summary_match else ""

    ost1_match = _OST1_SECTION_RE.search(text)
    ost1_section = ost1_match.group(0) if ost1_match else ""

    picture = ""
    for pattern in (
        re.compile(r"(?:\*\*)?picture\s*描述(?:\*\*)?[：:]\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:\*\*)?(?:场面描述|动作描述)(?:\*\*)?[：:]\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:\*\*)?爆燃场面名称(?:\*\*)?[：:]\s*(.+)", re.IGNORECASE),
    ):
        pic_match = pattern.search(opening_section)
        if pic_match:
            picture = pic_match.group(1).strip().strip("-· ")
            break
    if not picture:
        for line in opening_section.splitlines():
            stripped = line.strip().lstrip("-*· ")
            if stripped.startswith("爆燃场面") and "：" in stripped:
                picture = stripped.split("：", 1)[-1].strip()
                break

    reason = ""
    reason_match = _REASON_LINE_RE.search(opening_section)
    if reason_match:
        reason = reason_match.group(1).strip()

    ost1_opening_items = _parse_ost1_opening_items(ost1_section)

    quotes: list[str] = []
    for source in (opening_section, episode_summary, ost1_section):
        for quote in _extract_quoted_strings(str(source or "")):
            if quote not in quotes:
                quotes.append(quote)
    for item in ost1_opening_items:
        for quote in item.get("quotes") or []:
            if quote not in quotes:
                quotes.append(quote)

    timestamp_ranges = list(_extract_timestamp_ranges(opening_section))
    for item in ost1_opening_items:
        for item_range in item.get("timestamp_ranges") or []:
            if item_range not in timestamp_ranges:
                timestamp_ranges.append(item_range)

    return {
        "quotes": quotes,
        "timestamp_ranges": timestamp_ranges,
        "section": opening_section,
        "episode_summary": episode_summary,
        "opening_reason": reason,
        "picture": picture,
        "ost1_opening_items": ost1_opening_items,
        "plot_expects_late_climax": _plot_expects_late_climax(episode_summary, opening_section, reason),
    }


def _plot_expects_late_climax(*text_parts: str) -> bool:
    blob = " ".join(str(part or "") for part in text_parts)
    late_markers = (
        "片尾", "结尾", "最终", "结局", "跳楼", "牺牲", "纵身", "跃下",
        "天就快亮", "全剧第1集", "全剧第 1 集", "第一集",
    )
    return any(marker in blob for marker in late_markers)


def _parse_ost1_opening_items(ost1_section: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not (ost1_section or "").strip():
        return items
    for line in ost1_section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not any(marker in stripped for marker in _OPENING_USAGE_MARKERS):
            continue
        items.append(
            {
                "line": stripped,
                "quotes": _extract_quoted_strings(stripped),
                "timestamp_ranges": _extract_timestamp_ranges(stripped),
            }
        )
    return items


def _resolve_blueprint_clues(
    entries: list,
    *,
    quotes: list[str],
    timestamp_ranges: list[tuple[str, str]],
    picture: str = "",
    prefer_late: bool = False,
    settings: Optional[dict[str, Any]] = None,
) -> OpeningClimaxMatch | None:
    """将策划蓝图中的金句/时间戳落地到字幕条目。"""
    min_span_ms, max_span_ms = _opening_span_limits(settings)

    for start_text, end_text in timestamp_ranges:
        if end_text == start_text:
            start_ms = _timestamp_to_ms(start_text)
            end_ms = start_ms + min_span_ms
        else:
            start_ms, end_ms = parse_timestamp_range_ms(f"{start_text}-{end_text}")
        if quotes:
            matched = _best_quote_in_window(
                entries,
                _expand_quote_variants(quotes),
                window_start_ms=max(0, start_ms - 3000),
                window_end_ms=end_ms + 3000,
            )
            if matched:
                matched.source = "episode_blueprint"
                if picture:
                    matched.picture = picture
                q_start, q_end = parse_timestamp_range_ms(matched.timestamp)
                span_start = min(start_ms, q_start)
                span_end = max(end_ms, q_end)
                span_start, span_end = _expand_opening_span(
                    entries,
                    span_start,
                    span_end,
                    min_span_ms=min_span_ms,
                    max_span_ms=max_span_ms,
                )
                matched.timestamp = _format_range(span_start, span_end)
                return matched
        if end_ms > start_ms:
            span_start, span_end = _expand_opening_span(
                entries,
                start_ms,
                end_ms,
                min_span_ms=min_span_ms,
                max_span_ms=max_span_ms,
            )
            match = OpeningClimaxMatch(
                timestamp=_format_range(span_start, span_end),
                source="episode_blueprint",
                picture=picture,
            )
            if quotes:
                match.quote = quotes[0]
                match.original_line = f"「{quotes[0].strip('「」')}」"
            return match

    if quotes:
        matched = _match_from_quotes_only(
            entries,
            quotes,
            prefer_late=prefer_late,
        )
        if matched:
            matched.source = "episode_blueprint"
            if picture:
                matched.picture = picture
            return matched
    return None


def _match_from_episode_blueprint(
    entries: list,
    blueprint: dict[str, Any],
    *,
    settings: Optional[dict[str, Any]] = None,
) -> OpeningClimaxMatch | None:
    """优先采用策划蓝图（本集主线 + 开头高潮方案），再对齐字幕时间戳。"""
    if not entries or not blueprint:
        return None

    opening_section = str(blueprint.get("section") or "").strip()
    if not opening_section and not blueprint.get("ost1_opening_items"):
        return None

    picture = str(blueprint.get("picture") or "").strip()
    prefer_late = bool(blueprint.get("plot_expects_late_climax"))

    for item in blueprint.get("ost1_opening_items") or []:
        matched = _resolve_blueprint_clues(
            entries,
            quotes=list(item.get("quotes") or []),
            timestamp_ranges=list(item.get("timestamp_ranges") or []),
            picture=picture,
            prefer_late=prefer_late,
            settings=settings,
        )
        if matched:
            logger.info(f"开篇爆燃来自策划蓝图 OST=1 开头项: {item.get('line', '')[:80]}")
            return matched

    matched = _resolve_blueprint_clues(
        entries,
        quotes=list(blueprint.get("quotes") or []),
        timestamp_ranges=list(blueprint.get("timestamp_ranges") or []),
        picture=picture,
        prefer_late=prefer_late,
        settings=settings,
    )
    if matched:
        reason = str(blueprint.get("opening_reason") or "").strip()
        if reason:
            logger.info(f"开篇爆燃来自策划蓝图·开头高潮方案（{reason[:60]}）")
        else:
            logger.info("开篇爆燃来自策划蓝图·开头高潮方案")
    return matched


def _pick_picture_from_segment(segment: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("scene", "observation", "action", "key_visual"):
        value = str(segment.get(key) or "").strip()
        if value:
            parts.append(value)
    return "，".join(parts[:3])


def _score_segment_for_opening(
    segment: dict[str, Any],
    *,
    hint_expects_late: bool = False,
) -> float:
    blob = _segment_blob(segment)
    score = 0.0
    for keyword in _HIGH_ENERGY_KEYWORDS:
        if keyword in blob:
            score += 2.0
    if _segment_has_jump_cues(segment):
        score += 4.0
    if "打电话" in blob or "手机通话" in blob or "手持手机" in blob:
        if hint_expects_late:
            score -= 8.0
        else:
            score -= 2.0
    if _segment_is_rooftop(segment) and "夜色" in blob:
        score += 3.0
    time_range = resolve_segment_time_range(segment) or str(segment.get("timestamp") or "")
    start_ms = 0
    if time_range:
        start_ms, _ = parse_timestamp_range_ms(time_range.split("-", 1)[0])
        if hint_expects_late:
            if start_ms < 3 * 60 * 1000 and not _segment_has_jump_cues(segment):
                score -= 12.0
            elif start_ms < 10 * 60 * 1000 and not _segment_has_jump_cues(segment):
                score -= 6.0
            elif start_ms >= 15 * 60 * 1000 and _segment_has_jump_cues(segment):
                score += 5.0
        else:
            if start_ms < 5 * 60 * 1000:
                score -= 1.5
            if start_ms >= 10 * 60 * 1000:
                score += 1.0
    subtitle = resolve_segment_subtitle_text(segment)
    for marker in _LOW_ENERGY_EARLY_MARKERS:
        if marker in subtitle:
            score -= 3.0
    if hint_expects_late and start_ms < 2 * 60 * 1000:
        score -= 8.0
    return score


def _find_quote_span(
    entries: list,
    quote: str,
    *,
    near_start_ms: int | None = None,
    max_span_ms: int = 22_000,
) -> tuple[int, int] | None:
    if not entries or not (quote or "").strip():
        return None
    needle = dialogue_match_key(quote)
    if len(needle) < _MIN_OPENING_QUOTE_CHARS:
        return None
    if near_start_ms is None:
        span = find_subtitle_span_global(entries, quote, max_span_ms=max_span_ms)
    else:
        span = find_subtitle_span_for_line(
            entries,
            quote,
            near_start_ms=near_start_ms,
            max_span_ms=max_span_ms,
        )
    if not span:
        return None
    start_ms, end_ms = span
    anchor = near_start_ms if near_start_ms is not None else start_ms
    for entry in entries:
        if entry.end_ms < anchor - 8_000:
            continue
        if entry.start_ms > anchor + 8_000:
            break
        hay = dialogue_match_key(entry.text)
        if not _quote_keys_match(needle, hay):
            continue
        if abs(entry.start_ms - anchor) < abs(start_ms - anchor):
            start_ms, end_ms = entry.start_ms, entry.end_ms
    return start_ms, end_ms


def _collect_quote_matches(
    entries: list,
    quotes: list[str],
    *,
    prefer_late: bool,
) -> list[OpeningClimaxMatch]:
    matches: list[OpeningClimaxMatch] = []
    seen: set[str] = set()
    for quote in _expand_quote_variants(quotes):
        if any(marker in quote for marker in _LOW_ENERGY_EARLY_MARKERS):
            continue
        needle = dialogue_match_key(quote)
        if len(needle) < _MIN_OPENING_QUOTE_CHARS:
            continue
        for entry in entries:
            hay = dialogue_match_key(entry.text)
            if not _quote_keys_match(needle, hay):
                continue
            span = _find_quote_span(entries, quote, near_start_ms=entry.start_ms)
            if not span:
                continue
            start_ms, end_ms = span
            key = f"{start_ms}-{end_ms}-{quote}"
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                OpeningClimaxMatch(
                    timestamp=_format_range(start_ms, end_ms),
                    original_line=f"「{quote.rstrip('。！？!?')}。」",
                    source="subtitle_quote",
                    quote=quote,
                )
            )
    matches.sort(
        key=lambda item: (
            _score_quote(
                item.quote or "",
                _timestamp_to_ms(item.timestamp.split("-", 1)[0]),
                prefer_late=prefer_late,
            ),
            _timestamp_to_ms(item.timestamp.split("-", 1)[0]),
        ),
        reverse=True,
    )
    if prefer_late and matches:
        late = [item for item in matches if _timestamp_to_ms(item.timestamp.split("-", 1)[0]) >= 12 * 60 * 1000]
        if late:
            return late
    return matches


def _best_quote_in_window(
    entries: list,
    quotes: list[str],
    *,
    window_start_ms: int,
    window_end_ms: int,
) -> OpeningClimaxMatch | None:
    best: OpeningClimaxMatch | None = None
    best_score = -1.0
    for quote in quotes:
        span = _find_quote_span(entries, quote, near_start_ms=window_start_ms)
        if not span:
            continue
        start_ms, end_ms = span
        if end_ms < window_start_ms or start_ms > window_end_ms:
            continue
        energy = sum(1 for keyword in _HIGH_ENERGY_KEYWORDS if keyword in quote)
        score = energy + (end_ms - start_ms) / 1000.0
        if score > best_score:
            best_score = score
            best = OpeningClimaxMatch(
                timestamp=_format_range(start_ms, end_ms),
                original_line=f"「{quote}」",
                source="subtitle_quote",
                quote=quote,
            )
    return best


def _match_from_timestamp_ranges(
    entries: list,
    ranges: list[tuple[str, str]],
    quotes: list[str],
) -> OpeningClimaxMatch | None:
    for start_text, end_text in ranges:
        if end_text == start_text:
            start_ms = _timestamp_to_ms(start_text)
            end_ms = start_ms + 8000
        else:
            start_ms, end_ms = parse_timestamp_range_ms(f"{start_text}-{end_text}")
        if quotes:
            matched = _best_quote_in_window(
                entries,
                quotes,
                window_start_ms=max(0, start_ms - 2000),
                window_end_ms=end_ms + 2000,
            )
            if matched:
                matched.source = "analysis_timestamp"
                return matched
        if end_ms > start_ms:
            return OpeningClimaxMatch(
                timestamp=_format_range(start_ms, end_ms),
                source="analysis_timestamp",
            )
    return None


def _match_from_late_sacrifice_segments(
    entries: list,
    segments: list[dict[str, Any]],
    quotes: list[str],
    *,
    hint_expects_late: bool,
    settings: Optional[dict[str, Any]] = None,
) -> OpeningClimaxMatch | None:
    if not entries or not segments or not hint_expects_late:
        return None

    min_span_ms, max_span_ms = _opening_span_limits(settings)
    ranked = sorted(
        segments,
        key=lambda segment: _score_segment_for_opening(segment, hint_expects_late=True),
        reverse=True,
    )

    late_quotes = _collect_quote_matches(entries, quotes, prefer_late=True) if quotes else []
    if late_quotes:
        quote_match = late_quotes[0]
        quote_start, quote_end = parse_timestamp_range_ms(quote_match.timestamp)
        best_segment: dict[str, Any] | None = None
        best_score = -1.0
        for segment in ranked:
            time_range = resolve_segment_time_range(segment)
            if not time_range or "-" not in time_range:
                continue
            seg_start, seg_end = parse_timestamp_range_ms(time_range)
            if seg_end < quote_start - 8000 or seg_start > quote_end + 3000:
                continue
            if seg_start < 12 * 60 * 1000:
                continue
            score = _score_segment_for_opening(segment, hint_expects_late=True)
            if seg_start <= quote_start <= seg_end + 3000:
                score += 6.0
            if score > best_score:
                best_score = score
                best_segment = segment

        span_end = quote_end
        if best_segment is not None:
            seg_start, _seg_end = parse_timestamp_range_ms(
                resolve_segment_time_range(best_segment) or quote_match.timestamp
            )
            span_start = max(seg_start, span_end - max_span_ms)
            picture = _pick_picture_from_segment(best_segment)
        else:
            span_start = max(0, span_end - max_span_ms)
            picture = "夜色楼顶，胡小跃站在边缘，纵身跃下"

        span_start, span_end = _expand_opening_span(
            entries,
            span_start,
            span_end,
            min_span_ms=min_span_ms,
            max_span_ms=max_span_ms,
        )
        return OpeningClimaxMatch(
            timestamp=_format_range(span_start, span_end),
            original_line=quote_match.original_line,
            picture=picture,
            source="late_sacrifice_segment",
            quote=quote_match.quote,
        )

    for segment in ranked[:12]:
        score = _score_segment_for_opening(segment, hint_expects_late=True)
        if score < 2.0:
            continue
        time_range = resolve_segment_time_range(segment)
        if not time_range or "-" not in time_range:
            continue
        start_ms, end_ms = parse_timestamp_range_ms(time_range)
        if start_ms < 12 * 60 * 1000:
            continue
        if not (_segment_has_jump_cues(segment) or (_segment_is_rooftop(segment) and start_ms >= 15 * 60 * 1000)):
            continue

        quote_match: OpeningClimaxMatch | None = None
        if quotes:
            matched = _best_quote_in_window(
                entries,
                _expand_quote_variants(quotes),
                window_start_ms=max(0, start_ms - 3000),
                window_end_ms=end_ms + 5000,
            )
            if matched:
                quote_match = matched

        if quote_match:
            quote_start, quote_end = parse_timestamp_range_ms(quote_match.timestamp)
            span_start = min(start_ms, quote_start)
            span_end = max(end_ms, quote_end)
            original_line = quote_match.original_line
            quote = quote_match.quote
        else:
            span_start, span_end = start_ms, end_ms
            window_entries = [
                entry
                for entry in entries
                if not (entry.end_ms < start_ms or entry.start_ms > end_ms)
            ]
            if not window_entries:
                continue
            pick = max(window_entries, key=lambda entry: len((entry.text or "").strip()))
            quote = (pick.text or "").strip().replace("\n", " ")
            if not quote:
                continue
            span = _find_quote_span(entries, quote, near_start_ms=pick.start_ms) or (pick.start_ms, pick.end_ms)
            span_start, span_end = span
            original_line = f"「{quote}」"

        return OpeningClimaxMatch(
            timestamp=_format_range(span_start, span_end),
            original_line=original_line,
            picture=_pick_picture_from_segment(segment),
            source="late_sacrifice_segment",
            quote=quote or "",
        )
    return None


def _finalize_opening_match(
    entries: list,
    segments: list[dict[str, Any]],
    matched: OpeningClimaxMatch,
    *,
    hint_text: str,
    settings: Optional[dict[str, Any]] = None,
) -> OpeningClimaxMatch:
    if not matched or not matched.timestamp or "-" not in matched.timestamp:
        return matched

    start_ms, end_ms = parse_timestamp_range_ms(matched.timestamp)
    min_span_ms, max_span_ms = _opening_span_limits(settings)
    expanded_start, expanded_end = _expand_opening_span(
        entries,
        start_ms,
        end_ms,
        min_span_ms=min_span_ms,
        max_span_ms=max_span_ms,
    )

    blueprint_picture = str(matched.picture or "").strip()

    if _hint_expects_late_jump(hint_text):
        quote_ms = end_ms
        if matched.quote:
            quote_span = _find_quote_span(entries, matched.quote, near_start_ms=start_ms)
            if quote_span:
                quote_ms = quote_span[1]
        best_segment: dict[str, Any] | None = None
        best_score = -1.0
        for segment in segments:
            time_range = resolve_segment_time_range(segment)
            if not time_range or "-" not in time_range:
                continue
            seg_start, seg_end = parse_timestamp_range_ms(time_range)
            if seg_end < expanded_start - 5000 or seg_start > quote_ms + 5000:
                continue
            if seg_start < 12 * 60 * 1000 and not _segment_has_jump_cues(segment):
                continue
            score = _score_segment_for_opening(segment, hint_expects_late=True)
            if seg_start <= quote_ms <= seg_end + 3000:
                score += 3.0
            if score > best_score:
                best_score = score
                best_segment = segment
        if best_segment is not None:
            seg_start, _seg_end = parse_timestamp_range_ms(
                resolve_segment_time_range(best_segment) or matched.timestamp
            )
            expanded_start = min(expanded_start, seg_start)
            expanded_end = max(expanded_end, quote_ms)
            matched.picture = _prefer_opening_picture(
                blueprint_picture=blueprint_picture,
                segment_picture=_pick_picture_from_segment(best_segment),
                hint_expects_late_jump=True,
            )

    expanded_start, expanded_end = _expand_opening_span(
        entries,
        expanded_start,
        expanded_end,
        min_span_ms=min_span_ms,
        max_span_ms=max_span_ms,
    )
    if _hint_expects_late_jump(hint_text) and matched.quote:
        quote_span = _find_quote_span(entries, matched.quote, near_start_ms=expanded_start)
        if quote_span:
            expanded_end = max(expanded_end, quote_span[1])
            if expanded_end - expanded_start > max_span_ms:
                expanded_start = expanded_end - max_span_ms
    matched.timestamp = _format_range(expanded_start, expanded_end)
    if _hint_expects_late_jump(hint_text):
        matched.picture = _prefer_opening_picture(
            blueprint_picture=blueprint_picture or str(matched.picture or ""),
            segment_picture=str(matched.picture or ""),
            hint_expects_late_jump=True,
        )
        if not _picture_has_jump_cues(matched.picture):
            matched.picture = "夜色楼顶，胡小跃站在边缘，纵身跃下"
    return matched


def _match_from_frame_segments(
    entries: list,
    segments: list[dict[str, Any]],
    quotes: list[str],
    *,
    hint_expects_late: bool = False,
) -> OpeningClimaxMatch | None:
    if not segments:
        return None
    ranked = sorted(
        segments,
        key=lambda segment: _score_segment_for_opening(segment, hint_expects_late=hint_expects_late),
        reverse=True,
    )
    for segment in ranked[:8]:
        score = _score_segment_for_opening(segment, hint_expects_late=hint_expects_late)
        if score < 1.0:
            continue
        time_range = resolve_segment_time_range(segment)
        if not time_range or "-" not in time_range:
            continue
        start_ms, end_ms = parse_timestamp_range_ms(time_range)
        if hint_expects_late and start_ms < 10 * 60 * 1000 and not _segment_has_jump_cues(segment):
            continue
        if quotes:
            matched = _best_quote_in_window(
                entries,
                _expand_quote_variants(quotes),
                window_start_ms=max(0, start_ms - 3000),
                window_end_ms=end_ms + 3000,
            )
            if matched:
                matched.picture = _pick_picture_from_segment(segment)
                matched.source = "frame_segment"
                return matched
        window_entries = [
            entry
            for entry in entries
            if not (entry.end_ms < start_ms or entry.start_ms > end_ms)
        ]
        if not window_entries:
            continue
        pick = max(window_entries, key=lambda entry: len((entry.text or "").strip()))
        quote = (pick.text or "").strip().replace("\n", " ")
        if not quote:
            continue
        span = _find_quote_span(entries, quote, near_start_ms=pick.start_ms)
        if not span:
            span = (pick.start_ms, pick.end_ms)
        return OpeningClimaxMatch(
            timestamp=_format_range(span[0], span[1]),
            original_line=f"「{quote}」",
            picture=_pick_picture_from_segment(segment),
            source="frame_segment",
            quote=quote,
        )
    return None


def _match_from_quotes_only(
    entries: list,
    quotes: list[str],
    *,
    prefer_late: bool = False,
) -> OpeningClimaxMatch | None:
    matches = _collect_quote_matches(entries, quotes, prefer_late=prefer_late)
    return matches[0] if matches else None


def _score_quote(quote: str, start_ms: int, *, prefer_late: bool = False) -> float:
    score = 0.0
    for keyword in _HIGH_ENERGY_KEYWORDS:
        if keyword in quote:
            score += 2.0
    for marker in _LOW_ENERGY_EARLY_MARKERS:
        if marker in quote:
            score -= 4.0
    if prefer_late or "天就快亮" in quote or "天快亮" in quote:
        if start_ms >= 15 * 60 * 1000:
            score += 8.0
        elif start_ms >= 10 * 60 * 1000:
            score += 4.0
        elif start_ms < 3 * 60 * 1000:
            score -= 8.0
    elif start_ms >= 10 * 60 * 1000:
        score += 2.0
    elif start_ms < 3 * 60 * 1000:
        score -= 1.0
    return score


def resolve_opening_climax(
    subtitle_content: str,
    *,
    subtitle_frame_analysis: str = "",
    append_custom_prompt: str = "",
    opening_hint: str = "",
    frame_analysis_path: str = "",
    settings: Optional[dict[str, Any]] = None,
) -> OpeningClimaxMatch | None:
    """按优先级从字幕/对照分析/抽帧解析开篇爆燃段。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return None

    append_text = resolve_append_custom_prompt(append_custom_prompt, settings)
    hint_text = (opening_hint or resolve_fazu2_opening_climax_hint(settings)).strip()
    analysis_info = parse_episode_blueprint(subtitle_frame_analysis)
    plot_expects_late = bool(analysis_info.get("plot_expects_late_climax"))

    quotes: list[str] = []
    for quote in analysis_info.get("quotes") or []:
        if quote not in quotes:
            quotes.append(quote)
    for source in (append_text, hint_text, analysis_info.get("section", "")):
        for quote in _extract_quoted_strings(str(source or "")):
            if quote not in quotes:
                quotes.append(quote)

    timestamp_ranges = list(analysis_info.get("timestamp_ranges") or [])
    for source in (append_text, hint_text):
        for item in _extract_timestamp_ranges(str(source or "")):
            if item not in timestamp_ranges:
                timestamp_ranges.append(item)

    segments: list[dict[str, Any]] = []
    analysis_path = (frame_analysis_path or "").strip()
    if analysis_path and os.path.isfile(analysis_path):
        try:
            artifact = load_analysis_artifact(analysis_path)
            segments = _collect_top_level_segments(artifact)
        except Exception as exc:
            logger.warning(f"读取抽帧 JSON 用于开篇爆燃解析失败: {exc}")

    hint_expects_late = _hint_expects_late_jump(hint_text) or plot_expects_late
    resolvers: list[Any] = []

    if analysis_info.get("section") or analysis_info.get("ost1_opening_items"):
        resolvers.append(
            lambda: _match_from_episode_blueprint(
                entries,
                analysis_info,
                settings=settings,
            )
        )
    if timestamp_ranges:
        resolvers.append(
            lambda: _match_from_timestamp_ranges(entries, timestamp_ranges, quotes)
        )
    if quotes:
        resolvers.append(
            lambda: _match_from_quotes_only(
                entries,
                quotes,
                prefer_late=hint_expects_late,
            )
        )
    if hint_expects_late and segments:
        resolvers.append(
            lambda: _match_from_late_sacrifice_segments(
                entries,
                segments,
                quotes,
                hint_expects_late=True,
                settings=settings,
            )
        )
    if segments:
        resolvers.append(
            lambda: _match_from_frame_segments(
                entries,
                segments,
                quotes,
                hint_expects_late=hint_expects_late,
            )
        )
    for resolver in resolvers:
        matched = resolver()
        if matched and matched.timestamp:
            matched = _finalize_opening_match(
                entries,
                segments,
                matched,
                hint_text=hint_text,
                settings=settings,
            )
            logger.info(
                f"开篇爆燃段已解析（{matched.source}）: "
                f"{matched.timestamp} {matched.original_line or matched.quote}"
            )
            return matched
    return None


def _first_item_needs_opening_fix(
    first: dict[str, Any],
    resolved: OpeningClimaxMatch,
    *,
    hint_text: str = "",
) -> bool:
    if int(first.get("OST", 0)) != 1:
        return True

    narration = str(first.get("narration") or "")
    if "宝子们" in narration:
        return True

    combined = " ".join(
        [
            narration,
            str(first.get("original_line") or ""),
            str(first.get("picture") or ""),
        ]
    )
    if any(marker in combined for marker in _LOW_ENERGY_EARLY_MARKERS):
        return True

    if _hint_expects_late_jump(hint_text):
        try:
            current_start, _ = parse_timestamp_range(str(first.get("timestamp") or ""))
            if current_start < 2 * 60 * 1000:
                return True
        except Exception:
            return True
        current_picture = str(first.get("picture") or "")
        if _picture_has_phone_cues(current_picture) and not _picture_has_jump_cues(current_picture):
            return True
        if resolved.picture and _picture_has_jump_cues(resolved.picture):
            if current_picture != resolved.picture and not _picture_has_jump_cues(current_picture):
                return True

    if resolved.original_line:
        current_key = dialogue_match_key(
            str(first.get("original_line") or first.get("narration") or "")
        )
        target_key = dialogue_match_key(resolved.original_line)
        if current_key and target_key and current_key != target_key:
            try:
                current_start, _ = parse_timestamp_range(str(first.get("timestamp") or ""))
                target_start, _ = parse_timestamp_range(resolved.timestamp)
            except Exception:
                return True
            if abs(current_start - target_start) > 5000:
                return True

    try:
        current_start, _ = parse_timestamp_range(str(first.get("timestamp") or ""))
        target_start, _ = parse_timestamp_range(resolved.timestamp)
    except Exception:
        return False

    if target_start >= 8 * 60 * 1000 and current_start < 3 * 60 * 1000:
        return True
    if abs(current_start - target_start) > 120_000:
        return True
    return False


def apply_opening_climax_fix(
    items: list[dict[str, Any]],
    *,
    subtitle_content: str = "",
    subtitle_frame_analysis: str = "",
    append_custom_prompt: str = "",
    opening_hint: str = "",
    frame_analysis_path: str = "",
    settings: Optional[dict[str, Any]] = None,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """将第 1 段校正为解析出的开篇爆燃 OST=1 原声。"""
    if not enabled or not items:
        return items

    resolved = resolve_opening_climax(
        subtitle_content,
        subtitle_frame_analysis=subtitle_frame_analysis,
        append_custom_prompt=append_custom_prompt,
        opening_hint=opening_hint,
        frame_analysis_path=frame_analysis_path,
        settings=settings,
    )
    if not resolved:
        return items

    hint_text = (opening_hint or resolve_fazu2_opening_climax_hint(settings)).strip()
    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    first = ordered[0]
    if not _first_item_needs_opening_fix(first, resolved, hint_text=hint_text):
        return items

    updated = dict(first)
    updated["OST"] = 1
    first_id = int(updated.get("_id") or 1)
    if settings and is_compact_documentary_settings(settings):
        updated["narration"] = "播放原片"
    else:
        updated["narration"] = f"播放原片{first_id}"
    updated["timestamp"] = resolved.timestamp
    if resolved.original_line:
        line = resolved.original_line.strip()
        if not line.startswith("「"):
            line = f"「{line.strip('「」')}」"
        updated["original_line"] = line
    blueprint_picture = str(resolved.picture or "").strip()
    if blueprint_picture:
        updated["picture"] = blueprint_picture
    elif _hint_expects_late_jump(hint_text):
        updated["picture"] = "夜色楼顶，胡小跃站在边缘，纵身跃下"
    elif not str(updated.get("picture") or "").strip():
        updated["picture"] = "开篇名场面，高能冲突瞬间"
    elif _hint_expects_late_jump(hint_text) and _picture_has_phone_cues(str(updated.get("picture") or "")):
        updated["picture"] = "夜色楼顶，胡小跃站在边缘，纵身跃下"

    ordered[0] = updated
    logger.info(
        f"已校正第 1 段爆燃原声: {resolved.timestamp} "
        f"({resolved.source})，替换原 timestamp={first.get('timestamp')!r}"
    )
    return ordered


_REPLAY_PICTURE_PREFIX = "【复现】"
_OPENING_REPLAY_PRE_WINDOW_MS = 30_000
_OPENING_REPLAY_START_TOLERANCE_MS = 5_000
_OPENING_REPLAY_TIMESTAMP_TOLERANCE_MS = 2_000


def _opening_timestamps_match(
    left: str,
    right: str,
    *,
    tolerance_ms: int = _OPENING_REPLAY_TIMESTAMP_TOLERANCE_MS,
) -> bool:
    try:
        left_start, left_end = parse_timestamp_range(left)
        right_start, right_end = parse_timestamp_range(right)
    except Exception:
        return False
    return (
        abs(left_start - right_start) <= tolerance_ms
        and abs(left_end - right_end) <= tolerance_ms
    )


def _opening_replay_already_present(
    items: list[dict[str, Any]],
    opening_timestamp: str,
) -> bool:
    for item in items[1:]:
        if int(item.get("OST", 0)) != 1:
            continue
        if _opening_timestamps_match(
            str(item.get("timestamp") or ""),
            opening_timestamp,
        ):
            return True
    return False


def _find_chronological_replay_insert_index(
    items: list[dict[str, Any]],
    opening_start_ms: int,
) -> int:
    """在播放顺序中定位「正叙走到开篇高潮原片时刻」的插入点。"""
    threshold_ms = opening_start_ms - _OPENING_REPLAY_START_TOLERANCE_MS
    pre_window_ms = opening_start_ms - _OPENING_REPLAY_PRE_WINDOW_MS

    for idx in range(1, len(items)):
        timestamp = str(items[idx].get("timestamp") or "")
        if not timestamp:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range(timestamp)
        except Exception:
            continue

        if start_ms >= threshold_ms:
            return idx

        if (
            start_ms >= pre_window_ms
            and end_ms >= opening_start_ms - 60_000
            and end_ms <= opening_start_ms + _OPENING_REPLAY_START_TOLERANCE_MS
        ):
            return idx + 1

    return max(1, len(items) - 1)


def _build_opening_climax_replay_item(opening: dict[str, Any]) -> dict[str, Any]:
    replay = {
        key: value
        for key, value in opening.items()
        if not str(key).startswith("_")
    }
    picture = str(replay.get("picture") or "").strip().strip('"')
    if picture and not picture.startswith(_REPLAY_PICTURE_PREFIX):
        replay["picture"] = f"{_REPLAY_PICTURE_PREFIX}{picture}"
    replay["OST"] = 1
    narration = str(replay.get("narration") or "").strip()
    if not narration or narration == "播放原片":
        replay["narration"] = "播放原片"
    replay["_opening_climax_replay"] = True
    return replay


def _renumber_script_item_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, item in enumerate(items, start=1):
        item["_id"] = index
    return items


def apply_opening_climax_chronological_replay(
    items: list[dict[str, Any]],
    *,
    settings: Optional[dict[str, Any]] = None,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """正叙时间线走到第 1 段开篇高潮时，再插入同一片段的 OST=1 复现。"""
    if not enabled or len(items) < 2:
        return items

    cfg = settings or {}
    if not cfg.get("enable_opening_climax_chronological_replay", True):
        return items

    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    opening = ordered[0]
    if int(opening.get("OST", 0)) != 1:
        return items

    opening_timestamp = str(opening.get("timestamp") or "").strip()
    if not opening_timestamp:
        return items

    if _opening_replay_already_present(ordered, opening_timestamp):
        logger.info("开篇高潮已在正叙相应位置复现，跳过插入")
        return ordered

    try:
        opening_start_ms, _ = parse_timestamp_range(opening_timestamp)
    except Exception:
        return ordered

    insert_idx = _find_chronological_replay_insert_index(ordered, opening_start_ms)
    replay = _build_opening_climax_replay_item(opening)
    updated = ordered[:insert_idx] + [replay] + ordered[insert_idx:]
    updated = _renumber_script_item_ids(updated)
    logger.info(
        f"已在正叙位置插入开篇高潮复现（原 timestamp={opening_timestamp!r}，"
        f"插入于播放顺序第 {insert_idx + 1} 段前）"
    )
    return updated
