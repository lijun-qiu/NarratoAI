#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧/短剧「完美剧情构思方案」输出校验（时间戳边界、OST=1 时长等）。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import FAZU2_CHARACTER_ROLES
from app.services.documentary.video_episode_segment_schedule import segment_policy_summary
from app.services.short_drama_drama_knowledge import find_name_mistakes_in_text
from app.services.short_drama_plot_analysis_validator import (
    _CLIP_TS_RANGE_RE,
    _REQUIRED_SECTIONS_LEGACY,
    _SRT_ARROW_TS_RE,
    _SCENE_INDEX_RE,
    _clip_ts_duration_sec,
    _collect_ost1_section_timestamp_durations,
    _count_ost1_entries,
    _count_scene_segments,
    _estimate_min_ost1_entries,
    _has_scene_section,
    format_plot_analysis_validation_report,
)

_SINGLE_TS_RE = re.compile(r"(?<![\d,:])(\d{2}:\d{2}:\d{2}[,.]\d{3})(?![\d,:])")
_VIDEO_GRID_RE = re.compile(
    r"(?<![\d,:])(\d{2}:\d{2}:\d{2})-(?!\d{2}:\d{2}:\d{2},\d{3})(\d{2}:\d{2}:\d{2})(?![\d,:])"
)
_FEMALE_PRONOUN_RE = re.compile(r"(胡小跃|小跃)[^。\n]{0,24}(她|女警|女性)")
_MALE_ROLE_AS_FEMALE_RE = re.compile(
    r"胡小跃[^。\n]{0,12}(?:\(女\)|（女）|，女，|，女\)|性别[：:]\s*女)"
)


def _extract_section(text: str, header: str) -> str:
    start = text.find(header)
    if start < 0:
        return ""
    rest = text[start + len(header) :]
    next_header = re.search(r"\n##\s+", rest)
    if next_header:
        return rest[: next_header.start()]
    return rest


def _ts_to_ms(text: str) -> int:
    from app.services.srt_utils import parse_timestamp_range

    label = str(text or "").strip().replace(".", ",")
    if "-" in label:
        start_ms, _ = parse_timestamp_range(label)
        return start_ms
    start_ms, _ = parse_timestamp_range(f"{label}-{label}")
    return start_ms


def _format_ms_label(ms: int) -> str:
    from app.services.srt_utils import format_timestamp_ms

    return format_timestamp_ms(max(0, int(ms)))


def _normalize_video_grid_range(value: str) -> str:
    return str(value or "").strip().replace("—", "-").replace(",", "")


def _clip_overlaps_srt(
    start_ms: int,
    end_ms: int,
    srt_entries: list[dict[str, Any]],
    *,
    min_overlap_ms: int = 400,
) -> bool:
    for entry in srt_entries:
        try:
            entry_start = int(entry.get("start_ms") or 0)
            entry_end = int(entry.get("end_ms") or 0)
        except (TypeError, ValueError):
            continue
        overlap = min(end_ms, entry_end) - max(start_ms, entry_start)
        if overlap >= min_overlap_ms:
            return True
    return False


def _count_timeline_table_rows(timeline: str) -> int:
    count = 0
    for line in timeline.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        if re.match(r"^\|[-:\s|]+\|$", stripped):
            continue
        if "视频格" in stripped or "---" in stripped:
            continue
        count += 1
    return count


def _validate_timeline_granularity(
    content: str,
    issues: list[str],
    *,
    ost1_dur_min: float,
    max_timeline_rows: int = 16,
) -> None:
    timeline = _extract_section(content, "## 原片时间线")
    if not timeline:
        return
    row_count = _count_timeline_table_rows(timeline)
    bullet_rows = len(
        re.findall(r"(?m)^\s*[-*]\s+.*视频格", timeline)
    )
    total_rows = max(row_count, bullet_rows)
    if total_rows > max_timeline_rows:
        issues.append(
            f"「原片时间线」条目过多（约 {total_rows} 条 > {max_timeline_rows}），"
            f"应按**完整情节段**合并，禁止按每个 {segment_policy_summary()} 或每条字幕各写一行"
        )
    short_windows: list[str] = []
    for line in timeline.splitlines():
        if "OST=0" in line and "OST=1" not in line:
            continue
        for start, end in _CLIP_TS_RANGE_RE.findall(line):
            duration = _clip_ts_duration_sec(start, end)
            if 0 < duration < ost1_dur_min:
                short_windows.append(f"{start}-{end}({duration:.1f}s)")
    if short_windows:
        issues.append(
            f"「原片时间线」含过短字幕窗 {len(short_windows)} 处（< {ost1_dur_min:.0f}s，"
            f"如 {short_windows[:3]}），须合并同场连续对白为完整段落"
        )


def _validate_video_grid_citations(
    content: str,
    video_segment_ranges: list[str],
    issues: list[str],
    *,
    min_timeline_hits: int = 8,
) -> None:
    from app.services.documentary.video_episode_analysis import is_video_grid_span_allowed

    if not video_segment_ranges:
        return

    timeline = _extract_section(content, "## 原片时间线")
    if not timeline:
        issues.append("缺少「原片时间线」章节，无法校验视频格对齐")
        return

    hits = 0
    unknown: list[str] = []
    for start, end in _VIDEO_GRID_RE.findall(timeline):
        label = _normalize_video_grid_range(f"{start}-{end}")
        if is_video_grid_span_allowed(label, video_segment_ranges):
            hits += 1
        elif label not in unknown:
            unknown.append(label)

    if hits < min_timeline_hits:
        issues.append(
            f"「原片时间线」中合法视频格引用仅 {hits} 处，要求至少 {min_timeline_hits} 处"
            "（须对齐视频分析索引表，可跨连续多格）"
        )
    if unknown:
        issues.append(
            "「原片时间线」含未对齐视频分析索引表的视频格："
            + "、".join(f"`{item}`" for item in unknown[:5])
        )


def _validate_ost1_against_srt(
    content: str,
    srt_entries: list[dict[str, Any]],
    issues: list[str],
) -> None:
    if not srt_entries:
        return
    for header in (
        "## 建议保留原声 OST=1",
        "## OST=1 金句清单",
        "## 开头高潮方案",
    ):
        section = _extract_section(content, header)
        if not section:
            continue
        for start, end in _CLIP_TS_RANGE_RE.findall(section):
            try:
                start_ms = _ts_to_ms(start)
                end_ms = _ts_to_ms(end)
            except Exception:
                continue
            if not _clip_overlaps_srt(start_ms, end_ms, srt_entries):
                issues.append(
                    f"OST=1 时间戳 {start}-{end} 未与 SRT 对白时间窗重叠，"
                    "须从字幕索引选取连续对白区间"
                )


def collect_all_clip_ranges(text: str) -> list[tuple[str, str]]:
    ranges = list(_CLIP_TS_RANGE_RE.findall(text or ""))
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for start, end in ranges:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def validate_plot_blueprint(
    text: str,
    *,
    source_duration_ms: int | None = None,
    frame_max_ms: int | None = None,
    frame_min_ms: int = 0,
    srt_max_ms: int | None = None,
    srt_min_ms: int = 0,
    settings: dict[str, Any] | None = None,
    lexicon: dict | None = None,
    drama_known_names: set[str] | None = None,
    min_chars: int = 2000,
    require_all_sections: bool = True,
    video_segment_ranges: list[str] | None = None,
    srt_entries: list[dict[str, Any]] | None = None,
    use_video_episode_analysis: bool = False,
    relaxed: bool = True,
) -> dict[str, Any]:
    """校验剧情构思 Markdown；返回 ok / issues / warnings。"""
    content = (text or "").strip()
    issues: list[str] = []
    warnings: list[str] = []
    min_chars_required = max(1200, int(min_chars * 0.55)) if relaxed else min_chars

    if len(content) < min_chars_required:
        issues.append(f"篇幅过短（{len(content)} 字），要求不少于 {min_chars_required} 字")

    if relaxed:
        if "## 主要人物表" not in content:
            issues.append("缺少必填章节：## 主要人物表")
        if not _has_scene_section(content):
            issues.append("缺少必填章节：## 全片场景分段（或 ## 原片时间线）")
    elif require_all_sections:
        for header in _REQUIRED_SECTIONS_LEGACY:
            if header not in content:
                issues.append(f"缺少必填章节：{header}")

    scene_count = _count_scene_segments(content)
    if relaxed and scene_count and scene_count < 6:
        warnings.append(f"场景分段偏少（约 {scene_count} 段），建议按完整情节场切分 10 段以上")

    scene_hits = _SCENE_INDEX_RE.findall(content)
    if scene_hits and not relaxed:
        issues.append(
            "禁止引用抽帧采样编号「场景 N」（如 "
            + "、".join(sorted(set(scene_hits))[:5])
            + "），请改用原片时间段+地点描述"
        )

    if _SRT_ARROW_TS_RE.search(content):
        warnings.append("时间戳建议用剪辑格式 HH:MM:SS,mmm-HH:MM:SS,mmm，避免 SRT 箭头 `-->`")

    hard_cap_ms = frame_max_ms
    if source_duration_ms and source_duration_ms > 0:
        hard_cap_ms = (
            min(source_duration_ms, frame_max_ms)
            if frame_max_ms and frame_max_ms > 0
            else source_duration_ms
        )
    elif not hard_cap_ms or hard_cap_ms <= 0:
        hard_cap_ms = None

    dialogue_cap_ms = hard_cap_ms
    if srt_max_ms and srt_max_ms > 0:
        dialogue_cap_ms = (
            min(srt_max_ms, hard_cap_ms)
            if hard_cap_ms and hard_cap_ms > 0
            else srt_max_ms
        )

    visual_source_label = "整片视频分析" if use_video_episode_analysis else "抽帧"
    time_cap_label = "原片/视频分析" if use_video_episode_analysis else "原片/抽帧"

    if hard_cap_ms:
        cap_label = _format_ms_label(hard_cap_ms)
        for start, end in collect_all_clip_ranges(content):
            try:
                start_ms = _ts_to_ms(start)
                end_ms = _ts_to_ms(end)
            except Exception:
                issues.append(f"无法解析时间戳：{start}-{end}")
                continue
            if start_ms < max(0, frame_min_ms - 500):
                issues.append(
                    f"时间戳起点 {start} 早于{visual_source_label}覆盖范围（"
                    f"最早 {_format_ms_label(frame_min_ms)}）"
                )
            if end_ms > hard_cap_ms + 500:
                msg = f"时间戳 {start}-{end} 超出{time_cap_label}上限 {cap_label}"
                (warnings if relaxed else issues).append(msg)
            if end_ms <= start_ms:
                issues.append(f"时间戳区间无效（结束≤开始）：{start}-{end}")

        for single in _SINGLE_TS_RE.findall(content):
            try:
                point_ms = _ts_to_ms(single)
            except Exception:
                continue
            if point_ms > hard_cap_ms + 500:
                issues.append(
                    f"时间点 {single} 超出{time_cap_label}上限 {cap_label}"
                )

    if dialogue_cap_ms and srt_max_ms and srt_max_ms > 0 and not relaxed:
        srt_cap_label = _format_ms_label(dialogue_cap_ms)
        for header in (
            "## 建议保留原声 OST=1",
            "## OST=1 金句清单",
            "## 开头高潮方案",
        ):
            section = _extract_section(content, header)
            for start, end in _CLIP_TS_RANGE_RE.findall(section):
                try:
                    end_ms = _ts_to_ms(end)
                except Exception:
                    continue
                if end_ms > dialogue_cap_ms + 500:
                    issues.append(
                        f"对白时间戳 {start}-{end} 超出 SRT 上限 {srt_cap_label}"
                    )
        narrative = _extract_section(content, "## 成片叙事顺序方案")
        for start, end in _CLIP_TS_RANGE_RE.findall(narrative):
            block_start = max(0, narrative.find(start))
            block = narrative[block_start : block_start + 240]
            if "OST=1" not in block and "OST = 1" not in block:
                continue
            try:
                end_ms = _ts_to_ms(end)
            except Exception:
                continue
            if end_ms > dialogue_cap_ms + 500:
                issues.append(
                    f"成片 OST=1 时间戳 {start}-{end} 超出 SRT 上限 {srt_cap_label}"
                )

    cfg = settings or {}
    ost1_dur_min = float(cfg.get("ost1_duration_min", 8) or 8)
    ost1_dur_max = float(cfg.get("ost1_duration_max", 18) or 18)
    ts_durations = _collect_ost1_section_timestamp_durations(content)
    short_ts = [round(sec, 1) for sec in ts_durations if 0 < sec < ost1_dur_min]
    long_ts = [round(sec, 1) for sec in ts_durations if sec > ost1_dur_max + 2]
    if short_ts:
        msg = (
            f"部分 OST=1 时间戳较短（{len(short_ts)} 处 < {ost1_dur_min:.0f}s，"
            f"如 {short_ts[:4]}），写脚本时可酌情合并同场对白"
        )
        (warnings if relaxed else issues).append(msg)
    if long_ts:
        warnings.append(
            f"部分 OST=1 时间戳偏长（{len(long_ts)} 处 > {ost1_dur_max:.0f}s）"
        )

    min_ost1 = _estimate_min_ost1_entries(cfg)
    ost1_count = _count_ost1_entries(content)
    if ost1_count < min_ost1:
        msg = (
            f"建议保留原声 OST=1 条目偏少（约 {ost1_count} 条），"
            f"后续写脚本时可参考 ≥{min_ost1} 条"
        )
        (warnings if relaxed else issues).append(msg)

    if not relaxed:
        _validate_timeline_granularity(
            content,
            issues,
            ost1_dur_min=ost1_dur_min,
        )
    if video_segment_ranges and not relaxed:
        _validate_video_grid_citations(content, video_segment_ranges, issues)
    if srt_entries and not relaxed:
        _validate_ost1_against_srt(content, srt_entries, issues)

    for mistake in find_name_mistakes_in_text(content):
        issues.append(mistake)

    if _FEMALE_PRONOUN_RE.search(content) or _MALE_ROLE_AS_FEMALE_RE.search(content):
        issues.append(
            "胡小跃/小跃在本项目视频分析与对照表中为**男性刑警**，"
            "禁止写成女性/「她」；字幕「小月/胡小月/胡晓月」须归并为胡小跃(男)"
        )

    for name, role, note in FAZU2_CHARACTER_ROLES:
        if "（男）" in role and f"{name}（女）" in content:
            issues.append(f"人物 {name} 性别与对照表不符（应为男）：{note}")

    lex = lexicon or {}
    known_names = {str(x) for x in lex.get("names") or set()}
    drama_names = {str(x) for x in (drama_known_names or set()) if str(x).strip()}

    opening_section = _extract_section(content, "## 开头高潮方案")
    if opening_section and "跳楼" not in opening_section and "纵身" not in opening_section:
        if "天台" in opening_section and "绝路" in opening_section:
            warnings.append(
                "开头高潮未选用默认跳楼 sacrifice，若视频分析中无跳楼场面须在「声画对位注意」说明依据"
            )

    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "ost1_count": ost1_count,
        "min_ost1_expected": min_ost1,
        "scene_count": scene_count,
        "char_count": len(content),
        "hard_cap_ms": hard_cap_ms,
        "known_names": known_names | drama_names,
    }


def emit_plot_blueprint_validation_report(validation: dict[str, Any]) -> None:
    logger.info(format_plot_analysis_validation_report(validation))
