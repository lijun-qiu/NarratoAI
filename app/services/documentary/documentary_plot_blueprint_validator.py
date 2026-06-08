#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧/短剧「完美剧情构思方案」输出校验（时间戳边界、OST=1 时长等）。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import FAZU2_CHARACTER_ROLES
from app.services.short_drama_drama_knowledge import find_name_mistakes_in_text
from app.services.short_drama_plot_analysis_validator import (
    _CLIP_TS_RANGE_RE,
    _REQUIRED_SECTIONS,
    _SRT_ARROW_TS_RE,
    _SCENE_INDEX_RE,
    _clip_ts_duration_sec,
    _collect_ost1_section_timestamp_durations,
    _count_ost1_entries,
    _estimate_min_ost1_entries,
    format_plot_analysis_validation_report,
)

_SINGLE_TS_RE = re.compile(r"(?<![\d,:])(\d{2}:\d{2}:\d{2}[,.]\d{3})(?![\d,:])")
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
    settings: dict[str, Any] | None = None,
    lexicon: dict | None = None,
    drama_known_names: set[str] | None = None,
    min_chars: int = 2000,
    require_all_sections: bool = True,
) -> dict[str, Any]:
    """校验剧情构思 Markdown；返回 ok / issues / warnings。"""
    content = (text or "").strip()
    issues: list[str] = []
    warnings: list[str] = []

    if len(content) < min_chars:
        issues.append(f"篇幅过短（{len(content)} 字），要求不少于 {min_chars} 字")

    if require_all_sections:
        for header in _REQUIRED_SECTIONS:
            if header not in content:
                issues.append(f"缺少必填章节：{header}")

    scene_hits = _SCENE_INDEX_RE.findall(content)
    if scene_hits:
        issues.append(
            "禁止引用抽帧采样编号「场景 N」（如 "
            + "、".join(sorted(set(scene_hits))[:5])
            + "），请改用原片时间段+地点描述"
        )

    if _SRT_ARROW_TS_RE.search(content):
        issues.append(
            "时间戳须用剪辑格式 HH:MM:SS,mmm-HH:MM:SS,mmm，禁止 SRT 箭头 `-->`"
        )

    hard_cap_ms = frame_max_ms
    if source_duration_ms and source_duration_ms > 0:
        hard_cap_ms = (
            min(source_duration_ms, frame_max_ms)
            if frame_max_ms and frame_max_ms > 0
            else source_duration_ms
        )
    elif not hard_cap_ms or hard_cap_ms <= 0:
        hard_cap_ms = None

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
                    f"时间戳起点 {start} 早于抽帧覆盖范围（"
                    f"最早 {_format_ms_label(frame_min_ms)}）"
                )
            if end_ms > hard_cap_ms + 500:
                issues.append(
                    f"时间戳 {start}-{end} 超出原片/抽帧上限 {cap_label}"
                )
            if end_ms <= start_ms:
                issues.append(f"时间戳区间无效（结束≤开始）：{start}-{end}")

        for single in _SINGLE_TS_RE.findall(content):
            try:
                point_ms = _ts_to_ms(single)
            except Exception:
                continue
            if point_ms > hard_cap_ms + 500:
                issues.append(
                    f"时间点 {single} 超出原片/抽帧上限 {cap_label}"
                )

    cfg = settings or {}
    ost1_dur_min = float(cfg.get("ost1_duration_min", 8) or 8)
    ost1_dur_max = float(cfg.get("ost1_duration_max", 18) or 18)
    ts_durations = _collect_ost1_section_timestamp_durations(content)
    short_ts = [round(sec, 1) for sec in ts_durations if 0 < sec < ost1_dur_min]
    long_ts = [round(sec, 1) for sec in ts_durations if sec > ost1_dur_max + 2]
    if short_ts:
        issues.append(
            f"蓝图 OST=1 时间戳过短（{len(short_ts)} 处 < {ost1_dur_min:.0f}s，"
            f"如 {short_ts[:4]}），须合并相邻 subtitle_entries 为 "
            f"{ost1_dur_min:.0f}–{ost1_dur_max:.0f}s 连续对白块"
        )
    if long_ts:
        warnings.append(
            f"蓝图 OST=1 时间戳偏长（{len(long_ts)} 处 > {ost1_dur_max:.0f}s）"
        )

    min_ost1 = _estimate_min_ost1_entries(cfg)
    ost1_count = _count_ost1_entries(content)
    if ost1_count < min_ost1:
        issues.append(
            f"建议保留原声 OST=1 条目过少（约 {ost1_count} 条），"
            f"建议至少 {min_ost1} 条（含完整时间戳区间）"
        )

    for mistake in find_name_mistakes_in_text(content):
        issues.append(mistake)

    if _FEMALE_PRONOUN_RE.search(content) or _MALE_ROLE_AS_FEMALE_RE.search(content):
        issues.append(
            "胡小跃/小跃在本项目抽帧与对照表中为**男性刑警**，"
            "禁止写成女性/「她」；字幕「小月/胡小月」须归并为胡小跃(男)"
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
                "开头高潮未选用默认跳楼 sacrifice，若抽帧中无跳楼场面须在「声画对位注意」说明依据"
            )

    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "ost1_count": ost1_count,
        "min_ost1_expected": min_ost1,
        "char_count": len(content),
        "hard_cap_ms": hard_cap_ms,
        "known_names": known_names | drama_names,
    }


def emit_plot_blueprint_validation_report(validation: dict[str, Any]) -> None:
    logger.info(format_plot_analysis_validation_report(validation))
