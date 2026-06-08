#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧「字幕×抽帧联合剧情构思」输出校验。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.short_drama_drama_knowledge import find_name_mistakes_in_text
from app.services.short_drama_settings import get_short_drama_settings

_REQUIRED_SECTIONS = (
    "## 主要人物表",
    "## 开头高潮方案",
    "## 原片时间线",
    "## 成片叙事顺序方案",
    "## 建议保留原声 OST=1",
    "## 解说 OST=0 脉络规划",
    "## 声画对位注意",
)

_SCENE_INDEX_RE = re.compile(r"场景\s*\d+")
_SRT_ARROW_TS_RE = re.compile(
    r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}"
)
_INVALID_OST_RE = re.compile(r"OST\s*=\s*[^01\]\s]")
_OST1_ENTRY_RE = re.compile(
    r"^\s*(?:\d+\.|[-*])\s*(?:\*\*)?(?:说话人|台词|OST)",
    re.MULTILINE,
)
_PERSON_TABLE_NAME_RE = re.compile(
    r"^[-*]\s*\**\s*([\u4e00-\u9fffA-Za-z·]{2,8})",
    re.MULTILINE,
)
_OST1_SPEAKER_RE = re.compile(
    r"说话人[：:]\s*([\u4e00-\u9fffA-Za-z·]{2,8})"
)
_CLIP_TS_RANGE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*[-–—]\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def _clip_ts_duration_sec(start: str, end: str) -> float:
    from app.services.srt_utils import parse_timestamp_range

    try:
        start_ms, end_ms = parse_timestamp_range(f"{start}-{end}")
        return max(0, end_ms - start_ms) / 1000.0
    except Exception:
        return 0.0


def _collect_ost1_section_timestamp_durations(text: str) -> list[float]:
    durations: list[float] = []
    for header in (
        "## 建议保留原声 OST=1",
        "## OST=1 金句清单",
        "## 成片叙事顺序方案",
        "## 开头高潮方案",
    ):
        section = _extract_section(text, header)
        if not section:
            continue
        for start, end in _CLIP_TS_RANGE_RE.findall(section):
            if header == "## 成片叙事顺序方案":
                block_start = max(0, section.find(f"{start}"))
                block = section[block_start : block_start + 240]
                if "OST=1" not in block and "OST = 1" not in block:
                    continue
            durations.append(_clip_ts_duration_sec(start, end))
    return durations


def _count_ost1_entries(text: str) -> int:
    section = _extract_section(text, "## 建议保留原声 OST=1")
    if not section:
        return 0
    timestamps = re.findall(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*[-–—]\s*\d{2}:\d{2}:\d{2}[,.]\d{3}",
        section,
    )
    if timestamps:
        return len(timestamps)
    return len(_OST1_ENTRY_RE.findall(section))


def _extract_section(text: str, header: str) -> str:
    start = text.find(header)
    if start < 0:
        return ""
    rest = text[start + len(header) :]
    next_header = re.search(r"\n##\s+", rest)
    if next_header:
        return rest[: next_header.start()]
    return rest


def _estimate_min_ost1_entries(settings: dict[str, Any] | None) -> int:
    cfg = settings or get_short_drama_settings()
    min_minutes = float(cfg.get("target_output_minutes_min", 8) or 8)
    orig_pct = float(cfg.get("original_audio_percent", 70) or 70) / 100.0
    ost1_max = float(cfg.get("ost1_duration_max", 18) or 18)
    target_orig_sec = min_minutes * 60 * orig_pct
    return max(10, int(target_orig_sec / ost1_max * 0.6))


def estimate_min_ost1_entries_for_plot(settings: dict[str, Any] | None = None) -> int:
    return _estimate_min_ost1_entries(settings)


def _collect_referenced_names(text: str) -> set[str]:
    names: set[str] = set()
    person_section = _extract_section(text, "## 主要人物表")
    for match in _PERSON_TABLE_NAME_RE.finditer(person_section):
        names.add(match.group(1).strip())
    ost1_section = _extract_section(text, "## 建议保留原声 OST=1")
    for match in _OST1_SPEAKER_RE.finditer(ost1_section):
        names.add(match.group(1).strip())
    return {name for name in names if name}


def _name_in_lexicon(name: str, known_names: set[str], snippets: list[str]) -> bool:
    if not name:
        return True
    if name in known_names:
        return True
    if any(name in known for known in known_names):
        return True
    blob = "\n".join(snippets)
    return name in blob


def validate_short_drama_plot_analysis(
    text: str,
    *,
    lexicon: dict | None = None,
    drama_known_names: set[str] | None = None,
    settings: dict[str, Any] | None = None,
    min_chars: int = 2000,
) -> dict[str, Any]:
    """校验联合剧情构思 Markdown；返回 ok / issues / warnings。"""
    content = (text or "").strip()
    issues: list[str] = []
    warnings: list[str] = []

    if len(content) < min_chars:
        issues.append(f"篇幅过短（{len(content)} 字），要求不少于 {min_chars} 字")

    if content and not content.rstrip().endswith(("。", "！", "？", "」", "）", ".", "!", "?")):
        if len(content) > min_chars * 0.8:
            warnings.append("输出疑似被截断（末尾未完整收束），请补全 OST=1 与声画对位章节")

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

    invalid_ost = _INVALID_OST_RE.findall(content)
    if invalid_ost:
        issues.append("存在非法 OST 标记（仅允许 OST=0 或 OST=1）")

    min_ost1 = _estimate_min_ost1_entries(settings)
    ost1_count = _count_ost1_entries(content)
    if ost1_count < min_ost1:
        issues.append(
            f"建议保留原声 OST=1 条目过少（约 {ost1_count} 条），"
            f"8 分钟成片原声约 70% 时建议至少 {min_ost1} 条（含时间戳）"
        )

    cfg = settings or get_short_drama_settings()
    ost1_dur_min = float(cfg.get("ost1_duration_min", 8) or 8)
    ost1_dur_max = float(cfg.get("ost1_duration_max", 18) or 18)
    ts_durations = _collect_ost1_section_timestamp_durations(content)
    short_ts = [
        round(sec, 1)
        for sec in ts_durations
        if 0 < sec < ost1_dur_min
    ]
    long_ts = [
        round(sec, 1)
        for sec in ts_durations
        if sec > ost1_dur_max + 2
    ]
    if short_ts:
        issues.append(
            f"蓝图 OST=1 时间戳过短（{len(short_ts)} 处 < {ost1_dur_min:.0f}s，"
            f"如 {short_ts[:3]}），须合并为 {ost1_dur_min:.0f}–{ost1_dur_max:.0f}s 连续对白块"
        )
    if long_ts:
        warnings.append(
            f"蓝图 OST=1 时间戳偏长（{len(long_ts)} 处 > {ost1_dur_max:.0f}s），"
            f"建议拆分为多条或缩短至配置上限"
        )

    lex = lexicon or {}
    known_names = {str(x) for x in lex.get("names") or set()}
    snippets = [str(x) for x in lex.get("subtitle_snippets") or []]
    drama_names = {str(x) for x in (drama_known_names or set()) if str(x).strip()}

    for mistake in find_name_mistakes_in_text(content):
        issues.append(mistake)

    allowed_names = known_names | drama_names
    if allowed_names:
        for name in sorted(_collect_referenced_names(content)):
            if not _name_in_lexicon(name, allowed_names, snippets):
                issues.append(
                    f"人物「{name}」未出现在剧集对照表或抽帧字幕索引中，"
                    "请核对是否张冠李戴或臆造姓名"
                )
    elif known_names:
        for name in sorted(_collect_referenced_names(content)):
            if not _name_in_lexicon(name, known_names, snippets):
                issues.append(
                    f"人物「{name}」未出现在抽帧字幕索引中，"
                    "请核对是否张冠李戴或臆造姓名"
                )

    ok = not issues
    return {
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "ost1_count": ost1_count,
        "min_ost1_expected": min_ost1,
        "char_count": len(content),
    }


def format_plot_analysis_validation_report(validation: dict[str, Any]) -> str:
    status = "通过" if validation.get("ok") else "未达标"
    lines = [
        "",
        "-" * 72,
        f"【短剧联合构思校验】{status}",
        "-" * 72,
        f"字数: {validation.get('char_count', 0)}",
        f"OST=1 条目（含时间戳）: {validation.get('ost1_count', 0)} "
        f"/ 建议 ≥{validation.get('min_ost1_expected', 0)}",
    ]
    for issue in validation.get("issues") or []:
        lines.append(f"问题: {issue}")
    for warning in validation.get("warnings") or []:
        lines.append(f"提示: {warning}")
    lines.append("-" * 72)
    return "\n".join(lines)


def emit_plot_analysis_full_text(content: str, *, title: str = "完美剧情构思方案") -> None:
    """将构思方案全文输出到日志（校验未达标时也打印）。"""
    text = (content or "").strip()
    separator = "=" * 72
    if not text:
        logger.info(f"\n{separator}\n{title}（空）\n{separator}")
        return
    logger.info(f"\n{separator}\n{title}\n{separator}\n{text}\n{separator}")
