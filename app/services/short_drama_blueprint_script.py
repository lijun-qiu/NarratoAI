#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""场景分段蓝图 → JSON 脚本执行要点与高燃 moment 解析。"""

from __future__ import annotations

import re
from typing import Any

_SCENE_SECTION_RE = re.compile(
    r"##\s*全片场景分段[^\n]*\n(.*?)(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SCRIPT_REF_RE = re.compile(
    r"##\s*写脚本参考[^\n]*\n(.*?)(?=\n##\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SCENE_HEADER_RE = re.compile(
    r"###\s*场景\s*\d+\s*·\s*`([^`]+)`(?:\s*·\s*(.+))?",
    re.MULTILINE,
)
_TIMESTAMP_RANGE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s*[-–—~至到]\s*(\d{2}:\d{2}:\d{2}(?:,\d{3})?)"
)
_QUOTE_RE = re.compile(r"「([^」]+)」")
_HIGH_ENERGY_MARKERS = (
    "爆燃", "高能", "高潮", "名场面", "枪战", "追逐", "对峙", "牺牲", "跳楼",
    "纵身", "跃下", "反转", "冲突", "爆炸", "告白", "崩溃", "对决", "抓捕",
)


def _normalize_ts(ts: str) -> str:
    text = (ts or "").strip()
    if "," not in text and text.count(":") == 2:
        return f"{text},000"
    return text


def _extract_timestamp_ranges(text: str) -> list[tuple[str, str]]:
    ranges: list[tuple[str, str]] = []
    for start, end in _TIMESTAMP_RANGE_RE.findall(text or ""):
        pair = (_normalize_ts(start), _normalize_ts(end))
        if pair not in ranges:
            ranges.append(pair)
    return ranges


def _extract_quotes(text: str) -> list[str]:
    quotes: list[str] = []
    for quote in _QUOTE_RE.findall(text or ""):
        cleaned = quote.strip()
        if cleaned and cleaned not in quotes:
            quotes.append(cleaned)
    return quotes


def _score_high_energy(text: str) -> float:
    blob = (text or "").strip()
    if not blob:
        return 0.0
    score = 0.0
    for marker in _HIGH_ENERGY_MARKERS:
        if marker in blob:
            score += 2.0
    if "关键对白" in blob or "名场面" in blob:
        score += 1.0
    return score


def parse_scene_segment_blueprint(markdown: str) -> dict[str, Any]:
    """从「全片场景分段」蓝图解析场景块、时间窗与高燃线索。"""
    text = (markdown or "").strip()
    if not text:
        return {}

    scene_section = ""
    match = _SCENE_SECTION_RE.search(text)
    if match:
        scene_section = match.group(1)

    script_ref = ""
    ref_match = _SCRIPT_REF_RE.search(text)
    if ref_match:
        script_ref = ref_match.group(1)

    scenes: list[dict[str, Any]] = []
    if scene_section:
        headers = list(_SCENE_HEADER_RE.finditer(scene_section))
        for index, header in enumerate(headers):
            start_pos = header.start()
            end_pos = (
                headers[index + 1].start()
                if index + 1 < len(headers)
                else len(scene_section)
            )
            block = scene_section[start_pos:end_pos]
            time_label = (header.group(1) or "").strip()
            place = (header.group(2) or "").strip()
            ts_ranges = _extract_timestamp_ranges(time_label + "\n" + block)
            quotes = _extract_quotes(block)
            picture_hint = ""
            for line in block.splitlines():
                if "画面" in line or "环境要点" in line:
                    picture_hint = line.split("：", 1)[-1].strip().lstrip("-* ")
                    break
            scenes.append(
                {
                    "time_range": time_label,
                    "place": place,
                    "timestamp_ranges": ts_ranges,
                    "quotes": quotes,
                    "picture_hint": picture_hint,
                    "energy_score": _score_high_energy(block),
                    "block": block.strip(),
                }
            )

    high_energy_moments: list[dict[str, Any]] = []
    for scene in sorted(scenes, key=lambda row: -float(row.get("energy_score") or 0)):
        if float(scene.get("energy_score") or 0) >= 2.0:
            high_energy_moments.append(scene)
    for line in (script_ref or "").splitlines():
        stripped = line.strip().lstrip("-* ")
        if not stripped or _score_high_energy(stripped) < 2.0:
            continue
        high_energy_moments.append(
            {
                "time_range": "",
                "place": "",
                "timestamp_ranges": _extract_timestamp_ranges(stripped),
                "quotes": _extract_quotes(stripped),
                "picture_hint": stripped,
                "energy_score": _score_high_energy(stripped),
                "block": stripped,
            }
        )

    all_quotes: list[str] = []
    all_ranges: list[tuple[str, str]] = []
    for scene in scenes:
        for quote in scene.get("quotes") or []:
            if quote not in all_quotes:
                all_quotes.append(quote)
        for item_range in scene.get("timestamp_ranges") or []:
            if item_range not in all_ranges:
                all_ranges.append(item_range)
    for moment in high_energy_moments:
        for quote in moment.get("quotes") or []:
            if quote not in all_quotes:
                all_quotes.append(quote)
        for item_range in moment.get("timestamp_ranges") or []:
            if item_range not in all_ranges:
                all_ranges.append(item_range)

    opening_candidate = high_energy_moments[0] if high_energy_moments else (
        scenes[-1] if scenes else None
    )

    return {
        "scene_count": len(scenes),
        "scenes": scenes,
        "high_energy_moments": high_energy_moments,
        "quotes": all_quotes,
        "timestamp_ranges": all_ranges,
        "opening_candidate": opening_candidate,
        "script_ref_section": script_ref.strip(),
    }


def enrich_blueprint_parse_from_scenes(
    markdown: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    """在旧版蓝图解析结果上叠加场景分段字段。"""
    scene_info = parse_scene_segment_blueprint(markdown)
    if not scene_info:
        return base

    merged = dict(base)
    for key in ("quotes", "timestamp_ranges"):
        values = list(merged.get(key) or [])
        for item in scene_info.get(key) or []:
            if item not in values:
                values.append(item)
        merged[key] = values

    merged["scene_segments"] = scene_info.get("scenes") or []
    merged["high_energy_moments"] = scene_info.get("high_energy_moments") or []
    merged["scene_count"] = scene_info.get("scene_count") or 0

    if not (merged.get("section") or "").strip() and scene_info.get("opening_candidate"):
        candidate = scene_info["opening_candidate"]
        merged["section"] = str(candidate.get("block") or "")
        if candidate.get("picture_hint"):
            merged["picture"] = str(candidate["picture_hint"])

    if scene_info.get("script_ref_section"):
        merged["script_ref_section"] = scene_info["script_ref_section"]

    return merged


def build_blueprint_script_execution_note(
    *,
    plot_blueprint: str = "",
    has_video_analysis: bool = False,
) -> str:
    """第二步 JSON 生成：结合场景分段蓝图的执行要点。"""
    scene_info = parse_scene_segment_blueprint(plot_blueprint)
    scene_count = int(scene_info.get("scene_count") or 0)
    high_energy = scene_info.get("high_energy_moments") or []

    lines = [
        "## 蓝图执行要点（JSON 脚本须严格落实）",
        "- **解说旁白为主（≈85% 成片时长）**：蓝图场景的剧情、冲突、人物关系用 OST=0 **概括**",
        "- **原声 OST=1 极少**：全片 ≤10 段、每段 ≤5 秒，仅情绪顶点（如关键一句台词）",
        "- **picture**：写可执行景别+画面（特写/中景/航拍 + 主体动作）；**禁止无依据写「某某家中」**，不明则写「室内·家中」",
        "- **旁白质量**：每句提供新信息或升华情绪，禁止空泛「而这时+动作复述」",
        "- **时间戳**：禁止重叠；正叙段尽量首尾相接",
    ]
    if scene_count:
        lines.append(
            f"- 蓝图共 **{scene_count}** 个场景；JSON 应覆盖主线场景，"
            "高冲突场景优先保留原声"
        )
    if high_energy:
        preview = high_energy[:3]
        hints = []
        for moment in preview:
            label = (moment.get("time_range") or moment.get("place") or "").strip()
            if label:
                hints.append(label[:48])
        if hints:
            lines.append(f"- **蓝图高燃 moment 参考**：{'；'.join(hints)}")

    if has_video_analysis:
        lines.extend(
            [
                "- **picture 旁白（OST=1）**：对照整片视频分析同时间格 `旁白`/`环境`，"
                "≤配置字数、双引号包裹、承上启下、禁止复述对白",
                "- **OST=0 画面感**：可引用视频分析 `关键事件`/`环境`，但 timestamp 仍以 SRT 为准",
            ]
        )
    else:
        lines.append(
            "- **picture 旁白**：参照蓝图「画面/环境要点」，≤配置字数、双引号包裹"
        )

    lines.extend(
        [
            "- **旁白字幕贴合**：OST=1 的 picture 须能在该段 timestamp 画面内成立；"
            "OST=0 的 timestamp 跨度须 ≥ 解说 TTS 估算时长",
        ]
    )
    return "\n".join(lines)
