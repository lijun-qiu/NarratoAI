#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说：字幕分析与抽帧分析结合。"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Callable, Optional

from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import (
    compute_ost1_segment_bounds,
    get_documentary_compact_settings,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_append_custom_prompt,
)
from app.services.documentary.frame_timeline_sampling import (
    build_frame_subtitle_lexicon_markdown,
    collect_scene_segments_from_analysis,
)
from app.services.llm.migration_adapter import _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService
from app.services.short_drama_plot_analysis_validator import (
    estimate_min_ost1_entries_for_plot,
    emit_plot_analysis_full_text,
    format_plot_analysis_validation_report,
    validate_short_drama_plot_analysis,
)
from app.services.documentary.documentary_plot_blueprint_validator import (
    emit_plot_blueprint_validation_report,
    validate_plot_blueprint,
)
from app.services.short_drama_drama_knowledge import (
    build_plot_blueprint_character_relationship_table_section,
    build_plot_blueprint_name_unification_section,
    build_short_drama_drama_knowledge_section,
    correct_name_mistakes_in_text,
    find_name_mistakes_in_text,
)
from app.services.short_drama_settings import get_short_drama_settings
from app.services.documentary.video_episode_analysis import (
    build_video_episode_analysis_markdown,
    build_video_episode_character_lexicon_markdown,
    build_video_episode_time_bounds_section,
    collect_video_episode_time_bounds,
    load_video_episode_analysis_artifact,
    summarize_video_episode_markdown,
    video_episode_summary_usable,
)
from app.services.documentary.video_episode_segment_schedule import segment_policy_summary
from app.services.srt_utils import SrtEntry, entries_to_srt, parse_srt, _time_str_to_ms
from app.utils import utils

_TRAILING_CLAUSE_PUNCT = re.compile(r"[，。！？、；]+$")
_PHANTOM_SUBTITLE_FRAGMENT_RE = re.compile(
    r"^[的了啊哦呢吧吗呀嘛哈嗯呐哇么之个]$|^[的了啊哦呢吧吗呀嘛哈嗯呐哇么之个][。，！？、；]$"
)


def is_phantom_subtitle_fragment(text: str) -> bool:
    """过滤 ASR/SRT 窗口误挂的碎片（如「了。」「啊，」），非画面硬字幕。"""
    cleaned = clean_subtitle_punctuation(str(text or "").strip())
    if not cleaned:
        return True
    if _PHANTOM_SUBTITLE_FRAGMENT_RE.match(cleaned):
        return True
    if len(cleaned) <= 3 and cleaned[-1] in "。，！？、；":
        core = cleaned[:-1].strip()
        if len(core) <= 1:
            return True
    return False


def _normalize_subtitle_dedupe_key(text: str) -> str:
    return re.sub(r"\s+", "", clean_subtitle_punctuation(str(text or "").strip()))


def observation_burned_in_text(observation: dict[str, Any]) -> str:
    """逐帧硬字幕原文（视觉模型/OCR）；无硬字幕则返回空。"""
    if not isinstance(observation, dict):
        return ""
    if not observation.get("has_burned_in_subtitle"):
        return ""
    return str(observation.get("burned_in_subtitle") or "").strip()


def collect_burned_in_texts_for_segment(
    segment: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    pad_ms: int = 200,
) -> list[str]:
    """收集 segment 时间窗内画面硬字幕（去重、去碎片）。"""
    if not isinstance(segment, dict):
        return []
    time_range = str(segment.get("timestamp") or "").strip()
    if not time_range or "-" not in time_range:
        return []
    try:
        seg_start, seg_end = parse_timestamp_range_ms(time_range)
    except Exception:
        return []
    window_start = max(0, seg_start - pad_ms)
    window_end = seg_end + pad_ms

    seen: set[str] = set()
    collected: list[tuple[int, str]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        ts = str(observation.get("timestamp") or "").strip()
        if not ts:
            continue
        try:
            ts_ms = _timestamp_to_ms(ts)
        except Exception:
            continue
        if ts_ms < window_start or ts_ms > window_end:
            continue
        text = observation_burned_in_text(observation)
        if not text or is_phantom_subtitle_fragment(text):
            continue
        key = _normalize_subtitle_dedupe_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        collected.append((ts_ms, text))
    collected.sort(key=lambda item: item[0])
    return [text for _, text in collected]


def strip_subtitle_entries_from_artifact(artifact: dict[str, Any]) -> None:
    """移除 scene/batch 上的 subtitle_entries，仅保留 subtitle 文本字段。"""
    if not isinstance(artifact, dict):
        return
    for segment in artifact.get("scene_segments") or []:
        if isinstance(segment, dict):
            segment.pop("subtitle_entries", None)
            segment.pop("time_range", None)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch.pop("subtitle_entries", None)
        batch.pop("subtitle_excerpt", None)
        for segment in batch.get("scene_segments") or []:
            if isinstance(segment, dict):
                segment.pop("subtitle_entries", None)
                segment.pop("time_range", None)


def attach_burned_in_subtitles_to_artifact(
    artifact: dict[str, Any],
    *,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """从逐帧 burned_in_subtitle 汇总 scene 字幕，仅写入 subtitle 文本（无 subtitle_entries）。"""
    if not isinstance(artifact, dict):
        return artifact

    observations_by_batch: dict[int, list[dict[str, Any]]] = {}
    for observation in artifact.get("frame_observations") or []:
        if isinstance(observation, dict):
            batch_index = int(observation.get("batch_index", 0))
            observations_by_batch.setdefault(batch_index, []).append(observation)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_index = int(batch.get("batch_index", 0))
        batch_observations = list(batch.get("frame_observations") or [])
        if batch_observations:
            observations_by_batch.setdefault(batch_index, batch_observations)

    all_observations = _iter_frame_observation_dicts(artifact)

    unique_segments = _collect_unique_scene_segments(artifact)
    for segment in unique_segments:
        batch_index = int(segment.get("batch_index", 0))
        batch_obs = observations_by_batch.get(batch_index, [])
        texts = collect_burned_in_texts_for_segment(segment, batch_obs)
        if not texts and batch_obs is not all_observations:
            texts = collect_burned_in_texts_for_segment(segment, all_observations)
        segment.pop("subtitle_entries", None)
        segment.pop("time_range", None)
        if texts:
            segment["subtitle"] = join_subtitle_texts(texts)
        else:
            segment.pop("subtitle", None)

    batch_subtitles: dict[int, list[str]] = {}
    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        subtitle = resolve_segment_subtitle_text(segment)
        if not subtitle:
            continue
        batch_index = int(segment.get("batch_index", 0))
        key = _normalize_subtitle_dedupe_key(subtitle)
        existing = batch_subtitles.setdefault(batch_index, [])
        if key and key not in {_normalize_subtitle_dedupe_key(t) for t in existing}:
            existing.append(subtitle)

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_index = int(batch.get("batch_index", 0))
        parts = batch_subtitles.get(batch_index) or []
        batch.pop("subtitle_entries", None)
        batch.pop("subtitle_excerpt", None)
        if parts:
            batch["subtitle"] = join_subtitle_texts(parts)
        else:
            batch.pop("subtitle", None)

    strip_subtitle_entries_from_artifact(artifact)
    artifact["subtitle_attached"] = True
    artifact["subtitle_source"] = "burned_in_only"
    return artifact


def clean_subtitle_punctuation(text: str) -> str:
    """合并多句字幕后去掉「，；」「。；」等重复标点。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"([，。！？、])；+", "；", cleaned)
    cleaned = re.sub(r"；{2,}", "；", cleaned)
    cleaned = re.sub(r"；+$", "", cleaned).strip()
    return cleaned


def join_subtitle_texts(parts: list[str] | tuple[str, ...]) -> str:
    """多句字幕用分号连接，并去掉各句尾标点避免「，；」叠用。"""
    snippets: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        snippets.append(_TRAILING_CLAUSE_PUNCT.sub("", text))
    if not snippets:
        return ""
    return clean_subtitle_punctuation("；".join(snippets))


def resolve_segment_subtitle_text(segment: dict[str, Any]) -> str:
    """从 segment 的 subtitle 字段得到清洗后的合并字幕。"""
    if not isinstance(segment, dict):
        return ""
    return clean_subtitle_punctuation(str(segment.get("subtitle") or "").strip())


def resolve_segment_time_range(segment: dict[str, Any]) -> str:
    """
    剪辑用时间范围：subtitle_entries 首条 start 至末条 end（一小段完整对白）；
    无条目时回退 timestamp。
    """
    if not isinstance(segment, dict):
        return ""

    entries = segment.get("subtitle_entries")
    if isinstance(entries, list) and entries:
        starts: list[str] = []
        ends: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            end = str(item.get("end") or "").strip()
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        if ends:
            end = ends[-1]
            if starts:
                start = starts[0]
            else:
                base = str(segment.get("timestamp") or "").strip()
                start = base.split("-", 1)[0].strip() if "-" in base else base
            if start and end:
                return f"{start}-{end}"

    return str(segment.get("timestamp") or "").strip()


def _ms_to_hhmmss(ms: int) -> str:
    return utils.seconds_to_time(ms / 1000.0).replace(".", ",")


def _timestamp_to_ms(timestamp: str) -> int:
    text = (timestamp or "").strip()
    try:
        if "," in text:
            time_part, ms_part = text.split(",", 1)
            milliseconds = int(ms_part)
        else:
            time_part = text
            milliseconds = 0
        parts = [int(part) for part in time_part.split(":") if part]
        while len(parts) < 3:
            parts.insert(0, 0)
        hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
        return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds
    except Exception:
        return 0


def parse_timestamp_range_ms(time_range: str) -> tuple[int, int]:
    text = (time_range or "").strip()
    if "-" not in text:
        ms = _timestamp_to_ms(text)
        return ms, ms
    start_text, end_text = text.split("-", 1)
    start_ms = _timestamp_to_ms(start_text.strip())
    end_ms = _timestamp_to_ms(end_text.strip())
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    return start_ms, end_ms


def truncate_subtitle_content(subtitle_content: str, max_chars: int) -> str:
    text = (subtitle_content or "").strip()
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…（字幕已截断）"


def _iter_frame_observation_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """合并 batches 与顶层 frame_observations，按时间排序。"""
    merged: list[dict[str, Any]] = []
    seen_ts: set[str] = set()

    def add_obs(obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            return
        ts = str(obs.get("timestamp") or "").strip()
        key = ts or str(id(obs))
        if key in seen_ts:
            return
        seen_ts.add(key)
        merged.append(obs)

    for batch in data.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for obs in batch.get("frame_observations") or []:
            add_obs(obs)
    for obs in data.get("frame_observations") or []:
        add_obs(obs)

    def sort_key(item: dict[str, Any]) -> int:
        ts = str(item.get("timestamp") or "").strip()
        if not ts:
            return 0
        try:
            return _time_str_to_ms(ts)
        except Exception:
            return 0

    return sorted(merged, key=sort_key)


def extract_subtitle_entries_from_frame_analysis(data: dict[str, Any]) -> list[SrtEntry]:
    """
    从抽帧 JSON 还原字幕条目（原样保留 text）：
    scene/batch 的 subtitle_entries → 硬字幕 burned_in_subtitle。
    """
    if not isinstance(data, dict):
        return []

    collected: list[SrtEntry] = []
    seen: set[tuple[int, str]] = set()
    default_duration_ms = 2500

    def append_entry(
        start_label: str,
        end_label: str,
        text: str,
        *,
        verbatim: bool = False,
    ) -> None:
        body = str(text or "").strip()
        if not verbatim:
            body = clean_subtitle_punctuation(body)
        if not body or not start_label:
            return
        try:
            start_ms = _time_str_to_ms(start_label.strip())
            end_ms = _time_str_to_ms(end_label.strip()) if end_label else start_ms + default_duration_ms
        except Exception:
            return
        if end_ms <= start_ms:
            end_ms = start_ms + default_duration_ms
        key = (start_ms, body)
        if key in seen:
            return
        seen.add(key)
        collected.append(SrtEntry(start_ms=start_ms, end_ms=end_ms, text=body))

    def append_subtitle_entry_dict(item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        append_entry(
            str(item.get("start") or ""),
            str(item.get("end") or ""),
            str(item.get("text") or ""),
            verbatim=True,
        )

    for batch in data.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for item in batch.get("subtitle_entries") or []:
            append_subtitle_entry_dict(item)

    for segment in collect_scene_segments_from_analysis(data):
        entries = segment.get("subtitle_entries")
        has_entry_list = isinstance(entries, list) and bool(entries)
        if has_entry_list:
            for item in entries:
                append_subtitle_entry_dict(item)
        elif resolve_segment_subtitle_text(segment):
            ts_range = resolve_segment_time_range(segment)
            if "-" in ts_range:
                start_label, end_label = ts_range.split("-", 1)
                append_entry(
                    start_label,
                    end_label,
                    resolve_segment_subtitle_text(segment),
                    verbatim=True,
                )

    prev_ms: int | None = None
    for obs in _iter_frame_observation_dicts(data):
        burned = str(obs.get("burned_in_subtitle") or "").strip()
        has_burned = bool(obs.get("has_burned_in_subtitle")) and bool(burned)
        attached = str(obs.get("subtitle") or "").strip()
        text = burned if has_burned else attached
        if not text:
            continue
        start_label = str(obs.get("subtitle_start") or obs.get("timestamp") or "").strip()
        end_label = str(obs.get("subtitle_end") or "").strip()
        if not end_label and start_label:
            try:
                start_ms = _time_str_to_ms(start_label)
                end_ms = (
                    (prev_ms + start_ms) // 2
                    if prev_ms is not None and start_ms > prev_ms
                    else start_ms + default_duration_ms
                )
                end_label = utils.seconds_to_time(end_ms / 1000.0).replace(".", ",")
                prev_ms = start_ms
            except Exception:
                end_label = ""
        append_entry(start_label, end_label, text, verbatim=True)

    collected.sort(key=lambda item: (item.start_ms, item.end_ms, item.text))
    return collected


def extract_subtitle_srt_from_frame_analysis(frame_json_path: str) -> str:
    """将抽帧 JSON 内字幕/硬字幕还原为 SRT 文本（供构思蓝图对照）。"""
    path = (frame_json_path or "").strip()
    if not path or not os.path.isfile(path):
        return ""
    try:
        from app.services.documentary.frame_analysis_pairing import load_analysis_artifact

        data = load_analysis_artifact(path)
    except (OSError, ValueError) as exc:
        logger.warning(f"读取抽帧 JSON 字幕失败 {path}: {exc}")
        return ""

    entries = extract_subtitle_entries_from_frame_analysis(data)
    if not entries:
        return ""
    return entries_to_srt(entries).strip()


def resolve_subtitle_content_for_plot_analysis(
    *,
    subtitle_content: str = "",
    frame_json_path: str | None = None,
) -> tuple[str, str]:
    """
    脚本生成用字幕源：有 SRT 文件内容则优先；否则从抽帧 JSON 提取。
    构思蓝图请用 resolve_subtitles_for_plot_blueprint（SRT 与抽帧内字幕分别提取）。
    返回 (字幕正文, 来源标签 srt_file | frame_analysis)。
    """
    srt_text = (subtitle_content or "").strip()
    if srt_text:
        return srt_text, "srt_file"

    extracted = extract_subtitle_srt_from_frame_analysis(frame_json_path or "")
    if extracted.strip():
        logger.info(
            f"未提供 SRT，已改用抽帧 JSON 内字幕（约 {len(extracted)} 字）"
        )
        return extracted.strip(), "frame_analysis"
    return "", ""


def resolve_frame_subtitle_for_plot_blueprint(
    frame_json_path: str | None,
) -> str:
    """构思蓝图：从抽帧 JSON 提取 subtitle_entries / 硬字幕。"""
    extracted = extract_subtitle_srt_from_frame_analysis(frame_json_path or "")
    text = extracted.strip()
    if text:
        logger.info(f"构思蓝图：已取抽帧内字幕（约 {len(text)} 字）")
    return text


def resolve_subtitles_for_plot_blueprint(
    *,
    subtitle_content: str = "",
    frame_json_path: str | None = None,
) -> tuple[str, str, str]:
    """构思蓝图：分别提取 SRT 与抽帧内字幕。返回 (srt_text, frame_text, primary_source)。"""
    srt_text = (subtitle_content or "").strip()
    frame_text = resolve_frame_subtitle_for_plot_blueprint(frame_json_path)
    if srt_text:
        return srt_text, frame_text, "srt_file"
    if frame_text:
        return "", frame_text, "frame_analysis"
    return "", "", ""


def collect_subtitle_time_bounds(subtitle_content: str) -> dict[str, int]:
    """从 SRT 文本汇总对白可用时间范围（毫秒）。"""
    entries = parse_subtitle_entries_for_blueprint(subtitle_content)
    if not entries:
        return {"min_ms": 0, "max_ms": 0}
    return {
        "min_ms": min(entry["start_ms"] for entry in entries),
        "max_ms": max(entry["end_ms"] for entry in entries),
    }


def parse_subtitle_entries_for_blueprint(subtitle_content: str) -> list[dict[str, Any]]:
    """解析 SRT 为蓝图校验用的对白条目列表。"""
    from app.services.srt_utils import format_timestamp_ms, parse_srt

    entries = parse_srt(subtitle_content or "")
    parsed: list[dict[str, Any]] = []
    for entry in entries:
        text = str(entry.text or "").strip()
        if not text:
            continue
        parsed.append(
            {
                "start_ms": int(entry.start_ms),
                "end_ms": int(entry.end_ms),
                "text": text,
                "time_range": (
                    f"{format_timestamp_ms(entry.start_ms)}-"
                    f"{format_timestamp_ms(entry.end_ms)}"
                ),
            }
        )
    return parsed


def build_subtitle_cue_index_for_blueprint(
    subtitle_content: str,
    *,
    max_entries: int = 80,
) -> str:
    """构思蓝图用：SRT 对白时间窗索引（OST=1 须从此表选取）。"""
    entries = parse_subtitle_entries_for_blueprint(subtitle_content)
    if not entries:
        return (
            "### 字幕对白时间窗索引\n"
            "> 未提供 SRT；OST=1 时间戳须来自整片视频分析 `important_dialogues` 中的真实时间。"
        )
    if len(entries) > max_entries:
        step = max(1, len(entries) // max_entries)
        sampled = [entries[index] for index in range(0, len(entries), step)][:max_entries]
        note = (
            f"\n> SRT 共 **{len(entries)}** 条，上表采样 **{len(sampled)}** 条；"
            "OST=1 的 timestamp 须与 SRT 真实区间一致。"
        )
    else:
        sampled = entries
        note = f"\n> SRT 共 **{len(entries)}** 条，OST=1 须逐字引用对白并匹配时间窗。"
    lines = [
        "### 字幕对白时间窗索引（蓝图「字幕窗」/ OST=1 须从此表选取）",
        "| 字幕窗 time_range | 对白摘要 |",
        "|---|---|",
    ]
    for entry in sampled:
        preview = str(entry.get("text") or "").replace("|", "/").replace("\n", " ")[:40]
        lines.append(f"| `{entry.get('time_range', '')}` | {preview} |")
    lines.append(note)
    return "\n".join(lines)


def build_plot_blueprint_dual_time_alignment_section(
    video_artifact: dict[str, Any],
    subtitle_content: str,
) -> str:
    """视频分析固定格 + SRT 字幕窗 + 双轴对齐规则。"""
    video_section = build_video_episode_time_bounds_section(video_artifact)
    subtitle_section = build_subtitle_cue_index_for_blueprint(subtitle_content)
    if not video_section and not subtitle_section:
        return ""
    return f"{video_section}\n\n{subtitle_section}"


def _raw_scene_segments_from_artifact(artifact: dict) -> list[dict]:
    top_level = artifact.get("scene_segments")
    if isinstance(top_level, list) and top_level:
        return [segment for segment in top_level if isinstance(segment, dict)]
    return []


def _scan_segment_time_bounds(segments: list[dict]) -> tuple[int, int]:
    min_ms = 0
    max_ms = 0
    for segment in segments:
        clip_range = resolve_segment_time_range(segment)
        if not clip_range or "-" not in clip_range:
            clip_range = str(segment.get("timestamp") or "")
        if not clip_range or "-" not in clip_range:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range_ms(clip_range)
        except Exception:
            continue
        if not max_ms or end_ms > max_ms:
            max_ms = end_ms
        if not min_ms or start_ms < min_ms:
            min_ms = start_ms
    return min_ms, max_ms


def collect_frame_analysis_time_bounds(
    frame_json_path: str | None,
) -> dict[str, Any]:
    """从抽帧 JSON 汇总可用时间范围与场景锚点。"""
    from app.services.documentary.frame_analysis_pairing import load_analysis_artifact

    empty: dict[str, Any] = {
        "min_ms": 0,
        "max_ms": 0,
        "anchors": [],
    }
    path = (frame_json_path or "").strip()
    if not path:
        return empty
    try:
        artifact = load_analysis_artifact(path)
    except Exception as exc:
        logger.warning(f"读取抽帧 JSON 时间边界失败: {exc}")
        return empty

    raw_segments = _raw_scene_segments_from_artifact(artifact)
    if raw_segments:
        min_ms, max_ms = _scan_segment_time_bounds(raw_segments)
    else:
        min_ms, max_ms = 0, 0

    segments = collect_scene_segments_from_analysis(artifact)
    if not segments and not raw_segments:
        return empty
    if not max_ms and segments:
        min_ms, max_ms = _scan_segment_time_bounds(segments)

    anchors: list[dict[str, str]] = []
    for segment in segments:
        clip_range = resolve_segment_time_range(segment)
        if not clip_range or "-" not in clip_range:
            clip_range = str(segment.get("timestamp") or "")
        if not clip_range or "-" not in clip_range:
            continue
        try:
            start_ms, end_ms = parse_timestamp_range_ms(clip_range)
        except Exception:
            continue
        scene = str(segment.get("scene") or "").strip()
        if "切换至" in scene:
            scene = scene.split("切换至")[-1].strip()
        while scene.startswith("从"):
            scene = scene[1:].strip()
        observation = str(segment.get("observation") or segment.get("action") or "").strip()
        if observation:
            observation = observation.split("；")[0][:72]
        anchors.append(
            {
                "time_range": clip_range,
                "scene": scene[:48],
                "observation": observation,
            }
        )

    return {
        "min_ms": min_ms,
        "max_ms": max_ms,
        "anchors": anchors,
    }


def build_frame_analysis_time_bounds_section(
    frame_json_path: str | None,
    *,
    source_duration_sec: float | None = None,
    srt_max_ms: int | None = None,
    srt_min_ms: int = 0,
    max_anchor_rows: int = 28,
) -> str:
    """注入蓝图 prompt：原片/抽帧可用时间上限与场景锚点索引。"""
    from app.services.srt_utils import format_timestamp_ms

    bounds = collect_frame_analysis_time_bounds(frame_json_path)
    max_ms = int(bounds.get("max_ms") or 0)
    min_ms = int(bounds.get("min_ms") or 0)
    if max_ms <= 0:
        return (
            "## 抽帧时间边界（硬性）\n"
            "- 未能从抽帧 JSON 解析时间范围；所有 timestamp 须来自下方抽帧摘要中的真实时间，禁止编造\n"
        )

    frame_max_label = format_timestamp_ms(max_ms)
    frame_min_label = format_timestamp_ms(min_ms)
    cap_ms = max_ms
    cap_note = f"抽帧覆盖 **{frame_min_label}–{frame_max_label}**"
    if source_duration_sec and source_duration_sec > 0:
        video_ms = int(source_duration_sec * 1000)
        cap_ms = min(video_ms, max_ms) if max_ms > 0 else video_ms
        video_label = format_timestamp_ms(video_ms)
        cap_label = format_timestamp_ms(cap_ms)
        cap_note = (
            f"原片时长 **{video_label}**；抽帧覆盖 **{frame_min_label}–{frame_max_label}**；"
            f"**画面 timestamp 结束时间不得超过 {cap_label}**"
        )
    if srt_max_ms and srt_max_ms > 0:
        srt_max_label = format_timestamp_ms(srt_max_ms)
        srt_min_label = format_timestamp_ms(max(0, srt_min_ms))
        dialogue_cap = min(cap_ms, srt_max_ms) if cap_ms else srt_max_ms
        dialogue_label = format_timestamp_ms(dialogue_cap)
        cap_note += (
            f"；SRT 对白覆盖 **{srt_min_label}–{srt_max_label}**；"
            f"**OST=1 原声 timestamp 结束时间不得超过 {dialogue_label}**"
        )

    anchors = bounds.get("anchors") or []
    if len(anchors) > max_anchor_rows:
        step = max(1, len(anchors) // max_anchor_rows)
        sampled = [anchors[index] for index in range(0, len(anchors), step)]
    else:
        sampled = anchors

    rows: list[str] = []
    for item in sampled[:max_anchor_rows]:
        scene = str(item.get("scene") or "—")
        obs = str(item.get("observation") or "—")
        rows.append(
            f"| {item.get('time_range', '—')} | {scene} | {obs} |"
        )
    table = "\n".join(rows) if rows else "| — | — | — |"

    lead_sec = 10
    return f"""## 抽帧时间边界与场景锚点（硬性 · 第二依据）
- {cap_note}
- **禁止**写出超过上限的时间戳；禁止仅用 `00:02:51` 单点而无结束时间（OST=1 须写完整区间）
- **画面描述**须与下表同时间段抽帧 observation/action 一致，禁止臆造表中未出现的地点/动作/昼夜
- OST=0 铺垫下一段 OST=1：取画起点 = 下一段原声开始 **− 约 {lead_sec} 秒**（仍须落在上表覆盖范围内）

| 抽帧时间段 | 场景 | 画面要点（摘自抽帧） |
|------------|------|----------------------|
{table}
"""


def subtitle_excerpt_for_time_range(
    subtitle_content: str,
    time_range: str,
    *,
    pad_ms: int = 5000,
    max_lines: int = 8,
) -> str:
    """取某时间段（含前后缓冲）内的字幕对白摘要。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return ""

    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms

    lines: list[str] = []
    for entry in entries:
        if entry.end_ms < window_start or entry.start_ms > window_end:
            continue
        text = (entry.text or "").strip().replace("\n", " ")
        if not text:
            continue
        lines.append(f"{_ms_to_hhmmss(entry.start_ms)} {text}")
        if len(lines) >= max_lines:
            break

    if not lines:
        return ""
    excerpt = "；".join(lines)
    return excerpt[:400] + ("…" if len(excerpt) > 400 else "")


def _normalize_subtitle_compare_text(text: str) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    for ch in "，,。．！？、；：""''…—- ":
        cleaned = cleaned.replace(ch, "")
    return cleaned


def merge_subtitle_text_with_burned_in(srt_text: str, burned_text: str) -> tuple[str, str]:
    """
    字幕文字冲突时以画面硬字幕为准，时间仍用 SRT。
    返回 (最终文本, text_source)；source 为 srt | burned_in_corrected。
    """
    srt = (srt_text or "").strip().replace("\n", " ")
    burned = (burned_text or "").strip().replace("\n", " ")
    if not srt:
        return burned, "burned_in_only" if burned else "srt"
    if not burned:
        return srt, "srt"
    if _normalize_subtitle_compare_text(srt) == _normalize_subtitle_compare_text(burned):
        return srt, "srt"
    return burned, "burned_in_corrected"


def find_srt_entry_at_timestamp(
    entries: list[SrtEntry],
    timestamp: str,
    *,
    pad_ms: int = 800,
) -> SrtEntry | None:
    if not entries:
        return None

    ts_ms = _timestamp_to_ms(timestamp)
    for entry in entries:
        if entry.start_ms <= ts_ms <= entry.end_ms:
            return entry

    nearest: SrtEntry | None = None
    nearest_distance = pad_ms + 1
    for entry in entries:
        distance = min(abs(entry.start_ms - ts_ms), abs(entry.end_ms - ts_ms))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = entry

    if nearest is None or nearest_distance > pad_ms:
        return None
    return nearest


def subtitle_at_timestamp(
    subtitle_content: str,
    timestamp: str,
    *,
    pad_ms: int = 800,
) -> str:
    """取某一时间点对应的 SRT 对白（优先命中区间，否则取 pad 内最近一条）。"""
    entries = parse_srt(subtitle_content or "")
    entry = find_srt_entry_at_timestamp(entries, timestamp, pad_ms=pad_ms)
    if entry is None:
        return ""
    return (entry.text or "").strip().replace("\n", " ")


def subtitle_entries_for_time_range(
    subtitle_content: str,
    time_range: str,
    *,
    pad_ms: int = 0,
    max_entries: int = 0,
    text_overrides_by_start_ms: dict[int, str] | None = None,
) -> list[dict[str, str]]:
    """取时间范围内的 SRT 条目，含起止时间与对白文本。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return []

    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms
    overrides = text_overrides_by_start_ms or {}

    result: list[dict[str, str]] = []
    for entry in entries:
        if entry.end_ms < window_start or entry.start_ms > window_end:
            continue
        srt_text = (entry.text or "").strip().replace("\n", " ")
        if not srt_text and entry.start_ms not in overrides:
            continue
        text, text_source = merge_subtitle_text_with_burned_in(
            srt_text,
            overrides.get(entry.start_ms, ""),
        )
        if not text:
            continue
        payload: dict[str, str] = {
            "start": _ms_to_hhmmss(entry.start_ms),
            "end": _ms_to_hhmmss(entry.end_ms),
            "text": text,
        }
        if text_source == "burned_in_corrected":
            payload["text_source"] = text_source
        result.append(payload)
        if max_entries > 0 and len(result) >= max_entries:
            break
    return result


def build_burned_in_overrides_for_entries(
    observations: list[dict[str, Any]],
    entries: list[SrtEntry],
) -> dict[int, str]:
    """根据逐帧硬字幕，为重叠的 SRT 条目收集文字修正（以画面为准）。"""
    overrides: dict[int, str] = {}
    if not observations or not entries:
        return overrides

    for observation in observations:
        if not isinstance(observation, dict):
            continue
        burned = str(observation.get("burned_in_subtitle") or "").strip()
        if not burned:
            continue
        ts_ms = _timestamp_to_ms(str(observation.get("timestamp") or ""))
        if ts_ms <= 0:
            continue
        for entry in entries:
            if entry.start_ms <= ts_ms <= entry.end_ms:
                srt_text = (entry.text or "").strip().replace("\n", " ")
                if _normalize_subtitle_compare_text(srt_text) != _normalize_subtitle_compare_text(burned):
                    overrides[entry.start_ms] = burned
                break
    return overrides


def apply_subtitle_fields_to_observation(
    observation: dict[str, Any],
    subtitle_content: str,
    *,
    pad_ms: int = 800,
    parsed_entries: list[SrtEntry] | None = None,
) -> None:
    """为单帧写入 subtitle / subtitle_start / subtitle_end（抽帧阶段调用）。"""
    if not isinstance(observation, dict):
        return

    timestamp = str(observation.get("timestamp") or "").strip()
    burned = str(observation.get("burned_in_subtitle") or "").strip()
    entries = parsed_entries if parsed_entries is not None else parse_srt(subtitle_content or "")

    entry = find_srt_entry_at_timestamp(entries, timestamp, pad_ms=pad_ms) if timestamp else None
    if entry is not None:
        srt_text = (entry.text or "").strip().replace("\n", " ")
        text, text_source = merge_subtitle_text_with_burned_in(srt_text, burned)
        if text:
            observation["subtitle"] = text
            observation["subtitle_start"] = _ms_to_hhmmss(entry.start_ms)
            observation["subtitle_end"] = _ms_to_hhmmss(entry.end_ms)
            observation["subtitle_text_source"] = text_source
        return

    if burned:
        observation["subtitle"] = burned
        observation["subtitle_start"] = ""
        observation["subtitle_end"] = ""
        observation["subtitle_text_source"] = "burned_in_only"


def apply_subtitle_fields_to_segment(
    segment: dict[str, Any],
    subtitle_content: str,
    *,
    observations: list[dict[str, Any]] | None = None,
    pad_ms: int = 500,
    max_entries: int = 20,
) -> None:
    """为场景段写入 subtitle_entries 与合并 subtitle 文本。"""
    if not isinstance(segment, dict):
        return

    time_range = str(segment.get("timestamp") or "").strip()
    if not time_range:
        return

    all_entries = parse_srt(subtitle_content or "")
    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms
    window_entries = [
        entry
        for entry in all_entries
        if not (entry.end_ms < window_start or entry.start_ms > window_end)
    ]
    overrides = build_burned_in_overrides_for_entries(observations or [], window_entries)
    entries = subtitle_entries_for_time_range(
        subtitle_content,
        time_range,
        pad_ms=pad_ms,
        max_entries=max_entries,
        text_overrides_by_start_ms=overrides,
    )
    if not entries:
        return

    segment["subtitle_entries"] = entries
    segment["subtitle"] = join_subtitle_texts(str(item.get("text") or "") for item in entries)
    segment.pop("time_range", None)


def _subtitle_entry_start_ms(entry: dict[str, Any]) -> int:
    return _timestamp_to_ms(str(entry.get("start") or ""))


def _subtitle_entry_end_ms(entry: dict[str, Any]) -> int:
    return _timestamp_to_ms(str(entry.get("end") or ""))


def _segment_timestamp_bounds_ms(segment: dict[str, Any]) -> tuple[int, int]:
    return parse_timestamp_range_ms(str(segment.get("timestamp") or ""))


def _score_subtitle_entry_for_segment(entry: dict[str, Any], segment: dict[str, Any]) -> float:
    """条目与 scene 时间窗重叠越多、中心越近，得分越高。"""
    entry_start = _subtitle_entry_start_ms(entry)
    entry_end = _subtitle_entry_end_ms(entry)
    if entry_end <= entry_start:
        return -1.0
    seg_start, seg_end = _segment_timestamp_bounds_ms(segment)
    if seg_end <= seg_start:
        return -1.0
    overlap = min(entry_end, seg_end) - max(entry_start, seg_start)
    if overlap <= 0:
        return -1.0
    entry_mid = (entry_start + entry_end) / 2.0
    seg_mid = (seg_start + seg_end) / 2.0
    overlap_ratio = overlap / max(entry_end - entry_start, 1)
    distance_sec = abs(entry_mid - seg_mid) / 1000.0
    return overlap_ratio * 10.0 - distance_sec


def _build_parsed_subtitle_payload(
    entry: SrtEntry,
    *,
    text_override: str = "",
) -> dict[str, str] | None:
    srt_text = (entry.text or "").strip().replace("\n", " ")
    text, text_source = merge_subtitle_text_with_burned_in(srt_text, text_override)
    if not text and not text_override:
        return None
    payload: dict[str, str] = {
        "start": _ms_to_hhmmss(entry.start_ms),
        "end": _ms_to_hhmmss(entry.end_ms),
        "text": text,
    }
    if text_source == "burned_in_corrected":
        payload["text_source"] = text_source
    return payload


def assign_subtitle_entries_to_segments(
    segments: list[dict[str, Any]],
    parsed_entries: list[SrtEntry],
    *,
    observations_by_batch: dict[int, list[dict[str, Any]]] | None = None,
) -> None:
    """每条 SRT 字幕只分配给最匹配的一个 scene_segment（全局唯一，不重复挂载）。"""
    cleaned = [segment for segment in segments if isinstance(segment, dict)]
    if not cleaned or not parsed_entries:
        return

    obs_by_batch = observations_by_batch or {}
    overrides_by_batch: dict[int, dict[int, str]] = {}
    for batch_index, observations in obs_by_batch.items():
        overrides_by_batch[batch_index] = build_burned_in_overrides_for_entries(
            observations,
            parsed_entries,
        )

    assignments: dict[int, list[dict[str, str]]] = {
        index: [] for index in range(len(cleaned))
    }

    for entry in parsed_entries:
        best_index = -1
        best_score = -1.0
        best_payload: dict[str, str] | None = None
        for index, segment in enumerate(cleaned):
            batch_index = int(segment.get("batch_index", 0))
            override = overrides_by_batch.get(batch_index, {}).get(entry.start_ms, "")
            payload = _build_parsed_subtitle_payload(entry, text_override=override)
            if payload is None:
                continue
            score = _score_subtitle_entry_for_segment(payload, segment)
            if score > best_score:
                best_score = score
                best_index = index
                best_payload = payload
        if best_index >= 0 and best_payload is not None and best_score > 0:
            assignments[best_index].append(best_payload)

    for index, segment in enumerate(cleaned):
        entries = sorted(assignments[index], key=_subtitle_entry_start_ms)
        if entries:
            segment["subtitle_entries"] = entries
            segment["subtitle"] = join_subtitle_texts(
                str(item.get("text") or "") for item in entries
            )
        else:
            segment.pop("subtitle_entries", None)
            segment.pop("subtitle", None)
        segment.pop("time_range", None)


def _collect_unique_scene_segments(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for segment in artifact.get("scene_segments") or []:
        if isinstance(segment, dict) and id(segment) not in seen:
            seen.add(id(segment))
            unique.append(segment)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for segment in batch.get("scene_segments") or []:
            if isinstance(segment, dict) and id(segment) not in seen:
                seen.add(id(segment))
                unique.append(segment)
    return unique


def _sync_batch_subtitles_from_segments(
    artifact: dict[str, Any],
    *,
    subtitle_content: str = "",
    pad_ms: int = 0,
) -> None:
    """从 scene_segment 汇总各 batch 的字幕字段，避免对同一 SRT 重复拉取。"""
    segments_by_batch: dict[int, list[dict[str, Any]]] = {}
    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        batch_index = int(segment.get("batch_index", 0))
        segments_by_batch.setdefault(batch_index, []).append(segment)

    text = (subtitle_content or "").strip()
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_index = int(batch.get("batch_index", 0))
        batch_segments = segments_by_batch.get(batch_index) or []

        merged_entries: list[dict[str, str]] = []
        seen_starts: set[int] = set()
        for segment in batch_segments:
            if not isinstance(segment, dict):
                continue
            for item in segment.get("subtitle_entries") or []:
                if not isinstance(item, dict):
                    continue
                start_ms = _subtitle_entry_start_ms(item)
                if start_ms <= 0 or start_ms in seen_starts:
                    continue
                seen_starts.add(start_ms)
                merged_entries.append(item)
        merged_entries.sort(key=_subtitle_entry_start_ms)

        if merged_entries:
            batch["subtitle_entries"] = merged_entries
            batch["subtitle"] = join_subtitle_texts(
                str(item.get("text") or "") for item in merged_entries
            )
        else:
            batch.pop("subtitle_entries", None)
            batch.pop("subtitle", None)

        time_range = str(batch.get("time_range") or "").strip()
        if time_range and text:
            batch["subtitle_excerpt"] = subtitle_excerpt_for_time_range(
                text,
                time_range,
                pad_ms=pad_ms,
            )
        else:
            batch.pop("subtitle_excerpt", None)


def partition_subtitle_entries_across_segments(segments: list[dict[str, Any]]) -> None:
    """
    每条 SRT 字幕（按 start 唯一）只归属一个 scene_segment。

    视觉模型常输出 timestamp 重叠的多场景；按窗口独立挂载会导致 subtitle_entries 重复。
    """
    cleaned = [segment for segment in segments if isinstance(segment, dict)]
    if len(cleaned) <= 1:
        return

    entry_by_start: dict[int, dict[str, Any]] = {}
    for segment in cleaned:
        for entry in segment.get("subtitle_entries") or []:
            if not isinstance(entry, dict):
                continue
            start_ms = _subtitle_entry_start_ms(entry)
            if start_ms <= 0:
                continue
            entry_by_start.setdefault(start_ms, entry)

    if not entry_by_start:
        return

    owner_by_start: dict[int, int] = {}
    for start_ms, entry in entry_by_start.items():
        best_index = -1
        best_score = -1.0
        for index, segment in enumerate(cleaned):
            score = _score_subtitle_entry_for_segment(entry, segment)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0:
            owner_by_start[start_ms] = best_index

    for index, segment in enumerate(cleaned):
        entries = segment.get("subtitle_entries")
        if not isinstance(entries, list):
            continue
        kept = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and owner_by_start.get(_subtitle_entry_start_ms(entry)) == index
        ]
        kept.sort(key=_subtitle_entry_start_ms)
        if kept:
            segment["subtitle_entries"] = kept
            segment["subtitle"] = join_subtitle_texts(
                str(item.get("text") or "") for item in kept
            )
        else:
            segment.pop("subtitle_entries", None)
            segment.pop("subtitle", None)
            segment.pop("time_range", None)


def _partition_subtitle_entries_in_artifact(artifact: dict[str, Any]) -> None:
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for segment in batch.get("scene_segments") or []:
            if isinstance(segment, dict) and id(segment) not in seen:
                seen.add(id(segment))
                unique.append(segment)
    for segment in artifact.get("scene_segments") or []:
        if isinstance(segment, dict) and id(segment) not in seen:
            seen.add(id(segment))
            unique.append(segment)
    partition_subtitle_entries_across_segments(unique)


def attach_subtitles_to_frame_analysis_artifact(
    artifact: dict[str, Any],
    subtitle_content: str,
    *,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """抽帧完成后：仅从画面硬字幕写入 segment.subtitle（不挂 SRT subtitle_entries）。"""
    if not isinstance(artifact, dict):
        return artifact
    if (subtitle_content or "").strip():
        logger.debug("attach_subtitles：已忽略 SRT，scene 字幕仅取自 burned_in_subtitle")
    return attach_burned_in_subtitles_to_artifact(artifact, settings=settings)


def build_subtitle_cross_validation_instructions(
    settings: Optional[dict[str, Any]] = None,
) -> str:
    cfg = settings or get_documentary_settings()
    lines = [
        "## 素材优先级（有字幕时必须遵守）",
        "",
        "**生成脚本以字幕为主，抽帧为辅：**",
        "- **字幕（主）**：剧情主线、`narration` 内容、`original_line`、人名、**所有 `timestamp`**",
        "- **抽帧（辅）**：仅用于 `picture` 画面描述，以及截取时间在字幕区间内的**对齐参考**",
        "",
        "## 字幕 × 抽帧 对照规则",
        "",
        "- **对白内容、剧情、人名、台词时间戳** → 以 `<subtitles>` 为准",
        "- **`picture` 画面描述** → 参考抽帧（人物表情、动作、场景氛围）",
        "- **昼夜、天气、光线** → `picture` 以抽帧为准；`narration` 剧情仍以字幕为准",
        "- 两者冲突时：**剧情/台词/时间戳以字幕为准**，画面描述以抽帧为准，勿互相覆盖",
        "- 写 `timestamp` 必须从字幕复制；可参考抽帧在字幕区间内对齐起止，严禁重叠",
    ]
    if is_fazu2_compact_settings(cfg):
        min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
        lines.extend(
            [
                "",
                "### 精剪 · 罚罪2 V2",
                "- 生成 JSON 前须完成「字幕×抽帧 对照分析」蓝图，并严格遵循",
                "- 标出本集**最炸裂名场面**（供第 1 段纯原声开头）：金句+时间戳+动作",
                "- 列出正叙剧情情节点（按时间顺序）；多数对白写入 OST=0 解说",
                f"- 从字幕列出 **{min_ost1}–{max_ost1}** 个 OST=1（约 50%）；timestamp 覆盖整句对白，播完再接解说",
                "- 标出收尾情节与道别；人名用胡小跃/秦枫/伟业/罗博等（禁止胡小月/小月）",
            ]
        )
    elif is_compact_documentary_settings(cfg):
        lines.extend(
            [
                "",
                "### 精剪 · 原声对位",
                "- 标注可作 OST=1 的字幕对白 moment 与时间戳",
            ]
        )
    else:
        lines.extend(
            [
                "- 标记 `[高光原声]` 或字幕中有力对白 + 画面张力强的 moment，优先 OST=1",
            ]
        )
    return "\n".join(lines)


def summarize_frame_markdown(
    frame_markdown: str,
    max_chars: int,
    *,
    sampling: str = "head",
    frame_json_path: str | None = None,
) -> str:
    text = (frame_markdown or "").strip()
    if not text and not frame_json_path:
        return "（无抽帧描述）"

    strategy = (sampling or "head").strip().lower()
    if strategy == "timeline_uniform":
        json_path = (frame_json_path or "").strip()
        if json_path and os.path.isfile(json_path):
            from app.services.documentary.frame_timeline_sampling import (
                frame_analysis_to_timeline_sampled_markdown,
            )

            sampled = frame_analysis_to_timeline_sampled_markdown(json_path, max_chars)
            if sampled and not sampled.startswith("错误:"):
                return sampled
        logger.warning("时间轴均匀采样失败，回退为 Markdown 顺序截断")

    if not text:
        return "（无抽帧描述）"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…（抽帧摘要已截断）"


def _frame_summary_usable(frame_summary: str) -> bool:
    text = (frame_summary or "").strip()
    return bool(text) and text not in {"（无抽帧描述）", "（无）"}


_PERSON_HOME_IN_SCENE_RE = re.compile(r"([一-龥]{2,8})家中")
_HOME_GENERIC_SKIP_NAMES = frozenset(
    {"国家", "作家", "专家", "大家", "居家", "自家", "国家", "商家", "厂家"}
)


def build_plot_blueprint_location_naming_rules() -> str:
    """场景分段蓝图：地点命名约束（禁止无依据的「某某家中」）。"""
    return """## 场景地点命名（硬性）
- **禁止臆测归属**：SRT / 整片视频分析 **未明确写出**「某某家/某某住处」时，**不得**写「罗博家中」「秦枫家中」「胡小跃家中」等
- **室内归属不明**：场景标题用 **「室内·家中」** 或 **「室内·私宅（归属未明）」**；正文写「胡小跃与罗博在一处室内用餐…」，不要替素材认定是谁家
- **允许写明的场所**：字幕或 `environment_description` **原文出现**的可写（如「秦枫家」「罗马酒店」「警局审讯室」「龙湾祠堂」）
- **公共/职能场所**：警局、审讯室、会议室、天台、祠堂、广场、街道等 — 须有画面/字幕依据，勿张冠李戴
- **错误示例**：仅见两人室内吃饭 → ❌「罗博家中」；✅「室内·家中」+ 叙述「罗博招待胡小跃…」
"""


def collect_verified_home_location_tokens(
    *,
    srt_text: str = "",
    visual_summary: str = "",
) -> set[str]:
    """从素材原文收集可写「某某家/家中」的已验证表述。"""
    blob = f"{srt_text or ''}\n{visual_summary or ''}"
    tokens: set[str] = set()
    for match in re.finditer(r"([一-龥]{2,6})家(?:中|里|的)?", blob):
        name = match.group(1)
        if name in _HOME_GENERIC_SKIP_NAMES:
            continue
        tokens.add(f"{name}家")
        tokens.add(f"{name}家中")
    return tokens


def sanitize_blueprint_home_locations(
    text: str,
    *,
    verified_tokens: set[str] | None = None,
) -> str:
    """将无素材依据的「某某家中」改为中性「室内·家中」。"""
    if not (text or "").strip():
        return text
    allowed = verified_tokens or set()

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _HOME_GENERIC_SKIP_NAMES:
            return match.group(0)
        token = f"{name}家中"
        if token in allowed or f"{name}家" in allowed:
            return token
        return "室内·家中"

    return _PERSON_HOME_IN_SCENE_RE.sub(_replace, text)


def build_plot_blueprint_material_principles(
    *,
    has_srt_subtitle: bool = False,
    has_frame_subtitle: bool = False,
    use_video_episode_analysis: bool = False,
    theme: str = "",
    settings: dict[str, Any] | None = None,
) -> str:
    """构思蓝图：整片视频分析或抽帧为主（画面/剧情），SRT 为辅（对白/时间戳）。"""
    if use_video_episode_analysis:
        visual_line = (
            f"- **整片视频分析（主·画面/剧情·{segment_policy_summary()}）**：下方 `<video_episode_analysis>` 的 "
            "`episodic_segments`（time_range / key_events / narration / environment_description）"
            "为剧情主线与画面环境第一依据"
        )
    else:
        visual_line = (
            "- **抽帧（主·画面/场景）**：时间线、action/observation、场景/昼夜/人物动作 — **画面须与抽帧一致**"
        )

    if has_srt_subtitle:
        srt_line = (
            "- **SRT 字幕（对白/时间戳主）**：下方 `<subtitles>` 为原始 SRT；"
            "剧情主线、场景关键对白、时间戳**以 SRT 为准**"
        )
    else:
        srt_line = ""

    if has_frame_subtitle:
        if has_srt_subtitle:
            frame_sub_line = (
                "- **抽帧内字幕（辅）**：`<frame_subtitles>` 与「抽帧字幕人物索引」用于声画对位；"
                "与 SRT 冲突时**台词/时间戳以 SRT 为准**，画面以抽帧为准"
            )
        else:
            frame_sub_line = (
                "- **对白字幕（取自抽帧）**：使用下方 `<subtitles>` 与「抽帧字幕人物索引」中的 "
                "subtitle_entries.text / burned_in_subtitle"
            )
    elif has_srt_subtitle:
        frame_sub_line = (
            "- **视频分析对白索引（辅）**：「整片视频分析人物索引」中的 speaker / "
            "important_dialogues 用于声画对位；与 SRT 冲突时**台词/时间戳以 SRT 为准**"
            if use_video_episode_analysis
            else ""
        )
    else:
        frame_sub_line = (
            "- **对白字幕（暂无）**：未提供 SRT；时间戳与对白须从整片视频分析 "
            "important_dialogues 推断，禁止编造"
            if use_video_episode_analysis
            else "- **对白字幕（暂无）**：未提供 SRT，且抽帧 JSON 中未提取到 subtitle_entries / 硬字幕；"
            "时间戳与对白须从抽帧 scene_segments 时间段与 observation 推断，禁止编造"
        )

    subtitle_lines = "\n".join(
        line for line in (srt_line, frame_sub_line) if line
    )
    name_unification = build_plot_blueprint_name_unification_section(
        theme=theme,
        settings=settings,
        use_video_episode_analysis=use_video_episode_analysis,
    )
    evidence_line = (
        "每个场景描述须能在整片视频分析与字幕中找到依据；禁止编造未支持的情节"
        if use_video_episode_analysis
        else "每个场景描述须能在抽帧与字幕中找到依据；禁止编造未支持的情节"
    )
    subtitle_ref = (
        "SRT 字幕"
        if has_srt_subtitle
        else ("对白线索（整片视频分析）" if use_video_episode_analysis else "对白线索（抽帧）")
    )
    return f"""## 蓝图目标（硬性）
- **主任务**：结合**人物关系表（若有） + {subtitle_ref} + {("整片视频分析" if use_video_episode_analysis else "抽帧摘要")}**，对**整段视频**做**场景分段**，并**详细描述每个场景发生的事**
- **不是**写 OST 清单或成片剪辑顺序；后者在后续「生成脚本 JSON」步骤再做

## 素材优先级
{visual_line}
- **人物关系表（文字 · 必读）**：校正人名、身份、亲属/阵营/对立关系
{subtitle_lines}
- {evidence_line}

{build_plot_blueprint_location_naming_rules()}

## 整片视频分析时间格（自适应场景）
- **段内各窗独立**：同一段约 300s 上传视频内，各 time_range 按该窗口实际画面填写，**不要**整段机械「承接上窗」
- **上传分段边界**：约 5 分钟/300 秒切多段上传时，**仅各段连接处**须衔接；同场景误换人会自动校正，见 `continuity_note` / `coverage_warnings`
- 蓝图场景分段应**跨多窗合并**为一场戏（约 30 秒–3 分钟），勿按每个短窗机械切场景
- 读视频分析时若见边界 `continuity_note` 或 `coverage_warnings`，**以字幕与连续画面为准**

{name_unification}
"""


def _append_plot_blueprint_material_sections(
    prompt: str,
    *,
    frame_summary: str,
    subtitle_excerpt: str,
    has_subtitle: bool,
    source_note: str,
    for_plot_blueprint: bool,
    frame_subtitle_excerpt: str = "",
    has_srt_subtitle: bool = False,
    has_frame_subtitle: bool = False,
    visual_summary: str = "",
    use_video_episode_analysis: bool = False,
    frame_supplement_summary: str = "",
) -> str:
    primary_summary = (visual_summary or frame_summary or "").strip()
    if for_plot_blueprint:
        if has_srt_subtitle:
            subtitle_block = (
                f"<subtitles>\n"
                f"<!-- 来源：SRT 字幕文件（对白/时间戳第一依据） -->\n"
                f"{subtitle_excerpt}\n"
                f"</subtitles>"
            )
            if has_frame_subtitle and (frame_subtitle_excerpt or "").strip():
                subtitle_block += (
                    f"\n\n<frame_subtitles>\n"
                    f"<!-- 来源：抽帧 JSON subtitle_entries / 硬字幕（声画对位参照） -->\n"
                    f"{frame_subtitle_excerpt}\n"
                    f"</frame_subtitles>"
                )
        elif has_subtitle:
            subtitle_block = (
                f"<subtitles>\n"
                f"<!-- 来源：抽帧 JSON 内 subtitle_entries / 硬字幕 -->\n"
                f"{subtitle_excerpt}\n"
                f"</subtitles>"
            )
        else:
            subtitle_block = (
                "<subtitles>\n"
                + (
                    "<!-- 暂无 SRT；对白见整片视频分析人物索引与 important_dialogues -->\n"
                    if use_video_episode_analysis
                    else "<!-- 暂无 SRT 与抽帧内字幕；对白见抽帧摘要与字幕人物索引 -->\n"
                )
                + "</subtitles>"
            )
        visual_block = ""
        if use_video_episode_analysis:
            visual_block = (
                f"<video_episode_analysis>\n"
                f"<!-- 构思蓝图 · 整片视频分析第一依据 · 须逐项对照 -->\n"
                f"{primary_summary}\n"
                f"</video_episode_analysis>"
            )
            if (frame_supplement_summary or "").strip():
                visual_block += (
                    f"\n\n<video_frame_summary>\n"
                    f"<!-- 抽帧补充参照（非第一依据） -->\n"
                    f"{frame_supplement_summary.strip()}\n"
                    f"</video_frame_summary>"
                )
        else:
            visual_block = (
                f"<video_frame_summary>\n"
                f"<!-- 构思蓝图 · 画面/场景第一依据 · 须逐项对照 -->\n"
                f"{primary_summary}\n"
                f"</video_frame_summary>"
            )
        return (
            prompt
            + f"""

{visual_block}

{subtitle_block}
"""
        )
    return (
        prompt
        + f"""

<video_frame_summary>
{primary_summary}
</video_frame_summary>

<subtitles>
<!-- 来源：{source_note} -->
{subtitle_excerpt}
</subtitles>
"""
    )


def analyze_subtitle_with_frames(
    *,
    subtitle_content: str,
    frame_markdown: str,
    video_theme: str = "",
    append_custom_prompt: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
    documentary_settings: Optional[dict[str, Any]] = None,
    analysis_style: str = "documentary",
    frame_json_path: str | None = None,
    for_plot_blueprint: bool = False,
    source_duration_sec: float | None = None,
    video_episode_json_path: str | None = None,
    video_episode_markdown: str = "",
) -> str:
    """文本模型：结合字幕与整片视频分析/抽帧摘要做对照分析。

    for_plot_blueprint=True 时优先以整片视频分析为主；否则以抽帧画面为主。
    有 SRT 时对白/时间戳以 SRT 为准。
    """
    cfg = documentary_settings or get_documentary_settings()
    is_short_drama = (analysis_style or "").strip().lower() == "short_drama"

    has_subtitle = False
    has_srt_subtitle = False
    has_frame_subtitle = False
    subtitle_excerpt = ""
    frame_subtitle_excerpt = ""
    subtitle_source = ""
    if for_plot_blueprint:
        srt_text, frame_text, subtitle_source = resolve_subtitles_for_plot_blueprint(
            subtitle_content=subtitle_content,
            frame_json_path=frame_json_path,
        )
        has_srt_subtitle = bool(srt_text)
        has_frame_subtitle = bool(frame_text)
        has_subtitle = has_srt_subtitle or has_frame_subtitle
        max_sub_chars = int(cfg.get("subtitle_analysis_max_subtitle_chars", 12000) or 12000)
        if has_srt_subtitle:
            subtitle_excerpt = truncate_subtitle_content(srt_text, max_sub_chars)
        if has_frame_subtitle:
            frame_subtitle_excerpt = truncate_subtitle_content(frame_text, max_sub_chars)
            if not has_srt_subtitle:
                subtitle_excerpt = frame_subtitle_excerpt
    else:
        resolved_content, subtitle_source = resolve_subtitle_content_for_plot_analysis(
            subtitle_content=subtitle_content,
            frame_json_path=frame_json_path,
        )
        has_subtitle = bool((resolved_content or "").strip())
        if has_subtitle:
            subtitle_excerpt = truncate_subtitle_content(
                resolved_content,
                int(cfg.get("subtitle_analysis_max_subtitle_chars", 12000) or 12000),
            )

    video_episode_path = (video_episode_json_path or "").strip()
    video_episode_raw_markdown = (video_episode_markdown or "").strip()
    video_episode_artifact: dict[str, Any] = {}
    use_video_episode_analysis = False
    video_episode_summary = ""
    if video_episode_path and os.path.isfile(video_episode_path):
        try:
            video_episode_artifact = load_video_episode_analysis_artifact(video_episode_path)
            if not video_episode_raw_markdown:
                video_episode_raw_markdown = build_video_episode_analysis_markdown(
                    video_episode_artifact
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"读取整片视频分析失败: {video_episode_path} ({exc})")

    frame_max_chars = int(cfg.get("subtitle_analysis_max_frame_chars", 8000))
    frame_sampling = "head"
    if for_plot_blueprint or is_short_drama:
        frame_sampling = str(
            cfg.get("subtitle_analysis_frame_sampling", "timeline_uniform")
            or "timeline_uniform"
        ).strip().lower()
    if for_plot_blueprint:
        frame_max_chars = max(
            frame_max_chars,
            int(cfg.get("subtitle_analysis_max_frame_chars", 20000) or 20000),
        )
    frame_summary = summarize_frame_markdown(
        frame_markdown,
        frame_max_chars,
        sampling=frame_sampling,
        frame_json_path=frame_json_path,
    )
    if video_episode_raw_markdown:
        video_max_chars = int(cfg.get("subtitle_analysis_max_video_episode_chars", 30000) or 30000)
        video_episode_summary = summarize_video_episode_markdown(
            video_episode_raw_markdown,
            video_max_chars,
        )
        use_video_episode_analysis = video_episode_summary_usable(video_episode_summary)

    visual_summary = video_episode_summary if use_video_episode_analysis else frame_summary
    visual_source_label = "整片视频分析" if use_video_episode_analysis else "抽帧"
    frame_supplement_summary = frame_summary if use_video_episode_analysis else ""

    if use_video_episode_analysis:
        blueprint_read_hint = (
            "请**深读 SRT 字幕（对白/时间戳第一依据）**并**对照下方整片视频分析（剧情/画面/环境第一依据）**"
            if has_srt_subtitle
            else "请**深读下方整片视频分析（第一依据）**"
        )
        blueprint_analysis_rules = (
            f"""- **先通读 SRT 字幕**：梳理剧情主线、对白与关键台词
- **再对照整片视频分析**：各 time_range 窗的 key_events、narration、environment_description 须逐项利用
- **交叉验证**：台词/时间戳以 SRT 为准，画面/环境/旁白以整片视频分析为准"""
            if has_srt_subtitle
            else """- **先通读整片视频分析**：按 episodic_segments 时间线梳理场景、人物、动作与环境"""
        )
        blueprint_timestamp_rule = (
            "**优先来自 SRT**，格式 **HH:MM:SS,mmm-HH:MM:SS,mmm**（禁止 `-->`）；须落在整片视频分析时间边界内"
            if has_srt_subtitle
            else "**仅**能使用整片视频分析与 important_dialogues 中的真实时间；格式 **HH:MM:SS,mmm-HH:MM:SS,mmm**（禁止 `-->`）"
        )
        blueprint_timeline_label = (
            "SRT + 整片视频分析 chronology" if has_srt_subtitle else "整片视频分析 chronology"
        )
        blueprint_visual_anchor = "整片视频分析同段 episodic_segments"
        blueprint_evidence_source = "整片视频分析"
    else:
        blueprint_read_hint = (
            "请**深读 SRT 字幕（对白/时间戳第一依据）**并**对照下方抽帧摘要（画面/场景第一依据）**"
            if has_srt_subtitle
            else "请**深读下方抽帧摘要（第一依据）**"
        )
        blueprint_analysis_rules = (
            """- **先通读 SRT 字幕**：梳理剧情主线、对白与关键台词
- **再对照抽帧**：场景、人物、动作、昼夜须与锚点表一致
- **交叉验证**：抽帧 subtitle_entries 与 SRT 不一致时，**台词/时间戳以 SRT 为准**，画面以抽帧为准"""
            if has_srt_subtitle
            else """- **先通读抽帧**：按时间线梳理场景、人物、动作、subtitle_entries / 硬字幕"""
        )
        blueprint_timestamp_rule = (
            "**优先来自 SRT**，格式 **HH:MM:SS,mmm-HH:MM:SS,mmm**（禁止 `-->`）；须落在抽帧时间边界上限内"
            if has_srt_subtitle
            else "**仅**能使用上方「抽帧时间边界与场景锚点」与 subtitle_entries；格式 **HH:MM:SS,mmm-HH:MM:SS,mmm**（禁止 `-->`）"
        )
        blueprint_timeline_label = "SRT + 抽帧 chronology" if has_srt_subtitle else "抽帧 chronology"
        blueprint_visual_anchor = "锚点表同段抽帧"
        blueprint_evidence_source = "抽帧"

    if for_plot_blueprint:
        if use_video_episode_analysis:
            if not video_episode_summary_usable(video_episode_summary):
                logger.warning("构思蓝图：整片视频分析摘要为空，无法生成")
                return ""
        elif not _frame_summary_usable(frame_summary):
            logger.warning("构思蓝图：抽帧摘要为空，无法生成")
            return ""
    elif not has_subtitle:
        return ""
    theme = (video_theme or "").strip() or "本视频"
    min_chars = max(200, int(cfg.get("subtitle_analysis_min_chars", 500) or 500))
    if for_plot_blueprint and is_fazu2_compact_settings(cfg):
        min_chars = max(min_chars, 1800)
    max_tokens = max(1024, int(cfg.get("subtitle_analysis_max_tokens", 4096) or 4096))
    character_lexicon_block = ""
    character_lexicon_data: dict = {}
    drama_knowledge_block = ""
    drama_known_names: set[str] = set()
    sd_settings = get_short_drama_settings()

    if is_short_drama or (for_plot_blueprint and is_fazu2_compact_settings(cfg)):
        if is_short_drama:
            min_chars = max(
                min_chars,
                int(cfg.get("subtitle_analysis_short_drama_min_chars", 2000) or 2000),
            )
            max_tokens = max(
                max_tokens,
                int(cfg.get("subtitle_analysis_short_drama_max_tokens", 8192) or 8192),
            )
        if for_plot_blueprint:
            drama_knowledge_block, drama_known_names = (
                build_plot_blueprint_character_relationship_table_section(
                    theme,
                    cfg,
                    use_video_episode_analysis=use_video_episode_analysis,
                )
            )
        else:
            drama_knowledge_block, drama_known_names = build_short_drama_drama_knowledge_section(
                theme,
                cfg,
                use_video_episode_analysis=use_video_episode_analysis,
            )

    json_path = (frame_json_path or "").strip()
    lexicon_max = int(cfg.get("subtitle_analysis_frame_lexicon_chars", 4000) or 4000)
    if use_video_episode_analysis and video_episode_artifact:
        character_lexicon_block, character_lexicon_data = (
            build_video_episode_character_lexicon_markdown(
                video_episode_artifact,
                max_chars=lexicon_max,
            )
        )
    elif json_path and (is_short_drama or is_fazu2_compact_settings(cfg)):
        character_lexicon_block, character_lexicon_data = build_frame_subtitle_lexicon_markdown(
            json_path,
            max_chars=lexicon_max,
        )

    source_note = "SRT 字幕文件" if subtitle_source == "srt_file" else "抽帧 JSON 内硬字幕/对位字幕"
    time_bounds_section = ""
    frame_time_bounds: dict[str, Any] = {}
    srt_time_bounds: dict[str, int] = {"min_ms": 0, "max_ms": 0}
    if for_plot_blueprint:
        if has_srt_subtitle:
            srt_time_bounds = collect_subtitle_time_bounds(srt_text)
    video_time_bounds: dict[str, Any] = {}
    srt_entries_for_validation: list[dict[str, Any]] = []
    if for_plot_blueprint and use_video_episode_analysis and video_episode_artifact:
        video_time_bounds = collect_video_episode_time_bounds(video_episode_artifact)
        time_bounds_section = build_plot_blueprint_dual_time_alignment_section(
            video_episode_artifact,
            srt_text if has_srt_subtitle else "",
        )
        if source_duration_sec and source_duration_sec > 0:
            time_bounds_section += f"\n- 源视频实测时长：**{source_duration_sec:.1f}s**"
        if has_srt_subtitle:
            srt_entries_for_validation = parse_subtitle_entries_for_blueprint(srt_text)
    elif for_plot_blueprint and json_path:
        frame_time_bounds = collect_frame_analysis_time_bounds(json_path)
        time_bounds_section = build_frame_analysis_time_bounds_section(
            json_path,
            source_duration_sec=source_duration_sec,
            srt_max_ms=int(srt_time_bounds.get("max_ms") or 0) or None,
            srt_min_ms=int(srt_time_bounds.get("min_ms") or 0),
        )
    analysis_label = (
        (
            "字幕×整片视频分析×场景分段"
            if use_video_episode_analysis
            else "字幕×抽帧×场景分段"
        )
        if for_plot_blueprint
        else (
            (
                "字幕×整片视频分析×剧情构思"
                if use_video_episode_analysis
                else "字幕×抽帧×剧情构思"
            )
            if is_short_drama
            else (
                "字幕×整片视频分析对照分析"
                if use_video_episode_analysis
                else "字幕×抽帧对照分析"
            )
        )
    )
    lexicon_log_label = (
        "视频分析人物索引"
        if use_video_episode_analysis
        else "抽帧字幕索引"
    )
    if for_plot_blueprint:
        srt_info = (
            f"SRT 字幕有（{len(subtitle_excerpt)} 字）"
            if has_srt_subtitle
            else "SRT 字幕无"
        )
        if use_video_episode_analysis:
            name_count = len(character_lexicon_data.get("names") or [])
            extra_info = (
                f"视频分析人物 {name_count} 个"
                if name_count
                else "视频分析人物索引无"
            )
        else:
            extra_info = (
                f"抽帧内字幕有（{len(frame_subtitle_excerpt)} 字）"
                if has_frame_subtitle
                else "抽帧内字幕无"
            )
        visual_input = (
            f"整片视频分析摘要 {len(video_episode_summary)} 字"
            if use_video_episode_analysis
            else f"抽帧摘要 {len(frame_summary)} 字（采样 {frame_sampling}）"
        )
        logger.info(
            f"{analysis_label}输入：{visual_input}，"
            f"{srt_info}，{extra_info}，要求输出 ≥{min_chars} 字"
            + (
                f"，{lexicon_log_label} {len(character_lexicon_block)} 字"
                if character_lexicon_block
                else ""
            )
            + (
                f"，人物关系表 {len(drama_knowledge_block)} 字"
                if drama_knowledge_block and for_plot_blueprint
                else (f"，剧集知识库 {len(drama_knowledge_block)} 字" if drama_knowledge_block else "")
            )
        )
        if not has_subtitle:
            fallback_note = (
                "对白与时间戳只能从整片视频分析 important_dialogues 推断"
                if use_video_episode_analysis
                else "对白与时间戳只能从抽帧 observation 推断"
            )
            logger.warning(
                "构思蓝图：未提供 SRT"
                + (
                    "，须从整片视频分析 important_dialogues 推断对白与时间戳"
                    if use_video_episode_analysis
                    else " 且抽帧 JSON 无 subtitle_entries / 硬字幕，"
                    + fallback_note
                )
            )
    else:
        visual_log = (
            f"整片视频分析摘要 {len(video_episode_summary)} 字"
            if use_video_episode_analysis
            else f"抽帧摘要 {len(frame_summary)} 字（采样 {frame_sampling}）"
        )
        logger.info(
            f"{analysis_label}输入：{visual_log}，"
            f"字幕 {'有' if has_subtitle else '无'}"
            + (f"（{source_note}，{len(subtitle_excerpt)} 字）" if has_subtitle else "")
            + f"，要求输出 ≥{min_chars} 字"
            + (
                f"，{lexicon_log_label} {len(character_lexicon_block)} 字"
                if character_lexicon_block
                else ""
            )
            + (f"，剧集知识库 {len(drama_knowledge_block)} 字" if drama_knowledge_block else "")
        )
        if has_subtitle and len(subtitle_excerpt) < 100:
            source_hint = (
                "请确认 SRT 已选用或整片视频分析 JSON 有效"
                if use_video_episode_analysis
                else "请确认 SRT 已选用或抽帧 JSON 含 subtitle_entries"
            )
            logger.warning(
                f"字幕内容过短（{len(subtitle_excerpt)} 字，来源 {source_note}），"
                + source_hint
            )

    if progress_callback:
        if for_plot_blueprint:
            rel_hint = "人物关系表、" if drama_knowledge_block.strip() else ""
            if use_video_episode_analysis:
                if has_srt_subtitle:
                    progress_callback(
                        f"正在联合分析{rel_hint}字幕与整片视频分析做场景分段..."
                    )
                else:
                    progress_callback(
                        f"正在联合分析{rel_hint}整片视频分析做场景分段..."
                    )
            elif has_srt_subtitle:
                progress_callback(
                    f"正在联合分析{rel_hint}字幕与抽帧做场景分段..."
                )
            else:
                progress_callback(f"正在联合分析{rel_hint}抽帧做场景分段...")
        else:
            progress_callback("正在分析字幕并对照画面素材...")

    compact = is_compact_documentary_settings(cfg)
    append_text = resolve_append_custom_prompt(append_custom_prompt, cfg)
    append_block = ""
    if append_text:
        append_block = f"""## 用户追加要求（最高优先级 · 策划蓝图须优先落实）
{append_text}
若已指定开头高潮/爆燃场面，「开头高潮方案」必须写该场面，不得另选。

"""
    if is_fazu2_compact_settings(cfg):
        min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
        ost1_dur_min = int(cfg.get("ost1_duration_min", 8) or 8)
        ost1_dur_max = int(cfg.get("ost1_duration_max", 18) or 18)
        lexicon_section = (
            f"\n{character_lexicon_block.strip()}\n\n"
            if character_lexicon_block.strip()
            else ""
        )
        knowledge_section = (
            f"\n{drama_knowledge_block.strip()}\n\n"
            if drama_knowledge_block.strip()
            else ""
        )
        if for_plot_blueprint:
            min_chars = max(1500, int(min_chars * 0.75))
            principles = build_plot_blueprint_material_principles(
                has_srt_subtitle=has_srt_subtitle,
                has_frame_subtitle=has_frame_subtitle,
                use_video_episode_analysis=use_video_episode_analysis,
                theme=theme,
                settings=cfg,
            )
            timestamp_rule = blueprint_timestamp_rule
            prompt = f"""{append_block}{principles}
{time_bounds_section}
{knowledge_section}{lexicon_section}你是资深剧集内容分析师。{blueprint_read_hint}，**先通读上方人物关系表**，再结合字幕与{visual_source_label}，输出**全片场景分段蓝图**（**2500–6000 字**，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

## 分析原则
{blueprint_analysis_rules}
- **核心**：按时间顺序切分**完整场景**（一场戏 = 一段），**详细写清每段发生什么**（剧情、动作、冲突、人物关系互动、环境/昼夜）
- 人名/关系须与**人物关系表**一致；谐音/ASR 错字须归并（胡晓月→胡小跃、罗伯→罗博）；画面/环境须与{visual_source_label}一致
- **时间戳**：{timestamp_rule}

必须按以下标题填写：

## 主要人物表
- 本集/本片**实际出现**的人物：规范全名 + 身份/关系 + 性别（对照人物关系表，每人一行；禁止 ASR 谐音拆成两人）

## 全片场景分段
按**原片时间顺序**覆盖整段视频，约 **10–25 个场景**（按情节密度划分，勿按短窗机械切分）。

每个场景用三级标题，格式如下（**须写满各字段**）：

### 场景 1 · `00:00:00-00:01:30` · 地点/环境简述（勿臆测「某某家中」，不明则写「室内·家中」）
- **出场人物**：
- **本场景发生的事**：（**不少于 80 字**，详细叙述：谁做了什么、冲突/转折、情绪、与前后情节的因果；综合 SRT + {visual_source_label}）
- **关键对白**：（原文 1–3 句 + 时间戳；无对白写「无对白」）
- **画面/环境要点**：（摘自{visual_source_label}：光线、场景、人物动作表情）

（依次写场景 2、场景 3 … 直至覆盖全片）

## 剧情主线摘要
- 用 3–5 句话概括全片故事线与核心冲突

## 写脚本参考（可选）
- 名场面/高潮 moment（时间段 + 一句话说明；如楼顶跳楼等）
- 人名/关系/声画易错提醒

禁忌：不要警员1/说话人1；不要镜头/导演分析；不要编造人物关系表与素材中不存在的人名/情节/时间戳；**不要无依据写「某某家中」**
"""
            system_prompt = (
                f"你是剧集内容分析师，擅长结合人物关系表、字幕与{visual_source_label}做全片场景分段与详述。"
                "输出 Markdown，不要 JSON，不要中途截断。"
            )
        else:
            subtitle_source_hint = (
                "原始 SRT 字幕文件"
                if subtitle_source == "srt_file"
                else "抽帧 JSON 内硬字幕（原样提取，无 SRT 时以此为准）"
            )
            prompt = f"""{append_block}{lexicon_section}你是电视剧「高潮前置型」解说策划。请**以完整字幕为第一依据**（当前来源：{subtitle_source_hint}），对照抽帧摘要补充画面信息，输出一份**脚本生成蓝图**（800–1200 字，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

必须按以下标题逐项填写（缺一不可）：

## 本集识别
- 作品名；是否**全剧第 1 集**（决定转场句用固定句还是自拟）

## 主要人物表
- 姓名/昵称/关系称呼 + 身份/关系 + **性别（据字幕/抽帧）**（如胡小跃-刑警-男、老叶-局长-男、文妈-养母-女）
- 字幕中的**小名、昵称、关系称呼**（老叶、小跃、师傅等）须原样使用，可对照人物关系表映射到全名，禁止擅自改名

## 开头高潮方案（→ JSON 第 1 段 OST=1）
- **默认优先**：胡小跃**楼顶跳楼牺牲**（夜色楼顶纵身跃下），金句优先「天就快亮了。」+ 字幕精确时间戳
- **禁止**用中段台词（如「你跟我说这是狗贩子」）顶替跳楼作第 1 段；此类放正叙 OST=1
- 动作描述（结合抽帧；昼夜/光线须与画面一致）
- **转场句建议**：第2段用「宝子们」+「故事，得从头讲起。」
- 若用户追加要求已指定开头名场面，本节**只写该场面**

## 正叙时间线（→ JSON 主体 OST=0，按片头时间顺序）
- 至少 **8–12 个情节点**，每点含：时间段、人名、事件摘要
- 标注哪些点适合加小钩子（「您猜怎么着？」等）

## OST=1 金句清单（{min_ost1}–{max_ost1} 条，时间戳必须来自字幕）
- 每条：说话人姓名 + 台词原文 + **精确时间戳** + 用途（开头/中段爆点/高潮复现）
- **timestamp 须为连续对白块，时长 {ost1_dur_min}–{ost1_dur_max} 秒**；禁止 2–6 秒单句碎段
- 若单句不足 {ost1_dur_min} 秒，须合并相邻字幕为一条连续区间

## 高潮复现方案（→ JSON 正叙相应位置 · 硬性）
- 正叙推进到**第 1 段开篇高潮**的原片时刻时，须规划 **1 个 OST=1 复现段**
- 复现段与第 1 段 **timestamp / 台词 / 画面相同**（picture 可加「【复现】」）；不是另选中段台词
- 收尾可再次呼应，但**不能替代**正叙中的这次复现

## 高潮之后 + 下集钩子（→ JSON 末段）
- 本集高潮之后的关键情节（2–3 点）
- 下集悬念/预告（1–2 句）

## 声画对位注意
- 字幕与画面不一致处、需以字幕为准的台词、需以画面为准的动作
- **环境对位**：逐段标注昼夜/光线/天气（如「天台·白天·日光」）；标出易写错的镜头（画面白天勿写夜风刺骨）
- 正叙开场勿写「第几集」，用「宝子们，一起来看《作品名》。」即可

禁忌：不要警员1/说话人1；不要镜头/导演分析；不要然后/接着；不要编造时间戳；不要臆造与抽帧不符的环境氛围
"""
            system_prompt = (
                "你是影视解说策划，擅长高潮前置型短视频脚本。"
                "输出结构化策划蓝图，时间戳必须来自字幕原文，不要 JSON。"
            )
    elif compact:
        if for_plot_blueprint:
            principles = build_plot_blueprint_material_principles(
                has_srt_subtitle=has_srt_subtitle,
                has_frame_subtitle=has_frame_subtitle,
                use_video_episode_analysis=use_video_episode_analysis,
                theme=theme,
                settings=cfg,
            )
            compact_subtitle_hint = (
                f"SRT 字幕与{visual_source_label}摘要"
                if has_srt_subtitle
                else f"{visual_source_label}摘要"
            )
            prompt = f"""{append_block}{principles}请以**{compact_subtitle_hint}**为主、剧集对照为辅，输出供精剪脚本使用的对照分析（350–550 字）。

作品/主题：{theme}

请包含：剧情主线（{"SRT + 抽帧时间线" if has_srt_subtitle else "抽帧时间线"}）、建议 OST=1 时刻（带时间戳）、写脚本注意点。
"""
            system_prompt = (
                "你是资深影视解说分析师，擅长字幕与抽帧交叉验证，输出简洁可执行，不要 JSON。"
                if has_srt_subtitle
                else "你是资深影视解说分析师，擅长以抽帧梳理剧情，输出简洁可执行，不要 JSON。"
            )
        else:
            prompt = f"""请以**原始字幕为主**、抽帧画面摘要为辅，输出供精剪脚本使用的对照分析（350–550 字）。

作品/主题：{theme}

请包含剧情主线、建议 OST=1 时刻（带时间戳）、写脚本注意点。
"""
            system_prompt = (
                "你是资深影视解说分析师，输出简洁可执行，不要 JSON。"
            )
    elif is_short_drama:
        min_ost1_plot = estimate_min_ost1_entries_for_plot(sd_settings)
        min_minutes = int(sd_settings.get("target_output_minutes_min", 8) or 8)
        max_minutes = int(sd_settings.get("target_output_minutes_max", 13) or 13)
        ost1_dur_min = int(sd_settings.get("ost1_duration_min", 8) or 8)
        ost1_dur_max = int(sd_settings.get("ost1_duration_max", 18) or 18)
        lexicon_section = (
            f"\n{character_lexicon_block.strip()}\n\n"
            if character_lexicon_block.strip()
            else ""
        )
        knowledge_section = (
            f"\n{drama_knowledge_block.strip()}\n\n"
            if drama_knowledge_block.strip()
            else ""
        )
        if for_plot_blueprint:
            min_chars = max(1500, int(min_chars * 0.75))
            principles = build_plot_blueprint_material_principles(
                has_srt_subtitle=has_srt_subtitle,
                has_frame_subtitle=has_frame_subtitle,
                use_video_episode_analysis=use_video_episode_analysis,
                theme=theme,
                settings=cfg,
            )
            prompt = f"""{append_block}{principles}
{time_bounds_section}
{knowledge_section}{lexicon_section}你是资深剧集内容分析师。{blueprint_read_hint}，**先通读人物关系表**，再联合字幕与{visual_source_label}，输出**全片场景分段蓝图**（**2500–6000 字**，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

## 分析原则
{blueprint_analysis_rules}
- **核心**：按时间顺序切分**完整场景**（一场戏 = 一段），**详细写清每段发生什么**（剧情、动作、冲突、人物关系互动、环境/昼夜）
- 人名/关系须与**人物关系表**一致；画面/环境须与{visual_source_label}一致；对白摘录须与 SRT 一致
- 时间戳格式：**HH:MM:SS,mmm-HH:MM:SS,mmm** 或视频格 **HH:MM:SS-HH:MM:SS**（禁止 `-->`）

必须按以下标题填写：

## 主要人物表
- 本集/本片**实际出现**的人物：规范全名 + 身份/关系 + 性别（对照人物关系表，每人一行）

## 全片场景分段
按**原片时间顺序**覆盖整段视频，约 **10–25 个场景**（按情节密度划分，勿按短窗机械切分）。

每个场景用三级标题，格式如下（**须写满各字段**）：

### 场景 1 · `00:00:00-00:01:30` · 地点/环境简述（勿臆测「某某家中」，不明则写「室内·家中」）
- **出场人物**：
- **本场景发生的事**：（**不少于 80 字**，详细叙述：谁做了什么、冲突/转折、情绪、与前后情节的因果；综合 SRT + {visual_source_label}）
- **关键对白**：（原文 1–3 句 + 时间戳；无对白写「无对白」）
- **画面/环境要点**：（摘自{visual_source_label}：光线、场景、人物动作表情）

（依次写场景 2、场景 3 … 直至覆盖全片）

## 剧情主线摘要
- 用 3–5 句话概括全片故事线与核心冲突

## 写脚本参考（可选）
- 名场面/高潮 moment（时间段 + 一句话说明）
- 人名/关系/声画易错提醒

禁忌：不要警员1/说话人1；不要编造人物关系表与素材中不存在的人名/情节；**不要无依据写「某某家中」**
"""
            system_prompt = (
                f"你是剧集内容分析师，擅长结合人物关系表、字幕与{visual_source_label}做全片场景分段与详述。"
                "输出 Markdown，不要 JSON，不要中途截断。"
            )
        else:
            character_index_label = (
                "整片视频分析人物索引"
                if use_video_episode_analysis
                else "抽帧字幕人物索引"
            )
            visual_material_label = visual_source_label
            prompt = f"""{append_block}{knowledge_section}{lexicon_section}你是顶级短剧解说策划。请**先通读上方「人物关系表」**，再**同时深读字幕、{visual_material_label}与「{character_index_label}」**，交叉验证后输出供写 JSON 脚本的**完美剧情构思方案**（**2000–3500 字**，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

## 分析原则（硬性）
- **先读人物关系表，再对照字幕/{visual_material_label}**：人物身份、亲属、阵营、对立关系须与人物关系表一致；禁止臆造或张冠李戴
- **人名与对白以人物关系表 + {character_index_label} + 原始字幕为准**：含**全名、小名、昵称、关系称呼**（老叶、小跃、师傅、文妈等）；**禁止** ASR 谐音（胡晓月→胡小跃、罗伯→罗博）
- **所有时间戳**必须来自字幕，格式统一为 **HH:MM:SS,mmm-HH:MM:SS,mmm**（用连字符 `-`，**禁止** SRT 箭头 `-->`）
- **{visual_material_label}画面**：人物性别/表情/动作/场景/昼夜光线写入画面要点；**禁止**写「场景1/场景2」等采样编号，改用 **原片时间段+地点**
- 字幕与画面冲突：**台词/时间戳以字幕为准**，画面描述以{visual_material_label}为准
- 产出**可执行脚本蓝图**，不是 3–5 段空泛总结

必须按以下标题逐项填写（缺一不可）：

## 主要人物表
- 姓名（**须出现在人物关系表或{character_index_label}中**）+ 身份/关系（**与人物关系表一致**）+ **性别** + 外貌/气质（{visual_material_label}）
- 相似或简称须与人物关系表一致（如胡小跃禁止写胡小月/胡晓月；秦枫禁止秦峰；罗博禁止罗伯），**禁止混用不同角色的名字**

## 开头高潮方案（→ JSON 第 1 段 OST=1）
- 从**全片**选最爆燃段落；金句原文 + **HH:MM:SS,mmm-HH:MM:SS,mmm** + 画面（时间段+地点+动作）
- 第 1 段纯原声，禁止旁白

## 原片时间线（按字幕 chronology，供选材）
- 至少 **12–16 个情节点**；每点：**时间段 + 事件 + 画面要点（勿写场景N）**
- 标注适合 OST=0 串场 vs OST=1 原声

## 成片叙事顺序方案（→ JSON 的 `_id` 播放顺序 · 重要）
- **`_id` = 播放顺序**；`timestamp` = 原片裁剪区间，二者独立
- 写出至少 **18–28 个 `_id` 段**规划（含 OST=0/1），估算能否支撑 **{min_minutes}–{max_minutes} 分钟**成片
- 每段标注 **OST=0 或 OST=1**（仅写 0/1，勿写错别字）
- 第 1 段倒叙爆点 → 第 2 段「宝子们，我们开始看{theme}。」→ 末段「宝子们，我们下期再见！」

## 建议保留原声 OST=1（成片原声时长约 70%）
- **至少 {min_ost1_plot} 条**（含时间戳）；每条：说话人 + 台词原文 + **HH:MM:SS,mmm-HH:MM:SS,mmm** + 画面张力 + 用途
- **每条 timestamp 须为连续对白块，时长 {ost1_dur_min}–{ost1_dur_max} 秒**（禁止 2–6 秒碎段）
- 单句对白不足 {ost1_dur_min} 秒时，须合并相邻字幕为一条连续区间
- 同场连续对白可成块规划

## 解说 OST=0 脉络规划（成片解说时长约 30%）
- 每条 ≥20 字；注明对应原片时间段与承上启下
- **铺垫下一段 OST=1**：写清取画起点 = 该 OST=1 开始时间 − 约 {int(cfg.get("ost0_lead_before_ost1_sec", 10) or 10)} 秒
- **点评上一段 OST=1**：写清与上一段原声同场 `timestamp`

## 声画对位注意
- 字幕/画面不一致处；昼夜光线易错点；声画反差 moment

禁忌：不要警员1/说话人1；不要「场景N」编号；不要 `-->` 时间戳；不要编造人名/时间戳
"""
            system_prompt = (
                "你是短剧解说策划，擅长结合人物关系表与字幕×画面联合分析。"
                "须先依据上方人物关系表熟悉全剧，再输出完整 Markdown 蓝图；"
                "人名、关系、对白归属必须准确，不要 JSON，不要中途截断。"
            )
    else:
        prompt = f"""你是一位拥有 30 年经验的资深解说员，请以**原始字幕为主**、抽帧画面摘要为辅，输出供写脚本使用的对照分析（300–500 字）。

作品/主题：{theme}

请包含：
1. 剧情主线、关键人物与冲突（结合字幕与画面）
2. 字幕对白与画面是否一致或有反差（可点出「声画对位」moment）
3. **建议保留原声 OST=1** 的时刻（标注大致时间戳，来自字幕真实范围）
4. 写解说时的注意点（哪些地方以字幕为准、哪些以画面为准）
"""
        system_prompt = (
            "你是资深影视解说分析师，擅长字幕与画面交叉验证。"
            "输出简洁、可执行，不要 JSON。"
        )

    prompt = _append_plot_blueprint_material_sections(
        prompt,
        frame_summary=frame_summary,
        subtitle_excerpt=subtitle_excerpt,
        has_subtitle=has_subtitle,
        source_note=source_note,
        for_plot_blueprint=for_plot_blueprint,
        frame_subtitle_excerpt=frame_subtitle_excerpt,
        has_srt_subtitle=has_srt_subtitle,
        has_frame_subtitle=has_frame_subtitle,
        visual_summary=visual_summary,
        use_video_episode_analysis=use_video_episode_analysis,
        frame_supplement_summary=frame_supplement_summary,
    )

    text_provider = config.app.get("text_llm_provider", "openai").lower()
    api_key = config.app.get(f"text_{text_provider}_api_key")
    model = config.app.get(f"text_{text_provider}_model_name")
    base_url = config.app.get(f"text_{text_provider}_base_url", "")
    if not api_key or not model:
        logger.warning("未配置文本模型，跳过字幕×画面对照分析")
        return ""

    analysis_temperature = 0.35 if is_fazu2_compact_settings(cfg) else 0.7
    fazu2 = is_fazu2_compact_settings(cfg)
    validation_settings = cfg if (fazu2 or for_plot_blueprint) else sd_settings
    if for_plot_blueprint:
        max_attempts = 1
    elif is_short_drama:
        max_attempts = 3
    else:
        max_attempts = 3 if fazu2 else 2
    current_prompt = prompt
    analysis = ""
    last_validation: dict = {}
    source_duration_ms = (
        int(float(source_duration_sec) * 1000)
        if source_duration_sec and source_duration_sec > 0
        else None
    )
    if use_video_episode_analysis and video_time_bounds.get("max_ms"):
        frame_max_ms = int(video_time_bounds.get("max_ms") or 0) or None
        frame_min_ms = int(video_time_bounds.get("min_ms") or 0)
    else:
        frame_max_ms = int(frame_time_bounds.get("max_ms") or 0) or None
        frame_min_ms = int(frame_time_bounds.get("min_ms") or 0)
    video_segment_ranges = (
        list(video_time_bounds.get("segment_ranges") or [])
        if use_video_episode_analysis
        else None
    )
    character_index_label = (
        "整片视频分析人物索引"
        if use_video_episode_analysis
        else "抽帧字幕人物索引"
    )
    canonical_names_list = sorted(
        str(name) for name in (character_lexicon_data.get("names") or set()) if str(name).strip()
    )
    canonical_names_retry_hint = ""
    if canonical_names_list:
        canonical_names_retry_hint = (
            f"- 规范人名须严格使用（来自{character_index_label}）："
            + "、".join(canonical_names_list[:40])
            + "\n"
        )

    try:
        for attempt in range(max_attempts):
            result = _run_async_safely(
                UnifiedLLMService.generate_text,
                prompt=current_prompt,
                system_prompt=system_prompt,
                provider=text_provider,
                temperature=analysis_temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=base_url,
                for_script=True,
            )
            analysis = correct_name_mistakes_in_text((result or "").strip())
            if for_plot_blueprint:
                verified_homes = collect_verified_home_location_tokens(
                    srt_text=srt_text if has_srt_subtitle else "",
                    visual_summary=visual_summary or frame_summary or "",
                )
                analysis = sanitize_blueprint_home_locations(
                    analysis,
                    verified_tokens=verified_homes,
                )

            if for_plot_blueprint:
                last_validation = validate_plot_blueprint(
                    analysis,
                    source_duration_ms=source_duration_ms,
                    frame_max_ms=frame_max_ms,
                    frame_min_ms=frame_min_ms,
                    srt_max_ms=int(srt_time_bounds.get("max_ms") or 0) or None,
                    srt_min_ms=int(srt_time_bounds.get("min_ms") or 0),
                    settings=validation_settings,
                    lexicon=character_lexicon_data,
                    drama_known_names=drama_known_names,
                    min_chars=min_chars,
                    require_all_sections=is_short_drama
                    or not is_fazu2_compact_settings(cfg),
                    video_segment_ranges=video_segment_ranges,
                    srt_entries=srt_entries_for_validation or None,
                    use_video_episode_analysis=use_video_episode_analysis,
                    relaxed=True,
                )
                emit_plot_blueprint_validation_report(last_validation)
                if last_validation.get("ok"):
                    emit_plot_analysis_full_text(
                        analysis,
                        title=f"{analysis_label} · 场景分段蓝图",
                    )
                    logger.info(f"{analysis_label}完成，约 {len(analysis)} 字")
                    return analysis
                if attempt + 1 >= max_attempts:
                    logger.warning(
                        "场景分段蓝图校验未完全通过，返回首次输出（请人工核对场景与人名）"
                    )
                    emit_plot_analysis_full_text(
                        analysis,
                        title=f"{analysis_label} · 场景分段蓝图（校验未通过）",
                    )
                    return analysis
                issue_lines = "\n".join(
                    f"- {item}" for item in (last_validation.get("issues") or [])
                )
                cap_label = ""
                if last_validation.get("hard_cap_ms"):
                    from app.services.srt_utils import format_timestamp_ms

                    cap_label = format_timestamp_ms(int(last_validation["hard_cap_ms"]))
                cap_hint = (
                    f"所有 timestamp 不得超过 **{cap_label}**；"
                    if cap_label
                    else ""
                )
                time_source_hint = (
                    "「全片场景分段」须覆盖整片并按情节密度划分（约 10–25 段）；"
                    "每段「本场景发生的事」不少于 80 字；"
                    if use_video_episode_analysis
                    else "「全片场景分段」须覆盖整片；画面描述须与抽帧锚点一致；"
                )
                current_prompt = (
                    prompt
                    + f"\n\n## 重试要求（第 {attempt + 2} 次 · 须修正以下问题）\n"
                    f"{issue_lines}\n"
                    f"{canonical_names_retry_hint}"
                    f"{cap_hint}"
                    f"必须输出完整方案，**不少于 {min_chars} 字**；"
                    f"{time_source_hint}"
                    "人名须与人物关系表一致；禁止胡晓月/罗伯等 ASR 谐音。"
                )
                continue

            if is_short_drama:
                last_validation = validate_short_drama_plot_analysis(
                    analysis,
                    lexicon=character_lexicon_data,
                    drama_known_names=drama_known_names,
                    settings=sd_settings,
                    min_chars=min_chars,
                    use_video_episode_analysis=use_video_episode_analysis,
                    relaxed=True,
                )
                logger.info(format_plot_analysis_validation_report(last_validation))
                if last_validation.get("ok"):
                    logger.info(f"{analysis_label}完成，约 {len(analysis)} 字")
                    return analysis
                if attempt + 1 >= max_attempts:
                    logger.warning(
                        "短剧联合构思校验未完全通过，返回最后一次输出（最佳努力）"
                    )
                    return analysis
                issue_lines = "\n".join(
                    f"- {item}" for item in (last_validation.get("issues") or [])
                )
                dual_time_hint = (
                    f"「原片时间线」须双时间轴：视频格（{segment_policy_summary()}索引表）+ 字幕窗（SRT）；"
                    if use_video_episode_analysis
                    else ""
                )
                current_prompt = (
                    prompt
                    + f"\n\n## 重试要求（第 {attempt + 2} 次 · 须修正以下问题）\n"
                    f"{issue_lines}\n"
                    f"{canonical_names_retry_hint}"
                    f"必须输出完整方案，**不少于 {min_chars} 字**，"
                    "按全部标题逐项填写；人名/关系须与人物关系表及"
                    f"{character_index_label}一致、对白以字幕为准；"
                    f"{dual_time_hint}"
                    "时间戳用 HH:MM:SS,mmm-HH:MM:SS,mmm；禁止「场景N」。"
                )
                continue

            if len(analysis) >= min_chars or (not fazu2 and len(analysis) >= 200):
                logger.info(f"{analysis_label}完成，约 {len(analysis)} 字")
                return analysis

            preview = analysis[:120].replace("\n", " ")
            logger.warning(
                f"{analysis_label}过短（{len(analysis)} 字），"
                f"预览: {preview!r}，重试 {attempt + 1}/{max_attempts}"
            )
            if attempt + 1 >= max_attempts:
                break
            current_prompt = (
                prompt
                + f"\n\n## 重试要求（第 {attempt + 2} 次）\n"
                f"上次输出仅 {len(analysis)} 字，无效。"
                f"必须输出完整策划蓝图，**不少于 {min_chars} 字**，"
                "按上方全部标题逐项填写，禁止只回复一句话或空泛总结。"
            )

        if analysis:
            logger.warning(f"{analysis_label}仍未达标，最终仅 {len(analysis)} 字")
        return analysis
    except Exception as exc:
        logger.warning(f"{analysis_label}失败: {exc}")
        return ""


def build_subtitle_narration_sections(
    *,
    subtitle_content: str,
    subtitle_analysis: str = "",
    settings: Optional[dict[str, Any]] = None,
) -> list[str]:
    """构建写入解说 prompt 的字幕相关区块。"""
    cfg = settings or get_documentary_settings()
    if not cfg.get("enable_subtitle_enrichment", True):
        return []
    if not (subtitle_content or "").strip():
        if (subtitle_analysis or "").strip():
            if is_fazu2_compact_settings(cfg):
                title = "## 完美剧情构思方案（脚本生成主依据 · 无原始字幕附件）"
            else:
                title = "## 剧情构思方案（脚本生成主依据）"
            return [f"{title}\n{subtitle_analysis.strip()}"]
        return []

    sections: list[str] = [
        build_subtitle_cross_validation_instructions(cfg).strip()
    ]

    excerpt = truncate_subtitle_content(
        subtitle_content,
        int(cfg.get("subtitle_max_chars", 15000)),
    )
    sections.append(
        "## 原始字幕（脚本生成第一依据 · timestamp / narration / 台词均以此为准）\n"
        f"<subtitles>\n{excerpt}\n</subtitles>"
    )

    if subtitle_analysis.strip():
        if is_fazu2_compact_settings(cfg):
            sections.append(
                "## 字幕×抽帧 对照分析（策划蓝图 · 剧情与时间戳仍以字幕为准）\n"
                + subtitle_analysis.strip()
            )
        else:
            sections.append(
                "## 字幕×抽帧 对照分析（剧情与时间戳以字幕为准）\n"
                + subtitle_analysis.strip()
            )

    return sections
