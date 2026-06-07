#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""对照抽帧分析校正 ASR/转录字幕（保留时间轴，修正错字与断句）。"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.documentary.documentary_subtitle_enrichment import parse_timestamp_range_ms
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.documentary.subtitle_typo_calibration import should_apply_typo_correction
from app.services.llm.migration_adapter import _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService
from app.services.srt_utils import SrtEntry, parse_srt_file, write_srt_file
from app.utils import utils


def get_refined_subtitle_path(video_path: str) -> str:
    """与视频配对的校正字幕默认路径。"""
    if not video_path:
        return ""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(utils.subtitle_dir(), f"{stem}_refined.srt")


def _ms_to_hhmmss(ms: int) -> str:
    return utils.seconds_to_time(ms / 1000.0).replace(".", ",")


def _sorted_batches(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    from app.services.documentary.frame_analysis_compact import rebuild_batches_from_artifact

    rebuilt = rebuild_batches_from_artifact(artifact)
    if rebuilt:
        return rebuilt

    batches = artifact.get("batches")
    if not isinstance(batches, list):
        return []

    def sort_key(batch: dict[str, Any]) -> tuple[int, int]:
        time_range = str(batch.get("time_range") or "")
        start_ms, _ = parse_timestamp_range_ms(time_range.split("-", 1)[0] if "-" in time_range else time_range)
        return start_ms, int(batch.get("batch_index", 0) or 0)

    return sorted(
        [batch for batch in batches if isinstance(batch, dict)],
        key=sort_key,
    )


def _build_batch_frame_context(batch: dict[str, Any]) -> str:
    time_range = str(batch.get("time_range") or "").strip()
    summary = (
        batch.get("overall_activity_summary")
        or batch.get("summary")
        or batch.get("fallback_summary")
        or ""
    )
    lines = [f"时间范围：{time_range}"]
    if summary:
        lines.append(f"片段摘要：{summary}")

    observations = batch.get("frame_observations") or batch.get("observations") or []
    batch_segments = batch.get("scene_segments") or []
    if batch_segments:
        lines.append("场景片段（结构化）：")
        for segment in batch_segments[:12]:
            if not isinstance(segment, dict):
                continue
            timestamp = str(segment.get("timestamp") or "").strip()
            scene = str(segment.get("scene") or "").strip()
            action = str(segment.get("action") or "").strip()
            key_visual = str(segment.get("key_visual") or "").strip()
            importance = str(segment.get("importance") or "").strip()
            summary_parts = [part for part in (scene, action, key_visual, importance) if part]
            if timestamp or summary_parts:
                lines.append(f"- {timestamp}: {' | '.join(summary_parts)}")

    if observations:
        burned_lines: list[str] = []
        observation_lines: list[str] = []
        for obs in observations[:20]:
            if not isinstance(obs, dict):
                continue
            timestamp = str(obs.get("timestamp") or "").strip()
            burned = str(obs.get("burned_in_subtitle") or "").strip()
            if burned:
                burned_lines.append(f"- {timestamp}: {burned}")
            observation = str(obs.get("observation") or "").strip()
            if observation:
                observation_lines.append(f"- {timestamp}: {observation}")
        if burned_lines:
            lines.append("画面硬字幕（校对错字依据，优先于下方观察）：")
            lines.extend(burned_lines)
        if observation_lines:
            lines.append("画面观察（辅助理解，勿据此整句改写原字幕）：")
            lines.extend(observation_lines)
    return "\n".join(lines)


def _entries_in_range(
    entries: list[SrtEntry],
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, SrtEntry]]:
    if end_ms <= start_ms:
        return []
    matched: list[tuple[int, SrtEntry]] = []
    for index, entry in enumerate(entries):
        if entry.end_ms <= start_ms or entry.start_ms >= end_ms:
            continue
        matched.append((index, entry))
    return matched


def _chunk_items(items: list[tuple[int, SrtEntry]], chunk_size: int) -> list[list[tuple[int, SrtEntry]]]:
    if chunk_size <= 0:
        return [items]
    return [items[offset : offset + chunk_size] for offset in range(0, len(items), chunk_size)]


def _format_entries_for_prompt(chunk: list[tuple[int, SrtEntry]]) -> str:
    lines: list[str] = []
    for local_id, (_, entry) in enumerate(chunk, 1):
        lines.append(
            f'{local_id}. [{_ms_to_hhmmss(entry.start_ms)} --> {_ms_to_hhmmss(entry.end_ms)}] '
            f'{entry.text.replace(chr(10), " ")}'
        )
    return "\n".join(lines)


def _parse_refinement_response(response_text: str, expected_count: int) -> dict[int, str]:
    text = (response_text or "").strip()
    if not text:
        return {}

    candidates: list[str] = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    rows: list[Any] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            rows = parsed
            break
        if isinstance(parsed, dict):
            nested = parsed.get("entries") or parsed.get("items") or parsed.get("subtitles")
            if isinstance(nested, list):
                rows = nested
                break

    corrections: dict[int, str] = {}
    for row in rows[:expected_count]:
        if not isinstance(row, dict):
            continue
        local_id = int(row.get("index") or row.get("id") or row.get("local_id") or 0)
        corrected = str(row.get("text") or row.get("corrected_text") or "").strip()
        if local_id > 0 and corrected:
            corrections[local_id] = corrected
    return corrections


def _refine_subtitle_chunk(
    *,
    chunk: list[tuple[int, SrtEntry]],
    frame_context: str,
    video_theme: str,
    temperature: float,
    min_similarity: float,
    max_length_ratio_delta: float,
) -> dict[int, str]:
    if not chunk:
        return {}

    prompt_body = _format_entries_for_prompt(chunk)
    theme = (video_theme or "").strip() or "本视频"
    prompt = f"""你是专业影视字幕编辑。请**以原字幕文件为准**，对照画面硬字幕（burned_in_subtitle）修正 ASR 错字、同音字。

作品/主题：{theme}

## 抽帧画面分析
{frame_context}

## 待校正字幕（local_id 为批次内序号；时间轴勿改）
{prompt_body}

## 要求（必须遵守）
1. **原字幕优先**：保留原条目文字、断句与时间轴；仅改可确认的错别字/缺字
2. **画面硬字幕为准**：只有 burned_in_subtitle 与原句明显同一句时，才用硬字幕替换错字
3. 勿根据画面 observation 猜测对白、勿整句重写、勿合并/拆分/增删条目
4. 硬字幕缺失或与原句差异过大时，**保持原文**
5. 只输出 JSON 数组，长度必须为 {len(chunk)}：
[{{"index":1,"text":"校正后文本"}}, ...]
index 为上方 local_id（1 起），text 为校正后单行对白。"""

    system_prompt = (
        "你是字幕错别字校对专家：原字幕优先，画面硬字幕仅用于确认同句错字。"
        "只输出合法 JSON 数组，不要 markdown 或解释。"
    )

    text_provider = config.app.get("text_llm_provider", "openai").lower()
    api_key = config.app.get(f"text_{text_provider}_api_key")
    model = config.app.get(f"text_{text_provider}_model_name")
    base_url = config.app.get(f"text_{text_provider}_base_url", "")
    if not api_key or not model:
        raise ValueError(
            f"未配置文本模型，无法校正字幕。请配置 text_{text_provider}_api_key / model_name"
        )

    result = _run_async_safely(
        UnifiedLLMService.generate_text,
        prompt=prompt,
        system_prompt=system_prompt,
        provider=text_provider,
        temperature=temperature,
        api_key=api_key,
        api_base=base_url,
        for_script=True,
    )
    local_corrections = _parse_refinement_response(
        result if isinstance(result, str) else str(result),
        len(chunk),
    )

    global_corrections: dict[int, str] = {}
    for local_id, (global_index, entry) in enumerate(chunk, 1):
        corrected = local_corrections.get(local_id)
        if corrected and should_apply_typo_correction(
            entry.text,
            corrected,
            min_similarity=min_similarity,
            max_length_ratio_delta=max_length_ratio_delta,
        ):
            global_corrections[global_index] = corrected
    return global_corrections


def refine_subtitle_with_frame_analysis(
    *,
    subtitle_path: str,
    analysis_json_path: str,
    output_path: str | None = None,
    video_theme: str = "",
    documentary_settings: dict | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    以原字幕文件为准，对照画面硬字幕校正同句错别字，输出新 SRT（默认 video_refined.srt）。

    保留原时间轴与条目结构，仅更新可确认的错字。
    """
    if not subtitle_path or not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
    if not analysis_json_path or not os.path.isfile(analysis_json_path):
        raise FileNotFoundError(f"抽帧分析文件不存在: {analysis_json_path}")

    cfg = documentary_settings or get_documentary_settings()
    pad_ms = int(cfg.get("subtitle_batch_pad_sec", 5) or 5) * 1000
    max_entries = max(5, int(cfg.get("subtitle_refinement_max_entries_per_call", 25) or 25))
    temperature = float(cfg.get("subtitle_refinement_temperature", 0.3) or 0.3)
    min_similarity = float(cfg.get("subtitle_refinement_min_similarity", 0.5) or 0.5)
    max_length_ratio_delta = float(cfg.get("subtitle_refinement_max_length_ratio_delta", 0.4) or 0.4)

    entries = parse_srt_file(subtitle_path)
    if not entries:
        raise ValueError(f"字幕文件为空或无法解析: {subtitle_path}")

    artifact = load_analysis_artifact(analysis_json_path)
    batches = _sorted_batches(artifact)
    if not batches:
        raise ValueError("抽帧分析缺少 batches，无法对照校正字幕")

    if not output_path:
        video_path = str(artifact.get("video_path") or "").strip()
        if video_path:
            output_path = get_refined_subtitle_path(video_path)
        else:
            stem = os.path.splitext(os.path.basename(subtitle_path))[0]
            if stem.endswith("_transcribed"):
                stem = stem[: -len("_transcribed")]
            output_path = os.path.join(utils.subtitle_dir(), f"{stem}_refined.srt")

    corrections: dict[int, str] = {}
    total_batches = len(batches)

    for batch_index, batch in enumerate(batches, 1):
        time_range = str(batch.get("time_range") or "").strip()
        if not time_range:
            continue
        start_ms, end_ms = parse_timestamp_range_ms(time_range)
        window_start = max(0, start_ms - pad_ms)
        window_end = end_ms + pad_ms
        matched = _entries_in_range(entries, window_start, window_end)
        if not matched:
            continue

        if progress_callback:
            progress_callback(
                f"校正字幕批次 {batch_index}/{total_batches}（{len(matched)} 条）..."
            )

        frame_context = _build_batch_frame_context(batch)
        for chunk in _chunk_items(matched, max_entries):
            chunk_corrections = _refine_subtitle_chunk(
                chunk=chunk,
                frame_context=frame_context,
                video_theme=video_theme,
                temperature=temperature,
                min_similarity=min_similarity,
                max_length_ratio_delta=max_length_ratio_delta,
            )
            corrections.update(chunk_corrections)

    refined_entries: list[SrtEntry] = []
    changed_count = 0
    for index, entry in enumerate(entries):
        new_text = corrections.get(index)
        if new_text and new_text != entry.text:
            refined_entries.append(
                SrtEntry(
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    text=new_text,
                    label=entry.label,
                )
            )
            changed_count += 1
        else:
            refined_entries.append(entry)

    write_srt_file(refined_entries, output_path)
    logger.info(
        f"字幕校正完成: {output_path}（共 {len(refined_entries)} 条，修改 {changed_count} 条）"
    )
    return output_path
