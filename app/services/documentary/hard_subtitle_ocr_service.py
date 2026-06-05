#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""硬字幕校准：以字幕文件为准，用画面 burned_in_subtitle 修正同句错别字（保留时间轴）。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional

import PIL.Image
from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.documentary.documentary_subtitle_enrichment import (
    _timestamp_to_ms,
    parse_timestamp_range_ms,
)
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.llm.migration_adapter import _run_async_safely, create_vision_analyzer
from app.services.documentary.subtitle_typo_calibration import (
    calibrate_typo_from_screen_subtitle,
    normalize_subtitle_text,
)
from app.services.srt_utils import SrtEntry, parse_srt_file, write_srt_file
from app.utils import utils


_OCR_PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧底部字幕区域的裁剪图，按时间顺序排列。
每张图仅包含画面底部硬字幕（烧录字幕）区域。

请逐张识别**屏幕上实际显示的字幕文字**：
- 只输出画面底部字幕带内的对白/旁白文字
- 若无字幕、仅 logo/水印、或无法辨认，has_subtitle 设为 false，text 为空字符串
- 不要猜测听不清的内容，不要输出画面描述
- 保留原文标点；多行字幕合并为一行

务必输出 JSON，且 frame_results 长度必须为 {frame_count}：
{{
  "frame_results": [
    {{"index": 1, "has_subtitle": true, "text": "字幕原文"}},
    {{"index": 2, "has_subtitle": false, "text": ""}}
  ]
}}
只返回 JSON，不要 markdown 或解释。
""".strip()

_KEYFRAME_TS_RE = re.compile(r"keyframe_\d{6}_(\d{9})\.jpg$", re.IGNORECASE)


@dataclass
class OcrFrameHit:
    frame_path: str
    timestamp_ms: int
    timestamp: str
    text: str
    has_subtitle: bool


def get_ocr_refined_subtitle_path(video_path: str) -> str:
    if not video_path:
        return ""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(utils.subtitle_dir(), f"{stem}_ocr_refined.srt")


def _timestamp_from_keyframe_name(filename: str) -> str:
    match = _KEYFRAME_TS_RE.search(os.path.basename(filename))
    if not match:
        return "00:00:00,000"
    token = match.group(1)
    hours = int(token[0:2])
    minutes = int(token[2:4])
    seconds = int(token[4:6])
    milliseconds = int(token[6:9])
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _iter_frame_records_from_artifact(
    artifact: dict[str, Any],
    *,
    require_existing_files: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    batches = artifact.get("batches")
    if isinstance(batches, list):
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            observations = batch.get("frame_observations") or batch.get("observations") or []
            frame_paths = batch.get("frame_paths") or []
            if isinstance(frame_paths, list):
                for index, frame_path in enumerate(frame_paths):
                    if not isinstance(frame_path, str) or not frame_path:
                        continue
                    obs: dict[str, Any] = {}
                    if index < len(observations) and isinstance(observations[index], dict):
                        obs = observations[index]
                    timestamp = str(obs.get("timestamp") or "").strip()
                    records.append(
                        {
                            "frame_path": frame_path,
                            "timestamp": timestamp,
                            "burned_in_subtitle": str(obs.get("burned_in_subtitle") or "").strip(),
                            "has_burned_in_subtitle": bool(obs.get("has_burned_in_subtitle")),
                        }
                    )

    flat_observations = artifact.get("frame_observations")
    if isinstance(flat_observations, list):
        for obs in flat_observations:
            if not isinstance(obs, dict):
                continue
            frame_path = str(obs.get("frame_path") or "").strip()
            if frame_path:
                records.append(
                    {
                        "frame_path": frame_path,
                        "timestamp": str(obs.get("timestamp") or "").strip(),
                        "burned_in_subtitle": str(obs.get("burned_in_subtitle") or "").strip(),
                        "has_burned_in_subtitle": bool(obs.get("has_burned_in_subtitle")),
                    }
                )

    sorted_records: list[dict[str, Any]] = []
    for record in records:
        frame_path = str(record.get("frame_path") or "").strip()
        if not frame_path or frame_path in seen_paths:
            continue
        if require_existing_files and not os.path.isfile(frame_path):
            continue
        seen_paths.add(frame_path)
        timestamp = str(record.get("timestamp") or "").strip()
        if not timestamp:
            timestamp = _timestamp_from_keyframe_name(frame_path)
        timestamp_ms = _timestamp_to_ms(timestamp)
        burned_text = str(record.get("burned_in_subtitle") or "").strip()
        has_burned = bool(record.get("has_burned_in_subtitle"))
        if burned_text and not has_burned:
            has_burned = True
        sorted_records.append(
            {
                "frame_path": frame_path,
                "timestamp": timestamp,
                "timestamp_ms": timestamp_ms,
                "burned_in_subtitle": burned_text,
                "has_burned_in_subtitle": has_burned and bool(burned_text),
            }
        )

    sorted_records.sort(key=lambda item: int(item.get("timestamp_ms") or 0))
    return sorted_records


def _collect_ocr_frames(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return _iter_frame_records_from_artifact(artifact, require_existing_files=True)


def extract_ocr_hits_from_artifact(artifact: dict[str, Any]) -> list[OcrFrameHit]:
    """从抽帧分析 JSON 内嵌的 burned_in_subtitle 字段提取 OCR 结果（无需二次视觉调用）。"""
    hits: list[OcrFrameHit] = []
    for record in _iter_frame_records_from_artifact(artifact, require_existing_files=False):
        text = str(record.get("burned_in_subtitle") or "").strip()
        has_subtitle = bool(record.get("has_burned_in_subtitle")) and bool(text)
        hits.append(
            OcrFrameHit(
                frame_path=str(record["frame_path"]),
                timestamp_ms=int(record["timestamp_ms"]),
                timestamp=str(record["timestamp"]),
                text=text if has_subtitle else "",
                has_subtitle=has_subtitle,
            )
        )
    return hits


def _crop_subtitle_band(image_path: str, crop_ratio: float) -> PIL.Image.Image:
    ratio = float(crop_ratio)
    if ratio <= 0:
        ratio = 0.22
    if ratio > 0.5:
        ratio = 0.5

    with PIL.Image.open(image_path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        crop_height = max(1, int(round(height * ratio)))
        top = max(0, height - crop_height)
        cropped = rgb.crop((0, top, width, height))
        if max(cropped.size) > 1280:
            cropped.thumbnail((1280, 1280), PIL.Image.Resampling.LANCZOS)
        return cropped.copy()


def _parse_ocr_batch_response(response_text: str, expected_count: int) -> list[dict[str, Any]]:
    text = (response_text or "").strip()
    if not text:
        return []

    candidates: list[str] = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    rows: list[Any] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            nested = parsed.get("frame_results") or parsed.get("results") or parsed.get("items")
            if isinstance(nested, list):
                rows = nested
                break
        if isinstance(parsed, list):
            rows = parsed
            break

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:expected_count]):
        if not isinstance(row, dict):
            results.append({"has_subtitle": False, "text": ""})
            continue
        has_subtitle = bool(row.get("has_subtitle", row.get("has_text", False)))
        ocr_text = str(row.get("text") or row.get("subtitle") or "").strip()
        if ocr_text and not has_subtitle:
            has_subtitle = True
        results.append({"has_subtitle": has_subtitle, "text": ocr_text})
    while len(results) < expected_count:
        results.append({"has_subtitle": False, "text": ""})
    return results


def _extract_batch_response(raw_results: list[Any]) -> str:
    if not raw_results:
        return ""
    first = raw_results[0]
    if isinstance(first, dict):
        return str(first.get("response") or "")
    return str(first or "")


async def _ocr_frame_batches_async(
    frames: list[dict[str, Any]],
    *,
    crop_ratio: float,
    batch_size: int,
    max_concurrency: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[OcrFrameHit]:
    if not frames:
        return []

    provider = config.app.get("vision_llm_provider", "openai").lower()
    api_key = config.app.get(f"vision_{provider}_api_key")
    model_name = config.app.get(f"vision_{provider}_model_name")
    base_url = config.app.get(f"vision_{provider}_base_url", "")
    if not api_key or not model_name:
        raise ValueError(
            f"未配置视觉模型，无法 OCR 硬字幕。请配置 vision_{provider}_api_key / model_name"
        )

    analyzer = create_vision_analyzer(
        provider=provider,
        api_key=api_key,
        model=model_name,
        base_url=base_url,
    )

    batches: list[list[dict[str, Any]]] = [
        frames[index : index + batch_size] for index in range(0, len(frames), batch_size)
    ]
    total_batches = len(batches)
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    hits: list[OcrFrameHit] = []

    async def run_batch(batch_index: int, batch_frames: list[dict[str, Any]]) -> list[OcrFrameHit]:
        cropped_images: list[PIL.Image.Image] = []
        for frame in batch_frames:
            cropped_images.append(_crop_subtitle_band(frame["frame_path"], crop_ratio))

        prompt = _OCR_PROMPT_TEMPLATE.format(frame_count=len(batch_frames))
        async with semaphore:
            raw_results = await analyzer.analyze_images(
                images=cropped_images,
                prompt=prompt,
                batch_size=max(1, len(batch_frames)),
                max_concurrency=1,
            )
        response_text = _extract_batch_response(raw_results)
        parsed_rows = _parse_ocr_batch_response(response_text, len(batch_frames))

        batch_hits: list[OcrFrameHit] = []
        for frame, row in zip(batch_frames, parsed_rows):
            text = str(row.get("text") or "").strip()
            has_subtitle = bool(row.get("has_subtitle")) and bool(text)
            batch_hits.append(
                OcrFrameHit(
                    frame_path=frame["frame_path"],
                    timestamp_ms=int(frame["timestamp_ms"]),
                    timestamp=str(frame["timestamp"]),
                    text=text if has_subtitle else "",
                    has_subtitle=has_subtitle,
                )
            )
        if progress_callback:
            progress_callback(f"硬字幕 OCR 批次 {batch_index + 1}/{total_batches} 完成")
        return batch_hits

    gathered = await asyncio.gather(
        *(run_batch(index, batch) for index, batch in enumerate(batches))
    )
    for batch_hits in gathered:
        hits.extend(batch_hits)
    hits.sort(key=lambda item: item.timestamp_ms)
    return hits


def _ocr_hits_for_entry(
    entry: SrtEntry,
    ocr_hits: list[OcrFrameHit],
    pad_ms: int,
) -> list[OcrFrameHit]:
    window_start = max(0, entry.start_ms - pad_ms)
    window_end = entry.end_ms + pad_ms
    matched: list[OcrFrameHit] = []
    for hit in ocr_hits:
        if not hit.has_subtitle or not hit.text:
            continue
        if hit.timestamp_ms < window_start or hit.timestamp_ms > window_end:
            continue
        matched.append(hit)
    return matched


def _pick_ocr_text(hits: list[OcrFrameHit], min_frames: int) -> str:
    if not hits:
        return ""

    normalized_counts: Counter[str] = Counter()
    raw_by_normalized: dict[str, str] = {}
    for hit in hits:
        normalized = normalize_subtitle_text(hit.text)
        if not normalized:
            continue
        normalized_counts[normalized] += 1
        if normalized not in raw_by_normalized:
            raw_by_normalized[normalized] = hit.text.strip()

    if not normalized_counts:
        return ""

    best_normalized, count = normalized_counts.most_common(1)[0]
    if count < max(1, min_frames):
        return ""
    return raw_by_normalized.get(best_normalized, "")


def _apply_ocr_to_entries(
    entries: list[SrtEntry],
    ocr_hits: list[OcrFrameHit],
    *,
    pad_ms: int,
    min_frames: int,
    min_similarity: float,
    max_length_ratio_delta: float,
) -> tuple[list[SrtEntry], int]:
    """原字幕优先：仅在同句相似度足够时用画面硬字幕修正错字。"""
    calibrated: list[SrtEntry] = []
    changed_count = 0

    for entry in entries:
        hits = _ocr_hits_for_entry(entry, ocr_hits, pad_ms)
        screen_subtitle = _pick_ocr_text(hits, min_frames)
        corrected = calibrate_typo_from_screen_subtitle(
            entry.text,
            screen_subtitle,
            min_similarity=min_similarity,
            max_length_ratio_delta=max_length_ratio_delta,
        )
        if corrected:
            calibrated.append(
                SrtEntry(
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    text=corrected,
                    label=entry.label,
                )
            )
            changed_count += 1
        else:
            calibrated.append(entry)

    return calibrated, changed_count


def calibrate_subtitle_with_hard_subtitle_ocr(
    *,
    subtitle_path: str,
    analysis_json_path: str,
    output_path: str | None = None,
    documentary_settings: dict | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    allow_vision_ocr_fallback: bool = False,
) -> str:
    """
    以字幕文件为准，用画面硬字幕修正同句错别字（保留时间轴）。

    优先使用抽帧分析 JSON 内嵌的 burned_in_subtitle（与抽帧同一次视觉调用）；
    仅当 allow_vision_ocr_fallback=true 且无内嵌字段时，才二次调用视觉模型 OCR。

    输出默认 {stem}_ocr_refined.srt。
    """
    if not subtitle_path or not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
    if not analysis_json_path or not os.path.isfile(analysis_json_path):
        raise FileNotFoundError(f"抽帧分析文件不存在: {analysis_json_path}")

    cfg = documentary_settings or get_documentary_settings()
    crop_ratio = float(cfg.get("subtitle_ocr_crop_ratio", 0.22) or 0.22)
    batch_size = max(1, int(cfg.get("subtitle_ocr_batch_size", 10) or 10))
    max_concurrency = max(1, int(cfg.get("subtitle_ocr_max_concurrency", 2) or 2))
    pad_ms = max(0, int(cfg.get("subtitle_ocr_match_pad_ms", 1500) or 1500))
    min_frames = max(1, int(cfg.get("subtitle_ocr_min_confidence_frames", 1) or 1))
    min_similarity = float(cfg.get("subtitle_ocr_min_similarity", 0.5) or 0.5)
    max_length_ratio_delta = float(cfg.get("subtitle_ocr_max_length_ratio_delta", 0.35) or 0.35)

    entries = parse_srt_file(subtitle_path)
    if not entries:
        raise ValueError(f"字幕文件为空或无法解析: {subtitle_path}")

    artifact = load_analysis_artifact(analysis_json_path)
    ocr_hits = extract_ocr_hits_from_artifact(artifact)
    subtitle_hit_count = sum(1 for hit in ocr_hits if hit.has_subtitle and hit.text)

    if subtitle_hit_count == 0 and allow_vision_ocr_fallback:
        frames = _collect_ocr_frames(artifact)
        if not frames:
            raise ValueError("抽帧分析中未找到可用关键帧路径（frame_path），请先完成抽帧分析")

        if progress_callback:
            progress_callback(f"正在 OCR {len(frames)} 帧硬字幕区域...")

        ocr_hits = _run_async_safely(
            _ocr_frame_batches_async,
            frames,
            crop_ratio=crop_ratio,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
        )
        subtitle_hit_count = sum(1 for hit in ocr_hits if hit.has_subtitle and hit.text)
        logger.info(f"硬字幕 OCR（二次视觉）完成：{subtitle_hit_count}/{len(ocr_hits)} 帧识别到字幕")
    elif subtitle_hit_count > 0:
        logger.info(
            f"使用抽帧分析内嵌硬字幕：{subtitle_hit_count}/{len(ocr_hits)} 帧有效，跳过二次视觉 OCR"
        )

    if subtitle_hit_count == 0:
        raise ValueError(
            "未识别到硬字幕。"
            "请确认视频含烧录字幕；新抽帧分析会在同一次视觉调用中识读 burned_in_subtitle。"
        )

    if progress_callback:
        progress_callback("正在对照画面硬字幕修正原字幕错字...")

    calibrated_entries, changed_count = _apply_ocr_to_entries(
        entries,
        ocr_hits,
        pad_ms=pad_ms,
        min_frames=min_frames,
        min_similarity=min_similarity,
        max_length_ratio_delta=max_length_ratio_delta,
    )

    if not output_path:
        video_path = str(artifact.get("video_path") or "").strip()
        if video_path:
            output_path = get_ocr_refined_subtitle_path(video_path)
        else:
            stem = os.path.splitext(os.path.basename(subtitle_path))[0]
            if stem.endswith("_transcribed"):
                stem = stem[: -len("_transcribed")]
            elif stem.endswith("_refined"):
                stem = stem[: -len("_refined")]
            output_path = os.path.join(utils.subtitle_dir(), f"{stem}_ocr_refined.srt")

    write_srt_file(calibrated_entries, output_path)
    logger.info(
        f"硬字幕 OCR 校准完成: {output_path}（共 {len(calibrated_entries)} 条，修改 {changed_count} 条）"
    )
    return output_path
