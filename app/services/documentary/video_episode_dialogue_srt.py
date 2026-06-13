#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：SRT 作时间/对位参照，台词文字以视频分析（画面字幕）为准。"""

from __future__ import annotations

import os
import re
from typing import Any

from app.models import const
from app.services.srt_utils import SrtEntry, clean_subtitle_dialogue_text, parse_srt

_UNNAMED_SPEAKER = "剧中未明确交代"
_DEFAULT_DEDUPE_WINDOW_SECONDS = 10
_DEFAULT_TIMESTAMP_PAD_MS = 2000


def _clean_subtitle_punctuation(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for punct in sorted(const.PUNCTUATIONS, key=len, reverse=True):
        cleaned = cleaned.replace(punct, "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_subtitle_dedupe_key(text: str) -> str:
    return re.sub(r"\s+", "", _clean_subtitle_punctuation(str(text or "").strip()))


def is_phantom_subtitle_fragment(text: str) -> bool:
    cleaned = _clean_subtitle_punctuation(str(text or "").strip())
    if not cleaned:
        return True
    if len(cleaned) <= 1:
        return True
    if cleaned[-1] in "了啊呀嘛呢吧" and len(cleaned) <= 2:
        return True
    if cleaned.endswith("，") or cleaned.endswith("。"):
        core = cleaned[:-1].strip()
        if len(core) <= 1:
            return True
    return False


def _timestamp_to_ms(timestamp: str) -> int:
    cleaned = str(timestamp or "").strip().split("-", 1)[0].strip()
    parts = cleaned.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return int(
                round(
                    (float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])) * 1000
                )
            )
        if len(parts) == 2:
            return int(round((float(parts[0]) * 60 + float(parts[1])) * 1000))
        return int(round(float(parts[0]) * 1000))
    except (TypeError, ValueError):
        return 0


def _timestamp_to_seconds(value: str) -> int:
    return _timestamp_to_ms(value) // 1000


def _seconds_to_timestamp(seconds: int) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def _quote_similarity(left: str, right: str) -> float:
    a = _normalize_subtitle_dedupe_key(left)
    b = _normalize_subtitle_dedupe_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer:
        return len(shorter) / max(len(longer), 1)
    overlap = sum(1 for ch in shorter if ch in longer)
    return overlap / max(len(shorter), len(longer), 1)


def _quotes_are_near_duplicate(left: str, right: str, *, threshold: float = 0.82) -> bool:
    return _quote_similarity(left, right) >= threshold


def _pick_best_srt_entry(
    entries: list[SrtEntry],
    *,
    timestamp: str,
    quote: str,
    timestamp_pad_ms: int,
) -> SrtEntry | None:
    if not entries:
        return None

    direct = find_srt_entry_at_timestamp(entries, timestamp, pad_ms=timestamp_pad_ms)
    if direct is not None:
        return direct

    ts_ms = _timestamp_to_ms(timestamp)
    if ts_ms <= 0 and not quote.strip():
        return None

    candidates: list[tuple[float, SrtEntry]] = []
    window_start_ms = max(0, ts_ms - timestamp_pad_ms)
    window_end_ms = ts_ms + timestamp_pad_ms
    for entry in entries:
        if entry.end_ms < window_start_ms or entry.start_ms > window_end_ms:
            continue
        text = clean_subtitle_dialogue_text(entry.text) or (entry.text or "").strip()
        if not text or is_phantom_subtitle_fragment(text):
            continue
        time_distance = min(abs(entry.start_ms - ts_ms), abs(entry.end_ms - ts_ms))
        text_score = _quote_similarity(quote, text) if quote.strip() else 0.0
        score = text_score * 1000 - time_distance
        candidates.append((score, entry))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_entry = candidates[0]
    if quote.strip() and best_score < 0:
        return None
    return best_entry


def _merge_speakers(left: str, right: str) -> str:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a and not b:
        return ""
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    if a == _UNNAMED_SPEAKER:
        return b
    if b == _UNNAMED_SPEAKER:
        return a
    return _UNNAMED_SPEAKER


def dedupe_important_dialogues(
    dialogues: list[dict[str, Any]],
    *,
    window_seconds: int = _DEFAULT_DEDUPE_WINDOW_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not dialogues:
        return [], []

    ordered = sorted(
        dialogues,
        key=lambda item: _timestamp_to_seconds(str(item.get("timestamp") or "")),
    )
    merged: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in ordered:
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        ts_seconds = _timestamp_to_seconds(str(item.get("timestamp") or ""))
        duplicate_index = -1
        for index, existing in enumerate(merged):
            existing_ts = _timestamp_to_seconds(str(existing.get("timestamp") or ""))
            if abs(existing_ts - ts_seconds) > window_seconds:
                continue
            if _quotes_are_near_duplicate(quote, str(existing.get("quote") or "")):
                duplicate_index = index
                break

        if duplicate_index < 0:
            merged.append(dict(item))
            continue

        keeper = merged[duplicate_index]
        existing_quote = str(keeper.get("quote") or "")
        if len(quote) > len(existing_quote):
            keeper["quote"] = quote
        keeper_ts = _timestamp_to_seconds(str(keeper.get("timestamp") or ""))
        if ts_seconds and (not keeper_ts or ts_seconds < keeper_ts):
            keeper["timestamp"] = _seconds_to_timestamp(ts_seconds)

        old_speaker = str(keeper.get("speaker") or "").strip()
        new_speaker = str(item.get("speaker") or "").strip()
        merged_speaker = _merge_speakers(old_speaker, new_speaker)
        if merged_speaker != old_speaker and old_speaker and new_speaker and old_speaker != new_speaker:
            warnings.append(
                f"合并重复台词（{_seconds_to_timestamp(min(keeper_ts, ts_seconds))} 附近）："
                f"「{existing_quote[:18]}…」说话人 {old_speaker}/{new_speaker} → {merged_speaker}"
            )
        keeper["speaker"] = merged_speaker

        existing_sig = str(keeper.get("significance") or "").strip()
        new_sig = str(item.get("significance") or "").strip()
        if new_sig and (not existing_sig or len(new_sig) > len(existing_sig)):
            keeper["significance"] = new_sig

    return merged, warnings


def enrich_important_dialogues_with_srt(
    dialogues: list[dict[str, Any]],
    subtitle_content: str,
    *,
    timestamp_pad_ms: int = _DEFAULT_TIMESTAMP_PAD_MS,
    dedupe_window_seconds: int = _DEFAULT_DEDUPE_WINDOW_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    SRT 参照 + 视频分析台词为主：
    - timestamp：优先对齐 SRT 时间轴
    - quote：**严格保留**视觉模型从画面读到的原话，不用 SRT 替换
    - speaker / significance：保留视觉分析
    - 相近重复：去重合并，quote 仍取视频分析侧
    """
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return list(dialogues or []), []

    aligned: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in dialogues or []:
        if not isinstance(item, dict):
            continue
        video_quote = str(item.get("quote") or item.get("text") or "").strip()
        timestamp = str(item.get("timestamp") or "").strip()
        if not video_quote and not timestamp:
            continue

        entry = _pick_best_srt_entry(
            entries,
            timestamp=timestamp,
            quote=video_quote,
            timestamp_pad_ms=timestamp_pad_ms,
        )
        if entry is None:
            if video_quote:
                aligned.append(
                    {
                        "speaker": str(item.get("speaker") or "").strip(),
                        "timestamp": timestamp or "00:00:00",
                        "quote": video_quote,
                        "significance": str(item.get("significance") or "").strip(),
                    }
                )
                preview = video_quote[:20] + ("…" if len(video_quote) > 20 else "")
                warnings.append(f"未匹配 SRT，保留视频分析台词: {preview}")
            continue

        srt_quote = clean_subtitle_dialogue_text(entry.text) or (entry.text or "").strip()
        if srt_quote and is_phantom_subtitle_fragment(srt_quote):
            srt_quote = ""

        aligned_timestamp = _seconds_to_timestamp(entry.start_ms // 1000)
        if (
            srt_quote
            and video_quote
            and _quote_similarity(video_quote, srt_quote) < 0.55
        ):
            warnings.append(
                "视频台词与 SRT 差异较大，已保留视频分析原文 "
                f"（{aligned_timestamp}）：「{video_quote[:24]}…」"
            )

        aligned.append(
            {
                "speaker": str(item.get("speaker") or "").strip(),
                "timestamp": aligned_timestamp,
                "quote": video_quote,
                "significance": str(item.get("significance") or "").strip(),
            }
        )

    deduped, dedupe_warnings = dedupe_important_dialogues(
        aligned,
        window_seconds=dedupe_window_seconds,
    )
    return deduped, warnings + dedupe_warnings


def apply_important_dialogues_srt_enrichment(
    analysis: dict[str, Any],
    *,
    video_path: str,
    subtitle_path: str = "",
    enabled: bool = True,
) -> list[str]:
    """SRT 作参照对齐时间轴；important_dialogues.quote 以视频分析为准。"""
    if not enabled:
        return []

    from app.services.subtitle_video_pairing import (
        load_subtitle_content,
        resolve_subtitle_path_for_video,
    )

    resolved_subtitle = resolve_subtitle_path_for_video(
        video_path,
        explicit_path=subtitle_path,
    )
    subtitle_content = load_subtitle_content(resolved_subtitle)
    if not subtitle_content.strip():
        return []

    dialogues = list(analysis.get("important_dialogues") or [])
    if not dialogues:
        return []

    enriched, warnings = enrich_important_dialogues_with_srt(dialogues, subtitle_content)
    analysis["important_dialogues"] = enriched
    analysis["important_dialogues_source"] = "video_quote_srt_reference"
    if resolved_subtitle:
        analysis["important_dialogues_srt_path"] = os.path.abspath(resolved_subtitle)
    return warnings
