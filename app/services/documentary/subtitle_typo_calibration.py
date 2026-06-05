#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""字幕错别字校准：以原字幕为准，画面硬字幕仅作同句错字修正依据。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_subtitle_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for punct in "，。！？、；：""''（）【】《》…—·,.!?;:'\"()[]":
        cleaned = cleaned.replace(punct, "")
    return re.sub(r"\s+", "", cleaned)


def subtitle_text_similarity(left: str, right: str) -> float:
    left_norm = normalize_subtitle_text(left)
    right_norm = normalize_subtitle_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _length_delta_ratio(left: str, right: str) -> float:
    left_len = len(normalize_subtitle_text(left))
    right_len = len(normalize_subtitle_text(right))
    max_len = max(left_len, right_len)
    if max_len <= 0:
        return 0.0
    return abs(left_len - right_len) / max_len


def calibrate_typo_from_screen_subtitle(
    original: str,
    screen_subtitle: str,
    *,
    min_similarity: float = 0.5,
    max_length_ratio_delta: float = 0.35,
) -> str | None:
    """
    原字幕优先；画面硬字幕与原文判定为同一句时，返回用于替换的校正文本，否则 None（保持原文）。
    """
    original_text = (original or "").strip()
    screen_text = (screen_subtitle or "").strip()
    if not original_text or not screen_text:
        return None

    orig_norm = normalize_subtitle_text(original_text)
    screen_norm = normalize_subtitle_text(screen_text)
    if not orig_norm or not screen_norm:
        return None
    if orig_norm == screen_norm:
        return None

    length_delta = _length_delta_ratio(original_text, screen_text)
    if length_delta > max_length_ratio_delta:
        return None

    similarity = subtitle_text_similarity(original_text, screen_text)
    if screen_norm in orig_norm or orig_norm in screen_norm:
        if length_delta <= max_length_ratio_delta:
            return screen_text
        return None

    if similarity >= min_similarity:
        return screen_text

    return None


def should_apply_typo_correction(
    original: str,
    corrected: str,
    *,
    min_similarity: float = 0.5,
    max_length_ratio_delta: float = 0.4,
) -> bool:
    """LLM 校正结果仅在同句错字级别时采纳，防止整句被改写。"""
    original_text = (original or "").strip()
    corrected_text = (corrected or "").strip()
    if not original_text or not corrected_text:
        return False
    if original_text == corrected_text:
        return False
    if _length_delta_ratio(original_text, corrected_text) > max_length_ratio_delta:
        return False
    return subtitle_text_similarity(original_text, corrected_text) >= min_similarity
