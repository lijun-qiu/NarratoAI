#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧分析完成后自动校准字幕（复用分析 JSON 中的硬字幕字段，不重复调用视觉模型）。"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from loguru import logger

from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.documentary.hard_subtitle_ocr_service import calibrate_subtitle_with_hard_subtitle_ocr
from app.services.documentary.subtitle_refinement_service import refine_subtitle_with_frame_analysis
from app.services.subtitle_video_pairing import (
    find_paired_subtitle_path,
    has_subtitle_for_video,
    resolve_subtitle_path_for_video,
)


def requires_subtitle_before_frame_analysis(settings: dict | None = None) -> bool:
    """抽帧前是否必须先有字幕（以便同一步完成自动校准）。"""
    cfg = settings or get_documentary_settings()
    if not cfg.get("auto_subtitle_calibration_on_frame_analysis", True):
        return False
    return bool(
        cfg.get("enable_hard_subtitle_ocr", True)
        or cfg.get("enable_subtitle_refinement", True)
    )


def calibrate_subtitles_after_frame_analysis(
    *,
    analysis_json_path: str,
    video_path: str = "",
    subtitle_path: str | None = None,
    video_theme: str = "",
    documentary_settings: dict | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    allow_vision_ocr_fallback: bool = False,
) -> dict[str, Any]:
    """
    抽帧分析后联动字幕校准：优先用 JSON 内嵌的 burned_in_subtitle，再可选 LLM 校正。

    返回 dict，含 ocr_refined_path / refined_path / subtitle_path / skipped_reason 等。
    """
    cfg = documentary_settings or get_documentary_settings()
    result: dict[str, Any] = {
        "ocr_refined_path": None,
        "refined_path": None,
        "subtitle_path": None,
        "skipped_reason": None,
    }

    if not cfg.get("auto_subtitle_calibration_on_frame_analysis", True):
        result["skipped_reason"] = "auto_subtitle_calibration_on_frame_analysis=false"
        return result

    if not analysis_json_path or not os.path.isfile(analysis_json_path):
        result["skipped_reason"] = "missing_analysis_json"
        return result

    resolved_subtitle = resolve_subtitle_path_for_video(
        video_path,
        explicit_path=subtitle_path,
    )
    if not resolved_subtitle:
        result["skipped_reason"] = "no_subtitle"
        return result

    current_subtitle = resolved_subtitle

    if cfg.get("enable_hard_subtitle_ocr", True):
        if progress_callback:
            progress_callback("正在用抽帧分析中的硬字幕校准 ASR...")
        try:
            ocr_path = calibrate_subtitle_with_hard_subtitle_ocr(
                subtitle_path=current_subtitle,
                analysis_json_path=analysis_json_path,
                documentary_settings=cfg,
                progress_callback=progress_callback,
                allow_vision_ocr_fallback=allow_vision_ocr_fallback,
            )
            result["ocr_refined_path"] = ocr_path
            current_subtitle = ocr_path
        except ValueError as exc:
            logger.info(f"硬字幕 OCR 校准跳过: {exc}")

    if cfg.get("enable_subtitle_refinement", True):
        if progress_callback:
            progress_callback("正在对照抽帧分析 LLM 校正字幕...")
        try:
            refined_path = refine_subtitle_with_frame_analysis(
                subtitle_path=current_subtitle,
                analysis_json_path=analysis_json_path,
                video_theme=video_theme,
                documentary_settings=cfg,
                progress_callback=progress_callback,
            )
            result["refined_path"] = refined_path
            current_subtitle = refined_path
        except Exception as exc:
            logger.warning(f"LLM 字幕校正失败: {exc}")

    result["subtitle_path"] = current_subtitle
    return result
