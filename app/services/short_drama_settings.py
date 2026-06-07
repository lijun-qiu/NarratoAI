#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧解说规则与成片输出默认值。"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

SHORT_DRAMA_DEFAULTS: Dict[str, Any] = {
    # 解说 : 原声 = 3 : 7
    "narration_percent": 30,
    "original_audio_percent": 70,
    "narration_script_temperature": 0.4,
    "ost1_duration_min": 8,
    "ost1_duration_max": 18,
    "max_consecutive_ost1": 4,
    "ost0_segment_min": 8,
    "ost0_segment_max": 14,
    "ost1_segment_max": 24,
    "narration_chars_min": 40,
    "narration_chars_max": 120,
    # 原声段 picture 旁白烧录
    "enable_picture_narration": True,
    "picture_narration_font_size": 28,
    "picture_narration_color": "#000000",
    "picture_narration_max_chars": 16,
    "picture_narration_duration": 2.0,
    "picture_wrap_double_quotes": True,
}


def compute_short_drama_segment_bounds(source_duration_sec: float) -> tuple[int, int]:
    """按原片时长估算短剧解说 items 合理段数。"""
    minutes = max(1.0, float(source_duration_sec or 0) / 60.0)
    min_segments = max(18, int(round(minutes * 0.75)))
    max_segments = max(min_segments + 8, int(round(minutes * 1.35)))
    return min_segments, max_segments


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


def _read_short_drama_config_section() -> Dict[str, Any]:
    try:
        from app.config.config import _cfg

        section = _cfg.get("short_drama", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("short_drama", {})
        except Exception:
            section = {}
    return dict(section) if isinstance(section, dict) else {}


def get_short_drama_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = deepcopy(SHORT_DRAMA_DEFAULTS)
    for key, value in _read_short_drama_config_section().items():
        if key in SHORT_DRAMA_DEFAULTS:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in SHORT_DRAMA_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def compute_short_drama_ost_bounds(
    total_items: int,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """按 items 总数与 3:7 比例推算 OST=0/1 段数上下限。"""
    cfg = settings or get_short_drama_settings()
    total = max(1, int(total_items or 0))
    narr_pct = float(cfg.get("narration_percent", 30) or 30) / 100.0
    orig_pct = float(cfg.get("original_audio_percent", 70) or 70) / 100.0

    ost0_min_cfg = int(cfg.get("ost0_segment_min", 8) or 8)
    ost0_max_cfg = int(cfg.get("ost0_segment_max", 14) or 14)
    ost1_max_cfg = int(cfg.get("ost1_segment_max", 24) or 24)

    ost0_min = max(ost0_min_cfg, int(round(total * narr_pct * 0.85)))
    ost0_max = max(ost0_min, min(ost0_max_cfg, int(round(total * narr_pct * 1.15)) or ost0_min))
    ost1_max = min(ost1_max_cfg, max(1, int(round(total * orig_pct * 1.05))))
    ost1_min = max(1, int(round(total * orig_pct * 0.75)))

    return {
        "ost0_min": ost0_min,
        "ost0_max": ost0_max,
        "ost1_min": ost1_min,
        "ost1_max": ost1_max,
        "total": total,
    }


def get_short_drama_script_prompt_params(
    *,
    source_duration_sec: float = 0,
    expected_total_segments: int = 0,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """构建短剧解说 LLM 提示词中的段数/比例参数。"""
    cfg = get_short_drama_settings(settings)
    if expected_total_segments <= 0 and source_duration_sec > 0:
        min_seg, max_seg = compute_short_drama_segment_bounds(source_duration_sec)
        expected_total_segments = (min_seg + max_seg) // 2
    if expected_total_segments <= 0:
        expected_total_segments = 30

    bounds = compute_short_drama_ost_bounds(expected_total_segments, cfg)
    return {
        "narration_percent": str(int(cfg.get("narration_percent", 30))),
        "original_audio_percent": str(int(cfg.get("original_audio_percent", 70))),
        "ost0_segment_min": str(bounds["ost0_min"]),
        "ost0_segment_max": str(bounds["ost0_max"]),
        "ost1_segment_max": str(bounds["ost1_max"]),
        "narration_chars_min": str(int(cfg.get("narration_chars_min", 40))),
        "narration_chars_max": str(int(cfg.get("narration_chars_max", 120))),
        "max_consecutive_ost1": str(int(cfg.get("max_consecutive_ost1", 2))),
    }


def get_short_drama_video_output_overrides(
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """短剧解说成片时覆盖 [video_output] 旁白相关项。"""
    cfg = settings or get_short_drama_settings()
    return {
        "enable_picture_narration": bool(cfg.get("enable_picture_narration", True)),
        "picture_narration_font_size": int(cfg.get("picture_narration_font_size", 28)),
        "picture_narration_color": str(cfg.get("picture_narration_color", "#000000")),
        "picture_narration_max_chars": int(cfg.get("picture_narration_max_chars", 16)),
        "picture_narration_duration": float(cfg.get("picture_narration_duration", 2.0)),
    }


def resolve_video_output_for_script_mode(
    base: Optional[Dict[str, Any]] = None,
    *,
    script_path: str = "",
) -> Dict[str, Any]:
    """按脚本模式合并成片输出配置。"""
    from app.services.video_output_settings import get_video_output_settings

    merged = dict(base or get_video_output_settings())
    path = (script_path or "").strip().lower()
    if path == "summary":
        merged.update(get_short_drama_video_output_overrides())
    return merged


def save_short_drama_settings_to_config(settings: Dict[str, Any]) -> bool:
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}

        section = {}
        for key in SHORT_DRAMA_DEFAULTS:
            section[key] = settings.get(key, SHORT_DRAMA_DEFAULTS[key])
        config_data["short_drama"] = section

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))

        try:
            import app.config.config as config_py

            config_py._cfg["short_drama"] = section
        except Exception:
            pass

        logger.info("短剧解说配置已保存到 config.toml [short_drama]")
        return True
    except Exception as exc:
        logger.error(f"保存短剧解说配置失败: {exc}")
        return False
