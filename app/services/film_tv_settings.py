#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说规则参数：默认值、读取 config、UI 覆盖与持久化。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

from app.services.film_tv_rule_presets import (
    DEFAULT_PRESET_ID,
    apply_preset_to_settings,
    format_style_directive,
)

FILM_TV_DEFAULTS: Dict[str, Any] = {
    # 默认方案：《罚罪2》悬疑脉络（原声 48% / 解说 52%）
    "preset_id": DEFAULT_PRESET_ID,
    "target_duration_percent": 30,
    "ost1_duration_min": 5,
    "ost1_duration_max": 12,
    "ost1_duration_long_max": 12,
    "ost1_segment_min": 13,
    "ost1_segment_max": 18,
    "ost0_segment_min": 13,
    "ost0_segment_max": 18,
    "min_total_segments": 30,
    "max_total_segments": 36,
    "picture_chars_max": 12,
    "original_audio_percent": 48,
    "narration_percent": 52,
    "allow_consecutive_ost1": True,
    "enforce_narration_after_ost1": True,
    "narration_chars_min": 48,
    "narration_chars_max": 72,
    "opening_chars_max": 110,
    # 视觉模型增强（字幕 + 关键帧）
    "enable_vision_enrichment": True,
    "vision_scene_interval_sec": 30,
    "vision_max_scene_samples": 80,
    "vision_enrich_picture": True,
    "vision_enrich_narration": True,
    "vision_picture_max_items": 30,
    "vision_segment_max_items": 30,
    # 开场白 / 结尾固定话术
    "enable_opening_closing_hook": True,
    "opening_hook_template": "宝子们，今天咱们一起追《{work_name}》。",
    "closing_hook_template": (
        "本集的核心冲突、留下的悬念和下一集的火药桶，就先帮大家梳理到这儿。"
        "宝子们，觉得讲清楚了点个赞，咱们下期再见。"
    ),
}


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


def _read_film_tv_config_section() -> Dict[str, Any]:
    """仅读取 config.toml 中 [film_tv] 已显式配置的项。"""
    try:
        from app.config.config import _cfg
        film_tv = _cfg.get("film_tv", {})
    except Exception:
        try:
            film_tv = toml.load(_config_file_path()).get("film_tv", {})
        except Exception:
            film_tv = {}
    return dict(film_tv) if isinstance(film_tv, dict) else {}


def _load_film_tv_from_config() -> Dict[str, Any]:
    """兼容旧调用：默认 + config 覆盖（不含方案预设）。"""
    settings = deepcopy(FILM_TV_DEFAULTS)
    for key, value in _read_film_tv_config_section().items():
        if key in FILM_TV_DEFAULTS:
            settings[key] = value
    return settings


def get_film_tv_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并方案预设、config.toml 与页面微调（config 优先于预设）。"""
    config_layer = _read_film_tv_config_section()
    preset_id = (
        (overrides or {}).get("preset_id")
        or config_layer.get("preset_id")
        or FILM_TV_DEFAULTS.get("preset_id")
        or DEFAULT_PRESET_ID
    )
    settings = apply_preset_to_settings(deepcopy(FILM_TV_DEFAULTS), preset_id)
    for key, value in config_layer.items():
        if key in FILM_TV_DEFAULTS:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in FILM_TV_DEFAULTS:
                settings[key] = value
            elif key == "preset_id" and value:
                settings = apply_preset_to_settings(settings, value)
                for ck, cv in config_layer.items():
                    if ck in FILM_TV_DEFAULTS and ck != "preset_id":
                        settings[ck] = cv
    return settings


def get_film_tv_script_prompt_params(
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """构建影视解说 LLM 提示词参数。"""
    cfg = get_film_tv_settings(settings)
    if source_duration_sec and source_duration_sec > 0:
        source_minutes = source_duration_sec / 60
        target_minutes = source_duration_sec * cfg["target_duration_percent"] / 100 / 60
    else:
        source_minutes = 5.0
        target_minutes = 2.0

    return {
        "source_duration_minutes": f"{source_minutes:.1f}",
        "target_output_minutes": f"{target_minutes:.1f}",
        "target_duration_percent": str(int(cfg["target_duration_percent"])),
        "ost1_duration_min": str(int(cfg["ost1_duration_min"])),
        "ost1_duration_max": str(int(cfg["ost1_duration_max"])),
        "ost1_duration_long_max": str(int(cfg["ost1_duration_long_max"])),
        "ost1_segment_min": str(cfg["ost1_segment_min"]),
        "ost1_segment_max": str(cfg["ost1_segment_max"]),
        "ost0_segment_min": str(cfg["ost0_segment_min"]),
        "ost0_segment_max": str(cfg["ost0_segment_max"]),
        "total_segment_min": str(
            max(
                int(cfg["ost1_segment_min"]) + int(cfg["ost0_segment_min"]),
                int(cfg.get("min_total_segments") or 0),
            )
        ),
        "min_total_segments": str(int(cfg.get("min_total_segments") or 30)),
        "total_segment_max": str(int(cfg.get("max_total_segments") or 36)),
        "max_total_segments": str(int(cfg.get("max_total_segments") or 36)),
        "picture_chars_max": str(int(cfg.get("picture_chars_max") or 12)),
        "original_audio_percent": str(int(cfg["original_audio_percent"])),
        "narration_percent": str(int(cfg["narration_percent"])),
        "narration_chars_min": str(int(cfg["narration_chars_min"])),
        "narration_chars_max": str(int(cfg["narration_chars_max"])),
        "opening_chars_max": str(int(cfg["opening_chars_max"])),
        "editing_mode_name": str(cfg.get("preset_name") or "均衡解说"),
        "editor_persona": str(cfg.get("editor_persona") or ""),
        "style_directive": format_style_directive(
            str(cfg.get("style_directive") or ""),
            {
                "ost1_duration_min": str(int(cfg["ost1_duration_min"])),
                "ost1_duration_max": str(int(cfg["ost1_duration_max"])),
                "ost1_segment_min": str(int(cfg["ost1_segment_min"])),
                "ost1_segment_max": str(int(cfg["ost1_segment_max"])),
                "ost0_segment_min": str(int(cfg["ost0_segment_min"])),
                "ost0_segment_max": str(int(cfg["ost0_segment_max"])),
                "max_total_segments": str(int(cfg.get("max_total_segments") or 36)),
                "min_total_segments": str(int(cfg.get("min_total_segments") or 30)),
                "picture_chars_max": str(int(cfg.get("picture_chars_max") or 12)),
                "narration_chars_min": str(int(cfg["narration_chars_min"])),
                "narration_chars_max": str(int(cfg["narration_chars_max"])),
                "opening_chars_max": str(int(cfg["opening_chars_max"])),
            },
        ),
    }


def save_film_tv_settings_to_config(settings: Dict[str, Any]) -> bool:
    """将当前参数写入 config.toml 的 [film_tv] 段。"""
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}

        film_tv = {}
        for key in FILM_TV_DEFAULTS:
            film_tv[key] = settings.get(key, FILM_TV_DEFAULTS[key])
        if settings.get("preset_id"):
            film_tv["preset_id"] = settings["preset_id"]
        config_data["film_tv"] = film_tv

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))

        try:
            from app.config import config as config_module
            config_module.film_tv = film_tv
            import app.config.config as config_py
            config_py._cfg["film_tv"] = film_tv
            config_py.film_tv = film_tv
        except Exception:
            pass

        logger.info("影视解说规则已保存到 config.toml")
        return True
    except Exception as e:
        logger.error(f"保存影视解说配置失败: {e}")
        return False
