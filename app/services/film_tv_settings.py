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

FILM_TV_DEFAULTS: Dict[str, Any] = {
    # 方案 B：高燃精剪（长剧单集约 45 分钟 → 约 10–12 分钟成片）
    "target_duration_percent": 25,
    "ost1_duration_min": 8,
    "ost1_duration_max": 15,
    "ost1_duration_long_max": 20,
    "ost1_segment_min": 30,
    "ost1_segment_max": 42,
    "ost0_segment_min": 6,
    "ost0_segment_max": 10,
    "original_audio_percent": 80,
    "narration_percent": 20,
    "allow_consecutive_ost1": True,
    "enforce_narration_after_ost1": True,
    "narration_chars_min": 35,
    "narration_chars_max": 60,
    "opening_chars_max": 80,
}


def _config_file_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root, "config.toml")


def _load_film_tv_from_config() -> Dict[str, Any]:
    settings = deepcopy(FILM_TV_DEFAULTS)
    try:
        from app.config.config import _cfg
        film_tv = _cfg.get("film_tv", {})
    except Exception:
        try:
            film_tv = toml.load(_config_file_path()).get("film_tv", {})
        except Exception:
            film_tv = {}

    for key in FILM_TV_DEFAULTS:
        if key in film_tv:
            settings[key] = film_tv[key]
    return settings


def get_film_tv_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并 config.toml 默认值与运行时覆盖（如页面调节）。"""
    settings = _load_film_tv_from_config()
    if overrides:
        for key, value in overrides.items():
            if key in FILM_TV_DEFAULTS and value is not None:
                settings[key] = value
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
        "total_segment_min": str(int(cfg["ost1_segment_min"]) + int(cfg["ost0_segment_min"])),
        "original_audio_percent": str(int(cfg["original_audio_percent"])),
        "narration_percent": str(int(cfg["narration_percent"])),
        "narration_chars_min": str(int(cfg["narration_chars_min"])),
        "narration_chars_max": str(int(cfg["narration_chars_max"])),
        "opening_chars_max": str(int(cfg["opening_chars_max"])),
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
