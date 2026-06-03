#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""成片输出配置：水印、原声段旁白字幕等。"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

VIDEO_OUTPUT_DEFAULTS: Dict[str, Any] = {
    "watermark_text": "@小超剪辑",
    "enable_picture_narration": True,
    "picture_narration_font_size": 44,
    "picture_narration_color": "#FFE066",
    "picture_narration_max_chars": 12,
    "picture_narration_duration": 2.0,
}


def _config_file_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root, "config.toml")


def get_video_output_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = deepcopy(VIDEO_OUTPUT_DEFAULTS)
    try:
        from app.config.config import _cfg
        section = _cfg.get("video_output", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("video_output", {})
        except Exception:
            section = {}

    if isinstance(section, dict):
        for key in VIDEO_OUTPUT_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]

    if overrides:
        for key, value in overrides.items():
            if key in VIDEO_OUTPUT_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def is_watermark_enabled(settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or get_video_output_settings()
    text = str(cfg.get("watermark_text") or "").strip()
    return bool(text)


def is_picture_narration_enabled(settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or get_video_output_settings()
    return bool(cfg.get("enable_picture_narration", True))


def save_video_output_settings_to_config(settings: Dict[str, Any]) -> bool:
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}

        video_output = {}
        for key in VIDEO_OUTPUT_DEFAULTS:
            video_output[key] = settings.get(key, VIDEO_OUTPUT_DEFAULTS[key])
        config_data["video_output"] = video_output

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))

        try:
            import app.config.config as config_py
            config_py._cfg["video_output"] = video_output
            config_py.video_output = video_output
            from app.config import config as config_module
            config_module.video_output = video_output
        except Exception:
            pass

        logger.info("成片输出配置已保存到 config.toml")
        return True
    except Exception as exc:
        logger.error(f"保存成片输出配置失败: {exc}")
        return False
