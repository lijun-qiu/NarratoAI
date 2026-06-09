#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：时间格与上传转码参数（独立模块，避免循环导入）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import toml

SEGMENT_INTERVAL_SECONDS = 4

# 上传给视觉模型的转码档位（宽 px · fps · CRF · 音频 · x264 preset）
UPLOAD_TRANSCODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "high": {
        "width": 720,
        "fps": 15,
        "crf": 26,
        "audio_bitrate": "64k",
        "preset": "fast",
    },
    "standard": {
        "width": 640,
        "fps": 15,
        "crf": 28,
        "audio_bitrate": "48k",
        "preset": "fast",
    },
    "compact": {
        "width": 480,
        "fps": 12,
        "crf": 32,
        "audio_bitrate": "32k",
        "preset": "veryfast",
    },
}

VIDEO_EPISODE_UPLOAD_DEFAULTS: Dict[str, Any] = {
    # 单段上传体积上限（MB）；提高可保留更高清画面
    "max_upload_mb": 24.0,
    # 长片按此时长切段上传（秒）
    "chunk_seconds": 300.0,
    # 片长短于此值时用 high 档优先，否则 standard 档优先
    "short_video_high_profile_sec": 300.0,
}

_PROFILE_FALLBACK_ORDER = ("high", "standard", "compact")


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


def get_video_episode_upload_settings(
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = deepcopy(VIDEO_EPISODE_UPLOAD_DEFAULTS)
    try:
        from app.config.config import _cfg

        section = _cfg.get("video_episode_analysis", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("video_episode_analysis", {})
        except Exception:
            section = {}
    if isinstance(section, dict):
        for key in VIDEO_EPISODE_UPLOAD_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]
    if overrides:
        for key, value in overrides.items():
            if key in VIDEO_EPISODE_UPLOAD_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def resolve_upload_profile_chain(*, short_video: bool) -> tuple[str, ...]:
    """按片长返回尝试顺序：先清晰档，仍超限再降级 compact。"""
    if short_video:
        return _PROFILE_FALLBACK_ORDER
    return ("standard", "compact")


def get_upload_transcode_profile(name: str) -> Dict[str, Any]:
    profile = UPLOAD_TRANSCODE_PROFILES.get(name)
    if not profile:
        raise ValueError(f"未知上传转码档位: {name}")
    return dict(profile)
